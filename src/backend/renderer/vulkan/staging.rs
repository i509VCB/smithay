use std::ffi::CStr;

use ash::vk::{self, Handle};
use gpu_allocator::{
    vulkan::{AllocationCreateDesc, AllocationScheme},
    MemoryLocation,
};

use super::{Staging, StagingBuffer, VulkanError, VulkanRenderer};

// TODO: Separate upload queue

impl VulkanRenderer {
    pub(super) fn init_staging(&mut self) -> Result<(), VulkanError> {
        if self.staging.is_some() {
            return Ok(());
        }

        let command_buffer = self
            .command_buffers
            .pop_front()
            .expect("TODO: Handle error/allow creating more buffers, all buffers were consumed");

        // Begin recording the command buffer.
        let begin_info = vk::CommandBufferBeginInfo::builder();

        unsafe {
            self.device
                .begin_command_buffer(command_buffer, &begin_info)
                .expect("Handle error");
        }

        let staging = Staging {
            command_buffer,
            uploads: Vec::new(),
            upload_buffer: self.get_staging_buffer()?,
            upload_overflow: Vec::new(),
        };
        self.staging = Some(staging);

        Ok(())
    }

    fn get_staging_buffer(&mut self) -> Result<StagingBuffer, VulkanError> {
        self.staging_buffers
            .pop_front()
            .map(Ok)
            .unwrap_or_else(|| self.allocate_staging_buffer(Self::STAGING_BUFFER_SIZE))
    }

    pub(super) fn allocate_staging_buffer(
        &mut self,
        size: vk::DeviceSize,
    ) -> Result<StagingBuffer, VulkanError> {
        // FIXME: Handle failure midway through allocations.

        // VUID-VkBufferCreateInfo-size-00912
        assert!(size > 0);

        // VUID-VkBufferCreateInfo-size-06409
        if size > self.limits.max_buffer_size {
            todo!("Buffer allocation is too big");
        }

        // TODO: Check buffer creation limits for max memory size (just like image creation limits)

        let cpu_buffer_create_info = vk::BufferCreateInfo::builder()
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            // The buffer is going to be used for upload on the CPU so it should be a valid source for transfer.
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
            .size(size);
        let cpu = unsafe { self.device.create_buffer(&cpu_buffer_create_info, None) }.unwrap();
        let requirements = unsafe { self.device.get_buffer_memory_requirements(cpu) };

        let cpu_allocation = self
            .allocator
            .allocate(&AllocationCreateDesc {
                name: "CPU side memory image staging buffer",
                requirements,
                // This buffer should be used for uploading data to the GPU.
                location: MemoryLocation::CpuToGpu,
                linear: true, // CPU side memory is always in a linear tiling
                // Staging buffers should be suballocated.
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .expect("Handle error");

        let gpu_buffer_create_info = vk::BufferCreateInfo::builder()
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            // The buffer is going to be used to copy to an image so it should be a valid source for transfer.
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
            .size(size);
        let gpu = unsafe { self.device.create_buffer(&gpu_buffer_create_info, None) }.unwrap();
        let requirements = unsafe { self.device.get_buffer_memory_requirements(gpu) };

        let gpu_allocation = self
            .allocator
            .allocate(&AllocationCreateDesc {
                name: "GPU side memory image staging buffer",
                requirements,
                // This buffer should be used for uploading data to the GPU.
                location: MemoryLocation::GpuOnly,
                linear: true, // Buffer
                // Staging buffers should be suballocated.
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .expect("Handle error");

        // Set the object names if supported
        if let Some(debug_utils) = self.debug_utils.as_ref() {
            // cpu buffer
            let name = CStr::from_bytes_with_nul(b"CPU side image staging buffer\0").unwrap();
            let name_info = vk::DebugUtilsObjectNameInfoEXT::builder()
                .object_handle(cpu.as_raw())
                .object_type(vk::ObjectType::BUFFER)
                .object_name(name);

            unsafe {
                debug_utils
                    .set_debug_utils_object_name(self.device.handle(), &name_info)
                    .expect("Handle error");
            }

            // gpu buffer
            let name = CStr::from_bytes_with_nul(b"GPU side image staging buffer\0").unwrap();
            let name_info = vk::DebugUtilsObjectNameInfoEXT::builder()
                .object_handle(gpu.as_raw())
                .object_type(vk::ObjectType::BUFFER)
                .object_name(name);

            unsafe {
                debug_utils
                    .set_debug_utils_object_name(self.device.handle(), &name_info)
                    .expect("Handle error");
            }
        }

        Ok(StagingBuffer {
            cpu,
            cpu_allocation,
            gpu,
            gpu_allocation,
            size,
            remaining_space: size,
        })
    }

    // TODO: Safety
    pub(super) unsafe fn destroy_staging_buffer(
        &mut self,
        staging: StagingBuffer,
    ) -> Result<(), VulkanError> {
        unsafe {
            self.device.destroy_buffer(staging.gpu, None);
            self.device.destroy_buffer(staging.cpu, None);
        }

        self.allocator.free(staging.gpu_allocation).expect("Handle error");
        self.allocator.free(staging.cpu_allocation).expect("Handle error");

        Ok(())
    }
}
