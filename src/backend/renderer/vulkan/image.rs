use std::{
    ffi::CString,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

use ash::vk::{self, Handle};
use drm_fourcc::DrmFourcc;
use gpu_allocator::{
    vulkan::{AllocationCreateDesc, AllocationScheme},
    MemoryLocation,
};
use scopeguard::ScopeGuard;

use crate::{
    backend::{allocator::vulkan, renderer::vulkan::ImageInfo},
    utils::{Buffer, Size},
};

use super::{ImageAllocationType, VulkanError, VulkanImage, VulkanRenderer};

impl VulkanRenderer {
    /// Validate image parameters shared by all types of vulkan images created by the renderer.
    ///
    /// **Note**: The following must be validated by the caller before creating an image after this function
    /// is called:
    ///
    /// 1. The width and height must be less than the limits defined in the [Image Creation Limits].
    ///
    /// [Image Creation Limits]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#resources-image-creation-limits
    fn common_image_validation(&self, format: DrmFourcc, size: Size<i32, Buffer>) -> Result<(), VulkanError> {
        // First the format needs to have an equivalent Vulkan format.
        //
        // This only checks that the image can actually be used in Vulkan.
        vulkan::format::get_vk_format(format).expect("Handle error");

        // VUID-VkImageCreateInfo-extent-00944
        // VUID-VkImageCreateInfo-extent-00945
        if size.w <= 0 || size.h <= 0 {
            todo!("Error")
        }

        // Possible change: If memory images can be a valid Bind target, then these VUIDs need to be checked for
        // (color attachment usage):
        // VUID-VkImageCreateInfo-usage-00964
        // VUID-VkImageCreateInfo-usage-00965

        // We don't actually want to check VkPhysicalDeviceLimits::maxImageDimension2D here since that limits
        // the maximum image size that is supported per the Vulkan specification:
        // > Some combinations of image parameters (format, usage, etc.) may allow support for larger
        // > dimensions, which can be queried using vkGetPhysicalDeviceImageFormatProperties.

        Ok(())
    }

    pub(super) fn create_mem_image(
        &mut self,
        format: DrmFourcc,
        size: Size<i32, Buffer>,
        // TODO: Use flipped
        _flipped: bool,
    ) -> Result<VulkanImage, VulkanError> {
        // TODO: Could loosen the usage requirements if ExportMem could fail to work for some formats
        // When impl const is available in ash, remove the from_raw and as_raw calls.
        const USAGE_FLAGS: vk::ImageUsageFlags = vk::ImageUsageFlags::from_raw(
            vk::ImageUsageFlags::TRANSFER_SRC.as_raw()
                | vk::ImageUsageFlags::TRANSFER_DST.as_raw()
                | vk::ImageUsageFlags::SAMPLED.as_raw(),
        );

        self.common_image_validation(format, size)?;

        let vk_format = vulkan::format::get_vk_format(format).expect("Already validated");

        let format_info = vk::PhysicalDeviceImageFormatInfo2::builder()
            .format(vk_format)
            .tiling(vk::ImageTiling::OPTIMAL)
            .ty(vk::ImageType::TYPE_2D)
            .usage(USAGE_FLAGS);

        let mut properties = vk::ImageFormatProperties2::builder();

        // Ensure the image will fulfill the image creation limits.
        unsafe {
            self.instance
                .handle()
                .get_physical_device_image_format_properties2(
                    self.physical_device,
                    &format_info,
                    &mut properties,
                )
                .expect("Handle error")
        };

        let max_extent = properties.image_format_properties.max_extent;

        // VUID-VkImageCreateInfo-extent-02252
        // VUID-VkImageCreateInfo-extent-02253
        // VUID-VkImageCreateInfo-extent-02254
        if size.w as u32 > max_extent.width || size.h as u32 > max_extent.height || 1 > max_extent.depth {
            todo!()
        }

        // VUID-VkImageCreateInfo-mipLevels-00947
        // VUID-VkImageCreateInfo-mipLevels-02255
        if properties.image_format_properties.max_mip_levels < 1 {
            todo!()
        }

        // VUID-VkImageCreateInfo-arrayLayers-00948
        // VUID-VkImageCreateInfo-arrayLayers-02256
        if properties.image_format_properties.max_array_layers < 1 {
            todo!()
        }

        // VUID-VkImageCreateInfo-samples-02258
        if !properties
            .image_format_properties
            .sample_counts
            .contains(vk::SampleCountFlags::TYPE_1)
        {
            todo!()
        }

        let create_info = vk::ImageCreateInfo::builder()
            .flags(vk::ImageCreateFlags::empty())
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk_format)
            .extent(vk::Extent3D {
                width: size.w as u32,
                height: size.h as u32,
                // VUID-VkImageCreateInfo-imageType-00957: 2D images always have a depth of 1
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(USAGE_FLAGS)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            // At the moment the renderer does not use multiple queues
            // .queue_family_indices(queue_family_indices)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        // Create a scope guard to prevent memory leaks if future Vulkan commands fail.
        let image = scopeguard::guard(
            unsafe { self.device.create_image(&create_info, None) }.expect("Handle error"),
            |image| unsafe { self.device.destroy_image(image, None) },
        );

        let image_requirements_info = vk::ImageMemoryRequirementsInfo2::builder()
            // This clone is free but we do not want to drop the scopeguard.
            .image(image.clone());
        let mut dedicated_requirements = vk::MemoryDedicatedRequirements::default();
        let mut requirements = vk::MemoryRequirements2::builder().push_next(&mut dedicated_requirements);

        // SAFETY: The image was just created and has no bound memory.
        unsafe {
            self.device
                .get_image_memory_requirements2(&image_requirements_info, &mut requirements)
        };
        let name = format!("Memory Image {}", self.next_image_id);

        // In order to avoid borrowing conflicts take the base requirements out ahead of time.
        let requirements = requirements.memory_requirements;

        // If the driver requires a dedicated allocation then we have no other choice.
        let allocation_scheme = if dedicated_requirements.requires_dedicated_allocation == vk::TRUE ||
            // If the driver prefers a dedicated allocation then use a dedicated allocation.
            //
            // On Unix-like platforms we aren't too concerned about running out of allocations unlike windows
            // where the max allocation count is around 4096.
            dedicated_requirements.prefers_dedicated_allocation == vk::TRUE
        {
            AllocationScheme::DedicatedImage(image.clone())
        } else {
            AllocationScheme::GpuAllocatorManaged
        };

        // Create a scope guard to prevent memory leaks if future Vulkan commands fail.
        let allocation = scopeguard::guard(
            self.allocator
                .allocate(&AllocationCreateDesc {
                    name: &name,
                    requirements,
                    location: MemoryLocation::GpuOnly,
                    // optimal tiling
                    linear: false,
                    allocation_scheme,
                })
                .expect("Handle error"),
            |allocation| {
                self.allocator
                    .free(allocation)
                    .expect("Handle error: Error while freeing failed allocation")
            },
        );

        unsafe {
            self.device
                .bind_image_memory(*image, allocation.memory(), allocation.offset())
        }
        .expect("Handle error");

        // Attach a name to allow for easier debugging with tools like Renderdoc
        if let Some(ref debug_utils) = self.debug_utils {
            let name = CString::new(name).expect("Unreachable");
            let name_info = vk::DebugUtilsObjectNameInfoEXT::builder()
                .object_handle(image.as_raw())
                .object_type(vk::ObjectType::IMAGE)
                .object_name(&name);

            unsafe { debug_utils.set_debug_utils_object_name(self.device.handle(), &name_info) }
                .expect("Handle error");
        }

        let id = self.next_image_id;
        self.next_image_id += 1;

        let info = self.images.entry(id).or_insert(ImageInfo {
            id,
            renderer_id: 0, // TODO
            // Initialize with a refcount of 1 since a new image handle is being created.
            refcount: Arc::new(AtomicUsize::new(1)),
            // Image creation was successful, disarm the scope guards.
            image: ScopeGuard::into_inner(image),
            underlying_memory: Some(ImageAllocationType::Allocator(ScopeGuard::into_inner(allocation))),
        });

        let image = VulkanImage {
            id,
            refcount: Arc::clone(&info.refcount),
            width: size.w as u32,
            height: size.h as u32,
            vk_format,
            drm_format: Some(format),
        };

        Ok(image)
    }

    /// Performs cleanup on image resources.
    ///
    /// # Safety
    ///
    /// If `destroy` is [`true`] then the renderer must getting dropped.
    pub(super) unsafe fn cleanup_images(&mut self, destroy: bool) {
        // TODO: Use HashMap::drain_filter when stabilized
        let keys = self
            .images
            .iter()
            .filter(|(_, info)| {
                // Handle destroy for drop
                let mut destroy = destroy;

                // If the refcount of the image has reached 0 then all image handles have been dropped and
                // the image is not being used in any commands.
                destroy |= info.refcount.load(Ordering::Acquire) > 0;
                destroy
            })
            .map(|(key, _)| key)
            .copied()
            .collect::<Vec<_>>();

        for key in keys {
            if let Some(image_data) = self.images.remove(&key) {
                // TODO: For guest image check if the renderer owns the image.
                unsafe {
                    // TODO: VUID-vkDestroyImage-image-01000 - If destroy is `true`, the currently executing command must finish
                    // VUID-vkDestroyImage-image-04882: Not a swapchain image
                    self.device.destroy_image(image_data.image, None);
                }

                // If the image owns it's memory, free the memory as well
                if let Some(allocation) = image_data.underlying_memory {
                    match allocation {
                        ImageAllocationType::Allocator(allocation) => {
                            self.allocator
                                .free(allocation)
                                .expect("Error while freeing image allocation");
                        }
                    }
                }
            }
        }
    }
}
