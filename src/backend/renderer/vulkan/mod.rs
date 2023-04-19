#![deny(unsafe_op_in_unsafe_fn)]
#![allow(missing_docs)]

mod image;
mod staging;

// TODO: Per frame cleanup
// TODO: Reset command buffers from Executing when destructured
// TODO: Buffer allocation for staging.
// TODO: Use VK_KHR_synchronization2 if available
// TODO: Common function to clean up an `Executing` instance (see VulkanRenderer::drop).

// TODO: Document when more concrete
// Required extensions:
// - VK_KHR_dedicated_allocation
//
// Not required but nice to have:
// - VK_KHR_maintenance4
//
// Required for ImportDma and ExportDma
// - VK_KHR_external_memory_fd
// - VK_EXT_external_memory_dma_buf
// - VK_EXT_image_drm_format_modifier

use std::{
    array,
    collections::{HashMap, HashSet, VecDeque},
    fmt,
    mem::ManuallyDrop,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

use ash::{extensions::ext, vk};
use drm_fourcc::DrmFourcc;
use gpu_allocator::{
    vulkan::{Allocation, Allocator, AllocatorCreateDesc},
    AllocatorDebugSettings,
};

use crate::{
    backend::{
        allocator::{format, vulkan::format::get_vk_format},
        vulkan::{Instance, PhysicalDevice, LIBRARY},
    },
    utils::{Buffer, Physical, Rectangle, Size, Transform},
};

use super::{DebugFlags, ExportMem, Frame, ImportMem, Renderer, Texture, TextureFilter, TextureMapping};

pub struct VulkanRenderer {
    images: HashMap<u64, ImageInfo>,

    next_image_id: u64,

    limits: Limits,

    /// Data associated with the staging buffer.
    ///
    /// This will be [`Some`] if a buffer upload or download has been recorded and until a frame is submitted.
    staging: Option<Staging>,

    /// The memory allocator.
    ///
    /// This is wrapped in a [`Box`] to reduce the size of the [`VulkanRenderer`] on the stack.
    ///
    /// This is must be [`ManuallyDrop`]ed to ensure the allocator is not dropped before the device.  
    allocator: ManuallyDrop<Box<Allocator>>,

    debug_utils: Option<ext::DebugUtils>,

    queue: vk::Queue,

    command_pool: vk::CommandPool,

    /// Command buffers are recycled and placed in this queue when a command buffer finishes execution.
    command_buffers: VecDeque<vk::CommandBuffer>,

    /// Data associated with currently executing commands
    executing: VecDeque<Submission>,

    staging_buffers: VecDeque<StagingBuffer>,

    /// Ids of images pending destruction.
    images_pending_destruction: Vec<u64>,

    instance: Instance,

    /// Raw handle to the physical device.
    physical_device: vk::PhysicalDevice,

    // The device is placed in an Arc since it quite large.
    device: Arc<ash::Device>,

    // A [`HashSet`] containing supported [`vk::Format`].
    supported_formats: HashSet<vk::Format>,
}

impl fmt::Debug for VulkanRenderer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VulkanRenderer")
            .field("images", &self.images)
            .field("next_image_id", &self.next_image_id)
            .field("limits", &self.limits)
            .field("allocator", &self.allocator)
            // .field("debug_utils", &self.debug_utils)
            .field("queue", &self.queue)
            .field("command_pool", &self.command_pool)
            .field("command_buffers", &self.command_buffers)
            .field("instance", &self.instance)
            .field("physical_device", &self.physical_device)
            .field("device", &self.device.handle())
            .finish()
    }
}

const MEM_FORMATS: &[DrmFourcc] = &[
    DrmFourcc::Argb8888,
    DrmFourcc::Bgra8888,
    DrmFourcc::Rgba8888,
    DrmFourcc::Xbgr8888,
    DrmFourcc::Argb2101010,
    DrmFourcc::Bgra1010102,
    DrmFourcc::Rgba1010102,
    DrmFourcc::Xrgb2101010,
];

impl VulkanRenderer {
    pub fn new(device: &PhysicalDevice) -> Result<Self, VulkanError> {
        // Check if the required extensions are available
        //
        // TODO: Is this too restrictive?
        // The Vulkan specification says the following in the specification about VkMemoryDedicatedRequirements:
        // > requiresDedicatedAllocation may be VK_TRUE if the image’s tiling is VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT.
        //
        // If the implementation does not support VK_KHR_dedicated_allocation it may be entirely possible that
        // ImportDma would be supported but is actually unusable. If this were to become optional for ImportDma
        // then there would need to be addition tests on formats and modifiers to determine if the ImportDma
        // implementation is even worth advertising support for.
        //
        // For reference wlroots always requires VK_KHR_dedicated_allocation for dmabuf import.
        if !device.has_device_extension(vk::KhrDedicatedAllocationFn::name()) {
            todo!("Missing extensions")
        }

        let physical_device = device.handle();
        let instance_ = device.instance();
        let instance = device.instance().handle();
        let limits = device.limits();

        let max_buffer_size = if device.has_device_extension(vk::KhrMaintenance4Fn::name()) {
            let mut maintenance4 = vk::PhysicalDeviceMaintenance4Properties::default();
            let mut props2 = vk::PhysicalDeviceProperties2::builder().push_next(&mut maintenance4);

            // SAFETY: VK_KHR_maintenance4 is supported by the device.
            unsafe { device.get_properties(&mut props2) }
            maintenance4.max_buffer_size
        } else {
            // We don't know what the driver's maximum buffer size is.
            vk::DeviceSize::MAX
        };

        let limits = Limits {
            max_framebuffer_width: limits.max_framebuffer_width,
            max_framebuffer_height: limits.max_framebuffer_height,
            max_buffer_size,
        };

        // Select a queue
        let queue_families = unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        let (queue_family_index, _) = queue_families
            .iter()
            .enumerate()
            .find(|(_, properties)| properties.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .expect("Handle this error");

        let queue_create_info = vk::DeviceQueueCreateInfo::builder()
            .queue_family_index(queue_family_index as u32)
            .queue_priorities(&[1.0])
            .build();

        let extensions = vec![vk::KhrDedicatedAllocationFn::name()];
        // Note: The `extensions` must live until after the device is created.
        let extensions_ptr = extensions.iter().map(|str| str.as_ptr()).collect::<Vec<_>>();

        let create_info = vk::DeviceCreateInfo::builder()
            .enabled_extension_names(&extensions_ptr)
            .queue_create_infos(array::from_ref(&queue_create_info));
        // SAFETY: TODO
        let device =
            unsafe { instance.create_device(physical_device, &create_info, None) }.expect("Handle error");

        // Now that the device was created we can safely drop the `extensions`. We drop the Vec of pointers
        // as well to prevent misuse.
        drop(extensions_ptr);
        drop(extensions);

        // SAFETY:
        // - VUID-vkGetDeviceQueue-queueFamilyIndex-00384: Queue family index was specified when device was created.
        // - VUID-vkGetDeviceQueue-queueIndex-00385: Only one queue was created, so index 0 is valid.
        let queue = unsafe { device.get_device_queue(queue_family_index as u32, 0) };

        let desc = AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device,
            // TODO: Allow configuring debug settings
            debug_settings: AllocatorDebugSettings::default(),
            buffer_device_address: false,
        };
        let allocator = ManuallyDrop::new(Box::new(Allocator::new(&desc).expect("Handle error")));

        // If the instance has enabled VK_EXT_debug_utils, load the debug utils
        let debug_utils = instance_
            .is_extension_enabled(ext::DebugUtils::name())
            .then(|| ext::DebugUtils::new(LIBRARY.as_ref().unwrap(), &instance));

        let command_pool_create_info = vk::CommandPoolCreateInfo::builder()
            .queue_family_index(queue_family_index as u32)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool = unsafe { device.create_command_pool(&command_pool_create_info, None) }
            .expect("TODO: Handle error");

        let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::builder()
            .command_buffer_count(4)
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY);
        let command_buffers = VecDeque::from(
            unsafe { device.allocate_command_buffers(&command_buffer_allocate_info) }
                .expect("TODO: Handle error"),
        );

        let mut supported_formats = HashSet::new();

        for format in MEM_FORMATS {
            if let Some(vk_format) = get_vk_format(*format) {
                let format_info = vk::PhysicalDeviceImageFormatInfo2::builder()
                    .usage(
                        vk::ImageUsageFlags::TRANSFER_SRC
                            | vk::ImageUsageFlags::TRANSFER_DST
                            | vk::ImageUsageFlags::SAMPLED,
                    )
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .format(vk_format)
                    .ty(vk::ImageType::TYPE_2D);

                let mut image_format_prop = vk::ImageFormatProperties2::default();
                let res = unsafe {
                    instance.get_physical_device_image_format_properties2(
                        physical_device,
                        &format_info,
                        &mut image_format_prop,
                    )
                };

                if res.is_ok() {
                    if image_format_prop.image_format_properties.max_extent.depth == 1 {
                        supported_formats.insert(vk_format);
                    };
                }
            }
        }

        let mut renderer = Self {
            images: HashMap::new(),
            next_image_id: 0,
            limits,
            staging: None,
            allocator,
            debug_utils,
            queue,
            command_pool,
            command_buffers,
            executing: VecDeque::with_capacity(command_buffer_allocate_info.command_buffer_count as usize),
            images_pending_destruction: Vec::new(),
            staging_buffers: VecDeque::with_capacity(2),
            instance: instance_.clone(),
            physical_device,
            device: Arc::new(device),
            supported_formats,
        };

        // Allocate the staging buffers
        let staging_buffers = [
            renderer.allocate_staging_buffer(Self::STAGING_BUFFER_SIZE)?,
            renderer.allocate_staging_buffer(Self::STAGING_BUFFER_SIZE)?,
        ];
        renderer.staging_buffers.extend(staging_buffers);

        Ok(renderer)
    }

    pub fn submit_staging_buffers(&mut self) -> Result<(), VulkanError> {
        // TODO: Return a syncobj
        let Some(Staging { command_buffer, uploads, upload_buffer, upload_overflow }) = self.staging.take() else {
            // Nothing to submit
            return Ok(());
        };

        // VUID-vkQueueSubmit-pCommandBuffers-00070: Finish recording the command buffer
        unsafe { self.device.end_command_buffer(command_buffer) }.expect("Handle error");

        let submit_info = vk::SubmitInfo::builder()
            .command_buffers(array::from_ref(&command_buffer))
            .build();

        // FIXME: What if submitting fails? How are we supposed to respond
        unsafe {
            self.device
                .queue_submit(self.queue, array::from_ref(&submit_info), vk::Fence::null())
                .expect("Handle error");
        }

        let submission = Submission {
            command_buffer,
            uploads,
            upload_buffer,
            upload_overflow,
        };

        self.executing.push_back(submission);

        Ok(())
    }
}

impl Renderer for VulkanRenderer {
    type Error = VulkanError;
    type TextureId = VulkanImage;
    type Frame<'frame> = VulkanFrame<'frame>;

    fn id(&self) -> usize {
        todo!("Smithay needs a global renderer id counter")
    }

    fn downscale_filter(&mut self, _filter: TextureFilter) -> Result<(), Self::Error> {
        todo!()
    }

    fn upscale_filter(&mut self, _filter: TextureFilter) -> Result<(), Self::Error> {
        todo!()
    }

    fn set_debug_flags(&mut self, _flags: DebugFlags) {
        todo!()
    }

    fn debug_flags(&self) -> DebugFlags {
        todo!()
    }

    fn render(
        &mut self,
        _output_size: Size<i32, Physical>,
        _dst_transform: Transform,
    ) -> Result<Self::Frame<'_>, Self::Error> {
        todo!()
    }
}

impl ImportMem for VulkanRenderer {
    fn import_memory(
        &mut self,
        data: &[u8],
        format: DrmFourcc,
        size: Size<i32, Buffer>,
        flipped: bool,
    ) -> Result<Self::TextureId, Self::Error> {
        let texture = self.create_mem_image(format, size, flipped)?;
        // The image contents are empty, so initialize the image content.
        self.update_memory(&texture, data, Rectangle::from_loc_and_size((0, 0), size))?;
        Ok(texture)
    }

    fn update_memory(
        &mut self,
        texture: &Self::TextureId,
        data: &[u8],
        region: Rectangle<i32, Buffer>,
    ) -> Result<(), Self::Error> {
        // Region bounds
        if region.loc.x < 0
            || region.loc.y < 0
            || region.size.w < 0
            || region.size.h < 0
            || region.size.w as u32 > texture.width()
            || region.size.h as u32 > texture.height()
        {
            todo!("Error with bounds")
        }

        let fourcc = texture
            .format()
            .expect("Remove message when non ImportMem buffers are forbidden");
        let _vk_format = texture.vk_format();

        let bpp = format::get_bpp(fourcc).expect("Handle unknown format");
        // In theory bpp / 8 could be technically wrong if the bpp had with a remainder when divided by 8
        let min_size = (bpp / 8) * texture.width() as usize * texture.height() as usize;
        let _data = data.get(0..min_size).expect("Handle error: Too small");

        // TODO: Forbid non ImportMem buffers.
        self.init_staging()?;

        let image = self.images.get(&texture.id).expect("Not possible");

        // Initializing the staging buffers can fail, defer incrementing the refcount until after the staging
        // buffer has been allocated, mapped, command was recorded.

        // TODO: Suballocate a buffer for upload (CpuToGpu).
        // TODO: Record copy command with image as target.

        let image_refcount = image.refcount.clone();
        image_refcount.fetch_add(1, Ordering::Acquire);

        self.staging.as_mut().unwrap().uploads.push(Upload {
            image_refcount,
            image_id: image.id,
        });

        Ok(())
    }

    fn mem_formats(&self) -> Box<dyn Iterator<Item = DrmFourcc>> {
        todo!()
    }
}

impl ExportMem for VulkanRenderer {
    type TextureMapping = VulkanTextureMapping;

    fn copy_framebuffer(
        &mut self,
        _region: Rectangle<i32, Buffer>,
    ) -> Result<Self::TextureMapping, Self::Error> {
        todo!()
    }

    fn copy_texture(
        &mut self,
        _texture: &Self::TextureId,
        _region: Rectangle<i32, Buffer>,
    ) -> Result<Self::TextureMapping, Self::Error> {
        todo!()
    }

    fn map_texture<'a>(
        &mut self,
        _texture_mapping: &'a Self::TextureMapping,
    ) -> Result<&'a [u8], Self::Error> {
        todo!()
    }
}

#[derive(Debug)]
pub struct VulkanTextureMapping {}

impl Texture for VulkanTextureMapping {
    fn width(&self) -> u32 {
        todo!()
    }

    fn height(&self) -> u32 {
        todo!()
    }

    fn format(&self) -> Option<DrmFourcc> {
        todo!()
    }
}

impl TextureMapping for VulkanTextureMapping {
    fn flipped(&self) -> bool {
        todo!()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum VulkanError {}

#[derive(Debug)]
pub struct VulkanImage {
    id: u64,
    refcount: Arc<AtomicUsize>,
    width: u32,
    height: u32,
    vk_format: vk::Format,
    drm_format: Option<DrmFourcc>,
}

impl Texture for VulkanImage {
    fn width(&self) -> u32 {
        self.width
    }

    fn height(&self) -> u32 {
        self.height
    }

    fn format(&self) -> Option<DrmFourcc> {
        self.drm_format
    }
}

impl VulkanImage {
    pub fn vk_format(&self) -> vk::Format {
        self.vk_format
    }
}

impl Clone for VulkanImage {
    fn clone(&self) -> Self {
        self.refcount.fetch_add(1, Ordering::AcqRel);

        Self {
            id: self.id,
            refcount: self.refcount.clone(),
            width: self.width,
            height: self.height,
            vk_format: self.vk_format,
            drm_format: self.drm_format,
        }
    }
}

impl Drop for VulkanImage {
    fn drop(&mut self) {
        self.refcount.fetch_sub(1, Ordering::AcqRel);
    }
}

#[derive(Debug)]
pub struct VulkanFrame<'frame> {
    _marker: std::marker::PhantomData<&'frame ()>,
}

impl<'frame> Frame for VulkanFrame<'frame> {
    type Error = <VulkanRenderer as Renderer>::Error;
    type TextureId = <VulkanRenderer as Renderer>::TextureId;

    fn id(&self) -> usize {
        todo!()
    }

    fn clear(&mut self, _color: [f32; 4], _at: &[Rectangle<i32, Physical>]) -> Result<(), Self::Error> {
        todo!()
    }

    fn render_texture_from_to(
        &mut self,
        _texture: &Self::TextureId,
        _src: Rectangle<f64, Buffer>,
        _dst: Rectangle<i32, Physical>,
        _damage: &[Rectangle<i32, Physical>],
        _src_transform: Transform,
        _alpha: f32,
    ) -> Result<(), Self::Error> {
        todo!()
    }

    fn transformation(&self) -> Transform {
        todo!()
    }

    fn finish(self) -> Result<(), Self::Error> {
        todo!()
    }
}

impl Drop for VulkanRenderer {
    fn drop(&mut self) {
        // FIXME: This should not wait for the device to become idle and instead wait for commands to finish
        // execution. This is because the host if run as guest renderer may never never let the device become
        // idle.
        let _ = unsafe { self.device.device_wait_idle() };

        // TODO: Move this to a common function.
        for submission in self.executing.drain(..).collect::<Vec<_>>() {
            // SAFETY: The commands have finished executing because we just waited for the device to become
            // idle.
            let _ = unsafe { self.cleanup_submission(submission) };
        }

        // Queue all images for destruction.
        self.images_pending_destruction.extend(self.images.keys());
        // SAFETY: TODO
        let _ = unsafe { self.destroy_images() };

        for staging_buffer in self.staging_buffers.drain(..).collect::<Vec<_>>() {
            // SAFETY: TODO
            let _ = unsafe { self.destroy_staging_buffer(staging_buffer) };
        }

        // SAFETY:
        // The allocator needs to be dropped before the device is destroyed
        unsafe { ManuallyDrop::drop(&mut self.allocator) };

        // SAFETY: TODO: VUID-vkDestroyCommandPool-commandPool-00041
        unsafe { self.device.destroy_command_pool(self.command_pool, None) };

        // SAFETY:
        // - VUID-vkDestroyDevice-device-00378: All child objects were destroyed above
        // - Since Drop requires &mut, destruction of the device is externally synchronized
        //   by Rust's type system since only one reference to the device exists.
        unsafe {
            // TODO: For guest renderer check if the renderer owns the device.
            self.device.destroy_device(None);
        }
    }
}

impl VulkanRenderer {
    /// The default size for a staging buffer in bytes.
    ///
    /// This size was chosen to allow a 1024 x 1024 image in a 32-bit color format to be uploaded before any
    /// overflow buffers are used. Assuming damage tracking is being used this is plenty of room for all the
    /// damage boxes from several memory allocated images to be uploaded without any overflow.
    const STAGING_BUFFER_SIZE: vk::DeviceSize = 1024 * 1024 * 4;

    /// # Safety
    ///
    /// - The command buffer must have finished execution.
    unsafe fn cleanup_submission(&mut self, mut submission: Submission) -> Result<(), VulkanError> {
        // Reset the command buffer in case of reuse.
        unsafe {
            self.device
                .reset_command_buffer(submission.command_buffer, vk::CommandBufferResetFlags::empty())
                .expect("Handle error")
        }

        for upload in submission.uploads.drain(..) {
            // Decrement the image refcount to allow future cleanup of image resources.
            let refcount = upload.image_refcount.fetch_sub(1, Ordering::AcqRel);

            // The image is no longer being used, queue the image for destruction.
            if refcount == 0 {
                self.images_pending_destruction.push(upload.image_id);
            }
        }

        // Free any overflow buffers
        for overflow in submission.upload_overflow {
            // TODO: Safety
            unsafe { self.destroy_staging_buffer(overflow) }?;
        }

        // Since the buffer is done being used, reset the remaining size
        submission.upload_buffer.remaining_space = submission.upload_buffer.size;

        // Return the command buffer and upload buffer to the queue
        self.staging_buffers.push_back(submission.upload_buffer);
        self.command_buffers.push_back(submission.command_buffer);

        Ok(())
    }

    unsafe fn destroy_images(&mut self) -> Result<(), VulkanError> {
        for image_id in self.images_pending_destruction.drain(..) {
            if let Some(mut info) = self.images.remove(&image_id) {
                // TODO: For guest image check if the renderer owns the image.
                // TODO: VUID-vkDestroyImage-image-01000 - If destroy is `true`, the currently executing command must finish
                // VUID-vkDestroyImage-image-04882: Not a swapchain image
                unsafe {
                    self.device.destroy_image(info.image, None);
                }

                if let Some(allocation) = info.underlying_memory.take() {
                    match allocation {
                        ImageAllocationType::Allocator(allocation) => {
                            self.allocator.free(allocation).expect("Handle error");
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

/// Internal limits used by the renderer
///
/// This is used instead of keeping a copy of [`vk::PhysicalDeviceLimits`] in the renderer because
/// that type is quite large in memory.
#[derive(Debug)]
struct Limits {
    /// [`vk::PhysicalDeviceLimits::max_framebuffer_width`]
    max_framebuffer_width: u32,
    /// [`vk::PhysicalDeviceLimits::max_framebuffer_height`]
    max_framebuffer_height: u32,
    /// [`vk::PhysicalDeviceMaintenance4Properties::max_buffer_size`]
    max_buffer_size: vk::DeviceSize,
}

/// The type of allocation an image is backed by.
#[derive(Debug)]
enum ImageAllocationType {
    /// The backing memory was allocated using the [`Allocator`].
    Allocator(Allocation),
    // TODO: Imported
}

struct ImageInfo {
    /// The internal id of the image.
    id: u64,

    /// The id of the renderer, used to ensure an image isn't used with the wrong renderer.
    renderer_id: usize,

    /// Number of references to the underlying image resource.
    ///
    /// The refcount is increased to ensure the underlying image resource is not freed while VulkanTexture
    /// handles exist or the texture is used in a command buffer.
    refcount: Arc<AtomicUsize>,

    /// The underlying image resource.
    image: vk::Image,

    /// The underlying memory of the image.
    ///
    /// This will be [`None`] if the renderer does not own the image.
    // TODO: This may be multiple instances of device memory for imported textures.
    underlying_memory: Option<ImageAllocationType>,
}

impl fmt::Debug for ImageInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ImageInfo")
            .field("id", &self.id)
            .field("renderer_id", &self.renderer_id)
            .field("refcount", &self.refcount.load(Ordering::Relaxed))
            .field("image", &self.image)
            .field("allocation", &self.underlying_memory)
            .finish()
    }
}

#[derive(Debug)]
struct Upload {
    /// The refcount of the image.
    ///
    /// This is used to keep the image alive until the commands are uploaded and submitted.
    image_refcount: Arc<AtomicUsize>,

    /// The image id of the image being uploaded.
    image_id: u64,
}

#[derive(Debug)]
struct StagingBuffer {
    /// The CPU side buffer.
    cpu: vk::Buffer,

    /// The CPU side buffer allocation.
    cpu_allocation: Allocation,

    /// The GPU side buffer.
    gpu: vk::Buffer,

    /// The GPU side buffer allocation.
    gpu_allocation: Allocation,

    size: vk::DeviceSize,

    /// Space remaining in the allocation.
    remaining_space: vk::DeviceSize,
}

#[derive(Debug)]
struct Staging {
    /// The staging command buffer.
    command_buffer: vk::CommandBuffer,

    /// Pending upload commands.
    uploads: Vec<Upload>,

    /// The CPU side upload staging buffer.
    upload_buffer: StagingBuffer,

    /// Overflow staging buffers for upload.
    ///
    /// Overflow buffers are created when a memory upload exceeds the size of the staging buffer or enough
    /// data has been uploaded to overflow the capacity of the primary staging buffer.
    upload_overflow: Vec<StagingBuffer>,
    // TODO: Download related stuff
}

/// Information about executing commands.
///
/// This type notably keeps objects alive that cannot be destroyed until command execution has finished.
#[derive(Debug)]
struct Submission {
    // TODO: A way to check if the command is executed (fence or timeline semaphore)
    command_buffer: vk::CommandBuffer,

    /// Upload commands that are executing.
    uploads: Vec<Upload>,

    upload_buffer: StagingBuffer,

    upload_overflow: Vec<StagingBuffer>,
}
