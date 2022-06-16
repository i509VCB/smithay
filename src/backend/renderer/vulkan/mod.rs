//! Implementation of the rendering traits using Vulkan.
//!
//! The features supported by the Vulkan renderer are enabled in groups.
//!
//! The Vulkan renderer requires the following device extensions:
//! - [`VK_KHR_timeline_semaphore`] or Vulkan 1.2
//!
//! The following device extensions are required for [`Dmabuf`] external memory:
//! - [`VK_EXT_image_drm_format_modifier`]
//! - [`VK_KHR_image_format_list`] or Vulkan 1.2
//! - [`VK_EXT_external_memory_dma_buf`]
//! - [`VK_KHR_external_memory_fd`]
//!
//! [`VK_EXT_image_drm_format_modifier`]: https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/VK_EXT_image_drm_format_modifier.html
//! [`VK_KHR_image_format_list`]: https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/VK_KHR_image_format_list.html
//! [`VK_EXT_external_memory_dma_buf`]: https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/VK_EXT_external_memory_dma_buf.html
//! [`VK_KHR_external_memory_fd`]: https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/VK_KHR_external_memory_fd.html
//! [`VK_KHR_timeline_semaphore`]: https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/VK_KHR_timeline_semaphore.html

/*
TODO:
- Staging buffers (upload and download)
- Binding targets
- Command execution
- Clear color
- Public allocator API
- TODO: Dmabuf import
*/

#![allow(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]

mod alloc;
mod dmabuf;
mod extension;
mod format;
mod frame;
mod image;
mod liveness;
mod memory;
mod renderer;
mod wl;

use std::{
    collections::{HashMap, VecDeque},
    ffi::CStr,
};

use ash::{
    extensions::{
        ext::ImageDrmFormatModifier,
        khr::{ExternalMemoryFd, TimelineSemaphore},
    },
    vk,
};
use drm_fourcc::{DrmFormat, DrmModifier};
use gpu_alloc::{AllocationError, GpuAllocator, MemoryBlock};

use crate::backend::vulkan::{version::Version, PhysicalDevice};

use self::{
    alloc::AshMemoryDevice,
    liveness::{Alive, Liveness},
};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Allocation(#[from] AllocationError),

    #[error("The format is not supported")]
    UnsupportedFormat,

    /// Required extensions are missing.
    #[error("Required renderer extensions are missing")]
    MissingRequiredExtensions,

    /// No graphics queue is available
    #[error("No graphics queue is available")]
    NoQueue,

    /// Vulkan API error.
    #[error(transparent)]
    Vulkan(#[from] vk::Result),
}

#[derive(Debug, Clone)]
pub struct VulkanTexture {
    data: ImageData,
    _alive: Alive,
}

#[derive(Debug)]
pub struct VulkanFrame {
    /// Handles used by the frame.
    command_submission: CommandSubmission,
}

/// A renderer utilizing Vulkan.
pub struct VulkanRenderer {
    formats: Formats,

    #[allow(dead_code)]
    valid_memory_types: u32,
    /// GPU memory allocator.
    mem_allocator: GpuAllocator<vk::DeviceMemory>,

    /// All textures managed by the Vulkan renderer.
    images: Vec<Image>,
    /// Staging buffers managed by the
    staging_buffers: Vec<StagingBuffer>,

    /// The timelime semaphore which is signalled when commands submitted to the GPU have completed.
    submit_semaphore: vk::Semaphore,
    /// The number of times commands have been submitted to the GPU.
    submit_count: u64,
    /// List of commands that have been submitted to the gpu.
    ///
    /// Each [`CommandSubmission`] contains what the submit semaphore payload must reach in order to free
    /// resources with the command submission.
    ///
    /// A VecDeque is used since this list of submissions effectively acts like a ring buffer.
    command_submissions: VecDeque<CommandSubmission>,

    queue_family_index: u32,
    queue: vk::Queue,
    command_pool: vk::CommandPool,

    extensions: ExtensionFns,
    enabled_extensions: Vec<&'static CStr>,
    // TODO: Consider allowing a renderer to be a guest.
    // This requires some additional checks, such as ensuring we don't exceed the maximum allocations.
    // - One option would be to state the creator of the renderer must ensure limits are not exceeded.
    //   - A great way to ensure this would be to tell the renderer about reduced limits so the host and
    //     instance the guest renderer won't overstep each other.
    device: ash::Device,
    phd: PhysicalDevice,
    logger: slog::Logger,
}

impl VulkanRenderer {
    pub const MAX_INSTANCE_VERSION: Version = Version::VERSION_1_2; // TODO: Vulkan 1.3 support

    /// Returns all device extensions the device must enable to use the renderer.
    ///
    /// The returned list of extensions is guaranteed to satisfy the valid usage requirements stated in
    /// `VUID-vkCreateDevice-ppEnabledExtensionNames-01387`.
    ///
    /// Returns [`Err`] if the device does not support all the required extensions.
    ///
    /// This is only applicable to [`VulkanRenderer::from_raw_device`]
    pub fn required_device_extensions(phd: &PhysicalDevice) -> Result<Vec<&'static CStr>, Error> {
        let mut extensions = Vec::new();

        // VK_KHR_timeline_semaphore
        if phd.api_version() < Version::VERSION_1_2 {
            // VK_KHR_timeline_semaphore is promoted in Vulkan 1.2
            if !phd.has_device_extension(vk::KhrImageFormatListFn::name()) {
                return Err(Error::MissingRequiredExtensions);
            }

            extensions.push(vk::KhrTimelineSemaphoreFn::name());
        }

        Ok(extensions)
    }

    // TODO: Extensions for features?

    // TODO: Required extensions for sets of capabilities?
    // Although how would we expose internal things like VK_EXT_4444_formats to be enabled?

    /// Returns all device extensions the device should enable to use all the supported features with the renderer.
    ///
    /// The returned list of extensions is guaranteed to satisfy the valid usage requirements stated in
    /// `VUID-vkCreateDevice-ppEnabledExtensionNames-01387`.
    ///
    /// Returns [`Err`] if the device does not support all the required extensions.
    ///
    /// This is only applicable to [`VulkanRenderer::from_raw_device`]
    pub fn optimal_device_extensions(phd: &PhysicalDevice) -> Result<Vec<&'static CStr>, Error> {
        // First the physical device must support the required extensions.
        let mut extensions = Self::required_device_extensions(phd)?;

        /* Dmabuf external memory */

        // VK_EXT_image_drm_format_modifier
        if phd.has_device_extension(vk::ExtImageDrmFormatModifierFn::name())
            // VK_EXT_external_memory_dmabuf
            && phd.has_device_extension(vk::ExtExternalMemoryDmaBufFn::name())
            // VK_KHR_external_memory_fd (dependency of VK_EXT_external_memory_dmabuf)
            && phd.has_device_extension(vk::KhrExternalMemoryFdFn::name())
            // VK_KHR_image_format_list (dependency of VK_EXT_image_drm_format_modifier) is promoted in Vulkan 1.2
            && (phd.has_device_extension(vk::KhrImageFormatListFn::name()) || phd.api_version() >= Version::VERSION_1_2)
        {
            if phd.api_version() < Version::VERSION_1_2 {
                extensions.push(vk::KhrImageFormatListFn::name());
            }

            extensions.extend([
                vk::ExtImageDrmFormatModifierFn::name(),
                vk::ExtExternalMemoryDmaBufFn::name(),
                vk::KhrExternalMemoryFdFn::name(),
            ]);
        }

        // VK_EXT_4444_formats is promoted in Vulkan 1.3
        if phd.api_version() < Version::VERSION_1_3 && phd.has_device_extension(vk::Ext4444FormatsFn::name())
        {
            extensions.push(vk::Ext4444FormatsFn::name());
        }

        Ok(extensions)
    }

    /// Returns true if the device supports the required physical device features to use the renderer.
    ///
    /// This function may be used to filter out physical devices which do not support the Vulkan renderer.
    pub fn supports_required_features(phd: &PhysicalDevice) -> bool {
        // If getting the extensions fails, assume the device is not supported
        Self::required_device_extensions(phd)
            .map(|extensions| Self::supports_required_features_with_extensions(phd, &extensions))
            .unwrap_or(false)
    }

    /// Returns true if the device supports the required physical device features to use the renderer assuming
    /// the device enables the specified extensions.
    ///
    /// This function may be used to filter out physical devices which do not support the Vulkan renderer.
    pub fn supports_required_features_with_extensions(
        phd: &PhysicalDevice,
        extensions: &[&'static CStr],
    ) -> bool {
        PhysicalDeviceFeatures::from_extensions_and_features(phd, extensions).is_supported()
    }

    // TODO: Renderer capabilities.

    pub fn new<L>(phd: &PhysicalDevice, logger: L) -> Result<Self, Error>
    where
        L: Into<Option<slog::Logger>>,
    {
        // Generic inner function used to avoid large amounts of monomorphized and duplicate code.
        unsafe { Self::new_inner(phd, crate::slog_or_fallback(logger)) }
    }

    /// Creates a Vulkan renderer from a raw device.
    ///
    /// This function should not be used over [`VulkanRenderer::new`] unless you need additional device
    /// features or extensions.
    ///
    /// # Safety
    ///
    /// - The `device` must be created from the same `phd`.
    /// - The `device` must support at least Vulkan 1.1.
    /// - The physical device must support all [required physical device features](VulkanRenderer::supports_required_features).
    /// - All safety requirements stated by each field of [`RendererCreateInfo`] must be satisfied.
    /// - The `enabled_extensions` must be a superset of [`VulkanRenderer::required_device_extensions`].
    /// - The created `device` must enable timeline semaphores (see [`PhysicalDeviceTimelineSemaphoreFeatures`]).
    ///   - Testing whether the [required physical device features](VulkanRenderer::supports_required_features)
    ///     are supported will ensure that the device does support timeline semaphores.
    ///
    /// [`PhysicalDeviceTimelineSemaphoreFeatures`]: vk::PhysicalDeviceTimelineSemaphoreFeatures
    pub unsafe fn from_raw_device<L>(
        device: ash::Device,
        phd: &PhysicalDevice,
        enabled_extensions: &[&'static CStr],
        create_info: RendererCreateInfo,
        logger: L,
    ) -> Result<Self, Error>
    where
        L: Into<Option<slog::Logger>>,
    {
        // Generic inner function used to avoid large amounts of monomorphized and duplicate code.
        unsafe {
            Self::from_raw_device_inner(
                device,
                phd,
                enabled_extensions,
                create_info,
                crate::slog_or_fallback(logger),
            )
        }
    }

    /// Returns all the enabled device extensions.
    pub fn enabled_extensions(&self) -> &[&'static CStr] {
        &self.enabled_extensions
    }

    /// Returns the [`Queue`](vk::Queue) used by the renderer.
    pub fn queue(&self) -> vk::Queue {
        self.queue
    }

    /// Returns the index of the queue family the queue belongs to.
    pub fn queue_family(&self) -> u32 {
        self.queue_family_index
    }

    /// Returns the underlying [`ash::Device`].
    ///
    /// The device must NOT be destroyed.
    pub fn device(&self) -> &ash::Device {
        &self.device
    }

    /// Returns the physical device used to create this renderer.
    pub fn physical_device(&self) -> &PhysicalDevice {
        &self.phd
    }

    // TODO: Renderer supported formats

    // TODO: Render graph based command recoding.
    // - This would allow recording commands in a batch for multiple framebuffers.
    // - This would be similar to Renderer::render but provide a way to change the currently bound framebuffer.

    /// Waits for all previously submitted commands to complete.
    ///
    /// Users are generally discouraged from using this function as the thread will block until completion of
    /// gpu commands. This function is intended for debugging purposes.
    pub fn wait(&mut self) -> Result<(), Error> {
        slog::trace!(self.logger, "Waiting for all device commands to complete");

        // All previously submitted commands are completed when the submit semaphore's value is equal to the
        // submit count.
        //
        // &mut reference is intentional to ensure no commands can be submitted async while this occurs.
        let semaphores = [self.submit_semaphore];
        let values = [self.submit_count];

        let wait_info = vk::SemaphoreWaitInfo::builder()
            // VUID-VkSemaphoreWaitInfo-pSemaphores-03256: self.submit_semaphore is a timeline semaphore.
            .semaphores(&semaphores)
            .values(&values);
        unsafe { self.wait_semaphores(&wait_info, u64::MAX) }?;

        // Handle resource destruction since some commands may have completed execution.
        self.cleanup()
    }

    /// Allocates a memory block according to the `request`.
    ///
    /// # Safety
    /// - The returned memory block is only valid for the lifetime of the renderer.
    /// - The returned memory block must be [deallocated](VulkanRenderer::dealloc) before dropping this renderer.
    pub unsafe fn alloc(
        &mut self,
        request: gpu_alloc::Request,
    ) -> Result<MemoryBlock<vk::DeviceMemory>, Error> {
        // SAFETY: The caller must satisfy all the safety requirements.
        unsafe {
            self.mem_allocator
                .alloc(AshMemoryDevice::wrap(&self.device), request)
        }
        .map_err(From::from)
    }

    /// Allocates a memory block according to the `request`.
    ///
    /// This function allows user to force specific allocation strategy. Improper use can lead to suboptimal
    /// performance or large amounts of overhead. Prefer [`VulkanRenderer::alloc`] if in doubt.
    ///
    /// # Safety
    /// - The returned memory block is only valid for the lifetime of the renderer.
    /// - The returned memory block must be [deallocated](VulkanRenderer::dealloc) before dropping this renderer.
    pub unsafe fn alloc_with_dedicated(
        &mut self,
        request: gpu_alloc::Request,
        dedicated: gpu_alloc::Dedicated,
    ) -> Result<MemoryBlock<vk::DeviceMemory>, Error> {
        // SAFETY: The caller must satisfy all the safety requirements.
        unsafe {
            self.mem_allocator
                .alloc_with_dedicated(AshMemoryDevice::wrap(&self.device), request, dedicated)
        }
        .map_err(From::from)
    }

    /// Deallocates a memory block previously allocated by this renderer.
    ///
    /// # Safety
    /// - The memory block must have been allocated by this renderer.
    /// - The underlying memory object of the memory block must not be used after deallocation.
    pub unsafe fn dealloc(&mut self, block: MemoryBlock<vk::DeviceMemory>) {
        // SAFETY: The caller must satisfy all the safety requirements.
        unsafe {
            self.mem_allocator
                .dealloc(AshMemoryDevice::wrap(&self.device), block)
        }
    }
}

#[derive(Debug)]
pub struct RendererCreateInfo {
    // Maintainers note: Adding any fields to this type is a breaking api change and as such must be noted.
    /// The queue family index the queue was created with.
    ///
    /// # Safety
    ///
    /// The index of the queue family must be a graphics queue family.
    pub queue_family_index: u32,

    /// The index of the queue the renderer should use in the specified queue family.
    ///
    /// # Safety
    ///
    /// The queue at this index must have been created from same queue family stated in
    /// [`RendererCreateInfo::queue_family_index`].
    pub queue_index: u32,

    /// The physical device limits.
    ///
    /// # Safety
    ///
    /// This must equal the physical device limits or a lower value.
    pub phd_limits: vk::PhysicalDeviceLimits,

    /// The maximum memory allocation size.
    ///
    /// # Safety
    ///
    /// This must be equal to the value of `maxMemoryAllocationSize` from [`PhysicalDeviceMaintenance3Properties`]
    /// or lower. This value can also be obtained in Vulkan 1.2 using [`PhysicalDeviceVulkan11Properties`].
    ///
    /// [`PhysicalDeviceMaintenance3Properties`]: vk::PhysicalDeviceMaintenance3Properties
    /// [`PhysicalDeviceVulkan11Properties`]: vk::PhysicalDeviceVulkan11Properties
    pub max_memory_size_allocation_size: u64,
}

/// Extra data associated with a Vulkan image.
pub trait VulkanImageExt {
    /// The Vulkan texture format.
    fn vk_format(&self) -> vk::Format;

    /// The drm texture format.
    // TODO: Should this return `Some` if shm is used (with a linear modifier)
    fn drm_format(&self) -> Option<DrmFormat>;

    /// The device memory backing this texture.
    ///
    /// This handle is only valid for the lifetime of the texture and it's renderer.
    fn memory(&self) -> vk::DeviceMemory;

    /// The image handle of this texture.
    ///
    /// This handle is only valid for the lifetime of the texture and it's renderer.
    fn image(&self) -> vk::Image;

    /// The image view of this texture.
    ///
    /// This handle is only valid for the lifetime of the texture and it's renderer.
    fn image_view(&self) -> vk::ImageView;

    /// The usage flags of the image.
    fn usage(&self) -> vk::ImageUsageFlags;

    /// Whether the image contents may be downloaded from the gpu.
    fn downloadable(&self) -> bool {
        self.usage().contains(vk::ImageUsageFlags::TRANSFER_SRC)
    }

    /// Returns `true` if the texture could be exported as a dmabuf.
    ///
    /// This will return `false` if the implementation does not support dmabuf external memory or the image's
    /// format is not exportable.
    fn exportable(&self) -> bool;

    /// Whether this texture was imported from some external memory.
    ///
    /// If this is true, you need to insert a memory barrier when using the texture in a command.
    fn is_imported(&self) -> bool;

    /// The queue family index that the texture is owned by.
    ///
    /// This will only be [`Some`] if [`is_imported`](VulkanTextureExt::is_imported) returns true.
    ///
    /// If this returns [`Some`], when using the image content the image must have ownership transferred from
    /// the returned queue family to the queue family the command is submitted on. Additionally you must also
    /// transfer the image back to the external/foreign queue family to release the acquired underlying image
    /// resource. The returned value will always be [`QUEUE_FAMILY_EXTERNAL`](ash::vk::QUEUE_FAMILY_EXTERNAL)
    /// or [`QUEUE_FAMILY_FOREIGN_EXT`](ash::vk::QUEUE_FAMILY_FOREIGN_EXT)
    fn imported_queue_family(&self) -> Option<u32>;
}

/*
Implementation details: Please keep functions in the `renderer` submodules.
*/

// TODO: Usages:
//
// In theory there are too many image usage combinations. Also usages must be explicitly requested when
// creating an image.
//
// # Shm
//
// General usage: SAMPLED + TRANSFER_DST - This is the minimum required
// Downloadable: SAMPLED + TRANSFER_DST + TRANSFER_SRC
//
// # Dma
// Texture usage: SAMPLED
// Downloadable texture: SAMPLED + TRANSFER_SRC

/// Image usage flags that must be supported for an image to be a render target.
///
/// Both shm and dmabuf have the same usage requirements here.
const RENDER_USAGE: vk::ImageUsageFlags = vk::ImageUsageFlags::COLOR_ATTACHMENT;

/// Image usage flags that must be supported for an image to be used as a valid shm texture.
///
/// A shm texture must be able to be sampled and the destination of a transfer command (for upload).
const SHM_TEXTURE_USAGE: vk::ImageUsageFlags = vk::ImageUsageFlags::from_raw(
    vk::ImageUsageFlags::SAMPLED.as_raw() | vk::ImageUsageFlags::TRANSFER_DST.as_raw(),
);

/// Image usage flags that must be supported for an image to be used as a valid shm texture that can be
/// downloaded.
///
/// A shm texture must be able to be sampled and the source and destination of a transfer command.
const SHM_TEXTURE_DOWNLOAD_USAGE: vk::ImageUsageFlags = vk::ImageUsageFlags::from_raw(
    vk::ImageUsageFlags::SAMPLED.as_raw()
        | vk::ImageUsageFlags::TRANSFER_DST.as_raw()
        | vk::ImageUsageFlags::TRANSFER_SRC.as_raw(),
);

/// Image usage flags that must be supported for a dmabuf backed image to be a valid texture.
const DMA_TEXTURE_USAGE: vk::ImageUsageFlags = vk::ImageUsageFlags::SAMPLED;

/// Aggregate of the `vk::PhysicalDevice*Features` used by the vulkan renderer.
///
/// This type should **never** contain any `Vulkan11`, `Vulkan12` or `Vulkan13` feature types unless the
/// minimum version smithay supports is risen to Vulkan 1.2 (yes the Vulkan11 type was added in Vulkan 1.2) or
/// Vulkan 1.3.
#[derive(Default)]
struct PhysicalDeviceFeatures {
    core: vk::PhysicalDeviceFeatures,
    timeline_semaphore: vk::PhysicalDeviceTimelineSemaphoreFeatures,
    // Optional/extensions
    formats_4444: Option<vk::PhysicalDevice4444FormatsFeaturesEXT>,
}

#[derive(Debug)]
struct StagingBuffer {
    liveness: Liveness,
    buffer: vk::Buffer,
    /// Underlying memory block of the staging buffer.
    ///
    /// This is always [`Some`] (unless the image is being destroyed)
    block: Option<MemoryBlock<vk::DeviceMemory>>,
}

#[derive(Debug)]
struct Image {
    /// Handle used to test if an image handle is alive.
    ///
    /// This is effectively a `Weak<()>`.
    liveness: Liveness,
    data: ImageData,
    /// Underlying memory block of the texture.
    ///
    /// This is always [`Some`] (unless the image is being destroyed)
    memory: Option<MemoryBlock<vk::DeviceMemory>>,
}

/// Internal image data for a Vulkan texture or render buffer.
#[derive(Debug, Clone, Copy)]
struct ImageData {
    width: u32,
    height: u32,
    memory: vk::DeviceMemory, // TODO: offset, size
    image: vk::Image,
    image_view: vk::ImageView,
    usage: vk::ImageUsageFlags,
    vk_format: vk::Format,
    drm_format: Option<DrmFormat>,
    /// The external/foreign queue family the texture belongs to.
    ///
    /// This must be `QUEUE_FAMILY_EXTERNAL` or `QUEUE_FAMILY_FOREIGN`.
    imported_queue_family: Option<u32>,
}

#[derive(Debug)]
struct CommandSubmission {
    /// Alive handles to keep objects alive that are used in this frame submission.
    handles: Vec<Alive>,
    /// The counter value that must be reached for the frame submission to be completed.
    counter_value: u64,
}

/// A wrapper type representing a set of functionality originally defined by extensions which has been promoted
/// to the Vulkan core specification in a later version.
enum MaybePromoted<T> {
    /// The loaded extension functions.
    ///
    /// The functions defined in the type should be used.
    Extension(T),

    /// The extension has been promoted.
    ///
    /// This means the functionality should be performed using one of the functions in [`ash::Device`].
    Promoted,
}

// TODO: Remove this warning before release
#[allow(dead_code)]
/// Extension functions used by the renderer.
struct ExtensionFns {
    /// Use `VulkanRenderer::get_semaphore_counter_value` and `VulkanRenderer::wait_semaphores` instead of
    /// this matching on this type.
    khr_timeline_semaphore: MaybePromoted<TimelineSemaphore>,

    /// Extension functions for dmabuf external memory support.
    dmabuf_external_memory: Option<DmabufMemoryFns>,
}

/// Extension functions related to dmabuf external memory.
struct DmabufMemoryFns {
    ext_image_drm_format_modifier: ImageDrmFormatModifier,
    ext_external_memory_dmabuf: ExternalMemoryFd,
}

/// Information about supported renderer formats.
#[derive(Debug)]
struct Formats {
    /// Image properties for shm formats.
    shm_formats: HashMap<vk::Format, ShmFormat>,

    drm_importable: Vec<DrmFormat>,

    /// A mapping of format to all supported DRM modifiers.
    ///
    /// The value of an entry is all modifiers the implementation supports for image tiling, (we provide a
    /// list for Vulkan to choose from).
    ///
    /// For import, this mapping is used to lookup all valid modifiers for a format.
    ///
    /// Why not use [`DrmModifier`]? Vulkan expects a slice of u64 and we try to avoid unneeded transmutes.
    drm_modifiers: HashMap<vk::Format, Vec<u64>>,
    // TODO: Iterator from dma_format_to_modifiers -> dma_importable
    /// All supported formats that support the required dmabuf formats to be imported as a texture.
    drm_properties: HashMap<vk::Format, DrmFormatInfo>,

    #[cfg(feature = "wayland_frontend")]
    wl_shm: Vec<wayland_server::protocol::wl_shm::Format>,
}

#[derive(Debug)]
struct ShmFormat {
    /// Properties for an shm format used as a texture.
    texture: ShmFormatProperties,
    /// Properties for an shm format used as a downloadable texture.
    download: Option<ShmFormatProperties>,
}

#[derive(Debug)]
struct ShmFormatProperties {
    format: vk::FormatProperties,
    image: vk::ImageFormatProperties,
}

#[derive(Debug)]
struct DrmFormatInfo {
    /// Modifier to modifier properties mapping
    texture: HashMap<DrmModifier, FormatModifierProperties>,
    // TODO:
    // - Download
    // - Renderbuffer
}

#[derive(Debug)]
struct FormatModifierProperties {
    image_properties: vk::ImageFormatProperties,
    modifier_properties: vk::DrmFormatModifierPropertiesEXT,
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use ash::vk;
    use slog::Drain;

    use crate::backend::{
        renderer::{ImportDma, ImportMem, ImportMemWl},
        vulkan::{Instance, PhysicalDevice},
    };

    use super::VulkanRenderer;

    #[test]
    fn create_renderer() {
        let logger = slog::Logger::root(Mutex::new(slog_term::term_full().fuse()).fuse(), slog::o!());

        let instance =
            Instance::new(VulkanRenderer::MAX_INSTANCE_VERSION, None, logger.clone()).expect("No instance");
        let phd = PhysicalDevice::enumerate(&instance)
            .unwrap()
            .filter(|phd| phd.ty() != vk::PhysicalDeviceType::CPU)
            // Do not select devices which do not support all the required features.
            .filter(VulkanRenderer::supports_required_features)
            .next()
            .expect("No devices");

        let mut renderer = VulkanRenderer::new(&phd, logger).unwrap();

        println!("Shm formats: {:#?}", renderer.shm_formats());

        let drm_formats = renderer.dmabuf_formats().copied().collect::<Vec<_>>();

        println!("Dmabuf formats: {:#?}", drm_formats,);

        let texture = renderer
            .import_memory(&[0xFF, 0xFF, 0xFA, 0xFA], (1, 1).into(), false)
            .expect("Failed to import memory");

        renderer
            .wait()
            .expect("Failed to wait for command execution to complete");

        drop(renderer);
        dbg!(&texture);
        drop(texture);
    }
}
