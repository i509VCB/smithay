#![deny(unsafe_op_in_unsafe_fn)]

use std::{
    collections::HashMap,
    ffi::CStr,
    fmt::{self, Display},
    mem,
    os::fd::OwnedFd,
    sync::{Arc, Weak},
    time::Duration,
};

use ash::vk;
use bitflags::bitflags;
use gpu_alloc::GpuAllocator;
use smithay::{
    backend::{
        allocator::Fourcc,
        renderer::{
            sync::{Fence, SyncPoint},
            DebugFlags, Frame, Renderer, Texture, TextureFilter,
        },
    },
    utils::{Buffer, Physical, Rectangle, Size, Transform},
};
use tracing::{
    span::{self, EnteredSpan},
    Level,
};

mod format;
mod pipeline;

#[derive(Debug)]
pub enum VulkanError {}

impl Display for VulkanError {
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!()
    }
}

impl std::error::Error for VulkanError {}

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct Capabilities: u64 {
        /// Dmabufs can be imported into the renderer.
        const IMPORT_EXPORT_DMABUF = 0x0000_0001;

        /// Fences can be imported into the renderer.
        const IMPORT_FENCE = 0x0000_0002;

        /// Fences can be exported from the renderer.
        const EXPORT_FENCE = 0x0000_0004;

        /// Whether the renderer's device can be created with a non-default priority.
        const PRIORITY = 0x0000_0008;
    }
}

/// Requirements for the instance in order to create a [`VulkanRenderer`].
#[non_exhaustive]
#[derive(Debug)]
pub struct InstanceRequirements {
    /// Instance extensions that must be enabled.
    pub extensions: Vec<&'static CStr>,
}

#[non_exhaustive]
#[derive(Debug)]
pub struct DeviceRequirements {
    /// Device extensions that must be enabled.
    pub extensions: Vec<&'static CStr>,
    // TODO: Fields for features such as:
    // - samplerYcbcrConversion
}

pub struct VulkanRenderer {
    span: tracing::Span,
    capabilities: Capabilities,

    samplers: HashMap<SamplerKey, vk::Sampler>,
    min_filter: TextureFilter,
    max_filter: TextureFilter,
    // GpuAllocator is boxed up due to the significant size.
    mem_allocator: Box<GpuAllocator<vk::DeviceMemory>>,
    queue: vk::Queue,
    device: Arc<ash::Device>,
    physical_device: vk::PhysicalDevice,
}

impl fmt::Debug for VulkanRenderer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VulkanRenderer")
            .field("span", &self.span)
            .field("capabilities", &self.capabilities)
            .field("mem_allocator", &self.mem_allocator)
            .field("queue", &self.queue)
            .field("device", &self.device.handle())
            .field("physical_device", &self.physical_device)
            .finish()
    }
}

impl VulkanRenderer {
    /// Capabilities a [`VulkanRenderer`] can support when an instance is created.
    ///
    /// A capability being supported by the instance does not necessarily mean a physical device will
    /// support that capability. If a capability is not supported by the instance, then no physical devices
    /// will support the capability.
    pub fn supported_instance_capabilities(
        entry: &ash::Entry,
        instance_version: u32,
    ) -> Result<Capabilities, ()> {
        assert!(instance_version <= vk::API_VERSION_1_3);

        // QUEUE_PRIORITY: Dictated by device.
        let mut capabilities = Capabilities::PRIORITY;

        let extensions = entry
            .enumerate_instance_extension_properties(None)
            .expect("error");

        if !find_extension(&extensions, vk::KhrGetPhysicalDeviceProperties2Fn::name()) {
            todo!("Error: VK_KHR_get_physical_device_properties2 is always required")
        }

        if find_extension(&extensions, vk::KhrExternalMemoryCapabilitiesFn::name()) {
            capabilities |= Capabilities::IMPORT_EXPORT_DMABUF;
        }

        if find_extension(&extensions, vk::KhrExternalFenceCapabilitiesFn::name()) {
            capabilities |= Capabilities::IMPORT_FENCE | Capabilities::EXPORT_FENCE;
        }

        Ok(capabilities)
    }

    /// Requirements an instance must meet to create a [`VulkanRenderer`].
    pub fn instance_requirements(capabilities: Capabilities) -> Result<InstanceRequirements, ()> {
        let mut extensions = Vec::new();

        // Or Vulkan 1.1
        extensions.push(vk::KhrGetPhysicalDeviceProperties2Fn::name());

        if capabilities.contains(Capabilities::IMPORT_EXPORT_DMABUF) {
            // Or Vulkan 1.1
            extensions.push(vk::KhrExternalMemoryCapabilitiesFn::name());
        }

        if capabilities.contains(Capabilities::IMPORT_FENCE)
            || capabilities.contains(Capabilities::EXPORT_FENCE)
        {
            // Or Vulkan 1.1
            extensions.push(vk::KhrExternalFenceCapabilitiesFn::name());
        }

        Ok(InstanceRequirements { extensions })
    }

    /// Capabilities a [`VulkanRenderer`] can support with a given physical device.
    pub unsafe fn supported_capabilities(
        entry: &ash::Entry,
        instance: &ash::Instance,
        instance_capabilities: Capabilities,
        physical_device: vk::PhysicalDevice,
        device_api_version: u32,
    ) -> Result<Capabilities, ()> {
        let device_extensions =
            unsafe { instance.enumerate_device_extension_properties(physical_device) }.expect("error");

        let mut capabilities = Capabilities::empty();

        if instance_capabilities.contains(Capabilities::IMPORT_EXPORT_DMABUF) {
            let memory_dmabuf = find_extension(&device_extensions, vk::ExtExternalMemoryDmaBufFn::name());
            let image_drm_modifier =
                find_extension(&device_extensions, vk::ExtImageDrmFormatModifierFn::name());
            // Technically not needed, but there is no real way at the moment to query what device a dmabuf
            // belongs to: therefore we must assume the worst case and do a foreign queue transfer.
            let queue_family_foreign =
                find_extension(&device_extensions, vk::ExtQueueFamilyForeignFn::name());

            // Extensions or Vulkan 1.2
            let image_format_list = find_extension(&device_extensions, vk::KhrImageFormatListFn::name());

            // For VK_KHR_sampler_ycbcr_conversion explicitly check for Vulkan 1.1 since we do not require
            // the samplerYcbcrConversion capability.
            //
            // v3dv for some time was only able to expose VK_EXT_image_drm_format_modifier because the driver
            // was Vulkan 1.1 capable. However in Vulkan 1.0, if the VK_KHR_sampler_ycbcr_conversion is enabled,
            // you need to also enable the samplerYcbcrConversion capability.
            //
            // Smithay doesn't necessarily require ycbcr conversions for dmabuf import and export and such does
            // not mandate the samplerYcbcrConversion capability,
            let sampler_ycbcr_conversion =
                find_extension(&device_extensions, vk::KhrSamplerYcbcrConversionFn::name())
                    || device_api_version >= vk::API_VERSION_1_1;

            // Extensions or Vulkan 1.1
            let bind_memory2 = find_extension(&device_extensions, vk::KhrBindMemory2Fn::name());
            let maintenance1 = find_extension(&device_extensions, vk::KhrMaintenance1Fn::name());
            let get_mem_reqs2 = find_extension(&device_extensions, vk::KhrGetMemoryRequirements2Fn::name());
            let external_mem = find_extension(&device_extensions, vk::KhrExternalMemoryFn::name());

            if memory_dmabuf
                && image_drm_modifier
                && queue_family_foreign
                && image_format_list
                && sampler_ycbcr_conversion
                && bind_memory2
                && maintenance1
                && get_mem_reqs2
                && external_mem
            {
                capabilities |= Capabilities::IMPORT_EXPORT_DMABUF;
            }
        }

        if instance_capabilities.contains(Capabilities::IMPORT_FENCE)
            || instance_capabilities.contains(Capabilities::EXPORT_FENCE)
        {
            if find_extension(&device_extensions, vk::KhrExternalFenceFdFn::name())
                && find_extension(&device_extensions, vk::KhrExternalFenceFn::name())
            {
                // Although the extensions may be available, we still need to know what operations the device
                // supports.
                //
                // SAFETY: If the instance capabilities contains IMPORT_FENCE or EXPORT_FENCE, the caller must
                //         ensure the instance requirements for that capability are met when creating the instance.
                let props =
                    unsafe { get_sync_fd_fence_properties(entry, instance.handle(), physical_device) };

                // Quoting from the Vulkan specification:
                //
                // > If handleType is not supported by the implementation, then VkExternalFenceProperties::externalFenceFeatures
                // > will be set to zero.
                //
                // In particular we do not want to care about OPAQUE_FD due to the external fence handle types
                // compatibility requirements which mandate device and driver uuids match when sharing a fence.
                // See VkExternalFenceHandleTypeFlagBits in Vulkan specification for the table.
                if props
                    .compatible_handle_types
                    .contains(vk::ExternalFenceHandleTypeFlags::SYNC_FD)
                {
                    if props
                        .external_fence_features
                        .contains(vk::ExternalFenceFeatureFlags::IMPORTABLE)
                    {
                        capabilities |= Capabilities::IMPORT_FENCE;
                    }

                    if props
                        .external_fence_features
                        .contains(vk::ExternalFenceFeatureFlags::EXPORTABLE)
                    {
                        capabilities |= Capabilities::EXPORT_FENCE;
                    }
                }
            }
        }

        if instance_capabilities.contains(Capabilities::PRIORITY) {
            // Or Vulkan 1.2
            if find_extension(&device_extensions, vk::KhrGlobalPriorityFn::name()) {
                capabilities |= Capabilities::PRIORITY;
            }
        }

        Ok(capabilities)
    }

    /// Requirements a device must meet to create a [`VulkanRenderer`].
    pub unsafe fn device_requirements(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        device_api_version: u32,
        capabilities: Capabilities,
    ) -> Result<DeviceRequirements, ()> {
        assert!(device_api_version <= vk::API_VERSION_1_3);

        let supported_extensions =
            unsafe { instance.enumerate_device_extension_properties(physical_device) }.expect("Handle error");

        let mut extensions = Vec::new();

        if capabilities.contains(Capabilities::IMPORT_EXPORT_DMABUF) {
            extensions.extend([
                vk::ExtImageDrmFormatModifierFn::name(),
                vk::ExtExternalMemoryDmaBufFn::name(),
                vk::KhrExternalMemoryFdFn::name(),
                // Or Vulkan 1.2
                vk::KhrImageFormatListFn::name(),
                // Or Vulkan 1.1
                vk::KhrBindMemory2Fn::name(),
                vk::KhrMaintenance1Fn::name(),
                vk::KhrGetMemoryRequirements2Fn::name(),
                vk::KhrExternalMemoryFn::name(),
            ]);

            if device_api_version < vk::API_VERSION_1_1 {
                // For v3dv do not require ycbcr conversion if Vulkan 1.1+ is used
                extensions.push(vk::KhrSamplerYcbcrConversionFn::name());
            }

            // VK_KHR_dedicated_allocation is needed to handle disjoint dmabuf import: require if available.
            if find_extension(&supported_extensions, vk::KhrDedicatedAllocationFn::name()) {
                extensions.push(vk::KhrDedicatedAllocationFn::name());
            }
        }

        if capabilities.contains(Capabilities::IMPORT_FENCE)
            || capabilities.contains(Capabilities::EXPORT_FENCE)
        {
            extensions.push(vk::KhrExternalFenceFdFn::name());
            extensions.push(vk::KhrExternalFenceFn::name());
        }

        // Optional extensions that are preferable to enable.

        // Driver info is very useful for debugging purposes: require if available.
        if find_extension(&supported_extensions, vk::KhrDriverPropertiesFn::name()) {
            extensions.push(vk::KhrDriverPropertiesFn::name());
        }

        // If available require 4444 formats. This is core and optional in Vulkan 1.3.
        if find_extension(&supported_extensions, vk::Ext4444FormatsFn::name()) {
            extensions.push(vk::Ext4444FormatsFn::name());
        }

        // If available, let the allocator query the max allocation size.
        if find_extension(&supported_extensions, vk::KhrMaintenance3Fn::name()) {
            extensions.push(vk::KhrMaintenance3Fn::name());
        }

        // And the same for maxBufferSize
        if find_extension(&supported_extensions, vk::KhrMaintenance4Fn::name()) {
            extensions.push(vk::KhrMaintenance4Fn::name());
        }

        // TODO: VK_KHR_maintenance5 for VK_FORMAT_A1B5G5R5_UNORM_PACK16_KHR and VK_FORMAT_A8_UNORM_KHR.

        Ok(DeviceRequirements { extensions })
    }

    pub fn capabilities(&self) -> Capabilities {
        self.capabilities
    }

    #[profiling::function]
    pub fn import_fence(&mut self, _fd: OwnedFd) -> Result<VulkanFence, VulkanError> {
        todo!()
    }

    /// Create a new fence.
    ///
    /// If the EXPORT_FENCE capability is supported, then the fence can be exported.
    #[profiling::function]
    pub fn create_fence(&mut self) -> Result<VulkanFence, VulkanError> {
        todo!()
    }
}

impl Renderer for VulkanRenderer {
    type Error = VulkanError;
    type TextureId = VulkanImage;
    type Frame<'frame> = VulkanFrame<'frame>;

    fn id(&self) -> usize {
        todo!()
    }

    fn downscale_filter(&mut self, filter: TextureFilter) -> Result<(), Self::Error> {
        self.min_filter = filter;
        Ok(())
    }

    fn upscale_filter(&mut self, filter: TextureFilter) -> Result<(), Self::Error> {
        self.max_filter = filter;
        Ok(())
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
        // Load sampler for descriptors based on set filters.
        let _sampler = self
            .samplers
            .get(&SamplerKey {
                min_filter: self.min_filter,
                max_filter: self.max_filter,
            })
            .copied()
            .unwrap();

        // TODO: Framebuffer and Renderpass setup

        // TODO: Descriptor writes

        let span = tracing::span!(parent: &self.span, Level::DEBUG, "renderer_vulkan_frame").entered();

        Ok(VulkanFrame {
            renderer: self,
            _span: span,
        })
    }

    #[profiling::function]
    fn wait(&mut self, sync: &SyncPoint) -> Result<(), Self::Error> {
        if let Some(_fence) = sync.get::<VulkanFence>() {
            // The fence is indeed from a vulkan renderer, check if it is from ours.
            todo!()
        }

        // TODO: Check if the sync point contains a fence belonging to the same device.
        if let Some(_native) = self
            .capabilities
            .contains(Capabilities::IMPORT_FENCE)
            .then(|| sync.export())
            .flatten()
        {
            todo!("Import fence and try to wait")
        }

        // if everything above failed we can only
        // block until the sync point has been reached
        sync.wait();
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct VulkanImage {}

impl Texture for VulkanImage {
    fn width(&self) -> u32 {
        todo!()
    }

    fn height(&self) -> u32 {
        todo!()
    }

    fn format(&self) -> Option<Fourcc> {
        todo!()
    }
}

pub struct VulkanFrame<'frame> {
    renderer: &'frame mut VulkanRenderer,
    _span: EnteredSpan,
}

impl Frame for VulkanFrame<'_> {
    type Error = VulkanError;
    type TextureId = VulkanImage;

    fn id(&self) -> usize {
        todo!()
    }

    fn clear(&mut self, _color: [f32; 4], _at: &[Rectangle<i32, Physical>]) -> Result<(), Self::Error> {
        todo!()
    }

    fn draw_solid(
        &mut self,
        _dst: Rectangle<i32, Physical>,
        _damage: &[Rectangle<i32, Physical>],
        _color: [f32; 4],
    ) -> Result<(), Self::Error> {
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

    fn finish(self) -> Result<SyncPoint, Self::Error> {
        todo!()
    }
}

#[derive(Debug)]
pub struct VulkanFence {
    // TODO: Drop handling?
    fence: vk::Fence,
    device: Weak<ash::Device>,
}

impl VulkanFence {
    #[profiling::function]
    pub fn wait_with_timeout(&self, timeout: Option<Duration>) -> bool {
        let timeout = timeout.map(|t| t.as_nanos() as u64).unwrap_or(u64::MAX);

        if let Some(device) = self.device.upgrade() {
            let result = unsafe { device.wait_for_fences(&[self.fence], true, timeout) };
            return result != Err(vk::Result::TIMEOUT);
        }

        // Device no longer exists, assume the fence is signalled.
        true
    }
}

impl Fence for VulkanFence {
    #[profiling::function]
    fn is_signaled(&self) -> bool {
        let Some(device) = self.device.upgrade() else {
            // Let's assume the fence was signalled if the device no longer exists
            return true;
        };

        unsafe { device.get_fence_status(self.fence) }
            .ok()
            // If the device was lost assume the fence is dead and say it is signalled.
            .unwrap_or(true)
    }

    #[profiling::function]
    fn wait(&self) {
        let _ = self.wait_with_timeout(None);
    }

    fn is_exportable(&self) -> bool {
        false // TODO: Not yet
    }

    #[profiling::function]
    fn export(&self) -> Option<OwnedFd> {
        None // TODO: Not yet
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct SamplerKey {
    min_filter: TextureFilter,
    max_filter: TextureFilter,
}

/// # Safety
///
/// The caller must ensure the instance has enabled VK_KHR_external_fence_capabilities or the instance must
/// have been created using Vulkan 1.1.
unsafe fn get_sync_fd_fence_properties(
    entry: &ash::Entry,
    instance: vk::Instance,
    physical_device: vk::PhysicalDevice,
) -> vk::ExternalFenceProperties {
    // Loading a function can be quite expensive due to the loader creating a chain, but this function is only
    // called in initialization code.
    let fns = vk::KhrExternalFenceCapabilitiesFn::load(|name| unsafe {
        mem::transmute(entry.get_instance_proc_addr(instance, name.as_ptr()))
    });

    let info = vk::PhysicalDeviceExternalFenceInfo::builder()
        .handle_type(vk::ExternalFenceHandleTypeFlags::SYNC_FD)
        .build();
    let mut properties = vk::ExternalFenceProperties::default();

    unsafe {
        (fns.get_physical_device_external_fence_properties_khr)(physical_device, &info, &mut properties)
    };

    properties
}

fn find_extension(extensions: &[vk::ExtensionProperties], extension: &CStr) -> bool {
    extensions
        .iter()
        .any(|props| cstr_from_bytes_until_nul(&props.extension_name) == Some(extension))
}

/// Construct a `CStr` from a byte slice, up to the first zero byte.
///
/// Return a `CStr` extending from the start of `bytes` up to and
/// including the first zero byte. If there is no zero byte in
/// `bytes`, return `None`.
///
/// This can be removed when `CStr::from_bytes_until_nul` is stabilized.
/// ([#95027](https://github.com/rust-lang/rust/issues/95027))
fn cstr_from_bytes_until_nul(bytes: &[std::ffi::c_char]) -> Option<&std::ffi::CStr> {
    if bytes.contains(&0) {
        // Safety for `CStr::from_ptr`:
        // - We've ensured that the slice does contain a null terminator.
        // - The range is valid to read, because the slice covers it.
        // - The memory won't be changed, because the slice borrows it.
        unsafe { Some(CStr::from_ptr(bytes.as_ptr())) }
    } else {
        None
    }
}
