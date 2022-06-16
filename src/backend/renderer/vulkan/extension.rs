//! Extension function loading.

use std::ffi::CStr;

use ash::{
    extensions::{
        ext::ImageDrmFormatModifier,
        khr::{ExternalMemoryFd, TimelineSemaphore},
    },
    vk,
};

use crate::backend::vulkan::{version::Version, PhysicalDevice};

use super::MaybePromoted;

impl super::ExtensionFns {
    /// # Safety
    ///
    /// - The device must support Vulkan 1.2 or support VK_KHR_timeline_semaphore
    pub(super) unsafe fn load(
        instance: &ash::Instance,
        phd: &PhysicalDevice,
        device: &ash::Device,
        enabled_extensions: &[&CStr],
    ) -> Self {
        let khr_timeline_semaphore = if phd.api_version() < Version::VERSION_1_2 {
            MaybePromoted::Extension(TimelineSemaphore::new(instance, device))
        } else {
            MaybePromoted::Promoted
        };

        let dmabuf_external_memory =
            unsafe { super::DmabufMemoryFns::load(instance, phd, device, enabled_extensions) };

        Self {
            khr_timeline_semaphore,
            dmabuf_external_memory,
        }
    }
}

impl super::DmabufMemoryFns {
    unsafe fn load(
        instance: &ash::Instance,
        phd: &PhysicalDevice,
        device: &ash::Device,
        enabled_extensions: &[&CStr],
    ) -> Option<Self> {
        const REQUIRED_EXTENSIONS: &[&CStr] = &[
            vk::ExtImageDrmFormatModifierFn::name(),
            vk::ExtExternalMemoryDmaBufFn::name(),
            vk::KhrExternalMemoryFdFn::name(),
        ];

        // If the device uses Vulkan 1.1, then VK_KHR_image_format_list must be supported.
        let supports_image_format_list = phd.api_version() >= Version::VERSION_1_2
            || enabled_extensions.contains(&vk::KhrImageFormatListFn::name());

        if REQUIRED_EXTENSIONS
            .iter()
            .all(|&ext| enabled_extensions.iter().any(|&name| name == ext))
            && supports_image_format_list
        {
            Some(Self {
                ext_image_drm_format_modifier: ImageDrmFormatModifier::new(instance, device),
                ext_external_memory_dmabuf: ExternalMemoryFd::new(instance, device),
            })
        } else {
            None
        }
    }
}
