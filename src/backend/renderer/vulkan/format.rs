use std::collections::{hash_map::Entry, HashMap};

use ash::{prelude::VkResult, vk};
use drm_fourcc::{DrmFormat, DrmModifier};

use crate::backend::{
    allocator::vulkan::format::{get_vk_format, known_formats},
    renderer::vulkan::{ShmFormat, DMA_TEXTURE_USAGE, SHM_TEXTURE_USAGE},
    vulkan::version::Version,
};

use super::{DrmFormatInfo, FormatModifierProperties, ShmFormatProperties, SHM_TEXTURE_DOWNLOAD_USAGE};

impl super::VulkanRenderer {
    /// Initializes the supported formats for the renderer.
    ///
    /// This should only be invoked once.
    pub(super) fn init_formats(&mut self) -> VkResult<()> {
        // Test every known fourcc format
        for &fourcc in known_formats() {
            if let Some(vk_format) = get_vk_format(fourcc) {
                // It's not clear whether unknown/added in extension formats are strictly forbidden in the spec:
                //
                // Looking at the VU for vkGetPhysicalDeviceFormatProperties:
                // > VUID-vkGetPhysicalDeviceFormatProperties-format-parameter
                // >
                // > format must be a valid VkFormat value
                //
                // This would seem to agree with sentiment that an ICD would just return an entirely zeroed out
                // structure to us?
                //
                // However the section on valid usage rules in the specification conflicts with the above VU:
                // > Physical-device-level functionality or behavior added by a device extension to the API
                // > must not be used unless the conditions described in Extending Physical Device Core
                // > Functionality are met.
                //
                // This issue has been reported to Khronos but is not fixable without a new core version or a
                // maintenance extension (which itself may take time to propagate across the ecosystem, and of
                // course abandoned drivers will always exhibit these bugs).
                //
                // GitHub issue: https://github.com/KhronosGroup/Vulkan-Docs/issues/1730

                // Smithay assumes the worst and not allow formats added in extensions we don't enable.
                if vk_format == vk::Format::A4R4G4B4_UNORM_PACK16
                    || vk_format == vk::Format::B4G4R4A4_UNORM_PACK16
                {
                    // VK_EXT_4444_formats is promoted in Vulkan 1,3
                    if !self.enabled_extensions().contains(&vk::Ext4444FormatsFn::name())
                        || self.phd.api_version() < Version::VERSION_1_3
                    {
                        continue;
                    }
                }

                // Compute what usages are supported for shm formats.
                if let Some(texture) =
                    unsafe { self.get_shm_format_properties(vk_format, SHM_TEXTURE_USAGE) }?
                {
                    // If the required usages are supported, then test for download usages
                    let download =
                        unsafe { self.get_shm_format_properties(vk_format, SHM_TEXTURE_DOWNLOAD_USAGE) }?;

                    self.formats
                        .shm_formats
                        .insert(vk_format, ShmFormat { texture, download });

                    #[cfg(feature = "wayland_frontend")]
                    {
                        let wl_shm = match fourcc {
                            drm_fourcc::DrmFourcc::Argb8888 => {
                                Some(wayland_server::protocol::wl_shm::Format::Argb8888)
                            }
                            drm_fourcc::DrmFourcc::Xrgb8888 => {
                                Some(wayland_server::protocol::wl_shm::Format::Xrgb8888)
                            }
                            fourcc => wayland_server::protocol::wl_shm::Format::try_from(fourcc as u32).ok(),
                        };

                        if let Some(wl_shm) = wl_shm {
                            self.formats.wl_shm.push(wl_shm);
                        }
                    }
                }

                // Compute usages for dmabuf external memory
                if self.extensions.dmabuf_external_memory.is_some() {
                    // Get all supported modifiers for the format.
                    let format_properties = vk::FormatProperties::default();
                    let supported_drm_modifiers =
                        unsafe { self.get_dma_format_modifier_properties_list(vk_format, format_properties) };

                    if !supported_drm_modifiers.is_empty() {
                        self.formats.drm_modifiers.insert(
                            vk_format,
                            supported_drm_modifiers
                                .iter()
                                .map(|props| props.drm_format_modifier)
                                .collect(),
                        );
                    }

                    for drm_modifier_properties in supported_drm_modifiers {
                        let modifier = DrmModifier::from(drm_modifier_properties.drm_format_modifier);
                        let format = DrmFormat {
                            code: fourcc,
                            modifier,
                        };

                        if let Some(dma_texture_properties) =
                            unsafe { self.get_drm_format_properties(vk_format, modifier, DMA_TEXTURE_USAGE) }?
                        {
                            // Properties were returned, add the format to the importable list.
                            self.formats.drm_importable.push(format);

                            // Hash map insert mess...
                            let mut entry = self.formats.drm_properties.entry(vk_format);
                            let info = match entry {
                                Entry::Occupied(ref mut entry) => entry.get_mut(),
                                Entry::Vacant(entry) => entry.insert(DrmFormatInfo {
                                    texture: HashMap::new(),
                                }),
                            };

                            info.texture.insert(
                                modifier,
                                FormatModifierProperties {
                                    image_properties: dma_texture_properties,
                                    modifier_properties: drm_modifier_properties,
                                },
                            );
                        }
                    }
                }
            }
        }

        Ok(())
    }

    unsafe fn get_shm_format_properties(
        &self,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
    ) -> VkResult<Option<ShmFormatProperties>> {
        let format_info = vk::PhysicalDeviceImageFormatInfo2::builder()
            .format(format)
            .ty(vk::ImageType::TYPE_2D)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .flags(vk::ImageCreateFlags::empty());
        let mut props = vk::ImageFormatProperties2::builder();

        let result = unsafe {
            self.physical_device()
                .instance()
                .handle()
                .get_physical_device_image_format_properties2(
                    self.physical_device().handle(),
                    &format_info,
                    &mut props,
                )
        };

        if let Err(vk::Result::ERROR_FORMAT_NOT_SUPPORTED) = result {
            return Ok(None);
        }

        result?;

        let format_properties = unsafe {
            self.physical_device()
                .instance()
                .handle()
                .get_physical_device_format_properties(self.physical_device().handle(), format)
        };

        Ok(Some(ShmFormatProperties {
            format: format_properties,
            image: props.image_format_properties,
        }))
    }

    unsafe fn get_dma_format_modifier_properties_list(
        &self,
        format: vk::Format,
        format_properties: vk::FormatProperties,
    ) -> Vec<vk::DrmFormatModifierPropertiesEXT> {
        let mut list = vk::DrmFormatModifierPropertiesListEXT::default();

        // Get the number of entries
        unsafe {
            let mut format_properties2 = vk::FormatProperties2::builder()
                .format_properties(format_properties)
                .push_next(&mut list);

            self.phd
                .instance()
                .handle()
                .get_physical_device_format_properties2(self.phd.handle(), format, &mut format_properties2)
        };

        let mut data = Vec::with_capacity(list.drm_format_modifier_count as usize);
        list.p_drm_format_modifier_properties = data.as_mut_ptr();

        // Read the properties into the vector
        unsafe {
            let mut format_properties2 = vk::FormatProperties2::builder()
                .format_properties(format_properties)
                .push_next(&mut list);

            self.phd
                .instance()
                .handle()
                .get_physical_device_format_properties2(self.phd.handle(), format, &mut format_properties2);

            // SAFETY: Vulkan just initialized the elements of the vector.
            data.set_len(list.drm_format_modifier_count as usize);
        }

        data
    }

    unsafe fn get_drm_format_properties(
        &self,
        format: vk::Format,
        modifier: DrmModifier,
        usage: vk::ImageUsageFlags,
    ) -> VkResult<Option<vk::ImageFormatProperties>> {
        let mut image_drm_format_info = vk::PhysicalDeviceImageDrmFormatModifierInfoEXT::builder()
            .drm_format_modifier(modifier.into())
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let format_info = vk::PhysicalDeviceImageFormatInfo2::builder()
            .format(format)
            .ty(vk::ImageType::TYPE_2D)
            .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
            .usage(usage)
            .flags(vk::ImageCreateFlags::empty())
            // VUID-VkPhysicalDeviceImageFormatInfo2-tiling-02249
            .push_next(&mut image_drm_format_info);
        let mut props = vk::ImageFormatProperties2::builder();

        let result = unsafe {
            self.physical_device()
                .instance()
                .handle()
                // VUID-vkGetPhysicalDeviceImageFormatProperties-tiling-02248: Must use vkGetPhysicalDeviceImageFormatProperties2
                .get_physical_device_image_format_properties2(
                    self.physical_device().handle(),
                    &format_info,
                    &mut props,
                )
        };

        if let Err(vk::Result::ERROR_FORMAT_NOT_SUPPORTED) = result {
            return Ok(None);
        }

        result?;

        Ok(Some(props.image_format_properties))
    }
}
