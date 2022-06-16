use ash::{
    prelude::VkResult,
    vk::{self, Extent3D},
};
use drm_fourcc::{DrmFormat, DrmFourcc};
use gpu_alloc::UsageFlags;

use crate::{
    backend::{
        allocator::{format::has_alpha, vulkan::format::get_vk_format},
        renderer::Texture,
    },
    utils::{Buffer as BufferCoord, Size},
};

use super::{
    liveness::Liveness, Error, Image, ImageData, VulkanTexture, SHM_TEXTURE_DOWNLOAD_USAGE, SHM_TEXTURE_USAGE,
};

impl Texture for super::VulkanTexture {
    fn width(&self) -> u32 {
        self.data.width
    }

    fn height(&self) -> u32 {
        self.data.height
    }
}

impl super::VulkanImageExt for super::VulkanTexture {
    fn vk_format(&self) -> vk::Format {
        self.data.vk_format
    }

    fn drm_format(&self) -> Option<DrmFormat> {
        self.data.drm_format
    }

    fn memory(&self) -> vk::DeviceMemory {
        self.data.memory
    }

    fn image(&self) -> vk::Image {
        self.data.image
    }

    fn image_view(&self) -> vk::ImageView {
        self.data.image_view
    }

    fn usage(&self) -> vk::ImageUsageFlags {
        self.data.usage
    }

    fn exportable(&self) -> bool {
        todo!("VulkanTextureExt::exportable")
    }

    fn is_imported(&self) -> bool {
        self.data.imported_queue_family.is_some()
    }

    fn imported_queue_family(&self) -> Option<u32> {
        self.data.imported_queue_family
    }
}

impl super::VulkanRenderer {
    /// # Safety
    ///
    /// - The caller must ensure the `usage` is supported for the specified `format`.
    pub(super) unsafe fn validate_image(
        &self,
        format: vk::Format,
        width: u32,
        height: u32,
        usage: vk::ImageUsageFlags,
        image_properties: vk::ImageFormatProperties,
    ) -> Result<(), Error> {
        // Texture must be at least 1x1
        if width == 0 || height == 0 {
            slog::error!(self.logger, "Failed to create image: size must be at least 1x1");
            return Err(Error::UnsupportedFormat);
        }

        // Ensure the image is not too large
        let max_extent = image_properties.max_extent;

        // VUID-VkImageCreateInfo-extent-02252
        // VUID-VkImageCreateInfo-extent-02253
        if max_extent.width < width || max_extent.height < height {
            slog::error!(
                self.logger,
                "Failed to create image: too large (max size: {}x{})",
                max_extent.width,
                max_extent.height
                ; "format" => ?format, "size" => format!("{}x{}", width, height)
            );
            return Err(Error::UnsupportedFormat);
        }

        // VUID-VkImageCreateInfo-extent-02254 since our images are 2D
        //
        // Although it may be superfluous, guard against drivers returning an invalid value
        // (VUID-VkImageCreateInfo-imageType-00957). This may occur if the driver is given an unknown format
        // (the format properties could then just be zeroed out) and we somehow got here.
        if max_extent.depth != 1 {
            slog::error!(
                self.logger,
                "Driver reported unexpected max depth for image of format {:?}",
                format ; "format" => ?format
            );
            return Err(Error::UnsupportedFormat);
        }

        /* Usage */

        // VUID-VkImageCreateInfo-usage-00964
        // VUID-VkImageCreateInfo-usage-00965
        //
        // Images with color attachment usage must be limited by
        // VkPhysicalDeviceLimits::maxFramebufferWidth/maxFramebufferHeight
        if usage.contains(vk::ImageUsageFlags::COLOR_ATTACHMENT) {
            let limits = self.physical_device().limits();

            if limits.max_framebuffer_width < width || limits.max_framebuffer_height < height {
                slog::error!(
                    self.logger,
                    "Failed to create image: exceeds maximum framebuffer size ({}x{})",
                    limits.max_framebuffer_width, limits.max_framebuffer_height
                    ; "format" => ?format, "size" => format!("{}x{}", width, height)
                );
                return Err(Error::UnsupportedFormat);
            }
        }

        Ok(())
    }

    /// Creates an image handle with an optimal image tiling.
    ///
    /// The returned image handle is never exportable.
    ///
    /// # Safety
    ///
    /// - The caller must ensure the `width`, `height`, `format` and `usage` are valid for the format.
    /// - The caller must also uphold any valid usage requirements.
    pub(super) unsafe fn create_image_optimal(
        &self,
        width: u32,
        height: u32,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
    ) -> VkResult<vk::Image> {
        let create_info = vk::ImageCreateInfo::builder()
            .flags(vk::ImageCreateFlags::empty())
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            // tiling is added later
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .tiling(vk::ImageTiling::OPTIMAL);

        // SAFETY: The caller upholds any valid usage requirements.
        unsafe { self.device.create_image(&create_info, None) }
    }

    /// Creates an image handle with a DRM modifier.
    ///
    /// # Safety
    ///
    /// - The caller must ensure dmabuf external memory is supported.
    /// - The caller must ensure the `width`, `height`, `format` and `usage` are valid for the format.
    /// - The device must support the equivalent DRM format with the specified `modifiers`.
    /// - The caller must also uphold any valid usage requirements.
    pub(super) unsafe fn create_drm_image(
        &self,
        width: u32,
        height: u32,
        vk_format: vk::Format,
        modifiers: &[u64],
        usage: vk::ImageUsageFlags,
    ) -> VkResult<vk::Image> {
        assert!(
            self.extensions.dmabuf_external_memory.is_some(),
            "create_image_drm is not supported without dmabuf external memory"
        );

        // Tell the driver what DRM format modifiers are acceptable to use.
        let mut drm_modifier_list =
            vk::ImageDrmFormatModifierListCreateInfoEXT::builder().drm_format_modifiers(modifiers);

        // Tell the driver we want to create an image that can be exported as a dmabuf.
        let mut external_memory_image = vk::ExternalMemoryImageCreateInfo::builder()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);

        let create_info = vk::ImageCreateInfo::builder()
            .flags(vk::ImageCreateFlags::empty())
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk_format)
            .extent(Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            // tiling is added later
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
            // TODO: VUID required by tiling
            .push_next(&mut drm_modifier_list)
            .push_next(&mut external_memory_image);

        // SAFETY: The caller upholds any valid usage requirements.
        unsafe { self.device.create_image(&create_info, None) }
    }

    /// Returns the valid modifiers of an image that may be exportable with the given format, modifiers and
    /// usage.
    ///
    /// Returns [`None`] if the image is not exportable, either because the format is not supported or the
    /// implementation does not support the required extensions.
    pub(super) unsafe fn get_drm_image_modifiers(
        &self,
        width: u32,
        height: u32,
        format: vk::Format,
        modifiers: &[u64],
        feature_flags: vk::FormatFeatureFlags,
        _usage: vk::ImageUsageFlags,
    ) -> Option<Vec<u64>> {
        // 1. Are the required extensions supported.
        let _ = self.extensions.dmabuf_external_memory.as_ref()?;

        // 2. Some valid modifiers exist for the format
        let properties = self
            .formats
            .drm_properties
            .get(&format)
            // VUID-VkImageDrmFormatModifierListCreateInfoEXT-drmFormatModifierCount-arraylength
            .filter(|info| !info.texture.is_empty())?;

        // TODO: Select usage type to lookup for
        let modifiers = properties
            .texture
            .iter()
            // 3. Filter out modifiers we do not want to want.
            .filter(|(modifier, _)| modifiers.contains(&u64::from(**modifier)))
            // 4. Filter out any modifier listings that do not support the required FormatFeatureFlags
            .filter(|(_, props)| {
                props
                    .modifier_properties
                    .drm_format_modifier_tiling_features
                    .contains(feature_flags)
            })
            // 5. Filter out any modifiers that do not support the required extents
            .filter(|(_, props)| {
                let max_extent = props.image_properties.max_extent;
                max_extent.depth == 1 && max_extent.width >= width && max_extent.height >= height
            })
            .map(|(modifier, _)| modifier)
            .copied()
            .map(u64::from)
            .collect::<Vec<_>>();

        Some(modifiers).filter(Vec::is_empty)
    }

    /// Creates an texture which has contents uploaded from memory.
    pub(super) fn create_mem_texture(
        &mut self,
        format: DrmFourcc,
        size: Size<i32, BufferCoord>,
        // TODO: Handle flipped
        _flipped: bool,
    ) -> Result<VulkanTexture, Error> {
        // Ensure size is not negative
        if size.w.is_negative() || size.h.is_negative() {
            slog::error!(self.logger, "cannot create texture with negative size");
            return Err(Error::UnsupportedFormat);
        }

        let vk_format = get_vk_format(format)
            .ok_or(Error::UnsupportedFormat)
            // TODO: Replace with inspect_err when stabilized
            .map_err(|err| {
                slog::error!(
                    self.logger,
                    "cannot create texture: no Vulkan equivalent for {}",
                    format ; "format" => %format
                );
                err
            })?;

        let shm_image_properties = self
            .formats
            .shm_formats
            .get(&vk_format)
            .ok_or(Error::UnsupportedFormat)
            // TODO: Replace with inspect_err when stabilized
            .map_err(|err| {
                slog::error!(
                    self.logger,
                    "cannot create texture: unsupported format {}",
                    format ; "format" => %format
                );
                err
            })?;

        // Pick the download usage if available, falling back to plain texture usage.
        let usage = shm_image_properties
            .download
            .is_some()
            .then_some(SHM_TEXTURE_DOWNLOAD_USAGE)
            .unwrap_or(SHM_TEXTURE_USAGE);

        // Warn if the download usage is not supported.
        if usage == SHM_TEXTURE_USAGE {
            slog::trace!(
                self.logger,
                "{} shm format does not support texture download",
                format ; "format" => %format
            );
        }

        // Validate the image parameters
        unsafe {
            self.validate_image(
                vk_format,
                size.w as u32,
                size.h as u32,
                usage,
                shm_image_properties
                    .download
                    .as_ref()
                    .map(|props| props.image)
                    .unwrap_or(shm_image_properties.texture.image),
            )
        }?;

        let image = unsafe { self.create_image_optimal(size.w as u32, size.h as u32, vk_format, usage) }
            .map_err(|err| {
                slog::error!(self.logger, "failed to create image handle" ; "format" => %format);
                err
            })?;

        // TODO: Move code into another function
        // TODO: Try to create an exportable image
        let mut image_data = ImageData {
            width: size.w as u32,
            height: size.h as u32,
            memory: vk::DeviceMemory::null(),
            image,
            image_view: vk::ImageView::null(),
            usage,
            vk_format,
            drm_format: None,
            imported_queue_family: None,
        };

        let image_mem_requirements = unsafe { self.device.get_image_memory_requirements(image_data.image) };
        let request = gpu_alloc::Request {
            size: image_mem_requirements.size,
            align_mask: image_mem_requirements.alignment - 1,
            usage: UsageFlags::empty(),
            memory_types: image_mem_requirements.memory_type_bits,
        };

        let block = match unsafe { self.alloc(request) } {
            Ok(memory) => memory,
            Err(err) => {
                slog::error!(self.logger, "Failed to allocate image memory");

                // Failed to allocate, destroy the image handle
                unsafe {
                    self.device.destroy_image(image_data.image, None);
                }

                return Err(err);
            }
        };

        image_data.memory = *block.memory();

        // Bind memory to image
        if let Err(err) = unsafe {
            self.device
                .bind_image_memory(image_data.image, image_data.memory, block.offset())
        } {
            slog::error!(self.logger, "Failed to bind image to memory");

            // Failed to bind, destroy the image handle and free the memory
            unsafe {
                self.device.destroy_image(image_data.image, None);
                self.dealloc(block);
            }

            return Err(err.into());
        }

        // Alpha swizzle
        let a = has_alpha(format)
            // If there is an alpha channel, make sure the alpha component is swizzled to zero
            .then_some(vk::ComponentSwizzle::ZERO)
            // Otherwise the format should use identity.
            .unwrap_or(vk::ComponentSwizzle::IDENTITY);

        let components = vk::ComponentMapping {
            r: vk::ComponentSwizzle::IDENTITY,
            g: vk::ComponentSwizzle::IDENTITY,
            b: vk::ComponentSwizzle::IDENTITY,
            a,
        };

        let subresource_range = vk::ImageSubresourceRange::builder()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1)
            .build();

        let image_view_info = vk::ImageViewCreateInfo::builder()
            .image(image_data.image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk_format)
            .components(components)
            .subresource_range(subresource_range);

        image_data.image_view = match unsafe { self.device.create_image_view(&image_view_info, None) } {
            Ok(view) => view,
            Err(err) => {
                slog::error!(self.logger, "Failed to create image view");

                // Failed to create image view, destroy the image handle and free the memory
                unsafe {
                    self.device.destroy_image(image_data.image, None);
                    self.dealloc(block);
                }

                return Err(err.into());
            }
        };

        // Image creation was successful
        let (liveness, alive) = Liveness::new();

        self.images.push(Image {
            liveness,
            data: image_data,
            memory: Some(block),
        });

        let texture = VulkanTexture {
            data: image_data,
            _alive: alive,
        };

        Ok(texture)
    }
}
