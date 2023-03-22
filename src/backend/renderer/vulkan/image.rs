use std::sync::{atomic::AtomicUsize, Arc};

use ash::vk;
use drm_fourcc::DrmFourcc;

use crate::{
    backend::{allocator::vulkan, renderer::vulkan::ImageInfo},
    utils::{Buffer, Size},
};

use super::{VulkanError, VulkanImage, VulkanRenderer};

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

        let image = unsafe { self.device.create_image(&create_info, None) }.expect("Handle error");

        // TODO: Bind memory to the image

        let id = self.next_image_id;
        self.next_image_id += 1;

        let info = self.images.entry(id).or_insert(ImageInfo {
            id,
            renderer_id: 0, // TODO
            // Initialize with a refcount of 1 since a new image handle is being created.
            refcount: Arc::new(AtomicUsize::new(1)),
            image,
            underlying_memory: None,
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
}
