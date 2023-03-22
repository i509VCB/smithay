use ash::vk;
use drm_fourcc::DrmFourcc;

use crate::{
    backend::allocator::vulkan,
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
        flipped: bool,
    ) -> Result<VulkanImage, VulkanError> {
        self.common_image_validation(format, size)?;

        let vk_format = vulkan::format::get_vk_format(format).expect("Already validated");

        let format_info = vk::PhysicalDeviceImageFormatInfo2::builder()
            .format(vk_format)
            .tiling(vk::ImageTiling::OPTIMAL)
            .ty(vk::ImageType::TYPE_2D)
            // TODO: Could loosen the usage requirements
            .usage(
                vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::TRANSFER_DST
                    | vk::ImageUsageFlags::SAMPLED,
            );

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

        todo!()
    }
}
