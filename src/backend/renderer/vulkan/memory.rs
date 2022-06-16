use drm_fourcc::DrmFourcc;

use crate::{
    backend::renderer::ImportMem,
    utils::{Buffer as BufferCoord, Rectangle, Size},
};

use super::Error;

impl ImportMem for super::VulkanRenderer {
    fn import_memory(
        &mut self,
        data: &[u8],
        size: Size<i32, BufferCoord>,
        flipped: bool,
    ) -> Result<Self::TextureId, Self::Error> {
        // Validate the data is the correct size

        // Negative size means the data size is wrong.
        if size.w.is_negative() || size.h.is_negative() {
            return Err(Error::UnsupportedFormat);
        }

        let width = size.w as usize;
        let height = size.h as usize;

        // Guard against overflows for size calculation.
        let min_size = width
            .checked_mul(height)
            .and_then(|s| s.checked_mul(4))
            .ok_or(Error::UnsupportedFormat)
            // TODO: Replace with inspect_err when stabilized
            .map_err(|err| {
                slog::error!(self.logger, "expected size of memory import is too large");
                err
            })?;

        // Test the data is large enough
        if data.len() < min_size {
            slog::error!(self.logger, "memory buffer to import is too small");
            return Err(Error::UnsupportedFormat);
        }

        // Truncate the data to upload if it is too large.
        let data = &data[..min_size];
        let texture = self.create_mem_texture(DrmFourcc::Argb8888, size, flipped)?;

        // initialize the texture memory
        self.update_memory(&texture, data, Rectangle::from_loc_and_size((0, 0), size))?;
        Ok(texture)
    }

    fn update_memory(
        &mut self,
        _texture: &Self::TextureId,
        _data: &[u8],
        _region: Rectangle<i32, BufferCoord>,
    ) -> Result<(), Self::Error> {
        // TODO

        // Create a staging buffer for upload.
        // Create device buffer as target for upload
        // Begin recording commands
        // Transfer from host buffer to device buffer
        // Copy from device buffer to texture.

        Ok(())
    }
}

// impl ExportMem for super::VulkanRenderer {
//     type TextureMapping;

//     fn copy_framebuffer(
//         &mut self,
//         region: Rectangle<i32, BufferCoord>,
//     ) -> Result<Self::TextureMapping, Self::Error> {
//         todo!()
//     }

//     fn copy_texture(
//         &mut self,
//         texture: &Self::TextureId,
//         region: Rectangle<i32, BufferCoord>,
//     ) -> Result<Self::TextureMapping, Self::Error> {
//         todo!()
//     }

//     fn map_texture<'a>(
//         &mut self,
//         texture_mapping: &'a Self::TextureMapping,
//     ) -> Result<&'a [u8], Self::Error> {
//         todo!()
//     }
// }

impl super::VulkanRenderer {}
