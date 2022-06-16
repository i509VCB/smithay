use drm_fourcc::DrmFormat;

use crate::{
    backend::{
        allocator::dmabuf::Dmabuf,
        renderer::{ExportDma, ImportDma},
    },
    utils::{Buffer as BufferCoord, Rectangle, Size},
};

impl ImportDma for super::VulkanRenderer {
    fn dmabuf_formats<'a>(&'a self) -> Box<dyn Iterator<Item = &'a DrmFormat> + 'a> {
        Box::new(self.formats.drm_importable.iter())
    }

    fn import_dmabuf(
        &mut self,
        _dmabuf: &Dmabuf,
        _damage: Option<&[Rectangle<i32, BufferCoord>]>,
    ) -> Result<Self::TextureId, Self::Error> {
        todo!()
    }
}

impl ExportDma for super::VulkanRenderer {
    fn export_framebuffer(&mut self, _size: Size<i32, BufferCoord>) -> Result<Dmabuf, Self::Error> {
        todo!()
    }

    fn export_texture(&mut self, _texture: &Self::TextureId) -> Result<Dmabuf, Self::Error> {
        todo!()
    }
}
