#![cfg(feature = "wayland_frontend")]

use wayland_server::protocol::{wl_buffer, wl_shm};

use crate::{
    backend::renderer::{ImportDmaWl, ImportMemWl},
    utils::{Buffer, Rectangle},
    wayland::compositor,
};

impl ImportMemWl for super::VulkanRenderer {
    fn import_shm_buffer(
        &mut self,
        _buffer: &wl_buffer::WlBuffer,
        _surface: Option<&compositor::SurfaceData>,
        _damage: &[Rectangle<i32, Buffer>],
    ) -> Result<Self::TextureId, Self::Error> {
        todo!()
    }

    fn shm_formats(&self) -> &[wl_shm::Format] {
        &self.formats.wl_shm
    }
}

impl ImportDmaWl for super::VulkanRenderer {}
