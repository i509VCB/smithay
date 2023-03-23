#![deny(unsafe_op_in_unsafe_fn)]
#![allow(missing_docs)]

mod image;

use std::{
    array,
    collections::HashMap,
    fmt,
    mem::ManuallyDrop,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

use ash::{extensions::ext, vk};
use drm_fourcc::DrmFourcc;
use gpu_allocator::{
    vulkan::{Allocation, Allocator, AllocatorCreateDesc},
    AllocatorDebugSettings,
};

use crate::{
    backend::vulkan::{Instance, PhysicalDevice},
    utils::{Buffer, Physical, Rectangle, Size, Transform},
};

use super::{DebugFlags, ExportMem, Frame, ImportMem, Renderer, Texture, TextureFilter, TextureMapping};

pub struct VulkanRenderer {
    images: HashMap<u64, ImageInfo>,

    next_image_id: u64,

    limits: Limits,

    /// The memory allocator.
    ///
    /// This is wrapped in a [`Box`] to reduce the size of the [`VulkanRenderer`] on the stack.
    ///
    /// This is wrapped in a [`ManuallyDrop`]  
    allocator: ManuallyDrop<Box<Allocator>>,

    debug_utils: Option<ext::DebugUtils>,

    queue: vk::Queue,

    instance: Instance,

    /// Raw handle to the physical device.
    physical_device: vk::PhysicalDevice,

    // The device is placed in an Arc since it quite large.
    device: Arc<ash::Device>,
}

impl VulkanRenderer {
    pub fn new(device: &PhysicalDevice) -> Result<Self, VulkanError> {
        let physical_device = device.handle();
        let instance_ = device.instance();
        let instance = device.instance().handle();
        let limits = device.limits();

        let limits = Limits {
            max_framebuffer_width: limits.max_framebuffer_width,
            max_framebuffer_height: limits.max_framebuffer_height,
        };

        // Select a queue
        let queue_families = unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        let (queue_family_index, _) = queue_families
            .iter()
            .enumerate()
            .find(|(_, properties)| properties.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .expect("Handle this error");

        let queue_create_info = vk::DeviceQueueCreateInfo::builder()
            .queue_family_index(queue_family_index as u32)
            .queue_priorities(&[1.0])
            .build();

        let create_info =
            vk::DeviceCreateInfo::builder().queue_create_infos(array::from_ref(&queue_create_info));
        // SAFETY: TODO
        let device =
            unsafe { instance.create_device(physical_device, &create_info, None) }.expect("Handle error");
        // SAFETY:
        // - VUID-vkGetDeviceQueue-queueFamilyIndex-00384: Queue family index was specified when device was created.
        // - VUID-vkGetDeviceQueue-queueIndex-00385: Only one queue was created, so index 0 is valid.
        let queue = unsafe { device.get_device_queue(queue_family_index as u32, 0) };

        let desc = AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device,
            // TODO: Allow configuring debug settings
            debug_settings: AllocatorDebugSettings::default(),
            buffer_device_address: false,
        };
        let allocator = ManuallyDrop::new(Box::new(Allocator::new(&desc).expect("Handle error")));

        let renderer = Self {
            images: HashMap::new(),
            next_image_id: 0,
            limits,
            allocator,
            // TODO: Initialize debug utils if available
            debug_utils: None,
            queue,
            instance: instance_.clone(),
            physical_device,
            device: Arc::new(device),
        };

        Ok(renderer)
    }
}

impl Renderer for VulkanRenderer {
    type Error = VulkanError;
    type TextureId = VulkanImage;
    type Frame<'frame> = VulkanFrame<'frame>;

    fn id(&self) -> usize {
        todo!("Smithay needs a global renderer id counter")
    }

    fn downscale_filter(&mut self, filter: TextureFilter) -> Result<(), Self::Error> {
        todo!()
    }

    fn upscale_filter(&mut self, filter: TextureFilter) -> Result<(), Self::Error> {
        todo!()
    }

    fn set_debug_flags(&mut self, flags: DebugFlags) {
        todo!()
    }

    fn debug_flags(&self) -> DebugFlags {
        todo!()
    }

    fn render(
        &mut self,
        output_size: Size<i32, Physical>,
        dst_transform: Transform,
    ) -> Result<Self::Frame<'_>, Self::Error> {
        todo!()
    }
}

impl ImportMem for VulkanRenderer {
    fn import_memory(
        &mut self,
        data: &[u8],
        format: DrmFourcc,
        size: Size<i32, Buffer>,
        flipped: bool,
    ) -> Result<Self::TextureId, Self::Error> {
        let texture = self.create_mem_image(format, size, flipped)?;
        // The image contents are empty, so initialize the image content.
        self.update_memory(&texture, data, Rectangle::from_loc_and_size((0, 0), size))?;
        Ok(texture)
    }

    fn update_memory(
        &mut self,
        texture: &Self::TextureId,
        data: &[u8],
        region: Rectangle<i32, Buffer>,
    ) -> Result<(), Self::Error> {
        // TODO: Validate size of buffer.
        // TODO: Actually do this
        Ok(())
    }

    fn mem_formats(&self) -> Box<dyn Iterator<Item = DrmFourcc>> {
        todo!()
    }
}

impl ExportMem for VulkanRenderer {
    type TextureMapping = VulkanTextureMapping;

    fn copy_framebuffer(
        &mut self,
        region: Rectangle<i32, Buffer>,
    ) -> Result<Self::TextureMapping, Self::Error> {
        todo!()
    }

    fn copy_texture(
        &mut self,
        texture: &Self::TextureId,
        region: Rectangle<i32, Buffer>,
    ) -> Result<Self::TextureMapping, Self::Error> {
        todo!()
    }

    fn map_texture<'a>(
        &mut self,
        texture_mapping: &'a Self::TextureMapping,
    ) -> Result<&'a [u8], Self::Error> {
        todo!()
    }
}

pub struct VulkanTextureMapping {}

impl Texture for VulkanTextureMapping {
    fn width(&self) -> u32 {
        todo!()
    }

    fn height(&self) -> u32 {
        todo!()
    }

    fn format(&self) -> Option<DrmFourcc> {
        todo!()
    }
}

impl TextureMapping for VulkanTextureMapping {
    fn flipped(&self) -> bool {
        todo!()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum VulkanError {}

#[derive(Debug)]
pub struct VulkanImage {
    id: u64,
    refcount: Arc<AtomicUsize>,
    width: u32,
    height: u32,
    vk_format: vk::Format,
    drm_format: Option<DrmFourcc>,
}

impl Texture for VulkanImage {
    fn width(&self) -> u32 {
        self.width
    }

    fn height(&self) -> u32 {
        self.height
    }

    fn format(&self) -> Option<DrmFourcc> {
        self.drm_format
    }
}

impl VulkanImage {
    pub fn vk_format(&self) -> vk::Format {
        self.vk_format
    }
}

impl Clone for VulkanImage {
    fn clone(&self) -> Self {
        self.refcount.fetch_add(1, Ordering::AcqRel);

        Self {
            id: self.id,
            refcount: self.refcount.clone(),
            width: self.width,
            height: self.height,
            vk_format: self.vk_format,
            drm_format: self.drm_format,
        }
    }
}

impl Drop for VulkanImage {
    fn drop(&mut self) {
        self.refcount.fetch_sub(1, Ordering::AcqRel);
    }
}

pub struct VulkanFrame<'frame> {
    _marker: std::marker::PhantomData<&'frame ()>,
}

impl<'frame> Frame for VulkanFrame<'frame> {
    type Error = <VulkanRenderer as Renderer>::Error;
    type TextureId = <VulkanRenderer as Renderer>::TextureId;

    fn id(&self) -> usize {
        todo!()
    }

    fn clear(&mut self, color: [f32; 4], at: &[Rectangle<i32, Physical>]) -> Result<(), Self::Error> {
        todo!()
    }

    fn render_texture_from_to(
        &mut self,
        texture: &Self::TextureId,
        src: Rectangle<f64, Buffer>,
        dst: Rectangle<i32, Physical>,
        damage: &[Rectangle<i32, Physical>],
        src_transform: Transform,
        alpha: f32,
    ) -> Result<(), Self::Error> {
        todo!()
    }

    fn transformation(&self) -> Transform {
        todo!()
    }

    fn finish(self) -> Result<(), Self::Error> {
        todo!()
    }
}

impl Drop for VulkanRenderer {
    fn drop(&mut self) {
        // SAFETY: The render is being dropped, meaning `true` is allowed.
        unsafe { self.cleanup_images(true) };

        // SAFETY:
        // The allocator needs to be dropped before the device is destroyed
        unsafe { ManuallyDrop::drop(&mut self.allocator) };

        // SAFETY:
        // - VUID-vkDestroyDevice-device-00378: All child objects were destroyed above
        // - Since Drop requires &mut, destruction of the device is externally synchronized
        //   by Rust's type system since only one reference to the device exists.
        unsafe {
            // TODO: For guest renderer check if the renderer owns the device.
            self.device.destroy_device(None);
        }
    }
}

/// Internal limits used by the renderer
///
/// This is used instead of keeping a copy of [`vk::PhysicalDeviceLimits`] in the renderer because
/// that type is quite large in memory.
#[derive(Debug)]
struct Limits {
    /// [`vk::PhysicalDeviceLimits::max_framebuffer_width`]
    max_framebuffer_width: u32,
    /// [`vk::PhysicalDeviceLimits::max_framebuffer_height`]
    max_framebuffer_height: u32,
}

struct ImageInfo {
    /// The internal id of the image.
    id: u64,

    /// The id of the renderer, used to ensure an image isn't used with the wrong renderer.
    renderer_id: usize,

    /// Number of references to the underlying image resource.
    ///
    /// The refcount is increased to ensure the underlying image resource is not freed while VulkanTexture
    /// handles exist or the texture is used in a command buffer.
    refcount: Arc<AtomicUsize>,

    /// The underlying image resource.
    image: vk::Image,

    /// The underlying memory of the image.
    ///
    /// This will be [`None`] if the renderer does not own the image.
    // TODO: This may be multiple instances of device memory for imported textures.
    underlying_memory: Option<Allocation>,
}

impl fmt::Debug for ImageInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ImageInfo")
            .field("id", &self.id)
            .field("renderer_id", &self.renderer_id)
            .field("refcount", &self.refcount.load(Ordering::Relaxed))
            .field("image", &self.image)
            .field("allocation", &self.underlying_memory)
            .finish()
    }
}
