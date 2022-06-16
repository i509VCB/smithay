use std::{
    collections::{HashMap, VecDeque},
    ffi::CStr,
    fmt,
};

use ash::{prelude::VkResult, vk};
use gpu_alloc::GpuAllocator;

use crate::{
    backend::{
        renderer::{Renderer, TextureFilter},
        vulkan::{version::Version, PhysicalDevice},
    },
    utils::{Physical, Size, Transform},
};

use super::{
    alloc::AshMemoryDevice, Error, ExtensionFns, Formats, MaybePromoted, PhysicalDeviceFeatures,
    RendererCreateInfo, VulkanRenderer,
};

impl super::VulkanRenderer {}

impl Renderer for super::VulkanRenderer {
    type Error = super::Error;
    type TextureId = super::VulkanTexture;
    type Frame = super::VulkanFrame;

    fn id(&self) -> usize {
        todo!("id counter")
    }

    fn downscale_filter(&mut self, _filter: TextureFilter) -> Result<(), Self::Error> {
        todo!("not implemented yet")
    }

    fn upscale_filter(&mut self, _filter: TextureFilter) -> Result<(), Self::Error> {
        todo!("not implemented yet")
    }

    fn render<F, R>(
        &mut self,
        _size: Size<i32, Physical>,
        _dst_transform: Transform,
        _rendering: F,
    ) -> Result<R, Self::Error>
    where
        F: FnOnce(&mut Self, &mut Self::Frame) -> R,
    {
        // Handle resource destruction before command submission.
        self.cleanup()?;

        todo!()
    }
}

impl super::VulkanRenderer {
    pub(super) unsafe fn new_inner(phd: &PhysicalDevice, logger: slog::Logger) -> Result<Self, Error> {
        let instance = phd.instance().handle();

        let device_extensions = Self::optimal_device_extensions(phd)?;

        // Select the appropriate queue
        let queue_properties = unsafe { instance.get_physical_device_queue_family_properties(phd.handle()) };
        let queue_family_index = queue_properties
            .iter()
            .position(|properties| properties.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .ok_or(Error::NoQueue)? as u32;

        // Ensure the required features are supported
        let mut features = PhysicalDeviceFeatures::from_extensions_and_features(phd, &device_extensions);

        if !features.is_supported() {
            todo!()
        }

        /* Physical device features */

        // VUID-VkDeviceCreateInfo-ppEnabledLayerNames-parameter: Must allocate Vec to provide a valid pointer.
        let extensions_ptr = device_extensions
            .iter()
            .map(|name| name.as_ptr())
            .collect::<Vec<_>>();

        let queue_create_info = [vk::DeviceQueueCreateInfo::builder()
            .queue_family_index(queue_family_index)
            // VUID-VkDeviceQueueCreateInfo-pQueuePriorities-00383
            .queue_priorities(&[1.0])
            .build()];

        // Add the features to the builder
        let create_info = features.add_to_builder(
            vk::DeviceCreateInfo::builder()
                .queue_create_infos(&queue_create_info)
                // VUID-vkCreateDevice-ppEnabledExtensionNames-01387: satisfied using required_device_extensions
                .enabled_extension_names(&extensions_ptr),
        );

        unsafe {
            let device = instance.create_device(phd.handle(), &create_info, None)?;

            let create_info = RendererCreateInfo {
                queue_family_index,
                queue_index: 0, // Only one queue is created
                phd_limits: phd.limits(),
                max_memory_size_allocation_size: phd.properties_maintenance_3().max_memory_allocation_size,
            };

            Self::from_raw_device(device, phd, &device_extensions, create_info, logger)
        }
    }

    pub(super) unsafe fn from_raw_device_inner(
        device: ash::Device,
        phd: &PhysicalDevice,
        enabled_extensions: &[&'static CStr],
        create_info: RendererCreateInfo,
        logger: slog::Logger,
    ) -> Result<Self, Error> {
        let logger = logger.new(slog::o!("smithay_module" => "vulkan_renderer"));

        slog::info!(logger, "Initializing Vulkan renderer");
        slog::info!(logger, "Device Name: {}", phd.name());

        if let Some(driver) = phd.driver() {
            slog::info!(logger, "Device Driver: {}, {}", driver.name, driver.info);
        } else {
            slog::warn!(logger, "Vulkan implementation does not provide driver info");
        }

        slog::info!(logger, "Enabled device extensions: {:?}", enabled_extensions);

        // SAFETY: Caller has guaranteed the queue exists.
        let queue =
            unsafe { device.get_device_queue(create_info.queue_family_index, create_info.queue_index) };

        // Load extension functions
        let extensions =
            unsafe { ExtensionFns::load(phd.instance().handle(), phd, &device, enabled_extensions) };

        let known_memory_flags = vk::MemoryPropertyFlags::DEVICE_LOCAL
            | vk::MemoryPropertyFlags::HOST_VISIBLE
            | vk::MemoryPropertyFlags::HOST_COHERENT
            | vk::MemoryPropertyFlags::HOST_CACHED
            | vk::MemoryPropertyFlags::LAZILY_ALLOCATED;

        let mem_properties = unsafe {
            phd.instance()
                .handle()
                .get_physical_device_memory_properties(phd.handle())
        };
        let memory_types = &mem_properties.memory_types[..mem_properties.memory_type_count as usize];
        let valid_memory_types = memory_types.iter().enumerate().fold(0, |u, (i, mem)| {
            if known_memory_flags.contains(mem.property_flags) {
                u | (1 << i)
            } else {
                u
            }
        });

        let mem_allocator = {
            // TODO: Some middle ground on a scale of `i_am_potato` to `i_am_prototyping`
            let config = gpu_alloc::Config::i_am_prototyping();
            let properties = gpu_alloc::DeviceProperties {
                memory_types: memory_types
                    .iter()
                    .map(|memory_type| gpu_alloc::MemoryType {
                        props: gpu_alloc::MemoryPropertyFlags::from_bits_truncate(
                            memory_type.property_flags.as_raw() as u8,
                        ),
                        heap: memory_type.heap_index,
                    })
                    .collect(),
                memory_heaps: mem_properties.memory_heaps[..mem_properties.memory_heap_count as usize]
                    .iter()
                    .map(|&memory_heap| gpu_alloc::MemoryHeap {
                        size: memory_heap.size,
                    })
                    .collect(),
                max_memory_allocation_count: create_info.phd_limits.max_memory_allocation_count,
                max_memory_allocation_size: create_info.max_memory_size_allocation_size,
                non_coherent_atom_size: create_info.phd_limits.non_coherent_atom_size,
                // TODO: Validate this is correct with the requirements that gpu-alloc-ash sets.
                buffer_device_address: false,
            };

            GpuAllocator::new(config, properties)
        };

        let mut renderer = VulkanRenderer {
            formats: Formats {
                shm_formats: HashMap::new(),
                drm_importable: Vec::new(),
                drm_modifiers: HashMap::new(),
                drm_properties: HashMap::new(),
                #[cfg(feature = "wayland_frontend")]
                wl_shm: Vec::new(),
            },
            valid_memory_types,
            mem_allocator,
            images: Vec::new(),
            staging_buffers: Vec::new(),
            submit_semaphore: vk::Semaphore::null(),
            submit_count: 0,
            command_submissions: VecDeque::new(),
            queue_family_index: create_info.queue_family_index,
            queue,
            command_pool: vk::CommandPool::null(),
            extensions,
            enabled_extensions: enabled_extensions.into(),
            device,
            phd: phd.clone(),
            logger,
        };

        // Command pool
        let command_pool_info =
            vk::CommandPoolCreateInfo::builder().queue_family_index(renderer.queue_family_index);
        renderer.command_pool = unsafe { renderer.device.create_command_pool(&command_pool_info, None) }?;

        // Construct the timeline semaphore used to track command completion.
        let mut semaphore_type_info = vk::SemaphoreTypeCreateInfo::builder()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        let semaphore_info = vk::SemaphoreCreateInfo::builder().push_next(&mut semaphore_type_info);
        renderer.submit_semaphore = unsafe { renderer.device.create_semaphore(&semaphore_info, None) }?;

        renderer.init_formats()?;

        Ok(renderer)
    }

    /// <https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/vkGetSemaphoreCounterValue.html>
    pub(super) unsafe fn get_semaphore_counter_value(&self, semaphore: vk::Semaphore) -> VkResult<u64> {
        // SAFETY: Vulkan 1.2+ or VK_KHR_timeline_semaphore is enabled.
        unsafe {
            match self.extensions.khr_timeline_semaphore {
                MaybePromoted::Extension(ref ext) => ext.get_semaphore_counter_value(semaphore),
                MaybePromoted::Promoted => self.device.get_semaphore_counter_value(semaphore),
            }
        }
    }

    /// <https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/vkWaitSemaphores.html>
    pub(super) unsafe fn wait_semaphores(
        &self,
        wait_info: &vk::SemaphoreWaitInfo,
        timeout: u64,
    ) -> VkResult<()> {
        // SAFETY: Vulkan 1.2+ or VK_KHR_timeline_semaphore is enabled.
        unsafe {
            match self.extensions.khr_timeline_semaphore {
                MaybePromoted::Extension(ref ext) => ext.wait_semaphores(wait_info, timeout),
                MaybePromoted::Promoted => self.device.wait_semaphores(wait_info, timeout),
            }
        }
    }

    /// Cleanup resources from previous command submissions.
    pub(super) fn cleanup(&mut self) -> Result<(), Error> {
        slog::trace!(self.logger, "Cleaning up resources");

        // 1. Acquire the submit timeline semaphore counter value. The counter value is used to know which
        //    command submissions have completed.
        let value = unsafe { self.get_semaphore_counter_value(self.submit_semaphore) }?;

        // 2. Drop all Alive handles associated with the command submissions that have completed. If the
        //    counter value is greater than the current semaphore payload, then the command is still executing
        //    (and therefore retain returns true to keep the Alive handles around.).
        self.command_submissions
            .retain(|submission| value < submission.counter_value);

        // 3. Drop all alive handles that correspond to a Dmabufs that are gone. We try to upgrade every
        //    WeakDmabuf that the renderer has cloned when importing a dmabuf and remove any entries that fail
        //    to upgrade (since the actual strong Dmabuf was dropped).
        // TODO

        // 4. Retain all resources which are still alive. If a resource has been destroyed, it's Liveness
        //    handle will state the object has been dropped and therefore we can safely destroy the resource.
        //    This step will also destroy any dropped Texture handles that were not used in any commands.

        // First destroy dead texture resources.
        self.images.retain_mut(|image| {
            // Check if the image resource is dead.
            let dropped = image.liveness.is_dropped();

            if dropped {
                let data = image.data;
                unsafe {
                    // Destroy in reverse order since construction implies DeviceMemory -> Image -> ImageView.
                    // TODO: What are the VUIDs of these three statements?
                    self.device.destroy_image_view(data.image_view, None);
                    self.device.destroy_image(data.image, None);

                    self.mem_allocator
                        .dealloc(AshMemoryDevice::wrap(&self.device), image.memory.take().unwrap());
                }
            }

            !dropped
        });

        // Next destroy staging buffers that have completed.
        self.staging_buffers.retain_mut(|staging_buffer| {
            let dropped = staging_buffer.liveness.is_dropped();

            if dropped {
                unsafe {
                    // Destroy in reverse order since construction implies DeviceMemory -> Buffer.
                    self.device.destroy_buffer(staging_buffer.buffer, None);

                    self.mem_allocator.dealloc(
                        AshMemoryDevice::wrap(&self.device),
                        staging_buffer.block.take().unwrap(),
                    );
                }
            }

            !dropped
        });

        Ok(())
    }
}

impl fmt::Debug for super::VulkanRenderer {
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        unimplemented!("Debug not implemented yet")
    }
}

impl Drop for super::VulkanRenderer {
    fn drop(&mut self) {
        unsafe {
            // Wait until all commands have finished on the gpu.
            let _ = self.wait();

            self.device.destroy_command_pool(self.command_pool, None);

            // Now that commands have completed execution, destroy the semaphore
            self.device.destroy_semaphore(self.submit_semaphore, None);

            // Destroy all texture and staging buffer resources
            for mut texture in self.images.drain(..) {
                // Destroy in reverse order since construction implies DeviceMemory -> Image -> ImageView.
                // TODO: What are the VUIDs of these three statements?
                self.device.destroy_image_view(texture.data.image_view, None);
                self.device.destroy_image(texture.data.image, None);

                self.mem_allocator.dealloc(
                    AshMemoryDevice::wrap(&self.device),
                    texture.memory.take().unwrap(),
                );
            }

            for mut staging_buffer in self.staging_buffers.drain(..) {
                // Destroy in reverse order since construction implies DeviceMemory -> Buffer.
                self.device.destroy_buffer(staging_buffer.buffer, None);

                self.mem_allocator.dealloc(
                    AshMemoryDevice::wrap(&self.device),
                    staging_buffer.block.take().unwrap(),
                );
            }

            // Clean up leftover memory objects in the allocator
            self.mem_allocator.cleanup(AshMemoryDevice::wrap(&self.device));

            // After the device is destroyed, the drop implementation of PhysicalDevice may result in the Instance
            // being destroyed if the PhysicalDevice holds the final reference to the Instance.
            self.device.destroy_device(None);
        }
    }
}

impl super::PhysicalDeviceFeatures {
    pub fn from_extensions_and_features(phd: &PhysicalDevice, enabled_extensions: &[&'static CStr]) -> Self {
        let mut features = Self::default();

        let mut vk_features = vk::PhysicalDeviceFeatures2::builder()
            .features(vk::PhysicalDeviceFeatures::default())
            .push_next(&mut features.timeline_semaphore);

        // VK_EXT_4444_formats promoted in Vulkan 1.3.
        if phd.api_version() <= Version::VERSION_1_2
            && enabled_extensions.contains(&vk::Ext4444FormatsFn::name())
        {
            vk_features = vk_features.push_next(
                features
                    .formats_4444
                    .insert(vk::PhysicalDevice4444FormatsFeaturesEXT::default()),
            );
        }

        // TODO: get_features on PhysicalDevice
        unsafe {
            phd.instance()
                .handle()
                .get_physical_device_features2(phd.handle(), &mut vk_features)
        };

        features.core = vk_features.features;
        features
    }

    /// Whether the device properties support the minimum requirements.
    pub fn is_supported(&self) -> bool {
        // Timeline semaphores
        self.timeline_semaphore.timeline_semaphore == vk::TRUE
    }

    pub fn add_to_builder<'a>(
        &'a mut self,
        mut info: vk::DeviceCreateInfoBuilder<'a>,
    ) -> vk::DeviceCreateInfoBuilder<'_> {
        info = info
            .enabled_features(&self.core)
            .push_next(&mut self.timeline_semaphore);

        info
    }
}
