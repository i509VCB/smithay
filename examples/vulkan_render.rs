use smithay::backend::{
    renderer::vulkan::VulkanRenderer,
    vulkan::{version::Version, Instance, PhysicalDevice},
};

fn main() {
    let instance = Instance::new(Version::VERSION_1_1, None).unwrap();
    let device = PhysicalDevice::enumerate(&instance).unwrap().next().unwrap();
    let mut renderer = VulkanRenderer::new(&device).unwrap();
}
