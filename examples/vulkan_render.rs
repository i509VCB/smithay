use smithay::backend::{
    renderer::{vulkan::VulkanRenderer, ImportMem},
    vulkan::{version::Version, Instance, PhysicalDevice},
};

fn main() {
    if let Ok(env_filter) = tracing_subscriber::EnvFilter::try_from_default_env() {
        tracing_subscriber::fmt().with_env_filter(env_filter).init();
    } else {
        tracing_subscriber::fmt().init();
    }

    let instance = Instance::new(Version::VERSION_1_1, None).unwrap();
    let device = PhysicalDevice::enumerate(&instance).unwrap().next().unwrap();
    let mut renderer = VulkanRenderer::new(&device).unwrap();

    let _image = renderer
        .import_memory(
            &[0x00, 0x00, 0x00, 0x00],
            drm_fourcc::DrmFourcc::Argb8888,
            (1, 1).into(),
            false,
        )
        .expect("Failed to create image");

    renderer.submit_staging_buffers().unwrap();
}
