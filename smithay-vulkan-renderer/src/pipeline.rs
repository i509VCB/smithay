use ash::vk;
use smithay::backend::renderer::TextureFilter;

const fn texture_filter_to_vk(filter: TextureFilter) -> vk::Filter {
    match filter {
        TextureFilter::Linear => vk::Filter::LINEAR,
        TextureFilter::Nearest => vk::Filter::NEAREST,
    }
}

fn create_sampler(
    device: &ash::Device,
    min_filter: TextureFilter,
    max_filter: TextureFilter,
) -> Result<vk::Sampler, vk::Result> {
    let create_info = vk::SamplerCreateInfo::builder()
        .flags(vk::SamplerCreateFlags::empty())
        .min_filter(texture_filter_to_vk(min_filter))
        .mag_filter(texture_filter_to_vk(max_filter))
        .anisotropy_enable(false)
        .border_color(vk::BorderColor::FLOAT_OPAQUE_BLACK)
        // Quoting from the specification:
        //
        // VUID-VkSamplerCreateInfo-unnormalizedCoordinates-01072:
        // > If unnormalizedCoordinates is VK_TRUE, minFilter and magFilter must be equal
        .unnormalized_coordinates(false);

    unsafe { device.create_sampler(&create_info, None) }
}

fn create_solid_pipeline_layout(device: &ash::Device) -> Result<vk::PipelineLayout, ()> {
    let set_layouts = [];

    let create_info = vk::PipelineLayoutCreateInfo::builder().set_layouts(&set_layouts);

    let layout = unsafe { device.create_pipeline_layout(&create_info, None) }.expect("Error");
    Ok(layout)
}

fn create_solid_pipeline(device: &ash::Device, layout: vk::PipelineLayout) -> Result<vk::Pipeline, ()> {
    let create_info = vk::GraphicsPipelineCreateInfo::builder();
    todo!()
}
