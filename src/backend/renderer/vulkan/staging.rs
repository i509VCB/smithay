use ash::vk;

use super::{Staging, VulkanError, VulkanRenderer};

// TODO: Separate upload queue

impl VulkanRenderer {
    pub(super) fn init_staging(&mut self) -> Result<(), VulkanError> {
        if self.staging.is_some() {
            return Ok(());
        }

        let command_buffer = self
            .command_buffers
            .pop_front()
            .expect("TODO: Handle error/allow creating more buffers, all buffers were consumed");

        // Begin recording the command buffer.
        let begin_info = vk::CommandBufferBeginInfo::builder();

        unsafe {
            self.device
                .begin_command_buffer(command_buffer, &begin_info)
                .expect("Handle error");
        }

        let staging = Staging {
            command_buffer,
            uploads: Vec::new(),
        };
        self.staging = Some(staging);

        Ok(())
    }
}
