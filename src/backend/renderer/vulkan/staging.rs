use super::{Staging, VulkanError, VulkanRenderer};

impl VulkanRenderer {
    pub(super) fn init_staging(&mut self) -> Result<(), VulkanError> {
        if self.staging.is_some() {
            return Ok(());
        }

        let command_buffer = self
            .command_buffers
            .pop_front()
            .expect("TODO: Handle error/allow creating more buffrs, all buffers were consumed");

        let staging = Staging {
            command_buffer,
            uploads: Vec::new(),
        };
        self.staging = Some(staging);

        Ok(())
    }
}
