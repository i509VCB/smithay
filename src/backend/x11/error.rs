use std::io;

use nix::errno::Errno;
use x11rb::rust_connection::{ConnectError, ConnectionError, ReplyError, ReplyOrIdError};

use crate::backend::allocator::gbm::GbmConvertError;

/// An error emitted by the X11 backend during setup.
#[derive(Debug, thiserror::Error)]
pub enum X11Error {
    /// Connecting to the X server failed.
    #[error("Connecting to the X server failed")]
    ConnectionFailed(ConnectError),

    /// A required X11 extension was not present or has the right version.
    #[error("{0}")]
    MissingExtension(MissingExtensionError),

    /// Some protocol error occured during setup.
    #[error("Some protocol error occured during setup")]
    ProtocolError(ReplyOrIdError),

    /// Creating the window failed.
    #[error("Creating the window failed")]
    CreateWindow(CreateWindowError),
    /// Failed to allocate buffers needed to present to the window.
    #[error("Failed to allocate buffers needed to present to the window")]
    Allocation(AllocateBuffersError),
}

impl From<ConnectError> for X11Error {
    fn from(err: ConnectError) -> Self {
        Self::ConnectionFailed(err)
    }
}

impl From<ReplyError> for X11Error {
    fn from(err: ReplyError) -> Self {
        Self::ProtocolError(err.into())
    }
}

impl From<ConnectionError> for X11Error {
    fn from(err: ConnectionError) -> Self {
        Self::ProtocolError(err.into())
    }
}

impl From<ReplyOrIdError> for X11Error {
    fn from(err: ReplyOrIdError) -> Self {
        Self::ProtocolError(err)
    }
}

/// An error that occurs when a required X11 extension is not present.
#[derive(Debug, thiserror::Error)]
pub enum MissingExtensionError {
    /// An extension was not found.
    #[error("Extension \"{name}\" version {major}.{minor} was not found.")]
    NotFound {
        /// The name of the required extension.
        name: &'static str,
        /// The minimum required major version of extension.
        major: u32,
        /// The minimum required minor version of extension.
        minor: u32,
    },

    /// An extension was present, but the version is too low.
    #[error("Extension \"{name}\" version {required_major}.{required_minor} is required but only version {available_major}.{available_minor} is available.")]
    WrongVersion {
        /// The name of the extension.
        name: &'static str,
        /// The minimum required major version of extension.
        required_major: u32,
        /// The minimum required minor version of extension.
        required_minor: u32,
        /// The major version of the extension available on the X server.
        available_major: u32,
        /// The minor version of the extension available on the X server.
        available_minor: u32,
    },
}

impl From<MissingExtensionError> for X11Error {
    fn from(err: MissingExtensionError) -> Self {
        Self::MissingExtension(err)
    }
}

/// An error which may occur when creating an X11 window.
#[derive(Debug, thiserror::Error)]
pub enum CreateWindowError {
    /// No depth fulfilling the pixel format requirements was found.
    #[error("No depth fulfilling the requirements was found")]
    NoDepth,

    /// No visual fulfilling the pixel format requirements was found.
    #[error("No visual fulfilling the requirements was found")]
    NoVisual,
}

impl From<CreateWindowError> for X11Error {
    fn from(err: CreateWindowError) -> Self {
        Self::CreateWindow(err)
    }
}

/// An error which may occur when allocating buffers for presentation to the window.
#[derive(Debug, thiserror::Error)]
pub enum AllocateBuffersError {
    /// Failed to open the DRM device to allocate buffers.
    #[error("Failed to open the DRM device to allocate buffers.")]
    OpenDevice(io::Error),

    /// The device used to allocate buffers is not the correct drm node type.
    #[error("The device used to allocate buffers is not the correct drm node type.")]
    UnsupportedDrmNode,

    /// Exporting a dmabuf failed.
    #[error("Exporting a dmabuf failed.")]
    ExportDmabuf(GbmConvertError),
}

impl From<Errno> for AllocateBuffersError {
    fn from(err: Errno) -> Self {
        Self::OpenDevice(err.into())
    }
}

impl From<io::Error> for AllocateBuffersError {
    fn from(err: io::Error) -> Self {
        Self::OpenDevice(err)
    }
}

impl From<GbmConvertError> for AllocateBuffersError {
    fn from(err: GbmConvertError) -> Self {
        Self::ExportDmabuf(err)
    }
}

impl From<AllocateBuffersError> for X11Error {
    fn from(err: AllocateBuffersError) -> Self {
        Self::Allocation(err)
    }
}
