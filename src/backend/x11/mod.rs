//! Implementation of the backend types using X11.
//!
//! This backend provides the appropriate backend implementations to run a Wayland compositor
//! directly as an X11 client.
//!

/*
A note from i509 for future maintainers and contributors:

Grab yourself the nearest copy of the ICCCM.

You should follow this document religiously, or else you will easily get shot in the foot.
Specifically look out for "Section 4: Client to Window Manager Communication"

A link to the ICCCM Section 4: https://tronche.com/gui/x/icccm/sec-4.html

Useful reading:

DRI3 protocol documentation: https://cgit.freedesktop.org/xorg/proto/dri3proto/tree/dri3proto.txt

TODO: Possible future changes:

- Migrate to x11rb's Wrapper types for Pixmaps and windows when the next version releases
*/

mod buffer;
pub mod connection;
mod drm;
mod event_source;
pub mod input;
pub mod window;

use self::connection::{ConnectToXError, XConnection};
use self::window::{Window, WindowInner};
use super::allocator::Allocator;
use super::allocator::dmabuf::Dmabuf;
use super::input::{Axis, ButtonState, KeyState, MouseButton};
use crate::backend::input::InputEvent;
use crate::backend::x11::drm::{DRM_NODE_RENDER, get_drm_node_type};
use crate::backend::x11::event_source::X11Source;
use crate::backend::x11::input::*;
use crate::utils::{Logical, Size};
use calloop::{EventSource, Poll, PostAction, Readiness, Token, TokenFactory};
use gbm::Device;
use nix::fcntl;
use slog::{error, info, o, Logger};
use std::io;
use std::os::unix::prelude::{AsRawFd, RawFd};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use x11rb::connection::{Connection, RequestConnection};
use x11rb::errors::{ConnectError, ConnectionError, ReplyError};
use x11rb::protocol::dri3::{self, ConnectionExt};
use x11rb::protocol::xproto::{ColormapAlloc, ConnectionExt as _, Depth, VisualClass};
use x11rb::rust_connection::ReplyOrIdError;
use x11rb::x11_utils::X11Error as ImplError;
use x11rb::{atom_manager, protocol as x11};

/// An error that may occur when initializing the backend.
#[derive(Debug, thiserror::Error)]
pub enum InitializationError {}

/// An error emitted by the X11 backend.
#[derive(Debug, thiserror::Error)]
pub enum X11Error {
    /// An error that may occur when initializing the backend.
    #[error("Error while initializing backend")]
    Connect(ConnectToXError),

    /// Connecting to the X server failed.
    #[error("Connecting to the X server failed")]
    ConnectionFailed(ConnectError),

    /// An X11 error packet was encountered.
    #[error("An X11 error packet was encountered.")]
    Protocol(ReplyOrIdError),

    /// The window was destroyed.
    #[error("The window was destroyed")]
    WindowDestroyed,
}

impl From<ConnectToXError> for X11Error {
    fn from(e: ConnectToXError) -> Self {
        X11Error::Connect(e)
    }
}

impl From<ConnectError> for X11Error {
    fn from(e: ConnectError) -> Self {
        X11Error::ConnectionFailed(e)
    }
}

impl From<ConnectionError> for X11Error {
    fn from(e: ConnectionError) -> Self {
        ReplyOrIdError::from(e).into()
    }
}

impl From<ImplError> for X11Error {
    fn from(e: ImplError) -> Self {
        ReplyOrIdError::from(e).into()
    }
}

impl From<ReplyError> for X11Error {
    fn from(e: ReplyError) -> Self {
        ReplyOrIdError::from(e).into()
    }
}

impl From<ReplyOrIdError> for X11Error {
    fn from(e: ReplyOrIdError) -> Self {
        X11Error::Protocol(e)
    }
}

/// Properties defining information about the window created by the X11 backend.
#[derive(Debug, Clone, Copy)]
#[allow(missing_docs)] // Self explanatory fields
pub struct WindowProperties<'a> {
    pub width: u16,
    pub height: u16,
    pub title: &'a str,
}

impl Default for WindowProperties<'_> {
    fn default() -> Self {
        WindowProperties {
            width: 1280,
            height: 800,
            title: "Smithay",
        }
    }
}

/// An event emitted by the X11 backend.
#[derive(Debug)]
pub enum X11Event {
    /// The X server has required the compositor to redraw the window.
    Refresh,

    /// An input event occurred.
    Input(InputEvent<X11Input>),

    /// The window was resized.
    Resized(Size<u16, Logical>),

    /// The window was requested to be closed.
    CloseRequested,
}

// TODO:
// Figure out how to expose this to the outside world after inserting in the event loop so buffers can be allocated?

/// An abstraction representing a connection to the X11 server.
#[derive(Debug)]
pub struct X11Backend {
    log: Logger,
    connection: Arc<XConnection>,
    source: X11Source,
    window: Arc<WindowInner>,
    key_counter: Arc<AtomicU32>,
    depth: Depth,
    visual_id: u32,
    gbm_device: Device<RawFd>,
}

atom_manager! {
    pub(crate) Atoms: AtomCollectionCookie {
        WM_PROTOCOLS,
        WM_DELETE_WINDOW,
        WM_CLASS,
        _NET_WM_NAME,
        UTF8_STRING,
        _SMITHAY_X11_BACKEND_CLOSE,
    }
}

impl X11Backend {
    /// Initializes the X11 backend, connecting to the X server and creating the window the compositor may output to.
    pub fn new<L>(properties: WindowProperties<'_>, logger: L) -> Result<X11Backend, X11Error>
    where
        L: Into<Option<slog::Logger>>,
    {
        let logger = crate::slog_or_fallback(logger).new(o!("smithay_module" => "backend_x11"));

        info!(logger, "Connecting to the X server");

        let (connection, screen_number) = XConnection::new(&logger)?;
        let connection = Arc::new(connection);
        info!(logger, "Connected to screen {}", screen_number);
        let xcb = connection.xcb_connection();

        if xcb.extension_information(dri3::X11_EXTENSION_NAME)?.is_none() {
            todo!("DRI3 is not present")
        }

        // Does the X server support dri3?
        let (dri3_major, dri3_minor) = {
            // DRI3 will only return the highest version we request.
            // TODO: We might need to request a higher version?
            let version = xcb.dri3_query_version(1, 2)?.reply()?;

            if version.minor_version < 2 {
                todo!("DRI3 version too low")
            }

            (version.major_version, version.minor_version)
        };

        let screen = &xcb.setup().roots[screen_number];

        // Now that we've initialized the connection to the X server, we need to determine which
        // drm-device the Display is using.
        let dri3 = xcb.dri3_open(screen.root, 0)?.reply()?;
        // This file descriptor points towards the DRM device that the X server is using.
        let drm_device_fd = dri3.device_fd;

        // Duplicate the drm_device_fd
        let drm_device_fd: RawFd = fcntl::fcntl(
            drm_device_fd.as_raw_fd(),
            fcntl::FcntlArg::F_DUPFD_CLOEXEC(3), // Set to 3 so the fd cannot become stdin, stdout or stderr
        ).expect("TODO");

        let fd_flags =
            nix::fcntl::fcntl(drm_device_fd.as_raw_fd(), nix::fcntl::F_GETFD).expect("Handle this error");
        // No need to check if ret == 1 since nix handles that.

        // Enable the close-on-exec flag.
        nix::fcntl::fcntl(
            drm_device_fd.as_raw_fd(),
            nix::fcntl::F_SETFD(
                nix::fcntl::FdFlag::from_bits_truncate(fd_flags) | nix::fcntl::FdFlag::FD_CLOEXEC,
            ),
        )
        .expect("Handle this result");

        if get_drm_node_type(drm_device_fd.as_raw_fd()).expect("TODO") != DRM_NODE_RENDER {
            todo!("Attempt to get the render device by name for the DRM node that isn't a render node")
        }

        // Finally create a GBMDevice to manage the buffers.
        let gbm_device = crate::backend::allocator::gbm::GbmDevice::new(drm_device_fd.as_raw_fd())
            .expect("Failed to create gbm device");

        // We want 32 bit color
        let depth = screen
            .allowed_depths
            .iter()
            .find(|depth| depth.depth == 32)
            .cloned()
            .expect("TODO");

        // Next find a visual using the supported depth
        let visual_id = depth
            .visuals
            .iter()
            .find(|visual| visual.class == VisualClass::TRUE_COLOR)
            .expect("TODO")
            .visual_id;

        // Find a supported format.
        // TODO

        // Make a colormap
        let colormap = xcb.generate_id()?;
        xcb.create_colormap(ColormapAlloc::NONE, colormap, screen.root, visual_id)?;

        let atoms = Atoms::new(xcb)?.reply()?;

        let window = Arc::new(WindowInner::new(
            connection.clone(),
            screen,
            properties,
            atoms,
            depth.clone(),
            visual_id,
            colormap,
        )?);

        let source = X11Source::new(
            connection.clone(),
            window.inner,
            atoms._SMITHAY_X11_BACKEND_CLOSE,
            logger.clone(),
        );

        info!(logger, "Window created");

        Ok(X11Backend {
            log: logger,
            source,
            connection,
            window,
            key_counter: Arc::new(AtomicU32::new(0)),
            depth,
            visual_id,
            gbm_device,
        })
    }

    /// Returns the underlying connection to the X server.
    pub fn connection(&self) -> &XConnection {
        &self.connection
    }

    /// Returns a handle to the X11 window this input backend handles inputs for.
    pub fn window(&self) -> Window {
        self.window.clone().into()
    }

    /// Returns a reference to the GBM device used to allocate buffers used to present to the window.
    pub fn gbm_device(&self) -> &Device<RawFd> {
        &self.gbm_device
    }
}

impl EventSource for X11Backend {
    type Event = X11Event;

    type Metadata = Window;

    type Ret = ();

    fn process_events<F>(
        &mut self,
        readiness: Readiness,
        token: Token,
        mut callback: F,
    ) -> std::io::Result<PostAction>
    where
        F: FnMut(Self::Event, &mut Self::Metadata) -> Self::Ret,
    {
        use self::X11Event::Input;

        let connection = self.connection.clone();
        let window = self.window.clone();
        let key_counter = self.key_counter.clone();
        let log = self.log.clone();
        let mut event_window = window.clone().into();

        self.source
            .process_events(readiness, token, |event, _| {
                match event {
                    x11::Event::ButtonPress(button_press) => {
                        if button_press.event == window.inner {
                            // X11 decided to associate scroll wheel with a button, 4, 5, 6 and 7 for
                            // up, down, right and left. For scrolling, a press event is emitted and a
                            // release is them immediately followed for scrolling. This means we can
                            // ignore release for scrolling.

                            // Ideally we would use `ButtonIndex` from XCB, however it does not cover 6 and 7
                            // for horizontal scroll and does not work nicely in match statements, so we
                            // use magic constants here:
                            //
                            // 1 => MouseButton::Left
                            // 2 => MouseButton::Middle
                            // 3 => MouseButton::Right
                            // 4 => Axis::Vertical +1.0
                            // 5 => Axis::Vertical -1.0
                            // 6 => Axis::Horizontal -1.0
                            // 7 => Axis::Horizontal +1.0
                            // Others => ??
                            match button_press.detail {
                                1..=3 => {
                                    // Clicking a button.
                                    callback(
                                        Input(InputEvent::PointerButton {
                                            event: X11MouseInputEvent {
                                                time: button_press.time,
                                                button: match button_press.detail {
                                                    1 => MouseButton::Left,

                                                    // Confusion: XCB docs for ButtonIndex and what plasma does don't match?
                                                    2 => MouseButton::Middle,

                                                    3 => MouseButton::Right,

                                                    _ => unreachable!(),
                                                },
                                                state: ButtonState::Pressed,
                                            },
                                        }),
                                        &mut event_window,
                                    )
                                }

                                4..=7 => {
                                    // Scrolling
                                    callback(
                                        Input(InputEvent::PointerAxis {
                                            event: X11MouseWheelEvent {
                                                time: button_press.time,
                                                axis: match button_press.detail {
                                                    // Up | Down
                                                    4 | 5 => Axis::Vertical,

                                                    // Right | Left
                                                    6 | 7 => Axis::Horizontal,

                                                    _ => unreachable!(),
                                                },
                                                amount: match button_press.detail {
                                                    // Up | Right
                                                    4 | 7 => 1.0,

                                                    // Down | Left
                                                    5 | 6 => -1.0,

                                                    _ => unreachable!(),
                                                },
                                            },
                                        }),
                                        &mut event_window,
                                    )
                                }

                                // Unknown mouse button
                                _ => callback(
                                    Input(InputEvent::PointerButton {
                                        event: X11MouseInputEvent {
                                            time: button_press.time,
                                            button: MouseButton::Other(button_press.detail),
                                            state: ButtonState::Pressed,
                                        },
                                    }),
                                    &mut event_window,
                                ),
                            }
                        }
                    }

                    x11::Event::ButtonRelease(button_release) => {
                        if button_release.event == window.inner {
                            match button_release.detail {
                                1..=3 => {
                                    // Releasing a button.
                                    callback(
                                        Input(InputEvent::PointerButton {
                                            event: X11MouseInputEvent {
                                                time: button_release.time,
                                                button: match button_release.detail {
                                                    1 => MouseButton::Left,

                                                    2 => MouseButton::Middle,

                                                    3 => MouseButton::Right,

                                                    _ => unreachable!(),
                                                },
                                                state: ButtonState::Released,
                                            },
                                        }),
                                        &mut event_window,
                                    )
                                }

                                // We may ignore the release tick for scrolling, as the X server will
                                // always emit this immediately after press.
                                4..=7 => (),

                                _ => callback(
                                    Input(InputEvent::PointerButton {
                                        event: X11MouseInputEvent {
                                            time: button_release.time,
                                            button: MouseButton::Other(button_release.detail),
                                            state: ButtonState::Released,
                                        },
                                    }),
                                    &mut event_window,
                                ),
                            }
                        }
                    }

                    x11::Event::KeyPress(key_press) => {
                        if key_press.event == window.inner {
                            callback(
                                Input(InputEvent::Keyboard {
                                    event: X11KeyboardInputEvent {
                                        time: key_press.time,
                                        // It seems as if X11's keycodes are +8 relative to the libinput
                                        // keycodes that are expected, so subtract 8 from each keycode
                                        // to match libinput.
                                        key: key_press.detail as u32 - 8,
                                        count: key_counter.fetch_add(1, Ordering::SeqCst) + 1,
                                        state: KeyState::Pressed,
                                    },
                                }),
                                &mut event_window,
                            )
                        }
                    }

                    x11::Event::KeyRelease(key_release) => {
                        if key_release.event == window.inner {
                            // atomic u32 has no checked_sub, so load and store to do the same.
                            let mut key_counter_val = key_counter.load(Ordering::SeqCst);
                            key_counter_val = key_counter_val.saturating_sub(1);
                            key_counter.store(key_counter_val, Ordering::SeqCst);

                            callback(
                                Input(InputEvent::Keyboard {
                                    event: X11KeyboardInputEvent {
                                        time: key_release.time,
                                        // It seems as if X11's keycodes are +8 relative to the libinput
                                        // keycodes that are expected, so subtract 8 from each keycode
                                        // to match libinput.
                                        key: key_release.detail as u32 - 8,
                                        count: key_counter_val,
                                        state: KeyState::Released,
                                    },
                                }),
                                &mut event_window,
                            );
                        }
                    }

                    x11::Event::MotionNotify(motion_notify) => {
                        if motion_notify.event == window.inner {
                            // Use event_x/y since those are relative the the window receiving events.
                            let x = motion_notify.event_x as f64;
                            let y = motion_notify.event_y as f64;

                            callback(
                                Input(InputEvent::PointerMotionAbsolute {
                                    event: X11MouseMovedEvent {
                                        time: motion_notify.time,
                                        x,
                                        y,
                                        size: window.size(),
                                    },
                                }),
                                &mut event_window,
                            )
                        }
                    }

                    x11::Event::ConfigureNotify(configure_notify) => {
                        if configure_notify.window == window.inner {
                            let previous_size = { *window.size.lock().unwrap() };

                            // Did the size of the window change?
                            let configure_notify_size: Size<u16, Logical> =
                                (configure_notify.width, configure_notify.height).into();

                            if configure_notify_size != previous_size {
                                // Intentionally drop the lock on the size mutex incase a user
                                // requests a resize or does something which causes a resize
                                // inside the callback.
                                {
                                    *window.size.lock().unwrap() = configure_notify_size;
                                }

                                (callback)(X11Event::Resized(configure_notify_size), &mut event_window);
                            }
                        }
                    }

                    x11::Event::ClientMessage(client_message) => {
                        if client_message.data.as_data32()[0] == window.atoms.WM_DELETE_WINDOW // Destroy the window?
                            && client_message.window == window.inner
                        // Same window
                        {
                            (callback)(X11Event::CloseRequested, &mut event_window);
                        }
                    }

                    x11::Event::Expose(expose) => {
                        // TODO: We would ideally use this to determine damage and render more efficiently that way.
                        //
                        // Although there is an Expose with damage event somewhere?
                        if expose.window == window.inner && expose.count == 0 {
                            (callback)(X11Event::Refresh, &mut event_window);
                        }
                    }

                    x11::Event::Error(e) => {
                        error!(log, "X11 error: {:?}", e);
                    }

                    _ => (),
                }

                // Flush the connection so changes to the window state during callbacks can be emitted.
                let _ = connection.xcb_connection().flush();
            })
            .expect("TODO");

        Ok(PostAction::Continue)
    }

    fn register(&mut self, poll: &mut Poll, token_factory: &mut TokenFactory) -> io::Result<()> {
        self.source.register(poll, token_factory)
    }

    fn reregister(&mut self, poll: &mut Poll, token_factory: &mut TokenFactory) -> io::Result<()> {
        self.source.reregister(poll, token_factory)
    }

    fn unregister(&mut self, poll: &mut Poll) -> io::Result<()> {
        self.source.unregister(poll)
    }
}
