//! Implementation of the backend types using X11.
//!
//! This backend provides the appropriate backend implementations to run a Wayland compositor
//! directly as an X11 client.
//!

/*
A note from i509 for future maintainers and contributors:

Grab yourself the nearest copy of the ICCCM.

You should follow this document religiously, or else you will easily get shot in the foot when doing anything window related.
Specifically look out for "Section 4: Client to Window Manager Communication"

A link to the ICCCM Section 4: https://tronche.com/gui/x/icccm/sec-4.html

Useful reading:

DRI3 protocol documentation: https://gitlab.freedesktop.org/xorg/proto/xorgproto/-/blob/master/dri3proto.txt
*/

mod buffer;
mod drm;
mod event_source;
pub mod surface;
pub mod input;
pub mod window;

use self::surface::X11Surface;
use self::window::{Window, WindowInner};
use super::input::{Axis, ButtonState, KeyState, MouseButton};
use crate::backend::input::InputEvent;
use crate::backend::x11::event_source::X11Source;
use crate::utils::{Logical, Size};
use calloop::{EventSource, Poll, PostAction, Readiness, Token, TokenFactory};
use slog::{error, info, o, Logger};
use std::io;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use x11rb::connection::Connection;
use x11rb::errors::{ConnectError, ConnectionError, ReplyError};
use x11rb::protocol::xproto::{ColormapAlloc, ConnectionExt as _, Depth, VisualClass};
use x11rb::rust_connection::{ReplyOrIdError, RustConnection};
use x11rb::x11_utils::X11Error as ImplError;
use x11rb::{atom_manager, protocol as x11};

pub use crate::backend::x11::input::*;

/// An error that may occur when initializing the backend.
#[derive(Debug, thiserror::Error)]
pub enum InitializationError {}

/// An error emitted by the X11 backend.
#[derive(Debug, thiserror::Error)]
pub enum X11Error {
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
    // TODO: Narrow access once the presentation is part of the window.
    pub(crate) connection: Arc<RustConnection>,
    source: X11Source,
    screen_number: usize,
    window: Arc<WindowInner>,
    key_counter: Arc<AtomicU32>,
    depth: Depth,
    visual_id: u32,
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
    pub fn new<L>(properties: WindowProperties<'_>, logger: L) -> Result<(X11Backend, X11Surface), X11Error>
    where
        L: Into<Option<slog::Logger>>,
    {
        let logger = crate::slog_or_fallback(logger).new(o!("smithay_module" => "backend_x11"));

        info!(logger, "Connecting to the X server");

        let (connection, screen_number) = RustConnection::connect(None)?;
        let connection = Arc::new(connection);
        info!(logger, "Connected to screen {}", screen_number);

        let screen = &connection.setup().roots[screen_number];

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
        let colormap = connection.generate_id()?;
        connection.create_colormap(ColormapAlloc::NONE, colormap, screen.root, visual_id)?;

        let atoms = Atoms::new(&*connection)?.reply()?;

        let window = Arc::new(WindowInner::new(
            Arc::downgrade(&connection),
            screen,
            properties,
            atoms,
            depth.clone(),
            visual_id,
            colormap,
        )?);

        let source = X11Source::new(
            connection.clone(),
            window.id,
            atoms._SMITHAY_X11_BACKEND_CLOSE,
            logger.clone(),
        );

        info!(logger, "Window created");

        let backend = X11Backend {
            log: logger,
            source,
            connection,
            window,
            key_counter: Arc::new(AtomicU32::new(0)),
            depth,
            visual_id,
            screen_number,
        };

        let surface = X11Surface::new(&backend)?;

        Ok((backend, surface))
    }

    /// Returns the default screen number of the X server.
    pub fn screen(&self) -> usize {
        self.screen_number
    }

    /// Returns the underlying connection to the X server.
    pub fn connection(&self) -> &RustConnection {
        &*self.connection
    }

    /// Returns a handle to the X11 window this input backend handles inputs for.
    pub fn window(&self) -> Window {
        self.window.clone().into()
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
                        if button_press.event == window.id {
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
                        if button_release.event == window.id {
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
                        if key_press.event == window.id {
                            callback(
                                Input(InputEvent::Keyboard {
                                    event: X11KeyboardInputEvent {
                                        time: key_press.time,
                                        // X11's keycodes are +8 relative to the libinput keycodes
                                        // that are expected, so subtract 8 from each keycode to
                                        // match libinput.
                                        //
                                        // https://github.com/freedesktop/xorg-xf86-input-libinput/blob/master/src/xf86libinput.c#L54
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
                        if key_release.event == window.id {
                            // atomic u32 has no checked_sub, so load and store to do the same.
                            let mut key_counter_val = key_counter.load(Ordering::SeqCst);
                            key_counter_val = key_counter_val.saturating_sub(1);
                            key_counter.store(key_counter_val, Ordering::SeqCst);

                            callback(
                                Input(InputEvent::Keyboard {
                                    event: X11KeyboardInputEvent {
                                        time: key_release.time,
                                        // X11's keycodes are +8 relative to the libinput keycodes
                                        // that are expected, so subtract 8 from each keycode to
                                        // match libinput.
                                        //
                                        // https://github.com/freedesktop/xorg-xf86-input-libinput/blob/master/src/xf86libinput.c#L54
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
                        if motion_notify.event == window.id {
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
                        if configure_notify.window == window.id {
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
                            && client_message.window == window.id
                        // Same window
                        {
                            (callback)(X11Event::CloseRequested, &mut event_window);
                        }
                    }

                    x11::Event::Expose(expose) => {
                        // TODO: Use details of expose event to tell the the compositor what has been damaged?
                        if expose.window == window.id && expose.count == 0 {
                            (callback)(X11Event::Refresh, &mut event_window);
                        }
                    }

                    x11::Event::Error(e) => {
                        error!(log, "X11 error: {:?}", e);
                    }

                    _ => (),
                }

                // Flush the connection so changes to the window state during callbacks can be emitted.
                let _ = connection.flush();
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
