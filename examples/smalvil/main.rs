//! Smalvil, the small Wayland compositor.
//!
//! Smalvil is a functional starting point from which larger compositors may be built. Smalvil prioritizes the
//! following:
//! - Small and easy to reason with.
//! - Implements the core Wayland protocols and the XDG shell, meaning Smalvil works with the vast majority of
//! clients.
//! - Follows recommended usage patterns for Smithay.
//!
//! Smalvil being an example compositor does not prioritize the following:
//! - Aesthetics
//! - Configuration
//! - Plentiful features
//!
//! Smalvil is only designed to be run inside an existing session (inside a window) for the sake of
//! simplicity.

use smithay::{
    delegate_compositor, delegate_xdg_shell,
    wayland::{
        compositor::{CompositorHandler, CompositorState},
        shell::xdg::{XdgRequest, XdgShellHandler, XdgShellState}, seat::{SeatState, KeyboardHandle, PointerHandle}, shm::ShmState, output::OutputManagerState,
    },
};
use wayland_server::{protocol::wl_surface, Display, DisplayHandle};

/// The main function.
///
/// TODO: High level overview.
fn main() {
    // Let's prepare for the compositor setup by creating the necessary output backends.
    //
    // In the smalvil example, we use the winit for the windowing and input backend.

    // TODO: winit setup

    // The first step of the compositor setup is creating the display.
    //
    // The display handles queuing and dispatch of events from clients.
    let mut display = Display::<Smalvil>::new().expect("failed to create display");

    // Next the globals for the core wayland protocol and the xdg shell are created. In particular, this
    // creates instances of "delegate type"s which are responsible for processing some group of Wayland
    // protocols.
    //
    // Smalvil needs to initialize delegate types to handle the core Wayland protocols and the XDG Shell
    // protocol.

    let protocols = ProtocolStates {
        // Delegate type for the compositor
        compositor_state: CompositorState::new(&mut display, None),
        // Delegate type for the xdg shell.
        //
        // The xdg shell is the primary windowing shell used in the Wayland ecosystem.
        xdg_shell_state: XdgShellState::new(&mut display, None).0, // TODO: Make GlobalId a member of XdgShellState.
        shm_state: ShmState::new(&mut display, Vec::new(), None),
        output_manager: OutputManagerState::new(),
        seat_state: SeatState::new(),
    };

    let keyboard = protocols.seat_state.

    let mut smalvil = Smalvil {
        protocols,
        keyboard: todo!(),
        pointer: todo!(),
    };

    // TODO: Socket setup.

    // TODO: Run loop
}

/// The primary compositor state data type.
///
/// This struct contains all the moving parts of the compositor and other data you need to track. This data
/// type is passed around to most parts of the compositor, meaning this is a reliable place to store data you
/// may need to access later.
pub struct Smalvil {
    protocols: ProtocolStates,
    seat: Seat,
    keyboard: KeyboardHandle,
    pointer: PointerHandle<Smalvil>,
}

/// All the protocol delegate types Smalvil uses.
pub struct ProtocolStates {
    compositor_state: CompositorState,
    xdg_shell_state: XdgShellState,
    shm_state: ShmState,
    output_manager: OutputManagerState,
    seat_state: SeatState<Smalvil>,
}

/*
  Trait implementations and delegate macros
*/

// In order to use the delegate types we have defined in the `Smalvil` type and created in our main function,
// we need to implement some traits and use some macros.
//
// The trait bounds on `D` required by `CompositorState::new` indicate that the Smalvil type needs to
// implement the `CompositorHandler` trait.
impl CompositorHandler for Smalvil {
    // Many wayland frontend abstractions require a way to get the delegate type from your data type.
    fn compositor_state(&mut self) -> &mut CompositorState {
        &mut self.protocols.compositor_state
    }

    // This function is called when a surface has sent a commit to indicate the state has changed.
    //
    // In this case Smalvil delegates the handling to the "desktop" abstractions. A compositor can use this
    // function to perform other tasks as well.
    fn commit(&mut self, _dh: &mut DisplayHandle<'_>, _surface: &wl_surface::WlSurface) {
        todo!("desktop")
    }
}

// In order to complete implementing everything needed for the compositor state, we need to use the
// "delegate_compositor" macro to implement the Dispatch and GlobalDispatch traits on Smalvil for all the
// compositor protocol types. This macro ensures that the compositor protocols are handled by the
// CompositorState delegate type.
delegate_compositor!(Smalvil);

// Xdg shell trait and delegate

impl XdgShellHandler for Smalvil {
    fn xdg_shell_state(&mut self) -> &mut XdgShellState {
        &mut self.protocols.xdg_shell_state
    }

    /// Called when an event generated by the xdg shell is received.
    fn request(&mut self, _dh: &mut DisplayHandle<'_>, _request: XdgRequest) {
        todo!("desktop")
    }
}

// Implement Dispatch and GlobalDispatch for Smalvil to handle the xdg shell.
delegate_xdg_shell!(Smalvil);
