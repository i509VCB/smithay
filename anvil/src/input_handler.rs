use std::{process::Command, sync::atomic::Ordering};

#[cfg(feature = "udev")]
use crate::udev::UdevData;
#[cfg(feature = "winit")]
use crate::winit::WinitData;
use crate::AnvilState;

use smithay::{
    backend::input::{
        self, Device, DeviceCapability, Event, InputBackend, InputEvent, KeyState, KeyboardKeyEvent,
        PointerAxisEvent, PointerButtonEvent, ProximityState, TabletToolButtonEvent, TabletToolEvent,
        TabletToolProximityEvent, TabletToolTipEvent, TabletToolTipState,
    },
    reexports::wayland_server::protocol::wl_pointer,
    wayland::{
        output::Mode,
        seat::{keysyms as xkb, AxisFrame, Keysym, ModifiersState},
        tablet_manager::{TabletDescriptor, TabletSeatTrait},
        SERIAL_COUNTER as SCOUNTER,
    },
};

#[cfg(feature = "winit")]
use smithay::backend::input::PointerMotionAbsoluteEvent;

#[cfg(feature = "udev")]
use smithay::backend::{input::PointerMotionEvent, session::Session};

impl<Backend> AnvilState<Backend> {
    fn keyboard_key_to_action<B: InputBackend>(&mut self, evt: B::KeyboardKeyEvent) -> KeyAction {
        let keycode = evt.key_code();
        let state = evt.state();
        debug!(self.log, "key"; "keycode" => keycode, "state" => format!("{:?}", state));
        let serial = SCOUNTER.next_serial();
        let log = &self.log;
        let time = Event::time(&evt);
        let mut action = KeyAction::None;
        let suppressed_keys = &mut self.suppressed_keys;
        self.keyboard
            .input(keycode, state, serial, time, |modifiers, keysym| {
                debug!(log, "keysym";
                    "state" => format!("{:?}", state),
                    "mods" => format!("{:?}", modifiers),
                    "keysym" => ::xkbcommon::xkb::keysym_get_name(keysym)
                );

                // If the key is pressed and triggered a action
                // we will not forward the key to the client.
                // Additionally add the key to the suppressed keys
                // so that we can decide on a release if the key
                // should be forwarded to the client or not.
                if let KeyState::Pressed = state {
                    action = process_keyboard_shortcut(*modifiers, keysym);

                    // forward to client only if action == KeyAction::Forward
                    let forward = matches!(action, KeyAction::Forward);

                    if !forward {
                        suppressed_keys.push(keysym);
                    }

                    forward
                } else {
                    let suppressed = suppressed_keys.contains(&keysym);

                    if suppressed {
                        suppressed_keys.retain(|k| *k != keysym);
                    }

                    !suppressed
                }
            });
        action
    }

    fn on_pointer_button<B: InputBackend>(&mut self, evt: B::PointerButtonEvent) {
        let serial = SCOUNTER.next_serial();
        let button = match evt.button() {
            input::MouseButton::Left => 0x110,
            input::MouseButton::Right => 0x111,
            input::MouseButton::Middle => 0x112,
            input::MouseButton::Other(b) => b as u32,
        };
        let state = match evt.state() {
            input::ButtonState::Pressed => {
                // change the keyboard focus unless the pointer is grabbed
                if !self.pointer.is_grabbed() {
                    let under = self
                        .window_map
                        .borrow_mut()
                        .get_surface_and_bring_to_top(self.pointer_location);
                    self.keyboard
                        .set_focus(under.as_ref().map(|&(ref s, _)| s), serial);
                }
                wl_pointer::ButtonState::Pressed
            }
            input::ButtonState::Released => wl_pointer::ButtonState::Released,
        };
        self.pointer.button(button, state, serial, evt.time());
    }

    fn on_pointer_axis<B: InputBackend>(&mut self, evt: B::PointerAxisEvent) {
        let source = match evt.source() {
            input::AxisSource::Continuous => wl_pointer::AxisSource::Continuous,
            input::AxisSource::Finger => wl_pointer::AxisSource::Finger,
            input::AxisSource::Wheel | input::AxisSource::WheelTilt => wl_pointer::AxisSource::Wheel,
        };
        let horizontal_amount = evt
            .amount(input::Axis::Horizontal)
            .unwrap_or_else(|| evt.amount_discrete(input::Axis::Horizontal).unwrap() * 3.0);
        let vertical_amount = evt
            .amount(input::Axis::Vertical)
            .unwrap_or_else(|| evt.amount_discrete(input::Axis::Vertical).unwrap() * 3.0);
        let horizontal_amount_discrete = evt.amount_discrete(input::Axis::Horizontal);
        let vertical_amount_discrete = evt.amount_discrete(input::Axis::Vertical);

        {
            let mut frame = AxisFrame::new(evt.time()).source(source);
            if horizontal_amount != 0.0 {
                frame = frame.value(wl_pointer::Axis::HorizontalScroll, horizontal_amount);
                if let Some(discrete) = horizontal_amount_discrete {
                    frame = frame.discrete(wl_pointer::Axis::HorizontalScroll, discrete as i32);
                }
            } else if source == wl_pointer::AxisSource::Finger {
                frame = frame.stop(wl_pointer::Axis::HorizontalScroll);
            }
            if vertical_amount != 0.0 {
                frame = frame.value(wl_pointer::Axis::VerticalScroll, vertical_amount);
                if let Some(discrete) = vertical_amount_discrete {
                    frame = frame.discrete(wl_pointer::Axis::VerticalScroll, discrete as i32);
                }
            } else if source == wl_pointer::AxisSource::Finger {
                frame = frame.stop(wl_pointer::Axis::VerticalScroll);
            }
            self.pointer.axis(frame);
        }
    }
}

#[cfg(feature = "winit")]
impl AnvilState<WinitData> {
    pub fn process_input_event<B>(&mut self, event: InputEvent<B>)
    where
        B: InputBackend<SpecialEvent = smithay::backend::winit::WinitEvent>,
    {
        use smithay::backend::winit::WinitEvent;

        match event {
            InputEvent::Keyboard { event, .. } => match self.keyboard_key_to_action::<B>(event) {
                KeyAction::None | KeyAction::Forward => {}
                KeyAction::Quit => {
                    info!(self.log, "Quitting.");
                    self.running.store(false, Ordering::SeqCst);
                }
                KeyAction::Run(cmd) => {
                    info!(self.log, "Starting program"; "cmd" => cmd.clone());
                    if let Err(e) = Command::new(&cmd).spawn() {
                        error!(self.log,
                            "Failed to start program";
                            "cmd" => cmd,
                            "err" => format!("{:?}", e)
                        );
                    }
                }
                action => {
                    warn!(self.log, "Key action {:?} unsupported on winit backend.", action);
                }
            },
            InputEvent::PointerMotionAbsolute { event, .. } => self.on_pointer_move_absolute::<B>(event),
            InputEvent::PointerButton { event, .. } => self.on_pointer_button::<B>(event),
            InputEvent::PointerAxis { event, .. } => self.on_pointer_axis::<B>(event),
            InputEvent::Special(WinitEvent::Resized { size, .. }) => {
                self.output_map.borrow_mut().update_mode(
                    crate::winit::OUTPUT_NAME,
                    Mode {
                        width: size.0 as i32,
                        height: size.1 as i32,
                        refresh: 60_000,
                    },
                );
            }
            _ => {
                // other events are not handled in anvil (yet)
            }
        }
    }

    fn on_pointer_move_absolute<B: InputBackend>(&mut self, evt: B::PointerMotionAbsoluteEvent) {
        // different cases depending on the context:
        let (x, y) = evt.position();
        self.pointer_location = (x, y);
        let serial = SCOUNTER.next_serial();
        let under = self.window_map.borrow().get_surface_under((x as f64, y as f64));
        self.pointer.motion((x, y), under, serial, evt.time());
    }
}

#[cfg(feature = "udev")]
impl AnvilState<UdevData> {
    pub fn process_input_event<B: InputBackend>(&mut self, event: InputEvent<B>) {
        match event {
            InputEvent::Keyboard { event, .. } => match self.keyboard_key_to_action::<B>(event) {
                KeyAction::None | KeyAction::Forward => {}
                KeyAction::Quit => {
                    info!(self.log, "Quitting.");
                    self.running.store(false, Ordering::SeqCst);
                }
                #[cfg(feature = "udev")]
                KeyAction::VtSwitch(vt) => {
                    info!(self.log, "Trying to switch to vt {}", vt);
                    if let Err(err) = self.backend_data.session.change_vt(vt) {
                        error!(self.log, "Error switching to vt {}: {}", vt, err);
                    }
                }
                KeyAction::Run(cmd) => {
                    info!(self.log, "Starting program"; "cmd" => cmd.clone());
                    if let Err(e) = Command::new(&cmd).spawn() {
                        error!(self.log,
                            "Failed to start program";
                            "cmd" => cmd,
                            "err" => format!("{:?}", e)
                        );
                    }
                }
                KeyAction::Screen(num) => {
                    let geometry = self
                        .output_map
                        .borrow()
                        .find_by_index(num, |_, geometry| geometry)
                        .ok();

                    if let Some(geometry) = geometry {
                        let x = geometry.x as f64 + geometry.width as f64 / 2.0;
                        let y = geometry.height as f64 / 2.0;
                        self.pointer_location = (x, y)
                    }
                }
            },
            InputEvent::PointerMotion { event, .. } => self.on_pointer_move::<B>(event),
            InputEvent::PointerButton { event, .. } => self.on_pointer_button::<B>(event),
            InputEvent::PointerAxis { event, .. } => self.on_pointer_axis::<B>(event),
            InputEvent::TabletToolAxis { event, .. } => self.on_tablet_tool_axis::<B>(event),
            InputEvent::TabletToolProximity { event, .. } => self.on_tablet_tool_proximity::<B>(event),
            InputEvent::TabletToolTip { event, .. } => self.on_tablet_tool_tip::<B>(event),
            InputEvent::TabletToolButton { event, .. } => self.on_tablet_button::<B>(event),
            InputEvent::DeviceAdded { device } => {
                if device.has_capability(DeviceCapability::TabletTool) {
                    self.seat
                        .tablet_seat()
                        .add_tablet(&TabletDescriptor::from(&device));
                }
            }
            InputEvent::DeviceRemoved { device } => {
                if device.has_capability(DeviceCapability::TabletTool) {
                    let tablet_seat = self.seat.tablet_seat();

                    tablet_seat.remove_tablet(&TabletDescriptor::from(&device));

                    // If there are no tablets in seat we can remove all tools
                    if tablet_seat.count_tablets() == 0 {
                        tablet_seat.clear_tools();
                    }
                }
            }
            _ => {
                // other events are not handled in anvil (yet)
            }
        }
    }

    fn on_pointer_move<B: InputBackend>(&mut self, evt: B::PointerMotionEvent) {
        let (x, y) = (evt.delta_x(), evt.delta_y());
        let serial = SCOUNTER.next_serial();
        self.pointer_location.0 += x as f64;
        self.pointer_location.1 += y as f64;

        // clamp to screen limits
        // this event is never generated by winit
        self.pointer_location = self.clamp_coords(self.pointer_location);

        let under = self.window_map.borrow().get_surface_under(self.pointer_location);
        self.pointer
            .motion(self.pointer_location, under, serial, evt.time());
    }

    fn on_tablet_tool_axis<B: InputBackend>(&mut self, evt: B::TabletToolAxisEvent) {
        let output_map = self.output_map.borrow();
        let pointer_location = &mut self.pointer_location;
        let tablet_seat = self.seat.tablet_seat();
        let window_map = self.window_map.borrow();

        output_map
            .with_primary(|_, rect| {
                pointer_location.0 = evt.x_transformed(rect.width as u32) + rect.x as f64;
                pointer_location.1 = evt.y_transformed(rect.height as u32) + rect.y as f64;

                let under = window_map.get_surface_under(*pointer_location);
                let tablet = tablet_seat.get_tablet(&TabletDescriptor::from(&evt.device()));
                let tool = tablet_seat.get_tool(&evt.tool());

                if let (Some(tablet), Some(tool)) = (tablet, tool) {
                    if evt.pressure_has_changed() {
                        tool.pressure(evt.pressure());
                    }
                    if evt.distance_has_changed() {
                        tool.distance(evt.distance());
                    }
                    if evt.tilt_has_changed() {
                        tool.tilt(evt.tilt());
                    }
                    if evt.slider_has_changed() {
                        tool.slider_position(evt.slider_position());
                    }
                    if evt.rotation_has_changed() {
                        tool.rotation(evt.rotation());
                    }
                    if evt.wheel_has_changed() {
                        tool.wheel(evt.wheel_delta(), evt.wheel_delta_discrete());
                    }

                    tool.motion(
                        *pointer_location,
                        under,
                        &tablet,
                        SCOUNTER.next_serial(),
                        evt.time(),
                    );
                }
            })
            .unwrap();
    }

    fn on_tablet_tool_proximity<B: InputBackend>(&mut self, evt: B::TabletToolProximityEvent) {
        let output_map = self.output_map.borrow();
        let pointer_location = &mut self.pointer_location;
        let tablet_seat = self.seat.tablet_seat();
        let window_map = self.window_map.borrow();

        output_map
            .with_primary(|_, rect| {
                let tool = evt.tool();
                tablet_seat.add_tool(&tool);

                pointer_location.0 = evt.x_transformed(rect.width as u32) + rect.x as f64;
                pointer_location.1 = evt.y_transformed(rect.height as u32) + rect.y as f64;

                let under = window_map.get_surface_under(*pointer_location);
                let tablet = tablet_seat.get_tablet(&TabletDescriptor::from(&evt.device()));
                let tool = tablet_seat.get_tool(&tool);

                if let (Some(under), Some(tablet), Some(tool)) = (under, tablet, tool) {
                    match evt.state() {
                        ProximityState::In => tool.proximity_in(
                            *pointer_location,
                            under,
                            &tablet,
                            SCOUNTER.next_serial(),
                            evt.time(),
                        ),
                        ProximityState::Out => tool.proximity_out(evt.time()),
                    }
                }
            })
            .unwrap();
    }

    fn on_tablet_tool_tip<B: InputBackend>(&mut self, evt: B::TabletToolTipEvent) {
        let tool = self.seat.tablet_seat().get_tool(&evt.tool());

        if let Some(tool) = tool {
            match evt.tip_state() {
                TabletToolTipState::Down => {
                    tool.tip_down(SCOUNTER.next_serial(), evt.time());

                    // change the keyboard focus unless the pointer is grabbed
                    if !self.pointer.is_grabbed() {
                        let under = self
                            .window_map
                            .borrow_mut()
                            .get_surface_and_bring_to_top(self.pointer_location);

                        let serial = SCOUNTER.next_serial();
                        self.keyboard
                            .set_focus(under.as_ref().map(|&(ref s, _)| s), serial);
                    }
                }
                TabletToolTipState::Up => {
                    tool.tip_up(evt.time());
                }
            }
        }
    }

    fn on_tablet_button<B: InputBackend>(&mut self, evt: B::TabletToolButtonEvent) {
        let tool = self.seat.tablet_seat().get_tool(&evt.tool());

        if let Some(tool) = tool {
            tool.button(
                evt.button(),
                evt.button_state(),
                SCOUNTER.next_serial(),
                evt.time(),
            );
        }
    }

    fn clamp_coords(&self, pos: (f64, f64)) -> (f64, f64) {
        if self.output_map.borrow().is_empty() {
            return pos;
        }

        let (pos_x, pos_y) = pos;
        let output_map = self.output_map.borrow();
        let max_x = output_map.width();
        let clamped_x = pos_x.max(0.0).min(max_x as f64);
        let max_y = output_map.height(clamped_x as i32);

        if let Some(max_y) = max_y {
            let clamped_y = pos_y.max(0.0).min(max_y as f64);

            (clamped_x, clamped_y)
        } else {
            (clamped_x, pos_y)
        }
    }
}

/// Possible results of a keyboard action
#[derive(Debug)]
enum KeyAction {
    /// Quit the compositor
    Quit,
    /// Trigger a vt-switch
    VtSwitch(i32),
    /// run a command
    Run(String),
    /// Switch the current screen
    Screen(usize),
    /// Forward the key to the client
    Forward,
    /// Do nothing more
    None,
}

fn process_keyboard_shortcut(modifiers: ModifiersState, keysym: Keysym) -> KeyAction {
    if modifiers.ctrl && modifiers.alt && keysym == xkb::KEY_BackSpace
        || modifiers.logo && keysym == xkb::KEY_q
    {
        // ctrl+alt+backspace = quit
        // logo + q = quit
        KeyAction::Quit
    } else if (xkb::KEY_XF86Switch_VT_1..=xkb::KEY_XF86Switch_VT_12).contains(&keysym) {
        // VTSwicth
        KeyAction::VtSwitch((keysym - xkb::KEY_XF86Switch_VT_1 + 1) as i32)
    } else if modifiers.logo && keysym == xkb::KEY_Return {
        // run terminal
        KeyAction::Run("weston-terminal".into())
    } else if modifiers.logo && keysym >= xkb::KEY_1 && keysym <= xkb::KEY_9 {
        KeyAction::Screen((keysym - xkb::KEY_1) as usize)
    } else {
        KeyAction::Forward
    }
}
