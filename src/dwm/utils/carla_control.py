import argparse
import carla
import tkinter
import tkinter.font
import tkinter.ttk


class KeyPanel():

    def __init__(
        self, master, title: str, detail: str, style_class: str = "Inactivated"
    ):
        default_font = tkinter.font.nametofont("TkDefaultFont")
        default_font_family = default_font.cget("family")
        self.key_panel = tkinter.ttk.Frame(
            master, style="{}.TFrame".format(style_class))
        self.label_group = tkinter.ttk.Frame(
            self.key_panel, style="{}.TFrame".format(style_class))
        self.title = tkinter.ttk.Label(
            self.label_group, text=title,
            style="{}.TLabel".format(style_class),
            font=(default_font_family, 18), padding=(0, -4, 0, -4))
        self.detail = tkinter.ttk.Label(
            self.label_group, text=detail,
            style="{}.TLabel".format(style_class),
            font=(default_font_family, 10), padding=(0, -2, 0, -2))

        self.label_group.place(relx=0.5, rely=0.5, anchor="center")
        self.title.pack(anchor="center")
        self.detail.pack(anchor="center")

    def set_style_class(self, style_class: str):
        self.key_panel.configure(style="{}.TFrame".format(style_class))
        self.label_group.configure(style="{}.TFrame".format(style_class))
        self.title.configure(style="{}.TLabel".format(style_class))
        self.detail.configure(style="{}.TLabel".format(style_class))


class KeyboardControlPanel():

    def __init__(self, master, hero_vehicle=None, style_config=None):
        self.master = master
        self.hero_vehicle = hero_vehicle

        default_style_config = {
            "Inactivated.TFrame": {
                "background": self.master.cget("background")
            },
            "Inactivated.TLabel": {
                "background": self.master.cget("background"),
                "foreground": "black",
            },
            "Activated.TFrame": {
                "background": "dimgray"
            },
            "Activated.TLabel": {
                "background": "dimgray",
                "foreground": "white",
            }
        }

        self.style = tkinter.ttk.Style()
        for k, v in (style_config or default_style_config).items():
            self.style.configure(k, **v)

        self.frame = tkinter.ttk.Frame(master, padding=2)
        self.label_reverse = KeyPanel(self.frame, title="Q", detail="Reverse")
        self.label_up = KeyPanel(self.frame, title="W", detail="Throttle")
        self.label_autopilot = KeyPanel(
            self.frame, title="E", detail="Auto pilot")
        self.label_left = KeyPanel(self.frame, title="A", detail="Left")
        self.label_down = KeyPanel(self.frame, title="S", detail="Brake")
        self.label_right = KeyPanel(self.frame, title="D", detail="Right")

        self.pressed_key = {}
        self.is_auto = False
        self.reverse = False

    def setup_layout(self):
        for i in range(2):
            self.frame.grid_rowconfigure(i, weight=1)

        for i in range(3):
            self.frame.grid_columnconfigure(i, weight=1)

        grid_args = {
            "padx": 2,
            "pady": 2,
            "sticky": tkinter.NSEW
        }
        self.frame.pack(fill=tkinter.BOTH, expand=True)
        self.label_reverse.key_panel.grid(column=0, row=0, **grid_args)
        self.label_up.key_panel.grid(column=1, row=0, **grid_args)
        self.label_autopilot.key_panel.grid(column=2, row=0, **grid_args)
        self.label_left.key_panel.grid(column=0, row=1, **grid_args)
        self.label_down.key_panel.grid(column=1, row=1, **grid_args)
        self.label_right.key_panel.grid(column=2, row=1, **grid_args)

    def update_manual_control(self):
        control = carla.VehicleControl()
        control.throttle = \
            0.8 if any([i in self.pressed_key for i in ["w", "Up"]]) else 0
        control.steer = (
            -0.8 if any([i in self.pressed_key for i in ["a", "Left"]]) else 0
        ) + (
            0.8 if any([i in self.pressed_key for i in ["d", "Right"]]) else 0
        )
        control.brake = \
            1.0 if any([i in self.pressed_key for i in ["s", "Down"]]) else 0
        control.reverse = self.reverse
        self.hero_vehicle.apply_control(control)

    def on_key_pressed_event(self, event):
        self.pressed_key[event.keysym] = True
        if event.keysym in ["w", "Up"]:
            self.label_up.set_style_class("Activated")
        elif event.keysym in ["a", "Left"]:
            self.label_left.set_style_class("Activated")
        elif event.keysym in ["d", "Right"]:
            self.label_right.set_style_class("Activated")
        elif event.keysym in ["s", "Down"]:
            self.label_down.set_style_class("Activated")

        if self.hero_vehicle is not None and not self.is_auto:
            self.update_manual_control()

    def on_key_released_event(self, event):
        if event.keysym == "e":
            self.is_auto = not self.is_auto
            self.label_autopilot.set_style_class(
                "Activated" if self.is_auto else "Inactivated")
            if self.hero_vehicle is not None:
                self.hero_vehicle.set_autopilot(self.is_auto)
        elif event.keysym == "q":
            self.reverse = not self.reverse
            self.label_reverse.set_style_class(
                "Activated" if self.reverse else "Inactivated")
        elif event.keysym in ["w", "Up"]:
            self.label_up.set_style_class("Inactivated")
        elif event.keysym in ["a", "Left"]:
            self.label_left.set_style_class("Inactivated")
        elif event.keysym in ["d", "Right"]:
            self.label_right.set_style_class("Inactivated")
        elif event.keysym in ["s", "Down"]:
            self.label_down.set_style_class("Inactivated")

        if event.keysym in self.pressed_key:
            del self.pressed_key[event.keysym]

        if self.hero_vehicle is not None and not self.is_auto:
            self.update_manual_control()


def create_parser():
    parser = argparse.ArgumentParser(
        description="Carla control Client")
    parser.add_argument(
        "--host", default="127.0.0.1", type=str,
        help="The host address of the Carla simulator.")
    parser.add_argument(
        "-p", "--port", default=2000, type=int,
        help="The port of the Carla simulator.")
    parser.add_argument(
        "--client-timeout", default=10.0, type=float,
        help="The timeout of the Carla client.")
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    client = carla.Client(args.host, args.port, 1)
    client.set_timeout(args.client_timeout)
    world = client.get_world()
    world.wait_for_tick()

    hero_vehicle, = [
        i for i in world.get_actors()
        if (
            i.type_id.startswith("vehicle") and
            i.attributes.get("role_name") == "hero"
        )
    ]
    print("Hero vehicle: {}".format(hero_vehicle.id))

    window_args = {
        "title": "Carla Control",
        "geometry": "244x124"
    }
    window = tkinter.Tk()
    for k, v in window_args.items():
        getattr(window, k)(v)

    control_panel = KeyboardControlPanel(window, hero_vehicle)
    control_panel.setup_layout()
    window.bind("<KeyPress>", control_panel.on_key_pressed_event)
    window.bind("<KeyRelease>", control_panel.on_key_released_event)
    window.mainloop()
