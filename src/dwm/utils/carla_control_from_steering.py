import argparse
import carla
import io
from PIL import Image, ImageTk
import pygame
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


class SteeringControlPanel():

    steering_icon_png_content = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\xc0\x00\x00\x00\xc0\x08\x03\x00\x00\x00e\x02\x9c5\x00\x00\x00\x01sRGB\x00\xae\xce\x1c\xe9\x00\x00\x00\x04gAMA\x00\x00\xb1\x8f\x0b\xfca\x05\x00\x00\x003PLTE\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xa3\x05F\xc9\x00\x00\x00\x10tRNS\x00\x10 0@P`p\x7f\x8f\x9f\xaf\xbf\xcf\xdf\xef\x05Q\x84l\x00\x00\x00\tpHYs\x00\x00\x0e\xc3\x00\x00\x0e\xc3\x01\xc7o\xa8d\x00\x00\tmIDATx^\xed\x9d\xdb\x96\xab \x0c\x867\xd5\xd6\xb3\xce\xfb?\xed\xd6\n\n9@@l{\xc1w3k\xd5V\x81$\x7f"\xa0\xf3\xafP(\x14\n\x85B\xa1P(\x14\n\x85B>T\xf5l\xba~\x18\xa7i^\x96y\x9e\xc6\xa1\xef\xdb\xba\xd2G\x7f\x1b\xf5l\xc7i\xf9\xa3\x99\x87\xaeQ\xfa\x8b\xbf\x88zv\x93n\xaa\x87\xa5\x7f\xfdd\'\xeaV\xd0x\xc3\xdc\xd5\xfag?\xc2#\xa6\xf5;s\xf7\xd0?\xfe:\xea\xc5:\xbd\x9f\xb1\xd1g\xf8*\xaaKk\xfd\xce\xd7\xcd\xf0\xb8\xd4\xfc\x8d\xafv\xa1\xba\xdc\xfc\x8d\xfe[]\xb8\xe6<6\xdf\xe9\xc2+W\xf37>\x1f\xceu\xb4n\xfa\x99>[h\xe4\xf3\x9e\x93N\x9f\xfb\x13\xd4\xb3\xbehV\x96\x8fe\xe7&\xff\xf0\xef\xb4\xfa\x02\xf7\xa2\x06}\xb9\x1b\x98>P\xe6\x89\xdcg\x99\xd6\xc2\xf9UW\x0f\xa5\x1e+\xf5\xf3\xd5\xf6\xa3>\xe6ey\xea\xcb\xdcF\xd8}\xa6\xae\xa9\xe8\x81\xac_\x82r\xfbf7j\xf5e\x18\x96\xb1\r\xc9\xe1+\xe4\x81\xbd\xfe\xe2\x1d\xa8^_\x84d\x19\x85w*O\xefi\xfeF\xfd\xb5\xfcx\xc3w\x0e\x8e\xbdM\xe3\x8b\xa4\xbbz\xe0k\xffX\xc7\xea\xc7\xd3\x13\xd4\xf7\x88\x91\xe2\x03pLJA\x15? \xf3\r=\xe0\xc7?\xad\xf9\x1bO\xd6\x91&\xfd\x8d|\xb0\xed_.M\x944\xfa,\x88\xecq\xc0\t\xc7x\xb5\x94gO\xac\x8fg\x82\xd1\xff\x1c\x89\xf3\xa5\xcf\x05\xc9\x9a\x0f\x18K\x8fYbM1z\x94\xf1&\xa7\xa6\xeb\x876\x97V0\xe3\x93\xad.R\xa4X\xe4\xac\xbb*r\x84\x96\\\xb7\xca\xa4\x00MYo\xc4+r\x8c2\x052i\xe0\xbc\xed\xe7\xd2d\x96\xd2\x944o\xfedO\x87r\x86\xbbLrh\xf2\xc8\x0f\x80\xeaA\x860\xa02\xc0Mw~T\x0f\x06},\x19\xca\x81\xee\xbas%\x8d}\xd5\x89\x88Q\xc9\xa6n\x08J\xafg},\x11B\x81\xee\x9c\xbey\x10\xf6\xbe\x94\x90\x15>\xe1\xbd\xf3\x06\x95\xbe\x8a\xc5%\x83w\xfa$\x167\xcf\x1a\x10&\xbf\x10\xc7\x84Eo\x11P\x1b"\xe8\xd2M\x80\r\xb0\xdc>\x89Lxm\xb2\t\x08\x03|`\x1e\xff\xa9/e\x91j\x02\x9c\xc3.\xa7\x15\t\xb8vL\xbc\xb7\xa1$\xe8\xf6\x05jE.\x98\xa7\x99\x80\xb9\x8d\x1c^\xfax~T\xdd\xe3A\xdbH2\x01a\x00\xc3\xd0\xdc\x90\x8cU3\xf2WL\xb9\x1e\x11L\x16c^g\n\xed\xb4H\x11\x0f~\xdeL\x93+ h\xb7wY\xf4w#x\xe8\x9f\xfa\x19\xaen\x9dQO\xc6\xed\x01\xf1>\x14X\t8\x19\xda\xe4\x80\xa8\x1a\xf16\x91\xf85\xcc\xa0U-\x92\x02"j\x83Q\xbc\x0f\x11e\xa1\x9f>\xa6JUu\xe7[ \xa0\x88-\x82\xc5\x1ed\xd1K\xe3a\x90:\x8eEl* \xe7\x08\x02\x80\x0cW\xb5\xdd8-\xcb4\xf4p\xf1\xa6\xd6?\x88!\xd2\x87\x94\xfeY\x0c\x8eV\x03\x0f\x07\x92\xeb\xcf14qR\x91p\x05[\'j"\x898Sa\xccl\xa8\x8f\xb8\\\xe6_G\xa4\xb0\xda\xcfm\x05\xb1\xb7\x05\x11\xf7z\x01\xe2*\xe1X\x8d\xb0\xef\xd4\xf8\xb5|{> \x98\xe8!QA\x80\xd2pP6\xce\xa6y\xd7\xf2O?\x08F\x19:ML\x10\xa0\x10x\x04\x9c\xf6t\xa0\x80\xfe\x9es\x02\xfe/.\r\x1a\xc4\x98 \x80\xcd]\xcdW\xf9\xf2\xe6r8P0:\x0f\xad\xa5W\x1d4\xe3*\xbc\xf0xL\x10@\x07}O\xd4{\xdav\x0c\x0e\xbdR\xe1p8\x1b\xaft\xfb\xca\'\x8c\xf3\x98\xd5\x02\xd8\xf9\xddC\xd8\x95\xe9c\xb2\x94\x9aY\x83\x9c\x13U\\\xb2\xdc\x86\x7f\x05v0&\x8a\xf5O\x0eL!\xc2\x18\xe1pl\x91\xfa\x1e\xae@\xafO\x1e\x0b\xcf(\xccO\xa1\x0b\x81*\xb9c\xd0\xe8\xf5nsX\xe0@\x1b\xa6\xb0 \x85\xc8Zx\x86g\x93\xcfIA\xe3\xd9\x0b\xff\xc4\xb0\x1d\x87\xa1O\x8cM\xa5\xd4\x03\xed\x10:\xbe\x8f\r\xe6\xac\xfb\xc3\xd3\xc9e\x08z\x8a\x13\xffx\xd3\x90\xf1 P\xa3MG\x05\x0c\x97\xf0\xcc\x014\x18\xee\xba?\x8cb\xf9\xb4,l"\xb8\x1d\x82{v\xcdU]ew\x8ak\xb7-\xa64\x06>\xb4\x80!\x86\x99B\xae\xa3\xb0\x03\xb0\xeb\xae\x11\x0e\x8fp\xc6\x19t\xda\xe9\xc1\xa1\'\x8es\xa1m\x17\xd0\x93\xe5\x1d\x80N\x8b\xa7\xb2\xec\x8d2f<\x9d\xd0G;~\x9c<hr\x81\xd5\xad\x85\xb8\x88>d\x90\'\x02\xd8\x01\xeav\xee4\x82\x19k\'r\xd0O\x9c\xca\xc0\x98\xf4t\x12j\xd7\x0b,&\xe4\x9b\x88`\xf8\x93\xfauD\xa6\x19:\xdb\xaf\x08kS\x87\xcd\x18\x13\xc3\xbf\x02eV\xbe\\\x06\xcb\x1ebt6\xb4\x03\x98\xc1\xb6\xedF(\x9e\xed\x10f,\xb5n17\xd3\xb0\x03\xf2T\x0c+\t.\x05\xeeF0\x0emw\x9b\xb0\x99\xdd\x1c\xd3\x94\xf7g\xec\xaa\xdb\xfd\x1d\xd8\x8d`\xecc\xff\x8a\xb2\x99>\xf4F\x7f\xb45\xd13\x97\xb1\x7f\xf7D\x7f\x1c\x06\xe6p\xc6\x856V#\x98\x06\xd8\xbf\xa2\x1a\xa5\x0f\xbd\xd1\x1f\xfd\xb3\xb2\x1d\x81\xfe\xf2\x81\xfe8\x0c\xb4\x80\xa7\x03\xab\x92\x98\xa3\x01\x17\xb25\xe5\x08\xc7\x8e\x1f\xfe\x158\x90\xfa\xe30\xc2 \x06\xd8\xda%\x0c\xe2\x00\xfa\xdb\x86\xec*\x04\xb0\x93-!\xa3\xb6H\xc9R\x12\xcc\x03\xf2\x0e\x88\xf2\x00\xc2)]\xd0O\x9cR\x1b\xd4\x19\x0c\xb0\xaa\x97\'\xb2p)A\xe1\xa8\x1e\x1c-w?\x93/rO\xe0\x0c\xa4\xbc\x94\x80\xc5\x9c\xb0\x10wb\x1f\x0c\xb2c\x9es\x0e\xc0\x0b\xac\xb6\xe5\x1dpJ\xc7\x15a!\xee\xfe\xcc\xee\x01\x98\xab\x13\x96\x95\xf0\xb6D>A\x9dX\x88\x03\x9f=\x9fm\x83S22\x0fB\x8e \x8b\x9c\x8d\xd4:\x16\xc6N\xffz\xfcS\x0f\xb4\xf8\x9bz:Y(n\xa4\x16!h2\x8d\xc6\x9a!\xf5\x02\xf3\xa9\xd0p\x1b\xfa\x17\x07\xb2D\x80\x8dN"tH<\x1c\xd2V\xac\xc0\xbeK\xe7\x03<\xab\xfb\x07\xe2\x1d;PE\xe5\xc5(\xced\xe2\xf8\x17,\x1e\x89\x1d\x01J\x89<\x8fa\x1d\x95w\x9e{\x1a\xe0@.%H\x12\xf4\xe7\x12\xa0\x0cE\xcc\xea\xc1\xbe\x03\xe4\xedG\xee(\x9f\x16\xc22\x14\xa1`\xbeI\xec\xa8F\xa0A\x94\x8a\xd7\x1b\x18\xc5R\xe5\xd8\xe0\xe3\x80\xbewg\x80\x8a\x16\x13\xc3W\x82`\xa5b\xd4TO\x9b\x0b\xb92\x86\x84\xfdb\xc6\x0e\xd7\x0e\x1b\x91\x8f\x9b!;\xc6\x84\x00\x11\x041\x12\xb0\x01v\xd1,\xc3S\xae\x03o\x90\x15#\xd2\xd8\x064\xa0\xb0\x02\xb6y\xb6\xfd8\xff-\xf34t\xcc\x03\xc6\x1e\x90\x06\xc5\x85\x00\xa1\x86\xd2d\x9c\t\xe4\xc3q!\x80o\xe7\xa2G\xe0"\xb0\x16\x88\xa9\xe4v\xa0\x0fE\x86\xf1EP\x08\xc7\xbb0\xf2!\xf9\r]\x06`\x19\x11\x91\xc1\rH\x87\x84s\x13Y@\x0e\x1c\x97\x86w\x90\x0f\x89M\xc0\xec\xedU\xb5\xb8\x9eD\x1a\x1aS\x89\x1ap])\x8d\xa35\x00\xd1.@U\xb7\xb3\xb8\x19\xd8\x00qYl\x07\xfb\x90tfl\x0f\x9f\xa1}U[\xf2Q\xaaz\xb6\xfb\xaeb\xa9\x14\xa2\x08H{\x0c\x05W4B!r\xe7\x80\xf4\xdf\ra$\xa2\x1c\x10\x9d\x04v\xb0\t\x84\xb9\x007@#\x1b\x00b\x1bK\xa2~`\x13\xc8F\x82\x9d\x9c\x90\x05\x11\xbe%J3\x00e\x02Y\x13\x88\xdf\xed\x88<\x99x\xfa:ACw\xb0\td\t\x117aG\x1f\xf6B<\x8f\x98j\x00r(E\'C\x85\xcc\x8eHE\xb1\x03\xa5\x1b\x802\x81(\x10\x89FlH:O\xdcQ\xa7\x1b\x80\xf6f\xc1x\xc0\x19\x1d\x8d@E\x89\x1dG\xd7\x1e]#\xa6y\x04\'dt4l<\xea\x81\xdc\x94$lA\x9c1\x9c\x90\x19\x1d\rJ\x18\xb5\x1f\xec\xea{b\xa8\xb6\x04\xab:FGC*J\xeeg\x8b\xbe\x91\x81\x10q\x1c\xee\x01\xad\xa3\xfa \x0bu\xa9+\x11\xbcC\xeeO\r\xf5\x80\xd4\xd1\x90\x8aR\xda\x95\xe3\xe9qr\xa2-\xd0\x03RG\xfd\x83I\xbf\xbf\xeb\xb2\x03m\x90\xad\xf1\xf7\x80\xd4Q\xaf\x8a\xd2\xfb9\xe3o$Ih\x87\xf0\x19\x97\xd4Q\x9f\x8a\xd2\xed\xcf\xf5\xfa\x04z\xdd\xc5\xf7r\x00RG=\xee@\xbfZ%\xdf\xd3\xd7\xcc|3?\xd7E\xea(o2\xe6\xdd\xabY\x02`\'\xba6 Z\xc4\xde\rq\xaf\xdf\xcb\x14\x00;\xcc5\xd8\xddbD\xd8p*\xca=\xb1\x92\xb5\xfd\xc4}\xb6\x86\xb9\x0c\xa1\\\xb4\x8a\xb2\xef\x8e\x1d2\x05\xf0\x01S\xe33\xab.\x84\xcf\x91]e\x1f\x18\xba\xe1\xfd?DY\xb7Cm{#t\x94\xe8(\xff\xe6\xe1;\xde?\xc3\xf7\xe0o@\x02C\xe8(\xeaf\xcd\xbfyx\xcePA`<=\xf8\x1b@R t\x14\xb4\xa9\xe6|r%\xf7\xfb\xaf\x0c\xdc\x0b\xe1\xdeLnV@\x83\xeb\xa8\xa8\xf2\xbe\xb6\xfa\xae\xf7\x17\xadpZ\xb4cO\x88\xa2\xbe\x9e*\xca\xbe|As\xeb\xeb[\x98\xdb\xf5\x83cb\x1a}Q\xab\xa8\xf2x\xfe\x8e\x7f+\xe9e\x98\x9cl1t\xaf\xb5\x86A\xdf\xeb\xd6\xb67\xbd\xcfsvnn\xff\x1a|\x81\x01\xdc\x19\x90\x0b\xf1\xff\x1b\xc5&j-?\x11o(_\xe4\xf2\xcb?e\x84\xdd(\x91\xdb\xdd\xc7 s\xa3X>\xe1>\x06\xff\x0b\xa8\xd3\xf8\x90\xfb\x18\xe8{\xa8t\xbc\xcf\x11\xdc\x83wcS$\xd7\xde<\x9cJ>?\xfa\xb0\xf7\x9cp;\x9b\xe2\x88\xdb\x07\x95\x99\xcb]X\xbe\xda\xfc\x8dK]\xf8~\xf3\xdf4\xe1\n\x87$\xfc\xcf\n>F\xd5F\xab\xea\xf2;\xad\xdfy\x86\xcad\x9be\x88\xdf\x84\xf6\x01\x1e/\xc9\x8b\x83\x96\xe1\xd7\xc6\xde\xa1n:\xfeuq\xcb\xd8\xb7?9\xf4\x88\xf7\xbfm\x19\xa7y\x8b\x8ce\xd9\xfe\xaf\xe3\xc0\xfeo\x94B\xa1P(\x14\n\x85B\xa1P(\x14\n\t\xfc\xfb\xf7\x1f\xec\xcfq\x06\xfal\xd8>\x00\x00\x00\x00IEND\xaeB`\x82'
    carla_axis_range = {
        "steer": [-1, 1],
        "throttle": [0, 1],
        "brake": [0, 1]
    }

    @staticmethod
    def joystick_value_to_carla(joystick_config, key, joystick_value):
        a = joystick_config[key]["range"]
        b = SteeringControlPanel.carla_axis_range[key]
        carla_value = (joystick_value - a[0]) / (a[1] - a[0]) * \
            (b[1] - b[0]) + b[0]
        return min(max(b[0], carla_value), b[1])

    def __init__(
        self, master, joystick, joystick_config: dict, hero_vehicle=None,
        style_config=None
    ):
        self.master = master
        self.joystick = joystick
        self.joystick_config = joystick_config
        self.hero_vehicle = hero_vehicle

        self.joystick.init()
        self.axis_state = {
            k: SteeringControlPanel.joystick_value_to_carla(
                self.joystick_config, k, v["default"])
            for k, v in self.joystick_config.items()
        }

        self.default_background_rgb = tuple([
            int((i + 0.5) * 256 / 65536)
            for i in self.master.winfo_rgb(self.master.cget("background"))
        ])
        default_background_hex = "#{:02x}{:02x}{:02x}".format(
            *self.default_background_rgb)

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
            },
            "Throttle.TFrame": {
                "background": default_background_hex
            },
            "Throttle.TLabel": {
                "background": default_background_hex
            },
            "Brake.TFrame": {
                "background": default_background_hex
            },
            "Brake.TLabel": {
                "background": default_background_hex
            }
        }

        default_font = tkinter.font.nametofont("TkDefaultFont")
        default_font_family = default_font.cget("family")

        self.style = tkinter.ttk.Style()
        for k, v in (style_config or default_style_config).items():
            self.style.configure(k, **v)

        self.frame = tkinter.ttk.Frame(master, padding=2)

        self.steering_frame = tkinter.ttk.Frame(self.frame)
        self.steering_label = tkinter.ttk.Label(
            self.steering_frame, text="Steering", style="Inactivated.TLabel",
            font=(default_font_family, 10), padding=(0, 2, 0, 2))
        self.steering_sub_frame = tkinter.ttk.Frame(self.steering_frame)
        self.steering_canvas = tkinter.Canvas(
            self.steering_sub_frame, width=192, height=192)
        self.steering_icon_image = Image.open(
            io.BytesIO(SteeringControlPanel.steering_icon_png_content))
        self.steering_icon_tk_image = ImageTk.PhotoImage(
            self.steering_icon_image)
        self.steering_canvas.create_image(
            (96, 96), image=self.steering_icon_tk_image)

        self.label_autopilot = KeyPanel(
            self.frame, title="▲", detail="Auto pilot")
        self.label_reverse = KeyPanel(self.frame, title="●", detail="Reverse")
        self.label_brake = KeyPanel(
            self.frame, title=" ", detail="Brake", style_class="Brake")
        self.label_throttle = KeyPanel(
            self.frame, title=" ", detail="Throttle", style_class="Throttle")

        self.pressed_key = {}
        self.is_auto = False
        self.reverse = False
        self.on_timer()

    def setup_layout(self):
        for i in range(2):
            self.frame.grid_rowconfigure(i, weight=1)

        self.frame.grid_columnconfigure(0, weight=3)
        self.frame.grid_columnconfigure(1, weight=2)
        self.frame.grid_columnconfigure(2, weight=2)

        grid_args = {
            "padx": 2,
            "pady": 2,
            "sticky": tkinter.NSEW
        }
        self.frame.pack(fill=tkinter.BOTH, expand=True)

        self.steering_frame.grid(column=0, row=0, rowspan=2, **grid_args)
        self.steering_label.pack(side="bottom")
        self.steering_sub_frame.pack(fill=tkinter.BOTH, expand=True)
        self.steering_canvas.place(relx=0.5, rely=1.0, anchor="center")

        self.label_autopilot.key_panel.grid(column=1, row=0, **grid_args)
        self.label_reverse.key_panel.grid(column=2, row=0, **grid_args)
        self.label_brake.key_panel.grid(column=1, row=1, **grid_args)
        self.label_throttle.key_panel.grid(column=2, row=1, **grid_args)

    def update_manual_control(self):
        control = carla.VehicleControl()
        for k, v in self.axis_state.items():
            setattr(control, k, v)

        control.reverse = self.reverse
        self.hero_vehicle.apply_control(control)

    def on_timer(self):
        for event in pygame.event.get():
            if event.type == pygame.JOYBUTTONUP:
                if event.button == 3:  # autopilot
                    self.is_auto = not self.is_auto
                    self.label_autopilot.set_style_class(
                        "Activated" if self.is_auto else "Inactivated")
                    if self.hero_vehicle is not None:
                        self.hero_vehicle.set_autopilot(self.is_auto)
                elif event.button == 1:  # reverse
                    self.reverse = not self.reverse
                    self.label_reverse.set_style_class(
                        "Activated" if self.reverse else "Inactivated")

        for i in self.axis_state.keys():
            jc = self.joystick_config[i]
            self.axis_state[i] = SteeringControlPanel.joystick_value_to_carla(
                self.joystick_config, i, self.joystick.get_axis(jc["id"]))

        icon_rotate = -450 * self.axis_state["steer"]
        self.steering_icon_tk_image.paste(
            self.steering_icon_image.rotate(icon_rotate))

        target_color_rgb = (0, 0, 0)
        throttle_color = "#{:02x}{:02x}{:02x}".format(*[
            int(
                a * (1 - self.axis_state["throttle"]) +
                b * self.axis_state["throttle"])
            for a, b in zip(self.default_background_rgb, target_color_rgb)
        ])
        self.style.configure("Throttle.TFrame", background=throttle_color)
        self.style.configure("Throttle.TLabel", background=throttle_color)

        brake_color = "#{:02x}{:02x}{:02x}".format(*[
            int(
                a * (1 - self.axis_state["brake"]) +
                b * self.axis_state["brake"])
            for a, b in zip(self.default_background_rgb, target_color_rgb)
        ])
        self.style.configure("Brake.TFrame", background=brake_color)
        self.style.configure("Brake.TLabel", background=brake_color)

        if self.hero_vehicle is not None and not self.is_auto:
            self.update_manual_control()

        self.master.after(100, self.on_timer)


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
    parser.add_argument(
        "--steer-axis-id-min-max-default", default="0,-1,1,0", type=str,
        help="The ID, min value, max value of steer axis of the joystick.")
    parser.add_argument(
        "--throttle-axis-id-min-max-default", default="5,0,1,0", type=str,
        help="The ID, min value, max value of throttle axis of the joystick.")
    parser.add_argument(
        "--brake-axis-id-min-max-default", default="1,0,1,0", type=str,
        help="The ID, min value, max value of brake axis of the joystick.")
    return parser


def parse_arg_joystick_axis_config(arg: str):
    id_str, min_str, max_str, default_str = arg.split(",")
    return {
        "id": int(id_str),
        "range": [float(min_str), float(max_str)],
        "default": float(default_str)
    }


if __name__ == "__main__":

    parser = create_parser()
    args = parser.parse_args()

    joystick_config = {
        "steer": parse_arg_joystick_axis_config(
            args.steer_axis_id_min_max_default),
        "throttle": parse_arg_joystick_axis_config(
            args.throttle_axis_id_min_max_default),
        "brake": parse_arg_joystick_axis_config(
            args.brake_axis_id_min_max_default)
    }

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

    pygame.init()
    pygame.joystick.init()

    assert pygame.joystick.get_count() >= 1
    if pygame.joystick.get_count() > 1:
        print(
            "Warning: only the 1st joystick device is connected to this app.")

    joystick = pygame.joystick.Joystick(0)
    window_args = {
        "title": joystick.get_name(),
        "geometry": "334x124"
    }
    window = tkinter.Tk()
    for k, v in window_args.items():
        getattr(window, k)(v)

    control_panel = SteeringControlPanel(
        window, joystick, joystick_config, hero_vehicle)
    control_panel.setup_layout()
    window.mainloop()
    pygame.joystick.quit()
