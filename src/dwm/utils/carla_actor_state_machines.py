import carla


class ClassicPedestrian:

    def __init__(self, controller: carla.Actor):
        self.controller = controller
        self.state = "idle"

    def update(self):
        if self.state == "idle":
            world = self.controller.get_world()
            self.destination = world.get_random_location_from_navigation()

            # should start by 1 tick after the creation of actor
            self.controller.start()
            self.controller.go_to_location(self.destination)
            self.controller.set_max_speed(
                float(self.controller.parent.attributes["speed"]))

            self.state = "acting"

        elif self.state == "acting":
            # TODO: stop and transfer to idle when arrival
            pass


class BevSpectator:

    def __init__(self, actor: carla.Actor):
        self.spectator = None
        self.hero = actor
        world = self.hero.get_world()
        self.spectator = world.get_spectator()

    def update(self):
        vehicle_transform = self.hero.get_transform()
        spectator_transform = carla.Transform(
            vehicle_transform.location + carla.Location(x=0, y=0, z=50),
            carla.Rotation(pitch=-90, yaw=0, roll=0)
        )
        self.spectator.set_transform(spectator_transform)
