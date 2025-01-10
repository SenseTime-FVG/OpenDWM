

class PromptGen():

    def __init__(self, world, ego):

        self.world = world
        self.ego = ego

        self._weather = self.world.get_weather()

        sun_altitude_angle = self._weather.sun_altitude_angle
        if sun_altitude_angle >= 0:
            self.time = "daytime"
            self.weather = "sunny"
        else:
            self.time = "evening"
            self.weather = "cloudy"
        
        cloudiness = self._weather.cloudiness
        if cloudiness > 50:
            self.weather = "cloudy"
        
        precipitation = self._weather.precipitation
        if precipitation > 30:
            self.weather = "rainy"

        fog_density = self._weather.precipitation
        if fog_density > 50:
            self.weather = "foggy"
        
        map_name = self.world.get_map().name
        # TODO
        self.env = "urban street scene"

        
        


