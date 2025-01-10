import carla
import numpy as np
import cv2
import time
import os

def build_projection_matrix(w, h, fov):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

def get_bounding_box_world_coords(bbox, transform):
    extent = bbox.extent
    vertices = [carla.Location(x=extent.x, y=extent.y, z=extent.z),
                carla.Location(x=-extent.x, y=extent.y, z=extent.z),
                carla.Location(x=extent.x, y=-extent.y, z=extent.z),
                carla.Location(x=-extent.x, y=-extent.y, z=extent.z),
                carla.Location(x=extent.x, y=extent.y, z=-extent.z),
                carla.Location(x=-extent.x, y=extent.y, z=-extent.z),
                carla.Location(x=extent.x, y=-extent.y, z=-extent.z),
                carla.Location(x=-extent.x, y=-extent.y, z=-extent.z)]
    world_coords = [transform.transform(vertex) for vertex in vertices]
    return world_coords

def world_to_camera(world_coords, camera_transform):
    camera_matrix = np.array(camera_transform.get_inverse_matrix()).reshape(4, 4)
    camera_coords = []
    for location in world_coords:
        loc = np.array([location.x, location.y, location.z, 1.0])
        transformed_loc = np.dot(camera_matrix, loc)
        if transformed_loc[3] != 0:  # 防止除以零
            transformed_loc /= transformed_loc[3]
        camera_coords.append(transformed_loc[:3])
    return camera_coords

def camera_to_image(camera_coords, camera_intrinsic_matrix):
    image_coords = []
    for coord in camera_coords:
        x, y, z = coord
        if z > 0:  # 确保只投影在前方的点
            u = camera_intrinsic_matrix[0, 0] * x / z + camera_intrinsic_matrix[0, 2]
            v = camera_intrinsic_matrix[1, 1] * y / z + camera_intrinsic_matrix[1, 2]
            image_coords.append((u, v))
        else:
            print(f"Point behind camera: {coord}")
    return image_coords

def draw_bounding_box(image, bbox_image_coords):
    image = np.copy(image)  # 确保图像是可写的
    h, w, _ = image.shape
    for i in range(len(bbox_image_coords)):
        for j in range(i + 1, len(bbox_image_coords)):
            p1 = (int(bbox_image_coords[i][0]), int(bbox_image_coords[i][1]))
            p2 = (int(bbox_image_coords[j][0]), int(bbox_image_coords[j][1]))
            # 确保绘制的点在图像范围内
            if 0 <= p1[0] < w and 0 <= p1[1] < h and 0 <= p2[0] < w and 0 <= p2[1] < h:
                cv2.line(image, p1, p2, (0, 255, 0), 2)
                print(f"Drawing line from {p1} to {p2}")
    return image

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.load_world('Town05')
    map = world.get_map()
    debug = world.debug
    settings = world.get_settings()
    settings.fixed_delta_seconds = None
    settings.synchronous_mode = False
    world.apply_settings(settings)
    
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('vehicle.*model3*')[0]
    
    spawn_point = world.get_map().get_spawn_points()[90]
    vehicle_transform = carla.Transform(spawn_point.location + carla.Location(x=50), spawn_point.rotation)
    vehicle = world.spawn_actor(vehicle_bp, vehicle_transform)
    
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '1920')
    camera_bp.set_attribute('image_size_y', '1080')
    camera_bp.set_attribute('fov', '120')
    
    camera_transform = carla.Transform(carla.Location(x=2.0, z=1.5), carla.Rotation(pitch=0))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    
    # 获取车辆位置和朝向
    vehicle_location = vehicle.get_location()
    vehicle_forward_vector = vehicle.get_transform().get_forward_vector()
    
    # 获取红绿灯对象
    traffic_lights = world.get_actors().filter('traffic.traffic_light')
    
    def is_in_front(vehicle_location, vehicle_forward_vector, object_location):
        relative_position = object_location - vehicle_location
        dot_product = relative_position.x * vehicle_forward_vector.x + relative_position.y * vehicle_forward_vector.y + relative_position.z * vehicle_forward_vector.z
        return dot_product > 0
    
    def get_first_traffic_light_in_front(vehicle_location, vehicle_forward_vector, traffic_lights):
        min_distance = float('inf')
        first_traffic_light = None
        for traffic_light in traffic_lights:
            if is_in_front(vehicle_location, vehicle_forward_vector, traffic_light.get_location()):
                distance = vehicle_location.distance(traffic_light.get_location())
                if distance < min_distance:
                    min_distance = distance
                    first_traffic_light = traffic_light
        return first_traffic_light
    
    traffic_light = get_first_traffic_light_in_front(vehicle_location, vehicle_forward_vector, traffic_lights)
    location = traffic_light.get_location()
    waypoint = map.get_waypoint(location, project_to_road=True, lane_type=carla.LaneType.Any)
    debug.draw_point(waypoint.transform.location + carla.Location(z=4.3), size=0.15, color=carla.Color(255, 255, 0), life_time=999)
    
    if traffic_light is None:
        print("No traffic light in front of the vehicle.")
        return
    
    bounding_box = traffic_light.bounding_box

    output_folder = './carla_output'
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    frame_count = 0
    
    def save_image(image):
        nonlocal frame_count
        image.convert(carla.ColorConverter.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        
        bbox_world_coords = get_bounding_box_world_coords(bounding_box, traffic_light.get_transform())
        print(f"Bounding box world coordinates: {bbox_world_coords}")
        
        bbox_camera_coords = world_to_camera(bbox_world_coords, camera.get_transform())
        print(f"Bounding box camera coordinates: {bbox_camera_coords}")
        
        camera_intrinsic_matrix = build_projection_matrix(image.width, image.height, float(camera_bp.get_attribute('fov').as_float()))
        print(f"Camera intrinsic matrix: {camera_intrinsic_matrix}")
        
        bbox_image_coords = camera_to_image(bbox_camera_coords, camera_intrinsic_matrix)
        print(f"Bounding box image coordinates: {bbox_image_coords}")
        
        if not bbox_image_coords:
            print("No valid image coordinates for bounding box.")
        
        image_with_bbox = draw_bounding_box(array, bbox_image_coords)
        
        image_filename = os.path.join(output_folder, f'traffic_light_with_bbox_frame_{frame_count:04d}.png')
        
        cv2.imwrite(image_filename, image_with_bbox)
        
        frame_count += 1
    
    camera.listen(lambda image: save_image(image))
    
    vehicle.apply_control(carla.VehicleControl(throttle=0.5))
    
    time.sleep(20)
    
    vehicle.apply_control(carla.VehicleControl(throttle=0.0))
    
    camera.stop()
    vehicle.destroy()
    camera.destroy()

if __name__ == '__main__':
    main()
