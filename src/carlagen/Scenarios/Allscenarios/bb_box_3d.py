# import carla
# import math
# import random
# import time
# import queue
# import numpy as np
# import cv2

# client = carla.Client('localhost', 2000)
# client.set_timeout(10.0)
# world = client.load_world('Town05')
# # world  = client.get_world()
# bp_lib = world.get_blueprint_library()

# # spawn vehicle
# vehicle_bp =bp_lib.find('vehicle.lincoln.mkz_2020')
# # Get the map spawn points
# spawn_points = world.get_map().get_spawn_points()
# # vehicle = world.try_spawn_actor(vehicle_bp, )
# vehicle = world.spawn_actor(vehicle_bp,random.choice(spawn_points))
# # spawn camera

# camera_bp = bp_lib.find('sensor.camera.rgb')
# camera_init_trans = carla.Transform(carla.Location(z=2))
# camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)
# vehicle.set_autopilot(True)

# # Set up the simulator in synchronous mode
# settings = world.get_settings()
# settings.synchronous_mode = True # Enables synchronous mode
# settings.fixed_delta_seconds = 0.05
# world.apply_settings(settings)



# # Create a queue to store and retrieve the sensor data
# image_queue = queue.Queue()
# camera.listen(image_queue.put)

# def build_projection_matrix(w, h, fov, is_behind_camera=False):
#     focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
#     K = np.identity(3)

#     if is_behind_camera:
#         K[0, 0] = K[1, 1] = -focal
#     else:
#         K[0, 0] = K[1, 1] = focal

#     K[0, 2] = w / 2.0
#     K[1, 2] = h / 2.0
#     return K


# def get_image_point(loc, K, w2c):
#         # Calculate 2D projection of 3D coordinate

#         # Format the input coordinate (loc is a carla.Position object)
#         point = np.array([loc.x, loc.y, loc.z, 1])
#         # transform to camera coordinates
#         point_camera = np.dot(w2c, point)

#         # New we must change from UE4's coordinate system to an "standard"
#         # (x, y ,z) -> (y, -z, x)
#         # and we remove the fourth componebonent also
#         point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

#         # now project 3D->2D using the camera matrix
#         point_img = np.dot(K, point_camera)
#         # normalize
#         point_img[0] /= point_img[2]
#         point_img[1] /= point_img[2]

# # Get the world to camera matrix
# world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

# # Get the attributes from the camera
# image_w = camera_bp.get_attribute("image_size_x").as_int()
# image_h = camera_bp.get_attribute("image_size_y").as_int()
# fov = camera_bp.get_attribute("fov").as_float()

# # Calculate the camera projection matrix to project from 3D -> 2D
# K = build_projection_matrix(image_w, image_h, fov)
# K_b = build_projection_matrix(image_w, image_h, fov, is_behind_camera=True)

# # actor.get_world_vertices(actor.get_transform())


# # Retrieve all bounding boxes for traffic lights within the level
# bounding_box_set = world.get_level_bbs(carla.CityObjectLabel.TrafficLight)

# # Filter the list to extract bounding boxes within a 50m radius
# # nearby_bboxes = []
# # for bbox in bounding_box_set:
# #     if bbox.location.distance(actor.get_transform().location) < 50:
# #         nearby_bboxes
     

# # Set up the set of bounding boxes from the level
# # We filter for traffic lights and traffic signs
# bounding_box_set = world.get_level_bbs(carla.CityObjectLabel.TrafficLight)
# bounding_box_set.extend(world.get_level_bbs(carla.CityObjectLabel.TrafficSigns))

# # Remember the edge pairs
# edges = [[0,1], [1,3], [3,2], [2,0], [0,4], [4,5], [5,1], [5,7], [7,6], [6,4], [6,2], [7,3]]


# # Retrieve the first image
# world.tick()
# image = image_queue.get()

# # Reshape the raw data into an RGB array
# img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4)) 

# # Display the image in an OpenCV display window
# cv2.namedWindow('ImageWindowName', cv2.WINDOW_AUTOSIZE)
# cv2.imshow('ImageWindowName',img)
# cv2.waitKey(1)



# while True:
#     # Retrieve and reshape the image
#     world.tick()
#     image = image_queue.get()

#     img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))

#     # Get the camera matrix 
#     world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

#     for bb in bounding_box_set:

#         # Filter for distance from ego vehicle
#         if bb.location.distance(vehicle.get_transform().location) < 50:

#             # Calculate the dot product between the forward vector
#             # of the vehicle and the vector between the vehicle
#             # and the bounding box. We threshold this dot product
#             # to limit to drawing bounding boxes IN FRONT OF THE CAMERA
#             forward_vec = vehicle.get_transform().get_forward_vector()
#             ray = bb.location - vehicle.get_transform().location

#             if forward_vec.dot(ray) > 0:
#                 # Cycle through the vertices
#                 verts = [v for v in bb.get_world_vertices(carla.Transform())]
#                 for edge in edges:
#                     # Join the vertices into edges
#                     p1 = get_image_point(verts[edge[0]], K, world_2_camera)
#                     p2 = get_image_point(verts[edge[1]],  K, world_2_camera)
#                     # Draw the edges into the camera output
#                     cv2.line(img, (int(p1[0]),int(p1[1])), (int(p2[0]),int(p2[1])), (0,0,255, 255), 1)

#     # Now draw the image into the OpenCV display window
#     cv2.imshow('ImageWindowName',img)
#     # Break the loop if the user presses the Q key
#     if cv2.waitKey(1) == ord('q'):
#         break

# # Close the OpenCV display window when the game loop stops
# cv2.destroyAllWindows()


import carla
import math
import random
import time
import queue
import numpy as np
import cv2

client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.load_world('Town05')
# world  = client.get_world()
bp_lib = world.get_blueprint_library()

# Spawn vehicle
vehicle_bp = bp_lib.find('vehicle.lincoln.mkz_2020')
# Get the map spawn points
spawn_points = world.get_map().get_spawn_points()
vehicle = world.spawn_actor(vehicle_bp, random.choice(spawn_points))
# Spawn camera
camera_bp = bp_lib.find('sensor.camera.rgb')
camera_init_trans = carla.Transform(carla.Location(z=2))
camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)
vehicle.set_autopilot(True)

# Set up the simulator in synchronous mode
settings = world.get_settings()
settings.synchronous_mode = True  # Enables synchronous mode
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)

# Create a queue to store and retrieve the sensor data
image_queue = queue.Queue()
camera.listen(image_queue.put)

def build_projection_matrix(w, h, fov, is_behind_camera=False):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)

    if is_behind_camera:
        K[0, 0] = K[1, 1] = -focal
    else:
        K[0, 0] = K[1, 1] = focal

    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

def get_image_point(loc, K, w2c):
    # 计算3D坐标的2D投影

    # 格式化输入坐标（loc是一个carla.Position对象）
    point = np.array([loc.x, loc.y, loc.z, 1])
    # 转换为摄像机坐标
    point_camera = np.dot(w2c, point)

    # 检查点是否在摄像机后面
    if point_camera[2] <= 0:
        return None

    # 将UE4的坐标系转换为标准坐标系 (x, y, z) -> (y, -z, x)
    point_camera = np.array([point_camera[1], -point_camera[2], point_camera[0]])

    # 使用摄像机矩阵将3D投影转换为2D
    point_img = np.dot(K, point_camera)
    # 归一化
    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]

    # 检查点是否在图像边界内
    if 0 <= point_img[0] < K[0, 2] * 2 and 0 <= point_img[1] < K[1, 2] * 2:
        return point_img[:2]
    return None

# 获取世界到摄像机的变换矩阵
world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

# 获取摄像机属性
image_w = camera_bp.get_attribute("image_size_x").as_int()
image_h = camera_bp.get_attribute("image_size_y").as_int()
fov = camera_bp.get_attribute("fov").as_float()

# 计算摄像机投影矩阵，用于从3D投影到2D
K = build_projection_matrix(image_w, image_h, fov)
K_b = build_projection_matrix(image_w, image_h, fov, is_behind_camera=True)

# 获取交通灯的所有边界框
bounding_box_set = world.get_level_bbs(carla.CityObjectLabel.TrafficLight)
bounding_box_set.extend(world.get_level_bbs(carla.CityObjectLabel.TrafficSigns))

# 记住边缘对
edges = [[0, 1], [1, 3], [3, 2], [2, 0], [0, 4], [4, 5], [5, 1], [5, 7], [7, 6], [6, 4], [6, 2], [7, 3]]

# 获取第一张图像
world.tick()
image = image_queue.get()

# 将原始数据重塑为RGB数组
img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))

# 在OpenCV显示窗口中显示图像
cv2.namedWindow('ImageWindowName', cv2.WINDOW_AUTOSIZE)
cv2.imshow('ImageWindowName', img)
cv2.waitKey(1)

while True:
    # 获取并重塑图像
    world.tick()
    image = image_queue.get()

    img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))

    # 获取摄像机矩阵 
    world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

    for bb in bounding_box_set:

        # 过滤与自车的距离
        if bb.location.distance(vehicle.get_transform().location) < 50:

            # 计算自车前向向量与自车和边界框之间的向量的点积。
            # 我们将这个点积设为阈值，以限制只绘制摄像头前方的边界框
            forward_vec = vehicle.get_transform().get_forward_vector()
            ray = bb.location - vehicle.get_transform().location

            if forward_vec.dot(ray) > 0:
                # 循环遍历顶点
                verts = [v for v in bb.get_world_vertices(carla.Transform())]
                for edge in edges:
                    # 将顶点连接成边
                    p1 = get_image_point(verts[edge[0]], K, world_2_camera)
                    p2 = get_image_point(verts[edge[1]], K, world_2_camera)
                    if p1 is not None and p2 is not None:
                        # 将边绘制到摄像头输出
                        cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 0, 255, 255), 1)

    # 现在将图像绘制到OpenCV显示窗口中
    cv2.imshow('ImageWindowName', img)
    # 如果用户按下Q键，则退出循环
    if cv2.waitKey(1) == ord('q'):
        break

# 在游戏循环停止时关闭OpenCV显示窗口
cv2.destroyAllWindows()
