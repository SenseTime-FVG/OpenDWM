# -*- coding: utf-8 -*-
# @Time    : 2024/05/14
# @Author  : H.Liu
# @File    : obstacledistribution.py
# @Desc    : 前车道分布式障碍物场景（右侧）
# @input   : 输入接口在第270行，可调整障碍物数量/类型/和类别生成概率
#          : 'num_trashcan': [10,20], 障碍物数量将落在（10，20）之间
#          :  'objects' :['trash_can', 'garbage', 'trash_bin', 'barrel', 'box'], 障碍物类型
#          : 'weights' :[0.2, 0.5, 0.1, 0.1, 0.1]，每类别障碍物生成概率                                               
                                                

from __future__ import print_function
import random
import carla
import py_trees
import operator
from Initialization.Standardimport.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest, ScenarioTimeoutTest
from Initialization.Standardimport.scenariomanager.scenarioatomics.atomic_trigger_conditions import (DriveDistance, StandStill,InTriggerDistanceToVehicle, TimeoutRaiseException, InTriggerDistanceToNextIntersection)
from Initialization.Standardimport.basic_scenario import BasicScenario
from Explainablelibrary.explainable_utils import *
from Explainablelibrary.basic_desc import *
from Scenarios.Tinyscenarios import *
from Logger import *

OBSTACLE_TYPE_DICT = {
    # traffic obstacles
    'static.prop.garbage01': ['garbage'],  # 建筑垃圾
    'static.prop.garbage02': ['garbage'],
    'static.prop.garbage03': ['garbage'],
    'static.prop.garbage04': ['garbage'],
    'static.prop.garbage05': ['garbage'],
    'static.prop.trashbag':['garbage'],
    'static.prop.busstop ': ['bus_stop'],   # 公交车站
    'static.prop.constructioncone': ['construction'],   # 施工锥，用于标记施工区域或指引行人和车辆
    'static.prop.streetbarrier': ['street_barrier'],   # 用于限制车辆通行或指引行人。
    'static.prop.warningconstruction': ['street_barrier'],   # 用于限制车辆通行或指引行人。
    'static.prop.trafficcone01': ['traffic_barrier'],  # 交通锥，用于标记道路施工区域或指引交通
    'static.prop.trafficcone02': ['traffic_barrier'],  # 交通锥，用于标记道路施工区域或指引交通
    'static.prop.warningaccident' :['accident'],
    'walker.pedestrian.0004': ['workers'],
    'walker.pedestrian.0003': ['workers'],
    'walker.pedestrian.0015': ['workers'],
    'walker.pedestrian.0019': ['workers'],
    'walker.pedestrian.0016': ['workers'],
    'walker.pedestrian.0023': ['workers'],
    'static.prop.creasedbox02': ['creasedbox'],
    'static.prop.ironplank': ['plank'],
    'static.prop.trashcan01': ['trash_can'],  #垃圾桶
    'static.prop.trashcan03': ['trash_can'],  #垃圾桶
    'static.prop.bin':['trash_bin'],  #绿色垃圾桶
    'static.prop.barrel': ['barrel'],  #底座垃圾桶
    'static.prop.box01': ['box'],  #纸箱
    'static.prop.box02': ['box'],  #纸箱

}
TYPE_OBSTACLE_DICT = {}
for bp_obstacle_name, bp_obstacle_filters in OBSTACLE_TYPE_DICT.items():
    for bp_obstacle_filters in bp_obstacle_filters:
        if bp_obstacle_filters not in TYPE_OBSTACLE_DICT:
            TYPE_OBSTACLE_DICT[bp_obstacle_filters] = []
        TYPE_OBSTACLE_DICT[bp_obstacle_filters].append(bp_obstacle_name)



class ObstacleDistribution(BasicScenario):
    """
    Desc: TODO
    Special: 补充TODO的数据
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=180, uniad=False, interfuser=False, assign_distribution=None):
        self._world = world
        self._map = CarlaDataProvider.get_map()
        self.timeout = timeout

        self.starting_wp = self._map.get_waypoint(config.trigger_points[0].location)
        self.distance_to_junction = distance_to_next_junction(self.starting_wp)
        self.predefined_vehicle_length = 8
        self.actor_desc = []
        self.traffic_manager = None

        # For: tick autopilot
        self.front_obstacle = None
        self.front_obstacle_wp = None
        self.front_obstacle_loc = None
        self.initialized = False
        self.lane_changed = False
        self.avoid_sign = False 
        self.stage = 0
        self.data_tags = {}                        #关键指标日志记录
        self.navi_command = random.choice([0, 2])  #1: left, 2: straight
        self.start_save_sign = False               #是否开始记录
        # self.navigation_cmds = ['Straight']   
        self.obstacle_index = -1
        self.t = 0                  #开始避让的参考点
        self.max_t = 25             #完成避让的参考点(即到障碍物wp的距离)
        self.start_save_speed_seed = random.random()


        # 根据到路口的距离制定两种PlanA 和 Plan B
        if self.distance_to_junction > 150:
            self.can_spawn_distance = random.randint(60,80)
            message(f'planA:{self.can_spawn_distance}')
            self.init_speed = random.randint(35,45)
            self.data_tags['scenario_length'] = self.can_spawn_distance + 45
        else: 
            self.can_spawn_distance = random.randint(30,40)
            message(f'planB:{self.can_spawn_distance}')
            self.init_speed = random.randint(20,30)
            self.data_tags['scenario_length'] = self.can_spawn_distance + 30

        # For: 始终保持最右车道
        while True:
            right_lane = self.starting_wp.get_right_lane()
            if right_lane is None or right_lane.lane_type != carla.LaneType.Driving:
                break
            self.starting_wp = right_lane
        # For: Is success
        self.passed_lane_ids = []

        # For: V5 description
        self.carla_desc = CarlaDesc(world, ego_vehicles[0])

        # For: autopilot stage
        self.stages = {
            '#fix1': (KEEP_L, [], KEEP_S, [], ''),
            '#dynamic1': (KEEP_L, [], STOP, [], ''),
            '#dynamic2': (LEFT_C, [], KEEP_S, [], ''),
            '#fix2': (KEEP_L, [], ACC, [], ''),
            '#dynamic3': (LEFT_C, [], DEC, [], ''),
            '#fix3': (KEEP_L, [], ACC, [], ''),
            '#dynamic4': (RIGHT_C, [], KEEP_S, [], ''),
            '#fix4': (KEEP_L, [], ACC, [], ''),
            '#dynamic5': (RIGHT_C, [], DEC, [], ''),
            '#fix5': (KEEP_L, [], KEEP_S, [], ''),
            '#fix6': (KEEP_L, [], KEEP_S, [], ''),
        }


        super().__init__("ObstacleDistribution", ego_vehicles, config, world, randomize, debug_mode,
                         criteria_enable=criteria_enable, uniad=uniad, interfuser=interfuser, assign_distribution=assign_distribution)
        
    
    def get_data_tags(self):
        return self.data_tags

    def is_success(self):
        # Desc: Autopilot是否在当前车道上速度最终变为0且不存在变道行为
        if len(self.passed_lane_ids) == 0:
            return False, 'Autopilot未行驶'

        last_speed = None
        change_num = 0
        first_lane_id = -1
        for item in self.passed_lane_ids:
            if first_lane_id == -1:
                first_lane_id = item

            if last_speed is not None and item != last_speed:
                change_num += 1

            last_speed = item

        if last_speed == 0 and change_num == 0:
            return True, ''
        else:
            return False, 'Autopilot在当前车道上速度未最终变为0或者进行了车道变换'

    def interfuser_tick_autopilot(self):

        # 场景主车辆
        ego = self.ego_vehicles[0]

        # 自驾车辆的速度。首先获取车辆的速度向量（x和y分量），然后计算欧几里得距离（即速度的大小），乘以3.6将速度从米/秒转换为千米/小时，最后结果保留两位小数
        ego_speed = round(math.sqrt(ego.get_velocity().x ** 2 + ego.get_velocity().y ** 2) * 3.6, 2)

        frame_data = {'navi_command': self.navi_command}
        if self.initialized is False:
            # self._tf_set_ego_route(command)   # 设置自驾车辆的导航路线，这里基于self.navigation_cmds，为直行
            self._tf_set_ego_speed(self.init_speed)        # 初始速度
            self._tf_disable_ego_auto_lane_change()        # 禁用自动变道
            self._set_ego_autopilot()                      # 启动自驾
            self.initialized = True

        if not self.start_save_sign:
            ego_speed = round(math.sqrt(ego.get_velocity().x ** 2 + ego.get_velocity().y ** 2) * 3.6, 2)
            if self.start_save_speed_seed < 0.5:
                if ego_speed > self.init_speed - random.randint(2, 7):
                    self.start_save_sign = True
                    success(f'开始保存数据，自车速度：{round(ego_speed, 2)}km/h')
            else:
                if ego_speed > self.init_speed - 2:
                    self.start_save_sign = True
                    success(f'开始保存数据，自车速度：{round(ego_speed, 2)}km/h')
        
        # 主车当前位置
        cur_ego_loc = ego.get_location()
        cur_wp = self._map.get_waypoint(cur_ego_loc)


        # 如果当前路点既不是交叉口也不是路口：
        if not cur_wp.is_junction and not cur_wp.is_intersection:
            self.passed_lane_ids.append(cur_wp.lane_id)


        info = f'Speed: {int(ego_speed)} km/h '
        if self.obstacle_location is not None:
            distance = round(self.obstacle_location.distance(cur_ego_loc), 2)
            message(f'初始距离{distance}')


            # Plan A
            if self.distance_to_junction > 150:
                decrease_p1 = (40, self.init_speed)
                decrease_p2 = (15, 15)
                increase_p1 = (7, 15)
            # Plan B
            else: 
                decrease_p1 = (25, self.init_speed)
                decrease_p2 = (15, 15)
                increase_p1 = (7, 15)

            offset_m = cur_wp.lane_width / 3.0 # 偏航距离

            # 比较阶段1的距离，保持自车的初始速度
            if distance > decrease_p1[0]:
                
                self._tf_set_ego_speed(self.init_speed)
                
            # 减速阶段,距离介于阶段1和阶段2之间，根据距离调整速度公式计算目标速度，并设置自车速度。
            elif distance > decrease_p2[0] and self.avoid_sign is False:

                message(f'前方有车,减速')
                target_speed = (self.init_speed - decrease_p2[1]) / (decrease_p1[0] - decrease_p2[0]) * (distance - decrease_p2[0]) + decrease_p2[1]
                print(f'目标速度：{target_speed}')
                self._tf_set_ego_speed(target_speed)


            #绕行障碍物阶段
            elif self.avoid_sign is True and distance < increase_p1[0]: #decrease_p2[0]:
                        message(f'正在绕行')
                        self._tf_set_ego_offset(-offset_m)
                        self._tf_set_ego_force_go(100)
                        self._tf_set_ego_speed(decrease_p2[1])

                        # self.traffic_manager.ignore_vehicles_percentage(ego, 100)
            # 距离小于减速阶段2，且尚未设置避让标志，执行避让操作，设置速度为阶段2的速度，计算并设置避让偏移量，忽略的其他车辆，设置避让标志为True，并记录。
            else:
                if self.avoid_sign is False and self.t <= self.max_t:  #False
                    target_offset = -offset_m  # 目标偏移量                    
                    message(f'正在绕行')

                    current_offset = sigmoid(self.t,target_offset,self.max_t)
                    self._tf_set_ego_offset(current_offset)
                    self._tf_set_ego_speed(decrease_p2[1])
                    self.t += 1  # 更新距离
                    # self._tf_set_ego_offset(-offset_m)
                    self._tf_set_ego_force_go(100)
                    # self.traffic_manager.ignore_vehicles_percentage(ego, 100)
                    message(f'向左避让{current_offset}米')

                    if self.t == self.max_t:
                        self.avoid_sign = True  # True

                # self._tf_set_ego_speed(decrease_p2[1])
                
                # 绕过障碍后的加速阶段，根据距离调整速度公式计算目标速度，最终加速到初始速度。
                if self.avoid_sign is True and distance >= increase_p1[0]:#decrease_p2[0]:
                        message(f'恢复车道')
                        self._tf_set_ego_offset(0)
                        target_speed = (self.init_speed - decrease_p2[1]) / (decrease_p1[0] - increase_p1[0]) * (distance - increase_p1[0]) + decrease_p2[1]
                        # target_speed = (self.init_speed - decrease_p2[1]) / (decrease_p1[0] - decrease_p2[0]) * (distance - decrease_p2[0]) + decrease_p2[1]

                        self._tf_set_ego_speed(target_speed)
                        self._tf_set_ego_force_go(100)

        else:
            reason = 'because there are no special circumstances, so normal driving.'
            self._tf_set_ego_speed(self.init_speed)


        return self.start_save_sign, frame_data
    def _initialize_actors(self, config):

        #障碍物生成
        self.obstacle_location, self.data_tags = ahead_obstacle_scenario(self._world,self.starting_wp,self.can_spawn_distance,input_data_tags=self.data_tags,
                                    scene_cfg={ 'num_trashcan': [15,15],
                                               'objects' :['trash_can', 'garbage', 'trash_bin', 'barrel', 'box'],
                                                'weights' :[0.2, 0.5, 0.1, 0.1, 0.1]},   # 想要生成的垃圾桶数量
                                    gen_cfg={'name_prefix':'distribution_01'})  #默认distribution01， 按x，y轴方向均匀分布
        count = self.data_tags['obstacle_num']
        
        
        # For: 左侧交通流
        left_traffic_flow_scenario(
            self.starting_wp, self.other_actors, self.actor_desc,
            scene_cfg={'filters': '+hcy1', 'idp': 0.5, 'lanes_num': 4, 'skip_num': 1, 'forward_num': random.randint(2, 5), 'backward_num': random.randint(0, 20)}, gen_cfg={'name_prefix': 'left'}
        )
        # For: 对向交通流
        opposite_traffic_flow_scenario(
            self.starting_wp, self.other_actors, self.actor_desc,
            scene_cfg={'filters': '+hcy1', 'idp': 0.4, 'forward_num': random.randint(5, 8), 'backward_num': random.randint(0, 20)},
            gen_cfg={'name_prefix': 'opposite'}
        )
        self.traffic_manager = CarlaDataProvider.get_trafficmanager()
        for a_index, (actor, actor_desc) in enumerate(zip(self.other_actors, self.actor_desc)):
            if actor.type_id.startswith('vehicle'):
                if a_index == 0 or 'park' in actor_desc:
                    actor.apply_control(carla.VehicleControl(brake=1.0))
                    continue
                actor.set_autopilot(enabled=True, tm_port=CarlaDataProvider.get_traffic_manager_port())
                self.traffic_manager.update_vehicle_lights(actor, True)
                self.traffic_manager.set_desired_speed(actor, random.randint(30, 40))
                self.traffic_manager.auto_lane_change(actor, False)

    def _create_behavior(self):
        root = py_trees.composites.Parallel(name="ObstacleDistribution",
                                            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        c1 = py_trees.composites.Sequence(name="ObstacleDistribution_c1")
        c1.add_child(DriveDistance(self.ego_vehicles[0], self.data_tags['scenario_length']))
        root.add_child(c1)
        root.add_child(TimeoutRaiseException(300))
        root.add_child(CollisionTest(self.ego_vehicles[0], terminate_on_failure=True))
        root.add_child(InTriggerDistanceToNextIntersection(self.ego_vehicles[0], 10))
        return root

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = [ScenarioTimeoutTest(self.ego_vehicles[0], self.config.name)]
        if not self.route_mode:
            criteria.append(CollisionTest(self.ego_vehicles[0]))
        return criteria

    def __del__(self):
        """
        Remove all actors and traffic lights upon deletion
        """
        pass
    
#按概率生成障碍物类型
def obsbp_name_choice(objects,weights):
    '''可选的物体列表
    objects = ['trash_can', 'garbage', 'trash_bin', 'barrel', 'box','trashbag']
    # 对应的选择权重
    weights = [1, 3, 1, 2, 3]
    '''
    # 使用random.choices()按权重选择一个物体
    chosen_object = random.choices(objects, weights=weights, k=1)[0]
    return chosen_object



def choose_obsbp_name(filters):
    """
    Desc: 根据障碍物类型选择对应的blueprint
    @param filters: +x: 添加类型 -x: 排除类型，按顺序计算
    """
    # garbage: 道路垃圾，废弃物
    # bus_stop: 公交车站
    # construction： 施工
    # street_barrier: 道路指引
    # traffic_barrier: 交通障碍物
    # trash_can: 垃圾桶

    filters = [item.strip() for item in re.split(r'([+\-])', filters.strip()) if item.strip()]

    # 不能为单数
    if len(filters) % 2 != 0:
        return ""

    candidate_obsbp_names = []
    for index in range(0, len(filters), 2):
        op = filters[index]
        filter_type = filters[index + 1]
        if op == '+':
            candidate_obsbp_names.extend(TYPE_OBSTACLE_DICT[filter_type])
        elif op == '-':
            candidate_obsbp_names = list(set(candidate_obsbp_names) - set(TYPE_OBSTACLE_DICT[filter_type]))
        else:
            print(f'Error: {op} is not supported in blueprint choosing.')
            return ""

    if len(candidate_obsbp_names) == 0:
        print(f'Error: candidate_bp_names is empty.')
        return ""

    return random.choice(candidate_obsbp_names)

def gen_can_distribution(world, obstacle_spawn, num_trashcan,objects,weights,input_data_tags):
    '''
    pitch绕横轴（X轴）的旋转
    Yaw: 绕Z轴的旋转
    Roll 绕Y轴旋转
    '''
    blueprint_library = world.get_blueprint_library()
    # can_spawn = move_waypoint_forward(obstacle_spawn, random.randint(2, 3))
    x_point_list=[] #记录每个生成障碍物的横坐标
    y_point_list=[] #记录每个生成障碍物的纵坐标

    random_choice = random.randint(num_trashcan[0],num_trashcan[1])
    message(f'随机生成{random_choice}个障碍物')
    input_data_tags['obstacle_num']=random_choice

    for i in range(random_choice):
        obstacle_name = obsbp_name_choice(objects,weights)
        trashcan_bp = choose_obsbp_name(f'+{obstacle_name}')
        print(f"trashcan_bp : {trashcan_bp}\n")
        trashcan_blueprint = blueprint_library.find(trashcan_bp)

        # 生成均匀向右偏移的位置
        k_pitch_soffset = random.choice([2, 3])
        k_yaw_soffset = random.choice([2, 3])
        x_can_soffset = random.uniform(obstacle_spawn.lane_width/3 + 0.35, 5)  # 从1/3路宽到5米的水平偏移
        y_can_soffset = random.uniform(0, 8)  # 从0到8米的垂直偏移
        x_left_min = obstacle_spawn.transform.location
        pitch_can_soffset = random.choice([-90,90,0]) #障碍物站立/横倒
        yaw_can_soffset = random.uniform(0, 360)  #障碍物Z轴自旋角
        height = 0
        
        x_point_list.append(x_can_soffset)
        y_point_list.append(y_can_soffset)
        
        #若障碍物是小体积垃圾不需要旋转
        if obstacle_name == 'garbage':
            pitch_can_soffset = 0

        #若障碍物横倒，提高0.3米，防止与地面重叠
        if pitch_can_soffset == -90 or pitch_can_soffset == 90:
            height = 0.3
        else:
            height = 0

        # 设置障碍物(垃圾桶)的位置和旋转
        right_vector = obstacle_spawn.transform.rotation.get_right_vector() #右向单位向量
        forward_vector = obstacle_spawn.transform.rotation.get_forward_vector() #前向单位向量
        left_vector = right_vector * -1
        theta = math.radians(obstacle_spawn.transform.rotation.yaw)
    

        x_move_distance = x_can_soffset * right_vector  # 修改x偏移量的应用逻辑
        y_move_distance = y_can_soffset * forward_vector  # 修改y偏移量的应用逻辑

        road_right_location = obstacle_spawn.transform.location + x_move_distance + y_move_distance
        spawn_trashcan_point = carla.Transform(
                location=carla.Location(
                x=road_right_location.x, 
                y=road_right_location.y,
                z=obstacle_spawn.transform.location.z + height), #保持和汽车高度平面相同
            rotation=carla.Rotation(
                pitch=obstacle_spawn.transform.rotation.pitch + (-1) ** (k_pitch_soffset) * pitch_can_soffset,
                yaw=obstacle_spawn.transform.rotation.yaw + (-1) ** (k_yaw_soffset) * yaw_can_soffset,
                roll=obstacle_spawn.transform.rotation.roll) 
        )
        
        # 在仿真世界中生成障碍物，并禁用物理模拟
        while True:
            if obstacle_name in ('box', 'garbage'):
                trashcan = world.spawn_actor(trashcan_blueprint, spawn_trashcan_point)
                trashcan.set_simulate_physics(True)
                time.sleep(1)
                if two_wp_x_distance(trashcan,obstacle_spawn) <  0.2: #obstacle_spawn.lane_width/5:    #障碍物翻滚超出横坐标分布范围
                    trashcan.destroy()
                    message(f'删除X')
                elif two_wp_y_distance(trashcan,obstacle_spawn) > 15:    #障碍物翻滚超出纵坐标分布范围
                    trashcan.destroy()
                    message(f'删除y')
                               
            else: 
                trashcan = world.spawn_actor(trashcan_blueprint, spawn_trashcan_point)
                trashcan.set_simulate_physics(False)
            break
    
    distribution_area = (max(x_point_list) - min(y_point_list)) * (max(y_point_list) - min(y_point_list))  #记录分布面积
    message(f'障碍物分布面积:{distribution_area}')
    input_data_tags['Obstacle_distribution_area']= distribution_area
    return obstacle_spawn, input_data_tags

def two_wp_x_distance(actor,waypoint2):
    x1 = actor.get_transform().location.x
    x2 = waypoint2.transform.location.x
    
    # 计算沿X轴的距离
    x_distance = abs(x1 - x2)
    return x_distance

def two_wp_y_distance(waypoint1, waypoint2):
    y1 = waypoint1.get_transform().location.y
    y2 = waypoint2.transform.location.y
    
    # 计算沿X轴的距离
    y_distance = abs(y1 - y2)
    return y_distance

def gen_can_fix(world, obstacle_spawn):
    
    pass
def ahead_obstacle_scenario_distribution_01(world,wp, can_spawn_distance,input_data_tags,scene_cfg={}, gen_cfg={}):
    # Desc: 自动在当前车道的前方生成随机障碍物分布
    num_trashcan = scene_cfg.get('num_trashcan')
    objects = scene_cfg.get('objects')
    weights = scene_cfg.get('weights')
    obstacle_spawn = move_waypoint_forward(wp, can_spawn_distance)
 
    
    #生成障碍物分布
    gen_can_distribution(world,obstacle_spawn,num_trashcan,objects,weights,input_data_tags)
    return obstacle_spawn.transform.location, input_data_tags

def ahead_obstacle_scenario_distribution_02(world,wp, can_spawn_distance,scene_cfg={}, gen_cfg={}):
    # Desc: 在当前车道的前方生成固定障碍物分布

    obstacle_spawn = move_waypoint_forward(wp, can_spawn_distance)
    
    gen_can_fix(world,obstacle_spawn)

    return obstacle_spawn.transform.location
def ahead_obstacle_scenario(world,wp, can_spawn_distance,input_data_tags,scene_cfg={}, gen_cfg={}) :    
    # 检查gen_cfg字典中是否有name_prefix键，根据其值决定调用哪个函数
    name_prefix = gen_cfg.get('name_prefix')  # 默认值为distribution_01
    if name_prefix == 'distribution_01':
        return ahead_obstacle_scenario_distribution_01(world, wp, can_spawn_distance,input_data_tags,scene_cfg, gen_cfg)
    elif name_prefix == 'distribution_02':
        return ahead_obstacle_scenario_distribution_02(world, wp, can_spawn_distance, scene_cfg, gen_cfg)
    else:
        raise ValueError("Invalid generation configuration. 'name_prefix' must be 'distribution_01' or 'distribution_02'.")

def sigmoid(x,target_offset,max_t):
    # Sigmoid函数，a控制曲线的陡峭程度，b控制偏移范围
    a = 0.1  # 调整以控制平滑程度
    return 2 * target_offset / (1 + math.exp(-a * (x - max_t / 2))) - target_offset

