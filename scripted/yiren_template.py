from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy
from collections import deque

from pysc2.agents import base_agent
from pysc2.lib import actions, features, units
from pysc2.env.sc2_env import Race

from pysc2.static_data.unit_dependency import DATA as _UnitDependency
from pysc2.static_data.unit_numeric import DATA as _UnitNumeric

FUNCTIONS = actions.FUNCTIONS

class GeneralAgent(base_agent.BaseAgent):
  MINERAL_TYPES = [ units.Neutral.MineralField, units.Neutral.MineralField750
                  , units.Neutral.LabMineralField, units.Neutral.LabMineralField750
                  , units.Neutral.RichMineralField, units.Neutral.RichMineralField750
                  #, units.Neutral.PurifierMineralField, units.Neutral.PurifierMineralField750
                  #, units.Neutral.PurifierRichMineralField, units.Neutral.PurifierRichMineralField750
                  ]
  VESPENE_TYPES = [ units.Neutral.VespeneGeyser
                  , units.Neutral.SpacePlatformGeyser
                  , units.Neutral.RichVespeneGeyser
                  #, units.Neutral.PurifierVespeneGeyser
                  #, ProtossVespeneGeyser
                  #, ShakurasVespeneGeyser
                  ]
  RACE_ID = 0
  WORKER_TYPE = None
  GAS_PLANT_TYPE = None
  TOWNHALL_TYPES = []

  def __init__(self):
    super(GeneralAgent, self).__init__()
    self.reset()


  def reset(self):
    # game constants, detected when the game starts
    self.ViewportSize = (0, 0)
    self.ScreenSize = (0, 0)
    self.MinimapSize = (0, 0)
    self.CameraOffset = None
    self.FirstViewport = None
    # useful hidden states    
    self._current_camera = None
    self._expected_selected = None
    # useful hidden states for strategy
    self._world_coordinate = {}
    self._structures = {}
    self._neutral_regions = set()
    # scheduled
    self._scheduled_actions_on_camera = {}
    self._scheduled_actions_on_unit = {}


  def calculate_world_relative_coordinate(self, local_coordinate1, local_coordinate2):
    (camera_minimap1, location_screen1) = local_coordinate1
    (camera_minimap2, location_screen2) = local_coordinate2
    viewport_size = self.ViewportSize
    screen_size = self.ScreenSize
    relative_minimap = (camera_minimap2[0]-camera_minimap1[0], camera_minimap2[1]-camera_minimap1[1])
    # offset_screen = screen_size / viewport_size * offset_minimap
    relative_screen = ( int(round(screen_size[0]*relative_minimap[0]/viewport_size[0])), int(round(screen_size[1]*relative_minimap[1]/viewport_size[1])))
    relative_world = ( location_screen2[0]+relative_screen[0]-location_screen1[0], location_screen2[1]+relative_screen[1]-location_screen1[1])
    return relative_world


  def calculate_world_absolute_coordinate(self, local_coordinate):
    viewport_size = self.ViewportSize
    camera_offset = self.CameraOffset
    viewport_center = (int(math.floor((viewport_size[0]+1)/2)), int(math.floor((viewport_size[1]+1)/2)))
    camera_minimap = ((viewport_center[0]-camera_offset[0]), (viewport_center[1]-camera_offset[1]))
    return self.calculate_world_relative_coordinate((camera_minimap, (0, 0)), local_coordinate)


  def _schedule_job(self, camera_minimap, unit_type_id, function_call_arguments, preemptive=False):
    if unit_type_id is None or 0 == unit_type_id:
      if camera_minimap not in self._scheduled_actions_on_camera:
        self._scheduled_actions_on_camera[camera_minimap] = deque([function_call_arguments])
      else:
        if preemptive:
          self._scheduled_actions_on_camera[camera_minimap].appendleft(function_call_arguments)
        else:
          self._scheduled_actions_on_camera[camera_minimap].append(function_call_arguments)
    else:
      if camera_minimap not in self._scheduled_actions_on_unit:
        self._scheduled_actions_on_unit[camera_minimap] = {unit_type_id: deque([function_call_arguments])}
      elif unit_type_id not in self._scheduled_actions_on_unit[camera_minimap]:
        self._scheduled_actions_on_unit[camera_minimap][unit_type_id] = deque([function_call_arguments])
      else:
        if preemptive:
          self._scheduled_actions_on_unit[camera_minimap][unit_type_id].appendleft(function_call_arguments)
        else:
          self._scheduled_actions_on_unit[camera_minimap][unit_type_id].append(function_call_arguments)

      
  @classmethod
  def calculate_distance_square(cls, p1, p2):
    (x1, y1) = p1
    (x2, y2) = p2
    dx = x2-x1
    dy = y2-y1
    return dx*dx + dy*dy


  @classmethod  
  def get_coordinates(cls, mask):
    """Mask should be a set of bools from comparison with a feature layer."""
    y, x = mask.nonzero()
    return list(zip(x, y))


  @classmethod
  def aggregate_points(cls, mask, neighbor_distance_squre):
    point_list = cls.get_coordinates(mask)
    grouped_points_serial = 0
    grouped_points = {}
    for new_p in point_list:
      related_serial = set()
      for exist_serial in grouped_points.keys():
        for old_p in grouped_points[exist_serial]:
          if cls.calculate_distance_square(old_p, new_p) <= neighbor_distance_squre:
            related_serial.add(exist_serial)
            break
      grouped_points_serial += 1
      grouped_points[grouped_points_serial] = set([new_p])
      for exist_serial in related_serial:
        grouped_points[grouped_points_serial] |= grouped_points[exist_serial]
        del grouped_points[exist_serial]
    return grouped_points


  @classmethod
  def get_locations_screen(cls, mask):
    grouped_points = cls.aggregate_points(mask, 2)
    return_locs = []
    for exist_serial in grouped_points.keys():
      if len(grouped_points[exist_serial]) > 0:
        top, left, bottom, right = None, None, None, None
        for (x, y) in grouped_points[exist_serial]:
          if left is None or x < left:
            left = x
          if right is None or x > right:
            right = x
          if top is None or y < top:
            top = y
          if bottom is None or y > bottom:
            bottom = y
        center = (int(round((left+right)/2)), int(round((top+bottom)/2)))
        #return_locs.append( (center, (left,top), (right,bottom) ) )
        return_locs.append( center )
    return return_locs


  def _get_unit_screen(self, obs, player_role, unit_type_list):
    return_locs = []
    player_relative = obs.observation.feature_screen.player_relative
    player_relative_mask = (player_relative == player_role)
    unit_type = obs.observation.feature_screen.unit_type
    for unit_type_id in unit_type_list:
      unit_type_mask = (unit_type == unit_type_id)
      my_unit_mask = numpy.logical_and(player_relative_mask, unit_type_mask)
      return_locs.extend(self.get_locations_screen(my_unit_mask))
    return return_locs


  def _get_my_unit_screen(self, obs, unit_type_list):
    return self._get_unit_screen(obs, features.PlayerRelative.SELF, unit_type_list)


  def _get_my_townhall_screen(self, obs):
    return self._get_my_unit_screen(obs, self.TOWNHALL_TYPES)


  def _execute_moving_camera(self, obs, camera_minimap):
    if self._current_camera != camera_minimap:
      self._current_camera = camera_minimap
      return FUNCTIONS.move_camera(camera_minimap)


  def _execute_training_army(self, obs, army_type_id):
    for dependency in _UnitDependency[army_type_id]:      
      action_id = dependency['perform']
      if action_id in obs.observation.available_actions:
        return actions.FunctionCall.init_with_validation(action_id, ['now'])
    return FUNCTIONS.no_op()

    
  def _execute_training_worker(self, obs):
    if type(self) == GeneralAgent:
      return FUNCTIONS.no_op()
    return self._execute_training_army(obs, self.WORKER_TYPE)


  def _execute_training_worker_from_townhall(self, obs):
    return FUNCTIONS.no_op()

      
  def _execute_scheduled_actions(self, obs, scheduled_actions):
    while len(scheduled_actions) > 0:
      next_action = scheduled_actions.pop()
      if isinstance(next_action[0], int):
        if next_action[0] == FUNCTIONS.move_camera.id:
          if self._current_camera != next_action[1][0]:
            self._current_camera = next_action[1][0]
            return actions.FunctionCall.init_with_validation(next_action[0], next_action[1])
        elif next_action[0] in obs.observation.available_actions:
          return actions.FunctionCall.init_with_validation(next_action[0], next_action[1])
      elif isinstance(next_action[0], str):
        method = getattr(next_action[1], next_action[0])
        return method(obs, *next_action[2])
    return FUNCTIONS.no_op()
    
    
  def _main_step(self, obs):
    # 
    #視窗切換控制:
    #1. 部隊編隊(正規/偷襲)
    #2. 無編隊之偵察/蓋分基地之工兵
    #3. 發現敵情
    #4. 生兵建築(軍營/主堡) : 沒有軍隊被打時/撤退中/
    #5. 建築或研發科技 : 沒有敵情才看一下

    #new_queue: 新的單次排程
    #old_queue: 舊的循環排程
    #如果有新的就從新的拿，沒有就從舊的拿，做完有需要再塞回舊佇列。
    #
    # 檢查敵情：
    # 在小地圖上確定看得到的敵方單位 
    # obs.observation.feature_minimap.visibility_map == features.Visibility.VISIBLE
    # obs.observation.feature_minimap.player_relative == features.PlayerRelative.ENEMY
    # 需要記錄與更新敵方建築的 minimap 的位置資訊（何時從小地圖發現，何時切換過鏡頭去看過）
    # 看過的部份，如果敵方單位都不動，可以暫時不再看
    # 看過的部份，如果我方單位是安全且不發動攻擊的，可以暫時不看
    # 看過的部份，要記錄
    # obs.observation.feature_screen.player_id
    # obs.observation.feature_screen.unit_type
    # obs.observation.feature_screen.creep

    # 如果有想要選到的單位
    #     目前實際選到的單位 obs.observation.single_select/obs.observation.multi_select
    #     如果實際選到的單位 和 想要選到的單位 一致
    #         檢查有無鏡頭的排程
    # 該單位要做的事已經做完，想要選到的單位設為 None
    # 如果沒有敵情/交戰中
    # 是否出兵去攻擊/偷襲/騷擾
    # 是否偵查
    # 是否生兵
    # 是否蓋新建築
    # 是否升級科技
    # 是否擴張分基地
    # 是否多採礦
    # 檢查新建築是否完成
    # 檢查科技是否升級完成
    ready_function_call = FUNCTIONS.no_op()
    if self._current_camera is not None:
      if self._current_camera in self._scheduled_actions_on_unit:
        scheduled_actions_on_unit = self._scheduled_actions_on_unit[self._current_camera]

        if self._expected_selected is not None:
          really_selected = False
          count_multi_select = len(obs.observation.multi_select)
          if count_multi_select > 0:
            if obs.observation.multi_select[0].unit_type == self._expected_selected:
              really_selected = True
          elif len(obs.observation.single_select) > 0 and obs.observation.single_select[0].unit_type == self._expected_selected:
            really_selected = True
          if really_selected:
            if self._expected_selected in scheduled_actions_on_unit:
              schedule_actions = scheduled_actions_on_unit[self._expected_selected]
              ready_function_call = self._execute_scheduled_actions(obs, schedule_actions)
      if ready_function_call.function == FUNCTIONS.no_op.id:
        if self._current_camera in self._scheduled_actions_on_camera:
          schedule_actions = self._scheduled_actions_on_camera[self._current_camera]
          ready_function_call = self._execute_scheduled_actions(obs, schedule_actions)
    return ready_function_call

   
  def _get_neutral_regions_minimap(self, obs):
    player_relative = obs.observation.feature_minimap.player_relative
    grouped_points = self.aggregate_points(player_relative == features.PlayerRelative.NEUTRAL, 5)
    return_locs = []
    for exist_serial in grouped_points.keys():
      points_count = len(grouped_points[exist_serial])
      if points_count > 0:
        x = []
        y = []
        for p in grouped_points[exist_serial]:
          x.append(p[0])
          y.append(p[1])
        x_arr = numpy.array(x)
        y_arr = numpy.array(y)
        center_location = [int(round(x_arr.mean())), int(round(y_arr.mean()))]
        #loop = 3
        #while loop>0:
        #  expect_point_count = 0
        #  sum_x = 0
        #  sum_y = 0
        #  for index in range(points_count):
        #    distance = self.calculate_distance_square([x[index], y[index]], center_location)
        #    if distance>0:
        #      sqrt_distance = math.sqrt(distance)
        #      if distance < 8:
        #        sum_x += (center_location[0]-x[index]) / sqrt_distance * 6 +x[index]
        #        sum_y += (center_location[1]-y[index]) / sqrt_distance * 6 +y[index]
        #        expect_point_count += 1
        #      elif distance >= 49:
        #        sum_x += (center_location[0]-x[index]) / sqrt_distance * 5 +x[index]
        #        sum_y += (center_location[1]-y[index]) / sqrt_distance * 5 +y[index]
        #        expect_point_count += 1         
        #  if expect_point_count > 0:
        #    center_location[0] = int(sum_x / expect_point_count)
        #    center_location[1] = int(sum_y / expect_point_count)
        #    loop -= 1
        #  else:  
        #    break
        return_locs.append(tuple(center_location))
    return return_locs


  def _game_start(self, obs):
    ready_function_call = FUNCTIONS.no_op()
    owner = obs.observation.player.player_id
    townhall_location_list = self._get_my_townhall_screen(obs)
    self._world_coordinate[owner] = {}
    # 中立礦區的計算，等生完第一隻工兵再做
    #self._neutral_regions = set(self._get_neutral_regions_minimap(obs))
    if 1 == len(townhall_location_list):
      unit_type_id = self.TOWNHALL_TYPES[0]
      townhall_location = townhall_location_list[0]
      world_absolute_coordinate = self.calculate_world_absolute_coordinate((self._current_camera, townhall_location))
      self._world_coordinate[owner][unit_type_id] = [townhall_location]
      self._structures[world_absolute_coordinate] = {'owner': owner, 'unit_type':unit_type_id}
      self._expected_selected = unit_type_id
      ready_function_call = FUNCTIONS.select_point('select', townhall_location)
    return ready_function_call

  
  def step(self, obs):
    if type(self) == GeneralAgent:
      return FUNCTIONS.no_op()
    else:
      if self.CameraOffset is None:
        camera = obs.observation.feature_minimap.camera
        y, x = (camera == 1).nonzero()
        viewport = ((x.min(),y.min()), (x.max(),y.max()))
        if self.FirstViewport is None:
          self.FirstViewport = viewport
          self.ViewportSize =  (self.FirstViewport[1][0]-self.FirstViewport[0][0], self.FirstViewport[1][1]-self.FirstViewport[0][1])
          screen_height_map = obs.observation.feature_screen.height_map
          self.ScreenSize = (screen_height_map.shape[1], screen_height_map.shape[0])
          minimap_height_map = obs.observation.feature_minimap.height_map
          self.MinimapSize = (minimap_height_map.shape[1], minimap_height_map.shape[0])
          self._current_camera = (int(math.floor((self.FirstViewport[0][0]+self.FirstViewport[1][0]+1)/2)), int(math.floor((self.FirstViewport[0][1]+self.FirstViewport[1][1]+1)/2)))
        else:
          self.CameraOffset = ((viewport[0][0]-self.FirstViewport[0][0]), (viewport[0][1]-self.FirstViewport[0][1]))
          self._current_camera = ((self._current_camera[0]-self.CameraOffset[0]), (self._current_camera[1]-self.CameraOffset[1]))
          self._schedule_job(self._current_camera, None, ['_game_start', self, []])
        return FUNCTIONS.move_camera(self._current_camera)
      else:
        return self._main_step(obs)
        

IdleAgent1 = GeneralAgent
IdleAgent2 = GeneralAgent

class PracticeProtossAgent(GeneralAgent):
  RACE_ID = Race.protoss
  WORKER_TYPE = units.Protoss.Probe
  GAS_PLANT_TYPE = units.Protoss.Assimilator
  TOWNHALL_TYPES = [units.Protoss.Nexus]
  def __init__(self):
    super(PracticeProtossAgent, self).__init__()

    
  def reset(self):
    super(PracticeProtossAgent, self).reset()


  def _execute_training_worker_from_townhall(self, obs):
    return self._execute_training_worker(obs)


  def _game_start(self, obs):
    ready_function_call = super(PracticeProtossAgent, self)._game_start(obs)
    self._schedule_job(self._current_camera, self.TOWNHALL_TYPES[0], ['_execute_training_worker_from_townhall', self, []])
    return ready_function_call


  def do_step(self, obs):
    pass

class PracticeTerranAgent(GeneralAgent):
  RACE_ID = Race.terran
  WORKER_TYPE = units.Terran.SCV
  GAS_PLANT_TYPE = units.Terran.Refinery
  TOWNHALL_TYPES = [units.Terran.CommandCenter, units.Terran.OrbitalCommand, units.Terran.PlanetaryFortress]
 
  def __init__(self):
    super(PracticeTerranAgent, self).__init__()

    
  def reset(self):
    super(PracticeTerranAgent, self).reset()


  def _execute_training_worker_from_townhall(self, obs):
    return self._execute_training_worker(obs)


  def _game_start(self, obs):
    ready_function_call = super(PracticeTerranAgent, self)._game_start(obs)
    self._schedule_job(self._current_camera, self.TOWNHALL_TYPES[0], ['_execute_training_worker_from_townhall', self, []])
    return ready_function_call

  def do_step(self, obs):
    pass


class PracticeZergAgent(GeneralAgent):
  RACE_ID = Race.zerg
  WORKER_TYPE = units.Zerg.Drone
  GAS_PLANT_TYPE = units.Zerg.Extractor
  TOWNHALL_TYPES = [units.Zerg.Hatchery, units.Zerg.Lair, units.Zerg.Hive]

  def __init__(self):
    super(PracticeZergAgent, self).__init__()

    
  def reset(self):
    super(PracticeZergAgent, self).reset()


  def _execute_selecting_larva(self, obs):
    if FUNCTIONS.select_larva.id in obs.observation.available_actions:
      self._expected_selected = units.Zerg.Larva
      return FUNCTIONS.select_larva()
    return FUNCTIONS.no_op()


  def _execute_training_army_from_larva(self, obs, army_type_id):
    self._expected_selected = None
    return super(PracticeZergAgent, self)._execute_training_army(obs, army_type_id)

  
  def _execute_training_army(self, obs, army_type_id):
    valid_actors = _UnitDependency[army_type_id][0]['actor'][0]
    if army_type_id == units.Zerg.Queen:
      if self._expected_selected in valid_actors:
        return super(PracticeZergAgent, self)._execute_training_army(obs, army_type_id)
    else:
      if self._expected_selected in valid_actors:
        return super(PracticeZergAgent, self)._execute_training_army(obs, army_type_id)
      elif self._expected_selected in _UnitDependency[units.Zerg.Larva][0]['actor'][0]:
        self._schedule_job(self._current_camera, units.Zerg.Larva, ['_execute_training_army_from_larva', self, [army_type_id]], True)
        return self._execute_selecting_larva(obs)
    return FUNCTIONS.no_op()
        
      
  def _execute_training_worker_from_townhall(self, obs):
    self._schedule_job(self._current_camera, units.Zerg.Larva, ['_execute_training_army_from_larva', self, [self.WORKER_TYPE]], True)
    return self._execute_selecting_larva(obs)


  def _game_start(self, obs):
    ready_function_call = super(PracticeZergAgent, self)._game_start(obs)
    self._schedule_job(self._current_camera, self.TOWNHALL_TYPES[0], ['_execute_training_worker_from_townhall', self, []])
    return ready_function_call


  def do_step(self, obs):
    pass


class PracticeRandomRaceAgent(base_agent.BaseAgent):
  ACTUAL_AGENT_CLS = { PracticeProtossAgent.TOWNHALL_TYPES[0]: PracticeProtossAgent
                     , PracticeTerranAgent.TOWNHALL_TYPES[0]: PracticeTerranAgent
                     , PracticeZergAgent.TOWNHALL_TYPES[0]: PracticeZergAgent 
                     }
  def reset(self):
    super(PracticeRandomRaceAgent, self).reset()
    self._actual_agent = None

  @classmethod
  def _detect_my_race(cls, obs):
    unit_type = obs.observation.feature_screen.unit_type
    bin_counts = numpy.bincount(unit_type.flatten())
    bin_counts[0] = 0
    townhall_type = numpy.argmax(bin_counts)
    return cls.ACTUAL_AGENT_CLS[townhall_type]

  def step(self, obs):
    super(PracticeRandomRaceAgent, self).step(obs)
    if self._actual_agent is not None:
      return self._actual_agent.step(obs)
    agent_cls = self._detect_my_race(obs)
    if agent_cls is not None:
      self._actual_agent = agent_cls()
      return self._actual_agent.step(obs)
    return FUNCTIONS.no_op()
  
