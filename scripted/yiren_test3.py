from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pysc2.agents import base_agent
from pysc2.lib import actions, features, units
from pysc2.custom import unit_numeric as _Numeric, unit_dependency as _Dependency, unit_logic as _Logic
from pysc2.env.sc2_env import Race
from scipy.ndimage import morphology
import numpy
import math
import random

FUNCTIONS = actions.FUNCTIONS

class GeneralAgentCore(object):
  def __init__(self, actual_agent):
    super(GeneralAgentCore, self).__init__()
    self._mineral_source = [ units.Neutral.MineralField, units.Neutral.MineralField750
                           , units.Neutral.LabMineralField, units.Neutral.LabMineralField750
                           , units.Neutral.RichMineralField, units.Neutral.RichMineralField750
                           #, units.Neutral.PurifierMineralField, units.Neutral.PurifierMineralField750
                           #, units.Neutral.PurifierRichMineralField, units.Neutral.PurifierRichMineralField750
                           ]
    self._vespene_source = [ units.Neutral.VespeneGeyser
                           , units.Neutral.SpacePlatformGeyser
                           , units.Neutral.RichVespeneGeyser
                           #, units.Neutral.PurifierVespeneGeyser
                           #, ProtossVespeneGeyser
                           #, ShakurasVespeneGeyser
                           ]
    self._actual_agent = actual_agent
    self.reset()


  def reset(self):
    self._game_step_counter = 0
    self._myRace = self._actual_agent.RACE_ID
    self._myWorker = self._actual_agent.WORKER_TYPE
    self._myCastle = self._actual_agent.CASTLE_TYPE
    self._myGasPlant = self._actual_agent.GAS_PLANT_TYPE
    
    self._init_viewport = None
    self._viewport_size = None
    self._screen_size = None
    self._minimap_size = None
    self._camera_offset = None
    # camera and selected based scheduling
    self._current_camera = None
    self._current_selected_point = None
    self._last_camera = []
    self._last_selected_point = {}
    self._scheduled_actions_by_screen = {}
    self._scheduled_actions_by_minimap = {}
    # game status
    self._locations = {}
    self._structures = {}
    self._neutral_regions = set()
    self._occupied_neutral_regions = {}
    self._my_infrastructures = []
    # need polling when idle
    self._my_waiting = {}      # schedule when busy
    self._my_producing = []    # schedule when poor
    self._my_building = {}     # executed the 'build' actions and wait
    # test debug

  @classmethod
  def calculate_distance_square(cls, p1, p2):
    (x1, y1) = p1
    (x2, y2) = p2
    dx = x2-x1
    dy = y2-y1
    return dx*dx + dy*dy

  def calculate_world_relative_coordinate(self, local_coordinate1, local_coordinate2):
    (camera_minimap1, location_screen1) = local_coordinate1
    (camera_minimap2, location_screen2) = local_coordinate2
    viewport_size = self._viewport_size
    screen_size = self._screen_size
    relative_minimap = (camera_minimap2[0]-camera_minimap1[0], camera_minimap2[1]-camera_minimap1[1])
    # offset_screen = screen_size / viewport_size * offset_minimap
    relative_screen = ( int(round(screen_size[0]*relative_minimap[0]/viewport_size[0])), int(round(screen_size[1]*relative_minimap[1]/viewport_size[1])))
    relative_world = ( location_screen2[0]+relative_screen[0]-location_screen1[0], location_screen2[1]+relative_screen[1]-location_screen1[1])
    return relative_world
  
  def calculate_world_distance_square(self, local_coordinate1, local_coordinate2):
    relative_world = self.calculate_world_relative_coordinate(local_coordinate1, local_coordinate2)
    return self.calculate_distance_square( (0,0), relative_world)
  
  def calculate_world_absolute_coordinate(self, local_coordinate):
    viewport_size = self._viewport_size
    viewport_center = (int(math.floor((viewport_size[0]+1)/2)), int(math.floor((viewport_size[1]+1)/2)))
    camera_minimap = ((viewport_center[0]-self._camera_offset[0]), (viewport_center[1]-self._camera_offset[1]))
    return self.calculate_world_relative_coordinate((camera_minimap, (0, 0)), local_coordinate)

  def calculate_local_coordinate(self, world_coordinate):
    screen_size = self._screen_size
    estimative_referenced_world = (world_coordinate[0]-int(math.floor((screen_size[0]+1)/2)), world_coordinate[1]-int(math.floor((screen_size[1]+1)/2)))
    viewport_size = self._viewport_size
    minimap_offset_x = int(math.floor(estimative_referenced_world[0]*viewport_size[0]/screen_size[0]))
    minimap_offset_y = int(math.floor(estimative_referenced_world[1]*viewport_size[1]/screen_size[1]))
    boundary = (self._minimap_size[0]-viewport_size[0], self._minimap_size[1]-viewport_size[1])
    if minimap_offset_x < 0:
      minimap_offset_x = 0
    elif minimap_offset_x > boundary[0]:
      minimap_offset_x = boundary[0]
    if minimap_offset_y < 0:
      minimap_offset_y = 0
    elif minimap_offset_y > boundary[1]:
      minimap_offset_y = boundary[1]
    viewport_center = (int(math.floor((viewport_size[0]+1)/2)), int(math.floor((viewport_size[1]+1)/2)))
    origin_minimap = ((viewport_center[0]-self._camera_offset[0]), (viewport_center[1]-self._camera_offset[1]))      
    camera_minimap = (origin_minimap[0]+minimap_offset_x, origin_minimap[1]+minimap_offset_y)
    calculated_referenced_world = self.calculate_world_absolute_coordinate( (camera_minimap, (0,0) ) )
    location_screen = (world_coordinate[0]-calculated_referenced_world[0], world_coordinate[1]-calculated_referenced_world[1])
    return (camera_minimap, location_screen)
          
  @classmethod  
  def get_coordinates(cls, mask):
    """Mask should be a set of bools from comparison with a feature layer."""
    y, x = mask.nonzero()
    return list(zip(x, y))

  @classmethod
  def create_circle_mask(cls, height, width, center, radius_square):
    y, x = numpy.ogrid[(-center[1]):(height-center[1]), (-center[0]):(width-center[0])]
    
    mask = (x**2 + y**2 <= radius_square)
    masked_array = numpy.full( (height, width), False)
    masked_array[mask] = True
    return masked_array
    
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
        return_locs.append( (center, (left,top), (right,bottom) ) )
    return return_locs


  def _get_my_unit_screen(self, obs, unit_type_id):
    unit_type = obs.observation.feature_screen.unit_type
    unit_type_mask = (unit_type == unit_type_id)
    player_relative = obs.observation.feature_screen.player_relative
    player_relative_mask = (player_relative == features.PlayerRelative.SELF)
    my_unit_mask = numpy.logical_and(player_relative_mask, unit_type_mask)
    return self.get_locations_screen(my_unit_mask)
  def _select_my_unit_screen(self, obs, unit_type_id):
    unit_location_list = self._get_my_unit_screen(obs, unit_type_id)
    count_location = len(unit_location_list)
    if count_location > 0:
      chosen_location = unit_location_list[random.randrange(count_location)][0]
      self._schedule_job(self._current_camera, None, [FUNCTIONS.select_point.id, ['select_all_type', chosen_location ]], True)
      
  def _get_my_castle_screen(self, obs):
    return self._get_my_unit_screen(obs, self._myCastle)


  def _get_Neutral_regions_minimap(self, obs):
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
  def _find_castle_reserved_area(self, obs):
    mineral_source_list = self._find_mineral_source(obs)
    count_mineral_source = len(mineral_source_list)
    if count_mineral_source > 0:
      (left_most, top_most) = mineral_source_list[0][1]
      (right_most, bottom_most) = mineral_source_list[0][2]
      for i in range(1, count_mineral_source):
        (left, top) = mineral_source_list[i][1]
        (right, bottom) = mineral_source_list[i][2]
        if left < left_most:
          left_most = left
        if top < top_most:
          top_most = top
        if right > right_most:
          right_most = right
        if bottom > bottom_most:
          bottom_most = bottom
      center = ( int(round((left_most+right_most)/2)), int(round((top_most+bottom_most)/2)) )
      candidate_point = [center[0], center[1]]
      
      unit_data = _Numeric.DATA[self._myCastle]
      square_length = unit_data['square_length']
      double_square_length = square_length*2
      mineral_distance_from_castle = double_square_length+12
      direction_count = {'left':0, 'top':0, 'right':0, 'bottom':0}
      for mineral_loc in mineral_source_list:
        if mineral_loc[0][0] < center[0] and center[0]-mineral_distance_from_castle <= mineral_loc[2][0]:
          if mineral_loc[1][1]-4 < center[1] and center[1] < mineral_loc[2][1]+4:
            direction_count['left'] += 1 
        if mineral_loc[0][0] > center[0] and center[0]+mineral_distance_from_castle >= mineral_loc[1][0]:
          if mineral_loc[1][1]-4 < center[1] and center[1] < mineral_loc[2][1]+4:
            direction_count['right'] += 1
        if mineral_loc[0][1] < center[1] and center[1]-mineral_distance_from_castle <= mineral_loc[2][1]:
          if mineral_loc[1][0]-4 < center[0] and center[0] < mineral_loc[2][0]+4:
            direction_count['top'] += 1 
        if mineral_loc[0][1] > center[1] and center[1]+mineral_distance_from_castle >= mineral_loc[1][1]:
          if mineral_loc[1][0]-4 < center[0] and center[0] < mineral_loc[2][0]+4:
            direction_count['bottom'] += 1
      mineral_distance_square_from_castle = mineral_distance_from_castle**2
      zero_list = []
      for direction in direction_count.keys():
        if 0 == direction_count[direction]:
          zero_list.append(direction)
      zero_count = len(zero_list)
      
      if zero_count == 2:
        nearest_point = None
        if 'left' in zero_list and 'top' in zero_list:
          nearest_point = (left_most, top_most)
          vertical_range = range(nearest_point[1], center[1])
          horizontal_range = range(nearest_point[0], center[0])
        elif 'left' in zero_list and 'bottom' in zero_list:
          nearest_point = (left_most, bottom_most)
          vertical_range = range(nearest_point[1], center[1], -1)
          horizontal_range = range(nearest_point[0], center[0])
          nearest_distance = self.calculate_distance_square(center, nearest_point)
        elif 'right' in zero_list and 'bottom' in zero_list:
          nearest_point = (right_most, bottom_most)
          vertical_range = range(nearest_point[1], center[1], -1)
          horizontal_range = range(nearest_point[0], center[0], -1)
          nearest_distance = self.calculate_distance_square(center, nearest_point)
        elif 'right' in zero_list and 'top' in zero_list:
          nearest_point = (right_most, top_most)
          vertical_range = range(nearest_point[1], center[1])
          horizontal_range = range(nearest_point[0], center[0], -1)
        if nearest_point is not None:
          nearest_distance = self.calculate_distance_square(center, nearest_point)
          for y in vertical_range:
            for x in horizontal_range:
              chosen_point = (x, y)
              overlapped = False
              for mineral_loc in mineral_source_list:
                distance = self.calculate_distance_square(chosen_point, mineral_loc[0])
                if distance < mineral_distance_square_from_castle:
                  overlapped = True
                  break
              if not overlapped:
                distance = self.calculate_distance_square(center, chosen_point)
                if distance < nearest_distance:
                  nearest_point = chosen_point
                  nearest_distance = distance
          candidate_point = nearest_point
      elif zero_count == 1 or zero_count == 3:
        direction_list = ['left', 'top', 'right', 'bottom']
        direction_offset = [(-1, 0), (0, -1), (1, 0), (0, 1)]
        if zero_count == 1:
          direction_index = -1
          for i in range(4):
            if direction_list[i] in zero_list:
              direction_index = i
              break
        else:
          direction_index = -1
          for i in range(4):
            if direction_list[i] not in zero_list:
              direction_index = (i+2) % 4
              break
        chosen_point = (center[0], center[1])
        while True:
          overlapped = False
          for mineral_loc in mineral_source_list:
            distance = self.calculate_distance_square(chosen_point, mineral_loc[0])
            if distance < mineral_distance_square_from_castle:
              overlapped = True
              break
          if overlapped:
            chosen_point = (chosen_point[0]+direction_offset[direction_index][0], chosen_point[1]+direction_offset[direction_index][1])
          else:
            break
        candidate_point = chosen_point
      
      #height_map = obs.observation.feature_screen.height_map
      #height_histogram = numpy.bincount(height_map.flatten())
      #altitude = numpy.argmax(height_histogram)
      #buildable_mask = (height_map == altitude)
      #unit_type = obs.observation.feature_screen.unit_type
      #for field_type in self._mineral_source:
      #  mineral_mask = (unit_type == field_type)
      #  too_close_mask = morphology.binary_dilation(input=mineral_mask, iterations=12)
      #  buildable_mask = numpy.logical_and(buildable_mask, numpy.logical_not(too_close_mask))
      #unit_data = _Numeric.DATA[self._myCastle]
      #square_length = unit_data['square_length']
      #buildable_mask = morphology.binary_erosion(input=buildable_mask, iterations=square_length*2)
      #y, x = buildable_mask.nonzero()
      #point_list = numpy.asarray(list(zip(x, y)))
      #distance_square_list = numpy.sum((point_list - center)**2, axis=1)
      #min_index = numpy.argmin(distance_square_list)
      #candidate_point = point_list[min_index]      
      self._schedule_job(self._current_camera, None, [FUNCTIONS.Move_screen.id, ['now', candidate_point]])
      #self._schedule_job(self._current_camera, None, [FUNCTIONS.Patrol_screen.id, ['queued', center]])
        
    
    
  def _test_plan_castle_place(self, obs, castle_screen):
    if not self._can_afford_unit(obs, self._myCastle):
      return
    free_neutral_regions = self._neutral_regions - set(self._occupied_neutral_regions.keys())
    chosen_loc = random.sample(free_neutral_regions, 1)[0]
    self._schedule_job(self._current_camera, None, ['_select_gathering_mineral_worker', self, [castle_screen] ])  
    self._schedule_job(self._current_camera, None, [FUNCTIONS.Move_minimap.id, ['now', chosen_loc]])
    self._schedule_job(self._current_camera, None, [FUNCTIONS.move_camera.id, [ chosen_loc ]])
    #self._schedule_job(chosen_loc, None, [FUNCTIONS.move_camera.id, [ self._current_camera ]], True)
    self._schedule_job(chosen_loc, None, ['_find_castle_reserved_area', self, []], True)
    
    

  def _schedule_job(self, camera_minimap, location_screen, function_call_arguments, preemptive=False):
    if location_screen is None:
      if camera_minimap not in self._scheduled_actions_by_minimap:
        self._scheduled_actions_by_minimap[camera_minimap] = [function_call_arguments]
      else:
        if preemptive:
          self._scheduled_actions_by_minimap[camera_minimap].insert(0, function_call_arguments)
        else:
          self._scheduled_actions_by_minimap[camera_minimap].append(function_call_arguments)
    else:
      if camera_minimap not in self._scheduled_actions_by_screen:
        self._scheduled_actions_by_screen[camera_minimap] = {location_screen:[function_call_arguments]}
      elif location_screen not in self._scheduled_actions_by_screen[camera_minimap]:
        self._scheduled_actions_by_screen[camera_minimap][location_screen] = [function_call_arguments]
      else:
        if preemptive:
          self._scheduled_actions_by_screen[camera_minimap][location_screen].insert(0, function_call_arguments)
        else:
          self._scheduled_actions_by_screen[camera_minimap][location_screen].append(function_call_arguments)


  def _select_gathering_mineral_worker(self, obs, castle_screen):
    unit_type = obs.observation.feature_screen.unit_type    
    selected = obs.observation.feature_screen.selected
    gas_plant_list = self.get_locations_screen(unit_type == self._myGasPlant)
    worker_mask = (unit_type == self._myWorker)
    selected_mask = (selected == 0)
    unselected_worker_mask = numpy.logical_and(worker_mask, selected_mask)
    worker_list = self.get_locations_screen(unselected_worker_mask)
    (castle_x, castle_y) = castle_screen
    nearest_loc = None
    nearest_distance = None
    base_axis = []
    for gas_plant in gas_plant_list:
      (gas_plant_x, gas_plant_y) = gas_plant[0]
      base_axis_vec = (castle_x - gas_plant_x, castle_y - gas_plant_y)
      base_axis_len = math.sqrt(base_axis_vec[0]*base_axis_vec[0]+base_axis_vec[1]*base_axis_vec[1])
      base_axis_vec = (base_axis_vec[0]/base_axis_len, base_axis_vec[1]/base_axis_len)
      base_axis.append((base_axis_vec, base_axis_len))
    
    for worker in worker_list:
      (worker_x, worker_y) = worker[0]
      if unit_type[worker_y][worker_x] != self._myWorker:
        continue
      gathering_vespene = False
      for g in range(len(gas_plant_list)):
        gas_plant = gas_plant_list[g]
        (gas_plant_x, gas_plant_y) = gas_plant[0]
        (base_axis_vec, base_axis_len) = base_axis[g]
        worker_vec = (worker_x - gas_plant_x, worker_y - gas_plant_y)
        worker_len = math.sqrt(worker_vec[0]*worker_vec[0]+worker_vec[1]*worker_vec[1])
        worker_vec = (worker_vec[0]/worker_len, worker_vec[1]/worker_len)
        if worker_len < 5.0:
          gathering_vespene = True
          break        
        inner_product = base_axis_vec[0]*worker_vec[0] + base_axis_vec[1]*worker_vec[1]
        #cosine_value = inner_product / (base_axis_len*worker_len)
        cosine_value = inner_product
        degree = math.degrees(math.acos(cosine_value))
        if degree < 15.0:
          gathering_vespene = True
          break
      if gathering_vespene:
        continue
      if worker_x < castle_x-12 or worker_x > castle_x+12 or worker_y < castle_y-12 or worker_y > castle_y+12:
        distance = self.calculate_distance_square(worker[0], castle_screen)
        if nearest_distance is None or distance<nearest_distance:
          nearest_loc = worker
          nearest_distance = distance
          
    if nearest_loc is not None:
      #self._schedule_job(self._current_camera, None, [FUNCTIONS.Stop_quick.id, ['now']], True)
      self._schedule_job(self._current_camera, None, [FUNCTIONS.no_op.id, [] ], True)
      self._schedule_job(self._current_camera, None, [FUNCTIONS.no_op.id, [] ], True)
      self._schedule_job(self._current_camera, None, [FUNCTIONS.Harvest_Return_quick.id, ['queued']], True)
      self._schedule_job(self._current_camera, None, [FUNCTIONS.Move_screen.id, ['now', castle_screen ]], True)
      self._schedule_job(self._current_camera, None, [FUNCTIONS.select_point.id, ['select', nearest_loc[0] ]], True)
    else:
      self._schedule_job(self._current_camera, None, ['_select_gathering_mineral_worker', self, [castle_screen] ], True)
      self._schedule_job(self._current_camera, None, [FUNCTIONS.no_op.id, [] ], True)
      #pass

  def _can_afford_unit(self, obs, unit_type_id):
    if unit_type_id in _Numeric.DATA:
      unit_data = _Numeric.DATA[unit_type_id]
      mineral_cost = unit_data['mineral_cost']
      vespene_cost = unit_data['vespene_cost']
      mineral_hold = obs.observation.player.minerals
      vespene_hold = obs.observation.player.vespene
      if mineral_hold>=mineral_cost and vespene_hold>=vespene_cost:
        return True
    return False


  def _find_buildable_mask(self, obs, altitude, creep_filter, need_power, building_type_id):
    height_map = obs.observation.feature_screen.height_map
    height_map_mask = (height_map == altitude)
    creep = obs.observation.feature_screen.creep
    creep_mask = (creep == creep_filter)
    land_mask = numpy.logical_and(height_map_mask, creep_mask)
    if need_power:
      power = obs.observation.feature_screen.power
      power_mask = (power == 1)
      land_mask = numpy.logical_and(land_mask, power_mask)
    player_relative = obs.observation.feature_screen.player_relative
    neutral_mask = (player_relative == features.PlayerRelative.NEUTRAL)
    if building_type_id == self._myCastle:
      neutral_mask = morphology.binary_dilation(input=neutral_mask, iterations=12)    
    buildable_mask = numpy.logical_and(land_mask, numpy.logical_not(neutral_mask))
    unit_type = numpy.array(obs.observation.feature_screen.unit_type)
    self_mask = (player_relative == features.PlayerRelative.SELF)
    self_unit_type = unit_type[self_mask]
    bin_counts = numpy.bincount(self_unit_type.flatten())
    (my_unit_types, ) = bin_counts.nonzero()
    unit_data = _Logic.DATA
    for unit_type_id in my_unit_types:
      if unit_type_id in unit_data:
        if True == unit_data[unit_type_id]['is_structure'] and False == unit_data[unit_type_id]['is_flying']:
          except_mask = (unit_type == unit_type_id)
          self_unit_mask = numpy.logical_and(except_mask, self_mask)
          buildable_mask = numpy.logical_and(buildable_mask, numpy.logical_not(self_unit_mask))
    ally_mask = (player_relative == features.PlayerRelative.ALLY)
    enemy_mask = (player_relative == features.PlayerRelative.ENEMY)
    someone_mask = numpy.logical_or(ally_mask, enemy_mask)
    someone_unit_type = unit_type[someone_mask]
    bin_counts = numpy.bincount(someone_unit_type.flatten())
    (others_unit_types, ) = bin_counts.nonzero()
    for unit_type_id in others_unit_types:
      if unit_type_id in unit_data:
        if False == unit_data[unit_type_id]['is_flying']:
          except_mask = (unit_type == unit_type_id)
          someone_unit_mask = numpy.logical_and(except_mask, someone_mask)
          buildable_mask = numpy.logical_and(buildable_mask, numpy.logical_not(someone_unit_mask))
    return buildable_mask
                  

  def _find_mineral_source(self, obs):
    unit_type = obs.observation.feature_screen.unit_type
    unit_density = obs.observation.feature_screen.unit_density
    density_mask = (unit_density == 1)
    field_loc_list = []
    for field_type in self._mineral_source:
      unit_type_mask = (unit_type == field_type)
      isolated_unit_type_mask = numpy.logical_and(unit_type_mask, density_mask)
      loc_list = self.get_locations_screen(isolated_unit_type_mask)
      #erosion_mask = morphology.binary_erosion(input=mask, iterations=2)
      #loc_list = self.get_locations_screen(erosion_mask)
      field_loc_list.extend(loc_list)
    return field_loc_list

    
  def _find_vespene_source(self, obs):
    unit_type = obs.observation.feature_screen.unit_type
    geyser_loc_list = []
    for geyser_type in self._vespene_source:
      loc_list = self.get_locations_screen(unit_type == geyser_type)
      geyser_loc_list.extend(loc_list)
    return geyser_loc_list

  def _register_when_poor(self, requirement, resource_demand, candidate_actors, script, preemptive=True):
    self._my_producing.append([preemptive, requirement, resource_demand, candidate_actors, script])
  def _register_producing_unit(self, unit_type_id, camera_minimap, location_screen, script, preemptive=None):
    if unit_type_id in _Numeric.DATA:
      unit_data = _Numeric.DATA[unit_type_id]
      mineral_cost = unit_data['mineral_cost']
      vespene_cost = unit_data['vespene_cost']
      food_required = unit_data['food_required']
      resource_demand = (mineral_cost, vespene_cost, food_required)
      requirement = []
      if unit_type_id in _Dependency.DATA:
        requirement = [_Dependency.DATA[unit_type_id]['require'], _Dependency.DATA[unit_type_id]['actor'] ]
      candidate_actor = (camera_minimap, location_screen)
      if preemptive is None:
        preemptive = _Logic.DATA[unit_type_id]['is_structure']
      self._register_when_poor(requirement, resource_demand, candidate_actor, script, preemptive)

  def _register_building_structure(self, building):
    building_type = building[1][0]
    if building_type not in self._my_building:
      self._my_building[building_type] = []
    # 要檢查是否有其他預定要蓋的建物!!
    coordinate2 = building[0]
    square_length2 = _Numeric.DATA[building_type]['square_length']
    for exist_unit_type_id in self._my_building.keys():
      for exist_building in self._my_building[exist_unit_type_id]:
        coordinate1 = exist_building[0]
        square_length1 = _Numeric.DATA[exist_building[1][0]]['square_length']
        distance = self.calculate_world_distance_square(coordinate1, coordinate2)
        safe_distance = (square_length1+square_length2) * 2
        if distance < safe_distance**2:
          return False
    self._my_building[building_type].append(building)
    return True
    


  def _build_gas_plant(self, obs, castle_screen):
    if not self._can_afford_unit(obs, self._myGasPlant):
      return
    geyser_loc_list = self._find_vespene_source(obs)
    count_geyser_loc = len(geyser_loc_list)
    if count_geyser_loc <= 0:
      return

    unit_data = _Dependency.DATA[self._myGasPlant]
    loc_set = set()
    for loc in geyser_loc_list:
      loc_set.add(loc[0])
    unit_type = obs.observation.feature_screen.unit_type
    built = False
    while len(loc_set) > 0:
      chosen_loc = random.sample(loc_set, 1)[0]
      loc_set.discard(chosen_loc)
      original_unit_type_id = unit_type[chosen_loc[1]][chosen_loc[0]]
      building = [(self._current_camera, chosen_loc), (self._myGasPlant, original_unit_type_id), self._game_step_counter, None]
      if self._register_building_structure(building):      
        self._schedule_job(self._current_camera, None, ['_select_gathering_mineral_worker', self, [castle_screen] ])  
        self._schedule_job(self._current_camera, None, [FUNCTIONS.Move_screen.id, ['now', chosen_loc]])
        self._schedule_job(self._current_camera, None, [unit_data['perform'], ['queued', chosen_loc]])
        built = True
        break
    if built:
      if self._myRace == Race.protoss:
        mineral_field_loc = self._find_mineral_source(obs)
        count_mineral_field_loc = len(mineral_field_loc)
        if count_mineral_field_loc > 0:
          ideal_field_location = mineral_field_loc[0][0]
          nearest_distance = self.calculate_distance_square(ideal_field_location, chosen_loc)
          for i in range(1, count_mineral_field_loc):            
            distance = self.calculate_distance_square(mineral_field_loc[i][0], chosen_loc)
            if distance < nearest_distance:
              ideal_field_location = mineral_field_loc[i][0]
              nearest_distance = distance
          self._schedule_job(self._current_camera, None, [FUNCTIONS.Harvest_Gather_screen.id, ['queued', ideal_field_location]])
      else:
        # 晶礦兵會少一個
        pass


        
  def _backup_viewpoint(self):
    if self._current_camera not in self._last_selected_point:
      self._last_selected_point[self._current_camera] = [self._current_selected_point]
    elif len(self._last_selected_point[self._current_camera]) == 0 or self._last_selected_point[self._current_camera][-1] != self._current_selected_point:
      self._last_selected_point[self._current_camera].append(self._current_selected_point)
    if len(self._last_camera) == 0 or self._last_camera[-1] != self._current_camera:
      self._last_camera.append(self._current_camera)


  def _execute_scheduled_actions(self, obs, scheduled_actions):
    while len(scheduled_actions) > 0:
      next_action = scheduled_actions.pop(0)
      if isinstance(next_action[0], int):
        if next_action[0] == FUNCTIONS.move_camera.id:
          if self._current_camera != next_action[1][0]:
            self._backup_viewpoint()
            self._current_camera = next_action[1][0]
            self._current_selected_point = None
            return actions.FunctionCall.init_with_validation(next_action[0], next_action[1])
        elif next_action[0] == FUNCTIONS.select_point.id:
           if self._current_selected_point is None:
             self._current_selected_point = next_action[1][1]
             return actions.FunctionCall.init_with_validation(next_action[0], next_action[1])
           elif self._current_camera in self._structures and self._current_selected_point in self._structures[self._current_camera]:
             if self._current_selected_point != next_action[1][1]:
               self._backup_viewpoint()
               self._current_selected_point = next_action[1][1]
               return actions.FunctionCall.init_with_validation(next_action[0], next_action[1])
           else:
             self._current_selected_point = next_action[1][1]
             return actions.FunctionCall.init_with_validation(next_action[0], next_action[1])
        elif next_action[0] in obs.observation.available_actions:
          return actions.FunctionCall.init_with_validation(next_action[0], next_action[1])
      elif isinstance(next_action[0], str):
        method = getattr(next_action[1], next_action[0])
        # when DEBUG, return method(obs, *next_action[2])
        # otherwise, don't return what the method returned
        method(obs, *next_action[2])
    return None


  def _notify_gas_plant_is_built(self, obs, building, castle_screen):
    gas_plant_screen = building[0][1]
    gathering_workers = 2 if self._myRace == Race.terran else 3
    # 晶礦兵會少 gathering_workers 個
    for i in range(gathering_workers):
      self._schedule_job(self._current_camera, None, ['_select_gathering_mineral_worker', self, [castle_screen]])
      self._schedule_job(self._current_camera, None, [FUNCTIONS.Move_screen.id, ['now', gas_plant_screen]])
      self._schedule_job(self._current_camera, None, [FUNCTIONS.Harvest_Gather_screen.id, ['queued', gas_plant_screen]])

  
  def _notify_building_complete(self, obs, building):
    owner = obs.observation.player.player_id
    unit_type_id = building[1][0]
    original_unit_type_id = building[1][1]
    if original_unit_type_id is not None and original_unit_type_id in self._locations[owner]:
      old_index = None
      for index in range(len(self._locations[owner][original_unit_type_id])):
        old_location = self._locations[owner][original_unit_type_id][index]
        if old_location == building[0]:
          old_index = index
          break
      if old_index is not None:
        self._locations[owner][original_unit_type_id].pop(old_index)
    if unit_type_id not in self._locations[owner]:
      self._locations[owner][unit_type_id] = [building[0]]
    else:
      self._locations[owner][unit_type_id].append(building[0])
    if building[0][0] not in self._structures:
      self._structures[building[0][0]] = {}
    self._structures[building[0][0]][building[0][1]] = {'owner':owner, 'unit_type':unit_type_id }

    count_building = len(self._my_building[unit_type_id])
    construct_index = -1
    if count_building == 1:
      construct_index = 0
      del self._my_building[unit_type_id]
    else:
      for i in range(count_building):
        check_building = self._my_building[unit_type_id][i]
        if check_building[0] == building[0]:
          construct_index = i
          break
      if construct_index > -1:
        self._my_building[unit_type_id].pop(construct_index)
    if -1 == construct_index:
      debug = 1/0
    
    if unit_type_id == self._myCastle:
      self._structures[building[0][0]][building[0][1]]['mineral_workers'] = 0
      free_neutral_regions = self._neutral_regions - set(self._occupied_neutral_regions.keys())      
      for region_location in free_neutral_regions:
        if self.calculate_distance_square(building[0][0], region_location) <= 13:
          self._occupied_neutral_regions[region_location] = owner
          self._my_infrastructures.append( [building[0][0], building[0][1], region_location] )
          break
      # special call back for Castle
    elif unit_type_id == self._myGasPlant:
      camera_minimap = building[0][0]
      count_bases = len(self._my_infrastructures)
      castle_screen = None
      for my_base in self._my_infrastructures:
        if camera_minimap == my_base[0]:
          castle_screen = my_base[1]
          break
      self._notify_gas_plant_is_built(obs, building, castle_screen)
    else:
      self._actual_agent._notify_building_complete(obs, building)
  
  def _check_building_structures_screen(self, obs, under_constructing):
    unit_type = obs.observation.feature_screen.unit_type
    player_relative = obs.observation.feature_screen.player_relative
    hp_percent = obs.observation.feature_screen.unit_hit_points_ratio
    for building in under_constructing:
      (loc_x, loc_y) = building[0][1]
      if player_relative[loc_y][loc_x] == features.PlayerRelative.SELF:
        if unit_type[loc_y][loc_x] == building[1][0]:
          building[2] = self._game_step_counter
          if building[3] is None or hp_percent[loc_y][loc_x]>building[3]:
            building[3] = hp_percent[loc_y][loc_x]
            if building[3] > 254:
              self._notify_building_complete(obs, building)
              #call back, building complete
              pass
          elif hp_percent[loc_y][loc_x]>254:
            self._notify_building_complete(obs, building)
            
        elif building[1][1] is not None and unit_type[loc_y][loc_x] == building[1][1]:
          # 升級建物中, 請稍候
          pass
        else:
          # 可移動的單位就要叫他走開，不能移動的，就要取消這個building，還不知道細節怎麼做
          pass
      elif player_relative[loc_y][loc_x] == features.PlayerRelative.NONE:
        pass
      else:
        # 敵軍或友軍，都要變更計劃
        pass
  
  def _check_building_structures(self, obs):
    # unit_type_id: [[(minimap, screen), unit_type, last_check, progress]], ]
    # want to check in the same camera, sort by camera first
    constructing = {}
    for unit_type_id in self._my_building.keys():
      for building in self._my_building[unit_type_id]:
        camera_minimap = building[0][0]
        if camera_minimap not in constructing:
          constructing[camera_minimap] = [building]
        else:
          constructing[camera_minimap].append(building)
    if self._current_camera in constructing:
      self._check_building_structures_screen(obs, constructing[camera_minimap])
      
    chosen_structure = None
    for unit_type_id in self._my_building.keys():
      for building in self._my_building[unit_type_id]:
        if building[0][0] == self._current_camera:
          continue
        if chosen_structure is None:
          chosen_structure = building
        elif building[3] is None:
          if chosen_structure[3] is None:
            if building[2] < chosen_structure[2]:
              chosen_structure = building
          else:
            chosen_structure = building
        elif building[2] < chosen_structure[2]:
          chosen_structure = building
    if chosen_structure is not None:
      temp_camera_minimap = chosen_structure[0][0]
      self._schedule_job(temp_camera_minimap, None, [FUNCTIONS.move_camera.id, [ self._current_camera ]], True)
      self._schedule_job(temp_camera_minimap, None, ['_check_building_structures_screen', self, [constructing[temp_camera_minimap]] ], True)
      self._backup_viewpoint()
      self._current_camera = temp_camera_minimap
      self._current_selected_point = None
      return FUNCTIONS.move_camera(temp_camera_minimap)
    return None

      
  def _check_producing_units(self, obs):
    count_my_producing = len(self._my_producing)
    if 0 == count_my_producing:
      return None
    mineral_hold = obs.observation.player.minerals
    vespene_hold = obs.observation.player.vespene
    food_remain = obs.observation.player.food_cap - obs.observation.player.food_used
    owner = obs.observation.player.player_id   
    chosen_index = None
    built_structures = set(self._locations[owner].keys())
    for i in range(count_my_producing):
      requirement = set(self._my_producing[i][1][0])
      if len(requirement) > 0 and len(requirement & built_structures) == 0:
        continue
      (mineral_cost, vespene_cost, food_required) = self._my_producing[i][2]
      if mineral_hold>=mineral_cost and vespene_hold>=vespene_cost and food_remain>=food_required:
        chosen_index = i
        break
      elif True == self._my_producing[i][0]:
        break
    if chosen_index is None:
      return None
    producing = self._my_producing.pop(chosen_index)
    actor = producing[3]
    (camera_minimap, location_screen) = actor
    script = producing[4]
    if location_screen is None:
      for function_call_arguments in script:
        self._schedule_job(camera_minimap, location_screen, function_call_arguments)
      # backup current camera and select_point, then switch to camera_minimap
      if camera_minimap != self._current_camera:
        self._schedule_job(self._current_camera, location_screen, [FUNCTIONS.move_camera.id, [ camera_minimap ]])
    else:
      # TODO: should register to busy queue, the following is for testing
      for function_call_arguments in script:
        self._schedule_job(camera_minimap, location_screen, function_call_arguments)
      if camera_minimap != self._current_camera:
        self._schedule_job(self._current_camera, None, [FUNCTIONS.move_camera.id, [ camera_minimap ]])
        self._schedule_job(camera_minimap, None, [FUNCTIONS.select_point.id, [ 'select', location_screen ]])
      elif location_screen != self._current_selected_point:
        self._schedule_job(camera_minimap, None, [FUNCTIONS.select_point.id, [ 'select', location_screen ]])
    return None
      
  def main_step(self, obs):
    if self._current_camera is not None :
      if self._current_camera in self._scheduled_actions_by_screen:
        scheduled_actions_by_screen = self._scheduled_actions_by_screen[self._current_camera]
        if self._current_selected_point is not None and self._current_selected_point in scheduled_actions_by_screen:
          scheduled_actions = scheduled_actions_by_screen[self._current_selected_point]
          ready_action = self._execute_scheduled_actions(obs, scheduled_actions)
          if ready_action is not None:
            return ready_action
      if self._current_camera in self._scheduled_actions_by_minimap:
        scheduled_actions = self._scheduled_actions_by_minimap[self._current_camera]
        ready_action = self._execute_scheduled_actions(obs, scheduled_actions)
        if ready_action is not None:
          return ready_action
      self._check_producing_units(obs)

      # nothing to do? check which is idle or waiting complete
      #ready_action = self._check_producing_units(obs)
      #if ready_action is not None:
      #  return ready_action
      ready_action = self._check_building_structures(obs)
      if ready_action is not None:
        return ready_action
    return None


  def _move_to_the_slope(self, obs, castle_info_list):
    original_view = castle_info_list[0]
    height_map = original_view[1]
    castle_screen = original_view[0][1]
    altitude = height_map[castle_screen[1]][castle_screen[0]]
    
    count_castle_info = len(castle_info_list)
    nearest_slope = []
    for i in range(1, count_castle_info):
      view = castle_info_list[i]
      height_map = view[1]
      x_velocity = numpy.diff(height_map, axis=1) # 沿著x軸做, 橫向相減(右減左)
      zeros = numpy.zeros( (height_map.shape[0], 1) )
      x_positive = numpy.concatenate( (x_velocity, zeros), axis=1)
      x_negitive = numpy.concatenate( (zeros, x_velocity), axis=1)
      x_continue_minus = numpy.logical_and(numpy.logical_and( (x_positive<0), (x_positive>=-8) ), numpy.logical_and( (x_negitive<0), (x_negitive>=-8) ))
      x_continue_plus = numpy.logical_and(numpy.logical_and( (x_positive>0), (x_positive<=8) ), numpy.logical_and( (x_negitive>0), (x_negitive<=8) ))      
      x_has_velocity = numpy.logical_or(x_continue_minus, x_continue_plus)
                  
      y_velocity = numpy.diff(height_map, axis=0) # 沿著y軸做, 緃向相減(下減上)
      zeros = numpy.zeros( (1, height_map.shape[1]) )
      y_positive = numpy.concatenate( (y_velocity, zeros), axis=0)
      y_negitive = numpy.concatenate( (zeros, y_velocity), axis=0)
      y_continue_minus = numpy.logical_and(numpy.logical_and( (y_positive<0), (y_positive>=-8) ), numpy.logical_and( (y_negitive<0), (y_negitive>=-8) ))
      y_continue_plus = numpy.logical_and(numpy.logical_and( (y_positive>0), (y_positive<=8) ), numpy.logical_and( (y_negitive>0), (y_negitive<=8) ))
      y_has_velocity = numpy.logical_or(y_continue_minus, y_continue_plus)
      
      original_decision = numpy.logical_or(x_has_velocity, y_has_velocity)
      decision = morphology.binary_erosion(input=original_decision, iterations=4)
      if decision.any():      
        abs_height_difference = numpy.absolute(numpy.subtract(height_map, altitude))
        minabs = numpy.amin(abs_height_difference[decision])
        minabs_mask = (abs_height_difference == minabs)
        decision = numpy.logical_and(decision, minabs_mask)
        y, x = decision.nonzero()
        nearest_point = (int(round(x.mean())), int(round(y.mean())))
        nearest_slope.append((view[0][0], nearest_point))
    
    count_nearest_slope = len(nearest_slope)
    if count_nearest_slope > 0:
      chosen_slope = nearest_slope[0]
      nearest_camera_distance = self.calculate_distance_square(chosen_slope[0], original_view[0][0])
      for i in range(1, count_nearest_slope):
        camera_distance = self.calculate_distance_square(nearest_slope[i][0], original_view[0][0])
        if camera_distance < nearest_camera_distance:
          chosen_slope = nearest_slope[i]
          nearest_camera_distance = camera_distance      
      #self._schedule_job(self._current_camera, None, [FUNCTIONS.no_op.id, [] ])
      #self._schedule_job(self._current_camera, None, ['_select_gathering_mineral_worker', self, [castle_screen] ])  
      self._schedule_job(chosen_slope[0], None, [FUNCTIONS.move_camera.id, [self._current_camera]], True)
      self._schedule_job(chosen_slope[0], None, [FUNCTIONS.Move_screen.id, ['now', chosen_slope[1]]], True)
      self._schedule_job(self._current_camera, None, [FUNCTIONS.move_camera.id, [ chosen_slope[0] ]])
    else:
      pass
      #self._schedule_job(self._current_camera, None, [FUNCTIONS.move_camera.id, [ (32,32) ]])
          
  def _record_altitude(self, obs, old_record):
    height_map = obs.observation.feature_screen.height_map
    new_info = [[self._current_camera, None], numpy.array(height_map)]
    old_record.append(new_info)
  
  
  def _look_around(self, obs, old_record):  
    camera = obs.observation.feature_minimap.camera
    y, x = (camera == 1).nonzero()
    viewport = ((x.min(),y.min()), (x.max(),y.max()))
    viewport_size = (viewport[1][0]-viewport[0][0], viewport[1][1]-viewport[0][1])
    distance_ratio = 2.1
    horizontal_distance = int(math.ceil(viewport_size[0]/distance_ratio))
    vertical_distance = int(math.ceil(viewport_size[1]/distance_ratio))
    margin_ratio = 1.6
    horizontal_margin = int(math.floor(viewport_size[0]*margin_ratio))
    vertical_margin = int(math.floor(viewport_size[1]*margin_ratio))

    vertical_direction = []
    horizontal_direction = []
    if viewport[0][0] >= horizontal_margin:
      horizontal_direction.append((-2-horizontal_distance,0))
    if viewport[1][0]+horizontal_margin < camera.shape[1]:
      horizontal_direction.append((2+horizontal_distance,0))
    count_horizontal = len(horizontal_direction)
    if count_horizontal > 1:
      random.shuffle(horizontal_direction)
    if viewport[0][1] >= vertical_margin:
      vertical_direction.append((0,-2-vertical_distance))
    if viewport[1][1]+vertical_margin < camera.shape[0]:
      vertical_direction.append((0,2+vertical_distance))
    count_vertical = len(vertical_direction)
    if count_vertical > 1:
      random.shuffle(vertical_direction)
    if count_vertical > count_horizontal:
      main_sequence = [vertical_direction[0], horizontal_direction[0], vertical_direction[1]]
    elif count_vertical < count_horizontal:
      main_sequence = [horizontal_direction[0], vertical_direction[0], horizontal_direction[1]]
    else:
      flip_coin = random.randrange(1024)
      if 0 == (flip_coin & 1):
        main_sequence = [vertical_direction[0], horizontal_direction[0]]
        if count_horizontal > 1:
          main_sequence.extend([vertical_direction[1], horizontal_direction[1]])
      else:
        main_sequence = [horizontal_direction[0], vertical_direction[0]]
        if count_vertical > 1:
          main_sequence.extend([horizontal_direction[1], vertical_direction[1]])
    
    count_main_sequence = len(main_sequence)
    sequence = [(main_sequence[0], True)]
    for i in range(1, count_main_sequence):
      sequence.append((tuple(numpy.add(main_sequence[i-1], main_sequence[i])), True))
      sequence.append((main_sequence[i], True))
    if 4 == count_main_sequence:
      sequence.append((tuple(numpy.add(main_sequence[3], main_sequence[0])), True))
      sequence.append((main_sequence[0], False))

    sequence.reverse()
    next_camera = self._current_camera
    looked_offset = set()
    for last_offset in sequence:
      last_camera = tuple(numpy.add(self._current_camera, last_offset[0]))
      self._schedule_job(last_camera, None, [FUNCTIONS.move_camera.id, [ next_camera ]], True)
      if last_offset[1]:
        self._schedule_job(last_camera, None, ['_record_altitude', self, [ old_record ]], True)
      next_camera = last_camera
    self._schedule_job(self._current_camera, None, [FUNCTIONS.move_camera.id, [ next_camera ]], True)
                                        
  def initialize(self, obs):
    castle_info_list = []
    self._record_altitude(obs, castle_info_list)    
    castle_screen_list = self._get_my_castle_screen(obs)
    castle_screen = castle_screen_list[0][0]
    castle_info_list[0][0][1] = castle_screen
    owner = obs.observation.player.player_id    
    self._locations[owner] = {}    
    self._locations[owner][self._myCastle] = [castle_info_list]
    self._structures[self._current_camera] = {}
    self._structures[self._current_camera][castle_screen] = {'owner':owner, 'unit_type':self._myCastle, 'mineral_workers':12}        
    self._neutral_regions = set(self._get_Neutral_regions_minimap(obs))
    free_neutral_regions = self._neutral_regions - set(self._occupied_neutral_regions.keys())
    for region_location in free_neutral_regions:
      if self.calculate_distance_square(self._current_camera, region_location) <= 13:
        self._occupied_neutral_regions[region_location] = owner        
        self._my_infrastructures.append( [self._current_camera, castle_screen, region_location] )
        break
    if 1 == len(self._my_infrastructures):
      world_absolute_coordinate = self.calculate_world_absolute_coordinate((self._current_camera, castle_screen))
      local_coordinate = self.calculate_local_coordinate(world_absolute_coordinate)
      #self._schedule_job(self._current_camera, None, [FUNCTIONS.move_camera.id, [ local_coordinate[0] ]])
      #self._schedule_job(local_coordinate[0], None, [FUNCTIONS.select_point.id, ['select', local_coordinate[1] ]])
      #if local_coordinate[0] != self._current_camera:
      #  self._schedule_job(self._current_camera, None, [FUNCTIONS.move_camera.id, [ (32,32) ]])
      self._schedule_job(self._current_camera, None, [FUNCTIONS.select_point.id, ['select', castle_screen ]])
      self._actual_agent.initialize(obs)

  def step(self, obs):
    self._game_step_counter += 1
    if self._camera_offset is None:
      camera = obs.observation.feature_minimap.camera
      y, x = (camera == 1).nonzero()
      viewport = ((x.min(),y.min()), (x.max(),y.max()))
      if self._init_viewport is None:
        self._init_viewport = viewport
        self._viewport_size = (self._init_viewport[1][0]-self._init_viewport[0][0], self._init_viewport[1][1]-self._init_viewport[0][1])
        self._screen_size = (obs.observation.feature_screen.height_map.shape[1], obs.observation.feature_screen.height_map.shape[0])
        self._minimap_size = (obs.observation.feature_minimap.height_map.shape[1], obs.observation.feature_minimap.height_map.shape[0])
        self._current_camera = (int(math.floor((self._init_viewport[0][0]+self._init_viewport[1][0]+1)/2)), int(math.floor((self._init_viewport[0][1]+self._init_viewport[1][1]+1)/2)))
      else:
        self._camera_offset = ((viewport[0][0]-self._init_viewport[0][0]), (viewport[0][1]-self._init_viewport[0][1]))
        self._current_camera = ((self._current_camera[0]-self._camera_offset[0]), (self._current_camera[1]-self._camera_offset[1]))
        self._schedule_job(self._current_camera, None, ['initialize', self, []])
      return FUNCTIONS.move_camera(self._current_camera)
    else:
      return self.main_step(obs)
           
class PracticeProtossAgent(base_agent.BaseAgent):
  RACE_ID = Race.protoss
  WORKER_TYPE = units.Protoss.Probe
  CASTLE_TYPE = units.Protoss.Nexus
  GAS_PLANT_TYPE = units.Protoss.Assimilator
  def __init__(self):
    super(PracticeProtossAgent, self).__init__()
    self.common = GeneralAgentCore(self)

    
  def reset(self):
    super(PracticeProtossAgent, self).reset()
    self.common.reset()

  def _notify_building_complete(self, obs, building):
    pass

  def initialize(self, obs):
    castle_screen = self.common._my_infrastructures[0][1]
    self.common._schedule_job(self.common._current_camera, castle_screen, [FUNCTIONS.Train_Probe_quick.id, ['now']])
    self.common._schedule_job(self.common._current_camera, castle_screen, [FUNCTIONS.Effect_ChronoBoostEnergyCost_screen.id, ['now', castle_screen]])
    script = [ [_Dependency.DATA[self.WORKER_TYPE]['perform'], ['now']] ]
    self.common._register_producing_unit(self.WORKER_TYPE, self.common._current_camera, castle_screen, script)
    script = [ ['_build_gas_plant', self.common, [castle_screen] ] ]    
    self.common._register_producing_unit(self.GAS_PLANT_TYPE, self.common._current_camera, None, script)
    #for i in range(40):
    #  self.common._schedule_job(self.common._current_camera, None, [FUNCTIONS.no_op.id, [] ])
    #self.common._schedule_job(self.common._current_camera, None, ['_build_gas_plant', self.common, [castle_screen] ])

    
  def step(self, obs):
    ready_action = self.common.step(obs)
    if ready_action is not None:
      return ready_action
    else:
      pass
    return FUNCTIONS.no_op()



class PracticeTerranAgent(base_agent.BaseAgent):
  RACE_ID = Race.terran
  WORKER_TYPE = units.Terran.SCV
  CASTLE_TYPE = units.Terran.CommandCenter
  GAS_PLANT_TYPE = units.Terran.Refinery
  def __init__(self):
    super(PracticeTerranAgent, self).__init__()
    self.common = GeneralAgentCore(self)

        
  def reset(self):
    super(PracticeTerranAgent, self).reset()
    self.common.reset()

  def _build_Barracks(self, obs, castle_screen):
    unit_type_id = units.Terran.Barracks
    
  def _build_SupplyDepot(self, obs, castle_screen):
    unit_type_id = units.Terran.SupplyDepot
    if not self.common._can_afford_unit(obs, unit_type_id):
      return
    # build the first supply depot should be careful
    #castle_vertical_diameter = 20
    mineral_source_list = self.common._find_mineral_source(obs)
    candidate_locations = [(castle_screen[0], castle_screen[1]-26), (castle_screen[0], castle_screen[1]+26)]
    decision = [True, True]
    count_candidate_locations = len(candidate_locations)
    
    for field in mineral_source_list:
      for i in range(count_candidate_locations):
        if self.common.calculate_distance_square(field[0], candidate_locations[i]) < 81:
          decision[i] = False
        
    unit_data = _Dependency.DATA[unit_type_id]
    if True==decision[0] and False==decision[1]:
      self.common._schedule_job(self.common._current_camera, None, ['_select_gathering_mineral_worker', self.common, [castle_screen]])
      self.common._schedule_job(self.common._current_camera, None, [unit_data['perform'], ['now', candidate_locations[0]]])
    elif False==decision[0] and True==decision[1]:
      self.common._schedule_job(self.common._current_camera, None, ['_select_gathering_mineral_worker', self.common, [castle_screen]])
      self.common._schedule_job(self.common._current_camera, None, [unit_data['perform'], ['now', candidate_locations[1]]])
    else:
      self.common._schedule_job(self.common._current_camera, None, [FUNCTIONS.move_camera.id, [ (32,32) ]])
        
  def _notify_building_complete(self, obs, building):
    pass
    
  def initialize(self, obs):
    castle_screen = self.common._my_infrastructures[0][1]
    owner = obs.observation.player.player_id
    castle_info_list = self.common._locations[owner][self.CASTLE_TYPE][0]
    self.common._schedule_job(self.common._current_camera, castle_screen, [FUNCTIONS.Train_SCV_quick.id, ['now']])
    self.common._schedule_job(self.common._current_camera, None, ['_look_around', self.common, [castle_info_list]])
    #script = [ ['_build_SupplyDepot', self, [castle_screen] ] ]
    #self.common._register_producing_unit(units.Terran.SupplyDepot, self.common._current_camera, None, script)
    script = [ ['_build_gas_plant', self.common, [castle_screen] ] ]
    self.common._register_producing_unit(self.GAS_PLANT_TYPE, self.common._current_camera, None, script)
    script = [ ['_build_gas_plant', self.common, [castle_screen] ] ]
    self.common._register_producing_unit(self.GAS_PLANT_TYPE, self.common._current_camera, None, script)
    #self.common._schedule_job(self.common._current_camera, None, ['_select_gathering_mineral_worker', self.common, [castle_screen] ])  
    #self.common._schedule_job(self.common._current_camera, None, ['_move_to_the_slope', self.common, [castle_info_list]])

    script = [ ['_test_plan_castle_place', self.common, [castle_screen] ] ]
    self.common._register_producing_unit(self.CASTLE_TYPE, self.common._current_camera, None, script)


  def step(self, obs):
    ready_action = self.common.step(obs)
    if ready_action is not None:
      return ready_action
    else:
      pass
    return FUNCTIONS.no_op()



class PracticeZergAgent(base_agent.BaseAgent):
  RACE_ID = Race.zerg
  WORKER_TYPE = units.Zerg.Drone
  CASTLE_TYPE = units.Zerg.Hatchery
  GAS_PLANT_TYPE = units.Zerg.Extractor
  def __init__(self):
    super(PracticeZergAgent, self).__init__()
    self.common = GeneralAgentCore(self)

    
  def reset(self):
    super(PracticeZergAgent, self).reset()
    self.common.reset()

  def _find_buildable_locations(self, obs, altitude, building_type_id):
    buildable_mask = self.common._find_buildable_mask(obs, altitude, 1, False, building_type_id)
    if building_type_id in _Numeric.DATA:
      unit_data = _Numeric.DATA[building_type_id]
      square_length = unit_data['square_length']
      location_mask = morphology.binary_erosion(input=buildable_mask, iterations=square_length*2+2)
      y, x = location_mask.nonzero()
      return list(zip(x, y))
    return []

  def _upgrade_castle(self, obs, castle_screen):
    unit_type = obs.observation.feature_screen.unit_type
    current_castle_type = unit_type[castle_screen[1]][castle_screen[0]]
    if current_castle_type == units.Zerg.Hive:
      return
    elif current_castle_type == units.Zerg.Lair:
      target_castle_type = units.Zerg.Hive
    else:
      target_castle_type = units.Zerg.Lair
    if not self.common._can_afford_unit(obs, target_castle_type):
      return
    building = [(self.common._current_camera, castle_screen), (target_castle_type, current_castle_type), self.common._game_step_counter, None]
    self.common._register_building_structure(building)
    unit_data = _Dependency.DATA[target_castle_type]
    self.common._schedule_job(self.common._current_camera, castle_screen, [unit_data['perform'], ['now']])
    
  def _build_structures(self, obs, castle_screen, unit_type_id):
    if not self.common._can_afford_unit(obs, unit_type_id):
      return
    height_map = obs.observation.feature_screen.height_map
    altitude = height_map[castle_screen[1]][castle_screen[0]]
    buildable_locations = self._find_buildable_locations(obs, altitude, unit_type_id)
    count_locations = len(buildable_locations)
    if count_locations > 0:
      
      unit_data = _Dependency.DATA[unit_type_id]
      loc_set = set(buildable_locations)
      while len(loc_set) > 0:
        chosen_loc = random.sample(loc_set, 1)[0]
        loc_set.discard(chosen_loc)
        building = [(self.common._current_camera, chosen_loc), (unit_type_id, None), self.common._game_step_counter, None]
        if self.common._register_building_structure(building):
          self.common._schedule_job(self.common._current_camera, None, ['_select_gathering_mineral_worker', self.common, [castle_screen]])
          self.common._schedule_job(self.common._current_camera, None, [unit_data['perform'], ['now', chosen_loc]])
          break

  def _train_Zerg_Queen(self, obs, castle_screen):
    unit_type_id = units.Zerg.Queen
    unit_data = _Dependency.DATA[unit_type_id]
    selected = obs.observation.feature_screen.selected
    selected_mask = (selected == 1)
    unit_type = obs.observation.feature_screen.unit_type
    count_multi_select = len(obs.observation.multi_select)
    if count_multi_select > 0:
      script = [ ['_train_Zerg_Queen', self, [castle_screen] ] ]
      self.common._register_producing_unit(unit_type_id, self.common._current_camera, castle_screen, script)
      self.common._schedule_job(self.common._current_camera, castle_screen, ['remove_current_selected_point', self, []], True)
    elif len(obs.observation.single_select) > 0 and obs.observation.single_select[0].unit_type==units.Zerg.Larva:
      script = [ ['_train_Zerg_Queen', self, [castle_screen] ] ]
      self.common._register_producing_unit(unit_type_id, self.common._current_camera, castle_screen, script)
      self.common._schedule_job(self.common._current_camera, castle_screen, ['remove_current_selected_point', self, []], True)
    else:
      castle_types = unit_data['actor']
      any_castle_selected = False      
      for (castle_type_id, ) in castle_types:
        castle_mask = (unit_type == castle_type_id)
        castle_selected = numpy.logical_and(selected_mask, castle_mask)
        if castle_selected.any():
          any_castle_selected = True
          break
      if any_castle_selected:
        if len(obs.observation.build_queue) > 0:
          return
        elif unit_data['perform'] in obs.observation.available_actions:
          self.common._schedule_job(self.common._current_camera, castle_screen, [unit_data['perform'], ['now']], True)
        else:
          script = [ ['_train_Zerg_Queen', self, [castle_screen] ] ]
          self.common._register_producing_unit(unit_type_id, self.common._current_camera, castle_screen, script)        
      else:
        script = [ ['_train_Zerg_Queen', self, [castle_screen] ] ]
        self.common._register_producing_unit(unit_type_id, self.common._current_camera, castle_screen, script)
    
  def _train_Zerg_units(self, obs, castle_screen, unit_type_id):
    selected = obs.observation.feature_screen.selected
    selected_mask = (selected == 1)
    unit_type = obs.observation.feature_screen.unit_type
    unit_data = _Dependency.DATA[unit_type_id]
    count_multi_select = len(obs.observation.multi_select)
    if count_multi_select > 0:
      for i in range(count_multi_select):
        selected_unit = obs.observation.multi_select[i]
        if selected_unit.unit_type==units.Zerg.Larva:
          #self.common._schedule_job(self.common._current_camera, None, [FUNCTIONS.select_point.id, ['select', castle_screen]], True)
          #self.common._schedule_job(self.common._current_camera, castle_screen, ['remove_current_selected_point', self, []], True)
          self.common._schedule_job(self.common._current_camera, castle_screen, [unit_data['perform'], ['now']], True)
          self.common._schedule_job(self.common._current_camera, castle_screen, [FUNCTIONS.select_unit.id, ['select_all_type', i]], True)
          break    
    elif len(obs.observation.single_select) > 0 and obs.observation.single_select[0].unit_type==units.Zerg.Larva:
      self.common._schedule_job(self.common._current_camera, None, [FUNCTIONS.select_point.id, ['select', castle_screen]], True)
      self.common._schedule_job(self.common._current_camera, castle_screen, ['remove_current_selected_point', self, []], True)
      self.common._schedule_job(self.common._current_camera, castle_screen, [unit_data['perform'], ['now']], True)
    else:
      castle_types = _Dependency.DATA[units.Zerg.Larva]['actor']        
      any_castle_selected = False      
      for (castle_type_id, ) in castle_types:
        castle_mask = (unit_type == castle_type_id)
        castle_selected = numpy.logical_and(selected_mask, castle_mask)
        if castle_selected.any():
          any_castle_selected = True
          break
      if any_castle_selected:
        if obs.observation.player.larva_count > 0:
          #FUNCTIONS.select_larva.id in obs.observation.available_actions
          self.common._schedule_job(self.common._current_camera, castle_screen, [unit_data['perform'], ['now']], True)
          self.common._schedule_job(self.common._current_camera, castle_screen, [FUNCTIONS.select_larva.id, []], True)
        else: # a kind of busy, for production structure, re-schedule
          script = [ ['_train_Zerg_units', self, [castle_screen, unit_type_id] ] ]
          self.common._register_producing_unit(unit_type_id, self.common._current_camera, castle_screen, script)
      else:
        if self.common._current_selected_point == castle_screen and not any_castle_selected:
          self.common._current_selected_point = None
        script = [ ['_train_Zerg_units', self, [castle_screen, unit_type_id] ] ]
        self.common._register_producing_unit(unit_type_id, self.common._current_camera, castle_screen, script)


  def remove_current_selected_point(self, obs):    
    self.common._current_selected_point = None

  def _notify_building_complete(self, obs, building):
    unit_type_id = building[1][0]
    camera_minimap = building[0][0]
    location_screen = building[0][1]
    owner = obs.observation.player.player_id 
    #if unit_type_id == units.Zerg.SpawningPool and len(self.common._locations[owner][unit_type_id]) == 1:
    #  queen_data = _Dependency.DATA[units.Zerg.Queen]
    #  script = [ [queen_data['perform'], ['now']] ] 
    #  castle_camera = self.common._my_infrastructures[0][0]
    #  castle_screen = self.common._my_infrastructures[0][1]
    #  self.common._register_producing_unit(units.Zerg.Queen, castle_camera, castle_screen, script)
      
  def initialize(self, obs):  
    castle_screen = self.common._my_infrastructures[0][1]
    owner = obs.observation.player.player_id
    castle_info_list = self.common._locations[owner][self.CASTLE_TYPE][0]
    self.common._schedule_job(self.common._current_camera, castle_screen, ['_train_Zerg_units', self, [castle_screen, self.WORKER_TYPE] ])
    self.common._schedule_job(self.common._current_camera, None, ['_look_around', self.common, [castle_info_list]])
    self.common._schedule_job(self.common._current_camera, None, ['_select_my_unit_screen', self.common, [units.Zerg.Overlord]])
    self.common._schedule_job(self.common._current_camera, None, ['_move_to_the_slope', self.common, [castle_info_list]])
    script = [ ['_build_structures', self, [castle_screen, units.Zerg.SpawningPool] ] ]
    self.common._register_producing_unit(units.Zerg.SpawningPool, self.common._current_camera, None, script)
    script = [ ['_train_Zerg_units', self, [castle_screen, self.WORKER_TYPE] ] ]
    self.common._register_producing_unit(self.WORKER_TYPE, self.common._current_camera, castle_screen, script, True)
    script = [ ['_train_Zerg_units', self, [castle_screen, units.Zerg.Overlord] ] ]
    self.common._register_producing_unit(units.Zerg.Overlord, self.common._current_camera, castle_screen, script, True)
    script = [ ['_train_Zerg_units', self, [castle_screen, self.WORKER_TYPE] ] ]
    self.common._register_producing_unit(self.WORKER_TYPE, self.common._current_camera, castle_screen, script, True)
    #script = [ ['_build_gas_plant', self.common, [castle_screen] ] ]
    #self.common._register_producing_unit(self.GAS_PLANT_TYPE, self.common._current_camera, None, script)   
    #for i in range(4):
    #  script = [ ['_train_Zerg_units', self, [castle_screen, self.WORKER_TYPE] ] ]
    #  self.common._register_producing_unit(self.WORKER_TYPE, self.common._current_camera, castle_screen, script, True)
    #script = [ ['_build_gas_plant', self.common, [castle_screen] ] ]
    #self.common._register_producing_unit(self.GAS_PLANT_TYPE, self.common._current_camera, None, script)
    #for i in range(4):
    #  script = [ ['_train_Zerg_units', self, [castle_screen, self.WORKER_TYPE] ] ]
    #  self.common._register_producing_unit(self.WORKER_TYPE, self.common._current_camera, castle_screen, script, True)
    #script = [ ['_train_Zerg_units', self, [castle_screen, units.Zerg.Overlord] ] ]
    #self.common._register_producing_unit(units.Zerg.Overlord, self.common._current_camera, castle_screen, script, True)

    #script = [ ['_upgrade_castle', self, [castle_screen] ] ]
    #self.common._register_producing_unit(units.Zerg.Hive, self.common._current_camera, castle_screen, script)

    #script = [ ['_build_structures', self, [castle_screen, units.Zerg.InfestationPit] ] ]
    #self.common._register_producing_unit(units.Zerg.InfestationPit, self.common._current_camera, None, script)
    
    #queen_data = _Dependency.DATA[units.Zerg.Queen]
    #script = [ ['_train_Zerg_Queen', self, [castle_screen]] ] 
    #self.common._register_producing_unit(units.Zerg.Queen, self.common._current_camera, castle_screen, script, True)

    #script = [ ['_upgrade_castle', self, [castle_screen] ] ]
    #self.common._register_producing_unit(units.Zerg.Lair, self.common._current_camera, castle_screen, script)

    #script = [ ['_build_structures', self, [castle_screen, units.Zerg.RoachWarren] ] ]
    #self.common._register_producing_unit(units.Zerg.RoachWarren, self.common._current_camera, None, script)
    #script = [ ['_train_Zerg_units', self, [castle_screen, self.WORKER_TYPE] ] ]
    #self.common._register_producing_unit(self.WORKER_TYPE, self.common._current_camera, castle_screen, script, True)
    #script = [ ['_build_structures', self, [castle_screen, units.Zerg.BanelingNest] ] ]
    #self.common._register_producing_unit(units.Zerg.BanelingNest, self.common._current_camera, None, script)
    #script = [ ['_train_Zerg_units', self, [castle_screen, self.WORKER_TYPE] ] ]
    #self.common._register_producing_unit(self.WORKER_TYPE, self.common._current_camera, castle_screen, script, True)
    #script = [ ['_build_structures', self, [castle_screen, units.Zerg.EvolutionChamber] ] ]
    #self.common._register_producing_unit(units.Zerg.EvolutionChamber, self.common._current_camera, None, script)
    #script = [ ['_train_Zerg_units', self, [castle_screen, self.WORKER_TYPE] ] ]
    #self.common._register_producing_unit(self.WORKER_TYPE, self.common._current_camera, castle_screen, script, True)


    script = [ ['_test_plan_castle_place', self.common, [castle_screen] ] ]
    self.common._register_producing_unit(self.CASTLE_TYPE, self.common._current_camera, None, script)
            

    
  def step(self, obs):
    ready_action = self.common.step(obs)
    if ready_action is not None:
      return ready_action
    else:
      pass
    return FUNCTIONS.no_op()



class PracticeRandomRaceAgent(base_agent.BaseAgent):
  ACTUAL_AGENT_CLS = { units.Protoss.Nexus: PracticeProtossAgent, units.Terran.CommandCenter: PracticeTerranAgent, units.Zerg.Hatchery: PracticeZergAgent}        
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
