from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import json
import numpy
from collections import deque
from skimage.feature import canny
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.draw import circle_perimeter
from skimage import io as skimage_io
from skimage import color as skimage_color
from skimage import img_as_ubyte as skimage_img_as_ubyte
from matplotlib import pyplot as plt


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
                  , units.Neutral.PurifierMineralField, units.Neutral.PurifierMineralField750
                  , units.Neutral.PurifierRichMineralField, units.Neutral.PurifierRichMineralField750
                  , units.Neutral.BattleStationMineralField, units.Neutral.BattleStationMineralField750
                  ]

  VESPENE_TYPES = [ units.Neutral.VespeneGeyser
                  , units.Neutral.SpacePlatformGeyser
                  , units.Neutral.RichVespeneGeyser
                  , units.Neutral.PurifierVespeneGeyser
                  , units.Neutral.ProtossVespeneGeyser
                  , units.Neutral.ShakurasVespeneGeyser
                  ]
  MINERAL_BIAS = (0, 0)
  VESPENE_BIAS = (0, 0)
  # 遊戲畫面的輔助格邊長，約等於 feature_scren 的 3 像素長
  GRID_SIDE_LENGTH = 3
  DEBUG_OUTPUT_PATH = '/home/yiren/Pictures'
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
    self.ViewportCenter = (0, 0)
    self.ScreenSize = (0, 0)
    self.MinimapSize = (0, 0)
    self.CameraBias = None
    self.FirstViewport = None
    self.NeutralMinimap = None
    self.CameraBoundary = [None, None]
    self.AccumulatedOffset = [[0], [0]]
    self.MapShape = (0, 0)
    # useful hidden states
    self._current_camera = None
    self._expected_selected = None
    # useful hidden states for strategy
    self._world_coordinate = {}
    self._structures = {}
    self._neutral_regions = set()
    self._speculated_resource_region_list = []
    self._calculated_resource_region_list = []
    self._occupied_resource_regions = {}
    self._holding_resource_region_list = []
    self._height_map_on_camera = {}
    # for debug_accumulate
    self._resource_image = None
    # scheduled
    self._scheduled_actions_on_camera = {}
    self._scheduled_actions_on_unit = {}


  def calculate_world_absolute_coordinate(self, local_coordinate):
    (camera_minimap, location_screen) = local_coordinate
    p = [self.AccumulatedOffset[i][camera_minimap[i]-self.CameraBoundary[0][i]]+location_screen[i] for i in range(2)]
    return tuple(p)


  def calculate_world_relative_coordinate(self, local_coordinate1, local_coordinate2):
    world_coordinate1 = self.calculate_world_absolute_coordinate(local_coordinate1)
    world_coordinate2 = self.calculate_world_absolute_coordinate(local_coordinate2)
    return (world_coordinate2[0]-world_coordinate1[0], world_coordinate2[1]-world_coordinate1[1])


  def calculate_local_coordinate(self, world_coordinate):
    screen_size = self.ScreenSize

    camera_offset = [0, 0]
    location_base = [0, 0]
    for axis_index in range(2):
      axis_offset = numpy.array(self.AccumulatedOffset[axis_index])
      camera_offset[axis_index] = abs(axis_offset-(world_coordinate[axis_index]-self.ScreenSize[axis_index]//2)).argmin()
      new_location_base = self.AccumulatedOffset[axis_index][camera_offset[axis_index]]
      while world_coordinate[axis_index]-new_location_base<0 and camera_offset[axis_index]>0:
        camera_offset[axis_index] -= 1
        new_location_base = self.AccumulatedOffset[axis_index][camera_offset[axis_index]]      
      boundary_index = len(self.AccumulatedOffset[axis_index])-2
      if camera_offset[axis_index] < 1:
        new_location_base = self.AccumulatedOffset[axis_index][1]
        if world_coordinate[axis_index]-new_location_base >= 0:
          camera_offset[axis_index] = 1
      elif camera_offset[axis_index] > boundary_index:
        new_location_base = self.AccumulatedOffset[axis_index][boundary_index]
        if world_coordinate[axis_index]-new_location_base < self.ScreenSize[axis_index]:
          camera_offset[axis_index] = boundary_index
      location_base[axis_index] = self.AccumulatedOffset[axis_index][camera_offset[axis_index]]

    camera_minimap = (self.CameraBoundary[0][0]+camera_offset[0], self.CameraBoundary[0][1]+camera_offset[1])
    location_screen = (world_coordinate[0]-location_base[0], world_coordinate[1]-location_base[1])
    return (camera_minimap, location_screen)


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
  def create_circle_mask(cls, mask_shape, center, radius_square):
    (height, width) = mask_shape
    y, x = numpy.ogrid[(-center[1]):(height-center[1]), (-center[0]):(width-center[0])]

    mask = (x**2 + y**2 <= radius_square)
    masked_array = numpy.full( mask_shape, False)
    masked_array[mask] = True
    return masked_array


  @classmethod
  def create_mineral_circle_mask(cls, screen_shape, center):
    circle_mask = numpy.full(screen_shape, False)
    circle_mask[center[1]-4:center[1]+4, center[0]-4:center[0]+4] = True
    circle_mask[center[1]-4:center[1]-2, center[0]-4:center[0]-2] = False
    circle_mask[center[1]-4:center[1]-2, center[0]+2:center[0]+4] = False
    circle_mask[center[1]+2:center[1]+4, center[0]-4:center[0]-2] = False
    circle_mask[center[1]+2:center[1]+4, center[0]+2:center[0]+4] = False
    circle_mask[center[1]-3, center[0]-3] = True
    circle_mask[center[1]-3, center[0]+2] = True
    circle_mask[center[1]+2, center[0]-3] = True
    circle_mask[center[1]+2, center[0]+2] = True
    return circle_mask


  @classmethod
  def _create_resource_density(cls, mineral_field_list, vespene_geyser_list):
    SCREEN_SHAPE = (self.ScreenSize[1], self.ScreenSize[0])
    density = numpy.zeros(SCREEN_SHAPE, dtype=numpy.uint8)
    for center in mineral_field_list:
      circle_mask = cls.create_mineral_circle_mask(SCREEN_SHAPE, center)
      density[circle_mask] += 1
    for center in vespene_geyser_list:
      circle_mask = cls.create_circle_mask(SCREEN_SHAPE, center, 5**2+2**2)
      density[circle_mask] += 1
    return density


  @classmethod
  def create_mineral_field_list_mask(cls, screen_shape, mineral_field_list):
    resource_mask = numpy.full(screen_shape, False)
    for center in mineral_field_list:
      (mineral_x, mineral_y) = (center[0]+cls.MINERAL_BIAS[0], center[1]+cls.MINERAL_BIAS[1])
      (mineral_left, mineral_top) = (mineral_x-3, mineral_y-2)
      (mineral_right, mineral_bottom) = (mineral_x+2, mineral_y)
      resource_mask[mineral_top:mineral_bottom+1, mineral_left:mineral_right+1] = True
    return resource_mask


  @classmethod
  def create_vespene_geyser_list_mask(cls, screen_shape, vespene_geyser_list):
    resource_mask = numpy.full(screen_shape, False)
    for center in vespene_geyser_list:
      (vespene_x, vespene_y) = (center[0]+cls.VESPENE_BIAS[0], center[1]+cls.VESPENE_BIAS[1])
      (vespene_left, vespene_top) = (vespene_x-4, vespene_y-4)
      (vespene_right, vespene_bottom) = (vespene_x+4, vespene_y+4)
      resource_mask[vespene_top:vespene_bottom+1, vespene_left:vespene_right+1] = True
    return resource_mask


  @classmethod
  def _calculate_circles_intersection_points(cls, circle1, circle2):
    (x1, y1, r1) = circle1
    (x2, y2, r2) = circle2
    direct_vec = (x1-x2, y1-y2)
    determinant_value = x2*y1-y2*x1
    secant_equation_constant = ((r2**2-r1**2) + (x1**2-x2**2) + (y1**2-y2**2)) / 2.0
    square_sum = direct_vec[0]**2 + direct_vec[1]**2
    secant_middle_x = (direct_vec[0]*secant_equation_constant+direct_vec[1]*determinant_value) / square_sum
    secant_middle_y = (direct_vec[1]*secant_equation_constant-direct_vec[0]*determinant_value) / square_sum

    parametric_square = (r1**2 - ((x1-secant_middle_x)**2+(y1-secant_middle_y)**2)) / square_sum
    positive_parametric = math.sqrt(parametric_square)
    negative_parametric = -positive_parametric

    intersection_points = [(secant_middle_x-direct_vec[1]*t, secant_middle_y+direct_vec[0]*t) for t in (positive_parametric, negative_parametric)]
    return intersection_points

  def create_townhall_margin_mask(self, screen_shape, townhall_location):
    GRID_COUNT_townhall = self._get_grid_count_townhall()
    (LEFT, TOP, RIGHT, BOTTOM) = range(4)

    RADII = [self.GRID_SIDE_LENGTH*2, self.GRID_SIDE_LENGTH*4, self.GRID_SIDE_LENGTH*5, self.GRID_SIDE_LENGTH*6, self.GRID_SIDE_LENGTH*6+1]
    values = []
    for radius in RADII:
      values.append([townhall_location[i%2]+(i//2*2-1)*radius for i in range(4)])

    LOOSE = len(RADII)-1
    for i in (LEFT, TOP):
      for j in range(LOOSE, -1, -1):
        if values[j][i] < 0:
          values[j][i] = 0
        else:
          break
    for i in (RIGHT, BOTTOM):
      MAX_VALUE = screen_shape[3-i]-1
      for j in range(LOOSE, -1, -1):
        if values[j][i] > MAX_VALUE:
          values[j][i] = MAX_VALUE
        else:
          break

    margin_mask = numpy.full(screen_shape, False)
    for j in range(LOOSE+1):
      margin_mask[values[LOOSE-j][TOP]:values[LOOSE-j][BOTTOM]+1, values[j][LEFT]:values[j][RIGHT]+1] = True
    return margin_mask


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
        for old_p in list(grouped_points[exist_serial]):
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
  def get_locations_screen(cls, mask, neighbor_distance_squre=2, tuple_form=False):
    grouped_points = cls.aggregate_points(mask, neighbor_distance_squre)
    return_locs = []
    for exist_serial in grouped_points.keys():
      if len(grouped_points[exist_serial]) > 0:
        point_arr = numpy.array(list(grouped_points[exist_serial]))
        left, top = point_arr.min(axis=0)
        right, bottom = point_arr.max(axis=0)
        if tuple_form:
          center = ((left+right)/2.0, (top+bottom)/2.0)
          return_locs.append( (center, (left,top), (right,bottom) ) )
        else:
          center = (int(round((left+right)/2.0)), int(round((top+bottom)/2.0)))
          return_locs.append( center )
    return return_locs


  def _get_grid_count_townhall(self):
    if type(self) == GeneralAgent:
      return 5
    return _UnitNumeric[self.TOWNHALL_TYPES[0]]['grid_count']


  def _get_grid_count_gas_plant(self):
    if type(self) == GeneralAgent:
      return 3
    return _UnitNumeric[self.GAS_PLANT_TYPE]['grid_count']


  def _get_unit_screen(self, obs, player_role, unit_type_list, tuple_form=False):
    return_locs = []
    player_relative = obs.observation.feature_screen.player_relative
    player_relative_mask = (player_relative == player_role)
    unit_type = obs.observation.feature_screen.unit_type
    for unit_type_id in unit_type_list:
      unit_type_mask = (unit_type == unit_type_id)
      my_unit_mask = numpy.logical_and(player_relative_mask, unit_type_mask)
      return_locs.extend(self.get_locations_screen(my_unit_mask, 2, tuple_form))
    return return_locs


  def _get_my_unit_screen(self, obs, unit_type_list, tuple_form=False):
    return self._get_unit_screen(obs, features.PlayerRelative.SELF, unit_type_list, tuple_form)


  def _get_my_townhall_screen(self, obs, tuple_form=False):
    unit_type = obs.observation.feature_screen.unit_type
    return self._get_my_unit_screen(obs, self.TOWNHALL_TYPES, tuple_form)


  def _get_mineral_screen(self, obs, tuple_form=False):
    unit_type = obs.observation.feature_screen.unit_type
    unit_density = obs.observation.feature_screen.unit_density
    density_mask = (unit_density == 1)
    field_loc_list = []
    for field_type in self.MINERAL_TYPES:
      unit_type_mask = (unit_type == field_type)
      isolated_unit_type_mask = numpy.logical_and(unit_type_mask, density_mask)
      loc_list = self.get_locations_screen(isolated_unit_type_mask, 1, tuple_form)
      field_loc_list.extend(loc_list)
    return field_loc_list


  def _get_vespene_screen(self, obs, tuple_form=False):
    unit_type = obs.observation.feature_screen.unit_type
    return_locs = []
    for geyser_type in self.VESPENE_TYPES:
      unit_type_mask = (unit_type == geyser_type)
      return_locs.extend(self.get_locations_screen(unit_type_mask, 2, tuple_form))
    return return_locs


  def _detect_mineral_screen(self, union_mineral_mask):
    if union_mineral_mask.any():
      y, x = union_mineral_mask.nonzero()
      left, right = x.min(), x.max()
      top, bottom = y.min(), y.max()
      for maybe_y in (bottom-3, top+4):
        for maybe_x in range(right-3, left+3, -1):
          maybe_center = (maybe_x, maybe_y)
          circle_mask = self.create_mineral_circle_mask(union_mineral_mask.shape, maybe_center)
          contradiction = numpy.logical_and(numpy.logical_not(union_mineral_mask), circle_mask)
          if not contradiction.any():
            return maybe_center
      for maybe_x in (right-3, left+4):
        for maybe_y in range(bottom-3, top+3, -1):
          maybe_center = (maybe_x, maybe_y)
          circle_mask = self.create_mineral_circle_mask(union_mineral_mask.shape, maybe_center)
          contradiction = numpy.logical_and(numpy.logical_not(union_mineral_mask), circle_mask)
          if not contradiction.any():
            return maybe_center
    return None


  @classmethod
  def set_color_in_density_image(cls, image, density):
    image[(density==1)] = (255, 255, 255)
    image[(density==2)] = (191, 191, 191)
    image[(density==3)] = (127, 127, 127)
    image[(density==4)] = (63, 63, 63)


  def _get_resource_screen(self, unit_type, unit_density, player_relative, debug_output=False):
    DEBUG_OUTPUT = debug_output
    SCREEN_SHAPE = (self.ScreenSize[1], self.ScreenSize[0])
    neutral_mask = (player_relative == features.PlayerRelative.NEUTRAL)

    unanalyzed_density = numpy.array(unit_density)
    unanalyzed_density[numpy.logical_not(neutral_mask)] = 0

    if DEBUG_OUTPUT:
      fig0, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, nrows=1, sharex=True, sharey=True, figsize=(8, 2))
      image = skimage_color.gray2rgb(numpy.zeros(SCREEN_SHAPE, dtype=numpy.uint8))
      self.set_color_in_density_image(image, unanalyzed_density)
      ax1.imshow(image)

    union_mineral_mask = numpy.full(SCREEN_SHAPE, False)
    resource_type_list = self.MINERAL_TYPES
    for resource_type in resource_type_list:
      unit_type_mask = (unit_type == resource_type)
      union_mineral_mask = numpy.logical_or(union_mineral_mask, unit_type_mask)

    if DEBUG_OUTPUT:
      image = skimage_color.gray2rgb(numpy.zeros(SCREEN_SHAPE, dtype=numpy.uint8))
      image[union_mineral_mask] = (0, 0, 255)
      ax2.imshow(image)

    union_vespene_mask = numpy.full(SCREEN_SHAPE, False)
    resource_type_list = self.VESPENE_TYPES
    for resource_type in resource_type_list:
      unit_type_mask = (unit_type == resource_type)
      union_vespene_mask = numpy.logical_or(union_vespene_mask, unit_type_mask)

    if DEBUG_OUTPUT:
      image = skimage_color.gray2rgb(numpy.zeros(SCREEN_SHAPE, dtype=numpy.uint8))
      image[union_vespene_mask] = (0, 255, 0)
      ax3.imshow(image)

    union_resource_mask = numpy.logical_or(union_mineral_mask, union_vespene_mask)

    if DEBUG_OUTPUT:
      image = skimage_color.gray2rgb(numpy.zeros(SCREEN_SHAPE, dtype=numpy.uint8))
      image[numpy.logical_and(numpy.logical_not(union_resource_mask), (unanalyzed_density>0))] = (255, 0, 0)
      ax4.imshow(image)
      filename = 'debug_original_density_%02d_%02d.png' % (self._current_camera[0], self._current_camera[1])
      plt.savefig(self.DEBUG_OUTPUT_PATH + '/%s' % filename)
      plt.close(fig0)

    mineral_fragment_density = numpy.zeros(SCREEN_SHAPE, dtype=numpy.int32)
    mineral_fragment_list = []
    mineral_field_list = []

    if DEBUG_OUTPUT:
      fig1, axs = plt.subplots(ncols=5, nrows=2, sharex=True, sharey=True, figsize=(10, 4))
      ax_index = 0
      image = skimage_color.gray2rgb(numpy.zeros(SCREEN_SHAPE, dtype=numpy.uint8))
      self.set_color_in_density_image(image, unanalyzed_density)
      axs[ax_index//5][ax_index%5].imshow(image)
      ax_index += 1

    while union_mineral_mask.any():
      mineral_center = self._detect_mineral_screen(union_mineral_mask)
      if mineral_center is None:
        mineral_temp_mask = numpy.logical_and(union_mineral_mask, (mineral_fragment_density==0))
        mineral_center = self._detect_mineral_screen(mineral_temp_mask)
      if mineral_center is None:
        mineral_temp_mask = numpy.logical_or(union_mineral_mask, (mineral_fragment_density>0))
        mineral_center = self._detect_mineral_screen(mineral_temp_mask)
      if mineral_center is None:
        for old_fragment in mineral_fragment_list:
          fragment_mask = numpy.logical_and((mineral_fragment_density>0), old_fragment)
          mineral_temp_mask = numpy.logical_and(union_mineral_mask, numpy.logical_not(fragment_mask))
          mineral_center = self._detect_mineral_screen(mineral_temp_mask)
          if mineral_center is not None:
            break
          mineral_temp_mask = numpy.logical_and(union_mineral_mask, fragment_mask)
          mineral_center = self._detect_mineral_screen(mineral_temp_mask)
          if mineral_center is not None:
            break
      if mineral_center is not None:
        mineral_field_list.append(mineral_center)
        circle_mask = self.create_mineral_circle_mask(SCREEN_SHAPE, mineral_center)
        unanalyzed_density[circle_mask] -= 1
        if DEBUG_OUTPUT:
          image = skimage_color.gray2rgb(numpy.zeros(SCREEN_SHAPE, dtype=numpy.uint8))
          self.set_color_in_density_image(image, unanalyzed_density)
          axs[ax_index//5][ax_index%5].imshow(image)
          ax_index += 1
        mineral_density = numpy.array(unanalyzed_density)
        mineral_density[numpy.logical_not(union_mineral_mask)] = 0
        mineral_fragment_density[numpy.logical_and((mineral_fragment_density>0), circle_mask)] -= 1
        count_fragments = len(mineral_fragment_list)
        for i in range(count_fragments-1, -1, -1):
          old_fragment = mineral_fragment_list[i]
          if not (mineral_fragment_density[old_fragment]>0).any():
            mineral_fragment_list.pop(i)
        new_fragment = mineral_density[circle_mask]
        if (new_fragment>0).any():
          mineral_fragment_list.append(circle_mask)
          mineral_fragment_density[numpy.logical_and((mineral_density>0),circle_mask)] += 1
        union_mineral_mask = (mineral_density>0)
      else:
        break
    if numpy.logical_and(union_mineral_mask, (mineral_fragment_density==0)).any():
      while union_mineral_mask.any():
        vespene_overlapped_mask = numpy.logical_and(union_vespene_mask, (unanalyzed_density>1))
        stone_overlapped_mask = numpy.logical_and(numpy.logical_not(union_resource_mask), (unanalyzed_density>1))
        mineral_temp_mask = numpy.logical_or(union_mineral_mask, vespene_overlapped_mask)
        mineral_center = self._detect_mineral_screen(mineral_temp_mask)
        if mineral_center is None:
          mineral_temp_mask = numpy.logical_or(union_mineral_mask, stone_overlapped_mask)
          mineral_center = self._detect_mineral_screen(mineral_temp_mask)
        if mineral_center is None:
          mineral_temp_mask = numpy.logical_or(numpy.logical_or(union_mineral_mask, vespene_overlapped_mask), stone_overlapped_mask)
          mineral_center = self._detect_mineral_screen(mineral_temp_mask)
        if mineral_center is None:
          for old_fragment in mineral_fragment_list:
            fragment_mask = numpy.logical_and((mineral_fragment_density>0), old_fragment)
            mineral_temp_mask = numpy.logical_and(union_mineral_mask, numpy.logical_not(fragment_mask))
            mineral_center = self._detect_mineral_screen(mineral_temp_mask)
            if mineral_center is not None:
              break
            mineral_temp_mask = numpy.logical_and(union_mineral_mask, fragment_mask)
            mineral_center = self._detect_mineral_screen(mineral_temp_mask)
            if mineral_center is not None:
              break
        if mineral_center is not None:
          mineral_field_list.append(mineral_center)
          circle_mask = self.create_mineral_circle_mask(SCREEN_SHAPE, mineral_center)
          unanalyzed_density[circle_mask] -= 1
          if DEBUG_OUTPUT:
            image = skimage_color.gray2rgb(numpy.zeros(SCREEN_SHAPE, dtype=numpy.uint8))
            self.set_color_in_density_image(image, unanalyzed_density)
            axs[ax_index//5][ax_index%5].imshow(image)
            ax_index += 1
          mineral_density = numpy.array(unanalyzed_density)
          mineral_density[numpy.logical_not(union_mineral_mask)] = 0
          mineral_fragment_density[numpy.logical_and((mineral_fragment_density>0), circle_mask)] -= 1
          count_fragments = len(mineral_fragment_list)
          for i in range(count_fragments-1, -1, -1):
            old_fragment = mineral_fragment_list[i]
            if not (mineral_fragment_density[old_fragment]>0).any():
              mineral_fragment_list.pop(i)
          new_fragment = mineral_density[circle_mask]
          if (new_fragment>0).any():
            mineral_fragment_list.append(circle_mask)
            mineral_fragment_density[numpy.logical_and((mineral_density>0),circle_mask)] += 1
          union_mineral_mask = (mineral_density>0)
        else:
          break
    if DEBUG_OUTPUT:
      filename = 'debug_mineral_gathering_%02d_%02d.png' % (self._current_camera[0], self._current_camera[1])
      fig1.savefig(self.DEBUG_OUTPUT_PATH + '/%s' % filename)
      plt.close(fig1)

    cloned_density = numpy.array(unit_density)
    cloned_density[numpy.logical_not(neutral_mask)] = 0

    if DEBUG_OUTPUT:
      image = skimage_color.gray2rgb(numpy.zeros(SCREEN_SHAPE, dtype=numpy.uint8))
      self.set_color_in_density_image(image, cloned_density)

    point_arr = numpy.array(mineral_field_list)
    barycenter = point_arr.mean(axis=0)
    for center in mineral_field_list:
      circle_mask = self.create_mineral_circle_mask(SCREEN_SHAPE, center)
      cloned_density[circle_mask] -= 1

    vespene_temp_mask = numpy.logical_and(union_resource_mask, (cloned_density>0))
    temp_vespene_geyser_list = self.get_locations_screen(vespene_temp_mask, 2, False)

    distances = [(self.calculate_distance_square(barycenter, c), c) for c in temp_vespene_geyser_list if union_vespene_mask[c[1], c[0]] ]
    count_temp_vespene_geyser = len(temp_vespene_geyser_list)
    vespene_geyser_list = [ c for d,c in sorted(distances)[0:2 if count_temp_vespene_geyser>=2 else count_temp_vespene_geyser] ]

    if DEBUG_OUTPUT:
      for center in mineral_field_list:
        image[center[1], center[0]] = (0, 0, 255)
      for center in vespene_geyser_list:
        image[center[1], center[0]] = (0, 255, 0)
      p = [self.AccumulatedOffset[i][self._current_camera[i]-self.CameraBoundary[0][i]] for i in range(2)]
      #(pixel_left, pixel_top) = tuple(p)
      #self.resource_image[pixel_top:pixel_top+self.ScreenSize[1], pixel_left:pixel_left+self.ScreenSize[0]] = image[:,:]
      filename = 'debug_center_%02d_%02d_.png' % (self._current_camera[0], self._current_camera[1])
      skimage_io.imsave(self.DEBUG_OUTPUT_PATH + '/%s' % filename, image)

    return (mineral_field_list, vespene_geyser_list)


  def _check_resource_overlapping_townhall(self, screen_shape, townhall_location, mineral_field_list, vespene_geyser_list):
    townhall_margin_mask = self.create_townhall_margin_mask(screen_shape, townhall_location)
    mineral_field_list_mask = self.create_mineral_field_list_mask(screen_shape, mineral_field_list)
    vespene_geyser_list_mask = self.create_vespene_geyser_list_mask(screen_shape, vespene_geyser_list)
    resource_mask = numpy.logical_or(mineral_field_list_mask, vespene_geyser_list_mask)
    conflict_mask = numpy.logical_and(townhall_margin_mask, resource_mask)
    return conflict_mask.any()


  def _draw_debug_camera_height(self, obs):
    camera = obs.observation.feature_minimap.camera
    height_map = obs.observation.feature_screen.height_map
    camera_image = skimage_color.gray2rgb(numpy.zeros(camera.shape, dtype=numpy.uint8))
    camera_image[(camera==1)] = (255,255,255)
    camera_image[self._current_camera[1], self._current_camera[0]] = (255,0,0)
    filename = 'debug_camera_%02d_%02d.png' % (self._current_camera[0], self._current_camera[1])
    skimage_io.imsave(self.DEBUG_OUTPUT_PATH + '/%s' % filename, camera_image)

    height_image = skimage_color.gray2rgb(numpy.array(height_map, dtype=numpy.uint8))

    filename = 'debug_height_%02d_%02d.png' % (self._current_camera[0], self._current_camera[1])
    skimage_io.imsave(self.DEBUG_OUTPUT_PATH + '/%s' % filename, height_image)
    return FUNCTIONS.no_op()


  def _draw_debug_figure(self, townhall_location, mineral_field_list, vespene_geyser_list, append_filename='default'):
    SCREEN_SHAPE = (self.ScreenSize[1], self.ScreenSize[0])
    townhall_image = skimage_color.gray2rgb(numpy.zeros(SCREEN_SHAPE, dtype=numpy.uint8))
    townhall_margin_mask = self.create_townhall_margin_mask(SCREEN_SHAPE, townhall_location)
    townhall_image[townhall_margin_mask] = (85, 0, 0)
    townhall_mask = self.create_circle_mask(SCREEN_SHAPE, townhall_location, 9**2+3**2)
    townhall_image[townhall_mask] = (255, 255, 0)

    resource_density = self._create_resource_density(mineral_field_list, vespene_geyser_list)
    self.set_color_in_density_image(townhall_image, resource_density)

    mineral_field_list_mask = self.create_mineral_field_list_mask(SCREEN_SHAPE, mineral_field_list)
    townhall_image[mineral_field_list_mask] = (0, 0, 170)
    for center in mineral_field_list:
      (mineral_x, mineral_y) = (center[0]+self.MINERAL_BIAS[0], center[1]+self.MINERAL_BIAS[1])
      townhall_image[center[1], center[0]] = (0, 0, 85)

    vespene_geyser_list_mask = self.create_vespene_geyser_list_mask(SCREEN_SHAPE, vespene_geyser_list)
    townhall_image[vespene_geyser_list_mask] = (0, 170, 0)

    for center in vespene_geyser_list:
      (vespene_x, vespene_y) = (center[0]+self.VESPENE_BIAS[0], center[1]+self.VESPENE_BIAS[1])
      townhall_image[center[1], center[0]] = (0, 85, 0)

    filename = 'debug_townhall_%02d_%02d_%s.png' % (self._current_camera[0], self._current_camera[1], append_filename)
    skimage_io.imsave(self.DEBUG_OUTPUT_PATH + '/%s' % filename, townhall_image)
    return townhall_image


  def _calculate_townhall_best_location(self, mineral_field_list, vespene_geyser_list, debug_output=False):
    DEBUG_OUTPUT = debug_output

    count_mineral_source = len(mineral_field_list)
    count_vespene_source = len(vespene_geyser_list)

    if count_mineral_source < 2:
      return None
    GRID_COUNT_townhall = self._get_grid_count_townhall()
    GRID_COUNT_gas_plant = self._get_grid_count_gas_plant()
    GRID_COUNT_PRESERVED = 4.8
    MINERAL_DISTANCE_FROM_TOWNHALL = [self.GRID_SIDE_LENGTH*(GRID_COUNT_townhall/2+GRID_COUNT_PRESERVED+1), self.GRID_SIDE_LENGTH*(GRID_COUNT_townhall/2+GRID_COUNT_PRESERVED+0.5)]
    VESPENE_DISTANCE_FROM_TOWNHALL = self.GRID_SIDE_LENGTH*((GRID_COUNT_townhall+GRID_COUNT_gas_plant)/2+GRID_COUNT_PRESERVED)
    TOWNHALL_DIAMETER = GRID_COUNT_townhall*(self.GRID_SIDE_LENGTH+1)-1
    SCREEN_CENTER = (self.ScreenSize[0]/2, self.ScreenSize[1]/2)
    SCREEN_SHAPE = (self.ScreenSize[1], self.ScreenSize[0])

    point_arr = numpy.array(mineral_field_list)
    barycenter = point_arr.mean(axis=0)
    left_most_index, top_most_index = point_arr.argmin(axis=0)
    right_most_index, bottom_most_index = point_arr.argmax(axis=0)
    (left_most, top_most) = (mineral_field_list[left_most_index][0]-4, mineral_field_list[top_most_index][1]-4)
    (right_most, bottom_most) = (mineral_field_list[right_most_index][0]+3, mineral_field_list[bottom_most_index][1]+3)
    mineral_region = (right_most-left_most, bottom_most-top_most)
    vertical_middle = (top_most+bottom_most)/2.0
    horizontal_middle = (left_most+right_most)/2.0
    center = (horizontal_middle, vertical_middle)
    candidate_point = None
    intersection_points = None
    if count_vespene_source == 2:
      circles = [(*vespene_geyser_list[i], VESPENE_DISTANCE_FROM_TOWNHALL) for i in (0, 1) ]
      intersection_points = [ (int(round(x)), int(round(y))) for (x, y) in self._calculate_circles_intersection_points(*circles)]
    if mineral_region[0] < TOWNHALL_DIAMETER and mineral_region[1] < TOWNHALL_DIAMETER:
      pass
    elif mineral_region[0] >= TOWNHALL_DIAMETER and mineral_region[1] >= TOWNHALL_DIAMETER:    # 礦區像 L 型
      corners = [(left_most, top_most), (right_most, top_most), (left_most, bottom_most), (right_most, bottom_most)]
      distance_sum = [0] * 4
      chosen_index = -1
      for k in range(4):
        for mineral_center in mineral_field_list:
          distance_sum[k] += math.sqrt(self.calculate_distance_square(corners[k], mineral_center))
        if -1 == chosen_index or distance_sum[k] > distance_sum[chosen_index]:
          chosen_index = k

      corner_point = corners[chosen_index]
      if intersection_points is None:
        corner_minerals = (mineral_field_list[left_most_index if 0==chosen_index%2 else right_most_index], mineral_field_list[top_most_index if 0==chosen_index//2 else bottom_most_index])
        circles = [(*corner_minerals[i], MINERAL_DISTANCE_FROM_TOWNHALL[i]) for i in (0, 1) ]
        intersection_points = [ (int(round(x)), int(round(y))) for (x, y) in self._calculate_circles_intersection_points(*circles)]

      distance_from_intersection = [self.calculate_distance_square(corner_point, p) for p in intersection_points]
      if distance_from_intersection[0] < distance_from_intersection[1]:
        candidate_point = intersection_points[0]
      else:
        candidate_point = intersection_points[1]
      if True == self._check_resource_overlapping_townhall(SCREEN_SHAPE, candidate_point, mineral_field_list, vespene_geyser_list):
        if DEBUG_OUTPUT:
          self._draw_debug_figure(candidate_point, mineral_field_list, vespene_geyser_list, 'calculated_%02d_%02d'% (candidate_point[0], candidate_point[1]))
        nearest_point = candidate_point
        nearest_distance = None
        direction_offset = (1-chosen_index%2*2, 1-chosen_index//2*2)
        vertical_range = range(candidate_point[1]-direction_offset[1]*6, candidate_point[1]+direction_offset[1]*3, direction_offset[1])
        horizontal_range = range(candidate_point[0]-direction_offset[0]*6, candidate_point[0]+direction_offset[0]*3, direction_offset[0])
        if DEBUG_OUTPUT:
          fig, subfig = plt.subplots(nrows=len(list(vertical_range)), ncols=len(list(horizontal_range)), sharex=True, sharey=True)
          plt.subplots_adjust(wspace=0.6, hspace=0.6)
          flg_row = 0
          for y in vertical_range:
            flg_column = 0
            for x in horizontal_range:
              chosen_point = (x, y)
              if not self._check_resource_overlapping_townhall(SCREEN_SHAPE, chosen_point, mineral_field_list, vespene_geyser_list):
                chosen_img = self._draw_debug_figure(chosen_point, mineral_field_list, vespene_geyser_list, 'choose_%02d_%02d_valid' % (x, y))
                subfig[flg_row][flg_column].set_title('Yes', fontdict={'fontsize': 8, 'fontweight': 'medium'})
                distance = self.calculate_distance_square(barycenter, chosen_point)
                if nearest_distance is None or distance < nearest_distance:
                  nearest_point = chosen_point
                  nearest_distance = distance
              else:
                chosen_img = self._draw_debug_figure(chosen_point, mineral_field_list, vespene_geyser_list, 'choose_%02d_%02d_invalid' % (x, y))
                subfig[flg_row][flg_column].set_title('No', fontdict={'fontsize': 8, 'fontweight': 'medium'})
              subfig[flg_row][flg_column].imshow(chosen_img)
              flg_column += 1
            flg_row += 1
          filename = 'debug_choose_%02d_%02d.png' % (self._current_camera[0], self._current_camera[1])
          plt.savefig(self.DEBUG_OUTPUT_PATH + '/%s' % filename)
          plt.close(fig)
          self._draw_debug_figure(nearest_point, mineral_field_list, vespene_geyser_list, 'choose_%02d_%02d_final' % (nearest_point[0], nearest_point[1]))
        else:
          for y in vertical_range:
            for x in horizontal_range:
              chosen_point = (x, y)
              if not self._check_resource_overlapping_townhall(SCREEN_SHAPE, chosen_point, mineral_field_list, vespene_geyser_list):
                distance = self.calculate_distance_square(barycenter, chosen_point)
                if nearest_distance is None or distance < nearest_distance:
                  nearest_point = chosen_point
                  nearest_distance = distance
        candidate_point = nearest_point
      else:
        if DEBUG_OUTPUT:
          self._draw_debug_figure(candidate_point, mineral_field_list, vespene_geyser_list, 'calculated_%02d_%02d_final'% (candidate_point[0], candidate_point[1]))
    else:
      direction_offset = None

      MINERAL_DIAGONAL_DISTANCE_FROM_TOWNHALL = math.sqrt(MINERAL_DISTANCE_FROM_TOWNHALL[1]**2+MINERAL_DISTANCE_FROM_TOWNHALL[1]**2)
      if mineral_region[0] < TOWNHALL_DIAMETER:    # 礦區像直的(狹長)
        center_left = (left_most, vertical_middle)
        center_right = (right_most, vertical_middle)
        distance_from_left = self.calculate_distance_square(SCREEN_CENTER, center_left)
        distance_from_right = self.calculate_distance_square(SCREEN_CENTER, center_right)
        if distance_from_left < distance_from_right:
          center = center_left
          direction_offset = (-1, 0)
        else:
          center = center_right
          direction_offset = (1, 0)
        if intersection_points is None:
          corner_minerals = (mineral_field_list[top_most_index], mineral_field_list[bottom_most_index])
          circles = [(*corner_minerals[i], MINERAL_DIAGONAL_DISTANCE_FROM_TOWNHALL) for i in (0, 1) ]
      elif mineral_region[1] < TOWNHALL_DIAMETER:    # 礦區像橫的(扁平)
        center_top = (horizontal_middle, top_most)
        center_bottom = (horizontal_middle, bottom_most)
        distance_from_top = self.calculate_distance_square(SCREEN_CENTER, center_top)
        distance_from_bottom = self.calculate_distance_square(SCREEN_CENTER, center_bottom)
        if distance_from_top < distance_from_bottom:
          center = center_top
          direction_offset = (0, -1)
        else:
          center = center_bottom
          direction_offset = (0, 1)
        if intersection_points is None:
          corner_minerals = (mineral_field_list[left_most_index], mineral_field_list[right_most_index])
          circles = [(*corner_minerals[i], MINERAL_DIAGONAL_DISTANCE_FROM_TOWNHALL) for i in (0, 1) ]
      if intersection_points is None:
        intersection_points = [ (int(round(x)), int(round(y))) for (x, y) in self._calculate_circles_intersection_points(*circles)]

      farthest_point = (center[0]+direction_offset[0]*MINERAL_DISTANCE_FROM_TOWNHALL[0], center[1]+direction_offset[1]*MINERAL_DISTANCE_FROM_TOWNHALL[1])
      distance_from_intersection = [self.calculate_distance_square(farthest_point, p) for p in intersection_points]
      if distance_from_intersection[0] < distance_from_intersection[1]:
        candidate_point = intersection_points[0]
      else:
        candidate_point = intersection_points[1]
      if DEBUG_OUTPUT:
        self._draw_debug_figure(candidate_point, mineral_field_list, vespene_geyser_list, 'calculated_%02d_%02d'% (candidate_point[0], candidate_point[1]))
      while True == self._check_resource_overlapping_townhall(SCREEN_SHAPE, candidate_point, mineral_field_list, vespene_geyser_list):
        candidate_point = (candidate_point[0]+direction_offset[0], candidate_point[1]+direction_offset[1])
      if DEBUG_OUTPUT:
        self._draw_debug_figure(candidate_point, mineral_field_list, vespene_geyser_list, 'choose_%02d_%02d_final' % (candidate_point[0], candidate_point[1]))
    return candidate_point


  def _execute_moving_camera(self, obs, camera_minimap, debug_func=None):
    if debug_func is not None:
      debug_func()
    if self._current_camera != camera_minimap:
      self._current_camera = camera_minimap
      return FUNCTIONS.move_camera(camera_minimap)
    return FUNCTIONS.no_op()


  def _record_townhall_best_location(self, obs, scheduled_camera, debug_output=False):
    DEBUG_OUTPUT = debug_output
    height_map = obs.observation.feature_screen.height_map
    if self._current_camera not in self._height_map_on_camera:
      self._height_map_on_camera[self._current_camera] = numpy.array(height_map)
    unit_type = obs.observation.feature_screen.unit_type
    unit_density = obs.observation.feature_screen.unit_density
    player_relative = obs.observation.feature_screen.player_relative
    mineral_field_list, vespene_geyser_list = self._get_resource_screen(unit_type, unit_density, player_relative, debug_output)
    townhall_best_location = self._calculate_townhall_best_location(mineral_field_list, vespene_geyser_list, debug_output)
    world_coordinate = self.calculate_world_absolute_coordinate((self._current_camera, townhall_best_location))
    local_coordinate = self.calculate_local_coordinate(world_coordinate)
    self._calculated_resource_region_list.append( [local_coordinate[0], local_coordinate[1]] )
    #self._calculated_resource_region_list.append( [local_coordinate[0], None] )

    #self._calculated_resource_region_list.append( [local_coordinate[0], None] )
    #self._calculated_resource_region_list.append( [self._current_camera, None] )
    count_remaining_schedule = len(scheduled_camera)
    if count_remaining_schedule > 1:
      next_camera = scheduled_camera.pop()
      self._schedule_job( next_camera , None, ['_record_townhall_best_location', self, [scheduled_camera]], True)
    elif 1 == count_remaining_schedule:
      self._draw_debug_world_height_map()
      next_camera = scheduled_camera.pop()
    else:
      return FUNCTIONS.no_op()
    return self._execute_moving_camera(obs, next_camera)


  def _record_townhall_best_locations(self, obs):
    resource_region_list = self._speculate_resource_regions()
    #self.resource_image = skimage_color.gray2rgb(numpy.zeros(self.MapShape, dtype=numpy.uint8))
    NEIGHBOR_DISTANCE_SQURE = self.ViewportSize[0]*self.ViewportSize[1]*9.0/64.0
    center_minimap = ((self.CameraBoundary[1][0]+self.CameraBoundary[0][0])/2.0, (self.CameraBoundary[1][1]+self.CameraBoundary[0][1])/2.0)
    # 檢查礦區是否點(旋轉)對稱
    remaining_region = resource_region_list[:]
    symmetry_style = [None, None, None]
    opposite_mapping = {}
    while(len(remaining_region)>1):
      chosen_region = remaining_region.pop()
      opposite_index = None
      for test_index in range(len(remaining_region)):
        opposite_region = remaining_region[test_index]
        middle = ((chosen_region[0]+opposite_region[0])/2.0, (chosen_region[1]+opposite_region[1])/2.0)
        if self.calculate_distance_square(center_minimap, middle) <= NEIGHBOR_DISTANCE_SQURE:
          opposite_index = test_index
          break
      if opposite_index is not None:
        opposite_region = remaining_region[opposite_index]
        opposite_mapping[chosen_region] = opposite_region
        opposite_mapping[opposite_region] = chosen_region
        remaining_region.pop(opposite_index)
      else:
        break
    if len(remaining_region)==0:
      symmetry_style[2] = opposite_mapping
    # 檢查礦區是否線(垂直翻轉、水平翻轉)對稱
    for axis_index in range(2):
      remaining_region = resource_region_list[:]
      opposite_mapping = {}
      while(len(remaining_region)>1):
        chosen_region = remaining_region.pop()
        opposite_index = None
        for test_index in range(len(remaining_region)):
          opposite_region = remaining_region[test_index]
          if abs(chosen_region[1-axis_index]-opposite_region[1-axis_index]) < self.ViewportSize[1-axis_index]/2.0:
            middle = (chosen_region[axis_index]+opposite_region[axis_index])/2.0
            if abs(center_minimap[axis_index] - middle) < self.ViewportSize[axis_index]/2.0:
              opposite_index = test_index
              break
        if opposite_index is not None:
          opposite_region = remaining_region[opposite_index]
          opposite_mapping[chosen_region] = opposite_region
          opposite_mapping[opposite_region] = chosen_region
          remaining_region.pop(opposite_index)
        else:
          break
      if len(remaining_region)==0:
        symmetry_style[axis_index] = opposite_mapping
    symmetry_debug = [ False if d is None else True for d in symmetry_style ]
    txt_filename = 'debug_symmetry.json.txt'
    with open(self.DEBUG_OUTPUT_PATH + '/%s' % txt_filename, "w") as outfile:
      json.dump({'horizontal_symmetry':symmetry_debug[0], 'vertical_symmetry':symmetry_debug[1], 'rotational':symmetry_debug[2]}, outfile)
    
    near_region_list = deque()
    far_region_list = deque()
    remaining_region = set(resource_region_list)
    holding_index = None
    for i in range(len(resource_region_list)):
      resource_region_camera = resource_region_list[i]
      if self.FirstViewport[0][0]<resource_region_camera[0] and resource_region_camera[0]<self.FirstViewport[1][0] and self.FirstViewport[0][1]<resource_region_camera[1] and resource_region_camera[1]<self.FirstViewport[1][1]:
        holding_index = i
        break
    holding_region = None
    if holding_index is not None:
      holding_region = resource_region_list[holding_index]
    if symmetry_style[2] is not None:
      nearest_region = holding_region
      while nearest_region is not None:
        opposite_region = symmetry_style[2][nearest_region]
        remaining_region.discard(nearest_region)
        remaining_region.discard(opposite_region)
        near_region_list.append(nearest_region)
        far_region_list.append(opposite_region)      
        nearest_distance = None
        nearest_region = None
        barycenter = numpy.array(near_region_list).mean(axis=0)
        for region_coordinate in list(remaining_region):
          distance = self.calculate_distance_square(barycenter, region_coordinate)
          if nearest_distance is None or distance < nearest_distance:
            nearest_distance = distance
            nearest_region = region_coordinate
    elif symmetry_style[0] is not None and symmetry_style[1] is not None and len(resource_region_list)%4==0:
      nearest_region = holding_region
      process_times = 0
      while nearest_region is not None:
        horizontal_symmetry = symmetry_style[0][nearest_region]
        vertical_symmetry = symmetry_style[1][nearest_region]
        if symmetry_style[1][horizontal_symmetry] == symmetry_style[0][vertical_symmetry]:
          opposite_region = symmetry_style[0][vertical_symmetry]
          remaining_region.discard(nearest_region)
          remaining_region.discard(opposite_region)
          near_region_list.append(nearest_region)
          far_region_list.append(opposite_region)
          if self.calculate_distance_square(nearest_region, horizontal_symmetry) < self.calculate_distance_square(holding_region, vertical_symmetry):
            near_region_list.appendleft(horizontal_symmetry)
            far_region_list.appendleft(vertical_symmetry)
          else:
            near_region_list.appendleft(vertical_symmetry)
            far_region_list.appendleft(horizontal_symmetry)
          remaining_region.discard(horizontal_symmetry)
          remaining_region.discard(vertical_symmetry)
          process_times += 1
          nearest_distance = None
          nearest_region = None
          for region_coordinate in list(remaining_region):
            distance = self.calculate_distance_square(holding_region, region_coordinate)
            if nearest_distance is None or distance < nearest_distance:
              nearest_distance = distance
              nearest_region = region_coordinate
        else:
          break
      near_region_list.rotate(process_times)
      far_region_list.rotate(process_times)
    else:
      symmetry_axis = None
      if symmetry_style[0] is not None:    #左右鏡射
        symmetry_axis = 0
      elif symmetry_style[1] is not None:    #上下鏡射
        symmetry_axis = 1
      if symmetry_axis is not None:
        threshold_distance = (self.CameraBoundary[1][symmetry_axis]-self.CameraBoundary[0][symmetry_axis])/2.0
        nearest_region = holding_region
        while nearest_region is not None:
          opposite_region = symmetry_style[symmetry_axis][nearest_region]
          remaining_region.discard(nearest_region)
          remaining_region.discard(opposite_region)
          near_region_list.append(nearest_region)
          far_region_list.append(opposite_region)
          nearest_distance = None
          nearest_region = None
          barycenter = numpy.array(near_region_list).mean(axis=0)
          for region_coordinate in list(remaining_region):
            if abs(region_coordinate[symmetry_axis]-holding_region[symmetry_axis]) < threshold_distance:
              distance = self.calculate_distance_square(barycenter, region_coordinate)
              if nearest_distance is None or distance < nearest_distance:
                nearest_distance = distance
                nearest_region = region_coordinate
      else:
        nearest_region = holding_region
        while nearest_region is not None:
          remaining_region.discard(nearest_region)
          near_region_list.append(nearest_region)
          nearest_distance = None
          nearest_region = None
          barycenter = numpy.array(near_region_list).mean(axis=0)
          for region_coordinate in list(remaining_region):
            distance = self.calculate_distance_square(barycenter, region_coordinate)
            if nearest_distance is None or distance < nearest_distance:
              nearest_distance = distance
              nearest_region = region_coordinate
    far_region_list.reverse()
    reordered_region_list = list(near_region_list) + list(far_region_list)

    #holding_index = None
    #for i in range(len(resource_region_list)):
    #  resource_region_camera = resource_region_list[i]
    #  if self.FirstViewport[0][0]<resource_region_camera[0] and resource_region_camera[0]<self.FirstViewport[1][0] and self.FirstViewport[0][1]<resource_region_camera[1] and resource_region_camera[1]<self.FirstViewport[1][1]:
    #    holding_index = i
    #    break

    #if holding_index is not None:
    #  referenced_coordinate = resource_region_list[holding_index]
    #  self._speculated_resource_region_list.append(referenced_coordinate)
    #  resource_region_list.pop(holding_index)
    #  distace_square_values = []
    #  for i in range(len(resource_region_list)):
    #    distace_square_value = self.calculate_distance_square(referenced_coordinate, resource_region_list[i])
    #    distace_square_values.append((distace_square_value, i))
    #  for (distace_square_value, i) in sorted(distace_square_values):
    #    self._speculated_resource_region_list.append(resource_region_list[i])

    scheduled_camera = [self._current_camera] + reordered_region_list[:0:-1]
    next_camera = scheduled_camera.pop()
    self._schedule_job( next_camera , None, ['_record_townhall_best_location', self, [scheduled_camera]], True)
    return self._execute_moving_camera(obs, next_camera)


  def _generate_world_height_map(self):
    world_height_map = numpy.zeros(self.MapShape, dtype=numpy.uint8)
    for camera_minimap in self._height_map_on_camera.keys():
      height_map = self._height_map_on_camera[camera_minimap]
      p = [self.AccumulatedOffset[i][camera_minimap[i]-self.CameraBoundary[0][i]] for i in range(2)]
      (pixel_left, pixel_top) = tuple(p)
      world_height_map[pixel_top:pixel_top+self.ScreenSize[1], pixel_left:pixel_left+self.ScreenSize[0]] = height_map[:,:]
    return world_height_map


  def _draw_debug_world_height_map(self, filename='debug_world_height.png'):
    txt_filename = 'debug_accumulate.json.txt'
    with open(self.DEBUG_OUTPUT_PATH + '/%s' % txt_filename, "w") as outfile:
      json.dump({'column_offset':self.AccumulatedOffset[0], 'row_offset':self.AccumulatedOffset[1]}, outfile)
    world_height_image = skimage_color.gray2rgb(self._generate_world_height_map())
    skimage_io.imsave(self.DEBUG_OUTPUT_PATH + '/%s' % filename, world_height_image)


  def _draw_debug_world_resource(self, filename='debug_world_resource.png'):
    skimage_io.imsave(self.DEBUG_OUTPUT_PATH + '/%s' % filename, self.resource_image)


  def _calculate_world_map(self, obs):
    reserved_offset = (2, 2)
    pixel_offset = [None, None]
    for axis_index in range(2):
      pixel_offset[axis_index] = [0] * (self.CameraBoundary[1][axis_index]-self.CameraBoundary[0][axis_index]+1)
      prepared_point = [list(self.CameraBoundary[1]), list(self.CameraBoundary[0])]
      prepared_point[1][axis_index] += reserved_offset[axis_index]
      matching_times = [self.ViewportSize[axis_index]+reserved_offset[axis_index], reserved_offset[axis_index]]
      for z in range(2):
        current_point = prepared_point[z]
        current_height_map = self._height_map_on_camera[ tuple(current_point) ]
        current_height_sum = current_height_map.sum(axis=axis_index, dtype=numpy.uint32)
        for t in range(matching_times[z]):
          previous_height_sum = current_height_sum
          current_point[axis_index] -= 1
          current_height_map = self._height_map_on_camera[ tuple(current_point) ]
          current_height_sum = current_height_map.sum(axis=axis_index, dtype=numpy.uint32)
          overlapped_length = 0
          for current_start_index in range(self.ScreenSize[axis_index]):
            perfect_overlapped = True
            for line_count in range(self.ScreenSize[axis_index]-current_start_index):
              if previous_height_sum[line_count] != current_height_sum[current_start_index+line_count]:
                perfect_overlapped = False
                break
            if perfect_overlapped:
              overlapped_length = current_start_index
              break
          pixel_offset[axis_index][current_point[axis_index]-self.CameraBoundary[0][axis_index]+1] = overlapped_length
      for k in range(self.CameraBoundary[1][axis_index]-self.ViewportSize[axis_index]-reserved_offset[axis_index], self.CameraBoundary[0][axis_index]+reserved_offset[axis_index], -1):
        pixel_offset[axis_index][k-self.CameraBoundary[0][axis_index]] = pixel_offset[axis_index][k-self.CameraBoundary[0][axis_index]+self.ViewportSize[axis_index]]
      for k in range(1, len(pixel_offset[axis_index])):
        pixel_offset[axis_index][k] += pixel_offset[axis_index][k-1]
    self.AccumulatedOffset = pixel_offset

    self.MapShape = (self.AccumulatedOffset[1][-1]+self.ScreenSize[1], self.AccumulatedOffset[0][-1]+self.ScreenSize[0])
    #self._draw_debug_world_height_map()

  def _look_around_world(self, obs, scheduled_camera):
    height_map = obs.observation.feature_screen.height_map
    #self._draw_debug_camera_height(obs)
    world_corner = (self.MinimapSize[0]-1,self.MinimapSize[1]-1)
    next_camera = self._current_camera
    if scheduled_camera is None:
      scheduled_camera = [ self._current_camera, (0,0), world_corner ]
      next_camera = scheduled_camera.pop()
      self._schedule_job( next_camera , None, ['_look_around_world', self, [scheduled_camera]], True)
    elif self._current_camera == world_corner:
      camera = obs.observation.feature_minimap.camera
      y, x = (camera == 1).nonzero()
      viewport = ((x.min(),y.min()), (x.max(),y.max()))
      self.CameraBoundary[1] = (viewport[0][0]+self.ViewportCenter[0]+1, viewport[0][1]+self.ViewportCenter[1]+1)
      scheduled_camera.append(self.CameraBoundary[1])
      for y in range(self.CameraBoundary[1][1]-(self.ViewportSize[1]+2), self.CameraBoundary[1][1]):
        scheduled_camera.append( (self.CameraBoundary[1][0], y) )
      for x in range(self.CameraBoundary[1][0]-(self.ViewportSize[0]+2), self.CameraBoundary[1][0]):
        scheduled_camera.append( (x, self.CameraBoundary[1][1]) )
      next_camera = scheduled_camera.pop()
      self._schedule_job( next_camera , None, ['_look_around_world', self, [scheduled_camera]], True)
    elif self._current_camera == (0,0):
      camera = obs.observation.feature_minimap.camera
      y, x = (camera == 1).nonzero()
      viewport = ((x.min(),y.min()), (x.max(),y.max()))
      self.CameraBoundary[0] = (viewport[0][0]+self.ViewportCenter[0]-1, viewport[0][1]+self.ViewportCenter[1]-1)
      scheduled_camera.append(self.CameraBoundary[0])
      for y in range(self.CameraBoundary[0][1]+2, self.CameraBoundary[0][1], -1):
        scheduled_camera.append( (self.CameraBoundary[0][0], y) )
      for x in range(self.CameraBoundary[0][0]+2, self.CameraBoundary[0][0], -1):
        scheduled_camera.append( (x, self.CameraBoundary[0][1]) )
      next_camera = scheduled_camera.pop()
      self._schedule_job( next_camera , None, ['_look_around_world', self, [scheduled_camera]], True)
    elif len(scheduled_camera) > 1:
      self._height_map_on_camera[self._current_camera] = numpy.array(height_map)
      next_camera = scheduled_camera.pop()
      self._schedule_job( next_camera , None, ['_look_around_world', self, [scheduled_camera]], True)
    else:
      self._height_map_on_camera[self._current_camera] = numpy.array(height_map)
      self._calculate_world_map(obs)
      next_camera = scheduled_camera.pop()
    return self._execute_moving_camera(obs, next_camera)


  def _test_build_townhall(self, obs, local_coordinate):
    townhall_mineral_cost = _UnitNumeric[self.TOWNHALL_TYPES[0]]['mineral_cost']
    camera_minimap = local_coordinate[0]
    if local_coordinate[1] is None:
      local_coordinate[1] = self._calculate_townhall_best_location(obs, True)
    townhall_best_location = local_coordinate[1]
    if obs.observation.player.minerals >= townhall_mineral_cost:
      action_id = _UnitDependency[self.TOWNHALL_TYPES[0]][0]['perform']
      self._schedule_job(camera_minimap, self.WORKER_TYPE, [action_id, ['queued', townhall_best_location]], True)
    else:
      self._schedule_job(camera_minimap, self.WORKER_TYPE, ['_test_build_townhall', self, [local_coordinate]], True)
    return FUNCTIONS.no_op()


  def _walk_through_townhall_best_locations(self, obs):
    if len(self._calculated_resource_region_list) <= 1:
      return None
    last_camera_minimap = self._current_camera
    for i in range(len(self._calculated_resource_region_list)-1, 0, -1):
      local_coordinate = self._calculated_resource_region_list[i]
      camera_minimap = local_coordinate[0]
      self._schedule_job(camera_minimap, self._expected_selected, ['_execute_moving_camera', self, [last_camera_minimap]], True)
      #self._schedule_job(camera_minimap, self._expected_selected, [FUNCTIONS.Move_screen.id, ['queued', location_screen]], True)
      #action_id = _UnitDependency[self.TOWNHALL_TYPES[0]][0]['perform']
      #self._schedule_job(camera_minimap, self._expected_selected, [action_id, ['queued', location_screen]], True)
      self._schedule_job(camera_minimap, self.WORKER_TYPE, ['_test_build_townhall', self, [local_coordinate]], True)
      last_camera_minimap = camera_minimap
    self._schedule_job(self._current_camera, self._expected_selected, ['_execute_moving_camera', self, [last_camera_minimap]], True)
    return FUNCTIONS.no_op()


  def _speculate_resource_regions(self):
    NEIGHBOR_DISTANCE_SQURE = self.ViewportSize[0]*self.ViewportSize[1]*9.0/64.0
    grouped_points = self.aggregate_points((self.NeutralMinimap == features.PlayerRelative.NEUTRAL), NEIGHBOR_DISTANCE_SQURE)
    return_locs = []
    for exist_serial in grouped_points.keys():
      if len(grouped_points[exist_serial]) > 0:
        point_arr = numpy.array(list(grouped_points[exist_serial]))
        left, top = point_arr.min(axis=0)
        right, bottom = point_arr.max(axis=0)
        center = [ int(round((left+right)/2.0)), int(round((top+bottom)/2.0)) ]
        for index in range(2):
          if center[index] < self.CameraBoundary[0][index]:
            center[index] = self.CameraBoundary[0][index]
          elif center[index] > self.CameraBoundary[1][index]:
            center[index] = self.CameraBoundary[1][index]
        return_locs.append(tuple(center))
    return return_locs


  def _execute_returning_harvest(self, obs, townhall_location):
    if FUNCTIONS.Harvest_Return_quick.id in obs.observation.available_actions:
      self._schedule_job(self._current_camera, self.WORKER_TYPE, ['_execute_returning_harvest', self, [townhall_location] ], True)
      return FUNCTIONS.Harvest_Return_quick('now')
    else:
      return FUNCTIONS.Move_screen('now', townhall_location)


  def _select_gathering_mineral_worker(self, obs, townhall_location):
    unit_type = obs.observation.feature_screen.unit_type
    selected = obs.observation.feature_screen.selected
    gas_plant_list = self.get_locations_screen(unit_type == self.GAS_PLANT_TYPE)
    worker_mask = (unit_type == self.WORKER_TYPE)
    selected_mask = (selected == 0)
    unselected_worker_mask = numpy.logical_and(worker_mask, selected_mask)
    worker_list = self.get_locations_screen(unselected_worker_mask)
    (townhall_x, townhall_y) = townhall_location
    nearest_loc = None
    nearest_distance = None
    base_axis = []
    for gas_plant in gas_plant_list:
      (gas_plant_x, gas_plant_y) = gas_plant
      base_axis_vec = (townhall_x - gas_plant_x, townhall_y - gas_plant_y)
      base_axis_len = math.sqrt(base_axis_vec[0]*base_axis_vec[0]+base_axis_vec[1]*base_axis_vec[1])
      base_axis_vec = (base_axis_vec[0]/base_axis_len, base_axis_vec[1]/base_axis_len)
      base_axis.append((base_axis_vec, base_axis_len))

    GRID_COUNT_townhall = self._get_grid_count_townhall()
    RADIUS_townhall = int(math.floor(self.GRID_SIDE_LENGTH*GRID_COUNT_townhall/2.0))
    NEAR_townhall = RADIUS_townhall+self.GRID_SIDE_LENGTH
    FAR_townhall = RADIUS_townhall+self.GRID_SIDE_LENGTH*3

    for worker in worker_list:
      (worker_x, worker_y) = worker
      if unit_type[worker_y][worker_x] != self.WORKER_TYPE:
        continue
      gathering_vespene = False
      for g in range(len(gas_plant_list)):
        gas_plant = gas_plant_list[g]
        (gas_plant_x, gas_plant_y) = gas_plant
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
      distance = self.calculate_distance_square(worker, townhall_location)
      if distance<=NEAR_townhall**2 or distance>FAR_townhall**2:
        continue
      #if worker_x < townhall_x-12 or worker_x > townhall_x+12 or worker_y < townhall_y-12 or worker_y > townhall_y+12:
      #  distance = self.calculate_distance_square(worker, townhall_location)
      if nearest_distance is None or distance<nearest_distance:
        nearest_loc = worker
        nearest_distance = distance

    if nearest_loc is not None:
      #self._schedule_job(self._current_camera, None, [FUNCTIONS.Stop_quick.id, ['now']], True)
      #self._schedule_job(self._current_camera, self.WORKER_TYPE, [FUNCTIONS.no_op.id, [] ], True)
      #self._schedule_job(self._current_camera, self.WORKER_TYPE, [FUNCTIONS.no_op.id, [] ], True)
      #self._schedule_job(self._current_camera, self.WORKER_TYPE, [FUNCTIONS.Harvest_Return_quick.id, ['queued']], True)
      #self._schedule_job(self._current_camera, self.WORKER_TYPE, [FUNCTIONS.Move_screen.id, ['now', townhall_location ]], True)
      #self._schedule_job(self._current_camera, None, [FUNCTIONS.select_point.id, ['select', nearest_loc[0] ]], True)
      self._schedule_job(self._current_camera, self.WORKER_TYPE, ['_execute_returning_harvest', self, [townhall_location] ], True)
      self._expected_selected = self.WORKER_TYPE
      return FUNCTIONS.select_point('select', nearest_loc)
    else:
      self._schedule_job(self._current_camera, None, ['_select_gathering_mineral_worker', self, [townhall_location] ], True)
      #self._schedule_job(self._current_camera, None, [FUNCTIONS.no_op.id, [] ], True)
      return FUNCTIONS.no_op()


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
      next_action = scheduled_actions.popleft()
      if isinstance(next_action[0], int):
        #if next_action[0] == FUNCTIONS.move_camera.id:
        #  if self._current_camera != next_action[1][0]:
        #    self._current_camera = next_action[1][0]
        #    return actions.FunctionCall.init_with_validation(next_action[0], next_action[1])
        if next_action[0] in obs.observation.available_actions:
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


  def _game_start(self, obs):
    ready_function_call = FUNCTIONS.no_op()
    owner = obs.observation.player.player_id
    townhall_location_list = self._get_my_townhall_screen(obs, False)
    self._world_coordinate[owner] = {}

    if 1 == len(townhall_location_list):
      unit_type_id = self.TOWNHALL_TYPES[0]
      townhall_location = townhall_location_list[0]

      #townhall_best_location = self._calculate_townhall_best_location(obs)

      #world_absolute_coordinate = self.calculate_world_absolute_coordinate((self._current_camera, townhall_location))
      self._calculated_resource_region_list.append( (self._current_camera, townhall_location) )
      self._occupied_resource_regions[self._current_camera] = {'owner':owner}
      self._holding_resource_region_list.append( (self._current_camera, townhall_location) )
      #self._world_coordinate[owner][unit_type_id] = [world_absolute_coordinate]
      #self._structures[world_absolute_coordinate] = {'owner':owner, 'unit_type':unit_type_id}
      self._expected_selected = unit_type_id
      self._schedule_job(self._current_camera, unit_type_id, [FUNCTIONS.select_control_group.id, ['set', 0]])
      ready_function_call = FUNCTIONS.select_point('select', townhall_location)
    return ready_function_call

  def step(self, obs):
    if type(self) == GeneralAgent:
      return FUNCTIONS.no_op()
    else:
      if self.CameraBias is None:
        camera = obs.observation.feature_minimap.camera
        y, x = (camera == 1).nonzero()
        viewport = ((x.min(),y.min()), (x.max(),y.max()))
        if self.FirstViewport is None:
          self.FirstViewport = viewport
          self.NeutralMinimap = numpy.array(obs.observation.feature_minimap.player_relative)
          self.ViewportSize =  (self.FirstViewport[1][0]-self.FirstViewport[0][0]+1, self.FirstViewport[1][1]-self.FirstViewport[0][1]+1)
          screen_height_map = obs.observation.feature_screen.height_map
          self.ScreenSize = (screen_height_map.shape[1], screen_height_map.shape[0])
          minimap_height_map = obs.observation.feature_minimap.height_map
          self.MinimapSize = (minimap_height_map.shape[1], minimap_height_map.shape[0])
          self._current_camera = (int(math.floor((self.FirstViewport[0][0]+self.FirstViewport[1][0]+1)/2)), int(math.floor((self.FirstViewport[0][1]+self.FirstViewport[1][1]+1)/2)))
        else:
          self.CameraBias = ((viewport[0][0]-self.FirstViewport[0][0]), (viewport[0][1]-self.FirstViewport[0][1]))
          self._current_camera = ((self._current_camera[0]-self.CameraBias[0]), (self._current_camera[1]-self.CameraBias[1]))
          self.ViewportCenter = (self._current_camera[0]-self.FirstViewport[0][0], self._current_camera[1]-self.FirstViewport[0][1])
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
    townhall_location = self._holding_resource_region_list[0][1]
    self._schedule_job(self._current_camera, self.TOWNHALL_TYPES[0], ['_execute_training_worker_from_townhall', self, []])
    self._schedule_job(self._current_camera, None, ['_record_townhall_best_locations', self, []])
    self._schedule_job(self._current_camera, self.TOWNHALL_TYPES[0], ['_execute_training_worker_from_townhall', self, []])
    self._schedule_job(self._current_camera, self.TOWNHALL_TYPES[0], ['_execute_training_worker_from_townhall', self, []])
    self._schedule_job(self._current_camera, None, ['_select_gathering_mineral_worker', self, [townhall_location]])
    self._schedule_job(self._current_camera, self.WORKER_TYPE, ['_walk_through_townhall_best_locations', self, []])
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
    townhall_location = self._holding_resource_region_list[0][1]
    self._schedule_job(self._current_camera, self.TOWNHALL_TYPES[0], ['_execute_training_worker_from_townhall', self, []])
    self._schedule_job(self._current_camera, None, ['_look_around_world', self, [None]])
    self._schedule_job(self._current_camera, None, ['_record_townhall_best_locations', self, []])
    self._schedule_job(self._current_camera, None, ['_execute_training_worker_from_townhall', self, []])
    self._schedule_job(self._current_camera, None, ['_execute_training_worker_from_townhall', self, []])
    self._schedule_job(self._current_camera, None, ['_select_gathering_mineral_worker', self, [townhall_location]])
    self._schedule_job(self._current_camera, self.WORKER_TYPE, ['_walk_through_townhall_best_locations', self, []])


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
    townhall_location = self._holding_resource_region_list[0][1]
    self._schedule_job(self._current_camera, self.TOWNHALL_TYPES[0], ['_execute_training_worker_from_townhall', self, []])
    self._schedule_job(self._current_camera, None, ['_record_townhall_best_locations', self, []])
    self._schedule_job(self._current_camera, self.TOWNHALL_TYPES[0], ['_execute_training_worker_from_townhall', self, []])
    self._schedule_job(self._current_camera, None, ['_select_gathering_mineral_worker', self, [townhall_location]])
    self._schedule_job(self._current_camera, self.WORKER_TYPE, ['_walk_through_townhall_best_locations', self, []])
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

