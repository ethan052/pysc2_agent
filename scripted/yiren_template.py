from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
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
                  ]
                  #, units.Neutral.PurifierMineralField, units.Neutral.PurifierMineralField750
                  #, units.Neutral.PurifierRichMineralField, units.Neutral.PurifierRichMineralField750
                  #, units.Neutral.BattleStationMineralField, units.Neutral.BattleStationMineralField750
                  
  VESPENE_TYPES = [ units.Neutral.VespeneGeyser
                  , units.Neutral.SpacePlatformGeyser
                  , units.Neutral.RichVespeneGeyser
                  ]
                  #, units.Neutral.PurifierVespeneGeyser
                  #, units.Neutral.ProtossVespeneGeyser
                  #, units.Neutral.ShakurasVespeneGeyser
  MINERAL_BIAS = (-3, -1)
  VESPENE_BIAS = (-1, -1)
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
    self.ScreenSize = (0, 0)
    self.MinimapSize = (0, 0)
    self.CameraOffset = None
    self.FirstViewport = None
    self.UnitGridSize = None
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


  def calculate_local_coordinate(self, world_coordinate):
    screen_size = self.ScreenSize
    estimative_referenced_world = (world_coordinate[0]-int(math.floor((screen_size[0]+1)/2)), world_coordinate[1]-int(math.floor((screen_size[1]+1)/2)))
    viewport_size = self.ViewportSize
    minimap_offset_x = int(math.floor(estimative_referenced_world[0]*viewport_size[0]/screen_size[0]))
    minimap_offset_y = int(math.floor(estimative_referenced_world[1]*viewport_size[1]/screen_size[1]))
    boundary = (self.MinimapSize[0]-viewport_size[0], self.MinimapSize[1]-viewport_size[1])
    if minimap_offset_x < 0:
      minimap_offset_x = 0
    elif minimap_offset_x > boundary[0]:
      minimap_offset_x = boundary[0]
    if minimap_offset_y < 0:
      minimap_offset_y = 0
    elif minimap_offset_y > boundary[1]:
      minimap_offset_y = boundary[1]
    viewport_center = (int(math.floor((viewport_size[0]+1)/2)), int(math.floor((viewport_size[1]+1)/2)))
    origin_minimap = ((viewport_center[0]-self.CameraOffset[0]), (viewport_center[1]-self.CameraOffset[1]))      
    camera_minimap = (origin_minimap[0]+minimap_offset_x, origin_minimap[1]+minimap_offset_y)
    calculated_referenced_world = self.calculate_world_absolute_coordinate( (camera_minimap, (0,0) ) )
    location_screen = (world_coordinate[0]-calculated_referenced_world[0], world_coordinate[1]-calculated_referenced_world[1])
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
  def create_circle_mask(cls, height, width, center, radius_square):
    y, x = numpy.ogrid[(-center[1]):(height-center[1]), (-center[0]):(width-center[0])]
    
    mask = (x**2 + y**2 <= radius_square)
    masked_array = numpy.full( (height, width), False)
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
      
  
  def create_townhall_margin_mask(self, screen_shape, townhall_location):
    townhall_grid_length = _UnitNumeric[self.TOWNHALL_TYPES[0]]['square_length']
    (CORE, BOUND, TIGHT, MEDIUM, LOOSE) = range(5)
    (LEFT, TOP, RIGHT, BOTTOM) = range(4)
    # 遊戲畫面的輔助格邊長，約等於 feature_scren 的 3.7 像素長
    grid_size = 3
    radius_bias = 2
    townhall_radius = int(math.floor(grid_size*townhall_grid_length/2.0))+radius_bias
    offset = [townhall_radius//2, townhall_radius+1]
    values = []
    for j in (CORE, BOUND):
      values.append([townhall_location[i%2]+(i//2*2-1)*offset[j] for i in range(4)])    
    for j in (TIGHT, MEDIUM, LOOSE):
      values.append([values[BOUND][i]+(i//2*2-1)*grid_size*j for i in range(4)])
    
    for i in (LEFT, TOP):
      for j in range(4, -1, -1):
        if values[j][i] < 0:
          values[j][i] = 0
        else:
          break
    for i in (RIGHT, BOTTOM):
      MAX_VALUE = screen_shape[3-i]-1
      for j in range(4, -1, -1):
        if values[j][i] > MAX_VALUE:
          values[j][i] = MAX_VALUE
        else:
          break
    margin_mask = numpy.full(screen_shape, False)
    margin_mask[values[LOOSE][TOP]:values[LOOSE][BOTTOM]+1, values[CORE][LEFT]:values[CORE][RIGHT]+1] = True
    margin_mask[values[CORE][TOP]:values[CORE][BOTTOM]+1, values[LOOSE][LEFT]:values[LOOSE][RIGHT]+1] = True
    margin_mask[values[TIGHT][TOP]:values[TIGHT][BOTTOM]+1, values[TIGHT][LEFT]:values[TIGHT][RIGHT]+1] = True
    margin_mask[values[MEDIUM][TOP]:values[MEDIUM][BOTTOM]+1, values[BOUND][LEFT]:values[BOUND][RIGHT]+1] = True
    margin_mask[values[BOUND][TOP]:values[BOUND][BOTTOM]+1, values[MEDIUM][LEFT]:values[MEDIUM][RIGHT]+1] = True
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
  def get_locations_screen(cls, mask, neighbor_distance_squre=2, tuple_form=False):
    grouped_points = cls.aggregate_points(mask, neighbor_distance_squre)
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
        
        if tuple_form:
          center = ((left+right)/2.0, (top+bottom)/2.0)
          return_locs.append( (center, (left,top), (right,bottom) ) )
        else:
          center = (int(round((left+right)/2)), int(round((top+bottom)/2)))
          return_locs.append( center )
    return return_locs


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
    #townhall_image = skimage_color.gray2rgb(numpy.zeros(unit_type.shape, dtype=numpy.uint8))
    #townhall_image[(unit_type==self.TOWNHALL_TYPES[0])] = (255, 255, 0)
    #filename = 'debug_townhall_%02d_%02d.png' % (self._current_camera[0], self._current_camera[1])
    #skimage_io.imsave(self.DEBUG_OUTPUT_PATH + '/%s' % filename, townhall_image)    
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


  def _get_resource_screen(self, obs):
    DEBUG_OUTPUT = False
    unit_type = obs.observation.feature_screen.unit_type    
    unit_density = obs.observation.feature_screen.unit_density
    cloned_density = numpy.array(unit_density)
    
    union_resource_mask = numpy.full( unit_type.shape, False)
    resource_type_list = self.MINERAL_TYPES + self.VESPENE_TYPES
    for resource_type in resource_type_list:
      unit_type_mask = (unit_type == resource_type)
      union_resource_mask = numpy.logical_or(union_resource_mask, unit_type_mask)
    cloned_density[numpy.logical_not(union_resource_mask)] = 0
    
    if DEBUG_OUTPUT:
      fig0, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, nrows=1, sharex=True, sharey=True, figsize=(6, 2))
      original_density_image = skimage_color.gray2rgb(numpy.zeros(unit_type.shape, dtype=numpy.uint8))
      original_density_image[(cloned_density==1)] = (255, 255, 255)
      original_density_image[(cloned_density==2)] = (170, 170, 170)
      original_density_image[(cloned_density==3)] = (85, 85, 85)
      ax1.imshow(original_density_image)

    for geyser_type in self.VESPENE_TYPES:
      unit_type_mask = (unit_type == geyser_type)
      y_arr, x_arr = unit_type_mask.nonzero()
      point_list = list(zip(y_arr, x_arr))
      for (y, x) in point_list:
        cloned_density[y, x] -= 1
          
    y_arr, x_arr = (cloned_density>1).nonzero()
    geyser_centers = self._get_vespene_screen(obs, False)
    point_list = list(zip(y_arr, x_arr)) 
    for barycenter in geyser_centers:
      for (y, x) in point_list:
        if self.calculate_distance_square(barycenter, (x,y) ) <= 40.0:            
          cloned_density[y, x] -= 1

    if DEBUG_OUTPUT:
      patch1_density_image = skimage_color.gray2rgb(numpy.zeros(unit_type.shape, dtype=numpy.uint8))
      patch1_density_image[(cloned_density==1)] = (255, 255, 255)
      patch1_density_image[(cloned_density==2)] = (170, 170, 170)
      patch1_density_image[(cloned_density==3)] = (85, 85, 85)
      ax3.imshow(patch1_density_image)          
      for center in geyser_centers:
        original_density_image[int(round(center[1])), int(round(center[0]))] = (255,0,0)
      for (y, x) in point_list:
        original_density_image[y, x] = (0,255,0)
        for center in geyser_centers:
          if self.calculate_distance_square(center, (x,y) ) <= 40.0:
            original_density_image[y, x] = (0,0,255)

      ax2.imshow(original_density_image)
      filename = 'debug_original_density_%02d_%02d.png' % (self._current_camera[0], self._current_camera[1])
      skimage_io.imsave(self.DEBUG_OUTPUT_PATH + '/%s' % filename, original_density_image)

      patch2_density_image = skimage_color.gray2rgb(numpy.zeros(unit_type.shape, dtype=numpy.uint8))
      patch2_density_image[(cloned_density==1)] = (255, 255, 255)
      patch2_density_image[(cloned_density==2)] = (170, 170, 170)
      patch2_density_image[(cloned_density==3)] = (85, 85, 85)
      ax4.imshow(patch2_density_image)

      filename = 'debug_density_%02d_%02d.png' % (self._current_camera[0], self._current_camera[1])
      plt.savefig(self.DEBUG_OUTPUT_PATH + '/%s' % filename)
      plt.close(fig0)

    union_mineral_mask = (cloned_density>0)
    mineral_field_list = []      

    if DEBUG_OUTPUT:
      fig1, axs = plt.subplots(ncols=5, nrows=2, sharex=True, sharey=True, figsize=(10, 4))
      for i in range(10):
        if not union_mineral_mask.any():
          break
        union_mineral_image = numpy.zeros(unit_type.shape, dtype=numpy.uint8)
        union_mineral_image[union_mineral_mask] = 255
        debug_image = skimage_color.gray2rgb(union_mineral_image)
        axs[i//5][i%5].imshow(debug_image)

        y, x = union_mineral_mask.nonzero()
        left = x.min()
        top = y.min()
        right = x.max()
        bottom = y.max()
        for maybe_x in range(right-3, left+3, -1):
          maybe_center = (maybe_x, bottom-3)
          circle_mask = self.create_mineral_circle_mask(union_mineral_mask.shape, maybe_center)
          contradiction = numpy.logical_and(numpy.logical_not(union_mineral_mask), circle_mask)
          if not contradiction.any():
            mineral_field_list.append((maybe_center[0], maybe_center[1]))
            cloned_density[circle_mask] -= 1
            union_mineral_mask = (cloned_density > 0)
            break
      
      filename = 'debug_mineral_%02d_%02d.png' % (self._current_camera[0], self._current_camera[1])
      plt.savefig(self.DEBUG_OUTPUT_PATH + '/%s' % filename)
      plt.close(fig1)
    else:
      for i in range(10):
        if not union_mineral_mask.any():
          break
        y, x = union_mineral_mask.nonzero()
        left, right = x.min(), x.max()
        top, bottom = y.min(), y.max()
        for maybe_x in range(right-3, left+3, -1):
          maybe_center = (maybe_x, bottom-3)
          circle_mask = self.create_mineral_circle_mask(union_mineral_mask.shape, maybe_center)
          contradiction = numpy.logical_and(numpy.logical_not(union_mineral_mask), circle_mask)
          if not contradiction.any():
            mineral_field_list.append((maybe_center[0], maybe_center[1]))
            cloned_density[circle_mask] -= 1
            union_mineral_mask = (cloned_density > 0)
            break
    
    cloned_density = numpy.array(unit_density)
    cloned_density[numpy.logical_not(union_resource_mask)] = 0

    if DEBUG_OUTPUT:
      original_density_image = skimage_color.gray2rgb(numpy.zeros(unit_type.shape, dtype=numpy.uint8))
      original_density_image[(cloned_density==1)] = (255, 255, 255)
      original_density_image[(cloned_density==2)] = (170, 170, 170)
      original_density_image[(cloned_density==3)] = (85, 85, 85)
    
    for center in mineral_field_list:
      circle_mask = self.create_mineral_circle_mask(cloned_density.shape, center)
      cloned_density[circle_mask] -= 1

    if DEBUG_OUTPUT:
      for center in mineral_field_list:
        original_density_image[center[1], center[0]] = (0, 0, 255)

    union_vespene_mask = (cloned_density == 1)
    
    vespene_geyser_list = self.get_locations_screen(union_vespene_mask, 2, False)
    if DEBUG_OUTPUT:
      for center in vespene_geyser_list:
        original_density_image[center[1], center[0]] = (0, 255, 0)
      filename = 'debug_center_%02d_%02d_.png' % (self._current_camera[0], self._current_camera[1])
      skimage_io.imsave(self.DEBUG_OUTPUT_PATH + '/%s' % filename, original_density_image)
    return (mineral_field_list, vespene_geyser_list)

      
  def _check_resource_overlapping_townhall(self, screen_shape, townhall_location, mineral_field_list, vespene_geyser_list):    
    townhall_margin_mask = self.create_townhall_margin_mask(screen_shape, townhall_location)
    mineral_field_list_mask = self.create_mineral_field_list_mask(screen_shape, mineral_field_list)
    vespene_geyser_list_mask = self.create_vespene_geyser_list_mask(screen_shape, vespene_geyser_list)
    resource_mask = numpy.logical_or(mineral_field_list_mask, vespene_geyser_list_mask)
    conflict_mask = numpy.logical_and(townhall_margin_mask, resource_mask)
    return conflict_mask.any()

    
  def _draw_debug_figure(self, obs, townhall_location,  mineral_field_list, vespene_geyser_list, append_filename='default'):
    unit_type = obs.observation.feature_screen.unit_type
    townhall_image = skimage_color.gray2rgb(numpy.zeros(unit_type.shape, dtype=numpy.uint8))
    townhall_margin_mask = self.create_townhall_margin_mask(unit_type.shape, townhall_location)
    townhall_image[townhall_margin_mask] = (85, 0, 0)
    townhall_mask = self.create_circle_mask(unit_type.shape[1], unit_type.shape[0], townhall_location, 9**2+3.5**2)
    townhall_image[townhall_mask] = (255, 255, 0)

    unit_density = obs.observation.feature_screen.unit_density
    cloned_density = numpy.array(unit_density)
    
    union_resource_mask = numpy.full( unit_type.shape, False)
    resource_type_list = self.MINERAL_TYPES + self.VESPENE_TYPES
    for resource_type in resource_type_list:
      unit_type_mask = (unit_type == resource_type)
      union_resource_mask = numpy.logical_or(union_resource_mask, unit_type_mask)
    cloned_density[numpy.logical_not(union_resource_mask)] = 0

    townhall_image[(cloned_density==1)] = (255, 255, 255)
    townhall_image[(cloned_density==2)] = (170, 170, 170)
    townhall_image[(cloned_density==3)] = (85, 85, 85)

    mineral_field_list_mask = self.create_mineral_field_list_mask(unit_type.shape, mineral_field_list)
    townhall_image[mineral_field_list_mask] = (0, 0, 170)
    for center in mineral_field_list:
      (mineral_x, mineral_y) = (center[0]+self.MINERAL_BIAS[0], center[1]+self.MINERAL_BIAS[1])
      townhall_image[center[1], center[0]] = (0, 0, 85)

    vespene_geyser_list_mask = self.create_vespene_geyser_list_mask(unit_type.shape, vespene_geyser_list)
    townhall_image[vespene_geyser_list_mask] = (0, 170, 0)
    
    for center in vespene_geyser_list:
      (vespene_x, vespene_y) = (center[0]+self.VESPENE_BIAS[0], center[1]+self.VESPENE_BIAS[1])
      townhall_image[center[1], center[0]] = (0, 85, 0)
    
    filename = 'debug_townhall_%02d_%02d_%s.png' % (self._current_camera[0], self._current_camera[1], append_filename)
    skimage_io.imsave(self.DEBUG_OUTPUT_PATH + '/%s' % filename, townhall_image)
    return townhall_image
  
        
  def _calculate_townhall_best_location(self, obs):
    DEBUG_OUTPUT = False
    unit_type = obs.observation.feature_screen.unit_type
    mineral_source_list, vespene_source_list = self._get_resource_screen(obs)
    count_mineral_source = len(mineral_source_list)    
    count_vespene_source = len(vespene_source_list)
    
    if count_vespene_source != 2:
      return None
    if count_mineral_source <= 0:
      return None
    (barycenter_x, barycenter_y) = mineral_source_list[0]
    (left_most, top_most) = (mineral_source_list[0][0]-4, mineral_source_list[0][1]-4)
    (right_most, bottom_most) = (mineral_source_list[0][0]+3, mineral_source_list[0][1]+3)
    for i in range(1, count_mineral_source):
      barycenter_x += mineral_source_list[i][0]
      barycenter_y += mineral_source_list[i][1]
      (left, top) = (mineral_source_list[i][0]-4, mineral_source_list[i][1]-4)
      (right, bottom) = (mineral_source_list[i][0]+3, mineral_source_list[i][1]+3)
      if left < left_most:
        left_most = left
      if top < top_most:
        top_most = top
      if right > right_most:
        right_most = right
      if bottom > bottom_most:
        bottom_most = bottom
    barycenter_x /= count_mineral_source
    barycenter_y /= count_mineral_source
    barycenter = (barycenter_x, barycenter_y)
    mineral_region = (right_most-left_most, bottom_most-top_most)
    townhall_grid_length = _UnitNumeric[self.TOWNHALL_TYPES[0]]['square_length']
    gas_plant_grid_length = _UnitNumeric[self.GAS_PLANT_TYPE]['square_length']
    townhall_radius = townhall_grid_length*2-1
    townhall_diameter = townhall_grid_length*4-1
    mineral_distance_from_townhall = townhall_radius+12
    #mineral_distance_square_from_townhall = mineral_distance_from_townhall**2
    screen_center = (self.ScreenSize[0]/2, self.ScreenSize[1]/2)
    vertical_middle = (top_most+bottom_most)/2.0
    horizontal_middle = (left_most+right_most)/2.0
    center = (horizontal_middle, vertical_middle)
    vespene_distance_from_townhall = townhall_grid_length*2+gas_plant_grid_length*2+12
    vespene_distance_square_from_townhall = vespene_distance_from_townhall**2
    (x1, y1) = vespene_source_list[0]
    (x2, y2) = vespene_source_list[1]
    let_a = (x2-x1)/2.0
    let_b = (y2-y1)/2.0
    let_sum_square = let_a**2 + let_b**2
    quad_square_t = (vespene_distance_square_from_townhall / let_sum_square) - 1
    positive_2t = math.sqrt(quad_square_t)
    negative_2t = -positive_2t
    let_c = ( let_a+x1+self.VESPENE_BIAS[0], let_b+y1+self.VESPENE_BIAS[1] )
    positive_offset = (let_b*positive_2t, let_a*positive_2t)
    negative_offset = (let_b*negative_2t, let_a*negative_2t)
    intersection_points = [ (int(round(let_c[0]-positive_offset[0])), int(round(let_c[1]+positive_offset[1])))
                          , (int(round(let_c[0]-negative_offset[0])), int(round(let_c[1]+negative_offset[1])))
                          ]
       
    if mineral_region[0] < townhall_diameter and mineral_region[1] < townhall_diameter:
      return None
    elif mineral_region[0] >= townhall_diameter and mineral_region[1] >= townhall_diameter:    # 礦區像 L 型
      corners = [(left_most, top_most), (right_most, top_most), (left_most, bottom_most), (right_most, bottom_most)]
      distance_from_corners = [ self.calculate_distance_square(barycenter, corner) for corner in corners ]
      max_distance = 0
      chosen_index = -1
      for i in range(len(corners)):
        if distance_from_corners[i] > max_distance:
          max_distance = distance_from_corners[i]
          chosen_index = i
          
      corner_point = corners[chosen_index]
      distance_from_intersection = [self.calculate_distance_square(corner_point, p) for p in intersection_points]
      if distance_from_intersection[0] < distance_from_intersection[1]:
        candidate_point = intersection_points[0]
      else:
        candidate_point = intersection_points[1]
      if DEBUG_OUTPUT:
        self._draw_debug_figure(obs, candidate_point,  mineral_source_list, vespene_source_list, 'calculated_%02d_%02d'% (candidate_point[1], candidate_point[0]))
      if True == self._check_resource_overlapping_townhall(unit_type.shape, candidate_point, mineral_source_list, vespene_source_list):
        nearest_point = candidate_point
        nearest_distance = None        
        direction_offset = (1-chosen_index%2*2, 1-chosen_index//2*2)
        vertical_range = range(candidate_point[1]-direction_offset[1]*6, candidate_point[1]+direction_offset[1], direction_offset[1])
        horizontal_range = range(candidate_point[0]-direction_offset[0]*6, candidate_point[0]+direction_offset[0], direction_offset[0])
        if DEBUG_OUTPUT:
          fig, subfig = plt.subplots(nrows=len(list(vertical_range)), ncols=len(list(horizontal_range)), sharex=True, sharey=True)
          plt.subplots_adjust(wspace=0.6, hspace=0.6)
          flg_row = 0
          for y in vertical_range:
            flg_column = 0
            for x in horizontal_range:
              chosen_point = (x, y)
              if not self._check_resource_overlapping_townhall(unit_type.shape, chosen_point, mineral_source_list, vespene_source_list):
                chosen_img = self._draw_debug_figure(obs, chosen_point, mineral_source_list, vespene_source_list, 'choose_%02d_%02d_valid' % (y, x))
                subfig[flg_row][flg_column].set_title('Yes', fontdict={'fontsize': 8, 'fontweight': 'medium'})
                distance = self.calculate_distance_square(barycenter, chosen_point)
                if nearest_distance is None or distance < nearest_distance:
                  nearest_point = chosen_point
                  nearest_distance = distance
              else:
                chosen_img = self._draw_debug_figure(obs, chosen_point, mineral_source_list, vespene_source_list, 'choose_%02d_%02d_invalid' % (y, x))
                subfig[flg_row][flg_column].set_title('No', fontdict={'fontsize': 8, 'fontweight': 'medium'})
              subfig[flg_row][flg_column].imshow(chosen_img)
              flg_column += 1
            flg_row += 1
          filename = 'debug_choose_%02d_%02d.png' % (self._current_camera[0], self._current_camera[1])
          plt.savefig(self.DEBUG_OUTPUT_PATH + '/%s' % filename)
          plt.close(fig)
          self._draw_debug_figure(obs, nearest_point,  mineral_source_list, vespene_source_list, 'choose_%02d_%02d_final' % (nearest_point[1], nearest_point[0]))
        else:
          for y in vertical_range:
            for x in horizontal_range:
              chosen_point = (x, y)
              if not self._check_resource_overlapping_townhall(unit_type.shape, chosen_point, mineral_source_list, vespene_source_list):
                distance = self.calculate_distance_square(barycenter, chosen_point)
                if nearest_distance is None or distance < nearest_distance:
                  nearest_point = chosen_point
                  nearest_distance = distance        
        candidate_point = nearest_point
    else:
      direction_offset = None
      if mineral_region[0] < townhall_diameter:    # 礦區像直的(狹長)
        center_left = (left_most, vertical_middle)
        center_right = (right_most, vertical_middle)
        distance_from_left = self.calculate_distance_square(screen_center, center_left)
        distance_from_right = self.calculate_distance_square(screen_center, center_right)
        if distance_from_left < distance_from_right:
          center = center_left
          direction_offset = (-1, 0)
        else:
          center = center_right
          direction_offset = (1, 0)
      elif mineral_region[1] < townhall_diameter:    # 礦區像橫的(扁平)    
        center_top = (horizontal_middle, top_most)
        center_bottom = (horizontal_middle, bottom_most)
        distance_from_top = self.calculate_distance_square(screen_center, center_top)
        distance_from_bottom = self.calculate_distance_square(screen_center, center_bottom)
        if distance_from_top < distance_from_bottom:
          center = center_top
          direction_offset = (0, -1)
        else:
          center = center_bottom
          direction_offset = (0, 1)      
      farthest_point = (center[0]+direction_offset[0]*mineral_distance_from_townhall, center[1]+direction_offset[1]*mineral_distance_from_townhall)
      distance_from_intersection = [self.calculate_distance_square(farthest_point, p) for p in intersection_points]
      if distance_from_intersection[0] < distance_from_intersection[1]:
        candidate_point = intersection_points[0]
      else:
        candidate_point = intersection_points[1]
      if DEBUG_OUTPUT:
        self._draw_debug_figure(obs, candidate_point,  mineral_source_list, vespene_source_list, 'calculated_%02d_%02d'% (candidate_point[1], candidate_point[0]))
      while True == self._check_resource_overlapping_townhall(unit_type.shape, candidate_point, mineral_source_list, vespene_source_list):
        candidate_point = (candidate_point[0]+direction_offset[0], candidate_point[1]+direction_offset[1])
      if DEBUG_OUTPUT:
        self._draw_debug_figure(obs, candidate_point,  mineral_source_list, vespene_source_list, 'choose_%02d_%02d_final' % (candidate_point[1], candidate_point[0]))
    return candidate_point


  def _execute_moving_camera(self, obs, camera_minimap):
    if self._current_camera != camera_minimap:
      self._current_camera = camera_minimap
      return FUNCTIONS.move_camera(camera_minimap)
    return FUNCTIONS.no_op()


  def _record_townhall_best_location(self, obs, next_camera):
    townhall_best_location = self._calculate_townhall_best_location(obs)
    if townhall_best_location is not None:
      existed_world_coordinates = set()
      for local_coordinate in self._calculated_resource_region_list:
        world_coordinate = self.calculate_world_absolute_coordinate(local_coordinate)
        existed_world_coordinates.add(world_coordinate)
      world_coordinate = self.calculate_world_absolute_coordinate((self._current_camera, townhall_best_location))
      if world_coordinate not in existed_world_coordinates:
        local_coordinate = self.calculate_local_coordinate(world_coordinate)
        self._calculated_resource_region_list.append( local_coordinate )
    return self._execute_moving_camera(obs, next_camera)
    
    
  def _record_townhall_best_locations(self, obs):
    last_camera_minimap = self._current_camera
    for i in range(len(self._speculated_resource_region_list)-1, 0, -1):
      camera_minimap = self._speculated_resource_region_list[i]      
      self._schedule_job(camera_minimap, None, ['_record_townhall_best_location', self, [last_camera_minimap]], True)
      for j in range(2):
        self._schedule_job(camera_minimap, None, [FUNCTIONS.no_op.id, []], True)
      last_camera_minimap = camera_minimap
    return self._execute_moving_camera(obs, last_camera_minimap)


  def _test_build_townhall(self, obs, camera_minimap, townhall_best_location):
    townhall_mineral_cost = _UnitNumeric[self.TOWNHALL_TYPES[0]]['mineral_cost']
    if obs.observation.player.minerals >= townhall_mineral_cost:
      action_id = _UnitDependency[self.TOWNHALL_TYPES[0]][0]['perform']
      self._schedule_job(camera_minimap, self.WORKER_TYPE, [action_id, ['queued', townhall_best_location]], True)
    else:
      self._schedule_job(camera_minimap, self.WORKER_TYPE, ['_test_build_townhall', self, [camera_minimap, townhall_best_location]], True)
    return FUNCTIONS.no_op()
    
    
  def _walk_through_townhall_best_locations(self, obs):
    if len(self._calculated_resource_region_list) <= 1:
      return None
    last_camera_minimap = self._current_camera
    for i in range(len(self._calculated_resource_region_list)-1, 0, -1):
      (camera_minimap, location_screen) = self._calculated_resource_region_list[i]
      self._schedule_job(camera_minimap, self._expected_selected, ['_execute_moving_camera', self, [last_camera_minimap]], True)
      #self._schedule_job(camera_minimap, self._expected_selected, [FUNCTIONS.Move_screen.id, ['queued', location_screen]], True)
      #action_id = _UnitDependency[self.TOWNHALL_TYPES[0]][0]['perform']
      #self._schedule_job(camera_minimap, self._expected_selected, [action_id, ['queued', location_screen]], True)
      self._schedule_job(camera_minimap, self.WORKER_TYPE, ['_test_build_townhall', self, [camera_minimap, location_screen]], True)
      last_camera_minimap = camera_minimap
    self._schedule_job(self._current_camera, self._expected_selected, ['_execute_moving_camera', self, [last_camera_minimap]], True)
    return FUNCTIONS.no_op()

    
  def _speculate_resource_regions(self, obs):
    player_relative = obs.observation.feature_minimap.player_relative
    #neighbor_distance_x = self.MinimapSize[0]/4.0/self.ViewportSize[0]
    #neighbor_distance_y = self.MinimapSize[1]/4.0/self.ViewportSize[1]
    #neighbor_distance_squre = neighbor_distance_x**2 + neighbor_distance_y**2
    neighbor_distance_squre = 8
    grouped_points = self.aggregate_points(player_relative == features.PlayerRelative.NEUTRAL, neighbor_distance_squre)
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
        center_location = [int(round((x_arr.max()+x_arr.min())/2.0)), int(round((y_arr.max()+y_arr.min())/2.0))]
        #center_location = [int(round(x_arr.mean())), int(round(y_arr.mean()))]
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
      
    square_length = _UnitNumeric[self.TOWNHALL_TYPES[0]]['square_length']
    double_square_length = square_length*2
    
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
      if distance<=(double_square_length+2)**2 or distance>(double_square_length+10)**2:
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
    townhall_location_list = self._get_my_townhall_screen(obs, True)
    self._world_coordinate[owner] = {}
    resource_region_list = self._speculate_resource_regions(obs)
    #self._speculated_resource_region_list = self._speculate_resource_regions(obs)
    if 1 == len(townhall_location_list):
      resource_region_list = self._speculate_resource_regions(obs)
      holding_index = None
      for i in range(len(resource_region_list)):
        resource_region_camera = resource_region_list[i]
        if self.FirstViewport[0][0]<resource_region_camera[0] and resource_region_camera[0]<self.FirstViewport[1][0] and self.FirstViewport[0][1]<resource_region_camera[1] and resource_region_camera[1]<self.FirstViewport[1][1]:
          holding_index = i
          break
      if holding_index is not None:
        referenced_coordinate = resource_region_list[holding_index]
        self._speculated_resource_region_list.append(referenced_coordinate)
        resource_region_list.pop(holding_index)
        distace_square_values = []
        for i in range(len(resource_region_list)):
          distace_square_value = self.calculate_distance_square(referenced_coordinate, resource_region_list[i])
          distace_square_values.append((distace_square_value, i))
        for (distace_square_value, i) in sorted(distace_square_values):
          self._speculated_resource_region_list.append(resource_region_list[i])
          
      unit_type_id = self.TOWNHALL_TYPES[0]
      first_townhall = townhall_location_list[0]
      townhall_location = first_townhall[0]
      townhall_size = (first_townhall[2][0]-first_townhall[1][0], first_townhall[2][1]-first_townhall[1][1])
      townhall_grid_length = _UnitNumeric[unit_type_id]['square_length']
      self.UnitGridSize = (townhall_size[0]/townhall_grid_length, townhall_size[1]/townhall_grid_length)
      #townhall_best_location = self._calculate_townhall_best_location(obs)
      world_absolute_coordinate = self.calculate_world_absolute_coordinate((self._current_camera, townhall_location))
      self._calculated_resource_region_list.append( (self._current_camera, townhall_location) )
      self._occupied_resource_regions[self._current_camera] = {'owner':owner}
      self._holding_resource_region_list.append( (self._current_camera, townhall_location) )
      self._world_coordinate[owner][unit_type_id] = [world_absolute_coordinate]
      self._structures[world_absolute_coordinate] = {'owner':owner, 'unit_type':unit_type_id}
      self._expected_selected = unit_type_id
      self._schedule_job(self._current_camera, unit_type_id, [FUNCTIONS.select_control_group.id, ['set', 0]])
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
  
