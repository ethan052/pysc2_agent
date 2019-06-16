from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from collections import deque

from pysc2.agents import base_agent
from pysc2.lib import actions, features, units
from pysc2.env.sc2_env import Race

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
    self._current_selected_point = None
    # scheduled

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

    # 如果有選到單位
    #     檢查有無鏡頭的排程
    # 該單位要做的事已經做完，檢查敵情
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
          #self._schedule_job(self._current_camera, None, ['initialize', self, []])
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
  
