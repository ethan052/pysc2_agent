from pysc2.lib import units, actions
from pysc2.env.sc2_env import Race
FUNCTIONS = actions.FUNCTIONS

DATA = {}
## 神族兵種
### 探測機
DATA[units.Protoss.Probe] = [ {'actor':[set([units.Protoss.Nexus])], 'perform': FUNCTIONS.Train_Probe_quick.id, 'require':None} ]
### 狂戰士
DATA[units.Protoss.Zealot] = [ {'actor':[set([units.Protoss.Gateway])], 'perform': FUNCTIONS.Train_Zealot_quick.id, 'require':None}
                             , {'actor':[set([units.Protoss.WarpGate])], 'perform': FUNCTIONS.TrainWarp_Zealot_screen.id, 'require':None}
                             ]
### 哨衛
DATA[units.Protoss.Sentry] = [ {'actor':[set([units.Protoss.Gateway])], 'perform': FUNCTIONS.Train_Sentry_quick.id, 'require':set([units.Protoss.CyberneticsCore])}
                             , {'actor':[set([units.Protoss.WarpGate])], 'perform': FUNCTIONS.TrainWarp_Sentry_screen.id, 'require':set([units.Protoss.CyberneticsCore])}
                             ]
### 追獵者
DATA[units.Protoss.Stalker] = [ {'actor':[set([units.Protoss.Gateway])], 'perform': FUNCTIONS.Train_Stalker_quick.id, 'require':set([units.Protoss.CyberneticsCore])}
                              , {'actor':[set([units.Protoss.WarpGate])], 'perform': FUNCTIONS.TrainWarp_Stalker_screen.id, 'require':set([units.Protoss.CyberneticsCore])}
                              ]
### 教士
DATA[units.Protoss.Adept] = [ {'actor':[set([units.Protoss.Gateway])], 'perform': FUNCTIONS.Train_Adept_quick.id, 'require':set([units.Protoss.CyberneticsCore])}
                            , {'actor':[set([units.Protoss.WarpGate])], 'perform': FUNCTIONS.TrainWarp_Adept_screen.id, 'require':set([units.Protoss.CyberneticsCore])}
                            ]
### 高階聖堂武士
DATA[units.Protoss.HighTemplar] = [ {'actor':[set([units.Protoss.Gateway])], 'perform': FUNCTIONS.Train_HighTemplar_quick.id, 'require':set([units.Protoss.TemplarArchive])}
                                  , {'actor':[set([units.Protoss.WarpGate])], 'perform': FUNCTIONS.TrainWarp_HighTemplar_screen.id, 'require':set([units.Protoss.TemplarArchive])}
                                  ]
### 暗影聖堂武士
DATA[units.Protoss.DarkTemplar] = [ {'actor':[set([units.Protoss.Gateway])], 'perform': FUNCTIONS.Train_DarkTemplar_quick.id, 'require':set([units.Protoss.DarkShrine])}
                                  , {'actor':[set([units.Protoss.WarpGate])], 'perform': FUNCTIONS.TrainWarp_DarkTemplar_screen.id, 'require':set([units.Protoss.DarkShrine])}
                                  ]
### 執政官 (破壞能)
DATA[units.Protoss.Archon] = [ {'actor':[set([units.Protoss.HighTemplar, units.Protoss.DarkTemplar]), set([units.Protoss.HighTemplar, units.Protoss.DarkTemplar])], 'perform': FUNCTIONS.Morph_Archon_quick.id, 'require':None} ]
### 觀察者
DATA[units.Protoss.Observer] = [ {'actor':[set([units.Protoss.RoboticsFacility])], 'perform': FUNCTIONS.Train_Observer_quick.id, 'require':None} ]
### 傳輸稜鏡
DATA[units.Protoss.WarpPrism] = [ {'actor':[set([units.Protoss.RoboticsFacility])], 'perform': FUNCTIONS.Train_WarpPrism_quick.id, 'require':None} ]
### 不朽者
DATA[units.Protoss.Immortal] = [ {'actor':[set([units.Protoss.RoboticsFacility])], 'perform': FUNCTIONS.Train_Immortal_quick.id, 'require':None} ]
### 巨像
DATA[units.Protoss.Colossus] = [ {'actor':[set([units.Protoss.RoboticsFacility])], 'perform': FUNCTIONS.Train_Colossus_quick.id, 'require':set([units.Protoss.RoboticsBay])} ]
### 裂光球
DATA[units.Protoss.Disruptor] = [ {'actor':[set([units.Protoss.RoboticsFacility])], 'perform': FUNCTIONS.Train_Disruptor_quick.id, 'require':set([units.Protoss.RoboticsBay])} ]
### 鳳凰戰機
DATA[units.Protoss.Phoenix] = [ {'actor':[set([units.Protoss.Stargate])], 'perform': FUNCTIONS.Train_Phoenix_quick.id, 'require':None} ]
### 先知艦
DATA[units.Protoss.Oracle] = [ {'actor':[set([units.Protoss.Stargate])], 'perform': FUNCTIONS.Train_Oracle_quick.id, 'require':None} ]
### 虛空艦
DATA[units.Protoss.VoidRay] = [ {'actor':[set([units.Protoss.Stargate])], 'perform': FUNCTIONS.Train_VoidRay_quick.id, 'require':None} ]
### 暴風艦
DATA[units.Protoss.Tempest] = [ {'actor':[set([units.Protoss.Stargate])], 'perform': FUNCTIONS.Train_Tempest_quick.id, 'require':set([units.Protoss.FleetBeacon])} ]
### 航空母艦
DATA[units.Protoss.Carrier] = [ {'actor':[set([units.Protoss.Stargate])], 'perform': FUNCTIONS.Train_Carrier_quick.id, 'require':set([units.Protoss.FleetBeacon])} ]
### 聖母艦
DATA[units.Protoss.Mothership] = [ {'actor':[set([units.Protoss.Nexus])], 'perform': FUNCTIONS.Train_Mothership_quick.id, 'require':set([units.Protoss.FleetBeacon])} ]


## 神族建築
### 星核
DATA[units.Protoss.Nexus] = [ {'actor':[set([units.Protoss.Probe])], 'perform':  FUNCTIONS.Build_Nexus_screen.id, 'require':None} ]
### 瓦斯處理廠
DATA[units.Protoss.Assimilator] = [ {'actor':[set([units.Protoss.Probe])], 'perform': FUNCTIONS.Build_Assimilator_screen.id, 'require':None} ]
### 水晶塔
DATA[units.Protoss.Pylon] = [ {'actor':[set([units.Protoss.Probe])], 'perform': FUNCTIONS.Build_Pylon_screen.id, 'require':None} ]
### 傳送門
DATA[units.Protoss.Gateway] = [ {'actor':[set([units.Protoss.Probe])], 'perform': FUNCTIONS.Build_Gateway_screen.id, 'require':set([units.Protoss.Nexus])} ]
### 冶煉廠
DATA[units.Protoss.Forge] = [ {'actor':[set([units.Protoss.Probe])], 'perform': FUNCTIONS.Build_Forge_screen.id, 'require':set([units.Protoss.Nexus])} ]
### 光子加農砲
DATA[units.Protoss.PhotonCannon] = [ {'actor':[set([units.Protoss.Probe])], 'perform': FUNCTIONS.Build_PhotonCannon_screen.id, 'require':set([units.Protoss.Forge])} ]
### 機械控制核心
DATA[units.Protoss.CyberneticsCore] = [ {'actor':[set([units.Protoss.Probe])], 'perform': FUNCTIONS.Build_CyberneticsCore_screen.id, 'require':set([units.Protoss.Gateway, units.Protoss.WarpGate])} ]
### 護盾充能器
DATA[units.Protoss.ShieldBattery] = [ {'actor':[set([units.Protoss.Probe])], 'perform': FUNCTIONS.Build_ShieldBattery_screen.id, 'require':set([units.Protoss.CyberneticsCore])} ]
### 機械製造廠
DATA[units.Protoss.RoboticsFacility] = [ {'actor':[set([units.Protoss.Probe])], 'perform': FUNCTIONS.Build_RoboticsFacility_screen.id, 'require':set([units.Protoss.CyberneticsCore])} ]
### 機械研究所
DATA[units.Protoss.RoboticsBay] = [ {'actor':[set([units.Protoss.Probe])], 'perform': FUNCTIONS.Build_RoboticsBay_screen.id, 'require':set([units.Protoss.RoboticsFacility])} ]
### 星際之門
DATA[units.Protoss.Stargate] = [ {'actor':[set([units.Protoss.Probe])], 'perform': FUNCTIONS.Build_Stargate_screen.id, 'require':set([units.Protoss.CyberneticsCore])} ]
### 艦隊導航台
DATA[units.Protoss.FleetBeacon] = [ {'actor':[set([units.Protoss.Probe])], 'perform': FUNCTIONS.Build_FleetBeacon_screen.id, 'require':set([units.Protoss.Stargate])} ]
### 暮光議會
DATA[units.Protoss.TwilightCouncil] = [ {'actor':[set([units.Protoss.Probe])], 'perform': FUNCTIONS.Build_TwilightCouncil_screen.id, 'require':set([units.Protoss.CyberneticsCore])} ]
### 聖堂文庫
DATA[units.Protoss.TemplarArchive] = [ {'actor':[set([units.Protoss.Probe])], 'perform': FUNCTIONS.Build_TemplarArchive_screen.id, 'require':set([units.Protoss.TwilightCouncil])} ]
### 黑暗神殿
DATA[units.Protoss.DarkShrine] = [ {'actor':[set([units.Protoss.Probe])], 'perform': FUNCTIONS.Build_DarkShrine_screen.id, 'require':set([units.Protoss.TwilightCouncil])} ]



## 人類兵種
### 太空工程車
DATA[units.Terran.SCV] = [ {'actor':[set([units.Terran.CommandCenter, units.Terran.OrbitalCommand, units.Terran.PlanetaryFortress])], 'perform': FUNCTIONS.Train_SCV_quick.id, 'require':None} ]
### 陸戰隊
DATA[units.Terran.Marine] = [ {'actor':[set([units.Terran.Barracks])], 'perform': FUNCTIONS.Train_Marine_quick.id, 'require':None} ]
### 死神
DATA[units.Terran.Reaper] = [ {'actor':[set([units.Terran.Barracks])], 'perform': FUNCTIONS.Train_Reaper_quick.id, 'require':None} ]
### 掠奪者
DATA[units.Terran.Marauder] = [ {'actor':[set([units.Terran.Barracks])], 'perform': FUNCTIONS.Train_Marauder_quick.id, 'require':None, 'accompany':set([units.Terran.BarracksTechLab])} ]
### 幽靈特務
DATA[units.Terran.Ghost] = [ {'actor':[set([units.Terran.Barracks])], 'perform': FUNCTIONS.Train_Ghost_quick.id, 'require':set([units.Terran.GhostAcademy]), 'accompany':set([units.Terran.BarracksTechLab])} ]
### 惡狼
DATA[units.Terran.Hellion] = [ {'actor':[set([units.Terran.Factory])], 'perform': FUNCTIONS.Train_Hellion_quick.id, 'require':None} ]
### 戰狼
DATA[units.Terran.Hellbat] = [ {'actor':[set([units.Terran.Factory])], 'perform': FUNCTIONS.Train_Hellbat_quick.id, 'require':set([units.Terran.Armory])} ]
### 寡婦詭雷
DATA[units.Terran.WidowMine] = [ {'actor':[set([units.Terran.Factory])], 'perform': FUNCTIONS.Train_WidowMine_quick.id, 'require':None} ]
### 颶風飛彈車
DATA[units.Terran.Cyclone] = [ {'actor':[set([units.Terran.Factory])], 'perform': FUNCTIONS.Train_Cyclone_quick.id, 'require':None, 'accompany':set([units.Terran.FactoryTechLab])} ]
### 攻城坦克
DATA[units.Terran.SiegeTank] = [ {'actor':[set([units.Terran.Factory])], 'perform': FUNCTIONS.Train_SiegeTank_quick.id, 'require':None, 'accompany':set([units.Terran.FactoryTechLab])} ]
### 雷神
DATA[units.Terran.Thor] = [ {'actor':[set([units.Terran.Factory])], 'perform': FUNCTIONS.Train_SiegeTank_quick.id, 'require':set([units.Terran.Armory]), 'accompany':set([units.Terran.FactoryTechLab])} ]
### 維京戰機
DATA[units.Terran.VikingFighter] = [ {'actor':[set([units.Terran.Starport])], 'perform': FUNCTIONS.Train_VikingFighter_quick.id, 'require':None} ]
### 醫療艇
DATA[units.Terran.Medivac] = [ {'actor':[set([units.Terran.Starport])], 'perform': FUNCTIONS.Train_Medivac_quick.id, 'require':None} ]
### 解放者
DATA[units.Terran.Liberator] = [ {'actor':[set([units.Terran.Starport])], 'perform': FUNCTIONS.Train_Liberator_quick.id, 'require':None} ]
### 渡鴉
DATA[units.Terran.Raven] = [ {'actor':[set([units.Terran.Starport])], 'perform': FUNCTIONS.Train_Raven_quick.id, 'require':set([units.Terran.StarportTechLab])} ]
### 女妖轟炸機
DATA[units.Terran.Banshee] = [ {'actor':[set([units.Terran.Starport])], 'perform': FUNCTIONS.Train_Raven_quick.id, 'require':set([units.Terran.StarportTechLab])} ]
### 戰巡艦
DATA[units.Terran.Battlecruiser] = [ {'actor':[set([units.Terran.Starport])], 'perform': FUNCTIONS.Train_Raven_quick.id, 'require':set([units.Terran.FusionCore]), 'accompany':set([units.Terran.StarportTechLab])} ]


## 人類建築
### 指揮中心
DATA[units.Terran.CommandCenter] = [ {'actor':[set([units.Terran.SCV])], 'perform': FUNCTIONS.Build_CommandCenter_screen.id, 'require':None} ]
### 星軌指揮總部
DATA[units.Terran.OrbitalCommand] = [ {'actor':[set([units.Terran.CommandCenter])], 'perform': FUNCTIONS.Morph_OrbitalCommand_quick.id, 'require':set([units.Terran.Barracks])} ]
### 行星要塞
DATA[units.Terran.PlanetaryFortress] = [ {'actor':[set([units.Terran.CommandCenter])], 'perform': FUNCTIONS.Morph_PlanetaryFortress_quick.id, 'require':set([units.Terran.EngineeringBay])} ]
### 瓦斯精煉廠
DATA[units.Terran.Refinery] = [ {'actor':[set([units.Terran.SCV])], 'perform': FUNCTIONS.Build_Refinery_screen.id, 'require':None} ]
### 補給站 !!! 經蟲族神經寄生證明, 不需要主堡就能蓋
DATA[units.Terran.SupplyDepot] = [ {'actor':[set([units.Terran.SCV])], 'perform': FUNCTIONS.Build_SupplyDepot_screen.id, 'require':None} ]
### 科技實驗室
DATA[units.Terran.BarracksTechLab] = [ {'actor':[set([units.Terran.Barracks])], 'perform': FUNCTIONS.Build_TechLab_quick.id, 'require':None}
                                     , {'actor':[set([units.Terran.BarracksFlying])], 'perform': FUNCTIONS.Build_TechLab_screen.id, 'require':None}
                                     ]
DATA[units.Terran.FactoryTechLab] = [ {'actor':[set([units.Terran.Factory])], 'perform': FUNCTIONS.Build_TechLab_quick.id, 'require':None}
                                    , {'actor':[set([units.Terran.FactoryFlying])], 'perform': FUNCTIONS.Build_TechLab_screen.id, 'require':None}
                                    ]
DATA[units.Terran.StarportTechLab] = [ {'actor':[set([units.Terran.Starport])], 'perform': FUNCTIONS.Build_TechLab_quick.id, 'require':None}
                                     , {'actor':[set([units.Terran.StarportFlying])], 'perform': FUNCTIONS.Build_TechLab_screen.id, 'require':None}
                                     ]
### 反應爐
DATA[units.Terran.BarracksReactor] = [ {'actor':[set([units.Terran.Barracks])], 'perform': FUNCTIONS.Build_Reactor_quick.id, 'require':None}
                                     , {'actor':[set([units.Terran.BarracksFlying])], 'perform': FUNCTIONS.Build_Reactor_screen.id, 'require':None}
                                     ] 
DATA[units.Terran.FactoryReactor] = [ {'actor':[set([units.Terran.Factory])], 'perform': FUNCTIONS.Build_Reactor_quick.id, 'require':None}
                                    , {'actor':[set([units.Terran.FactoryFlying])], 'perform': FUNCTIONS.Build_Reactor_screen.id, 'require':None}
                                    ]
DATA[units.Terran.StarportReactor] = [ {'actor':[set([units.Terran.Starport])], 'perform': FUNCTIONS.Build_Reactor_quick.id, 'require':None}
                                     , {'actor':[set([units.Terran.StarportFlying])], 'perform': FUNCTIONS.Build_Reactor_screen.id, 'require':None}
                                     ]
### 兵營
DATA[units.Terran.Barracks] = [ {'actor':[set([units.Terran.SCV])], 'perform': FUNCTIONS.Build_Barracks_screen.id, 'require':set([units.Terran.SupplyDepot])} ]
### 碉堡
DATA[units.Terran.Bunker] = [ {'actor':[set([units.Terran.SCV])], 'perform': FUNCTIONS.Build_Bunker_screen.id, 'require':set([units.Terran.Barracks])} ]
### 電機工程所
DATA[units.Terran.EngineeringBay] = [ {'actor':[set([units.Terran.SCV])], 'perform': FUNCTIONS.Build_EngineeringBay_screen.id, 'require':set([units.Terran.CommandCenter, units.Terran.OrbitalCommand, units.Terran.PlanetaryFortress])} ]
### 感應塔
DATA[units.Terran.SensorTower] = [ {'actor':[set([units.Terran.SCV])], 'perform': FUNCTIONS.Build_SensorTower_screen.id, 'require':set([units.Terran.EngineeringBay])} ]
### 飛彈砲台
DATA[units.Terran.MissileTurret] = [ {'actor':[set([units.Terran.SCV])], 'perform': FUNCTIONS.Build_MissileTurret_screen.id, 'require':set([units.Terran.EngineeringBay])} ]
### 軍工廠
DATA[units.Terran.Factory] = [ {'actor':[set([units.Terran.SCV])], 'perform': FUNCTIONS.Build_Factory_screen.id, 'require':set([units.Terran.Barracks])} ]
### 兵工廠
DATA[units.Terran.Armory] = [ {'actor':[set([units.Terran.SCV])], 'perform': FUNCTIONS.Build_Armory_screen.id, 'require':set([units.Terran.Factory])} ]
### 星際港
DATA[units.Terran.Starport] = [ {'actor':[set([units.Terran.SCV])], 'perform': FUNCTIONS.Build_Starport_screen.id, 'require':set([units.Terran.Factory])} ]
### 核融合核心
DATA[units.Terran.FusionCore] = [ {'actor':[set([units.Terran.SCV])], 'perform': FUNCTIONS.Build_FusionCore_screen.id, 'require':set([units.Terran.Starport])} ]
### 幽靈特務學院
DATA[units.Terran.GhostAcademy] = [ {'actor':[set([units.Terran.SCV])], 'perform': FUNCTIONS.Build_GhostAcademy_screen.id, 'require':set([units.Terran.Barracks])} ]



## 蟲族兵種
### 幼蟲
DATA[units.Zerg.Larva] = [ {'actor':[set([units.Zerg.Hatchery, units.Zerg.Lair, units.Zerg.Hive])], 'perform': FUNCTIONS.select_larva.id, 'require':None} ]
### 工蟲
DATA[units.Zerg.Drone] = [ {'actor':[set([units.Zerg.Larva])], 'perform': FUNCTIONS.Train_Drone_quick.id, 'require':None} ]
### 王蟲
DATA[units.Zerg.Overlord] = [ {'actor':[set([units.Zerg.Larva])], 'perform': FUNCTIONS.Train_Overlord_quick.id, 'require':None} ]
### 王蟲 (可運載蟲)
DATA[units.Zerg.OverlordTransport] = [ {'actor':[set([units.Zerg.Overlord])], 'perform': FUNCTIONS.Morph_OverlordTransport_quick.id, 'require':set([units.Zerg.Lair, units.Zerg.Hive])} ]
### 監察王蟲
DATA[units.Zerg.Overseer] = [ {'actor':[set([units.Zerg.Overlord, units.Zerg.OverlordTransport])], 'perform': FUNCTIONS.Morph_Overseer_quick.id, 'require':set([units.Zerg.Lair, units.Zerg.Hive])} ]
### 后蟲
DATA[units.Zerg.Queen] = [ {'actor':[set([units.Zerg.Hatchery, units.Zerg.Lair, units.Zerg.Hive])], 'perform': FUNCTIONS.Train_Queen_quick.id, 'require':set([units.Zerg.SpawningPool])} ]
### 異化蟲(狗)
DATA[units.Zerg.Zergling] = [ {'actor':[set([units.Zerg.Larva])], 'perform': FUNCTIONS.Train_Zergling_quick.id, 'require':set([units.Zerg.SpawningPool])} ]
### 蟑螂
DATA[units.Zerg.Roach] = [ {'actor':[set([units.Zerg.Larva])], 'perform': FUNCTIONS.Train_Roach_quick.id, 'require':set([units.Zerg.RoachWarren])} ]
### 劫毀蟲
DATA[units.Zerg.Ravager] = [ {'actor':[set([units.Zerg.Roach])], 'perform': FUNCTIONS.Morph_Ravager_quick.id, 'require':set([units.Zerg.RoachWarren])} ]
### 毒爆蟲 !! Train_Baneling_quick ??
DATA[units.Zerg.Baneling] = [ {'actor':[set([units.Zerg.Zergling])], 'perform': FUNCTIONS.Train_Baneling_quick.id, 'require':set([units.Zerg.BanelingNest])} ]
### 刺蛇
DATA[units.Zerg.Hydralisk] = [ {'actor':[set([units.Zerg.Larva])], 'perform': FUNCTIONS.Train_Hydralisk_quick.id, 'require':set([units.Zerg.HydraliskDen])} ]
### 遁地獸
DATA[units.Zerg.Lurker] = [ {'actor':[set([units.Zerg.Hydralisk])], 'perform': FUNCTIONS.Morph_Lurker_quick.id, 'require':set([units.Zerg.LurkerDen])} ]
### 飛螳
DATA[units.Zerg.Mutalisk] = [ {'actor':[set([units.Zerg.Larva])], 'perform': FUNCTIONS.Train_Mutalisk_quick.id, 'require':set([units.Zerg.Spire])} ]
### 腐化飛蟲
DATA[units.Zerg.Corruptor] = [ {'actor':[set([units.Zerg.Larva])], 'perform': FUNCTIONS.Train_Corruptor_quick.id, 'require':set([units.Zerg.Spire])} ]
### 寄生王蟲
DATA[units.Zerg.BroodLord] = [ {'actor':[set([units.Zerg.Corruptor])], 'perform': FUNCTIONS.Morph_BroodLord_quick.id, 'require':set([units.Zerg.GreaterSpire])} ]
### 感染蟲
DATA[units.Zerg.Infestor] = [ {'actor':[set([units.Zerg.Larva])], 'perform': FUNCTIONS.Train_Infestor_quick.id, 'require':set([units.Zerg.InfestationPit])} ]
### 百生獸
DATA[units.Zerg.SwarmHost] = [ {'actor':[set([units.Zerg.Larva])], 'perform': FUNCTIONS.Train_SwarmHost_quick.id, 'require':set([units.Zerg.InfestationPit])} ]
### 飛蟒
DATA[units.Zerg.Viper] = [ {'actor':[set([units.Zerg.Larva])], 'perform': FUNCTIONS.Train_Viper_quick.id, 'require':set([units.Zerg.Hive])} ]
### 雷獸
DATA[units.Zerg.Ultralisk] = [ {'actor':[set([units.Zerg.Larva])], 'perform': FUNCTIONS.Train_Ultralisk_quick.id, 'require':set([units.Zerg.UltraliskCavern])} ]



## 蟲族建築
### 孵化所
DATA[units.Zerg.Hatchery] = [ {'actor':[set([units.Zerg.Drone])], 'perform': FUNCTIONS.Build_Hatchery_screen.id, 'require':None} ]
### 蟲穴
DATA[units.Zerg.Lair] = [ {'actor':[set([units.Zerg.Hatchery])], 'perform': FUNCTIONS.Morph_Lair_quick.id, 'require':set([units.Zerg.SpawningPool])} ]
### 蟲巢
DATA[units.Zerg.Hive] = [ {'actor':[set([units.Zerg.Lair])], 'perform': FUNCTIONS.Morph_Hive_quick.id, 'require':set([units.Zerg.InfestationPit])} ]
### 瓦斯萃取巢
DATA[units.Zerg.Extractor] = [ {'actor':[set([units.Zerg.Drone])], 'perform': FUNCTIONS.Build_Extractor_screen.id, 'require':None} ]
### 孵化池
DATA[units.Zerg.SpawningPool] = [ {'actor':[set([units.Zerg.Drone])], 'perform': FUNCTIONS.Build_SpawningPool_screen.id, 'require':set([units.Zerg.Hatchery, units.Zerg.Lair, units.Zerg.Hive])} ]
### 進化室
DATA[units.Zerg.EvolutionChamber] = [ {'actor':[set([units.Zerg.Drone])], 'perform': FUNCTIONS.Build_EvolutionChamber_screen.id, 'require':set([units.Zerg.Hatchery, units.Zerg.Lair, units.Zerg.Hive])} ]
### 孢子爬行蟲
DATA[units.Zerg.SporeCrawler] = [ {'actor':[set([units.Zerg.Drone])], 'perform': FUNCTIONS.Build_SporeCrawler_screen.id, 'require':set([units.Zerg.SpawningPool])} ]
### 脊刺爬行蟲
DATA[units.Zerg.SpineCrawler] = [ {'actor':[set([units.Zerg.Drone])], 'perform': FUNCTIONS.Build_SpineCrawler_screen.id, 'require':set([units.Zerg.SpawningPool])} ]
### 蟑螂繁殖場
DATA[units.Zerg.RoachWarren] = [ {'actor':[set([units.Zerg.Drone])], 'perform': FUNCTIONS.Build_RoachWarren_screen.id, 'require':set([units.Zerg.SpawningPool])} ]
### 毒爆蟲巢
DATA[units.Zerg.BanelingNest] = [ {'actor':[set([units.Zerg.Drone])], 'perform': FUNCTIONS.Build_BanelingNest_screen.id, 'require':set([units.Zerg.SpawningPool])} ]
### 刺蛇巢穴
DATA[units.Zerg.HydraliskDen] = [ {'actor':[set([units.Zerg.Drone])], 'perform': FUNCTIONS.Build_HydraliskDen_screen.id, 'require':set([units.Zerg.Lair, units.Zerg.Hive])} ]
### 遁地獸穴
DATA[units.Zerg.LurkerDen] = [ {'actor':[set([units.Zerg.Drone])], 'perform': FUNCTIONS.Build_LurkerDen_screen.id, 'require':set([units.Zerg.HydraliskDen])} ]
### 螺旋塔
DATA[units.Zerg.Spire] = [ {'actor':[set([units.Zerg.Drone])], 'perform': FUNCTIONS.Build_Spire_screen.id, 'require':set([units.Zerg.Lair, units.Zerg.Hive])} ]
### 巨型螺旋塔
DATA[units.Zerg.GreaterSpire] = [ {'actor':[set([units.Zerg.Spire])], 'perform': FUNCTIONS.Morph_GreaterSpire_quick.id, 'require':set([units.Zerg.Hive])} ]
### 感染巢
DATA[units.Zerg.InfestationPit] = [ {'actor':[set([units.Zerg.Drone])], 'perform': FUNCTIONS.Build_InfestationPit_screen.id, 'require':set([units.Zerg.Lair, units.Zerg.Hive])} ]
### 雷獸洞穴
DATA[units.Zerg.UltraliskCavern] = [ {'actor':[set([units.Zerg.Drone])], 'perform': FUNCTIONS.Build_UltraliskCavern_screen.id, 'require':set([units.Zerg.Hive])} ]
### 蟲殖地網
DATA[units.Zerg.NydusNetwork] = [ {'actor':[set([units.Zerg.Drone])], 'perform': FUNCTIONS.Build_NydusNetwork_screen.id, 'require':set([units.Zerg.Lair, units.Zerg.Hive])} ]
### 地下蠕蟲 !!! NydusWorm
DATA[units.Zerg.NydusCanal] = [ {'actor':[set([units.Zerg.NydusNetwork])], 'perform': FUNCTIONS.Build_NydusWorm_screen.id, 'require':None} ]
### 蟲苔瘤
# DATA[units.Zerg.CreepTumor] = [ {'actor':[set([units.Zerg.Queen, units.Zerg.CreepTumorBurrowed])], 'perform': FUNCTIONS.Build_CreepTumor_screen.id, 'require':[]} ]

