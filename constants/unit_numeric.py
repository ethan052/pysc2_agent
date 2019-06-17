from pysc2.lib import units
from pysc2.env.sc2_env import Race

DATA = {}
## 神族兵種
### 探測機
DATA[units.Protoss.Probe] = {'mineral_cost':50, 'vespene_cost':0, 'food_required': 1, 'build_time': 12}
### 狂戰士
DATA[units.Protoss.Zealot] = {'mineral_cost':100, 'vespene_cost':0, 'food_required': 2, 'build_time': 27}
### 哨衛
DATA[units.Protoss.Sentry] = {'mineral_cost':50, 'vespene_cost':100, 'food_required': 2, 'build_time': 26}
### 追獵者
DATA[units.Protoss.Stalker] = {'mineral_cost':125, 'vespene_cost':50, 'food_required': 2, 'build_time': 30}
### 教士
DATA[units.Protoss.Adept] = {'mineral_cost':100, 'vespene_cost':25, 'food_required': 1, 'build_time': 30}
### 高階聖堂武士
DATA[units.Protoss.HighTemplar] = {'mineral_cost':50, 'vespene_cost':150, 'food_required': 2, 'build_time': 39}
### 暗影聖堂武士
DATA[units.Protoss.DarkTemplar] = {'mineral_cost':125, 'vespene_cost':125, 'food_required': 2, 'build_time': 39}
### 執政官 (破壞能)
DATA[units.Protoss.Archon] = {'mineral_cost':0, 'vespene_cost':0, 'food_required': 0, 'build_time': 9}
### 觀察者
DATA[units.Protoss.Observer] = {'mineral_cost':25, 'vespene_cost':75, 'food_required': 1, 'build_time': 21}
### 傳輸稜鏡
DATA[units.Protoss.WarpPrism] = {'mineral_cost':200, 'vespene_cost':0, 'food_required': 2, 'build_time': 36}
### 不朽者
DATA[units.Protoss.Immortal] = {'mineral_cost':275, 'vespene_cost':100, 'food_required': 4, 'build_time': 39}
### 巨像
DATA[units.Protoss.Colossus] = {'mineral_cost':300, 'vespene_cost':200, 'food_required': 6, 'build_time': 54}
### 裂光球
DATA[units.Protoss.Disruptor] = {'mineral_cost':150, 'vespene_cost':150, 'food_required': 3, 'build_time': 36}
### 鳳凰戰機
DATA[units.Protoss.Phoenix] = {'mineral_cost':150, 'vespene_cost':100, 'food_required': 2, 'build_time': 25}
### 先知艦
DATA[units.Protoss.Oracle] = {'mineral_cost':150, 'vespene_cost':150, 'food_required': 3, 'build_time': 37}
### 虛空艦
DATA[units.Protoss.VoidRay] = {'mineral_cost':250, 'vespene_cost':150, 'food_required': 4, 'build_time': 43}
### 暴風艦
DATA[units.Protoss.Tempest] = {'mineral_cost':250, 'vespene_cost':175, 'food_required': 5, 'build_time': 43}
### 航空母艦
DATA[units.Protoss.Carrier] = {'mineral_cost':350, 'vespene_cost':250, 'food_required': 6, 'build_time': 64}
### 聖母艦
DATA[units.Protoss.Mothership] = {'mineral_cost':400, 'vespene_cost':400, 'food_required': 8, 'build_time': 114}


## 神族建築
### 星核
DATA[units.Protoss.Nexus] = {'mineral_cost':400, 'vespene_cost':0, 'food_required': 0, 'build_time': 71, 'square_length':5}
### 瓦斯處理廠
DATA[units.Protoss.Assimilator] = {'mineral_cost':75, 'vespene_cost':0, 'food_required': 0, 'build_time': 21, 'square_length':3}
### 水晶塔
DATA[units.Protoss.Pylon] = {'mineral_cost':100, 'vespene_cost':0, 'food_required': 0, 'build_time': 18, 'square_length':2}
### 傳送門
DATA[units.Protoss.Gateway] = {'mineral_cost':150, 'vespene_cost':0, 'food_required': 0, 'build_time': 46, 'square_length':3}
### 空間之門
DATA[units.Protoss.WarpGate] = {'mineral_cost':0, 'vespene_cost':0, 'food_required': 0, 'build_time': 7, 'square_length':3}
### 冶煉廠
DATA[units.Protoss.Forge] = {'mineral_cost':150, 'vespene_cost':0, 'food_required': 0, 'build_time': 32, 'square_length':3}
### 光子加農砲
DATA[units.Protoss.PhotonCannon] = {'mineral_cost':150, 'vespene_cost':0, 'food_required': 0, 'build_time': 29, 'square_length':2}
### 機械控制核心
DATA[units.Protoss.CyberneticsCore] = {'mineral_cost':150, 'vespene_cost':0, 'food_required': 0, 'build_time': 36, 'square_length':3}
### 護盾充能器
DATA[units.Protoss.ShieldBattery] = {'mineral_cost':100, 'vespene_cost':0, 'food_required': 0, 'build_time': 29, 'square_length':2}
### 機械製造廠
DATA[units.Protoss.RoboticsFacility] = {'mineral_cost':150, 'vespene_cost':100, 'food_required': 0, 'build_time': 46, 'square_length':3}
### 機械研究所
DATA[units.Protoss.RoboticsBay] = {'mineral_cost':150, 'vespene_cost':150, 'food_required': 0, 'build_time': 46, 'square_length':3}
### 星際之門
DATA[units.Protoss.Stargate] = {'mineral_cost':150, 'vespene_cost':150, 'food_required': 0, 'build_time': 43, 'square_length':3}
### 艦隊導航台
DATA[units.Protoss.FleetBeacon] = {'mineral_cost':300, 'vespene_cost':200, 'food_required': 0, 'build_time': 43, 'square_length':3}
### 暮光議會
DATA[units.Protoss.TwilightCouncil] = {'mineral_cost':150, 'vespene_cost':100, 'food_required': 0, 'build_time': 36, 'square_length':3}
### 聖堂文庫
DATA[units.Protoss.TemplarArchive] = {'mineral_cost':150, 'vespene_cost':200, 'food_required': 0, 'build_time': 36, 'square_length':3}
### 黑暗神殿
DATA[units.Protoss.DarkShrine] = {'mineral_cost':150, 'vespene_cost':150, 'food_required': 0, 'build_time': 71, 'square_length':2}


## 人類兵種
### 太空工程車
DATA[units.Terran.SCV] = {'mineral_cost':50, 'vespene_cost':0, 'food_required': 1, 'build_time': 12}
### 陸戰隊
DATA[units.Terran.Marine] = {'mineral_cost':50, 'vespene_cost':0, 'food_required': 1, 'build_time': 18}
### 死神
DATA[units.Terran.Reaper] = {'mineral_cost':50, 'vespene_cost':50, 'food_required': 1, 'build_time': 32}
### 掠奪者
DATA[units.Terran.Marauder] = {'mineral_cost':100, 'vespene_cost':25, 'food_required': 2, 'build_time': 21}
### 幽靈特務
DATA[units.Terran.Ghost] = {'mineral_cost':150, 'vespene_cost':125, 'food_required': 2, 'build_time': 29}
### 惡狼
DATA[units.Terran.Hellion] = {'mineral_cost':100, 'vespene_cost':0, 'food_required': 2, 'build_time': 21}
### 戰狼
DATA[units.Terran.Hellbat] = {'mineral_cost':100, 'vespene_cost':0, 'food_required': 2, 'build_time': 21}
### 寡婦詭雷
DATA[units.Terran.WidowMine] = {'mineral_cost':75, 'vespene_cost':25, 'food_required': 2, 'build_time': 21}
### 颶風飛彈車
DATA[units.Terran.Cyclone] = {'mineral_cost':150, 'vespene_cost':100, 'food_required': 3, 'build_time': 32}
### 攻城坦克
DATA[units.Terran.SiegeTank] = {'mineral_cost':150, 'vespene_cost':125, 'food_required': 3, 'build_time': 32}
### 雷神
DATA[units.Terran.Thor] = {'mineral_cost':300, 'vespene_cost':200, 'food_required': 6, 'build_time': 43}
### 維京戰機
DATA[units.Terran.VikingFighter] = {'mineral_cost':150, 'vespene_cost':75, 'food_required': 2, 'build_time': 30}
### 醫療艇
DATA[units.Terran.Medivac] = {'mineral_cost':100, 'vespene_cost':100, 'food_required': 2, 'build_time': 30}
### 解放者
DATA[units.Terran.Liberator] = {'mineral_cost':150, 'vespene_cost':150, 'food_required': 3, 'build_time': 43}
### 渡鴉
DATA[units.Terran.Raven] = {'mineral_cost':100, 'vespene_cost':200, 'food_required': 2, 'build_time': 43}
### 女妖轟炸機
DATA[units.Terran.Banshee] = {'mineral_cost':150, 'vespene_cost':100, 'food_required': 3, 'build_time': 43}
### 戰巡艦
DATA[units.Terran.Battlecruiser] = {'mineral_cost':400, 'vespene_cost':300, 'food_required': 6, 'build_time': 64}


## 人類建築
### 指揮中心
DATA[units.Terran.CommandCenter] = {'mineral_cost':400, 'vespene_cost':0, 'food_required': 0, 'build_time': 71, 'square_length':5}
### 星軌指揮總部
DATA[units.Terran.OrbitalCommand] = {'mineral_cost':150, 'vespene_cost':0, 'food_required': 0, 'build_time': 25, 'square_length':5}
### 行星要塞
DATA[units.Terran.PlanetaryFortress] = {'mineral_cost':150, 'vespene_cost':150, 'food_required': 0, 'build_time': 36, 'square_length':5}
### 瓦斯精煉廠
DATA[units.Terran.Refinery] = {'mineral_cost':75, 'vespene_cost':0, 'food_required': 0, 'build_time': 21, 'square_length':3}
### 補給站
DATA[units.Terran.SupplyDepot] = {'mineral_cost':100, 'vespene_cost':0, 'food_required': 0, 'build_time': 21, 'square_length':2}
### 科技實驗室
DATA[units.Terran.TechLab] = {'mineral_cost':50, 'vespene_cost':25, 'food_required': 0, 'build_time': 18, 'square_length':2}
DATA[units.Terran.BarracksTechLab] = DATA[units.Terran.TechLab]
DATA[units.Terran.FactoryTechLab] = DATA[units.Terran.TechLab]
DATA[units.Terran.StarportTechLab] = DATA[units.Terran.TechLab]
DATA[units.Terran.TechLab] = {'mineral_cost':0, 'vespene_cost':0, 'food_required': 0, 'build_time': 0, 'square_length':2}
### 反應爐
DATA[units.Terran.Reactor] = {'mineral_cost':50, 'vespene_cost':50, 'food_required': 0, 'build_time': 36, 'square_length':2}
DATA[units.Terran.BarracksReactor] = DATA[units.Terran.Reactor]
DATA[units.Terran.FactoryReactor] = DATA[units.Terran.Reactor]
DATA[units.Terran.StarportReactor] = DATA[units.Terran.Reactor]
DATA[units.Terran.Reactor] = {'mineral_cost':0, 'vespene_cost':0, 'food_required': 0, 'build_time': 0, 'square_length':2}
### 兵營
DATA[units.Terran.Barracks] = {'mineral_cost':150, 'vespene_cost':0, 'food_required': 0, 'build_time': 46, 'square_length':3}
### 碉堡
DATA[units.Terran.Bunker] = {'mineral_cost':100, 'vespene_cost':0, 'food_required': 0, 'build_time': 29, 'square_length':3}
### 電機工程所
DATA[units.Terran.EngineeringBay] = {'mineral_cost':125, 'vespene_cost':0, 'food_required': 0, 'build_time': 25, 'square_length':3}
### 感應塔
DATA[units.Terran.SensorTower] = {'mineral_cost':125, 'vespene_cost':100, 'food_required': 0, 'build_time': 18, 'square_length':1}
### 飛彈砲台
DATA[units.Terran.MissileTurret] = {'mineral_cost':100, 'vespene_cost':0, 'food_required': 0, 'build_time': 18, 'square_length':2}
### 軍工廠
DATA[units.Terran.Factory] = {'mineral_cost':150, 'vespene_cost':100, 'food_required': 0, 'build_time': 43, 'square_length':3}
### 兵工廠
DATA[units.Terran.Armory] = {'mineral_cost':150, 'vespene_cost':100, 'food_required': 0, 'build_time': 46, 'square_length':3}
### 星際港
DATA[units.Terran.Starport] = {'mineral_cost':150, 'vespene_cost':100, 'food_required': 0, 'build_time': 36, 'square_length':3}
### 核融合核心
DATA[units.Terran.FusionCore] = {'mineral_cost':150, 'vespene_cost':150, 'food_required': 0, 'build_time': 46, 'square_length':3}
### 幽靈特務學院
DATA[units.Terran.GhostAcademy] = {'mineral_cost':150, 'vespene_cost':50, 'food_required': 0, 'build_time': 29, 'square_length':3}



## 蟲族兵種
### 幼蟲
DATA[units.Zerg.Larva] = {'mineral_cost':0, 'vespene_cost':0, 'food_required': 0, 'build_time': 11}
### 工蟲
DATA[units.Zerg.Drone] = {'mineral_cost':50, 'vespene_cost':0, 'food_required': 1, 'build_time': 12}
### 王蟲
DATA[units.Zerg.Overlord] = { 'mineral_cost':100, 'vespene_cost':0, 'food_required': 0, 'build_time': 18}
### 王蟲 (可運載蟲)
DATA[units.Zerg.OverlordTransport] = { 'mineral_cost':25, 'vespene_cost':25, 'food_required': 0, 'build_time': 12}
### 監察王蟲
DATA[units.Zerg.Overseer] = { 'mineral_cost':50, 'vespene_cost':50, 'food_required': 0, 'build_time': 12}
### 后蟲
DATA[units.Zerg.Queen] = {'mineral_cost':150, 'vespene_cost':0, 'food_required': 2, 'build_time': 36}
### 異化蟲(狗)
DATA[units.Zerg.Zergling] = {'mineral_cost':50, 'vespene_cost':0, 'food_required': 1, 'build_time': 17}
### 蟑螂
DATA[units.Zerg.Roach] = {'mineral_cost':75, 'vespene_cost':25, 'food_required': 2, 'build_time': 19}
### 劫毀蟲
DATA[units.Zerg.Ravager] = {'mineral_cost':25, 'vespene_cost':75, 'food_required': 1, 'build_time': 9}
### 毒爆蟲
DATA[units.Zerg.Baneling] = {'mineral_cost':25, 'vespene_cost':25, 'food_required': 0, 'build_time': 14}
### 刺蛇
DATA[units.Zerg.Hydralisk] = {'mineral_cost':100, 'vespene_cost':50, 'food_required': 2, 'build_time': 24}
### 遁地獸
DATA[units.Zerg.Lurker] = {'mineral_cost':50, 'vespene_cost':100, 'food_required': 1, 'build_time': 18}
### 飛螳
DATA[units.Zerg.Mutalisk] = {'mineral_cost':100, 'vespene_cost':100, 'food_required': 2, 'build_time': 24}
### 腐化飛蟲
DATA[units.Zerg.Corruptor] = {'mineral_cost':150, 'vespene_cost':100, 'food_required': 2, 'build_time': 29}
### 寄生王蟲
DATA[units.Zerg.BroodLord] = {'mineral_cost':150, 'vespene_cost':150, 'food_required': 2, 'build_time': 24}
### 感染蟲
DATA[units.Zerg.Infestor] = {'mineral_cost':100, 'vespene_cost':150, 'food_required': 2, 'build_time': 29}
### 百生獸
DATA[units.Zerg.SwarmHost] = {'mineral_cost':100, 'vespene_cost':75, 'food_required': 3, 'build_time': 29}
### 飛蟒
DATA[units.Zerg.Viper] = {'mineral_cost':100, 'vespene_cost':200, 'food_required': 3, 'build_time': 29}
### 雷獸
DATA[units.Zerg.Ultralisk] = {'mineral_cost':300, 'vespene_cost':200, 'food_required': 6, 'build_time': 39}


## 蟲族建築
### 孵化所
DATA[units.Zerg.Hatchery] = {'mineral_cost':300, 'vespene_cost':0, 'food_required': 0, 'build_time': 71, 'square_length':5}
### 蟲穴
DATA[units.Zerg.Lair] = {'mineral_cost':150, 'vespene_cost':100, 'food_required': 0, 'build_time': 57, 'square_length':5}
### 蟲巢
DATA[units.Zerg.Hive] = {'mineral_cost':200, 'vespene_cost':150, 'food_required': 0, 'build_time': 71, 'square_length':5}
### 瓦斯萃取巢
DATA[units.Zerg.Extractor] = {'mineral_cost':25, 'vespene_cost':0, 'food_required': 0, 'build_time': 21, 'square_length':3}
### 孵化池
DATA[units.Zerg.SpawningPool] = {'mineral_cost':200, 'vespene_cost':0, 'food_required': 0, 'build_time': 46, 'square_length':3}
### 進化室
DATA[units.Zerg.EvolutionChamber] = {'mineral_cost':75, 'vespene_cost':0, 'food_required': 0, 'build_time': 25, 'square_length':3}
### 孢子爬行蟲
DATA[units.Zerg.SporeCrawler] = {'mineral_cost':75, 'vespene_cost':0, 'food_required': 0, 'build_time': 21, 'square_length':2}
### 脊刺爬行蟲
DATA[units.Zerg.SpineCrawler] = {'mineral_cost':100, 'vespene_cost':0, 'food_required': 0, 'build_time': 36, 'square_length':2}
### 蟑螂繁殖場
DATA[units.Zerg.RoachWarren] = {'mineral_cost':150, 'vespene_cost':0, 'food_required': 0, 'build_time': 39, 'square_length':3}
### 毒爆蟲巢
DATA[units.Zerg.BanelingNest] = {'mineral_cost':100, 'vespene_cost':50, 'food_required': 0, 'build_time': 43, 'square_length':3}
### 刺蛇巢穴
DATA[units.Zerg.HydraliskDen] = {'mineral_cost':100, 'vespene_cost':100, 'food_required': 0, 'build_time': 29, 'square_length':3}
### 遁地獸穴
DATA[units.Zerg.LurkerDen] = {'mineral_cost':100, 'vespene_cost':150, 'food_required': 0, 'build_time': 86, 'square_length':3}
### 螺旋塔
DATA[units.Zerg.Spire] = {'mineral_cost':200, 'vespene_cost':200, 'food_required': 0, 'build_time': 71, 'square_length':2}
### 巨型螺旋塔
DATA[units.Zerg.GreaterSpire] = {'mineral_cost':100, 'vespene_cost':150, 'food_required': 0, 'build_time': 71, 'square_length':2}
### 感染巢
DATA[units.Zerg.InfestationPit] = {'mineral_cost':100, 'vespene_cost':100, 'food_required': 0, 'build_time': 36, 'square_length':3}
### 雷獸洞穴
DATA[units.Zerg.UltraliskCavern] = {'mineral_cost':150, 'vespene_cost':200, 'food_required': 0, 'build_time': 46, 'square_length':3}
### 蟲殖地網
DATA[units.Zerg.NydusNetwork] = {'mineral_cost':150, 'vespene_cost':150, 'food_required': 0, 'build_time': 36, 'square_length':3}
### 地下蠕蟲
DATA[units.Zerg.NydusCanal] = {'mineral_cost':50, 'vespene_cost':50, 'food_required': 0, 'build_time': 14, 'square_length':3}
### 蟲苔瘤
DATA[units.Zerg.CreepTumorBurrowed] = {'mineral_cost':0, 'vespene_cost':0, 'food_required': 0, 'build_time': 11, 'square_length':1}
