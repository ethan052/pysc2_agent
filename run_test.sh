#!/bin/sh
### Simple64(簡單64)
MAP=Simple64
### YearZero(紀年起源)
# MAP=YearZero
### PortAleksander(亞歷克桑德港)
MAP=PortAleksander
### NewRepugnancy(新生衝突點)
# MAP=NewRepugnancy
### KingsCove(國王海灣)
# MAP=KingsCove
### KairosJunction(凱羅斯交會點)
# MAP=KairosJunction
### CyberForest(電子叢林)
# MAP=CyberForest
### Automaton(自動化)
# MAP=Automaton

#AGENT1_CLS=pysc2.agents.yiren.idle_agent.IdleAgent1
AGENT1_CLS=pysc2.agents.yiren.yiren_template.PracticeRandomRaceAgent
AGENT1_RACE=random

#AGENT1_CLS=pysc2.agents.yiren.yiren_template.PracticeProtossAgent
#AGENT1_RACE=protoss
AGENT1_CLS=pysc2.agents.yiren.yiren_template.PracticeTerranAgent
AGENT1_RACE=terran
#AGENT1_CLS=pysc2.agents.yiren.yiren_template.PracticeZergAgent
#AGENT1_RACE=zerg

AGENT2_CLS=pysc2.agents.yiren.idle_agent.IdleAgent2
#AGENT2_RACE=protoss
#AGENT2_RACE=terran
#AGENT2_RACE=zerg
AGENT2_RACE=random

python3 -m pysc2.bin.agent --map ${MAP} --render True --save_replay True --agent ${AGENT1_CLS} --agent_race ${AGENT1_RACE} --agent2 ${AGENT2_CLS} --agent2_race ${AGENT2_RACE}
mkdir -p ~yiren/Pictures/${MAP}
mv ~yiren/Pictures/debug_* ~yiren/Pictures/${MAP}/
