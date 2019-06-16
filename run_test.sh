#!/bin/sh
python3 -m pysc2.bin.agent --map Simple64 --render True --save_replay True --agent pysc2.agents.yiren.idle_agent.IdleAgent1 --agent_race terran --agent2 pysc2.agents.yiren.idle_agent.IdleAgent2 --agent2_race zerg
