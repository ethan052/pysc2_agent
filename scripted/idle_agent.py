from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pysc2.agents import base_agent
from pysc2.lib import actions

FUNCTIONS = actions.FUNCTIONS

class IdleAgent1(base_agent.BaseAgent):
  def step(self, obs):
    return FUNCTIONS.no_op()

class IdleAgent2(base_agent.BaseAgent):
  def step(self, obs):
    return FUNCTIONS.no_op()
