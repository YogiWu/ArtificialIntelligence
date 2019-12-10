#%%
from enum import Enum

class NodeEnum():
  accept = 0
  reject = 1
  init = 2
  normal = 3

class MoveDirection():
  left = -1
  right = 1

#%%
class StateNode():
  def __init__(self, node_type, id):    
    self.state_transfer_list = []
    self.node_type = node_type
    self._id = id

  def get_id(self):
    return self._id

  def add_transfer(self, state_transfer):
    self.state_transfer_list.append(state_transfer)

#%%
class Transfer():
  def __init__(self, state, cover_state, move_direction):
    self.state = state
    self.cover_state = cover_state
    self.move_direction = move_direction

class StateTransfer(Transfer):
  def __init__(self, state, cover_state, move_direction, start_node_id, next_node_id):
    super().__init__(state, cover_state, move_direction)

    self.start_node_id = start_node_id
    self.next_node_id = next_node_id

#%%
class CaculateGraph():
  def __init__(self):
    self.node_list = []

  def add_state_node(self, node):
    if len(self.node_list) != node.get_id():
      print("the node index not match the node id")
    self.node_list.append(node)

  def add_state_transfer(self, state, cover_state, move_direction, start_node_id, next_node_id):
    for node in self.node_list:
      if node.get_id() == start_node_id:
        node.add_transfer(StateTransfer(state, cover_state, move_direction, start_node_id, next_node_id))
        return

  def caculate(self, state_list):
    if len(self.node_list) == 0:
      return False

    begin_node = self.node_list[0]
    if begin_node.node_type != NodeEnum.init:
      print("graph is illegal")
      return False
    
    node = begin_node
    current_index = 0
    while True:
      if node.node_type == NodeEnum.reject:
        return False

      if node.node_type == NodeEnum.accept:
        return True

      state = state_list[current_index]
      for transfer in node.state_transfer_list:
        if transfer.state == state:
          state_list[current_index] = transfer.cover_state
          current_index += transfer.move_direction
          node = self.node_list[transfer.next_node_id]
          break

#%%
class StateEnums():
  beigin = 8
  end = 9
  true = 1
  false = 0

node_list = [
  NodeEnum.init,
  NodeEnum.normal,
  NodeEnum.normal,
  NodeEnum.reject,
  NodeEnum.accept
]

transfer_list = [
  # (state, cover_state, move_direction, start_node_id, next_node_id)
  (StateEnums.beigin, StateEnums.beigin, MoveDirection.right, 0, 1),
  (0, 0, MoveDirection.right, 1, 1),
  (1, 1, MoveDirection.right, 1, 2),
  (StateEnums.end, StateEnums.end, MoveDirection.right, 1, 3),
  (0, 0, MoveDirection.right, 2, 1),
  (1, 1, MoveDirection.right, 2, 2),
  (StateEnums.end, StateEnums.end, MoveDirection.right, 2, 4),
]

cal_graph = CaculateGraph()
for i in range(len(node_list)):
  cal_graph.add_state_node(StateNode(node_list[i], i))

for transfer in transfer_list:
  cal_graph.add_state_transfer(*transfer)

input_list = [StateEnums.beigin, 0, 0, 1, 1, 1, StateEnums.end]
print(cal_graph.caculate(input_list))

#%%
