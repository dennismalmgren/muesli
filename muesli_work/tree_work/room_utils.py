from typing import Tuple

import torch
from tensordict import TensorDict
from torchrl.data.map.hash import SipHash

class HashTensorWrapper():
    def __init__(self, tensor):
        self.tensor = tensor

    def __hash__(self):
        return self.tensor

    def __eq__(self, other):
        return torch.all(self.tensor == other.tensor)
    
TYPE_LOOKUP = {
    0: 'wall',
    1: 'empty space',
    2: 'box target',
    3: 'box on target',
    4: 'box not on target',
    5: 'player'
}

ACTION_LOOKUP = {
    0: 'push up',
    1: 'push down',
    2: 'push left',
    3: 'push right',
    4: 'move up',
    5: 'move down',
    6: 'move left',
    7: 'move right',
}

# Moves are mapped to coordinate changes as follows
# 0: Move up
# 1: Move down
# 2: Move left
# 3: Move right
CHANGE_COORDINATES = torch.tensor([[-1, 0],
                                   [1, 0],
                                   [0, -1],
                                   [0, 1]])


# Global variables used for reverse playing.
explored_states = set()
global_num_boxes = 0
best_room_score = -1
best_room = None
best_box_mapping = None

def box_displacement_score(box_mapping):
    """
    Calculates the sum of all Manhattan distances, between the boxes
    and their origin box targets.
    :param box_mapping:
    :return:
    """
    score = torch.norm((box_mapping["target"] - box_mapping["source"]).float(), p=1)
 
    return score


def reverse_move(room_state, room_structure, box_mapping, last_pull, action):
    global CHANGE_COORDINATES
    """
    Perform reverse action. Where all actions in the range [0, 3] correspond to
    push actions and the ones greater 3 are simmple move actions.
    :param room_state:
    :param room_structure:
    :param box_mapping:
    :param last_pull:
    :param action:
    :return:
    """
    player_position = torch.where(room_state == 5)
    player_position = torch.cat(player_position)
    
    CHANGE_COORDINATES = CHANGE_COORDINATES.to(room_state.device)
    change = CHANGE_COORDINATES[action % 4]
    next_position = player_position + change

    # Check if next position is an empty floor or an empty box target
    if room_state[*next_position] in [1, 2]:

        # Move player, independent of pull or move action.
        room_state[*player_position] = room_structure[*player_position]
        room_state[*next_position] = 5

        # In addition try to pull a box if the action is a pull action
        if action < 4:
            possible_box_location = -1 * change
            possible_box_location += player_position

            if room_state[*possible_box_location] in [3, 4]:
                # Perform pull of the adjacent box
                room_state[*player_position] = 3
                room_state[*possible_box_location] = room_structure[*possible_box_location]

                # Update the box mapping
                for ind, (target, source) in enumerate(zip(box_mapping["target"].unbind(0), box_mapping["source"].unbind(0))):
                    if (source == possible_box_location).all():
                        box_mapping["source"][ind] = player_position
                        last_pull = ind

    return room_state, box_mapping, last_pull

def depth_first_search(room_state, room_structure, box_mapping, box_swaps = 0, last_pull = (-1, 1), ttl=300):
    global explored_states, global_num_boxes, best_room_score, best_room, best_box_mapping
    ttl -= 1
    if ttl <= 0 or len(explored_states) >= 300000:
        return
    hash_fn = SipHash(as_tensor=True)
    state_tohash = hash_fn(room_state)
    state_tohash = tuple(state_tohash.tolist())
    if not (state_tohash in explored_states):
        room_score = box_swaps * box_displacement_score(box_mapping)
        if torch.where(room_state == 2)[0].shape[0] != global_num_boxes:
            room_score = 0
        if room_score > best_room_score:
            best_room = room_state
            best_room_score = room_score
            best_box_mapping = box_mapping

        explored_states.add(state_tohash)
        
        for action in ACTION_LOOKUP.keys():
            # The state and box mapping  need to be copied to ensure
            # every action start from a similar state.
            room_state_next = room_state.clone()
            box_mapping_next = box_mapping.clone()
            room_state_next, box_mapping_next, last_pull_next = \
                reverse_move(room_state_next, room_structure, box_mapping_next, last_pull, action)
            
            box_swaps_next = box_swaps
            if last_pull_next != last_pull:
                box_swaps_next += 1

            depth_first_search(room_state_next, room_structure,
                               box_mapping_next, box_swaps_next,
                               last_pull, ttl)

def reverse_playing(room_state, room_structure, num_boxes, search_depth=100, batch_size=[]):
    global explored_states, global_num_boxes, best_room_score, best_room, best_box_mapping

    num_boxes_unbatched = num_boxes[0]

    B = batch_size[0]
    
    #num_boxes = (room_structure == 2).float().sum(dim=-1).sum(dim=-1)
    box_locations = torch.where(room_structure == 2) #tuple of (B, X, Y) tensors
    
    #num_boxes = len(box_locations[0])

    the_box_locations = torch.stack((box_locations[1], box_locations[2]), dim=-1).int().reshape(-1, num_boxes_unbatched, 2)
    #final box mapping structure
    box_mapping  = TensorDict({
        "target": the_box_locations, #x, y locations for each box in each batch itemF
        "source": the_box_locations.clone(), #x, y locations for each box in each batch itemF
        },
        batch_size=B)

    global_num_boxes = num_boxes_unbatched
    best_room = room_state
    batch_best_room_score = torch.ones(B, device=room_state.device)
    for batch_ind in range(batch_size[0]):
        explored_states = set()
        best_room_score = -1
        best_box_mapping = box_mapping[batch_ind]
        best_room = room_state[batch_ind]

        depth_first_search(room_state[batch_ind], room_structure[batch_ind], box_mapping[batch_ind], box_swaps = 0, last_pull = (-1, 1), ttl=300)
        batch_best_room_score[batch_ind] = best_room_score
        box_mapping[batch_ind] = best_box_mapping
        room_state[batch_ind] = best_room

    return room_state, batch_best_room_score, box_mapping

def place_boxes_and_player(room, num_boxes, second_player, batch_size=[]):
    """
    Places the player and the boxes into the floors in a room (supports batched rooms).

    :param room: Tensor of shape (batch_size, dim_x, dim_y)
    :param num_boxes: Number of boxes to place
    :param second_player: Boolean indicating whether to place a second player
    :return: Tensor of rooms with players and boxes placed
    """
    dim_x, dim_y = room.shape[-2:]
    second_player_unbatched = second_player[0]
    num_boxes_unbatched = num_boxes[0]
    num_players = 2 if second_player_unbatched else 1
    
    B = batch_size[0]

    room_flat = room.view(B, -1)
    valid_mask = (room_flat == 1)  # (B, X*Y)
    free_spots_per_batch = valid_mask.sum(dim=-1)  # (B,)
    if (free_spots_per_batch < num_players).any():
        raise RuntimeError(
            f"Not enough free floor spots for {num_players} players in one or more batches."
        )
    
    scores = torch.rand(B, dim_x * dim_y, device=room.device)

    scores[~valid_mask] = float('-inf')
    
    _, topk_indices = scores.topk(num_players, dim=-1, largest=True)
    # Decode flattened indices back to (x, y)
    x_coords = topk_indices // dim_y
    y_coords = topk_indices % dim_y
    room[torch.arange(B, device=room.device).unsqueeze(-1), x_coords, y_coords] = 5

    #-----------------------------
    # 2) Place Boxes
    #-----------------------------
    # Re-flatten and compute new valid mask (floors still == 1)
    room_flat = room.view(B, -1)
    valid_mask = (room_flat == 1)  # (B, X*Y)
    
     # Check each batch again for enough free spots
    free_spots_per_batch = valid_mask.sum(dim=-1)
    if (free_spots_per_batch < num_boxes).any():
        raise RuntimeError(
            f"Not enough free floor spots for {num_boxes} boxes in one or more batches."
        )
    
     # Random scores for boxes
    scores = torch.rand(B, dim_x * dim_y, device=room.device)
    scores[~valid_mask] = float('-inf')

      # topk to choose 'num_boxes' positions per batch
    _, topk_indices = scores.topk(num_boxes_unbatched, dim=-1, largest=True)
    x_coords = topk_indices // dim_y
    y_coords = topk_indices % dim_y

    room[torch.arange(B, device=room.device).unsqueeze(1), x_coords, y_coords] = 2
    return room

def room_topology_generation(dim=(10, 10), p_change_directions=0.35, num_steps=15, batch_size=[]) -> torch.Tensor:
    dim_unbatched = dim[0]
    dim_x, dim_y = dim_unbatched
    num_steps_unbatched = num_steps[0]
     # The ones in the mask represent all fields which will be set to floors
    # during the random walk. The centered one will be placed over the current
    # position of the walk.

    masks = torch.tensor([
        [
            [0, 0, 0],
            [1, 1, 1],
            [0, 0, 0]
        ],
        [
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0]
        ],
        [
            [0, 0, 0],
            [1, 1, 0],
            [0, 1, 0]
        ],
        [
            [0, 0, 0],
            [1, 1, 0],
            [1, 1, 0]
        ],
        [
            [0, 0, 0],
            [0, 1, 1],
            [0, 1, 0]
        ]
    ],
    dtype=torch.int)

    # Possible directions during the walk
    directions = torch.tensor([[1, 0], 
                [0, 1], 
                [-1, 0], 
                [0, -1]])
    direction_indices = torch.randint(low=0, high=len(directions), size=batch_size)
    direction = directions[direction_indices] #batch x 2

    # Starting position of random walk
    x_positions = torch.randint(low=0, high=dim_x, size=batch_size)
    y_positions = torch.randint(low=0, high=dim_y, size=batch_size)

    position = torch.stack((x_positions, y_positions), dim=-1)
    position_limits_min = torch.tensor([[1, 1]], dtype=torch.int)
    position_limits_max = torch.tensor([[dim_x - 2, dim_y - 2]], dtype=torch.int)
  # Precompute 3x3 neighborhood offsets
    offsets = torch.tensor([[-1, -1], [-1, 0], [-1, 1],
                            [0, -1], [0, 0], [0, 1],
                            [1, -1], [1, 0], [1, 1]], dtype=torch.int)  # shape: (9, 2)


    level = torch.zeros(size=(*batch_size, *dim_unbatched), dtype=torch.int)
    for s in range(num_steps_unbatched):
        change_directions = torch.rand(size=batch_size) < p_change_directions
        direction_indices = torch.randint(low=0, high=len(directions), size=batch_size)
        direction[change_directions] = directions[direction_indices][change_directions]

        #update position
        position = position + direction
        position.clamp_(min=position_limits_min, max=position_limits_max)

        #apply mask
        mask_indices = torch.randint(low=0, high=len(masks), size=batch_size)
        selected_masks = masks[mask_indices]  # shape: (batch_size, 3, 3)
               # Apply offsets to get 9 positions for each sample in the batch
        neighbor_positions = position.unsqueeze(1) + offsets.unsqueeze(0)  # shape: (batch_size, 9, 2)

        # Flatten neighbor positions and selected masks for scatter_add
        flat_positions = neighbor_positions.view(-1, 2)  # shape: (batch_size * 9, 2)
        flat_masks = selected_masks.view(*batch_size, -1).flatten()#.repeat_interleave(9, dim=0)  # shape: (batch_size * 9,)

        # Create index tensors for scatter_add
        batch_indices = torch.arange(batch_size[0]).unsqueeze(1).repeat(1, 9).view(-1)  # shape: (batch_size * 9,)
        x_indices, y_indices = flat_positions[:, 0], flat_positions[:, 1]

        # Scatter add the masks onto the level map
        level.index_put_((batch_indices, x_indices, y_indices), flat_masks, accumulate=True)

        
    # Ensure values are either 0 or 1 (binary map)
    level = torch.clamp(level, max=1)

     # Set borders to 0
    level[:, :, [0, dim_y - 1]] = 0
    level[:, [0, dim_x - 1], :] = 0

    return level


def generate_room(dim=(13, 13), p_change_directions=0.35, num_steps=25, num_boxes=3, tries=4, second_player=False, batch_size=[]) -> Tuple[torch.Tensor, torch.Tensor, TensorDict]:
    batch_added = False
    if len(batch_size) == 0:
        batch_added = True
        batch_size = torch.Size([1])
        dim = dim.unsqueeze(0)
        num_boxes = num_boxes.unsqueeze(0)
        second_player = second_player.unsqueeze(0)
        num_steps = num_steps.unsqueeze(0)

    dim_unbatched = dim[0] if len(batch_size) > 0 else dim
    
    room_state = torch.zeros(size=(*batch_size, *dim_unbatched))
    room_structure = torch.zeros(size=(*batch_size, *dim_unbatched))
    #a bit risky, only try with low batch sizes...otherwise we want conditional re-generation to be added.
    for t in range(tries):
        room = room_topology_generation(dim, p_change_directions, num_steps, batch_size)
        room = place_boxes_and_player(room, num_boxes=num_boxes, second_player=second_player, batch_size=batch_size)

        room_structure = torch.clone(room)
        room_structure[room_structure == 5] = 1

        room_state = torch.clone(room)
        room_state[room_state == 2] = 4

        room_state, score, box_mapping = reverse_playing(room_state, room_structure, num_boxes, batch_size=batch_size)
        room_state[room_state == 3] = 4

        if torch.all(score > 0):
            break

    if torch.any(score == 0):
        raise RuntimeWarning('Generated Model with scores == 0')
    
    if batch_added:
        room_structure = room_structure[0]
        room_state = room_state[0]
        box_mapping = box_mapping[0]

    return room_structure, room_state, box_mapping

# structure, state, mapping = generate_room()
# print(mapping)