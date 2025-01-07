from typing import Optional
import torch

import torchrl

from torchrl.envs import EnvBase
from tensordict import TensorDict
from torchrl.data import Bounded, Composite, Unbounded, Categorical
from torchrl.envs.utils import check_env_specs, step_mdp

from room_utils import generate_room, ACTION_LOOKUP, CHANGE_COORDINATES

def make_composite_from_td(td):
    # custom function to convert a ``tensordict`` in a similar spec structure
    # of unbounded values.
    composite = Composite(
        {
            key: make_composite_from_td(tensor)
            if isinstance(tensor, TensorDict)
            else Unbounded(
                dtype=tensor.dtype, device=tensor.device, shape=tensor.shape
            )
            for key, tensor in td.items()
        },
        shape=td.shape,
    )
    return composite


class SokobanEnv(EnvBase):
    batch_locked = False

    metadata = {
        "render_modes": ["human", "rgb_array", "tiny_human", "tiny_rgb_array", "raw"]
    }

    def __init__(self,
                 td_params = None,
                 seed = None,
                 batch_size = None,
                 device="cpu"):
        if td_params is None:
            td_params = self.gen_params(batch_size=batch_size)
        super().__init__(device=device, batch_size=batch_size)
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)

        num_boxes_unbatched = td_params["params", "num_boxes"][0] if batch_size is not None and len(batch_size) > 0 else td_params["params", "num_boxes"]
        box_mapping_template = TensorDict({
        "target": torch.zeros(*td_params.shape, num_boxes_unbatched, 2, dtype=torch.int),
        "source": torch.zeros(*td_params.shape, num_boxes_unbatched, 2, dtype=torch.int),
        })
        dim_room_unbatched = td_params["params", "dim_room"][0] if batch_size is not None and len(batch_size) > 0 else td_params["params", "dim_room"]
        #TODO: state? self.boxes_on_target = 0
        #now the specs
        self.observation_spec = Composite(
            observed_room=Bounded(low=0., high=255., shape=(*td_params.shape, dim_room_unbatched[1], dim_room_unbatched[0], 3), dtype=torch.float32),
            params = make_composite_from_td(td_params["params"]),
            room_fixed = Unbounded(shape=(*td_params.shape, *dim_room_unbatched), dtype=torch.int),
            room_state = Unbounded(shape=(*td_params.shape, *dim_room_unbatched), dtype=torch.int),
            player_position = Unbounded(shape=(*td_params.shape, 2), dtype=torch.int),
            box_mapping = make_composite_from_td(box_mapping_template),
            num_env_steps = Unbounded(shape=(*td_params.shape, 1), dtype=torch.int),
            boxes_on_target = Unbounded(shape=(*td_params.shape, 1), dtype=torch.int),
            shape=self.batch_size,
        )

        self.state_spec = self.observation_spec.clone()

        self.action_spec = Categorical(n=len(ACTION_LOOKUP), shape=td_params.shape)
        self.reward_spec = Unbounded(shape=(*td_params.shape, 1))

        # Penalties and Rewards
        self.penalty_for_step = -0.1
        self.penalty_box_off_target = -1
        self.reward_box_on_target = 1
        self.reward_finished = 10
        self.reward_last = 0

    def room_to_tiny_world_rgb(self, room, room_structure=None, scale=1, batch_size=[]):
        room = room.clone()    
        if not room_structure is None:
            # Change the ID of a player on a target
            room[(room == 5) & (room_structure == 2)] = 6

        wall = [0, 0, 0]
        floor = [243, 248, 238]
        box_target = [254, 126, 125]
        box_on_target = [254, 95, 56]
        box = [142, 121, 56]
        player = [160, 212, 56]
        player_on_target = [219, 212, 56]

        surfaces = torch.tensor([wall, floor, box_target, box_on_target, box, player, player_on_target], device=room.device)

        # Assemble the new rgb_room, with all loaded images
        room_small_rgb = torch.zeros(size=(*batch_size, room.shape[-2]*scale, room.shape[-1]*scale, 3), dtype=torch.float32)
        for i in range(room.shape[-2]):
            x_i = i * scale
            for j in range(room.shape[-1]):
                y_j = j * scale
                surfaces_id = room[..., i, j]
                room_small_rgb[..., x_i:(x_i+scale), y_j:(y_j+scale), :] = surfaces[surfaces_id].unsqueeze(-2).unsqueeze(-2)

        return room_small_rgb
    
    def gen_params(self, batch_size = None) -> TensorDict:
        if batch_size is None:
            batch_size = []
        #TODO: num_gen_steps can also be specified
        dim_room = (10, 10)
        td = TensorDict(
            {
                "params": TensorDict(
                    {
                        "dim_room": (7, 7),
                        "num_gen_steps": int(1.7 * (dim_room[0] + dim_room[1])),
                        "num_boxes": 2,
                        "max_steps": 120,    
                        "screen_height": dim_room[0] * 16,
                        "screen_width": dim_room[1] * 16,
                        "second_player": False              
                    }
                )
            }
        )

        if batch_size:
            td = td.expand(batch_size).contiguous()
        return td

    def _reset(self, tensordict):
        if tensordict is None or tensordict.is_empty():
            tensordict = self.gen_params(batch_size=self.batch_size)
        batch_dim_added = False
        if len(tensordict.shape) == 0:
            tensordict = tensordict.unsqueeze(0)
            batch_dim_added = True

        try:
            room_fixed, room_state, box_mapping = generate_room(dim=tensordict["params", "dim_room"], num_steps=tensordict["params", "num_gen_steps"],
                                num_boxes=tensordict["params", "num_boxes"],
                                second_player=tensordict["params", "second_player"],
                                batch_size=tensordict.shape)
        except (RuntimeError, RuntimeWarning) as e:
            print("[SOKOBAN] Runtime Error/Warning: {}".format(e))
            print("[SOKOBAN] Retry . . .")
            if batch_dim_added:
                tensordict = tensordict.squeeze(0)
            return self._reset(tensordict)
        
        player_position = torch.argwhere(room_state == 5)[..., 1:].type(torch.int)

        starting_observation = self.room_to_tiny_world_rgb(room_state, room_fixed, batch_size=tensordict.shape)
        num_env_steps = torch.zeros((*tensordict.shape, 1), dtype=torch.int)
        
        out = TensorDict({
                "observed_room": starting_observation,
                "params": tensordict["params"],
                "room_fixed": room_fixed,
                "room_state": room_state,
                "box_mapping": box_mapping,
                "player_position": player_position,
                "num_env_steps": num_env_steps,
                "boxes_on_target": torch.zeros(*tensordict.shape, 1, dtype=torch.int)
            },
            batch_size=tensordict.shape
        )

        if batch_dim_added:
            out = out.squeeze(0)
        return out
    
    def _move(self, tensordict, action, player_position, dims, room_state, room_fixed):
        global CHANGE_COORDINATES
        CHANGE_COORDINATES = CHANGE_COORDINATES.to(action.device)
        dims_unbatched = dims[0]

        change = CHANGE_COORDINATES[(action.squeeze(-1) - 1) % 4]
        new_position = player_position + change
        current_position = player_position.clone()
        valid_move_environments = (new_position[..., 0] >= 0) & (new_position[..., 0] < dims_unbatched[0]) \
                                    & (new_position[..., 1] >= 0) & (new_position[..., 1] < dims_unbatched[1])
        valid_move_indices = valid_move_environments.nonzero(as_tuple=True)[0] 

        batch_indices = torch.arange(room_state.shape[0])[valid_move_environments]
        x_indices = new_position[..., 0][valid_move_environments]
        y_indices = new_position[..., 1][valid_move_environments]
        can_move = (room_state[batch_indices, x_indices, y_indices] == 1) |  (room_state[batch_indices, x_indices, y_indices] == 2)
        valid_move_environments[valid_move_indices] = can_move

        #now move.
        batch_indices = torch.arange(room_state.shape[0])[valid_move_environments]
        x_indices = new_position[..., 0][valid_move_environments]
        y_indices = new_position[..., 1][valid_move_environments]
        room_state[batch_indices, x_indices, y_indices] = 5
        x_indices = current_position[..., 0][valid_move_environments]
        y_indices = current_position[..., 1][valid_move_environments]
        room_state[batch_indices, x_indices, y_indices] = room_fixed[batch_indices, x_indices, y_indices]
        return valid_move_environments, room_state


    def _push(self, tensordict, action, player_position, dims, room_state, room_fixed):
        global CHANGE_COORDINATES
        CHANGE_COORDINATES = CHANGE_COORDINATES.to(action.device)
        dims_unbatched = dims[0]

        change = CHANGE_COORDINATES[(action.squeeze(-1) - 1) % 4]
        new_position = player_position + change
        current_position = player_position.clone()
        assert len(dims_unbatched) == 2, "Incorrect dims: " + str(len(tensordict.shape))
        # No push, if the push would get the box out of the room's grid
        new_box_position = new_position + change
        valid_push_environments = (new_box_position[..., 0] >= 0) & (new_box_position[..., 0] < dims_unbatched[0]) \
                                    & (new_box_position[..., 1] >= 0) & (new_box_position[..., 1] < dims_unbatched[1])
        valid_push_indices = valid_push_environments.nonzero(as_tuple=True)[0] 

        batch_indices = torch.arange(room_state.shape[0])[valid_push_environments]
        x_indices = new_position[..., 0][valid_push_environments]
        y_indices = new_position[..., 1][valid_push_environments]
        can_push_box = (room_state[batch_indices, x_indices, y_indices] == 3) |  (room_state[batch_indices, x_indices, y_indices] == 4)
        x_indices = new_box_position[..., 0][valid_push_environments]
        y_indices = new_box_position[..., 1][valid_push_environments]
        can_push_box &= (room_state[batch_indices, x_indices, y_indices] == 1) |  (room_state[batch_indices, x_indices, y_indices] == 2)
        valid_push_environments[valid_push_indices] = can_push_box

        # Now we're going to push the valid ones.
        batch_indices = torch.arange(room_state.shape[0])[valid_push_environments]
        #Move player
        x_indices = new_position[..., 0][valid_push_environments]
        y_indices = new_position[..., 1][valid_push_environments]
        room_state[batch_indices, x_indices, y_indices] = 5
        x_indices = current_position[..., 0][valid_push_environments]
        y_indices = current_position[..., 1][valid_push_environments]
        room_state[batch_indices, x_indices, y_indices] = room_fixed[batch_indices, x_indices, y_indices]

        #Move box
        box_types = torch.ones_like(room_fixed) * 4
        box_types[room_fixed == 2] = 3
        batch_indices = torch.arange(room_state.shape[0])[valid_push_environments]
        x_indices = new_box_position[..., 0][valid_push_environments]
        y_indices = new_box_position[..., 1][valid_push_environments]        
        room_state[batch_indices, x_indices, y_indices] = box_types[batch_indices, x_indices, y_indices]

        #return filter, new box position, old box position, player position, room state
        return valid_push_environments, room_state
    
    def _step(self, tensordict):
        batch_dim_added = False
        if len(tensordict.shape) == 0:
            tensordict = tensordict.unsqueeze(0)
            batch_dim_added = True

        #todo: check if done, for batched execution
        num_env_steps = tensordict["num_env_steps"] + 1
        action = tensordict["action"]
        pause_environments = (action == 0)
        pause_environments = pause_environments
        push_environments = (action > 0) & (action < 5)
        push_environments = push_environments
        player_position = tensordict["player_position"]
        dims = tensordict["params", "dim_room"]
        room_state = tensordict["room_state"]
        room_fixed = tensordict["room_fixed"]
        if torch.any(push_environments):
            assert len(dims[push_environments].shape) == 2, "Was: " + str(len(dims[push_environments].shape)) + ", batch: " + str(batch_dim_added) + ", orig: " + str(len(dims.shape))
            pushed_environments, new_room_state = self._push(tensordict[push_environments], 
                                                         action[push_environments], 
                                                         player_position[push_environments], 
                                                         dims[push_environments], 
                                                         room_state[push_environments],
                                                         room_fixed[push_environments])
            push_indices = push_environments.nonzero(as_tuple=True)[0] 

            push_environments[push_indices] = pushed_environments
            room_state[push_environments] = new_room_state[pushed_environments]

        move_environments = ~push_environments & ~pause_environments
        if torch.any(move_environments):
            assert len(dims[move_environments].shape) == 2, "Was: " + str(len(dims[move_environments].shape)) + ", batch: " + str(batch_dim_added) + ", orig: " + str(len(dims.shape))
            moved_environments, new_room_state = self._move(tensordict[move_environments], 
                                                    action[move_environments], 
                                                    player_position[move_environments], 
                                                    dims[move_environments], 
                                                    room_state[move_environments],
                                                    room_fixed[move_environments])
            move_indices = move_environments.nonzero(as_tuple=True)[0] 
            move_environments[move_indices] = moved_environments

            room_state[move_environments] = new_room_state[moved_environments]

        room_observation = self.room_to_tiny_world_rgb(room_state, room_fixed, batch_size=tensordict.shape)

        player_position = torch.argwhere(room_state == 5)[..., 1:].type(torch.int)
        #TODO: reward


            
        out = TensorDict({
                "observed_room": room_observation,
                "params": tensordict["params"],
                "room_fixed": room_fixed,
                "room_state": room_state,
                "box_mapping": tensordict["box_mapping"],
                "player_position": player_position,
                "num_env_steps": num_env_steps,
                "boxes_on_target": tensordict["boxes_on_target"] #this will be removed later
            },
            batch_size=tensordict.shape
        )

        reward, boxes_on_target = self._calc_reward_and_completion(out)
        out["reward"] = reward
        out["boxes_on_target"] = boxes_on_target

        terminated = self._check_if_terminated(out)
        out["done"] = terminated
        out["terminated"] = terminated.clone()

        #TODO: new boxes'on'target
        if batch_dim_added:
            out = out.squeeze(0)
        return out
    
    def _calc_reward_and_completion(self, tensordict):
        num_boxes_unbatched = tensordict["params", "num_boxes"][0]
        room_state = tensordict["room_state"]
        room_fixed = tensordict["room_fixed"]
        reward = self.penalty_for_step * torch.ones(*tensordict.shape, 1)
        empty_targets = room_state == 2

        player_on_target = (room_fixed == 2) & (room_state == 5)
        total_targets = empty_targets | player_on_target
        current_boxes_on_target = num_boxes_unbatched - total_targets.sum(-1).sum(-1, keepdims=True)
        old_boxes_on_target = tensordict["boxes_on_target"]
        reward[current_boxes_on_target > old_boxes_on_target] += self.reward_box_on_target
        reward[current_boxes_on_target < old_boxes_on_target] += self.penalty_box_off_target

        games_won = self._check_if_all_boxes_on_target(tensordict)
        reward[games_won] += self.reward_finished
        return reward, current_boxes_on_target.type(torch.int)


    def _check_if_terminated(self, tensordict):
        max_steps_reached = self._check_if_maxsteps(tensordict)
        boxes_on_target = self._check_if_all_boxes_on_target(tensordict)
        return boxes_on_target | max_steps_reached
    
    def _check_if_all_boxes_on_target(self, tensordict):
        room_state = tensordict["room_state"]
        room_fixed = tensordict["room_fixed"]
        empty_targets = room_state == 2
        player_hiding_target = (room_fixed == 2) & (room_fixed == 5)
        are_all_boxes_on_targets = (empty_targets | player_hiding_target).float().sum(-1).sum(-1) == 0
        return are_all_boxes_on_targets.unsqueeze(-1)

    def _check_if_maxsteps(self, tensordict):
        max_steps = tensordict["params", "max_steps"].unsqueeze(-1)
        num_env_steps = tensordict["num_env_steps"]
        return (num_env_steps == max_steps)
    
    def _set_seed(self, seed: Optional[int]):
        rng = torch.manual_seed(seed)
        self.rng = rng

# # #test
# env = SokobanEnv()
# # check_env_specs(env)
# td = env.reset()
# done = td["done"]
# while not torch.all(done):
#     action = env.action_spec.sample()
#     td['action'] = action
#     td = env.step(td)
#     done = td["next", "done"]
#     td = step_mdp(td)
# print('done')
# print("Boxes on target: ", td["boxes_on_target"].item())
