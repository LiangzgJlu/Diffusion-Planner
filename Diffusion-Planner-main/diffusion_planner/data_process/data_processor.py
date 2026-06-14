import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters

from diffusion_planner.data_process.roadblock_utils import route_roadblock_correction
from diffusion_planner.data_process.agent_process import (
    agent_past_process,
    sampled_tracked_objects_to_array_list,
    sampled_static_objects_to_array_list,
    all_agent_past_future_process,
)
from diffusion_planner.data_process.map_process import get_neighbor_vector_set_map, map_process
from diffusion_planner.data_process.ego_process import get_ego_past_array_from_scenario, get_ego_future_array_from_scenario, calculate_additional_ego_states
from diffusion_planner.data_process.utils import convert_to_model_inputs


_PROCESSOR = None


def _init_worker(processor):
    global _PROCESSOR
    _PROCESSOR = processor


def _process_scenario_worker(scenario):
    try:
        return _PROCESSOR.process_and_save_scenario(scenario)
    except Exception as exc:
        map_name = getattr(scenario, "_map_name", "unknown_map")
        token = getattr(scenario, "token", "unknown_token")
        raise RuntimeError(f"Failed to process scenario {map_name}_{token}") from exc


def ego_past_to_agent_feature(ego_agent_past):
    """
    Convert ego history to the same feature format used by neighbor agents.

    Args:
        ego_agent_past: [T_h, 7], [x, y, heading, vx, vy, ax, ay]

    Returns:
        [T_h, 11], [x, y, cos_h, sin_h, vx, vy, width, length, type...]
    """
    vehicle_params = get_pacifica_parameters()
    t_h = ego_agent_past.shape[0]
    ego_feature = np.zeros((t_h, 11), dtype=np.float32)

    ego_feature[:, 0] = ego_agent_past[:, 0]
    ego_feature[:, 1] = ego_agent_past[:, 1]
    ego_feature[:, 2] = np.cos(ego_agent_past[:, 2])
    ego_feature[:, 3] = np.sin(ego_agent_past[:, 2])
    ego_feature[:, 4] = ego_agent_past[:, 3]
    ego_feature[:, 5] = ego_agent_past[:, 4]
    ego_feature[:, 6] = vehicle_params.width
    ego_feature[:, 7] = vehicle_params.length
    ego_feature[:, 8] = 1.0

    return ego_feature


class DataProcessor(object):
    def __init__(self, config):

        self._save_dir = getattr(config, "save_path", None) 

        self.past_time_horizon = 2 # [seconds]
        self.num_past_poses = 10 * self.past_time_horizon 
        self.future_time_horizon = 8 # [seconds]
        self.num_future_poses = 10 * self.future_time_horizon

        self.num_agents = config.agent_num
        self.num_static = config.static_objects_num
        self.max_ped_bike = 10 # Limit the number of pedestrians and bicycles in the agent.
        self._radius = 100 # [m] query radius scope relative to the current pose.

        self._map_features = ['LANE', 'LEFT_BOUNDARY', 'RIGHT_BOUNDARY', 'ROUTE_LANES'] # name of map features to be extracted.
        self._max_elements = {'LANE': config.lane_num, 'LEFT_BOUNDARY': config.lane_num, 'RIGHT_BOUNDARY': config.lane_num, 'ROUTE_LANES': config.route_num} # maximum number of elements to extract per feature layer.
        self._max_points = {'LANE': config.lane_len, 'LEFT_BOUNDARY': config.lane_len, 'RIGHT_BOUNDARY': config.lane_len, 'ROUTE_LANES': config.route_len} # maximum number of points per feature to extract per feature layer.

        self.num_workers = max(1, int(getattr(config, "num_workers", os.cpu_count()) or 1))

    def _format_saved_agents(
        self,
        ego_agent_past,
        ego_agent_future,
        selected_indices,
        all_neighbor_agents_past,
        all_neighbor_agents_future,
        all_neighbor_history_mask,
        all_neighbor_future_mask,
        all_neighbor_agent_type,
    ):
        num_saved_agents = self.num_agents + 1
        history_steps = self.num_past_poses + 1
        future_steps = self.num_future_poses

        history_trajectory = np.zeros((num_saved_agents, history_steps, 8), dtype=np.float32)
        future_trajectory = np.zeros((num_saved_agents, future_steps, 3), dtype=np.float32)
        history = np.zeros((num_saved_agents, history_steps), dtype=np.bool_)
        future_mask = np.zeros((num_saved_agents, future_steps), dtype=np.bool_)
        agent_type = np.zeros((num_saved_agents, 3), dtype=np.float32)

        ego_agent_past_feature = ego_past_to_agent_feature(ego_agent_past)
        history_trajectory[0] = ego_agent_past_feature[:, :8]
        future_trajectory[0] = ego_agent_future.astype(np.float32)
        history[0] = True
        future_mask[0] = True
        agent_type[0] = [1.0, 0.0, 0.0]

        for output_idx, agent_idx in enumerate(selected_indices[:self.num_agents], start=1):
            if agent_idx >= all_neighbor_agents_past.shape[0]:
                continue

            history_trajectory[output_idx] = all_neighbor_agents_past[agent_idx, :, :8]
            future_trajectory[output_idx] = all_neighbor_agents_future[agent_idx]
            history[output_idx] = all_neighbor_history_mask[agent_idx]
            future_mask[output_idx] = all_neighbor_future_mask[agent_idx]
            agent_type[output_idx] = all_neighbor_agent_type[agent_idx]

        return history_trajectory, future_trajectory, history, future_mask, agent_type

    # Use for inference
    def observation_adapter(self, history_buffer, traffic_light_data, map_api, route_roadblock_ids, device='cpu'):

        '''
        ego
        '''
        ego_agent_past = None # inference no need ego_agent_past
        ego_state = history_buffer.current_state[0]
        ego_coords = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)
        anchor_ego_state = np.array([ego_state.rear_axle.x, ego_state.rear_axle.y, ego_state.rear_axle.heading], dtype=np.float64)

        '''
        neighbor
        '''
        observation_buffer = history_buffer.observation_buffer # Past observations including the current
        neighbor_agents_past, neighbor_agents_types = sampled_tracked_objects_to_array_list(observation_buffer)
        static_objects, static_objects_types = sampled_static_objects_to_array_list(observation_buffer[-1])
        _, neighbor_agents_past, _, static_objects = \
            agent_past_process(ego_agent_past, neighbor_agents_past, neighbor_agents_types, self.num_agents, static_objects, static_objects_types, self.num_static, self.max_ped_bike, anchor_ego_state)

        '''
        Map
        '''
        # Simply fixing disconnected routes without pre-searching for reference lines
        route_roadblock_ids = route_roadblock_correction(
            ego_state, map_api, route_roadblock_ids
        )
        coords, traffic_light_data, speed_limit, lane_route = get_neighbor_vector_set_map(
            map_api, self._map_features, ego_coords, self._radius, traffic_light_data
        )
        vector_map = map_process(route_roadblock_ids, anchor_ego_state, coords, traffic_light_data, speed_limit, lane_route, self._map_features, 
                                    self._max_elements, self._max_points)

        
        data = {"neighbor_agents_past": neighbor_agents_past[:, -21:],
                "ego_current_state": np.array([0., 0., 1. ,0., 0., 0., 0., 0., 0., 0.], dtype=np.float32), # ego centric x, y, cos, sin, vx, vy, ax, ay, steering angle, yaw rate, we only use x, y, cos, sin during inference
                "static_objects": static_objects}
        data.update(vector_map)
        data = convert_to_model_inputs(data, device)

        return data
    
    # Use for data preprocess
    def work(self, scenarios, num_workers=None, chunksize=1):
        num_workers = self.num_workers if num_workers is None else max(1, int(num_workers))
        total = len(scenarios) if hasattr(scenarios, "__len__") else None
        if total == 0:
            return
        if total is not None:
            num_workers = min(num_workers, total)

        if num_workers == 1:
            for scenario in tqdm(scenarios, total=total):
                self.process_and_save_scenario(scenario)
            return

        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_init_worker,
            initargs=(self,),
        ) as executor:
            results = executor.map(
                _process_scenario_worker,
                scenarios,
                chunksize=max(1, int(chunksize)),
            )
            for _ in tqdm(results, total=total):
                pass

    def process_and_save_scenario(self, scenario):
        data = self.process_scenario(scenario)
        return self.save_to_disk(self._save_dir, data)

    def process_scenario(self, scenario):
        map_name = scenario._map_name
        token = scenario.token
        map_api = scenario.map_api        

        '''
        ego & agents past
        '''
        ego_state = scenario.initial_ego_state
        ego_coords = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)
        anchor_ego_state = np.array([ego_state.rear_axle.x, ego_state.rear_axle.y, ego_state.rear_axle.heading], dtype=np.float64)
        ego_agent_past, time_stamps_past = get_ego_past_array_from_scenario(scenario, self.num_past_poses, self.past_time_horizon)

        present_tracked_objects = scenario.initial_tracked_objects.tracked_objects
        past_tracked_objects = [
            tracked_objects.tracked_objects
            for tracked_objects in scenario.get_past_tracked_objects(
                iteration=0, time_horizon=self.past_time_horizon, num_samples=self.num_past_poses
            )
        ]
        sampled_past_observations = past_tracked_objects + [present_tracked_objects]
        neighbor_agents_past_raw, neighbor_agents_types_raw = \
            sampled_tracked_objects_to_array_list(sampled_past_observations)
        neighbor_agents_past_for_topk = [agents.copy() for agents in neighbor_agents_past_raw]
        neighbor_agents_past_for_all = [agents.copy() for agents in neighbor_agents_past_raw]
        
        static_objects, static_objects_types = sampled_static_objects_to_array_list(present_tracked_objects)

        ego_agent_past, neighbor_agents_past, neighbor_indices, static_objects = \
            agent_past_process(ego_agent_past, neighbor_agents_past_for_topk, neighbor_agents_types_raw, self.num_agents, static_objects, static_objects_types, self.num_static, self.max_ped_bike, anchor_ego_state)
        
        '''
        Map
        '''
        route_roadblock_ids = scenario.get_route_roadblock_ids()
        traffic_light_data = list(scenario.get_traffic_light_status_at_iteration(0))

        if route_roadblock_ids != ['']:
            route_roadblock_ids = route_roadblock_correction(
                ego_state, map_api, route_roadblock_ids
            )

        coords, traffic_light_data, speed_limit, lane_route = get_neighbor_vector_set_map(
            map_api, self._map_features, ego_coords, self._radius, traffic_light_data
        )

        vector_map = map_process(route_roadblock_ids, anchor_ego_state, coords, traffic_light_data, speed_limit, lane_route, self._map_features, 
                                self._max_elements, self._max_points)

        '''
        ego & agents future
        '''
        ego_agent_future = get_ego_future_array_from_scenario(scenario, ego_state, self.num_future_poses, self.future_time_horizon)

        present_tracked_objects = scenario.initial_tracked_objects.tracked_objects
        future_tracked_objects = [
            tracked_objects.tracked_objects
            for tracked_objects in scenario.get_future_tracked_objects(
                iteration=0, time_horizon=self.future_time_horizon, num_samples=self.num_future_poses
            )
        ]

        sampled_future_observations = [present_tracked_objects] + future_tracked_objects
        future_tracked_objects_array_list, _ = sampled_tracked_objects_to_array_list(sampled_future_observations)
        future_tracked_objects_array_list_for_all = [agents.copy() for agents in future_tracked_objects_array_list]

        (
            all_neighbor_agents_past,
            all_neighbor_agents_future,
            all_neighbor_history_mask,
            all_neighbor_future_mask,
            all_neighbor_agent_type,
        ) = all_agent_past_future_process(
            anchor_ego_state=anchor_ego_state,
            past_tracked_objects=neighbor_agents_past_for_all,
            past_tracked_objects_types=neighbor_agents_types_raw,
            future_tracked_objects=future_tracked_objects_array_list_for_all,
        )

        history_trajectory, future_trajectory, history_mask, future_mask, agent_type = self._format_saved_agents(
            ego_agent_past=ego_agent_past,
            ego_agent_future=ego_agent_future,
            selected_indices=neighbor_indices,
            all_neighbor_agents_past=all_neighbor_agents_past,
            all_neighbor_agents_future=all_neighbor_agents_future,
            all_neighbor_history_mask=all_neighbor_history_mask,
            all_neighbor_future_mask=all_neighbor_future_mask,
            all_neighbor_agent_type=all_neighbor_agent_type,
        )


        '''
        ego current
        '''
        ego_current_state = calculate_additional_ego_states(ego_agent_past, time_stamps_past)

        # gather data
        data = {
            "map_name": map_name,
            "token": token,
            "ego_current_state": ego_current_state.astype(np.float32),
            "history_trajectory": history_trajectory,
            "future_trajectory": future_trajectory,
            "history_mask": history_mask,
            "future_mask": future_mask,
            "agent_type": agent_type,
            "static_objects": static_objects,
            "lanes": vector_map["lanes"],
            "lanes_speed_limit": vector_map["lanes_speed_limit"],
            "lanes_has_speed_limit": vector_map["lanes_has_speed_limit"],
            "route_lanes": vector_map["route_lanes_mask"],
            "route_lanes_speed_limit": vector_map["route_lanes_speed_limit"],
            "route_lanes_has_speed_limit": vector_map["route_lanes_has_speed_limit"],
        }

        return data

    def save_to_disk(self, dir, data):
        file_path = f"{dir}/{data['map_name']}_{data['token']}.npz"
        np.savez(file_path, **data)
        return file_path
