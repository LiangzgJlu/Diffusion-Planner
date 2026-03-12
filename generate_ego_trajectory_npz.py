import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.config import Config


MODEL_INPUT_KEYS = [
    "ego_current_state",
    "neighbor_agents_past",
    "lanes",
    "lanes_speed_limit",
    "lanes_has_speed_limit",
    "route_lanes",
    "route_lanes_speed_limit",
    "route_lanes_has_speed_limit",
    "static_objects",
]


def _to_tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(array)
    if tensor.dtype == torch.float64:
        tensor = tensor.float()
    return tensor.unsqueeze(0).to(device)


def load_model(args_file: str, model_path: str, device: torch.device) -> tuple[Diffusion_Planner, Config]:
    config = Config(args_file, None)
    config.device = str(device)

    model = Diffusion_Planner(config).to(device)
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()
    return model, config


def generate_and_save(model: Diffusion_Planner, config: Config, input_path: Path, output_path: Path, device: torch.device) -> None:
    npz_data = np.load(input_path, allow_pickle=True)

    missing_keys = [k for k in MODEL_INPUT_KEYS if k not in npz_data.files]
    if missing_keys:
        raise KeyError(f"{input_path} 缺少模型输入字段: {missing_keys}")

    model_inputs = {key: _to_tensor(npz_data[key], device) for key in MODEL_INPUT_KEYS}

    with torch.no_grad():
        _, outputs = model(model_inputs)

    prediction = outputs["prediction"][0].detach().cpu().numpy().astype(np.float32)
    ego_agent_future = prediction[0]
    neighbor_agents_future = prediction[1 : 1 + config.predicted_neighbor_num]

    save_data = {key: npz_data[key] for key in npz_data.files}
    save_data["ego_agent_future"] = ego_agent_future
    save_data["neighbor_agents_future"] = neighbor_agents_future

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **save_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成自车轨迹，并将自车/他车轨迹与环境信息按原格式保存到 npz")
    parser.add_argument("--input_dir", required=True, type=str, help="输入 npz 目录")
    parser.add_argument("--output_dir", required=True, type=str, help="输出 npz 目录")
    parser.add_argument("--args_file", required=True, type=str, help="模型配置文件，例如 checkpoints/args.json")
    parser.add_argument("--model_path", required=True, type=str, help="模型权重路径，例如 checkpoints/model.pth")
    parser.add_argument("--device", default="cuda", type=str, help="推理设备，默认 cuda")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    npz_files = sorted(input_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"{input_dir} 下没有找到 npz 文件")

    model, config = load_model(args.args_file, args.model_path, device)

    for npz_file in tqdm(npz_files, desc="Generating ego trajectories"):
        target_path = output_dir / npz_file.name
        generate_and_save(model, config, npz_file, target_path, device)

    print(f"Done. Saved {len(npz_files)} files to {output_dir}")
