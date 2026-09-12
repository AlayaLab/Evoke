

import json
import os
import random
from typing import Dict as TDict, Iterable, List

import numpy as np
import torch
from addict import Dict
from tqdm import tqdm

from depth_anything_3.bench.print_metrics import MetricsPrinter
from depth_anything_3.utils.parallel_utils import parallel_execution
from depth_anything_3.bench.registries import MV_REGISTRY
from depth_anything_3.utils.constants import EVAL_REF_VIEW_STRATEGY


class Evaluator:


    VALID_MODES = {"pose", "recon_unposed", "recon_posed", "view_syn"}

    def __init__(
        self,
        work_dir: str = "./eval_workspace",
        datas: List[str] = ("dtu",),
        modes: List[str] = ("recon_unposed",),
        ref_view_strategy: str = EVAL_REF_VIEW_STRATEGY,
        scenes: List[str] = None,
        debug: bool = False,
        num_fusion_workers: int = 4,
        max_frames: int = 100,
        gpu_id: int = 0,
        total_gpus: int = 1,
    ):


        self.work_dir = work_dir
        self.datas = list(datas)
        self.modes = set(modes)
        self.ref_view_strategy = ref_view_strategy
        self.scenes_filter = scenes
        self.debug = debug
        self.num_fusion_workers = num_fusion_workers
        self.max_frames = max_frames
        self.gpu_id = gpu_id
        self.total_gpus = total_gpus


        unknown = self.modes - self.VALID_MODES
        if unknown:
            raise ValueError(f"Unknown modes: {unknown}. Valid: {sorted(self.VALID_MODES)}")

        os.makedirs(self.work_dir, exist_ok=True)


        self.datasets = Dict()
        for data in self.datas:
            if not MV_REGISTRY.has(data):
                available = list(MV_REGISTRY.all().keys())
                raise ValueError(f"Dataset '{data}' not found. Available: {available}")
            self.datasets[data] = MV_REGISTRY.get(data)()


        self._printer = MetricsPrinter()


    def all(self, api) -> TDict[str, dict]:


        self.infer(api)
        return self.eval()

    def _get_scenes(self, dataset) -> List[str]:

        all_scenes = dataset.SCENES
        if self.scenes_filter:
            scenes = [s for s in all_scenes if s in self.scenes_filter]
            if self.debug:
                print(f"[DEBUG] Filtered scenes: {scenes} (from {len(all_scenes)} total)")
            return scenes
        return all_scenes

    def infer(self, api, model_path: str = None) -> None:


        need_unposed = {"pose", "recon_unposed"} & self.modes
        need_posed = {"recon_posed", "view_syn"} & self.modes
        export_format = "mini_npz-glb" if self.debug else "mini_npz"


        all_tasks = []
        for data in self.datas:
            dataset = self.datasets[data]
            for scene in self._get_scenes(dataset):
                all_tasks.append((data, scene))


        if self.total_gpus > 1:
            tasks = [t for i, t in enumerate(all_tasks) if i % self.total_gpus == self.gpu_id]
            print(f"[INFO] GPU {self.gpu_id}/{self.total_gpus}: {len(tasks)}/{len(all_tasks)} tasks")
        else:
            tasks = all_tasks
            print(f"[INFO] Total inference tasks: {len(tasks)}")

        for data, scene in tqdm(tasks, desc=f"Inference (GPU {self.gpu_id})"):
            dataset = self.datasets[data]
            scene_data = dataset.get_data(scene)
            scene_data = self._sample_frames(scene_data, scene)

            if need_unposed:
                export_dir = self._export_dir(data, scene, posed=False)
                api.inference(
                    scene_data.image_files,
                    export_dir=export_dir,
                    export_format=export_format,
                    ref_view_strategy=self.ref_view_strategy,
                )
                self._save_gt_meta(export_dir, scene_data)

            if need_posed:
                export_dir = self._export_dir(data, scene, posed=True)
                api.inference(
                    scene_data.image_files,
                    scene_data.extrinsics,
                    scene_data.intrinsics,
                    export_dir=export_dir,
                    export_format=export_format,
                    ref_view_strategy=self.ref_view_strategy,
                )
                self._save_gt_meta(export_dir, scene_data)

    def eval(self) -> TDict[str, dict]:


        summary: TDict[str, dict] = {}


        if "pose" in self.modes:
            print(f"\n{'='*60}")
            print(f"📊 Evaluating POSE for all datasets...")
            print(f"{'='*60}")
            for data, result in self._eval_pose():
                summary[f"{data}_pose"] = result

        if "recon_unposed" in self.modes:
            print(f"\n{'='*60}")
            print(f"📊 Evaluating RECON_UNPOSED for all datasets...")
            print(f"{'='*60}")
            for data, result in self._eval_reconstruction("recon_unposed"):
                summary[f"{data}_recon_unposed"] = result

        if "recon_posed" in self.modes:
            print(f"\n{'='*60}")
            print(f"📊 Evaluating RECON_POSED for all datasets...")
            print(f"{'='*60}")
            for data, result in self._eval_reconstruction("recon_posed"):
                summary[f"{data}_recon_posed"] = result

        if "view_syn" in self.modes:

            pass

        return summary

    def print_metrics(self, metrics: TDict[str, dict] = None) -> None:


        if metrics is None:
            metrics = self._load_metrics()

        self._printer.print_results(metrics)


    def _eval_pose(self) -> Iterable[tuple]:

        os.makedirs(self._metric_dir, exist_ok=True)

        for data in tqdm(self.datas, desc="Datasets (pose eval)"):
            dataset = self.datasets[data]
            dataset_results = Dict()
            scenes = self._get_scenes(dataset)

            for scene in tqdm(scenes, desc=f"{data} scenes", leave=False):
                export_dir = self._export_dir(data, scene, posed=False)
                result_path = os.path.join(export_dir, "exports", "mini_npz", "results.npz")


                if not os.path.exists(result_path):
                    print(f"\n[ERROR] Result file not found: {result_path}")
                    print(f"[ERROR] CWD: {os.getcwd()}")
                    print(f"[ERROR] Please run inference first (remove --eval_only)")
                    continue

                try:

                    gt_meta = self._load_gt_meta(export_dir)
                    if gt_meta is not None:
                        result = self._compute_pose_with_gt(result_path, gt_meta)
                    else:

                        result = dataset.eval_pose(scene, result_path)
                    dataset_results[scene] = self._to_float_dict(result)
                except Exception as e:
                    print(f"\n[ERROR] Failed to evaluate pose for {data}/{scene}: {e}")
                    print(f"[ERROR] File path: {os.path.abspath(result_path)}")
                    if self.debug:
                        import traceback
                        traceback.print_exc()
                    continue

            if not dataset_results:
                print(f"[WARNING] No valid results for {data}")
                continue

            dataset_results["mean"] = self._mean_of_dicts(dataset_results.values())
            out_path = os.path.join(self._metric_dir, f"{data}_pose.json")
            self._dump_json(out_path, dataset_results)
            yield data, dataset_results

    def _eval_reconstruction(self, mode: str) -> Iterable[tuple]:


        assert mode in {"recon_unposed", "recon_posed"}
        os.makedirs(self._metric_dir, exist_ok=True)

        posed_flag = mode == "recon_posed"


        recon_datas = [d for d in self.datas if d != "dtu64"]

        for data in tqdm(recon_datas, desc=f"Datasets ({mode} eval)"):
            dataset = self.datasets[data]
            dataset_results = Dict()
            scenes = self._get_scenes(dataset)


            scene_list = []
            result_paths = []
            fuse_paths = []
            for scene in scenes:
                export_dir = self._export_dir(data, scene, posed=posed_flag)
                result_path = os.path.join(export_dir, "exports", "mini_npz", "results.npz")
                fuse_path = os.path.join(export_dir, "exports", "fuse", "pcd.ply")
                scene_list.append(scene)
                result_paths.append(result_path)
                fuse_paths.append(fuse_path)


            use_sequential = (data == "dtu")
            parallel_execution(
                scene_list,
                result_paths,
                fuse_paths,
                action=lambda s, rp, fp: dataset.fuse3d(s, rp, fp, mode),
                num_processes=self.num_fusion_workers,
                print_progress=True,
                desc=f"{data} fusion",
                sequential=use_sequential,
            )


            for scene, fuse_path in zip(scene_list, fuse_paths):

                if data == "dtu" and hasattr(dataset, "eval3d"):
                    result = dataset.eval3d(scene, fuse_path)
                else:
                    result = dataset.eval3d(scene, fuse_path)
                dataset_results[scene] = self._to_float_dict(result)
                print(f"  {mode} | {data} | {scene}: {result}")

            dataset_results["mean"] = self._mean_of_dicts(dataset_results.values())
            out_path = os.path.join(self._metric_dir, f"{data}_{mode}.json")
            self._dump_json(out_path, dataset_results)
            yield data, dataset_results


    def _save_gt_meta(self, export_dir: str, scene_data: Dict) -> None:


        meta_path = os.path.join(export_dir, "exports", "gt_meta.npz")
        os.makedirs(os.path.dirname(meta_path), exist_ok=True)
        np.savez_compressed(
            meta_path,
            extrinsics=scene_data.extrinsics,
            intrinsics=scene_data.intrinsics,
            image_files=np.array(scene_data.image_files, dtype=object),
        )

    def _load_gt_meta(self, export_dir: str) -> Dict:


        meta_path = os.path.join(export_dir, "exports", "gt_meta.npz")
        if os.path.exists(meta_path):
            data = np.load(meta_path)
            return Dict({
                "extrinsics": data["extrinsics"],
                "intrinsics": data["intrinsics"],
            })
        return None

    def _compute_pose_with_gt(self, result_path: str, gt_meta: Dict) -> TDict[str, float]:


        from depth_anything_3.bench.dataset import _wait_for_file_ready
        from depth_anything_3.bench.utils import compute_pose
        from depth_anything_3.utils.geometry import as_homogeneous

        _wait_for_file_ready(result_path)
        pred = np.load(result_path)
        return compute_pose(
            torch.from_numpy(as_homogeneous(pred["extrinsics"])),
            torch.from_numpy(as_homogeneous(gt_meta["extrinsics"])),
        )

    def _sample_frames(self, scene_data: Dict, scene: str) -> Dict:


        if self.max_frames <= 0:
            return scene_data

        num_frames = len(scene_data.image_files)
        if num_frames <= self.max_frames:
            return scene_data


        random.seed(42)
        indices = list(range(num_frames))
        random.shuffle(indices)
        sampled_indices = sorted(indices[:self.max_frames])

        print(f"  [Sampling] {scene}: {num_frames} -> {self.max_frames} frames")


        sampled = Dict()
        sampled.image_files = [scene_data.image_files[i] for i in sampled_indices]
        sampled.extrinsics = scene_data.extrinsics[sampled_indices]
        sampled.intrinsics = scene_data.intrinsics[sampled_indices]


        sampled.aux = Dict()
        for key, val in scene_data.aux.items():
            if isinstance(val, list) and len(val) == num_frames:
                sampled.aux[key] = [val[i] for i in sampled_indices]
            elif isinstance(val, np.ndarray) and len(val) == num_frames:
                sampled.aux[key] = val[sampled_indices]
            else:
                sampled.aux[key] = val

        return sampled

    @property
    def _metric_dir(self) -> str:

        return os.path.join(self.work_dir, "metric_results")

    def _export_dir(self, data: str, scene: str, posed: bool) -> str:


        suffix = "posed" if posed else "unposed"
        export_dir = os.path.join(self.work_dir, "model_results", data, scene, suffix)
        os.makedirs(export_dir, exist_ok=True)
        return export_dir

    @staticmethod
    def _to_float_dict(d: TDict[str, float]) -> dict:

        return {k: float(v) for k, v in d.items()}

    @staticmethod
    def _mean_of_dicts(dicts: Iterable[dict]) -> dict:

        dicts = list(dicts)
        if not dicts:
            return {}
        keys = dicts[0].keys()
        return {k: float(np.mean([d[k] for d in dicts]).item()) for k in keys}

    @staticmethod
    def _dump_json(path: str, obj: dict, indent: int = 4) -> None:

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=indent, ensure_ascii=False)

    def _load_metrics(self) -> TDict[str, dict]:

        metrics = {}
        metric_dir = self._metric_dir

        if not os.path.exists(metric_dir):
            return metrics

        for filename in os.listdir(metric_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(metric_dir, filename)
                try:
                    with open(filepath, encoding="utf-8") as f:
                        data = json.load(f)
                    key = filename[:-5]
                    metrics[key] = data
                except Exception as e:
                    print(f"Warning: Failed to read metrics file: {filename} - {e}")

        return metrics


if __name__ == "__main__":
    import sys
    from omegaconf import OmegaConf
    from depth_anything_3.cfg import load_config


    _default_config = os.path.join(
        os.path.dirname(__file__), "configs", "eval_bench.yaml"
    )


    if "--help" in sys.argv or "-h" in sys.argv:
        pass


    argv = sys.argv[1:]


    config_path = _default_config
    if "--config" in argv:
        config_idx = argv.index("--config")
        if config_idx + 1 < len(argv):
            config_path = argv[config_idx + 1]

            argv = argv[:config_idx] + argv[config_idx + 2:]


    if "--help" in sys.argv or "-h" in sys.argv:
        print("""
DepthAnything3 Benchmark Evaluation

Usage:
  python -m depth_anything_3.bench.evaluator [OPTIONS] [KEY=VALUE ...]

Configuration:
  --config PATH                      Config YAML file (default: bench/configs/eval_bench.yaml)

Config Overrides (using dotlist notation):
  model.path=VALUE                   Model path or HuggingFace ID
  workspace.work_dir=VALUE           Working directory for outputs
  eval.datasets=[dataset1,dataset2]  Datasets to evaluate (eth3d,7scenes,scannetpp,hiroom,dtu,dtu64)
  eval.modes=[mode1,mode2]           Evaluation modes (pose,recon_unposed,recon_posed)
  eval.scenes=[scene1,scene2]        Specific scenes to evaluate (null=all)
  eval.max_frames=VALUE              Max frames per scene (-1=no limit, default: 100)
  eval.ref_view_strategy=VALUE       Reference view strategy (default: first)
  eval.eval_only=VALUE               Only run evaluation (skip inference) (true/false)
  eval.print_only=VALUE              Only print saved metrics (true/false)
  inference.num_fusion_workers=VALUE Number of parallel workers (default: 4)
  inference.debug=VALUE              Enable debug mode (true/false)

Special Flags:
  --help, -h                         Show this help message

Multi-GPU:
  Use CUDA_VISIBLE_DEVICES to specify GPUs (auto-detected and distributed)

Examples:
  # Use default config
  python -m depth_anything_3.bench.evaluator

  # Override model path
  python -m depth_anything_3.bench.evaluator model.path=depth-anything/DA3-LARGE

  # Evaluate specific datasets and modes
  python -m depth_anything_3.bench.evaluator \\
      eval.datasets=[eth3d,hiroom] \\
      eval.modes=[pose]

  # Use custom config with overrides
  python -m depth_anything_3.bench.evaluator \\
      --config my_config.yaml \\
      model.path=/path/to/model \\
      eval.max_frames=50

  # Multi-GPU inference (auto-distributed)
  CUDA_VISIBLE_DEVICES=0,1,2,3 python -m depth_anything_3.bench.evaluator

  # Debug specific scenes
  python -m depth_anything_3.bench.evaluator \\
      eval.datasets=[eth3d] \\
      eval.scenes=[courtyard] \\
      inference.debug=true

  # Only evaluate (skip inference)
  python -m depth_anything_3.bench.evaluator eval.eval_only=true

  # Only print saved metrics
  python -m depth_anything_3.bench.evaluator eval.print_only=true

          """)
        sys.exit(0)


    config = load_config(config_path, argv=argv)


    work_dir = config.workspace.work_dir
    model_path = config.model.path
    datasets = config.eval.datasets
    modes = config.eval.modes
    ref_view_strategy = config.eval.ref_view_strategy
    scenes = config.eval.scenes
    max_frames = config.eval.max_frames
    eval_only = config.eval.eval_only
    print_only = config.eval.print_only
    debug = config.inference.debug
    num_fusion_workers = config.inference.num_fusion_workers


    gpu_id = 0
    total_gpus = 1
    for arg in argv:
        if arg.startswith("gpu_id="):
            gpu_id = int(arg.split("=")[1])
        elif arg.startswith("total_gpus="):
            total_gpus = int(arg.split("=")[1])


    if scenes:
        print(f"[INFO] Running on specific scenes: {scenes}")

    evaluator = Evaluator(
        work_dir=work_dir,
        datas=datasets,
        modes=modes,
        ref_view_strategy=ref_view_strategy,
        scenes=scenes,
        debug=debug,
        num_fusion_workers=num_fusion_workers,
        max_frames=max_frames,
        gpu_id=gpu_id,
        total_gpus=total_gpus,
    )

    if print_only:
        evaluator.print_metrics()
    elif eval_only:
        metrics = evaluator.eval()
        evaluator.print_metrics(metrics)
    else:


        cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_devices is not None and cuda_devices.strip():
            gpu_list = [g.strip() for g in cuda_devices.split(",") if g.strip()]
        else:

            num_available = torch.cuda.device_count()
            gpu_list = [str(i) for i in range(num_available)] if num_available > 0 else ["0"]


        is_worker = os.environ.get("_DA3_WORKER") == "1"

        if len(gpu_list) > 1 and not is_worker:

            import subprocess

            num_gpus = len(gpu_list)
            print(f"[INFO] Detected {num_gpus} GPUs: {gpu_list}")
            print(f"[INFO] Launching {num_gpus} workers...")


            base_cmd = [sys.executable, "-m", "depth_anything_3.bench.evaluator"]

            if config_path != _default_config:
                base_cmd += ["--config", config_path]
            base_cmd += [f"model.path={model_path}"]
            base_cmd += [f"workspace.work_dir={work_dir}"]
            base_cmd += [f"eval.datasets=[{','.join(datasets)}]"]
            base_cmd += [f"eval.modes=[{','.join(modes)}]"]
            if scenes:
                base_cmd += [f"eval.scenes=[{','.join(scenes)}]"]
            base_cmd += [f"eval.max_frames={max_frames}"]
            base_cmd += [f"eval.ref_view_strategy={ref_view_strategy}"]
            base_cmd += [f"inference.debug={str(debug).lower()}"]
            base_cmd += [f"inference.num_fusion_workers={num_fusion_workers}"]


            processes = []
            for idx, gpu_id in enumerate(gpu_list):
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = gpu_id
                env["_DA3_WORKER"] = "1"

                cmd = base_cmd.copy()

                cmd += [f"gpu_id={idx}", f"total_gpus={num_gpus}"]

                print(f"[INFO] Starting worker {idx} on GPU {gpu_id}")
                p = subprocess.Popen(cmd, env=env)
                processes.append(p)


            for p in processes:
                p.wait()

            print(f"[INFO] All {num_gpus} workers completed")


            metrics = evaluator.eval()
            evaluator.print_metrics(metrics)
        else:

            from depth_anything_3.api import DepthAnything3

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            api = DepthAnything3.from_pretrained(model_path)
            api = api.to(device)

            evaluator.infer(api, model_path=model_path)


            if not is_worker:
                metrics = evaluator.eval()
                evaluator.print_metrics(metrics)
