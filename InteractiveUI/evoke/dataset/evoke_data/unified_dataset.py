from .operators import *
import torch, json, pandas
from einops import rearrange
from scipy.spatial.transform import Rotation as R


HUNYUAN_ACTION_MAPPING = {
    (0, 0, 0, 0): 0,
    (1, 0, 0, 0): 1,
    (0, 1, 0, 0): 2,
    (0, 0, 1, 0): 3,
    (0, 0, 0, 1): 4,
    (1, 0, 1, 0): 5,
    (1, 0, 0, 1): 6,
    (0, 1, 1, 0): 7,
    (0, 1, 0, 1): 8,
}


def hunyuan_one_hot_to_one_dimension(one_hot):

    y = torch.tensor([HUNYUAN_ACTION_MAPPING[tuple(row.tolist())] for row in one_hot])
    return y


_DEFAULT_NORM_INTRINSIC = np.array(
    [[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]], dtype=np.float32
)

class UnifiedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        repeat=1,
        data_file_keys=tuple(),
        main_data_operator=lambda x: x,
        special_operator_map=None,
        jsonl_key_map=None,
        pose_dir=None,
        target_h=None,
        target_w=None,
        source_h=None,
        source_w=None,
        fallback_default_intrinsic=False,
        event_window_keys=None,
        skill_source=False,
        video_loader=None,
    ):
        self.base_path = base_path
        self.metadata_path = metadata_path
        self.repeat = repeat
        self.data_file_keys = data_file_keys
        self.main_data_operator = main_data_operator
        self.cached_data_operator = LoadTorchPickle()
        self.special_operator_map = {} if special_operator_map is None else special_operator_map
        self.jsonl_key_map = jsonl_key_map if jsonl_key_map else {}
        self.pose_dir = pose_dir
        self._target_h = target_h
        self._target_w = target_w

        self._source_h = source_h
        self._source_w = source_w


        self.fallback_default_intrinsic = fallback_default_intrinsic


        self.event_window_keys = event_window_keys
        self.skill_source = bool(skill_source)
        self.video_loader = video_loader
        self.data = []
        self.cached_data = []
        self.load_from_cache = metadata_path is None
        self.load_metadata(metadata_path)

    @staticmethod
    def default_image_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
    ):
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor)),
            (list, SequencialProcess(ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor))),
        ])

    @staticmethod
    def default_video_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        num_frames=81, time_division_factor=4, time_division_remainder=1,
        target_fps=None,
    ):
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> RouteByExtensionName(operator_map=[
                (("jpg", "jpeg", "png", "webp"), LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor) >> ToList()),
                (("gif",), LoadGIF(
                    num_frames, time_division_factor, time_division_remainder,
                    frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor),
                )),
                (("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"), LoadVideo(
                    num_frames, time_division_factor, time_division_remainder,
                    frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor),
                    target_fps=target_fps,
                )),
            ])),
        ])

    def search_for_cached_data_files(self, path):
        for file_name in os.listdir(path):
            subpath = os.path.join(path, file_name)
            if os.path.isdir(subpath):
                self.search_for_cached_data_files(subpath)
            elif subpath.endswith(".pth"):
                self.cached_data.append(subpath)

    def _apply_key_map(self, metadata):

        if not self.jsonl_key_map:
            return metadata
        for entry in metadata:
            for src_key, dst_key in self.jsonl_key_map.items():
                if src_key in entry and dst_key not in entry:
                    entry[dst_key] = entry[src_key]
        return metadata

    def load_metadata(self, metadata_path):
        if metadata_path is None:
            print("No metadata_path. Searching for cached data files.")
            self.search_for_cached_data_files(self.base_path)
            print(f"{len(self.cached_data)} cached data files found.")
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = self._apply_key_map(metadata)
        elif metadata_path.endswith(".jsonl"):
            metadata = []
            with open(metadata_path, 'r') as f:
                for line in f:
                    metadata.append(json.loads(line.strip()))
            self.data = self._apply_key_map(metadata)
        else:
            metadata = pandas.read_csv(metadata_path)
            self.data = self._apply_key_map([metadata.iloc[i].to_dict() for i in range(len(metadata))])


    def load_camera_parameters(
        self,
        pose_path: str,
        frame_ids: list = None,
        pose_type: str = "vipe",
        normalize_translation: bool = True
    ):

        if pose_type not in ["vipe", "game_gt_normed"]:
            raise ValueError(f"only pose_type in ['vipe', 'game_gt_normed'] is supported, got: {pose_type}")

        if pose_path is None:
            raise ValueError("pose_path is required: TA2V/VA2V training needs camera parameters")

        data = np.load(pose_path, allow_pickle=True)


        intrinsic_is_default = False
        if "intrinsics" in data:
            intrinsics_all = data["intrinsics"]
        elif "intrinsic" in data:
            intrinsics_all = data["intrinsic"]
        elif "K" in data:
            intrinsics_all = data["K"]
        elif self.fallback_default_intrinsic:
            intrinsics_all = _DEFAULT_NORM_INTRINSIC
            intrinsic_is_default = True
        else:
            raise KeyError(f"pose npz has no intrinsics/intrinsic/K key: {pose_path}")

        if frame_ids is not None and intrinsics_all.ndim == 3:
            intrinsic = intrinsics_all[frame_ids][0]
        elif intrinsics_all.ndim == 3:
            intrinsic = intrinsics_all[0]
        else:
            intrinsic = intrinsics_all

        if intrinsic_is_default:
            normalized_intrinsic = intrinsic
        elif pose_type == "vipe":
            normalized_intrinsic = normalize_intrinsic(intrinsic, width=1280.0, height=720.0)
        elif pose_type == "game_gt_normed":
            normalized_intrinsic = intrinsic


        if "cam_c2w" in data:
            cam_c2w_all = data["cam_c2w"]
        elif "extrinsic" in data:
            cam_c2w_all = data["extrinsic"]
        elif "data" in data:
            cam_c2w_all = data["data"]
        else:
            raise KeyError(f"pose npz has no cam_c2w/extrinsic/data key: {pose_path}")
        if frame_ids is not None:
            cam_c2w = cam_c2w_all[frame_ids]
        else:
            cam_c2w = cam_c2w_all


        normalized_cam_c2w = normalize_cam_c2w(cam_c2w, normalize_translation=normalize_translation)

        return normalized_intrinsic, normalized_cam_c2w

    def load_camera_parameters_lingbot(
        self,
        pose_path: str,
        frame_ids: list,
        h_org: int = None,
        w_org: int = None,
        h_target: int = 480,
        w_target: int = 832,
        pose_type: str = "vipe",
    ):

        data = np.load(pose_path, allow_pickle=True)


        if "intrinsics" in data:
            intrinsics_all = data["intrinsics"]
        elif "intrinsic" in data:
            intrinsics_all = data["intrinsic"]
        elif "K" in data:
            intrinsics_all = data["K"]
        elif self.fallback_default_intrinsic:
            intrinsics_all = _DEFAULT_NORM_INTRINSIC
        else:
            raise KeyError(f"pose npz has no intrinsics/intrinsic/K key: {pose_path}")
        intrinsic = intrinsics_all[0] if intrinsics_all.ndim == 3 else intrinsics_all


        if "cam_c2w" in data:
            cam_c2w_all = data["cam_c2w"]
        elif "extrinsic" in data:
            cam_c2w_all = data["extrinsic"]
        elif "data" in data:
            cam_c2w_all = data["data"]
        else:
            raise KeyError(f"pose npz has no cam_c2w/extrinsic/data key: {pose_path}")
        if frame_ids is not None and len(frame_ids) > 0:

            valid_ids = [min(fid, cam_c2w_all.shape[0] - 1) for fid in frame_ids]
            cam_c2w = cam_c2w_all[valid_ids]
        else:
            cam_c2w = cam_c2w_all


        relative_c2ws = torch.from_numpy(cam_c2w).float()


        h_final, w_final = resolve_intrinsic_source_resolution(intrinsic, h_org, w_org, tag=pose_path)


        Ks_flat = transform_intrinsic_for_crop_resize(
            intrinsic, h_final, w_final, h_target, w_target)

        return Ks_flat, relative_c2ws

    def process_input_for_astra_v1(self, cam_data_seq):

        f_num = (len(cam_data_seq) - 1) // 4 + 1

        all_keyframe_indices = []
        for compressed_idx in range(0, f_num-1):
            all_keyframe_indices.append(compressed_idx * 4 + 1)


        relative_cams = [torch.zeros_like(torch.as_tensor(cam_data_seq[0][:3, :]))]
        for idx in all_keyframe_indices:
            cam_prev = cam_data_seq[0]
            cam_next = cam_data_seq[idx]
            relative_cam = compute_relative_pose(cam_prev, cam_next)
            relative_cams.append(torch.as_tensor(relative_cam[:3, :]))

        pose_embedding = torch.stack(relative_cams, dim=0)
        pose_embedding = rearrange(pose_embedding, 'b c d -> b (c d)')
        pose_embedding = pose_embedding.to(torch.bfloat16)


        mask = torch.ones(f_num, 1, dtype=torch.float32)
        mask[1:] = 0
        mask = mask.view(-1, 1)
        camera_embedding = torch.cat([pose_embedding, mask], dim=1)
        return camera_embedding.to(torch.bfloat16)

    def process_input_for_astra_v2(self, cam_data_seq):


        L = len(cam_data_seq)
        f_num = (L - 1) // 4 + 1


        def rel12(idx: int) -> torch.Tensor:
            cam_prev = cam_data_seq[0]
            cam_next = cam_data_seq[idx]
            rel = compute_relative_pose(cam_prev, cam_next)
            rel_3x4 = torch.as_tensor(rel[:3, :])
            return rel_3x4.reshape(-1)

        tokens = []
        for group_i in range(f_num):
            if group_i == 0:

                base12 = torch.zeros_like(torch.as_tensor(cam_data_seq[0][:3, :])).reshape(-1)
                four = torch.cat([base12, base12, base12, base12], dim=0)
            else:


                start = (group_i - 1) * 4 + 1
                idxs = [start + j for j in range(4)]

                idxs = [min(i, L - 1) for i in idxs]

                emb_list = [rel12(i) for i in idxs]
                four = torch.cat(emb_list, dim=0)

            tokens.append(four)

        pose_embedding = torch.stack(tokens, dim=0)


        mask = torch.zeros(f_num, 1, dtype=torch.float32, device=pose_embedding.device)
        mask[0, 0] = 1.0

        camera_embedding = torch.cat([pose_embedding.to(torch.bfloat16), mask.to(torch.bfloat16)], dim=1)
        return camera_embedding

    def process_input_for_hunyuan_camctrl(self, cam_c2w, intrinsic, tps=False,
                                          auto_move_thresh=True, move_thresh_factor=0.05, move_thresh_min=1e-6,move_thresh_eps=1e-12):


        num_frames = len(cam_c2w)
        latent_num = (num_frames - 1) // 4 + 1


        latent_frame_indices = [0]
        for compressed_idx in range(0, latent_num-1):
            latent_frame_indices.append(compressed_idx * 4 + 1)


        w2c_list = []
        intrinsic_list = []

        for i in latent_frame_indices:
            c2w = cam_c2w[i]
            w2c = np.linalg.inv(c2w)
            w2c_list.append(w2c)

            norm_intrinsic = intrinsic.copy()
            if norm_intrinsic[0, 2] > 1.0:
                norm_intrinsic[0, 0] /= norm_intrinsic[0, 2] * 2
                norm_intrinsic[1, 1] /= norm_intrinsic[1, 2] * 2
                norm_intrinsic[0, 2] = 0.5
                norm_intrinsic[1, 2] = 0.5
            intrinsic_list.append(norm_intrinsic)

        w2c_list = np.array(w2c_list)
        intrinsic_list = np.array(intrinsic_list)


        c2ws = np.linalg.inv(w2c_list)


        relative_c2w = np.zeros_like(c2ws)
        relative_c2w[0, ...] = c2ws[0, ...]

        if len(c2ws) > 1:
            C_inv = np.linalg.inv(c2ws[:-1])
            relative_c2w[1:, ...] = np.einsum('nij,njk->nik', C_inv, c2ws[1:, ...])


        trans_one_hot = np.zeros((latent_num, 4), dtype=np.int32)
        rotate_one_hot = np.zeros((latent_num, 4), dtype=np.int32)


        if auto_move_thresh and latent_num > 1:
            step_norms = np.linalg.norm(relative_c2w[1:, :3, 3], axis=1)
            nonzero = step_norms[step_norms > move_thresh_eps]

            if nonzero.size > 0:
                median_step = np.median(nonzero)
                move_norm_valid = max(move_thresh_min, move_thresh_factor * median_step)
            else:
                move_norm_valid = move_thresh_min
        else:
            move_norm_valid = 0.0001

        for i in range(1, latent_num):

            move_dirs = relative_c2w[i, :3, 3]
            move_norms = np.linalg.norm(move_dirs)

            if move_norms > move_norm_valid:
                move_norm_dirs = move_dirs / move_norms
                angles_rad = np.arccos(np.clip(move_norm_dirs, -1.0, 1.0))
                trans_angles_deg = np.degrees(angles_rad)
            else:
                trans_angles_deg = np.zeros(3)


            R_rel = relative_c2w[i, :3, :3]
            r = R.from_matrix(R_rel)
            rot_angles_deg = r.as_euler("xyz", degrees=True)


            if move_norms > move_norm_valid:

                if (not tps) or (
                    tps and abs(rot_angles_deg[1]) < 5e-2 and abs(rot_angles_deg[0]) < 5e-2
                ):

                    if trans_angles_deg[2] < 60:
                        trans_one_hot[i, 0] = 1
                    elif trans_angles_deg[2] > 120:
                        trans_one_hot[i, 1] = 1


                    if trans_angles_deg[0] < 60:
                        trans_one_hot[i, 2] = 1
                    elif trans_angles_deg[0] > 120:
                        trans_one_hot[i, 3] = 1


            if rot_angles_deg[1] > 5e-2:
                rotate_one_hot[i, 0] = 1
            elif rot_angles_deg[1] < -5e-2:
                rotate_one_hot[i, 1] = 1


            if rot_angles_deg[0] > 5e-2:
                rotate_one_hot[i, 2] = 1
            elif rot_angles_deg[0] < -5e-2:
                rotate_one_hot[i, 3] = 1


        trans_one_hot = torch.tensor(trans_one_hot)
        rotate_one_hot = torch.tensor(rotate_one_hot)

        trans_one_label = hunyuan_one_hot_to_one_dimension(trans_one_hot)
        rotate_one_label = hunyuan_one_hot_to_one_dimension(rotate_one_hot)


        action_one_label = trans_one_label * 9 + rotate_one_label

        return (
            torch.as_tensor(w2c_list, dtype=torch.float32),
            torch.as_tensor(intrinsic_list, dtype=torch.float32),
            action_one_label
        )

    def __getitem__(self, data_id):
        if self.load_from_cache:
            data = self.cached_data[data_id % len(self.cached_data)]
            data = self.cached_data_operator(data)
        else:
            data = self.data[data_id % len(self.data)].copy()


            _ev_src = None
            if self.skill_source and self.event_window_keys is not None:
                _es = data.get(self.event_window_keys.get("start"))
                _ee = data.get(self.event_window_keys.get("end"))
                if _es is not None and _ee is not None:
                    _ev_src = (int(_es), int(_ee))
                    if self.video_loader is not None:
                        self.video_loader.event_hint = _ev_src
            for key in self.data_file_keys:
                if key in data:
                    if key == "pose":

                        if self.pose_dir is not None:
                            pose_filename = os.path.basename(data[key])
                            loadpath = os.path.join(self.pose_dir, pose_filename)
                            if not os.path.exists(loadpath):
                                loadpath = os.path.join(self.pose_dir, data[key])
                                if not os.path.exists(loadpath):
                                    loadpath = os.path.join(self.base_path, data[key])
                        else:
                            loadpath = os.path.join(self.base_path, data[key])
                        pose_type = data.get("pose_type", "vipe")

                        data["intrinsic"], data["cam_c2w"] = self.load_camera_parameters(loadpath, pose_type=pose_type)
                        data["cam_emb_astra"] = self.process_input_for_astra_v2(data["cam_c2w"])
                        data["hy_viewmats"], data["hy_Ks"], data["hy_action"] = self.process_input_for_hunyuan_camctrl(data["cam_c2w"], data["intrinsic"])

                        frame_indices = data.get("__frame_indices__")
                        if self._target_h is not None and self._target_w is not None:
                            data["lingbot_Ks"], data["lingbot_c2ws"] = \
                                self.load_camera_parameters_lingbot(
                                    loadpath,
                                    frame_ids=frame_indices,
                                    h_org=self._source_h,
                                    w_org=self._source_w,
                                    h_target=self._target_h,
                                    w_target=self._target_w,
                                    pose_type=pose_type,
                                )
                    elif key in self.special_operator_map:
                        data[key] = self.special_operator_map[key](data[key])
                    elif key in self.data_file_keys:
                        result = self.main_data_operator(data[key])

                        if key == "video" and isinstance(result, tuple):
                            if len(result) == 4:
                                data[key] = result[0]
                                data["__start_time__"] = result[1]
                                data["__frame_indices__"] = result[2]
                                data["first_frame"] = result[3]
                            elif len(result) == 3:
                                data[key] = result[0]
                                data["__start_time__"] = result[1]
                                data["__frame_indices__"] = result[2]
                            elif len(result) == 2:
                                data[key] = result[0]
                                data["__start_time__"] = result[1]
                            else:
                                data[key] = result
                        else:
                            data[key] = result

            if "segment_prompts" in data and isinstance(data["segment_prompts"], str):
                try:
                    data["segment_prompts"] = json.loads(data["segment_prompts"])
                except (json.JSONDecodeError, TypeError):
                    data["segment_prompts"] = None


            if self.skill_source:
                data["__skill__"] = True
                fidx = data.get("__frame_indices__")
                if _ev_src is not None and fidx:
                    _es, _ee = _ev_src
                    hits = [i for i, sf in enumerate(fidx) if _es <= sf <= _ee]
                    data["__event_window__"] = (min(hits), max(hits)) if hits else None
                else:
                    data["__event_window__"] = None

                if self.video_loader is not None:
                    self.video_loader.event_hint = None
        return data

    def __len__(self):
        if self.load_from_cache:
            return len(self.cached_data) * self.repeat
        else:
            return len(self.data) * self.repeat

    def check_data_equal(self, data1, data2):

        if len(data1) != len(data2):
            return False
        for k in data1:
            if data1[k] != data2[k]:
                return False
        return True
