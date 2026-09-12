import os
import json
import random
import re
from .operators import DataProcessingOperator


_CAMERA_BLOCK_RE = re.compile(r"<camera>.*?</camera>", re.DOTALL | re.IGNORECASE)
_CAMERA_BARE_RE = re.compile(r"</?camera>", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")


def strip_camera_tags_text(text: str) -> str:

    if not isinstance(text, str):
        return text
    t = _CAMERA_BLOCK_RE.sub(" ", text)
    t = _CAMERA_BARE_RE.sub(" ", t)
    return _WS_RE.sub(" ", t).strip()


class ResolveCaptionFile(DataProcessingOperator):


    INTERLEAVE_PREFIXES = ("interleave_caption",)


    SEGMENTS_LIST_KEYS = ("segments", "merged_segments")

    def __init__(self, caption_dir, caption_key="overall_caption", target_fps=24, inline=False,
                 segment_text_key="full_prompt", strip_camera_tags=False):
        self.caption_dir = caption_dir
        self.target_fps = target_fps


        self.segment_text_key = segment_text_key
        self.strip_camera_tags = bool(strip_camera_tags)


        self.inline = inline


        if isinstance(caption_key, str):
            self.caption_keys = {caption_key: 1.0}
        elif isinstance(caption_key, dict):
            self.caption_keys = caption_key
        else:
            self.caption_keys = {"overall_caption": 1.0}

        self._keys = list(self.caption_keys.keys())
        self._weights = list(self.caption_keys.values())

    def _is_interleave(self, key):
        return any(key.startswith(p) for p in self.INTERLEAVE_PREFIXES)

    def _is_segments_list(self, key):
        return key in self.SEGMENTS_LIST_KEYS

    def _parse_segments_list(self, segments_data, path=""):


        segments = []
        for seg in sorted(segments_data, key=lambda x: float(x["time_range_s"][0])):
            if self.segment_text_key not in seg:
                raise ValueError(
                    f"segment is missing segment_text_key '{self.segment_text_key}' (no silent fallback). "
                    f"path={path}, segment_index={seg.get('segment_index')}, "
                    f"fields present in this segment={sorted(seg.keys())}. Fix: 1) change segment_text_key in "
                    f"the data yaml, or 2) regenerate the caption json so the field exists."
                )
            _p = seg[self.segment_text_key]
            if self.strip_camera_tags:
                _p = strip_camera_tags_text(_p)
            if not isinstance(_p, str) or not _p.strip():

                continue
            segments.append({
                "start_frame": int(float(seg["time_range_s"][0]) * self.target_fps),
                "end_frame": int(float(seg["time_range_s"][1]) * self.target_fps),
                "prompt": _p,
            })
        return segments

    def _parse_interleave(self, interleave_data):

        segments = []
        for seg_key in sorted(interleave_data.keys()):
            seg = interleave_data[seg_key]
            _p = seg["caption"]
            if self.strip_camera_tags:
                _p = strip_camera_tags_text(_p)
                if not isinstance(_p, str) or not _p.strip():
                    continue
            segments.append({
                "start_frame": int(seg["start_time"] * self.target_fps),
                "end_frame": int(seg["end_time"] * self.target_fps),
                "prompt": _p,
            })
        return segments

    @staticmethod
    def _lookup(cap, dotted_key):


        cur = cap
        for part in dotted_key.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return None, False
            cur = cur[part]
        return cur, True

    def __call__(self, caption_value):

        if self.inline:
            if not isinstance(caption_value, str) or not caption_value.strip():
                raise ValueError(f"inline caption must be a non-empty string, got: {caption_value!r}")
            return {"text": caption_value}


        path = os.path.join(self.caption_dir, caption_value)
        if not os.path.exists(path):
            path = os.path.join(self.caption_dir, os.path.basename(caption_value))
        with open(path, "r", encoding="utf-8") as f:
            cap = json.load(f)


        chosen_key = random.choices(self._keys, weights=self._weights)[0]


        value, found = self._lookup(cap, chosen_key)
        if not found:
            raise ValueError(
                f"caption_key '{chosen_key}' is missing from the caption JSON; no silent fallback. "
                f"path={path}, top_keys={sorted(list(cap.keys()))}. "
                f"configured caption_key={self._keys} (dotted nesting is supported, e.g. 'overall.description'). "
                f"Fix: 1) check the spelling / nesting of caption_key, 2) regenerate the caption json so the "
                f"field exists, or 3) use path_replace to point the prompt path at a directory that has it."
            )

        result = {}

        if self._is_segments_list(chosen_key):

            if not isinstance(value, list) or not value:
                raise ValueError(
                    f"caption_key '{chosen_key}' must be a non-empty list (segmented schema), "
                    f"path={path}, value type={type(value).__name__}"
                )
            result["segment_prompts"] = self._parse_segments_list(value, path=path)
            if not result["segment_prompts"]:
                raise ValueError(
                    f"caption_key '{chosen_key}' parsed to no usable segments: every segment's "
                    f"'{self.segment_text_key}' is empty, or became empty after stripping camera tags. path={path}"
                )
            result["text"] = result["segment_prompts"][0]["prompt"]
        elif self._is_interleave(chosen_key):

            interleave_data = value
            if not isinstance(interleave_data, dict) or not interleave_data:
                raise ValueError(
                    f"caption_key '{chosen_key}' is of interleave type, but the JSON field is not a non-empty dict. "
                    f"path={path}, value type={type(interleave_data).__name__}"
                )
            result["segment_prompts"] = self._parse_interleave(interleave_data)
            if not result["segment_prompts"]:
                raise ValueError(
                    f"caption_key '{chosen_key}' parsed to an empty segment list. path={path}"
                )

            result["text"] = result["segment_prompts"][0]["prompt"]
        else:

            text = value
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"caption_key '{chosen_key}' is not a non-empty string. path={path}, value={text!r}"
                )
            if self.strip_camera_tags:
                text = strip_camera_tags_text(text)
                if not text:
                    raise ValueError(
                        f"caption_key '{chosen_key}' became empty after stripping camera tags. path={path}"
                    )
            result["text"] = text

        return result
