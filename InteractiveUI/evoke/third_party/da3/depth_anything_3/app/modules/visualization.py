

import os
from typing import Any, Dict, List, Optional, Tuple
import cv2
import gradio as gr
import numpy as np


class VisualizationHandler:


    def __init__(self):
        pass

    def update_view_selectors(
        self, processed_data: Optional[Dict[int, Dict[str, Any]]]
    ) -> Tuple[gr.Dropdown, gr.Dropdown]:


        if processed_data is None or len(processed_data) == 0:
            choices = ["View 1"]
        else:
            num_views = len(processed_data)
            choices = [f"View {i + 1}" for i in range(num_views)]

        return (
            gr.Dropdown(choices=choices, value=choices[0]),
            gr.Dropdown(choices=choices, value=choices[0]),
        )

    def get_view_data_by_index(
        self, processed_data: Optional[Dict[int, Dict[str, Any]]], view_index: int
    ) -> Optional[Dict[str, Any]]:


        if processed_data is None or len(processed_data) == 0:
            return None

        view_keys = list(processed_data.keys())
        if view_index < 0 or view_index >= len(view_keys):
            view_index = 0

        return processed_data[view_keys[view_index]]

    def update_depth_view(
        self, processed_data: Optional[Dict[int, Dict[str, Any]]], view_index: int
    ) -> Optional[str]:


        view_data = self.get_view_data_by_index(processed_data, view_index)
        if view_data is None or view_data.get("depth_image") is None:
            return None


        return view_data["depth_image"]

    def navigate_depth_view(
        self,
        processed_data: Optional[Dict[int, Dict[str, Any]]],
        current_selector_value: str,
        direction: int,
    ) -> Tuple[str, Optional[str]]:


        if processed_data is None or len(processed_data) == 0:
            return "View 1", None


        try:
            current_view = int(current_selector_value.split()[1]) - 1
        except:
            current_view = 0

        num_views = len(processed_data)
        new_view = (current_view + direction) % num_views

        new_selector_value = f"View {new_view + 1}"
        depth_vis = self.update_depth_view(processed_data, new_view)

        return new_selector_value, depth_vis

    def update_measure_view(
        self, processed_data: Optional[Dict[int, Dict[str, Any]]], view_index: int
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List]:


        view_data = self.get_view_data_by_index(processed_data, view_index)
        if view_data is None:
            return None, None, []


        if "image" in view_data and view_data["image"] is not None:
            image = view_data["image"].copy()
        else:
            return None, None, []


        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)


        depth_image_path = view_data.get("depth_image", None)
        depth_right_half = None

        if depth_image_path and os.path.exists(depth_image_path):
            try:

                depth_combined = cv2.imread(depth_image_path)
                depth_combined = cv2.cvtColor(depth_combined, cv2.COLOR_BGR2RGB)
                if depth_combined is not None:
                    height, width = depth_combined.shape[:2]

                    depth_right_half = depth_combined[:, width // 2 :]
            except Exception as e:
                print(f"Error extracting depth right half: {e}")

        return image, depth_right_half, []

    def navigate_measure_view(
        self,
        processed_data: Optional[Dict[int, Dict[str, Any]]],
        current_selector_value: str,
        direction: int,
    ) -> Tuple[str, Optional[np.ndarray], Optional[str], List]:


        if processed_data is None or len(processed_data) == 0:
            return "View 1", None, None, []


        try:
            current_view = int(current_selector_value.split()[1]) - 1
        except:
            current_view = 0

        num_views = len(processed_data)
        new_view = (current_view + direction) % num_views

        new_selector_value = f"View {new_view + 1}"
        measure_image, depth_right_half, measure_points = self.update_measure_view(
            processed_data, new_view
        )

        return new_selector_value, measure_image, depth_right_half, measure_points

    def populate_visualization_tabs(
        self, processed_data: Optional[Dict[int, Dict[str, Any]]]
    ) -> Tuple[Optional[str], Optional[np.ndarray], Optional[str], List]:


        if processed_data is None or len(processed_data) == 0:
            return None, None, None, []


        depth_vis = self.update_depth_view(processed_data, 0)
        measure_img, depth_right_half, _ = self.update_measure_view(processed_data, 0)

        return depth_vis, measure_img, depth_right_half, []

    def reset_measure(
        self, processed_data: Optional[Dict[int, Dict[str, Any]]]
    ) -> Tuple[Optional[np.ndarray], List, str]:


        if processed_data is None or len(processed_data) == 0:
            return None, [], ""


        first_view = list(processed_data.values())[0]
        return first_view["image"], [], ""

    def measure(
        self,
        processed_data: Optional[Dict[int, Dict[str, Any]]],
        measure_points: List,
        current_view_selector: str,
        event: gr.SelectData,
    ) -> List:


        try:
            print(f"Measure function called with selector: {current_view_selector}")

            if processed_data is None or len(processed_data) == 0:
                return [None, [], "No data available"]


            try:
                current_view_index = int(current_view_selector.split()[1]) - 1
            except:
                current_view_index = 0

            print(f"Using view index: {current_view_index}")


            if current_view_index < 0 or current_view_index >= len(processed_data):
                current_view_index = 0

            view_keys = list(processed_data.keys())
            current_view = processed_data[view_keys[current_view_index]]

            if current_view is None:
                return [None, [], "No view data available"]

            point2d = event.index[0], event.index[1]
            print(f"Clicked point: {point2d}")

            measure_points.append(point2d)


            image, depth_right_half, _ = self.update_measure_view(
                processed_data, current_view_index
            )
            if image is None:
                return [None, [], "No image available"]

            image = image.copy()


            try:
                if image.dtype != np.uint8:
                    if image.max() <= 1.0:

                        image = (image * 255).astype(np.uint8)
                    else:

                        image = image.astype(np.uint8)
            except Exception as e:
                print(f"Image conversion error: {e}")
                return [None, [], f"Image conversion error: {e}"]


            try:
                for p in measure_points:
                    if 0 <= p[0] < image.shape[1] and 0 <= p[1] < image.shape[0]:
                        image = cv2.circle(image, p, radius=5, color=(255, 0, 0), thickness=2)
            except Exception as e:
                print(f"Drawing error: {e}")
                return [None, [], f"Drawing error: {e}"]


            depth_text = ""
            try:
                for i, p in enumerate(measure_points):
                    if (
                        current_view["depth"] is not None
                        and 0 <= p[1] < current_view["depth"].shape[0]
                        and 0 <= p[0] < current_view["depth"].shape[1]
                    ):
                        d = current_view["depth"][p[1], p[0]]
                        depth_text += f"- **P{i + 1} depth: {d:.2f}m**\n"
                    else:
                        depth_text += f"- **P{i + 1}: Click position ({p[0]}, {p[1]}) - No depth information**\n"
            except Exception as e:
                print(f"Depth text error: {e}")
                depth_text = f"Error computing depth: {e}\n"

            if len(measure_points) == 2:
                try:
                    point1, point2 = measure_points

                    if (
                        0 <= point1[0] < image.shape[1]
                        and 0 <= point1[1] < image.shape[0]
                        and 0 <= point2[0] < image.shape[1]
                        and 0 <= point2[1] < image.shape[0]
                    ):
                        image = cv2.line(image, point1, point2, color=(255, 0, 0), thickness=2)


                    distance_text = "- **Distance: Unable to calculate 3D distance**"
                    if (
                        current_view["depth"] is not None
                        and 0 <= point1[1] < current_view["depth"].shape[0]
                        and 0 <= point1[0] < current_view["depth"].shape[1]
                        and 0 <= point2[1] < current_view["depth"].shape[0]
                        and 0 <= point2[0] < current_view["depth"].shape[1]
                    ):
                        try:

                            d1 = current_view["depth"][point1[1], point1[0]]
                            d2 = current_view["depth"][point2[1], point2[0]]


                            if current_view["intrinsics"] is not None:

                                K = current_view["intrinsics"]
                                fx, fy = K[0, 0], K[1, 1]
                                cx, cy = K[0, 2], K[1, 2]


                                u1, v1 = point1[0], point1[1]
                                x1 = (u1 - cx) * d1 / fx
                                y1 = (v1 - cy) * d1 / fy
                                z1 = d1


                                u2, v2 = point2[0], point2[1]
                                x2 = (u2 - cx) * d2 / fx
                                y2 = (v2 - cy) * d2 / fy
                                z2 = d2


                                p1_3d = np.array([x1, y1, z1])
                                p2_3d = np.array([x2, y2, z2])
                                distance_3d = np.linalg.norm(p1_3d - p2_3d)

                                distance_text = f"- **Distance: {distance_3d:.2f}m**"
                            else:

                                pixel_distance = np.sqrt(
                                    (point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2
                                )
                                avg_depth = (d1 + d2) / 2
                                scale_factor = avg_depth / 1000
                                estimated_3d_distance = pixel_distance * scale_factor
                                distance_text = f"- **Distance: {estimated_3d_distance:.2f}m (estimated, no intrinsics)**"

                        except Exception as e:
                            print(f"Distance computation error: {e}")
                            distance_text = f"- **Distance computation error: {e}**"

                    measure_points = []
                    text = depth_text + distance_text
                    print(f"Measurement complete: {text}")
                    return [image, depth_right_half, measure_points, text]
                except Exception as e:
                    print(f"Final measurement error: {e}")
                    return [None, [], f"Measurement error: {e}"]
            else:
                print(f"Single point measurement: {depth_text}")
                return [image, depth_right_half, measure_points, depth_text]

        except Exception as e:
            print(f"Overall measure function error: {e}")
            return [None, [], f"Measure function error: {e}"]
