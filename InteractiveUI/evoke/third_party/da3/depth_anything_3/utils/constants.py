

DEFAULT_MODEL = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
DEFAULT_EXPORT_DIR = "workspace/gallery/scene"
DEFAULT_GALLERY_DIR = "workspace/gallery"
DEFAULT_GRADIO_DIR = "workspace/gradio"
THRESH_FOR_REF_SELECTION = 3


DEFAULT_EVAL_WORKSPACE = "workspace/evaluation"


EVAL_REF_VIEW_STRATEGY = "first"


DTU_EVAL_DATA_ROOT = "workspace/benchmark_dataset/dtu"


DTU_SCENES = [
    "scan1",
    "scan4",
    "scan9",
    "scan10",
    "scan11",
    "scan12",
    "scan13",
    "scan15",
    "scan23",
    "scan24",
    "scan29",
    "scan32",
    "scan33",
    "scan34",
    "scan48",
    "scan49",
    "scan62",
    "scan75",
    "scan77",
    "scan110",
    "scan114",
    "scan118",
]


DTU_DIST_THRESH = 0.2
DTU_NUM_CONSIST = 4
DTU_MAX_POINTS = 4_000_000


DTU_DOWN_DENSE = 0.2
DTU_PATCH_SIZE = 60
DTU_MAX_DIST = 20


DTU64_EVAL_DATA_ROOT = "workspace/benchmark_dataset/dtu64"
DTU64_CAMERA_ROOT = "workspace/benchmark_dataset/dtu64/Cameras"


DTU64_SCENES = [
    "scan105",
    "scan114",
    "scan118",
    "scan122",
    "scan24",
    "scan37",
    "scan40",
    "scan55",
    "scan63",
    "scan65",
    "scan69",
    "scan83",
    "scan97",
]


ETH3D_EVAL_DATA_ROOT = "workspace/benchmark_dataset/eth3d"


ETH3D_SCENES = [
    "courtyard",
    "electro",
    "kicker",
    "pipes",
    "relief",

    "delivery_area",
    "facade",

    "office",
    "playground",
    "relief_2",
    "terrains",
]


ETH3D_FILTER_KEYS = {
    "delivery_area": ["711.JPG", "712.JPG", "713.JPG", "714.JPG"],
    "electro": ["9289.JPG", "9290.JPG", "9291.JPG", "9292.JPG", "9293.JPG", "9298.JPG"],
    "playground": ["587.JPG", "588.JPG", "589.JPG", "590.JPG", "591.JPG", "592.JPG"],
    "relief": [
        "427.JPG", "428.JPG", "429.JPG", "430.JPG", "431.JPG", "432.JPG",
        "433.JPG", "434.JPG", "435.JPG", "436.JPG", "437.JPG", "438.JPG",
    ],
    "relief_2": [
        "458.JPG", "459.JPG", "460.JPG", "461.JPG", "462.JPG", "463.JPG",
        "464.JPG", "465.JPG", "466.JPG", "467.JPG", "468.JPG",
    ],
}


ETH3D_VOXEL_LENGTH = 4.0 / 512.0 * 5
ETH3D_SDF_TRUNC = 0.04 * 5
ETH3D_MAX_DEPTH = 100000.0


ETH3D_SAMPLING_NUMBER = 1_000_000


ETH3D_EVAL_THRESHOLD = 0.05 * 5
ETH3D_DOWN_SAMPLE = 4.0 / 512.0 * 5


SEVENSCENES_EVAL_DATA_ROOT = "workspace/benchmark_dataset/7scenes"


SEVENSCENES_SCENES = [
    "chess",
    "fire",
    "heads",
    "office",
    "pumpkin",
    "redkitchen",
    "stairs",
]


SEVENSCENES_FX = 585.0
SEVENSCENES_FY = 585.0
SEVENSCENES_CX = 320.0
SEVENSCENES_CY = 240.0


SEVENSCENES_VOXEL_LENGTH = 4.0 / 512.0
SEVENSCENES_SDF_TRUNC = 0.04
SEVENSCENES_MAX_DEPTH = 1000000.0


SEVENSCENES_SAMPLING_NUMBER = 1_000_000


SEVENSCENES_EVAL_THRESHOLD = 0.05
SEVENSCENES_DOWN_SAMPLE = 4.0 / 512.0


SCANNETPP_EVAL_DATA_ROOT = "workspace/benchmark_dataset/scannetpp"


SCANNETPP_SCENES = [
    "09c1414f1b",
    "1ada7a0617",
    "40aec5fffa",
    "3e8bba0176",
    "acd95847c5",
    "578511c8a9",
    "5f99900f09",
    "c4c04e6d6c",
    "f3d64c30f8",
    "7bc286c1b6",
    "c5439f4607",
    "286b55a2bf",
    "fb5a96b1a2",
    "7831862f02",
    "38d58a7a31",
    "bde1e479ad",
    "9071e139d9",
    "21d970d8de",
    "bcd2436daf",
    "cc5237fd77",
]


SCANNETPP_INPUT_H = 768
SCANNETPP_INPUT_W = 1024


SCANNETPP_VOXEL_LENGTH = 0.02
SCANNETPP_SDF_TRUNC = 0.15
SCANNETPP_MAX_DEPTH = 5.0


SCANNETPP_SAMPLING_NUMBER = 1_000_000


SCANNETPP_EVAL_THRESHOLD = 0.05
SCANNETPP_DOWN_SAMPLE = 0.02


HIROOM_EVAL_DATA_ROOT = "workspace/benchmark_dataset/hiroom/data"
HIROOM_GT_ROOT_PATH = "workspace/benchmark_dataset/hiroom/fused_pcd"
HIROOM_SCENE_LIST_PATH = "workspace/benchmark_dataset/hiroom/selected_scene_list_val.txt"


HIROOM_VOXEL_LENGTH = 4.0 / 512.0
HIROOM_SDF_TRUNC = 0.04
HIROOM_MAX_DEPTH = 10000.0


HIROOM_SAMPLING_NUMBER = 1_000_000


HIROOM_EVAL_THRESHOLD = 0.05
HIROOM_DOWN_SAMPLE = 4.0 / 512.0
