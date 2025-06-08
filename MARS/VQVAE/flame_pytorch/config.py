from dataclasses import dataclass, field

@dataclass
class VertexArguments:
    flame_model_path: str = field(default="flame_pytorch/flame_models/generic_model.pkl", metadata={"help": "FLAME model path", "tyro_name": "flame-model-path"})
    static_landmark_embedding_path: str = field(default="flame_pytorch/flame_models/flame_static_embedding.pkl", metadata={"help": "Static landmark embeddings path for FLAME", "tyro_name": "static-landmark-embedding-path"})
    dynamic_landmark_embedding_path: str = field(default="flame_pytorch/flame_models/flame_dynamic_embedding.npy", metadata={"help": "Dynamic contour embedding path for FLAME", "tyro_name": "dynamic-landmark-embedding-path"})
    shape_params: int = field(default=100, metadata={"help": "The number of shape parameters", "tyro_name": "shape-params"})
    expression_params: int = field(default=50, metadata={"help": "The number of expression parameters", "tyro_name": "expression-params"})
    pose_params: int = field(default=6, metadata={"help": "The number of pose parameters", "tyro_name": "pose-params"})
    use_face_contour: bool = field(default=True, metadata={"help": "If true, apply landmark loss also on the face contour.", "tyro_name": "use-face-contour"})
    use_3D_translation: bool = field(default=True, metadata={"help": "If true, apply landmark loss on 3D translation.", "tyro_name": "use-3D-translation"})
    optimize_eyeballpose: bool = field(default=True, metadata={"help": "If true, optimize for eyeball pose.", "tyro_name": "optimize-eyeballpose"})
    optimize_neckpose: bool = field(default=True, metadata={"help": "If true, optimize for neck pose.", "tyro_name": "optimize-neckpose"})
    num_worker: int = field(default=4, metadata={"help": "Number of PyTorch workers.", "tyro_name": "num-worker"})
    batch_size: int = field(default=1, metadata={"help": "Training batch size.", "tyro_name": "batch-size"})
    ring_margin: float = field(default=0.5, metadata={"help": "Ring margin.", "tyro_name": "ring-margin"})
    ring_loss_weight: float = field(default=1.0, metadata={"help": "Weight on ring loss.", "tyro_name": "ring-loss-weight"})
    output_video_dir: str = field(default="/home/MIR_LAB/EMOTE/rend", metadata={"help": "Output video directory", "tyro_name": "output-video-dir"})
    input_param_path: str = field(default="/home/MIR_LAB/EMOTE/flame_param", metadata={"help": "Path for parameters", "tyro_name": "input-param-path"})
    with_shape: bool = field(default=False, metadata={"help": "Whether to use shape.", "tyro_name": "with-shape"})
    with_shape_each: bool = field(default=False, metadata={"help": "Whether to use shape for each frame.", "tyro_name": "with-shape-each"})
    with_jaw: bool = field(default=False, metadata={"help": "Whether to use jaw.", "tyro_name": "with-jaw"})
    with_global_rot: bool = field(default=False, metadata={"help": "Whether to use global rotation.", "tyro_name": "with-global-rot"})
    with_neck: bool = field(default=False, metadata={"help": "Whether to use neck.", "tyro_name": "with-neck"})
    shape_params: int = field(default=100, metadata={"help": "The number of shape parameters.", "tyro_name": "shape-params"})
    
    
    use_savgol_filter: bool = field(default=False, metadata={"help": "Whether to use Savitzky-Golay filter.", "tyro_name": "use-savgol-filter"})
    use_bilateral_filter: bool = field(default=False, metadata={"help": "Whether to use bilateral filter.", "tyro_name": "use-bilateral-filter"})
    audio_path: str = field(default='', metadata={"help": "Audio path.", "tyro_name": "audio-path"})
    output_image_path: str = field(default='', metadata={"help": "Image output path.", "tyro_name": "output-image-path"})
    output_vertices_path: str = field(default='', metadata={"help": "Vertices output path.", "tyro_name": "output-vertices-path"})
    original_video_path: str = field(default='', metadata={"help": "Original video path.", "tyro_name": "original-video-path"})
    video_width: int = field(default=800, metadata={"help": "Width of the video.", "tyro_name": "width"})
    video_height: int = field(default=800, metadata={"help": "Height of the video.", "tyro_name": "height"})
    fps: int = field(default=25, metadata={"help": "Frames per second of the video.", "tyro_name": "fps"})
    data_type: str = field(default='', metadata={"help": "Data type.", "tyro_name": "data-type"})
    users: bool = field(default=False, metadata={"help": "User flag.", "tyro_name": "users"})
    only_mesh: bool = field(default=False, metadata={"help": "Whether to use only mesh.", "tyro_name": "only-mesh"})

def get_config() -> VertexArguments:
    return VertexArguments()
