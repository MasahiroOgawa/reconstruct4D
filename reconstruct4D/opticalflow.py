import inspect
import sys
import os
import cv2
import numpy as np
import torch
from ext.unimatch import utils
from ext.unimatch.utils import frame_utils
from ext.unimatch.utils import flow_viz
from PIL import Image


class UnimatchFlow:
    """
    compute optical flow using unimatch algorithm
    """

    def __init__(self, FLOW_RESULT_DIR) -> None:
        self.FLOW_RESULT_DIR = FLOW_RESULT_DIR

    def compute(self, imgname):
        """
        compute optical flow from 2 consecutive images.
        currently just read flow from files.
        args:
            imgname: image file name. e.g. 00000.jpg
        result:
            self.flow: size = h x w x 2. 2 means flow vector (u,v).
            self.flow_img: size = h x w x 3. 3 means RGB channel which represents flow orientation.
        """
        imgnum = imgname.split(".")[0]
        flow_file = os.path.join(self.FLOW_RESULT_DIR, f"{imgnum}_pred.flo")
        if flow_file is None:
            raise ValueError(f"flow file {flow_file} does not exist.")
        self.flow = utils.frame_utils.readFlow(flow_file)

        self.flow_img = utils.flow_viz.flow_to_image(self.flow)


class MemFlow:
    """
    Compute optical flow using MemFlow algorithm
    """

    def __init__(self, model_name="MemFlowNet", stage="things", weights_path=None):
        """
        Initialize MemFlow model
        args:
            model_name: MemFlowNet or MemFlowNet_T
            stage: things, sintel, kitti, spring
            weights_path: path to pretrained weights
        """
        # Add memflow to path
        memflow_path = os.path.join(os.path.dirname(__file__), 'ext/memflow')
        if memflow_path not in sys.path:
            sys.path.append(memflow_path)
            sys.path.append(os.path.join(memflow_path, 'core'))

        # Import MemFlow components
        if stage == 'things':
            from configs.things_memflownet import get_cfg
        elif stage == 'sintel':
            from configs.sintel_memflownet import get_cfg
        elif stage == 'kitti':
            from configs.kitti_memflownet import get_cfg
        elif stage == 'spring':
            from configs.spring_memflownet import get_cfg
        else:
            raise ValueError(f"Unknown stage: {stage}")

        from core.Networks import build_network
        from utils.utils import InputPadder
        from utils import flow_viz as memflow_viz
        from inference import inference_core_skflow as inference_core

        # Store imports for later use
        self.InputPadder = InputPadder
        self.memflow_viz = memflow_viz
        self.inference_core = inference_core

        # Get configuration
        self.cfg = get_cfg()
        self.cfg.restore_ckpt = weights_path

        # Check if CUDA is available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create model
        print(f"Creating {model_name} model...")
        self.model = build_network(self.cfg).to(self.device)

        # Load pretrained weights if available
        if weights_path and os.path.exists(weights_path):
            print(f"Loading pretrained model from {weights_path}")
            checkpoint = torch.load(weights_path, map_location=self.device)

            # Handle different checkpoint formats
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint

            # Remove 'module.' prefix if present
            if 'module' in list(state_dict.keys())[0]:
                new_state_dict = {}
                for key in state_dict.keys():
                    new_state_dict[key.replace('module.', '', 1)] = state_dict[key]
                state_dict = new_state_dict

            self.model.load_state_dict(state_dict, strict=False)
            print("MemFlow model loaded successfully!")
        else:
            print(f"Warning: Pretrained model not found at {weights_path}")

        self.model.eval()

        # Create processor for inference
        self.processor = self.inference_core.InferenceCore(self.model, config=self.cfg)

        # Initialize flow storage
        self.flow = None
        self.flow_img = None

    def compute_from_images(self, img1_path, img2_path):
        """
        Compute optical flow from two image paths
        args:
            img1_path: path to first image
            img2_path: path to second image
        result:
            self.flow: size = h x w x 2. 2 means flow vector (u,v).
            self.flow_img: size = h x w x 3. 3 means RGB channel which represents flow orientation.
        """
        # Load images
        img1 = np.array(Image.open(img1_path)).astype(np.uint8)
        img2 = np.array(Image.open(img2_path)).astype(np.uint8)

        # Convert to tensors
        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()

        # Stack images
        images = torch.stack([img1, img2]).to(self.device)

        # Pad images for inference
        padder = self.InputPadder(images.shape)
        images = padder.pad(images)

        # Normalize images
        images = 2 * (images / 255.0) - 1.0

        # Run inference
        with torch.no_grad():
            # Process image pair (batch_size=1, time=2)
            images_batch = images.unsqueeze(0)  # Add batch dimension
            flow_low, flow_pred = self.processor.step(images_batch, end=True, add_pe=False)
            self.flow = padder.unpad(flow_pred[0]).cpu().numpy()  # Remove padding and convert to numpy

        # Convert flow to h x w x 2 format (transpose from 2 x h x w)
        self.flow = np.transpose(self.flow, (1, 2, 0))

        # Generate flow visualization
        self.flow_img = self.memflow_viz.flow_to_image(self.flow)

        return self.flow, self.flow_img

    def compute_from_arrays(self, img1_array, img2_array):
        """
        Compute optical flow from two numpy arrays
        args:
            img1_array: first image as numpy array (h x w x 3)
            img2_array: second image as numpy array (h x w x 3)
        result:
            self.flow: size = h x w x 2. 2 means flow vector (u,v).
            self.flow_img: size = h x w x 3. 3 means RGB channel which represents flow orientation.
        """
        # Convert to tensors
        img1 = torch.from_numpy(img1_array).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2_array).permute(2, 0, 1).float()

        # Stack images
        images = torch.stack([img1, img2]).to(self.device)

        # Pad images for inference
        padder = self.InputPadder(images.shape)
        images = padder.pad(images)

        # Normalize images
        images = 2 * (images / 255.0) - 1.0

        # Run inference
        with torch.no_grad():
            # Process image pair (batch_size=1, time=2)
            images_batch = images.unsqueeze(0)  # Add batch dimension
            flow_low, flow_pred = self.processor.step(images_batch, end=True, add_pe=False)
            self.flow = padder.unpad(flow_pred[0]).cpu().numpy()  # Remove padding and convert to numpy

        # Convert flow to h x w x 2 format (transpose from 2 x h x w)
        self.flow = np.transpose(self.flow, (1, 2, 0))

        # Generate flow visualization
        self.flow_img = self.memflow_viz.flow_to_image(self.flow)

        return self.flow, self.flow_img


class UndominantFlowAngleExtractor:
    def __init__(
        self, THRE_FLOWLENGTH=4.0, THRE_ANGLE=10 * np.pi / 180, LOG_LEVEL=0
    ) -> None:
        # constants
        # if flow length is lower than this value, the flow is ignored.
        self.THRE_FLOWLENGTH = THRE_FLOWLENGTH
        self.LOG_LEVEL = LOG_LEVEL
        self.THRE_ANGLE = THRE_ANGLE  # radian

    def compute(self, flow: np.ndarray, nonsky_static_mask: np.ndarray):
        """
        compute undominant orientation mask from optical flow.
        args:
            flow: size = h x w x 2. 2 means flow vector (u,v).
        result:
            self.undominant_flow_prob: size = h x w.
        """
        # compute flow angle and length
        flow_angle = np.arctan2(flow[:, :, 1], flow[:, :, 0])
        flow_length = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)

        # extract median angle
        median_angle = np.median(flow_angle[nonsky_static_mask])

        # compute mask from median angle.
        self.undominant_flow_prob = np.zeros(
            (flow.shape[0], flow.shape[1]), dtype=np.float16
        )

        self.undominant_flow_prob[
            (flow_length > self.THRE_FLOWLENGTH)
            & (np.abs(flow_angle - median_angle) > self.THRE_ANGLE)
        ] = 0.9  # 0.9 means outlier
        self.undominant_flow_prob[
            (flow_length > self.THRE_FLOWLENGTH)
            & (np.abs(flow_angle - median_angle) <= self.THRE_ANGLE)
        ] = 0.1  # 0.1 means inlier