

import torch


def _create(name, pretrained=True, channels=3, classes=80, autoshape=True, verbose=True, device=None):
   
    Loads a custom or local YOLOv5 model from a given path with optional autoshaping and device specification.

    Args:
        path (str): Path to the custom model file (e.g., 'path/to/model.pt').
        autoshape (bool): Apply YOLOv5 .autoshape() wrapper to model if True, enabling compatibility with various input
            types (default is True).
        _verbose (bool): If True, prints all informational messages to the screen; otherwise, operates silently
            (default is True).
        device (str | torch.device | None): Device to load the model on, e.g., 'cpu', 'cuda', torch.device('cuda:0'), etc.
            (default is None, which automatically selects the best available device).

    Returns:
        torch.nn.Module: A YOLOv5 model loaded with the specified parameters.

    Notes:
        For more details on loading models from PyTorch Hub:
        https://docs.ultralytics.com/yolov5/tutorials/pytorch_hub_model_loading

    Examples:
        ```python
        # Load model from a given path with autoshape enabled on the best available device
        model = torch.hub.load('ultralytics/yolov5', 'custom', 'yolov5s.pt')

        # Load model from a local path without autoshape on the CPU device
        model = torch.hub.load('.', 'custom', 'yolov5s.pt', source='local', autoshape=False, device='cpu')
        ```
 
    Instantiates the YOLOv5-nano model with options for pretraining, input channels, class count, autoshaping,
    verbosity, and device.

    Args:
        pretrained (bool): If True, loads pretrained weights into the model. Defaults to True.
        channels (int): Number of input channels for the model. Defaults to 3.
        classes (int): Number of classes for object detection. Defaults to 80.
        autoshape (bool): If True, applies the YOLOv5 .autoshape() wrapper to the model for various formats (file/URI/PIL/
            cv2/np) and non-maximum suppression (NMS) during inference. Defaults to True.
        _verbose (bool): If True, prints detailed information to the screen. Defaults to True.
        device (str | torch.device | None): Specifies the device to use for model computation. If None, uses the best device
            available (i.e., GPU if available, otherwise CPU). Defaults to None.

    Returns:
        DetectionModel | ClassificationModel | SegmentationModel: The instantiated YOLOv5-nano model, potentially with
            pretrained weights and autoshaping applied.

    Notes:
        For further details on loading models from PyTorch Hub, refer to [PyTorch Hub models](https://pytorch.org/hub/
        ultralytics_yolov5).

    Examples:
        ```python
        import torch
        from ultralytics import yolov5n

        # Load the YOLOv5-nano model with defaults
        model = yolov5n()

        # Load the YOLOv5-nano model with a specific device
        model = yolov5n(device='cuda')
        ```

    Create a YOLOv5-small (yolov5s) model with options for pretraining, input channels, class count, autoshaping,
    verbosity, and device configuration.

    Args:
        pretrained (bool, optional): Flag to load pretrained weights into the model. Defaults to True.
        channels (int, optional): Number of input channels. Defaults to 3.
        classes (int, optional): Number of model classes. Defaults to 80.
        autoshape (bool, optional): Whether to wrap the model with YOLOv5's .autoshape() for handling various input formats.
            Defaults to True.
        _verbose (bool, optional): Flag to print detailed information regarding model loading. Defaults to True.
        device (str | torch.device | None, optional): Device to use for model computation, can be 'cpu', 'cuda', or
            torch.device instances. If None, automatically selects the best available device. Defaults to None.

    Returns:
        torch.nn.Module: The YOLOv5-small model configured and loaded according to the specified parameters.

    Example:
        ```python
        import torch

        # Load the official YOLOv5-small model with pretrained weights
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

        # Load the YOLOv5-small model from a specific branch
        model = torch.hub.load('ultralytics/yolov5:master', 'yolov5s')

        # Load a custom YOLOv5-small model from a local checkpoint
        model = torch.hub.load('ultralytics/yolov5', 'custom', 'yolov5s.pt')

        # Load a local YOLOv5-small model specifying source as local repository
        model = torch.hub.load('.', 'custom', 'yolov5s.pt', source='local')
        ```

    Notes:
        For more details on model loading and customization, visit
        the [YOLOv5 PyTorch Hub Documentation](https://pytorch.org/hub/ultralytics_yolov5/).

    Instantiates the YOLOv5-medium model with customizable pretraining, channel count, class count, autoshaping,
    verbosity, and device.

    Args:
        pretrained (bool, optional): Whether to load pretrained weights into the model. Default is True.
        channels (int, optional): Number of input channels. Default is 3.
        classes (int, optional): Number of model classes. Default is 80.
        autoshape (bool, optional): Apply YOLOv5 .autoshape() wrapper to the model for handling various input formats.
            Default is True.
        _verbose (bool, optional): Whether to print detailed information to the screen. Default is True.
        device (str | torch.device | None, optional): Device specification to use for model parameters (e.g., 'cpu', 'cuda').
            Default is None.

    Returns:
        torch.nn.Module: The instantiated YOLOv5-medium model.

    Usage Example:
        ```python
        import torch

        model = torch.hub.load('ultralytics/yolov5', 'yolov5m')  # Load YOLOv5-medium from Ultralytics repository
        model = torch.hub.load('ultralytics/yolov5:master', 'yolov5m')  # Load from the master branch
        model = torch.hub.load('ultralytics/yolov5', 'custom', 'yolov5m.pt')  # Load a custom/local YOLOv5-medium model
        model = torch.hub.load('.', 'custom', 'yolov5m.pt', source='local')  # Load from a local repository
        ```

    For more information, visit https://pytorch.org/hub/ultralytics_yolov5.
    """
    return _create("yolov5m", pretrained, channels, classes, autoshape, _verbose, device)


def yolov5l(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    """
    Creates YOLOv5-large model with options for pretraining, channels, classes, autoshaping, verbosity, and device
    selection.

    Args:
        pretrained (bool): Load pretrained weights into the model. Default is True.
        channels (int): Number of input channels. Default is 3.
        classes (int): Number of model classes. Default is 80.
        autoshape (bool): Apply YOLOv5 .autoshape() wrapper to model. Default is True.
        _verbose (bool): Print all information to screen. Default is True.
        device (str | torch.device | None): Device to use for model parameters, e.g., 'cpu', 'cuda', or a torch.device instance.
            Default is None.

    Returns:
        YOLOv5 model (torch.nn.Module): The YOLOv5-large model instantiated with specified configurations and possibly
        pretrained weights.

    Examples:
        ```python
        import torch
        model = torch.hub.load('ultralytics/yolov5', 'yolov5l')
        ```

    Notes:
        For additional details, refer to the PyTorch Hub models documentation:
        https://pytorch.org/hub/ultralytics_yolov5
 
    Perform object detection using the YOLOv5-xlarge model with options for pretraining, input channels, class count,
    autoshaping, verbosity, and device specification.

    Args:
        pretrained (bool): If True, loads pretrained weights into the model. Defaults to True.
        channels (int): Number of input channels for the model. Defaults to 3.
        classes (int): Number of model classes for object detection. Defaults to 80.
        autoshape (bool): If True, applies the YOLOv5 .autoshape() wrapper for handling different input formats. Defaults to
            True.
        _verbose (bool): If True, prints detailed information during model loading. Defaults to True.
        device (str | torch.device | None): Device specification for computing the model, e.g., 'cpu', 'cuda:0', torch.device('cuda').
            Defaults to None.

    Returns:
        torch.nn.Module: The YOLOv5-xlarge model loaded with the specified parameters, optionally with pretrained weights and
        autoshaping applied.

    Example:
        ```python
        import torch
        model = torch.hub.load('ultralytics/yolov5', 'yolov5x')
        ```

    For additional details, refer to the official YOLOv5 PyTorch Hub models documentation:
    https://pytorch.org/hub/ultralytics_yolov5
  
    # Results
    results.print()
    results.save()

