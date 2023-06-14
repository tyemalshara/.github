
<a name="readme-top"></a>

<!-- PROJECT LOGO -->
<div align="center">
  <h1 align="center">State-of-the-art Computer Vision as ready-to-use algorithms</h1>
</div>
<p align="center">
    <a href="https://github.com/Ikomia-hub">
        <img alt="Stars" src="https://img.shields.io/github/stars/Ikomia-hub?style=social">
    </a>
    <a href="https://ikomia.com/en/computer-vision-api/">
        <img alt="Website" src="https://img.shields.io/website/http/ikomia.com/en.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/Ikomia-hub">
        <img alt="Python" src="https://img.shields.io/badge/os-win%2C%20linux-9cf">
    </a>
     <br>
    <a href="https://discord.com/invite/82Tnw9UGGc">
        <img alt="Discord community" src="https://img.shields.io/badge/Discord-white?style=social&logo=discord">
    </a> 
    <a href="https://www.linkedin.com/company/ikomia">
        <img src="https://img.shields.io/badge/LinkedIn-white?logo=linkedin&style=social" alt="linkedin community">
    </a> 
    <a href="https://twitter.com/IkomiaOfficial">
        <img alt="Twitter Follow" src="https://img.shields.io/twitter/follow/IkomiaOfficial?style=social">
    </a>  <a href="https://www.youtube.com/channel/UC0nIasJy6f-b-f0SOEsBlQw">
        <img alt="YouTube Channel Subscribers" src="https://img.shields.io/youtube/channel/subscribers/UC0nIasJy6f-b-f0SOEsBlQw?style=social">
    </a>
</p>
<p align="center">
    <a href="https://github.com/Ikomia-hub">
        <img alt="Ecological futuristic city in the sky" src="https://user-images.githubusercontent.com/42171814/202030473-de4f4498-2ce8-4bfe-96f9-3baa3caabf4e.jpg">
    </a>
</p>

# Introduction

At Ikomia, we deeply believe that sharing scientific knowledge is the key to success, that's why we make research-based algorithms ready-to-use for developers. 

The main goal of Ikomia is to take existing Python code and wrap it as ready-to-use algorithm for [Ikomia API](https://github.com/Ikomia-dev/IkomiaApi) (our Python library) and [Ikomia STUDIO](https://github.com/Ikomia-dev/IkomiaStudio) (our desktop software). With this approach, we can easily integrate individual repos from researchers or labs and also awesome frameworks like [OpenCV](https://github.com/opencv/opencv), [Detectron2](https://github.com/facebookresearch/detectron2), [OpenMMLab](https://github.com/open-mmlab) or [Hugging Face](https://github.com/huggingface/transformers) so that developers can benefit from the best state-of-the-art algorithms in a single framework.



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#dataset-loader">Dataset loader</a></li>
    <li><a href="#classification">Classification</a></li>
    <li><a href="#object-detection">Object detection</a></li>
    <li><a href="#instance-segmentation">Instance segmentation</a></li>
    <li><a href="#semantic-segmentation">Semantic segmentation</a></li>
    <li><a href="#panoptic-segmentation">Panoptic segmentation</a></li>
    <li><a href="#pose-estimation">Pose estimation</a></li>
    <li><a href="#text-detection">Text detection</a></li>
    <li><a href="#text-recognition">Text recognition</a></li>
    <li><a href="#background-matting">Background matting</a></li>
    <li><a href="#style-transfer">Style transfer</a></li>
    <li><a href="#colorization">Colorization</a></li>
    <li><a href="#inpainting">Inpainting</a></li>
    <li><a href="#tracking">Tracking</a></li>
    <li><a href="#optical-flow">Optical flow</a></li>
    <li><a href="#action-recognition">Action recognition</a></li>
    <li><a href="#emotion-recognition">Emotion recognition</a></li>
    <li><a href="#scikit">Scikit</a></li>
  </ol>
</details>


## Dataset loader

| Name | Language | Description | Source code |
| --- | --- | --- | --- |
| [dataset_coco](https://github.com/Ikomia-hub/dataset_coco) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Load COCO 2017 dataset  | Made by Ikomia |
| [dataset_cwfid](https://github.com/Ikomia-hub/dataset_cwfid) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Load Crop/Weed Field Image Dataset (CWFID) for semantic segmentation  | Made by Ikomia |
| [dataset_pascal_voc](https://github.com/Ikomia-hub/dataset_pascal_voc) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Load PascalVOC dataset  | Made by Ikomia |
| [dataset_via](https://github.com/Ikomia-hub/dataset_via) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Load VGG Image Annotator dataset  | Made by Ikomia |
| [dataset_wgisd](https://github.com/Ikomia-hub/dataset_wgisd) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Load Wine Grape Instance Segmentation Dataset (WGISD) | Made by Ikomia |
| [dataset_wildreceipt](https://github.com/Ikomia-hub/dataset_wildreceipt) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Load the WildReceipt dataset (OCR) | Made by Ikomia |
| [dataset_yolo](https://github.com/Ikomia-hub/dataset_yolo) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Loader for datasets in YOLO format | Made by Ikomia |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Classification

| Name | Language | Description | Source code |
| --- | --- | --- | --- |
| [infer_covidnet](https://github.com/Ikomia-hub/infer_covidnet) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | A tailored Deep Convolutional Neural Network Design for detection of COVID-19 cases from chest radiography images | [lindawangg/COVID-Net](https://github.com/lindawangg/COVID-Net) |
| [infer_inception_v3](https://github.com/Ikomia-hub/infer_inception_v3) | ![C++](https://img.shields.io/badge/c++-%2300599C.svg?style=for-the-badge&logo=c%2B%2B&logoColor=white) | Deep neural network classifier trained on ImageNet dataset | Made with OpenCV |
| [train_timm_image_classification](https://github.com/Ikomia-hub/train_timm_image_classification) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Train timm image classification models | [rwightman/pytorch-image-models](https://github.com/rwightman/pytorch-image-models) |
| [infer_timm_image_classification](https://github.com/Ikomia-hub/infer_timm_image_classification) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Infer timm image classification models | [rwightman/pytorch-image-models](https://github.com/rwightman/pytorch-image-models) |
| [train_torchvision_mnasnet](https://github.com/Ikomia-hub/train_torchvision_mnasnet) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Training process for MnasNet convolutional network | [pytorch/vision](https://github.com/pytorch/vision) |
| [infer_torchvision_mnasnet](https://github.com/Ikomia-hub/infer_torchvision_mnasnet) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | MnasNet inference model for image classification | [pytorch/vision](https://github.com/pytorch/vision) |
| [train_torchvision_resnet](https://github.com/Ikomia-hub/train_torchvision_resnet) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Training process for ResNet convolutional network | [pytorch/vision](https://github.com/pytorch/vision) |
| [infer_torchvision_resnet](https://github.com/Ikomia-hub/infer_torchvision_resnet) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | ResNet inference model for image classification | [pytorch/vision](https://github.com/pytorch/vision) |
| [train_torchvision_resnext](https://github.com/Ikomia-hub/train_torchvision_resnext) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Training process for ResNeXt convolutional network | [pytorch/vision](https://github.com/pytorch/vision) |
| [infer_torchvision_resnext](https://github.com/Ikomia-hub/infer_torchvision_resnext) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | ResNeXt inference model for image classification | [pytorch/vision](https://github.com/pytorch/vision) |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Object detection

| Name | Language | Description | Source code |
| --- | --- | --- | --- |
| [infer_face_detection_kornia](https://github.com/Ikomia-hub/infer_face_detection_kornia) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Face detection using kornia | [kornia/kornia](https://github.com/kornia/kornia) |
| [infer_face_detector](https://github.com/Ikomia-hub/infer_face_detector) | ![C++](https://img.shields.io/badge/c++-%2300599C.svg?style=for-the-badge&logo=c%2B%2B&logoColor=white) | Deep learning based face detector | Made with OpenCV |
| [infer_detectron2_detection](https://github.com/Ikomia-hub/infer_detectron2_detection) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Inference for Detectron2 detection models | [facebookresearch/detectron2](https://github.com/facebookresearch/detectron2) |
| [infer_detectron2_retinanet](https://github.com/Ikomia-hub/infer_detectron2_retinanet) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | RetinaNet inference model of Detectron2 for object detection | [facebookresearch/detectron2](https://github.com/facebookresearch/detectron2) |
| [infer_detectron2_tridentnet](https://github.com/Ikomia-hub/infer_detectron2_tridentnet) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | TridentNet inference model of Detectron2 for object detection | [facebookresearch/detectron2](https://github.com/facebookresearch/detectron2/tree/main/projects/TridentNet) |
| [train_mmlab_detection](https://github.com/Ikomia-hub/train_mmlab_detection) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Train MMDET Detection models | [open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection) |
| [infer_mmlab_detection](https://github.com/Ikomia-hub/infer_mmlab_detection) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Inference for MMDET Detection models | [open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection) |
| [infer_mobilenet_ssd](https://github.com/Ikomia-hub/infer_mobilenet_ssd) | ![C++](https://img.shields.io/badge/c++-%2300599C.svg?style=for-the-badge&logo=c%2B%2B&logoColor=white) | Single Shot Detector (SSD) for mobile and embedded vision applications | Made with OpenCV |
| [train_torchvision_faster_rcnn](https://github.com/Ikomia-hub/train_torchvision_faster_rcnn) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Training process for Faster R-CNN convolutional network | [pytorch/vision](https://github.com/pytorch/vision) |
| [infer_torchvision_faster_rcnn](https://github.com/Ikomia-hub/infer_torchvision_faster_rcnn) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Faster R-CNN inference model for object detection | [pytorch/vision](https://github.com/pytorch/vision) |
| [train_yolo](https://github.com/Ikomia-hub/train_yolo) | ![C++](https://img.shields.io/badge/c++-%2300599C.svg?style=for-the-badge&logo=c%2B%2B&logoColor=white) | Train YOLO neural network with darknet framework | [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet) |
| [infer_yolo_v3](https://github.com/Ikomia-hub/infer_yolo_v3) | ![C++](https://img.shields.io/badge/c++-%2300599C.svg?style=for-the-badge&logo=c%2B%2B&logoColor=white) | Object detection using YOLO V3 neural network | [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet) |
| [infer_yolo_v4](https://github.com/Ikomia-hub/infer_yolo_v4) | ![C++](https://img.shields.io/badge/c++-%2300599C.svg?style=for-the-badge&logo=c%2B%2B&logoColor=white) | Object detection using YOLO V4 neural network | [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet) |
| [train_yolo_v5](https://github.com/Ikomia-hub/train_yolo_v5) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Train Ultralytics YoloV5 object detection models | [ultralytics/yolov5](https://github.com/ultralytics/yolov5) |
| [infer_yolo_v5](https://github.com/Ikomia-hub/infer_yolo_v5) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Ultralytics YoloV5 object detection models | [ultralytics/yolov5](https://github.com/ultralytics/yolov5) |
| [train_yolo_v7](https://github.com/Ikomia-hub/train_yolo_v7) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Train YOLOv7 object detection models | [WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7) |
| [infer_yolo_v7](https://github.com/Ikomia-hub/infer_yolo_v7) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | YOLOv7 object detection inference | [WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7) |
| [train_yolor](https://github.com/Ikomia-hub/train_yolor) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Train YoloR object detection models | [WongKinYiu/yolor](https://github.com/WongKinYiu/yolor) |
| [infer_yolor](https://github.com/Ikomia-hub/infer_yolor) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Inference for YoloR object detection models | [WongKinYiu/yolor](https://github.com/WongKinYiu/yolor) |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Instance segmentation

| Name | Language | Description | Source code |
| --- | --- | --- | --- |
| [infer_detectron2_instance_segmentation](https://github.com/Ikomia-hub/infer_detectron2_instance_segmentation) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Infer Detectron2 instance segmentation models | [facebookresearch/detectron2](https://github.com/facebookresearch/detectron2) |
| [infer_detectron2_pointrend](https://github.com/Ikomia-hub/infer_detectron2_pointrend) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | PointRend inference model of Detectron2 for instance segmentation | [facebookresearch/detectron2](https://github.com/facebookresearch/detectron2/tree/main/projects/PointRend) |
| [infer_mask_rcnn](https://github.com/Ikomia-hub/infer_mask_rcnn) | ![C++](https://img.shields.io/badge/c++-%2300599C.svg?style=for-the-badge&logo=c%2B%2B&logoColor=white) | Semantic segmentation based on Faster R-CNN method  | Made with OpenCV |
| [train_torchvision_mask_rcnn](https://github.com/Ikomia-hub/train_torchvision_mask_rcnn) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Training process for Mask R-CNN convolutional network | [pytorch/vision](https://github.com/pytorch/vision) |
| [infer_torchvision_mask_rcnn](https://github.com/Ikomia-hub/infer_torchvision_mask_rcnn) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Mask R-CNN inference model for object detection and segmentation | [pytorch/vision](https://github.com/pytorch/vision) |
| [infer_yolact](https://github.com/Ikomia-hub/infer_yolact) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | A simple, fully convolutional model for real-time instance segmentation | [dbolya/yolact](https://github.com/dbolya/yolact) |
| [train_yolo_v7_instance_segmentation](https://github.com/Ikomia-hub/train_yolo_v7_instance_segmentation) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Train YOLOv7 instance segmentation models | [WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7/tree/u7/seg) |
| [infer_yolo_v7_instance_segmentation](https://github.com/Ikomia-hub/infer_yolo_v7_instance_segmentation) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | YOLOv7 instance segmentation inference | [WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7/tree/u7/seg) |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Semantic segmentation

| Name | Language | Description | Source code |
| --- | --- | --- | --- |
| [train_detectron2_deeplabv3plus](https://github.com/Ikomia-hub/train_detectron2_deeplabv3plus) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Training process for DeepLabv3+ model of Detectron2 | [facebookresearch/detectron2](https://github.com/facebookresearch/detectron2/tree/main/projects/DeepLab) |
| [infer_detectron2_deeplabv3plus](https://github.com/Ikomia-hub/infer_detectron2_deeplabv3plus) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | DeepLabv3+ inference model of Detectron2 for semantic segmentation | [facebookresearch/detectron2](https://github.com/facebookresearch/detectron2/tree/main/projects/DeepLab) |
| [train_transunet](https://github.com/Ikomia-hub/train_transunet) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Training process for TransUNet model | [Beckschen/TransUNet](https://github.com/Beckschen/TransUNet) |
| [infer_transunet](https://github.com/Ikomia-hub/infer_transunet) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | TransUNet inference for semantic segmentation | [Beckschen/TransUNet](https://github.com/Beckschen/TransUNet) |
| [infer_hugginface_semantic_segmentation](https://github.com/Ikomia-hub/infer_huggingface_semantic_segmentation) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Infer semantic segmentation models from HuggingFace | [Hugging Face](https://huggingface.co/docs/transformers/tasks/semantic_segmentation) |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Panoptic segmentation

| Name | Language | Description | Source code |
| --- | --- | --- | --- |
| [infer_detectron2_panoptic_segmentation](https://github.com/Ikomia-hub/infer_detectron2_panoptic_segmentation) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Infer Detectron2 panoptic segmentation models | [facebookresearch/detectron2](https://github.com/facebookresearch/detectron2) |
| [infer_yolop_v2](https://github.com/Ikomia-hub/infer_yolop_v2) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Infer YOLOPv2 models for Panoptic driving Perception | [CAIC-AD/YOLOPv2](https://github.com/CAIC-AD/YOLOPv2) |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Pose estimation

| Name | Language | Description | Source code |
| --- | --- | --- | --- |
| [infer_detectron2_densepose](https://github.com/Ikomia-hub/infer_detectron2_densepose) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Detectron2 inference model for human pose detection | [facebookresearch/detectron2](https://github.com/facebookresearch/detectron2/tree/main/projects/DensePose)
| [infer_detectron2_keypoints](https://github.com/Ikomia-hub/infer_detectron2_keypoints) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Inference for Detectron2 keypoint models | [facebookresearch/detectron2](https://github.com/facebookresearch/detectron2) |
| [infer_facemark_lbf](https://github.com/Ikomia-hub/infer_facemark_lbf) | ![C++](https://img.shields.io/badge/c++-%2300599C.svg?style=for-the-badge&logo=c%2B%2B&logoColor=white) | Facial landmark detection using Local Binary Features (LBF) | Made with OpenCV |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Text detection

| Name | Language | Description | Source code |
| --- | --- | --- | --- |
| [train_mmlab_text_detection](https://github.com/Ikomia-hub/train_mmlab_text_detection) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Training process for MMOCR Text Detection | [open-mmlab/mmocr](https://github.com/open-mmlab/mmocr) |
| [infer_mmlab_text_detection](https://github.com/Ikomia-hub/infer_mmlab_text_detection) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Inference for MMOCR Text Detection | [open-mmlab/mmocr](https://github.com/open-mmlab/mmocr) |
| [infer_text_detector_east](https://github.com/Ikomia-hub/infer_text_detector_east) | ![C++](https://img.shields.io/badge/c++-%2300599C.svg?style=for-the-badge&logo=c%2B%2B&logoColor=white) | Fast and accurate text detection in natural scenes using single neural network | Made with OpenCV |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Text recognition

| Name | Language | Description | Source code |
| --- | --- | --- | --- |
| [train_mmlab_text_recognition](https://github.com/Ikomia-hub/train_mmlab_text_recognition) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Inference for MMOCR Text Recognition   | [open-mmlab/mmocr](https://github.com/open-mmlab/mmocr) |
| [infer_mmlab_text_recognition](https://github.com/Ikomia-hub/infer_mmlab_text_recognition) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Inference for MMOCR Text Recognition   | [open-mmlab/mmocr](https://github.com/open-mmlab/mmocr) |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Background matting

| Name | Language | Description | Source code |
| --- | --- | --- | --- |
| [infer_background_matting](https://github.com/Ikomia-hub/infer_background_matting) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Real-Time High-Resolution Background Matting | [PeterL1n/BackgroundMattingV2](https://github.com/PeterL1n/BackgroundMattingV2) |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Style transfer

| Name | Language | Description | Source code |
| --- | --- | --- | --- |
| [infer_neural_style_transfer](https://github.com/Ikomia-hub/infer_neural_style_transfer) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Neural network method to paint given image in the style of the reference image | [PyImageSearch](https://pyimagesearch.com/2018/08/27/neural-style-transfer-with-opencv/) |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Colorization

| Name | Language | Description | Source code |
| --- | --- | --- | --- |
| [infer_colorful_image_colorization](https://github.com/Ikomia-hub/infer_colorful_image_colorization) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Automatic colorization of grayscale image based on neural network | [richzhang/colorization](https://github.com/richzhang/colorization) |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Inpainting

| Name | Language | Description | Source code |
| --- | --- | --- | --- |
| [infer_lama](https://github.com/Ikomia-hub/infer_lama) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Resolution-robust Large Mask Inpainting with Fourier Convolutions | [saic-mdal/lama](https://github.com/saic-mdal/lama) |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Tracking

| Name | Language | Description | Source code |
| --- | --- | --- | --- |
| [infer_deepsort](https://github.com/Ikomia-hub/infer_deepsort) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Simple Online and Realtime Tracking with a deep association metric | [nwojke/deep_sort](https://github.com/nwojke/deep_sort) |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Optical flow

| Name | Language | Description | Source code |
| --- | --- | --- | --- |
| [infer_raft_optical_flow](https://github.com/Ikomia-hub/infer_raft_optical_flow) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Estimate the optical flow from a video using a RAFT model | [princeton-vl/RAFT](https://github.com/princeton-vl/RAFT) |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Action recognition

| Name | Language | Description | Source code |
| --- | --- | --- | --- |
| [infer_resnet_action_recognition](https://github.com/Ikomia-hub/infer_resnet_action_recognition) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Human action recognition with spatio-temporal 3D CNNs | [kenshohara/3D-ResNets-PyTorch](https://github.com/kenshohara/3D-ResNets-PyTorch) |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Emotion recognition

| Name | Language | Description | Source code |
| --- | --- | --- | --- |
| [infer_emotion_fer_plus](https://github.com/Ikomia-hub/infer_emotion_fer_plus) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Facial emotion recognition using DNN trained from crowd-sourced label distribution | [microsoft/FERPlus](https://github.com/microsoft/FERPlus) |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Scikit

| Name | Language | Description | Source code |
| --- | --- | --- | --- |
| [skimage_morpho_snakes](https://github.com/Ikomia-hub/skimage_morpho_snakes) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Morphological active contour segmentation from scikit-image library | [scikit-image/scikit-image](https://github.com/scikit-image/scikit-image) |
| [skimage_rolling_ball](https://github.com/Ikomia-hub/skimage_rolling_ball) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | The rolling-ball algorithm estimates the background intensity of a grayscale image in case of uneven exposure | [scikit-image/scikit-image](https://github.com/scikit-image/scikit-image) |
| [skimage_threshold](https://github.com/Ikomia-hub/skimage_threshold) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Compilation of well-known thresholding methods from scikit-image library | [scikit-image/scikit-image](https://github.com/scikit-image/scikit-image) |

<p align="right">(<a href="#readme-top">back to top</a>)</p>
