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
    <li><a href="#DATASET_LOADER<">Dataset loader</a></li><li><a href="#CLASSIFICATION">Classification</a></li>
<li><a href="#COLORIZATION">Colorization</a></li>
<li><a href="#IMAGE_CAPTIONING">Image captioning</a></li>
<li><a href="#IMAGE_GENERATION">Image generation</a></li>
<li><a href="#IMAGE_MATTING">Image matting</a></li>
<li><a href="#INPAINTING">Inpainting</a></li>
<li><a href="#INSTANCE_SEGMENTATION">Instance segmentation</a></li>
<li><a href="#KEYPOINTS_DETECTION">Keypoints detection</a></li>
<li><a href="#OBJECT_DETECTION">Object Detection</a></li>
<li><a href="#OBJECT_TRACKING">Object tracking</a></li>
<li><a href="#OCR">OCR</a></li>
<li><a href="#OPTICAL_FLOW">Optical flow</a></li>
<li><a href="#OTHER">Other</a></li>
<li><a href="#PANOPTIC_SEGMENTATION">Panoptic segmentation</a></li>
<li><a href="#SEMANTIC_SEGMENTATION">Semantic segmentation</a></li>
<li><a href="#SUPER_RESOLUTION">Super resolution</a></li>
  </ol>
</details>

<a name="DATASET_LOADER"></a>

## Dataset loader

| Name | Language | Description | Original repository |
| --- | --- | --- | --- |
| [auto_annotate](https://github.com/Ikomia-hub/auto_annotate) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Auto-annotate images with GroundingDINO and SAM models | Made by Ikomia |
| [dataset_classification](https://github.com/Ikomia-hub/dataset_classification) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Load classification dataset | Made by Ikomia |
| [dataset_coco](https://github.com/Ikomia-hub/dataset_coco) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Load COCO 2017 dataset | Made by Ikomia |
| [dataset_cwfid](https://github.com/Ikomia-hub/dataset_cwfid) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Load Crop/Weed Field Image Dataset (CWFID) for semantic segmentation | [Link](https://github.com/cwfid/dataset) |
| [dataset_pascal_voc](https://github.com/Ikomia-hub/dataset_pascal_voc) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Load PascalVOC dataset | Made by Ikomia |
| [dataset_via](https://github.com/Ikomia-hub/dataset_via) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Load VGG Image Annotator dataset | Made by Ikomia |
| [dataset_wgisd](https://github.com/Ikomia-hub/dataset_wgisd) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Load Wine Grape Instance Segmentation Dataset (WGISD) | [Link](https://github.com/thsant/wgisd) |
| [dataset_wildreceipt](https://github.com/Ikomia-hub/dataset_wildreceipt) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Load Wildreceipt dataset | Made by Ikomia |
| [dataset_yolo](https://github.com/Ikomia-hub/dataset_yolo) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Load YOLO dataset | Made by Ikomia |
<p align="right">(<a href="#readme-top">Back to top</a>)</p>



<a name=CLASSIFICATION></a>

## Classification

| Name | Language | Description | Original repository |
| --- | --- | --- | --- |
| [infer_covidnet](https://github.com/Ikomia-hub/infer_covidnet) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | A tailored Deep Convolutional Neural Network Design for detection of COVID-19 cases from chest radiography images. | [Link](https://github.com/lindawangg/COVID-Net) |
| [infer_emotion_fer_plus](https://github.com/Ikomia-hub/infer_emotion_fer_plus) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Facial emotion recognition using DNN trained from crowd-sourced label distribution. | [Link](https://github.com/microsoft/FERPlus) |
| [infer_resnet_action_recognition](https://github.com/Ikomia-hub/infer_resnet_action_recognition) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Human action recognition with spatio-temporal 3D CNNs. | [Link](https://github.com/kenshohara/3D-ResNets-PyTorch) |
| [infer_timm_image_classification](https://github.com/Ikomia-hub/infer_timm_image_classification) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Infer timm image classification models | [Link](https://github.com/rwightman/pytorch-image-models) |
| [infer_torchvision_mnasnet](https://github.com/Ikomia-hub/infer_torchvision_mnasnet) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | MnasNet inference model for image classification. | [Link](https://github.com/pytorch/vision) |
| [infer_torchvision_resnet](https://github.com/Ikomia-hub/infer_torchvision_resnet) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | ResNet inference model for image classification. | [Link](https://github.com/pytorch/vision) |
| [infer_torchvision_resnext](https://github.com/Ikomia-hub/infer_torchvision_resnext) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | ResNeXt inference model for image classification. | [Link](https://github.com/pytorch/vision) |
| [infer_yolo_v8_classification](https://github.com/ultralytics/ultralytics) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Inference with YOLOv8 image classification models | Made by Ikomia |
| [train_timm_image_classification](https://github.com/Ikomia-hub/train_timm_image_classification) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Train timm image classification models | [Link](https://github.com/rwightman/pytorch-image-models) |
| [train_torchvision_mnasnet](https://github.com/Ikomia-hub/train_torchvision_mnasnet) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Training process for MnasNet convolutional network. | [Link](https://github.com/pytorch/vision) |
| [train_torchvision_resnet](https://github.com/Ikomia-hub/train_torchvision_resnet) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Training process for ResNet convolutional network. | [Link](https://github.com/pytorch/vision) |
| [train_torchvision_resnext](https://github.com/Ikomia-hub/train_torchvision_resnext) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Training process for ResNeXt convolutional network. | [Link](https://github.com/pytorch/vision) |
| [train_yolo_v8_classification](https://github.com/ultralytics/ultralytics) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Train YOLOv8 classification models. | Made by Ikomia |
<p align="right">(<a href="#readme-top">Back to top</a>)</p>

<a name=COLORIZATION></a>

## Colorization

| Name | Language | Description | Original repository |
| --- | --- | --- | --- |
| [infer_colorful_image_colorization](https://github.com/Ikomia-hub/infer_colorful_image_colorization) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Automatic colorization of grayscale image based on neural network. | [Link](https://github.com/richzhang/colorization) |
<p align="right">(<a href="#readme-top">Back to top</a>)</p>

<a name=IMAGE_GENERATION></a>

## Image generation

| Name | Language | Description | Original repository |
| --- | --- | --- | --- |
| [infer_hf_stable_diffusion](https://github.com/Ikomia-hub/infer_hf_stable_diffusion) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Stable diffusion models from Hugging Face. | [Link](https://github.com/Stability-AI/stablediffusion) |
| [infer_kandinsky_2](https://github.com/Ikomia-hub/infer_kandinsky_2) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Kandinsky 2.2 text2image diffusion model. | [Link](https://github.com/ai-forever/Kandinsky-2) |
| [infer_kandinsky_2_controlnet_depth](https://github.com/Ikomia-hub/infer_kandinsky_2_controlnet_depth) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Kandinsky 2.2 controlnet depth diffusion model. | [Link](https://github.com/ai-forever/Kandinsky-2) |
| [infer_kandinsky_2_image_mixing](https://github.com/Ikomia-hub/infer_kandinsky_2_image_mixing) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Kandinsky 2.2 image mixing diffusion model. | [Link](https://github.com/ai-forever/Kandinsky-2) |
| [infer_kandinsky_2_img2img](https://github.com/Ikomia-hub/infer_kandinsky_2_img2img) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Kandinsky 2.2 image-to-image diffusion model. | [Link](https://github.com/ai-forever/Kandinsky-2) |
| [infer_neural_style_transfer](https://github.com/Ikomia-hub/infer_neural_style_transfer) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Neural network method to paint given image in the style of the reference image. | [Link](https://github.com/jcjohnson/fast-neural-style) |
| [infer_pulid](https://github.com/Ikomia-hub/infer_pulid) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Pure and Lightning ID customization (PuLID) is a novel tuning-free ID customization method for text-to-image generation. | [Link](https://github.com/ToTheBeginning/PuLID) |
| [infer_stable_cascade](https://github.com/Ikomia-hub/infer_stable_cascade) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Stable Cascade is a diffusion model trained to generate images given a text prompt. | [Link](https://github.com/Stability-AI/StableCascade) |
<p align="right">(<a href="#readme-top">Back to top</a>)</p>

<a name=IMAGE_MATTING></a>

## Image matting

| Name | Language | Description | Original repository |
| --- | --- | --- | --- |
| [infer_background_matting](https://github.com/Ikomia-hub/infer_background_matting) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Real-Time High-Resolution Background Matting | [Link](https://github.com/PeterL1n/BackgroundMattingV2) |
| [infer_modnet_portrait_matting](https://github.com/Ikomia-hub/infer_modnet_portrait_matting) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Inference of MODNet Portrait Matting. | [Link](https://github.com/ZHKKKe/MODNet) |
| [infer_p3m_portrait_matting](https://github.com/Ikomia-hub/infer_p3m_portrait_matting) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Inference of Privacy-Preserving Portrait Matting (P3M) | [Link](https://github.com/ViTAE-Transformer/P3M-Net) |
<p align="right">(<a href="#readme-top">Back to top</a>)</p>

<a name=INPAINTING></a>

## Inpainting

| Name | Language | Description | Original repository |
| --- | --- | --- | --- |
| [infer_face_inpainting](https://github.com/Ikomia-hub/infer_face_inpainting) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Face inpainting using Segformer for segmentation and RealVisXL for inpainting. | Made by Ikomia |
| [infer_hf_stable_diffusion_inpaint](https://github.com/Ikomia-hub/infer_hf_stable_diffusion_inpaint) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Stable diffusion inpainting models from Hugging Face. | [Link](https://github.com/Stability-AI/stablediffusion) |
| [infer_kandinsky_2_inpaint](https://github.com/Ikomia-hub/infer_kandinsky_2_inpaint) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Kandinsky 2.2 inpainting diffusion model. | [Link](https://github.com/ai-forever/Kandinsky-2) |
<p align="right">(<a href="#readme-top">Back to top</a>)</p>

<a name=INSTANCE_SEGMENTATION></a>

## Instance segmentation

| Name | Language | Description | Original repository |
| --- | --- | --- | --- |
| [infer_detectron2_instance_segmentation](https://github.com/Ikomia-hub/infer_detectron2_instance_segmentation) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Infer Detectron2 instance segmentation models | [Link](https://github.com/facebookresearch/detectron2) |
| [infer_detectron2_pointrend](https://github.com/Ikomia-hub/infer_detectron2_pointrend) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | PointRend inference model of Detectron2 for instance segmentation. | [Link](https://github.com/facebookresearch/detectron2/tree/master/projects/PointRend) |
| [infer_hf_instance_seg](https://github.com/Ikomia-hub/infer_hf_instance_seg) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Instance segmentation using models from Hugging Face. | [Link](https://github.com/huggingface/transformers) |
| [infer_sparseinst](https://github.com/Ikomia-hub/infer_sparseinst) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Infer Sparseinst instance segmentation models | [Link](https://github.com/hustvl/SparseInst) |
| [infer_torchvision_mask_rcnn](https://github.com/Ikomia-hub/infer_torchvision_mask_rcnn) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Mask R-CNN inference model for object detection and segmentation. | [Link](https://github.com/pytorch/vision) |
| [infer_yolact](https://github.com/Ikomia-hub/infer_yolact) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | A simple, fully convolutional model for real-time instance segmentation. | [Link](https://github.com/dbolya/yolact) |
| [infer_yolo_v7_instance_segmentation](https://github.com/Ikomia-hub/infer_yolo_v7_instance_segmentation) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Inference for YOLO v7 instance segmentation models | [Link](https://github.com/WongKinYiu/yolov7/tree/u7/seg) |
| [infer_yolo_v8_seg](https://github.com/Ikomia-hub/infer_yolo_v8_seg) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Inference with YOLOv8 segmentation models | [Link](https://github.com/ultralytics/ultralytics) |
| [infer_yolop_v2](https://github.com/Ikomia-hub/infer_yolop_v2) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Panoptic driving Perception using YoloPv2 | [Link](https://github.com/CAIC-AD/YOLOPv2) |
| [train_detectron2_instance_segmentation](https://github.com/Ikomia-hub/train_detectron2_instance_segmentation) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Train Detectron2 instance segmentation models | [Link](https://github.com/facebookresearch/detectron2) |
| [train_mmlab_segmentation](https://github.com/Ikomia-hub/train_mmlab_segmentation) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Train for MMLAB segmentation models | [Link](https://github.com/open-mmlab/mmsegmentation) |
| [train_sparseinst](https://github.com/Ikomia-hub/train_sparseinst) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Train Sparseinst instance segmentation models | [Link](https://github.com/hustvl/SparseInst) |
| [train_torchvision_mask_rcnn](https://github.com/Ikomia-hub/train_torchvision_mask_rcnn) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Training process for Mask R-CNN convolutional network. | [Link](https://github.com/pytorch/vision) |
| [train_yolo_v7_instance_segmentation](https://github.com/Ikomia-hub/train_yolo_v7_instance_segmentation) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Train for YOLO v7 instance segmentation models | [Link](https://github.com/WongKinYiu/yolov7/tree/u7/seg) |
| [train_yolo_v8_seg](https://github.com/Ikomia-hub/train_yolo_v8_seg) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Train YOLOv8 instance segmentation models. | [Link](https://github.com/ultralytics/ultralytics) |
<p align="right">(<a href="#readme-top">Back to top</a>)</p>

<a name=KEYPOINTS_DETECTION></a>

## Keypoints detection

| Name | Language | Description | Original repository |
| --- | --- | --- | --- |
| [infer_detectron2_densepose](https://github.com/Ikomia-hub/infer_detectron2_densepose/) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Detectron2 inference model for human pose detection. | [Link](https://github.com/facebookresearch/detectron2/tree/master/projects/DensePose) |
| [infer_detectron2_keypoints](https://github.com/Ikomia-hub/infer_detectron2_keypoints) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Inference for Detectron2 keypoint models | [Link](https://github.com/facebookresearch/detectron2) |
| [infer_mmlab_pose_estimation](https://github.com/Ikomia-hub/infer_mmlab_pose_estimation) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Inference for pose estimation models from mmpose | [Link](https://github.com/open-mmlab/mmpose) |
| [infer_yolo_v7_keypoints](https://github.com/Ikomia-hub/infer_yolo_v7_keypoints) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | YOLOv7 pose estimation models. | [Link](https://github.com/WongKinYiu/yolov7) |
| [infer_yolo_v8_pose_estimation](https://github.com/Ikomia-hub/infer_yolo_v8_pose_estimation) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Inference with YOLOv8 pose estimation models | [Link](https://github.com/ultralytics/ultralytics) |
<p align="right">(<a href="#readme-top">Back to top</a>)</p>

<a name=OBJECT_DETECTION></a>

## Object Detection

| Name | Language | Description | Original repository |
| --- | --- | --- | --- |
| [infer_detectron2_detection](https://github.com/Ikomia-hub/infer_detectron2_detection) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Inference for Detectron2 detection models | [Link](https://github.com/facebookresearch/detectron2) |
| [infer_detectron2_retinanet](https://github.com/Ikomia-hub/infer_detectron2_retinanet) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | RetinaNet inference model of Detectron2 for object detection. | [Link](https://github.com/facebookresearch/detectron2) |
| [infer_detectron2_tridentnet](https://github.com/Ikomia-hub/infer_detectron2_tridentnet) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | TridentNet inference model of Detectron2 for object detection. | [Link](https://github.com/facebookresearch/detectron2/tree/master/projects/TridentNet) |
| [infer_face_detection_kornia](https://github.com/Ikomia-hub/infer_face_detection_kornia) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Face detection using the Kornia API | [Link](https://github.com/kornia/kornia/tree/master/examples/face_detection) |
| [infer_google_vision_face_detection](https://github.com/Ikomia-hub/infer_google_vision_face_detection) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Face detection using Google cloud vision API. | [Link](https://github.com/googleapis/python-vision) |
| [infer_google_vision_landmark_detection](https://github.com/Ikomia-hub/infer_google_vision_landmark_detection) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Landmark Detection detects popular natural and human-made structures within an image. | [Link](https://github.com/googleapis/python-vision) |
| [infer_google_vision_logo_detection](https://github.com/Ikomia-hub/infer_google_vision_logo_detection) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Logo Detection detects popular product logos within an image using the Google cloud vision API. | [Link](https://github.com/googleapis/python-vision) |
| [infer_google_vision_object_localization](https://github.com/Ikomia-hub/infer_google_vision_object_localization) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | The Vision API can detect and extract multiple objects in an image with Object Localization. | [Link](https://github.com/googleapis/python-vision) |
| [infer_grounding_dino](https://github.com/Ikomia-hub/infer_grounding_dino) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Inference of the Grounding DINO model | [Link](https://github.com/IDEA-Research/GroundingDINO) |
| [infer_mmlab_detection](https://github.com/Ikomia-hub/infer_mmlab_detection) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Inference for MMDET from MMLAB detection models | [Link](https://github.com/open-mmlab/mmdetection) |
| [infer_torchvision_faster_rcnn](https://github.com/Ikomia-hub/infer_torchvision_faster_rcnn) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Faster R-CNN inference model for object detection. | [Link](https://github.com/pytorch/vision) |
| [infer_yolo_v5](https://github.com/ultralytics/yolov5) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Ultralytics YoloV5 object detection models. | Made by Ikomia |
| [infer_yolo_v7](https://github.com/Ikomia-hub/infer_yolo_v7) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | YOLOv7 object detection models. | [Link](https://github.com/WongKinYiu/yolov7) |
| [infer_yolo_v8](https://github.com/Ikomia-hub/infer_yolo_v8) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Inference with YOLOv8 models | [Link](https://github.com/ultralytics/ultralytics) |
| [infer_yolo_v9](https://github.com/Ikomia-hub/infer_yolo_v9) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Object detection with YOLOv9 models | [Link](https://github.com/WongKinYiu/yolov9) |
| [infer_yolo_v10](https://github.com/Ikomia-hub/infer_yolo_v10) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Run inference with YOLOv10 models | [Link](https://github.com/THU-MIG/yolov10) |
| [infer_yolo_world](https://github.com/Ikomia-hub/infer_yolo_world) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | YOLO-World is a real-time zero-shot object detection modelthat leverages the power of open-vocabulary learning to recognize and localize a wide range of objects in images. | [Link](https://github.com/AILab-CVC/YOLO-World) |
| [infer_yolop_v2](https://github.com/Ikomia-hub/infer_yolop_v2) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Panoptic driving Perception using YoloPv2 | [Link](https://github.com/CAIC-AD/YOLOPv2) |
| [infer_yolor](https://github.com/Ikomia-hub/infer_yolor) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Inference for YoloR object detection models | [Link](https://github.com/WongKinYiu/yolor) |
| [train_detectron2_detection](https://github.com/Ikomia-hub/train_detectron2_detection) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Train for Detectron2 detection models | [Link](https://github.com/facebookresearch/detectron2) |
| [train_mmlab_detection](https://github.com/Ikomia-hub/train_mmlab_detection) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Train for MMLAB detection models | [Link](https://github.com/open-mmlab/mmdetection) |
| [train_torchvision_faster_rcnn](https://github.com/Ikomia-hub/train_torchvision_faster_rcnn) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Training process for Faster R-CNN convolutional network. | [Link](https://github.com/pytorch/vision) |
| [train_yolo_v5](https://github.com/Ikomia-hub/train_yolo_v5) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Train Ultralytics YoloV5 object detection models. | [Link](https://github.com/ultralytics/yolov5) |
| [train_yolo_v7](https://github.com/Ikomia-hub/train_yolo_v7) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Train YOLOv7 object detection models. | [Link](https://github.com/WongKinYiu/yolov7) |
| [train_yolo_v8](https://github.com/Ikomia-hub/train_yolo_v8) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Train YOLOv8 object detection models. | [Link](https://github.com/ultralytics/ultralytics) |
| [train_yolo_v9](https://github.com/Ikomia-hub/train_yolo_v9) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Train YOLOv9 models | [Link](https://github.com/WongKinYiu/yolov9) |
| [train_yolo_v10](https://github.com/Ikomia-hub/infer_yolo_v10) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Train YOLOv10 object detection models. | [Link](https://github.com/THU-MIG/yolov10) |
| [train_yolor](https://github.com/Ikomia-hub/train_yolor) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Train YoloR object detection models | [Link](https://github.com/WongKinYiu/yolor) |
<p align="right">(<a href="#readme-top">Back to top</a>)</p>

<a name=OBJECT_TRACKING></a>

## Object tracking

| Name | Language | Description | Original repository |
| --- | --- | --- | --- |
| [infer_bytetrack](https://github.com/Ikomia-hub/infer_bytetrack) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Infer ByteTrack for object tracking | [Link](https://github.com/ifzhang/ByteTrack) |
| [infer_deepsort](https://github.com/Ikomia-hub/infer_deepsort) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Multiple Object Tracking algorithm (MOT) combining a deep association metricwith the well known SORT algorithm for better performance. | [Link](https://github.com/nwojke/deep_sort) |
<p align="right">(<a href="#readme-top">Back to top</a>)</p>

<a name=OCR></a>

## OCR

| Name | Language | Description | Original repository |
| --- | --- | --- | --- |
| [infer_google_vision_ocr](https://github.com/Ikomia-hub/infer_google_vision_ocr) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Detects and extracts text from any image. | [Link](https://github.com/googleapis/python-vision) |
| [infer_mmlab_text_detection](https://github.com/Ikomia-hub/infer_mmlab_text_detection) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Inference for MMOCR from MMLAB text detection models | [Link](https://github.com/open-mmlab/mmocr) |
| [infer_mmlab_text_recognition](https://github.com/Ikomia-hub/infer_mmlab_text_recognition) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Inference for MMOCR from MMLAB text recognition models | [Link](https://github.com/open-mmlab/mmocr) |
| [train_mmlab_kie](https://github.com/Ikomia-hub/train_mmlab_kie) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Train for MMOCR from MMLAB KIE models | [Link](https://github.com/open-mmlab/mmocr) |
| [train_mmlab_text_detection](https://github.com/Ikomia-hub/train_mmlab_text_detection) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Training process for MMOCR from MMLAB in text detection | [Link](https://github.com/open-mmlab/mmocr) |
| [train_mmlab_text_recognition](https://github.com/Ikomia-hub/train_mmlab_text_recognition) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Training process for MMOCR from MMLAB in text recognition | [Link](https://github.com/open-mmlab/mmocr) |
<p align="right">(<a href="#readme-top">Back to top</a>)</p>

<a name=OPTICAL_FLOW></a>

## Optical flow

| Name | Language | Description | Original repository |
| --- | --- | --- | --- |
| [infer_raft_optical_flow](https://github.com/Ikomia-hub/infer_raft_optical_flow) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Estimate the optical flow from a video using a RAFT model. | [Link](https://github.com/princeton-vl/RAFT) |
<p align="right">(<a href="#readme-top">Back to top</a>)</p>

<a name=OTHER></a>

## Other

| Name | Language | Description | Original repository |
| --- | --- | --- | --- |
| [infer_depth_anything](https://github.com/Ikomia-hub/infer_depth_anything) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Depth Anything is a highly practical solution for robust monocular depth estimation | [Link](https://github.com/LiheYoung/Depth-Anything) |
| [infer_google_vision_image_properties](https://github.com/Ikomia-hub/infer_google_vision_image_properties) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Image Properties feature detects general attributes of the image, such as dominant color. | [Link](https://github.com/googleapis/python-vision) |
| [infer_google_vision_label_detection](https://github.com/Ikomia-hub/infer_google_vision_label_detection) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Detect and extract information about entities in an image, across a broad group of categories. | [Link](https://github.com/googleapis/python-vision) |
| [infer_google_vision_safe_search](https://github.com/Ikomia-hub/infer_google_vision_safe_search) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Safe Search detects explicit content such as adult content or violent content within an image. | [Link](https://github.com/googleapis/python-vision) |
| [infer_google_vision_web_detection](https://github.com/Ikomia-hub/infer_google_vision_web_detection) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Web Detection detects Web references to an image. | [Link](https://github.com/googleapis/python-vision) |
<p align="right">(<a href="#readme-top">Back to top</a>)</p>

<a name=PANOPTIC_SEGMENTATION></a>

## Panoptic segmentation

| Name | Language | Description | Original repository |
| --- | --- | --- | --- |
| [infer_detectron2_panoptic_segmentation](https://github.com/facebookresearch/detectron2) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Infer Detectron2 panoptic segmentation models | Made by Ikomia |
| [infer_hf_image_seg](https://github.com/huggingface/transformers) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Panoptic segmentation using models from Hugging Face.  | Made by Ikomia |
| [infer_mmlab_segmentation](https://github.com/Ikomia-hub/infer_mmlab_segmentation) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Inference for MMLAB segmentation models | [Link](https://github.com/open-mmlab/mmsegmentation) |
| [train_mmlab_segmentation](https://github.com/Ikomia-hub/train_mmlab_segmentation) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Train for MMLAB segmentation models | [Link](https://github.com/open-mmlab/mmsegmentation) |
<p align="right">(<a href="#readme-top">Back to top</a>)</p>

<a name=SEMANTIC_SEGMENTATION></a>

## Semantic segmentation

| Name | Language | Description | Original repository |
| --- | --- | --- | --- |
| [infer_detectron2_deeplabv3plus](https://github.com/Ikomia-hub/infer_detectron2_deeplabv3plus) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | DeepLabv3+ inference model of Detectron2 for semantic segmentation. | [Link](https://github.com/facebookresearch/detectron2) |
| [infer_hf_semantic_seg](https://github.com/Ikomia-hub/infer_hf_semantic_seg) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Semantic segmentation using models from Hugging Face. | [Link](https://github.com/huggingface/transformers) |
| [infer_mmlab_segmentation](https://github.com/Ikomia-hub/infer_mmlab_segmentation) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Inference for MMLAB segmentation models | [Link](https://github.com/open-mmlab/mmsegmentation) |
| [infer_mobile_segment_anything](https://github.com/Ikomia-hub/infer_mobile_segment_anything) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Inference for Mobile Segment Anything Model (SAM). | [Link](https://github.com/ChaoningZhang/MobileSAM) |
| [infer_segment_anything](https://github.com/Ikomia-hub/infer_segment_anything) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Inference for Segment Anything Model (SAM). | [Link](https://github.com/facebookresearch/segment-anything) |
| [infer_transunet](https://github.com/Ikomia-hub/infer_transunet) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | TransUNet inference for semantic segmentation | [Link](https://github.com/Beckschen/TransUNet) |
| [infer_unet](https://github.com/Ikomia-hub/infer_unet) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Multi-class semantic segmentation using Unet, the default model was trained on Kaggle's Carvana Images dataset | [Link](https://github.com/milesial/Pytorch-UNet) |
| [infer_yolop_v2](https://github.com/Ikomia-hub/infer_yolop_v2) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Panoptic driving Perception using YoloPv2 | [Link](https://github.com/CAIC-AD/YOLOPv2) |
| [train_detectron2_deeplabv3plus](https://github.com/Ikomia-hub/train_detectron2_deeplabv3plus) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Training process for DeepLabv3+ model of Detectron2. | [Link](https://github.com/facebookresearch/detectron2) |
| [train_hf_semantic_seg](https://github.com/Ikomia-hub/train_hf_semantic_seg) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Train models for semantic segmentationwith transformers from HuggingFace. | [Link](https://github.com/huggingface/transformers) |
| [train_mmlab_segmentation](https://github.com/Ikomia-hub/train_mmlab_segmentation) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Train for MMLAB segmentation models | [Link](https://github.com/open-mmlab/mmsegmentation) |
| [train_transunet](https://github.com/Ikomia-hub/train_transunet) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Training process for TransUNet model.  | [Link](https://github.com/Beckschen/TransUNet) |
| [train_unet](https://github.com/Ikomia-hub/train_unet) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | multi-class semantic segmentation using Unet | [Link](https://github.com/milesial/Pytorch-UNet) |
<p align="right">(<a href="#readme-top">Back to top</a>)</p>

<a name=SUPER_RESOLUTION></a>

## Super resolution

| Name | Language | Description | Original repository |
| --- | --- | --- | --- |
| [infer_swinir_super_resolution](https://github.com/JingyunLiang/SwinIR) | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | Image restoration algorithms with Swin Transformer | Made by Ikomia |
<p align="right">(<a href="#readme-top">Back to top</a>)</p>


