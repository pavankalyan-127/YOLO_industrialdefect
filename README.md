#  Industrial Surface Defect Detection using YOLO

This project focuses on detecting **six types of industrial surface defects** using a YOLO-based object detection model.
Surface quality inspection is essential in manufacturing industries like steel, aluminum, automotive, and metal sheet production.
Automating defect detection helps reduce manual inspection errors and improves production accuracy.

The system identifies **6 defect categories**:

- **crazing**
- **inclusion**
- **patches**
- **pitted_surface**
- **rolled-in_scale**
- **scratches**
- 
##  Features

- ✔️ YOLO-based detection for high-speed inference  
- ✔️ Supports **all 6 industrial defect classes**  
- ✔️ Real-time detection for factory automation  
- ✔️ High accuracy on grayscale manufacturing images  
- ✔️ Overlay bounding boxes + confidence scores  
- ✔️ Streamlit interface for easy visualization  

##  Model Architecture

This project uses **YOLOv8 / YOLOv5** style detection with:

- CSPDarknet / Conv blocks  
- PANet feature fusion  
- Multi-scale prediction heads  
- Grayscale input optimization  

### Why YOLO?

- High FPS for real-time inspection  
- Strong small-object performance  
- Lightweight and deployable on edge devices  
- Accurate on texture-based defects
- 
##  Dataset (6 Classes)

The dataset contains **6 types of steel/metal defects** commonly found in industrial manufacturing:


Image format: **grayscale, 200x200**  
Dataset size: **~1500+ images (balanced)**  

##  Output Results

deployment link
https://yoloindustrialdefect.streamlit.app/

![image alt](https://github.com/pavankalyan-127/YOLO_industrialdefect/blob/main/ind_1.jpg?raw=true) 
![image alt](https://github.com/pavankalyan-127/YOLO_industrialdefect/blob/main/ind_1.1.jpg?raw=true) 
