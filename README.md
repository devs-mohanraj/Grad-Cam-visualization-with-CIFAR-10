# GRAD-CAM Visualization of FGSM on CIFAR-10 Trained ResNet-18 Model

## Contents
- [Project Overview](#project-overview)
- [Key Concepts](#key-concepts)
- [Project Workflow](#project-workflow)
- [Model Visualization](#model-visualization)
- [Results](#results)
- [Installation](#installation)
- [References](#references)

##  Project Overview
This project visualizes the effects of **Fast Gradient Sign Method (FGSM) adversarial attacks** on a **ResNet-18 model trained on CIFAR-10** using **Grad-CAM**. The goal is to understand how adversarial attacks shift the model’s attention and affect predictions.

##  Key Concepts
- **Grad-CAM (Gradient-weighted Class Activation Mapping)**: A technique to highlight important image regions influencing model decisions.
- **Fast Gradient Sign Method (FGSM)**: A white-box adversarial attack that perturbs an image in the direction of the gradient to fool the model.
- **ResNet-18 on CIFAR-10**: A convolutional neural network with 18 layers trained on CIFAR-10, achieving ~92% accuracy.

## Project Workflow
1. **Train ResNet-18 on CIFAR-10**
   - Load and preprocess CIFAR-10 dataset.
   - Modify ResNet-18 for 10 classes and train the model.

2. **Generate FGSM Adversarial Examples**
   - Compute gradients of input images using FGSM.
   - Add small perturbations to create adversarial images that fool the model.

3. **Apply Grad-CAM for Explainability**
   - Extract feature maps from a **convolutional layer** in ResNet-18.
   - Compute **gradient-based importance weights**.
   - Generate **heatmaps** to visualize the regions influencing the model’s decision.

4. **Compare Clean vs. Adversarial Predictions**
   - Use Grad-CAM to **visualize attention maps** for both **original and adversarial images**.
   - Analyze **how adversarial attacks shift the model's focus**.

## Model Visualization
- **Grad-CAM heatmaps** highlight key image regions.
- **Matplotlib** is used for visualizing results.
- **Training accuracy/loss plots** help track model performance.

## Results
**Clean image**

![alt text](<__pycache__\readme_images\clean_image.jpg>)

**Adversarial image**

![alt text](<__pycache__\readme_images\adversarial_image.jpg>)

## Installation
Run the following commands to set up the environment:

```bash
# Create a virtual environment (optional)
python -m venv env
source env/bin/activate  # On Windows, use `env\Scripts\activate`

# Install required dependencies
pip install torch torchvision numpy matplotlib adversarial-robustness-toolbox
```

## References

- Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization
[https://arxiv.org/pdf/1610.02391]

- Explaining and Harnessing Adversarial Examples
[https://arxiv.org/pdf/1412.6572]