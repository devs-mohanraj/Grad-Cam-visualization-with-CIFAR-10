import torch
from data_loader import get_dataloaders
from model_loader import load_model, create_classifier
from grad_cam import GradCAM
from adversarial_attack import generate_adversarial_examples
from visualization import plot_cam

# Load data, model, and classifier
train_loader, test_loader = get_dataloaders()
model = load_model()
classifier = create_classifier(model)

# Initialize Grad-CAM
grad_cam = GradCAM(model, model.layer3)

# Load test images and generate adversarial images
test_images, test_labels = next(iter(test_loader))
test_images = test_images.to(torch.float32)
adv_images = generate_adversarial_examples(classifier, test_images)

# Process and visualize Grad-CAM for multiple images
num_images = 5
for i in range(num_images):
    # Grad-CAM for Clean Image
    input_image = test_images[i:i+1]
    class_idx_clean = torch.argmax(model(input_image)).item()
    cam_map_clean = grad_cam.generate_cam(input_image, class_idx_clean)
    plot_cam(test_images[i], cam_map_clean, title=f"Clean Image CAM {i + 1}")

    # Grad-CAM for Adversarial Image
    adv_image = torch.tensor(adv_images[i:i+1], dtype=torch.float32)
    class_idx_adv = torch.argmax(model(adv_image)).item()
    cam_map_adv = grad_cam.generate_cam(adv_image, class_idx_adv)
    plot_cam(torch.tensor(adv_images[i]), cam_map_adv, title=f"Adversarial Image CAM {i + 1}")
