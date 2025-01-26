import torch.nn.functional as F
import matplotlib.pyplot as plt

def plot_cam(image, cam_map, title):
    image = image.permute(1, 2, 0).cpu().numpy()  # Convert image to HWC format
    cam_map = F.interpolate(cam_map, size=(224, 224), mode="bilinear", align_corners=False).squeeze(0).squeeze(0)  # Resize CAM to (224, 224)
    cam_map = cam_map.detach().cpu().numpy()  # Convert CAM to NumPy array
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow((image * 0.5 + 0.5).clip(0, 1))  # De-normalize
    axes[0].set_title("Original Image")
    axes[1].imshow(cam_map, cmap="jet", alpha=0.5)
    axes[1].set_title(title)
    plt.show()
