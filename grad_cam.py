import torch

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.register_hooks()

    def register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
        
    def generate_cam(self, input_image, class_idx):
        # Forward pass
        output = self.model(input_image)
        self.model.zero_grad()
        class_score = output[0, class_idx]
        class_score.backward(retain_graph=True)
    
        # Validate gradients and activations
        if self.gradients is None or self.activations is None:
            raise RuntimeError("Gradients or activations are not captured properly.")
    
        # Compute weights and CAM
        gradients = self.gradients
        activations = self.activations
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * activations, dim=1).squeeze(0)
        cam = torch.relu(cam)
    
        # Ensure CAM is 2D
        if len(cam.shape) != 2:
            raise ValueError(f"Expected CAM to be 2D, but got shape {cam.shape}.")
    
        # Normalize CAM
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-10)
        
        # Reshape CAM to (1, 1, H, W) for interpolation
        cam = cam.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        
        return cam

