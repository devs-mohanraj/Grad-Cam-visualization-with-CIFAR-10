from art.attacks.evasion import FastGradientMethod
import torch

def generate_adversarial_examples(classifier, test_images, eps=0.2):
    fgsm = FastGradientMethod(estimator=classifier, eps=eps)
    adv_images = fgsm.generate(x=test_images.numpy())
    adv_images = (adv_images - 0.5) / 0.5  # Normalize to match preprocessing
    adv_image = torch.tensor(adv_images[0:1], dtype=torch.float32)

    return adv_images
