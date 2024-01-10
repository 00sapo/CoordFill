from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from . import models

THIS_DIR = Path(__file__).parent
WEIGHTS_PATH = THIS_DIR / "encoder-epoch-last.pth"
MODEL_CONFIG = {
    "name": "gan",
    "args": {
        "encoder_spec": {
            "name": "coordfill",
            "multi_res_training": True,
            "mask_prediction": True,
            "attffc": True,
            "scale_injection": True,
            "args": {"no_upsampling": True},
        }
    },
}


def resize_fn(img, size):
    return transforms.ToTensor()(transforms.Resize(size)(transforms.ToPILImage()(img)))


def to_mask(mask):
    return transforms.ToTensor()(
        transforms.Grayscale(num_output_channels=1)(transforms.ToPILImage()(mask))
    )


def load_image(path):
    return transforms.ToTensor()(Image.open(path).convert("RGB"))


def save_image(img, path):
    return transforms.ToPILImage()(img).save(path)


def load_model(weights_path=WEIGHTS_PATH):
    model = models.model_factory(MODEL_CONFIG)
    model.encoder.load_state_dict(torch.load(weights_path))
    return model


def inpaint(model, image: torch.Tensor, mask: torch.Tensor, resize=(2048, 2048)):
    """
    Input must be Torch tensors of shape (3, H, W) and (1, H, W) respectively with dtype float32 and values in [0, 1].
    """
    if resize is not None:
        original_size = image.shape[-2:]
        image = resize_fn(image, resize)
        mask = resize_fn(mask, resize)

    # normalize image
    image = (image - 0.5) / 0.5

    # convert mask to binary
    if mask.shape[-1] > 1:
        mask = to_mask(mask)

    mask[mask > 0] = 1
    mask = 1 - mask

    with torch.no_grad():
        pred = model.encoder.mask_predict([image.unsqueeze(0), mask.unsqueeze(0)])

    pred = (pred * 0.5 + 0.5).clamp(0, 1).view(3, resize[0], resize[1])

    # resize to original size
    if resize is not None:
        pred = resize_fn(pred.cpu(), original_size).to(pred.device)
    return pred


def main(img, mask, output, weights=WEIGHTS_PATH, gpu=False, resize=None):
    """
    Performs inpainting on the given image using the given mask and weights and saves the result to the given output path.
    """
    img = load_image(img)
    mask = load_image(mask)
    model = load_model(weights)
    if gpu:
        model = model.cuda()
        img = img.cuda()
        mask = mask.cuda()
    pred = inpaint(model, img, mask, resize=resize).cpu()
    save_image(pred, output)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
