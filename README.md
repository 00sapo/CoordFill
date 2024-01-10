# CoordFill

This is a simple repo that makes CoordFill easily installable via pip, including the
pre-trained weights.

I have also cleaned the code from the training-related parts, but this is far from being
complete.

## Installation

### API

`pip install ....`

### CLI

`pipx install ...`

## Usage

### API

```python
from coordfill import load_model, inpaint, load_image, save_image

img = load_image(img)
mask = load_image(mask)
model = load_model() # you can specify the path to your custom weights if you like

if gpu: # using GPU or not is up to you
    model = model.cuda()
    img = img.cuda()
    mask = mask.cuda()

# perform Inpainting
# reshape before inpainting if the image is too large
pred = inpaint(model, img, mask, resize=(1024, 1024))

# if you used cuda, the returned value will be there...
save_image(pred.cpu(), "out.png")
```

### CLI

- `coordfill - --help` for help
- `coordfill input.png mask.png output.png --gpu`
- Options: `--gpu`, `--resize`, `--weights`

## 🛎 Citation

If you find our work useful in your research, please consider citing:

```

@inproceedings{liu2023coordfill,
title={CoordFill: Efficient High-Resolution Image Inpainting via Parameterized Coordinate Querying},
author={Liu, Weihuang and Cun, Xiaodong and Pun, Chi-Man and Xia, Menghan and Zhang, Yong and Wang, Jue},
booktitle={AAAI},
year={2023}
}

```

## 💗 Acknowledgements

The original code is: https://github.com/NiFangBaAGe/CoordFill/
