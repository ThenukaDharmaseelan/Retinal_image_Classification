# FAF / IR / OTHER retinal image classifier: demo

Predicts whether a retinal image is **FAF** (fundus autofluorescence), **IR** (infrared SLO) or **OTHER**.

The model is a DenseNet-121, pretrained on ImageNet and fine-tuned on 253 labelled images
(`model/final_model_pretrained.pt`, 28 MB).

## Contents

| File | What it is |
|---|---|
| `predict.py` | Command-line script: predicts images or folders and writes a CSV |
| `FAF_IR_Demo.ipynb` | The same thing as a notebook, plus a grid of images with their predictions |
| `model/final_model_pretrained.pt` | Trained weights and the preprocessing settings |
| `requirements.txt` | Python packages |

## Setup

Python 3.9 or newer. A GPU is optional; on CPU it takes well under a second per image.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run it

```bash
# a single image
python predict.py path/to/image.png

# one or more folders (searched recursively), results saved to a CSV
python predict.py path/to/folder another/folder --out my_predictions.csv

# force CPU
python predict.py path/to/folder --device cpu
```

Output: one row per image, giving the predicted class, its confidence, and the probability of each class:

```
image                 prediction  confidence  p_FAF   p_IR    p_OTHER
.../0063_AF_B-0_0.png FAF         0.8755      0.8755  0.0023  0.1222
```

For the notebook, open `FAF_IR_Demo.ipynb` from this folder, set `IMAGE_PATH` in the first code cell, and run all cells.

Supported formats: PNG, JPG, TIFF, BMP. Grayscale images are fine; they are converted to 3 channels.

## Preprocessing

The script applies the training preprocessing automatically. The settings are stored in the checkpoint:
1. Keep the aspect ratio, scale the longer side to 330 px, and pad to a black square.
2. Black out everything outside the centred circle. This is done so the model can't use the image outline as a shortcut.
3. Apply ImageNet normalisation.

Don't resize or crop the images yourself first. Pass the original files.

## How well it works

| | Result |
|---|---|
| 5-fold cross-validation (253 images) | macro-F1 0.992 ± 0.011 |
| Held-out test set (55 images: 22 FAF, 13 IR, 20 OTHER) | 55 / 55 correct |

Before relying on these numbers, note the following limitations:
- **The test set is small.** With 55 images, 100% accuracy is still consistent with a true accuracy of about 95%.
- **3 test images are exact copies of training images**, and 1 is a near-duplicate. Without them it scores 51 / 51.
- **There is no patient ID in the data**, so we can't confirm that train and test contain different patients.
- **The OTHER images come from different sources from the FAF/IR images.** The model may partly recognise the source rather than the modality.
- **It has not been tested on images from another clinic or device.** Treat a low confidence (below about 0.8) as "check by eye".

If you run it on new data, especially from another site, please send back the CSV and any errors.
