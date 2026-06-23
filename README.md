# ai-nst-project
# AI Neural Style Transfer Project

This project implements an adaptive instance normalization (AdaIN) neural style transfer web application using PyTorch and Flask. It includes training utilities, an interactive Flask interface, and model definitions for content-style blending.

## Repository Structure

- `NST_Code/`
  - `app.py` - Flask web app for uploading images and applying style transfer.
  - `train.py` - Training script for the decoder using AdaIN loss.
  - `vgg_normalised.pth` - Pre-trained VGG encoder weights.
  - `content_data/` - Content image dataset for training.
  - `style_data/` - Style image dataset for training.
  - `experiment/` - Output folder for saved models and training artifacts.
  - `experiment/final_exp/decoder_final.pth` - Saved decoder checkpoint used by the app.
  - `static/` - Static files served by Flask.
    - `uploads/` - Uploaded content/style images and generated outputs.
  - `templates/` - HTML templates for the Flask application.
    - `index.html` - Main UI for the style transfer web app.
  - `utils/`
    - `models.py` - VGG encoder and decoder model architecture.
    - `utils.py` - Dataset utilities, transforms, and AdaIN helper functions.

## Features

- Upload content and style images via a web interface.
- Generate stylized images using AdaIN style transfer.
- Control the blending strength with the `alpha` parameter.
- Train a decoder model from scratch using custom content and style datasets.

## Requirements

Install Python dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

> Note: This project uses PyTorch and torchvision. Pick the right CUDA-enabled wheel if running on GPU.

## Running the Web App

1. Ensure the model files are available:
   - `NST_Code/vgg_normalised.pth`
   - `NST_Code/experiment/final_exp/decoder_final.pth`

2. Start the Flask app from the `NST_Code` folder:

```bash
cd NST_Code
python app.py
```

3. Open your browser at:

```text
http://localhost:5000
```

4. Upload a content image and a style image, then click `Transfer Style`.

## Training the Decoder

The training pipeline optimizes the decoder to rebuild stylized images from AdaIN features.

Example training command:

```bash
cd NST_Code
python train.py \
  --content_dir content_data \
  --style_dir style_data \
  --vgg vgg_normalised.pth \
  --experiment experiment/final_exp \
  --epochs 10 \
  --batch_size 4 \
  --content_weight 1.0 \
  --style_weight 5.0
```

Key options:
- `--content_dir`: path to content images
- `--style_dir`: path to style images
- `--vgg`: path to pre-trained VGG weights
- `--experiment`: output directory for checkpoints
- `--epochs`: number of training epochs
- `--batch_size`: batch size
- `--content_weight`: content loss weight
- `--style_weight`: style loss weight

## Notes

- The Flask app currently loads the encoder and decoder by hardcoded paths in `app.py`. Update the decoder path if you move the model checkpoint.
- Uploaded images are saved to `static/uploads/` and served from there.
- The app supports `.png`, `.jpg`, and `.jpeg` images.

## Troubleshooting

- If the app cannot load the decoder checkpoint, verify the path in `NST_Code/app.py`.
- If CUDA is unavailable, the app automatically runs on CPU.
- Ensure `static/uploads/` exists or allow the app to create it.

## License

This repository is provided as-is for neural style transfer experimentation.
