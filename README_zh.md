# CNN-Receptive-Field-Visualization
<h4 align="center">
  <p style="text-align: center;">
  <b> English</b> | <a href="README_zh.md">中文文档</a>
  </p>
</h4>

<p align="center">

**Receptive Field Visualizer: Make receptive fields of CNN  models more clear. And it is helpful to design your own model architecture.**

## Supported Models
✅ resnet18

✅ pixelcnn

✅ toy models (SimpleCNN && SimpleStrideCNN)

⏳ MobileNet [TODO]

## Quick Start

### Usage
```shell
git clone https://github.com/Phi-C/CNN-Receptive-Field-Visualization.git
pip install -r requirements.txt

cd cnnrfvis

# Trace your model and save receptive field info
python save_model_layers.py models/PixelCNN.py

# Visualize receptive fields
streamlit run web.py 
```
### Custom Models
If you want to visualize your own vision model, take following steps:
1. Put your model architecture code in `cnnrfvis/models/{model_name}.py`
2. Add `get_{model_name}_rf_info` function in `cnnrfvis/model/{model_name}.py` 
3. Add `from cnnrfvis.models.{model_name} import get_{model_name}_rf_info` in `save_model_layers.py`

## Visualization Example
Take PixelCNN for example, after `streamlit run web.py` is runned, you are directed to a web page.

1. Choose your interested layer.
2. Click `Visualize Receptive Field` button
3. Put your mouse on some pixel on feature map(left)

Then you will see the corresponding receptive fields on the input image.

![Example](assets/CNNRFVis.png)