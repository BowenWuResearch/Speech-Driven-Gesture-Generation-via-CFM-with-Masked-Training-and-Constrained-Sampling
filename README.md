# Speech-Driven Gesture Generation via Conditional Flow Matching with Masked Training and Constrained Sampling

Following instructions below you can host a webpage for generating gestures for your uploaded audio.

## Dependencies
### For Ubuntu
(Optional) For rendering on Ubuntu\
`sudo apt install xvfb ffmpeg -y`

### Create environment
Run the follows to configure environment:
```bash
conda create -n gcfm python=3.11
conda activate gcfm
pip install -U pip
# (Optional) For rendering: conda install libstdcxx-ng -c conda-forge -y
# (Optional) For GPU, change to your cuda version: conda install cuda-toolkit -c nvidia/label/cuda-12.1.1 -y
# Install torch for GPU
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
# (Optional) Install torch for CPU: pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0
# Install project
pip install -e .
```

### Download necessary files
Download from [URL](https://drive.google.com/drive/folders/13TG1igXPMdG48TgyI8Zfo_pYDV_xEfeO?usp=sharing) and place them as follows:
```
chkpts/250320.pth
chkpts/stats.pkl
assets/smplx_models/smplx/SMPLX_NEUTRAL_2020.npz
```

## Gradio Demo
Run: `python src/scripts/demo.py`