# Co-speech-gesture-generation-using-conditional-flow-matching-with-masked-training
A conditional flow matching model for co-speech gesture generation, with masked training for motion conditioned generation. 

## Dependencies
### For Ubuntu env
For rendering on Ubuntu\
`sudo apt install xvfb ffmpeg -y`

### Conda env
python=3.11

For rendering\
`conda install libstdcxx-ng -c conda-forge`

For cuda-toolkit\
`conda install cuda-toolkit -c nvidia/label/cuda-12.1.1`

Pytorch installation\
`pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121`

Other packages\
`pip install -r requirements.txt`

## Quick start
1. Download checkpoints from [here](https://www.dropbox.com/scl/fi/3tthgolqh3i8g0fhrj9pm/250320.zip?rlkey=22wdfmny1mn0okj3rm7kuv963&st=i6fa5yj3&dl=0).
Put it to `./assets/release/xxx.zip` and unzip to the current directory.
2. Download smplx model from [here](https://smpl-x.is.tue.mpg.de/).
Downlaod SMPL-X 2020 neutral and put it to `./modules/smplx_models/smplx/SMPLX_NEUTRAL_2020.npz`
3. Run: `python generate_from_wav.py`