# Co-speech-gesture-generation-using-conditional-flow-matching-with-masked-training
A conditional flow matching model for co-speech gesture generation, with masked training for motion conditioned generation. 

## Dependencies
### For Ubuntu env
For rendering on Ubuntu\
`sudo apt install xvfb ffmpeg -y`

### Conda env
python=3.11\
`conda create -n cfm python=3.11`

Activate
`conda activate cfm`

For rendering\
`conda install libstdcxx-ng -c conda-forge`

For cuda-toolkit\
`conda install cuda-toolkit -c nvidia/label/cuda-12.1.1`

Packages\
`pip install -r requirements.txt`

Checkpoints and files for rendering\
`git lfs pull`

## Gradio Demo
Run: `python demo.py`