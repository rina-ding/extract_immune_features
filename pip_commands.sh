apt-get update
apt-get install ffmpeg libsm6 libxext6  -y
apt-get -y install libopenjp2-7-dev libopenjp2-tools openslide-tools
apt-get install python3-openslide -y
pip install openslide-python 
pip install -r requirements.txt
pip install opencv-python==4.5.5.64
pip install scikit-image
pip install zarr
pip install albumentations
pip install natsort