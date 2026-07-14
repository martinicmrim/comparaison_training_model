FROM jupyter/datascience-notebook:52a999a554fe
# Digest: jupyter/datascience-notebook@sha256:8c6c051f059d51a10b1919dafc5ee8058fda0d5550bbb3d43ca3a3cd2da7835a

RUN pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

RUN pip install \
        scikit-learn==1.3.0 \
        prettytable==3.9.0 \
        opencv-python==4.8.0.76 \ 
        grad-cam==1.4.8
