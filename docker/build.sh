# log to nvidia ngc --> need for gpu images
# sudo docker login nvcr.io

# Image Building
export DOCKER_IMAGE_NAME=gpu_pytorch_jupyter
export VERSION=3.0.0
docker build --pull --rm --no-cache -t $DOCKER_IMAGE_NAME:$VERSION .
