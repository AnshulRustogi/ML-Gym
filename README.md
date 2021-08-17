# ML-Gym


## Description
ML-Gym is an AI automated gym built using python and openCV. 
The app can estimate the number of pushup and squats done by the user in realtime.

## Demo 

[![Watch the video](https://media.giphy.com/media/GewSyQxwPB76f2u8KY/giphy.gif)](https://www.youtube.com/watch?v=3d-xEj4x860)

## Model and Libraries
The app is built using openCV-python library and uses the <a href="https://github.com/CMU-Perceptual-Computing-Lab/openpose">openpose MPI model</a> to estimate the human pose. 
## Run the app
To run the app use one of the following methods:
* Using conda environment
  - Clone the repository 
    ```
    git clone https://github.com/AnshulRustogi/ML-Gym.git
    ```
  - Create a new conda environment using requirements.txt and activate it
    ```
    conda create --name ml-gym --file requirements.txt
    conda activate ml-gym
    ```
  - Run the application using
    ```bash
    python main.py --device=gpu --videoInput=cam --activity=squats --displaySkeleton=1 
    ```
    
* To run in docker on cpu:
```bash
docker run --rm -it -e DISPLAY=$DISPLAY --privileged --volume ~/.Xauthority -v /tmp/.X11-unix:/tmp/.X11-unix:ro --device /dev/video0 anshulrustogi/ml-gym --device=cpu
```
* To run in docker on nvidia-gpu:
```bash
docker run --rm -it -e DISPLAY=$DISPLAY --privileged --volume ~/.Xauthority -v /tmp/.X11-unix:/tmp/.X11-unix:ro --device /dev/video0 --runtime=nvidia --gpus all ml-gym:gpu --device=gpu
```
## Optional Arugments
1) --device=gpu/cpu
    - Option to run the app on gpu or cpu
    - Note: to run the gpu, opencv with cuda is required
2) --videoInput=cam/\<source\>
    - Source of the input video file
    - cam: Take input from the device webcam
    - \<source\>: location of the video file for input. Eg: ```--videoInput=Video/sampleVideo.mp4```
3) --activity=pushup/squats
    - Specifc the activity done by the user either squats or pushup
4) --displaySkeleton=0/
    - 1: The skeleton would be displayed
    - 0: The skeleton would not be displayed
## Contributors
* Rishabh Rustogi
* Anshul Rustogi
