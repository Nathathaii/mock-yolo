# Create conda env

create conda environment for the project 


Note: Ultralytics and label-studio require Python 3.8 or later
##
    conda create -n cap-detect python=3.9
<br>

activate conda environment
##
    conda activate cap-detect
<br>

Install Ultralytics for running yolov8 and yolov10
##
    pip install -q git+http://github.com/THU-MIG/yolov10.git
<br>

install from requirement
##
    pip install requirement.txt
    
<br>

# Start Label Studio
label studio will be started at http://localhost:8080/
##
    python -m label_studio.server

<br>


# Label Studio ml backend

initiate label studio ml backend
##
    label-studio-ml init my_ml_backend
<br>

start label studio ml backend
##
    label-studio-ml start my_ml_backend

<br>