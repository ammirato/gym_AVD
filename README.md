
# AVD Gym Environment
A starting point for working with the [AVD dataset](http://cs.unc.edu/~ammirato/active_vision_dataset_website/index.html) with an OpenAI Gym style environment. It should be fairly straight forward to change the rewards given, or add in getting depth images from AVD. Ideally this will all be added here before too long.

##The code
The interesting parts of the code are in `gym_AVD/gym_AVD/envs/AVD_env.py`

## External Requirements
* Python 2.X (or probably Python 3)
* [AVD Data](http://www.cs.unc.edu/~ammirato/active_vision_dataset_website/get_data.html) Parts 1, 2 and 3
* [AVD processing code](https://github.com/ammirato/active_vision_dataset_processing)

##  Dependencies and Data:

1. Get the [AVD processing code](https://github.com/ammirato/active_vision_dataset_processing), and make sure it is included in your PYTHONPATH
2. Download the [AVD Data](http://www.cs.unc.edu/~ammirato/active_vision_dataset_website/get_data.html) into a path of your choosing, we will refer to is as `AVD_ROOT_DIR`.
3. Make sure to also get the [instance id map](https://drive.google.com/file/d/1UmhAr-l-CL3CeBq6U8V973jX5BPWkrlK/view?usp=sharing) and put it in the `AVD_ROOT_DIR`
4.  Optionally download the [target images](https://drive.google.com/file/d/1uV2I-SYWQvJb0PqzDdg8ESwRdQoVpSWr/view?usp=sharing) into a path of your choosing, we will refer to is as `TARGET_IMAGE_DIR`.

## Installation
1. `git clone https://github.com/ammirato/gym_AVD.git`
2. cd `gym_AVD/`
3. pip install -e .

## Usage



##TODO
1. Add depth images option
2. Add instructions for how to adjust rewards based on things like distance to object, etc. 
