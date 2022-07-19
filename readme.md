# Evaluation Scripts for GigaVisiiiion Challenge
Official evaluation scripts for GigaVision Challenge. 

## Installation
The evaluation scripts in this repo. is compatible with python3. Conda environment and additional dependencies including pytorch3d can be installed by running:

`pip install -r requirements.txt`

## GigaMOT
To evaluate on the MOT challenge, your results should be organized into a `results.zip` file according to the [official submission guidance](https://www.gigavision.cn/other/page?page=f6cdadbba43645a2a64d1ccfcccbbba5&anchor=results&from=Tracking) and the ground truth data is arranged into `./tracking/mot_anno_full`. The results can be evaluated by running:

`python evaluate.py /path/to/gt /path/to/result /url/to/post/score Tracking union_users_key`

and scores of MOTA and MOTP will be returned. See [here](https://www.gigavision.cn/other/page?page=f6cdadbba43645a2a64d1ccfcccbbba5&anchor=evaluation&from=Tracking) for more details about evaluation metrics.

## GigaDetection
Evaluation codes for Detection are coming soon.

## GigaTrajectory
Evaluation codes for Trajectory are coming soon.
