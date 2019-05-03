# Fingerprint Core Point Detection
## Requirements
- Data in `./path/Data` folder
- Groundtruth in `./<path>/Ground_Truth` folder with file named `<image_name>.txt`
- python3 and tf libraries
## How to run the code?
### Train
- Execute the command `python3 python3corePoint.py --phase=train --epochs=2` Here, number of epochs has been set to 2.
- Code will ask for input folder - `Enter the training folder:`
- Enter the path to data as mentioned in requirements and hit enter

### Test
- Execute the command `python3 --phase=test`
- Code will ask for input folder: `Enter the training folder:`
- Enter the path to data as mentioned in requirements and hit enter

## Output
### Train
The model file named `q3.h5` will be saved in current directory.

### Test
Outputs will be saved in `./results/<image_name>/pred.txt` file. Each file will contain two numbers indicating position of core point on image.