The data for the training and testing images of the danger dataset are stored in `.pt`.
These are tensors ready for training and testing with the `YoloThreat` model.
They are not included in this database because this GitHub is shared between multiple projects, and many users do not need the large data payload uploaded to their devices.

In particular, the ML/src/make_danger.py script does not work unless the raw image data is loaded into the appropriate directories. This is a few Gb of data as is.
The final result of this is the `.pt` files for the train and test datasets which also are similarly large.
To ameliorate this issue, the links to the publicly available original datasets are provided in the `make_danger` script.
Similarly the final results can be downloaded for those who need to ue it for production or for review at the following link:

https://www.kaggle.com/datasets/augustgaribay/580-threat-dectection

It is 2.15 Gb.

These files should be included in the directory `ML/data/danger/` for full integration into the other scripts and notebooks in this codebase. 