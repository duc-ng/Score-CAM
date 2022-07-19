import torch
import torch.nn.functional as F
import torchvision.models as models
import matplotlib.pyplot as plt
import sys
import getopt
import pathlib
import shutil
import numpy as np
from matplotlib.image import imsave
from glob import glob
from os.path import join
from utils import load_image, apply_transforms, basic_visualize, visualize, save_output
from cam.scorecam import ScoreCAM
import warnings

warnings.filterwarnings("ignore")


# load paths from cmdline args
try:
  opts, args = getopt.getopt(sys.argv[1:],"hm:i:",["model=","input="])
except getopt.GetoptError:
  print('run.py -m <model> -i <imagefolder>')
  print("yoh")
  sys.exit(2)
for opt, arg in opts:
  if opt == '-h':
      print('run.py -m <model> -i <imagefolder>')
      sys.exit()
  elif opt in ("-m", "--model"):
      path_model = arg
  elif opt in ("-i", "--input"):
      path_imagedir = arg
print('Model:', path_model)
print('Image folder:', path_imagedir)

# load model and images
model = torch.load(path_model)
images = glob(join(path_imagedir, "*"))

# make result directory
path_results = pathlib.Path('./results')
if path_results.exists() and path_results.is_dir():
    shutil.rmtree(path_results) #rm previous results
path_results.mkdir(parents=True, exist_ok=True) 

# create scorecam heatmap
for image in images:
  # load image
  input_image = load_image(image)
  input_ = apply_transforms(input_image)
  if torch.cuda.is_available():
    input_ = input_.cuda()

  # save heatmap
  model_scorecam = ScoreCAM(dict(arch=model, input_size=input_image.size))
  heatmap = model_scorecam(input_)
  path_heatmap = join(path_results, "heatmap_"+pathlib.Path(image).stem)
  with torch.no_grad():
    if heatmap is not None:
      save_output(input_.cpu(), heatmap.type(torch.FloatTensor).cpu(),save_path=path_heatmap)


print("Heatmaps can be found in ./results")
print("FINISH")