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