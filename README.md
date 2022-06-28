## XAI: Score-CAM

This project implements Score-CAM, a visual explanation method for convolutional neural networks.

Paper: [Score-CAM: Score-Weighted Visual Explanations for Convolutional Neural Networks](http://openaccess.thecvf.com/content_CVPRW_2020/papers/w1/Wang_Score-CAM_Score-Weighted_Visual_Explanations_for_Convolutional_Neural_Networks_CVPRW_2020_paper.pdf)


<img src="more/index.png" width="100%" height="100%">

## Prerequisites
Development
- Python
- Pytorch
- Numpy
- Matplotlib
- other ML libraries

Score-CAM
- Trained neural network model with at least one 2D convolutional layer
- folder with images for prediction


## Run
Run an example on [Google Colab](https://colab.research.google.com/drive/1Ar32WDBcEG5UUF8qPfSTtFEvfFiw_E2J?usp=sharing)


```bash
git clone https://github.com/duc-ng/Score-CAM
cd Score-CAM
python run.py -m model.pt -i images
```

Results can be found in ./results .

# Args
| Arg        | Name           | Description  |
| ------------- |:-------------:| -----:|
| -m    | Model | Trained pytorch model with at least one 2Dconv layer|
| -i     | Image folder      |   folder with images for prediction (.jpegs, .pngs, ..)  |

## Credits
This code is forked from the original [Score-CAM repo](https://github.com/haofanwang/Score-CAM).

