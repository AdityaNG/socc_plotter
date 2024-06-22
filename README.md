# socc_plotter

[![codecov](https://codecov.io/gh/AdityaNG/socc_plotter/branch/main/graph/badge.svg?token=socc_plotter_token_here)](https://codecov.io/gh/AdityaNG/socc_plotter)
[![CI](https://github.com/AdityaNG/socc_plotter/actions/workflows/main.yml/badge.svg)](https://github.com/AdityaNG/socc_plotter/actions/workflows/main.yml)

Semantic Occupancy 3D Plotter. This is the plotter made by the [SOccDPT](https://adityang.github.io/SOccDPT) project to create fancy 3D visuals. You can use this for your own AV or Robotics visualization!

![demo](media/demo.gif)

## Install it from PyPI

```bash
pip install socc_plotter
```

## Usage

The socc_plotter works on a callback mechanism since the GUI must be run on the main thread.
```py
from socc_plotter.plotter import Plotter
import time

def callback(plot: Plotter):
    time.sleep(1)
    print("in callback")
    graph_region = plot.graph_region

    points = np.array([[1, 0, 0]])
    colors = np.array([[1, 1, 1]])

    graph_region.setData(pos=points, color=colors)

plotter = Plotter(
    callback=callback,
)
plotter.start()
```


## NuScenes Demo

Start by downloading the NuScenes mini datset
```
mkdir -p data/nuscenes/
cd data/nuscenes/
wget -c https://www.nuscenes.org/data/v1.0-mini.tgz
tar -xf v1.0-mini.tgz
```

Install a few dependencies for the demo
```
pip install nuscenes-devkit==1.1.10
pip install transformers torch torchvision timm accelerate general_navigation
```

Run the demo
```bash
$ python -m socc_plotter
#or
$ socc_plotter
```

## Development

Read the [CONTRIBUTING.md](CONTRIBUTING.md) file.

## Cite

Cite our work if you find it useful

```bibtex
@article{NG2024SOccDPT,
  title={SOccDPT: 3D Semantic Occupancy from Dense Prediction Transformers trained under memory constraints},
  author={NG, Aditya},
  journal={Advances in Artificial Intelligence and Machine Learning},
  volume={ISSN: 2582-9793, Source Id: 21101164612},
  year={2024},
  url={https://www.oajaiml.com/}
}
``` 

## TODO

- [x] Demo
    - [x] RGB Frame
    - [x] Depth perception
    - [x] Semantic segmentation
    - [x] NuScenes Calibration
    - [x] NuScenes Vehicle trajectory
    - [x] Semantic Occupancy Grid
- [ ] Ensure demo dependencies are seperate from the module
- [ ] Demo is to prompt the user to install dependencies
- [ ] Demo is to auto download NuScenes and unarchive it
- [ ] Test Cases
- [ ] PiPy deployment
