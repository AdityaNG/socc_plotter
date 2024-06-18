# socc_plotter

[![codecov](https://codecov.io/gh/AdityaNG/socc_plotter/branch/main/graph/badge.svg?token=socc_plotter_token_here)](https://codecov.io/gh/AdityaNG/socc_plotter)
[![CI](https://github.com/AdityaNG/socc_plotter/actions/workflows/main.yml/badge.svg)](https://github.com/AdityaNG/socc_plotter/actions/workflows/main.yml)

Semantic Occupancy 3D Plotter

## Install it from PyPI

```bash
pip install socc_plotter
```

## Usage

The socc_plotter works on a callback mechanism since the GUI must be run on the main thread.
```py
from socc_plotter.plotter import Plotter
import time
def callback():
    time.sleep(1)
    print('in callback')

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

pip install nuscenes-devkit
```

Run the demo
```bash
$ python -m socc_plotter
#or
$ socc_plotter
```

## Development

Read the [CONTRIBUTING.md](CONTRIBUTING.md) file.
