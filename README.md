# dApp Library

Fork from  original [dApp-library](https://github.com/wineslab/dApp-library), with extra applications and some extra parameters for testing.

A complete tutorial on how to deploy a dApp can be found on the [OpenRAN Gym website](https://openrangym.com/tutorials/dapps-oai). Please refer to that guide to instrument your system.

## Installation

Clone the repository

```
hatch build
pip3 install dist/*.tar.gz
```

### Launch the Spectrum Sharing dApp

OAI should start _before_ running the dApp

```python 
python3 examples/spectrum_dapp.py
```

This dApp implements a spectrum sharing use case discussed in [our paper](https://arxiv.org/pdf/2501.16502).

The dApp can be controlled through the following command-line arguments:
- `ota` bool (false): If true, use the OAI and spectrum configurations for OTA else use the ones of Colosseum
- `control` bool (false): If set to true, performs the PRB masking
- `energy-gui` bool (false): If set to true, creates and show the energy spectrum
- `iq-plotter-gui` bool (false): If set to true, creates and show the sensed spectrum
- `save-iqs` bool (false): Specify if this is data collection run or not. In the first case I/Q samples will be saved
- `timed` bool (false): Run with a 5-minute time limit
- `timed` int (1): ID of the existing dApp
- `model-deployment` str ("gpu"): Specify if the model is intended to be run on GPU or CPU
- `model-type` str ("tf"): Specify runtime option for model (standard Tensorflow, TensorRT, LiteRT)
- `input-size` int (1536): size of I/Q samples expected

