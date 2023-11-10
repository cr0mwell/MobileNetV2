# MobileNetV2
Implementation of MobileNetV2 architecture in Tensorflow 2.12.

## Usage
It's assumed you have Tensorflow 2.12 environment [installed](https://www.tensorflow.org/install) with all necessary libraries.<br>
Install all third-party modules from the _requirements.txt_ file.<br>
To launch a MobileNetV2 model run the following command:
```
python MobileNetV2.py
```
By default the model runs in an _inference_ mode.<br>
Here is a full list of optional arguments:

`-t` - launch a model in a train mode<br>
`-e` - number of epochs to train (defaults to 50)<br>
`-b` - batch size for the training (defaults to 100)<br>

In the _train_ mode model's artifacts will be saved to `./src/models/MobileNetV2_cifar10` directory.
