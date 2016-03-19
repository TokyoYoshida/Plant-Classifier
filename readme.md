## Plant Classifier

plant(trees) classifier implements with tensorflow.

This software can classifier a plant jpeg image file on web brwoser.
Now only 10 type tree can calssify.
However ,you can train this software and increase the tree type more.

## Installation(MAC)

This description is represent for MAC.
But you can install any platform by installing following software.
 tensorflow,python,opencv

1.install tensorflow

see Google page.

https://www.tensorflow.org/versions/r0.7/get_started/os_setup.html

2.install opencv

```
brew install opencv
```

3.install python packages


```
pip install -r requirements.txt
```

4.clone this repository

## Usage

you can start only type following command.
```
ptyhon plantClassifier.py
```

and access by your browser
http://localhost:3000

Result string is by Japanese.
You can translate a result to any language with google translate.

## New Training

Training is very easy.

1.Assambling images

Assemble vary many tree images,and save on TreeImages Directory.

2.Making train.txt ,test.txt

train.txt and test.txt is to teach tensorflow plant type from image file.

file format is fllowing style.
```
[image file Name] [tree type index]
```

for example.
```
./TreeImages/kusunoki.jpg 0
./TreeImages/ichou.jpg 1
```

If you want to increase tree type,you must modify imageClassifierTrainer.py and ./plantClassifier.py.

for example.
```
NUM_CLASSES = 100
```
3.Training tensorflow

```
ptyhon imageClassifierTrainer.py
```

If the model.ckpt file is update, training is success.

## License

 MIT
