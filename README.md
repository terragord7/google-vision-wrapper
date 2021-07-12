# google-vision-wrapper
A Tiny Python Wrapper for Google Vision API. 🐍

#### _Introduction_

Google Vision API is a service provided by Google Cloud, with the intent of giving to the vast public the possibility to access complex deep learning models, trained with huge datasets, without the inconvenience of collecting a proper dataset and training a model from scratch.

All pythonist computer vision ethusiasts (as I am), knows that there are computer vision libraries that comes already loaded with potent pre-trained models. An example is dlib with Haar Cascades, HOG for face recognition and Caffe models for object detection. But sometimes you need an API that is accessible from the web (i.e. if you want to deploy a web app or a lightweight service). Then you have two possibilities:

1. You deploy this already trained model on a web app in the cloud, with the incovinence of hosting a potential huge model on physical space, paying a lot of attention to subtle cloud Pricing and billings. 
2. Rely on a third party service, already shipped with all you need, with less attention to costs since this kind of service is really cheap.

For some kind of projects I prefer relying on the second option. Google Vision API is a perfect solution in this context. There is a list of tasks that the API can accomplish on an image:

- Optical character recognition
- Detect crop hints
- Detect faces
- Detect image properties
- Detect labels
- Detect landmarks
- Detect logos
- Detect multiple objects
- Detect explicit content
- Detect web entities and pages

For more information I strongly suggest to read the [Google Official User Guide](https://cloud.google.com/vision/docs/how-to) to the API.

Even if the guide is exhaustive and the API is straightforward to use, I found that it requires a bit of manipulation to setup and to retrieve the information from the server response in a useable way (a Pandas DataFrame or an array for plotting). This is why I started creating a wrapper around the API, that handles the request under-the-hood and which interface is more compact than the original one. 

I started for a small project of aesthetics medicine I'm working on and I though that maybe someone else could find it useful. I am therefore extending my original tiny class for face landmark recognition, with all the features provided by Google Vision API, with the possibility of retrieving the desired information in 3-lines code, in array format or as a Pandas DataFrame.

#### _Classes and Methods_

***
```python
class GVisionAPI():  
  def __init__(self,keyfile=None):
```
#### Description:
Main Class Constructor. set credentials as environmental variable and instantiate a Google ImageAnnotator Client. See [1](https://cloud.google.com/vision/docs/before-you-begin) and [2](https://cloud.google.com/vision/docs/detecting-faces) for more info.

#### Parameters:
**keyfile**: path to the .json auth file returned by google vision API. Refer to [here](https://cloud.google.com/vision/docs/before-you-begin) to know how to get it

#### Usage:
```python
from gvision import GVisionAPI

authfile = 'path_to_authfile'
gvision = GVisionAPI(authfile)
```

***

