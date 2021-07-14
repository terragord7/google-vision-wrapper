# google-vision-wrapper
A Tiny Python Wrapper 🐍 for Google Vision API. 👁

### _Introduction_

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

### _Classes and Methods_



***

### Class Initialization

```python
class GVisionAPI():  
  def __init__(self,keyfile=None):
   '''
   set credentials as environmental variable and 
   instantiate a Google ImageAnnotator Client
   see https://cloud.google.com/vision/docs/before-you-begin
   and https://cloud.google.com/vision/docs/detecting-faces 
   for more info
   '''
```

#### _Description_:
Main Class Constructor. set credentials as environmental variable and instantiate a Google ImageAnnotator Client. See [1](https://cloud.google.com/vision/docs/before-you-begin) and [2](https://cloud.google.com/vision/docs/detecting-faces) for more info.

#### _Parameters_:
**keyfile**: path to the .json auth file returned by google vision API. Refer to [here](https://cloud.google.com/vision/docs/before-you-begin) to know how to get it

#### _Usage_:
```python
from gvision import GVisionAPI

authfile = 'path_to_authfile'
gvision = GVisionAPI(authfile)
```

***

### Perform a Request

```python
def perform_request(self,img=None,request_type=None):
    '''
    given and imput image in either numpy array or bytes format
    chek type and perform conversion nparray->bytes (if needed).
    Provides the bytes content to the Google client and perform
    an API request based on the "request_type" parameter.
    Response can be accessed using the self.response attribute.

    Parameters:
    - img : input imange of type numpy.ndarray or bytes
    - request_type : a string in ['face detection','landmark detection','logo detection',
                                  'object detection','label detection','image properties',
                                  'text detection']
      representing the type of request to perform
    '''
```
#### _Description_:
Given and imput image in either numpy array or bytes format checks type and perform conversion nparray->bytes (if needed). Provides the bytes content to the Google client and performs an API request based on the "request_type" parameter. Response can be accessed using the ```self.response``` attribute.

#### _Parameters_:
**img** : input imange of type numpy.ndarray or bytes
**request_type** : a string representing the type of request to perform. Possible values: ['face detection','landmark detection','logo detection','object detection','label detection','image properties','text detection']

#### _Usage_:
```python
import cv2

# replace 'path_to_image' with what you want
img   = cv2.imread('path_to_image')

gvision.perform_request('face detection')
print(gvision.response)
```

***

### Obtain Data as list

```python
    def face_landmarks(self):
        '''
        Loop on the face annotations and, for each face,
        append a 2-tuple list with (x,y) coordinates of detected points.
        append also a list with the corresponding point names.
        return the two lists created
        '''

    def head(self):
        '''
        Loop on the face annotations and, for each face,
        append a 2-tuple list with (x,y) coordinates of head bounding box.
        append also a list with the corresponding point names.
        return the two lists created
        '''
        
    def face(self):
        '''
        Loop on the face annotations and, for each face,
        append a 2-tuple list with (x,y) coordinates of face bounding box.
        append also a list with the corresponding point names.
        return the two lists created
        '''
        
    def angles(self):
        '''
        Loops on the face annotations and, for each face,
        appends a list with the rotation angles on the 3 axis.
        appends also a list with the corresponding angle names.
        return the two lists created
        '''     
    
    def objects(self):
        '''
        Loops on the detected objects. For each,
        appends a list with name, confidence and
        (x,y) 2-tuples with the bounding box coordinates.
        Appends also a list with the corresponding headers.
        Returns the two lists created.
        '''
        
    def colors(self):
        '''
        Loops on the detected colors. For each,
        appends a list with: 
        - (r,g,b) 3-tuple with values of the color channels.
        - pixel fraction.
        - score
        Appends also a list with the corresponding headers.
        Returns the two lists created.
        '''
        
    def crop_hints(self):
        '''
        Loops on the crop_hints. For each,
        appends a list with: 
        - (x,y) 2-tuple with coordinates of the cropped image.
        - confidence.
        - importance fraction.
        Appends also a list with the corresponding headers.
        Returns the two lists created.
        '''
        
    def logos(self):
        '''
        Loops on the detected logos. For each,
        appends a list with name, confidence and
        (x,y) 2-tuples with the bounding box coordinates.
        Appends also a list with the corresponding headers.
        Returns the two lists created.
        '''
    
    def texts(self):
        '''
        Loops on the detected texts. For each,
        appends a list with description, language and
        (x,y) 2-tuples with the bounding box coordinates.
        Appends also a list with the corresponding headers.
        Returns the two lists created.
        '''
        
    def pages(self):
        '''
        Loops on the detected pages. For each,
        appends a list with language, confidence, 
        height and width.
        Appends also a list with the corresponding headers.
        Returns the two lists created.
        ''' 
        
    def blocks(self):
        '''
        Loops on the blocks in the detected pages. For each,
        appends a list with description, language, confidence, block type 
        and (x,y) 2-tuples with the bounding box coordinates.
        Appends also a list with the corresponding headers.
        Returns the two lists created.
        '''
    
    def paragraphs(self):
        '''
        Loops on the paragraphs in the blocks in the detected pages. For each,
        appends a list with description, language, confidence
        and (x,y) 2-tuples with the bounding box coordinates.
        Appends also a list with the corresponding headers.
        Returns the two lists created.
        '''
    
    def words(self):
        '''
        Loops on the words in the paragraphs 
        in the blocks in the detected pages. For each,
        appends a list with language
        and (x,y) 2-tuples with the bounding box coordinates.
        Appends also a list with the corresponding headers.
        Returns the two lists created.
        '''
    
    def symbols(self):
        '''
        Loops on the symbols in the words, in the paragraphs,
        in the blocks, in the detected pages. For each,
        appends a list with text, language
        and (x,y) 2-tuples with the bounding box coordinates.
        Appends also a list with the corresponding headers.
        Returns the two lists created.
        '''
     
    def landmarks(self):
        '''
        Loops on the detected landmarks. For each,
        appends a list with: 
        - name
        - confidence
        - (x,y) 2-tuples with coordinates of the bounding box
        - 2-tuple with latitude and logitude.
        Appends also a list with the corresponding headers.
        Returns the two lists created.
        '''
        
    def labels(self):
        '''
        Loops on the detected labels. For each,
        appends a list with name, 
        confidence and topicality.
        Appends also a list with the corresponding headers.
        Returns the two lists created.
        '''
```
#### _Description_:
Response data in form of list. For more detailed information regarding the headers meaning or type of information please refer to the corresponding guides:
- [face detection](https://cloud.google.com/vision/docs/detecting-faces)
- [object detection](https://cloud.google.com/vision/docs/object-localizer)
- [landmarks detection](https://cloud.google.com/vision/docs/detecting-landmarks)
- [logo detection](https://cloud.google.com/vision/docs/detecting-logos)
- [label detection](https://cloud.google.com/vision/docs/labels)
- [image properties](https://cloud.google.com/vision/docs/detecting-properties)
- [text in images detection](https://cloud.google.com/vision/docs/ocr)

**N.B.**: to each request type correspond different information that can be retrieved. I.e. ```gvision.perform_request('face detection')``` must me used to retrieve ```face_landmarks, face, head, angles``` information. If another type of request has been performed the API will throw a "key" error. Refere to the section _Usage_ below for a full explanation.

#### _Return types_:
**headers** : list. **data** : list

#### _Usage_:
```python

# to each type of request corresponds 
# different information that can be retrieved

gvision.perform_request('face detection')
headers,   data   = gvision.face_landmarks()
h_headers, h_data = gvision.head()
f_headers, f_data = gvision.face()
a_headers, a_data = gvision.angles()

gvision.perform_request('object detection')
headers, data = gvision.objects()

gvision.perform_request('landmark detection')
headers, data = gvision.landmarks()

gvision.perform_request('label detection')
headers, data = gvision.labels()

gvision.perform_request('logo detection')
headers, data = gvision.logos()

gvision.perform_request('text detection')
headers,    data    = gvision.texts()
p_headers,  p_data  = gvision.pages()
b_headers,  b_data  = gvision.blocks()
pr_headers, pr_data = gvision.paragraphs()
w_headers,  w_data  = gvision.words()
s_headers,  s_data  = gvision.symbols()


gvision.perform_request('image properties')
headers,    data    = gvision.colors()
ch_headers, ch_data = gvision.crop_hints()

```

***
