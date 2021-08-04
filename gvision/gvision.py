# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 10:31:07 2021

@github: gcgrossi
@email: giulio.cornelio.grossi@gmail.com
"""

import os
import pandas as pd
import numpy as np
from google.cloud import vision
import cv2


def google_version(lib=None):

    # if the supplied library is None, import OpenCV
    if lib is None:
        import google.cloud as lib

    # return the major version number
    return int(lib.__version__.split(".")[0])


class GVisionAPI():
    def __init__(self, keyfile=None):
        
        if keyfile == None:
            raise Exception("required argument keyfile. Must contain a valid path to authentication json file from Google Vision API. See https://cloud.google.com/vision/docs/before-you-begin on how to get it.")
        
        self.keyfile = keyfile
        self.request_category = None
        self.client = None

        # setup Vision API environment
        self.set_environ()
        if self.client == None: 
            raise Exception("Client was not setup correctly.")
        
        # define a dictionary with all possible resquests
        self.requests_dict = {
            'face detection' :       self.client.face_detection,
            'landmark detection':    self.client.landmark_detection,
            'logo detection':        self.client.logo_detection,
            'object detection':      self.client.object_localization,
            'label detection':       self.client.label_detection,
            'image properties':      self.client.image_properties,
            'text detection':        self.client.text_detection,
            'handwriting detection': self.client.document_text_detection,
            'web detection':         self.client.web_detection  
        }

        # define a dictionary with the methods available
        self.methods_dict = {
            'face landmarks': self.face_landmarks,
            'face': self.face,
            'head': self.head,
            'angles': self.angles,
            'objects': self.objects,
            'landmarks':self.landmarks,
            'logos': self.logos,
            'labels':self.labels,
            'colors':self.colors,
            'crop hints':self.crop_hints,
            'texts':self.texts,
            'pages':self.pages,
            'blocks':self.blocks,
            'paragraphs':self.paragraphs,
            'words':self.words,
            'symbols':self.symbols,
            'web entities':self.web_entities,
            'matching images':self.matching_images,
            'similar images':self.similar_images
        }
        
        # the options are the keys of the dictionaries
        self.request_types = self.requests_dict.keys()
        self.df_types      = self.methods_dict.keys()
        
        return
    
    def set_environ(self):
        """
        set credentials as environmental variable and 
        instantiate a Google ImageAnnotator Client
        see https://cloud.google.com/vision/docs/before-you-begin
        and https://cloud.google.com/vision/docs/detecting-faces 
        for more info
        """
        
        # set credentials
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"]=self.keyfile
        
        # initialize client
        self.client = vision.ImageAnnotatorClient()
        return
    
    def request_options(self):
        print('Possible Request Options: \n* '+'\n* '.join(self.request_types))
        return
    
    def perform_request(self,img=None,request_type=None):
        """
        given and imput image in either numpy array or bytes format
        chek type and perform conversion nparray->bytes (if needed).
        Provides the bytes content to the Google client and perform
        an API request based on the "request_type" parameter.
        Response can be accessed using the self.response attribute.

        Parameters:
        - img : input imange of type numpy.ndarray or bytes
        - request_type : a string in ['face detection','landmark detection','logo detection']
          representing the type of request to perform
        """

        # init bytestream image content
        content = None
        self.img = img
         
        # check input parameters
        if img is None:
            raise Exception('Required Parameter "image". Accepted types:\n* numpy.ndarray\n* bytes')
        if request_type is None:
            raise Exception('Required Parameter "request_type".\nYou should specify one of the following: \n* '+'\n* '.join(self.request_types))
        if request_type not in self.request_types:
            raise Exception(('Bad argument "{}" supplied for "request_type".\nYou should specify one of the following: \n* '+'\n* '.join(self.request_types)).format(request_type))

        # if type is numpy array -> convert to bytes
        if type(img) == np.ndarray:
            content = cv2.imencode('.jpg', img)[1].tobytes()
        
        # if type is bytes -> do nothing
        if type(img) == bytes:
            # convert bytestream to nparray image for further manipulation
            content = img
            nparr = np.frombuffer(content, np.uint8)
            self.img  = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # if none of the two -> raise Exception
        if content is None:
            raise Exception('The image supplied has not the correct type or cannot be converted in bytestream.\nAccepted types:\n* numpy.ndarray\n* bytes')
        
        # Supply the byte stream image
        image = vision.Image(content=content)

        # perform actual request based on input
        self.response = self.requests_dict[request_type](image=image)
        self.request_category = 'face'

        if self.response.error.message:
            raise Exception('{}\nFor more info on error messages, check: ''https://cloud.google.com/apis/design/errors'.format(self.response.error.message))
                
        return
                    
    def face_landmarks(self):
        """
        Loop on the face annotations and, for each face,
        append a 2-tuple list with (x,y) coordinates of detected points.
        append also a list with the corresponding point names.
        return the two lists created
        """
        self.faces = self.response.face_annotations
        types,vtx = [],[]
        for i,face in enumerate(self.faces):
            vx = []
            for vertex in face.landmarks:
                vx.append((vertex.position.x,vertex.position.y))
                if i==0: types.append("{}".format(vertex.type_))
            vtx.append(vx)
                
        return types,vtx

    def head(self):
        """
        Loop on the face annotations and, for each face,
        append a 2-tuple list with (x,y) coordinates of head bounding box.
        append also a list with the corresponding point names.
        return the two lists created
        """
        self.faces = self.response.face_annotations
        types,vtx = [],[]
        for i,face in enumerate(self.faces):
            vx = []
            for v,vertex in enumerate(face.bounding_poly.vertices):
                vx.append((vertex.x,vertex.y))
                if i==0: types.append('head{}'.format(v+1))
            vtx.append(vx)
                
        return types,vtx
    
    def face(self):
        """
        Loop on the face annotations and, for each face,
        append a 2-tuple list with (x,y) coordinates of face bounding box.
        append also a list with the corresponding point names.
        return the two lists created
        """
        self.faces = self.response.face_annotations
        types, vtx= [],[]
        for i,face in enumerate(self.faces):
            vx = []
            for v,vertex in enumerate(face.fd_bounding_poly.vertices):
                vx.append((vertex.x,vertex.y))
                if i==0: types.append('face{}'.format(v+1))
            vtx.append(vx)
        
        return types,vtx
    
    def angles(self):
        """
        Loops on the face annotations and, for each face,
        appends a list with the rotation angles on the 3 axis.
        appends also a list with the corresponding angle names.
        return the two lists created
        """
        self.faces = self.response.face_annotations
        angles, types = [], ['roll_angle','tilt_angle','pan_angle']
        for face in self.faces:
            angles.append([face.roll_angle,face.tilt_angle,face.pan_angle])
            
        return types,angles
    
    def objects(self):
        """
        Loops on the detected objects. For each,
        appends a list with name, confidence and
        (x,y) 2-tuples with the bounding box coordinates.
        Appends also a list with the corresponding headers.
        Returns the two lists created.
        """
        types,vtx = ["OBJECT_NAME","CONFIDENCE"],[]
        h,w,d = self.img.shape

        objects = self.response.localized_object_annotations
        for i,obj in enumerate(objects):
            vx = []
            vx.append(obj.name)
            vx.append(obj.score)

            for v,vertex in enumerate(obj.bounding_poly.normalized_vertices):
                vx.append((vertex.x*w,vertex.y*h))
                if i==0: types.append("BBOX_{}".format(v+1))
            vtx.append(vx)

        return types,vtx

    def colors(self):
        """
        Loops on the detected colors. For each,
        appends a list with: 
        - (r,g,b) 3-tuple with values of the color channels.
        - pixel fraction.
        - score
        Appends also a list with the corresponding headers.
        Returns the two lists created.
        """
        types,vtx = ["COLOR","PIXEL_FRACTION","SCORE"],[]

        colors = self.response.image_properties_annotation.dominant_colors
        for col in colors.colors: 
            vx = []

            vx.append((col.color.red,col.color.green,col.color.blue))
            vx.append(col.pixel_fraction)
            vx.append(col.score)

            vtx.append(vx)

        return types,vtx
    
    def crop_hints(self):
        """
        Loops on the crop_hints. For each,
        appends a list with: 
        - (x,y) 2-tuple with coordinates of the cropped image.
        - confidence.
        - importance fraction.
        Appends also a list with the corresponding headers.
        Returns the two lists created.
        """
        types,vtx = [],[]

        hints = self.response.crop_hints_annotation.crop_hints
        for i,hint in enumerate(hints): 
            vx = []

            for v,vertex in enumerate(hint.bounding_poly.vertices):
                vx.append((vertex.x,vertex.y))
                if i==0: types.append("BBOX_{}".format(v+1))
            
            if i==0:
               for t in ["CONFIDENCE","IMPORTANCE_FRACTION"]: types.append(t)
            vx.append(hint.confidence)
            vx.append(hint.importance_fraction)

            vtx.append(vx)

        return types,vtx
    
    def logos(self):
        """
        Loops on the detected logos. For each,
        appends a list with name, confidence and
        (x,y) 2-tuples with the bounding box coordinates.
        Appends also a list with the corresponding headers.
        Returns the two lists created.
        """
        types,vtx = ["LOGO_NAME","CONFIDENCE"],[]

        logos = self.response.logo_annotations
        for i,logo in enumerate(logos):
            vx = []
            vx.append(logo.description)
            vx.append(logo.score)

            for v,vertex in enumerate(logo.bounding_poly.vertices):
                vx.append((vertex.x,vertex.y))
                if i==0: types.append("BBOX_{}".format(v+1))
            vtx.append(vx)

        return types,vtx
    
    def texts(self):
        """
        Loops on the detected texts. For each,
        appends a list with description, language and
        (x,y) 2-tuples with the bounding box coordinates.
        Appends also a list with the corresponding headers.
        Returns the two lists created.
        """
        types,vtx = ["DESCRIPTION","LANGUAGE"],[]

        texts = self.response.text_annotations
        for i,text in enumerate(texts):
            vx = []
            vx.append(text.description)
            vx.append(text.locale)

            for v,vertex in enumerate(text.bounding_poly.vertices):
                vx.append((vertex.x,vertex.y))
                if i==0: types.append("BBOX_{}".format(v+1))
            vtx.append(vx)

        return types,vtx
    
    def pages(self):
        """
        Loops on the detected pages. For each,
        appends a list with language, confidence, 
        height and width.
        Appends also a list with the corresponding headers.
        Returns the two lists created.
        """
        types,vtx = ["LANGUAGE","CONFIDENCE","HEIGHT","WIDTH"],[]

        pages = self.response.full_text_annotation.pages
        for page in pages:
            
            vx,lang,conf = [],[],[]

            for p in page.property.detected_languages:
                lang.append(p.language_code)
                conf.append(p.confidence)
            
            vx.append(lang)
            vx.append(conf)
            vx.append(page.height)
            vx.append(page.width)

            vtx.append(vx)

        return types,vtx
    
    def blocks(self):
        """
        Loops on the blocks in the detected pages. For each,
        appends a list with description, language, confidence, block type 
        and (x,y) 2-tuples with the bounding box coordinates.
        Appends also a list with the corresponding headers.
        Returns the two lists created.
        """
        types,vtx = ["BLOCK_TYPE","LANGUAGE","CONFIDENCE"],[]

        pages = self.response.full_text_annotation.pages
        for p,page in enumerate(pages):
            for b,block in enumerate(page.blocks):
                
                vx,lang,conf = [],[],[]
                
                vx.append(block.block_type)

                properties = block.property.detected_languages
                for prop in properties:
                   lang.append(prop.language_code)
                   conf.append(prop.confidence)

                vx.append(lang)
                vx.append(conf)
                
                vertices = block.bounding_box.vertices
                for v,vertex in enumerate(vertices):
                    vx.append((vertex.x,vertex.y))
                    if [p,b]==[0,0]: types.append('BBOX_{}'.format(v+1))

                vtx.append(vx)
            
        return types,vtx
    
    def paragraphs(self):
        """
        Loops on the paragraphs in the blocks in the detected pages. For each,
        appends a list with description, language, confidence
        and (x,y) 2-tuples with the bounding box coordinates.
        Appends also a list with the corresponding headers.
        Returns the two lists created.
        """
        types,vtx = ["LANGUAGE","CONFIDENCE"],[]

        pages = self.response.full_text_annotation.pages
        for p,page in enumerate(pages):
            for b,block in enumerate(page.blocks):
                for par,paragraph in enumerate(block.paragraphs):
                    
                    vx,lang,conf = [],[],[]
 
                    properties = paragraph.property.detected_languages
                    for prop in properties:
                        lang.append(prop.language_code)
                        conf.append(prop.confidence)
                    
                    vx.append(lang)
                    vx.append(conf)
                
                    vertices = paragraph.bounding_box.vertices
                    for v,vertex in enumerate(vertices):
                        vx.append((vertex.x,vertex.y))
                        if [p,b,par]==[0,0,0]: types.append('BBOX_{}'.format(v+1))

                    vtx.append(vx)
            
        return types,vtx

    def words(self):
        """
        Loops on the words in the paragraphs 
        in the blocks in the detected pages. For each,
        appends a list with language
        and (x,y) 2-tuples with the bounding box coordinates.
        Appends also a list with the corresponding headers.
        Returns the two lists created.
        """
        types,vtx = ["LANGUAGE"],[]

        pages = self.response.full_text_annotation.pages
        for p,page in enumerate(pages):
            for b,block in enumerate(page.blocks):
                for par,paragraph in enumerate(block.paragraphs):
                    for w,word in enumerate(paragraph.words):

                        vx,lang = [],[]
 
                        properties = word.property.detected_languages
                        for prop in properties:
                            lang.append(prop.language_code)
                        
                        vx.append(lang)

                        vertices = word.bounding_box.vertices
                        for v,vertex in enumerate(vertices):
                            vx.append((vertex.x,vertex.y))
                            if [p,b,par,w]==[0,0,0,0]: types.append('BBOX_{}'.format(v+1))

                        vtx.append(vx)
            
        return types,vtx

    
    def symbols(self):
        """
        Loops on the symbols in the words, in the paragraphs,
        in the blocks, in the detected pages. For each,
        appends a list with text, language
        and (x,y) 2-tuples with the bounding box coordinates.
        Appends also a list with the corresponding headers.
        Returns the two lists created.
        """
        types,vtx = ["TEXT","LANGUAGE"],[]

        pages = self.response.full_text_annotation.pages
        for p,page in enumerate(pages):
            for b,block in enumerate(page.blocks):
                for par,paragraph in enumerate(block.paragraphs):
                    for w,word in enumerate(paragraph.words):
                        for s,symbol in enumerate(word.symbols):
                            
                            vx,lang = [],[]

                            vx.append(symbol.text)
 
                            properties = symbol.property.detected_languages
                            for prop in properties:
                                lang.append(prop.language_code)

                            vx.append(lang)
                            
                            vertices = symbol.bounding_box.vertices
                            for v,vertex in enumerate(vertices):
                                vx.append((vertex.x,vertex.y))
                                if [p,b,par,w,s]==[0,0,0,0,0]: types.append('BBOX_{}'.format(v+1))

                            vtx.append(vx)
            
        return types,vtx
    


    def landmarks(self):
        """
        Loops on the detected landmarks. For each,
        appends a list with: 
        - name
        - confidence
        - (x,y) 2-tuples with coordinates of the bounding box
        - 2-tuple with latitude and logitude.
        Appends also a list with the corresponding headers.
        Returns the two lists created.
        """
        types,vtx = ["DESCRIPTION","CONFIDENCE"],[]

        lands = self.response.landmark_annotations
        for i,landmark in enumerate(lands): 
            vx = []
            vx.append(landmark.description)
            vx.append(landmark.score)

            for v,vertex in enumerate(landmark.bounding_poly.vertices):
                vx.append((vertex.x,vertex.y))
                if i==0: types.append("BBOX_{}".format(v+1))
    
            for l,ll in enumerate(landmark.locations):
                if i==0: types.append("LATITUDE_LONGITUDE_{}".format(l+1))
                vx.append((ll.lat_lng.latitude,ll.lat_lng.longitude))

            vtx.append(vx)

        return types,vtx

    def labels(self):
        """
        Loops on the detected lables. For each,
        appends a list with name, 
        confidence and topicality.
        Appends also a list with the corresponding headers.
        Returns the two lists created.
        """
        types,vtx = ["DESCRIPTION","SCORE","TOPICALITY"],[]

        labels = self.response.label_annotations
        for label in labels:
            vx = []
            vx.append(label.description)
            vx.append(label.score)
            vx.append(label.topicality)

            vtx.append(vx)

        return types,vtx
    
    def web_entities(self):
        """
        Loops on the detected web entities. For each,
        appends a list with description and score.
        Appends also a list with the corresponding headers.
        Returns the two lists created.
        """
        types,vtx = ["DESCRIPTION","SCORE"],[]

        entities = self.response.web_detection.web_entities
        for i,entity in enumerate(entities):
            vx = []
            vx.append(entity.description)
            vx.append(entity.score)
            vtx.append(vx)

        return types,vtx

    def matching_images(self):
        """
        Loops on the detected pages with matching images. 
        For each, appends a list with the page title and url,
        plus the fully or partially matched images found (url).
        Appends also a list with the corresponding headers.
        Returns the two lists created.
        """
        types,vtx = ["URL","PAGE_TITLE","TYPE","MATCHING_IMAGE"],[]

        pages = self.response.web_detection.pages_with_matching_images
        
        for i,page in enumerate(pages):
            for v,img in enumerate(page.partial_matching_images):
                vx = []
                vx.append(page.url)
                vx.append(page.page_title)
                vx.append("partial matching")
                vx.append(img.url)
                vtx.append(vx)

            for v,img in enumerate(page.full_matching_images):
                vx = []
                vx.append(page.url)
                vx.append(page.page_title)
                vx.append("full matching")
                vx.append(img.url)
                vtx.append(vx)

        return types,vtx
    
    def similar_images(self):
        """
        Loops on the suggested similar images. For each,
        appends a list with the url of that image.
        Appends also a list with the corresponding headers.
        Returns the two lists created.
        """
        types,vtx = ["URL"],[]

        images = self.response.web_detection.visually_similar_images
        for image in images:
            vx = []
            vx.append(image.url)
            vtx.append(vx)

        return types,vtx

    
    def df_options(self):
        print('Possible DataFrame Options: \n* '+'\n* '.join(self.df_types))
        return
    
    def prepare_face_df(self,types,vertex,name):
        """
        Prepares DataFrame 
        specific for the case 'face' detection.
        """
        # First: prepare the column names
        # IMAGE_NAME is the first column
        # followed by all the point's name provided
        columns=['IMAGE_NAME']
        for t in types: columns.append(t)

        # Second: prepare the rows with values
        # Loop on each face detected: image name 
        # is always the first value
        # followed by all the point's x,y coordinates
        rows = []
        for face in vertex:
            row = [name]
            for vtx in face: row.append(vtx)  
            rows.append(row)
        
        return pd.DataFrame(rows, columns=columns)
    
    def to_df(self,option=None,name='image'):
        """
        Parameters:
        - option: a string in ['face landmarks','face','head','angles',
                               'objects','landmarks','logos','labels',
                               'colors', 'crop hints','texts','pages',
                               'blocks','paragraphs','words','symbols']
          precifing the type of information to dump in the DataFrame
        - name: (optional) the name of the image used in the request. 
          default is 'image'.

        Returns:
        a DataFrame with information for the specific option.
        """

        if option is None: raise Exception('Required Parameter "option".\nYou should specify one of the following: \n* '+'\n* '.join(self.df_types))

        # retrieve the information based on the different options
        types,vertex = [],[]
        if option in self.df_types:
            types, vertex = self.methods_dict[option]()
        else:
            raise Exception('The option you specified is not valid.\nYou should specify one of the following: \n* '+'\n* '.join(self.df_types))

        # prepare the specific DataFrame
        if self.request_category == 'face':
            outdf=self.prepare_face_df(types,vertex,name)
    
        return outdf






