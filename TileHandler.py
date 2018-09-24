'''

###############################################################################
Author: Tanner Bohn, Joey

Purpose: used to generate random Bongard-like image tiles for training a ML
          model which is then used to produce features for solving Bongard
          problems

Usage:
```
from TileHandler import *

# set shape value to None to prevent drawing that shape
TH = TileHandler(n_shapes = 1, size = 300, line_k_range = (1, 2),
                 circle = True, dot = True, curve = True, polygon_k_range = None,
                 ellipses = True, equilateral_polygon_range = (3, 5))

new_tile = TH.generate_tile()

# to view the tile
new_tile['img'].show()

# to convert the tile img and description to numpy arrays
img_vec, description_vec = TH.preprocess_tile(new_tile)


```

###############################################################################
'''
from __future__ import print_function


import math
import random
import numpy as np
from PIL import Image, ImageFilter, ImageDraw

from utils import *

class TileHandler:
    
    def __init__(self, n_shapes = 1, size = 300, line_k_range = None,
                 circle = None, dot = None, curve = None, polygon_k_range = None,
                 ellipses = None, equilateral_polygon_range = None):
        # if shape parameter is None, then do not draw that shape
        self.n_shapes = n_shapes

        self.size = size

        self.line_k_range = line_k_range

        self.polygon_k_range = polygon_k_range

        self.ellipses = ellipses

        self.circle = circle

        self.dot = dot

        self.curve = curve

        self.equilateral_polygon_range = equilateral_polygon_range

        self.valid_shapes = []

        if line_k_range:
            self.valid_shapes.append("line")

        if circle:
            self.valid_shapes.append('circle')

        if curve:
            self.valid_shapes.append("curve")

        if dot:
            self.valid_shapes.append("dot")

        if polygon_k_range:
            self.valid_shapes.append("polygon")

        if ellipses:
            self.valid_shapes.append("ellipses")

        if equilateral_polygon_range:
            self.valid_shapes.append("equilateral_polygon")
             
    def vectorize_img(self, img):
        dim = img.size[0]
        pix = np.array(img)/255.
        img_vec = pix.reshape(dim, dim, 1)
        return img_vec
             
    def preprocess_tile(self, tile):
        img_vec = self.vectorize_img(tile['img'])
        description_vec = np.array([tile['description'][key] for key in sorted(tile['description'])])
        return img_vec, description_vec


    def generate_tile(self):

        # https://pillow.readthedocs.io/en/3.1.x/reference/ImageDraw.html

        # http://cglab.ca/~sander/misc/ConvexGeneration/convex.html

        # size: side length of images


        n_shapes = self.n_shapes

        size = self.size

        line_k_range = self.line_k_range

        ellipses = self.ellipses

        polygon_k_range = self.polygon_k_range

        equilateral_polygon_range = self.equilateral_polygon_range

        dot = self.dot

        curve = self.curve

        circle = self.circle


        description = dict()


        if self.line_k_range:
            for k in range(line_k_range[0], line_k_range[1]+1):
                description['line_{}'.format(k)] = 0


        if self.polygon_k_range:
            for k in range(polygon_k_range[0], polygon_k_range[1]+1):
                description['polygon_filled_{}'.format(k)] = 0

            for k in range(polygon_k_range[0], polygon_k_range[1]+1):
                description['polygon_unfilled_{}'.format(k)] = 0


        if self.dot:
            description['dot'] = 0

        if self.curve:
            description['curve'] = 0

        if self.circle:
            description['circle_filled'] = 0

            description['circle_unfilled'] = 0

        if self.ellipses:
            description['ellipses_filled'] = 0

            description['ellipses_unfilled'] = 0

        if self.equilateral_polygon_range:
            for k in range(equilateral_polygon_range[0], equilateral_polygon_range[1]+1):

                description['equilateral_polygon_filled_{}'.format(k)] = 0

            for k in range(equilateral_polygon_range[0], equilateral_polygon_range[1]+1):

                description['equilateral_polygon_unfilled_{}'.format(k)] = 0

        img_size = (size, size)

        img = Image.new('L', (img_size), "white")

        img_draw = ImageDraw.Draw(img, 'L')

        for i_shape in range(n_shapes):

            shape_type = random.choice(self.valid_shapes)

            if (shape_type == "line") :

                k = random.randint(line_k_range[0], line_k_range[1]) 
                
                k_p = k + 2
                
                radius = rand_uniform(0.1, 0.2)

                origin = (rand_uniform(radius, 1.-radius), rand_uniform(radius, 1.-radius))