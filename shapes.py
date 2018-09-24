from __future__ import print_function

from PIL import Image, ImageFilter, ImageDraw

import random

from matplotlib.pyplot import imshow

import random

import numpy as np

from PIL import Image, ImageFilter, ImageDraw
 

def draw_ellipse(image, bounds, width=1, outline='white', antialias=4):


 

     """
         Improved ellipse drawing function, based on PIL.ImageDraw.
         
    """

 

     # Use a single channel image (mode='L') as mask.

     # The size of the mask can be increased relative to the input image

     # to get smoother looking results.
     """
         shape.shape
    """

     mask = Image.new(

         size=[int(dim * antialias) for dim in image.size],

         mode='L', color='black')

     draw = ImageDraw.Draw(mask)

 

     # draw outer shape in white (color) and inner shape in black (transparent)

     for offset, fill in (width/-2.0, 'white'), (width/2.0, 'black'):

         left, top = [(value + offset) * antialias for value in bounds[0]]

         right, bottom = [(value - offset) * antialias for value in bounds[1]]

         draw.ellipse([left, top, right, bottom], fill=fill)

 

     # downsample the mask using PIL.Image.LANCZOS

     # (a high-quality downsampling filter).

     mask = mask.resize(image.size, Image.LANCZOS)

     # paste outline color to input image through the mask

     image.paste(outline, mask=mask)

     return image

 

def rand_uniform(m, M):

 

     return random.random()*(M - m) + m

 

class TileHandler:

 

     def __init__(self, n_shapes=10, size=100, line_k_range=(1, 3), polygon_k_range=(3, 5)):

 

         self.n_shapes = n_shapes

         self.size = size

         self.line_k_range = line_k_range

         self.polygon_k_range = polygon_k_range

 

 

 

     def preprocess_tile(self, tile):

        

         dim = tile['img'].size[0]

         pix = np.array(tile['img'])/255.

         img_vec = pix.reshape(dim, dim, 1)

         description_vec = np.array([tile['description'][key] for key in sorted(tile['description'])])

 

         return img_vec, description_vec

 

     def generate_tile(self):

         # https://pillow.readthedocs.io/en/3.1.x/reference/ImageDraw.html

         # http://cglab.ca/~sander/misc/ConvexGeneration/convex.html

        

         # size: side length of images

 

         n_shapes=self.n_shapes

         size=self.size

         line_k_range=self.line_k_range

         polygon_k_range=self.polygon_k_range

 

         description = dict()

 

         for k in range(line_k_range[0], line_k_range[1]+1):

              description['line_{}'.format(k)] = 0

 

         for k in range(polygon_k_range[0], polygon_k_range[1]+1):

              description['polygon_filled_{}'.format(k)] = 0

 

         for k in range(polygon_k_range[0], polygon_k_range[1]+1):

              description['polygon_unfilled_{}'.format(k)] = 0

 

         description['ellipses_filled'] = 0

 

         description['ellipses_unfilled'] = 0

        

 

 

 

         img_size = (size, size)

 

         img = Image.new('L', (img_size), "white")

 

         img_draw = ImageDraw.Draw(img, 'L')

 

         for i_shape in range(n_shapes):

 

              shape_type = random.choice(["line", "polygon", "ellipse"])

 

              if shape_type == "line":

 

 

 

                  k = random.randint(line_k_range[0], line_k_range[1]) # number of pieces the line is made of

                 

                  # generate normalized list of coords

                  # TODO: objects are usually localized, instead of having uniform mass distribution

                  coords = [(random.random(), random.random()) for _ in range(k+1)]

 

                  # convert coords to pixels

                  coords = [(int(x*size), int(y*size)) for x, y in coords]

 

                  img_draw.line(coords, width=5)

 

                  description['line_'+str(k)] += 1

 

              elif shape_type == "polygon":

 

                  k = random.randint(polygon_k_range[0], polygon_k_range[1]) # number of polygon sides

 

                  coords = [(random.random(), random.random()) for _ in range(k)]

 

                  # convert coords to pixels

                  coords = [(int(x*size), int(y*size)) for x, y in coords]

 

                  filled = random.random() > 0.5

                  if filled:

                       img_draw.polygon(coords, fill="black", outline="black")

 

                       description['polygon_filled_'+str(k)] += 1

                  else:

                       coords.append(coords[0])

                       img_draw.line(coords, width=5)

 

                       description['polygon_unfilled_'+str(k)] += 1

 

              elif shape_type == "ellipse":

 

                  # TODO: make these better... right now expected mass distribution is non-uniform

                  #c = (random.random(), random.random())

                  top_left = (random.random(), random.random())

                  bot_right = rand_uniform(top_left[0], 1), rand_uniform(top_left[1], 1)

 

                  coords = [top_left, bot_right]

 

                  # convert coords to pixels

                  coords = [(int(x*size), int(y*size)) for x, y in coords]

                 

                  filled = random.random() > 0.5

                  if filled:

                       img_draw.ellipse(coords, fill="black", outline="black")

 

                       description['ellipses_filled'] += 1

 

                  else:

                       img = draw_ellipse(img, coords, width=5, outline='black', antialias=4)

 

                       description['ellipses_unfilled'] += 1

 

         del img_draw

 

 

         return {"img": img, "description": description}

 

 

print("meow")

 

TH = TileHandler(n_shapes=1, size=300, line_k_range=(1, 3), polygon_k_range=(3, 5))

tile = TH.generate_tile()

 

tile_x, tile_y = TH.preprocess_tile(tile)

 

print("description:", tile['description'])

print("   -> vec:", tile_y)

tile['img'].save('test.png')

def view_cdimg(img):
    imshow(np.asarray(tile['img']))
    
TH = TileHandler(n_shapes=1, size=300, line_k_range=(1, 10), polygon_k_range=(1, 6))

view_img(TH.generate_tile()['img'])
