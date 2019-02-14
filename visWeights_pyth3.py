# takes a saved SOM weight matrix and creates a png visualizezation of it
# parameters: 
# nIn - how many distinct blocks does one input contain (in the sense of DynaMos Connections, ie how many maps does the SOM connection map to a 2D rep?)
# outX outY - SOM domensions
# blockX - size of a single block. For simplicity we assume here that all blocks have equal size, need not always hold PROBLEM
# border - how many pixels in the visuialization between individual input blocks. The border between total RFs of a sin,gle
#          output are 2 times that value

import numpy,os,sys;
#import cv2	#for python 2.7

# filename nIn outX outY blockX blockY

dataf = numpy.load(sys.argv[1]) ;
data = dataf['arr_0'] ;
sh = data.shape
print("read", sh)

print("wmi/max=", data.min(), data.max())

nIn  = int(sys.argv[2])
outX  = int(sys.argv[3])
outY  = int(sys.argv[4])
blockX  = int(sys.argv[5])
border  = int(sys.argv[6])

len1d=sh[1] /nIn;
#len1d=sh[1] /(outX*outY);
blockY = len1d/blockX ;

print('single block size is', blockY, blockX) ;

complRFX =  (nIn*(blockX+border)-border+2*border) ;
complRFY =  blockY + 2*border

print("one vis RF is ", complRFY, complRFX) ;
totalX = outX * complRFX ;
totalY = outY * complRFY ;
print("Creating", totalY, totalX)

img = numpy.zeros([int(totalY),totalX]) ;
img [:,:] = 0.3*255. ;

data = (data-data.min())/(data.max()-data.min()) * 255. ;
#data = (data-data.min())/(data.min()-data.max()) * 255 ;

print(outY, outX, nIn, len1d, blockY, blockX)

for outy in range(0,outY):
  for outx in range(0,outX):
    for inblock in range(0,nIn):
      wBlock = (data[outy*outX+outx,:][inblock*int(len1d):(inblock+1)*int(len1d)]).reshape(int(blockY),int(blockX))
       
      #print wBlock.shape, blockY,blockX
      starty = outy*complRFY ;
      startx = outx*complRFX + inblock*(blockX+border);
      img[int(starty):int(starty+blockY),int(startx):int(startx+blockX)] = wBlock ;



#sigmas = (numpy.load('sigmas.npz')['arr_0']).reshape ([outY,outX]) ;
#sigmas/=sigmas.max() ;
#sigmas*= 255.
from PIL import Image

result = Image.fromarray((img * 255).astype(numpy.uint8))
result.save('w.png')

#cv2.imwrite("w.png", (img).astype("uint8"))	## For python 2.7
#cv2.imwrite("sigmas.png", sigmas );


  
            



