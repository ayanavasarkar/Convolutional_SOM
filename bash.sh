
#@author: ayanava

#!/bin/bash

START=0
END=99

for (( c=$START; c<=$END; c++ ));
do
	echo  "$c "

    
	python mnist_pixel_vec_tensors.py c
		

done
