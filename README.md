## Identifying 2D brain slices in a 3D reference atlas using Siamese Networks

## Data

The [Nissl](http://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/ara_nissl/) and [average template](http://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/average_template/) volumes from the Allen Reference Atlas can be downloaded directly from the Allen Institute server. 

2D plates from a volume can be sliced at different angles and pre-processed using the script `utils/atlas_utils.py`



## Dataset Directory Structure
        
	|-- data
	    |-- avg                     # The name of volume
	        |-- 760                 # The position of 2D plate
	            |-- 760_-15.jpg     # The position of 2D plate and the angle of rotation
	            |-- ...
	        |-- 765
			|-- ...
		|-- nissl
	        |-- 760
	            |-- 760_-15.jpg
	            |-- ...
	        |-- 765
			|-- ...

Paths to the images are specified in the file `paths.py`.

## Training

Check file `trainer/train.py` for training Siamese CNNs.