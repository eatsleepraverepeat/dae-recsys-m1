# Trying ML with M1

Attempt to train a recommender system on Apple M1, using [Apple's TF fork](https://github.com/apple/tensorflow_macos) and Neural Engine unit. The model is 
denoising autoencoder, as described in [Variational Autoencoders for Collaborative Filtering, Liang et al.](https://arxiv.org/abs/1802.05814)
The data is [Echo Nest Taste Profile Subset from Million Songs Dataset](http://millionsongdataset.com/tasteprofile/).

Code works fine, but training is taking forever, especially considering that number of modeling items is very limited. 

Tested on MBP'13, 2020.

## How to run
Install ML Compute accelerated TF2 package of 0.1a3 version by following this [instructions](https://github.com/apple/tensorflow_macos/issues/153)
(installation must be done with miniforge), and then fire:
```bash
conda activate YOUR_ENVIRONMENT_NAME
conda install -c conda-forge --yes sqlalchemy==1.4.4 more-itertools=8.7.0 pandas=1.2.3 tqdm==4.59.0 && mkdir data 
wget http://millionsongdataset.com/sites/default/files/challenge/train_triplets.txt.zip data/ && unzip train_triplets.txt.zip
wget http://labrosa.ee.columbia.edu/millionsong/sites/default/files/AdditionalFiles/track_metadata.db data/
python src/data/insert_data.py
python main.py
```

## Sources
- [Echo Nest meta backup](https://github.com/MTG/echonest-backup)
- [Original implementation of Vartiational Autoencoders for Collabarative Filtering](https://github.com/dawenl/vae_cf);
- [NVIDIA NGC's Variational Autoencoder for Collabarative Filtering](https://ngc.nvidia.com/catalog/resources/nvidia:vae_for_tensorflow)
