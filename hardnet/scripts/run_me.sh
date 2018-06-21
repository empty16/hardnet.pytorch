#!/bin/bash

wxbs_folder="wxbs-descriptors-benchmark"

if [ ! -d "$wxbs_folder" ]; then
    git clone https://github.com/ducha-aiki/wxbs-descriptors-benchmark.git
    chmod +x wxbs-descriptors-benchmark/data/download_W1BS_dataset.sh
    ./wxbs-descriptors-benchmark/data/download_W1BS_dataset.sh
    mv W1BS wxbs-descriptors-benchmark/data/
fi
cd hardnet

echo "HardNet demo ..."
# python HardNet_exp.py --fliprot=False --gor=True 

# python HardNet_exp.py --fliprot=True --gor=True 

# n-triplets=5000,000
#SRN first
python HardNet_exp.py --fliprot=False --gor=True --use-srn=True --gpu-id=4

python HardNet_exp.py --fliprot=False --gor=True --use-arf=True --gpu-id=4

python HardNet_exp.py --fliprot=True --gor=True --use-srn=True --gpu-id=4 

python HardNet_exp.py --fliprot=True --gor=True --use-arf=True --gpu-id=4


# n-triplets=30,000,000
python HardNet_exp.py --fliprot=False --gor=True --use-srn=True --n-triplets=30000000 --gpu-id=4

python HardNet_exp.py --fliprot=False --gor=True --use-arf=True --n-triplets=30000000 --gpu-id=4

python HardNet_exp.py --fliprot=True --gor=True --use-srn=True --n-triplets=30000000 --gpu-id=4

python HardNet_exp.py --fliprot=True --gor=True --use-arf=True --n-triplets=30000000 --gpu-id=4
