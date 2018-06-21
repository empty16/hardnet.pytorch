# To reproduce the results in HardNet-NIPS paper
1. training-set=notredame, test-set=liberty, yosemite

python HardNet.py --fliprot=False --gpu-id=4 | tee -a ../logs/log_HardNet_notredame.log

python HardNet.py --fliprot=True --experiment-name=notredame_train --training-set=notredame --gpu-id=4 | tee -a ../logs/log_HardNet_notredame_flip.log

2. training-set=liberty, test-set=notredame, yosemite

python HardNet.py --fliprot=False --experiment-name=liberty_train --training-set=liberty --gpu-id=7 --dataroot="../data/photo/" | tee -a ../logs/log_HardNet_liberty.log

python HardNet.py --fliprot=True --experiment-name=liberty_train --training-set=liberty --gpu-id=4 | tee -a ../logs/log_HardNet_liberty_flipp.log

python HardNet.py --fliprot=False --experiment-name=liberty_train_arf --training-set=liberty --gpu-id=7 --use-arf=True --orientation=8 | tee -a ../logs/log_HardNet_arf_liberty.log