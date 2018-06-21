#!/bin/bash
cd ..

python HardNet.py --dataroot /local/temporary/mishkdmy/descriptors/PhotoTourism/ \
--w1bsroot /local/temporary/mishkdmy/descriptors/wxbs-descriptors-benchmark/code/ \
--fliprot=False --experiment-name=/liberty_train/ \
--loss=triplet_margin --batch-reduce=min | tee -a log_HardNet_Lib_as_is.log

python HardNet.py --dataroot /local/temporary/mishkdmy/descriptors/PhotoTourism/ \
--w1bsroot /local/temporary/mishkdmy/descriptors/wxbs-descriptors-benchmark/code/  \
--fliprot=False --experiment-name=/liberty_train/ --loss=triplet_margin \
--batch-reduce=average | tee -a log_HardNet_Lib_average_negative.log

python HardNet.py --dataroot /local/temporary/mishkdmy/descriptors/PhotoTourism/ \
--w1bsroot /local/temporary/mishkdmy/descriptors/wxbs-descriptors-benchmark/code/ \
--fliprot=False --experiment-name=/liberty_train/ --loss=triplet_margin \
--batch-reduce=random | tee -a log_HardNet_Lib_random_negative.log

python HardNet.py --dataroot /local/temporary/mishkdmy/descriptors/PhotoTourism/ \
--w1bsroot /local/temporary/mishkdmy/descriptors/wxbs-descriptors-benchmark/code/ \
--fliprot=False --experiment-name=/liberty_train/ --loss=softmax --batch-reduce=min | tee -a log_HardNet_Lib_softmax.log
