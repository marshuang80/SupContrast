export CUDA_VISIBLE_DEVICES=0,1,2
python3 -W ignore main_chexpert.py --img_type Frontal \
                         --num_workers 8 \
                         --batch_size 48 \
                         --iters_per_eval 100 \
                         --gpu_ids 0,1,2 \
                         --num_epoch 3 \
                         --resize_shape 320 \
                         --crop_shape 320 \
                         --optimizer adam \
                         --lr 0.001 \

