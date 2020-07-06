export CUDA_VISIBLE_DEVICES=0
python3 main_chexpert.py --img_type Frontal \
                         --num_workers 8 \
                         --batch_size 8 \
                         --iters_per_eval 1000 \
                         --gpu_ids 0 \
                         --num_epoch 10 \
                         --resize_shape 320 \
                         --crop_shape 320 \
                         --optimizer adam \
                         --lr 0.001 \

