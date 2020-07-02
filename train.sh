export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 main_supcon.py --batch_size 16 \
                      --learning_rate 0.0001 \
                      --lr_decay_rate 0.00001 \
                      --trial 3 \
                      --model densenet121 \
                      --match_type all \
                      --epoch 10 \
                      --dataset chexpert \
                      --temp 0.07 \
                      --cosine


