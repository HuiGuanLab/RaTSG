python main.py --task charades_RF --save_dir datasets --model_dir ckpt --model_name RaTSG --test_name charades_sta_test2.0.json --val_name charades_sta_val2.0.json --batch_size 16 --init_lr 2e-3  --warmup_proportion 0. --epochs 100 --mode train --threshold 0.5 --gama 6. --beta 6. --n_heads 8