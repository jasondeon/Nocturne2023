@echo off
call .\venv\Scripts\activate.bat && python train.py^
 --train_file "maestro_train.pickle"^
 --test_file "maestro_test.pickle"^
 --vocab_size 396^
 --seq_len 512^
 --embed_dim 512^
 --layers 8^
 --batch_size 8^
 --lr 3e-5^
 --weight_decay 1e-5^
 --dropout 0.1^
 --pretrained 1^
 --device "cuda:0"^
 --print_every 100^
 --checkpoint_path "model.pt"