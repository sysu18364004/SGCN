cd ..
python main.py --train_size 1.0 \
--lambda1 0.5 \
--resrate 3e-5 \
--lambda2 1e-6 \
--epochs 4000 \
--learning_rate 5e-4 \
--gpu_order cuda:6 \
--multiple_times 1 \
--output_dir lstm \
--remove_limit 15 \
--use_load 0 \
--emb_path ./embeddings/aapd \
--loc_path ./graphs \
--norm_edge 1