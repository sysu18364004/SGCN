## How to Use
### Reproducing results
Modify the dataset path in `build_dataset.py` and other paramters in `main.py`
Simply run `python main.py --train_size 1.0`
### Arguments description
| Argument     | Default   | Description |
| ----------- | ----------- |----------- |
| train_size  | 1 | If it is larger than 1, it means the number of training samples. If it is from 0 to 1, it means the proportion of the original training set.        |
| test_size  | 1 | If it is larger than 1, it means the number of training samples. If it is from 0 to 1, it means the proportion of the original test set.|
| remove_limit  | 10 | Remove the words showing fewer than 10 times |
| use_gpu  | 1 | Whether to use GPU, 1 means True and 0 means False. If True and no GPU available, will use CPU instead. |
| shuffle_seed  | None | If not specified, train/val is shuffled differently in each experiment. |
| hidden_dim  | 300 | The hidden dimension of GCN model |
| dropout  | 0.5 | The dropout rate of GCN model |
| learning_rate  | 0.02 | Learning rate, and the optimizer is Adam |
| weight_decay  | 0 | Weight decay, normally it is 0 |
| epochs  | 1500 | Number of maximum epochs |
| multiple_times  | 10 | Running multiple experiments, each time the train/val split is different |
| easy_copy  | 1 | For easy copy of the experiment results. 1 means True and 0 means False. |
| lambda1  | 0.5 | The parameter that balance the 1 label and 0 label |
| dict_path  | dict_rmsc.json | The word dict of the dataset calculated by LSTM |
| emb_path  | ./embeddings/aapd | Path of the saved embedding of each document in the dataset |
## Acknowledgement
Part of the code is inspired by https://github.com/tkipf/pygcn , https://github.com/yao8839836/text_gcn and https://github.com/usydnlp/InductTGCN, but has been modified.
