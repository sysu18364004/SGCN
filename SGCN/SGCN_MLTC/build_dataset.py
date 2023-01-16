import pandas as pd
def load_tsv(filename):
    labels = []
    sentences = []
    with open(filename,'r',encoding='utf-8') as f:
        for line in f.readlines():
            # print(line.split())
            label,*word = line.split()

            labels.append(list(map(int,label)))
            sentences.append(" ".join(word))

    return sentences,labels
def samples(sentences,labels,rate):
    if rate > 1 or rate < 0:
        rate = 1
    leng = len(sentences)
    sample_size = int(leng*rate)
    return sentences[:sample_size],labels[:sample_size]
def get_dataset(args):
    train_sentences,train_labels = load_tsv('/home/zengdl/project/InductTGCN/AAPD/data/aapd_train.tsv')
    val_sentences,val_labels = load_tsv('/home/zengdl/project/InductTGCN/AAPD/data/aapd_validation.tsv')
    test_sentences,test_labels = load_tsv('/home/zengdl/project/InductTGCN/AAPD/data/aapd_test.tsv')

    train_sentences,train_labels = samples(train_sentences,train_labels,args.train_size)
    val_sentences,val_labels = samples(val_sentences,val_labels,args.train_size)
    test_sentences,test_labels = samples(test_sentences,test_labels,args.test_size)

    train_size = len(train_sentences)
    test_size = len(test_sentences)
    val_size = len(val_sentences)
    return train_sentences+val_sentences+test_sentences, train_labels+val_labels+test_labels, train_size,val_size, test_size
if __name__=="__main__":
    get_dataset(1)