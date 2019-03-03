# -*- coding: utf-8 -*-
import numpy as np
import pickle
import os
import string

def pad_dataset(dataset, bs):
    n_records = len(dataset)
    n_padded = bs - n_records % bs
    new_dataset = [t for t in dataset]
    new_dataset.extend(dataset[:n_padded])
    return new_dataset

def pad_seq(dataset, field, max_len, symbol):
    n_records = len(dataset)
    for i in range(n_records):
        assert isinstance(dataset[i][field], list)
        while len(dataset[i][field]) < max_len:
            dataset[i][field].append(symbol)
    return dataset

def read(path, attention_alpha=None):
    dataset = []
    sid = 0 
    with open(path, encoding='utf-8') as fp:
        for line in fp:
            record = {}
            tokens = line.strip().split()
            words, target_words = [], []
            d = []
            find_label = False
            for t in tokens:
                if '/p' in t or '/n' in t or '/0' in t:
                    end = 'xx'
                    y = 0
                    if '/p' in t:
                        end = '/p'
                        y = 1
                    elif '/n' in t:
                        end = '/n'
                        y = 0
                    elif '/0' in t:
                        end = '/0'
                        y = 2
                    words.append(t.strip(end))
                    target_words.append(t.strip(end))
                    if not find_label:
                        find_label = True
                        record['y'] = y
                        left_most = right_most = tokens.index(t)
                    else:
                        right_most += 1
                else:
                    words.append(t)
            for pos in range(len(tokens)):
                if pos < left_most:
                    d.append(right_most - pos)
                else:
                    d.append(pos - left_most)
            record['sent'] = line.strip()
            record['words'] = words.copy()
            record['twords'] = target_words.copy()
            record['wc'] = len(words)  
            record['wct'] = len(record['twords'])  
            record['dist'] = d.copy()  
            record['sid'] = sid
            record['beg'] = left_most
            record['end'] = right_most + 1
            sid += 1
            dataset.append(record)
    return dataset

def load_data(ds_name, a1_name=None, a2_name=None, a3_name=None, a4_name=None, a5_name=None, erasing_or_final=True):
    train_file = './dataset/%s/train.txt' % ds_name
    test_file = './dataset/%s/test.txt' % ds_name
    train_set = read(path=train_file)
    test_set = read(path=test_file)

    train_wc = [t['wc'] for t in train_set]
    test_wc = [t['wc'] for t in test_set]
    max_len = max(train_wc) if max(train_wc) > max(test_wc) else max(test_wc) 

    train_t_wc = [t['wct'] for t in train_set]
    test_t_wc = [t['wct'] for t in test_set]
    max_len_target = max(train_t_wc) if max(train_t_wc) > max(test_t_wc) else max(test_t_wc)

    train_set = pad_seq(dataset=train_set, field='dist', max_len=max_len, symbol=-1)
    test_set = pad_seq(dataset=test_set, field='dist', max_len=max_len, symbol=-1)

    train_set = calculate_position_weight(dataset=train_set)
    test_set = calculate_position_weight(dataset=test_set)
    
    vocab = build_vocab(dataset=train_set+test_set)

    train_set = set_wid(dataset=train_set, vocab=vocab, max_len=max_len)
    test_set = set_wid(dataset=test_set, vocab=vocab, max_len=max_len)
    
    train_set = set_tid(dataset=train_set, vocab=vocab, max_len=max_len_target)
    test_set = set_tid(dataset=test_set, vocab=vocab, max_len=max_len_target)
    
    alphas1 = np.loadtxt(a1_name) if a1_name != None else None
    alphas2 = np.loadtxt(a2_name) if a2_name != None else None
    alphas3 = np.loadtxt(a3_name) if a3_name != None else None
    alphas4 = np.loadtxt(a4_name) if a4_name != None else None
    alphas5 = np.loadtxt(a5_name) if a5_name != None else None
    
    if erasing_or_final:
        train_set = get_attention_mask_forerasing(dataset=train_set, alphas1=alphas1, alphas2=alphas2, alphas3=alphas3, alphas4=alphas4, alphas5=alphas5, a1_name=a1_name, a2_name=a2_name, a3_name=a3_name, a4_name=a4_name, a5_name=a5_name)
    else:
        train_set = get_attention_mask_final(dataset=train_set, alphas1=alphas1, alphas2=alphas2, alphas3=alphas3, alphas4=alphas4, alphas5=alphas5, a1_name=a1_name, a2_name=a2_name, a3_name=a3_name, a4_name=a4_name, a5_name=a5_name)
    test_set = get_attention_mask_fortest(dataset=test_set)
    
    dataset = [train_set, test_set]

    return dataset, vocab

def build_vocab(dataset):
    vocab = {}
    idx = 1
    n_records = len(dataset)
    for i in range(n_records):
        for w in dataset[i]['words']:
            if w not in vocab:
                vocab[w] = idx
                idx += 1
        for w in dataset[i]['twords']:
            if w not in vocab:
                vocab[w] = idx
                idx += 1
    return vocab

def set_wid(dataset, vocab, max_len):
    n_records = len(dataset)
    for i in range(n_records):
        sent = dataset[i]['words']
        dataset[i]['wids'] = word2id(vocab, sent, max_len)
    return dataset

def set_tid(dataset, vocab, max_len):
    n_records = len(dataset)
    for i in range(n_records):
        sent = dataset[i]['twords']
        dataset[i]['tids'] = word2id(vocab, sent, max_len)
    return dataset

def word2id(vocab, sent, max_len):
    wids = [vocab[w] for w in sent]
    while len(wids) < max_len:
        wids.append(0)
    return wids

def get_embedding(vocab, ds_name, dim_w):
    emb_file = './embeddings/glove_840B_300d.txt'
    pkl = './embeddings/%s_840B.pkl' % ds_name

    print("Load embeddings from %s or %s..." % (emb_file, pkl))
    n_emb = 0
    if not os.path.exists(pkl):
        embeddings = np.zeros((len(vocab)+1, dim_w), dtype='float32')
        with open(emb_file, encoding='utf-8') as fp:
            for line in fp:
                eles = line.strip().split()
                w = eles[0]
                n_emb += 1
                if w in vocab:
                    try:
                        embeddings[vocab[w]] = [float(v) for v in eles[1:]]
                    except ValueError:
                        pass
        print("Find %s word embeddings!!" % n_emb)
        pickle.dump(embeddings, open(pkl, 'wb'))
    else:
        embeddings = pickle.load(open(pkl, 'rb'))
    return embeddings

def build_dataset(ds_name, bs, dim_w, a1_name=None, a2_name=None, a3_name=None, a4_name=None, a5_name=None, erasing_or_final=True):
    dataset, vocab = load_data(ds_name=ds_name, a1_name=a1_name, a2_name=a2_name, a3_name=a3_name, a4_name=a4_name, a5_name=a5_name, erasing_or_final=erasing_or_final)
    n_train = len(dataset[0])
    n_test = len(dataset[1])
    embeddings = get_embedding(vocab, ds_name, dim_w)
    for i in range(len(embeddings)):
        if i and np.count_nonzero(embeddings[i]) == 0:
            embeddings[i] = np.random.uniform(-0.25, 0.25, embeddings.shape[1])
    embeddings = np.array(embeddings, dtype='float32')
    train_set = pad_dataset(dataset=dataset[0], bs=bs)
    test_set = pad_dataset(dataset=dataset[1], bs=bs)
    return [train_set, test_set], embeddings, n_train, n_test

def calculate_position_weight(dataset):
    tmax = 40
    n_tuples = len(dataset)
    for i in range(n_tuples):
        dataset[i]['pw'] = []
        weights = []
        for w in dataset[i]['dist']:
            if w == -1:
                weights.append(0.0)
            elif w > tmax:
                weights.append(0.0)
            else:
                weights.append(1.0 - float(w) / tmax)
        dataset[i]['pw'].extend(weights)
    return dataset

def get_attention_mask_forerasing(dataset, alphas1=None, alphas2=None, alphas3=None, alphas4=None, alphas5=None, a1_name=None, a2_name=None, a3_name=None, a4_name=None, a5_name=None):
    n_tuples = len(dataset)
    good_tuples = [0, 0, 0, 0, 0]
    bad_tuples = [0, 0, 0, 0, 0]
    max_entroy = 1.0
    
    for i in range(n_tuples):
        dataset[i]['mask'] = []
        masks = [] 
        for w in dataset[i]['dist']:
            if w == -1:
                masks.append(0.0)
            else:
                masks.append(1.0)
    
        if a1_name != None:
            this_alpha = alphas1[i]
            this_alpha = this_alpha[this_alpha!=0]
            if - np.sum(np.log2(abs(this_alpha)) * abs(this_alpha)) < max_entroy:
                index = abs(alphas1[i]).argmax()
                masks[index] = 0.0 
                dataset[i]['wids'][index] = 0 
                if alphas1[i][index] > 0:
                    good_tuples[0] += 1
                else:
                    bad_tuples[0] += 1

        if a2_name != None:
            this_alpha = alphas2[i]
            this_alpha = this_alpha[this_alpha!=0]
            if - np.sum(np.log2(abs(this_alpha)) * abs(this_alpha)) < max_entroy:
                index = abs(alphas2[i]).argmax()
                masks[index] = 0.0 
                dataset[i]['wids'][index] = 0 
                if alphas2[i][index] > 0:
                    good_tuples[1] += 1
                else:
                    bad_tuples[1] += 1

        if a3_name != None:
            this_alpha = alphas3[i]
            this_alpha = this_alpha[this_alpha!=0]
            if - np.sum(np.log2(abs(this_alpha)) * abs(this_alpha)) < max_entroy:
                index = abs(alphas3[i]).argmax()
                masks[index] = 0.0 
                dataset[i]['wids'][index] = 0 
                if alphas3[i][index] > 0:
                    good_tuples[2] += 1
                else:
                    bad_tuples[2] += 1

        if a4_name != None:
            this_alpha = alphas4[i]
            this_alpha = this_alpha[this_alpha!=0]
            if - np.sum(np.log2(abs(this_alpha)) * abs(this_alpha)) < max_entroy:
                index = abs(alphas4[i]).argmax()
                masks[index] = 0.0 
                dataset[i]['wids'][index] = 0 
                if alphas4[i][index] > 0:
                    good_tuples[3] += 1
                else:
                    bad_tuples[3] += 1
    
        if a5_name != None:
            this_alpha = alphas5[i]
            this_alpha = this_alpha[this_alpha!=0]
            if - np.sum(np.log2(abs(this_alpha)) * abs(this_alpha)) < max_entroy:
                index = abs(alphas5[i]).argmax()
                masks[index] = 0.0 
                dataset[i]['wids'][index] = 0 
                if alphas5[i][index] > 0:
                    good_tuples[4] += 1
                else:
                    bad_tuples[4] += 1
    
        dataset[i]['mask'].extend(masks)

    print ("Erasing ratio:\n")
    print ("1: good: %.2f, bad: %.2f\n" % (float(good_tuples[0]) / n_tuples, float(bad_tuples[0]) / n_tuples))
    print ("2: good: %.2f, bad: %.2f\n" % (float(good_tuples[1]) / n_tuples, float(bad_tuples[1]) / n_tuples))
    print ("3: good: %.2f, bad: %.2f\n" % (float(good_tuples[2]) / n_tuples, float(bad_tuples[2]) / n_tuples))
    print ("4: good: %.2f, bad: %.2f\n" % (float(good_tuples[3]) / n_tuples, float(bad_tuples[3]) / n_tuples))
    print ("5: good: %.2f, bad: %.2f\n" % (float(good_tuples[4]) / n_tuples, float(bad_tuples[4]) / n_tuples))
    
    return dataset

def get_attention_mask_final(dataset, alphas1=None, alphas2=None, alphas3=None, alphas4=None, alphas5=None, a1_name=None, a2_name=None, a3_name=None, a4_name=None, a5_name=None):
    n_tuples = len(dataset)
    good_tuples = [0, 0, 0, 0, 0]
    bad_tuples = [0, 0, 0, 0, 0]
    max_entroy = 1.0
    
    for i in range(n_tuples):
        dataset[i]['mask'] = []
        dataset[i]['amask'] = []
        dataset[i]['avalue'] = []
        
        masks = [] 
        amasks = []
        avalues = []
        for w in dataset[i]['dist']:
            if w == -1:
                masks.append(0.0)
            else:
                masks.append(1.0)
            amasks.append(0.0)
            avalues.append(0.0)
        
        if a1_name != None:
            this_alpha = alphas1[i]
            this_alpha = this_alpha[this_alpha!=0]
            if - np.sum(np.log2(abs(this_alpha)) * abs(this_alpha)) < max_entroy:
                index = abs(alphas1[i]).argmax()
                amasks[index] = 1.0
                if alphas1[i][index] > 0:
                    avalues[index] = 1.0
                    good_tuples[0] += 1
                else:
                    bad_tuples[0] += 1

        if a2_name != None:
            this_alpha = alphas2[i]
            this_alpha = this_alpha[this_alpha!=0]
            if - np.sum(np.log2(abs(this_alpha)) * abs(this_alpha)) < max_entroy:
                index = abs(alphas2[i]).argmax()
                amasks[index] = 1.0
                if alphas2[i][index] > 0:
                    avalues[index] = 1.0
                    good_tuples[1] += 1
                else:
                    bad_tuples[1] += 1
    
        if a3_name != None:
            this_alpha = alphas3[i]
            this_alpha = this_alpha[this_alpha!=0]
            if - np.sum(np.log2(abs(this_alpha)) * abs(this_alpha)) < max_entroy:
                index = abs(alphas3[i]).argmax()
                amasks[index] = 1.0
                if alphas3[i][index] > 0:
                    avalues[index] = 1.0
                    good_tuples[2] += 1
                else:
                    bad_tuples[2] += 1

        if a4_name != None:
            this_alpha = alphas4[i]
            this_alpha = this_alpha[this_alpha!=0]
            if - np.sum(np.log2(abs(this_alpha)) * abs(this_alpha)) < max_entroy:
                index = abs(alphas4[i]).argmax()
                amasks[index] = 1.0
                if alphas4[i][index] > 0:
                    avalues[index] = 1.0
                    good_tuples[3] += 1
                else:
                    bad_tuples[3] += 1
    
        if a5_name != None:
            this_alpha = alphas5[i]
            this_alpha = this_alpha[this_alpha!=0]
            if - np.sum(np.log2(abs(this_alpha)) * abs(this_alpha)) < max_entroy:
                index = abs(alphas5[i]).argmax()
                amasks[index] = 1.0
                if alphas5[i][index] > 0:
                    avalues[index] = 1.0
                    good_tuples[4] += 1
                else:
                    bad_tuples[4] += 1
    
        dataset[i]['mask'].extend(masks)
        dataset[i]['amask'].extend(amasks)
        dataset[i]['avalue'].extend(avalues)
    
    print ("Erasing ratio:\n")
    print ("1: good: %.2f, bad: %.2f\n" % (float(good_tuples[0]) / n_tuples, float(bad_tuples[0]) / n_tuples))
    print ("2: good: %.2f, bad: %.2f\n" % (float(good_tuples[1]) / n_tuples, float(bad_tuples[1]) / n_tuples))
    print ("3: good: %.2f, bad: %.2f\n" % (float(good_tuples[2]) / n_tuples, float(bad_tuples[2]) / n_tuples))
    print ("4: good: %.2f, bad: %.2f\n" % (float(good_tuples[3]) / n_tuples, float(bad_tuples[3]) / n_tuples))
    print ("5: good: %.2f, bad: %.2f\n" % (float(good_tuples[4]) / n_tuples, float(bad_tuples[4]) / n_tuples))
    
    return dataset

def get_attention_mask_fortest(dataset):
    n_tuples = len(dataset)
    for i in range(n_tuples):
        dataset[i]['mask'] = []
        masks = []
        for w in dataset[i]['dist']:
            if w == -1:
                masks.append(0.0)
            else:
                masks.append(1.0)
        dataset[i]['mask'].extend(masks)
    return dataset
