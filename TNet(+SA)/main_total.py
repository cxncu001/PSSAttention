# -*- coding: utf-8 -*-
import argparse
import math
import time
import os
from layer import TNet
from utils import *
from nn_utils import *
from evals import *

def train_erasing(a1_name, a2_name, a3_name, a4_name, a5_name, erasing, args):
    print ('#######################################################\n')
    print ('#######################################################\n')
    dataset, embeddings, n_train, n_test = build_dataset(ds_name=args.ds_name, bs=args.bs, dim_w=args.dim_w, a1_name=a1_name, a2_name=a2_name, a3_name=a3_name, a4_name=a4_name, a5_name=a5_name, erasing_or_final=True)

    args.dim_w = len(embeddings[1])
    print(args)
    args.embeddings = embeddings
    args.sent_len = len(dataset[0][0]['wids'])
    args.target_len = len(dataset[0][0]['tids'])
    print("sent length:", args.sent_len)
    print("target length:", args.target_len)
    print("length of padded training set:", len(dataset[0]))
    
    n_train_batches = math.ceil(n_train / args.bs)
    n_test_batches = math.ceil(n_test / args.bs)
    train_set, test_set = dataset
    
    cur_model_name = 'TNet-%s' % args.connection_type
    print("Current model name:", cur_model_name)
    model = TNet(args=args)
    print(model)
    
    result_strings = []
    result_store_train = []
    result_store_test = []
    for i in range(1, args.n_epoch + 1):
        beg = time.time()
        # ---------------training----------------
        print("In epoch %s/%s:" % (i, args.n_epoch))
        np.random.shuffle(train_set)
        train_y_pred, train_y_gold = [], []
        train_losses = []
        train_ids = []
        train_alphas = []
        for j in range(n_train_batches):
            train_id, train_x, train_xt, train_y, train_pw, train_mask = get_batch_input(dataset=train_set, bs=args.bs, idx=j)
            y_pred, y_gold, loss, alpha = model.train(train_x, train_xt, train_y, train_pw, train_mask, np.int32(1))
            train_y_pred.extend(y_pred)
            train_y_gold.extend(y_gold)
            train_losses.append(loss)
            train_ids.extend(train_id)
            train_alphas.extend(alpha)
        acc, f, _, _ = evaluate(pred=train_y_pred, gold=train_y_gold)
        result_store_train.append((acc * 100, f * 100, train_ids, train_alphas, train_y_pred, train_y_gold))
        print("\ttrain loss: %.4f, train acc: %.4f, train f1: %.4f" % (sum(train_losses), acc, f))
        

        test_y_pred, test_y_gold = [], []
        test_ids = []
        test_alphas = []
        for j in range(n_test_batches):
            test_id, test_x, test_xt, test_y, test_pw, test_mask = get_batch_input(dataset=test_set, bs=args.bs, idx=j)
            y_pred, y_gold, loss, alpha = model.test(test_x, test_xt, test_y, test_pw, test_mask, np.int32(0))
            test_y_pred.extend(y_pred)
            test_y_gold.extend(y_gold)
            test_ids.extend(test_id)
            test_alphas.extend(alpha)
        acc, f, _, _ = evaluate(pred=test_y_pred[:n_test], gold=test_y_gold[:n_test])
        result_store_test.append((acc * 100, f * 100, test_ids, test_alphas))
        print("\tperformance of prediction: acc: %.4f, f1: %.4f" % (acc, f))
        result_strings.append("In Epoch %s: accuracy: %.2f, macro-f1: %.2f\n" % (i, acc * 100, f * 100))
        end = time.time()
        print ("\tIn Epoch %s: cost %f s\n" % (i, end - beg))
    
    result_logs = ['-------------------------------------------------------\n']
    params_string = str(args)
    result_logs.append("Running model: %s\n" % cur_model_name)
    result_logs.append(params_string + "\n")
    result_logs.extend(result_strings)
    result_logs.append('-------------------------------------------------------\n')
    if not os.path.exists('./log'):
        os.mkdir('log')
    print ("".join(result_logs))

    best_index_test = result_store_test.index(max(result_store_test))
    best_result_test = result_store_test[best_index_test]
    print ("Best model in Epoch %s: test accuracy: %.2f, macro-f1: %.2f\n" % (best_index_test+1, best_result_test[0], best_result_test[1]))
    best_index_train = result_store_train.index(max(result_store_train))
    best_result_train = result_store_train[best_index_train]
    print ("Best model in Epoch %s: train accuracy: %.2f, macro-f1: %.2f\n" % (best_index_train+1, best_result_train[0], best_result_train[1]))

    # store train alpha
    write_alpha_to_file = np.zeros((len(train_set), args.sent_len))
    for (id, alpha, pred, gold) in zip(best_result_train[2], best_result_train[3], best_result_train[4], best_result_train[5]):
        if int(pred) == int(gold):
            write_alpha_to_file[id] = alpha
        else:
            write_alpha_to_file[id] = -alpha
    np.savetxt("log/%s/%s_%s.alpha.E%s" % (args.log_name, cur_model_name, args.ds_name, erasing), write_alpha_to_file)
    print ('#######################################################\n')
    print ('#######################################################\n')

def train_final(a1_name, a2_name, a3_name, a4_name, a5_name, erasing, args):
    print ('#######################################################\n')
    print ('#######################################################\n')
    dataset, embeddings, n_train, n_test = build_dataset(ds_name=args.ds_name, bs=args.bs, dim_w=args.dim_w, a1_name=a1_name, a2_name=a2_name, a3_name=a3_name, a4_name=a4_name, a5_name=a5_name, erasing_or_final=False)
    
    args.dim_w = len(embeddings[1])
    print(args)
    args.embeddings = embeddings
    args.sent_len = len(dataset[0][0]['wids'])
    args.target_len = len(dataset[0][0]['tids'])
    print("sent length:", args.sent_len)
    print("target length:", args.target_len)
    print("length of padded training set:", len(dataset[0]))
    
    n_train_batches = math.ceil(n_train / args.bs)
    n_test_batches = math.ceil(n_test / args.bs)
    train_set, test_set = dataset
    
    cur_model_name = 'TNet-%s' % args.connection_type
    print("Current model name:", cur_model_name)
    model = TNet(args=args)
    print(model)
    
    result_strings = []
    
    result_store_test = []

    for i in range(1, args.n_epoch + 1):
        beg = time.time()
        # ---------------training----------------
        print("In epoch %s/%s:" % (i, args.n_epoch))
        np.random.shuffle(train_set)
        train_y_pred, train_y_gold = [], []
        train_losses = []
        train_ids = []
        train_alosses = []
        for j in range(n_train_batches):
            train_id, train_x, train_xt, train_y, train_pw, train_mask, train_amask, train_avalue = get_batch_input_final(dataset=train_set, bs=args.bs, idx=j)
            y_pred, y_gold, loss, aloss = model.train_final(train_x, train_xt, train_y, train_pw, train_mask, train_amask, train_avalue, np.int32(1))
            train_y_pred.extend(y_pred)
            train_y_gold.extend(y_gold)
            train_losses.append(loss)
            train_alosses.append(aloss)
        acc, f, _, _ = evaluate(pred=train_y_pred, gold=train_y_gold)
        print("\ttrain loss: %.4f, train aloss: %.4f, train acc: %.4f, train f1: %.4f" % (sum(train_losses), sum(train_alosses), acc, f))
        
        # ---------------prediction----------------
        test_y_pred, test_y_gold = [], []
        test_ids = []
        test_alphas = []
        for j in range(n_test_batches):
            test_id, test_x, test_xt, test_y, test_pw, test_mask = get_batch_input(dataset=test_set, bs=args.bs, idx=j)
            y_pred, y_gold, loss, alpha = model.test(test_x, test_xt, test_y, test_pw, test_mask, np.int32(0))
            test_y_pred.extend(y_pred)
            test_y_gold.extend(y_gold)
            test_ids.extend(test_id)
            test_alphas.extend(alpha)
        acc, f, _, _ = evaluate(pred=test_y_pred[:n_test], gold=test_y_gold[:n_test])
        result_store_test.append((acc * 100, f * 100, test_ids, test_alphas))
        print("\tperformance of prediction: acc: %.4f, f1: %.4f" % (acc, f))
        result_strings.append("In Epoch %s: accuracy: %.2f, macro-f1: %.2f\n" % (i, acc * 100, f * 100))
        end = time.time()
        print ("\tIn Epoch %s: cost %f s\n" % (i, end - beg))
    
    result_logs = ['-------------------------------------------------------\n']
    params_string = str(args)
    result_logs.append("Running model: %s\n" % cur_model_name)
    result_logs.append(params_string + "\n")
    result_logs.extend(result_strings)
    result_logs.append('-------------------------------------------------------\n')
    if not os.path.exists('./log'):
        os.mkdir('log')
    
    print ("".join(result_logs))

    best_index_test = result_store_test.index(max(result_store_test))
    best_result_test = result_store_test[best_index_test]
    print ("Best model in Epoch %s: test accuracy: %.2f, macro-f1: %.2f\n" % (best_index_test+1, best_result_test[0], best_result_test[1]))

    # store alpha
    write_alpha_to_file = np.zeros((len(test_set), len(test_set[0]['wids'])))
    for (id, alpha) in zip(best_result_test[2], best_result_test[3]):
        write_alpha_to_file[id] = alpha
    np.savetxt("log/%s/%s_%s.alpha.Final%s" % (args.log_name, cur_model_name, args.ds_name, erasing), write_alpha_to_file)
    print ('#######################################################\n')
    print ('#######################################################\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TNet settings')
    parser.add_argument("-ds_name", type=str, default="14semeval_rest", help="dataset name")
    parser.add_argument("-bs", type=int, default=64, help="batch size")
    parser.add_argument("-dim_w", type=int, default=300, help="dimension of word embeddings")
    parser.add_argument("-dropout_rate", type=float, default=0.3, help="dropout rate for sentimental features")
    parser.add_argument("-dim_h", type=int, default=50, help="dimension of hidden state")
    parser.add_argument("-n_epoch", type=int, default=50, help="number of training epoch")
    parser.add_argument("-dim_y", type=int, default=3, help="dimension of label space")
    parser.add_argument("-connection_type", type=str, default="AS", help="connection type, only AS and LF are valid")
    parser.add_argument("-log_name", type=str, default="14semeval_rest", help="dataset name")
    
    args = parser.parse_args()
    cur_model_name = 'TNet-%s' % args.connection_type
    args.lamda = 0.1
    
    if args.ds_name == '14semeval_rest' or args.ds_name == '14semeval_rest_val':
        args.lamda = 0.5
        args.bs = 25
    
    a1_name = None
    a2_name = None
    a3_name = None
    a4_name = None
    a5_name = None

    train_erasing(a1_name, a2_name, a3_name, a4_name, a5_name, 1, args)
    a1_name = "log/%s/%s_%s.alpha.E%s" % (args.log_name, cur_model_name, args.ds_name, 1)

    train_erasing(a1_name, a2_name, a3_name, a4_name, a5_name, 2, args)
    a2_name = "log/%s/%s_%s.alpha.E%s" % (args.log_name, cur_model_name, args.ds_name, 2)
    
    train_erasing(a1_name, a2_name, a3_name, a4_name, a5_name, 3, args)
    a3_name = "log/%s/%s_%s.alpha.E%s" % (args.log_name, cur_model_name, args.ds_name, 3)

    train_erasing(a1_name, a2_name, a3_name, a4_name, a5_name, 4, args)
    a4_name = "log/%s/%s_%s.alpha.E%s" % (args.log_name, cur_model_name, args.ds_name, 4)

    train_erasing(a1_name, a2_name, a3_name, a4_name, a5_name, 5, args)
    a5_name = "log/%s/%s_%s.alpha.E%s" % (args.log_name, cur_model_name, args.ds_name, 5)


    a1_name = None
    a2_name = None
    a3_name = None
    a4_name = None
    a5_name = None

    train_final(a1_name, a2_name, a3_name, a4_name, a5_name, 0, args)
    a1_name = "log/%s/%s_%s.alpha.E%s" % (args.log_name, cur_model_name, args.ds_name, 1)

    train_final(a1_name, a2_name, a3_name, a4_name, a5_name, 1, args)
    a2_name = "log/%s/%s_%s.alpha.E%s" % (args.log_name, cur_model_name, args.ds_name, 2)

    train_final(a1_name, a2_name, a3_name, a4_name, a5_name, 2, args)
    a3_name = "log/%s/%s_%s.alpha.E%s" % (args.log_name, cur_model_name, args.ds_name, 3)

    train_final(a1_name, a2_name, a3_name, a4_name, a5_name, 3, args)
    a4_name = "log/%s/%s_%s.alpha.E%s" % (args.log_name, cur_model_name, args.ds_name, 4)

    train_final(a1_name, a2_name, a3_name, a4_name, a5_name, 4, args)
    a5_name = "log/%s/%s_%s.alpha.E%s" % (args.log_name, cur_model_name, args.ds_name, 5)

    train_final(a1_name, a2_name, a3_name, a4_name, a5_name, 5, args)
