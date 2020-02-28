#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'yp'

import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib.crf import viterbi_decode
from model import Model as Model
from sklearn.metrics import classification_report
from utils import DataProcess


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
TRAIN_MODE = 'train'

model_dir = os.path.join('../', '_'.join([Model.__name__, time.strftime("%Y%m%d%H%M%S")]))
print('model_dir:==={}'.format(model_dir))

if os.path.exists(model_dir):
    pass
else:
    os.mkdir(model_dir)

log_file = './log_file.txt'
_num = 0.00001

train_data_list = [
    '../data/normal_train/test_v2.txt'
]

test_data_list = ['../data/normal_train/test_v2.txt']

model = Model(learning_rate=0.0001, sequence_length_val=10, num_tags=9)

init = tf.global_variables_initializer()
saver = tf.train.Saver(tf.global_variables(), max_to_keep=40)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

if TRAIN_MODE == 'train':
    with open(log_file, mode='w', encoding='utf-8') as f1:
        with tf.Session(config=config) as sess:
            sess.run(init)

            train_data_process = DataProcess()
            train_data_process.load_data(file_list=train_data_list)
            train_data_process.get_feature()

            test_data_process = DataProcess()
            test_data_process.load_data(file_list=test_data_list, is_shuffle=False)
            test_data_process.get_feature()

            step = 0
            epoch = 40

            for i in range(epoch):

                for _, batch_x, batch_y in train_data_process.next_batch():
                    sum_counter = 0
                    right_counter = 0

                    model.is_training = True
                    _seq_len = np.array([len(_) for _ in batch_x])
                    _logits, _loss, _opt, transition_params = sess.run([model.logits,
                                                                        model.loss_val,
                                                                        model.train_op,
                                                                        model.transition_params
                                                                        ],
                                                                       feed_dict={model.input_x: batch_x,
                                                                                  model.input_y: batch_y,
                                                                                  model.sequence_lengths: _seq_len,
                                                                                  model.keep_prob: 0.8})

                    step += 1

                    for logit, seq_len, _y_label in zip(_logits, _seq_len, batch_y):
                        viterbi_seq, _ = viterbi_decode(logit[:seq_len], transition_params)

                        if viterbi_seq == list(_y_label):
                            right_counter += 1
                        sum_counter += 1

                    if step % 50 == 0:
                        a.info("step:{} ===loss:{} === acc: {}".format(step, _loss, str(right_counter / sum_counter)))
                        print("step:{} ===loss:{} === acc: {}".format(step, _loss, str(right_counter / sum_counter)))
                        f1.writelines("step:{} ===loss:{} === acc: {}\n".format(str(step),
                                                                                str(_loss),
                                                                                str(right_counter / sum_counter)))

                save_path = saver.save(sess, "%s/%s/model_epoch_%s" % (model_dir, str(i), str(i)))

                # test
                y_predict_list = []
                y_label_list = []

                sum_counter = 0
                right_counter = 0
                f1_statics = np.array([0 for i in range(12)])
                y_t = []
                y_p = []
                for batch_sentence, batch_x, batch_y in test_data_process.next_batch():
                    model.is_training = False
                    _seq_len = np.array([len(_) for _ in batch_x])
                    _logits, transition_params = sess.run([model.logits,
                                                           model.transition_params], feed_dict=
                                                          {model.input_x: batch_x,
                                                           model.input_y: batch_y,
                                                           model.sequence_lengths: _seq_len,
                                                           model.keep_prob: 1.0})
                    for _sentence_str, logit, seq_len, _y_label in zip(batch_sentence, _logits, _seq_len, batch_y):
                        viterbi_seq, _ = viterbi_decode(logit[:seq_len], transition_params)

                        y_p.extend(viterbi_seq)
                        y_t.extend(list(_y_label))

                        # predict
                        disease_out = get_disease_from_tag(sentence=_sentence_str, tag=viterbi_seq, target=[1, 2])
                        symp_out = get_disease_from_tag(sentence=_sentence_str, tag=viterbi_seq, target=[3, 4])
                        drug_out = get_disease_from_tag(sentence=_sentence_str, tag=viterbi_seq, target=[5, 6])
                        diagnosis_out = get_disease_from_tag(sentence=_sentence_str, tag=viterbi_seq, target=[7, 8])

                        # true
                        true_disease_out = get_disease_from_tag(sentence=_sentence_str, tag=_y_label, target=[1, 2])
                        true_symp_out = get_disease_from_tag(sentence=_sentence_str, tag=_y_label, target=[3, 4])
                        true_drug_out = get_disease_from_tag(sentence=_sentence_str, tag=_y_label, target=[5, 6])
                        true_diagnosis_out = get_disease_from_tag(sentence=_sentence_str, tag=_y_label, target=[7, 8])

                        # sum f1
                        f1_tmp = [len(set(disease_out).intersection(set(true_disease_out))),
                                  len(disease_out), len(true_disease_out)]
                        f1_tmp += [len(set(symp_out).intersection(set(true_symp_out))),
                                   len(symp_out), len(true_symp_out)]
                        f1_tmp += [len(set(drug_out).intersection(set(true_drug_out))),
                                   len(drug_out), len(true_drug_out)]
                        f1_tmp += [len(set(diagnosis_out).intersection(set(true_diagnosis_out))),
                                   len(diagnosis_out), len(true_diagnosis_out)]
                        f1_statics += np.array(f1_tmp)

                        if viterbi_seq == list(_y_label):
                            right_counter += 1
                        sum_counter += 1

                f1_statics = f1_statics.tolist()
                a.info("epoch: {}====f1_statics: {} \n".format(str(i), str(f1_statics)))
                f1.writelines("epoch: {}====f1_statics: {} \n".format(str(i), str(f1_statics)))

                P = f1_statics[0] / f1_statics[1]
                R = f1_statics[0] / f1_statics[2]
                print("epoch: {}====disease f1: {} \n".format(str(i), str(2 * P * R / (P + R + _num))))
                a.info("epoch: {}====disease f1: {} \n".format(str(i), str(2 * P * R / (P + R + _num))))
                f1.writelines("epoch: {}====disease f1: {} \n".format(str(i), str(2 * P * R / (P + R + _num))))

                P = f1_statics[3] / f1_statics[4]
                R = f1_statics[3] / f1_statics[5]
                print("epoch: {}====symp f1: {} \n".format(str(i), str(2 * P * R / (P + R + _num))))
                a.info("epoch: {}====symp f1: {} \n".format(str(i), str(2 * P * R / (P + R + _num))))
                f1.writelines("epoch: {}====symp f1: {} \n".format(str(i), str(2 * P * R / (P + R + _num))))

                P = f1_statics[6] / f1_statics[7]
                R = f1_statics[6] / f1_statics[8]
                print("epoch: {}====drug f1: {} \n".format(str(i), str(2 * P * R / (P + R + _num))))
                a.info("epoch: {}====drug f1: {} \n".format(str(i), str(2 * P * R / (P + R + _num))))
                f1.writelines("epoch: {}====drug f1: {} \n".format(str(i), str(2 * P * R / (P + R + _num))))

                P = f1_statics[9] / f1_statics[10]
                R = f1_statics[9] / f1_statics[11]
                print("epoch: {}====diag f1: {} \n".format(str(i), str(2 * P * R / (P + R + _num))))
                a.info("epoch: {}====diag f1: {} \n".format(str(i), str(2 * P * R / (P + R + _num))))
                f1.writelines("epoch: {}====diag f1: {} \n".format(str(i), str(2 * P * R / (P + R + _num))))

                print("epoch: {}====: \n".format(str(i)), classification_report(y_pred=y_p, y_true=y_t))
                a.info("epoch: {}==={}=: \n".format(str(i), classification_report(y_pred=y_p, y_true=y_t)))
                f1.writelines("epoch: {}==={}=: \n".format(str(i), classification_report(y_pred=y_p, y_true=y_t)))

                a.info("epoch:{}==macro_f1:{}==weight_f1:{}".format(str(i),
                                                                    str(count_macro_f1(f1_statics)),
                                                                    str(count_weight_f1(f1_statics))))

                print("epoch: {}======acc rate: {}\n".format(str(i), str(right_counter / sum_counter)))
                a.info("epoch: {}======acc rate: {}\n".format(str(i), str(right_counter / sum_counter)))
                f1.writelines("epoch: {}======acc rate: {}\n".format(str(i), str(right_counter / sum_counter)))

if TRAIN_MODE == 'prodict':
    predict_data_list = ['../data/medical_record/train_3w.txt']

    predict_data_process = DataProcess(feature_mode=FEATURE_MODE)
    predict_data_process.load_data(file_list=predict_data_list)
    predict_data_process.get_feature()

    with tf.Session(config=config) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, "../model/19/model_epoch_19")

        y_predict_list = []
        for batch_x, batch_y in predict_data_process.next_batch():
            model.is_training = False
            _seq_len = np.array([len(_) for _ in batch_x])
            _logits, transition_params = sess.run([model.logits,
                                                   model.transition_params],
                                                  feed_dict={model.input_x: batch_x,
                                                             model.sequence_lengths: _seq_len,
                                                             model.keep_prob: 1.0})

            for logit, seq_len in zip(_logits, _seq_len):
                viterbi_seq, _ = viterbi_decode(logit[:seq_len], transition_params)
                y_predict_list.append(viterbi_seq)

        _out_file = predict_data_process.data
        _out_file['y_pred'] = pd.Series(y_predict_list)
        _out_file.to_csv('./final_predict.tsv', sep='\t')
