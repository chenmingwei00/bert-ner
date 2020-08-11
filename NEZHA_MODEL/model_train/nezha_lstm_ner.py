#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Copyright 2018 The Google AI Language Team Authors.
BASED ON Google_BERT.
reference from :zhoukaiyin/

@Author:Macan
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import json
import tensorflow as tf
import codecs
import pickle
from NEZHA_TensorFlow.NEZHA_MODEL.utils.data_utils import *
from NEZHA_TensorFlow.NEZHA_MODEL.nezha_bert import tf_metrics
from NEZHA_TensorFlow.NEZHA_MODEL.nezha_bert import modeling
from NEZHA_TensorFlow.NEZHA_MODEL.nezha_bert import optimization
# import

__version__ = '0.1.0'


__all__ = ['__version__', 'DataProcessor', 'NerProcessor', 'write_tokens', 'convert_single_example',
           'filed_based_convert_examples_to_features', 'file_based_input_fn_builder',
           'model_fn_builder', 'train']



def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, args):
    """
    构建模型
    :param bert_config:
    :param num_labels:
    :param init_checkpoint:
    :param learning_rate:
    :param num_train_steps:
    :param num_warmup_steps:
    :param use_tpu:
    :param use_one_hot_embeddings:
    :return:
    """

    def model_fn(features, labels, mode, params):
        logger.info("*** Features ***")
        for name in sorted(features.keys()):
            logger.info("  name = %s, shape = %s" % (name, features[name].shape))
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        # words_parts_ids = features["words_parts_ids"]

        if mode != 'infer':
            label_ids = features["label_ids"]

            print('shape of input_ids', input_ids.shape)
            # label_mask = features["label_mask"]
            is_training = (mode == tf.estimator.ModeKeys.TRAIN)

            # 使用参数构建模型,input_idx 就是输入的样本idx表示，label_ids 就是标签的idx表示
            total_loss, logits, trans, pred_ids = create_model(
                bert_config, is_training, input_ids,input_mask, segment_ids,
                num_labels, False, args.dropout_rate, args.lstm_size, args.cell, args.num_layers, label_ids)
        else:
            guid = features['guid']
            (total_loss, logits, trans, pred_ids) = create_model(
                bert_config=bert_config, is_training=False, input_ids=input_ids, input_mask=input_mask,
                segment_ids=None,num_labels=num_labels, use_one_hot_embeddings=False, dropout_rate=1.0,
                lstm_size=args.lstm_size,cell= args.cell,num_layers=args.num_layers,labels=None)

        tvars = tf.trainable_variables()
        # 加载BERT模型
        if init_checkpoint:
            (assignment_map, initialized_variable_names) = \
                modeling.get_assignment_map_from_checkpoint(tvars,
                                                            init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        # 打印变量名
        # logger.info("**** Trainable Variables ****")
        #
        # # 打印加载模型的参数
        # for var in tvars:
        #     init_string = ""
        #     if var.name in initialized_variable_names:
        #         init_string = ", *INIT_FROM_CKPT*"
        #     logger.info("  name = %s, shape = %s%s", var.name, var.shape,
        #                     init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            # train_op = optimizer.optimizer(total_loss, learning_rate, num_train_steps)
            # train_op = optimization.create_optimizer(
            #     total_loss, learning_rate, num_train_steps, num_warmup_steps, False)
            train_op=optimization.create_optimizer(
                loss=total_loss, init_lr=learning_rate, num_train_steps=num_train_steps,
                num_warmup_steps=num_warmup_steps)
            hook_dict = {}
            hook_dict['loss'] = total_loss
            hook_dict['global_steps'] = tf.train.get_or_create_global_step()
            logging_hook = tf.train.LoggingTensorHook(
                hook_dict, every_n_iter=args.save_summary_steps)

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                training_hooks=[logging_hook])

        elif mode == tf.estimator.ModeKeys.EVAL:
            # 针对NER ,进行了修改
            def metric_fn(label_ids, pred_ids):
                # predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                accuracy = tf.metrics.accuracy(
                    labels=label_ids, predictions=pred_ids)

                precision, precision_update_op = tf.metrics.precision(labels=label_ids, predictions=pred_ids)
                recall, recall_update_op = tf.metrics.recall(labels=label_ids, predictions=pred_ids,
                                                             )

                f1_sco, f1_score_op = tf.metrics.mean((2 * precision_update_op * recall_update_op) /
                                                      (precision_update_op + recall_update_op), name='f1_score')
                return {
                    "eval_accuracy": accuracy,
                    "precision": (precision, precision_update_op),
                    'recall': (recall, recall_update_op),
                    'f1': (f1_sco, f1_score_op),
                }
                # return {
                #     "eval_loss": tf.metrics.mean_squared_error(labels=label_ids, predictions=pred_ids),
                #     "predilabel":pred_ids
                # }

            eval_metrics = metric_fn(label_ids, pred_ids)
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metric_ops=eval_metrics
            )
        else:
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={"probabilities": pred_ids, 'guid': guid},
            )
        return output_spec

    return model_fn

key_labels={'test_elements':'试验要素',  'performance':'性能指标', 'system_composition':'系统组成',
            "mission_scen":'任务场景'}
def train(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_map

    processors = {
        "ner": NerProcessor
    }
    bert_config = modeling.BertConfig.from_json_file(args.bert_config_file)

    if args.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (args.max_seq_length, bert_config.max_position_embeddings))

    # 在re train 的时候，才删除上一轮产出的文件，在predicted 的时候不做clean
    if args.clean and args.do_train:
        if os.path.exists(args.output_dir):
            def del_file(path):
                ls = os.listdir(path)
                for i in ls:
                    c_path = os.path.join(path, i)
                    if os.path.isdir(c_path):
                        del_file(c_path)
                    else:
                        os.remove(c_path)

            try:
                del_file(args.output_dir)
            except Exception as e:
                print(e)
                print('pleace remove the files of output dir and NERdata.conf')
                exit(-1)

    # check output dir exists
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    processor = processors[args.ner](args.output_dir)

    tokenizer = tokenization.FullTokenizer(
        vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)

    session_config = tf.ConfigProto(
        log_device_placement=False,
        inter_op_parallelism_threads=0,
        intra_op_parallelism_threads=0,
        allow_soft_placement=True)

    run_config = tf.estimator.RunConfig(
        model_dir=args.output_dir,
        save_summary_steps=500,
        save_checkpoints_steps=500,
        session_config=session_config
    )

    train_examples = None
    eval_examples = None
    num_train_steps = None
    num_warmup_steps = None

    if args.do_train and args.do_eval:
        # 加载训练数据
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_steps = int(
            len(train_examples) * 1.0 / args.batch_size * args.num_train_epochs)
        if num_train_steps < 1:
            raise AttributeError('training NERdata is so small...')
        num_warmup_steps = int(num_train_steps * args.warmup_proportion)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.batch_size)
        logger.info("  Num steps = %d", num_train_steps)

        eval_examples = processor.get_dev_examples(args.data_dir)

        # 打印验证集数据信息
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.batch_size)

    label_list = processor.get_labels()
    # 加载词性字典
    part_speech_dcit = {}
    # with open(args.partspeech_dict, 'r') as part_dict:
    #     lines = part_dict.readlines()
    #     for index, temp in enumerate(lines):
    #         part_speech_dcit[temp.strip()] = index
    # 返回的model_dn 是一个函数，其定义了模型，训练，评测方法，并且使用钩子参数，加载了BERT模型的参数进行了自己模型的参数初始化过程
    # tf 新的架构方法，通过定义model_fn 函数，定义模型，然后通过EstimatorAPI进行模型的其他工作，Es就可以控制模型的训练，预测，评估工作等。
    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list) + 1,
        init_checkpoint=args.init_checkpoint,
        learning_rate=args.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        args=args)

    params = {
        'batch_size': args.batch_size
    }

    estimator = tf.estimator.Estimator(
        model_fn,
        params=params,
        config=run_config)

    if args.do_train and args.do_eval:
        # 1. 将数据转化为tf_record 数据
        train_file = os.path.join(args.output_dir, "train.tf_record")
        if not os.path.exists(train_file):
            filed_based_convert_examples_to_features(
                train_examples, label_list, args.max_seq_length, tokenizer, train_file, args.output_dir,part_speech_dcit=part_speech_dcit)

        # 2.读取record 数据，组成batch
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=args.max_seq_length,
            is_training=True,
            drop_remainder=True)
        # estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

        eval_file = os.path.join(args.output_dir, "eval.tf_record")
        if not os.path.exists(eval_file):
            filed_based_convert_examples_to_features(
                eval_examples, label_list, args.max_seq_length, tokenizer, eval_file, args.output_dir,part_speech_dcit=part_speech_dcit)

        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=args.max_seq_length,
            is_training=True,
            drop_remainder=False)

        # train and eval togither
        # early stop hook
        early_stopping_hook = tf.estimator.experimental.stop_if_no_decrease_hook(
            estimator=estimator,
            metric_name='loss',
            max_steps_without_decrease=num_train_steps,
            eval_dir=None,
            min_steps=0,
            run_every_secs=None,
            run_every_steps=args.save_checkpoints_steps)

        train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=num_train_steps,
                                            hooks=[early_stopping_hook])
        eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    if args.do_predict:
        token_path = os.path.join(args.output_dir, "token_test.txt")
        if os.path.exists(token_path):
            os.remove(token_path)

        with codecs.open(os.path.join(args.output_dir, 'label2id.pkl'), 'rb') as rf:
            label2id = pickle.load(rf)
            id2label = {value: key for key, value in label2id.items()}

        predict_examples = processor.get_test_examples(args.data_dir,stype='predict')
        predict_file = os.path.join(args.output_dir, "predict.tf_record")
        filed_based_convert_examples_to_features(predict_examples, label_list,
                                                 args.max_seq_length, tokenizer,
                                                 predict_file, args.output_dir, mode="test",part_speech_dcit=part_speech_dcit)

        logger.info("***** Running prediction*****")
        logger.info("  Num examples = %d", len(predict_examples))
        logger.info("  Batch size = %d", args.batch_size)

        predict_drop_remainder = False
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=args.max_seq_length,
            is_training=False,
            drop_remainder=predict_drop_remainder)

        result = estimator.predict(input_fn=predict_input_fn)
        output_predict_file = os.path.join(args.output_dir, "label_test1.json")
        dev_predict_file = os.path.join(args.output_dir, "dev_label_test1.txt")

        def result_to_pair(writer,writer2):
            predict_result={}
            max_lens=[]
            for predict_line, prediction in zip(predict_examples, result):
                assert prediction['guid'].decode('utf-8') == predict_line.guid
                tokens_list_index = []  # 表示原始对应当前分词后的结果
                tokens = []
                if predict_line.label!=None:
                    assert len(predict_line.text.split("|"))==len(predict_line.label.split("|"))
                    for i, word in enumerate(predict_line.text.split("|")):
                        current_tokens_index = []
                        token = tokenizer.tokenize(word)
                        for toke in token:
                            current_tokens_index.append(len(tokens))
                            tokens.append(toke)
                        tokens_list_index.append(current_tokens_index)
                else:
                    for i, word in enumerate(list(predict_line.text.strip())):
                        current_tokens_index = []
                        token = tokenizer.tokenize(word)
                        for toke in token:
                            current_tokens_index.append(len(tokens))
                            tokens.append(toke)
                        tokens_list_index.append(current_tokens_index)

                len_seq = len(tokens)#这个是对应的预测结果实际长度
                labels_predict=[]
                idx = 0
                for id in prediction['probabilities']:
                    if idx >= len_seq:
                        break
                    if id == 0:
                        continue
                    curr_labels = id2label[id]
                    if curr_labels in ['[CLS]', '[SEP]']:
                        continue

                    labels_predict.append(curr_labels)

                    idx += 1

                if predict_line.label!=None:
                    tokens1 = predict_line.text.split("|")
                    true_labels = predict_line.label.split("|")
                else:
                    tokens1 = list(predict_line.text)
                if len(tokens)>len(labels_predict):
                    tokens_list_index=tokens_list_index[:len(labels_predict)]
                    max_lens.append(len(tokens))
                    if predict_line.label!=None:
                        true_labels=true_labels[:len(labels_predict)]
                        tokens1=tokens1[:len(labels_predict)]
                if predict_line.label!=None:
                    k=0
                    for current_index,input_id in enumerate(tokens_list_index):
                        if len(input_id)>=1:
                            temp_token=tokens1[k]
                            pre_tag=labels_predict[k]
                            true_tag=true_labels[current_index]
                            k+=len(input_id)
                        else:
                            temp_token = " "
                            pre_tag = labels_predict[k+1]
                            true_tag = true_labels[current_index]
                        writer2.write(temp_token+" "+true_tag+" "+pre_tag+'\n')
                    writer2.write('\n')
                else:
                    for temp_token, pre_tag in zip(tokens1, labels_predict):
                        writer2.write(temp_token + " " + pre_tag + '\n')
                    writer2.write('\n')
                temp_predict = []
                temp_entity_info = {}
                for index, temp_flag in enumerate(tokens_list_index):
                    for flags_index in temp_flag:
                        if 'B_' in labels_predict[flags_index]:
                            if len(temp_entity_info) > 0:
                                temp_predict.append(temp_entity_info)
                                temp_entity_info = {}
                            temp_entity_info["start_pos"] = index+1
                            temp_entity_info["end_pos"] = index+1
                            temp_entity_info["label_type"] = key_labels[labels_predict[flags_index].replace('B_', '')]
                            temp_entity_info["overlap"]=0
                        elif 'I_' in labels_predict[flags_index]:
                            temp_entity_info["end_pos"] = index+1
                if len(temp_entity_info)>0: #这个地方少了最后一个的预测结果
                    temp_predict.append(temp_entity_info)

                predict_result[predict_line.guid]=temp_predict
            json.dump(predict_result,writer,ensure_ascii=False)
            print(max_lens)
            print(len(max_lens))
        with codecs.open(output_predict_file, 'w', encoding='utf-8') as writer,codecs.open(
            dev_predict_file,'w',encoding='utf-8') as writer2:
            result_to_pair(writer,writer2)
    # filter model
    if args.filter_adam_var:
        adam_filter(args.output_dir)

def adam_filter(model_path):
    """
    去掉模型中的Adam相关参数，这些参数在测试的时候是没有用的
    :param model_path:
    :return:
    """
    last_name = get_last_checkpoint(model_path)
    if last_name is None:
        return
    sess = tf.Session()
    imported_meta = tf.train.import_meta_graph(os.path.join(model_path, last_name + '.meta'))
    imported_meta.restore(sess, os.path.join(model_path, last_name))
    need_vars = []
    for var in tf.global_variables():
        if 'adam_v' not in var.name and 'adam_m' not in var.name:
            need_vars.append(var)
    saver = tf.train.Saver(need_vars)
    saver.save(sess, os.path.join(model_path, 'model.ckpt'))

def get_last_checkpoint(model_path):
    if not os.path.exists(os.path.join(model_path, 'checkpoint')):
        logger.info('checkpoint file not exits:'.format(os.path.join(model_path, 'checkpoint')))
        return None
    last = None
    with codecs.open(os.path.join(model_path, 'checkpoint'), 'r', encoding='utf-8') as fd:
        for line in fd:
            line = line.strip().split(':')
            if len(line) != 2:
                continue
            if line[0] == 'model_checkpoint_path':
                last = line[1][2:-1]
                break
    return last

