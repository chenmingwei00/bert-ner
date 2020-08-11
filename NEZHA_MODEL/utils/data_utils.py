import tensorflow as tf
import os, codecs, pickle, collections, json
from NEZHA_TensorFlow.NEZHA_MODEL.nezha_bert import tokenization
from NEZHA_TensorFlow.NEZHA_MODEL.server.helper import set_logger
from NEZHA_TensorFlow.NEZHA_MODEL.nezha_bert.models import create_model, InputFeatures, InputExample

logger = set_logger('NER Training')


class DataProcessor(object):
    """Base class for NERdata converters for sequence classification NERdata sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this NERdata set."""
        raise NotImplementedError()

    @classmethod
    def _read_data(cls, input_file):
        """Reads a BIO NERdata."""
        with codecs.open(input_file, 'r', encoding='utf-8') as f:
            lines = []
            words = []
            labels = []
            for line in f:
                contends = line.strip()
                tokens = contends.split(' ')
                if len(tokens) == 2:
                    words.append(tokens[0])
                    labels.append(tokens[1])
                else:
                    if len(contends) == 0:
                        l = ' '.join([label for label in labels if len(label) > 0])
                        w = ' '.join([word for word in words if len(word) > 0])
                        lines.append([l, w])
                        words = []
                        labels = []
                        continue
                if contends.startswith("-DOCSTART-"):
                    words.append('')
                    continue
            return lines


class NerProcessor(DataProcessor):
    def __init__(self, output_dir):
        self.labels = set()
        self.output_dir = output_dir

    def get_train_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "train.txt")), "train"
        )

    def get_dev_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "dev.txt")), "dev"
        )

    def get_test_examples(self, data_dir, stype='predict'):
        if stype == 'predict':
            return self._create_example(
                self._read_test(os.path.join(data_dir, "validate_data.json")), "test")
        else:
            return self._create_example(
                self._read_test2(os.path.join(data_dir, "predict.txt")), "test")

    def get_labels(self, labels=None):
        if labels is not None:
            try:
                # 支持从文件中读取标签类型
                if os.path.exists(labels) and os.path.isfile(labels):
                    with codecs.open(labels, 'r', encoding='utf-8') as fd:
                        for line in fd:
                            self.labels.append(line.strip())
                else:
                    # 否则通过传入的参数，按照逗号分割
                    self.labels = labels.split(',')
                self.labels = set(self.labels)  # to set
            except Exception as e:
                print(e)
        # 通过读取train文件获取标签的方法会出现一定的风险。
        if os.path.exists(os.path.join(self.output_dir, 'label_list.pkl')):
            with codecs.open(os.path.join(self.output_dir, 'label_list.pkl'), 'rb') as rf:
                self.labels = pickle.load(rf)
        else:
            if len(self.labels) > 0:
                self.labels = self.labels.union(set(["X", "[CLS]", "[SEP]"]))
                with codecs.open(os.path.join(self.output_dir, 'label_list.pkl'), 'wb') as rf:
                    pickle.dump(self.labels, rf)
            else:
                self.labels = ["O", 'B-TIM', 'I-TIM', "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X",
                               "[CLS]", "[SEP]"]
        return self.labels

    def _create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if set_type != 'test':
                guid = "%s-%s" % (set_type, i)
                text = tokenization.convert_to_unicode(line[1])
                label = tokenization.convert_to_unicode(line[0])
                # if i == 0:
                #     print('label: ', label)
                examples.append(InputExample(guid=guid, text=text, label=label))
            else:

                guid = tokenization.convert_to_unicode(line[0])
                text = tokenization.convert_to_unicode(line[1])
                # if i == 0:
                #     print('label: ', label)
                examples.append(InputExample(guid=guid, text=text))
        return examples

    def _read_data(self, input_file):
        """Reads a BIO NERdata."""
        with codecs.open(input_file, 'r', encoding='utf-8') as f:
            lines = []
            words = []
            labels = []
            for line in f:
                contends = line
                tokens = contends.split('\t')
                if len(tokens) == 2:
                    words.append(tokens[0])
                    labels.append(tokens[-1].strip())
                else:
                    if contends.strip() == '---' and len(words) > 0:
                        label = []
                        word = []
                        for l, w in zip(labels, words):
                            if len(l) > 0 and len(w) > 0:
                                label.append(l)
                                self.labels.add(l)
                                word.append(w)
                        assert len(label) == len(word)
                        lines.append(['|'.join(label), '|'.join(word)])
                        words = []
                        labels = []
                        continue
                if contends.startswith("-DOCSTART-"):
                    continue
            return lines

    def _read_test(self, input_file):
        """Reads a BIO NERdata."""
        with codecs.open(input_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
            lines = []
            for key_ids, value_input in test_data.items():
                lines.append([key_ids, value_input])
            return lines

    def _read_test2(self, input_file):
        """Reads a BIO NERdata."""
        with codecs.open(input_file, 'r', encoding='utf-8') as f:
            test_data = f.readlines()
            lines = []
            for temp_text in test_data:
                key_ids, value_input = temp_text.split("|-")
                lines.append([str(key_ids), value_input])
            return lines


def write_tokens(tokens, output_dir, mode):
    """
    将序列解析结果写入到文件中
    只在mode=test的时候启用
    :param tokens:
    :param mode:
    :return:
    """
    if mode == "test":
        path = os.path.join(output_dir, "token_" + mode + ".txt")
        wf = codecs.open(path, 'a', encoding='utf-8')
        for token in tokens:
            if token != "**NULL**":
                wf.write(token + '\n')
        wf.close()


import jieba.posseg as pseg


def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, output_dir, mode,
                           part_speech_dcit):
    """
    将一个样本进行分析，然后将字转化为id, 标签转化为id,然后结构化到InputFeatures对象中
    :param ex_index: index
    :param example: 一个样本
    :param label_list: 标签列表
    :param max_seq_length:
    :param tokenizer:
    :param output_dir
    :param mode:
    :return:
    """
    label_map = {}
    # 1表示从1开始对label进行index化
    for (i, label) in enumerate(label_list, 1):
        label_map[label] = i
    # 保存label->index 的map
    if not os.path.exists(os.path.join(output_dir, 'label2id.pkl')):
        with codecs.open(os.path.join(output_dir, 'label2id.pkl'), 'wb') as w:
            pickle.dump(label_map, w)

    # cut_words = pseg.cut(''.join(example.text.split('|')))  # 里边包含空格符所以
    # words_parts = []
    # for w in cut_words:
    #     # if len(w.word.strip()) == 0: continue
    #     for k in range(len(w.word)):
    #         words_parts.append(part_speech_dcit[w.flag])  # bia
    #         # real_words.append(w.flag)
    if mode != 'test':
        labellist = example.label.split('|')
        textlist = example.text.split('|')
        assert len(textlist) == len(labellist)
        tokens = []
        labels = []
        for i, word in enumerate(textlist):
            # 分词，如果是中文，就是分字,但是对于一些不在BERT的vocab.txt中得字符会被进行WordPice处理（例如中文的引号），可以将所有的分字操作替换为list(input)
            token = tokenizer.tokenize(word)
            if len(token) > 1:
                print('1111111')
            tokens.extend(token)
            label_1 = labellist[i]
            for m in range(len(token)):
                if m == 0:
                    labels.append(label_1)
                else:  # 一般不会出现else
                    labels.append("X")
        assert len(tokens) == len(labels)
    else:
        tokens_list_index = []  # 表示原始对应当前分词后的结果
        tokens = []
        words_parts_ids=[]
        assert len(words_parts)==len(list(example.text))
        for i, word in enumerate(list(example.text)):
            current_tokens_index = []
            token = tokenizer.tokenize(word)
            for toke in token:
                current_tokens_index.append(len(tokens))
                words_parts_ids.append(words_parts[i])
                tokens.append(toke)
            tokens_list_index.append(current_tokens_index)
        words_parts=words_parts_ids
        # tokens = tokenizer.tokenize(example.text)
    try:
        assert len(words_parts)==len(tokens)
    except:
        print('1111111111111111111111')
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 2)]  # -2 的原因是因为序列需要加一个句首和句尾标志
        words_parts = words_parts[0:(max_seq_length - 2)]
        if mode != 'test':
            labels = labels[0:(max_seq_length - 2)]
    ntokens = []
    segment_ids = []
    if mode != 'test':
        label_ids = []
        label_ids.append(label_map["[CLS]"])  # O OR CLS 没有任何影响，不过我觉得O 会减少标签个数,不过拒收和句尾使用不同的标志来标注，使用LCS 也没毛病
    ntokens.append("[CLS]")  # 句子开始设置CLS 标志
    segment_ids.append(0)
    # words_parts.insert(0, part_speech_dcit['[CLS]'])
    # append("O") or append("[CLS]") not sure!
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        if mode != 'test':
            label_ids.append(label_map[labels[i]])
    ntokens.append("[SEP]")  # 句尾添加[SEP] 标志
    segment_ids.append(0)
    # words_parts.append(part_speech_dcit['[SEP]'])
    # append("O") or append("[SEP]") not sure!
    if mode != 'test':
        label_ids.append(label_map["[SEP]"])
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)  # 将序列中的字(ntokens)转化为ID形式
    input_mask = [1] * len(input_ids)
    # label_mask = [1] * len(input_ids)
    # padding, 使用
    # assert len(words_parts)==len(input_ids)
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        # words_parts.append(0)
        # we don't concerned about it!
        if mode != 'test':
            label_ids.append(0)
        ntokens.append("**NULL**")
        # label_mask.append(0)
    # print(len(input_ids))
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    # assert len(words_parts) == max_seq_length
    if mode != 'test':
        assert len(label_ids) == max_seq_length
    # assert len(label_mask) == max_seq_length

    # 打印部分样本数据信息
    if ex_index < 5:
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.guid))
        logger.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        # logger.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
        # logger.info("label_mask: %s" % " ".join([str(x) for x in label_mask]))

    # 结构化为一个类
    if mode != 'test':
        feature = InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            label_ids=label_ids,
            # label_mask = label_mask
        )
    else:
        feature = InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            label_ids=None,
            guid=example.guid,

        )
    # mode='test'的时候才有效
    write_tokens(ntokens, output_dir, mode)
    return feature


def filed_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file, output_dir, mode=None, part_speech_dcit=None):
    """
    将数据转化为TF_Record 结构，作为模型数据输入
    :param examples:  样本
    :param label_list:标签list
    :param max_seq_length: 预先设定的最大序列长度
    :param tokenizer: tokenizer 对象
    :param output_file: tf.record 输出路径
    :param mode:
    :return:
    """
    writer = tf.python_io.TFRecordWriter(output_file)
    # 遍历训练数据
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        # 对于每一个训练样本,
        if mode != 'test':
            assert len(example.text.split('|')) == len(example.label.split('|'))
        feature = convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, output_dir, mode,
                                         part_speech_dcit)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        def create_bytes_feature(value):
            return tf.train.Feature(bytes_list=tf.train.BytesList(
                value=[value.encode('utf - 8') if type(
                    value) == str else value]))

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        if mode != 'test':
            features["label_ids"] = create_int_feature(feature.label_ids)
        else:
            features["guid"] = create_bytes_feature(feature.guid)
        # features["words_parts_ids"] = create_int_feature(feature.words_parts)
        # features["label_mask"] = create_int_feature(feature.label_mask)
        # tf.train.Example/Feature 是一种协议，方便序列化？？？
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()

def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder):
    if is_training:
        name_to_features = {
            "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
            "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
            "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
            "label_ids": tf.FixedLenFeature([seq_length], tf.int64),
            # "words_parts_ids": tf.FixedLenFeature([seq_length], tf.int64),
            # "label_mask": tf.FixedLenFeature([seq_length], tf.int64),
        }

    else:
        name_to_features = {
            "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
            "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
            "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
            "guid": tf.FixedLenFeature([], tf.string),
            # "words_parts_ids": tf.FixedLenFeature([seq_length], tf.int64),
            # "label_ids": tf.FixedLenFeature([seq_length], tf.int64),
            # "label_ids":tf.VarLenFeature(tf.int64),
            # "label_mask": tf.FixedLenFeature([seq_length], tf.int64),
        }

    def _decode_record(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    def input_fn(params):
        batch_size = params["batch_size"]
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=300)
        d = d.apply(tf.data.experimental.map_and_batch(lambda record: _decode_record(record, name_to_features),
                                                       batch_size=batch_size,
                                                       num_parallel_calls=8,  # 并行处理数据的CPU核心数量，不要大于你机器的核心数
                                                       drop_remainder=drop_remainder))
        d = d.prefetch(buffer_size=4)
        return d

    return input_fn
