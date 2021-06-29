# -*- coding: utf-8 -*-
"""
Comment/uncomment line 270-272 to choose one persona fusion strategy
"""
import os
import collections
import tensorflow as tf
from tqdm import tqdm
import tokenization


tf.flags.DEFINE_string("train_file", "../../data/personachat_processed/processed_train_self_original.txt", 
	"path to train file")
tf.flags.DEFINE_string("valid_file", "../../data/personachat_processed/processed_valid_self_original.txt", 
	"path to valid file")
tf.flags.DEFINE_string("test_file", "../../data/personachat_processed/processed_test_self_original.txt", 
    "path to test file")

tf.flags.DEFINE_string("vocab_file", "../uncased_L-12_H-768_A-12/vocab.txt", 
    "path to vocab file")
tf.flags.DEFINE_integer("max_seq_length", 280, 
	"max sequence length of concatenated context and response")
tf.flags.DEFINE_bool("do_lower_case", True,
    "whether to lower case the input text")



def print_configuration_op(FLAGS):
    print('My Configurations:')
    for name, value in FLAGS.__flags.items():
        value=value.value
        if type(value) == float:
            print(' %s:\t %f'%(name, value))
        elif type(value) == int:
            print(' %s:\t %d'%(name, value))
        elif type(value) == str:
            print(' %s:\t %s'%(name, value))
        elif type(value) == bool:
            print(' %s:\t %s'%(name, value))
        else:
            print('%s:\t %s' % (name, value))
    print('End of configuration')


def load_dataset_train(fname, tfrecord_path="data_tfrecord/"):

    if not os.path.exists(tfrecord_path):
        os.makedirs(tfrecord_path)

    processed_fnames = []
    for epoch_id in range(19):
        fname_list = fname.split("/")[-1].split(".")
        processed_fname = tfrecord_path + fname_list[0] + "_" + str(epoch_id) + "." + fname_list[1]
        processed_fnames.append(processed_fname)
        dataset_size = 0
        print("Generating the file of {} ...".format(processed_fname))

        with open(processed_fname, 'w') as fw:
            with open(fname, 'rt') as fr:
                context_id = 0
                for line in fr:
                    line = line.strip()
                    fields = line.split('\t')
                    
                    context_id += 1
                    context = fields[0]

                    persona = ""
                    if fields[3] != "NA":
                        persona += " ".join(fields[3].split("|"))
                    if fields[4] != "NA":
                        persona += " ".join(fields[4].split("|"))
                    context = persona + " _eop_ " + context

                    responses = fields[1].split('|')
                    # positive
                    positive_response_id = int(fields[2])
                    positive_response = responses[positive_response_id]
                    fw.write("\t".join([str(context_id), context, str(positive_response_id), positive_response, "follow"]))
                    fw.write('\n')
                    dataset_size += 1
                    # negative
                    negative_response_id = epoch_id
                    if negative_response_id >= positive_response_id:
                        negative_response_id += 1
                    negative_response = responses[negative_response_id]
                    fw.write("\t".join([str(context_id), context, str(negative_response_id), negative_response, "unfollow"]))
                    fw.write('\n')
                    dataset_size += 1

                print("{} dataset_size: {}".format(processed_fname, dataset_size))
            
    return processed_fnames


def load_dataset_test(fname, tfrecord_path="data_tfrecord/"):

    if not os.path.exists(tfrecord_path):
        os.makedirs(tfrecord_path)

    processed_fname = tfrecord_path + fname.split("/")[-1]
    dataset_size = 0
    print("Generating the file of {} ...".format(processed_fname))

    with open(processed_fname, 'w') as fw:
        with open(fname, 'rt') as fr:
            context_id = 0
            for line in fr:
                line = line.strip()
                fields = line.split('\t')
                
                context_id += 1
                context = fields[0]
                responses = fields[1].split('|')
                label_index = int(fields[2])

                persona = ""
                if fields[3] != "NA":
                    persona += " ".join(fields[3].split("|"))
                if fields[4] != "NA":
                    persona += " ".join(fields[4].split("|"))
                context = persona + " _eop_ " + context

                for response_id, response in enumerate(responses):
                    if response_id == label_index:
                        label = "follow"
                    else:
                        label = "unfollow"
                    response_id = context_id*100 + response_id
                    dataset_size += 1
                    fw.write("\t".join([str(context_id), context, str(response_id), response, label]))
                    fw.write('\n')
    
    print("{} dataset_size: {}".format(processed_fname, dataset_size))            
    return processed_fname


class InputExample(object):
    def __init__(self, guid, text_a_id, text_a, text_b_id, text_b=None, label=None):
        """Constructs a InputExample."""
        self.guid = guid
        self.text_a_id = text_a_id
        self.text_a = text_a
        self.text_b_id = text_b_id
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, text_a_id, text_b_id, input_ids, input_mask, segment_ids, label_id):
        self.text_a_id = text_a_id
        self.text_b_id = text_b_id
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def read_processed_file(input_file):
    lines = []
    num_lines = sum(1 for line in open(input_file, 'r'))
    with open(input_file, 'r') as f:
        for line in tqdm(f, total=num_lines):
            concat = []
            temp = line.rstrip().split('\t')
            concat.append(temp[0]) # context id
            concat.append(temp[1]) # context
            concat.append(temp[2]) # response id
            concat.append(temp[3]) # response
            concat.append(temp[4]) # label
            lines.append(concat)
    return lines


def create_examples(lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
        guid = "%s-%s" % (set_type, str(i))
        text_a_id = line[0]
        text_a = tokenization.convert_to_unicode(line[1])
        text_b_id = line[2]
        text_b = tokenization.convert_to_unicode(line[3])
        label = tokenization.convert_to_unicode(line[-1])
        examples.append(InputExample(guid=guid, text_a_id=text_a_id, text_a=text_a, text_b_id=text_b_id, text_b=text_b, label=label))
    return examples


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {}  # label
    for (i, label) in enumerate(label_list):  # ['0', '1']
        label_map[label] = i

    features = []  # feature
    for (ex_index, example) in enumerate(examples):
        text_a_id = int(example.text_a_id)
        text_b_id = int(example.text_b_id)

        text_a_sentences = example.text_a.split(" _eop_ ")
        persona_sentence = text_a_sentences[0]
        context_sentence = text_a_sentences[1]

        persona_sentence_tokens = tokenizer.tokenize(persona_sentence)
        context_sentence_tokens = tokenizer.tokenize(context_sentence)
        response_sentence_tokens = tokenizer.tokenize(example.text_b)

        input_ids = []
        input_mask = []
        segment_ids = []

        # context-response pair
        _truncate_seq_pair(context_sentence_tokens, response_sentence_tokens, max_seq_length - 3)

        tokens_temp = []
        segment_ids_temp = []

        tokens_temp.append("[CLS]")
        segment_ids_temp.append(0)
        for token in context_sentence_tokens:
            tokens_temp.append(token)
            segment_ids_temp.append(0)
        tokens_temp.append("[SEP]")
        segment_ids_temp.append(0)

        for token in response_sentence_tokens:
            tokens_temp.append(token)
            segment_ids_temp.append(1)
        tokens_temp.append("[SEP]")
        segment_ids_temp.append(1)

        input_ids_temp = tokenizer.convert_tokens_to_ids(tokens_temp)

        input_mask_temp = [1] * len(input_ids_temp)  # mask

        # Zero-pad up to the sequence length.
        while len(input_ids_temp) < max_seq_length:
            input_ids_temp.append(0)
            input_mask_temp.append(0)
            segment_ids_temp.append(0)

        assert len(input_ids_temp) == max_seq_length
        assert len(input_mask_temp) == max_seq_length
        assert len(segment_ids_temp) == max_seq_length

        input_ids.extend(input_ids_temp)
        input_mask.extend(input_mask_temp)
        segment_ids.extend(segment_ids_temp)


        # persona fusion strategies
        X_sentence_tokens = []                          # none-aware
        # X_sentence_tokens = context_sentence_tokens   # context-aware
        # X_sentence_tokens = response_sentence_tokens  # response-aware
        _truncate_seq_pair(persona_sentence_tokens, X_sentence_tokens, max_seq_length - 3)

        tokens_temp = []
        segment_ids_temp = []

        tokens_temp.append("[CLS]")
        segment_ids_temp.append(0)
        for token in persona_sentence_tokens:
            tokens_temp.append(token)
            segment_ids_temp.append(0)
        tokens_temp.append("[SEP]")
        segment_ids_temp.append(0)

        for token in X_sentence_tokens:
            tokens_temp.append(token)
            segment_ids_temp.append(1)
        tokens_temp.append("[SEP]")
        segment_ids_temp.append(1)

        input_ids_temp = tokenizer.convert_tokens_to_ids(tokens_temp)

        input_mask_temp = [1] * len(input_ids_temp)  # mask

        # Zero-pad up to the sequence length.
        while len(input_ids_temp) < max_seq_length:
            input_ids_temp.append(0)
            input_mask_temp.append(0)
            segment_ids_temp.append(0)

        assert len(input_ids_temp) == max_seq_length
        assert len(input_mask_temp) == max_seq_length
        assert len(segment_ids_temp) == max_seq_length

        input_ids.extend(input_ids_temp)
        input_mask.extend(input_mask_temp)
        segment_ids.extend(segment_ids_temp)


        label_id = label_map[example.label]

        if ex_index%2000 == 0:
            print('convert_{}_examples_to_features'.format(ex_index))

        features.append(
            InputFeatures(  # object
                text_a_id=text_a_id,
                text_b_id=text_b_id,
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_id=label_id))

    return features


def write_instance_to_example_files(instances, output_files):
    writers = []

    for output_file in output_files:
        writers.append(tf.python_io.TFRecordWriter(output_file))

    writer_index = 0
    total_written = 0
    for (inst_index, instance) in enumerate(instances):
        features = collections.OrderedDict()
        features["text_a_id"] = create_int_feature([instance.text_a_id])
        features["text_b_id"] = create_int_feature([instance.text_b_id])
        features["input_ids"] = create_int_feature(instance.input_ids)
        features["input_mask"] = create_int_feature(instance.input_mask)
        features["segment_ids"] = create_int_feature(instance.segment_ids)
        features["label_ids"] = create_float_feature([instance.label_id])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))

        writers[writer_index].write(tf_example.SerializeToString())
        writer_index = (writer_index + 1) % len(writers)

        total_written += 1

        # if inst_index < 5:
        # 	print("*** Example ***")
        # 	print("text_a_id: %s" % instance.text_a_id)
        # 	print("text_b_id: %s"  % instance.text_b_id)
        # 	print("input_ids: %s" % " ".join([str(tokenization.printable_text(x)) for x in instance.input_ids]))
        # 	print("input_mask: %s" % " ".join([str(tokenization.printable_text(x)) for x in instance.input_mask]))
        # 	print("segment_ids: %s" % " ".join([str(tokenization.printable_text(x)) for x in instance.segment_ids]))
        # 	print("label_id: %s" % instance.label_id)

    print("write_{}_instance_to_example_files".format(total_written))

    for feature_name in features.keys():
        feature = features[feature_name]
        values = []
    if feature.int64_list.value:
        values = feature.int64_list.value
    elif feature.float_list.value:
        values = feature.float_list.value
    tf.logging.info(
        "%s: %s" % (feature_name, " ".join([str(x) for x in values])))

    for writer in writers:
        writer.close()


def create_int_feature(values):
	feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
	return feature

def create_float_feature(values):
	feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
	return feature



if __name__ == "__main__":

    FLAGS = tf.flags.FLAGS

    print_configuration_op(FLAGS)

    train_filenames = load_dataset_train(FLAGS.train_file)
    valid_filename = load_dataset_test(FLAGS.valid_file)
    test_filename = load_dataset_test(FLAGS.test_file)

    filenames = train_filenames + [valid_filename, test_filename]
    filetypes = ["train"] * len(train_filenames) + ["valid", "test"]
    files = zip(filenames, filetypes)

    label_list = ["unfollow", "follow"]
    tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    # [exp1, exp2...] example X is a class object: {guid, text_a(str), text_b(str), label(str)}
    for (filename, filetype) in files:
        examples = create_examples(read_processed_file(filename), filetype)
        features = convert_examples_to_features(examples, label_list, FLAGS.max_seq_length, tokenizer)
        new_filename = filename[:-4] + ".tfrecord"
        write_instance_to_example_files(features, [new_filename])
        print('Convert {} to {} done'.format(filename, new_filename))

    print("Sub-process(es) done.")
