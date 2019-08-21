from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import re
import numpy as np
import six
from os.path import join
from six.moves import zip
import sentencepiece as spm

import tensorflow as tf
from preprocessing import _get_spm_basename
import function_builder 
import model_utils

PRETRAINED_MODEL_DIR_SP = 'xlnet_cased_L-24_H-1024_A-16/spiece.model'
OUTPUT_DIR = 'proc_data/squad'
PROC_ID = 0
NUM_PROC = 1
MAX_SEQ_LENGTH = 512
MAX_QUERY_LENGTH = 64
TRAIN_FILE = 'train-v2.0.json'
DOC_STRIDE = 128
SAVE_STEPS = 1000
MAX_SAVE = 5
model_config = {
    "num_core_per_host":1,
    "model_dir":'experiment/squad',
    "output_dir": 'proc_data/squad',
    "output_dir1": '',
    "iterations":1000,
    "num_hosts":1,
    "max_save":5,
    "save_steps":1000,
    "max_seq_length":512,
    "max_query_length":64,
    "train_batch_size":8,
    "train_steps":12000,
    "init_checkpoint":'xlnet_cased_L-24_H-1024_A-16/xlnet_model.ckpt',
    'model_config_path':'xlnet_cased_L-24_H-1024_A-16/xlnet_config.json',
    'use_bfloat16':False,
    'use_tpu':False,
    'dropout':0.1,
    'dropatt':0.1,
    'init':'normal',
    'init_range':0.1,
    'init_std':0.02,
    'clamp_len':-1,
    'mem_len':None,
    'reuse_len':None,
    'bi_data':False,
    'clamp_len':-1,
    'same_length':False,
    'shuffle_buffer':2048,
    'warmup_steps':1000,
    'learning_rate':2e-5,
    'decay_method':'poly',
    'min_lr_ratio':0.0,
    'weight_decay':0.00,
    'adam_epsilon':1e-6,
    'clip':1.0,
    'lr_layer_decay_rate':0.75,
    'start_n_top':5,
    'end_n_top':5

}


def configure_tpu():
    tpu_cluster = None
    master = None

    session_config = tf.ConfigProto(allow_soft_placement=True)
    # Uncomment the following line if you hope to monitor GPU RAM growth
    # session_config.gpu_options.allow_growth = True

    if model_config['num_core_per_host'] == 1:
        strategy = None
        tf.logging.info('Single device mode.')
    else:
        strategy = tf.contrib.distribute.MirroredStrategy(
            num_gpus=model_config['num_core_per_host'])
        tf.logging.info('Use MirroredStrategy with %d devices.',
                    strategy.num_replicas_in_sync)

    per_host_input = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        master=master,
        model_dir=model_config['model_dir'],
        session_config=session_config,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=model_config['iterations'],
            num_shards=model_config['num_hosts'] * model_config['num_core_per_host'],
            per_host_input_for_training=per_host_input),
        keep_checkpoint_max=model_config['max_save'],
        save_checkpoints_secs=None,
        save_checkpoints_steps=model_config['save_steps'],
        train_distribute=strategy
    )
    return run_config

def input_fn_builder(input_glob, seq_length, is_training, drop_remainder,
                     num_hosts, num_threads=8):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""
    name_to_features = {
        "unique_ids": tf.FixedLenFeature([], tf.int64),
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.float32),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "cls_index": tf.FixedLenFeature([], tf.int64),
        "p_mask": tf.FixedLenFeature([seq_length], tf.float32)
    }

    if is_training:
        name_to_features["start_positions"] = tf.FixedLenFeature([], tf.int64)
        name_to_features["end_positions"] = tf.FixedLenFeature([], tf.int64)
        name_to_features["is_impossible"] = tf.FixedLenFeature([], tf.float32)

        tf.logging.info("Input tfrecord file glob {}".format(input_glob))
        global_input_paths = tf.gfile.Glob(input_glob)
        tf.logging.info("Find {} input paths {}".format(
            len(global_input_paths), global_input_paths))

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.cast(t, tf.int32)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        # if FLAGS.use_tpu:
        #     batch_size = params["batch_size"]
        # elif is_training:
        batch_size = model_config['train_batch_size']
        # else:
        #     batch_size = FLAGS.predict_batch_size

    # Split tfrecords across hosts
        if num_hosts > 1:
            host_id = params["context"].current_host
            num_files = len(global_input_paths)
            if num_files >= num_hosts:
                num_files_per_host = (num_files + num_hosts - 1) // num_hosts
                my_start_file_id = host_id * num_files_per_host
                my_end_file_id = min((host_id + 1) * num_files_per_host, num_files)
                input_paths = global_input_paths[my_start_file_id: my_end_file_id]
            tf.logging.info("Host {} handles {} files".format(host_id,
                                                        len(input_paths)))
        else:
            input_paths = global_input_paths

        if len(input_paths) == 1:
            d = tf.data.TFRecordDataset(input_paths[0])
            # For training, we want a lot of parallel reading and shuffling.
            # For eval, we want no shuffling and parallel reading doesn't matter.
            if is_training:
                d = d.shuffle(buffer_size=model_config['shuffle_buffer'])
                d = d.repeat()
        else:
            d = tf.data.Dataset.from_tensor_slices(input_paths)
            # file level shuffle
            d = d.shuffle(len(input_paths)).repeat()

            # `cycle_length` is the number of parallel files that get read.
            cycle_length = min(num_threads, len(input_paths))

            d = d.apply(
                tf.contrib.data.parallel_interleave(
                tf.data.TFRecordDataset,
                sloppy=is_training,
                cycle_length=cycle_length))

            if is_training:
                # sample level shuffle
                d = d.shuffle(buffer_size=model_config['shuffle_buffer'])

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                num_parallel_batches=num_threads,
                drop_remainder=drop_remainder))
        d = d.prefetch(1024)

        return d

    return input_fn

def get_model_fn():
    def model_fn(features, labels, mode, params):
        #### Training or Evaluation
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        #### Get loss from inputs
        outputs = function_builder.get_qa_outputs(model_config, features, is_training)

        #### Check model parameters
        num_params = sum([np.prod(v.shape) for v in tf.trainable_variables()])
        tf.logging.info('#params: {}'.format(num_params))

        scaffold_fn = None

        #### Evaluation mode
        if mode == tf.estimator.ModeKeys.PREDICT:
            if model_config['init_checkpoint']:
                tf.logging.info("init_checkpoint not being used in predict mode.")

            predictions = {
                "unique_ids": features["unique_ids"],
                "start_top_index": outputs["start_top_index"],
                "start_top_log_probs": outputs["start_top_log_probs"],
                "end_top_index": outputs["end_top_index"],
                "end_top_log_probs": outputs["end_top_log_probs"],
                "cls_logits": outputs["cls_logits"]
            }

            # if FLAGS.use_tpu:
            #     output_spec = tf.contrib.tpu.TPUEstimatorSpec(
            #     mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
            # else:
            output_spec = tf.estimator.EstimatorSpec(
            mode=mode, predictions=predictions)
            return output_spec

        ### Compute loss
        seq_length = tf.shape(features["input_ids"])[1]
        def compute_loss(log_probs, positions):
            one_hot_positions = tf.one_hot(
            positions, depth=seq_length, dtype=tf.float32)

            loss = - tf.reduce_sum(one_hot_positions * log_probs, axis=-1)
            loss = tf.reduce_mean(loss)
            return loss

        start_loss = compute_loss(
            outputs["start_log_probs"], features["start_positions"])
        end_loss = compute_loss(
            outputs["end_log_probs"], features["end_positions"])

        total_loss = (start_loss + end_loss) * 0.5

        cls_logits = outputs["cls_logits"]
        is_impossible = tf.reshape(features["is_impossible"], [-1])
        regression_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=is_impossible, logits=cls_logits)
        regression_loss = tf.reduce_mean(regression_loss)

        # note(zhiliny): by default multiply the loss by 0.5 so that the scale is
        # comparable to start_loss and end_loss
        total_loss += regression_loss * 0.5

        #### Configuring the optimizer
        train_op, learning_rate, _ = model_utils.get_train_op(model_config, total_loss)

        monitor_dict = {}
        monitor_dict["lr"] = learning_rate

        #### load pretrained models
        scaffold_fn = model_utils.init_from_checkpoint(model_config)

        #### Constucting training TPUEstimatorSpec with new cache.
        # if FLAGS.use_tpu:
        #     host_call = function_builder.construct_scalar_host_call(
        #         monitor_dict=monitor_dict,
        #         model_dir=FLAGS.model_dir,
        #         prefix="train/",
        #         reduce_fn=tf.reduce_mean)

        #     train_spec = tf.contrib.tpu.TPUEstimatorSpec(
        #         mode=mode, loss=total_loss, train_op=train_op, host_call=host_call,
        #         scaffold_fn=scaffold_fn)
        # else:
        train_spec = tf.estimator.EstimatorSpec(
            mode=mode, loss=total_loss, train_op=train_op)

        return train_spec

    return model_fn

def main(_):
    sp_model = spm.SentencePieceProcessor()
    sp_model.Load(PRETRAINED_MODEL_DIR_SP)

    ### TPU Configuration
    run_config = configure_tpu()

    model_fn = get_model_fn()
    spm_basename = _get_spm_basename()


    estimator = tf.estimator.Estimator(
    model_fn=model_fn,
    config=run_config)

    
    train_rec_glob = os.path.join(
        model_config['output_dir1'],
        "{}.*.slen-{}.qlen-{}.train.tf_record".format(
        spm_basename, model_config['max_seq_length'],
        model_config['max_query_length']))

    train_input_fn = input_fn_builder(
        input_glob=train_rec_glob,
        seq_length=model_config['max_seq_length'],
        is_training=True,
        drop_remainder=True,
        num_hosts=model_config['num_hosts'])

    estimator.train(input_fn=train_input_fn, max_steps=model_config['train_steps'])

if __name__ == "__main__":
  tf.app.run()