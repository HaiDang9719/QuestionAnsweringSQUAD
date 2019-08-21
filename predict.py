import preprocessing
import json
import os
import tensorflow as tf
import sentencepiece as spm
import six
import train
if six.PY2:
    import cPickle as pickle
else:
    import pickle

PRETRAINED_MODEL_DIR_SP = 'xlnet_cased_L-24_H-1024_A-16/spiece.model'
predict_config = {
    'predict_file':'dev-v2.0.json',
    'output_dir':'',
    'max_seq_length':512,
    'max_query_length':64,
    'overwrite_data': False

}

def main(_):

    sp_model = spm.SentencePieceProcessor()
    sp_model.Load(PRETRAINED_MODEL_DIR_SP)
    spm_basename = preprocessing._get_spm_basename()
    eval_examples = preprocessing.read_squad_examples(predict_config['predict_file'], is_training = False)

    with tf.gfile.Open(predict_config['predict_file']) as f:
        orig_data = json.load(f)["data"]
    
    eval_rec_file = os.path.join(predict_config['output_dir'],
        "{}.slen-{}.qlen-{}.eval.tf_record".format(
            spm_basename, predict_config['max_seq_length'], predict_config['max_query_length']))
    eval_feature_file = os.path.join(
        predict_config['output_dir'],
        "{}.slen-{}.qlen-{}.eval.features.pkl".format(
            spm_basename, predict_config['max_seq_length'], predict_config['max_query_length']))

    if tf.gfile.Exists(eval_rec_file) and tf.gfile.Exists(eval_feature_file) and not predict_config['overwrite_data']:
        tf.logging.info("Loading eval features from {}".format(eval_feature_file))
        with tf.gfile.Open(eval_feature_file, 'rb') as fin:
            eval_features = pickle.load(fin)
    else:
        eval_writer = preprocessing.FeatureWriter(filename=eval_rec_file, is_training=False)
        eval_features = []

    def append_feature(feature):
        eval_features.append(feature)
        eval_writer.process_feature(feature)

        preprocessing.convert_examples_to_features(
            examples=eval_examples,
            sp_model=sp_model,
            max_seq_length=predict_config['max_seq_length'],
            doc_stride=predict_config['doc_stride'],
            max_query_length=predict_config['max_query_length'],
            is_training=False,
            output_fn=append_feature)
        eval_writer.close()
        with tf.gfile.Open(eval_feature_file, 'wb') as fout:
            pickle.dump(eval_features, fout)
    eval_input_fn = train.input_fn_builder(
        input_glob=eval_rec_file,
        seq_length=predict_config['max_seq_length'],
        is_training=False,
        drop_remainder=False,
        num_hosts=1)

    cur_results = []
    for result in estimator.predict(input_fn=eval_input_fn, yield_single_examples=True):

        if len(cur_results) % 1000 == 0:
            tf.logging.info("Processing example: %d" % (len(cur_results)))

        unique_id = int(result["unique_ids"])
        start_top_log_probs = ([float(x) for x in result["start_top_log_probs"].flat])
        start_top_index = [int(x) for x in result["start_top_index"].flat]
        end_top_log_probs = ([float(x) for x in result["end_top_log_probs"].flat])
        end_top_index = [int(x) for x in result["end_top_index"].flat]

        cls_logits = float(result["cls_logits"].flat[0])

        cur_results.append(
            RawResult(
                unique_id=unique_id,
                start_top_log_probs=start_top_log_probs,
                start_top_index=start_top_index,
                end_top_log_probs=end_top_log_probs,
                end_top_index=end_top_index,
                cls_logits=cls_logits))
    output_prediction_file = os.path.join(FLAGS.predict_dir, "predictions.json")
    output_nbest_file = os.path.join(FLAGS.predict_dir, "nbest_predictions.json")
    output_null_log_odds_file = os.path.join(FLAGS.predict_dir, "null_odds.json")

    ret = write_predictions(eval_examples, eval_features, cur_results,
                            FLAGS.n_best_size, FLAGS.max_answer_length,
                            output_prediction_file,
                            output_nbest_file,
                            output_null_log_odds_file,
                            orig_data)
    # Log current result
    tf.logging.info("=" * 80)
    log_str = "Result | "
    for key, val in ret.items():
        log_str += "{} {} | ".format(key, val)
    tf.logging.info(log_str)
    tf.logging.info("=" * 80)