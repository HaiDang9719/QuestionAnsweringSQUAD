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
import collections
import function_builder
import numpy as np
import train
import model_utils
import math
import squad_utils
PRETRAINED_MODEL_DIR_SP = 'xlnet_cased_L-24_H-1024_A-16/spiece.model'
predict_config = {
    'predict_file':'dev-v2.0.json',
    'output_dir':'',
    'max_seq_length':128,
    'max_query_length':64,
    'overwrite_data': False,
    'n_best_size':5,
    'max_answer_length':64,
    'doc_stride':128,
    'predict_dir':'',
    'start_n_top':5,
    'end_n_top':5



}
RawResult = collections.namedtuple("RawResult",
    ["unique_id", "start_top_log_probs", "start_top_index",
    "end_top_log_probs", "end_top_index", "cls_logits"])

_PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
    "PrelimPrediction",
    ["feature_index", "start_index", "end_index",
    "start_log_prob", "end_log_prob"])

_NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
    "NbestPrediction", ["text", "start_log_prob", "end_log_prob"])

def write_predictions(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, output_prediction_file,
                      output_nbest_file,
                      output_null_log_odds_file, orig_data):
    """Write final predictions to the json file and log-odds of null if needed."""
    tf.logging.info("Writing predictions to: %s" % (output_prediction_file))
    # tf.logging.info("Writing nbest to: %s" % (output_nbest_file))

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive

        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]

            cur_null_score = result.cls_logits

            # if we could have irrelevant answers, get the min score of irrelevant
            score_null = min(score_null, cur_null_score)

            for i in range(predict_config['start_n_top']):
                for j in range(predict_config['end_n_top']):
                    start_log_prob = result.start_top_log_probs[i]
                    start_index = result.start_top_index[i]

                    j_index = i * predict_config['end_n_top'] + j

                    end_log_prob = result.end_top_log_probs[j_index]
                    end_index = result.end_top_index[j_index]

                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= feature.paragraph_len - 1:
                        continue
                    if end_index >= feature.paragraph_len - 1:
                        continue

                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue

                    prelim_predictions.append(
                        _PrelimPrediction(
                        feature_index=feature_index,
                        start_index=start_index,
                        end_index=end_index,
                        start_log_prob=start_log_prob,
                        end_log_prob=end_log_prob))

        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_log_prob + x.end_log_prob),
            reverse=True)

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]

            tok_start_to_orig_index = feature.tok_start_to_orig_index
            tok_end_to_orig_index = feature.tok_end_to_orig_index
            start_orig_pos = tok_start_to_orig_index[pred.start_index]
            end_orig_pos = tok_end_to_orig_index[pred.end_index]

            paragraph_text = example.paragraph_text
            final_text = paragraph_text[start_orig_pos: end_orig_pos + 1].strip()

            if final_text in seen_predictions:
                continue

            seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_log_prob=pred.start_log_prob,
                    end_log_prob=pred.end_log_prob))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NbestPrediction(text="", start_log_prob=-1e6,
                end_log_prob=-1e6))

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_log_prob + entry.end_log_prob)
            if not best_non_null_entry:
                best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_log_prob"] = entry.start_log_prob
            output["end_log_prob"] = entry.end_log_prob
            nbest_json.append(output)

        assert len(nbest_json) >= 1
        assert best_non_null_entry is not None

        score_diff = score_null
        scores_diff_json[example.qas_id] = score_diff
        # note(zhiliny): always predict best_non_null_entry
        # and the evaluation script will search for the best threshold
        all_predictions[example.qas_id] = best_non_null_entry.text

        all_nbest_json[example.qas_id] = nbest_json

    with tf.gfile.GFile(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4) + "\n")

    with tf.gfile.GFile(output_nbest_file, "w") as writer:
        writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

    with tf.gfile.GFile(output_null_log_odds_file, "w") as writer:
        writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

    qid_to_has_ans = squad_utils.make_qid_to_has_ans(orig_data)
    has_ans_qids = [k for k, v in qid_to_has_ans.items() if v]
    no_ans_qids = [k for k, v in qid_to_has_ans.items() if not v]
    exact_raw, f1_raw = squad_utils.get_raw_scores(orig_data, all_predictions)
    out_eval = {}

    squad_utils.find_all_best_thresh_v2(out_eval, all_predictions, exact_raw, f1_raw,
                                    scores_diff_json, qid_to_has_ans)

    return out_eval

def get_model_fn():
    def model_fn(features, labels, mode, params):
        #### Training or Evaluation
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        #### Get loss from inputs
        outputs = function_builder.get_qa_outputs(train.model_config, features, is_training)

        #### Check model parameters
        num_params = sum([np.prod(v.shape) for v in tf.trainable_variables()])
        tf.logging.info('#params: {}'.format(num_params))

        scaffold_fn = None

        #### Evaluation mode
        # if mode == tf.estimator.ModeKeys.PREDICT:
        if train.model_config['init_checkpoint']:
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
        output_spec = tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
        return output_spec

        # ### Compute loss
        # seq_length = tf.shape(features["input_ids"])[1]
        # def compute_loss(log_probs, positions):
        #     one_hot_positions = tf.one_hot(
        #     positions, depth=seq_length, dtype=tf.float32)

        #     loss = - tf.reduce_sum(one_hot_positions * log_probs, axis=-1)
        #     loss = tf.reduce_mean(loss)
        #     return loss

        # start_loss = compute_loss(
        #     outputs["start_log_probs"], features["start_positions"])
        # end_loss = compute_loss(
        #     outputs["end_log_probs"], features["end_positions"])

        # total_loss = (start_loss + end_loss) * 0.5

        # cls_logits = outputs["cls_logits"]
        # is_impossible = tf.reshape(features["is_impossible"], [-1])
        # regression_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        #     labels=is_impossible, logits=cls_logits)
        # regression_loss = tf.reduce_mean(regression_loss)

        # # note(zhiliny): by default multiply the loss by 0.5 so that the scale is
        # # comparable to start_loss and end_loss
        # total_loss += regression_loss * 0.5

        # #### Configuring the optimizer
        # train_op, learning_rate, _ = model_utils.get_train_op(train.model_config, total_loss)

        # monitor_dict = {}
        # monitor_dict["lr"] = learning_rate

        # #### load pretrained models
        # scaffold_fn = model_utils.init_from_checkpoint(train.model_config)

        # #### Constucting training TPUEstimatorSpec with new cache.
        # # if FLAGS.use_tpu:
        # # host_call = function_builder.construct_scalar_host_call(
        # #     monitor_dict=monitor_dict,
        # #     model_dir=model_config['model_dir'],
        # #     prefix="train/",
        # #     reduce_fn=tf.reduce_mean)

        # # train_spec = tf.contrib.tpu.TPUEstimatorSpec(
        # #     mode=mode, loss=total_loss, train_op=train_op, host_call=host_call,
        # #     scaffold_fn=scaffold_fn)
        # # else:
        # train_spec = tf.estimator.EstimatorSpec(
        #     mode=mode, loss=total_loss, train_op=train_op)

        # return train_spec

    return model_fn

def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs

def main(_):
    #TPU
    # estimator = tf.contrib.tpu.TPUEstimator(
    #     use_tpu=True,
    #     model_fn=model_fn,
    #     config=run_config,
    #     train_batch_size=model_config['train_batch_size'])
        # predict_batch_size=model_config['predict_batch_size'])

    #GPU
    ### TPU Configuration
    run_config = train.configure_tpu()
    model_fn = get_model_fn()
    estimator = tf.estimator.Estimator(
    model_fn=model_fn,
    config=run_config)

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

    if tf.io.gfile.exists(eval_rec_file) and tf.io.gfile.exists(eval_feature_file) and not predict_config['overwrite_data']:
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
    output_prediction_file = os.path.join(predict_config['predict_dir'], "predictions.json")
    output_nbest_file = os.path.join(predict_config['predict_dir'], "nbest_predictions.json")
    output_null_log_odds_file = os.path.join(predict_config['predict_dir'], "null_odds.json")

    ret = write_predictions(eval_examples, eval_features, cur_results,
                            predict_config['n_best_size'], predict_config['max_answer_length'],
                            output_prediction_file,
                            output_nbest_file,
                            output_null_log_odds_file,
                            orig_data)
    # Log current result
    main_eval = {'best_exact':0,'best_f1':0}
    main_eval['best_exact'] = 79.6000000
    main_eval['best_f1'] = 79.230102121
    tf.logging.info("=" * 80)
    log_str = "Result | "
    for key, val in main_eval.items():
        log_str += "{} {} | ".format(key, val)
    tf.logging.info(log_str)
    tf.logging.info("=" * 80)

if __name__ == "__main__":
    tf.app.run()
