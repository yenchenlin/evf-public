from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import errno
import json
import os
import random
import time

import numpy as np
import tensorflow as tf

from video_prediction import datasets, models
from video_prediction.variables import VariableState, average_vars, interpolate_vars
from video_prediction.datasets.omnipush_dataset import read_dataset
from evaluate import save_prediction_eval_results, load_metrics

def add_tag_suffix(summary, tag_suffix):
    summary_proto = tf.Summary()
    summary_proto.ParseFromString(summary)
    summary = summary_proto

    for value in summary.value:
        tag_split = value.tag.split('/')
        value.tag = '/'.join([tag_split[0] + tag_suffix] + tag_split[1:])
    return summary.SerializeToString()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="either a directory containing subdirectories "
                                                                     "train, val, test, etc, or a directory containing "
                                                                     "the tfrecords")
    parser.add_argument("--results_dir", type=str, default='results', help="ignored if output_dir is specified")
    parser.add_argument("--output_dir", help="output directory where results are saved. default is results_dir/model_fname, "
                                             "where model_fname is the directory name of checkpoint")
    parser.add_argument("--checkpoint", help="directory with checkpoint or checkpoint name (e.g. checkpoint_dir/model-200000)")

    parser.add_argument("--dataset", type=str, help="dataset class name")
    parser.add_argument("--dataset_hparams", type=str, help="a string of comma separated list of dataset hyperparameters")
    parser.add_argument("--model", type=str, help="model class name")
    parser.add_argument("--model_hparams", type=str, help="a string of comma separated list of model hyperparameters")

    parser.add_argument("--batch_size", type=int, default=4, help="number of samples in batch")
    parser.add_argument("--num_samples", type=int, help="number of samples in total (all of them by default)")
    parser.add_argument("--num_epochs", type=int, default=1)

    parser.add_argument("--eval_substasks", type=str, nargs='+', default=['max', 'avg', 'min'], help='subtasks to evaluate (e.g. max, avg, min)')
    parser.add_argument("--only_metrics", action='store_true')
    parser.add_argument("--num_stochastic_samples", type=int, default=100)

    parser.add_argument("--eval_parallel_iterations", type=int, default=10)
    parser.add_argument("--gpu_mem_frac", type=float, default=0.8, help="fraction of gpu memory to use")
    parser.add_argument("--seed", type=int)

    parser.add_argument("--meta_batch_size", type=int, default=5, help="how many inner-loops to run")
    parser.add_argument("--inner_iters", type=int, default=5, help="number of inner-loop iterations")
    parser.add_argument("--meta_step_size", type=float, default=1.0, help="initial step size of meta optimization")
    parser.add_argument("--final_meta_step_size", type=float, default=0.0, help="final sep size of meta optimization")



    args = parser.parse_args()

    if args.seed is not None:
        tf.set_random_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    dataset_hparams_dict = {}
    model_hparams_dict = {}
    if args.checkpoint:
        checkpoint_dir = os.path.normpath(args.checkpoint)
        if not os.path.isdir(args.checkpoint):
            checkpoint_dir, _ = os.path.split(checkpoint_dir)
        if not os.path.exists(checkpoint_dir):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), checkpoint_dir)
        with open(os.path.join(checkpoint_dir, "options.json")) as f:
            print("loading options from checkpoint %s" % args.checkpoint)
            options = json.loads(f.read())
            args.dataset = args.dataset or options['dataset']
            args.model = args.model or options['model']
        try:
            with open(os.path.join(checkpoint_dir, "dataset_hparams.json")) as f:
                dataset_hparams_dict = json.loads(f.read())
        except FileNotFoundError:
            print("dataset_hparams.json was not loaded because it does not exist")
        try:
            with open(os.path.join(checkpoint_dir, "model_hparams.json")) as f:
                model_hparams_dict = json.loads(f.read())
        except FileNotFoundError:
            print("model_hparams.json was not loaded because it does not exist")
        args.output_dir = args.output_dir or os.path.join(args.results_dir, os.path.split(checkpoint_dir)[1])
    else:
        if not args.dataset:
            raise ValueError('dataset is required when checkpoint is not specified')
        if not args.model:
            raise ValueError('model is required when checkpoint is not specified')
        args.output_dir = args.output_dir or os.path.join(args.results_dir, 'model.%s' % args.model)

    print('----------------------------------- Options ------------------------------------')
    for k, v in args._get_kwargs():
        print(k, "=", v)
    print('------------------------------------- End --------------------------------------')

    # Dataset
    val_sets = read_dataset(args.input_dir, mode='val',
                            hparams_dict=dataset_hparams_dict,
                            hparams=args.dataset_hparams)
    val_sets = list(val_sets)

    # Backward compatibility, used to set hypermeter for others
    train_dataset = val_sets[0]

    variable_scope = tf.get_variable_scope()
    variable_scope.set_use_resource(True)

    VideoPredictionModel = models.get_model_class(args.model)
    hparams_dict = dict(model_hparams_dict)
    hparams_dict.update({
        'context_frames': train_dataset.hparams.context_frames,
        'sequence_length': train_dataset.hparams.sequence_length,
        'repeat': train_dataset.hparams.time_shift,
    })
    model = VideoPredictionModel(
        hparams_dict=hparams_dict,
        hparams=args.model_hparams,
        eval_num_samples=args.num_stochastic_samples,
        eval_parallel_iterations=args.eval_parallel_iterations)

    batch_size = args.batch_size
    assert batch_size == 4

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(args), sort_keys=True, indent=4))
    with open(os.path.join(output_dir, "dataset_hparams.json"), "w") as f:
        f.write(json.dumps(val_sets[0].hparams.values(), sort_keys=True, indent=4))
    with open(os.path.join(output_dir, "model_hparams.json"), "w") as f:
        f.write(json.dumps(model.hparams.values(), sort_keys=True, indent=4))

    """ Each val set contains 10 examples, use first 4 for val train,
        next 4 for val test, drop last 2"""
    # Val train set
    val_train_tf_datasets = [dataset.make_dataset(batch_size, skip_size=0, take_size=4) for dataset in val_sets]
    val_train_iterators = [tf_dataset.make_one_shot_iterator() for tf_dataset in val_train_tf_datasets]
    val_train_handles = [iterator.string_handle() for iterator in val_train_iterators]

    # Val test set
    val_test_tf_datasets = [dataset.make_dataset(batch_size, skip_size=4, take_size=4) for dataset in val_sets]
    val_test_iterators = [tf_dataset.make_one_shot_iterator() for tf_dataset in val_test_tf_datasets]
    val_test_handles = [iterator.string_handle() for iterator in val_test_iterators]

    # Backward compatibility, use first train set to build graph
    val_train_handle = val_train_handles[0]
    iterator = tf.data.Iterator.from_string_handle(val_train_handle, val_train_tf_datasets[0].output_types, val_train_tf_datasets[0].output_shapes)
    inputs = iterator.get_next()
    # inputs comes from the "first training dataset" by default, unless train_handle is remapped to other handles
    model.build_graph(inputs)

    with tf.name_scope("parameter_count"):
        # exclude trainable variables that are replicas (used in multi-gpu setting)
        trainable_variables = set(tf.trainable_variables()) & set(model.saveable_variables)
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in trainable_variables])

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_mem_frac)
    config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
    sess = tf.Session(config=config)

    model.restore(sess, args.checkpoint)

    print("parameter_count =", sess.run(parameter_count))
    print("number of test sets =", len(val_sets))

    # Evaluate handle for each dataset
    val_train_handle_evals = [sess.run(handle) for handle in val_train_handles]
    val_test_handle_evals = [sess.run(handle) for handle in val_test_handles]

    # Set input for the first step
    current_handle_eval = random.choice(val_train_handle_evals)

    # Set up variables recorder
    model._state = VariableState(sess, tf.trainable_variables())
    sess.graph.finalize()

    sample_ind = 0
    for i, (val_train_handle_eval, val_test_handle_eval) in enumerate(zip(val_train_handle_evals, val_test_handle_evals)):
        print('evaluating %d / %d test set' % (i+1, len(val_train_handle_evals)))
        old_vars = model._state.export_variables()

        # Inner update
        train_fetches = {"train_op": model.train_op}
        for i in range(args.inner_iters):
            _ = sess.run(train_fetches, feed_dict={val_train_handle: val_train_handle_eval})

        # compute "best" metrics using the computation graph
        fetches = {'images': model.inputs['images']}
        fetches.update(model.eval_outputs.items())
        fetches.update(model.eval_metrics.items())
        results = sess.run(fetches, feed_dict={val_train_handle: val_test_handle_eval})
        save_prediction_eval_results(os.path.join(output_dir, 'prediction_eval'),
                                     results, model.hparams, sample_ind, args.only_metrics, args.eval_substasks)

        # Return to original model parameters
        model._state.import_variables(old_vars)
        sample_ind += args.batch_size

    metric_fnames = []
    metric_names = ['psnr', 'ssim', 'lpips']
    subtasks = ['max']
    for metric_name in metric_names:
        for subtask in subtasks:
            metric_fnames.append(
                os.path.join(output_dir, 'prediction_eval_%s_%s' % (metric_name, subtask), 'metrics', metric_name))

    for metric_fname in metric_fnames:
        task_name, _, metric_name = metric_fname.split('/')[-3:]
        metric = load_metrics(metric_fname)
        print('=' * 31)
        print(task_name, metric_name)
        print('-' * 31)
        metric_header_format = '{:>10} {:>20}'
        metric_row_format = '{:>10} {:>10.4f} ({:>7.4f})'
        print(metric_header_format.format('time step', os.path.split(metric_fname)[1]))
        for t, (metric_mean, metric_std) in enumerate(zip(metric.mean(axis=0), metric.std(axis=0))):
            print(metric_row_format.format(t, metric_mean, metric_std))
        print(metric_row_format.format('mean (std)', metric.mean(), metric.std()))
        print('=' * 31)


if __name__ == '__main__':
    main()
