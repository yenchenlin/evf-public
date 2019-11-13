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
    parser.add_argument("--val_input_dir", type=str, help="directories containing the tfrecords. default: input_dir")
    parser.add_argument("--logs_dir", default='logs', help="ignored if output_dir is specified")
    parser.add_argument("--output_dir", help="output directory where json files, summary, model, gifs, etc are saved. "
                                             "default is logs_dir/model_fname, where model_fname consists of "
                                             "information from model and model_hparams")
    parser.add_argument("--output_dir_postfix", default="")
    parser.add_argument("--checkpoint", help="directory with checkpoint or checkpoint name (e.g. checkpoint_dir/model-200000)")
    parser.add_argument("--resume", action='store_true', help='resume from lastest checkpoint in output_dir.')

    parser.add_argument("--dataset", type=str, help="dataset class name")
    parser.add_argument("--dataset_hparams", type=str, help="a string of comma separated list of dataset hyperparameters")
    parser.add_argument("--dataset_hparams_dict", type=str, help="a json file of dataset hyperparameters")
    parser.add_argument("--model", type=str, help="model class name")
    parser.add_argument("--model_hparams", type=str, help="a string of comma separated list of model hyperparameters")
    parser.add_argument("--model_hparams_dict", type=str, help="a json file of model hyperparameters")
    parser.add_argument("--debug_num_datasets", type=int, default=-1, help="number of dataset to use")

    parser.add_argument("--summary_freq", type=int, default=10, help="save frequency of summaries (except for image and eval summaries) for train/validation set")
    parser.add_argument("--image_summary_freq", type=int, default=50, help="save frequency of image summaries for train/validation set")
    parser.add_argument("--eval_summary_freq", type=int, default=100, help="save frequency of eval summaries for train/validation set")
    parser.add_argument("--accum_eval_summary_freq", type=int, default=400, help="save frequency of accumulated eval summaries for validation set only")
    parser.add_argument("--progress_freq", type=int, default=10, help="display progress every progress_freq steps")
    parser.add_argument("--save_freq", type=int, default=50, help="save frequence of model, 0 to disable")

    parser.add_argument("--aggregate_nccl", type=int, default=0, help="whether to use nccl or cpu for gradient aggregation in multi-gpu training")
    parser.add_argument("--gpu_mem_frac", type=float, default=0.8, help="fraction of gpu memory to use")
    parser.add_argument("--seed", type=int)

    parser.add_argument("--meta_batch_size", type=int, default=8, help="how many inner-loops to run")
    parser.add_argument("--exp_size", type=int, default=5, help="how many videos to compute embedding")
    parser.add_argument("--inner_iters", type=int, default=1, help="number of inner-loop iterations")
    parser.add_argument("--meta_step_size", type=float, default=1.0, help="initial step size of meta optimization")
    parser.add_argument("--final_meta_step_size", type=float, default=1.0, help="final sep size of meta optimization")

    args = parser.parse_args()

    if args.seed is not None:
        tf.set_random_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    if args.output_dir is None:
        list_depth = 0
        model_fname = ''
        for t in ('model=%s,%s' % (args.model, args.model_hparams)):
            if t == '[':
                list_depth += 1
            if t == ']':
                list_depth -= 1
            if list_depth and t == ',':
                t = '..'
            if t in '=,':
                t = '.'
            if t in '[]':
                t = ''
            model_fname += t
        args.output_dir = os.path.join(args.logs_dir, model_fname) + args.output_dir_postfix

    if args.resume:
        if args.checkpoint:
            raise ValueError('resume and checkpoint cannot both be specified')
        args.checkpoint = args.output_dir

    dataset_hparams_dict = {}
    model_hparams_dict = {}
    if args.dataset_hparams_dict:
        with open(args.dataset_hparams_dict) as f:
            dataset_hparams_dict.update(json.loads(f.read()))
    if args.model_hparams_dict:
        with open(args.model_hparams_dict) as f:
            model_hparams_dict.update(json.loads(f.read()))
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
                dataset_hparams_dict.update(json.loads(f.read()))
        except FileNotFoundError:
            print("dataset_hparams.json was not loaded because it does not exist")
        try:
            with open(os.path.join(checkpoint_dir, "model_hparams.json")) as f:
                model_hparams_dict.update(json.loads(f.read()))
        except FileNotFoundError:
            print("model_hparams.json was not loaded because it does not exist")

    print('----------------------------------- Options ------------------------------------')
    for k, v in args._get_kwargs():
        print(k, "=", v)
    print('------------------------------------- End --------------------------------------')

    # Dataset
    train_sets = read_dataset(args.input_dir, mode='train',
                              hparams_dict=dataset_hparams_dict,
                              hparams=args.dataset_hparams)
    val_sets = read_dataset(args.input_dir, mode='val',
                            hparams_dict=dataset_hparams_dict,
                            hparams=args.dataset_hparams)
    train_sets = list(train_sets)
    val_sets = list(val_sets)

    # Backward compatibility, used to set hypermeter for others
    train_dataset = train_sets[0]

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
        aggregate_nccl=args.aggregate_nccl)

    model.exp_size = args.exp_size
    batch_size = model.hparams.batch_size

    # Train set
    train_tf_datasets = [dataset.make_dataset(batch_size) for dataset in train_sets]
    train_iterators = [tf_dataset.make_one_shot_iterator() for tf_dataset in train_tf_datasets]
    train_handles = [iterator.string_handle() for iterator in train_iterators]

    # Val train set
    val_tf_datasets = [dataset.make_dataset(batch_size) for dataset in val_sets]
    val_iterators = [tf_dataset.make_one_shot_iterator() for tf_dataset in val_tf_datasets]
    val_handles = [iterator.string_handle() for iterator in val_iterators]

    # Backward compatibility, use first train set to build graph
    train_handle = train_handles[0]
    iterator = tf.data.Iterator.from_string_handle(train_handle, train_tf_datasets[0].output_types, train_tf_datasets[0].output_shapes)
    inputs = iterator.get_next()
    # inputs comes from the "first training dataset" by default, unless train_handle is remapped to other handles
    model.build_graph(inputs)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(os.path.join(args.output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(args), sort_keys=True, indent=4))
    with open(os.path.join(args.output_dir, "dataset_hparams.json"), "w") as f:
        f.write(json.dumps(train_dataset.hparams.values(), sort_keys=True, indent=4))
    with open(os.path.join(args.output_dir, "model_hparams.json"), "w") as f:
        f.write(json.dumps(model.hparams.values(), sort_keys=True, indent=4))

    with tf.name_scope("parameter_count"):
        # exclude trainable variables that are replicas (used in multi-gpu setting)
        trainable_variables = set(tf.trainable_variables()) & set(model.saveable_variables)
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in trainable_variables])

    saver = tf.train.Saver(var_list=model.saveable_variables, max_to_keep=2)

    # None has the special meaning of evaluating at the end, so explicitly check for non-equality to zero
    if (args.summary_freq != 0 or args.image_summary_freq != 0 or
            args.eval_summary_freq != 0 or args.accum_eval_summary_freq != 0):
        summary_writer = tf.summary.FileWriter(args.output_dir)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_mem_frac)
    config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
    global_step = tf.train.get_or_create_global_step()

    max_steps = model.hparams.max_steps // (args.inner_iters * args.meta_batch_size)

    with tf.Session(config=config) as sess:

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        model.restore(sess, args.checkpoint)
        sess.run(model.post_init_ops)

        # Evaluate handle for each dataset
        # Note: this step is super slow, so we only use a few datasets for debugging.
        if args.debug_num_datasets == -1:
            train_handle_evals = [sess.run(handle) for handle in train_handles]
            val_handle_evals = [sess.run(handle) for handle in val_handles]
        else:
            train_handle_evals = [sess.run(handle) for handle in train_handles[:args.debug_num_datasets]]
            val_handle_evals = [sess.run(handle) for handle in val_handles[:args.debug_num_datasets]]

        print("parameter_count =", sess.run(parameter_count))
        print("number of train sets =", len(train_handle_evals))
        print("number of test sets =", len(val_handle_evals))

        # Set input for the first step
        current_handle_eval = random.choice(train_handle_evals)

        # Set up variables recorder
        model._state = VariableState(sess, tf.trainable_variables())

        sess.graph.finalize()
        start_step = sess.run(global_step)

        def should(step, freq):
            if freq is None:
                return (step + 1) == (max_steps - start_step)
            else:
                return freq and ((step + 1) % freq == 0 or (step + 1) in (0, max_steps - start_step))

        def should_eval(step, freq):
            # never run eval summaries at the beginning since it's expensive, unless it's the last iteration
            return should(step, freq) and (step >= 0 or (step + 1) == (max_steps - start_step))

        # start at one step earlier to log everything without doing any training
        # step is relative to the start_step
        for step in range(-1, max_steps - start_step):
            if step == 1:
                # skip step -1 and 0 for timing purposes (for warmstarting)
                start_time = time.time()

            fetches = {"global_step": global_step}
            if step >= 0:
                # Set up train fetches
                fetches["train_op"] = model.train_op

                # Linearly decreased the meta step size
                frac_done = step / max_steps
                cur_meta_step_size = args.meta_step_size * (1 - frac_done) + args.final_meta_step_size * frac_done

                # Start meta training
                old_vars = model._state.export_variables()
                new_vars = []
                for meta_idx in range(args.meta_batch_size):
                    print("step %d, meta batch %d / %d" %
                          (step, meta_idx+1, args.meta_batch_size))

                    # Sample task (videos from one specific object)
                    current_handle_eval = random.choice(train_handle_evals)
                    for i in range(args.inner_iters):

                        # Run inner update
                        run_start_time = time.time()
                        results = sess.run(fetches, feed_dict={train_handle: current_handle_eval})
                        run_elapsed_time = time.time() - run_start_time
                        if run_elapsed_time > 1.5 and step > 0 and set(fetches.keys()) == {"global_step", "train_op"}:
                            print('running train_op took too long (%0.1fs)' % run_elapsed_time)

                    # Record parameters after doing inner update for each task
                    new_vars.append(model._state.export_variables())
                    model._state.import_variables(old_vars)
                new_vars = average_vars(new_vars)
                # Perform meta update
                model._state.import_variables(interpolate_vars(old_vars, new_vars, cur_meta_step_size))

            fetches = {"global_step": global_step}
            if should(step, args.progress_freq):
                fetches['d_loss'] = model.d_loss
                fetches['g_loss'] = model.g_loss
                fetches['d_losses'] = model.d_losses
                fetches['g_losses'] = model.g_losses
                if isinstance(model.learning_rate, tf.Tensor):
                    fetches["learning_rate"] = model.learning_rate
            if should(step, args.summary_freq):
                fetches["summary"] = model.summary_op
            if should(step, args.image_summary_freq):
                fetches["image_summary"] = model.image_summary_op
            if should_eval(step, args.eval_summary_freq):
                fetches["eval_summary"] = model.eval_summary_op
            results = sess.run(fetches, feed_dict={train_handle: current_handle_eval})
            print(step)

            # Val
            if (should(step, args.summary_freq) or
                    should(step, args.image_summary_freq) or
                    should_eval(step, args.eval_summary_freq)): 
                # Set up val fetches for summary
                current_handle_eval = val_handle_evals[0]
                val_fetches = {"global_step": global_step}
                if should(step, args.summary_freq):
                    val_fetches["summary"] = model.summary_op
                if should(step, args.image_summary_freq):
                    val_fetches["image_summary"] = model.image_summary_op
                if should_eval(step, args.eval_summary_freq):
                    val_fetches["eval_summary"] = model.eval_summary_op

                # Eval
                val_results = sess.run(val_fetches, feed_dict={train_handle: current_handle_eval})
                for name, summary in val_results.items():
                    if name == 'global_step':
                        continue
                    val_results[name] = add_tag_suffix(summary, '_1')

            if should(step, args.summary_freq):
                print("recording summary")
                summary_writer.add_summary(results["summary"], results["global_step"])
                summary_writer.add_summary(val_results["summary"], val_results["global_step"])
                print("done")
            if should(step, args.image_summary_freq):
                print("recording image summary")
                summary_writer.add_summary(results["image_summary"], results["global_step"])
                summary_writer.add_summary(val_results["image_summary"], val_results["global_step"])
                print("done")
            if should_eval(step, args.eval_summary_freq):
                print("recording eval summary")
                summary_writer.add_summary(results["eval_summary"], results["global_step"])
                summary_writer.add_summary(val_results["eval_summary"], val_results["global_step"])
                print("done")
            if should_eval(step, args.accum_eval_summary_freq):
                sess.run(model.accum_eval_metrics_reset_op)
                val_fetches = {"global_step": global_step, "accum_eval_summary": model.accum_eval_summary_op}
                for i, val_handle_eval in enumerate(val_handle_evals):
                    # traverse (roughly up to rounding based on the batch size) all the validation dataset
                    print('evaluating %d / %d test set' % (i, len(val_handle_evals)))
                    val_results = sess.run(val_fetches, feed_dict={train_handle: val_handle_eval})
                accum_eval_summary = add_tag_suffix(val_results["accum_eval_summary"], '_inner_update')
                print("recording accum eval summary")
                summary_writer.add_summary(accum_eval_summary, val_results["global_step"])
                print("done")

            if (should(step, args.summary_freq) or should(step, args.image_summary_freq) or
                    should_eval(step, args.eval_summary_freq) or should_eval(step, args.accum_eval_summary_freq)):
                summary_writer.flush()
            if should(step, args.progress_freq):
                # global_step will have the correct step count if we resume from a checkpoint
                # global step is read before it's incremented
                steps_per_epoch = train_dataset.num_examples_per_epoch() / batch_size
                train_epoch = results["global_step"] / steps_per_epoch
                print("progress  global step %d  epoch %0.1f" % (results["global_step"] + 1, train_epoch))
                if step > 0:
                    elapsed_time = time.time() - start_time
                    average_time = elapsed_time / step
                    images_per_sec = batch_size / average_time
                    remaining_time = (max_steps - (start_step + step + 1)) * average_time
                    print("          image/sec %0.1f  remaining %dm (%0.1fh) (%0.1fd)" %
                          (images_per_sec, remaining_time / 60, remaining_time / 60 / 60, remaining_time / 60 / 60 / 24))

                if results['d_losses']:
                    print("d_loss", results["d_loss"])
                for name, loss in results['d_losses'].items():
                    print("  ", name, loss)
                if results['g_losses']:
                    print("g_loss", results["g_loss"])
                for name, loss in results['g_losses'].items():
                    print("  ", name, loss)
                if isinstance(model.learning_rate, tf.Tensor):
                    print("learning_rate", results["learning_rate"])

            if should(step, args.save_freq):
                print("saving model to", args.output_dir)
                saver.save(sess, os.path.join(args.output_dir, "model"), global_step=global_step)
                print("done")


if __name__ == '__main__':
    main()
