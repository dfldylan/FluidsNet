import tensorflow as tf


def GPUInitial(GPU_MEMORY_FRACTION, GPU_AVAILABLE):
    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=GPU_MEMORY_FRACTION,
                                  visible_device_list=GPU_AVAILABLE,
                                  allow_growth=True),
        device_count={
            "GPU": len(GPU_AVAILABLE.split(',')),
        },
        allow_soft_placement=True,
    )
    return config


def paramInitial(model, sess, save_model_dir):
    # param init/restore
    if tf.train.get_checkpoint_state(save_model_dir):
        print("Reading model parameters from %s" % save_model_dir)
        model.saver.restore(
            sess, tf.train.latest_checkpoint(save_model_dir))

    else:
        print("Created model with fresh parameters.")
        tf.global_variables_initializer().run()
