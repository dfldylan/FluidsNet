from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)

import prepare as cfg
from model import *
from tools import *
from tf_init import *
from build import *

# 准备数据集文件
train_list = cfg.predict_csv_list

with tf.Graph().as_default():
    with tf.Session(config=GPUInitial(cfg.GPU_MEMORY_FRACTION, cfg.GPU_AVAILABLE)) as sess:
        # 定义模型
        model = Model(
            cfg.learning_rate,
            cfg.GPU_AVAILABLE.split(','),
            cfg.mini_max_particles_voxel,
            cfg.max_particles_voxel,
            cfg.grid_size,
            cfg.VOXEL_NUMBER,
            cfg.RANDOM_NUMBER
        )

        # 初始化模型/恢复模型
        paramInitial(model, sess, cfg.save_model_dir)

        for index in range(0, len(train_list), cfg.GPU_USE_COUNT):
            if index + cfg.GPU_USE_COUNT > len(train_list):
                break
            else:
                start = index
                end = index + cfg.GPU_USE_COUNT

            # [final_data, final_neighbor, final_label, final_voxel_index, fps, timestep]
            data = build_input(train_list, start, end)

            # train_input[idx, N, 7] & [idx, N, K, 7], train_output[idx, N, 3], voxel_index[idx, N]
            ret = model.train(sess, data, train=False, summary=False)

            # print_screen(data[4], ret[1], ret[0], data[2])
            write_data(data[4], data[0], data[1], data[2], data[3], ret[0], cfg.trans_data_dir)
