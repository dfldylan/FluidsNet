from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)

import prepare as cfg
from model import *
from tools import *
from tf_init import *
from build import *
from datasets_loop import *
import time

# 准备数据集文件
train_loader = PredDataLoader()

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

        for index in range(0, len(train_loader), cfg.GPU_USE_COUNT):
            if index + cfg.GPU_USE_COUNT > len(train_loader):
                break
            else:
                start = index
                end = index + cfg.GPU_USE_COUNT
            # [final_data, final_neighbor, final_label, final_voxel_index, fps, timestep]
            data = build_input(train_loader, start, end)
            train_loader.write_csv(data[0][0][:, :3], data[0][0][:, 3:6], data[0][0][:, 6:7])
            for step in range(train_loader.pred_loop_num):
                [pred] = model.train(sess, data, onlypred=True)

                vel = pred[0]
                pos = vel * data[5][0] + data[0][0][:, :3]
                isfluidsolid = data[0][0][:, 6:7]
                train_loader.fps += 1
                path = train_loader.write_csv(pos, vel, isfluidsolid)
                data = build_input([path], 0, 1)






            # train_input[idx, N, 7] & [idx, N, K, 7], train_output[idx, N, 3], voxel_index[idx, N]
            # time0 = time.time()
            # time1 = time.time()
            # print_screen(data[4], ret[1], ret[0], data[2])
            # write_data(data[4], data[0], data[1], data[2], data[3], ret[0], cfg.trans_data_dir)
            # train_loader.save_time(time1 - time0)
