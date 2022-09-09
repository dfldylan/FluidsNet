import numpy as np
import pandas as pd
import os
import csv


def Find_csv(dir, fps_range=None):
    items = os.listdir(dir)
    csv_files = []
    if fps_range is None:
        for item in items:
            if item.split('/')[-1].split('.')[-1] == 'csv':
                csv_files.append(item)
    elif fps_range[0] == fps_range[1]:
        return [os.path.join(dir, 'all_particles_' + str(fps_range[0]) + '.csv')]
    else:
        for item in items:
            if item.split('/')[-1].split('.')[-1] == 'csv':
                fps = int(item.split('/')[-1].split('_')[-1].split('.')[0])
                if fps_range[1] > fps >= fps_range[0]:
                    csv_files.append(item)

    return list(map(lambda x: os.path.join(dir, x), csv_files))


def Find_csv_from_parent_folder(train_data_dirs, csv_range):
    train_list = []
    for train_data_dir in train_data_dirs:
        train_files = Find_csv(train_data_dir, csv_range)
        train_list += train_files
    return train_list


def get_fps(filename):
    return int(filename.split('/')[-1].split('_')[-1].split('.')[0])


def Load_csv(filename):
    df = pd.read_csv(filename, dtype=float)
    df['isFluidSolid'] = df['isFluidSolid'].astype(int)
    particles = df.values
    data = np.concatenate((particles[:, :6], particles[:, 7:8], particles[:, 12:15]), axis=1)
    timestep = particles[0, 6]
    return data, timestep  # input[:, :7], output[:, 7:]


def Draw_voxels(point_cloud, voxel_size, grid_size, lidar_coord):
    shifted_coord = point_cloud[:, :3] + lidar_coord
    voxel_index = np.floor(shifted_coord[:, :] / voxel_size).astype(np.int)  # int lower than num

    bound_z = np.logical_and(voxel_index[:, 2] >= 0, voxel_index[:, 2] < grid_size[2])
    bound_y = np.logical_and(voxel_index[:, 1] >= 0, voxel_index[:, 1] < grid_size[1])
    bound_x = np.logical_and(voxel_index[:, 0] >= 0, voxel_index[:, 0] < grid_size[0])

    bound_box = np.logical_and(np.logical_and(bound_x, bound_y), bound_z)

    point_cloud = point_cloud[bound_box]
    voxel_index = voxel_index[bound_box]

    voxel_index = voxel_index[:, 0] * grid_size[1] * grid_size[2] + voxel_index[:, 1] * grid_size[2] + voxel_index[:, 2]

    return point_cloud, voxel_index


def build_voxel(voxel_index, RANDOM_NUMBER, VOXEL_NUMBER, max_particles_voxel):
    _voxel_index = np.random.randint(-RANDOM_NUMBER, 0, size=[VOXEL_NUMBER, max_particles_voxel])
    _voxel_number = np.zeros(shape=[VOXEL_NUMBER], dtype=np.int32)
    for i in range(voxel_index.shape[0]):
        _voxel = voxel_index[i]
        index = _voxel_number[_voxel]
        if index == max_particles_voxel:
            continue
        _voxel_index[_voxel, index] = i
        _voxel_number[_voxel] += 1
    return _voxel_index


def build_mini_voxel(MINI_VOXEL_NUMBER, mini_max_particles_voxel, mini_voxel_size, data, voxel_index, bias_voxel):
    mini_voxel = [[] for i in range(MINI_VOXEL_NUMBER)]
    num = data.shape[0]
    particle_voxel = np.tile(np.expand_dims(voxel_index, axis=1), (1, 27))
    neighbor_voxel = particle_voxel + bias_voxel  # [N, 27]
    for i in range(num):
        mini_voxel[voxel_index[i]].append(data[i])

    neighbor_particles = []
    for i in range(num):
        neighbor_particle = [x[:7] for j in neighbor_voxel[i] for x in mini_voxel[j]]  # [27, 8, 7]
        neighbor_particle = np.reshape(neighbor_particle, [-1, 7])
        neighbor_particle[:, :3] = abs(neighbor_particle[:, :3] - data[i, :3])

        _num = neighbor_particle.shape[0]
        fill_num = 27 * mini_max_particles_voxel - _num
        if fill_num > 0:
            fill_particles = np.concatenate((
                (np.random.randint(0, 2, size=(fill_num, 3)) * 2 - 1) * mini_voxel_size[0] * 3,
                np.zeros([fill_num, 4])), axis=1)
            neighbor_particle = np.concatenate((neighbor_particle, fill_particles), axis=0)
        elif fill_num < 0:
            print('voxel contains too many particles')
            np.random.shuffle(neighbor_particle)
            neighbor_particle = neighbor_particle[:27 * mini_max_particles_voxel]
        neighbor_particles.append(neighbor_particle)
    return np.array(neighbor_particles)
