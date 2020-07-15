from tools import *
import prepare as cfg



# 携带近邻网格的局部特征，逐帧计算
def build_input(files, start, end):
    final_data = []
    final_label = []
    final_neighbor = []
    final_voxel_index = []
    fps = []
    timestep = []

    for i in range(start, end):
        file = files[i]
        _fps = get_fps(file)
        print("choose the fps: " + str(_fps))
        fps.append(_fps)
        data, _timestep = Load_csv(file)  # input[:, :7], output[:, 7:]
        _, large_voxel_index = Draw_voxels(data, cfg.voxel_size, cfg.grid_size, cfg.lidar_coord)
        data, voxel_index = Draw_voxels(_, cfg.mini_voxel_size, cfg.mini_grid_size, cfg.mini_lidar_coord)
        assert data.shape[0] == _.shape[0]

        timestep.append(_timestep)
        final_data.append(data[:, :7])
        final_label.append(data[:, 7:])
        final_voxel_index.append(
            build_voxel(large_voxel_index, cfg.RANDOM_NUMBER, cfg.VOXEL_NUMBER, cfg.max_particles_voxel))
        final_neighbor.append(
            build_mini_voxel(cfg.MINI_VOXEL_NUMBER, cfg.mini_max_particles_voxel, cfg.mini_voxel_size, data,
                             voxel_index, cfg.bias_voxel))
    return [final_data, final_neighbor, final_label, final_voxel_index, fps, timestep]

def write_data(fps_list, feature, knn_index, y_truth, voxel_index, y_pred, trans_data_dir):
    for i in range(len(fps_list)):
        fps = fps_list[i]
        pred_out = trans_data_dir + r"/all_particles_" + str(fps) + ".csv"
        with open(pred_out, 'w', newline='') as f:
            writer = csv.writer(f)
            row = ['px', 'py', 'pz', 'vx', 'vy', 'vz', 'isFluid', 'ox', 'oy', 'oz', 'pred_x', 'pred_y', 'pred_z', '']
            writer.writerow(row)

            for j in range(len(feature[i])):
                row = []
                for k in range(0, 7):
                    row.append(str(feature[i][j][k]))
                for k in range(0, 3):
                    row.append(str(y_truth[i][j][k]))
                for k in range(0, 3):
                    row.append(str(y_pred[i][j][k]))
                row.append('')
                writer.writerow(row)
