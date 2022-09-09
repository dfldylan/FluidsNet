import os
import pandas as pd

root_path = r'D:\dufeilong\datasets\normal'
log_path = os.path.join(root_path, 'log.csv')
pred_fps = [350, 450]


class PredDataLoader(object):
    def __init__(self):
        self.df = pd.read_csv(log_path)
        self.index = None

    def __len__(self):
        return len(pred_fps) * len(self.df)

    def __getitem__(self, item):
        # self.index = item
        folder_num = item // len(pred_fps)
        fps_num = item % len(pred_fps)
        folder = str(self.df.iloc[folder_num, 0])
        fps = str(pred_fps[fps_num])
        path = os.path.join(root_path, folder)
        path = os.path.join(path, r'all_particles_' + fps + '.csv')
        self.current_folder_index = folder_num
        self.current_fps = fps
        return path

    def save_time(self, time):
        self.df.loc[self.current_folder_index, self.current_fps] = time
        self.df.to_csv(r'log.csv', index=False)
