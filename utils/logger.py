import matplotlib
import os

matplotlib.use('agg')
# import seaborn as sns
import matplotlib.pyplot as plt

class nnUNetLogger(object):
    """
    This class is really trivial. Don't expect cool functionality here. This is my makeshift solution to problems
    arising from out-of-sync epoch numbers and numbers of logged loss values. It also simplifies the trainer class a
    little

    YOU MUST LOG EXACTLY ONE VALUE PER EPOCH FOR EACH OF THE LOGGING ITEMS! DONT FUCK IT UP
    """
    def __init__(self, loss_info: dict, verbose: bool = False):
        self.my_fantastic_logging = {
            'train_losses': list(),
            'val_losses': list(),
            'lrs': list(),
            'epoch_start_timestamps': list(),
            'epoch_end_timestamps': list(),
            'ap_01_test': list(),
            'ap_01_train': list(), 
            'recall_test': list(), 
            'precision_test': list(), 
            'recall_train': list(), 
            'precision_train': list(), 
        }
        self._loss_dict = loss_info
        self.verbose = verbose
        self.train_loss = self._loss_dict['train_loss']
        self.train_loss_epoch = self._loss_dict['train_loss_epoch']
        self.valid_loss = self._loss_dict['valid_loss']
        self.valid_loss_epoch = self._loss_dict['valid_loss_epoch']
        self.ap_01_test = self._loss_dict['ap_01_test']
        self.ap_01_test_epoch = self._loss_dict['ap_01_test_epoch']
        self.lr = self._loss_dict['lr']
        self.lr_epoch = self._loss_dict['lr_epoch']
        self.ap_01_train = self._loss_dict['ap_01_train']
        self.ap_01_train_epoch = self._loss_dict['ap_01_train_epoch']
        self.recall_test = self._loss_dict['recall_test']
        self.recall_test_epoch = self._loss_dict['recall_test_epoch']
        self.precision_test = self._loss_dict['precision_test']
        self.precision_test_epoch = self._loss_dict['precision_test_epoch']
        self.recall_train = self._loss_dict['recall_train']
        self.recall_train_epoch = self._loss_dict['recall_train_epoch']
        self.precision_train = self._loss_dict['precision_train']
        self.precision_train_epoch = self._loss_dict['precision_train_epoch']
        self.epoch_time = self._loss_dict['epoch_time']
        self.epoch_time_epoch = self._loss_dict['epoch_time_epoch']

    def log(self, key, value, epoch: int):
        """
        sometimes shit gets messed up. We try to catch that here
        """
        if key == 'train_losses':
            self.train_loss.append(value)
            self.train_loss_epoch.append(epoch)
        if key == 'val_losses':
            self.valid_loss.append(value)
            self.valid_loss_epoch.append(epoch)
        if key == 'lrs':
            self.lr.append(value)
            self.lr_epoch.append(epoch)
        if key == 'ap_01_train':
            self.ap_01_train.append(value)
            self.ap_01_train_epoch.append(epoch)
        if key == 'ap_01_test':
            self.ap_01_test.append(value)
            self.ap_01_test_epoch.append(epoch)
        if key == 'recall_test':
            self.recall_test.append(value)
            self.recall_test_epoch.append(epoch)
        if key == 'precision_test':
            self.precision_test.append(value)
            self.precision_test_epoch.append(epoch)
        if key == 'recall_train':
            self.recall_train.append(value)
            self.recall_train_epoch.append(epoch)
        if key == 'precision_train':
            self.precision_train.append(value)
            self.precision_train_epoch.append(epoch)
        if key == 'epoch_end_timestamps':
            self.epoch_time.append(value)
            self.epoch_time_epoch.append(epoch)
        # if key == 'Batch_losses':
        #     self.batch_loss.append(value)
        #     self.batch_loss_epoch.append(epoch)

    def plot_progress_png(self, output_folder, timesteamp):
        # we infer the epoch form our internal logging

        fig, ax_all = plt.subplots(2, 2, figsize=(54, 30))  # 修改布局为 2x2
        for ax in ax_all.flatten():
            # ax.set_title("Title", fontsize=30)
            ax.set_xlabel("X-axis label", fontsize=30)
            ax.set_ylabel("Y-axis label", fontsize=30)
            # 设置刻度标签的大小
            ax.tick_params(axis='both', which='major', labelsize=30)

        # Loss and accuracy curves
        ax = ax_all[0, 0]
        # self.valid_loss_epoch = [i+1 for i in self.valid_loss_epoch]
        # Plot loss
        if len(self.train_loss_epoch) > 4:
            ax.plot(self.train_loss_epoch[3:], self.train_loss[3:], color='b', ls='-', label="loss_tr", linewidth=6)
        else:
            ax.plot(self.train_loss_epoch, self.train_loss, color='b', ls='-', label="loss_tr", linewidth=6)

        if len(self.train_loss_epoch) > 4:
            ax.plot(self.valid_loss_epoch[3:], self.valid_loss[3:], color='r', ls='-', label="loss_val", linewidth=6)
        else:
            ax.plot(self.valid_loss_epoch, self.valid_loss, color='r', ls='-', label="loss_val", linewidth=6)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")

        # Epoch times
        ax = ax_all[0, 1]
        ax.plot(self.epoch_time_epoch, self.epoch_time, color='b', ls='-', label="epoch duration", linewidth=6)
        ylim = [0] + [ax.get_ylim()[1]]
        ax.set(ylim=ylim)
        ax.set_xlabel("epoch")
        ax.set_ylabel("time [s]")

        # Learning rate
        ax = ax_all[1, 0]
        ax.plot(self.lr_epoch, self.lr, color='b', ls='-', label="learning rate", linewidth=6)
        ax.set_xlabel("epoch")
        ax.set_ylabel("learning rate")

        # Recall, Precision, and AP 0.1
        ax = ax_all[1, 1]
        ax.plot(self.recall_train_epoch, self.recall_train, color='g', ls='dotted', label="recall_train", linewidth=6)
        ax.plot(self.recall_test_epoch, self.recall_test, color='g', ls='-', label="recall_test", linewidth=6)
        # ax.plot(self.precision_test_epoch, self.precision_test, color='r', ls='-', label="precision_test", linewidth=6)
        # ax.plot(self.precision_train_epoch, self.precision_train, color='r', ls='dotted', label="precision_train", linewidth=6)
        # ax.plot(self.ap_01_test_epoch, self.ap_01_test, color='b', ls='-', label="ap_01_test", linewidth=6)
        # ax.plot(self.ap_01_train_epoch, self.ap_01_train, color='b', ls='dotted', label="ap_01_train", linewidth=6)
        ax.set_xlabel("epoch")
        ax.set_ylabel("metrics")
        plt.tight_layout()
        for i in range(2):
            for j in range(2):
                ax_all[i, j].legend(prop={'size': 30})
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        path_save = os.path.join(output_folder, f'{timesteamp}-progress.png')
        fig.savefig(path_save)
        # print(f'progress.png save done in the {output_folder}')
        plt.close()


    def return_loss_info_dict(self):
        return self._loss_dict

    def load_checkpoint(self, checkpoint: dict):
        self.my_fantastic_logging = checkpoint


