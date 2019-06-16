import torch.utils.data
import numpy as np
import torch
import os
from config import consts, args, DirsAndLocksSingleton, lock_file, release_file
#from preprocess import lock_file, release_file
import time

class Memory(torch.utils.data.Dataset):

    def __init__(self):
        super(Memory, self).__init__()
        self.n_steps = args.n_steps
        self.action_space = consts.action_space

    def __len__(self):
        return args.n_tot

    def __getitem__(self, sample):

        sample, next_sample = sample

        return {'r': torch.from_numpy(np.array(sample['r'])), 't': torch.from_numpy(np.array(sample['t'])),
                'pi': torch.from_numpy(sample['pi']), 'pi_explore': torch.from_numpy(sample['pi_explore']),
                'acc': torch.from_numpy(np.array(sample['acc'])), 'pi_tag': torch.from_numpy(next_sample['pi'])}


class ReplayBatchSampler(object):

    def __init__(self, exp_name):

        self.batch = args.batch

        self.dirs_locks = DirsAndLocksSingleton(exp_name)

        self.trajectory_dir = self.dirs_locks.trajectory_dir
        self.list_old_path = self.dirs_locks.list_old_path

        #TODO Mor: ?
        self.replay_updates_interval = args.replay_updates_interval
        self.replay_memory_size = args.replay_memory_size
        self.readlock = self.dirs_locks.readlock

        self.rec_type = consts.rec_type
        self.n_steps = args.n_steps

    def __iter__(self):

        traj_old = 0
        replay_buffer = np.array([], dtype=self.rec_type)

        # flag = True
        while True:

            # load new memory

            fread = lock_file(self.readlock)
            traj_sorted = np.load(fread)
            fread.seek(0)
            np.save(fread, [])
            release_file(fread)
            # if flag:
            #traj_sorted = list(range(10))*50
            #
            if not len(traj_sorted):
                #print("traj_sorted empty")
                #time.sleep(5)
            #     if flag:
            #         break
                #traj_sorted = list(range(1000))
                continue
            try:
                replay = np.concatenate([np.load(os.path.join(self.trajectory_dir, "%d.npy" % traj)) for traj in traj_sorted], axis=0)
            except Exception as e:
                import sys
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
                print(e)
                print("traj_sorted", traj_sorted)
                print("traj_old", traj_old)

            replay_buffer = np.concatenate([replay_buffer, replay], axis=0)[-self.replay_memory_size:]

            # save previous traj_old to file
            np.save(self.list_old_path, np.array([traj_old]))
            traj_old = replay_buffer[0]['traj']
            print("Old trajectory: %d" % traj_old)
            print("New Sample size is: %d" % len(replay))

            len_replay_buffer = len(replay_buffer)

            minibatches = min(self.replay_updates_interval, int(len_replay_buffer / self.batch) - self.n_steps)

            shuffle_indexes = np.random.choice(len_replay_buffer - self.n_steps, (minibatches, self.batch),
                                               replace=True)

            print("Explorer:Replay Buffer size is: %d" % len_replay_buffer)

            for i in range(minibatches):
                samples = shuffle_indexes[i]
                try:
                    yield list(zip(replay_buffer[samples], replay_buffer[samples + self.n_steps]))
                except Exception as e:
                    import sys
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    print(exc_type, fname, exc_tb.tb_lineno)
                    print(e)

            time.sleep(5)

    def __len__(self):
        return np.inf
