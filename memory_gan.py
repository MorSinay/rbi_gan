import torch.utils.data
import numpy as np
import torch
import os
from config import consts, args
#from preprocess import lock_file, release_file

class Memory(torch.utils.data.Dataset):

    def __init__(self):
        super(Memory, self).__init__()
        self.n_steps = args.n_steps
        # TODO Mor: update
        self.screen_dir = consts.screen_dir

    def __len__(self):
        return args.n_tot

    def __getitem__(self, sample):

        sample, next_sample = sample
        print("x")
        return {'s': torch.from_numpy(sample['st']), 'r': torch.from_numpy(np.array(sample['r'])),
                'a': torch.from_numpy(np.array(sample['a'])), 't': torch.from_numpy(np.array(sample['t'])),
                'pi': torch.from_numpy(sample['pi']),
                's_tag': torch.from_numpy(next_sample['st']), 'pi_tag': torch.from_numpy(next_sample['pi'])}

class ReplayBatchSampler(object):

    def __init__(self):

        self.batch = args.batch

        self.screen_dir = consts.screen_dir
        self.trajectory_dir = consts.trajectory_dir
        self.list_old_path = consts.list_old_path

        #TODO Mor: ?
        self.replay_updates_interval = args.replay_updates_interval
        self.replay_memory_size = args.replay_memory_size
        self.readlock = consts.readlock

        #self.rec_type = consts.rec_type
        self.n_steps = args.n_steps

    def __iter__(self):

        traj_old = 0
        #replay_buffer = np.array([], dtype=self.rec_type)
        replay_buffer = np.array([],dtype=consts.rec_type)
        flag = True
        while True:

            # load new memory

            fread = open(self.readlock,"r+b")
            traj_sorted = np.load(fread)
            fread.seek(0)
            np.save(fread, [])
            fread.close()
            if flag:
                traj_sorted = list(range(4))

            if not len(traj_sorted):
                if flag:
                    break
                continue

            replay = np.concatenate([np.load(os.path.join(self.trajectory_dir, "%d.npy" % traj)) for traj in traj_sorted], axis=0)

            replay_buffer = np.concatenate([replay_buffer, replay], axis=0)[-self.replay_memory_size:]

            # save previous traj_old to file
            np.save(self.list_old_path, np.array([traj_old]))
            traj_old = replay_buffer[0]['traj']
            print("Old trajectory: %d" % traj_old)
            print("New Sample size is: %d" % len(replay))

            len_replay_buffer = len(replay_buffer)

            minibatches = min(self.replay_updates_interval, int(len_replay_buffer / self.batch) - self.n_steps)

            #TODO Mor: the indexes can return
            shuffle_indexes = np.random.choice(len_replay_buffer - self.n_steps, (minibatches, self.batch),
                                               replace=True)

            print("Explorer:Replay Buffer size is: %d" % len_replay_buffer)

            for i in range(minibatches):
                samples = shuffle_indexes[i]
                print("y")
                yield list(zip(replay_buffer[samples], replay_buffer[samples + self.n_steps]))

    def __len__(self):
        return np.inf
