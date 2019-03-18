#from environment import Env,train_primarily_model
#import torch
#train_primarily_model(250, 128, 5)
import sys,os

# env = Env()
# action_space = 10
# device = "cuda"
# for _ in range(10):
#     pi_rand = torch.ones(action_space, dtype=torch.float).to(device) / action_space
#     env.step(pi_rand)
from config import consts, args
from gan_rl_agent import GANAgent

#agent_p = GANAgent(player=True, choose=False, checkpoint=None)
#agent_p.play(4)

try:
    agent_l = GANAgent(player=False, choose=False, checkpoint=None)
except Exception as error:
    print('Caught this error: ' + repr(error))

agent_l.learn(1,1)
#except Exception as e:
 #   exc_type, exc_obj, exc_tb = sys.exc_info()
  #  fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
   # print(exc_type, fname, exc_tb.tb_lineno)
    #print(e)

#agent.play(16)
#agent.play(args.play_episodes_interval)
