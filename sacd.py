import numpy as np
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import argparse
import os#, shutil
# from datetime import datetime
from home import Homes
import pickle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class RandomBuffer(object):
	def __init__(self, state_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0 #store index
		self.size = 0 #current size of buffer

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, 1))
		self.reward = np.zeros((max_size, 1))
		self.next_state = np.zeros((max_size, state_dim))
		self.done = np.zeros((max_size, 1))  #mask of dead&win

		self.device = device

	def add(self, state, action, reward, next_state, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.reward[self.ptr] = reward
		self.next_state[self.ptr] = next_state
		self.done[self.ptr] = done #0,0,0，...，1

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)

	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)
		with torch.no_grad():
			return (
				torch.FloatTensor(self.state[ind]).to(self.device),
				torch.Tensor(self.action[ind]).long().to(self.device),
				torch.FloatTensor(self.reward[ind]).to(self.device),
				torch.FloatTensor(self.next_state[ind]).to(self.device),
				torch.FloatTensor(self.done[ind]).to(self.device)
			)

def evaluate_policy(env, model):
	scores = 0
	for j in range(3):
		s, done= env.reset(mode = 'validation', set_num = j), False
		while not done:
			# Take deterministic actions at test time
			a = model.select_action(s, deterministic=True)
			s_next, r, done, _ = env.step(a)
			scores += r
			s = s_next
	return scores/3, env.acts


#You can just ignore this funciton. Is not related to the RL.
def str2bool(v):
	'''transfer str to bool for argparse'''
	if isinstance(v, bool):
		return v
	if v.lower() in ('yes', 'True','true','TRUE', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'False','false','FALSE', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')
	
def build_net(layer_shape, hid_activation, output_activation):
	'''build net with for loop'''
	layers = []
	for j in range(len(layer_shape)-1):
		act = hid_activation if j < len(layer_shape)-2 else output_activation
		layers += [nn.Linear(layer_shape[j], layer_shape[j+1]), act()]
	return nn.Sequential(*layers)


class Q_Net(nn.Module):
	def __init__(self, state_dim, action_dim, hid_shape):
		super(Q_Net, self).__init__()
		layers = [state_dim] + list(hid_shape) + [action_dim]

		self.Q1 = build_net(layers, nn.ReLU, nn.Identity)
		self.Q2 = build_net(layers, nn.ReLU, nn.Identity)

	def forward(self, s):
		q1 = self.Q1(s)
		q2 = self.Q2(s)
		return q1,q2


class Policy_Net(nn.Module):
	def __init__(self, state_dim, action_dim, hid_shape):
		super(Policy_Net, self).__init__()
		layers = [state_dim] + list(hid_shape) + [action_dim]
		self.P = build_net(layers, nn.ReLU, nn.Identity)

	def forward(self, s):
		logits = self.P(s)
		probs = F.softmax(logits, dim=1)
		return probs


class SACD_Agent(object):
	def __init__(self, opt):
		self.action_dim = opt.action_dim
		self.batch_size = opt.batch_size
		self.gamma = opt.gamma
		self.tau = 0.005

		self.actor = Policy_Net(opt.state_dim, opt.action_dim, opt.hid_shape).to(device)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=opt.lr)

		self.q_critic = Q_Net(opt.state_dim, opt.action_dim, opt.hid_shape).to(device)
		self.q_critic_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr=opt.lr)

		self.q_critic_target = copy.deepcopy(self.q_critic)
		for p in self.q_critic_target.parameters(): p.requires_grad = False

		self.alpha = opt.alpha
		self.adaptive_alpha = opt.adaptive_alpha
		if opt.adaptive_alpha:
			# We use 0.6 because the recommended 0.98 will cause alpha explosion.
			self.target_entropy = 0.6 * (-np.log(1 / opt.action_dim))  # H(discrete)>0
			self.log_alpha = torch.tensor(np.log(opt.alpha), dtype=float, requires_grad=True, device=device)
			self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=opt.lr)

		self.H_mean = 0

	def select_action(self, state, deterministic):
		with torch.no_grad():
			state = torch.FloatTensor([state]).to(device) #from (s_dim,) to (1, s_dim)
			probs = self.actor(state)
			if deterministic:
				a = probs.argmax(-1).item()
			else:
				a = Categorical(probs).sample().item()
			return a

	def train(self,replay_buffer):
		s, a, r, s_next, done = replay_buffer.sample(self.batch_size)

		#------------------------------------------ Train Critic ----------------------------------------#
		'''Compute the target soft Q value'''
		with torch.no_grad():
			next_probs = self.actor(s_next) #[b,a_dim]
			next_log_probs = torch.log(next_probs+1e-8) #[b,a_dim]
			next_q1_all, next_q2_all = self.q_critic_target(s_next)  # [b,a_dim]
			min_next_q_all = torch.min(next_q1_all, next_q2_all)
			v_next = torch.sum(next_probs * (min_next_q_all - self.alpha * next_log_probs), dim=1, keepdim=True) # [b,1]
			target_Q = r + (1 - done) * self.gamma * v_next

		'''Update soft Q net'''
		q1_all, q2_all = self.q_critic(s) #[b,a_dim]
		q1, q2 = q1_all.gather(1, a), q2_all.gather(1, a) #[b,1]
		q_loss = F.mse_loss(q1, target_Q) + F.mse_loss(q2, target_Q)
		self.q_critic_optimizer.zero_grad()
		q_loss.backward()
		self.q_critic_optimizer.step()

		#------------------------------------------ Train Actor ----------------------------------------#
		for params in self.q_critic.parameters():
			#Freeze Q net, so you don't waste time on computing its gradient while updating Actor.
			params.requires_grad = 	False

		probs = self.actor(s) #[b,a_dim]
		log_probs = torch.log(probs + 1e-8) #[b,a_dim]
		with torch.no_grad():
			q1_all, q2_all = self.q_critic(s)  #[b,a_dim]
		min_q_all = torch.min(q1_all, q2_all)

		a_loss = torch.sum(probs * (self.alpha*log_probs - min_q_all), dim=1, keepdim=True) #[b,1]

		self.actor_optimizer.zero_grad()
		a_loss.mean().backward()
		self.actor_optimizer.step()

		for params in self.q_critic.parameters():
			params.requires_grad = 	True

		#------------------------------------------ Train Alpha ----------------------------------------#
		if self.adaptive_alpha:
			with torch.no_grad():
				self.H_mean = -torch.sum(probs * log_probs, dim=1).mean()
			alpha_loss = self.log_alpha * (self.H_mean - self.target_entropy)

			self.alpha_optim.zero_grad()
			alpha_loss.backward()
			self.alpha_optim.step()

			self.alpha = self.log_alpha.exp().item()

		#------------------------------------------ Update Target Net ----------------------------------#
		for param, target_param in zip(self.q_critic.parameters(), self.q_critic_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

	def save(self,b_envname):
		torch.save(self.actor.state_dict(), "./model/{}_actor.pth".format(b_envname))
		torch.save(self.q_critic.state_dict(), "./model/{}_critic.pth".format(b_envname))


	def load(self, b_envname):
		self.actor.load_state_dict(torch.load("./model/{}_actor.pth".format(b_envname)))
		self.q_critic.load_state_dict(torch.load("./model/{}_critic.pth".format(b_envname)))
		
'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--EnvIdex', type=int, default=0, help='Homes_SACD')
parser.add_argument('--Option', type=str, default='WCO', help='WCO or FXD')
parser.add_argument('--Tar', type=str2bool, default=False, help='True or False')
parser.add_argument('--Rate', type=float, default=99, help='Consumer rate (rate + 1 = Total)')
parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')
parser.add_argument('--ModelIdex', type=int, default=50, help='which model to load')

parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--Max_train_steps', type=int, default=4e6, help='Max training steps')
parser.add_argument('--save_interval', type=int, default=1e6, help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=2e3, help='Model evaluating interval, in steps.')
parser.add_argument('--random_steps', type=int, default=1e4, help='steps for random policy to explore')
parser.add_argument('--update_every', type=int, default=50, help='training frequency')

parser.add_argument('--gamma', type=float, default=0.97, help='Discounted Factor')
parser.add_argument('--hid_shape', type=list, default=[200, 1024, 512, 200], help='Hidden net shape')
parser.add_argument('--lr', type=float, default=3e-3, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--alpha', type=float, default=0.2, help='init alpha')
parser.add_argument('--adaptive_alpha', type=str2bool, default=True, help='Use adaptive alpha turning')
opt = parser.parse_args()
print(opt)

def main():
	#create Env
	EnvName = ['Homes_SACD']
	BriefEnvName = ['Homes_SACD']
	env = Homes(tar= opt.Tar, option= opt.Option, rate = opt.Rate)
	eval_env = Homes(tar= opt.Tar, option= opt.Option, rate = opt.Rate)
	opt.state_dim = env.observation_space.shape[0]
	opt.action_dim = env.action_space.n
	opt.max_e_steps = 31

	#Seed everything
	torch.manual_seed(opt.seed)
	# env.seed(opt.seed)
	# env.action_space.seed(opt.seed)
	# eval_env.seed(opt.seed)
	# eval_env.action_space.seed(opt.seed)
	np.random.seed(opt.seed)

	print('Algorithm: SACD','  Env:',BriefEnvName[opt.EnvIdex],'  state_dim:',opt.state_dim,
		  '  action_dim:',opt.action_dim,'  Random Seed:',opt.seed, '  max_e_steps:',opt.max_e_steps, '\n')

	#Build model and replay buffer
	if not os.path.exists('model'): os.mkdir('model')
	model = SACD_Agent(opt)
	if opt.Loadmodel: model.load(BriefEnvName[opt.EnvIdex])
	buffer = RandomBuffer(opt.state_dim, max_size=int(1e6))

	if opt.render:
		score, Homes_hist = evaluate_policy(eval_env, model)
		with open(f'res/acts_{EnvName[opt.EnvIdex]}.pkl', 'wb') as file:
			pickle.dump(acts, file)
		print('EnvName:', BriefEnvName[opt.EnvIdex], 'seed:', opt.seed, 'score:', score)
	else:
		total_steps, acts, rew, scores_hist = 0, [], [], []
		while total_steps < opt.Max_train_steps:
			s, done, ep_r, steps, best_score, rr = env.reset(), False, 0, 0, -np.inf, 0
			while not done:
				steps += 1  # steps in current episode

				# interact with Env
				if buffer.size < opt.random_steps:
					a = env.action_space.sample()
				else:
					a = model.select_action(s, deterministic=False)
				s_next, r, done, _ = env.step(a)
				rr += r

				buffer.add(s, a, r, s_next, done)
				s = s_next
				ep_r += r

				'''update if its time'''
				# train 50 times every 50 steps rather than 1 training per step. Better!
				if total_steps >= opt.random_steps and total_steps % opt.update_every == 0:
					for j in range(opt.update_every):
						model.train(buffer)

				'''record & log'''
				if (total_steps) % opt.eval_interval == 0:
					score, act = evaluate_policy(eval_env, model)
					rew.append(score)
					if (total_steps) % (10*opt.eval_interval) == 0:
						acts.append(act)
					if score > best_score:
						model.save(BriefEnvName[opt.EnvIdex])
						best_score = score
					print('EnvName:', BriefEnvName[opt.EnvIdex], 'seed:', opt.seed,
						  'steps: {}k'.format(int(total_steps / 1000)), 'score:', score)
				total_steps += 1
			scores_hist.append(rr)
		with open(f'res/acts_{EnvName[opt.EnvIdex]}.pkl', 'wb') as file:
			pickle.dump(acts, file)
		with open(f'res/rew_{EnvName[opt.EnvIdex]}.pkl', 'wb') as file:
			pickle.dump(rew, file)
		with open(f'res/scores_hist_{EnvName[opt.EnvIdex]}.pkl', 'wb') as file:
			pickle.dump(scores_hist, file)

	env.close()
	eval_env.close()


if __name__ == '__main__':
	main()
