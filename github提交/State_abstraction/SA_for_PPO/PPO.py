from utils import PPO_Actor, V_Critic,PPO_ReplayBuffer
from Abstraction_module4 import abstraction_module
import numpy as np
import copy
import torch
import math


class PPO_agent(object):
	def __init__(self, **kwargs):
		# Init hyperparameters for PPO agent, just like "self.gamma = opt.gamma, self.lambd = opt.lambd, ..."
		self.__dict__.update(kwargs)

		# Choose distribution for the actor
		self.actor = PPO_Actor(self.state_size, self.action_size, self.neu_size).to(self.device)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.a_lr)

		# Build Critic
		self.critic = V_Critic(self.state_size, self.neu_size).to(self.device)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.c_lr)
		self.replay_buffer = PPO_ReplayBuffer(self.state_size1,self.action_size,self.num_envs,self.max_steps,self.device)
		if self.abs:
			self.abs_module = abstraction_module(self.state_size1, self.state_size, self.neu_size, self.action_size,self.c_lr, self.device)
		# Build Trajectory holder

	def act(self, state, deterministic,with_logprob = True):
		with torch.no_grad():
			state = torch.FloatTensor(state).to(self.device);
			if self.abs:
				state = self.abs_module.phi_net(state)
			if deterministic:
				a = self.actor.get_act(state, deterministic, with_logprob = False);
				return a.cpu().numpy()
			else:
				a, logprob_a = self.actor.get_act(state, deterministic, with_logprob);
				return a.cpu().numpy(),logprob_a.detach().cpu().numpy()

	def process_action(self, a):
		return self.action_expand*2*(a - 0.5)
	def train(self):
		# self.entropy_coef*=self.entropy_coef_decay

		'''Prepare PyTorch data from Numpy data'''
		s,a,logprob_a,r,s_next,done,dw = self.replay_buffer.get_data()
		# print(s.shape[0])
		if self.abs:
			# self.abs_module.replay_buffer.add(s,a,r,s_next,done)
			for _ in range(self.train_abs):
				# self.abs_module.train(self.abs_train_size)
				self.abs_module.train_with_data(s,a,r,s_next,batch_size=self.abs_train_size)
			with torch.no_grad():
				s = self.abs_module.phi_net(s)
				s_next = self.abs_module.phi_net(s_next)

		''' Use TD+GAE+LongTrajectory to compute Advantage and TD target'''
		with torch.no_grad():
			vs = self.critic(s)
			vs_ = self.critic(s_next)

			'''dw for TD_target and Adv'''
			deltas = r + self.gamma * vs_ * (~done) - vs
			deltas = deltas.cpu().flatten().numpy()
			adv = [0]

			'''done for GAE'''
			for dlt, mask in zip(deltas[::-1], dw.cpu().flatten().numpy()[::-1]):
				advantage = dlt + self.gamma * self.lambd * adv[-1] * (~mask)
				adv.append(advantage)
			adv.reverse()
			adv = copy.deepcopy(adv[0:-1])
			adv = torch.tensor(adv).unsqueeze(1).float().to(self.device)
			td_target = adv + vs
			adv = (adv - adv.mean()) / ((adv.std()+1e-4))  #sometimes helps


		"""Slice long trajectopy into short trajectory and perform mini-batch PPO update"""
		a_optim_iter_num = int(math.ceil(s.shape[0] / self.a_optim_batch_size))
		c_optim_iter_num = int(math.ceil(s.shape[0] / self.c_optim_batch_size))
		for i in range(self.K_epochs):

			#Shuffle the trajectory, Good for training
			perm = np.arange(s.shape[0])
			np.random.shuffle(perm)
			perm = torch.LongTensor(perm).to(self.device)
			s, a, td_target, adv, logprob_a = \
				s[perm].clone(), a[perm].clone(), td_target[perm].clone(), adv[perm].clone(), logprob_a[perm].clone()

			'''update the actor'''
			for i in range(a_optim_iter_num):
				index = slice(i * self.a_optim_batch_size, min((i + 1) * self.a_optim_batch_size, s.shape[0]))

				distribution = self.actor.get_dist(s[index])
				dist_entropy = distribution.entropy().sum(1, keepdim=True)
				logprob_a_now = distribution.log_prob(a[index])
				ratio = torch.exp(logprob_a_now.sum(1,keepdim=True) - logprob_a[index].sum(1,keepdim=True))  # a/b == exp(log(a)-log(b))

				## - self.entropy_coef * dist_entropy

				surr1 = ratio * adv[index]
				surr2 = torch.clamp(ratio, 1 - self.clip_rate, 1 + self.clip_rate) * adv[index]
				a_loss = -torch.min(surr1, surr2)- self.entropy_coef * dist_entropy

				self.actor_optimizer.zero_grad()
				a_loss.mean().backward()
				torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 40)
				self.actor_optimizer.step()

			'''update the critic'''
			for i in range(c_optim_iter_num):
				index = slice(i * self.c_optim_batch_size, min((i + 1) * self.c_optim_batch_size, s.shape[0]))
				c_loss = (self.critic(s[index]) - td_target[index]).pow(2).mean()
				for name,param in self.critic.named_parameters():
					if 'weight' in name:
						c_loss += param.pow(2).sum() * self.l2_reg

				self.critic_optimizer.zero_grad()
				c_loss.backward()
				self.critic_optimizer.step()



	def save(self, path, num):
		torch.save(self.actor.state_dict(), path + f'ppo_actor{num}.pth')
		torch.save(self.critic.state_dict(), path + f'ppo_critic{num}.pth')
		if self.abs:
			self.abs_module.save(path, num)

	def load(self, path, num):
		self.actor.load_state_dict(torch.load(path + f'ppo_actor{num}.pth'))
		self.critic.load_state_dict(torch.load(path + f'ppo_critic{num}.pth'))
		if self.abs:
			self.abs_module.load(path, num)



