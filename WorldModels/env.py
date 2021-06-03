import numpy as np
import gym
import json
import os
import tensorflow as tf
import gc

from PIL import Image
from gym.spaces.box import Box
from gym.envs.box2d.car_racing import CarRacing

class CarRacingWrapper(CarRacing):
  def __init__(self, full_episode=False):
    super(CarRacingWrapper, self).__init__()
    self.full_episode = full_episode
    self.observation_space = Box(low=0, high=255, shape=(64, 64, 3)) # , dtype=np.uint8

  def _process_frame(self, frame):
    obs = frame[0:84, :, :]
    obs = Image.fromarray(obs, mode='RGB').resize((64, 64))
    obs = np.array(obs)
    return obs

  def _step(self, action):
    obs, reward, done, _ = super(CarRacingWrapper, self)._step(action)
    if self.full_episode:
      return self._process_frame(obs), reward, False, {}
    return self._process_frame(obs), reward, done, {}

from vae.vae import CVAE
from rnn.rnn import MDNRNN, rnn_next_state, rnn_init_state
class CarRacingMDNRNN(CarRacingWrapper):
  def __init__(self, args, load_model=True, full_episode=False, with_obs=False):
    super(CarRacingMDNRNN, self).__init__(full_episode=full_episode)
    self.with_obs = with_obs # whether or not to return the frame with the encodings
    self.vae = CVAE(args)
    self.rnn = MDNRNN(args)
     
    if load_model:
      self.vae.set_weights([param_i.numpy() for param_i in tf.saved_model.load('results/{}/{}/tf_vae'.format(args.exp_name, args.env_name)).variables])
      self.rnn.set_weights([param_i.numpy() for param_i in tf.saved_model.load('results/{}/{}/tf_rnn'.format(args.exp_name, args.env_name)).variables])
    self.rnn_states = rnn_init_state(self.rnn)
    
    self.full_episode = False 
    self.observation_space = Box(low=np.NINF, high=np.Inf, shape=(args.z_size+args.rnn_size*args.state_space))
  def encode_obs(self, obs):
    # convert raw obs to z, mu, logvar
    result = np.copy(obs).astype(np.float)/255.0
    result = result.reshape(1, 64, 64, 3)
    z = self.vae.encode(result)[0]
    return z
  def reset(self):
    self.rnn_states = rnn_init_state(self.rnn)
    if self.with_obs:
        [z_state, obs] = super(CarRacingMDNRNN, self).reset() # calls step
        self.N_tiles = len(self.track)
        return [z_state, obs]
    else:
        z_state = super(CarRacingMDNRNN, self).reset() # calls step
        self.N_tiles = len(self.track)
        return z_state
  def _step(self, action):
    obs, reward, done, _ = super(CarRacingMDNRNN, self)._step(action)
    z = tf.squeeze(self.encode_obs(obs))
    h = tf.squeeze(self.rnn_states[0])
    c = tf.squeeze(self.rnn_states[1])
    if self.rnn.args.state_space == 2:
        z_state = tf.concat([z, c, h], axis=-1)
    else:
        z_state = tf.concat([z, h], axis=-1)
    if action is not None: # don't compute state on reset
        self.rnn_states = rnn_next_state(self.rnn, z, action, self.rnn_states)
    if self.with_obs:
        return [z_state, obs], reward, done, {}
    else:
        return z_state, reward, done, {}
  def close(self):
    super(CarRacingMDNRNN, self).close()
    tf.keras.backend.clear_session()
    gc.collect()

def make_env(args, dream_env=False, seed=-1, render_mode=False, full_episode=False, with_obs=False, load_model=True):
  if args.env_name == 'DoomTakeCover-v0':
        raise ValueError('training for carracing is not yet supported')

  else:
    if dream_env:
      raise ValueError('training in dreams for carracing is not yet supported')
    else:
      print('makeing real CarRacing environment')
      env = CarRacingMDNRNN(args=args, full_episode=full_episode, with_obs=with_obs, load_model=load_model)
  if (seed >= 0):
    env.seed(seed)
  return env
