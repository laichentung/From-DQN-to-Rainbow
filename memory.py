# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import torch
import random
import operator


class ReplayMemory:
  def __init__(self, args, capacity):
    self.capacity = capacity
    self.memory = []
    self.discount = args.discount
    self.device = args.device
    self.nsteps = args.multi_step
    self.nstep_buffer = []

  def push(self, s, a, r, s_, done):
    s_ = None if done else s_
    self.nstep_buffer.append((s, a, r, s_))
    if done:
      while len(self.nstep_buffer) > 0:
        self._sum_reward_append(s_)
    else:
      if(len(self.nstep_buffer)<self.nsteps):
        return
      self._sum_reward_append(s_)

  def _sum_reward_append(self, s_):
    R = sum([self.nstep_buffer[i][2]*(self.discount**i) for i in range(len(self.nstep_buffer))])
    state, action, _, _ = self.nstep_buffer.pop(0)
    self.memory.append((state, action, R, s_))
    if len(self.memory) > self.capacity:
      del self.memory[0]

  def simple_push(self, s, a, r, s_):
    self.memory.append((s, a, r, s_))

  def sample(self, batch_size):
    # random transition batch is taken from experience replay memory
    transitions = random.sample(self.memory, batch_size)
    
    batch_state, batch_action, batch_reward, batch_next_state = zip(*transitions)

    # batch_state = torch.tensor(list(batch_state), device=self.device, dtype=torch.float).view(batch_size, 4, 84, 84)
    batch_state = torch.stack(batch_state).to(device=self.device)
    batch_action = torch.tensor(batch_action, device=self.device, dtype=torch.long).squeeze().view(-1, 1)
    batch_reward = torch.tensor(batch_reward, device=self.device, dtype=torch.float).squeeze().view(-1, 1)
    
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch_next_state)), device=self.device, dtype=torch.bool)
    try: #sometimes all next states are false
      non_final_next_states = torch.stack([s for s in batch_next_state if s is not None]).to(device=self.device)
      empty_next_state_values = False
    except:
      non_final_next_states = None
      empty_next_state_values = True

    return batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, None, None

  def __len__(self):
    return len(self.memory)

  def __iter__(self):
    self.current_idx = 0
    return self

  # Return valid states for validation
  def __next__(self):
    if self.current_idx == self.capacity:
      raise StopIteration
    data = self.memory[self.current_idx]
    state = torch.tensor(data[0], device=self.device, dtype=torch.float)  # Agent will turn into batch
    self.current_idx += 1
    return state


class SegmentTree(object):
  def __init__(self, capacity, operation, neutral_element):
    """Build a Segment Tree data structure.
    https://en.wikipedia.org/wiki/Segment_tree
    Can be used as regular array, but with two
    important differences:
        a) setting item's value is slightly slower.
           It is O(lg capacity) instead of O(1).
        b) user has access to an efficient ( O(log segment size) )
           `reduce` operation which reduces `operation` over
           a contiguous subsequence of items in the array.
    Paramters
    ---------
    capacity: int
        Total size of the array - must be a power of two.
    operation: lambda obj, obj -> obj
        and operation for combining elements (eg. sum, max)
        must form a mathematical group together with the set of
        possible values for array elements (i.e. be associative)
    neutral_element: obj
        neutral element for the operation above. eg. float('-inf')
        for max and 0 for sum.
    """
    assert capacity > 0 and capacity & (capacity - 1) == 0, "capacity must be positive and a power of 2."
    self._capacity = capacity
    self._value = [neutral_element for _ in range(2 * capacity)]
    self._operation = operation

  def _reduce_helper(self, start, end, node, node_start, node_end):
    if start == node_start and end == node_end:
      return self._value[node]
    mid = (node_start + node_end) // 2
    if end <= mid:
      return self._reduce_helper(start, end, 2 * node, node_start, mid)
    else:
      if mid + 1 <= start:
        return self._reduce_helper(start, end, 2 * node + 1, mid + 1, node_end)
      else:
        return self._operation(
          self._reduce_helper(start, mid, 2 * node, node_start, mid),
          self._reduce_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end)
        )

  def reduce(self, start=0, end=None):
    """Returns result of applying `self.operation`
    to a contiguous subsequence of the array.
        self.operation(arr[start], operation(arr[start+1], operation(... arr[end])))
    Parameters
    ----------
    start: int
        beginning of the subsequence
    end: int
        end of the subsequences
    Returns
    -------
    reduced: obj
        result of reducing self.operation over the specified range of array elements.
    """
    if end is None:
      end = self._capacity
    if end < 0:
      end += self._capacity
    end -= 1
    return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

  def __setitem__(self, idx, val):
    # index of the leaf
    idx += self._capacity
    self._value[idx] = val
    idx //= 2
    while idx >= 1:
      self._value[idx] = self._operation(
        self._value[2 * idx],
        self._value[2 * idx + 1]
      )
      idx //= 2

  def __getitem__(self, idx):
    assert 0 <= idx < self._capacity
    return self._value[self._capacity + idx]


class SumSegmentTree(SegmentTree):
  def __init__(self, capacity):
    super(SumSegmentTree, self).__init__(
      capacity=capacity,
      operation=operator.add,
      neutral_element=0.0
    )

  def sum(self, start=0, end=None):
    """Returns arr[start] + ... + arr[end]"""
    return super(SumSegmentTree, self).reduce(start, end)

  def find_prefixsum_idx(self, prefixsum):
    """Find the highest index `i` in the array such that
        sum(arr[0] + arr[1] + ... + arr[i - i]) <= prefixsum
    if array values are probabilities, this function
    allows to sample indexes according to the discrete
    probability efficiently.
    Parameters
    ----------
    perfixsum: float
        upperbound on the sum of array prefix
    Returns
    -------
    idx: int
        highest index satisfying the prefixsum constraint
    """
    try:
      assert 0 <= prefixsum <= self.sum() + 1e-5
    except AssertionError:
      print("Prefix sum error: {}".format(prefixsum))
      exit()
    idx = 1
    while idx < self._capacity:  # while non-leaf
      if self._value[2 * idx] > prefixsum:
        idx = 2 * idx
      else:
        prefixsum -= self._value[2 * idx]
        idx = 2 * idx + 1
    return idx - self._capacity


class MinSegmentTree(SegmentTree):
  def __init__(self, capacity):
    super(MinSegmentTree, self).__init__(
      capacity=capacity,
      operation=min,
      neutral_element=float('inf')
    )

  def min(self, start=0, end=None):
    """Returns min(arr[start], ...,  arr[end])"""

    return super(MinSegmentTree, self).reduce(start, end)


class PrioritizedReplayMemory(object):
  def __init__(self, args, capacity):
    super(PrioritizedReplayMemory, self).__init__()
    self._storage = []
    self._maxsize = capacity
    self._next_idx = 0

    assert args.priority_exponent >= 0
    self._alpha = args.priority_exponent

    self.beta_start = args.priority_weight
    self.beta_frames = args.T_max - args.learn_start
    self.frame = 1

    it_capacity = 1
    while it_capacity < capacity:
      it_capacity *= 2

    self._it_sum = SumSegmentTree(it_capacity)
    self._it_min = MinSegmentTree(it_capacity)
    self._max_priority = 1.0
    self.device = args.device
    self.nsteps = args.multi_step
    self.nstep_buffer = []
    self.discount = args.discount

  def beta_by_frame(self, frame_idx):
    return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)

  def push(self, s, a, r, s_, done):
    s_ = None if done else s_
    self.nstep_buffer.append((s, a, r, s_))
    if done:
      while len(self.nstep_buffer) > 0:
        R = sum([self.nstep_buffer[i][2] * (self.discount ** i) for i in range(len(self.nstep_buffer))])
        state, action, _, _ = self.nstep_buffer.pop(0)
    else:
      if (len(self.nstep_buffer) < self.nsteps):
        return
      R = sum([self.nstep_buffer[i][2] * (self.discount ** i) for i in range(len(self.nstep_buffer))])
      state, action, _, _ = self.nstep_buffer.pop(0)

    idx = self._next_idx

    if self._next_idx >= len(self._storage):
      self._storage.append((state, action, R, s_))
    else:
      self._storage[self._next_idx] = (state, action, R, s_)
    self._next_idx = (self._next_idx + 1) % self._maxsize

    self._it_sum[idx] = self._max_priority ** self._alpha
    self._it_min[idx] = self._max_priority ** self._alpha

  def _encode_sample(self, idxes):
    return [self._storage[i] for i in idxes]

  def _sample_proportional(self, batch_size):
    res = []
    for _ in range(batch_size):
      mass = random.random() * self._it_sum.sum(0, len(self._storage) - 1)
      idx = self._it_sum.find_prefixsum_idx(mass)
      res.append(idx)
    return res

  def sample(self, batch_size):
    idxes = self._sample_proportional(batch_size)

    weights = []

    # find smallest sampling prob: p_min = smallest priority^alpha / sum of priorities^alpha
    p_min = self._it_min.min() / self._it_sum.sum()

    beta = self.beta_by_frame(self.frame)
    self.frame += 1

    # max_weight given to smallest prob
    max_weight = (p_min * len(self._storage)) ** (-beta)

    for idx in idxes:
      p_sample = self._it_sum[idx] / self._it_sum.sum()
      weight = (p_sample * len(self._storage)) ** (-beta)
      weights.append(weight / max_weight)
    weights = torch.tensor(weights, device=self.device, dtype=torch.float)
    encoded_sample = self._encode_sample(idxes)
    return encoded_sample, idxes, weights

  def update_priorities(self, idxes, priorities):
    assert len(idxes) == len(priorities)
    for idx, priority in zip(idxes, priorities):
      assert 0 <= idx < len(self._storage)
      self._it_sum[idx] = (priority + 1e-5) ** self._alpha
      self._it_min[idx] = (priority + 1e-5) ** self._alpha

      self._max_priority = max(self._max_priority, (priority + 1e-5))



