# long_horizon_chainning

实验代码

agent:

(Pdb) agent.__dict__
{'training': True, '_parameters': OrderedDict(), '_buffers': OrderedDict(), '_non_persistent_buffers_set': set(), '_backward_pre_hooks': OrderedDict(), '_backward_hooks': OrderedDict(), '_is_full_backward_hook': None, '_forward_hooks': OrderedDict(), '_forward_hooks_with_kwargs': OrderedDict(), '_forward_hooks_always_called': OrderedDict(), '_forward_pre_hooks': OrderedDict(), '_forward_pre_hooks_with_kwargs': OrderedDict(), '_state_dict_hooks': OrderedDict(), '_state_dict_pre_hooks': OrderedDict(), '_load_state_dict_pre_hooks': OrderedDict(), '_load_state_dict_post_hooks': OrderedDict(), '_modules': OrderedDict([('actor', DeterministicActor(
  (trunk): MLP(
    (mlp): Sequential(
      (0): Linear(in_features=22, out_features=256, bias=True)
      (1): ReLU()
      (2): Linear(in_features=256, out_features=256, bias=True)
      (3): ReLU()
      (4): Linear(in_features=256, out_features=256, bias=True)
      (5): ReLU()
      (6): Linear(in_features=256, out_features=5, bias=True)
    )
  )
)), ('actor_target', DeterministicActor(
  (trunk): MLP(
    (mlp): Sequential(
      (0): Linear(in_features=22, out_features=256, bias=True)
      (1): ReLU()
      (2): Linear(in_features=256, out_features=256, bias=True)
      (3): ReLU()
      (4): Linear(in_features=256, out_features=256, bias=True)
      (5): ReLU()
      (6): Linear(in_features=256, out_features=5, bias=True)
    )
  )
)), ('critic', Critic(
  (q): MLP(
    (mlp): Sequential(
      (0): Linear(in_features=27, out_features=256, bias=True)
      (1): ReLU()
      (2): Linear(in_features=256, out_features=256, bias=True)
      (3): ReLU()
      (4): Linear(in_features=256, out_features=256, bias=True)
      (5): ReLU()
      (6): Linear(in_features=256, out_features=1, bias=True)
    )
  )
)), ('critic_target', Critic(
  (q): MLP(
    (mlp): Sequential(
      (0): Linear(in_features=27, out_features=256, bias=True)
      (1): ReLU()
      (2): Linear(in_features=256, out_features=256, bias=True)
      (3): ReLU()
      (4): Linear(in_features=256, out_features=256, bias=True)
      (5): ReLU()
      (6): Linear(in_features=256, out_features=1, bias=True)
    )
  )
))]), 'discount': 0.99, 'reward_scale': 1, 'update_epoch': 40, 'her_sampler': <experiment.algo.viskill_agents.modules.replay_buffer.HER_sampler_seq object at 0x000001DDCD1379A0>, 'device': 'cpu', 'noise_eps': 0.2, 'aux_weight': 5, 'p_dist': 2, 'soft_target_tau': 0.05, 'clip_obs': 200, 'norm_clip': 5, 'norm_eps': 0.01, 'dima': 5, 'dimo': 19, 'dimg': 3, 'max_action': 1.0, 'act_sampler': <bound method Box.sample of Box(-1.0, 1.0, (5,), float32)>, 'o_norm': <experiment.algo.viskill_agents.components.normalizer.Normalizer object at 0x000001DDCD198A60>, 'g_norm': <experiment.algo.viskill_agents.components.normalizer.Normalizer object at 0x000001DDCD198FA0>, 'actor_optimizer': Adam (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.999)
    capturable: False
    differentiable: False
    eps: 1e-08
    foreach: None
    fused: None
    lr: 0.001
    maximize: False
    weight_decay: 0
), 'critic_optimizer': Adam (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.999)
    capturable: False
    differentiable: False
    eps: 1e-08
    foreach: None
    fused: None
    lr: 0.001
    maximize: False
    weight_decay: 0
), 'k': 5}
