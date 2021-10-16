From DQN to Rainbow
=======
- [x] DQN [[1]](#references)
- [x] Double DQN [[2]](#references)
- [x] Prioritised Experience Replay [[3]](#references)
- [x] Dueling Network Architecture [[4]](#references)
- [x] Multi-step Returns [[5]](#references)
- [x] Distributional RL [[6]](#references)
- [x] Noisy Nets [[7]](#references)
- [x] Rainbow: Combining Improvements in Deep Reinforcement Learning [[8]](#references)

Experimental results and pretrained models will be released soon.

Run the ordinary DQN with the default arguments:

```
python main.py
```

Run the original Rainbow [[1]](#references) using the following options:

```
python main.py --double \
               --duel \
               --noisy \
               --distributional \
               --multi-step 3 \
               --prioritize
```

Data-efficient Rainbow [[9]](#references) can be run using the following options (note that the "unbounded" memory is implemented here in practice by manually setting the memory capacity to be the same as the maximum number of timesteps):

```
python main.py --double \
               --duel \
               --noisy \
               --distributional \
               --multi-step=3 \
               --prioritize \
               --target-update 2000 \
               --T-max 100000 \
               --learn-start 1600 \
               --memory-capacity 100000 \
               --replay-frequency 1 \
               --multi-step 20 \
               --architecture data-efficient \
               --hidden-size 256 \
               --learning-rate 0.0001 \
               --evaluation-interval 10000
```

Requirements
------------

- [atari-py](https://github.com/openai/atari-py)
- [OpenCV Python](https://pypi.python.org/pypi/opencv-python)
- [Plotly](https://plot.ly/)
- [PyTorch](http://pytorch.org/)

To install all dependencies with Anaconda run `conda env create -f environment.yml` and use `source activate rainbow` to activate the environment.

Available Atari games can be found in the [`atari-py` ROMs folder](https://github.com/openai/atari-py/tree/master/atari_py/atari_roms).

Acknowledgements
----------------

- [@Kaixhin](https://github.com/Kaixhin) for [Rainbow implemention](https://github.com/Kaixhin/Rainbow)

References
----------

[1] [Playing Atari with Deep Reinforcement Learning](http://arxiv.org/abs/1312.5602)  
[2] [Deep Reinforcement Learning with Double Q-learning](http://arxiv.org/abs/1509.06461)  
[3] [Prioritized Experience Replay](http://arxiv.org/abs/1511.05952)  
[4] [Dueling Network Architectures for Deep Reinforcement Learning](http://arxiv.org/abs/1511.06581)  
[5] [Reinforcement Learning: An Introduction](http://www.incompleteideas.net/sutton/book/ebook/the-book.html)  
[6] [A Distributional Perspective on Reinforcement Learning](https://arxiv.org/abs/1707.06887)  
[7] [Noisy Networks for Exploration](https://arxiv.org/abs/1706.10295)  
[8] [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/abs/1710.02298)  
[9] [When to Use Parametric Models in Reinforcement Learning?](https://arxiv.org/abs/1906.05243)  
