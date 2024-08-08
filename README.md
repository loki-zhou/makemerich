# makemerich

pip config set global.index-url http://pypi.tuna.tsinghua.edu.cn/simple
pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn
pip config unset global.index-url

pip install stable-baselines3[extra]

pip install stable-baselines3



tensorboard.exe --logdir tlog/FlappyBird/ppo

# convert   jupyter to  markdown
jupyter nbconvert --to markdown render_jupyter.ipynb