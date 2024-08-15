from graphviz import Digraph

# 创建一个有向图，并设置方向为从左到右
dot = Digraph(comment='Two Layer Neural Network', format='png', engine='dot')
dot.attr(rankdir='LR')

# 添加输入层节点
dot.node('I1', 'Input 1')
dot.node('I2', 'Input 2')
dot.node('I3', 'Input 3')

# 添加隐藏层节点
dot.node('H1', 'Hidden 1')
dot.node('H2', 'Hidden 2')
dot.node('H3', 'Hidden 3')
dot.node('H4', 'Hidden 4')

# 添加输出层节点
dot.node('O1', 'Output 1')

# 添加连接
dot.edges([('I1', 'H1'), ('I1', 'H2'), ('I1', 'H3'), ('I1', 'H4')])
dot.edges([('I2', 'H1'), ('I2', 'H2'), ('I2', 'H3'), ('I2', 'H4')])
dot.edges([('I3', 'H1'), ('I3', 'H2'), ('I3', 'H3'), ('I3', 'H4')])
dot.edges([('H1', 'O1'), ('H2', 'O1'), ('H3', 'O1'), ('H4', 'O1')])

# 保存并显示图形
dot.render('two_layer_nn_horizontal', view=True)