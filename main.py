#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install pgl==1.2.0')


# In[2]:


import pgl
#本任务仍使用旧版动态图模式完成
import paddle.fluid as fluid
import numpy as np
import time
import pandas as pd


# In[3]:


from easydict import EasyDict as edict

allocation = {
    "model_name": "GCN",
    "num_layers": 1,
    "dropout": 0.5,
    "learning_rate": 0.01,
    "weight_decay": 0.0005,
    "edge_dropout": 0.00,
}

config = edict(allocation)


# In[4]:


from collections import namedtuple

Dataset = namedtuple("Dataset", 
               ["graph", "num_classes", "train_index",
                "train_label", "valid_index", "valid_label", "test_index"])

def load_edges(num_nodes, self_loop=True, add_inverse_edge=True):
    #读取数据并获得边
    edges = pd.read_csv("/home/aistudio/edges.csv", header=None, names=["src", "dst"]).values

    if add_inverse_edge:
        edges = np.vstack([edges, edges[:, ::-1]])

    if self_loop:
        src = np.arange(0, num_nodes)
        dst = np.arange(0, num_nodes)
        self_loop = np.vstack([src, dst]).T
        edges = np.vstack([edges, self_loop])
    
    return edges

def load():
    # 数据加载与划分
    node_feat = np.load("/home/aistudio/feat.npy") #加载feat.npy
    num_nodes = node_feat.shape[0]
    edges = load_edges(num_nodes=num_nodes, self_loop=True, add_inverse_edge=True)
    graph = pgl.graph.Graph(num_nodes=num_nodes, edges=edges, node_feat={"feat": node_feat})
    
    indegree = graph.indegree()
    norm = np.maximum(indegree.astype("float32"), 1)
    norm = np.power(norm, -0.5)
    graph.node_feat["norm"] = np.expand_dims(norm, -1)
    
    df = pd.read_csv("/home/aistudio/train.csv") #读取训练文件
    node_index = df["nid"].values
    node_label = df["label"].values
    train_part = int(len(node_index) * 0.8)
    train_index = node_index[:train_part]
    train_label = node_label[:train_part]
    valid_index = node_index[train_part:]
    valid_label = node_label[train_part:]
    test_index = pd.read_csv("/home/aistudio/test.csv")["nid"].values #读取测试文件
    dataset = Dataset(graph=graph, 
                    train_label=train_label,
                    train_index=train_index,
                    valid_index=valid_index,
                    valid_label=valid_label,
                    test_index=test_index, num_classes=35)
    return dataset


# In[5]:


dataset = load()
# 文件格式设置
train_index = dataset.train_index
train_label = np.reshape(dataset.train_label, [-1 , 1])
train_index = np.expand_dims(train_index, -1)

val_index = dataset.valid_index
val_label = np.reshape(dataset.valid_label, [-1, 1])
val_index = np.expand_dims(val_index, -1)

test_index = dataset.test_index
test_index = np.expand_dims(test_index, -1)
test_label = np.zeros((len(test_index), 1), dtype="int64")


# In[9]:


from pgl import data_loader
import paddle.fluid as fluid # paddle.nn
import numpy as np
import time

def build_model(dataset, config, phase, main_prog):
    gw = pgl.graph_wrapper.GraphWrapper(
            name="graph",
            node_feat=dataset.graph.node_feat_info())

    GraphModel = getattr(model, config.model_name)
    m = GraphModel(config=config, num_class=dataset.num_classes) 
    logits = m.forward(gw, gw.node_feat["feat"], phase)

    node_index = fluid.layers.data(
            "node_index",
            shape=[None, 1],
            dtype="int64",
            append_batch_size=False)
    node_label = fluid.layers.data(
            "node_label",
            shape=[None, 1],
            dtype="int64",
            append_batch_size=False)

    pred = fluid.layers.gather(logits, node_index)
    loss, pred = fluid.layers.softmax_with_cross_entropy(
        logits=pred, label=node_label, return_softmax=True)
    acc = fluid.layers.accuracy(input=pred, label=node_label, k=1)
    pred = fluid.layers.argmax(pred, -1)
    loss = fluid.layers.mean(loss)

    if phase == "train":
        adam = fluid.optimizer.Adam(
            learning_rate=config.learning_rate,
            regularization=fluid.regularizer.L2DecayRegularizer(
                regularization_coeff=config.weight_decay)) # L2DecayRegularizer, Adam
        adam.minimize(loss)
    return gw, loss, acc, pred


# In[12]:


import pgl
import paddle.fluid as fluid
import numpy as np
import time
import model # model.py参考基线

place = fluid.CUDAPlace(0)

train_program = fluid.default_main_program()
startup_program = fluid.default_startup_program()
with fluid.program_guard(train_program, startup_program):
    with fluid.unique_name.guard():
        gw, loss, acc, pred = build_model(dataset,
                            config=config,
                            phase="train",
                            main_prog=train_program)

test_program = fluid.Program()
with fluid.program_guard(test_program, startup_program):
    with fluid.unique_name.guard():
        _gw, v_loss, v_acc, v_pred = build_model(dataset,
            config=config,
            phase="test",
            main_prog=test_program)


test_program = test_program.clone(for_test=True)

exe = fluid.Executor(place) #执行器设定


# In[13]:


epoch = 200 #调参：100精度过低，250~300精度提高有限、资源消耗增加
exe.run(startup_program) #开始执行

#处理图数据方便后续运行
feed_dict = gw.to_feed(dataset.graph)

for epoch in range(epoch):
    # node_index: 训练节点的nid    
    # node_label: 训练节点对应的标签
    feed_dict["node_index"] = np.array(train_index, dtype="int64")
    feed_dict["node_label"] = np.array(train_label, dtype="int64")
    
    train_loss, train_acc = exe.run(train_program,
                                feed=feed_dict,
                                fetch_list=[loss, acc],
                                return_numpy=True)

    feed_dict["node_index"] = np.array(val_index, dtype="int64")
    feed_dict["node_label"] = np.array(val_label, dtype="int64")
    val_loss, val_acc = exe.run(test_program,
                            feed=feed_dict,
                            fetch_list=[v_loss, v_acc],
                            return_numpy=True)
    print("Epoch", epoch, "Train Acc", train_acc[0], "Valid Acc", val_acc[0])


# In[14]:


feed_dict["node_index"] = np.array(test_index, dtype="int64")
feed_dict["node_label"] = np.array(test_label, dtype="int64") #假标签
test_prediction = exe.run(test_program,
                            feed=feed_dict,
                            fetch_list=[v_pred],
                            return_numpy=True)[0] # 运行执行器


# In[15]:


submission = pd.DataFrame(data={
                            "nid": test_index.reshape(-1),
                            "label": test_prediction.reshape(-1)
                        })
submission.to_csv("submission.csv", index=False) #写出为csv


# In[ ]:




