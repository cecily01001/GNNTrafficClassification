import numpy
import pandas
from pandas import read_csv
import matplotlib.pyplot as plt

recall = read_csv("malware_recall.csv")
# x = pandas.np.arange(1, 35, 1)
# plt.plot(x,recall['GNN1'],'-',color='r',marker='o',  label='GNN')
# plt.plot(x,recall['Deep Packet1'],'-',color='b',marker='o',  label='Deep Packet')
# plt.plot(x,recall['1D-CNN1'],'-',color='g',marker='o',  label='1D-CNN')
# plt.plot(x,recall['2D-CNN1'],'-',color='purple',marker='o', label='2D-CNN')
# plt.legend()  # 图例
# plt.show()#可视化展示
# gb = recall.groupby(
#     by = ['alg'],
#     as_index=False
# )['accuracy'].agg({
#     'accuracy':numpy.sum #求和
# })
# index = ['Not Standardized','standardized']
# value=[0.86,0.81]
# plt.bar(index,value,width=0.3, label="Accuracy", fc = "b")
# plt.xlabel(" ",fontsize=10)
# plt.ylabel("Accuracy",fontsize=10)
# for a,b in zip(index,value):   #柱子上的数字显示
#     plt.text(a,b,'%.3f'%b,ha='center',va='bottom',fontsize=10)
# plt.show()#可视化展示


# name = recall['app']
# x= list(range(len(recall['GNN'])))
# width = 0.2
# plt.bar(x, recall['GNN1'],width=width, color='r', label='GNN',tick_label = name)
# plt.xticks(rotation=45)
# for i in range(len(x)):
#     x[i] = x[i] + width
# plt.bar(x, recall['Deep Packet1'], width=width, color='b', label='Deep Packet')
# for i in range(len(x)):
#     x[i] = x[i] + width
# plt.bar(x, recall['1D-CNN1'], width=width,color='g', label='1D-CNN')
# for i in range(len(x)):
#     x[i] = x[i] + width
# plt.bar(x, recall['2D-CNN1'], width=width,color='purple', label='2D-CNN')
# plt.xlabel("Models",fontsize=10)
# plt.ylabel("precision",fontsize=10)
# plt.bar(x, recall['GNN2'],width=width, color='r', label='GNN',tick_label = name)
# plt.xticks(rotation=45)
# for i in range(len(x)):
#     x[i] = x[i] + width
# plt.bar(x, recall['Deep Packet2'], width=width, color='b', label='Deep Packet')
# for i in range(len(x)):
#     x[i] = x[i] + width
# plt.bar(x, recall['1D-CNN2'], width=width,color='g', label='1D-CNN')
# for i in range(len(x)):
#     x[i] = x[i] + width
# plt.bar(x, recall['2D-CNN2'], width=width,color='purple', label='2D-CNN')
# plt.xlabel("Models",fontsize=10)
# plt.ylabel("precision",fontsize=10)
# plt.legend(loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = 0.)  # 图例
# plt.show()  # 可视化展示

recall = read_csv("malware_accuracy.csv")
name = recall['model']
value=recall['acc']
plt.bar(name,value,width=0.3, label="Accuracy", fc = "b")
plt.xlabel("The number of layers of GNN",fontsize=10)
plt.ylabel("Accuracy",fontsize=10)
for a,b in zip(name,value):   #柱子上的数字显示
    plt.text(a,b,'%.3f'%b,ha='center',va='bottom',fontsize=10)
plt.show()#可视化展示