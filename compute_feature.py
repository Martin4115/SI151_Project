from gensim.models.word2vec import Word2Vec
import csv
import re
import collections
import numpy as np

model = Word2Vec.load("./gensim_w2v_sg1_mc6_model")
feature_dim = 200


print(model.wv.__contains__('happy'))
print(type(model['happy']))

def compute_feature(freq):
	global model, feature_dim
	result = np.zeros(feature_dim)
	total_freq = 0
	for key, f in freq.items():
		if key == '':
			continue
		if model.wv.__contains__(key):
			result = result + f*model.wv.__getitem__(key)
			total_freq += f
	result = result / total_freq
	return result

def encode_type(label):
	return label




features = []
i = 0
with open('./mbti.csv', 'r', encoding='UTF-8') as f:
	reader = csv.reader(f)
	print(type(reader))
	for row in reader:
		if i == 0:
			i += 1
			continue
		para = row[1].replace("|||", " ")
		para = re.split('[,.:?! /\'=]', para)
		freq = dict(collections.Counter(para))
		# print(freq)
		result = compute_feature(freq)
		# print(result)
		t = encode_type(row[0])
		features.append((t, result))
		print(len(features))



fileHeader = ["Type", "feature"]
csvFile = open("./feature.csv", "w", newline="")
writer = csv.writer(csvFile)
writer.writerow(fileHeader)
for data in features:
	data_to_write = [data[0]]
	for i in data[1]:
		data_to_write.append(i)
	writer.writerow(data_to_write)
csvFile.close()

'''
import csv
# 文件头，一般就是数据名
fileHeader = ["name", "score"]
# 假设我们要写入的是以下两行数据
d1 = ["Wang", "100"]
d2 = ["Li", "80"]
# 写入数据
csvFile = open("instance.csv", "w")
writer = csv.writer(csvFile)
# 写入的内容都是以列表的形式传入函数
writer.writerow(fileHeader)
writer.writerow(d1)
writer.writerow(d1)
csvFile.close()'''