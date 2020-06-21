from gensim.models.word2vec import Word2Vec
 
# 读取数据，用gensim中的word2vec训练词向量
file = open('./corpus.txt', encoding='UTF-8')
sss=[]
while True:
    ss=file.readline().replace('\n','').rstrip()
    if ss=='':
        break
    s1=ss.split(" ")
    sss.append(s1)
file.close()
model = Word2Vec(size=200, workers=4,sg=1, min_count=4) 
model.build_vocab(sss)
model.train(sss,total_examples = model.corpus_count,epochs = model.iter)
model.save('./gensim_w2v_sg1_mc6_model') 
model.wv.save_word2vec_format("./readable", binary = "Ture/False")
print("23333")