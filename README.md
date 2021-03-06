# SI151 Project: Social Media Based MBTI Personality Prediction

## Introduction

This is the course project of SI151: Optimization and Machine Learning at [Shanghaitech University](https://www.shanghaitech.edu.cn) lectured by professor [Lu Sun](https://sist.shanghaitech.edu.cn/2020/0707/c7499a53879/page.htm). The authors' information is mentioned in the report(report/report.pdf). </br>
Report and its source code are included in folder "report", and you are welcomed to view our methods and results. However, we strongly recommend you not to copy our work for your own project. </br>
The reference listed at the end of this doc are rather helpful, as you may have understanding about the background knowledge. </br>

### Dependencies
The project is implement with Python, with many third-party librarys installed. I was too lazy to list them so you may find out by yourself.

### Ways to run the program
Find out by yourself.

### Method of label encoding:
+ Four bits binary string.
+ First bit: 0-E 1-I
+ Second bit: 0-S 1-N
+ Third bit: 0-T 1-F
+ Fourth bit: 0-J 1-P

### Note:
If you find any mistakes in our work, don't be surprised. We know it is far away from perfect, and there is a lot to do to improve it. Life has been so hard that we just wanted to finish the course project.


## Roadmap: What we have done
:heavy_check_mark: Feature extraction </br>
:heavy_check_mark: SVM </br>
:heavy_check_mark: Logistic regression  </br>
:heavy_check_mark: Naive bayes </br>
:heavy_check_mark: Neural network </br>
:heavy_check_mark: Come up with something new </br>
:heavy_check_mark: Write the report. </br>
:heavy_check_mark: Record the vadio. </br>

## Result: Accuracy summary
It has been visualized in the report, as well as the picture in /report/fig/result.jpg.

| Method | Dimension 0 | Dimension 1 | Dimension 2 | Dimension 3 | Multi-class |
| ------ | ------ | ------ | ------ | ------ | ------ |
| Support Vector Machine | 77% | 86.7% | 73% | 68.7% | 33.9% |
| Logistic Regression | 67% | 70% | 73% | 66% | 16% |
| Naive Bayes | 38% | 75% | 59% | 47% |   |
| Sequential Network | 77% | 89% | 55% | 60% | 24% |
| LSTM Network | 82.3% | 91.7% | 74.7% | 73.2% | 41.2% |


## Reference:
### Other github repo
[Predicting-Myers-Briggs-Type-Indicator-with-Recurrent-Neural-Networks](https://github.com/ianscottknight/Predicting-Myers-Briggs-Type-Indicator-with-Recurrent-Neural-Networks)</br>
[LogisticRegression](https://github.com/perborgen/LogisticRegression)</br>
### Word2vec
[Official document for gensim library](https://radimrehurek.com/gensim/models/word2vec.html) </br>
[Introduction to theory of word2vec](https://blog.csdn.net/huacha__/article/details/84068653) </br>
[Introduction to word embedding from Zhihu](https://www.zhihu.com/question/32275069) </br>
### Relative Papers
[Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546) </br>
[Machine Learning Approach to Personality Type Prediction Based on the Myers-Briggs Type Indicator](https://www.researchgate.net/publication/339935842_Machine_Learning_Approach_to_Personality_Type_Prediction_Based_on_the_Myers-Briggs_Type_Indicator_R) </br>
### Neural network classifier
[Tensorflow inplementation](https://blog.csdn.net/sinat_29957455/article/details/78324082)</br>
### Logistic Regression
[Sklearn-LogisticRegression](https://blog.csdn.net/CherDW/article/details/54891073)</br>
[LogisticRegressionCV Arguments Introduction](https://blog.csdn.net/weixin_41690708/article/details/95171333)</br>
### XGboost
[Example of XGboost](https://blog.csdn.net/u011630575/article/details/79418138?ops_request_misc=&request_id=&biz_id=102&utm_term=XGBoost&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-0-79418138) </br>
[Theroy of XGboost](https://blog.csdn.net/sb19931201/article/details/52557382?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522159306924419195162532096%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=159306924419195162532096&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_click~default-2-52557382.first_rank_ecpm_v3_pc_rank_v2&utm_term=XGBoost)
