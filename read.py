import csv

sentence = []

with open('./mbti.csv', 'r', encoding='UTF-8') as f:
	reader = csv.reader(f)
	print(type(reader))
	for row in reader:
		sentence.append(row[1].replace("|||", " "))


with open('./vocalib.txt', 'w', encoding='UTF-8') as f:
	for item in sentence:
		f.write(item)
		f.write(" ")