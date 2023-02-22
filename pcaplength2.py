import csv
graph_path='test2.csv'
f1 = open(graph_path, 'w', encoding='utf-8', newline="")
f2 = csv.writer(f1)
with open("test.csv","r") as csvfile:
    read = csv.reader(csvfile)
    for value in read:

        value=str(value)[4:len(value)-3]
        print(value)
        list=value.split(' | ')
        appname=list[0].split(':')[0]
        precision=list[0].split(':')[2]
        recall=list[1].split(':')[1]
        f1c=list[2].split(':')[1]
        f2.writerow([appname,precision,recall,f1c])