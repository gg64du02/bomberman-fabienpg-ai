import matplotlib.pyplot as plt
loss = []
acc = []
val_loss = []
val_acc = []
with open('points_run6_clean.txt','r') as f:
    read_data = f.read()
    # print(read_data)
    lol1 = read_data.split('-')
    # lol2 = lol1.split('-')
    print(lol1)
    # lol3 = lol2.split('-')
    # print(lol2)
    for word in lol1:
        print('word',word)
        # if(not '-' in word):
        if(' loss' in word):
            loss.append(float(word.split(':')[1]))
        if(' acc' in word):
            lol4 = word.split('\n')
            # print('lol4',lol4)
            lol5 = lol4[0].split(' ')
            # print('lol5',lol5)
            acc.append(float(lol5[2]))
        if(' val_loss' in word):
            val_loss.append(float(word.split(':')[1]))
        if(' val_acc' in word):
            lol4 = word.split('\n')
            print('lol4',lol4)
            lol5 = lol4[0].split(' ')
            print('lol5',lol5)
            val_acc.append(float(lol5[2]))

            # lol4 = word.replace('\n',' ')
            # # val_acc.append(float(word.split(':')[1]))
            # val_acc.append(float(lol4[0]))
# print('loss',loss)
plt.plot(loss)
plt.plot(acc)
plt.plot(val_loss)
plt.plot(val_acc)
plt.ylabel('some numbers')
plt.show()
f.closed