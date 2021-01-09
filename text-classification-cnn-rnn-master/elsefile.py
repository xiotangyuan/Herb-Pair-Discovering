import matplotlib.pyplot as plt

def huatu(train_loss,train_acc):
    #定义两个数组
    Loss_list = []
    Accuracy_list = []

    Loss_list.append(train_loss / (len(train_dataset)))
    Accuracy_list.append(100 * train_acc / (len(train_dataset)))

    #我这里迭代了200次，所以x的取值范围为(0，200)，然后再将每次相对应的准确率以及损失率附在x上
    #横坐标为总个数，纵坐标为准确率和损失率
    x1 = range(0, 1100)
    x2 = range(0, 1100)
    y1 = Accuracy_list
    y2 = Loss_list
    plt.subplot(1)
    plt.plot(x1, y1, 'o-')
    plt.title('Test accuracy vs. epoches')
    plt.ylabel('Test accuracy')
    plt.show()
    plt.subplot(2)
    plt.plot(x2, y2, '.-')
    plt.xlabel('Test loss vs. epoches')
    plt.ylabel('Test loss')
    plt.show()
    plt.savefig("accuracy_loss.jpg")

#train_dev
# def savefile(epoch,loss_train, acc_train,loss_val, acc_val):
#     with open('/media/ubuntu/My Passport/tmy/xueqi/text-classification-cnn-rnn-master/data/1.txt','a',encoding="utf-8") as f:
#         ShuJustr=str(epoch)+"," +str(loss_train)+"," +str(acc_train)+"," +str(loss_val)+"," +str(acc_val)
#         c=f.writelines(ShuJustr+"\n")
#
#     f.close()
#     print("chenggong",loss_val, acc_val)

#test
def savefile(epoch,loss_test, acc_test):
    with open('/media/ubuntu/My Passport/tmy/xueqi/text-classification-cnn-rnn-master/data/1.txt','a',encoding="utf-8") as f:
        ShuJustr=str(epoch)+"," +str(loss_test)+"," +str(acc_test)
        c=f.writelines(ShuJustr+"\n")

    f.close()
    print("chenggong",loss_test, acc_test)