# coding: utf-8
import numpy as np
# import sys
# sys.path.append('/data/yjj/chineseocr-yolo')
import torch
from PIL import Image
import numpy as np
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import numpy as np
from warpctc_pytorch import CTCLoss
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from train.ocr.dataset import PathDataset,randomSequentialSampler,alignCollate
from crnn.keys import alphabetChinese
from glob import glob
from sklearn.model_selection import train_test_split
from crnn.models.crnn import CRNN
from config import LSTMFLAG, GPU
##优化器
from crnn.util import strLabelConverter
from train.ocr.generic_utils import Progbar
from train.ocr.dataset import resizeNormalize
from crnn.util import loadData
from random import shuffle
import logging
logging.basicConfig(filename='train.log', level=logging.DEBUG, format='%(asctime)s %(message)s')


trainP = glob('/data/share_data/crnn_train/gen_no_lstm/100w/*/*.jpg')
shuffle(trainP)
#train just partial data
trainP = trainP[:500000]
print('train data num: ', len(trainP))
testP = glob('/data/yjj/chineseocr-yolo/imgs/train/*.jpg')
tempP = trainP[-20:-1]
testP += tempP
nepochs = 10
batchSize = 64
learning_reate = 1e-4
ocrModel = './models/epoch2_step1000_model_dense.pth'
display_inter = 100
saving_per_step = 1000

# trainP = glob('/data/share_data/crnn_train/gen_from_360w/train/*/*.jpg')
# testP = glob('/data/share_data/crnn_train/gen_from_360w/val/*/*.jpg')
# testP = glob('/data/yjj/chineseocr-yolo/imgs/sample2/*.jpg')
##此处未考虑字符平衡划分
# trainP,testP = train_test_split(roots,test_size=0.1)
print('testP: ', testP, 'testP num: ', len(testP))
# tranP should be shuffled at every start
traindataset = PathDataset(trainP,alphabetChinese)
testdataset = PathDataset(testP,alphabetChinese)

workers = 0
imgH = 32
imgW = 280
keep_ratio = True
cuda = True
ngpu = 1
nh =256
sampler = randomSequentialSampler(traindataset, batchSize)
train_loader = torch.utils.data.DataLoader(
    traindataset, batch_size=batchSize,
    shuffle=False, sampler=sampler,
    num_workers=int(workers),
    collate_fn=alignCollate(imgH=imgH, imgW=imgW, keep_ratio=keep_ratio))

train_iter = iter(train_loader)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def trainBatch(net, criterion, optimizer, cpu_images, cpu_texts):
    # data = train_iter.next()
    # cpu_images, cpu_texts = data
    batch_size = cpu_images.size(0)
    loadData(image, cpu_images)
    t, l = converter.encode(cpu_texts)

    loadData(text, t)
    loadData(length, l)
    preds = net(image)
    preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
    cost = criterion(preds, text, preds_size, length) / batch_size
    net.zero_grad()
    cost.backward()
    optimizer.step()


    return cost


def predict(im):
    """
    预测
    """
    image = im.convert('L')
    scale = image.size[1] * 1.0 / 32
    w = image.size[0] / scale
    w = int(w)
    transformer = resizeNormalize((w, 32))

    image = transformer(image)
    if torch.cuda.is_available():
        image = image.cuda()
    image = image.view(1, *image.size())
    image = Variable(image)
    preds = model(image)
    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)
    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    # print('sim_pred: ', sim_pred)
    return sim_pred

# acc is computed by val images , so pe patient
def val(net, dataset, max_iter=100):
    for p in net.parameters():
        p.requires_grad = False
    net.eval()
    i = 0
    n_correct = 0
    N = len(dataset)

    max_iter = min(max_iter, N)
    for i in range(max_iter):
        # im, label = dataset[np.random.randint(0, N)]
        im, label = dataset[i]
        if im.size[0] > 1024:
            continue

        pred = predict(im)
        print('')
        # logging.debug('')
        # logging.debug('compare: true: ', label, 'pred:', str(pred))
        if pred is not None:
            log_msg = label + ' |pred: ' + str(pred)
            logging.debug(log_msg)
        print('compare: true: ', label, 'pred:', pred)
        if pred.strip() == label:
            n_correct += 1
    # print('n_correct: ', n_correct)
    accuracy = n_correct / float(max_iter)
    print('val_accruracy: ' , accuracy)
    return accuracy

print('lstm: ', LSTMFLAG)
model = CRNN(32, 1, len(alphabetChinese) + 1, 256, 1, lstmFlag=LSTMFLAG)
# just run this line when training from strach
# model.apply(weights_init)
print('load weights: ', ocrModel)
preWeightDict = torch.load(ocrModel, map_location=lambda storage, loc: storage)  ##加入项目训练的权重

modelWeightDict = model.state_dict()

for k, v in preWeightDict.items():
    name = k.replace('module.', '')  # remove `module.`
    if 'rnn.1.embedding' not in name:  ##不加载最后一层权重
        modelWeightDict[name] = v

model.load_state_dict(modelWeightDict)
print('model has been loaded')
#print(model)

# if dense optimizer = SGD; if lstm optimizer = adadelta
# lr = 0.1

# optimizer = optim.Adadelta(model.parameters(), lr=0.001)
optimizer = optim.SGD(model.parameters(), lr=learning_reate, momentum=0.6)
converter = strLabelConverter(''.join(alphabetChinese))
criterion = CTCLoss()


image = torch.FloatTensor(batchSize, 3, imgH, imgH)
text = torch.IntTensor(batchSize * 5)
length = torch.IntTensor(batchSize)

if torch.cuda.is_available():
    model.cuda()
    model = torch.nn.DataParallel(model, device_ids=[0])##转换为多GPU训练模型
    image = image.cuda()
    criterion = criterion.cuda()






acc = 0
interval = len(train_loader) // display_inter  ##评估模型
print("interval : " ,interval)

for i in range(nepochs):
    print('epoch:{}/{}'.format(i, nepochs))
    n = len(train_loader)
    # print(n, ' batchs in every epochs')
    pbar = Progbar(target=n, verbose=1)
    train_iter = iter(train_loader)
    loss = 0
    # for j in range(n):
    #     for p in model.named_parameters():
    #         p[1].requires_grad = True
    #         if 'rnn.1.embedding' in p[0]:
    #             p[1].requires_grad = True
    #         else:
    #             p[1].requires_grad = False  ##冻结模型层
    # loop for step, j = step num
    for j in range(n):
        for p in model.named_parameters():
            p[1].requires_grad = True

        model.train()
        cpu_images, cpu_texts = train_iter.next()
        cost = trainBatch(model, criterion, optimizer, cpu_images, cpu_texts)

        loss += cost.data.numpy()
        # print('loss: ', '%.10f' % loss)

        if (j + 1) % interval == 0:
            # print('do val.. ')
            curAcc = val(model, testdataset, max_iter=1024)
            print('loss sum' , loss)
            print('training loss: ', loss / ((j + 1) * batchSize))
            logging.debug('loss: ')
            logging.debug(str(loss / ((j + 1) * batchSize)))


            if curAcc > acc:
                acc = curAcc
                # torch.save(model.state_dict(), './save_weights/'+'epoch{}'.format(i)+'_model_dense.pth')
                # print('saved new weights, in epoch: ', i)

        pbar.update(j + 1, values=[('loss', loss / ((j + 1) * batchSize)), ('acc', acc)])

        #for every epoch save model
        if j % saving_per_step == 0:
            torch.save(model.state_dict(), './save_weights/' + 'epoch{}'.format(i) + '_step{}'.format(j)+'_model_dense.pth')


# print('begin predict: ')
# N = len(testdataset)
# im, label = testdataset[np.random.randint(0, N)]
# pred = predict(im)
# print('true:{},pred:{}'.format(label, pred))

