#在for循环推断中，将GPU中的tensor清理掉
del inputs, labels, outputs, preds #这些都是GPU中的变量
torch.cuda.empty_cache()



# 测试网络,获得输出的feature map:
m = nn.Linear(20, 30)
input = torch.randn(128, 20)
output = m(input)
print(output.size())
torch.Size([128, 30])

#格式化多张图片到一张图片：(20,w,h) -> (20,1,w,h)
images = [transform(image) for image in images]
images = torch.cat([t.unsqueeze(0) for t in images], 0)

#在训练OCR模型时，快速获得字典和字符的index
self.alphabet = alphabet + 'ç'  # for `-1` index
self.dict = {}
for i, char in enumerate(alphabet):
    # NOTE: 0 is reserved for 'blank' required by wrap_ctc
    self.dict[char] = i + 1
    
#绘制网络结构，并保存为pdf
out_HR = net(net_input)
dot = make_dot(out_HR, params=dict(net.named_parameters()))
dot.render(filename='./net')

#获得网络的summary，包括每一层输出
from torchsummary import summary

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
model = Net().to(device)
#有些情况不用输入第一个维度
summary(model, (1, 28, 28))

#将numpy转换为torch tensor，并且HWC to CHW
img_HR = torch.from_numpy(np.ascontiguousarray(np.transpose(img_HR, (2, 0, 1)))).float()


