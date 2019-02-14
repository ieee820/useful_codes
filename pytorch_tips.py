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

