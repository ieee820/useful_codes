# 测试网络,获得输出的feature map:
m = nn.Linear(20, 30)
input = torch.randn(128, 20)
output = m(input)
print(output.size())
torch.Size([128, 30])

#格式化多张图片到一张图片：(20,w,h) -> (20,1,w,h)
images = [transform(image) for image in images]
images = torch.cat([t.unsqueeze(0) for t in images], 0)
