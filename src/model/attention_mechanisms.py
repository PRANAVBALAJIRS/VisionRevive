class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
    def forward(self, x):
        y = self.pa(x)
        return x * y
    
class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y
    
class EfficientAttention(nn.Module):
    def __init__(self, channel):
        super(EfficientAttention, self).__init__()
        self.conv1 = nn.Conv2d(channel, channel, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(channel, channel, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        attn_map = self.sigmoid(self.conv1(x))
        refined = attn_map * self.conv2(x)
        return refined