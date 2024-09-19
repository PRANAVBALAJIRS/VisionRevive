class ProgressiveAttentionRefinement(nn.Module):
    def __init__(self, channels):
        super(ProgressiveAttentionRefinement, self).__init__()
        self.ca = nn.Sequential(
            nn.Conv2d(channels, channels // 8, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        self.pa = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        ca_weight = self.ca(x)
        pa_weight = self.pa(x)
        return x * ca_weight * pa_weight