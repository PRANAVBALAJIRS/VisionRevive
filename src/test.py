def test(net, loader_test, max_psnr, max_ssim, step):
    net.eval()
    torch.cuda.empty_cache()
    ssims, psnrs = [], []
    for i, (inputs, targets) in enumerate(loader_test):
        inputs = inputs.to(device); targets = targets.to(device)
        pred = net(inputs)
        # # print(pred)
        # tfs.ToPILImage()(torch.squeeze(targets.cpu())).save('111.png')
        # vutils.save_image(targets.cpu(),'target.png')
        # vutils.save_image(pred.cpu(),'pred.png')
        ssim1 = ssim(pred, targets).item()
        psnr1 = psnr(pred, targets)
        ssims.append(ssim1)
        psnrs.append(psnr1)
        #if (psnr1>max_psnr or ssim1 > max_ssim) and s :
#             ts=vutils.make_grid([torch.squeeze(inputs.cpu()),torch.squeeze(targets.cpu()),torch.squeeze(pred.clamp(0,1).cpu())])
#             vutils.save_image(ts,f'samples/{model_name}/{step}_{psnr1:.4}_{ssim1:.4}.png')
#             s=False
    return np.mean(ssims) ,np.mean(psnrs)

task = 'its'
test_imgs = '../input/synthetic-objective-testing-set-sots-reside/indoor/hazy/'

dataset = task
img_dir = test_imgs

output_dir = f'pred_FFA_{dataset}/'
print("pred_dir:",output_dir)

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

ckp = torch.load(model_dir, map_location=device)
net = FFA(gps=gps, blocks=blocks)
net = nn.DataParallel(net)
net.load_state_dict(ckp['model'])
net.to(device)  # Ensure the model is on the right device
net.eval()

for im in os.listdir(img_dir):
    haze = Image.open(img_dir+im)
    haze1 = tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize(mean=[0.64, 0.6, 0.58],std=[0.14,0.15, 0.152])
    ])(haze)[None,::]
    
    haze1 = haze1.to(device)
    haze_no = tfs.ToTensor()(haze)[None,::]
    
    with torch.no_grad():
        pred = net(haze1)
    
    ts = torch.squeeze(pred.clamp(0,1).cpu())
    
    haze_no = make_grid(haze_no, nrow=1, normalize=True)
    ts = make_grid(ts, nrow=1, normalize=True)
    image_grid = torch.cat((haze_no, ts), -1)
    vutils.save_image(image_grid, output_dir + im.split('.')[0] + '_FFA.png')