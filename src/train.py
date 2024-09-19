print('log_dir :', log_dir)
print('model_name:', model_name)

models_ = {'ffa': FFA(gps = gps, blocks = blocks)}
loaders_ = {'its_train': ITS_train_loader, 'its_test': ITS_test_loader}
# loaders_ = {'its_train': ITS_train_loader, 'its_test': ITS_test_loader, 'ots_train': OTS_train_loader, 'ots_test': OTS_test_loader}
start_time = time.time()
T = steps

def train(net, loader_train, loader_test, optim, criterion):
    losses = []
    start_step = 0
    max_ssim = max_psnr = 0
    ssims, psnrs = [], []
    if resume and os.path.exists(pretrained_model_dir):
        print(f'resume from {pretrained_model_dir}')
        ckp = torch.load(pretrained_model_dir)
        losses = ckp['losses']
        net.load_state_dict(ckp['model'])
        start_step = ckp['step']
        max_ssim = ckp['max_ssim']
        max_psnr = ckp['max_psnr']
        psnrs = ckp['psnrs']
        ssims = ckp['ssims']
        print(f'Resuming training from step: {start_step} ***')
    else :
        print('Training from scratch *** ')
    for step in range(start_step+1, steps+1):
        net.train()
        accumulation_steps = 4
        lr = learning_rate
        if not no_lr_sche:
            lr = lr_schedule_cosdecay(step,T)
            for param_group in optim.param_groups:
                param_group["lr"] = lr
        x, y = next(iter(loader_train))
        x = x.to(device); y = y.to(device)
        torch.cuda.empty_cache()
        out = net(x)
        loss = criterion[0](out,y)
#         loss = loss / accumulation_steps
#             loss.backward()  # Accumulate gradients
#             if (i + 1) % accumulation_steps == 0:  # Perform optimizer step every accumulation_steps
#                 optimizer.step()
#                 optimizer.zero_grad()
        if perloss:
            loss2 = criterion[1](out,y)
            loss = loss + 0.04*loss2

        loss.backward()

        optim.step()
        optim.zero_grad()
        losses.append(loss.item())
        print(f'\rtrain loss: {loss.item():.5f} | step: {step}/{steps} | lr: {lr :.7f} | time_used: {(time.time()-start_time)/60 :.1f}',end='',flush=True)
        if step % eval_step ==0 :
            with torch.no_grad():
                ssim_eval, psnr_eval = test(net, loader_test, max_psnr, max_ssim, step)
            print(f'\nstep: {step} | ssim: {ssim_eval:.4f} | psnr: {psnr_eval:.4f}')

            ssims.append(ssim_eval)
            psnrs.append(psnr_eval)
            if ssim_eval > max_ssim and psnr_eval > max_psnr :
                max_ssim = max(max_ssim,ssim_eval)
                max_psnr = max(max_psnr,psnr_eval)
                torch.save({
                            'step': step,
                            'max_psnr': max_psnr,
                            'max_ssim': max_ssim,
                            'ssims': ssims,
                            'psnrs': psnrs,
                            'losses': losses,
                            'model': net.state_dict()
                }, model_dir)
                print(f'\n model saved at step : {step} | max_psnr: {max_psnr:.4f} | max_ssim: {max_ssim:.4f}')

    np.save(f'./numpy_files/{model_name}_{steps}_losses.npy',losses)
    np.save(f'./numpy_files/{model_name}_{steps}_ssims.npy',ssims)
    np.save(f'./numpy_files/{model_name}_{steps}_psnrs.npy',psnrs)

%%time

loader_train = loaders_[trainset]
loader_test = loaders_[testset]
net = models_[network]
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
criterion = []
criterion.append(nn.L1Loss().to(device))
if perloss:
    vgg_model = vgg16(pretrained=True).features[:16]
    vgg_model = vgg_model.to(device)
    for param in vgg_model.parameters():
        param.requires_grad = False
    criterion.append(PerLoss(vgg_model).to(device))
optimizer = optim.Adam(params = filter(lambda x: x.requires_grad, net.parameters()), lr=learning_rate, betas=(0.9,0.999), eps=1e-08)
optimizer.zero_grad()
train(net, loader_train, loader_test, optimizer, criterion)