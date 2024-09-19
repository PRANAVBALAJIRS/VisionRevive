def tensorShow(tensors, titles=None):
    fig = plt.figure()
    for tensor, title, i in zip(tensors, titles, range(len(tensors))):
        img = make_grid(tensor)
        npimg = img.numpy()
        ax = fig.add_subplot(211 + i)
        ax.imshow(np.transpose(npimg, (1, 2, 0)))
        ax.set_title(title)
    plt.show()

def lr_schedule_cosdecay(t, T, init_lr=learning_rate):
    lr = 0.5 * (1 + math.cos(t * math.pi / T)) * init_lr
    return lr

