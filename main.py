import torch
import time
import numpy as np
from tqdm import tqdm
import math
from torch.optim import lr_scheduler

import get_data
import encoder
import encoder_disout
import arg_parser
import model_utils
import os



def load_model_and_optimizer(opt, num_GPU=None):
    resnet = encoder.ResNet(opt)
    optimizer = torch.optim.Adam(resnet.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
    resnet, num_GPU = model_utils.distribute_over_GPUs(opt, resnet, num_GPU=num_GPU)
    return resnet, optimizer



def train(opt, resnet):
    total_step = len(train_loader)

    starttime = time.time()
    print_idx = 100
    best_loss = 100
    print_loss = 100

    for epoch in range(opt.start_epoch, opt.num_epochs):
        for step, (img1, img2, target) in enumerate(train_loader):
            size = img1.size()[0]
            x_t1 = img1.to(opt.device)
            x_t2 = img2.to(opt.device)

            if step % print_idx == 0:
                print(
                    "Epoch [{}/{}], Step [{}/{}], Time (s): {:.1f}".format(
                        epoch + 1,
                        opt.num_epochs,
                        step,
                        total_step,
                        time.time() - starttime,
                    )
                )

            _, out_1 = resnet(x_t1)
            _, out_2 = resnet(x_t2)

            loss = contrast_loss(out_1, out_2, size, opt)
            resnet.zero_grad()

            loss.backward()
            optimizer.step()
            print_loss = loss.item()


        print("Epoch:[{}/{}]\t \t Loss: \t \t {:.4f}".format(epoch + 1, opt.num_epochs, print_loss))
        model_utils.logfile(opt, epoch, print_loss, optimizer.param_groups[0]['lr'])
        model_utils.save_checkpoint(opt, resnet, epoch, print_loss, best_loss)
        if print_loss < best_loss:
            best_loss = print_loss


def contrast_loss(out_1, out_2, size, opt):
    out = torch.cat([out_1, out_2], dim=0)
    n = size * 2
    loss = 0
    s = torch.matmul(out, torch.transpose(out, 0, 1)) / (
        opt.tem * torch.matmul(torch.norm(out, dim=1).unsqueeze(1), torch.norm(out, dim=1).unsqueeze(0))
    )
    a = torch.diag(s)
    l = -(s.exp() / (s.exp().sum(dim=0)-a.exp())).log()
    l1 = torch.diag(l[0: size, size: 2 * size])
    l2 = torch.diag(l[size: 2 * size, 0: size])
    loss = ((l1 + l2).sum())/n
    return loss


if __name__ == "__main__":

    opt = arg_parser.parse_args()
    opt.training_dataset = "unlabeled"

    # random seeds
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    torch.backends.cudnn.benchmark = True

    # load model
    train_loader, _, supervised_loader, _, test_loader, _ = get_data.get_dataloader(opt)
    resnet, optimizer = load_model_and_optimizer(opt)
    total = sum([param.nelement() for param in resnet.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
    if opt.reload:
        resnet.load_state_dict(torch.load(os.path.join(opt.output_dir, 'checkpoint_10.pt'))["model"])
    os.makedirs(opt.output_dir, exist_ok=True)
    train(opt, resnet)

