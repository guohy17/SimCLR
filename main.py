import torch
import time
import numpy as np
from tqdm import tqdm
import math
from torch.optim import lr_scheduler

# import logger
import get_data
import encoder
import encoder_disout
import arg_parser
import model_utils
import os



def load_model_and_optimizer(opt, num_GPU=None):
    resnet = encoder.ResNet(opt)
    # resnet = encoder_disout.ResNet(opt)
    optimizer = torch.optim.Adam(resnet.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
    schedule = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
    # base_optimizer = torch.optim.SGD(resnet.parameters(), lr=0.1)
    # optimizer = torchlars.LARS(optimizer=base_optimizer, eps=1e-8, trust_coef=0.001)
    # optimizer = lars_opt.LARS(
    #     params=resnet.parameters(), lr=0.5, weight_decay=opt.weight_decay, max_epoch=opt.num_epochs
    # )
    resnet, num_GPU = model_utils.distribute_over_GPUs(opt, resnet, num_GPU=num_GPU)
    return resnet, optimizer, schedule



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

            # loss, _, _ = resnet(x_t1, x_t2)
            # loss = resnet(x_t1, x_t2)
            resnet.zero_grad()

            loss.backward()
            optimizer.step()
            print_loss = loss.item()
            # schedule.step(print_loss)

        print("Epoch:[{}/{}]\t \t Loss: \t \t {:.4f}".format(epoch + 1, opt.num_epochs, print_loss))
        model_utils.logfile(opt, epoch, print_loss, optimizer.param_groups[0]['lr'])
        model_utils.save_checkpoint(opt, resnet, epoch, print_loss, best_loss)
        if print_loss < best_loss:
            best_loss = print_loss
        # if (epoch + 1) % 10 == 0:
        #     torch.save(resnet.state_dict(),
        #                '/lustre/home/hyguo/code/code/SimCLR/models/models_0513/model-{}.pth'.format(epoch+1))



def contrast_loss(out_1, out_2, size, opt):
    out = torch.cat([out_1, out_2], dim=0)
    n = size * 2
    loss = 0
    s = torch.matmul(out, torch.transpose(out, 0, 1)) / (
        opt.tem * torch.matmul(torch.norm(out, dim=1).unsqueeze(1), torch.norm(out, dim=1).unsqueeze(0))
    )
    a = torch.diag(s)
    l = -(s.exp() / (s.exp().sum(dim=0)-a.exp())).log()
    # for k in range(size):
    #     l1 = -((s[2 * k - 1, 2 * k].exp() / (s[2 * k - 1, :].exp().sum()-s[2 * k, 2 * k].exp())).log())
    #     l2 = -((s[2 * k, 2 * k - 1].exp() / (s[2 * k, :].exp().sum() - s[2 * k, 2 * k].exp())).log())
    #     loss += l1 + l2
    # for k in range(size):
    #     loss += l[2 * k - 1, 2 * k] + l[2 * k, 2 * k - 1]
    l1 = torch.diag(l[0: size, size: 2 * size])
    l2 = torch.diag(l[size: 2 * size, 0: size])
    # for k in range(size):
    #     loss += l[k, k + size] + l[k + size, k]
    loss = ((l1 + l2).sum())/n
    return loss


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.learning_rate
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.num_epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def test(net, train_loader, test_loader):
    net.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for data, target in tqdm(train_loader, desc='Feature extracting'):
            feature, out = net(data.cuda(non_blocking=True))
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(train_loader.dataset.targets, device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_loader)
        for data, target in test_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature, out = net(data)

            total_num += data.size(0)
            # compute cos similarity between each feature vector and feature bank ---> [B, N]
            sim_matrix = torch.mm(feature, feature_bank)
            # [B, K]
            sim_weight, sim_indices = sim_matrix.topk(k=200, dim=-1)
            # [B, K]
            sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
            sim_weight = (sim_weight / opt.tem).exp()

            # counts for each class
            one_hot_label = torch.zeros(data.size(0) * 200, 10, device=sim_labels.device)
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
            # weighted score ---> [B, C]
            pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, 10) * sim_weight.unsqueeze(dim=-1), dim=1)

            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            test_bar.set_description('Acc@1:{:.2f}% Acc@5:{:.2f}%'
                                     .format( total_top1 / total_num * 100, total_top5 / total_num * 100))

    return total_top1 / total_num * 100, total_top5 / total_num * 100


if __name__ == "__main__":

    opt = arg_parser.parse_args()
    # arg_parser.create_log_path(opt)
    opt.training_dataset = "unlabeled"

    # random seeds
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    torch.backends.cudnn.benchmark = True

    # load model
    train_loader, _, supervised_loader, _, test_loader, _ = get_data.get_dataloader(opt)
    resnet, optimizer, schedule = load_model_and_optimizer(opt)
    total = sum([param.nelement() for param in resnet.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
    if opt.reload:
        resnet.load_state_dict(torch.load(os.path.join(opt.output_dir, 'checkpoint_10.pt'))["model"])
    os.makedirs(opt.output_dir, exist_ok=True)
    train(opt, resnet)

