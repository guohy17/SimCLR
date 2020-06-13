import torch
import torch.nn as nn
import os
import logging
import traceback


def distribute_over_GPUs(opt, model, num_GPU):
    ## distribute over GPUs
    if opt.device.type != "cpu":
        if num_GPU is None:
            model = nn.DataParallel(model)
            num_GPU = torch.cuda.device_count()
            opt.batch_size_multiGPU = opt.batch_size * num_GPU
        else:
            assert (
                num_GPU <= torch.cuda.device_count()
            ), "You cant use more GPUs than you have."
            model = nn.DataParallel(model, device_ids=list(range(num_GPU)))
            opt.batch_size_multiGPU = opt.batch_size * num_GPU
    else:
        model = nn.DataParallel(model)
        opt.batch_size_multiGPU = opt.batch_size

    model = model.to(opt.device)
    print("Let's use", num_GPU, "GPUs!")

    return model, num_GPU


def logfile(args, epoch, loss ,lr, rtnext=True):
    output = "Epoch:[{}/{}]\t \t Loss: \t \t {:.4f}  Lr:{}".format(epoch + 1, args.num_epochs, loss, lr)
    filename = os.path.join(args.output_dir, 'log_file.txt')
    fo = open(filename, "a")
    if rtnext:
        fo.write(output+'\n')
    else:
        fo.write(output)
    fo.close()


def save_checkpoint(args, model, epoch, loss, best_loss):
    extra_state = {
        'train_iterator': epoch,
    }
    file = 'checkpoint_{:d}.pt'.format(epoch + 1)
    checkpoints = os.path.join(args.output_dir, file)
    state_dict = {
        'model': model.state_dict(),
        'extra_state': extra_state,
    }
    if (epoch + 1) % 10 == 0:
        torch_persistent_save(state_dict, checkpoints)
    elif epoch > 100 and loss < best_loss:
        torch_persistent_save(state_dict, os.path.join(args.output_dir, 'best_model.pt'))


def torch_persistent_save(*args, **kwargs):
    for i in range(3):
        try:
            return torch.save(*args, **kwargs)
        except Exception:
            if i == 2:
                logging.error(traceback.format_exc())



