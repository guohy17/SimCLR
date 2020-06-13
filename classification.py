import sys
import torch
import torch.nn as nn
import numpy as np
import time
import os

import get_data
import arg_parser
import main
import utils


class ClassificationModel(torch.nn.Module):
    def __init__(self, in_channels=128, num_classes=10):
        super().__init__()
        self.in_channels = in_channels
        self.layer = nn.Linear(self.in_channels, num_classes, bias=True)

        print(self.layer)

    def forward(self, x, *args):
        x = self.layer(x).squeeze()
        return x

def train_logistic_regression(opt, resnet, classification_model, train_loader):
    total_step = len(train_loader)
    classification_model.train()

    start_time = time.time()

    for epoch in range(opt.num_epochs):
        epoch_acc1 = 0
        epoch_acc5 = 0

        loss_epoch = 0
        for step, (img, target) in enumerate(train_loader):

            classification_model.zero_grad()
            model_input = img.to(opt.device)

            with torch.no_grad():
                z, h = resnet(model_input)
            z = z.detach()

            prediction = classification_model(z)


            target = target.to(opt.device)
            loss = criterion(prediction, target)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc1, acc5 = utils.accuracy(prediction.data, target, topk=(1, 5))
            epoch_acc1 += acc1
            epoch_acc5 += acc5
            sample_loss = loss.item()
            loss_epoch += sample_loss
        print("Overall accuracy for epoch [{}/{}]: {}".format(epoch + 1, opt.num_epochs, epoch_acc1 / total_step))


def test_logistic_regression(opt, resnet, classification_model, test_loader):
    total_step = len(test_loader)
    resnet.eval()
    classification_model.eval()

    starttime = time.time()

    loss_epoch = 0
    epoch_acc1 = 0
    epoch_acc5 = 0

    for step, (img, target) in enumerate(test_loader):

        model_input = img.to(opt.device)

        with torch.no_grad():
            z, h = resnet(model_input)

        z = z.detach()

        prediction = classification_model(z)

        target = target.to(opt.device)
        loss = criterion(prediction, target)

        # calculate accuracy
        acc1, acc5 = utils.accuracy(prediction.data, target, topk=(1, 5))
        epoch_acc1 += acc1
        epoch_acc5 += acc5

        sample_loss = loss.item()
        loss_epoch += sample_loss

    print("Testing Accuracy: ", epoch_acc1 / total_step)
    return epoch_acc1 / total_step, epoch_acc5 / total_step, loss_epoch / total_step


if __name__ == "__main__":

    opt = arg_parser.parse_args()

    add_path_var = "linear_model"

    opt.training_dataset = "train"

    # random seeds
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    # load pretrained model
    resnet, _, _ = main.load_model_and_optimizer(
        opt)
    resnet.load_state_dict(
        torch.load(os.path.join(opt.output_dir, 'checkpoint_300.pt'))["model"])
    resnet.eval()

    _, _, train_loader, _, test_loader, _ = get_data.get_dataloader(opt)


    classification_model = ClassificationModel(in_channels=2048, num_classes=10).to(opt.device)
    params = classification_model.parameters()
    optimizer = torch.optim.Adam(params)
    criterion = torch.nn.CrossEntropyLoss()

    # Train the model
    train_logistic_regression(opt, resnet, classification_model, train_loader)

    # Test the model

    acc1, acc5, _ = test_logistic_regression(
        opt, resnet, classification_model, test_loader
    )
