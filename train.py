import signal
import numpy as np
from tqdm import tqdm
import datetime as dt

import torch
import torchvision.models as models
import torch.nn.functional as F

from datasets import JIGSAWS



BATCH_SIZE = 16
NUM_WORKERS = 8

HIDDEN_DIM = 1024

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Handle ctrl+c gracefully
signal.signal(signal.SIGINT, lambda signum, frame: exit(0))



class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        # Layers
        self.base_model = torch.nn.Sequential(*list(models.resnet34(pretrained=True).children())[:-1])
        self.lstmlayer = torch.nn.LSTM(
            input_size  = list(self.base_model.parameters())[-1].size(0),
            hidden_size = HIDDEN_DIM,
        )

        # Freeze weights
        net_size = len(list(self.base_model.parameters()))
        for idx, param in enumerate(list(self.base_model.parameters())[:int(net_size - 3)]):
            param.requires_grad = False

    def forward(self, anchor, positive=None, negative=None):
        results = []
        vids = [anchor,] if positive is None and negative is None else [anchor, positive, negative]
        for vid in vids:
            base_model_out = [self.base_model(vid[:,i]) for i in range(vid.size()[1])]
            x = torch.stack(base_model_out)
            x = x.view(x.size(0), x.size(1), int(np.prod(x.size()[2:]))) # Flatten
            lstm_out,_ = self.lstmlayer(x)
            results.append(lstm_out[-1])

        return results




def train_epoch(net, optimizer, epoch, dataloader):
    # Loss funtion
    loss_func = torch.nn.TripletMarginLoss().to(DEVICE)

    # Create the dataloading iterator
    running_losses = []
    stat_dict = {"Epoch": epoch}
    iterator = tqdm(dataloader, postfix=stat_dict, ncols=100, desc='Training')

    # Loop over the data
    for data in iterator:
        # Forward pass, calculate loss, backward pass
        optimizer.zero_grad()
        outputs = net(data[0].to(DEVICE), data[1].to(DEVICE), data[2].to(DEVICE))
        loss = loss_func(*outputs)

        # Backward Pass
        loss.backward()
        optimizer.step()

        # Update loss
        running_losses.append(loss.item())
        stat_dict['Loss'] = "{:.5f}".format(np.mean(running_losses))
        iterator.set_postfix(stat_dict)

    # Return the loss
    return np.mean(running_losses)




def test_epoch(phase, net, train_dataloader, test_dataloader):
    # Build table of train data representations and their labels
    train_reps = []
    train_labels = []
    for i, data in enumerate(tqdm(train_dataloader, ncols=100, desc='Building query dictionary')):
        outputs = net(data[0].to(DEVICE))[0].cpu().detach()
        train_reps.append(outputs)
        train_labels += [int(x.numpy()) for x in data[1]]

    # Find closest match for all test/val data
    test_reps = []
    test_labels = []
    for data in tqdm(test_dataloader, ncols=100, desc='Building {} dictionary'.format(phase)):
        outputs = net(data[0].to(DEVICE))[0].cpu().detach()
        test_reps.append(outputs)
        test_labels += [int(x.numpy()) for x in data[1]]

    # Calculate accuracy
    train_reps = torch.cat(train_reps)
    test_reps = torch.cat(test_reps)

    # Get all pairwise distances
    pdist = torch.nn.PairwiseDistance()
    all_dists = [pdist(test_rep.expand(train_reps.size(0), -1), train_reps) for test_rep in test_reps]
    all_dists = torch.stack(all_dists).numpy()

    # Find minimum distance for each testing example
    closest_train_vid = np.argmin(all_dists, axis=1)
    closest_train_vid_label = [int(train_labels[x]) for x in closest_train_vid]
    is_equal = [closest_train_vid_label[i] == test_labels[i] for i in range(len(test_labels))]
    return 100.0 * np.mean(is_equal)


def main():
    net = Model().to(DEVICE)

    # Get dataloaders
    train_triplet_loader = torch.utils.data.DataLoader(JIGSAWS("train", return_triplets=True), BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    train_loader = torch.utils.data.DataLoader(JIGSAWS("train"), BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    valid_loader = torch.utils.data.DataLoader(JIGSAWS("val"), BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = torch.utils.data.DataLoader(JIGSAWS("test"), BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # Set up optimizer with auto-adjusting learning rate
    parameters = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=0.002)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.75)

    # Train + Validate
    best_valid_accuracy = 0
    best_test_accuracy = 0
    for epoch in range(1000):
        # Train
        net.train(True)
        torch.set_grad_enabled(True)
        train_acc = train_epoch(net, optimizer, epoch, train_triplet_loader)

        # Validate
        net.eval()
        with torch.no_grad():
            valid_acc = test_epoch('valid', net, train_loader, valid_loader)
            test_acc = test_epoch('test', net, train_loader, test_loader)
            print('Epoch {:02} valid-set accuracy: {:.1f}%   (best = {:.1f}%)'.format(epoch, valid_acc, best_valid_accuracy))
            print('Epoch {:02} test-set accuracy: {:.1f}%   (best = {:.1f}%)'.format(epoch, test_acc, best_test_accuracy))

            # if valid_acc > best_valid_accuracy:
            #     print('Best validation accuracy so far, saving model state')
            #     torch.save(net.state_dict(), 'models/torch_model_{}___{}'.format(
            #         dt.datetime.now().strftime("%D_%H_%M").replace("/", "_"),
            #         "{:.1f}".format(valid_acc).replace(".", "_")
            #     ))

            best_valid_accuracy = valid_acc if valid_acc > best_valid_accuracy else best_valid_accuracy
            best_test_accuracy = test_acc if test_acc > best_test_accuracy else best_test_accuracy
        scheduler.step()



if __name__ == '__main__':
    main()
