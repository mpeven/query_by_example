import signal
import numpy as np
from tqdm import tqdm
import datetime as dt
import pandas as pd

import torch
import torchvision.models as models
import torch.nn.functional as F

from datasets import JIGSAWS

TRIPLETS = False

NUM_WORKERS = 8
BATCH_SIZE = 16

HIDDEN_DIM = 512

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes = ["G1","G2","G3","G4","G5","G6","G8","G9","G10","G11","G12","G13","G14","G15"]
classes = [x + "_expert" for x in classes] + [x + "_novice" for x in classes]



# Handle ctrl+c gracefully
signal.signal(signal.SIGINT, lambda signum, frame: exit(0))




class Model(torch.nn.Module):
    def __init__(self, triplet_loss=False):
        super(Model, self).__init__()

        self.triplet_loss = triplet_loss

        # Layers
        self.base_model = torch.nn.Sequential(*list(models.resnet34(pretrained=True).children())[:-1])
        self.lstmlayer = torch.nn.GRU(
            input_size  = list(self.base_model.parameters())[-1].size(0),
            hidden_size = HIDDEN_DIM,
            num_layers = 4,
        )
        self.preds = torch.nn.Linear(HIDDEN_DIM, len(classes))

        # Freeze weights
        net_size = len(list(self.base_model.parameters()))
        for idx, param in enumerate(list(self.base_model.parameters())[:int(net_size - 54)]):
            param.requires_grad = False


    def forward(self, ims, pos_ims=None, neg_ims=None):
        results = []
        vids = [ims,] if pos_ims is None and neg_ims is None else [ims, pos_ims, neg_ims]

        for vid in vids:
            base_model_out = [self.base_model(vid[:,i]) for i in range(vid.size()[1])]
            x = torch.stack(base_model_out)
            x = x.view(x.size(0), x.size(1), int(np.prod(x.size()[2:]))) # Flatten
            lstm_out,_ = self.lstmlayer(x)
            if self.triplet_loss:
                results.append(lstm_out[-1])
            else:
                results.append(self.preds(lstm_out[-1]))

        return results




def train_epoch(phase, net, optimizer, epoch, dataloader):
    ''' Training/Testing/Validation epoch '''

    # Loss funtion
    if TRIPLETS:
        loss_func = torch.nn.TripletMarginLoss().to(DEVICE)
    else:
        loss_func = torch.nn.CrossEntropyLoss().to(DEVICE)

    # Single pass through training data
    accuracy = 0
    running_losses = []
    running_labels = []
    running_predict = []
    output_dicts = []

    # Create the dataloading iterator
    stat_dict = {"Epoch": epoch}
    iterator = tqdm(dataloader, postfix=stat_dict, ncols=100, desc=phase)

    # Loop over the data
    for i, data in enumerate(iterator):
        # Forward pass, calculate loss, backward pass
        optimizer.zero_grad()
        if TRIPLETS == False:
            outputs = net(data["images"].to(DEVICE))[0]
            loss = loss_func(outputs, data["gesture_id"].to(DEVICE))
        else:
            outputs = net(data["images"].to(DEVICE), data["pos_images"].to(DEVICE), data["neg_images"].to(DEVICE))
            loss = loss_func(*outputs)


        # Backward Pass
        if phase == 'train':
            loss.backward()
            optimizer.step()

        # Update loss
        running_losses.append(loss.item())
        stat_dict['Loss'] = "{:.5f}".format(np.mean(running_losses))

        # No predictions if triplet loss
        if TRIPLETS:
            iterator.set_postfix(stat_dict)
            continue

        # Update labels and predictions
        running_labels.append(np.array(data["gesture_id"], dtype=int))
        prediction = outputs.cpu().detach().numpy()
        running_predict.append(np.argmax(prediction, 1))

        # Update accuracy
        accuracy = 100.0 * np.mean(np.equal(np.concatenate(running_labels), np.concatenate(running_predict)))
        stat_dict['Acc'] = "{:.4f}".format(accuracy)
        iterator.set_postfix(stat_dict)

        if phase == 'test':
            for batch in range(len(data["gesture_skill"])):
                output_dicts.append({
                    "video": data["video"][batch],
                    "frame_start": int(data["frame_start"][batch]),
                    "ground_truth": data["gesture_skill"][batch],
                    "prediction": classes[np.argmax(prediction, 1)[batch]],
                    "skill": data['skill'][batch],
                })

    # Return the accuracy
    return accuracy, output_dicts





def test_query(phase, net, train_dataloader, test_dataloader):
    # Build table of train data representations and their labels
    train_reps = []
    train_labels = []
    for i, data in enumerate(tqdm(train_dataloader, ncols=100, desc='Building query dictionary')):
        outputs = net(data['images'].to(DEVICE))[0].cpu().detach()
        train_reps.append(outputs)
        train_labels += [int(x.numpy()) for x in data['gesture_id']]
    train_reps = torch.cat(train_reps)


    # Find closest match for all test/val data
    is_equal = []
    output_dicts = []
    pdist = torch.nn.PairwiseDistance()
    for data in tqdm(test_dataloader, ncols=100, desc='Building {} dictionary'.format(phase)):
        outputs = net(data['images'].to(DEVICE))[0].cpu().detach()

        # Get distance to all representations in the query dictionary
        all_dists = [pdist(test_rep.expand(train_reps.size(0), -1), train_reps) for test_rep in outputs]
        all_dists = torch.stack(all_dists).numpy()

        # Find label of closest representation
        closest_train_vid_label = [int(train_labels[x]) for x in np.argmin(all_dists, axis=1)]
        test_labels = [int(x.numpy()) for x in data['gesture_id']]
        is_equal += [closest_train_vid_label[i] == test_labels[i] for i in range(len(test_labels))]

        # Build output dictionary
        for batch in range(len(data['gesture_skill'])):
            output_dicts.append({
                "video": data["video"][batch],
                "frame_start": int(data["frame_start"][batch]),
                "ground_truth": data['gesture_skill'][batch],
                "prediction": classes[closest_train_vid_label[batch]],
                "skill": data['skill'][batch],
            })

    return 100.0 * np.mean(is_equal), output_dicts



def main(experiment=0):
    net = Model().to(DEVICE)
    net.load_state_dict(torch.load('models/torch_model_07_12_18_11_40___61_9'))
    net = net.to(DEVICE)

    # Get dataloaders
    train_loader_trips = torch.utils.data.DataLoader(JIGSAWS("train", experiment, return_triplets=True), BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    train_loader = torch.utils.data.DataLoader(JIGSAWS("train", experiment), BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    valid_loader = torch.utils.data.DataLoader(JIGSAWS("val", experiment), BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = torch.utils.data.DataLoader(JIGSAWS("test", experiment), BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # Set up optimizer with auto-adjusting learning rate
    parameters = [p for p in net.parameters() if p.requires_grad]
    # optimizer = torch.optim.Adam(parameters, lr=0.003)
    optimizer = torch.optim.SGD(parameters, lr = 0.01, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Train + Validate
    best_valid_accuracy = 0
    best_valid_test_accuracy = 0
    best_test_accuracy = 0
    for epoch in range(100):
        # Train
        if epoch > 0:
            net.train(True)
            torch.set_grad_enabled(True)
            if TRIPLETS:
                train_acc, _ = train_epoch('train', net, optimizer, epoch, train_loader_trips)
            else:
                train_acc, _ = train_epoch('train', net, optimizer, epoch, train_loader)

        # Validate
        net.eval()
        with torch.no_grad():
            if TRIPLETS:
                valid_acc, _ = test_query('valid', net, train_loader, valid_loader)
                test_acc, outputs = test_query('test', net, train_loader, test_loader)
            else:
                valid_acc, _ = train_epoch('valid', net, optimizer, epoch, valid_loader)
                test_acc, outputs = train_epoch('test', net, optimizer, epoch, test_loader)
            print('Epoch {:02} valid-set accuracy: {:.1f}%   (best = ({:.1f}%, {:.1f}%))'.format(epoch, valid_acc, best_valid_accuracy, best_valid_test_accuracy))
            print('Epoch {:02} test-set accuracy: {:.1f}%   (best = {:.1f}%)'.format(epoch, test_acc, best_test_accuracy))

            if valid_acc > best_valid_accuracy:
                print('Best validation accuracy so far, saving model state and test outputs')
                torch.save(net.state_dict(), 'models/torch_model_{}___{}'.format(
                    dt.datetime.now().strftime("%D_%H_%M").replace("/", "_"),
                    "{:.1f}".format(valid_acc).replace(".", "_")
                ))
                pd.DataFrame(outputs).to_csv("outputs.csv", index=False)

                best_valid_accuracy = valid_acc
                best_valid_test_accuracy = test_acc

            best_test_accuracy = test_acc if test_acc > best_test_accuracy else best_test_accuracy
        scheduler.step()



def run_all_experiments():
    for experiment in range(8):
        net = Model(triplet_loss=False).to(DEVICE)

        # Get dataloaders
        train_loader = torch.utils.data.DataLoader(JIGSAWS("train", experiment), BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        valid_loader = torch.utils.data.DataLoader(JIGSAWS("val", experiment), BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        test_loader = torch.utils.data.DataLoader(JIGSAWS("test", experiment), BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

        # Set up optimizer with auto-adjusting learning rate
        parameters = [p for p in net.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(parameters, lr = 0.01, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        # Train + Validate
        best_valid_accuracy = 0
        for epoch in range(100):
            net.train(True)
            torch.set_grad_enabled(True)
            train_acc, _ = train_epoch('train', net, optimizer, epoch, train_loader)

            # Validate
            net.eval()
            with torch.no_grad():
                valid_acc, _ = test_query('valid', net, valid_loader, test_loader)
                print('Epoch {:02} valid-set accuracy: {:.1f}%   (best = {:.1f}%)'.format(epoch, valid_acc, best_valid_accuracy))
                if valid_acc > best_valid_accuracy:
                    print('Best validation accuracy so far, saving model state and test outputs')
                    torch.save(net.state_dict(), 'models/torch_model_{:02}'.format(experiment))

                    best_valid_accuracy = valid_acc
            scheduler.step()

        # Test
        net = Model(triplet_loss=True).to(DEVICE)
        net.load_state_dict(torch.load('models/torch_model_{:02}'.format(experiment)))
        net.eval()
        with torch.no_grad():
            test_acc, outputs = test_query('test', net, train_loader, test_loader)
            print('Experiment {:02} test-set accuracy: {:.1f}%'.format(experiment, test_acc))
            pd.DataFrame(outputs).to_csv("outputs_{:02}.csv".format(experiment), index=False)


if __name__ == '__main__':
    run_all_experiments()
