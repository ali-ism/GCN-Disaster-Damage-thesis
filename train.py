import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import RandomNodeSampler
from tqdm import tqdm
from dataset import xBD
from model import GCN, DeeperGCN, CNNGCN
from metrics import score
from utils import get_class_weights

with open('exp_settings.json', 'r') as JSON:
    settings_dict = json.load(JSON)

batch_size = settings_dict['data']['batch_size']
delaunay = settings_dict['data']['delaunay']
name = settings_dict['model']['name']
train_disaster = settings_dict['data']['train_disasters']
train_paths = settings_dict['data']['train_paths']
train_root = settings_dict['data']['train_root']
test_root = "/home/ami31/scratch/datasets/delaunay/socal_test"
hold_root = "/home/ami31/scratch/datasets/delaunay/socal_hold"
edge_features = settings_dict['model']['edge_features']
n_epochs = settings_dict['epochs']
starting_epoch = settings_dict['starting_epoch']
path = settings_dict['model']['path']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def train(epoch):
    model.train()
    pbar = tqdm(total=len(train_dataset))
    pbar.set_description(f'Epoch {epoch:02d}')
    total_loss = 0
    total_examples = 0
    for data in train_dataset:
        if data.num_nodes > batch_size:
            sampler = RandomNodeSampler(data, num_parts=data.num_nodes//batch_size, num_workers=2)
            for subdata in sampler:
                subdata = subdata.to(device)
                optimizer.zero_grad()
                if edge_features:
                    out = model(subdata.x, subdata.edge_index, subdata.edge_attr)
                else:
                    out = model(subdata.x, subdata.edge_index)
                loss = F.nll_loss(input=out, target=subdata.y, weight=class_weights.to(device))
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * subdata.num_nodes
                total_examples += subdata.num_nodes
        else:
            data = data.to(device)
            optimizer.zero_grad()
            if edge_features:
                out = model(data.x, data.edge_index, data.edge_attr)
            else:
                out = model(data.x, data.edge_index)
            loss = F.nll_loss(input=out, target=data.y, weight=class_weights.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.num_nodes
            total_examples += data.num_nodes
        pbar.update()
    pbar.close()
    return total_loss / total_examples


@torch.no_grad()
def test(dataset):
    model.eval()
    y_true = []
    y_pred = []
    for data in dataset:
        if data.num_nodes > batch_size:
            sampler = RandomNodeSampler(data, num_parts=data.num_nodes//batch_size, num_workers=2)
            for subdata in sampler:
                subdata = subdata.to(device)
                if edge_features:
                    y_pred.append(model(subdata.x, subdata.edge_index, subdata.edge_attr).cpu())
                else:
                    y_pred.append(model(subdata.x, subdata.edge_index).cpu())
                y_true.append(subdata.y.cpu())
        else:
            data = data.to(device)
            if edge_features:
                y_pred.append(model(data.x, data.edge_index, data.edge_attr).cpu())
            else:
                y_pred.append(model(data.x, data.edge_index).cpu())
            y_true.append(data.y.cpu())
    y_pred = torch.cat(y_pred)
    y_true = torch.cat(y_true)
    accuracy, f1_macro, f1_weighted, auc = score(y_true, y_pred)
    if dataset is not train_dataset:
        loss = F.nll_loss(input=y_pred, target=y_true, weight=class_weights)
    else:
        loss = None
    return accuracy, f1_macro, f1_weighted, auc, loss


def save_results(hold=False) -> None:
    plt.figure()
    plt.plot(train_loss)
    plt.plot(test_loss)
    plt.legend(['train', 'test'])
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.savefig('results/'+name+'_loss.pdf')
    plt.figure()
    plt.plot(train_acc)
    plt.plot(test_acc)
    plt.legend(['train', 'test'])
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.savefig('results/'+name+'_acc.pdf')
    plt.figure()
    plt.plot(train_f1_macro)
    plt.plot(test_f1_macro)
    plt.legend(['train', 'test'])
    plt.xlabel('epochs')
    plt.ylabel('macro f1')
    plt.savefig('results/'+name+'_macro_f1.pdf')
    plt.figure()
    plt.plot(train_f1_weighted)
    plt.plot(test_f1_weighted)
    plt.legend(['train', 'test'])
    plt.xlabel('epochs')
    plt.ylabel('weighted f1')
    plt.savefig('results/'+name+'_weighted_f1.pdf')
    plt.figure()
    plt.plot(train_auc)
    plt.plot(test_auc)
    plt.legend(['train', 'test'])
    plt.xlabel('epochs')
    plt.ylabel('auc')
    plt.savefig('results/'+name+'_auc.pdf')
    plt.close('all')
    np.save('results/'+name+'_loss_train.npy', train_loss)
    np.save('results/'+name+'_loss_test.npy', test_loss)
    np.save('results/'+name+'_acc_train.npy', train_acc)
    np.save('results/'+name+'_acc_test.npy', test_acc)
    np.save('results/'+name+'_macro_f1_train.npy', train_f1_macro)
    np.save('results/'+name+'_macro_f1_test.npy', test_f1_macro)
    np.save('results/'+name+'_weighted_f1_train.npy', train_f1_weighted)
    np.save('results/'+name+'_weighted_f1_test.npy', test_f1_weighted)
    np.save('results/'+name+'_auc_train.npy', train_auc)
    np.save('results/'+name+'_auc_test.npy', test_auc)
    if hold:
        hold_dataset = xBD(hold_root, ['/home/ami31/scratch/datasets/xbd/hold_bldgs/'], ['socal-fire'])
        hold_scores = test(hold_dataset)
        print('\nHold results for last model.')
        print(f'Hold accuracy: {hold_scores[0]:.4f}')
        print(f'Hold macro F1: {hold_scores[1]:.4f}')
        print(f'Hold weighted F1: {hold_scores[2]:.4f}')
        print(f'Hold auc: {hold_scores[3]:.4f}')
        model_path = path + '/' + name + '_best.pt'
        model.load_state_dict(torch.load(model_path))
        hold_scores = test(hold_dataset)
        print('\nHold results for best model.')
        print(f'Hold accuracy: {hold_scores[0]:.4f}')
        print(f'Hold macro F1: {hold_scores[1]:.4f}')
        print(f'Hold weighted F1: {hold_scores[2]:.4f}')
        print(f'Hold auc: {hold_scores[3]:.4f}')


if __name__ == "__main__":

    train_dataset = xBD(train_root, train_paths, train_disaster).shuffle()
    test_dataset = xBD(test_root, ['/home/ami31/scratch/datasets/xbd/test_bldgs/'], ['socal-fire'])

    class_weights = get_class_weights(train_disaster, train_dataset)

    if settings_dict['model']['type'] == 'gcn':
        model = GCN(train_dataset.num_node_features,
                    settings_dict['model']['hidden_units'],
                    train_dataset.num_classes,
                    settings_dict['model']['num_layers'],
                    settings_dict['model']['dropout_rate'],
                    settings_dict['model']['fc_output'])
    elif settings_dict['model']['type'] == 'deepgcn':
        model = DeeperGCN(train_dataset.num_node_features,
                        train_dataset.num_edge_features,
                        settings_dict['model']['hidden_units'],
                        train_dataset.num_classes,
                        settings_dict['model']['num_layers'],
                        settings_dict['model']['dropout_rate'],
                        settings_dict['model']['msg_norm'])
    elif settings_dict['model']['type'] == 'cnngcn':
        model = CNNGCN(settings_dict['model']['hidden_units'],
                       train_dataset.num_classes,
                       settings_dict['model']['num_layers'],
                       settings_dict['model']['dropout_rate'],
                       settings_dict['model']['fc_output'])
    if starting_epoch != 1:
        model_path = path + '/' + name + '_best.pt'
        model.load_state_dict(torch.load(model_path))
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=settings_dict['model']['lr'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=0.00001, verbose=True)

    best_test_auc = best_epoch = 0

    if starting_epoch == 1:
        train_loss = np.empty(n_epochs)
        test_loss = np.empty(n_epochs)
        train_acc = np.empty(n_epochs)
        test_acc = np.empty(n_epochs)
        train_f1_macro = np.empty(n_epochs)
        test_f1_macro = np.empty(n_epochs)
        train_f1_weighted = np.empty(n_epochs)
        test_f1_weighted = np.empty(n_epochs)
        train_auc = np.empty(n_epochs)
        test_auc = np.empty(n_epochs)
    else:
        train_loss = np.load('results/'+name+'_loss_train.npy')
        test_loss = np.load('results/'+name+'_loss_test.npy')
        train_acc = np.load('results/'+name+'_acc_train.npy')
        test_acc = np.load('results/'+name+'_acc_test.npy')
        train_f1_macro = np.load('results/'+name+'_macro_f1_train.npy')
        test_f1_macro = np.load('results/'+name+'_macro_f1_test.npy')
        train_f1_weighted = np.load('results/'+name+'_weighted_f1_train.npy')
        test_f1_weighted = np.load('results/'+name+'_weighted_f1_test.npy')
        train_auc = np.load('results/'+name+'_auc_train.npy')
        test_auc = np.load('results/'+name+'_auc_test.npy')

    for epoch in range(starting_epoch, n_epochs+1):
        
        train_loss[epoch-1] = train(epoch)
        print('**********************************************')
        print(f'Epoch {epoch:02d}, Train Loss: {train_loss[epoch-1]:.4f}')
    
        if not settings_dict['save_best_only']:
            model_path = path + '/' + name + '.pt'
            torch.save(model.state_dict(), model_path)

        train_acc[epoch-1], train_f1_macro[epoch-1], train_f1_weighted[epoch-1], train_auc[epoch-1], _ = test(train_dataset)
        test_acc[epoch-1], test_f1_macro[epoch-1], test_f1_weighted[epoch-1], test_auc[epoch-1], test_loss[epoch-1] = test(test_dataset)
        scheduler.step(test_loss[epoch-1])

        if test_auc[epoch-1] > best_test_auc:
            best_test_auc = test_auc[epoch-1]
            best_epoch = epoch
            model_path = path + '/' + name + '_best.pt'
            print(f'New best model saved with AUC {best_test_auc} at epoch {best_epoch}.')
            torch.save(model.state_dict(), model_path)
        
        if not (epoch % 5):
            save_results()
    
    print(f'\nBest test AUC {best_test_auc} at epoch {best_epoch}.\n')
    save_results(hold=True)