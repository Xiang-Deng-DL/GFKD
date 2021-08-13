import argparse, time, random, os
import numpy as np
import torch
import torch.nn as nn

from model.GIN import GIN_dict
from model.GCN import GCN_dict
from Temp.dataset import GINDataset
from Temp.fake_dataset import FAKEGINDataset
from utils.GIN.data_loader import GraphDataLoader, collate
from utils.GIN.full_loader import GraphFullDataLoader
from utils.scheduler import LinearSchedule
from distiller_zoo import DistillKL

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    if args.gpu >= 0:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def evaluate(model, dataloader, loss_fcn):
    model.eval()

    total = 0
    total_loss = 0
    total_correct = 0
    with torch.no_grad():
        for data in dataloader:
            graphs, labels = data
            feat = graphs.ndata['attr'].cuda()
            labels = labels.cuda()
            total += len(labels)
            outputs = model(graphs, feat)
            _, predicted = torch.max(outputs.data, 1)

            total_correct += (predicted == labels.data).sum().item()
            loss = loss_fcn(outputs, labels)

            total_loss += loss * len(labels)

    loss, acc = 1.0 * total_loss / total, 1.0 * total_correct / total
    return loss, acc

def task_data(args, fake_path):

    # step 0: setting for gpu
    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)
        
    # step 1: prepare dataset
    
    dataset = GINDataset(args.dataset, args.self_loop, args.degree_as_label)
    
    fakedataset = FAKEGINDataset(fake_path, args.fake_dataset, args.self_loop, args.degree_as_label)
    
    print(dataset.dim_nfeats)

    print(fakedataset.dim_nfeats)
   
    # step 2: prepare data_loader
    _, valid_loader = GraphDataLoader(
        dataset, batch_size=args.batch_size, device=args.gpu,
        collate_fn=collate, seed=args.dataseed, shuffle=True,
        split_name=args.split_name).train_valid_loader()


    train_loader = GraphFullDataLoader(
            fakedataset, batch_size=args.batch_size, device=args.gpu).train_loader()

    return dataset, train_loader, valid_loader



def task_model(args, dataset, path):

    #  step 1: prepare model
    assert args.tmodel in ['GIN', 'GCN']
    assert args.smodel in ['GIN', 'GCN']
    
    if args.tmodel == 'GIN':
        modelt = GIN_dict[args.modelt](dataset)
    elif args.tmodel == 'GCN':
        modelt = GCN_dict[args.modelt](dataset)
    else:
        raise('Not supporting such model!')

    modelt.load_state_dict(torch.load(path)['model'])

    if args.smodel == 'GIN':
        models = GIN_dict[args.models](dataset)
    elif args.smodel == 'GCN':
        models = GCN_dict[args.models](dataset)
    else:
        raise('Not supporting such model!')


    # step 2: prepare loss
    cross_ent = nn.CrossEntropyLoss()
    
    kl_div = DistillKL(args.kd_T)

        
    if args.gpu >= 0:
        modelt = modelt.cuda()
        models = models.cuda()
        cross_ent = cross_ent.cuda()
        kl_div = kl_div.cuda()

    # step 3: prepare optimizer

    optimizer = torch.optim.Adam(models.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    return modelt, models, cross_ent, kl_div, optimizer

def train_kd(args, train_loader, valid_loader, modelt, models, cross_ent, kl_div, optimizer):

    scheduler = LinearSchedule(optimizer, args.epoch)

    dur = []
    best_acc = 0
    

    
    model_name = './students/T:{}_S:{}_Datset:{}'.format(args.modelt, args.models, args.dataset)
    if not os.path.isdir(model_name):
        os.makedirs(model_name)
        

    for epoch in range(1, args.epoch+1):
        
        models.train()
        
        modelt.eval()
        
        t0 = time.time()

        for graphs, labels in train_loader:
            labels = labels.cuda()
            features = graphs.ndata['attr'].cuda()
            
            outputs_s = models(graphs, features)
            
            with torch.no_grad():
                outputs_t = modelt(graphs, features)
            
            optimizer.zero_grad()

            loss_div = kl_div(outputs_s, outputs_t)
            
            loss = loss_div
            
            
            loss.backward()

            optimizer.step()


        dur.append(time.time() - t0)

        

        _, valid_acc = evaluate(models, valid_loader, cross_ent)
        _, train_acc = evaluate(models, train_loader, cross_ent)
        
        
        print('Average Epoch Time {:.4f}'.format(float(sum(dur)/len(dur))))
        print('Epoch: %d' %epoch)
        print('Test acc {:.4f}'.format(float(valid_acc)))
        print('Traing_loss {:.4f}'.format(float(loss.item())))
        
        if valid_acc > best_acc:
            best_acc = valid_acc
            state = {
                    'epoch': epoch,
                    'model': models.state_dict(),
                    'accuracy': valid_acc,
                    }
            save_file = os.path.join(model_name, 'best_{}.pth'.format(args.dataseed))
            print('saving the best model!')
            torch.save(state, save_file)



        scheduler.step()
    

    save_file = os.path.join(model_name, 'ckpt_epoch_last_{}.pth'.format(args.dataseed))
    state = {
            'epoch': epoch,
            'model': models.state_dict(),
            'accuracy': valid_acc,
            }
    torch.save(state, save_file)
    print('last_acc: %f' %valid_acc)
    print('best_acc: %f' %best_acc)
    
    return valid_acc, best_acc

def main(args):
    
    onehot_cof = 1e-6
    bn_reg_scale = 1e-2
                        
    print('------------------------------------')
                    
    print('onehot_cof: %.15f, bn_reg_scale: %.15f' %(onehot_cof, bn_reg_scale) )
                    
    fake_path1 = args.fake_path + '_bn' + str(bn_reg_scale) + '_oh' + str(onehot_cof)
                        
    fake_path = fake_path1 + '/' + args.modelt + 'fake_mutag' + str(args.dataseed) + '_' + str(args.trial) + '.txt'
    
    dataset, train_loader, valid_loader = task_data(args, fake_path)
            
    path_t = args.path_t+'/ckpt_best_'+str(args.dataseed)+'.pth'
                    
    modelt, models, cross_ent, kl_div, optimizer = task_model(args, dataset, path_t)
            
    _, tacc = evaluate(modelt, valid_loader, cross_ent)
                        
    print('Teacher Accuracy: %f' %tacc)
                        
    v_ac, b_ac = train_kd(args, train_loader, valid_loader, modelt, models, cross_ent, kl_div, optimizer)
    
                    
    return    
    
    
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Graph')

    # 1) general params
    parser.add_argument("--gpu", type=int, default=0,
                        help="gpu")
    
    parser.add_argument("--seed", type=int, default=3,
                        help='random seed')
    
    parser.add_argument("--dataseed", type=int, default=29,
                        help='random seed') # to avoid real data leakage, use the same random dataset_seed as that for training the teacher
    
    parser.add_argument("--self_loop", action='store_true',
                        help='add self_loop to graph data')
    
    parser.add_argument("--data_dir", type=str, default='/home/xiang/structure-inversion/baselines/random_baseline/PTC/data', help='path to the datas')

    parser.add_argument('--dataset', type=str, default='MUTAG',
                        choices=['MUTAG'], help='name of dataset')

    parser.add_argument('--fake_dataset', type=str, default='fake_mutag', choices=['fake_mutag'],
                        help='name of dataset (default: MUTAG)'
                        )

    parser.add_argument('--fake_path', type=str, default=None, help='fake graph path')

     
    parser.add_argument("--tmodel", type=str, default='GIN', choices=['GIN', 'GCN'], help='graph models')
    
    parser.add_argument("--modelt", type=str, default='GIN5_64', 
                        choices=['GIN5_64', 'GIN5_32', 'GIN3_64', 'GIN3_32', 'GIN2_64', 'GIN2_32', 
                                 'GCN5_64', 'GCN5_32', 'GCN3_64', 'GCN3_32', 'GCN2_64', 'GCN2_32'],help='graph models')

    parser.add_argument("--smodel", type=str, default='GIN', choices=['GIN', 'GCN'], help='graph models')
    
    parser.add_argument("--models", type=str, default='GIN5_64', 
                        choices=['GIN5_64', 'GIN5_32', 'GIN3_64', 'GIN3_32', 'GIN2_64', 'GIN2_32', 
                                 'GCN5_64', 'GCN5_32', 'GCN3_64', 'GCN3_32', 'GCN2_64', 'GCN2_32'],help='graph models')

    
    parser.add_argument('--path_t', type=str, default=None, help='teacher model snapshot')
    
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
   
    parser.add_argument("--epoch", type=int, default=300, help="number of training epochs") # or 400 epochs
    
    parser.add_argument("--weight_decay", type=float, default=0.0, help='Weight for L2 Loss')
    
    parser.add_argument('--kd_T', type=float, default=2, help='temperature for KD distillation')

    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training and validation (default: 32)')
    
    parser.add_argument('--trial', type=int, default=0, help='trial')

    parser.add_argument('--split_name', type=str, default='rand', choices=['rand'], help='rand split with dataseed')

    parser.add_argument('--degree_as_label', action='store_true', help='use node degree as node labels')



    args = parser.parse_args()
    
    set_seed(args.seed)
                        
    print('seed: %d' %args.seed)

    os.environ['DGL_DOWNLOAD_DIR'] = args.data_dir

    print(args)
    main(args)



