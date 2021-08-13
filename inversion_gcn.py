from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

#DGL 0.4.2

import argparse
import random
import torch
import torch.nn as nn
# import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
# import torch.utils.data

import numpy as np
import collections
import os
import copy
#from utils.scheduler import LinearSchedule
from utils.helper import adjust_learning_rate



from utils.helper import norm_loss, bernulli_fastgrad, task_model, task_data, evaluate, generate_graphs, create_folder




class DeepInversionFeatureHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss.
    Adopted from "Dreaming to distill: Data-free knowledge transfer via deepinversion"
    '''

    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]

        mean = input[0].mean([0])
        var = input[0].permute(1, 0).contiguous().view([nch, -1]).var(1, unbiased=False)

        # forcing mean and variance to match between two distributions
        # other ways might work better, e.g. KL divergence
        r_feature = torch.norm(module.running_var.data.type(var.type()) - var, 2) + torch.norm(
            module.running_mean.data.type(var.type()) - mean, 2)

        self.r_feature = r_feature
        # must have no output

    def close(self):
        self.hook.remove()


def inverse(modelname, lr_f, lr_stru, n, params, net, feature_paras, targets, criterion, criterion_stru, self_loop, degree_as_label, epochs, batch_size, 
            gpu, onehot, optimizer_stru = None, optimizer_f = None, bn_reg_scale = 0.0, onehot_cof=0.0):


    best_cost = 1e9


    ## Create hooks for feature statistics catching
    loss_r_feature_layers = []
    for module in net.modules():
        if isinstance(module, nn.BatchNorm1d):
            loss_r_feature_layers.append(DeepInversionFeatureHook(module))
    
    savefolder = './save/fakegraphs'+'_bn'+str(bn_reg_scale)+'_oh'+str(onehot_cof)+'/'
    create_folder(savefolder)
            
    for epoch in range(epochs):
        
        adjust_learning_rate(epoch, lr_f, optimizer_f, 'gcn')
     
        adjust_learning_rate(epoch, lr_stru, optimizer_stru, 'gcn')
        
        optimizer_f.zero_grad()
        
        loss_fea, strus = norm_loss(n, params, feature_paras, targets, criterion, self_loop, degree_as_label, net, 
                                    loss_r_feature_layers, bn_reg_scale, batch_size, gpu, onehot, onehot_cof)
        
        loss_target = loss_fea.item()

        if epoch%100==0:
            print('epoch: %d' %epoch)
            print('loss: %f' %loss_target)
        
        if best_cost > loss_target:
            best_cost = loss_target
            best_strus = strus
            
            bestfeapath = savefolder+'feature_paras'+modelname+'_'+str(args.dataseed)+'_'+str(args.trial)+'.pt'
            
            beststrupath = savefolder+'stru_paras'+modelname+'_'+str(args.dataseed)+'_'+str(args.trial)+'.pt'
            
            torch.save(feature_paras, bestfeapath)
            torch.save(params, beststrupath)
            
        
        loss_fea.backward()
        
        optimizer_f.step()
        
        optimizer_stru.zero_grad()
        
        grads = bernulli_fastgrad(params, feature_paras, targets, criterion_stru, self_loop, degree_as_label, 
                                  net, batch_size, gpu, onehot, loss_r_feature_layers, bn_reg_scale)
        
        for pk in range(len(params)):
            para = params[pk]
            para.grad = grads[pk]
        
        
        optimizer_stru.step()            

    return best_cost, best_strus



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch MUTAG Fake Graph Generation')
    
    parser.add_argument('--di_lr_f', default=0.01, type=float, help='lr for feature')
    
    parser.add_argument('--di_lr_stru', default=1.0, type=float, help='lr for structure')
    
    parser.add_argument("--epoch", type=int, default=2000, help="number of training iteration")

    parser.add_argument('--bn_reg_scale', default=0.01, type=float, help='weight for BN regularization statistic')
    
    #dataset
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    
    
    parser.add_argument("--dataseed", type=int, default=29, help='random seed')
    
    parser.add_argument("--self_loop", action='store_true', help='add self_loop to graph data')
    
    parser.add_argument("--onehot", type=int, default=1, help="one hot features")
    
    parser.add_argument("--onehot_cof", type=float, default=0.01, help="gpu")


    parser.add_argument('--dataset', type=str, default='MUTAG', choices=['MUTAG'], help='name of dataset (default: MUTAG)')
    
    # 2) model params    
    parser.add_argument("--tmodel", type=str, default='GCN', choices=['GIN', 'GCN'], help='graph models')
    
    parser.add_argument("--smodel", type=str, default='GCN', choices=['GIN', 'GCN'], help='not used in data generation')
    
    parser.add_argument("--modelt", type=str, default='GCN5_64',  choices=[ 'GCN5_64'],help='graph models')

    parser.add_argument("--models", type=str, default='GCN3_32', 
                        choices=['GIN5_64', 'GIN5_32', 'GIN3_64', 'GIN3_32', 'GIN2_64', 'GIN2_32', 
                                 'GCN5_64', 'GCN5_32', 'GCN3_64', 'GCN3_32', 'GCN2_64', 'GCN2_32'],help='not used in data generation')
    
    parser.add_argument('--path_t', type=str, default=None, help='teacher path')
    
    parser.add_argument('--batch_size', type=int, default=131, help='batch size')#188 = 169+19

    parser.add_argument('--total_num', type=int, default=131, help='number of fake graphs, defualt 131 on MUTAG, 188*0.7=131.6')

    parser.add_argument('--degree_as_label', action='store_true', help='use node degree as node labels')
    
    parser.add_argument('--savepath', type=str, default='./save/fakegraphs', help='save path for generated graphs')
    
    parser.add_argument('--data_dir', type=str, default='./data', help='data path')
    
    parser.add_argument('--split_name', type=str, default='rand', choices=['rand'], help='rand split with dataseed')
    
    
    parser.add_argument('--trial', type=int, default=0, help='data path')

    
    
    
    args = parser.parse_args()
    
    
    os.environ['DGL_DOWNLOAD_DIR'] = args.data_dir
    
    print('\n \n ')
    print(args)
    
    

    print("loading teacher and student")

    
    # loading
    dataset, valid_loader = task_data(args)

    net_teacher, net_student = task_model(args, dataset)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    criterion = nn.CrossEntropyLoss().cuda()# 
    
    criterion_stru = nn.CrossEntropyLoss(reduce=False).cuda()
    
    path = args.path_t+'/ckpt_best_'+str(args.dataseed)+'.pth'
    
    net_teacher.load_state_dict(torch.load(path)['model'])    
    
    net_teacher.eval()
    
    cudnn.benchmark = True
    
    # Checking teacher accuracy
    print("Checking teacher accuracy")
    _, tacc = evaluate(net_teacher, valid_loader, criterion)

    print('Teacher Accuracy: %f' %tacc)     

    if args.total_num%args.batch_size==0:
        total_batchnum = int(args.total_num/args.batch_size)
        
    else:
        total_batchnum = int(args.total_num/args.batch_size)+1
    
    
    print("Starting model inversion")


    for bat_num in range(total_batchnum):
        
        print('bat_num: %d' %bat_num)
        
        if bat_num!=total_batchnum-1 or args.total_num%args.batch_size==0:
            
            number_list = random.choices(range(10, 30), k=args.batch_size)# 133 graphs with different number of nodes
            structure_paras = []
            feature_paras = []
            
            targets = [random.randint(0,1) for _ in range(args.batch_size)]    
            
            for n in number_list:
                
                p = np.random.uniform(size = (n, n))
                p = np.log( (p+1e-7)/(1-p+1e-7) )
                p = torch.tensor(p, requires_grad=False, device='cuda', dtype=torch.float)
                structure_paras += [p]   
                
                f = np.random.normal(size = (n, 7))#np.ones((n, 7)).astype(np.float32)
                f = torch.tensor(f, requires_grad=True, device='cuda', dtype=torch.float)
                
                feature_paras+=[f]
                
            #print(feature_paras)
            optimizer_stru = optim.Adam(structure_paras, lr=args.di_lr_stru)
            optimizer_f = optim.Adam(feature_paras, lr=args.di_lr_f)
            
            sample_n = 1
            
            cost, best_strus = inverse(modelname = args.modelt, lr_f=args.di_lr_f, lr_stru=args.di_lr_stru, n=sample_n, 
                                       params=structure_paras, net=net_teacher, feature_paras=feature_paras, targets=targets, 
                                       criterion=criterion, criterion_stru=criterion_stru, self_loop=args.self_loop, 
                                       degree_as_label=args.degree_as_label, epochs=args.epoch, batch_size = args.batch_size, 
                                       gpu=args.gpu,onehot=args.onehot, optimizer_stru=optimizer_stru, optimizer_f=optimizer_f, 
                                       bn_reg_scale=args.bn_reg_scale, onehot_cof=args.onehot_cof)
            
            if args.onehot:
                features = []
                for fp in feature_paras:
                    fea = torch.softmax(fp,1)
                    features += [fea]
            else:
                features = feature_paras
        
            savept = args.savepath+'_bn'+str(args.bn_reg_scale)+'_oh'+str(args.onehot_cof)+'/'
    
            create_folder(savept)
    
            generate_graphs(best_strus, features, targets, savept, args.dataseed, args.trial, args.modelt, bat_num, args.total_num)

        elif bat_num==total_batchnum - 1 and args.total_num%args.batch_size!=0:
            
            print('bat_num: %d' %bat_num)
            
            b1 = args.total_num - (args.batch_size*(total_batchnum-1) )
            
            number_list = random.choices(range(10, 30), k=b1)# 133 graphs with different number of nodes
            structure_paras = []
            feature_paras = []
            
            targets = [random.randint(0,1) for _ in range(b1)]    
            
            for n in number_list:
                
                p = np.random.uniform(size = (n, n))
                p = np.log( (p+1e-7)/(1-p+1e-7) )
                p = torch.tensor(p, requires_grad=False, device='cuda', dtype=torch.float)
                structure_paras += [p]   
                
                f = np.random.normal(size = (n, 7))#np.ones((n, 7)).astype(np.float32)
                f = torch.tensor(f, requires_grad=True, device='cuda', dtype=torch.float)
                
                feature_paras+=[f]
                
            #print(feature_paras)
            optimizer_stru = optim.Adam(structure_paras, lr=args.di_lr_stru)
            optimizer_f = optim.Adam(feature_paras, lr=args.di_lr_f)
            
            sample_n = 1
            
            cost, best_strus = inverse(modelname = args.modelt, lr_f=args.di_lr_f, lr_stru=args.di_lr_stru, n=sample_n, 
                                       params=structure_paras, net=net_teacher, feature_paras=feature_paras, targets=targets, 
                                       criterion=criterion, criterion_stru=criterion_stru, self_loop=args.self_loop, 
                                       degree_as_label=args.degree_as_label, epochs=args.epoch, batch_size = args.batch_size, 
                                       gpu=args.gpu,onehot=args.onehot, optimizer_stru=optimizer_stru, optimizer_f=optimizer_f, 
                                       bn_reg_scale=args.bn_reg_scale, onehot_cof=args.onehot_cof)
            
            if args.onehot:
                features = []
                for fp in feature_paras:
                    fea = torch.softmax(fp,1)
                    features += [fea]
            else:
                features = feature_paras
        
            savept = args.savepath+'_bn'+str(args.bn_reg_scale)+'_oh'+str(args.onehot_cof)+'/'
    
            create_folder(savept)
    
            generate_graphs(best_strus, features, targets, savept, args.dataseed, args.trial, args.modelt, bat_num, args.total_num)