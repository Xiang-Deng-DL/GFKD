#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 01:33:09 2020

@author: xiangdeng
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals



import torch
import copy
from model.GIN import GIN_dict
from model.GCN import GCN_dict
from Temp.dataset import GINDataset
from utils.GIN.full_loader import GraphFullDataLoader
from Temp.stru_dataset import STRDataset
from utils.GIN.data_loader import GraphDataLoader, collate
import os



def adjust_learning_rate(epoch, learning_rate, optimizer, model):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    
    if model=='gin':
        step = int(epoch/700)
        new_lr = learning_rate * (0.1 ** step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr       
    elif model=='gcn':
        step = int(epoch/700)
        new_lr = learning_rate * (0.1 ** step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

def nernulli_sample(params):
    
    strus = []
    
    for parm in params:
        pro = torch.sigmoid(parm)
        stru = (pro>=0.5).type(torch.int).cuda() #or torch.bernoulli(pro)
        strus +=[stru]
        
    return strus


def norm_loss(n, params, feature_paras, targets, criterion, self_loop, degree_as_label, net, 
              loss_r_feature_layers, bn_reg_scale, batch_size, gpu, onehot, onehot_cof):

    
    strus = nernulli_sample(params)
    
        
    if onehot:
        graphfeatures = []
        for fp in feature_paras:
            fea = torch.softmax(fp,1)
            graphfeatures += [fea]
    else:
        graphfeatures = feature_paras
    
    dataset = STRDataset(strus, graphfeatures, targets, self_loop, degree_as_label)
    train_loader = GraphFullDataLoader(dataset, batch_size=batch_size, device=gpu).train_loader()
    for graphs, labels in train_loader:
        labels = labels.cuda()
        features = graphs.ndata['attr'].cuda()
        outputs = net(graphs, features)
        loss1 = criterion(outputs, labels)
        loss_distr = sum([mod.r_feature for mod in loss_r_feature_layers])        
        loss = loss1 + bn_reg_scale*loss_distr

    #print('start sample second')    
    for i in range(n-1):
        strus = nernulli_sample(params)
        dataset = STRDataset(strus, graphfeatures, targets, self_loop, degree_as_label)
        train_loader = GraphFullDataLoader(dataset, batch_size=batch_size, device=gpu).train_loader()
        for graphs, labels in train_loader:
            labels = labels.cuda()
            features = graphs.ndata['attr'].cuda()
            outputs = net(graphs, features)
            loss1 = criterion(outputs, labels)  
            loss_distr = sum([mod.r_feature for mod in loss_r_feature_layers])
            curloss = loss1+bn_reg_scale*loss_distr
            loss+=curloss
            
    loss = loss/n
    if onehot:
        allfeatures = torch.cat(graphfeatures, dim=0)
        b = allfeatures * torch.log(allfeatures)
        h = -1.0 * b.sum()/len(allfeatures)
        loss = loss + onehot_cof*h
    return loss, strus
        
  
def generate_b(param):
    
    num=len(param)
    first =[]
    second=[]
    noise=[]
    for i in range(num):
        temparam=param[i]
        noise_shape=temparam.shape
        u_noise = torch.rand(size=noise_shape).cuda()
        
        P1 = torch.sigmoid(-temparam)
        E1 = (u_noise>P1).type(torch.int).cuda()
         
        P2 = 1 - P1
        E2 = (u_noise<P2).type(torch.int).cuda()
        
        first+=[E1]
        second+=[E2]
        noise+=[u_noise]
      
    return first, second, noise


def bernulli_fastgrad(params, feature_paras, targets, criterion_stru, self_loop, degree_as_label, net, batch_size, gpu, 
                      onehot, loss_r_feature_layers, bn_reg_scale):
    
    first, second, noise = generate_b(params)
    
    if onehot:
        graphfeatures = []
        for fp in feature_paras:
            fea = torch.softmax(fp,1)
            graphfeatures += [fea]
    else:
        graphfeatures = feature_paras
    
    grads = []
   
    dataset1 = STRDataset(first, graphfeatures, targets, self_loop, degree_as_label)
    train_loader1 = GraphFullDataLoader(dataset1, batch_size=batch_size, device=gpu).train_loader()
    
    for graphs1, labels1 in train_loader1:
        labels1 = labels1.cuda()
        features1 = graphs1.ndata['attr'].cuda()
        outputs1 = net(graphs1, features1)
        loss_ce1 = criterion_stru(outputs1, labels1)
        loss_distr1 = sum([mod.r_feature for mod in loss_r_feature_layers])*bn_reg_scale
    
    
    dataset2 = STRDataset(second, graphfeatures, targets, self_loop, degree_as_label)
    train_loader2 = GraphFullDataLoader(dataset2, batch_size=batch_size, device=gpu).train_loader()
    
    for graphs2, labels2 in train_loader2:
        labels2 = labels2.cuda()
        features2 = graphs2.ndata['attr'].cuda()
        outputs2 = net(graphs2, features2)
        loss_ce2 = criterion_stru(outputs2, labels2)
        loss_distr2 = sum([mod.r_feature for mod in loss_r_feature_layers])*bn_reg_scale
        
        
    for i in range( len(noise) ):      
        
        grad = (loss_ce1[i]-loss_ce2[i] + loss_distr1-loss_distr2 )*(noise[i] - 0.5)
        grads+=[grad]
    
    return grads




def task_data(args):

    # step 0: setting for gpu
    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)
        
    # step 1: prepare dataset
    
    dataset = GINDataset(args.dataset, args.self_loop, args.degree_as_label)
    
    print(dataset.dim_nfeats)
   
    # step 2: prepare data_loader
    _, valid_loader = GraphDataLoader(
            dataset, batch_size=32, device=args.gpu,
            collate_fn=collate, seed=args.dataseed, shuffle=True,
            split_name=args.split_name).train_valid_loader()


    return dataset, valid_loader


def task_model(args, dataset):

    #  step 1: prepare model
    assert args.tmodel in ['GIN', 'GCN']
    assert args.smodel in ['GIN', 'GCN']
    
    if args.tmodel == 'GIN':
        modelt = GIN_dict[args.modelt](dataset)
    elif args.tmodel == 'GCN':
        modelt = GCN_dict[args.modelt](dataset)
    else:
        raise('Not supporting such model!')


    if args.smodel == 'GIN':
        models = GIN_dict[args.models](dataset)
    elif args.smodel == 'GCN':
        models = GCN_dict[args.models](dataset)
    else:
        raise('Not supporting such model!')
        
        
    modelt = modelt.cuda()
    models = models.cuda()


    return modelt, models


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


def generate_graphs(sturectures, features, targets, path, daseed, trial, modelname, bat_num, total_num):
    
    graph_num = len(sturectures)
 
    filep = path+modelname+'fake_mutag'+ str(daseed)+'_'+str(trial)+ '.txt'
    
    if bat_num ==0:
        open(filep, 'w').close()
        
    with open(filep,'a') as f:
        if bat_num==0:
            tnum = str(total_num)
            f.write(tnum)
            f.write('\n')
        
        for i in range(graph_num):
            
            # node num and label
            
            feas = features[i]
            feas = torch.argmax(feas, 1)
            feas = feas.to('cpu').numpy()
            
            stru = sturectures[i]
            node_number, label = stru.shape[0], targets[i]
            label = str(label)
            content = str(node_number)+' '+label
            #content = content.replace('/n', ' ')
            f.write(content)
            f.write('\n')
            
            #
            for j in range(node_number):
                
                cur_row = stru[j]
                
                neig = ((cur_row == 1).nonzero())
                
                neig = neig[neig!=j]
                
                
                num = len(neig)
                
                neig = neig.to('cpu').numpy()
                
                '''if num>7:
                    neig = list(neig)
                    num = 7
                    neig = np.array(random.sample(neig, 7))'''
                
                if num>0:
                    neig=str(neig)[1:-1]
                else:
                    neig = str(neig)
                
                #node_label = random.sample(range(0, 7), 1)[0]
                node_label = feas[j]
                
                node_inf = str(node_label)+' '+str(num)+' '+neig
                
                node_inf = node_inf.replace('\n', ' ').replace('\r', ' ')
                                
                f.write(node_inf)
                f.write('\n')



              
def create_folder(directory):
    # from https://stackoverflow.com/a/273227
    if not os.path.exists(directory):
        os.makedirs(directory)