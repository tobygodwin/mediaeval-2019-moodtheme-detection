# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import datetime
import tqdm
from sklearn import metrics
import pickle
import csv

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from model import MusicSelfAttModel, ClassClassifierMobileNetV2, ClassClassifierSelfAttention,\
DomainClassifierMobileNetV2, DomainClassifierSelfAttention

#######################################################
#         DOMAIN ADVERSARIAL TRAINING
#######################################################

LAMBDA = 1
GAMMA = 10

class Solver():
    def __init__(self, data_loader1, data_loader2, valid_loader, tag_list, config, LABELS_TXT, num_epochs=120, mode='train', target_loader=None):
        if mode == 'train_adversarial':
            # If adversarial training use source/target data
            # extra target_loader also required.
            self.source_dataloader1 = data_loader1
            self.source_dataloader2 = data_loader2
            self.target_dataloader = target_loader
            self.valid_loader = valid_loader
        else:
            # Data loader
            self.data_loader1 = data_loader1
            self.data_loader2 = data_loader2
            self.valid_loader = valid_loader

        # Training settings
        self.n_epochs = num_epochs #ORIGINALLY 120
        self.lr = 1e-4
        self.log_step = 100
        self.is_cuda = torch.cuda.is_available()
        self.model_save_path = config['log_dir']
        self.batch_size = config['batch_size']
        self.tag_list = tag_list
        self.num_class = 4
        self.writer = SummaryWriter(config['log_dir'])
        self.mode = mode
        self.labels_txt = LABELS_TXT
        # self.model_fn = os.path.join(self.model_save_path, 'best_model.pth')

        # Build model
        self.build_models()

    def build_models(self):
        # model and optimizer
        feature_extractor = MusicSelfAttModel()
        class_mobnet = ClassClassifierMobileNetV2()
        class_selfatt = ClassClassifierSelfAttention()


        if self.is_cuda:
            self.feature_extractor = feature_extractor
            self.class_mobnet = class_mobnet
            self.class_selfatt = class_selfatt

            self.feature_extractor.cuda()
            self.class_mobnet.cuda()
            self.class_selfatt.cuda()

        if self.mode == 'train_adversarial':
            # If adversarial training include domain classifiers
            self.domain_class_mobnet = DomainClassifierMobileNetV2().cuda()
            self.domain_class_selfatt = DomainClassifierSelfAttention().cuda()
            self.optimizer = torch.optim.Adam([{'params': self.feature_extractor.parameters()},
                                               {'params': self.class_mobnet.parameters()},
                                               {'params': self.class_selfatt.parameters()},
                                               {'params': self.domain_class_mobnet.parameters()},
                                               {'params': self.domain_class_selfatt.parameters()}], self.lr)


        else:
            self.optimizer = torch.optim.Adam([{'params': self.feature_extractor.parameters()},
                                           {'params': self.class_mobnet.parameters()},
                                           {'params': self.class_selfatt.parameters()}], self.lr)

    def load(self, filename, model):
        S = torch.load(filename)
        model.load_state_dict(S)

    def save(self, filename):
        torch.save({'model': self.class_selfatt.state_dict()}, filename)
        torch.save({'model': self.class_mobnet.state_dict()}, filename)
        torch.save({'model': self.feature_extractor.state_dict()}, filename)
        if self.mode == 'train_adversarial':
            torch.save({'model': self.domain_class_mobnet.state_dict()}, filename)
            torch.save({'model': self.domain_class_selfatt.state_dict()}, filename)

    def to_var(self, x):
        if self.is_cuda:
            x = x.cuda()
        return x

    def train(self,exp_name):
        #filename = args.filename
        filename = exp_name

        start_t = time.time()
        current_optimizer = 'adam'
        best_roc_auc = 0
        drop_counter = 0
        reconst_loss = nn.BCELoss()
        roc_auc, _ = self._validation(start_t, 0)
        for epoch in range(self.n_epochs):
            print('Training')
            drop_counter += 1
            # train
            self.feature_extractor.train()
            self.class_mobnet.train()
            self.class_selfatt.train()
            ctr = 0
            step_loss = 0
            epoch_loss = 0

            for i1, i2 in zip(self.data_loader1, self.data_loader2):
                ctr += 1

                # mixup---------
                alpha = 1
                mixup_vals = np.random.beta(alpha, alpha, i1[0].shape[0])
                
                lam = torch.Tensor(mixup_vals.reshape(mixup_vals.shape[0], 1, 1, 1))
                inputs = (lam * i1[0]) + ((1 - lam) * i2[0])
                
                lam = torch.Tensor(mixup_vals.reshape(mixup_vals.shape[0], 1))
                labels = (lam * i1[1]) + ((1 - lam) * i2[1])

                # variables to cuda
                x = self.to_var(inputs)
                y = self.to_var(labels)
                # predict
                att,clf = self.feature_extractor(x)
                att_pred = self.class_selfatt(att)
                mob_pred = self.class_mobnet(clf)

                # print(clf)
                # print(y)

                loss1 = reconst_loss(att_pred, y)
                loss2 = reconst_loss(mob_pred, y)
                loss = (loss1+loss2)/2

                step_loss += loss.item()
                epoch_loss += loss.item()

                # back propagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # print log
                if (ctr) % self.log_step == 0:
                    print("[%s] Epoch [%d/%d] Iter [%d/%d] train loss: %.4f Elapsed: %s" %
                            (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            epoch+1, self.n_epochs, ctr, len(self.data_loader1), (step_loss/self.log_step),
                            datetime.timedelta(seconds=time.time()-start_t)))
                    step_loss = 0

            self.writer.add_scalar('Loss/train', epoch_loss/len(self.data_loader1), epoch)

            # validation
            roc_auc, _ = self._validation(start_t, epoch)

            # save model
            if roc_auc > best_roc_auc:
                print('best model: %4f' % roc_auc)
                best_roc_auc = roc_auc
                torch.save(self.feature_extractor.state_dict(), os.path.join(self.model_save_path,
                                                                             filename+'_feature_extractor.pth'))
                torch.save(self.class_mobnet.state_dict(), os.path.join(self.model_save_path,
                                                                             filename + '_class_mobnet.pth'))
                torch.save(self.class_selfatt.state_dict(), os.path.join(self.model_save_path,
                                                                        filename + '_class_selfatt.pth'))

            # if epoch%10 == 0:
            #     print(f'Saving model at epoch {epoch}')
            #     torch.save(self.model.state_dict(), os.path.join(self.model_save_path, f'model_epoch_{epoch}.pth'))

            # schedule optimizer
            current_optimizer, drop_counter = self._schedule(current_optimizer, drop_counter,filename)

        print("[%s] Train finished. Elapsed: %s"
                % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    datetime.timedelta(seconds=time.time() - start_t)))

    def train_adversarial(self, exp_name):
        #filename = args.filename
        filename = exp_name

        start_t = time.time()
        current_optimizer = 'adam'
        best_roc_auc = 0
        drop_counter = 0

        # Define loss criteria
        class_criterion = nn.BCELoss()
        domain_criterion = nn.NLLLoss()
        roc_auc, _ = self._validation(start_t, 0)

        for epoch in range(self.n_epochs):
            print('Training')
            drop_counter += 1
            # train
            self.feature_extractor.train()
            self.class_mobnet.train()
            self.class_selfatt.train()
            self.domain_class_mobnet.train()
            self.domain_class_selfatt.train()

            ctr = 0
            step_class_loss = 0
            step_domain_loss = 0
            step_total_loss = 0
            epoch_loss = 0

            # steps
            start_steps = epoch * len(self.source_dataloader1)
            total_steps = self.n_epochs * len(self.source_dataloader1)

            # Step through batches
            for source1, source2, target in zip(self.source_dataloader1, self.source_dataloader2, self.target_dataloader):

                # constant for gradient reversal layer
                # from
                # http://sites.skoltech.ru/compvision/projects/grl/files/paper.pdf
                p = float(ctr + start_steps) / total_steps
                constant = 2. / (1. + np.exp(-GAMMA * p)) - 1

                ctr += 1

                # mixup---------
                alpha = 1
                mixup_vals = np.random.beta(alpha, alpha, source1[0].shape[0])

                lam = torch.Tensor(mixup_vals.reshape(mixup_vals.shape[0], 1, 1, 1))
                inputs = (lam * source1[0]) + ((1 - lam) * source2[0])

                lam = torch.Tensor(mixup_vals.reshape(mixup_vals.shape[0], 1))
                labels = (lam * source1[1]) + ((1 - lam) * source2[1])

                # variables to cuda
                x_source = self.to_var(inputs)
                y_source = self.to_var(labels)
                x_target = self.to_var(target[0])
                y_target = self.to_var(target[1])

                # set up domain labels
                source_labels = self.to_var(torch.zeros((x_source.size()[0])).type(torch.LongTensor))
                target_labels = self.to_var(torch.zeros((x_target.size()[0])).type(torch.LongTensor))

                # get features from source and target domain
                src_att_feat, src_clf_feat = self.feature_extractor(x_source)
                tgt_att_feat, tgt_clf_feat = self.feature_extractor(x_target)

                # get class prediction from source features
                class_preds_selfatt = self.class_selfatt(src_att_feat)
                class_preds_mobnet = self.class_mobnet(src_clf_feat)

                # compute class loss
                loss_selfatt = class_criterion(class_preds_selfatt, y_source)
                loss_mobnet = class_criterion(class_preds_mobnet, y_source)
                class_loss = (loss_mobnet + loss_selfatt)/2

                # get domain prediction from source and target features
                src_dom_pred_selfatt = self.domain_class_selfatt(src_att_feat, constant)
                src_dom_pred_mobnet = self.domain_class_mobnet(src_clf_feat, constant)

                tgt_dom_pred_selfatt = self.domain_class_selfatt(tgt_att_feat, constant)
                tgt_dom_pred_mobnet = self.domain_class_mobnet(tgt_clf_feat, constant)

                # compute domain loss
                source_loss_att = domain_criterion(src_dom_pred_selfatt, source_labels)
                source_loss_mobnet = domain_criterion(src_dom_pred_mobnet, source_labels)
                source_loss = (source_loss_att + source_loss_mobnet)/2

                target_loss_att = domain_criterion(tgt_dom_pred_selfatt, target_labels)
                target_loss_mobnet = domain_criterion(tgt_dom_pred_mobnet, target_labels)
                target_loss = (target_loss_att + target_loss_mobnet)/2

                domain_loss = target_loss + source_loss

                # compute total loss
                loss = class_loss + LAMBDA * domain_loss

                step_class_loss += class_loss.item()
                step_domain_loss += domain_loss.item()
                step_total_loss += loss.item()
                epoch_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # print log
                if (ctr) % self.log_step == 0:

                    print("[%s] Epoch [%d/%d] Iter [%d/%d] train total loss: %.4f train domain loss: %.4f train class loss: %.4f Elapsed: %s" %
                          (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                           epoch + 1, self.n_epochs, ctr, len(self.source_dataloader1), (step_total_loss / self.log_step),
                           (step_domain_loss / self.log_step),  (step_class_loss / self.log_step),
                           datetime.timedelta(seconds=time.time() - start_t)))
                    step_class_loss = 0
                    step_domain_loss = 0
                    step_total_loss = 0
            self.writer.add_scalar('Loss/train', epoch_loss / len(self.source_dataloader1), epoch)

            # validation
            roc_auc, _ = self._validation(start_t, epoch)

            # save model
            if roc_auc > best_roc_auc:
                print('best model: %4f' % roc_auc)
                best_roc_auc = roc_auc

                torch.save(self.feature_extractor.state_dict(), os.path.join(self.model_save_path,
                                                                             filename + '_feature_extractor.pth'))
                torch.save(self.class_mobnet.state_dict(), os.path.join(self.model_save_path,
                                                                        filename + '_class_mobnet.pth'))
                torch.save(self.class_selfatt.state_dict(), os.path.join(self.model_save_path,
                                                                         filename + '_class_selfatt.pth'))

                torch.save(self.domain_class_mobnet.state_dict(), os.path.join(self.model_save_path,
                                                                             filename + '_domain_mobnet.pth'))
                torch.save(self.domain_class_selfatt.state_dict(), os.path.join(self.model_save_path,
                                                                        filename + '_domain_selfatt.pth'))

            # schedule optimizer
            current_optimizer, drop_counter = self._schedule(current_optimizer, drop_counter, filename)

        print("[%s] Train finished. Elapsed: %s"
              % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                 datetime.timedelta(seconds=time.time() - start_t)))


    def _validation(self, start_t, epoch):
        prd1_array = []  # prediction
        prd2_array = []
        gt_array = []   # ground truth
        ctr = 0
        self.feature_extractor.eval()
        self.class_mobnet.eval()
        self.class_selfatt.eval()

        reconst_loss = nn.BCELoss()
        for x, y in self.valid_loader:
            ctr += 1
            #print(y)
            # variables to cuda
            x = self.to_var(x)
            y = self.to_var(y)

            # predict
            att, clf = self.feature_extractor(x)
            att_pred = self.class_selfatt(att)
            mob_pred = self.class_mobnet(clf)

            loss1 = reconst_loss(att_pred, y)
            loss2 = reconst_loss(mob_pred, y)
            loss = (loss1+loss2)/2

            # print log
            if (ctr) % self.log_step == 0:
                print("[%s] Epoch [%d/%d], Iter [%d/%d] valid loss: %.4f Elapsed: %s" %
                        (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        epoch+1, self.n_epochs, ctr, len(self.valid_loader), loss.item(),
                        datetime.timedelta(seconds=time.time()-start_t)))

            # append prediction
            att_pred = att_pred.detach().cpu()
            mob_pred = mob_pred.detach().cpu()
            y = y.detach().cpu()
            for prd1 in att_pred:
                prd1_array.append(list(np.array(prd1)))
            for prd2 in mob_pred:
                prd2_array.append(list(np.array(prd2)))
            for gt in y:
                gt_array.append(list(np.array(gt)))

        val_loss1 = reconst_loss(torch.Tensor(prd1_array), torch.Tensor(gt_array))
        val_loss2 = reconst_loss(torch.Tensor(prd2_array), torch.Tensor(gt_array))
        print(f'Val Loss: {val_loss1}, {val_loss2}')
        self.writer.add_scalar('Loss/val1', val_loss1, epoch)
        self.writer.add_scalar('Loss/val2', val_loss2, epoch)

        # get auc
        list_all = True if epoch==self.n_epochs else False

        roc_auc1, pr_auc1, _, _ = self.get_auc(prd1_array, gt_array, list_all)
        roc_auc2, pr_auc2, _, _ = self.get_auc(prd2_array, gt_array, list_all)
        self.writer.add_scalar('AUC/ROC2', roc_auc1, epoch)
        self.writer.add_scalar('AUC/PR2', pr_auc1, epoch)
        self.writer.add_scalar('AUC/ROC2', roc_auc2, epoch)
        self.writer.add_scalar('AUC/PR2', pr_auc2, epoch)
        return roc_auc1, pr_auc1

    def get_idx2label(self, labels_path):
        idx2label = {}
        tag_list = []
        with open(labels_path) as f:
            lines = f.readlines()

        for i, l in enumerate(lines):
            tag_list.append(l.strip())
            idx2label[i] = l.strip()

        return idx2label

    def write_metrics(self, pred_array, gt_array):

        roc_auc_class = metrics.roc_auc_score(gt_array, pred_array, average=None)
        roc_auc_macro = metrics.roc_auc_score(gt_array, pred_array, average='macro', multi_class='ovo')
        roc_auc_micro = metrics.roc_auc_score(gt_array, pred_array, average='micro')

        pr_auc_class = metrics.average_precision_score(gt_array, pred_array, average=None)
        pr_auc_macro = metrics.average_precision_score(gt_array, pred_array, average='macro')
        pr_auc_micro = metrics.average_precision_score(gt_array, pred_array, average='micro')

        class_pred = np.array(pred_array).argmax(axis=1)
        class_gt = np.array(gt_array).argmax(axis=1)

        conf_mat = metrics.confusion_matrix(class_gt, class_pred)

        f1_class = metrics.f1_score(class_gt, class_pred, average=None)
        f1_macro = metrics.f1_score(class_gt, class_pred, average='macro')
        f1_micro = metrics.f1_score(class_gt, class_pred, average='micro')

        idx2label = self.get_idx2label(self.labels_txt)
        save_path = self.model_save_path + "/metrics.txt"
        with open("metrics.txt", 'w', encoding='utf-8') as f:
            f.write(f"Confusion Matrix:\n {conf_mat} \n")
            f.write(f"ROC-AUC macro: {roc_auc_macro} \n")
            f.write(f"ROC-AUC micro: {roc_auc_micro} \n")
            [f.write(f"ROC-AUC {idx2label[idx]}: {pr_auc_class[idx]} \n") for idx in range(len(roc_auc_class))]
            f.write(f"PR-AUC macro: {pr_auc_macro} \n")
            f.write(f"PR-AUC micro: {pr_auc_micro} \n")
            [f.write(f"PR-AUC {idx2label[idx]}: {pr_auc_class[idx]} \n") for idx in range(len(roc_auc_class))]
            f.write(f"F1 macro: {f1_macro} \n")
            f.write(f"F1 micro: {f1_micro} \n")
            [f.write(f"F1 {idx2label[idx]}: {f1_class[idx]} \n") for idx in range(len(roc_auc_class))]

        print(f"ROC-AUC macro:\n {conf_mat} \n")
        print(f"ROC-AUC macro: {roc_auc_macro} \n")
        print(f"ROC-AUC micro: {roc_auc_micro} \n")
        [print(f"ROC-AUC {idx2label[idx]}: {pr_auc_class[idx]} \n") for idx in range(len(roc_auc_class))]
        print(f"PR-AUC macro: {pr_auc_macro} \n")
        print(f"PR-AUC micro: {pr_auc_micro} \n")
        [print(f"PR-AUC {idx2label[idx]}: {pr_auc_class[idx]} \n") for idx in range(len(roc_auc_class))]
        print(f"F1 macro: {f1_macro} \n")
        print(f"F1 micro: {f1_micro} \n")
        [print(f"F1 {idx2label[idx]}: {f1_class[idx]} \n") for idx in range(len(roc_auc_class))]
        return roc_auc_macro, pr_auc_macro, roc_auc_class, pr_auc_class



    def get_auc(self, prd_array, gt_array, list_all=False):
        prd_array = np.array(prd_array)
        gt_array = np.array(gt_array)

        roc_aucs = metrics.roc_auc_score(gt_array, prd_array, average='macro')
        pr_aucs = metrics.average_precision_score(gt_array, prd_array, average='macro')

        print('roc_auc: %.4f' % roc_aucs)
        print('pr_auc: %.4f' % pr_aucs)

        roc_auc_all = metrics.roc_auc_score(gt_array, prd_array, average=None)
        pr_auc_all = metrics.average_precision_score(gt_array, prd_array, average=None)

        if list_all==True:            
            for i in range(self.num_class):
                print('%s \t\t %.4f , %.4f' % (self.tag_list[i], roc_auc_all[i], pr_auc_all[i]))
        
        return roc_aucs, pr_aucs, roc_auc_all, pr_auc_all

    def get_conf_f1(self,prd_array, gt_array, list_all=False):
        prd_array = np.array(prd_array).argmax(axis=1)
        gt_array = np.array(gt_array).argmax(axis=1)

        print(prd_array)
        conf_mat = metrics.confusion_matrix(gt_array, prd_array)

        print(f"confusion matrix: \n  {conf_mat}")

        f1 = metrics.f1_score(gt_array, prd_array, average='macro')

        print(f"Macro F1: {f1}")

        return f1, conf_mat

    def _schedule(self, current_optimizer, drop_counter,filename):

        self.load(os.path.join(self.model_save_path, filename + '_feature_extractor.pth'), self.feature_extractor)
        self.load(os.path.join(self.model_save_path, filename + '_class_mobnet.pth'), self.class_mobnet)
        self.load(os.path.join(self.model_save_path, filename + '_class_selfatt.pth'), self.class_selfatt)

        if self.mode == 'train_adversarial':
            self.load(os.path.join(self.model_save_path, filename + '_domain_mobnet.pth'), self.domain_class_mobnet)
            self.load(os.path.join(self.model_save_path, filename + '_domain_selfatt.pth'), self.domain_class_selfatt)


        if current_optimizer == 'adam' and drop_counter == 60:
            if self.mode == 'train_adversarial':
                self.optimizer = torch.optim.SGD([{'params': self.feature_extractor.parameters()},
                                                  {'params': self.class_mobnet.parameters()},
                                                  {'params': self.class_selfatt.parameters()},
                                                  {'params': self.domain_class_mobnet.parameters()},
                                                  {'params': self.domain_class_selfatt.parameters()}],
                                                 0.001, momentum=0.9, weight_decay=0.0001, nesterov=True)
            else:
                self.optimizer = torch.optim.SGD([{'params': self.feature_extractor.parameters()},
                                               {'params': self.class_mobnet.parameters()},
                                               {'params': self.class_selfatt.parameters()}],
                                                0.001, momentum=0.9, weight_decay=0.0001, nesterov=True)
            current_optimizer = 'sgd_1'
            drop_counter = 0
            print('sgd 1e-3')
        # first drop
        if current_optimizer == 'sgd_1' and drop_counter == 20:
            for pg in self.optimizer.param_groups:
                pg['lr'] = 0.0001
            current_optimizer = 'sgd_2'
            drop_counter = 0
            print('sgd 1e-4')
        # second drop
        if current_optimizer == 'sgd_2' and drop_counter == 20:
            for pg in self.optimizer.param_groups:
                pg['lr'] = 0.00001
            current_optimizer = 'sgd_3'
            print('sgd 1e-5')
        return current_optimizer, drop_counter

    def test(self, exp_name):
        filename = exp_name
        start_t = time.time()
        reconst_loss = nn.BCELoss()
        epoch = 0

        self.load(os.path.join(self.model_save_path, filename + '_feature_extractor.pth'), self.feature_extractor)
        self.load(os.path.join(self.model_save_path, filename + '_class_mobnet.pth'), self.class_mobnet)
        self.load(os.path.join(self.model_save_path, filename + '_class_selfatt.pth'), self.class_selfatt)

        self.feature_extractor.eval()
        self.class_mobnet.eval()
        self.class_selfatt.eval()
        ctr = 0
        prd_array = []  # prediction
        gt_array = []   # ground truth
        for x, y in self.data_loader1:
            ctr += 1

            # variables to cuda
            x = self.to_var(x)
            y = self.to_var(y)

            # predict
            att, clf = self.feature_extractor(x)
            att_pred = self.class_selfatt(att)
            mob_pred = self.class_mobnet(clf)

            out = (att_pred + mob_pred)/2
            loss = reconst_loss(out, y)

            # predict
            # out1, out2 = self.model(x)
            # out = (out1+out2)/2
            # loss = reconst_loss(out, y)

            # print log
            if (ctr) % self.log_step == 0:
                print("[%s] Iter [%d/%d] test loss: %.4f Elapsed: %s" %
                        (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        ctr, len(self.data_loader1), loss.item(),
                        datetime.timedelta(seconds=time.time()-start_t)))

            # append prediction
            out = out.detach().cpu()
            y = y.detach().cpu()
            for prd in out:
                prd_array.append(list(np.array(prd)))
            for gt in y:
                gt_array.append(list(np.array(gt)))

        #np.save('./pred_array.npy', np.array(prd_array))
        #np.save('./gt_array.npy', np.array(gt_array))

        # get auc
        roc_auc, pr_auc, roc_auc_all, pr_auc_all = self.get_auc(prd_array, gt_array)
        f1, conf_mat = self.get_conf_f1(prd_array, gt_array)

        roc_auc, pr_auc, roc_auc_all, pr_auc_all = self.write_metrics(prd_array, gt_array)

        return (np.array(prd_array), np.array(gt_array), roc_auc, pr_auc)

        # save aucs
        #np.save(open(self.roc_auc_fn, 'wb'), roc_auc_all)
        #np.save(open(self.pr_auc_fn, 'wb'), pr_auc_all)

