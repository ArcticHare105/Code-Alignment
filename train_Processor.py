import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import os
import time
import numpy as np
import warnings
from gensim.models.word2vec import Word2Vec
from model import BatchProgramCC
from torch.autograd import Variable
from ccd_dataloader import CCDDataset
#from focalloss import FocalLoss
warnings.filterwarnings('ignore')

class Processor():
    def __init__(self, args):
        # parameters
        self.args = args
        # device
        self.device = torch.device(
            'cuda:' + str(self.args.gpu_ids[0]) if torch.cuda.is_available() and len(self.args.gpu_ids) > 0 else 'cpu')

        if self.args.run_type == 0:
            self.train_data_loader = torch.utils.data.DataLoader(CCDDataset(self.args, 'train'),
                                                                 batch_size=1,
                                                                 shuffle=False,
                                                                 #num_workers=2 * len(self.args.gpu_ids),
                                                                 drop_last=False,
                                                                 collate_fn=lambda x:x)
            self.test_data_loader = torch.utils.data.DataLoader(CCDDataset(self.args, 'test'),
                                                                batch_size=1,
                                                                shuffle=False,
                                                                drop_last=False,
                                                                collate_fn=lambda x:x)
        elif self.args.run_type == 1 or self.args.run_type == 2:
            self.test_data_loader = torch.utils.data.DataLoader(CCDDataset(self.args, 'test'),
                                                                batch_size=1,
                                                                shuffle=False,
                                                                drop_last=False,
                                                                collate_fn=lambda x:x)
        else:
            raise ValueError('Do not Exist This Processing')

        # Loss Function Setting
        self.loss_BCE = nn.BCELoss()
        #self.loss_focal = FocalLoss()

        # initialization
        word2vec = Word2Vec.load(args.root + args.lang + "/train/embedding/node_w2v_128").wv
        max_tokens = word2vec.syn0.shape[0]
        embedding_dim = word2vec.syn0.shape[1]
        embeddings = np.zeros((max_tokens + 1, embedding_dim), dtype="float32")
        embeddings[:word2vec.syn0.shape[0]] = word2vec.syn0

        self.model = BatchProgramCC(args, embedding_dim, max_tokens + 1, 1, True, embeddings).to(self.device)

        # total = sum([param.nelement() for param in self.model.parameters()])
        # print("Number of parameter: %.2fM" % (total / 1e6))
        # import pdb; pdb.set_trace()

        # # Model Parallel Setting
        # if len(self.args.gpu_ids) > 1:
            # self.model = nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            # self.model_module = self.model.module
        # else:
            # self.model_module = self.model

        # Loading Pretrain Model
        if self.args.pretrained:
            model_dir = './ckpt/' + self.args.lang + '/' + str(self.args.model_id) + '/' + str(
                self.args.load_epoch) + '.pkl'
            if os.path.isfile(model_dir):
                self.model.load_state_dict(torch.load(model_dir))
            else:
                raise ValueError('Do Not Exist This Pretrained File')

        # Optimizer Setting
        if self.args.optimizer == 'Adam':
            self.optimizer = torch.optim.Adamax(self.model.parameters(), lr=self.args.lr, betas=[0.9, 0.999])
        elif self.args.optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum,
                                             weight_decay=self.args.weight_decay, nesterov=True)
        else:
            raise ValueError('Do Not Exist This Optimizer')

        # Optimizer Parallel Setting
        if len(self.args.gpu_ids) > 1:
            self.optimizer = nn.DataParallel(self.optimizer, device_ids=self.args.gpu_ids)
            self.optimizer_module = self.optimizer.module
        else:
            self.optimizer_module = self.optimizer

    def processing(self):
        if self.args.run_type == 0:
            self.train()
        elif self.args.run_type == 1:
            self.val(self.args.load_epoch)
        elif self.args.run_type == 2:
            self.plot(self.args.load_epoch)
        else:
            raise ValueError('Do not Exist This Processing')

    def train(self):
        print('Start training!')
        self.model.train(mode=True)

        if self.args.pretrained:
            epoch_range = range(self.args.load_epoch, self.args.max_epoch)
        else:
            epoch_range = range(self.args.max_epoch)

        iter = 0
        step = 0
        current_lr = self.args.lr
        loss_recorder = {
            'cls': 0,
            'sum': 0,
        }
        for epoch in epoch_range:
            for num, sample in enumerate(self.train_data_loader):
                iter += 1
                id_x = [i['id_x'] for i in sample]
                id_y = [i['id_y'] for i in sample]
                code_x = [i['code_x'] for i in sample]
                code_y = [i['code_y'] for i in sample]
                labels = [i['labels'] for i in sample]

                labels = torch.FloatTensor(labels).to(self.device)
                # self.model.hidden = self.model.init_hidden()

                # start = torch.cuda.Event(enable_timing=True)
                # end = torch.cuda.Event(enable_timing=True)
                # start.record()
                output = self.model(code_x, code_y)
                # end.record()
                # torch.cuda.synchronize()
                # print(start.elapsed_time(end))

                cls_loss = self.loss_BCE(output, labels)
                #cls_loss = self.loss_focal(output, labels)
                total_loss = cls_loss 
                total_loss = total_loss / self.args.batch_size

                loss_recorder['cls'] += cls_loss.item()
                loss_recorder['sum'] += total_loss.item()

                total_loss.backward()

                if iter % self.args.batch_size == 0:
                    step += 1
                    print('Epoch: {}/{}, Iter: {:02d}, Lr: {:.6f}'.format(
                        epoch + 1,
                        self.args.max_epoch,
                        step,
                        current_lr), end=' ')
                    for k, v in loss_recorder.items():
                        print('Loss_{}: {:.4f}'.format(k, v / self.args.batch_size), end=' ')
                        loss_recorder[k] = 0

                    print()

                    self.optimizer_module.step()
                    self.optimizer_module.zero_grad()

                # if iter % 1000 == 0:
                    # out_dir = './ckpt/' + str(self.args.lang) + '/' + str(self.args.model_id) + '/' + str(
                        # epoch + 1) + '.pkl'
                    # torch.save(self.model.state_dict(), out_dir)
                    # self.model.eval()
                    # self.eval_code(epoch + 1, self.test_data_loader, self.args, self.model, self.device)
                    # self.model.train()

            if (epoch + 1) % self.args.save_interval == 0:
                out_dir = './ckpt/' + str(self.args.lang) + '/' + str(self.args.model_id) + '/' + str(
                    epoch + 1) + '.pkl'
                torch.save(self.model.state_dict(), out_dir)
                self.model.eval()
                self.eval_code(epoch + 1, self.test_data_loader, self.args, self.model, self.device)
                self.model.train()


    def val(self, epoch):
        print('Start testing!')
        self.model.eval()
        self.eval_code(epoch, self.test_data_loader, self.args, self.model, self.device)
        print('Finish testing!')


    def plot(self, epoch):
        print('Start testing!')
        self.model.eval()
        self.plot_code(epoch, self.test_data_loader, self.args, self.model, self.device)
        print('Finish testing!')


    def eval_code(self, epoch, dataloader, args, model, device):
        print('Start Evaluation!')

        predicts = []
        scores = []
        labels = []
        trues = []
        total = 0.0

        for num, sample in enumerate(dataloader):
            if (num + 1) % 500 == 0:
                print('Testing test data point %d of %d' % (num + 1, len(dataloader)))

            id_x = [i['id_x'] for i in sample]
            id_y = [i['id_y'] for i in sample]
            code_x = [i['code_x'] for i in sample]
            code_y = [i['code_y'] for i in sample]
            label = [i['labels'] for i in sample]
            label = torch.FloatTensor(label).to(self.device)

            with torch.no_grad():
                output = self.model(code_x, code_y)
                predicted = (output.squeeze(-1).data > 0.5).cpu().numpy()

                scores.append(np.squeeze(output.cpu().data.numpy(), axis=0))
                labels.append(np.squeeze(label.cpu().data.numpy(), axis=0))

                predicts.extend(predicted)
                trues.extend(label.cpu().numpy())
                total += 1

            # if predicted != labels.cpu().numpy():
            #     if labels.cpu().numpy() == 0:
            #         unsimilar_false.append((id_x, id_y, output.squeeze(-1).data))
            #     else:
            #         similar_false.append((id_x, id_y, output.squeeze(-1).data))

        # import pdb; pdb.set_trace()
        # scores = np.array(scores)
        # labels = np.array(labels)

        # np.save('att_scores5.npy', scores)
        # np.save('att_labels5.npy', labels)

        # import pdb; pdb.set_trace()
        # with open("unsimilar_false.txt", "w") as f:
        #     for pair in unsimilar_false:
        #         f.write(str(pair[0][0]) + ',' + str(pair[1][0]) + ',' + str(pair[2][0]))
        #         f.write('\r\n')

        # with open("similar_false.txt", "w") as f:
        #     for pair in similar_false:
        #         f.write(str(pair[0][0]) + ',' + str(pair[1][0]) + ',' + str(pair[2][0]))
        #         f.write('\r\n')

        #pdb.set_trace()
        # if args.lang == 'java':
            # weights = [0, 0.005, 0.001, 0.002, 0.010, 0.982]
            # p, r, f, _ = precision_recall_fscore_support(trues, predicts, average='binary')
            # precision += weights[t] * p
            # recall += weights[t] * r
            # f1 += weights[t] * f
            # print("Type-" + str(t) + ": " + str(p) + " " + str(r) + " " + str(f))
        # else:
            # precision, recall, f1, _ = precision_recall_fscore_support(trues, predicts, average='binary')
        precision, recall, f1, _ = precision_recall_fscore_support(trues, predicts, average='binary')
        write_to_file(args, precision, recall, f1, epoch)
        print("Total testing results(P,R,F1):%.3f, %.3f, %.3f" % (precision, recall, f1))


    def plot_code(self, epoch, dataloader, args, model, device):
        print('Start Evaluation!')

        outputs = []
        trues = []

        for num, sample in enumerate(dataloader):
            if (num + 1) % 500 == 0:
                print('Testing test data point %d of %d' % (num + 1, len(dataloader)))

            code_x = [i['code_x'] for i in sample]
            code_y = [i['code_y'] for i in sample]
            labels = [i['labels'] for i in sample]
            labels = torch.FloatTensor(labels).to(self.device)

            with torch.no_grad():
                output = self.model(code_x, code_y)
                output = output.squeeze(-1).data.cpu().numpy()
                outputs.extend(output)
                trues.extend(labels.cpu().numpy())
        # pos_out = [outputs[i] for i in range(len(trues)) if trues[i]==1.0]
        # neg_out = [outputs[i] for i in range(len(trues)) if trues[i]==0.0]

        out = np.stack((np.asarray(outputs), np.asarray(trues)), 0)
        np.save('unalign_out.npy', out)

        #import pdb; pdb.set_trace()
        # pos_count, pos_bin = np.histogram(np.asarray(pos_out), bins=100, range=[0.0, 1.0])
        # neg_count, neg_bin = np.histogram(np.asarray(neg_out), bins=100, range=[0.0, 1.0])
        # pos_bin = [(pos_bin[i] + pos_bin[i+1]) / 2 for i in range(len(pos_bin)-1)]
        # neg_bin = [(neg_bin[i] + neg_bin[i+1]) / 2 for i in range(len(neg_bin)-1)]
        # pos_out = np.stack((pos_count, pos_bin), 0)
        # neg_out = np.stack((neg_count, neg_bin), 0)
        # np.save('unalign_pos.npy', pos_out)
        # np.save('unalign_neg.npy', neg_out)
        # plt.hist(pos_out, 100, range=[0.0, 1.0], align='mid')
        # for rect in ax.patches:
            # height = rect.get_height()
            # ax.annotate(f'{int(height)}', xy=(rect.get_x()+rect.get_width()/2, height), 
                        # xytext=(0, 5), textcoords='offset points', ha='center', va='bottom') 
        # ax = fig.add_subplot(122)
        # plt.hist(neg_out, 100, range=[0.0, 1.0], align='mid')
        # for rect in ax.patches:
            # height = rect.get_height()
            # ax.annotate(f'{int(height)}', xy=(rect.get_x()+rect.get_width()/2, height), 
                        # xytext=(0, 5), textcoords='offset points', ha='center', va='bottom') 
        # plt.savefig('1.jpg')
        # plt.close()

def write_to_file(args, p, r, f, epoch):
    file_folder = './ckpt/' + args.lang + '/' + str(args.model_id) + '/'
    file_name = args.lang + '-results.log'
    fid = open(file_folder + file_name, 'a+')
    string_to_write = str(epoch)
    string_to_write += ' precision ' + '%.3f' % p
    string_to_write += ' recall ' + '%.3f' % r
    string_to_write += ' f1 ' + '%.3f' % f
    fid.write(string_to_write + '\n')
    fid.close()
