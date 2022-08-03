import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import random
import transformer
import pdb


class BatchTreeEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, encode_dim, batch_size, use_gpu, pretrained_weight=None):
        super(BatchTreeEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dim = embedding_dim
        self.encode_dim = encode_dim
        self.W_c = nn.Linear(embedding_dim, encode_dim)
        self.activation = F.relu
        self.stop = -1
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.node_list = []
        self.th = torch.cuda if use_gpu else torch
        self.batch_node = None
        self.max_index = vocab_size
        # pretrained  embedding
        if pretrained_weight is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))
            # self.embedding.weight.requires_grad = False

    def create_tensor(self, tensor):
        if self.use_gpu:
            return tensor.cuda()
        return tensor

    def traverse_mul(self, node, batch_index):
        size = len(node)
        if not size:
            return None
        batch_current = self.create_tensor(Variable(torch.zeros(size, self.embedding_dim)))

        index, children_index = [], []
        current_node, children = [], []
        for i in range(size):
            # if node[i][0] is not -1:
                index.append(i)
                current_node.append(node[i][0])
                temp = node[i][1:]
                c_num = len(temp)
                for j in range(c_num):
                    if temp[j][0] is not -1:
                        if len(children_index) <= j:
                            children_index.append([i])
                            children.append([temp[j]])
                        else:
                            children_index[j].append(i)
                            children[j].append(temp[j])

        batch_current = self.W_c(batch_current.index_copy(0, Variable(self.th.LongTensor(index)),
                                                          self.embedding(Variable(self.th.LongTensor(current_node)))))

        for c in range(len(children)):
            zeros = self.create_tensor(Variable(torch.zeros(size, self.encode_dim)))
            batch_children_index = [batch_index[i] for i in children_index[c]]
            tree = self.traverse_mul(children[c], batch_children_index)
            if tree is not None:
                batch_current += zeros.index_copy(0, Variable(self.th.LongTensor(children_index[c])), tree)
        # batch_index = [i for i in batch_index if i is not -1]
        b_in = Variable(self.th.LongTensor(batch_index))
        self.node_list.append(self.batch_node.index_copy(0, b_in, batch_current))
        return batch_current

    def forward(self, x, bs):
        self.batch_size = bs
        self.batch_node = self.create_tensor(Variable(torch.zeros(self.batch_size, self.encode_dim)))
        self.node_list = []
        self.traverse_mul(x, list(range(self.batch_size)))
        self.node_list = torch.stack(self.node_list)
        return torch.max(self.node_list, 0)[0]


class BiDirectionalConv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size, dilation=1):
        super(BiDirectionalConv, self).__init__()
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.padding = (kernel_size - 1) * dilation 
        self.conv1 = nn.Conv1d(self.inp_dim, self.out_dim, kernel_size=kernel_size, dilation=dilation, padding=self.padding)
        self.conv2 = nn.Conv1d(self.inp_dim, self.out_dim, kernel_size=kernel_size, dilation=dilation, padding=self.padding)

    def forward(self, x):
        forward_conv = self.conv1(x)[..., 0:-self.padding]
        backward_conv = self.conv2(torch.flip(x, [-1]))[..., 0:-self.padding]
        # return forward_conv + torch.flip(backward_conv, [-1])
        return torch.cat((forward_conv, torch.flip(backward_conv, [-1])), 1)


class MultiScaleBlock(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(MultiScaleBlock, self).__init__()
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.conv1 = nn.Conv1d(self.inp_dim, self.out_dim, kernel_size=1)
        self.conv2 = BiDirectionalConv(self.inp_dim, self.out_dim // 2, kernel_size=3)
        self.conv3 = BiDirectionalConv(self.inp_dim, self.out_dim // 2, kernel_size=3, dilation=2)
        self.activation = nn.Tanh()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        out = conv1 + conv2 + conv3
        return self.activation(out)


class BatchProgramCC(nn.Module):
    def __init__(self, args, embedding_dim, vocab_size, batch_size, use_gpu=True, pretrained_weight=None):
        super(BatchProgramCC, self).__init__()
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.num_layers = 2

        self.embedding_dim = embedding_dim
        self.input_dim = args.input_dim        
        self.encode_dim1 = args.encode_dim1
        self.encode_dim2 = args.encode_dim2
        self.encoder = BatchTreeEncoder(self.vocab_size, self.embedding_dim, self.input_dim,
                                        self.batch_size, self.use_gpu, pretrained_weight)

         # embedding
        self.embedding_layer = nn.Sequential(
                               MultiScaleBlock(self.input_dim, self.encode_dim1),
                               MultiScaleBlock(self.encode_dim1, self.encode_dim2),
                               )

        self.w_q = nn.Linear(self.encode_dim2, self.encode_dim2, bias=False)
        self.w_k = nn.Linear(self.encode_dim2, self.encode_dim2, bias=False)


        self.hidden2label = nn.Linear(self.encode_dim2 * 2, 1)


    def init_hidden(self):
        if self.use_gpu is True:
            if isinstance(self.bigru, nn.LSTM):
                h0 = Variable(torch.zeros(self.num_layers * 2, self.batch_size, self.encode_dim).cuda())
                c0 = Variable(torch.zeros(self.num_layers * 2, self.batch_size, self.encode_dim).cuda())
                return h0, c0
            return Variable(torch.zeros(self.num_layers * 2, self.batch_size, self.encode_dim)).cuda()
        else:
            return Variable(torch.zeros(self.num_layers * 2, self.batch_size, self.encode_dim))

    def encode(self, x):
        lens = [len(item) for item in x]
        encodes = []
        for i in range(len(x)):
            for j in range(lens[i]):
                encodes.append(x[i][j])
        encodes = self.encoder(encodes, sum(lens))
        conv_out = self.embedding_layer(encodes[None, ...].permute(0,2,1)).permute(0,2,1)

        return conv_out

    def forward(self, x, y):
        x_embed, y_embed = self.encode(x), self.encode(y)

        # sparse reconstruction
        ####################################################
        # yty = torch.einsum('nbd,ncd->nbc', [y_embed, y_embed])
        # eye = yty.new_ones(yty.size(-1)).diag().expand_as(yty)
        # mat_inv1, _ = torch.solve(eye, yty + 0.01 * eye)
        # mat_yx = torch.einsum('nbd,ncd->nbc', [y_embed, x_embed])
        # mat_yx = torch.einsum('nbc,nck->nbk', [mat_inv1, mat_yx])
        # align_x_embed = torch.einsum('nbd,nbk->nkd', [y_embed, mat_yx])

        # xtx = torch.einsum('nbd,ncd->nbc', [x_embed, x_embed])
        # eye = xtx.new_ones(xtx.size(-1)).diag().expand_as(xtx)
        # mat_inv2, _ = torch.solve(eye, xtx + 0.01 * eye)
        # mat_xy = torch.einsum('nbd,ncd->nbc', [x_embed, y_embed])
        # mat_xy = torch.einsum('nbc,nck->nbk', [mat_inv2, mat_xy])        
        # align_y_embed = torch.einsum('nbd,nbk->nkd', [x_embed, mat_xy])

        # attention
        ####################################################
        x_embed_1 = self.w_k(x_embed)
        y_embed_1 = self.w_q(y_embed)
        mat_yx = torch.einsum('ntd,nkd->ntk', [y_embed_1, x_embed_1])
        mat_yx = F.softmax(mat_yx, -1)
        align_y_embed = torch.einsum('ntk,nkd->ntd', [mat_yx, x_embed])

        x_embed_2 = self.w_q(x_embed)
        y_embed_2 = self.w_k(y_embed)
        mat_xy = torch.einsum('ntd,nkd->ntk', [x_embed_2, y_embed_2])
        mat_xy = F.softmax(mat_xy, -1)
        align_x_embed = torch.einsum('ntk,nkd->ntd', [mat_xy, y_embed])

        ####################################################
        x_error = x_embed.max(1)[0] - align_x_embed.max(1)[0]
        y_error = y_embed.max(1)[0] - align_y_embed.max(1)[0]
        ##################################################
        abs_dist = torch.cat((torch.abs(x_error), torch.abs(y_error)), dim=-1)
        y = torch.sigmoid(self.hidden2label(abs_dist))
        return y
