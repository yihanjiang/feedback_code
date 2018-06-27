__author__ = 'yihanjiang'

# Adding the *Receiver Encoding*
import argparse
import math
import random
import torch
import torch.optim as optim

import numpy as np
import torch.nn.functional as F

def snr_db2sigma(train_snr):
    return 10**(-train_snr*1.0/20)

def get_args():
    ################################
    # Setup Parameters and get args
    ################################
    parser = argparse.ArgumentParser()

    parser.add_argument('-init_nw_weight', type=str, default='default')
    parser.add_argument('-code_rate', type=int, default=3)

    parser.add_argument('-learning_rate', type = float, default=0.01)
    parser.add_argument('-batch_size', type=int, default=100)
    parser.add_argument('-num_epoch', type=int, default=10)

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    parser.add_argument('-block_len', type=int, default=50)
    parser.add_argument('-num_block', type=int, default=5000)

    parser.add_argument('-enc_num_layer', type=int, default=2)
    parser.add_argument('-dec_num_layer', type=int, default=2)
    parser.add_argument('-fb_num_layer',  type=int, default=2)
    parser.add_argument('-enc_num_unit',  type=int, default=50)
    parser.add_argument('-dec_num_unit',  type=int, default=50)
    parser.add_argument('-fb_num_unit',   type=int, default=50)


    parser.add_argument('-train_snr', type=float, default= 0.0)
    parser.add_argument('-fb_snr', type=float, default= 0.0)

    parser.add_argument('-snr_test_start', type=float, default=-1.0)
    parser.add_argument('-snr_test_end', type=float, default=2.0)
    parser.add_argument('-snr_points', type=int, default=4)

    parser.add_argument('-channel_mode', choices=['normalize', 'lazy_normalize', 'tanh'], default='lazy_normalize')

    parser.add_argument('-fb_dropout', choices=['0.0', '0.1', '0.2', '0.5', 'every_two'], default='0.5')

    parser.add_argument('-enc_act', choices=['tanh', 'selu', 'relu', 'elu', 'sigmoid'], default='elu')

    parser.add_argument('--zero_padding', action='store_true', default=False,
                        help='enable zero padding')

    parser.add_argument('--no_weight_allocation', action='store_true', default=False,
                        help='enable power allocation')

    args = parser.parse_args()

    return args

class Power_reallocate(torch.nn.Module):
    def __init__(self, args):
        super(Power_reallocate, self).__init__()
        self.args = args

        req_grad = False if args.no_weight_allocation else True
        if args.zero_padding:
            self.weight = torch.nn.Parameter(torch.Tensor(args.block_len+1, args.code_rate),requires_grad = req_grad )
        else:
            self.weight = torch.nn.Parameter(torch.Tensor(args.block_len, args.code_rate),requires_grad = req_grad )
        self.weight.data.uniform_(1.0, 1.0)

    def forward(self, inputs):

        if args.zero_padding:
            self.wt   = torch.sqrt(self.weight**2 * ((args.block_len+1) * args.code_rate) / torch.sum(self.weight**2))
        else:
            self.wt   = torch.sqrt(self.weight**2 * (args.block_len * args.code_rate) / torch.sum(self.weight**2))
        # print torch.mean(self.weight), torch.std(self.weight)
        res = torch.mul(self.wt, inputs)
        # print wt[0][0], wt[-1][0],wt[0][1], wt[-1][1],  wt[0][2], wt[-1][2]
        # print torch.mean(wt), torch.std(wt)
        return res


class AE(torch.nn.Module):
    def __init__(self, args):
        super(AE, self).__init__()

        self.args             = args

        # Encoder
        self.enc_p1_rnn_fwd   = torch.nn.GRU(2, args.enc_num_unit,
                                           num_layers=args.enc_num_layer, bias=True, batch_first=True,
                                           dropout=0, bidirectional=False) # Raw bits & Immediate Feedback

        self.enc_p1_rnn_bwd   = torch.nn.GRU(1, args.enc_num_unit,
                                           num_layers=args.enc_num_layer, bias=True, batch_first=True,
                                           dropout=0, bidirectional=False) # Raw bits

        self.enc_p1_linear    = torch.nn.Linear(2*args.enc_num_unit, 1)

        self.enc_p2_rnn_fwd   = torch.nn.GRU(3, args.enc_num_unit,
                                           num_layers=2, bias=True, batch_first=True,
                                           dropout=0, bidirectional=False) # Raw bits & Phase1 feedback & Immediate Feedback
        self.enc_p2_rnn_bwd   = torch.nn.GRU(2, args.enc_num_unit,
                                           num_layers=2, bias=True, batch_first=True,
                                           dropout=0, bidirectional=False) # Raw bits & Phase1 feedback

        self.enc_p2_linear    = torch.nn.Linear(2*args.enc_num_unit, 1) # Generate two codewords per cell.

        self.enc_p3_rnn_fwd   = torch.nn.GRU(4, args.enc_num_unit,
                                           num_layers=2, bias=True, batch_first=True,
                                           dropout=0, bidirectional=False) # Raw bits & Phase1, 2 feedback &Immediate Feedback

        self.enc_p3_rnn_bwd   = torch.nn.GRU(3, args.enc_num_unit,
                                           num_layers=2, bias=True, batch_first=True,
                                           dropout=0, bidirectional=False) # Raw bits & Phase1,2  feedback

        self.enc_p3_linear    = torch.nn.Linear(2*args.enc_num_unit, 1) # Generate two codewords per cell.

        # Feedback Encoder
        self.enc_fb1_rnn       = torch.nn.GRU(1, args.fb_num_unit,
                                           num_layers=2, bias=True, batch_first=True,
                                           dropout=0, bidirectional=False) # Just received bits

        self.enc_fb1_linear    = torch.nn.Linear(args.fb_num_unit, 1)

        self.enc_fb2_rnn       = torch.nn.GRU(2, args.fb_num_unit,
                                           num_layers=2, bias=True, batch_first=True,
                                           dropout=0, bidirectional=False) # received bits & Phase 1 received bits

        self.enc_fb2_linear    = torch.nn.Linear(args.fb_num_unit, 1)

        self.enc_fb3_rnn       = torch.nn.GRU(3, args.fb_num_unit,
                                           num_layers=2, bias=True, batch_first=True,
                                           dropout=0, bidirectional=False) # received bits & Phase 1,2 received bits

        self.enc_fb3_linear    = torch.nn.Linear(args.fb_num_unit, 1)
        # Decoder
        self.total_power_reloc = Power_reallocate(args)
        self.dec_rnn           = torch.nn.GRU(args.code_rate,  args.dec_num_unit,
                                           num_layers=2, bias=True, batch_first=True,
                                           dropout=0, bidirectional=True)

        self.dec_output        = torch.nn.Linear(2*args.dec_num_unit, 1)


    # make power constraint with normalization. (to escape tanh's saturation issue, also make it casual)
    def power_constraint(self, inputs, historys = None):
        if self.args.channel_mode == 'normalize':
            this_mean = torch.mean(historys)
            this_std  = torch.std(historys)
            outputs   = (inputs - this_mean)*1.0/this_std

        elif self.args.channel_mode == 'tanh':
            outputs = F.tanh(inputs)

        elif self.args.channel_mode == 'lazy_normalize':
            this_mean = torch.mean(inputs)
            this_std  = torch.std(inputs)
            outputs   = (inputs - this_mean)*1.0/this_std

        else:
            print 'oh no I must make a type'

        return outputs

    def enc_act(self, inputs):
        if self.enc_act == 'tanh':
            return  F.tanh(inputs)
        elif self.enc_act == 'elu':
            return F.elu(inputs)
        elif self.enc_act == 'relu':
            return F.relu(inputs)
        elif self.enc_act == 'selu':
            return F.selu(inputs)
        elif self.enc_act == 'sigmoid':
            return F.sigmoid(inputs)
        else:
            return F.tanh(inputs)

    def fb_dropout(self, fb, idx):
        if self.args.fb_dropout == '0.0':
            pass
        elif self.args.fb_dropout == '0.1':
            fb = F.dropout(fb, p=0.1, training=True)
        elif self.args.fb_dropout == '0.2':
            fb = F.dropout(fb, p=0.2, training=True)
        elif self.args.fb_dropout == '0.5':
            fb = F.dropout(fb, p=0.5, training=True)
        elif self.args.fb_dropout == 'every_two':
            for idx in range(fb.shape[1]):
                if idx%2 == 1:
                    continue
                else:
                    fb[:,idx,:] = 0

        return fb


    def forward(self, input, fwd_noise, fb_noise):
        ###############################
        # half-BD-RNN case
        ###############################

        # encoder part: Phase 1
        #Reverse of copy of numpy array of given tensor

        idx = [i for i in range(input.size(1)-1, -1, -1)]
        idx = torch.LongTensor(idx).to(device)
        input_reverse = input.index_select(1, idx)

        x_bwd_p1_whole, _ = self.enc_p1_rnn_bwd(input_reverse)

        for idx in range(input.shape[1]):
            if idx == 0:
                input_tmp        = torch.cat([input[:,idx,:].view(self.args.batch_size, 1, 1),
                                              torch.zeros((self.args.batch_size, 1, 1)).to(device)], dim=2)
                x_fwd_p1, h_tmp  = self.enc_p1_rnn_fwd(input_tmp)
                x_bwd_p1         = x_bwd_p1_whole[:,input.shape[1] - idx - 1,:].view((args.batch_size, 1, args.enc_num_unit))
                x_tmp_p1         = torch.cat([x_fwd_p1, x_bwd_p1], dim = 2)
                x_tmp_p1         = self.enc_act(self.enc_p1_linear(x_tmp_p1))
                x_p1_history     = x_tmp_p1
            else:
                input_tmp        = torch.cat([input[:,idx,:].view(self.args.batch_size, 1, 1),
                                              fb_tmp.view((self.args.batch_size, 1, 1))], dim=2)

                x_fwd_p1, h_tmp  = self.enc_p1_rnn_fwd(input_tmp, h_tmp)
                x_bwd_p1         = x_bwd_p1_whole[:,input.shape[1] - idx - 1,:].view((args.batch_size, 1, args.enc_num_unit))
                x_tmp_p1         = torch.cat([x_fwd_p1, x_bwd_p1], dim = 2)
                x_tmp_p1         = self.enc_act(self.enc_p1_linear(x_tmp_p1))
                x_p1_history     = torch.cat([x_p1_history, x_tmp_p1], dim = 1)

            x_tmp_p1  = self.power_constraint(x_tmp_p1, x_p1_history)

            if not self.args.no_weight_allocation:
                if not self.training:
                    x_tmp_p1  = x_tmp_p1 * self.total_power_reloc.wt[idx, 0]

            x_p1_rec  = x_tmp_p1  + fwd_noise[:,idx,0].view(self.args.batch_size, 1, 1)

            # FB
            if idx == 0:
                fbenc_x_tmp_p1, fbenc_h_tmp = self.enc_fb1_rnn(x_p1_rec)
                fbenc_x_tmp_p1              = self.enc_act(self.enc_fb1_linear(fbenc_x_tmp_p1))
                fb_p1_history               = fbenc_x_tmp_p1
            else:
                fbenc_x_tmp_p1, fbenc_h_tmp = self.enc_fb1_rnn(x_p1_rec,fbenc_h_tmp)
                fbenc_x_tmp_p1              = self.enc_act(self.enc_fb1_linear(fbenc_x_tmp_p1))
                fb_p1_history               = torch.cat([fb_p1_history, fbenc_x_tmp_p1], dim=1)

            x_p1_fbenc  = self.power_constraint(fbenc_x_tmp_p1, fb_p1_history) # Feedback Encoder

            x_p1_fbenc  = self.fb_dropout(x_p1_fbenc, idx)

            x_p1_fb     = x_p1_fbenc  + fb_noise[:,idx, 0].view(self.args.batch_size, 1, 1)
            fb_tmp      = x_p1_fb




            if idx == 0:
                p1_code= x_tmp_p1
                p1_rec = x_p1_rec
                p1_fb  = x_p1_fb
            else:
                p1_code = torch.cat([p1_code,x_tmp_p1 ], dim = 1)
                p1_rec = torch.cat([p1_rec,x_p1_rec ], dim = 1)
                p1_fb  = torch.cat([p1_fb, x_p1_fb],   dim = 1)

        # encoder part: Phase 2
        idx = [i for i in range(p1_fb.size(1)-1, -1, -1)]
        idx = torch.LongTensor(idx).to(device)
        p1_fb_reverse = p1_fb.index_select(1, idx)

        x_bwd_p2_whole, _ = self.enc_p2_rnn_bwd(torch.cat([input_reverse,p1_fb_reverse], dim = 2))

        for idx in range(input.shape[1]):
            # ENC
            if idx == 0:
                input_tmp        = torch.cat([input[:,idx,:].view(self.args.batch_size, 1, 1),
                                              p1_fb[:,idx,:].view(self.args.batch_size, 1, 1),
                                              torch.zeros((self.args.batch_size, 1, 1)).to(device)], dim=2)

                x_fwd_p2, h_tmp  = self.enc_p2_rnn_fwd(input_tmp)
                x_bwd_p2         = x_bwd_p2_whole[:,input.shape[1] - idx - 1,:].view((args.batch_size, 1, args.enc_num_unit))
                x_tmp_p2         = torch.cat([x_fwd_p2, x_bwd_p2], dim = 2)
                x_tmp_p2         = self.enc_act(self.enc_p2_linear(x_tmp_p2))
                x_p2_history     = x_tmp_p2

            else:
                input_tmp        = torch.cat([input[:,idx,:].view(self.args.batch_size, 1, 1),
                                              p1_fb[:,idx,:].view(self.args.batch_size, 1, 1),
                                              fb_tmp.view((self.args.batch_size, 1, 1))], dim=2)

                x_fwd_p2, h_tmp  = self.enc_p2_rnn_fwd(input_tmp, h_tmp)
                x_bwd_p2         = x_bwd_p2_whole[:,input.shape[1] - idx - 1,:].view((args.batch_size, 1, args.enc_num_unit))
                x_tmp_p2         = torch.cat([x_fwd_p2, x_bwd_p2], dim = 2)
                x_tmp_p2         = self.enc_act(self.enc_p2_linear(x_tmp_p2))
                x_p2_history     = x_tmp_p2

            x_tmp_p2  = self.power_constraint(x_tmp_p2, x_p2_history)

            if not self.args.no_weight_allocation:
                if not self.training:
                    x_tmp_p2  = x_tmp_p2 * self.total_power_reloc.wt[idx, 1]

            x_p2_rec  = x_tmp_p2  + fwd_noise[:,idx, 1].view(self.args.batch_size, 1, 1)

            # FB
            if idx == 0:
                fb_tmp_input_p2             = torch.cat([x_p2_rec, p1_rec[:, idx, 0].view(args.batch_size, 1, 1) ], dim=2)

                fbenc_x_tmp_p2, fbenc_h_tmp = self.enc_fb2_rnn(fb_tmp_input_p2)
                fbenc_x_tmp_p2              = self.enc_act(self.enc_fb2_linear(fbenc_x_tmp_p2))
                fb_p2_history               = fbenc_x_tmp_p2
            else:
                fb_tmp_input_p2             = torch.cat([x_p2_rec, p1_rec[:, idx, 0].view(args.batch_size, 1, 1) ], dim=2)
                fbenc_x_tmp_p2, fbenc_h_tmp = self.enc_fb2_rnn(fb_tmp_input_p2,fbenc_h_tmp)
                fbenc_x_tmp_p2              = self.enc_act(self.enc_fb2_linear(fbenc_x_tmp_p2))
                fb_p2_history               = torch.cat([fb_p2_history, fbenc_x_tmp_p2], dim=1)

            x_p2_fbenc  = self.power_constraint(fbenc_x_tmp_p2, fb_p2_history) # Feedback Encoder

            x_p2_fbenc  = self.fb_dropout(x_p2_fbenc, idx)

            x_p2_fb     = x_p2_fbenc  + fb_noise[:,idx, 1].view(self.args.batch_size, 1, 1)

            fb_tmp      = x_p2_fb

            fb_tmp      = self.fb_dropout(fb_tmp, idx)

            if idx == 0:
                p2_code= x_tmp_p2
                p2_rec = x_p2_rec
                p2_fb  = x_p2_fb
            else:
                p2_code = torch.cat([p2_code,x_tmp_p2 ], dim = 1)
                p2_rec = torch.cat([p2_rec,x_p2_rec ], dim = 1)
                p2_fb  = torch.cat([p2_fb, x_p2_fb],   dim = 1)

        # encoder part: Phase 3

        idx = [i for i in range(p2_fb.size(1)-1, -1, -1)]
        idx = torch.LongTensor(idx).to(device)
        p2_fb_reverse = p2_fb.index_select(1, idx)

        x_bwd_p3_whole, _ = self.enc_p3_rnn_bwd(torch.cat([input_reverse,p1_fb_reverse, p2_fb_reverse], dim = 2))
        for idx in range(input.shape[1]):
            # ENC
            if idx == 0:
                input_tmp        = torch.cat([input[:,idx,:].view(self.args.batch_size, 1, 1),
                                              p1_fb[:,idx,:].view(self.args.batch_size, 1, 1),
                                              p2_fb[:,idx,:].view(self.args.batch_size, 1, 1),
                                              torch.zeros((self.args.batch_size, 1, 1)).to(device)], dim=2)

                x_fwd_p3, h_tmp  = self.enc_p3_rnn_fwd(input_tmp)
                x_bwd_p3         = x_bwd_p3_whole[:,input.shape[1] - idx - 1,:].view((args.batch_size, 1, args.enc_num_unit))
                x_tmp_p3         = torch.cat([x_fwd_p3, x_bwd_p3], dim = 2)
                x_tmp_p3         = self.enc_act(self.enc_p3_linear(x_tmp_p3))
                x_p3_history     = x_tmp_p3

            else:
                input_tmp        = torch.cat([input[:,idx,:].view(self.args.batch_size, 1, 1),
                                              p1_fb[:,idx,:].view(self.args.batch_size, 1, 1),
                                              p2_fb[:,idx,:].view(self.args.batch_size, 1, 1),
                                              fb_tmp.view((self.args.batch_size, 1, 1))], dim=2)

                x_fwd_p3, h_tmp  = self.enc_p3_rnn_fwd(input_tmp, h_tmp)
                x_bwd_p3         = x_bwd_p3_whole[:,input.shape[1] - idx - 1,:].view((args.batch_size, 1, args.enc_num_unit))
                x_tmp_p3         = torch.cat([x_fwd_p3, x_bwd_p3], dim = 2)
                x_tmp_p3         = self.enc_act(self.enc_p3_linear(x_tmp_p3))
                x_p3_history     = x_tmp_p3

            x_tmp_p3  = self.power_constraint(x_tmp_p3, x_p3_history)

            if not self.args.no_weight_allocation:
                if not self.training:
                    x_tmp_p3  = x_tmp_p3 * self.total_power_reloc.wt[idx, 2]

            x_p3_rec  = x_tmp_p3  + fwd_noise[:,idx, 2].view(self.args.batch_size, 1, 1)

            # FB
            if idx == 0:
                fb_tmp_input_p3             = torch.cat([x_p3_rec,
                                                         p1_rec[:, idx, 0].view(args.batch_size, 1, 1),
                                                         p2_rec[:, idx, 0].view(args.batch_size, 1, 1)], dim=2)

                fbenc_x_tmp_p3, fbenc_h_tmp = self.enc_fb3_rnn(fb_tmp_input_p3)
                fbenc_x_tmp_p3              = self.enc_act(self.enc_fb3_linear(fbenc_x_tmp_p3))
                fb_p3_history               = fbenc_x_tmp_p3
            else:
                fb_tmp_input_p3             = torch.cat([x_p3_rec,
                                                         p1_rec[:, idx, 0].view(args.batch_size, 1, 1),
                                                         p2_rec[:, idx, 0].view(args.batch_size, 1, 1)], dim=2)
                fbenc_x_tmp_p3, fbenc_h_tmp = self.enc_fb3_rnn(fb_tmp_input_p3, fbenc_h_tmp)
                fbenc_x_tmp_p3              = self.enc_act(self.enc_fb3_linear(fbenc_x_tmp_p3))
                fb_p3_history               = torch.cat([fb_p3_history, fbenc_x_tmp_p3], dim=1)

            x_p3_fbenc  = self.power_constraint(fbenc_x_tmp_p3, fb_p3_history) # Feedback Encoder

            x_p3_fbenc  = self.fb_dropout(x_p3_fbenc, idx)

            x_p3_fb     = x_p3_fbenc  + fb_noise[:,idx, 2].view(self.args.batch_size, 1, 1)

            fb_tmp      = x_p3_fb

            if idx == 0:
                p3_code = x_tmp_p3
                p3_rec = x_p3_rec
                p3_fb  = x_p3_fb
            else:
                p3_code = torch.cat([p3_code,x_tmp_p3 ], dim = 1)
                p3_rec = torch.cat([p3_rec,x_p3_rec ], dim = 1)
                p3_fb  = torch.cat([p3_fb, x_p3_fb],   dim = 1)

        if not self.args.no_weight_allocation:
            if self.training:
                codes_original = torch.cat([p1_code,p2_code,p3_code], dim = 2)
                codes_adjust   = self.total_power_reloc(codes_original)
                dec_input      = codes_adjust + fwd_noise
            else:
                dec_input = torch.cat([p1_rec,p2_rec, p3_rec], dim=2)
        else:
            dec_input = torch.cat([p1_rec,p2_rec, p3_rec], dim=2)


        x_dec, _  = self.dec_rnn(dec_input)
        x_dec     = F.sigmoid(self.dec_output(x_dec))

        return x_dec


###### MAIN
args = get_args()
print args


def errors_ber(y_true, y_pred):

    if args.zero_padding:
        t1 = np.round(y_true[:,:-1,:])
        t2 = np.round(y_pred[:,:-1,:])
    else:
        t1 = np.round(y_true[:,:,:])
        t2 = np.round(y_pred[:,:,:])

    myOtherTensor = np.not_equal(t1, t2).float()
    k = sum(sum(myOtherTensor))/(myOtherTensor.shape[0]*myOtherTensor.shape[1])
    return k

def errors_bler(y_true, y_pred):

    if args.zero_padding:
        t1 = np.round(y_true[:,:-1,:])
        t2 = np.round(y_pred[:,:-1,:])
    else:
        t1 = np.round(y_true[:,:,:])
        t2 = np.round(y_pred[:,:,:])

    decoded_bits = t1
    X_test       = t2
    tp0 = (abs(decoded_bits-X_test)).reshape([X_test.shape[0],X_test.shape[1]])
    tp0 = tp0.numpy()
    bler_err_rate = sum(np.sum(tp0,axis=1)>0)*1.0/(X_test.shape[0])
    return bler_err_rate

identity = str(np.random.random())[2:8]
print '[ID]', identity

use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

if use_cuda:
    model = AE(args).to(device)
else:
    model = AE(args)

print model

if args.init_nw_weight == 'default':
    pass
else:
    model = torch.load(args.init_nw_weight)
    model.args = args

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)

test_ratio = 1
num_train_block, num_test_block = args.num_block, args.num_block/test_ratio


my_train_snr = args.train_snr
my_train_sigma = 10**(-my_train_snr*1.0/20)#(this_sigma_low - this_sigma_high) * torch.rand((args.batch_size, args.block_len, args.code_rate)) + this_sigma_high

print 'Traning snr is', my_train_snr

my_fb_snr = args.fb_snr
my_fb_sigma = 10**(-my_fb_snr*1.0/20)#(this_sigma_low - this_sigma_high) * torch.rand((args.batch_size, args.block_len, args.code_rate)) + this_sigma_high

if my_fb_snr > 100:
    my_fb_sigma = 0.00


print 'FB sigma is', my_fb_sigma


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx in range(int(num_train_block/args.batch_size)):
        if args.zero_padding:
            X_train    = torch.randint(0, 2, (args.batch_size, args.block_len, 1), dtype=torch.float)
            X_train    = torch.cat([X_train, torch.zeros(args.batch_size, 1, 1)], dim=1)

            this_sigma = my_train_sigma
            fwd_noise  = this_sigma * torch.randn((args.batch_size, args.block_len+1, args.code_rate), dtype=torch.float)
            fb_noise   = my_fb_sigma * torch.randn((args.batch_size, args.block_len+1, args.code_rate), dtype=torch.float)
        else:
            X_train    = torch.randint(0, 2, (args.batch_size, args.block_len, 1), dtype=torch.float)
            this_sigma = my_train_sigma
            fwd_noise  = this_sigma * torch.randn((args.batch_size, args.block_len, args.code_rate), dtype=torch.float)
            fb_noise   = my_fb_sigma * torch.randn((args.batch_size, args.block_len+1, args.code_rate), dtype=torch.float)

        # use GPU
        X_train, fwd_noise, fb_noise = X_train.to(device), fwd_noise.to(device), fb_noise.to(device)

        optimizer.zero_grad()
        output = model(X_train, fwd_noise, fb_noise)

        loss = F.binary_cross_entropy(output, X_train)
        loss.backward()
        train_loss += loss.item()

        optimizer.step()
        if batch_idx % 1000 == 0:
            print('Train Epoch: {} [{}/{} Loss: {:.6f}'.format(
                epoch, batch_idx, num_train_block/args.batch_size, loss.item()))

    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss /(num_train_block/args.batch_size)) )
    print torch.min(model.total_power_reloc.wt[:-1,:]), torch.max(model.total_power_reloc.wt)
    print torch.mean(model.total_power_reloc.wt), torch.std(model.total_power_reloc.wt)
    print model.total_power_reloc.wt.shape

def test():
    model.eval()
    torch.manual_seed(random.randint(0,1000))

    snr_interval = (args.snr_test_end - args.snr_test_start)* 1.0 /  (args.snr_points-1)
    snrs = [snr_interval* item + args.snr_test_start for item in range(args.snr_points)]
    print('SNRS', snrs)
    sigmas = [snr_db2sigma(item) for item in snrs]

    num_train_block =  args.num_block

    for sigma, this_snr in zip(sigmas, snrs):
        test_ber, test_bler = .0, .0
        with torch.no_grad():
            num_test_batch = int(num_train_block/(args.batch_size*test_ratio))
            for batch_idx in range(num_test_batch):
                if args.zero_padding:
                    X_test     = torch.randint(0, 2, (args.batch_size, args.block_len, 1), dtype=torch.float)
                    X_test     = torch.cat([X_test, torch.zeros(args.batch_size, 1, 1)], dim=1)
                    fwd_noise  = sigma*torch.randn((args.batch_size, args.block_len+1, args.code_rate))
                    fb_noise   = my_fb_sigma * torch.randn((args.batch_size, args.block_len+1, args.code_rate))
                else:
                    X_test     = torch.randint(0, 2, (args.batch_size, args.block_len, 1), dtype=torch.float)
                    fwd_noise  = sigma*torch.randn((args.batch_size, args.block_len, args.code_rate))
                    fb_noise   = my_fb_sigma * torch.randn((args.batch_size, args.block_len, args.code_rate))

                # use GPU
                X_test, fwd_noise, fb_noise = X_test.to(device), fwd_noise.to(device), fb_noise.to(device)

                X_hat_test = model(X_test, fwd_noise, fb_noise)
                test_ber  += errors_ber(X_hat_test,X_test)
                test_bler += errors_bler(X_hat_test,X_test)

        test_ber  /= 1.0*num_test_batch
        test_bler /= 1.0*num_test_batch
        print('Test SNR',this_snr ,'with ber ', float(test_ber), 'with bler', float(test_bler))


#PATH='torch_model_791480.pt'
#model=torch.load(PATH)

for epoch in range(1, args.num_epoch + 1):
    train(epoch)
    test()

torch.save(model, './tmp/torch_model_'+identity+'.pt')

print('saved model', './tmp/torch_model_'+identity+'.pt')

