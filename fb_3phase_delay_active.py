__author__ = 'yihanjiang'

import argparse
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

    parser.add_argument('-learning_rate', type = float, default=0.001)
    parser.add_argument('-clip_norm', type = float, default=1.0)
    parser.add_argument('-batch_size', type=int, default=100)
    parser.add_argument('-num_epoch', type=int, default=1)

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')


    parser.add_argument('-block_len', type=int, default=50)
    parser.add_argument('-num_block', type=int, default=500)

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

    # not functional
    parser.add_argument('-act_function', choices=['tanh', 'relu', 'selu', 'elu'], default='tanh')
    parser.add_argument('-optimizer', choices=['sgd', 'adam', 'nadam', 'yihan'], default='adam')
    parser.add_argument('-loss', choices=['mean_absolute_error', 'mean_squared_error',
                                          'binary_crossentropy', 'mse+max_mse',
                                          'max_mse', 'max_bce'], default='mean_squared_error')

    parser.add_argument('-channel_mode', choices=['normalize', 'lazy_normalize', 'tanh'], default='lazy_normalize')



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

        self.mean = torch.nn.Parameter(torch.Tensor(1),requires_grad = req_grad )
        self.std = torch.nn.Parameter(torch.Tensor(1),requires_grad = req_grad )

        self.weight = torch.nn.Parameter(torch.Tensor(args.block_len, args.code_rate),requires_grad = req_grad )

        self.weight.data.uniform_(1.0, 1.0)

        self.mean.data.uniform_(0.0, 0.0)
        self.std.data.uniform_(1.0, 1.0)

    def forward(self, inputs, phase = -1):

        if phase == -1:
            if self.args.zero_padding:
                self.wt   = torch.sqrt(self.weight**2 * ((self.args.block_len+1) * self.args.code_rate) / torch.sum(self.weight**2))
            else:
                self.wt   = torch.sqrt(self.weight**2 * (self.args.block_len * self.args.code_rate) / torch.sum(self.weight**2))
            # print torch.mean(self.weight), torch.std(self.weight)
            res = torch.mul(self.wt, inputs)
        else:
            res = torch.mul(self.wt[:, phase], inputs.view(self.args.batch_size, self.args.block_len))
            res = res.view((self.args.batch_size, self.args.block_len, 1))


        if self.training:
            self.mean = torch.nn.Parameter(torch.mean(res))
            self.std  = torch.nn.Parameter(torch.std(res))


        return (res - self.mean)/self.std


class AE(torch.nn.Module):
    def __init__(self, args):
        super(AE, self).__init__()

        self.args             = args

        # Delayed Encoder

        # Phase 1 can be BD
        self.enc_p1_rnn       = torch.nn.GRU(1, args.enc_num_unit,
                                           num_layers=2, bias=True, batch_first=True,
                                           dropout=0, bidirectional=True)

        self.enc_p1_linear    = torch.nn.Linear(2*args.enc_num_unit, 1)


        # Phase 2 has to be SD
        self.enc_p2_rnn       = torch.nn.GRU(2, args.enc_num_unit,
                                           num_layers=2, bias=True, batch_first=True,
                                           dropout=0, bidirectional=False)

        self.enc_p2_linear    = torch.nn.Linear(args.enc_num_unit, 1)

        # Phase 3 has to be SD
        self.enc_p3_rnn       = torch.nn.GRU(3, args.enc_num_unit,
                                           num_layers=2, bias=True, batch_first=True,
                                           dropout=0, bidirectional=False)

        self.enc_p3_linear    = torch.nn.Linear(args.enc_num_unit, 1)

        self.total_power_reloc = Power_reallocate(args)


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
        self.dec_rnn       = torch.nn.GRU(args.code_rate,  args.dec_num_unit,
                                           num_layers=2, bias=True, batch_first=True,
                                           dropout=0, bidirectional=True)

        self.dec_output    = torch.nn.Linear(2*args.dec_num_unit, 1)

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


    def forward(self, input, fwd_noise, fb_noise):
        ###############################
        # Yihan's phase-wise feedback
        # No feedback affect during block (no causal feedback somehow)
        # ? Phase-wise normalization
        ###############################
        # Phase 1
        input_p1  = input
        x_p1, _   = self.enc_p1_rnn(input_p1)
        x_p1      = F.elu(self.enc_p1_linear(x_p1))

        x_p1      = self.power_constraint(x_p1)

        if not self.args.no_weight_allocation and not self.training:
            x_p1_norm  = self.total_power_reloc(x_p1, 0)
        else:
            x_p1_norm = x_p1

        x_p1_rec  = x_p1_norm + fwd_noise[:,:,0].view(self.args.batch_size, self.args.block_len, 1)

        # FB
        fb_p1, _  = self.enc_fb1_rnn(x_p1_rec)
        fb_p1     = self.enc_act(self.enc_fb1_linear(fb_p1))

        x_p1_fbenc  = self.power_constraint(fb_p1) # Feedback Encoder
        x_p1_fb     = x_p1_fbenc  + fb_noise[:,:, 0].view(self.args.batch_size, self.args.block_len, 1)


        # Phase 2
        input_p2  = torch.cat([input, x_p1_fb], dim=2)

        x_p2, _   = self.enc_p2_rnn(input_p2)
        x_p2      = F.elu(self.enc_p2_linear(x_p2))

        x_p2      = self.power_constraint(x_p2)

        if not self.args.no_weight_allocation and not self.training:
            x_p2_norm  = self.total_power_reloc(x_p2, 1)
        else:
            x_p2_norm = x_p2


        x_p2_rec  = x_p2_norm + fwd_noise[:,:,1].view(self.args.batch_size, self.args.block_len, 1)

        # FB
        fb_p2_input = torch.cat([x_p1_rec,x_p2_rec], dim = 2)
        fb_p2, _  = self.enc_fb2_rnn(fb_p2_input)
        fb_p2     = self.enc_act(self.enc_fb2_linear(fb_p2))

        x_p2_fbenc  = self.power_constraint(fb_p2) # Feedback Encoder
        x_p2_fb     = x_p2_fbenc  + fb_noise[:,:, 1].view(self.args.batch_size, self.args.block_len, 1)

        # Phase 3
        input_p3  = torch.cat([input, x_p1_fb, x_p2_fb], dim=2)
        x_p3, _   = self.enc_p3_rnn(input_p3)
        x_p3      = F.elu(self.enc_p3_linear(x_p3))

        x_p3      = self.power_constraint(x_p3)

        if not self.args.no_weight_allocation and not self.training:
            x_p3_norm  = self.total_power_reloc(x_p3, 2)
        else:
            x_p3_norm = x_p3

        x_p3_rec  = x_p3_norm + fwd_noise[:,:,2].view(self.args.batch_size, self.args.block_len, 1)

        # FB
        fb_p3_input = torch.cat([x_p1_rec,x_p2_rec, x_p3_rec], dim = 2)
        fb_p3, _  = self.enc_fb3_rnn(fb_p3_input)
        fb_p3     = self.enc_act(self.enc_fb2_linear(fb_p3))

        x_p3_fbenc  = self.power_constraint(fb_p3) # Feedback Encoder
        x_p3_fb     = x_p3_fbenc  + fb_noise[:,:, 2].view(self.args.batch_size, self.args.block_len, 1)


        if not self.args.no_weight_allocation and self.training:
            codes_original = torch.cat([x_p1,x_p2,x_p3], dim = 2)
            codes_adjust   = self.total_power_reloc(codes_original)
            dec_input      = codes_adjust + fwd_noise
        else:
            dec_input = torch.cat([x_p1_rec,x_p2_rec, x_p3_rec], dim=2)

        # Decoder
        x_dec, _  = self.dec_rnn(dec_input)
        x_dec     = F.sigmoid(self.dec_output(x_dec))

        return x_dec

def errors_ber(y_true, y_pred):
    myOtherTensor = np.not_equal(np.round(y_true), np.round(y_pred)).float()
    k = sum(sum(myOtherTensor))/(myOtherTensor.shape[0]*myOtherTensor.shape[1])
    return k

def errors_bler(y_true, y_pred):
    decoded_bits = np.round(y_pred)
    X_test       = np.round(y_true)
    tp0 = (abs(decoded_bits-X_test)).reshape([X_test.shape[0],X_test.shape[1]])
    tp0 = tp0.numpy()
    bler_err_rate = sum(np.sum(tp0,axis=1)>0)*1.0/(X_test.shape[0])
    return bler_err_rate


def main():

    args = get_args()
    print args

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


    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    test_ratio = 1
    num_train_block, num_test_block = args.num_block, args.num_block/test_ratio

    my_train_snr = args.train_snr
    my_train_sigma = 10**(-my_train_snr*1.0/20)#(this_sigma_low - this_sigma_high) * torch.rand((args.batch_size, args.block_len, args.code_rate)) + this_sigma_high

    print 'Traning snr is', my_train_snr

    my_fb_snr = args.fb_snr
    my_fb_sigma = 10**(-my_fb_snr*1.0/20)#(this_sigma_low - this_sigma_high) * torch.rand((args.batch_size, args.block_len, args.code_rate)) + this_sigma_high

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
                fb_noise   = my_fb_sigma * torch.randn((args.batch_size, args.block_len, args.code_rate), dtype=torch.float)

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


    def test():
        model.eval()
        torch.manual_seed(random.randint(0,1000))

        snr_interval = (args.snr_test_end - args.snr_test_start)* 1.0 /  (args.snr_points-1)
        snrs = [snr_interval* item + args.snr_test_start for item in range(args.snr_points)]

        print('0.0 FWD, with FB SNRS', snrs)
        sigmas = [snr_db2sigma(item) for item in snrs]

        num_train_block =  args.num_block

        fwd_sigma = 1.0

        for sigma, this_snr in zip(sigmas, snrs):
            test_ber, test_bler = .0, .0
            with torch.no_grad():
                num_test_batch = int(num_train_block/(args.batch_size*test_ratio))
                for batch_idx in range(num_test_batch):
                    if args.zero_padding:
                        X_test     = torch.randint(0, 2, (args.batch_size, args.block_len, 1), dtype=torch.float)
                        X_test     = torch.cat([X_test, torch.zeros(args.batch_size, 1, 1)], dim=1)
                        fwd_noise  = fwd_sigma*torch.randn((args.batch_size, args.block_len+1, args.code_rate))
                        fb_noise   = sigma * torch.randn((args.batch_size, args.block_len+1, args.code_rate))
                    else:
                        X_test     = torch.randint(0, 2, (args.batch_size, args.block_len, 1), dtype=torch.float)
                        fwd_noise  = fwd_sigma*torch.randn((args.batch_size, args.block_len, args.code_rate))
                        fb_noise   = sigma * torch.randn((args.batch_size, args.block_len, args.code_rate))

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




if __name__ == '__main__':
    main()