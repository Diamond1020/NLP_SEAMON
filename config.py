import sys
import torch

class Config(object):
    def __init__(self):
        self.data_path = "./data/train.tags.en-fr.en"
        self.level = "bpe"
        self.bpe_word_ratio = 0.7
        self.bpe_vocab_size = 7000
        self.batch_size = 32
        self.d_model = 512
        self.ff_dim = 2048
        self.num = 6
        self.n_heads = 8
        self.max_encoder_len = 80
        self.max_decoder_len = 80
        self.LAS_embed_dim = 512
        self.LAS_hidden_size = 512
        self.num_epochs = 127
        self.lr = 0.00003
        self.gradient_acc_steps = 2
        self.max_norm = 1.0
        self.T_max = 5000
        self.model_no = 1
        self.train = 1
        self.infer = 0

class StyleTransferConfig():
    data_path = './data/yelp/'
    log_dir = 'runs/exp'
    save_path = './data/style_transfer'
    pretrained_embed_path = './embedding/'
    device = torch.device('cuda' if True and torch.cuda.is_available() else 'cpu')
    discriminator_method = 'Multi' # 'Multi' or 'Cond'
    load_pretrained_embed = False
    min_freq = 3
    max_length = 16
    embed_size = 256
    d_model = 256
    h = 4
    num_styles = 2
    num_classes = num_styles + 1 if discriminator_method == 'Multi' else 2
    num_layers = 4
    batch_size = 64
    lr_F = 0.0001
    lr_D = 0.0001
    L2 = 0
    iter_D = 10
    iter_F = 5
    F_pretrain_iter = 500
    log_steps = 5
    eval_steps = 25
    learned_pos_embed = True
    dropout = 0
    drop_rate_config = [(1, 0)]
    temperature_config = [(1, 0)]

    slf_factor = 0.25
    cyc_factor = 0.5
    adv_factor = 1

    inp_shuffle_len = 0
    inp_unk_drop_fac = 0
    inp_rand_drop_fac = 0
    inp_drop_prob = 0
    
    def __init__(self, args):
        self.data_path = args.data_path
        self.num_styles = args.num_classes
        self.batch_size = args.batch_size
        self.max_length = args.max_features_length
        self.d_model = args.d_model
        self.embed_size = args.d_model
        self.h = args.n_heads
        self.lr_D = args.lr_D
        self.lr_F = args.lr_F
        self.num_layers = args.num
        self.num_iters = args.num_iters
        self.checkpoint_Fpath = args.checkpoint_Fpath
        self.checkpoint_Dpath = args.checkpoint_Dpath
        self.eval_steps = args.save_iters
        self.checkpoint_config = args.checkpoint_config
        self.gradient_acc_steps = args.gradient_acc_steps
        self.F_pretrain_iter = self.F_pretrain_iter*self.gradient_acc_steps
        self.train_from_checkpoint = args.train_from_checkpoint
