from utils.tools import dotdict
from hydroinformer.exp_informer_hydrology import Exp_Informer
import torch


seeds = [110, 111, 222, 333, 444, 555, 666, 777, 888, 999]

for seed in seeds:

    torch.manual_seed(seed)

    args = dotdict()

    args.model = 'informerD' # model of experiment, options: [informerD (only dynamic forcings), informerDS (dynamic and static forcings) with a new FC layer, informerDSI (dynamic + static) with original projection layer]
    #args.model = 'informerDS' # model of experiment, options: [informerD (only dynamic forcings), informerDS (dynamic and static forcings) with a new FC layer, informerDSI (dynamic + static) with original projection layer]
    #args.model = 'informerDSI' # model of experiment, options: [informerD (only dynamic forcings), informerDS (dynamic and static forcings) with a new FC layer, informerDSI (dynamic + static) with original projection layer]

    # camels_ds (static + dynamic) / camels_d (only dynamic)

    if args.model == 'informerD':
        args.data = 'camels_d'
    elif args.model == 'informerDS' or args.model == 'informerDSI':
        args.data = 'camels_ds'

    
    args.root_path = './dataset/' # root path of data file
    args.data_path_dynamic = 'dynamic.nc' # data file
    args.data_path_static = 'static.csv' # data file


    args.features = 'MS' # forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate
    args.target = 'q_obs' # target feature in S or MS task
    args.freq = 'h' # freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h
    args.checkpoints = './informer_checkpoints' # location of model checkpoints

    args.seq_len = 96 # input sequence length of Informer encoder
    args.label_len = 48 # start token length of Informer decoder
    args.pred_len = 1 # prediction sequence length
    # Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]

    args.enc_in = 12 # encoder input size
    args.dec_in = 12 # decoder input size
    args.c_out = 12 # output size
    args.d_static = 26 # static input size
    args.factor = 5 # probsparse attn factor
    if args.model == 'informerDSI':
        args.d_model = args.d_static # dimension of model
    else:
        args.d_model = 512 # dimension of model
    args.n_heads = 8 # num of heads
    args.e_layers = 2 # num of encoder layers
    args.d_layers = 1 # num of decoder layers
    args.d_ff = 2048 # dimension of fcn in model
    args.dropout = 0.05 # dropout
    args.attn = 'prob' # attention used in encoder, options:[prob, full]
    args.embed = 'timeF' # time features encoding, options:[timeF, fixed, learned]
    args.activation = 'gelu' # activation
    args.distil = True # whether to use distilling in encoder
    args.output_attention = False # whether to output attention in ecoder
    args.mix = True
    args.padding = 0
    args.freq = 'h'

    args.batch_size = 32 
    args.learning_rate = 0.0001
    args.loss = 'mse'
    args.lradj = 'type1'
    args.use_amp = False # whether to use automatic mixed precision training

    args.num_workers = 0
    args.itr = 1
    args.train_epochs = 2
    args.patience = 3
    args.des = 'exp'

    args.use_gpu = True if torch.cuda.is_available() else False
    args.gpu = 0

    args.use_multi_gpu = False
    args.devices = '0,1,2,3'

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ','')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    args.detail_freq = args.freq
    args.freq = args.freq[-1:]

    print('Args in experiment:')
    print(args)
    Exp = Exp_Informer
    for ii in range(args.itr):
        # setting record of experiments
        setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}_seed{}'.format(args.model, args.data, args.features, 
                    args.seq_len, args.label_len, args.pred_len,
                    args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.attn, args.factor, args.embed, args.distil, args.mix, args.des, ii,seed)

        # set experiments
        exp = Exp(args)
        
        # train
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)
        
        # test
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)

        torch.cuda.empty_cache()
