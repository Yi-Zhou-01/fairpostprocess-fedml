from torch import nn, sigmoid
import numpy as np
from torchvision import models 



class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return self.softmax(x)


class MLPAdult(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLPAdult, self).__init__()
        self.layer_1 = nn.Linear(in_features=dim_in, out_features=64)
        self.layer_2 = nn.Linear(in_features=64, out_features=128)
        self.layer_3 = nn.Linear(in_features=128, out_features=64)
        self.layer_4 = nn.Linear(in_features=64, out_features=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.dropout(self.relu(self.layer_1(x)))
        x = self.dropout(self.relu(self.layer_2(x)))
        x = self.dropout(self.relu(self.layer_3(x)))
        network = self.layer_4(x)
        return network
    


class MLPAdult2(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLPAdult2, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features=dim_in, out_features=64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(in_features=64, out_features=128), 
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.final_layer = nn.Linear(in_features=128, out_features=1)

    def forward(self, x):
        x = self.mlp(x)
        network = self.final_layer(x)

        return network
    
    def get_features(self, x):
        features = self.mlp(x)
        return features


    def set_grad(self, val):
        for param in self.mlp.parameters():
            param.requires_grad = val



class MLPCompas(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLPCompas, self).__init__()
        # print("Dim in: ", dim_in)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=dim_in, out_features=64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(in_features=64, out_features=128), 
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.final_layer = nn.Linear(in_features=128, out_features=1)

    def forward(self, x):
        x = self.mlp(x)
        network = self.final_layer(x)

        return network
    
    def get_features(self, x):
        features = self.mlp(x)
        return features


    def set_grad(self, val):
        for param in self.mlp.parameters():
            param.requires_grad = val


def _padding(downsample, kernel_size):
    """Compute required padding"""
    padding = max(0, int(np.floor((kernel_size - downsample + 1) / 2)))
    return padding


def _downsample(n_samples_in, n_samples_out):
    """Compute downsample rate"""
    downsample = int(n_samples_in // n_samples_out)
    if downsample < 1:
        raise ValueError("Number of samples should always decrease\n but we got {} in and {} out.".format(n_samples_in, n_samples_out))
    if n_samples_in % n_samples_out != 0:
        raise ValueError("Number of samples for two consecutive blocks "
                         "should always decrease by an integer factor.")
    return downsample


class ResBlock1d(nn.Module):
    """Residual network unit for unidimensional signals."""

    def __init__(self, n_filters_in, n_filters_out, downsample, kernel_size, dropout_rate):
        if kernel_size % 2 == 0:
            raise ValueError("The current implementation only support odd values for `kernel_size`.")
        super(ResBlock1d, self).__init__()
        # Forward path
        padding = _padding(1, kernel_size)
        self.conv1 = nn.Conv1d(n_filters_in, n_filters_out, kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(n_filters_out)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        padding = _padding(downsample, kernel_size)
        self.conv2 = nn.Conv1d(n_filters_out, n_filters_out, kernel_size,
                               stride=downsample, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(n_filters_out)
        self.dropout2 = nn.Dropout(dropout_rate)

        # Skip connection
        skip_connection_layers = []
        # Deal with downsampling
        if downsample > 1:
            maxpool = nn.MaxPool1d(downsample, stride=downsample)
            skip_connection_layers += [maxpool]
        # Deal with n_filters dimension increase
        if n_filters_in != n_filters_out:
            conv1x1 = nn.Conv1d(n_filters_in, n_filters_out, 1, bias=False)
            skip_connection_layers += [conv1x1]
        # Build skip conection layer
        if skip_connection_layers:
            self.skip_connection = nn.Sequential(*skip_connection_layers)
        else:
            self.skip_connection = None

    def forward(self, x, y):
        """Residual unit."""
        if self.skip_connection is not None:
            y = self.skip_connection(y)
        else:
            y = y
        # 1st layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        # 2nd layer
        x = self.conv2(x)
        x += y  # Sum skip connection and main connection
        y = x
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        return x, y

 
class ResNetPTB(nn.Module):
    def __init__(self, input_dim, blocks_dim, n_classes, kernel_size=17, dropout_rate=0.8):
        super(ResNetPTB, self).__init__()

        n_filters_in, n_filters_out = input_dim[0], blocks_dim[0][0]
        n_samples_in, n_samples_out = input_dim[1], blocks_dim[0][1]
        downsample = _downsample(n_samples_in, n_samples_out)
        padding = _padding(downsample, kernel_size)
        self.conv1 = nn.Conv1d(n_filters_in, n_filters_out, kernel_size, bias=False,
                               stride=downsample, padding=padding)
        self.bn1 = nn.BatchNorm1d(n_filters_out)

        # Residual block layers
        self.res_blocks = []
        for i, (n_filters, n_samples) in enumerate(blocks_dim):
            n_filters_in, n_filters_out = n_filters_out, n_filters
            n_samples_in, n_samples_out = n_samples_out, n_samples
            downsample = _downsample(n_samples_in, n_samples_out)
            resblk1d = ResBlock1d(n_filters_in, n_filters_out, downsample, kernel_size, dropout_rate)
            self.add_module('resblock1d_{0}'.format(i), resblk1d)
            self.res_blocks += [resblk1d]

        # Linear layer
        n_filters_last, n_samples_last = blocks_dim[-1]
        last_layer_dim = n_filters_last * n_samples_last
        self.final_layer = nn.Linear(last_layer_dim, n_classes)
        self.n_blk = len(blocks_dim)


    def forward(self, x):
        """Implement ResNet1d forward propagation"""
        # First layers
        x = self.conv1(x)
        x = self.bn1(x)

        # Residual blocks
        y = x
        for blk in self.res_blocks:
            x, y = blk(x, y)

        # Flatten array
        x = x.view(x.size(0), -1)

        # Fully conected layer
        # x = self.final_layer(x)
        x = sigmoid(self.final_layer(x))
        return x
    
    
    def get_features(self, x):
       # First layers
        x = self.conv1(x)
        x = self.bn1(x)

        # Residual blocks
        y = x
        for blk in self.res_blocks:
            x, y = blk(x, y)

        # Flatten array
        x = x.view(x.size(0), -1)

        return x

    def set_grad(self, val):
        # Freeze all paramters and reactive final layer parameter
        for param in self.parameters():
            param.requires_grad = False
        for param in self.final_layer.parameters():
            param.requires_grad = True




class ANN(nn.Module):
    def __init__(self, dim_in):
        super(ANN, self).__init__()

        self.input_size = dim_in
       
        self.features = nn.Sequential(
        nn.Dropout(p=0.3), 
        nn.Linear(in_features=dim_in, out_features=4096) ,
        nn.ReLU(),
        nn.Dropout(p=0.5), 
        nn.Linear(in_features=4096, out_features=4096) ,
        nn.ReLU(),
        nn.Dropout(p=0.3), 
        nn.Linear(in_features=4096, out_features=1024) ,
        nn.ReLU(),
        nn.Dropout(p=0.3), 
        nn.Linear(in_features=1024, out_features=256) ,
        nn.ReLU(),
        nn.Dropout(p=0.3), 
        nn.Linear(in_features=256, out_features=64) ,
        )

        self.final_layer = nn.Linear(in_features=64 , out_features=1)

    def forward(self, x):
        x = self.features(x)
        x = (self.final_layer(x))

        return x

    def get_features(self, x):
        return self.features(x)
    
    def set_grad(self,val):
        for param in self.parameters():
            param.requires_grad = False
        for param in self.final_layer.parameters():
            param.requires_grad = True
       





class VGG(nn.Module):
    def __init__(self, dim_in):
        super(VGG, self).__init__()

        self.input_size = dim_in
        self.vgg19 = models.vgg19(weights='DEFAULT')
        self.relu = nn.Sequential(
        nn.ReLU(),
        )

        self.final_layer = nn.Linear(1000, 1)

    def forward(self, x):
        # x = self.features(x)
        # x = (self.final_layer(x))
        x = self.relu(self.vgg19(x))
        # x = self.final_layer(x)
        x = sigmoid(self.final_layer(x))


        return x

    def get_features(self, x):
        return self.features(x)
    
    def set_grad(self,val):
        for param in self.parameters():
            param.requires_grad = False
        for param in self.final_layer.parameters():
            param.requires_grad = True
       



class VGG16(nn.Module):
    def __init__(self, dim_in):
        super(VGG16, self).__init__()

        self.input_size = dim_in
        self.vgg16 = models.vgg16(weights='DEFAULT')
        self.relu = nn.Sequential(
        nn.ReLU(),
        nn.Dropout(p=0.5), 
        )

        self.final_layer = nn.Linear(1000, 1)
       
    def forward(self, x):
        x = self.relu(self.vgg16(x))
        # x = self.final_layer(x)
        x = sigmoid(self.final_layer(x))

        return x
    
    def get_features(self, x):
        return self.vgg16(x)
    
    def set_grad(self, val):
        for param in self.parameters():
            param.requires_grad = False
        for param in self.final_layer.parameters():
            param.requires_grad = True
       



class MobileNet(nn.Module):
    def __init__(self, dim_in, kernel_size):
        super(MobileNet, self).__init__()

        self.input_size = dim_in
        self.kernel_size = kernel_size
       
        vgg_model = models.mobilenet_v2(weights='IMAGENET1K_V2')
        
        IN_FEATURES = 1280

        vgg_model.classifier = nn.Sequential(
        nn.Linear(in_features=IN_FEATURES, out_features=512) ,
        nn.ReLU(),
        nn.Dropout(p=0.6), 
        )

        self.features = vgg_model

        self.final_layer = nn.Linear(in_features=512 , out_features=1)

    def forward(self, x):
        x = self.features(x)
        # x = (self.final_layer(x))
        x = sigmoid(self.final_layer(x))

        return x

    def get_features(self, x):
        return self.features(x)
    
    def set_grad(self,val):
        for param in self.parameters():
            param.requires_grad = False
        for param in self.final_layer.parameters():
            param.requires_grad = True



class Plain_LR_Adult(nn.Module):
    def __init__(self, dim_in):
        super(Plain_LR_Adult, self).__init__()
        # self.input_size = dim_in
        self.final_layer = nn.Linear(dim_in, 1)

    def forward(self, x):
        x = sigmoid(self.final_layer(x))
        return x

    def get_features(self, x):
        return x
    
    def set_grad(self,val):
        return True

class Plain_LR_2(nn.Module):
    def __init__(self, dim_in):
        super(Plain_LR_2, self).__init__()
        
        self.features = nn.Sequential(
            nn.Linear(in_features=dim_in, out_features=dim_in),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.final_layer = nn.Linear(dim_in, 1)

    def forward(self, x):
        x = self.features(x)
        x = sigmoid(self.final_layer(x))
        # x=self.final_layer(x)
        return x

    def get_features(self, x):
        return self.features(x)
    
    def set_grad(self,val):
        for param in self.parameters():
            param.requires_grad = False
        for param in self.final_layer.parameters():
            param.requires_grad = True




def get_model(args, img_size):

    if args.dataset == 'ptb-xl':
        if args.model == "mobile":
            len_in = 1
            for x in img_size:
                len_in *= x
            kernel_size = 17
            global_model = MobileNet(dim_in=len_in, kernel_size=kernel_size)

        elif args.model == "ann":
            len_in = 1
            for x in img_size:
                len_in *= x
            kernel_size = 17
            global_model = ANN(dim_in=len_in)

        else:
            N_LEADS = 12  # the 12 leads
            N_CLASSES = 1  # just the age
            seq_length = 1000
            # net_filter_size=[64, 128, 196, 256, 320]
            # net_seq_lengh=[4096, 1024, 256, 64, 16]
            net_filter_size=[32, 64, 128, 196, 256]
            net_seq_lengh=[1000, 500, 250, 125, 25]
            dropout_rate=0.5
            kernel_size=17
            global_model = ResNetPTB(input_dim=(N_LEADS, seq_length),
                            blocks_dim=list(zip(net_filter_size, net_seq_lengh)),
                            n_classes=N_CLASSES,
                            kernel_size=kernel_size,
                            dropout_rate=dropout_rate)

    elif args.dataset == 'nih-chest' or args.dataset == 'nih-chest-h5' or args.dataset == 'nih-chest-eff':
        N_CLASSES = 1  # just the age
        seq_length = 256
        kernel_size=17
        if args.model == 'vgg16':
            global_model = VGG16(dim_in=(seq_length, seq_length))
        elif args.model == 'vgg':
            global_model = VGG(dim_in=(seq_length, seq_length))
        elif args.model == 'mobile':
            global_model = MobileNet(dim_in=(seq_length, seq_length), kernel_size=kernel_size)


    elif args.model == 'mlp':
        # Multi-layer preceptron
        if args.dataset == 'adult':
            # img_size = train_dataset[0][0].shape
            # print("img size: ", img_size)
            len_in = 1
            for x in img_size:
                len_in *= x
                # global_model = MLPAdult(dim_in=len_in, dim_hidden=64,
                #                 dim_out=args.num_classes)
            global_model = MLPAdult2(dim_in=len_in, dim_hidden=64,
                            dim_out=args.num_classes)
        
        elif args.dataset == 'compas':
            # img_size = train_dataset[0][0].shape
            len_in = 1
            for x in img_size:
                len_in *= x
                # global_model = MLPAdult(dim_in=len_in, dim_hidden=64,
                #                 dim_out=args.num_classes)
            global_model = MLPCompas(dim_in=len_in, dim_hidden=64,
                            dim_out=args.num_classes)
            
        elif args.dataset == 'wcld':
            # img_size = train_dataset[0][0].shape
            len_in = 1
            for x in img_size:
                len_in *= x
                global_model = MLPCompas(dim_in=len_in, dim_hidden=64,
                                dim_out=args.num_classes)
        else:
            # img_size = train_dataset[0][0].shape
            len_in = 1
            for x in img_size:
                len_in *= x
                global_model = MLP(dim_in=len_in, dim_hidden=64,
                                dim_out=args.num_classes)
    
    elif args.model == 'plain2':
        len_in = 1
        for x in img_size:
            len_in *= x
        global_model = Plain_LR_2(dim_in=len_in)

    elif args.model == 'plain':
        if args.dataset == 'adult':
            # img_size = train_dataset[0][0].shape
            len_in = 1
            for x in img_size:
                len_in *= x
            global_model = Plain_LR_Adult(dim_in=len_in)
        elif args.dataset == 'compas' or args.dataset == 'compas-binary':
            # img_size = train_dataset[0][0].shape
            len_in = 1
            for x in img_size:
                len_in *= x
            global_model = Plain_LR_Adult(dim_in=len_in)
        elif args.dataset == 'wcld':
            # img_size = train_dataset[0][0].shape
            len_in = 1
            for x in img_size:
                len_in *= x
            global_model = Plain_LR_Adult(dim_in=len_in)
        

    else:
        exit('Error: unrecognized model')

    return global_model

