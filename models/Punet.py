from torch import device
from models.usleep_lstm import *
from torch.distributions import Normal, Independent, kl

device = torch.device("cuda")

def generate_mask(mask):
    Mask = np.zeros(shape=(len(mask),1288))
    for i in range(len(mask)):
        Mask[i] = np.r_[np.ones(mask[i]),np.zeros(1288-mask[i])]
    return Mask
        
def masked_loss(out, rl, mask):
    
    if not mask == None:
        rl = rl.long().squeeze(1)
        rl = nn.functional.one_hot(rl, num_classes=5)
        rl = torch.transpose(rl,1,2)
        loss = -torch.log(out).mul(rl).sum(axis=1)
        loss = loss.mul(torch.tensor(generate_mask(mask)).to(device))
        return loss.sum()/mask.sum()
    else:
        output = torch.log(out)
        loss = torch.tensor(0.0).to(device)
        cnt = 0
        for rl_idx in range(rl.shape[0]):
            
            for idx, i in enumerate(rl[rl_idx,0,:]):
                cur = int(i)
                if cur==7: 
                    continue
                else:
                    loss += -torch.tensor(output[rl_idx,cur,idx].item())
                    cnt += 1
        return loss/cnt
    
def masked_loss_acc(out, rl):
    # generate a mask, maybe should put in the dataloader in advance：
    # n = rl.shape[-1]
    # rl = rl.reshape([rl.shape[0], n])
    mask = (rl!=7)
    
    # accuracy
    ans = out.argmax(1)
    crt = (ans==rl).sum()
    
    # one_hot
    rl = nn.functional.one_hot(rl.mul(mask).long().squeeze(1), num_classes=5)
    rl = torch.transpose(rl, 1, 2)
    out = out
    
    # loss
    loss = -torch.log(out).mul(rl).mul(mask).sum(axis=1)
    
    n = mask.sum()
    
    return loss.sum()/n, crt/n

class Encoder(nn.Module):
    """
    A CNN mapping input PSG to latent space of dim:2*f
    
    """
    def __init__(self, input_channels, num_filters, no_convs_per_block, padding = True, posterior = False):
        super(Encoder, self).__init__()
        self.input_channels = input_channels
        self.num_filters = num_filters
        self.input_channels = input_channels
        
        if posterior:#Setting posterior=True when training
            self.input_channels += 1 #concatnating truth with input
        
        layers, cnt = [], 0
        for i in num_filters:
            in_dim = self.input_channels if cnt==0 else out_dim
            out_dim = i
            
            if cnt != 0:
                layers.append(nn.AvgPool1d(kernel_size=256, stride=256, padding=0, ceil_mode=True))
            layers.append(nn.Conv1d(in_dim, out_dim, kernel_size=3, padding=int(padding)))
            layers.append(nn.ReLU(inplace=True))
            "Add some conv layers with same amount of filters"
            # for _ in range(no_convs_per_block-1):
            #     layers.append(nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=int(padding)))
            #     layers.append(nn.ReLU(inplace=True))
            
            cnt += 1
        self.layers = nn.Sequential(*layers).double()
        
    def forward(self, input):
        output = self.layers(input)
        return output

class AxisAlignedConvGaussian(nn.Module):
    """
    A convolutional net that parametrizes a Gaussian distribution with axis aligned covariance matrix.
    The output will be of shape [batch, f]
    
    """
    def __init__(self, input_channels, num_filters, no_convs_per_block, latent_dim, initializers, posterior=False):
        super(AxisAlignedConvGaussian, self).__init__()
        self.input_channels = input_channels
        self.channel_axis = 1
        self.num_filters = num_filters
        self.no_convs_per_block = no_convs_per_block
        self.latent_dim = latent_dim
        self.posterior = posterior
        if self.posterior:
            self.name = 'Posterior'
        else:
            self.name = 'Prior'
        self.encoder = Encoder(self.input_channels, self.num_filters, self.no_convs_per_block, posterior=self.posterior)
        self.conv_layer = nn.Conv1d(num_filters[-1], 2 * self.latent_dim, kernel_size=1, stride=1).double()
        self.show_img = 0
        self.show_seg = 0
        self.show_concat = 0
        self.show_enc = 0
        self.sum_input = 0

        nn.init.kaiming_normal_(self.conv_layer.weight, mode='fan_in', nonlinearity='relu')
        nn.init.normal_(self.conv_layer.bias)

    def forward(self, input, segm=None):

        #If segmentation is not none, concatenate the mask to the channel axis of the input
        if segm is not None:
            self.show_img = input
            self.show_seg = segm
            input = torch.cat((input, segm), dim=1)
            self.show_concat = input
            self.sum_input = torch.sum(input)

        encoding = self.encoder(input.double())
        self.show_enc = encoding

        #We only want the mean of the resulting hxw image
        encoding = torch.mean(encoding, dim=2, keepdim=True)
        #Convert encoding to 2 x latent dim and split up for mu and log_sigma
        mu_log_sigma = self.conv_layer(encoding)
        #We squeeze the second dimension twice, since otherwise it won't work when batch size is equal to 1
        mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)

        mu = mu_log_sigma[:,:self.latent_dim]
        log_sigma = mu_log_sigma[:,self.latent_dim:]

        #This is a multivariate normal with diagonal covariance matrix sigma
        #https://github.com/pytorch/pytorch/pull/11178
        dist = Independent(Normal(loc=mu, scale=torch.exp(log_sigma)),1)
        return dist

class Fcomb(nn.Module):
    """
    A function composed of no_convs_fcomb times a 1x1 convolution that combines the sample taken from the latent space,
    and output of the UNet (the feature map) by concatenating them along their channel axis.
    """
    def __init__(self, num_filters, latent_dim, num_output_channels, num_classes, no_convs_fcomb, initializers, use_tile=True):
        super(Fcomb, self).__init__()
        self.num_channels = num_output_channels #output channels
        self.num_classes = num_classes
        self.channel_axis = 1
        self.spatial_axes = [2,3]
        self.num_filters = num_filters
        self.latent_dim = latent_dim
        self.use_tile = use_tile
        self.no_convs_fcomb = no_convs_fcomb 
        self.name = 'Fcomb'

        if self.use_tile:
            layers = []

            #Decoder of N x a 1x1 convolution followed by a ReLU activation function except for the last layer
            layers.append(nn.Conv1d(5+self.latent_dim, 7, kernel_size=1))
            layers.append(nn.ReLU(inplace=True))

            for _ in range(no_convs_fcomb-2):
                layers.append(nn.Conv1d(7, 7, kernel_size=1))
                layers.append(nn.ReLU(inplace=True))

            self.layers = nn.Sequential(*layers).double()

            self.last_layer = nn.Sequential(
                nn.AvgPool1d(kernel_size=30*128, stride=30*128),
                nn.Conv1d(7, self.num_classes, kernel_size=1),
                nn.Softmax(dim=-2)
            ).double()

            # if initializers['w'] == 'orthogonal':
            #     self.layers.apply(init_weights_orthogonal_normal)
            #     self.last_layer.apply(init_weights_orthogonal_normal)
            # else:
            #     self.layers.apply(init_weights)
            #     self.last_layer.apply(init_weights)

    def tile(self, a, dim, n_tile):
        """
        This function is taken form PyTorch forum and mimics the behavior of tf.tile.
        Source: https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/3
        """
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx)).to(device)
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(device)
        return torch.index_select(a, dim, order_index)

    def forward(self, feature_map, z):
        """
        Z is batch_sizexlatent_dim and feature_map is batch_sizexno_channelsxHxW.
        So broadcast Z to batch_sizexlatent_dimxHxW. Behavior is exactly the same as tf.tile (verified)
        """
        if self.use_tile:
            z = torch.unsqueeze(z,2)
            z = self.tile(z, 2, feature_map.shape[self.spatial_axes[0]])


            #Concatenate the feature map (output of the UNet) and the sample taken from the latent space
            feature_map = torch.cat((feature_map, z), dim=self.channel_axis)
            output = self.layers(feature_map)
            return self.last_layer(output)


class ProbabilisticUnet(nn.Module):
    """
    A probabilistic UNet (https://arxiv.org/abs/1806.05034) implementation.
    input_channels: the number of channels in the image (1 for greyscale and 3 for RGB)
    num_classes: the number of classes to predict
    num_filters: is a list consisint of the amount of filters layer
    latent_dim: dimension of the latent space
    no_cons_per_block: no convs per block in the (convolutional) encoder of prior and posterior
    """

    def __init__(self, device, input_channels=2, num_classes=5, num_filters=[32,64,128,192], latent_dim=5, no_convs_fcomb=2, beta=0.1):
        super(ProbabilisticUnet, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.latent_dim = latent_dim
        self.no_convs_per_block = 3
        self.no_convs_fcomb = no_convs_fcomb
        self.initializers = {'w':'he_normal', 'b':'normal'}
        self.beta = beta
        self.z_prior_sample = 0

        self.unet = UNet(self.input_channels, self.num_classes, self.num_filters).to(device)
        self.prior = AxisAlignedConvGaussian(self.input_channels, self.num_filters, self.no_convs_per_block, self.latent_dim,  self.initializers).to(device)
        self.posterior = AxisAlignedConvGaussian(self.input_channels, self.num_filters, self.no_convs_per_block, self.latent_dim, self.initializers, posterior=True).to(device)
        self.fcomb = Fcomb(self.num_filters, self.latent_dim, self.input_channels, self.num_classes, self.no_convs_fcomb, {'w':'orthogonal', 'b':'normal'}, use_tile=True).to(device)

    def forward(self, patch, segm, training=True):
        """
        Construct prior latent space for patch and run patch through UNet,
        in case training is True also construct posterior latent space
        """
        if training:
            self.posterior_latent_space = self.posterior.forward(patch, segm)
        self.prior_latent_space = self.prior.forward(patch)
        self.unet_features = self.unet.forward(patch)

    def sample(self, testing=False):
        """
        Sample a segmentation by reconstructing from a prior sample
        and combining this with UNet features
        """
        if testing == False:
            z_prior = self.prior_latent_space.rsample()
            self.z_prior_sample = z_prior
        else:
            #You can choose whether you mean a sample or the mean here. For the GED it is important to take a sample.
            #z_prior = self.prior_latent_space.base_dist.loc 
            z_prior = self.prior_latent_space.sample()
            self.z_prior_sample = z_prior
        return self.fcomb.forward(self.unet_features,z_prior)


    def reconstruct(self, use_posterior_mean=False, calculate_posterior=False, z_posterior=None):
        """
        Reconstruct a segmentation from a posterior sample (decoding a posterior sample) and UNet feature map
        use_posterior_mean: use posterior_mean instead of sampling z_q
        calculate_posterior: use a provided sample or sample from posterior latent space
        """
        if use_posterior_mean:
            z_posterior = self.posterior_latent_space.loc
        else:
            if calculate_posterior:
                z_posterior = self.posterior_latent_space.rsample()
        return self.fcomb.forward(self.unet_features, z_posterior)

    def kl_divergence(self, analytic=True, calculate_posterior=False, z_posterior=None):
        """
        Calculate the KL divergence between the posterior and prior KL(Q||P)
        analytic: calculate KL analytically or via sampling from the posterior
        calculate_posterior: if we use samapling to approximate KL we can sample here or supply a sample
        """
        if analytic:
            #Neeed to add this to torch source code, see: https://github.com/pytorch/pytorch/issues/13545
            kl_div = kl.kl_divergence(self.posterior_latent_space, self.prior_latent_space)
        else:
            if calculate_posterior:
                z_posterior = self.posterior_latent_space.rsample()
            log_posterior_prob = self.posterior_latent_space.log_prob(z_posterior)
            log_prior_prob = self.prior_latent_space.log_prob(z_posterior)
            kl_div = log_posterior_prob - log_prior_prob
        return kl_div

    def elbo(self, segm, mask, analytic_kl=True, reconstruct_posterior_mean=False):
        """
        Calculate the evidence lower bound of the log-likelihood of P(Y|X)
        """

        z_posterior = self.posterior_latent_space.rsample()
        
        self.kl = torch.mean(self.kl_divergence(analytic=analytic_kl, calculate_posterior=False, z_posterior=z_posterior))

        #Here we use the posterior sample sampled above
        self.reconstruction = self.reconstruct(use_posterior_mean=reconstruct_posterior_mean, calculate_posterior=False, z_posterior=z_posterior)
        
        reconstruction_loss, acc = masked_loss_acc(self.reconstruction, segm)
        self.reconstruction_loss = reconstruction_loss
        self.mean_reconstruction_loss = torch.mean(reconstruction_loss)
       
        return -(self.reconstruction_loss + self.beta * self.kl), acc
    
    def val(self, segm):
        z_posterior = self.prior_latent_space.rsample()
        #Here we use the posterior sample sampled above
        self.reconstruction = self.reconstruct(use_posterior_mean=False, calculate_posterior=False, z_posterior=z_posterior)
        
        return self.reconstruction
    
        
        
        
     
