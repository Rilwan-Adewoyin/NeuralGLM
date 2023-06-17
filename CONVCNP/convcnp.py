import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse 
from torch.distributions.gamma import Gamma
from methodtools import lru_cache
from typing import Tuple
import numpy as np
from itertools import product


def generate_context_mask(batch_size, n_channels, x, y):
    """
    Generate a context mask - in this simple case this will be one 
    for all grid points
    """
    return torch.ones(batch_size, n_channels, x, y).cuda()

def get_dists(target_x, grid_x, grid_y):
    """
    Get the distances between the grid points and true points
    """
    # dimensions
    x_dim, y_dim = grid_x.shape

    # Init large scale grid
    total_grid = torch.zeros(target_x.shape[0], x_dim, y_dim)
    count = 0

    for point in target_x:
        # Calculate distance from point to each grid
        dists = (grid_x - point[0])**2+(grid_y - point[1])**2
        total_grid[count, :, :] = dists

        count += 1

    return total_grid.cuda()

def log_exp(x):
    """
    Fix overflow
    """
    lt = torch.where(torch.exp(x)<1000)
    if lt[0].shape[0] > 0:
        x[lt] = torch.log(1+torch.exp(x[lt])).to(x)
    return x

def force_positive(x):
    return 0.01+ (1-0.1)*log_exp(x)

def make_r_mask(target_vals):
    """
    Make the r mask for the Bernoulli precipitation distribution
    """
    # Make r mask
    r = target_vals.new_ones(target_vals.shape[0])
    no_rain_mask = target_vals==0
    r[no_rain_mask] = 0
    
    # Set the target vals to one to stop the pesky error
    # (It doesn't contribute anyway)
    target_vals[ no_rain_mask ] = 0.01

    return r, target_vals

def gamma_ll(target_vals, v, mask):
    """
    Evaluate gamma-bernoulli mixture likelihood
    Parameters:
    ----------
    v: torch.Tensor(batch,86,channels)
        parameters from model [rho, alpha, beta]
    target_vals: torch.Tensor(batch,86)
        target vals to eval at
    """

    # Reshape
    target_vals = target_vals.reshape(-1)
    v = v.reshape(-1, 3)
    mask = mask.reshape(-1)
    
    target_vals = torch.masked_select( target_vals, mask )
    v = v[torch.argwhere(mask).squeeze()] 
    
    # # Deal with cases where data is missing for a station
    v = v[~torch.isnan(target_vals), :]
    target_vals = target_vals[~torch.isnan(target_vals)]

    # Make r mask
    r, target_vals = make_r_mask(target_vals)

    gamma = Gamma(concentration = v[:,1], rate = v[:,2])
    logp = gamma.log_prob(target_vals)

    total = r*(torch.log(v[:,0])+logp)+(1-r)*torch.log(1-v[:,0])
    
    return torch.mean(total)

class GammaBiasConvCNPElev(nn.Module):
    """
    Bias correction for precipitation, including MLP for elevation
    Parameters:
    ----------
    decoder: convolutional architecture
    """

    def __init__(self,
                 in_channels = 1, 
                 ls = 0.1,
                 
                 dec_n_blocks = 6,
                 mlp_hidden_channels = 96, # 64,
                 
                 dec_n_channels = 160 # #128,
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.activation = torch.relu
        self.sigmoid = nn.Sigmoid()

        self.encoder = Encoder(in_channels = in_channels, 
                                out_channels = dec_n_channels)

        self.decoder = CNN( n_blocks=dec_n_blocks, 
                           n_channels=dec_n_channels,
                           Conv=torch.nn.Conv2d,
                           )
                
        self.mlp = MLP(in_channels = dec_n_channels,
            out_channels = 3,
            hidden_channels = mlp_hidden_channels,
            hidden_layers = 4)
        

        self.out_layer = GammaFinalLayer(
            init_ls = ls,
            n_params = 3
        )

        self.elev_mlp = MLP(4,
            out_channels = 3,
            hidden_channels = mlp_hidden_channels,
            hidden_layers = 4)

    def forward(self, h, mask, dists, elev):

        # Encode with set convolution
        h1 = self.activation(self.encoder(h, mask))
        # Decode with CNN
        h2 = self.activation(self.decoder(h1))
        # MLP 
        h3 = self.mlp(h2)
        # out layer
        rho, alpha, beta = self.out_layer(h3, dists) #(b, d)
        out = torch.cat([rho.view(*rho.shape, 1),
            alpha.view(*alpha.shape, 1),
            beta.view(*beta.shape, 1)], dim = 2)

        # Do elevation
        # elev = elev.repeat(out.shape[0], 1, 1)
        # out = torch.cat([out[...,0], elev], dim = 2)
        out = torch.cat([out, elev], dim = 2)
        
        out1 = self.elev_mlp(out)
        out1[...,0] = self.sigmoid(out1[...,0])
        out1[...,1:] = force_positive(out1[...,1:])
                       
        return out1

    def parse_model_args(parent_parser):
        model_parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=True, allow_abbrev=False)
        
        model_parser.add_argument("--in_channels", default=6,type=int)
        model_parser.add_argument("--ls", default=0.1, type=int)
                
        model_args = model_parser.parse_known_args()[0]
                
        return model_args
    
class Encoder(nn.Module):
    """
    ConvCNP encoder
    Elements of this class based on
    https://github.com/YannDubs
    Parameters:
    ----------
    in_channels: Int
        Total number of context variables 
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        Conv=lambda in_channels: self._make_abs_conv(nn.Conv2d)(
            in_channels, 
            in_channels, 
            groups=in_channels, 
            kernel_size=5, 
            padding=5 // 2, 
            bias=False
        )
        self.conv = Conv(in_channels)

        self.transform_to_cnn = nn.Linear(
            self.in_channels*2, out_channels
        )

        self.density_to_confidence = ProbabilityConverter(
            trainable_dim=self.in_channels
        )

    def _make_abs_conv(self, Conv):

        class AbsConv(Conv):
            def forward(self, input):
                return F.conv2d(
                    input,
                    self.weight.abs(),
                    self.bias,
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.groups,
                )

        return AbsConv

    def forward(self, x, mask):

        batch_size, n_channels, x_grid, y_grid = x.shape

        num = self.conv(x*mask)
        denom = self.conv(mask)

        h = num/torch.clamp(denom, min=1e-5)
        
        confidence = self.density_to_confidence(denom.view(-1, n_channels) * 0.1).view(
                batch_size, n_channels, x_grid, y_grid
            )

        h = torch.cat([h, confidence], dim=1)
        h = self.transform_to_cnn(h.permute(0, 2, 3, 1))

        return h


         
class ProbabilityConverter(nn.Module):
    """
    Convert from densities to probabilities
    From https://github.com/YannDubs
    """

    def __init__(
        self,
        trainable_dim=1,):

        super().__init__()
        self.min_p = 0.0
        self.trainable_dim = trainable_dim
        self.initial_temperature = 1.0
        self.initial_probability = 0.5
        self.initial_x = 0.0
        self.temperature_transformer = F.softplus

        self.reset_parameters()

    def reset_parameters(self):
        self.temperature = torch.tensor([self.initial_temperature] * self.trainable_dim)
        self.temperature = nn.Parameter(self.temperature)

        initial_bias = self._probability_to_bias(
            self.initial_probability, initial_x=self.initial_x
        )

        self.bias = torch.tensor([initial_bias] * self.trainable_dim)
        self.bias = nn.Parameter(self.bias)

    def _probability_to_bias(self, p, initial_x=0):
        """
        Compute the bias to use to satisfy the constraints.
        """
        assert p > self.min_p and p < 1 - self.min_p
        range_p = 1 - self.min_p * 2
        p = (p - self.min_p) / range_p
        p = torch.tensor(p, dtype=torch.float)

        bias = -(torch.log((1 - p) / p) / self.initial_temperature + initial_x)
        return bias

    def _rescale_range(self, p, init_range, final_range):
        """
        Rescale vec to be in new range
        """
        init_min, final_min = init_range[0], final_range[0]
        init_delta = init_range[1] - init_range[0]
        final_delta = final_range[1] - final_range[0]

        return (((p - init_min)*final_delta) / init_delta) + final_min

    def forward(self, x):
        self.temperature.to(x.device)
        self.bias.to(x.device)

        temperature = self.temperature_transformer(self.temperature)
        full_p = torch.sigmoid((x + self.bias) * temperature)
        p = self._rescale_range(full_p, (0, 1), (self.min_p, 1 - self.min_p))

        return p


class MLP(nn.Module):
    """
    MLP for the elevation data and raw CNN output
    Parameters:
    -----------
    in_channels: Int
        Number of input channels
    out_channels: Int
        Number of output channels
    hidden_channels: Int
        Number of hidden nodes
    hidden_layers: Int
        Number of hidden layers
    """

    def __init__(self, in_channels, 
                out_channels, 
                hidden_channels,
                hidden_layers):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.n_hidden_layers = hidden_layers
        self.relu = nn.ReLU()

        self.in_to_hidden = nn.Linear(self.in_channels, self.hidden_channels)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(self.hidden_channels, self.hidden_channels) 
            for _ in range(hidden_layers)]
            )
        self.hidden_to_out = nn.Linear(self.hidden_channels, self.out_channels) 

    def forward(self, h):
        # in -> hidden
        h = self.in_to_hidden(h)
        h = self.relu(h)
        # hidden
        for layer in self.hidden_layers:
            h = self.relu(layer(h))
        # hidden -> out
        h = self.hidden_to_out(h)

        return h

def channels_to_2nd_dim(h):
    return h.permute(*([0, h.dim() - 1] + list(range(1, h.dim() - 1))))

def channels_to_last_dim(h):
    return h.permute(*([0] + list(range(2, h.dim())) + [1]))

def make_depth_sep_conv(Conv):

    class DepthSepConv(nn.Module):
        """
        Make a depth separable conv. 
        """

        def __init__(self, 
            in_channels, 
            out_channels, 
            kernel_size, 
            confidence=False, 
            bias=True, **kwargs):

            super().__init__()
            self.depthwise = Conv(in_channels, 
                in_channels, 
                kernel_size, 
                groups=in_channels, 
                bias=bias, **kwargs)

            self.pointwise = Conv(in_channels, out_channels, 1, bias=bias)

        def forward(self, x):
            out = self.depthwise(x)
            out = self.pointwise(out)
            return out
    
    return DepthSepConv
    
class CNN(nn.Module):
    """
    Resnet CNN
    Adapted from https://github.com/YannDubs
    """

    def __init__(self, n_channels, n_blocks=6, **kwargs):

        super().__init__()
        self.n_blocks = n_blocks
        self.in_channels = n_channels
        self.out_channels = n_channels
        self.in_out_channels = self._get_in_out_channels(n_channels, n_blocks)
        self.conv_blocks = nn.ModuleList(
            [ResConvBlock(in_chan, out_chan, **kwargs) for in_chan, out_chan in self.in_out_channels]
        )

    def _get_in_out_channels(self, n_channels, n_blocks):
        """Return a list of tuple of input and output channels."""
        if isinstance(n_channels, int):
            channel_list = [n_channels] * (n_blocks + 1)
        else:
            channel_list = list(n_channels)

        return list(zip(channel_list, channel_list[1:]))

    def forward(self, h):

        h = channels_to_2nd_dim(h)

        for conv_block in self.conv_blocks:
            h = conv_block(h)

        h = channels_to_last_dim(h)

        return h

class ResConvBlock(nn.Module):
    """
    Residual block for Resnet CNN
    Adapted from https://github.com/YannDubs
    """

    def __init__(
        self,
        in_chan,
        out_chan,
        Conv,
        kernel_size=5,
        activation=nn.ReLU(),
        Normalization=nn.Identity,
        is_bias=True,
        Padder=None,
    ):
        super().__init__()
        self.activation = activation

        padding = kernel_size // 2

        if Padder is not None:
            self.padder = Padder(padding)
            padding = 0
        else:
            self.padder = nn.Identity()

        self.norm1 = Normalization(in_chan)
        self.conv1 = make_depth_sep_conv(Conv)(
            in_chan, in_chan, kernel_size, padding=padding, bias=is_bias
        )
        self.norm2 = Normalization(in_chan)
        self.conv2_depthwise = Conv(
            in_chan, in_chan, kernel_size, padding=padding, groups=in_chan, bias=is_bias
        )
        self.conv2_pointwise = Conv(in_chan, out_chan, 1, bias=is_bias)

    def forward(self, X):
        out = self.padder(X)
        out = self.conv1(self.activation(self.norm1(X)))
        out = self.padder(out)
        out = self.conv2_depthwise(self.activation(self.norm2(X)))
        out = out + X
        out = self.conv2_pointwise(out)
        return out

class GammaFinalLayer(nn.Module):
    """
    On-grid -> off-grid layer for Bernoulli-Gamma 
    mixture distribution
    """

    def __init__(self,  
                 init_ls,
                 n_params):

        # FinalLayer.__init__(self, 
        #          init_ls, 
        #          n_params)
        super().__init__()
        self.param_layers = nn.ModuleList(
            [ParamLayer(init_ls)
             for _ in range(n_params)]
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, h, dists):

        rho = self.param_layers[0](h[..., 0], dists)
        alpha = self.param_layers[1](h[..., 1], dists)
        beta = self.param_layers[2](h[..., 2], dists)

        # rho = self.sigmoid(rho).view(*rho.shape, 1)
        rho = self.sigmoid(rho)
        alpha1 = self._force_positive(alpha).view(*rho.shape)
        beta1 = self._force_positive(beta).view(*rho.shape)

        # clamp values
        rho = torch.clamp(rho, min = 1e-5, max=1-1e-5)
        alpha2 = torch.clamp(alpha1, min = 1e-5, max=1e5)
        beta2 = torch.clamp(beta1, min = 1e-5, max=1e5)
        

        return rho, alpha2, beta2

    def _force_positive(self, x):
        """
        Make values greater than zero
        """
        return 0.01+ (1-0.1)*self._log_exp(x)

    def _log_exp(self, x):
        """
        Fix overflow
        """
        lt = torch.where(torch.exp(x)<1000)
        if lt[0].shape[0] > 0:
            x[lt] = torch.log(1+torch.exp(x[lt]))
        return x
    
class ParamLayer(nn.Module):
    """
    Calculate predicted of a parameter from gridded output
    Parameters:
    -----------
    init_ls: float
        initial length scale for the RBF kernel
    """

    def __init__(self, init_ls):        
        super().__init__()
        self.init_ls = torch.nn.Parameter(torch.tensor([init_ls], dtype=torch.float))
        self.init_ls.requires_grad = True

    def forward(self, wt, dists):
        # Calculate rbf kernel
        kernel = torch.exp(-0.5 * dists / self.init_ls ** 2)
        
        #NOTE:
        # kernel = torch.nn.functional.normalize(kernel, p=1.0, dim=[1,2])
        
        vals = torch.einsum('bij,pij->bpij', wt, kernel)
        outp = torch.sum(vals, (2, 3))
        
        return outp
   
   
class InterpolatedLocationsDistance():
    """
        This class can be used to retreive the (h, w) shaped matrices reflecting the
            the position of point p_i relative to  position defined on a possibly non-uniform
            grid with x values defined by grid_x and y values defined by grid_y
        
        Given P points and an original grid shape of (h, w)
            the output will be an array of shape (p, h, w)
                    
    """
    #TODO: add droppign by mask e.g. not all positions should be reutrned with the 
    def __init__(self, grid_x, grid_y) -> None:
        
        self.grid_x = grid_x
        self.grid_y = grid_y
    
    @lru_cache(24)
    def get_distances(self, target_positions:Tuple[ Tuple[float, float], ...], device=None, dtype=None, scale=1e-2 ):
        
        # Init large scale grid
        total_grid = np.zeros( (len(target_positions), len(self.grid_x), len(self.grid_y) ) )
        
        for idx, lat_lon in enumerate(target_positions):
            dists_x = (self.grid_x - lat_lon[0])**2
            dists_y = (self.grid_y - lat_lon[1])**2
            
            total_grid[idx] =  dists_x[:, None] + dists_y[None, :]
        
        total_grid = total_grid * scale
            
        total_grid = torch.tensor(total_grid)
        if device is not None: total_grid = total_grid.to(device)
        if dtype is not None: total_grid = total_grid.to(dtype)
        
        return total_grid
    
    @lru_cache(10)
    def get_points_for_upscale(self, upscale_x:int, upscale_y:int) -> Tuple[ Tuple[float, float], ...]:

        grid_x_upscaled = np.linspace(self.grid_x[0], self.grid_x[-1], len(self.grid_x)*upscale_x )
        grid_y_upscaled = np.linspace(self.grid_y[0], self.grid_y[-1], len(self.grid_y)*upscale_y )
        
        
        points = tuple( product(grid_x_upscaled, grid_y_upscaled) )
        
        return points
        
            
        
        