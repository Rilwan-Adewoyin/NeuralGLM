
import torch
from torch import nn
import argparse
from  glm_utils import tuple_type
import einops

    
class VAEGAN(nn.Module):
    
    def __init__(self,
                 
                 # encoder params
                 filters_gen=64,
                 residual_layers=2,
                 conv_size = (3,3),
                 norm = None,             
                            
                 
                 filters_disc=64,
                 noise_channels=32,
                 
                 latent_variables=50,
                 forceconv=False
                 ) -> None:
        super().__init__()

        
        self.grouped_conv2d_reduce_time = torch.nn.Conv2d( 6*4, 6, (1,1), groups=6, bias=False )
        
        self.encoder = VAEGAN_encoder( filters_gen,
                                        residual_layers,
                                        conv_size,
                                        norm=norm,
                                        latent_variables=latent_variables,
                                        forceconv=forceconv)
        
        self.decoder = VAEGAN_decoder(
                  filters_gen,
                  norm=norm,
                  latent_variables=latent_variables,
                    forceconv=forceconv)
        
        
        self.discriminator = VAEGAN_discriminator( 
                                filters_disc=filters_disc,
                                filters_gen=filters_gen,
                                conv_size=(3,3),
                                padding_mode='reflect',
                                norm=norm,
                                forceconv=forceconv)
                                
        self.latent_variables = latent_variables
        self.noise_channels = noise_channels
        
    def forward(self, variable_fields, constant_fields, mask=None ): #ensemble_size=1
        
        image, variable_fields, z_mean, z_logvar = self.generator(variable_fields, constant_fields)
        score = self.discriminator(image, variable_fields, constant_fields, mask )

        return score , image, z_mean, z_logvar
    
    def generator(self, variable_fields, constant_fields ): #ensemble_size=1


        b, _,  h, w = constant_fields.shape 
        _, _, _, d = variable_fields.shape

        # grouped convolution to reduce time dimension in variable fields
        variable_fields = einops.rearrange( variable_fields, '(b t) d h w -> b (t d) h w', t=4)
        variable_fields = self.grouped_conv2d_reduce_time( variable_fields ) #(b, d, h, w )
                
                            
        z_mean, z_logvar = self.encoder(variable_fields, constant_fields)      

        noise = torch.randn( (variable_fields.shape[0],
                                self.noise_channels,
                                *variable_fields.shape[2:]),
                                device=z_logvar.device)
                    
        image = self.decoder(z_mean, z_logvar, constant_fields, noise )
        
        image = torch.where( image<2.0, image, torch.tensor(2.0, device=image.device ))
                        
        return image, variable_fields, z_mean, z_logvar
                    
    def parse_model_args(parent_parser):
        model_parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=True, allow_abbrev=False)
        
        model_parser.add_argument("--filters_gen", default=84,type=int)
        model_parser.add_argument("--residual_layers", default=3, type=int)
        model_parser.add_argument("--conv_size", default=(3,3), type=tuple_type)
        model_parser.add_argument("--latent_variables", default=32, type=int)
        model_parser.add_argument("--noise_channels", default=32, type=int)
        model_parser.add_argument("--forceconv", action='store_true', default=False )
        
        # model_parser.add_argument("--latent_variables", default=32, type=int)
        
        model_parser.add_argument("--filters_disc", default=None,type=int)
        model_parser.add_argument("--norm", default=None,type=str)
        model_args = model_parser.parse_known_args()[0]
        
        if model_args.filters_disc is None: 
            model_args.filters_disc = model_args.filters_gen * 4
        
        return model_args
        
class VAEGAN_encoder(nn.Module):
    
    def __init__(self,
                    filters_gen=64,
                    residual_layers=3,
                    # input_channels=9,
                    # img_shape=(100, 100),
                    conv_size=(3, 3),
                    # padding_mode='reflect',
                    padding_mode='zeros',
                    stride=1,
                    relu_alpha=0.2,
                    norm=None,
                    dropout_rate=None,
                    latent_variables=50,
                    forceconv=True
                    ) -> None:
        super().__init__()
                
        # Constant Field Input Encoder
            # Input field (100, 140) -> (20, 20)
            # Input variable field needs to be cropped to match downscale on 
        const_field_inp = 1
        var_field_inp = 6
        const_field_k1 = (5, 7)
        const_field_k2 = (1, 1)
        
        self.inp_conv_cf = torch.nn.Sequential(
            nn.Conv2d(in_channels=const_field_inp,
                      out_channels=filters_gen, 
                      kernel_size=const_field_k1,
                      padding='valid',
                      stride = const_field_k1), nn.ReLU(),
            nn.Conv2d(in_channels=filters_gen,
                      out_channels=filters_gen,
                      kernel_size=const_field_k2,
                      padding='valid',
                      stride = const_field_k2), nn.ReLU())         
                
        self.residual_block = nn.Sequential(
            *[ ResidualBlock(
                    filters_gen if idx!=0 else filters_gen+var_field_inp, filters_gen, conv_size=(3,3),
                    stride=stride, relu_alpha=relu_alpha, norm=norm,
                    dropout_rate=dropout_rate, padding_mode=padding_mode,
                    force_1d_conv=forceconv) 
                for idx in range(residual_layers)
                ]
        )
        
        self.conv2d_mean = nn.Sequential( 
                                nn.Conv2d( filters_gen, latent_variables , kernel_size=1, padding='valid'),
                                nn.LeakyReLU(relu_alpha))
        
        self.conv2d_logvars = nn.Sequential( 
                                nn.Conv2d( filters_gen, latent_variables , kernel_size=1, padding='valid'),
                                nn.LeakyReLU(relu_alpha)
                                )
            
    def forward(self, variable_fields, constant_fields):
        """_summary_

        Args:
            variable_fields (_type_): (b, c, h, w)
            constant_fields (_type_): (b, c1, h1, w1) 
        """
        # Scale the constant fields down / encode them    
        constant_fields = self.inp_conv_cf(constant_fields)
        
        # # Concat fields
        x = torch.cat([variable_fields, constant_fields], axis=-3 )
        
        # Residual Block
        x1 = self.residual_block(x)
        
        # Add noise to log_vars
        z_mean = self.conv2d_mean(x1)
        z_logvar = self.conv2d_logvars(x1)
                    
        return z_mean, z_logvar

class VAEGAN_decoder(nn.Module):
    def __init__(self,
                 filters_gen,
                 relu_alpha=0.2, 
                 stride=1,
                 norm=None,
                 dropout_rate=0.0,
                #  padding_mode='reflect',
                 padding_mode='zeros',
                 conv_size=(3,3),
                 num_layers_res1=3,
                 num_layers_res2=3,
                 forceconv=True,
                 latent_variables=50
                 ):
        super().__init__()
        
        # if arch == "forceconv-long"
        self.residual_block1 = nn.Sequential(
            *( ResidualBlock(
                    filters_gen if idx!=0 else latent_variables , filters_gen, conv_size=conv_size,
                    stride=stride, relu_alpha=relu_alpha, norm=norm,
                    dropout_rate=dropout_rate, padding_mode=padding_mode,
                    force_1d_conv=forceconv) 
                for idx in range(num_layers_res1) )
        )
        
        # Upsampling from (10,10) to (100,100) with alternating residual blocks
        us_block_channels = [2*filters_gen, filters_gen]
        
        const_field_inp = 1
        const_field_k1 = (1.0, 1.0 )
        const_field_k2 = (5.0, 7.0 )
        self.upsample_residual_block = nn.Sequential(
            nn.UpsamplingBilinear2d( scale_factor = const_field_k1),
            ResidualBlock(filters_gen, us_block_channels[0], conv_size, stride, relu_alpha, norm, dropout_rate, padding_mode, forceconv) ,
            nn.UpsamplingBilinear2d( scale_factor =  const_field_k2),
            # ResidualBlock(filters_gen, us_block_channels[1], conv_size, stride, relu_alpha, norm, dropout_rate, padding_mode, forceconv)            
            ResidualBlock(us_block_channels[0], us_block_channels[1], conv_size, stride, relu_alpha, norm, dropout_rate, padding_mode, forceconv)            
        )
                
        self.residual_block2 = nn.Sequential(
                                    *( ResidualBlock(
                                            us_block_channels[-1]+1 if idx==0 else filters_gen, 
                                            filters_gen, conv_size=(3,3),
                                            stride=stride, relu_alpha=relu_alpha, norm=None,
                                            dropout_rate=dropout_rate, padding_mode=padding_mode,
                                            force_1d_conv=forceconv) 
                                        for idx in range(num_layers_res2) )
                                    )
        
        self.output_layer = nn.Sequential(
            nn.Conv2d(filters_gen, 1, (1,1)),
            nn.Softplus()
            # nn.ReLU()
        )
    
    def forward(self, z_mean , z_logvar, constant_field, noise):
        """_summary_

            Args:
                x (c1, h1, w1): _description_
                constant_field (_type_): (c, h, w) )
        """
        
        # Generated noised output
        sample_z = torch.mul( noise, torch.exp(z_logvar*0.5) ) + z_mean
        
        # residual block 1
        x = self.residual_block1(sample_z)
        
        # upsampling residual block
        x1 = self.upsample_residual_block(x)
        
        # concatenate with constant field
        x2 = torch.cat([x1, constant_field], dim=-3 )
        
        # residual block 2
        x3 = self.residual_block2(x2)
        
        # softplus        
        x4 = self.output_layer(x3)    
        
        return x4
        
class VAEGAN_discriminator(nn.Module):
    def __init__(self,
                  filters_disc=64,
                  filters_gen=64,
                  conv_size=(3, 3),
                  padding=None,
                  stride=1,
                  relu_alpha=0.2,
                  norm=None,
                  dropout_rate=None,
                  num_resid_block_field=2,
                  padding_mode='reflective',
                  forceconv = False,
                  ) -> None:
        super().__init__()
        
        # 
        self.inp_conv_cf = nn.Conv2d(1,filters_disc, kernel_size=(5,7), stride=(5,7), padding='valid')
        torch.nn.init.xavier_uniform_(self.inp_conv_cf.weight )
        # Constant Field Input Encoder
        const_field_inp = constant_field_input_channels = 1
        const_field_k1 = (1,1) 
        const_field_k2 = (5,7)
        
        
        self.field_conv_block = torch.nn.Sequential(
            nn.Conv2d(in_channels=const_field_inp,
                      out_channels=filters_disc, 
                      kernel_size=const_field_k1,
                      padding='valid',stride = const_field_k1),nn.ReLU(),
            nn.Conv2d(in_channels=filters_disc,
                      out_channels=filters_disc*2, 
                      kernel_size=const_field_k2,
                      padding='valid',stride = const_field_k2),nn.ReLU() 
            )         
        
        # Residual Block for variable fields and downscaled topology field
        block_outp_channels = [filters_disc, filters_disc*2 ]
        block_inp_channels = [const_field_inp+filters_disc] + block_outp_channels[1:]
        
        self.residual_block_field = torch.nn.Sequential(
            *[
                ResidualBlock(filters_disc+6 if idx==0 else block_outp_channels[idx-1],
                              block_outp_channels[idx],
                              conv_size,
                              stride=stride, relu_alpha=relu_alpha, norm=None,
                              dropout_rate=dropout_rate, padding_mode=padding_mode,
                              force_1d_conv=forceconv) 
                for idx in range(num_resid_block_field)
            ]
        )


        # Downscaling Residual Block for
        block_outp_channels = [filters_disc, filters_disc*2 ]
        block_inp_channels = [1 +  const_field_inp ] + block_outp_channels[1:]
        self.ds_resid_block_img = nn.Sequential(
            nn.Conv2d(block_inp_channels[0], block_outp_channels[0],  (5,7), (5,7), padding='valid'),
            nn.ReLU(),
            ResidualBlock(block_outp_channels[0],block_inp_channels[1] ,conv_size, stride, relu_alpha, None, dropout_rate, padding_mode, forceconv),
            
            nn.Conv2d(block_inp_channels[1], block_outp_channels[1],  (1,1), (1,1), padding='valid'),
            nn.ReLU(),
            ResidualBlock( block_outp_channels[1], block_outp_channels[1], conv_size, stride, relu_alpha, None, dropout_rate, padding_mode, forceconv)            
        )
        
        # Residual block for output
        self.outp_residual_block = nn.Sequential(
            ResidualBlock(block_outp_channels[-1]*2, filters_disc, conv_size, stride, relu_alpha, None, dropout_rate, padding_mode, forceconv),
            ResidualBlock(filters_disc, filters_disc, conv_size, stride, relu_alpha, None, dropout_rate, padding_mode, forceconv)
        )
        
        self.output_layer = nn.Sequential(
            nn.Linear(filters_disc, filters_disc//2), nn.ReLU(),
            nn.Linear(filters_disc//2, 1)    
        )

    def forward(self, image, variable_fields, constant_fields, mask=None):
        
        # Scale the constant fields down / encode them    
        constant_fields_conv = self.inp_conv_cf(constant_fields)
        
        #Concatenations
        fields = torch.cat([variable_fields, constant_fields_conv], dim=-3)
        image = torch.cat([image, constant_fields], dim=-3)
        
        # Residual Block Field
        fields = self.residual_block_field(fields)
        
        # Downsampling residual block image
        image = self.ds_resid_block_img(image)
        
        x = torch.cat([image, fields], dim=-3)
        
        # Residual Block
        x = self.outp_residual_block(x)
        
        # Global Avg 2D Pooling
        if mask is not None:
            #Ignore masked values during Avg2D Pooling
            #scaling mask down
            mask_pooled = nn.functional.avg_pool2d(mask.to(torch.float), (5,7), stride=(5,7), count_include_pad=False  )            
            mask_pooled = mask_pooled[:,None]
            mask_pooled = torch.where( mask_pooled>=0.5, torch.ones_like(x), torch.zeros_like(x) )
            
            x = x * mask_pooled
            x = x.sum( dim=[-2,-1], keepdim=False) 
            x = x / mask_pooled.sum(dim=[-2,-1])
            
        else:
            x = torch.mean(x, dim=[-2,-1] )
        
        # Output Layer
        x = self.output_layer(x)
        
        return x
                   
def residualblock_innerblock(in_channels, filters, conv_size, padding_mode, relu_alpha, norm, dropout_rate):
    block_layers = nn.Sequential()
    block_layers.append( nn.LeakyReLU(relu_alpha) )
    # block_layers.append( nn.ReflectionPad2d( padding = tuple((s-1)//2 for s in conv_size)   ) )# only works if s is odd!
    
    # if (filters != in_channels):
    conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=filters,
            kernel_size=conv_size,
            padding='same',
            padding_mode=padding_mode #convert all padding tf words to equivalent pytorch ones
        )
    torch.nn.init.xavier_uniform_(conv2d.weight)
    block_layers.append(conv2d)
    
    if norm == "batch":
        block_layers.append(
            nn.BatchNorm2d(filters)
        )            
    if dropout_rate is not None:
        block_layers.append(
            nn.Dropout(dropout_rate)
            )
    return block_layers

class ResidualBlock_innerblock(nn.Module):
    
    def __init__(self, in_channels, filters, conv_size, padding_mode, relu_alpha, norm, dropout_rate) -> None:
        super().__init__()
        
        self.act = nn.LeakyReLU(relu_alpha)
        
        self.conv2d = nn.Conv2d(
                in_channels=in_channels,
                out_channels=filters,
                kernel_size=conv_size,
                padding='same',
                padding_mode=padding_mode #convert all padding tf words to equivalent pytorch ones
            )
        torch.nn.init.xavier_uniform_(self.conv2d.weight)

        self.bn = nn.BatchNorm2d(filters) if norm=="batch" else nn.Identity()           
        
        self.do = nn.Dropout(dropout_rate) if dropout_rate is not None else nn.Identity()
                

    def forward(self, x):
        
        x = self.act(x)
        x = self.conv2d(x)
        x = self.bn(x)
        x = self.do(x)        
        
        return x


def residualblock_highway(filters, in_channels, force_1d_conv, stride):
    
    highway_block = nn.Sequential()
    if stride > 1:
        highway_block.append( nn.AvgPool2d( kernel_size=(stride,stride))  )
    if (filters != in_channels) or force_1d_conv:
        highway_block.append( nn.Conv2d(in_channels=in_channels,
                                        out_channels=filters,
                                        kernel_size=(1, 1)) )
    if len(highway_block)==0:
        return torch.nn.Identity()
    
    return highway_block


class ResidualBlock_highway(nn.Module):
    
    def __init__(self, filters, in_channels, force_1d_conv, stride) -> None:
        super().__init__()
        
        
        self.ap2d =  nn.AvgPool2d( kernel_size=(stride,stride))  if stride > 1 else nn.Identity()
        
        self.conv2d = nn.Conv2d(in_channels=in_channels,
                                            out_channels=filters,
                                            kernel_size=(1, 1))  if (filters != in_channels) \
                        or force_1d_conv else nn.Identity()
                        
        # if len(highway_block)==0:
        #     return torch.nn.Identity()
    
    def forward(self, x):
        x = self.ap2d(x)
        x = self.conv2d(x)
        return x
    
    

class ResidualBlock(nn.Module):
    
    def __init__(self,
                 in_channels,
                 filters, conv_size=(3, 3), stride=1, 
                 relu_alpha=0.2, norm=None, dropout_rate=None, 
                 padding_mode='reflect', force_1d_conv=False
                 ) -> None:
        super().__init__()
        
        # in_channels = x.shape[-3]
        
        #inner highway_block
        # self.highway_block = residualblock_highway(filters, in_channels, force_1d_conv, stride)
        self.highway_block = ResidualBlock_highway(filters, in_channels, force_1d_conv, stride)
        
        # first block of activation and 3x3 convolution
        self.block1 = ResidualBlock_innerblock(in_channels, filters, conv_size, padding_mode, relu_alpha, norm, dropout_rate)
        # self.block1 = residualblock_innerblock(in_channels, filters, conv_size, padding_mode, relu_alpha, norm, dropout_rate)
        
        # second block of activation and 3x3 convolution
        self.block2 = ResidualBlock_innerblock(filters, filters, conv_size, padding_mode, relu_alpha, norm, dropout_rate)
        # self.block2 = residualblock_innerblock(filters, filters, conv_size, padding_mode, relu_alpha, norm, dropout_rate)
                 
    def forward(self, x):
        """_summary_

        Args:
            x (_type_): (b, c, h, w)
        """
        
                        
        x1 = self.block1(x)
        
        x2 = self.block2(x1)
        
        x3 = x2 + self.highway_block(x)
        
        return x3
        
    
