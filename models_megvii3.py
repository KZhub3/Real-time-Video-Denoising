"""
Definition of the FastDVDnet model

Copyright (C) 2019, Matias Tassano <matias.tassano@parisdescartes.fr>

This program is free software: you can use, modify and/or
redistribute it under the terms of the GNU General Public
License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later
version. You should have received a copy of this license along
this program. If not, see <http://www.gnu.org/licenses/>.
"""
import torch
import torch.nn as nn

class SepConv(nn.Module):
	'''(Conv2d => BN => ReLU) x 2'''
	def __init__(self, in_ch, out_ch, kernel_size, stride, padding1 = 1, padding2 =0):
		super(SepConv, self).__init__()
		self.convblock = nn.Sequential(
			##print(in_ch,out_ch,min(out_ch,in_ch)),
			nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, padding=padding1, stride = stride, groups = in_ch, bias=False),
			nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=padding2, bias=False)

		)

	def forward(self, x):
		return self.convblock(x)

class deconv(nn.Module):

	def __init__(self,in_ch,out_ch,kernel_size,stride,output_padding =1):
		super(deconv, self).__init__()
		self.convblcok = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride = stride,\
		output_padding=output_padding, padding = 0, bias=False)
		#nn.BatchNorm2d(out_ch),
	#	nn.ReLU(inplace=True),

	def forward(self,x):
		return self.convblcok(x)

class EncoderBlock(nn.Module):
	'''Downscale + (Conv2d => BN => ReLU)*2'''
	def __init__(self, in_ch):
		super(EncoderBlock, self).__init__()
		self.convblock = nn.Sequential(
			SepConv(in_ch, (in_ch//4), kernel_size = 5, stride = 1, padding1 = 2 ),
			#nn.BatchNorm2d(in_ch//4),
			nn.ReLU(inplace=True),
			SepConv((in_ch//4), in_ch, kernel_size = 5, stride = 1, padding1 = 2)
			#nn.BatchNorm2d(in_ch),
	#		nn.ReLU(inplace=True),
		)

	def forward(self, x):
		return self.convblock(x) + x

class DownSampleBlock(nn.Module):
	'''Downscale + (Conv2d => BN => ReLU)*2'''
	def __init__(self, in_ch, out_ch, stride = 2, padding = 2):
		super(DownSampleBlock, self).__init__()
		self.convblock = nn.Sequential(
			SepConv(in_ch, in_ch//4, kernel_size = 5, stride = stride, padding1 = padding),
			#nn.BatchNorm2d(in_ch//4),
			nn.ReLU(inplace=True),
		#	nn.BatchNorm2d(out_ch),
			SepConv(in_ch//4, out_ch, kernel_size = 5, stride = 1, padding1 = padding)
			#nn.BatchNorm2d(out_ch,
	#		nn.ReLU(inplace=True),
		)
		#super(DownSampleBlock, self).__init__()
		#self.convblock2 = nn.Sequential(SepConv(in_ch, out_ch, kernel_size = 3, stride = 2))

	def forward(self, x):
		return self.convblock(x)


class DownSampleBlock2(nn.Module):
	'''Downscale + (Conv2d => BN => ReLU)*2'''
	def __init__(self, in_ch, out_ch):
		super(DownSampleBlock2, self).__init__()
		self.convblock = nn.Sequential(
			SepConv(in_ch, in_ch//4, kernel_size = 5, stride = (2,3)),
		#	nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True),
			SepConv(in_ch//4, out_ch, kernel_size = 5, stride = 1)
		)
		self.convblock = SepConv(in_ch, out_ch, kernel_size = 3, stride = (2,3))

	def forward(self, x):
		return self.convblock(x)


class DownSampleBlocklink(nn.Module):
	'''Downscale + (Conv2d => BN => ReLU)*2'''
	def __init__(self, in_ch, out_ch, stride =2):
		super(DownSampleBlocklink, self).__init__()

		#super(DownSampleBlock, self).__init__()
		self.convblock= nn.Sequential(SepConv(in_ch, out_ch, kernel_size = 3, stride = stride))

	def forward(self, x):
		return self.convblock(x)

class DownSampleBlock2link(nn.Module):
	'''Downscale + (Conv2d => BN => ReLU)*2'''
	def __init__(self, in_ch, out_ch):
		super(DownSampleBlock2link, self).__init__()

		self.convblock = SepConv(in_ch, out_ch, kernel_size = 3, stride = (2,3))

	def forward(self, x):
		return self.convblock(x)

class UpSampleBlock(nn.Module):
	'''Downscale + (Conv2d => BN => ReLU)*2'''
	def __init__(self, in_ch, out_ch, stride =2, output_padding = 0):
		super(UpSampleBlock, self).__init__()
		self.convblock = deconv(in_ch, out_ch, kernel_size = 2,  stride = stride, output_padding = output_padding)

	def forward(self, x):
		return self.convblock(x)


class UpSampleBlock2(nn.Module):
	'''Downscale + (Conv2d => BN => ReLU)*2'''
	def __init__(self, in_ch, out_ch):
		super(UpSampleBlock2, self).__init__()
		self.convblock = deconv(in_ch, out_ch, kernel_size = 2,  stride = (2,3))

	def forward(self, x):
		return self.convblock(x)


class DecoderBlock(nn.Module):
	'''Downscale + (Conv2d => BN => ReLU)*2'''
	def __init__(self, in_ch):
		super(DecoderBlock, self).__init__()
		self.convblock = nn.Sequential(
			SepConv(in_ch, (in_ch//4), kernel_size = 3, stride = 1),
			#nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True),
			SepConv((in_ch//4), in_ch, kernel_size = 3, stride = 1)
		)

	def forward(self, x):
		return self.convblock(x) + x

class InputCvBlock(nn.Module):
	'''(Conv with num_in_frames groups => BN => ReLU) + (Conv => BN => ReLU)'''
	def __init__(self, num_in_frames, out_ch):
		super(InputCvBlock, self).__init__()
		self.interm_ch = 30
		self.convblock = nn.Sequential(
			nn.Conv2d(num_in_frames*(3+1), num_in_frames*self.interm_ch, \
					  kernel_size=3, padding=1, groups=num_in_frames, bias=False),
		#	nn.BatchNorm2d(num_in_frames*self.interm_ch),
			nn.ReLU(inplace=True),
			nn.Conv2d(num_in_frames*self.interm_ch, out_ch, kernel_size=3, padding=1, bias=False),
		#	nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		return self.convblock(x)

class InputStage(nn.Module):
	'''(Conv2d => BN => ReLU) x 2'''
	def __init__(self, num_in_frames, out_ch):
		super(InputStage, self).__init__()
		self.convblock = nn.Sequential(nn.Conv2d(num_in_frames*(3+1), out_ch, kernel_size=3, padding=1,stride = 1, bias=False))

	def forward(self, x):
		return self.convblock(x)

# Here are all the blocks and now we go to the net

class MegviiBlock(nn.Module):
	""" Definition of the denosing block of FastDVDnet.
	Inputs of constructor:
		num_input_frames: int. number of input frames
	Inputs of forward():
		xn: input frames of dim [N, C, H, W], (C=3 RGB)
		noise_map: array with noise map of dim [N, 1, H, W]
	"""

	def __init__(self, num_input_frames=3):
		super(MegviiBlock, self).__init__()
		self.chs_lyr0 = 16
		self.chs_lyr5 = 32
		self.chs_lyr1 = 64
		self.chs_lyr2 = 128
		self.chs_lyr3 = 256
		self.chs_lyr4 = 512

		#self.SepConv5 = SepConv(in_ch, out_ch, 5)

		self.inputSt = InputStage(num_in_frames=num_input_frames, out_ch=self.chs_lyr0)
		self.DownSampleBlock1 = DownSampleBlock(self.chs_lyr0, self.chs_lyr1)
		self.DownSampleBlocklink1 = DownSampleBlocklink(self.chs_lyr0, self.chs_lyr1)
		self.EncoderBlock1 = EncoderBlock(self.chs_lyr1)

		self.DownSampleBlock2 = DownSampleBlock(self.chs_lyr1, self.chs_lyr2)
		self.DownSampleBlocklink2 = DownSampleBlocklink(self.chs_lyr1, self.chs_lyr2)
		self.EncoderBlock2 = EncoderBlock(self.chs_lyr2)

		self.DownSampleBlock3 = DownSampleBlock(self.chs_lyr2, self.chs_lyr3, stride = 3)
		self.DownSampleBlocklink3 = DownSampleBlocklink(self.chs_lyr2, self.chs_lyr3, stride = 3)
		self.EncoderBlock3 = EncoderBlock(self.chs_lyr3)

		self.DownSampleBlock4 = DownSampleBlock(self.chs_lyr3, self.chs_lyr4, stride =5)
		self.DownSampleBlocklink4 = DownSampleBlocklink(self.chs_lyr3, self.chs_lyr4, stride = 5)
		self.EncoderBlock4 = EncoderBlock(self.chs_lyr4)
		#然而540只能除以4，到这里已经除以16 了,只能是2235，oR少来几层
		#两个办法，见草稿

		self.DecoderBlock1 = DecoderBlock(self.chs_lyr4)
		self.UpSampleBlock1 = UpSampleBlock(self.chs_lyr4, self.chs_lyr1, stride =5, output_padding = 3)
		self.link1 = SepConv(self.chs_lyr3, self.chs_lyr1, kernel_size = 3, stride = 1, padding1 = 1, padding2 =0)

		self.DecoderBlock2 = DecoderBlock(self.chs_lyr1)
		self.UpSampleBlock2 = UpSampleBlock(self.chs_lyr1, self.chs_lyr5, stride =3, output_padding = 1)
		self.link2 = SepConv(self.chs_lyr2, self.chs_lyr5, kernel_size = 3, stride = 1, padding1 = 1, padding2 =0)

		self.DecoderBlock3 = DecoderBlock(self.chs_lyr5)
		self.UpSampleBlock3 = UpSampleBlock(self.chs_lyr5, self.chs_lyr5)
		self.link3 = SepConv(self.chs_lyr1, self.chs_lyr5, kernel_size = 3, stride = 1, padding1 = 1, padding2 =0)

		self.DecoderBlock4 = DecoderBlock(self.chs_lyr5)
		self.UpSampleBlock4 = UpSampleBlock(self.chs_lyr5, self.chs_lyr0)
		self.link4 = SepConv(self.chs_lyr0, self.chs_lyr0, kernel_size = 3, stride = 1, padding1 = 1, padding2 =0)
		self.DecoderBlock5 = DecoderBlock(self.chs_lyr0)
		self.outconv = nn.Conv2d(self.chs_lyr0, 3, kernel_size=3, padding=1, bias=False)
		self.reset_params()

	@staticmethod
	def weight_init(m):
		if isinstance(m, nn.Conv2d):
			nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

	def reset_params(self):
		for _, m in enumerate(self.modules()):
			self.weight_init(m)

	def forward(self, in0, in1, in2, noise_map):
		'''Args:
			inX: Tensor, [N, C, H, W] in the [0., 1.] range
			noise_map: Tensor [N, 1, H, W] in the [0., 1.] range
		'''

		#Input Stage
		x0 = self.inputSt(torch.cat((in0, noise_map, in1, noise_map, in2, noise_map), dim=1))
		x11 = torch.cat((in1, noise_map), dim=1)
		# Downsampling
		x1 = self.DownSampleBlock1(x0)
		temp = self.DownSampleBlocklink1(x0)
	#	print(in0.shape, x0.shape,x1.shape, temp.shape)
		x1 = x1 + self.DownSampleBlocklink1(x0)
		x1 = self.EncoderBlock1(x1) 

		x2 = self.DownSampleBlock2(x1)
		x2 = x2 + self.DownSampleBlocklink2(x1)
		x2 = self.EncoderBlock2(x2) 
		#print(x2.shape)

		x3 = self.DownSampleBlock3(x2)
		temp = self.DownSampleBlocklink3(x2)
		#print(x3.shape, temp.shape)
		x3 = x3 + self.DownSampleBlocklink3(x2)
		x3 = self.EncoderBlock3(x3) 

		x4 = self.DownSampleBlock4(x3)
		x4 = x4 + self.DownSampleBlocklink4(x3)
		x4 = self.EncoderBlock4(x4)
		
		# Upsampling
		x5 = self.DecoderBlock1(x4)
		#print(x5.shape,x3.shape)
		x5 = self.UpSampleBlock1(x5)
	#	print(x5.shape,x3.shape)
		x5 = x5 + self.link1(x3)

		x6 = self.DecoderBlock2(x5)
		x6 = self.UpSampleBlock2(x6)
		x6 = x6 + self.link2(x2)

		x7 = self.DecoderBlock3(x6)
		x7 = self.UpSampleBlock3(x7)
		#print(x7.shape,x1.shape)
		x7 = x7 + self.link3(x1)

		x8 = self.DecoderBlock4(x7)
		x8 = self.UpSampleBlock4(x8)
		x8 = x8 + self.link4(x0)
		#print(x8.shape)

		x9 = self.DecoderBlock5(x8)
		x10 = self.outconv(x9)
		
		# Estimation
		##print(x10.shape,x11.shape)
		#x = x10+x11
		x = in1 - x10

		

		return x

class FastDVDnet(nn.Module):
	""" Definition of the FastDVDnet model.
	Inputs of forward():
		xn: input frames of dim [N, C, H, W], (C=3 RGB)
		noise_map: array with noise map of dim [N, 1, H, W]
	"""

	def __init__(self, num_input_frames=5):
		super(FastDVDnet, self).__init__()
		self.num_input_frames = num_input_frames
		# Define models of each denoising stage
		self.temp2 = MegviiBlock(num_input_frames=3)
		# Init weights
		self.reset_params()

	@staticmethod
	def weight_init(m):
		if isinstance(m, nn.Conv2d):
			nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

	def reset_params(self):
		for _, m in enumerate(self.modules()):
			self.weight_init(m)

	def forward(self, x, noise_map):
		'''Args:
			x: Tensor, [N, num_frames*C, H, W] in the [0., 1.] range
			noise_map: Tensor [N, 1, H, W] in the [0., 1.] range
		'''
		# Unpack inputs
		(x1, x2, x3) = tuple(x[:, 3*m:3*m+3, :, :] for m in range(self.num_input_frames))

		# First stage
		#x20 = self.temp1(x0, x1, x2, noise_map)
		#x21 = self.temp1(x1, x2, x3, noise_map)
		#x22 = self.temp1(x2, x3, x4, noise_map)

		#Second stage
		x = self.temp2(x1, x2, x3, noise_map)

		return x
