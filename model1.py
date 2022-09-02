import networks
import torch
import torch.nn as nn
import torch.nn.functional as F
import time


class CodeCycleGAN(nn.Module):
  def __init__(self, opts):
    super(CodeCycleGAN, self).__init__()

    # parameters
    lr = opts.lr
    self.nz = 8
    self.codebook_size = opts.codebook_size
    self.embed_dim = opts.embed_dim
    self.beta = 0.25
    self.concat = opts.concat
    self.lambdaB = opts.lambdaB
    self.lambdaI = opts.lambdaI

    # codebook
    self.quantize_s = networks.VectorQuantizer(self.codebook_size, self.embed_dim, self.beta)
    self.quantize_b = networks.VectorQuantizer(self.codebook_size, self.embed_dim, self.beta)

    # discriminators
    if opts.dis_scale > 1:
      self.disA = networks.MultiScaleDis(opts.input_dim_a, opts.dis_scale, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
      self.disB = networks.MultiScaleDis(opts.input_dim_b, opts.dis_scale, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
    else:
      self.disA = networks.Dis(opts.input_dim_a, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
      self.disB = networks.Dis(opts.input_dim_b, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
        
    # encoders
    self.enc = networks.E_content(opts.input_dim_a, opts.input_dim_b)

    # generators
    self.gen = networks.G_concat_codebook(opts.input_dim_a)

    # optimizers
    self.quantize_s_opt = torch.optim.Adam(self.quantize_s.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
    self.quantize_b_opt = torch.optim.Adam(self.quantize_b.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
 
    self.disA_opt = torch.optim.Adam(self.disA.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
    self.disB_opt = torch.optim.Adam(self.disB.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
    self.enc_opt = torch.optim.Adam(self.enc.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
    
    self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
    # self.gen_codebook_opt = torch.optim.Adam(self.gen_codebook.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)

    # Setup the loss function for training
    self.criterionL1 = torch.nn.L1Loss()
    self.criterionL2 = torch.nn.MSELoss()
    if opts.percep == 'default':
        self.perceptualLoss = networks.PerceptualLoss(nn.MSELoss(), opts.gpu, opts.percp_layer)
    elif opts.percep == 'face':
        self.perceptualLoss = networks.PerceptualLoss16(nn.MSELoss(), opts.gpu, opts.percp_layer)
    else:
        self.perceptualLoss = networks.MultiPerceptualLoss(nn.MSELoss(), opts.gpu)

  # load weights from scratch
  def initialize(self):
    self.quantize_s.apply(networks.gaussian_weights_init)
    self.quantize_b.apply(networks.gaussian_weights_init)
    self.disA.apply(networks.gaussian_weights_init)
    self.disB.apply(networks.gaussian_weights_init)
    self.gen.apply(networks.gaussian_weights_init)
    self.enc.apply(networks.gaussian_weights_init)


  # load weights when training from resume
  def set_scheduler(self, opts, last_ep=0):
    self.quantize_s_sch = networks.get_scheduler(self.quantize_s_opt, opts, last_ep)
    self.quantize_b_sch = networks.get_scheduler(self.quantize_b_opt, opts, last_ep)
    self.disA_sch = networks.get_scheduler(self.disA_opt, opts, last_ep)
    self.disB_sch = networks.get_scheduler(self.disB_opt, opts, last_ep)
    self.enc_sch = networks.get_scheduler(self.enc_opt, opts, last_ep)
    self.gen_sch = networks.get_scheduler(self.gen_opt, opts, last_ep)

  def setgpu(self, gpu):
    self.gpu = gpu
    self.quantize_s.cuda(self.gpu)
    self.quantize_b.cuda(self.gpu)
    self.disA.cuda(self.gpu)
    self.disB.cuda(self.gpu)
    self.enc.cuda(self.gpu)
    self.gen.cuda(self.gpu)


  
  # transfer blurred image to sharp image
  def test_forward(self, image, a2b=True):
    if a2b:
        self.z_content = self.enc_c.forward_b(image)
        self.mu_b, self.logvar_b = self.enc_a.forward(image)
        std_b = self.logvar_b.mul(0.5).exp_()
        eps_b = self.get_z_random(std_b.size(0), std_b.size(1), 'gauss')
        self.z_attr_b = eps_b.mul(std_b).add_(self.mu_b)
        output = self.gen.forward_D(self.z_content, self.z_attr_b)
    
    return output


  def forward(self):
    # input images
    real_I = self.input_I
    real_B = self.input_B

    self.real_I_encoded = real_I
    self.real_B_encoded = real_B

    # get z_content_i and z_content_b
    self.z_content_i, self.z_content_b = self.enc.forward(self.real_I_encoded, self.real_B_encoded)

    self.z_codebook_output_i_s, _, _ = self.quantize_s(self.z_content_i)
    self.z_codebook_output_i_b, _, _ = self.quantize_b(self.z_content_i)
    self.z_codebook_output_b_s, _, _ = self.quantize_s(self.z_content_b)
    self.z_codebook_output_b_b, _, _ = self.quantize_b(self.z_content_b)

    self.recon_Ii_encoded = self.gen.forward(self.z_codebook_output_i_s)
    self.recon_Ib_encoded = self.gen.forward(self.z_codebook_output_i_b)
    self.recon_Bi_encoded = self.gen.forward(self.z_codebook_output_b_s)
    self.recon_Bb_encoded = self.gen.forward(self.z_codebook_output_b_b)

    # get reconstructed encoded z_c
    self.z_content_recon_b, self.z_content_recon_i = self.enc.forward(self.recon_Bi_encoded, self.recon_Ib_encoded)

    self.z_codebook_output_recon_b, _, _ = self.quantize_b(self.z_content_recon_b)
    self.z_codebook_output_recon_i, _, _ = self.quantize_s(self.z_content_recon_i)

    # get final(cycle) reconstructed images
    self.recon_I_hat = self.gen.forward(self.z_codebook_output_recon_i)
    self.recon_B_hat = self.gen.forward(self.z_codebook_output_recon_b)  


  def update_D(self, image_a, image_b):
    self.input_I = image_a
    self.input_B = image_b
    self.forward()

    # update disA
    self.disA_opt.zero_grad()
    loss_D1_A = self.backward_D(self.disA, self.real_I_encoded, self.recon_Bi_encoded)
    self.disA_loss = loss_D1_A.item()
    self.disA_opt.step()

    # update disB
    self.disB_opt.zero_grad()
    loss_D1_B = self.backward_D(self.disB, self.real_B_encoded, self.recon_Ib_encoded)
    self.disB_loss = loss_D1_B.item()
    self.disB_opt.step()


  def backward_D(self, netD, real, fake):
    pred_fake = netD.forward(fake.detach())
    pred_real = netD.forward(real)
    loss_D = 0
    for it, (out_a, out_b) in enumerate(zip(pred_fake, pred_real)):
      out_fake = nn.functional.sigmoid(out_a)
      out_real = nn.functional.sigmoid(out_b)
      all0 = torch.zeros_like(out_fake).cuda(self.gpu)
      all1 = torch.ones_like(out_real).cuda(self.gpu)
      ad_fake_loss = nn.functional.binary_cross_entropy(out_fake, all0)
      ad_true_loss = nn.functional.binary_cross_entropy(out_real, all1)
      loss_D += ad_true_loss + ad_fake_loss
    loss_D.backward()
    return loss_D

  def update_ECG(self):
    # update G, E, codebook_s, codebook_b
    self.quantize_s_opt.zero_grad()
    self.quantize_b_opt.zero_grad()
    self.enc_opt.zero_grad()
    self.gen_opt.zero_grad()
    self.backward_ECG()
    self.quantize_s_opt.step()
    self.quantize_b_opt.step()
    self.enc_opt.step()
    self.gen_opt.step()

  def backward_ECG(self):
    # Ladv for generator
    loss_G_GAN_I = self.backward_G_GAN(self.recon_Bi_encoded, self.disA)
    loss_G_GAN_B = self.backward_G_GAN(self.recon_Ib_encoded, self.disB)

    # cross cycle consistency loss
    loss_G_L1_I = self.criterionL1(self.recon_Ii_encoded, self.real_I_encoded) * 10
    loss_G_L1_B = self.criterionL1(self.recon_Bb_encoded, self.real_B_encoded) * 10
    loss_G_L1_II = self.criterionL1(self.recon_I_hat, self.real_I_encoded) * 10
    loss_G_L1_BB = self.criterionL1(self.recon_B_hat, self.real_B_encoded) * 10

    # code feat loss
    _, code_feat_loss_1, _ = self.quantize_s(self.z_content_i)
    _, code_feat_loss_2, _ = self.quantize_b(self.z_content_i)
    _, code_feat_loss_3, _ = self.quantize_s(self.z_content_b)
    _, code_feat_loss_4, _ = self.quantize_b(self.z_content_b)
    _, code_feat_loss_5, _ = self.quantize_s(self.z_content_recon_i)
    _, code_feat_loss_6, _ = self.quantize_b(self.z_content_recon_b)
    code_feat_loss = code_feat_loss_1 + code_feat_loss_2 + \
                     code_feat_loss_3 + code_feat_loss_4 + \
                     code_feat_loss_6 + code_feat_loss_6
    
    # perceptual losses
    percp_loss_B = self.perceptualLoss.getloss(self.recon_Bi_encoded, self.real_B_encoded) * self.lambdaB
    percp_loss_I = self.perceptualLoss.getloss(self.recon_Ib_encoded, self.real_I_encoded) * self.lambdaI

    loss_G = loss_G_GAN_I + loss_G_GAN_B + \
             loss_G_L1_II + loss_G_L1_BB + \
             loss_G_L1_I + loss_G_L1_B + \
             percp_loss_B + percp_loss_I + code_feat_loss

    loss_G.backward(retain_graph=True)

    self.gan_loss_i = loss_G_GAN_I.item()
    self.gan_loss_b = loss_G_GAN_B.item()

    self.loss_code_feat = code_feat_loss.item()
    self.l1_recon_I_loss = loss_G_L1_I.item()
    self.l1_recon_B_loss = loss_G_L1_B.item()
    self.l1_recon_II_loss = loss_G_L1_II.item()
    self.l1_recon_BB_loss = loss_G_L1_BB.item()
    self.B_percp_loss = percp_loss_B.item()
    self.G_loss = loss_G.item()


  def backward_G_GAN(self, fake, netD=None):
    outs_fake = netD.forward(fake)
    loss_G = 0
    for out_a in outs_fake:
      outputs_fake = nn.functional.sigmoid(out_a)
      all_ones = torch.ones_like(outputs_fake).cuda(self.gpu)
      loss_G = loss_G + nn.functional.binary_cross_entropy(outputs_fake, all_ones)
    return loss_G

  def backward_G_alone(self):
    # Ladv for generator
    loss_G_GAN2_I = self.backward_G_GAN(self.fake_I_random, self.disA2)
    loss_G_GAN2_B = self.backward_G_GAN(self.fake_B_random, self.disB2)

    # latent regression loss
    if self.concat:
      loss_z_L1_b = torch.mean(torch.abs(self.mu2_b - self.z_random)) * 10
    else:
      loss_z_L1_b = torch.mean(torch.abs(self.z_attr_random_b - self.z_random)) * 10
    
    # perceptual losses
    percp_loss_B2 = self.perceptualLoss.getloss(self.fake_I_random, self.real_B_encoded) * self.lambdaB
    percp_loss_I2 = self.perceptualLoss.getloss(self.fake_B_random, self.real_I_encoded) * self.lambdaI


    loss_G2 = loss_z_L1_b + loss_G_GAN2_I + loss_G_GAN2_B + percp_loss_B2 + percp_loss_I2
    loss_G2.backward()
    self.gan2_loss_a = loss_G_GAN2_I.item()
    self.gan2_loss_b = loss_G_GAN2_B.item()

  def update_lr(self):
    self.quantize_s_sch.step()
    self.quantize_b_sch.step()
    self.disA_sch.step()
    self.disB_sch.step()
    self.enc_sch.step()
    self.gen_sch.step()

  def _l2_regularize(self, mu):
    mu_2 = torch.pow(mu, 2)
    encoding_loss = torch.mean(mu_2)
    return encoding_loss

  def resume(self, model_dir, train=True):
    checkpoint = torch.load(model_dir, map_location=lambda storage, loc: storage)

    # weight
    if train:
      self.disA.load_state_dict(checkpoint['disA'])
      self.disB.load_state_dict(checkpoint['disB'])
    self.quantize_s.load_state_dict(checkpoint['quantize_s'])
    self.quantize_b.load_state_dict(checkpoint['quantize_b'])
    self.enc.load_state_dict(checkpoint['enc'])
    self.gen.load_state_dict(checkpoint['gen'])
    # optimizer 
    if train:
      self.quantize_s_opt.load_state_dict(checkpoint['quantize_s_opt'])
      self.quantize_b_opt.load_state_dict(checkpoint['quantize_b_opt'])
      self.disA_opt.load_state_dict(checkpoint['disA_opt'])
      self.disB_opt.load_state_dict(checkpoint['disB_opt'])
      self.enc_opt.load_state_dict(checkpoint['enc_opt'])
      self.gen_opt.load_state_dict(checkpoint['gen_opt'])
    return checkpoint['ep'], checkpoint['total_it']

  def save(self, filename, ep, total_it):
    state = {
             'quantize_s': self.quantize_s.state_dict(),
             'quantize_b': self.quantize_b.state_dict(),
             'disA': self.disA.state_dict(),
             'disB': self.disB.state_dict(),
             'enc': self.enc.state_dict(),
             'gen': self.gen.state_dict(),
             'quantize_s_opt': self.quantize_s_opt.state_dict(),
             'quantize_b_opt': self.quantize_b_opt.state_dict(),
             'disA_opt': self.disA_opt.state_dict(),
             'disB_opt': self.disB_opt.state_dict(),
             'enc_opt': self.enc_opt.state_dict(),
             'gen_opt': self.gen_opt.state_dict(),
             'ep': ep,
             'total_it': total_it
              }
    time.sleep(10)
    torch.save(state, filename)
    return

  def assemble_outputs(self):
    images_a = self.normalize_image(self.real_I_encoded).detach()
    images_b = self.normalize_image(self.real_B_encoded).detach()
    images_a1 = self.normalize_image(self.recon_Ib_encoded).detach()
    images_a3 = self.normalize_image(self.recon_Ii_encoded).detach()
    images_a4 = self.normalize_image(self.recon_I_hat).detach()
    images_b1 = self.normalize_image(self.recon_Bi_encoded).detach()
    images_b3 = self.normalize_image(self.recon_Bb_encoded).detach()
    images_b4 = self.normalize_image(self.recon_B_hat).detach()
    row1 = torch.cat((images_a[0:1, ::], images_a1[0:1, ::], images_a3[0:1, ::], images_a4[0:1, ::]),3)
    row2 = torch.cat((images_b[0:1, ::], images_b1[0:1, ::], images_b3[0:1, ::], images_b4[0:1, ::]),3)
    return torch.cat((row1,row2),2)


  def normalize_image(self, x):
    return x[:,0:3,:,:]
