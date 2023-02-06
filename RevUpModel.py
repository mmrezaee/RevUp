################################################
#
# Semi-Supervised Discrete VAE (RevUp) - Main Module
# Code for the main module of the Dag VAE
#
################################################
import torch
import torch.nn as nn
import numpy as np
import math
from torch.autograd import Variable
from EncDec import Encoder, Decoder, Attention, fix_enc_hidden, gather_last
from  torch import distributions
import DAG
import torch.nn.functional as F
import data_utils
from data_utils import EOS_TOK, SOS_TOK, PAD_TOK, TUP_TOK

class RevUp(nn.Module):
    def __init__(self, emb_size, hsize, vocab,
                 latents, cell_type="GRU", layers=2, bidir=True,
                 pretrained=True, use_cuda=True, dropout=0.10,
                 frame_max=None,template=None,latent_dim=None,
                 latent_emb_dim=None,verb_max_idx=None):
        """
        Args:
            emb_size (int) : size of input word embeddings
            hsize (int or tuple) : size of the hidden state (for one direction of encoder). If this is an integer then it is assumed
            to be the size for the encoder, and decoder is set the same. If a Tuple, then it should contain (encoder size, dec size)
            latents (LatentNode) : The root of a latent node tree (Note: Size of latent embedding dims should be 2*hsize if bidir!)
            layers (int) : layers for encoder and decoder
            vocab (Vocab object)
            bidir (bool) : use bidirectional encoder?
            cell_type (str) : 'LSTM' or 'GRU'
            sos_idx (int) : id of the start of sentence token
            latent_vocab_embedding: [frame_max, vocab_size]  defined for p(w|h,f) (reconstruction)
                                    each clause has a frame (frame<= frame_max) we map each of them
                                    to a vocab token
        """
        super(RevUp, self).__init__()

        self.embd_size=emb_size
        self.vocab = vocab
        self.vocab_size=len(vocab.stoi.keys())
        self.cell_type = cell_type
        self.layers = layers
        self.bidir=bidir
        self.sos_idx = self.vocab.stoi[SOS_TOK]
        self.eos_idx = self.vocab.stoi[EOS_TOK]
        self.pad_idx = self.vocab.stoi[PAD_TOK]
        self.tup_idx = self.vocab.stoi[TUP_TOK]
        self.latent_root = latents
        self.latent_dim = latent_dim  #Num Frames
        print('RevUp latent_dim: ',self.latent_dim)
        self.latent_emb_dim = latent_emb_dim
        self.frame_max=frame_max
        self.template = template
        self.latents=latents
        self.use_cuda = use_cuda
        self.verb_max_idx = verb_max_idx

        if isinstance(hsize, tuple):
            self.enc_hsize, self.dec_hsize = hsize
        elif bidir:
            self.enc_hsize = hsize
            self.dec_hsize = 2*hsize
        else:
            self.enc_hsize = hsize
            self.dec_hsize = hsize
        in_embedding = nn.Embedding(self.vocab_size, self.embd_size, padding_idx=self.pad_idx)
        out_embedding = nn.Embedding(self.vocab_size, self.embd_size, padding_idx=self.pad_idx)
        self.template_to_frame = nn.Linear(self.template, self.frame_max,bias=False)
        self.template_to_vocab = nn.Linear(self.frame_max, self.vocab_size,bias=False)
        self.theta_layer=nn.Linear(self.layers*self.enc_hsize,self.template)

        if pretrained:
            print("Using Pretrained")
            in_embedding.weight.data = vocab.vectors
            out_embedding.weight.data = vocab.vectors


        self.encoder = Encoder(self.embd_size, self.enc_hsize, in_embedding, self.cell_type, self.layers, self.bidir, use_cuda=use_cuda)
        self.decoder = Decoder(self.embd_size, self.dec_hsize,self.vocab_size,out_embedding, self.cell_type, self.layers, attn_dim=(self.latent_dim, self.dec_hsize), use_cuda=use_cuda, dropout=dropout)



        self.logits_out= nn.Linear(self.dec_hsize, self.vocab_size) #Weights to calculate logits, out [batch, vocab_size]
        self.latent_in = nn.Linear(self.latent_dim, self.layers*self.dec_hsize) #Compute the query for the latents from the last encoder output vector
        if use_cuda:
            self.decoder = self.decoder.cuda()
            self.encoder = self.encoder.cuda()
            self.logits_out = self.logits_out.cuda()
            self.latent_in = self.latent_in.cuda()
            self.theta_layer = self.theta_layer.cuda()
            self.template_to_frame = self.template_to_frame.cuda()
            self.template_to_vocab = self.template_to_vocab.cuda()

        else:
            self.decoder = self.decoder
            self.encoder = self.encoder
            self.logits_out = self.logits_out
            self.latent_in = self.latent_in
            self.theta_layer = self.theta_layer

    def set_use_cuda(self, value):
        self.use_cuda = value
        self.encoder.use_cuda = value
        self.decoder.use_cuda = value
        self.decoder.attention.use_cuda = value
        self.latent_root.set_use_cuda(value)


    def forward(self,obsv_prob, input, seq_lens,summary,summary_lens,f_vals=None,beam_size=-1, str_out=False, max_len_decode=50, min_len_decode=0, n_best=1, encode_only=False):
        batch_size = input.size(0)
        if str_out: #use batch size 1 if trying to get actual output
            assert batch_size == 1

        # INIT THE ENCODER
        ehidden = self.encoder.initHidden(batch_size)
        enc_output, ehidden = self.encoder(input, ehidden, seq_lens)
        enc_theta = self.theta_layer(enc_output).mean(1) #[batch_size,template]
        p_theta_sampled = F.softmax(enc_theta,-1).cuda()
        template_input = F.tanh(self.template_to_frame(p_theta_sampled))
        self.template_decode_input = self.template_to_vocab(template_input)

        if self.use_cuda:
            enc_output_avg = torch.sum(enc_output, dim=1) / Variable(seq_lens.view(-1, 1).type(torch.FloatTensor).cuda())
        else:
            enc_output_avg = torch.sum(enc_output, dim=1) / Variable(seq_lens.view(-1, 1).type(torch.FloatTensor))
        initial_query = enc_output_avg
        latent_values, diffs,latent_embs,prop_entropy,revised_entropy,frames_to_frames, frame_classifier, frame_gumb_samples ,scores, z_KL = self.latent_root.forward(obsv_prob,enc_output, seq_lens, initial_query,f_vals,template_input=template_input) #[batch, num_clauses, num_frames]
        self.scores=scores
        self.latent_gumbels = latent_values
        self.frames_to_frames = frames_to_frames
        self.frame_classifier = frame_classifier
        self.frame_gumb_samples = frame_gumb_samples
        self.z_KL = z_KL


        self.prop_entropy=prop_entropy
        self.revised_entropy=revised_entropy

        top_level = latent_embs[:, 0, :]
        dhidden = torch.nn.functional.tanh(self.latent_in(top_level).view(self.layers, batch_size, self.dec_hsize))

        if encode_only:
            if self.use_cuda:
                self.decoder.init_feed_(Variable(torch.zeros(batch_size, self.decoder.attn_dim)).cuda())
            else:
                self.decoder.init_feed_(Variable(torch.zeros(batch_size, self.decoder.attn_dim)))
            return dhidden, latent_embs

        if str_out:
            if beam_size <=0:
                # GREEDY Decoding
                if self.use_cuda:
                    self.decoder.init_feed_(Variable(torch.zeros(batch_size, self.decoder.attn_dim).cuda())) #initialize the input feed 0
                else:
                    self.decoder.init_feed_(Variable(torch.zeros(batch_size, self.decoder.attn_dim)))

                return self.greedy_decode(input, dhidden, latent_embs, max_len_decode)
            else:
                # BEAM Decoding
                return self.beam_decode(input, dhidden, latent_embs, beam_size, max_len_decode, min_len_decode=min_len_decode)


        # This is for TRAINING, use teacher forcing
        if self.use_cuda:
            self.decoder.init_feed_(Variable(torch.zeros(batch_size, self.decoder.attn_dim).cuda())) #initialize the input feed 0
        else:
            self.decoder.init_feed_(Variable(torch.zeros(batch_size, self.decoder.attn_dim)))

        return self.train(summary, batch_size, dhidden, latent_embs, diffs)


    def train(self, summary, batch_size, dhidden, latent_embs, diffs, return_hid=False, use_eos=False):

        dec_outputs = []
        logits = []
        summary_size = summary.size(1) #Dont need to process last since no eos

        for i in range(summary_size):
            #Choose input for this step
            if i == 0:
                tens = torch.LongTensor(summary.shape[0]).zero_() + self.sos_idx
                if self.use_cuda:
                    dec_input = Variable(tens.cuda()) #Decoder input init with sos
                else:
                    dec_input = Variable(tens)
            else:
                dec_input = summary[:, i-1]
            dec_output, dhidden ,logit , frame_to_vocab = self.decoder(dec_input, dhidden,latent_embs,self.template_decode_input)


            dec_outputs += [dec_output]
            logits += [logit]

        dec_outputs = torch.stack(dec_outputs, dim=0) 
        logits = torch.stack(logits, dim=0) 
        self.logits=logits
        self.frame_to_vocab=frame_to_vocab
        if return_hid:
            return latent_embs, self.latent_root, dhidden, dec_outputs 
        else:
            self.decoder.reset_feed_() 
            return latent_embs, self.latent_root, diffs, dec_outputs 


