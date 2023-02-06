#########################################
#   module for training the RevUp model
#
#
########################################
import torch
import torch.nn as nn
from torchtext.data import Iterator as BatchIter
from  torch import distributions
from show_inf import *
import argparse
import numpy as np
import random
import math
from torch.autograd import Variable
from sklearn import metrics
from EncDec import Encoder, Decoder, Attention, fix_enc_hidden, kl_divergence
import torch.nn.functional as F
import data_utils as du
from RevUpModel import RevUp
from DAG import example_tree
from masked_cross_entropy import masked_cross_entropy
from data_utils import EOS_TOK, SOS_TOK, PAD_TOK
import time
from torchtext.vocab import GloVe
from report_md import *
import pickle
import gc
import glob
import sys
import os
import wandb
from datetime import datetime
from pathlib import Path
from sklearn.metrics import f1_score
now = datetime.now()
TODAY_DATE=now.strftime("%m-%d-%Y")

def tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % n_params)


def monolithic_compute_loss(iteration, model, target,
                            target_lens, latent_values, latent_root,
                            diff, dec_outputs, use_cuda, args,
                            train=True,topics_dict=None,real_sentence=None,
                            next_frames_dict=None,word_to_frame=None,
                            template_dict=None,show=False):
    """
    use this function for validation loss. NO backprop in this function.
    """
    logits = model.logits.transpose(0,1).contiguous() # convert to [batch, seq, vocab]
    prop_entropy= model.prop_entropy
    revised_entropy= model.revised_entropy

    frame_classifier = model.frame_classifier
    frame_classifier_total = -frame_classifier.sum((1,2)).mean()

    prop_entropy_total= prop_entropy.sum(-1).mean()
    revised_entropy_total= revised_entropy.sum(-1).mean()

    z_KL = model.z_KL
    z_KL_total = z_KL.sum(-1).mean()

    if use_cuda:
        ce_loss = masked_cross_entropy(logits, Variable(target.cuda()), Variable(target_lens.cuda()))
    else:
        ce_loss = masked_cross_entropy(logits, Variable(target), Variable(target_lens))

    #loss = ce_loss + 0.1*revised_entropy_total+ 0.1*prop_entropy_total + frame_classifier_total + 0.1*z_KL_total
    #loss = ce_loss + 0.4*revised_entropy_total+ 0.4*prop_entropy_total + 0.4*frame_classifier_total + 0.00001*z_KL_total
    #loss = ce_loss + 0.4*revised_entropy_total+ 0.4*prop_entropy_total + 0.8*frame_classifier_total + z_KL_total
    loss = ce_loss + revised_entropy_total+ prop_entropy_total + frame_classifier_total + z_KL_total
    #loss = ce_loss + 0.0*q_log_q_total + frame_classifier_total + 0.1*z_KL_total
    if train==True and show==True:
        print_iter_stats(iteration, loss, ce_loss, prop_entropy_total,topics_dict,real_sentence,next_frames_dict,frame_classifier_total,word_to_frame,template_dict,args,show=True)
    return loss,frame_classifier_total, ce_loss ,logits, z_KL_total, prop_entropy_total# tensor




def print_iter_stats(iteration, loss, ce_loss, 
                     prop_entropy_total,topics_dict,
                     real_sentence,next_frames_dict,
                     frame_classifier_total,word_to_frame,
                     template_dict,args,show=False):
    if iteration%10==0:
        print("Iteration: ", iteration)
        print("Total: ", loss.cpu().data)
        print("CE: ", ce_loss.cpu().data)
        print("prop_entropy_total",prop_entropy_total.cpu().data)
        print("frame_classifier_total: ",frame_classifier_total.cpu().data)
        print('-'*50)
        if True:
            print("sentence: "," ".join(real_sentence))
            topics_to_md('chain: ',topics_dict)
            templates=np.arange(args.template).reshape((-1,5))
            topics_to_md('words: ',word_to_frame)
            print('-'*50)




def check_save_model_path(save_model):
    save_model_path = os.path.abspath(save_model)
    model_dirname = os.path.dirname(save_model_path)
    if not os.path.exists(model_dirname):
        os.makedirs(model_dirname)



def classic_train(args,args_dict,args_info):
    """
    Train the model in the ol' fashioned way, just like grandma used to
    Args
        args (argparse.ArgumentParser)
    """
    if args.cuda and torch.cuda.is_available():
        print("Using cuda")
        use_cuda = True
    elif args.cuda and not torch.cuda.is_available():
        print("You do not have CUDA, turning cuda off")
        use_cuda = False
    else:
        use_cuda=False

    #Load the data
    print("\nLoading Vocab")
    print('args.vocab: ',args.vocab)
    #vocab , verb_max_idx = du.load_vocab(args.vocab)
    vocab = du.load_vocab(args.vocab)
    vocab_input = du.load_vocab(args.vocab_input)
    args.vocab_input_size = len(vocab_input)
    verb_max_idx = None
    print("Input Vocab Loaded, Size {}".format(len(vocab_input.stoi.keys())))
    print(vocab_input.itos[:40])
    print('\n')
    print("Vocab Loaded, Size {}".format(len(vocab.stoi.keys())))
    print(vocab.itos[:40])
    args_dict["vocab"]=len(vocab.stoi.keys())
    vocab2 = du.load_vocab(args.frame_vocab_address,is_Frame=True)
    print(vocab2.itos[:40])
    print("Frames-Vocab Loaded, Size {}".format(len(vocab2.stoi.keys())))
    vocab_summary = du.load_vocab(args.vocab_summary)
    args.vocab_summary_size = len(vocab_summary)
    print("Summary Vocab Loaded, Size {}".format(len(vocab_summary.stoi.keys())))
    print(vocab_summary.itos[:40])
    print('\n')
    total_frames=len(vocab2.stoi.keys())
    args.total_frames=total_frames
    args.num_latent_values=args.total_frames
    print('total frames: ',args.total_frames)

    # import wandb
    print('obsv_prob: {}'.format(args.obsv_prob))
    experiment_name = '0.3-classify_{}_noise_{}_use_{}_in_{}_out_{}_{}_eps_{}_num_{}_seed_{}_dropout_{}_date_{}'.format('RevUp',
                                                                                                                        args.noise_prob,
                                                                                                                        args.just_use, 
                                                                                                                        args.input_type, 
                                                                                                                        args.output_type, 
                                                                                                                        args.dataset,
                                                                                                                        args.obsv_prob,
                                                                                                                        args.exp_num,
                                                                                                                        args.seed,
                                                                                                                        args.dropout,
                                                                                                                        TODAY_DATE)
    if args.debug: experiment_name='DEBUG_'+experiment_name
    args.experiment_name = experiment_name
    wandb.init(project=args.wandb_project,name=args.experiment_name,tags=['RevUp'])
    if args.use_pretrained:
        pretrained = GloVe(name='6B', dim=args.emb_size, unk_init=torch.Tensor.normal_)
        vocab.load_vectors(pretrained)
        print("Vectors Loaded")

    print("Loading Dataset")
    #dataset = du.SentenceDataset(path=args.train_data,path2=args.train_frames,
    #vocab=vocab,vocab2=vocab2,num_clauses=args.num_clauses, add_eos=False,is_ref=True,obsv_prob=args.obsv_prob)
    dataset = du.VanillaInfoSentenceDataset(args,output_type=args.output_type,addition=args.addition,path=args.train_data,vocab=vocab,vocab2=vocab2,num_clauses=args.num_clauses,
                                            add_eos=False,unsupervised=None,obsv_prob=args.obsv_prob,noise_prob=args.noise_prob,debug=args.debug,candid=args.candid)


    print("Finished Loading Dataset {} examples".format(len(dataset)))
    batches = BatchIter(dataset, args.batch_size, sort_key=lambda x:len(x.text), train=True, sort_within_batch=True, device=-1)
    data_len = len(dataset)

    if args.load_model:
        print("Loading the Model")
        model = torch.load(args.load_model)
    else:
        print("Creating the Model")
        bidir_mod = 2 if args.bidir else 1
        latents = example_tree(args.num_latent_values, (bidir_mod*args.enc_hid_size, args.latent_dim),
                               frame_max=args.total_frames,padding_idx=vocab2.stoi['<pad>'],use_cuda=use_cuda, nohier_mode=args.nohier) #assume bidirectional

        hidsize = (args.enc_hid_size, args.dec_hid_size)
        model = RevUp(args.emb_size, hidsize, vocab, latents, layers=args.nlayers, use_cuda=use_cuda,
                      pretrained=args.use_pretrained, dropout=args.dropout,frame_max=args.total_frames,
                      template=args.template,latent_dim=args.latent_dim,verb_max_idx=verb_max_idx)



    #create the optimizer
    if args.load_opt:
        print("Loading the optimizer state")
        optimizer = torch.load(args.load_opt)
    else:
        print("Creating the optimizer anew")
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    start_time = time.time() #start of epoch 1
    curr_epoch = 1
    valid_loss = [0.0]
    min_ppl=1e10
    print("Loading Validation Dataset.")
    #val_dataset = du.SentenceDataset(path=args.valid_data,path2=args.valid_frames,
    #vocab=vocab,vocab2=vocab2,num_clauses=args.num_clauses, add_eos=False,is_ref=True,obsv_prob=0.0,print_valid=True)

    val_dataset = du.VanillaInfoSentenceDataset(args,output_type=args.output_type,addition=args.addition,path=args.valid_data,vocab=vocab,vocab2=vocab2,num_clauses=args.num_clauses,
                                                add_eos=False, unsupervised=None,obsv_prob=0.0,noise_prob=0.0,debug=args.debug,candid=args.candid)
    print("Finished Loading Validation Dataset {} examples.".format(len(val_dataset)))
    val_batches = BatchIter(val_dataset, args.batch_size, sort_key=lambda x:len(x.text), train=False, sort_within_batch=True, device=-1)
    args.train_size=len(dataset)
    args.valid_size=len(val_dataset)
    #samples_dataset_saved_address='{}/{}.txt'.format('./samplesDataset',args.experiment_name)
    #for idx,item in enumerate(val_batches):
    for idx,item in enumerate(batches):
        #sample_file_object = open(samples_dataset_saved_address, 'a')
        if idx==1:
            break
        for mini_idx in range(len(item.target[0])):
            text_rev=[vocab.itos[int(v.numpy())] for v in item.text[0][mini_idx]]
            target_rev=[vocab.itos[int(v.numpy())] for v in item.target[0][mini_idx]]
            frame_rev=[vocab2.itos[int(v.numpy())] for v in item.frame[0][mini_idx]]
            ref_frame=[vocab2.itos[int(v.numpy())] for v in item.ref[0][mini_idx]]
            #taret_rev=[vocab.itos[int(v.numpy())] for v in item.target[0][-1]]

            print('text_rev:'," ".join(text_rev),len(text_rev),"lengths: ",item.text[1][mini_idx])
            print('frame_rev:',frame_rev,len(frame_rev),"lengths: ",item.frame[1][mini_idx])
            print('ref_frame:',ref_frame,len(ref_frame),"lengths: ",item.ref[1][mini_idx])
            print('target_rev:'," ".join(target_rev),len(target_rev),"lengths: ",item.target[1][mini_idx])
            print('-'*50)
    print('Model_named_params:{}'.format(model.named_parameters()))

    min_ppl=1e10
    max_F1=-1
    max_Acc=-1
    max_precision=-1
    test_best_results={'min_ppl':min_ppl,
                       'max_F1':max_F1,
                       'max_Acc':max_Acc,
                       'max_precision':max_precision}

    wandb.config.update(args) # adds all of the arguments as config variables
    for iteration, bl in enumerate(batches): #this will continue on forever (shuffling every epoch) till epochs finished
        batch, batch_lens = bl.text
        f_vals,f_vals_lens = bl.frame
        approx_prob = Variable(f_vals>2).view(-1,1).cpu().data
        approx_prob = (approx_prob.sum()*1.0)/approx_prob.size()[0]
        target, target_lens = bl.target
        f_ref, _ = bl.ref
        if use_cuda:
            batch = Variable(batch.cuda())
            target = Variable(target.cuda())
            f_vals= Variable(f_vals.cuda())
        else:
            batch = Variable(batch)
            target = Variable(target)
            f_vals= Variable(f_vals)

        model.zero_grad()
        obsv_prob=args.obsv_prob
        #print('approx_prob: {}, obsv_prob: {}'.format(approx_prob,obsv_prob))
        latent_values, latent_root, diff, dec_outputs = model(approx_prob,batch, batch_lens,target, target_lens,f_vals=f_vals)

        topics_dict,real_sentence,next_frames_dict,word_to_frame,template_dict=show_inference(model,batch,vocab,vocab2,f_vals,f_ref,args) 
        loss,frame_classifier_total, _, _,z_KL_total,prop_entropy_total = monolithic_compute_loss(iteration, model, target, target_lens,
                                                                                                 latent_values, latent_root, 
                                                                                                 diff, dec_outputs, use_cuda,
                                                                                                 args=args,topics_dict=topics_dict,
                                                                                                 real_sentence=real_sentence,
                                                                                                 next_frames_dict=next_frames_dict, 
                                                                                                 word_to_frame=word_to_frame,
                                                                                                 template_dict=template_dict,
                                                                                                 train=True,show=True)

        args_dict_wandb={'train_loss':loss, 'z_KL':z_KL_total,'prop_entropy_total':prop_entropy_total,'train_classifier':frame_classifier_total}
        wandb.log(args_dict_wandb)
        # backward propagation
        loss.backward()
        # Gradient clipping torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        # Optimize
        optimizer.step()

        # End of an epoch - run validation
        if iteration%50==0:
            print("\nFinished Training Epoch/iteration {}/{}".format(curr_epoch, iteration))
            test_best_results=test_scores(args,val_batches,model,use_cuda,iteration,vocab,vocab_input,vocab_summary,vocab2,test_best_results)

def test_scores(args,val_batches,model,
                use_cuda,iteration,vocab,
                vocab_input,vocab_summary,
                vocab2,test_best_results):
    obsv_prob=0.0
    frame_pred,frame_gold= [],[]
    # do validation
    total_logprobs,total_lengths,total_loss,total_frame_classifier = 0.0,0.0,0.0,0.0
    sum_pred = []
    target_org = []
    with torch.no_grad():
        for v_iteration, bl in enumerate(val_batches):
            batch, batch_lens = bl.text
            f_vals,f_vals_lens = bl.frame
            target, target_lens = bl.target
            f_ref, _ = bl.ref
            batch_lens = batch_lens.cpu()
            if use_cuda:
                batch = Variable(batch.cuda())
                target = Variable(target.cuda())
                f_vals = Variable(f_vals.cuda())
            else:
                batch = Variable(batch) 
                target = Variable(target)
                f_vals = Variable(f_vals)
            latent_values, latent_root, diff, dec_outputs = model(obsv_prob,batch, batch_lens,target, target_lens,f_vals=f_vals)
            topics_dict,real_sentence,next_frames_dict,word_to_frame,template_dict=show_inference(model,batch,vocab,vocab2,f_vals,f_ref,args)
            loss,frame_classifier, ce_loss,logits,z_KL_total,_ = monolithic_compute_loss(iteration, model, target, target_lens,
                                                                                         latent_values, latent_root, 
                                                                                         diff, dec_outputs, use_cuda,
                                                                                         args=args,topics_dict=topics_dict,
                                                                                         real_sentence=real_sentence,
                                                                                         next_frames_dict=next_frames_dict, 
                                                                                         word_to_frame=word_to_frame,
                                                                                         template_dict=template_dict,
                                                                                         train=False,show=False)

            total_loss += loss.data.clone().cpu().data
            total_frame_classifier += frame_classifier.clone().cpu().data
            total_logprobs+=ce_loss.data.clone().cpu().numpy()*target_lens.sum().cpu().data.numpy()
            total_lengths+=target_lens.sum().cpu().data.numpy()
            sum_max = torch.argmax(logits,-1).cpu()
            summary_lens=target_lens.cpu()
            val_target = target
            for indx,val_item in enumerate(sum_max):
                sum_pred.append(" ".join([vocab_summary.itos[int(v.numpy())] for v_idx,v in enumerate(val_item) if v_idx<summary_lens[indx]]))
                target_org.append(" ".join([vocab_summary.itos[int(v.numpy())] for v_idx,v in enumerate(val_target[indx].cpu()) if summary_lens[indx]]))
            # print("valid_lengths: ",valid_lengths[0])
            batch_frame_pred = model.frame_gumb_samples.data.cpu()
            frame_gold.append(f_ref)
            frame_pred.append(torch.argmax(batch_frame_pred,-1))
    nll=total_logprobs/total_lengths
    ppl=np.exp(nll)
    avg_loss = total_loss/(v_iteration+1)
    avg_frame_classifier_loss = total_frame_classifier/(v_iteration+1)
    all_refs = torch.cat(frame_gold,0)
    all_preds = torch.cat(frame_pred,0)
    all_refs=np.array(all_refs.reshape((-1,))).astype(int)
    all_preds=np.array(all_preds.reshape((-1))).astype(int)
    valid_f1 = f1_score(all_refs, all_preds,average='macro')
    valid_acc = metrics.accuracy_score(all_refs, all_preds)
    valid_precision_macro = metrics.precision_score(all_refs, all_preds,average='macro')

    print("**Validation loss {:.2f}.**\n".format(avg_loss))
    print("**Validation classification loss {:.2f}.**\n".format(avg_frame_classifier_loss))
    print("**Validation NLL {:.2f}.**\n".format(nll))
    print("**Validation PPL {:.2f}.**\n".format(ppl))
    print("**Validation F1 {:.2f}.**\n".format(valid_f1))
    print("**Validation F1 {:.2f}.**\n".format(valid_f1))
    print("**Validation ACC {:.2f}.**\n".format(valid_acc))
    print("**Validation Precision {:.2f}.**\n".format(valid_precision_macro))
    args_dict_wandb = {"val_nll":nll,
                       "val_ppl":ppl,
                       "valid_loss":avg_loss,
                       "valid_F1-score":valid_f1,
                       "valid_Acc":valid_acc,
                       "valid_Precision":valid_precision_macro,
                       "valid_class_loss":avg_frame_classifier_loss
                       }
    wandb.log(args_dict_wandb) # wandb experiments log
    if valid_f1>test_best_results['max_F1']:
        max_F1 = valid_f1
        max_Acc = valid_acc
        max_precision = valid_precision_macro
        test_best_results['max_F1']= max_F1
        test_best_results['max_Acc']= max_Acc
        test_best_results['max_precision']= max_precision
        wandb.run.summary.update({"valid_max_F1":max_F1,"valid_max_Acc":max_Acc,"vaid_max_precision":max_precision}) #wandb runs summary
    if ppl<test_best_results['min_ppl']:
        min_ppl=ppl
        test_best_results['min_ppl']= min_ppl
        args_dict["min_ppl"]=min_ppl
        dir_path = os.path.dirname(os.path.realpath(__file__))
        args_to_md(model="RevUp",args_dict=args_dict)
        wandb.run.summary.update({"val_min_ppl":min_ppl}) #wandb runs summary
        if args.save_model:
            model_path="{}/{}.pt".format(args.model_save_address,args.experiment_name)
            torch.save(model,model_path)
    print('\t==> min_ppl {:4.4f} '.format(test_best_results['min_ppl']))
    return test_best_results



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='RevUp')
    parser.add_argument('--train_data', type=str)
    parser.add_argument('--valid_data', type=str)
    parser.add_argument("--wandb_project", default='approxProb', type=str, help="wandb project")
    parser.add_argument("--sh_file", default=None, type=str, help="The shell script running this python file.")
    parser.add_argument('--vocab', type=str, help='the vocabulary pickle file')
    parser.add_argument('--frame_vocab_address', type=str, help='frames vocabulary pickle file')
    parser.add_argument('--emb_size', type=int, default=300, help='size of word embeddings')
    parser.add_argument('--enc_hid_size', type=int, default=512, help='size of encoder hidden')
    parser.add_argument('--dec_hid_size', type=int, default=512, help='size of encoder hidden')
    parser.add_argument('--nlayers', type=int, default=2, help='number of layers')
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--teacher_forcing_ratio', type=float, default=0.5, help='Teacher Forcing Ratio')

    parser.add_argument('--latent_type', type=str, default='mean', help='mean,root,max')
    parser.add_argument('--feed_type', type=str, default='enc', help='enc,zero')


    parser.add_argument('--log_every', type=int, default=200)
    parser.add_argument('--save_after', type=int, default=500)
    parser.add_argument('--addition', type=int, default=5)
    parser.add_argument('--validate_after', type=int, default=2500)
    parser.add_argument('--optimizer', type=str, default='adam', help='adam, adagrad, sgd')
    parser.add_argument('--wandb_first', type=str, default='conc', help='wandb_name')
    parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=40, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=200, metavar='N', help='batch size')
    parser.add_argument('--seed', type=int, default=11, help='random seed')
    parser.add_argument('--cuda', action='store_true', help='Using Cuda')
    parser.add_argument('--bidir', type=bool, default=True, help='Use bidirectional encoder')
    # parser.add_argument('-src_seq_length', type=int, default=50, help="Maximum source sequence length")
    parser.add_argument('-max_decode_len', type=int, default=50, help='Maximum prediction length.')
    parser.add_argument('-save_model', default='model', help="""Model filename""")
    parser.add_argument('-num_latent_values', type=int, default=400, help='How many values for each categorical value')
    parser.add_argument('-latent_dim', type=int, default=603, help='The dimension of the latent embeddings')
    parser.add_argument('-use_pretrained', type=bool, default=True, help='Use pretrained glove vectors')
    parser.add_argument('-commit_c', type=float, default=0.25, help='loss hyperparameters')
    parser.add_argument('-commit2_c', type=float, default=0.15, help='loss hyperparameters')
    parser.add_argument('-dropout', type=float, default=0.0, help='loss hyperparameters')
    parser.add_argument('--load_model', type=str)
    parser.add_argument('--num_clauses', type=int,default=5)
    parser.add_argument('--load_opt', type=str)
    parser.add_argument('--just_use', type=str,default='frame', help='just use latents or frames')
    parser.add_argument('--nohier', action='store_true', help='use the nohier model instead')
    parser.add_argument('--frame_max', type=int, default=500)

    parser.add_argument('--obsv_prob', type=float, default=1.0,help='the percentage of observing frames')
    parser.add_argument('--noise_prob', type=float, default=0.0,help='the percentage of noisy frames')

    parser.add_argument('--template', type=int, default=20)
    parser.add_argument('--unsupervised', action='store_true', help='No frames observed.')
    parser.add_argument('--num_Frames', type=int, default=53)
    parser.add_argument('--exp_num', type=int, default=1)
    parser.add_argument('--debug' , type=bool, default=False, help='debug mode')
    parser.add_argument('--dataset', type=str, default='small,big,sent_300', help='wandb_name')
    parser.add_argument('--max_seq_len', type=int, default=150, help='max_seq_len')

    parser.add_argument('--output_type', type=str, default='abstract', help='summary or abstract')
    parser.add_argument('--input_type', type=str, default='event', help='event or summary input')
    parser.add_argument('--max_summary_len', type=int, default=20)
    parser.add_argument('--max_input_len', type=int, default=100)
    parser.add_argument('--save_model', action='store_true', help='saving the model')

    args = parser.parse_args()
    #path = os.path.dirname(os.path.realpath(__file__))
    args.model='RevUp'
    args.command = ' '.join(sys.argv)

    #args.num_Frames=args.frame_max
    args.frame_max=args.num_Frames
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path=str(Path(dir_path).parent)
    args.input_type='event'
    args.candid = True


    if args.save_model:
        model_save_address= './saved_models/{}'.format(TODAY_DATE)
        Path(model_save_address).mkdir(parents=True, exist_ok=True)
        args.model_save_address = model_save_address

    args.vocab_input=args.vocab
    args.vocab_summary=args.vocab_input
    args.latent_dim=args.frame_max
    args.num_latent_values=args.frame_max
    args_info={}
    for arg in vars(args):
        args_info[arg] = getattr(args, arg)
    print('parser_info:')
    for item in args_info:
        print(item,": ",args_info[item])
    print('-'*50)



    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(args.seed)

    keys=["model","emb_size","nlayers",
         "lr","batch_size","num_clauses","num_latent_values",
         "latent_dim","dropout","bidir","use_pretrained","obsv_prob","template","frame_max","exp_num","seed"]
    args_dict={key:str(value) for key,value in vars(args).items() if key in keys}

    classic_train(args,args_dict,args_info)



