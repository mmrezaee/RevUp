################################
# Data Utils for generating
# plain old text (The Book Corpus)
# The input is assumed to be a pretokenized text file
# with a single sentence per line
#
# Uses texttorch stuff, so make sure thats installed
################################
import torch
import torch.nn as nn
import numpy as np
import math
import pickle
import json
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torchtext.data as ttdata
import torchtext.datasets as ttdatasets
import jsonlines
from torchtext.vocab import Vocab
from collections import defaultdict, Counter
import unidecode
from tqdm import tqdm
from random import sample

#Reserved Special Tokens
PAD_TOK = "<pad>"
SOS_TOK = "<sos>" #start of sentence
EOS_TOK = "<eos>" #end of sentence
UNK_TOK = "<unk>"
TUP_TOK = "<tup>"
DIST_TOK = "<DIST>" # distractor token for NC task
NOFRAME_TOK = "__NOFRAME__"

#These are the values that should be used during evalution to keep things consistent
#MIN_EVAL_SEQ_LEN = 8
#MAX_EVAL_SEQ_LEN = 50
#MAX_CLAUSE = 15
#MAX_SUMMARY_LEN=20


#A Field for a single sentence from the Book Corpus (or any other corpus with a single item per line)

#PAD has an id of 1
#UNK has id of 0

def create_vocab(filename, max_size=None, min_freq=1, savefile=None, specials = [UNK_TOK, PAD_TOK, SOS_TOK, EOS_TOK]):
    """
    Create a vocabulary object
    Args
        filename (str) : filename to induce vocab from
        max_size (int) : max size of the vocabular (None = Unbounded)
        min_freq (int) : the minimum times a word must appear to be
        placed in the vocab
        savefile (str or None) : file to save vocab to (return it if None)
        specials (list) : list of special tokens
    returns Vocab object
    """
    count = Counter()
    with open(filename, 'r') as fi:
        for line in fi:
            for tok in line.split(" "):
                count.update([tok.rstrip('\n')])

    voc = Vocab(count, max_size=max_size, min_freq=min_freq, specials=specials)
    if savefile is not None:
        with open(savefile, 'wb') as fi:
            pickle.dump(voc, fi)
        return None
    else:
        return voc


def load_vocab(filename,is_Frame=False):
    #load vocab from json file
    with open(filename, 'rb') as fi:
        voc = pickle.load(fi)
        return voc



class ExtendableField(ttdata.Field):
    'A field class that allows the vocab object to be passed in'
    #This is to avoid having to calculate the vocab every time
    #we want to run
    def __init__(self, vocab, *args, **kwargs):
        """
        Args
            Same args as Field except
            vocab (torchtext Vocab) : vocab to init with
                set this to None to init later

            USEFUL ARGS:
            tokenize
            fix_length (int) : max size for any example, rest are padded to this (None is defualt, means no limit)
            include_lengths (bool) : Whether to return lengths with the batch output (for packing)
        """

        super(ExtendableField, self).__init__(*args, pad_token=PAD_TOK, batch_first=True, include_lengths=True,**kwargs)
        if vocab is not None:
            self.vocab = vocab
            self.vocab_created = True
        else:
            self.vocab_created = False

    def init_vocab(self, vocab):
        if not self.vocab_created:
            self.vocab = vocab
            self.vocab_created = True

    def build_vocab(self):
        raise NotImplementedError

    def numericalize(self, arr, device=None, train=True):
        """Turn a batch of examples that use this field into a Variable.

        If the field has include_lengths=True, a tensor of lengths will be
        included in the return value.

        Arguments:
            arr (List[List[str]], or tuple of (List[List[str]], List[int])):
                List of tokenized and padded examples, or tuple of List of
                tokenized and padded examples and List of lengths of each
                example if self.include_lengths is True.
            device (-1 or None): Device to create the Variable's Tensor on.
                Use -1 for CPU and None for the currently active GPU device.
                Default: None.
            train (boolean): Whether the batch is for a training set.
                If False, the Variable will be created with volatile=True.
                Default: True.
        """
        if self.include_lengths and not isinstance(arr, tuple):
            raise ValueError("Field has include_lengths set to True, but "
                             "input data is not a tuple of "
                             "(data batch, batch lengths).")
        if isinstance(arr, tuple):
            arr, lengths = arr
            lengths = torch.LongTensor(lengths)

        if self.use_vocab:
            if self.sequential:
                arr = [[self.vocab.stoi[x] for x in ex] for ex in arr]
            else:
                arr = [self.vocab.stoi[x] for x in arr]

            if self.postprocessing is not None:
                arr = self.postprocessing(arr, self.vocab, train)
        else:
            if self.tensor_type not in self.tensor_types:
                raise ValueError(
                    "Specified Field tensor_type {} can not be used with "
                    "use_vocab=False because we do not know how to numericalize it. "
                    "Please raise an issue at "
                    "https://github.com/pytorch/text/issues".format(self.tensor_type))
            numericalization_func = self.tensor_types[self.tensor_type]
            # It doesn't make sense to explictly coerce to a numeric type if
            # the data is sequential, since it's unclear how to coerce padding tokens
            # to a numeric type.
            if not self.sequential:
                arr = [numericalization_func(x) if isinstance(x, six.string_types)
                       else x for x in arr]
            if self.postprocessing is not None:
                arr = self.postprocessing(arr, None, train)

        arr = self.tensor_type(arr)
        if self.sequential and not self.batch_first:
            arr.t_()
        if device == -1:
            if self.sequential:
                arr = arr.contiguous()
        else:
            arr = arr.cuda(device)
            if self.include_lengths:
                lengths = lengths.cuda(device)
        #print("arr is {}".format(arr))
        if self.include_lengths:
            return arr, lengths
        return arr

class VanillaInfoSentenceDataset(ttdata.Dataset):
    'Data set which has a single sentence per line'
    def __init__(self,args,output_type,addition,path,vocab,vocab2,num_clauses,src_seq_length=None, min_seq_length=0,
                 n_cloze=False, add_eos=True, print_valid=False,unsupervised=True,obsv_prob=0.0,noise_prob=0.0,debug=False,candid=False,mode='train'):
        """
        Args
            path (str) : Filename of text file with dataset
            vocab (Torchtext Vocab object)
            filter_pred (callable) : Only use examples for which filter_pred(example) is TRUE
        """
        #assert vocab2.stoi[NOFRAME_TOK]==0, "__NOFRAME__ index in vocab2 is not zero"
        MAX_SUMMARY_LEN = args.max_summary_len
        MAX_INPUT_LEN = args.max_input_len
        text_field = ExtendableField(vocab)
        frame_field = ExtendableField(vocab2)
        ref_field = ExtendableField(vocab2) #True Frames

        examples = []

        if add_eos:
            target_field = ExtendableField(vocab, eos_token=EOS_TOK)
            #frame_field = ExtendableField(vocab2, eos_token=EOS_TOK)
        else:
            target_field = ExtendableField(vocab) # added this for narrative cloze

        fields = [('text', text_field), ('frame', frame_field),
                  ('ref',ref_field), ('target', target_field)]
        #cut_off=(5*num_clauses-1)

        #addition = 2
        voc_frame_list=vocab2.itos[3:]
        with jsonlines.open(path) as documents:
        #with open(path, 'r') as json_file:
        #    data = json.load(json_file)
        #    documents = data['documents']

            for num_doc,doc in enumerate(documents):
                abstracts = doc['abstract']
                headline = doc['headline']
                if debug:
                    if mode=='valid': 
                        NUM_DOC = 3
                    else: 
                        NUM_DOC=10000
                    if num_doc==NUM_DOC:
                        break

                abstract_clauses = [] #abstract elements
                #abstract_frames = [] #abstract elements
                ref_frames = [] #abstract elements
                summary_clauses = [] #headlines elements

                abstract_counter = 0
                #summary_counter = 0

                for num_text,text in enumerate(abstracts):
                    #if abstract_counter == (num_clauses+addition):
                    if candid: text=text['candid']
                    if abstract_counter == num_clauses:
                        break
                    temp_Semafor = text['event_tuple']['semantic_parses']['Semafor v2.1 (concrete-semafor)']['frame'].split('/')[1].lower()
                    #if temp_Semafor == '__noframe__':
                    if vocab2.stoi[temp_Semafor] == vocab2.stoi['__noframe__']:
                        continue
                    else:
                        abstract_counter += 1
                    abstracts_pred = unidecode.unidecode(str(text['event_tuple']['pred']).lower()).replace(' ','-')
                    abstracts_arg_0 = unidecode.unidecode(str(text['event_tuple']['arg0']).lower()).replace(' ','-')
                    abstracts_arg_1 = unidecode.unidecode(str(text['event_tuple']['arg1']).lower()).replace(' ','-')
                    abstracts_mod = unidecode.unidecode(str(text['event_tuple']['mod']).lower()).replace(' ','-')
                    args_clauses=" ".join([abstracts_pred,abstracts_arg_0,abstracts_arg_1,abstracts_mod,TUP_TOK])

                    abstract_clauses.append(args_clauses)
                    #abstract_frames.append([temp_Semafor])
                    ref_frames.append([temp_Semafor])
                #abstract_sent = str(text['sent']).lower()
                #print('num: {}, abstract_clauses: {}'.format(num_doc,abstract_clauses))
                #if len(abstract_clauses)<(num_clauses+addition):
                #    continue
                #temp_summary = abstract_clauses[num_clauses:]
                #abstract_clauses=abstract_clauses[0:num_clauses]
                #ref_frames=ref_frames[0:num_clauses]
                for num_summary,summary in enumerate(headline):
                    if num_summary == 1:
                        break
                    summary_pred = str(summary['event_tuple']['pred']).lower()
                    summary_arg_0 = str(summary['event_tuple']['arg0']).lower()
                    summary_arg_1 = str(summary['event_tuple']['arg1']).lower()
                    summary_mod = str(summary['event_tuple']['mod']).lower()
                    summ_clauses=" ".join([summary_pred,summary_arg_0,summary_arg_1,summary_mod,TUP_TOK])
                    summary_clauses.append(summ_clauses)
                #summary_clauses = abstract_clauses[num_clauses:]
                #summary_clauses = temp_summary

                #print('summary_clauses: {}'.format(summary_clauses))
                #print('len_summary_clauses: {}'.format(len(summary_clauses)))
                #summary_sent = str(summary['sent']).lower()
                if len(abstract_clauses)<num_clauses:
                    continue
                abstract_clauses = abstract_clauses[:num_clauses]
                ref_frames = ref_frames[:num_clauses]

                convert_abstract = " ".join(abstract_clauses)
                convert_summary = " ".join(summary_clauses)
                convert_summary = " ".join(convert_summary.split()[:MAX_SUMMARY_LEN])
                convert_abstract = " ".join(convert_abstract.split()[:MAX_INPUT_LEN])
                if len(convert_abstract.split())==0 or len(convert_summary.split())==0:
                    continue
                #print('convert_summary: {}'.format(convert_summary))
                #print('*'*20)
                debug_abstract = convert_abstract.split()
                if len(debug_abstract) != 25:
                    print('len(debug_abstract): {}'.format(len(debug_abstract)))
                    print(abstract_clauses)
                    print('\n')
                    print(debug_abstract)
                    print('='*50)
                num_frames = len(ref_frames)
                ref_frames = " ".join([" ".join(item) for item in ref_frames])
                ref_frames_list = ref_frames.split()

                #print('ref_frames_list: {}, len: {}'.format(ref_frames,len(ref_frames_list)))
                probs = obsv_prob*torch.ones(len(ref_frames_list))
                noisy_probs = noise_prob*torch.ones(len(ref_frames_list))
                noisy_candids = sample(voc_frame_list,len(ref_frames_list))
                #print('noisy_candids: ',noisy_candids)
                selector = torch.bernoulli(probs)
                noisy_selector = torch.bernoulli(noisy_probs)
                #print('noisy_selector: ',noisy_selector)

                obs_frames_select = [ref_frames_list[idx] if selector[idx]==1 else NOFRAME_TOK.lower() for idx,_ in enumerate(ref_frames_list)]
                #print('original obs_frames_select: {}'.format(obs_frames_select))
                obs_frames_select = [noisy_candids[idx] if noisy_selector[idx]==1 else ref_frames_list[idx] for idx,_ in enumerate(ref_frames_list)]
                #print('modified obs_frames_select: {}'.format(obs_frames_select))
                #print('*'*50)

                convert_frames = " ".join(obs_frames_select)

                if len(ref_frames.split()) == 0:
                    continue
                ref_frames += (num_clauses-len(ref_frames.split()))*" <pad>"
                convert_frames += (num_clauses-len(convert_frames.split()))*" <pad>"

                #examples.append(ttdata.Example.fromlist([convert_abstract, convert_frames, ref_frames, convert_summary], fields))

                if output_type=='abstract':
                    #print('output_type==abstract')
                    examples.append(ttdata.Example.fromlist([convert_abstract, convert_frames, ref_frames, convert_abstract], fields))
                elif output_type=='summary':
                    #print('output_type==summary')
                    examples.append(ttdata.Example.fromlist([convert_abstract, convert_frames, ref_frames, convert_summary], fields))

        def filter_pred(example):
            return len(example.text) > 0 and len(example.target) > 0
        super(VanillaInfoSentenceDataset, self).__init__(examples, fields, filter_pred=None)

