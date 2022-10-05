from typing import Tuple
import numpy as np
import torch
from torch import nn
import re
import random

device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token = 0
EOS_token = 1
teacher_forcing_ratio = 0.5

def load_file(file_path: str) -> Tuple[Tuple[str], Tuple[str]]:
    """loads the test file and extracts all functions/derivatives"""
    data = open(file_path, "r").readlines()
    functions, derivatives = zip(*[line.strip().split("=") for line in data])
    return functions, derivatives


class Lang:
    def __init__(self,name):
        self.word2index  = {}
        self. word2count = {}
        self.index2word  = {0: "SOS",1: "EOS"}
        self.n_words     = 2

    def addSentence(self, sentence):
        for word in re.findall(r"sin|cos|tan|cot|sec|cosec|exp|d\w|\d|\w|\(|\)|\+|\-|\*|\^|\/+", sentence):
            self.addWord(word)

    def sentence_to_words(self,sentence):
        return re.findall(r"sin|cos|tan|cot|sec|cosec|exp|d\w|\d|\w|\(|\)|\+|-|\*|\^|\/+", sentence)

    def addWord(self,word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in lang.sentence_to_words(sentence)]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def trainingexample(functions,true_derivatives):
    rand= random.randint(0,999999)
    x= tensorFromSentence(lang1,functions[rand])
    y= tensorFromSentence(lang2,true_derivatives[rand])
    return (x,y)

class ENCODER(nn.Module):
    def __init__(self, input_size,hidden_size):
        super(ENCODER,self).__init__()
        self.hidden_size = hidden_size
        self.embedding   = nn.Embedding(input_size,hidden_size)
        self.gru         = nn.GRU(hidden_size,hidden_size,2)

    def forward(self,input,hidden):
        embedded      = self.embedding(input).view(1,1,-1)
        output,hidden = self.gru(embedded,hidden)
        return output, hidden
    def initHidden(self):
        return torch.zeros(2,1,self.hidden_size,device=device)

class DECODER(nn.Module):
    def __init__(self,hidden_size,output_size):
        super(DECODER,self).__init__()
        self.hidden_size  = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size,2)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = torch.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden
    def initHidden(self):
        return torch.zeros(2, self.hidden_size, device=device)



def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    loss = 0
    #print(input_length)

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden= decoder(
                decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

def trainIters(encoder, decoder, n_iters, print_every=10000,  learning_rate=3e-3):
    print_loss_total = 0  

    encoder_optimizer = torch.optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer =torch.optim.SGD(decoder.parameters(), lr=learning_rate)
    for epoch in range(10):
        print('epoch',epoch)
        training_pairs = [trainingexample(functions,true_derivatives)
                          for i in range(n_iters)]
        criterion = nn.NLLLoss()
        c=0

        for iter in range(0, n_iters ):
            training_pair = training_pairs[iter]
            input_tensor = training_pair[0]
            target_tensor = training_pair[1]

            loss= train(input_tensor, target_tensor, encoder,
                         decoder, encoder_optimizer, decoder_optimizer, criterion)
            print_loss_total += loss
            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('loss',print_loss_avg)

functions,true_derivatives = load_file('train.txt')

lang1= Lang('function')
lang2= Lang('derivative')
for i in range(len(functions)):
    lang1.addSentence(functions[i])
    lang2.addSentence(true_derivatives[i])


encoder= ENCODER(lang1.n_words,512).to(device)
decoder= DECODER(512,lang2.n_words).to(device)

trainIters(encoder, decoder, 100000, print_every=10000)
torch.save(encoder.state_dict(),'encoder_final1.pt')
torch.save(decoder.state_dict(),'decoder_final1.pt')
