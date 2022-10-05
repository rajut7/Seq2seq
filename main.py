from typing import Tuple
import numpy as np
import torch
from torch import nn
import re


SOS_token = 0
EOS_token = 1
MAX_SEQUENCE_LENGTH = 30
TRAIN_URL = "https://drive.google.com/file/d/1ND_nNV5Lh2_rf3xcHbvwkKLbY2DmOH09/view?usp=sharing"
#device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
device='cpu'

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


encoder = ENCODER(61,512).to(device)
decoder= DECODER(512,40).to(device)
encoder.load_state_dict(torch.load('encoder_final1.pt'),strict=False)
decoder.load_state_dict(torch.load('decoder_final1.pt'),strict=False)
encoder.eval()
decoder.eval()



def load_file(file_path: str) -> Tuple[Tuple[str], Tuple[str]]:
    """loads the test file and extracts all functions/derivatives"""
    data = open(file_path, "r").readlines()
    functions, derivatives = zip(*[line.strip().split("=") for line in data])
    return functions, derivatives


def score(true_derivative: str, predicted_derivative: str) -> int:
    """binary scoring function for model evaluation"""
    #print(true_derivative, predicted_derivative)
    return int(true_derivative == predicted_derivative)


# --------- PLEASE FILL THIS IN --------- #
def predict(functions: str,lang1,lang2):
    with torch.no_grad():
        input_tensor = tensorFromSentence(lang1, functions)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(MAX_SEQUENCE_LENGTH, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)

        decoder_hidden = encoder_hidden

        decoded_words = []

        for di in range(MAX_SEQUENCE_LENGTH ):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                break
            else:
                decoded_words.append(lang2.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()
        output=''
        for i in decoded_words:
            output=output+i
    return output

# ----------------- END ----------------- #



def main(filepath: str = "test.txt"):
    """load, inference, and evaluate"""
    functions, true_derivatives = load_file(filepath)
    lang1= Lang('function')
    lang2= Lang('derivative')
    for i in range(len(functions)):
        lang1.addSentence(functions[i])
        lang2.addSentence(true_derivatives[i])
    predicted_derivatives = [predict(f,lang1,lang2) for f in functions]
    scores = [score(td, pd) for td, pd in zip(true_derivatives, predicted_derivatives)]
    print(np.mean(scores))


if __name__ == "__main__":
    main()
