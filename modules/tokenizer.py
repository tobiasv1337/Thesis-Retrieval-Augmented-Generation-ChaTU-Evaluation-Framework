import nltk
from nnsplit import NNSplit
nltk.download('punkt')

class Tokenizer:
    def __init__(self, language, model="nltk"):
        self.model = model
        if language == "deutsch":
            if model == "nnsplit":
                self.tokenizer = NNSplit.load("de")
                print(self.tokenize("Hallo! Das hier ist ein Test."))
            else:
                self.tokenizer = nltk.data.load('tokenizers/punkt/german.pickle')
                self.tokenizer._params.abbrev_types.update(['ggf', 'i.d.r', 'gem', '1', '2', 'bspw', 'bsc', 'msc', 'vllt', 'ca'])
        else:
            if model == "nnsplit":
                self.tokenizer = NNSplit.load("en")
            else:
                self.tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
                self.tokenizer._params.abbrev_types.update(['e.g', '1', '2'])

    def tokenize(self, input):
        if self.model == "nnsplit":
            return self.tokenizer.split(input)
        else:
            return self.tokenizer.tokenize(input)


tokenizer = {
    "deutsch": Tokenizer("deutsch"),
    "english": Tokenizer("english")
}