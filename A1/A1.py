
import torch, nltk, pickle
from torch import nn
from collections import Counter
from transformers import BatchEncoding, PretrainedConfig, PreTrainedModel
from datasets import load_dataset
from torch.utils.data import DataLoader
import numpy as np
import sys, time, os
from torch.utils.data import Subset

# Ensure required NLTK tokenizers are available
nltk.download('punkt', quiet=True)
try:
    nltk.download('punkt_tab', quiet=True)
except Exception:
    pass

###
### Part 1. Tokenization.
###
def lowercase_tokenizer(text):
    return [t.lower() for t in nltk.word_tokenize(text)]

def build_tokenizer(train_file, tokenize_fun=lowercase_tokenizer, max_voc_size=None, model_max_length=None,
                    pad_token='<PAD>', unk_token='<UNK>', bos_token='<BOS>', eos_token='<EOS>'):
    """ Build a tokenizer from the given file.

        Args:
             train_file:        The name of the file containing the training texts.
             tokenize_fun:      The function that maps a text to a list of string tokens.
             max_voc_size:      The maximally allowed size of the vocabulary.
             model_max_length:  Truncate texts longer than this length.
             pad_token:         The dummy string corresponding to padding.
             unk_token:         The dummy string corresponding to out-of-vocabulary tokens.
             bos_token:         The dummy string corresponding to the beginning of the text.
             eos_token:         The dummy string corresponding to the end the text.
    """

    # TODO: build the vocabulary, possibly truncating it to max_voc_size if that is specified.
    # Then return a tokenizer object (implemented below).
    token_counts = Counter()
    with open(train_file, 'r', encoding = 'utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                tokens = tokenize_fun(line)
                token_counts.update(tokens)

    # Building the vocabulary, 
    special_tokens = [pad_token, unk_token, bos_token, eos_token]
    vocab = special_tokens.copy()

  # The total size of the vocabulary (including the 4 symbols) should be at most max_voc_size, which is is a 
  # user-specified hyperparameter. 
  # If the number of unique tokens in the text is greater than max_voc_size, then use the most frequent ones.
    if max_voc_size is not None:
        num_tokens = max(0, max_voc_size - len(special_tokens))
        most_frequent_tokens = token_counts.most_common(num_tokens)
    else:
        most_frequent_tokens = token_counts.most_common()

    for word, count in most_frequent_tokens:
        if word not in special_tokens:
            vocab.append(word)


    str2int = {word: idx for idx, word in enumerate(vocab)}  # word → ID
    int2str = {idx: word for word, idx in str2int.items()}   # ID → word

    return  A1Tokenizer(word2id=str2int, id2word=int2str, pad_token= pad_token, unk_token=unk_token, bos_token=bos_token, eos_token=eos_token, model_max_length=model_max_length, tokenize_fun=tokenize_fun)



class A1Tokenizer:
    """A minimal implementation of a tokenizer similar to tokenizers in the HuggingFace library."""

    def __init__(self, model_max_length, word2id, id2word, pad_token, unk_token, bos_token, eos_token, tokenize_fun):
        # TODO: store all values you need in order to implement __call__ below.
        ## Maps
        self.word2id = word2id
        self.id2word = id2word

        ## spec tokens as strings
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token

        # spec tokens as IDs
        self.pad_token_id = word2id[pad_token]     # required
        self.unk_token_id = word2id[unk_token]
        self.bos_token_id = word2id[bos_token]
        self.eos_token_id = word2id[eos_token]

        # model length anf tokenizer function
        self.model_max_length = model_max_length   # Needed for truncation
        self.tokenize_fun = tokenize_fun

    def __call__(self, texts, truncation=False, padding=False, return_tensors=None):
        """Tokenize the given texts and return a BatchEncoding containing the integer-encoded tokens.
           
           Args:
             texts:           The texts to tokenize.
             truncation:      Whether the texts should be truncated to model_max_length.
             padding:         Whether the tokenized texts should be padded on the right side.
             return_tensors:  If None, then return lists; if 'pt', then return PyTorch tensors.

           Returns:
             A BatchEncoding where the field `input_ids` stores the integer-encoded texts.
        """
        if return_tensors and return_tensors != 'pt':
            raise ValueError('Should be pt')
        

        # if there is only one text in texts
        if isinstance(texts, str):
            texts = [texts]
        
        encoded_texts = []

        for text in texts:
            # splitting and adding the bos an eos token
            tokens = self.tokenize_fun (text)
            tokens = [self.bos_token] + tokens + [self.eos_token]
            # convert to ids, specify the unk_token as a default value with the .get call
            ids = [self.word2id.get(token, self.unk_token_id) for token in tokens]
            encoded_texts.append(ids)
        
        if truncation and self.model_max_length is not None:
            encoded_texts = [ids[:self.model_max_length] for ids in encoded_texts]

        if padding:
            max_len = max(len(ids) for ids in encoded_texts)
            encoded_texts = [ids + [self.pad_token_id]*(max_len - len(ids)) for ids in encoded_texts]

        # attention mask
        attention_mask = []
        for ids in encoded_texts: 
            lil_mask = []
            for token_id in ids:
                if token_id != self.pad_token_id:
                    lil_mask.append(1)
                else:
                    lil_mask.append(0)
            attention_mask.append(lil_mask)


        if return_tensors == 'pt':
            input_ids = torch.tensor(encoded_texts)
            attention_mask = torch.tensor(attention_mask)
        else:
            input_ids = encoded_texts
        
        return BatchEncoding({'input_ids': input_ids, "attention_mask": attention_mask})

    def __len__(self):
        """Return the size of the vocabulary."""
        return len(self.word2id)
    
    def save(self, filename):
        """Save the tokenizer to the given file."""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def from_file(filename):
        """Load a tokenizer from the given file."""
        with open(filename, 'rb') as f:
            return pickle.load(f)
   

###
### Part 3. Defining the model.
###

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(SCRIPT_DIR, "train.txt")
val_path = os.path.join(SCRIPT_DIR, "val.txt")

dataset = load_dataset('text', data_files={'train': train_path, 'val': val_path})
dataset = dataset.filter(lambda x: x['text'].strip() != '')

""" Comment this out to use the full dataset!!!"""
#print(len(dataset["train"]))
#for sec in ['train', 'val']:
#    dataset[sec] = Subset(dataset[sec], range(1000))



class A1RNNModelConfig(PretrainedConfig):
    """Configuration object that stores hyperparameters that define the RNN-based language model."""
    def __init__(self, vocab_size=None, embedding_size=None, hidden_size=None, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size

class A1RNNModel(PreTrainedModel):
    """The neural network model that implements a RNN-based language model."""
    config_class = A1RNNModelConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_size)
        self.rnn = nn.LSTM(config.embedding_size, config.hidden_size, batch_first=True)
        self.unembedding = nn.Linear(config.hidden_size, config.vocab_size)
        self.post_init()
        
    def forward(self, X):
        """The forward pass of the RNN-based language model.
        
           Args:
             X:  The input tensor (2D), consisting of a batch of integer-encoded texts.
           Returns:
             The output tensor (3D), consisting of logits for all token positions for all vocabulary items.
        """
        embedded = self.embedding(X)
        rnn_out, _ = self.rnn(embedded)
        out = self.unembedding(rnn_out)
        return out


###
### Part 4. Training the language model.
###


class A1Trainer:
    """A minimal implementation similar to a Trainer from the HuggingFace library."""

    def __init__(self, model, args, train_dataset, eval_dataset, tokenizer):
        """Set up the trainer.
           
           Args:
             model:          The model to train.
             args:           The training parameters stored in a TrainingArguments object.
             train_dataset:  The dataset containing the training documents.
             eval_dataset:   The dataset containing the validation documents.
             eval_dataset:   The dataset containing the validation documents.
             tokenizer:      The tokenizer.
        """
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer

        assert(args.optim == 'adamw_torch')
        assert(args.eval_strategy == 'epoch')

    def select_device(self):
        """Return the device to use for training, depending on the training arguments and the available backends."""
        if self.args.use_cpu:
            return torch.device('cpu')
        if not self.args.no_cuda and torch.cuda.is_available():
            return torch.device('cuda')
        if torch.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
            
    def train(self):
        """Train the model."""
        args = self.args

        device = self.select_device()
        print('Device:', device)
        self.model.to(device)
        V = self.model.config.vocab_size
        
        # Wrap for multi-GPU if multiple GPUs available
        #if torch.cuda.device_count() > 1:
        #    print(f"Using {torch.cuda.device_count()} GPUs!")
        #     self.model = nn.DataParallel(self.model)
    
        loss_func = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)


        optimizer = torch.optim.AdamW(
                        self.model.parameters(),
                        lr = args.learning_rate,
                        weight_decay=0.01,      # L2 regularization
                    )
                        

        train_loader = DataLoader(self.train_dataset, batch_size =args.per_device_train_batch_size, shuffle=True)
        val_loader = DataLoader(self.eval_dataset, batch_size =args.per_device_eval_batch_size, shuffle=False)

        # for epoch for batch; tokenizer logic forward pass backward pass store gradients start over
        # TODO: Your work here is to implement the training loop.
        #       
        # for each training epoch (use args.num_train_epochs here):
        #   for each batch B in the training set:
        #
        #       PREPROCESSING AND FORWARD PASS:
        #       input_ids = apply your tokenizer to B
	    #       X = all columns in input_ids except the last one
	    #       Y = all columns in input_ids except the first one
	    #       put X and Y onto the GPU (or whatever device you use)
        #       apply the model to X
        #   	compute the loss for the model output and Y
        #
        #       BACKWARD PASS AND MODEL UPDATE:
        #       optimizer.zero_grad()
        #       loss.backward()
        #       optimizer.step()


        for epoch in range(args.num_train_epochs):
            self.model.train()
            train_loss_sum, train_tok = 0.0, 0

            for batch in train_loader:
                texts = batch["text"]
                enc = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
                input_ids = enc["input_ids"].to(device)

                X = input_ids[:, :-1]
                Y = input_ids[:, 1:]

                logits = self.model(X)
                loss = loss_func(logits.reshape(-1, V), Y.reshape(-1))

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    num_tokens = (Y != self.tokenizer.pad_token_id).sum().item()
                    train_loss_sum += loss.item() * num_tokens
                    train_tok += num_tokens

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_loss_sum, val_tok = 0.0, 0
                for batch in val_loader:
                    texts = batch["text"]
                    enc = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
                    ids = enc["input_ids"].to(device)
                    Xv, Yv = ids[:, :-1], ids[:, 1:]
                    logits = self.model(Xv)
                    loss_v = loss_func(logits.reshape(-1, V), Yv.reshape(-1))
                    num_tokens = (Yv != self.tokenizer.pad_token_id).sum().item()
                    val_loss_sum += loss_v.item() * num_tokens
                    val_tok += num_tokens

            train_ce = train_loss_sum / max(1, train_tok)  ## hmm?
            val_ce = val_loss_sum / max(1, val_tok)
            train_ppl = float(np.exp(train_ce))
            val_ppl = float(np.exp(val_ce))

            print(f"Epoch {epoch+1}: train CE={train_ce:.4f} PPL={train_ppl:.1f} | val CE={val_ce:.4f} PPL={val_ppl:.1f}")
            self.model.train()



        print(f'Saving to {args.output_dir}.')
        self.model.save_pretrained(args.output_dir)

    

train_dataset = dataset["train"]
eval_dataset = dataset["val"]

tokenizer = build_tokenizer(
    train_file=train_path,
    tokenize_fun=lowercase_tokenizer,
    max_voc_size=30000,
    model_max_length=256,
    pad_token='<PAD>',
    unk_token='<UNK>',
    bos_token='<BOS>',
    eos_token='<EOS>'
)

config = A1RNNModelConfig(
    vocab_size=len(tokenizer),
    embedding_size=256,
    hidden_size=512
)

model = A1RNNModel(config)


class TrainingArguments:
    def __init__(self):
        self.optim = 'adamw_torch'
        self.eval_strategy = 'epoch'
        self.use_cpu = False
        self.no_cuda = False
        self.learning_rate = 2e-3
        self.num_train_epochs = 3
        self.per_device_train_batch_size = 32
        self.per_device_eval_batch_size = 32
        self.output_dir = 'trainer_output'


args = TrainingArguments()

trainer = A1Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer
)

trainer.train()

# Save tokenizer
tokenizer.save('tokenizer.pkl')


def predict_next(tokenizer: A1Tokenizer, model: PreTrainedModel, text: str, device: torch.device, k: int = 5):
    model.eval()
    with torch.no_grad():
        enc = tokenizer(text, return_tensors='pt', padding=False, truncation=True)
        ids = enc["input_ids"].to(device)
        X = ids[:, :-1]
        logits = model(X)
        last_logits = logits[:, -1, :]
        topk = torch.topk(last_logits, k=k, dim=-1)
        idxs = topk.indices[0].tolist()
        return [tokenizer.id2word.get(i, tokenizer.unk_token) for i in idxs]

device = trainer.select_device()
# Test predictions
test = [
    "She lives in San"
]

top5 = predict_next(tokenizer, model, test, device, k=5)
print(f"{test} → {top5}")
