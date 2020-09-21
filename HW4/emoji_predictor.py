import torch
import torch.nn as nn

class LongShortTermMemoryModel(nn.Module):
    def __init__(self, encoding_size):
        super(LongShortTermMemoryModel, self).__init__()
        self.hidden_state = torch.zeros(1,4,128)
        self.cell_state = torch.zeros(1, 4, 128)
        self.lstm = nn.LSTM(encoding_size, 128)  # 128 is the state size
        self.dense = nn.Linear(128*4, encoding_size)  # 128 is the state size

    def reset(self):  # Reset states prior to new input sequence
        zero_state = torch.zeros(1, 4, 128)  # Shape: (number of layers, batch size, state size)
        self.hidden_state = zero_state
        self.cell_state = zero_state

    def logits(self, x):  # x shape: (sequence length, batch size, encoding size)
        out, (self.hidden_state, self.cell_state) = self.lstm(x, (self.hidden_state, self.cell_state))
        return self.dense(out.reshape(-1,128*4))

    def forward(self, x):  # x shape: (sequence length, batch size, encoding size)
        return torch.softmax(self.logits(x), dim=0)

    def loss(self, x, y):  # x shape: (sequence length, batch size, encoding size), y shape: (sequence length, encoding size)
        return nn.functional.cross_entropy(self.logits(x), y.argmax(1))

model = LongShortTermMemoryModel(13)
state = torch.load('emoji_predictor_state.pt')
model.load_state_dict(state['model_state_dict'])
index_to_char = ['a', 'h', 't', 'r', 'c', 'f', 'l', 'm', 'p', 's', 'o', 'n', ' ']
char_encodings = [
    [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 'a' 0
    [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 'h' 1
    [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 't' 2
    [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 'r' 3
    [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],  # 'c' 4
    [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],  # 'f' 5
    [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],  # 'l' 6
    [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],  # 'm' 7
    [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],  # 'p' 8
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],  # 's' 9
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],  # 'o' 10
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],  # 'n' 11
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]   # ' ' 12
]

def decode_text(text):
    print(text)
    res = torch.zeros(4)
    for i in range(4):
        if text[i]:
            res[i] = char_encodings[index_to_char.index(text[i])]
        else:
            res[i] = char_encodings[12]

    return res

while True:
    inp = decode_text(input('PLease give a 4 char string\n'))
    print(inp)
