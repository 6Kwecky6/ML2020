import torch
import torch.nn as nn
import os

torch.cuda.set_device(0)
class LongShortTermMemoryModel(nn.Module):
    def __init__(self):
        super(LongShortTermMemoryModel, self).__init__()
        self.cross_loss = torch.nn.CrossEntropyLoss()
        self.hidden_state = torch.zeros(1, 1,state_size).cuda()
        self.cell_state = torch.zeros(1, 1, state_size).cuda()
        self.lstm = nn.LSTM(encoding_size*batch, state_size)  # 128 is the state size
        self.dense = nn.Linear(state_size, emoji_size)  # 128 is the state size

    def reset(self):  # Reset states prior to new input sequence
        zero_state = torch.zeros(1, 1, state_size)  # Shape: (number of layers, batch size, state size)
        self.hidden_state = zero_state.cuda()
        self.cell_state = zero_state.cuda()

    def logits(self, x):  # x shape: (sequence length, batch size, encoding size)
        x = x.view(-1,1, batch*encoding_size).cuda()
        out, (self.hidden_state, self.cell_state) = self.lstm(x, (self.hidden_state, self.cell_state))
        y = self.dense(out.view(-1, state_size))
        return y

    def forward(self, x):  # x shape: (sequence length, batch size, encoding size)
        x = x.cuda()
        y= self.logits(x)
        return torch.softmax(y,dim=1)

    def loss(self, x, y):  # x shape: (sequence length, batch size, encoding size), y shape: (sequence length, encoding size)
        x = x.cuda()
        y = y.cuda()
        x= self.logits(x)
        return torch.nn.functional.cross_entropy(x, y.argmax(1))

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


index_to_char = ['a', 'h', 't', 'r', 'c', 'f', 'l', 'm', 'p', 's', 'o', 'n', ' ']
index_to_emoji = ['hat ','rat ','cat ','flat','matt','cap ','son ']
emoji_encodings = torch.tensor([[1., 0., 0., 0., 0., 0., 0.],
                        [0., 1., 0., 0., 0., 0., 0.],
                        [0., 0., 1., 0., 0., 0., 0.],
                        [0., 0., 0., 1., 0., 0., 0.],
                        [0., 0., 0., 0., 1., 0., 0.],
                        [0., 0., 0., 0., 0., 1., 0.],
                        [0., 0., 0., 0., 0., 0., 1.]])

encoding_size = len(char_encodings)
state_size = 7
batch = 4
emoji_size = len(emoji_encodings)
#x_train shape: 7,4,13
x_train = torch.tensor([[char_encodings[1], char_encodings[0], char_encodings[2], char_encodings[12]],  # 'hat '
                        [char_encodings[3], char_encodings[0], char_encodings[2], char_encodings[12]],  # 'rat '
                        [char_encodings[4], char_encodings[0], char_encodings[2], char_encodings[12]],  # 'cat '
                        [char_encodings[5], char_encodings[6], char_encodings[0], char_encodings[2]], # 'flat'
                        [char_encodings[7], char_encodings[0], char_encodings[2], char_encodings[2]], # 'matt'
                        [char_encodings[4], char_encodings[0], char_encodings[8], char_encodings[12]], # 'cap '
                        [char_encodings[9], char_encodings[10], char_encodings[11], char_encodings[12]]])  # 'son '
#y_train shape: 7,7
y_train = emoji_encodings
model = LongShortTermMemoryModel()

optimizer = torch.optim.RMSprop(model.parameters(),lr=float('1.0e-5'))
x_train.cuda()
y_train.cuda()
model.cuda()

def decode_text(text):
    res = torch.zeros(4,13)
    text=text.ljust(4)
    for i in range(4):
        res[i] = torch.tensor(char_encodings[index_to_char.index(text[i])])
    return res

if os.path.isfile('emoji_predictor_state.pt'):
    state = torch.load('emoji_predictor_state.pt')
    model.load_state_dict(state['model_state_dict'])
    optimizer.load_state_dict(state['optimizer_state_dict'])
max_range = 10000
for epoch in range(max_range):
    model.reset()
    model.loss(x_train, y_train).backward()
    optimizer.step()
    optimizer.zero_grad()
    if epoch%100==0:
        model.reset()
        print('loss: %s\nprogress: %s%%'%(model.loss(x_train,y_train),(epoch*100)/max_range))
        model.reset()
        res = model.forward(decode_text('rat ').clone().detach())
        #print(res)
        print(index_to_emoji[res.argmax(1).detach()])

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, 'emoji_predictor_state.pt')


inp = input("Legal characters: ('a', 'h', 't', 'r', 'c', 'f', 'l', 'm', 'p', 's', 'o', 'n', ' ')\nPlease enter a 4 character word:\n")
while inp != '':
    model.reset()
    res = model.forward(decode_text(inp).clone().detach())
    #print(res)
    print('The emoji is going to be %s'%index_to_emoji[res.argmax(1).detach()])
    inp = input("Legal characters: ('a', 'h', 't', 'r', 'c', 'f', 'l', 'm', 'p', 's', 'o', 'n', ' ')\nPlease enter a 4 character word:\n")
