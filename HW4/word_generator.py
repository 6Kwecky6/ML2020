import torch
import torch.nn as nn
import os

torch.cuda.set_device(0)
class LongShortTermMemoryModel(nn.Module):
    def __init__(self, encoding_size):
        super(LongShortTermMemoryModel, self).__init__()
        self.hidden_state = torch.zeros(1,1,128).cuda()
        self.cell_state = torch.zeros(1, 1, 128).cuda()
        self.lstm = nn.LSTM(encoding_size, 128)  # 128 is the state size
        self.dense = nn.Linear(128, encoding_size)  # 128 is the state size

    def reset(self):  # Reset states prior to new input sequence
        zero_state = torch.zeros(1, 1, 128)  # Shape: (number of layers, batch size, state size)
        self.hidden_state = zero_state.cuda()
        self.cell_state = zero_state.cuda()

    def logits(self, x):  # x shape: (sequence length, batch size, encoding size)
        x = x.cuda()
        out, (self.hidden_state, self.cell_state) = self.lstm(x, (self.hidden_state, self.cell_state))
        return self.dense(out.reshape(-1, 128))

    def forward(self, x):  # x shape: (sequence length, batch size, encoding size)
        x = x.cuda()
        return torch.softmax(self.logits(x), dim=1)

    def loss(self, x, y):  # x shape: (sequence length, batch size, encoding size), y shape: (sequence length, encoding size)
        x = x.cuda()
        y = y.cuda()
        return nn.functional.cross_entropy(self.logits(x), y.argmax(1))


char_encodings = [
    [1., 0., 0., 0., 0., 0., 0., 0.],  # ' ' 0
    [0., 1., 0., 0., 0., 0., 0., 0.],  # 'h' 1
    [0., 0., 1., 0., 0., 0., 0., 0.],  # 'e' 2
    [0., 0., 0., 1., 0., 0., 0., 0.],  # 'l' 3
    [0., 0., 0., 0., 1., 0., 0., 0.],  # 'o' 4
    [0., 0., 0., 0., 0., 1., 0., 0.],  # 'w' 5
    [0., 0., 0., 0., 0., 0., 1., 0.],  # 'r' 6
    [0., 0., 0., 0., 0., 0., 0., 1.]  # 'd' 7
]
encoding_size = len(char_encodings)

index_to_char = [' ', 'h', 'e', 'l', 'o', 'w', 'r', 'd']

x_train = torch.tensor([[char_encodings[0]],[char_encodings[1]], [char_encodings[2]], [char_encodings[3]], [char_encodings[3]], [char_encodings[4]], [char_encodings[0]], # ' hello '
                        [char_encodings[5]], [char_encodings[4]], [char_encodings[6]], [char_encodings[3]], [char_encodings[7]],  # 'world'
                        [char_encodings[0]], [char_encodings[1]], [char_encodings[2]], [char_encodings[3]], [char_encodings[3]], [char_encodings[4]], [char_encodings[0]],  # ' hello '
                        [char_encodings[5]], [char_encodings[4]], [char_encodings[6]], [char_encodings[3]], [char_encodings[7]]])  # 'world'
y_train = torch.tensor([char_encodings[1], char_encodings[2], char_encodings[3], char_encodings[3], char_encodings[4], char_encodings[0], # 'hello '
                        char_encodings[5], char_encodings[4], char_encodings[6], char_encodings[3], char_encodings[7], char_encodings[0], # 'world '
                        char_encodings[1], char_encodings[2], char_encodings[3], char_encodings[3], char_encodings[4], char_encodings[0], # 'hello '
                        char_encodings[5], char_encodings[4], char_encodings[6], char_encodings[3], char_encodings[7], char_encodings[0]]) # 'world '

model = LongShortTermMemoryModel(encoding_size)

optimizer = torch.optim.RMSprop(model.parameters(), float('1.0e-4'))
x_train.cuda()
y_train.cuda()
model.cuda()

if os.path.isfile('word_generator_state.pt'):
    state = torch.load('word_generator_state.pt')
    model.load_state_dict(state['model_state_dict'])
    optimizer.load_state_dict(state['optimizer_state_dict'])

for epoch in range(1000):
    model.reset()
    model.loss(x_train, y_train).backward()
    optimizer.step()
    optimizer.zero_grad()

    if epoch % 100 == 0:
        # Generate characters from the initial characters ' h'
        model.reset()
        text = ' h'
        model.forward(torch.tensor([[char_encodings[0]]]))
        y = model.forward(torch.tensor([[char_encodings[1]]]))
        text += index_to_char[y.argmax(1).detach()]
        for c in range(50):
            y = model.forward(torch.tensor([[char_encodings[y.argmax(1).detach()]]]))
            text += index_to_char[y.argmax(1).detach()]
        print(text)
torch.save({
    'model_state_dict':model.state_dict(),
    'optimizer_state_dict':optimizer.state_dict()
},'word_generator_state.pt')
