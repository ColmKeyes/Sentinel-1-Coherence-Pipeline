
import torch
import torch.nn as nn
import xarray as xr

class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.conv = nn.Conv2d(in_channels=self.input_channels + self.hidden_channels,
                              out_channels=4 * self.hidden_channels,
                              kernel_size=self.kernel_size,
                              padding=self.padding)

    def forward(self, x, hidden):
        hx, cx = hidden
        combined = torch.cat((x, hx), dim=1)
        gates = self.conv(combined)

        ingate, forgetgate, cellgate, outgate = gates.chunk(4, dim=1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, cy

class ConvLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, num_layers):
        super(ConvLSTM, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers

        cell_list = []
        for i in range(self.num_layers):
            cur_input_channels = self.input_channels if i == 0 else self.hidden_channels

            cell_list.append(ConvLSTMCell(input_channels=cur_input_channels,
                                          hidden_channels=self.hidden_channels,
                                          kernel_size=self.kernel_size))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, x, hidden=None):
        seq_len, batch_size, _, height, width = x.size()

        if hidden is None:
            hidden = self.init_hidden(batch_size, height, width)

        layer_output_list = []
        last_states = []

        for layer_idx in range(self.num_layers):
            hidden_seq = []

            for t in range(seq_len):
                hx, cx = self.cell_list[layer_idx](x[t, :, :, :, :], hidden[layer_idx])
                hidden_seq.append(hx)

            hidden[layer_idx] = (hx, cx)
            x = torch.stack(hidden_seq, dim=0)
            layer_output_list.append(x)
            last_states.append((hx, cx))

        layer_output_list = torch.stack(layer_output_list, dim=1)

        last_states = [torch.stack(states) for states in last_states]
        last_states = torch.stack(last_states, dim=0)

        return layer_output_list, last_states

    def init_hidden(self, batch_size, height, width):
        init_states = []
        for _ in range(self.num_layers):
            init_states.append((torch.zeros(batch_size, self.hidden_channels, height, width),
                                torch.zeros(batch_size, self.hidden_channels, height, width)))
        return init_states

# Example usage
# input_channels = 3
# hidden_channels = 64
# kernel_size = 3
# num_layers = 2
# seq_len = 5
# batch_size = 1
# height = 128
# width = 128
#
# model = ConvLSTM(input_channels, hidden_channels, kernel_size, num_layers)
# input_data = torch.randn(seq_len, batch_size, input_channels, height, width)
# output, _ = model(input_data)
# print(output.shape)