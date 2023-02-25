import torch
import torch.nn as nn
import torch.nn.functional as F

# multiscale中加入Squeeze-and-Excitation Networks
class Attention(nn.Module):
    def __init__(self, channel=64,  ratio=8):
        super(Attention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
        self.shared_layer = nn.Sequential(
            nn.Linear(in_features=channel, out_features=channel // ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=channel // ratio, out_features=channel),
            nn.ReLU(inplace=True)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, F):
        b, c, _, _ = F.size()

        F_avg = self.shared_layer(self.avg_pool(F).reshape(b, c))
        F_max = self.shared_layer(self.max_pool(F).reshape(b, c))
        M = self.sigmoid(F_avg + F_max).reshape(b, c, 1, 1)

        return F * M


# input [64, 6, 99, 20]
class Lstm_DSMIL_Att(nn.Module):
    def __init__(self):
        super(Lstm_DSMIL_Att, self).__init__()

        # -----------------------------Conv2D版本----------------------------------------------------------------

        self.conv1_1 = nn.Conv1d(in_channels=20, out_channels=512, kernel_size=7)
        self.pool1_1 = nn.MaxPool1d(kernel_size=2)
        self.squeeze_1 = nn.AdaptiveAvgPool1d(output_size=1)
        self.excitation_1 = nn.Sequential(
            nn.Linear(in_features=512, out_features=16),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=16, out_features=512),
            nn.Sigmoid()
        )
        self.conv1_2 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=7)
        self.pool1_2 = nn.AdaptiveAvgPool1d(output_size=1)
        self.pool1_3 = nn.AdaptiveMaxPool1d(output_size=1)

        self.conv2_1 = nn.Conv1d(in_channels=20, out_channels=512, kernel_size=10)
        self.pool2_1 = nn.MaxPool1d(kernel_size=2)
        self.squeeze_2 = nn.AdaptiveAvgPool1d(output_size=1)
        self.excitation_2 = nn.Sequential(
            nn.Linear(in_features=512, out_features=16),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=16, out_features=512),
            nn.Sigmoid()
        )
        self.conv2_2 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=10)
        self.pool2_2 = nn.AdaptiveAvgPool1d(output_size=1)
        self.pool2_3 = nn.AdaptiveMaxPool1d(output_size=1)

        self.conv3_1 = nn.Conv1d(in_channels=20, out_channels=512, kernel_size=15)
        self.pool3_1 = nn.MaxPool1d(kernel_size=2)
        self.squeeze_3 = nn.AdaptiveAvgPool1d(output_size=1)
        self.excitation_3 = nn.Sequential(
            nn.Linear(in_features=512, out_features=16),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=16, out_features=512),
            nn.Sigmoid()
        )
        self.conv3_2 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=15)
        self.pool3_2 = nn.AdaptiveAvgPool1d(output_size=1)
        self.pool3_3 = nn.AdaptiveMaxPool1d(output_size=1)

        self.out_lstm = nn.LSTM(input_size=1536, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True)
        # self.out_lstm = nn.LSTM(input_size=768, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True)
        self.seq_linear = nn.Linear(192, 100)
        self.seq_linear2 = nn.Linear(100, 1)

        self.drop = nn.Dropout(0.5)
        self.fla = nn.Flatten(start_dim=1)
        # self.linear = nn.Linear(292, 100)
        self.linear = nn.Linear(256, 1)
        # self.linear2 = nn.Linear(100, 1)

        # -------------------------------------------------------------------------------------------------------

        # valid上可以0.82 测试集只有0.7953 0.7937 保存最好：0.7846 0.7824
        self.bert_conv1 = nn.Conv1d(in_channels=768, out_channels=512, kernel_size=1)
        self.bert_conv2 = nn.Conv1d(in_channels=512, out_channels=1, kernel_size=1)
        self.bert_pool = nn.AdaptiveMaxPool1d(output_size=1)
        self.bert_linear = nn.Linear(101, 256)
        self.bert_lstm = nn.LSTM(input_size=512, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True)
        self.squeeze_4 = nn.AdaptiveAvgPool1d(output_size=1)
        self.excitation_4 = nn.Sequential(
            nn.Linear(in_features=512, out_features=16),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=16, out_features=512),
            nn.Sigmoid()
        )

        self.residual = nn.Conv1d(in_channels=84, out_channels=192, kernel_size=1)


    def forward(self, feats, bert):

        # ------------------------------Conv2D---------------------------------------------------
        feats = feats.permute(0, 2, 1)

        seq_out1 = self.conv1_1(feats)
        seq_out1 = self.pool1_1(seq_out1)
        residual_1 = seq_out1
        b_1, c_1, _,  = seq_out1.size()  # 64 512 168
        y_1 = self.squeeze_1(seq_out1).view(b_1, c_1)  # [64, 512]
        z_1 = self.excitation_1(y_1).view(b_1, c_1, 1)  # [64, 512, 1]
        seq_out1 = seq_out1 * z_1.expand_as(seq_out1)
        # print(seq_out1.shape)
        # seq_out1 = self.conv1_2(seq_out1)
        seq_out1 = self.pool1_3(seq_out1)
        seq_out1 = seq_out1.permute(0, 2, 1)

        seq_out2 = self.conv2_1(feats)
        seq_out2 = self.pool2_1(seq_out2)
        residual_2 = seq_out2
        b_2, c_2, _, = seq_out2.size()
        y_2 = self.squeeze_2(seq_out2).view(b_2, c_2)
        z_2 = self.excitation_2(y_2).view(b_2, c_2, 1)
        seq_out2 = seq_out2 * z_2.expand_as(seq_out2)
        # print(seq_out1.shape)
        # seq_out2 = self.conv2_2(seq_out2)
        seq_out2 = self.pool2_3(seq_out2)
        seq_out2 = seq_out2.permute(0, 2, 1)

        seq_out3 = self.conv3_1(feats)
        seq_out3 = self.pool3_1(seq_out3)
        residual_3 = seq_out3
        b_3, c_3, _,  = seq_out3.size()
        y_3 = self.squeeze_3(seq_out3).view(b_3, c_3)
        z_3 = self.excitation_3(y_3).view(b_3, c_3, 1)
        seq_out3 = seq_out3 * z_3.expand_as(seq_out3)
        # print(seq_out1.shape)
        # seq_out3 = self.conv3_2(seq_out3)
        seq_out3 = self.pool3_3(seq_out3)
        seq_out3 = seq_out3.permute(0, 2, 1)

        seq_out = torch.cat((seq_out1, seq_out2, seq_out3), dim=2)
        # print(seq_out.shape)
        seq_out, _ = self.out_lstm(seq_out)
        seq_out = torch.squeeze(seq_out, 1)

        # bert特征原始版本
        # ds-ssb feature
        # ds-ssb = ds-ssb.permute(0, 2, 1)
        # bert_out = self.bert_conv1(ds-ssb)
        # bert_out = self.bert_pool(bert_out)
        # bert_out = bert_out.permute(0, 2, 1)
        # bert_out, _ = self.bert_lstm(bert_out)
        # bert_out = torch.squeeze(bert_out, 1)

        # bert特征使用SE模块版本
        bert = bert.permute(0, 2, 1)
        bert_out = self.bert_conv1(bert)
        bert_out = self.bert_pool(bert_out)
        residual = bert_out
        b_4, c_4, _, = bert_out.size()
        y_4 = self.squeeze_4(bert_out).view(b_4, c_4)
        z_4 = self.excitation_4(y_4).view(b_4, c_4, 1)
        bert_out = bert_out * z_4.expand_as(bert_out)
        bert_out = bert_out.permute(0, 2, 1)
        bert_out, _ = self.bert_lstm(bert_out)
        bert_out = torch.squeeze(bert_out, 1)

        out = torch.cat((seq_out, bert_out), dim=1)
        prediction_bag = self.linear(out)
        return prediction_bag


