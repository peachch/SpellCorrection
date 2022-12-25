import os
import time
import json
import torch
import Levenshtein
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import Counter
from generate_pinyin_seq import generate_pinyin_seq


def correct_pinyin_seq(pinyin_seq):
    encoded_pinyin_seq = [wrong_dict.get(w, 0) for w in pinyin_seq]
    # print(encoded_pinyin_seq)

    #（syx）得到encode之后拼音的矩阵
    mb_x = torch.from_numpy(np.array(encoded_pinyin_seq).reshape(1, -1)).long().to(device)
    #（syx）得到拼音矩阵的长度
    mb_x_len = torch.from_numpy(np.array([len(encoded_pinyin_seq)])).long().to(device)
    #（syx）得到拼音开端
    bos = torch.Tensor([[correct_dict["BOS"]]]).long().to(device)


    translation, attn = model.translate(mb_x, mb_x_len, bos)
    translation = [inv_correct_dict[i] for i in translation.data.cpu().numpy().reshape(-1)]
    trans = []
    for word in translation:
        if word != "EOS":
            trans.append(word)
        else:
            break

    res = " ".join(trans)

    return res

#（syx）读取错误query和正确query 并进行加头和尾的处理
def load_data(in_file):
    wrong = []
    correct = []
    with open(in_file, 'r', encoding='utf8') as f:
        for line in f:
            # 按tab分隔成内容列表
            line = line.strip().split("\t")
            # 对英文分词并前后加上BOS和EOS
            wrong.append(["BOS"] + line[0].split() + ["EOS"])
            # 对中文直接按字符拆分并前后加上BOS和EOS
            correct.append(["BOS"] + line[1].split() + ["EOS"])
    return wrong, correct

#（syx）为了得到wrong_dict和correct_dict，其中字典的大小为50000，且排序方式为从大到小
def build_dict(sentences, max_words=50000):
    # 词频统计对象
    word_count = Counter()
    # 对所有句子中每个单词进行计数，更新到Counter对象中
    for sentence_words_list in sentences:
        for s in sentence_words_list:
            word_count[s] += 1
    # 获取最高频的max_words个单词
    ls = word_count.most_common(max_words)
    # 给拿到的高频词长度加2得到最终词典长度
    total_words = len(ls) + 2
    # 构建最终的单词到id的映射关系字典
    word_dict = {w[0]: index + 2 for index, w in enumerate(ls)}
    word_dict["UNK"] = UNK_IDX
    word_dict["PAD"] = PAD_IDX
    # 返回最终的单词到id的映射关系字典和词典总长度
    return word_dict, total_words


#通过wrong_dict和correct_dict生成每个sentence的数据
def encode_sentence(wrong_sentences, correct_sentences, wrong_dict, correct_dict, sort_by_len=True):
    """
    Encode the sequences.
    :param wrong_sentences:
    :param correct_sentences:
    :param wrong_dict:
    :param correct_dict:
    :param sort_by_len:
    :return:
    """
    # 获取错误句子数据长度
    # length = len(wrong_sentences)

    # 将所有错误句子序列转为对应的id序列，不在词典中则以0（UNK的id）设置
    out_wrong_sentences = [[wrong_dict.get(w, 0) for w in sent] for sent in wrong_sentences]
    # 将所有正确句子序列转为对应的id序列，不在词典中则以0（UNK的id）设置
    out_correct_sentences = [[correct_dict.get(w, 0) for w in sent] for sent in correct_sentences]

    # sort sentences by english lengths
    # 将句子数据的下标按句子长度排序，返回排序后的下标索引序列
    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    # 把错误句子和正确句子按照同样的顺序（错误句子长度）排序
    if sort_by_len:
        sorted_index = len_argsort(out_wrong_sentences)
        out_wrong_sentences = [out_wrong_sentences[i] for i in sorted_index]
        out_correct_sentences = [out_correct_sentences[i] for i in sorted_index]
    # 返回排序后的正确句子和错误句子id序列数据
    return out_wrong_sentences, out_correct_sentences


# 构建带Attention的Seq2Seq模型
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, enc_hidden_size, dec_hidden_size, dropout=0.2):
        super(Encoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, enc_hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(enc_hidden_size * 2, dec_hidden_size)

    def forward(self, x, lengths):
        sorted_len, sorted_idx = lengths.sort(0, descending=True)
        x_sorted = x[sorted_idx.long()]
        embedded = self.dropout(self.embed(x_sorted))

        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, sorted_len.long().cpu().data.numpy(),
                                                            batch_first=True)
        packed_out, hid = self.rnn(packed_embedded)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        _, original_idx = sorted_idx.sort(0, descending=False)
        out = out[original_idx.long()].contiguous()
        hid = hid[:, original_idx.long()].contiguous()

        hid = torch.cat([hid[-2], hid[-1]], dim=1)
        hid = torch.tanh(self.fc(hid)).unsqueeze(0)

        return out, hid


# scoring function
class Attention(nn.Module):
    def __init__(self, enc_hidden_size, dec_hidden_size):
        super(Attention, self).__init__()

        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size

        self.linear_in = nn.Linear(enc_hidden_size * 2, dec_hidden_size, bias=False)
        self.linear_out = nn.Linear(enc_hidden_size * 2 + dec_hidden_size, dec_hidden_size)

    def forward(self, output, context, mask):
        # output: batch_size, output_len, dec_hidden_size
        # context: batch_size, context_len, 2*enc_hidden_size

        batch_size = output.size(0)
        output_len = output.size(1)
        input_len = context.size(1)

        # 将
        context_in = self.linear_in(context.view(batch_size * input_len, -1)).view(
            batch_size, input_len, -1)  # batch_size, context_len, dec_hidden_size

        # context_in.transpose(1,2): batch_size, dec_hidden_size, context_len
        # output: batch_size, output_len, dec_hidden_size
        attn = torch.bmm(output, context_in.transpose(1, 2))
        # batch_size, output_len, context_len

        # 有些位置不是单词而是mask，要设置为特别小的值，这样不会影响softmax
        attn.data.masked_fill(mask, -1e6)

        attn = F.softmax(attn, dim=2)
        # batch_size, output_len, context_len

        context = torch.bmm(attn, context)
        # batch_size, output_len, enc_hidden_size

        output = torch.cat((context, output), dim=2)  # batch_size, output_len, hidden_size*2

        output = output.view(batch_size * output_len, -1)
        output = torch.tanh(self.linear_out(output))
        output = output.view(batch_size, output_len, -1)
        return output, attn


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, enc_hidden_size, dec_hidden_size, dropout=0.2):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.attention = Attention(enc_hidden_size, dec_hidden_size)
        self.rnn = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.out = nn.Linear(dec_hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def create_mask(self, x_len, y_len):
        # a mask of shape x_len * y_len
        device = x_len.device
        max_x_len = x_len.max()
        max_y_len = y_len.max()
        x_mask = torch.arange(max_x_len, device=x_len.device)[None, :] < x_len[:, None]
        y_mask = torch.arange(max_y_len, device=x_len.device)[None, :] < y_len[:, None]
        mask = (~ x_mask[:, :, None] * y_mask[:, None, :]).byte()
        return mask

    def forward(self, ctx, ctx_lengths, y, y_lengths, hid):
        sorted_len, sorted_idx = y_lengths.sort(0, descending=True)
        y_sorted = y[sorted_idx.long()]
        hid = hid[:, sorted_idx.long()]

        y_sorted = self.dropout(self.embed(y_sorted))  # batch_size, output_length, embed_size

        packed_seq = nn.utils.rnn.pack_padded_sequence(y_sorted, sorted_len.long().cpu().data.numpy(), batch_first=True)
        out, hid = self.rnn(packed_seq, hid)
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        _, original_idx = sorted_idx.sort(0, descending=False)
        output_seq = unpacked[original_idx.long()].contiguous()
        hid = hid[:, original_idx.long()].contiguous()

        mask = self.create_mask(y_lengths, ctx_lengths)

        output, attn = self.attention(output_seq, ctx, mask)
        output = F.log_softmax(self.out(output), -1)

        return output, hid, attn


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, x_lengths, y, y_lengths):
        encoder_out, hid = self.encoder(x, x_lengths)
        output, hid, attn = self.decoder(ctx=encoder_out,
                                         ctx_lengths=x_lengths,
                                         y=y,
                                         y_lengths=y_lengths,
                                         hid=hid)
        return output, attn

    def translate(self, x, x_lengths, y, max_length=100):
        encoder_out, hid = self.encoder(x, x_lengths)
        preds = []
        batch_size = x.shape[0]
        attns = []
        for i in range(max_length):
            output, hid, attn = self.decoder(ctx=encoder_out,
                                             ctx_lengths=x_lengths,
                                             y=y,
                                             y_lengths=torch.ones(batch_size).long().to(y.device),
                                             hid=hid)
            y = output.max(2)[1].view(batch_size, 1)
            preds.append(y)
            attns.append(attn)
        return torch.cat(preds, 1), torch.cat(attns, 1)


if __name__ == '__main__':
    # 记录读取数据开始时间
    start = time.time()

    # 设置工作路径
    WORK_DIR = os.path.dirname(os.path.abspath(__file__))
    os.chdir(WORK_DIR)

    # 设定UNK和PAD符号的词典id
    UNK_IDX = 0
    PAD_IDX = 1

    # # 设置数据所在路径
    # train_file = "./data/train.txt"
    # # dev_file = "./data/dev.txt"
    # dev_file = "./data/test.txt"
    # train_wrong, train_correct = load_data(train_file)
    # dev_wrong, dev_correct = load_data(dev_file)
    #
    # # 从训练集构建错误句子到id字典
    # wrong_dict, wrong_total_words = build_dict(train_wrong)
    # print('wrong_total_words: ', wrong_total_words)
    # # 从训练集构建正确句子到id字典
    # correct_dict, correct_total_words = build_dict(train_correct)
    # print('correct_total_words: ', correct_total_words)
    # # 构建错误句子和正确句子的id到单词映射关系字典
    # inv_wrong_dict = {v: k for k, v in wrong_dict.items()}
    # inv_correct_dict = {v: k for k, v in correct_dict.items()}
    #
    # train_wrong, train_correct = encode_sentence(train_wrong, train_correct, wrong_dict, correct_dict)
    # dev_wrong, dev_correct = encode_sentence(dev_wrong, dev_correct, wrong_dict, correct_dict)
    #
    # # train_length = len(train_wrong)
    # dev_length = len(dev_wrong)
    #
    # # 保存必要的字典
    # f_wrong_dict = open('./data/wrong_dict.json', 'w', encoding='utf-8')
    # f_correct_dict = open('./data/correct_dict.json', 'w', encoding='utf-8')
    # f_inv_correct_dict = open('./data/inv_correct_dict.json', 'w', encoding='utf-8')
    #
    # json.dump(wrong_dict, f_wrong_dict, indent=4)
    # json.dump(correct_dict, f_correct_dict, indent=4)
    # json.dump(inv_correct_dict, f_inv_correct_dict, indent=4)
    #
    # f_wrong_dict.close()
    # f_correct_dict.close()
    # f_inv_correct_dict.close()

    # %% 生成正确拼音序列到名词的映射关系
    # query_output_path = "./data/movie_names_process_query_unsorted.txt"
    # pinyin_output_path = "./data/movie_names_process_pinyin.txt"

    # f_pinyin = open(pinyin_output_path, 'r', encoding='utf-8')
    # f_query = open(query_output_path, 'r', encoding='utf-8')
    #
    # f_mapping_dict = open(mapping_dict_path, 'w', encoding='utf-8')
    #
    # # 初始化映射关系字典
    # answer_mapping_dict = {}
    #
    # for p in f_pinyin:
    #     answer_mapping_dict[p.strip()] = next(f_query).strip()
    #
    # json.dump(answer_mapping_dict, f_mapping_dict, indent=4)
    #
    # f_pinyin.close()
    # f_query.close()
    # f_mapping_dict.close()

    # %% 加载正确拼音序列到名词的映射关系
    mapping_dict_path = "./data/mapping_dict.json"

    with open(mapping_dict_path, 'r', encoding='utf-8') as fb:
        mapping_dict = json.load(fb)

    # %% 加载模型
    model_path = './model/seq2seq-model-191213.pt'

    # 设定加载模型需要的参数
    dropout = 0.2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embed_size = hidden_size = 100

    wrong_total_words = correct_total_words = 61

    # 加载需要的字典
    f_wrong_dict = open('./data/wrong_dict.json', 'r', encoding='utf-8')
    f_correct_dict = open('./data/correct_dict.json', 'r', encoding='utf-8')
    f_inv_correct_dict = open('./data/inv_correct_dict.json', 'r', encoding='utf-8')

    wrong_dict = json.load(f_wrong_dict)
    correct_dict = json.load(f_correct_dict)
    inv_correct_dict = json.load(f_inv_correct_dict)
    inv_correct_dict = {int(k): v for k, v in inv_correct_dict.items()}

    f_wrong_dict.close()
    f_correct_dict.close()
    f_inv_correct_dict.close()

    # 还原构建模型
    encoder = Encoder(vocab_size=wrong_total_words,
                      embed_size=embed_size,
                      enc_hidden_size=hidden_size,
                      dec_hidden_size=hidden_size,
                      dropout=dropout)
    decoder = Decoder(vocab_size=correct_total_words,
                      embed_size=embed_size,
                      enc_hidden_size=hidden_size,
                      dec_hidden_size=hidden_size,
                      dropout=dropout)
    model = Seq2Seq(encoder, decoder)
    model = model.to(device)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # %% -------在dev集上对模型预测结果进行检测
    while True:
        input_query = input('请输入待纠错句子：')
        dev_test_start = time.time()
        query_pinyin_seq = 'BOS ' + generate_pinyin_seq(input_query.strip()) + ' EOS'
        query_pinyin_seq_list = query_pinyin_seq.split()
        result = correct_pinyin_seq(query_pinyin_seq_list)
        print('纠错后的拼音序列：', result)
        # 在词典中
        if result in mapping_dict:
            # print('纠错后的句子为(在词典中)：', mapping_dict.get(result, 'not found in mapping dict!'))
            print('纠错后的句子为(在词典中)：', mapping_dict[result])
        # 不在词典中，则计算序列编辑距离排序
        else:
            best_pinyin_seq_key = None
            best_similarity = 0
            result_seq = result.split()
            for k in mapping_dict:
                k_similarity = Levenshtein.seqratio(result_seq, k.split())
                if k_similarity > best_similarity:
                    best_similarity = k_similarity
                    best_pinyin_seq_key = k
            if best_similarity >= 0.7:
                print('纠错后的句子为(不在词典中,返回最小距离句子)：', mapping_dict[best_pinyin_seq_key])
                print('相似度为：', best_similarity)
            else:
                print('未在词典中找到纠错后相似度超过阈值的句子!')
        print()
