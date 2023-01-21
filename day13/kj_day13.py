### Transformer ###

# module
import torch
import torch.nn as nn
import math
from torch.autograd import Variable
from copy import deepcopy

# Tokenization
# 연산을 수행하기 위해 텍스트를 숫자로 나타낼 방법이 필요함
# 토큰화 : 텍스트 문자열을 압축된 기호의 시퀀스로 구문 분석하는 프로세스
# 이 프로세스에서 각 정수가 텍스트의 일부를 나타내는 정수의 벡터가 생성됨
# 여기서 사용하는 토크나이저는 WordPiece임 (BERT와 같은 최근의 언어 모델이 WordPiece 토크나이저를 사용함)
# WordPiece는 전체 단어를 하나의 토큰으로 토큰화하는 것으로 디코딩이 더 쉽고 직관적으로 보임
from transformers import BertTokenizer
# BERT는 2018년 구글이 공개한 사전 훈련된(pretrained) 모델임
# 트랜스포머를 이용해 구현되었으며 위키피디아와 BooksCorpus와 같은 레이블이 없는 텍스트 데이터로 훈련되었음
# 기본 구조는 트랜스포머의 인코더를 쌓아올린 구조임
tok = BertTokenizer.from_pretrained("bert-base-uncased")

#tok("Hello, how are you doing?")['inputs_ids']
tok("The Frenchman spoke in the [MASK] language and ate 🥖")['input_ids']
tok("[CLS] [SEP] [MASK] [UNK]")['input_ids']


# Embeddings
# 텍스트의 적절한 표현을 학습하기 위해, 시퀀스의 각 개별 토큰은 임베딩을 통해 벡터로 변환됨
# 시퀀스의 각 개별 토큰은 신경망 계층의 한 종류로 볼 수 있음
# 임베딩에 대한 가중치는 트랜스포머 모델의 나머지 부분과 함께 학습되기 때문
# 이는 어휘(vocabulary)의 각 단어에 대한 벡터를 포함하고 있으며
# 이러한 가중치는 표준정규분포에서 초기화됨
# 모델 E ∈ Rvocab×dmodel 을 초기화 할 때 vocab의 크기 및 모델(dmodel=512)의 차원을 지정해야 함
# 마지막으로 정규화 단계로 가중치에 sqrt(dmodel)을 곱함
import torch
import torch.nn as nn
import math

class Embed(nn.Module):
    def __init__(self, vocab: int, d_model: int = 512):
        super(Embed, self).__init__()
        self.d_model = d_model
        self.vocab = vocab
        self.emb = nn.Embedding(self.vocab, self.d_model)
        # nn.Embedding()은 num_embeddings, embedding_dim 두 가지 인자를 받음
        # num_embeddings : 임베딩을 할 단어들의 개수, 즉 단어 집합의 크기
        # embedding_dim : 임베딩 할 벡터의 차원, 사용자가 정해주는 하이퍼파라미터임
        
        # d_model : Transformer의 Encoder와 Decoder에서의 정해진 입력과 출력의 크기를 의미, default=512
        
        self.scaling = math.sqrt(self.d_model)

    def forward(self, x):
        return self.emb(x) * self.scaling


# Positional Encoding
# CNN이나 RNN과는 다르게, 트랜스포머는 시퀀스에 임베드된 토큰의 상대 위치에 대한 정보를 가지고 있지 않음
# 따라서 인코더와 디코더에 대한 입력 임베딩에 인코딩을 추가함으로써 이 정보를 입력해야함
# 상대 위치에 대한 정보는 다양한 방법으로 추가할 수 있으며 정적이거나 학습될 수 있음
# 트랜스포머는 각 위치(pos)에 대한 sin, cos 변환을 사용함
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int = 512, dropout: float = .1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout) # 레이어간 연결 중 일부를 랜덤하게 삭제함, 0.1이므로 10%를 랜덤하게 삭제

        # Compute the positional encodings in log space
        pe = torch.zeros(max_len, d_model) # shape : (5000,512)
        position = torch.arange(0, max_len).unsqueeze(1) # shape : (5000, 1)
        # 수식에서 pos에 해당, 단어의 시점

        div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.Tensor([10000.0])) / d_model))
        # 수식에서 10000^(2i/d_model)에 해당
        # log를 썼다가 exp를 다시 써준 이유는 숫자 오버 플로우 방지를 위해 log space에서 연산하기 위함

        pe[:, 0::2] = torch.sin(position * div_term) # sin은 짝수 차원(2i)에서 사용
        pe[:, 1::2] = torch.cos(position * div_term) # cos은 홀수 차원(2i+1)에서 사용
        # 0::2는 step=2로 0부터 끝까지

        pe = pe.unsqueeze(0) # shape : (1,5000,512)
        self.register_buffer('pe', pe) # parameter가 아니라 buffer를 수행하기 위한 목적으로 활용
        # backpropagation을 진행하지 않고, optimization에 사용되지 않음
        # optimizer.step 적용 X, 즉 update 하지 않음
        # state_dict에는 저장되므로 나중에 다시 사용하기에 용이함
        # GPU에서 작동함
        # 쉽게 생각해서 모델의 parameter로 등록하지 않기 위한 함수라고 보면 됨
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        # Variable은 Tensor의 Wrapper로 연산 그래프에서 노드로 표현됨
        # Tensor를 감싸며, requires_grad=False이므로 역전파 중에 이 Variable들에 대한 gradient를 계산할 필요가 없음을 나타냄
        # 현재는 모든 Tensor가 자동적으로 Variable의 성질을 가지기 때문에 굳이 사용할 필요 X
        return self.dropout(x)


# Multi-Head Attention

# 참고
# Transformer 이전에, Sequence에서 학습하기 위한 AI연구의 패러다임은 CNN, RNN, LSTM 중 하나를 사용하는 것이었음
# Attention은 Transformer 이전에 이미 몇몇 NLP 성과를 이뤘으나, 그 당시에는 Convolution이나 Recurrent 없이 효율적인 모델을 구축할 수 있는지가 불분명했음

# 우선 Attention부터 확인
# Attention Layer는 Query와 Key, Value간의 맵핑을 학습할 수 있음
# 텍스트 생성의 맥락 기준 (의미가 특정 NLP 응용프로그램에 따라 달라지므로 기준 설정)
# Q는 입력의 임베딩, V와 K는 target (일반적으로 V=K)
class Attention:
    def __init__(self, dropout: float = 0.):
        super(Attention, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value, mask=None):
        d_k = query.size(-1) # 연산값이 너무 커지는 것을 방지하기 위한 조정 상수
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        # attention score계산 (QxK.transpose)/sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        # 만약 mask가 None이 아니라면, mask가 0인 부분은 -1e9(매우 큰 음수, 때로는 -inf)로 채움
        # 이렇게 하면 softmax를 통과했을 때의 값이 0이 되므로, Attention 매커니즘에 반영되지 않음
        # 이는 후속 위치를 처리함으로써 모델이 cheating(부정행위)을 저지르는 것을 예방하기 위한 것임
        # 이를 통해 모델은 다음 토큰을 예측하려 할 때 이전 위치의 단어에만 주의를 기울일 수 있음

        p_attn = self.dropout(self.softmax(scores)) # attention score를 softmax함수에 통과시킴

        return torch.matmul(p_attn, value) # attention value(or context vector)를 출력
        # attention value = softmax(attention score) * V
    
    def __call__(self, query, key, value, mask=None):
        return self.forward(query, key, value, mask)
        # __call__ 은 클래스를 함수처럼 사용할 수 있게 해주는 함수임
        # 클래스를 사용하려면 클래스의 객체를 만들어야하는데 call을 넣어주면 그렇게 할 필요없이 바로 선언하는 것 가능

# 단일 Attentino Layer는 하나의 표현만을 허용하므로 Transformer에서는 Multi-Head Attention이 사용됨
# Multi-Head Attention은 여러번의 Attention을 병렬로 사용하는 방법
# 이를 통해 모델이 다중 패턴 및 표현을 학습할 수 있음
class MultiHeadAttention(nn.Module):
    def __init__(self, h: int = 8, d_model: int = 512, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_k = d_model // h # d_model의 차원을 num_heads로 나누어서 d_model/num_heads의 차원을 가지는 Q,K,V에 대해서 num_heads개의 병렬 Attention 수행
        # 각 head의 차원을 h로 나누기 때문에 총 연산은 완전한 차원을 가진 하나의 Attention head를 사용하는 것과 유사함
        self.h = h # num_heads = 8, 즉 8개의 병렬 Attention
        self.attn = Attention(dropout)
        self.lindim = (d_model, d_model) # (512,512)
        self.linears = nn.ModuleList([deepcopy(nn.Linear(*self.lindim)) for _ in range(4)])
        # deepcopy는 내부 객체들까지 모두 copy되는 것, 즉 deepcopy는 원본 배열을 보존하기 위해서 사용
        # ModuleList에 nn.Linear(512,512) 모듈을 4개 저장함
        # 이렇게 해두면 이를 리스트처럼 순서대로 iterable하게 접근하여 사용이 가능함 

        self.final_linear = nn.Linear(*self.lindim, bias=False)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        
        query, key, value = [l(x).view(query.size(0), -1, self.h, self.d_k).transpose(1, 2) \
                             for l, x in zip(self.linears, (query, key, value))]
        # self.linears에 4개의 nn.Linear(512,512)가 저장되어 있으므로 for문이 총 4번 돌아감
        # l, x를 4번 만들어서 위와 같은 연산을 통해 query, key, value를 각각 만듦

        nbatches = query.size(0)
        x = self.attn(query, key, value, mask=mask)
        # 각 query, key, value를 Attention 클래스에 넣어서 계산
        
        # Concatenate and multiply by W^O
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        # view나 transpose와 같은 함수는 메모리를 따로 할당하지 않는 tensor 객체 연산
        # 이는 column 기준으로 축을 바꿔버려서 나중에 바뀐 축단위 포인터 연산이 필요할 시 문제가 생길 수 있음
        # contiguous는 새로운 메모리 공간에 데이터를 복사하여 주소값 연속성을 가변적이게 만들어줌
        # 따라서 위에서 발생하는 문제 해결
        # 이 과정들을 통해 나눠졌던 head들을 연결

        return self.final_linear(x) # 가중치 행렬 W_o에 곱해지게끔 설정
        # 최종 결과물은 인코더의 입력이었던 행렬의 크기와 동일


# Residuals and Layer Normalization
# Residual connection 및 배치 정규화와 같은 개념이 퍼포먼스를 향상시키고, 훈련 시간을 단축하며, 보다 심층적인 네트워크의 훈련을 가능케 함
# 따라서 모든 Attention Layer 및 Feed Forward Layer 다음에 이를 갖추고 있음
# 각 레이어에 dropout이 추가되어 있는 이유 : 더 나은 일반화를 위해서

# Normalization
# 현대 딥러닝 기반 모델은 보통 배치 정규화를 포함하고 있음
# 그러나 배치 정규화는 배치 사이즈에 의해 좌우되므로 대신 레이어 정규화를 사용
# 레이어 정규화는 배치 크기가 작더라도 안정적임
# 레이어 정규화 : 각 인스턴스에서 나온 feature를 모든 channel에 걸쳐 한 번

class LayerNorm(nn.Module):
    def __init__(self, features: int, eps: float = 1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps
        # features로 받은 수 만큼 1벡터와 0벡터를 만듦
        # gamma = 1벡터의 parameter
        # beta = 0벡터의 parameter
        # gamma와 beta는 학습 가능한 매개변수임 (역전파를 통해 학습)

    def forward(self, x):
        mean = x.mean(-1, keepdim=True) # 미니배치의 평균
        std = x.std(-1, keepdim=True) # 미니배치의 분산
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
        # 표준편차가 0일수도 있기 때문에 수치 안정성을 위해 eps가 추가됨
        # 위 식은 일반적인 정규화 식이 아니라 레이어 정규화 식임
        # 미니배치의 평균과 분산을 이용해서 정규화 한 뒤
        # scale 및 shift를 gamma, beta 값을 통해 실행함
        
# Residual Connection
# 이전 레이어의 출력을 현재 레이어의 출력에 추가하는 것을 의미
# 네트워크는 본질적으로 특정 레이어를 건너뛰기 할 수 있으므로, 이는 매우 깊은 네트워크를 허용함
# 그림에서 Add에 해당
# 쉽게 생각해서 블록 계산(ex.Feed Forward)을 건너뛰는 경로를 두는 것을 의미함

class ResidualConnection(nn.Module):
    def __init__(self, size: int = 512, dropout: float = .1):
        super(ResidualConnection,  self).__init__()
        self.norm = LayerNorm(size) # Layernorm의 feature size가 size가 됨
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
        # 그 다음 각 레이어의 최종 출력이 위 return과 같음


# Feed Forward
# 모든 Attention Layer의 위에 Feed Forward가 추가됨
# Feed Forward는 ReLU와 내부 레이어 드롭아웃을 통해 fully-connected Layer로 구성됨
# 논문에서 사용한 표준 차원은 입력 레이어의 경우 d_model=512, 내부 레이어의 경우 d_ff=2048
# d_ff = dim_feedforward : FFN model의 차원, FFN의 은닉층의 크기 (default=2048)
# FeedForward(x) = W2max(0, xW1 + B1) + B2

class FeedForward(nn.Module):
    def __init__(self, d_model: int = 512, d_ff: int = 2048, dropout: float = .1):
        super(FeedForward, self).__init__()
        self.l1 = nn.Linear(d_model, d_ff) # nn.Linear(512,2048)
        self.l2 = nn.Linear(d_ff, d_model) # nn.Linear(2048, 512)
        self.relu = nn.ReLU() # max(0,x)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.l2(self.dropout(self.relu(self.l1(x))))


# Encoder
# 단일 Encoder Layer는 Feed Forward가 뒤따르는 Multi-Head Attention Layer로 구성됨
# 여기에 Residual connection 및 Layer 정규화 또한 포함됨
# Enconding(x, mask) = FeedForward(MultiHeadAttention(x))

class EncoderLayer(nn.Module):
    def __init__(self, size: int, self_attn: MultiHeadAttention, feed_forward: FeedForward, dropout: float = .1):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn # Multi-Head Attention (Self-attention)
        self.feed_forward = feed_forward
        self.sub1 = ResidualConnection(size, dropout)
        self.sub2 = ResidualConnection(size, dropout)
        self.size = size

    def forward(self, x, mask):
        x = self.sub1(x, lambda x: self.self_attn(x, x, x, mask))
        # x, x, x는 각각 query, key, value로 MultiHeadAttention의 인자로서 작용함
        # 최종적으로 하나의 x로 출력되고, 그 값에 ResidualConnection 적용
        # mask : Padding mask
        return self.sub2(x, self.feed_forward)
        # 위 x를 Feed Forward와 같이 ResidualConnection을 적용시켜서 반환

# 논문의 최종 Transformer Encoder는 Layer 정규화가 뒤따르는 6개의 동일한 Layer 정규화로 구성됨
class Encoder(nn.Module):
    def __init__(self, layer, n: int = 6):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([deepcopy(layer) for _ in range(n)])
        # layer로 Encoder Layer를 받아서 n개만큼 똑같이 생성
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
        # layers에서 layer를 하나씩 가져와서 layer에 x와 mask대입
        # 최종적으로 나온 x를 Layer 정규화

# Decoder
# Decoding Layer는 메모리를 포함한 Multi-Head Attention Layer가 뒤따르는 Masked Multi-Head Attention Layer임
# Memory는 Encoder의 출력
# 마지막으로 역시 Feed Forward를 통과함
# 또한 모든 구성요소는 Encoder와 마찬가지로 Residual Connection 및 Layer 정규화를 포함함
# Decoding(x, memory, mask1, mask2) = FeedForward(MultiHeadAttention(MultiHeadAttention(x, mask1), memory, mask2))

class DecoderLayer(nn.Module):
    def __init__(self, size: int, self_attn: MultiHeadAttention, src_attn: MultiHeadAttention, 
                 feed_forward: FeedForward, dropout: float = .1):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn # Masked MultiHeadAttention (Self-attention)
        self.src_attn = src_attn # MultiHeadAttention (일반 attention)
        self.feed_forward = feed_forward
        self.sub1 = ResidualConnection(size, dropout) # 첫번째 서브층
        self.sub2 = ResidualConnection(size, dropout) # 두번째 서브층
        self.sub3 = ResidualConnection(size, dropout) # 세번째 서브층
 
    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.sub1(x, lambda x: self.self_attn(x, x, x, tgt_mask)) # 첫번째 서브층 적용
        # x, x, x는 각각 query, key, value로 MultiHeadAttention의 인자로서 작용함
        # tgt_mask : look-ahead mask
        # RNN의 경우 입력 단어를 매 시점마다 순차적으로 입력받으므로 다음 단어 예측에 현재 시점을 포함한 이전 시점에 입력된 단어들만 참고 가능
        # but Transformer는 문장 행렬로 이미 입력을 한 번에 받았음
        # 따라서 현재 시점의 단어를 예측하고자 할 때, 미래 시점의 단어까지도 참고할 수 있는 현상이 발생할 수 있음
        # 이를 방지하기 위해서 사용하는 것이 look-ahead mask임
        # 마스킹을 하고자 하는 위치에는 1, 마스킹을 하지 않는 위치에는 0
        x = self.sub2(x, lambda x: self.src_attn(x, memory, memory, src_mask)) # 두번째 서브층 적용
        # memory는 Encoder의 출력, x는 첫번째 서브층의 출력
        # src_mask : Padding mask
        # Key에 PAD가 있는 경우 해당 열 전체를 마스킹하기 위해 사용
        # PAD의 경우 softmax를 지나고 나면 합이 0이 되어 어떠한 유의미한 값도 가지고 있지 않게 됨
        return self.sub3(x, self.feed_forward)
        # 위 x를 Feed Forward와 같이 ResidualConnection을 적용시켜서 반환

# 최종 Encoder에서와 같이, 논문의 Decoder는 Layer 정규화가 뒤따르는 6개의 동일한 Layer를 가지고 있음
class Decoder(nn.Module):
    def __init__(self, layer: DecoderLayer, n: int = 6):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([deepcopy(layer) for _ in range(n)])
        # layer로 Decoder Layer를 받아서 n개만큼 똑같이 생성
        self.norm = LayerNorm(layer.size) # Layer 정규화
        
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)
    # 각 Layer마다 Decoder Layer 수행 후 Layer정규화까지 함


# Output
# 마지막으로, Decoder의 벡터 출력은 최종 출력으로 변환되어야 함
# 언어 번역과 같은 Seq2Seq 문제의 경우 출력은 각 위치에 대한 총 어위에 관한 확률 분포임
# Fully-Connected Layer는 Decoder Output을 logits의 행렬로 변환하며, 이는 target어휘의 차원을 가지고 있음
# 이러한 숫자를 softmax를 통해 어휘에 대한 확률 분포로 변환함
# 만약 번역된 문장이 20개의 토큰을 갖고 있고, 총 어휘는 30000개의 토큰이라고 가정하면
# 결과 출력은 행렬 M ∈ R(20×30000)이 됨
# 그 다음 마지막 차원에 대한 argmax를 취하여 Tokenizer를 통해 텍스트 열로 Decoding 할 수 있는 출력 토큰 T ∈ N(20)의 벡터를 얻을 수 있음
# Output(x) = LogSoftmax(max(0, xW1 + B1)) 

class Output(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(Output, self).__init__()
        self.l1 = nn.Linear(input_dim, output_dim)
        self.log_softmax = nn.LogSoftmax(dim=-1) # 더 빠르고 수치적으로 더 안정적이기 때문에 LogSoftmax 사용

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.l1(x)
        return self.log_softmax(logits)


# Encoder-Decoder
# 보다 높은 수준의 Encoder 및 Decoder의 표현을 통해, 최종 Encoder-Decoder 블록을 쉽게 공식화 할 수 있음

class EncoderDecoder(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, 
                 src_embed: Embed, tgt_embed: Embed, final_layer: Output):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder # Encoder
        self.decoder = decoder # Decoder
        self.src_embed = src_embed # Embed
        self.tgt_embed = tgt_embed # Embed
        self.final_layer = final_layer # Output
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.final_layer(self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask))
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


# Model Initialization
# 논문에서의 경우와 같이 동일한 차원으로 Transformer모델을 구축
# 초기화 전략은 Xavier/Glorot 초기화 : fan in과 fan out을 모두 고려하여 확률 분포를 조정해줌
# tanh를 activation function으로 사용하는 신경망에서 많이 사용됨
# Xavier(W) ~ U[-1/n, 1/n], B=0 (균일 분포, 모든 bias 0으로 초기화)

def make_model(input_vocab: int, output_vocab: int, d_model: int = 512):
    encoder = Encoder(EncoderLayer(d_model, MultiHeadAttention(), FeedForward()))
    decoder = Decoder(DecoderLayer(d_model, MultiHeadAttention(), MultiHeadAttention(), FeedForward()))
    input_embed= nn.Sequential(Embed(vocab=input_vocab), PositionalEncoding())
    output_embed = nn.Sequential(Embed(vocab=output_vocab), PositionalEncoding())
    output = Output(input_dim=d_model, output_dim=output_vocab)
    model = EncoderDecoder(encoder, decoder, input_embed, output_embed, output)
    # 아래 예시 적용시 input_vocab=10, output_vocab=10, d_model=512
    
    # Initialize parameters with Xavier uniform 
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


# Tokenized symbols for source and target
src = torch.tensor([[1, 2, 3, 4, 5]]) # 토큰화된 input
src_mask = torch.tensor([[1, 1, 1, 1, 1]]) # Padding mask
tgt = torch.tensor([[6, 7, 8, 0, 0]]) # 토큰화된 output
tgt_mask = torch.tensor([[1, 1, 1, 0, 0]]) # Look-ahead mask

# Create PyTorch model
model = make_model(input_vocab=10, output_vocab=10)
# 입력과 출력에 대한 어휘가 10개만 있다고 가정

# Do inference and take tokens with highest probability through argmax along the vocabulary axis (-1)
result = model(src, tgt, src_mask, tgt_mask)
result
result.shape # 1,5,10
result.argmax(dim=-1)
# argmax : arguments of the maxima
# tensor에 있는 모든 element들 중에서 가장 큰 값을 가지는 element의 index반환
# dim=-1이므로 마지막 차원을 기준으로 max값이 있는 index를 출력