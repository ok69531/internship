from transformers import BertTokenizer

import torch
from torch import nn
import math
from torch.autograd import Variable
from copy import deepcopy

#토큰화(Tokenization)
#바이터 페어 인코딩(BPE)가 아닌 Wordpiece를 사용
#WordPiece는 전체 단어를 하나의 단어로 토큰화해 디코딩이 직관적
#단어의 조각으로 토큰화하는 BPE처럼 알 수 없는 토큰 X
#실무자들의 대부분 사전훈련된 WordPiece 토크나이저 사용



tok = BertTokenizer.from_pretrained("bert-base-uncased")

#[CLS] : 인코딩의 시작 = 101
#[SEP] : 인코딩의 종료 즉, 분리 = 102
#[MASK] : 기타 특수 토큰 = 103
#[UNK] : 알 수 없는 기호(빵 이모티콘처럼) = 100

tok("Hello, how ar you doing")['input_ids']

tok("The Frenchman spoke in the [MASK] language and ate 🥖")['input_ids']

tok("[CLS] [SEP] [MASK] [UNK]")['input_ids']

# 임베딩(Embeddings)
# 텍스트의 학습을 위해 각 토큰들은 임베딩 벡터로 변환
# 임베딩에 대한 가중치는 트랜스포머 모델의 나머지 부분과 학습
# 가중치는 정규분포 N(0,1)에서 초기화
# 모델의 초기화 시에는 어휘의 크기 및 모델의 차원을 지정
# 마지막으로 정규화 단계로 가중치에 d_model을 곱함

class Embed(nn.Module):
    def __init__(self, vocab: int, d_model: int = 512): #512는 트랜스포머가 소개된 논문에서 정한 수치(모델 차원)
        super(Embed, self).__init__()
        self.d_model = d_model # 모델 차원 설정
        self.vocab = vocab # 입력할 어휘의 크기
        self.emb = nn.Embedding(self.vocab, self.d_model) # 어휘를 512차원으로 임베딩
        self.scaling = math.sqrt(self.d_model) # 근데 왜 루트 512를 곱해야함???

    def forward(self, x):
        return self.emb(x) * self.scaling #임베딩된 단어의 정규화
    

#포지셔널 인코딩(Positional Encoding)
#순환 및 합성곱 신경망과는 대조적으로, 모델 자체는 시퀀스에 임베드된 토큰의 상대위치정보 없음
#따라서 인코더와 디코더에 대한 입력 임베딩에 인코딩을 추가함으로써 이 정보를 입력
# 각 위치에 대한 삼각함수 변환을 이용.
# sin은 짝수차원, cos는 홀수 차원에 이용
# 숫자 오버플로우 방지를 위해 로그 공간에서 연산

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int = 512, dropout: float = .1, max_len: int = 5000): #10퍼센트 노드는 사용 X
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Compute the positional encodings in log space
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1) #0부터 max_len까지 위치들을 표현하기 위하여
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.Tensor([10000.0])) / d_model)) 
        # 식에서 분모부분 로그 씌웠다가 다시 exp 준거 (지수족 분포 할때처럼)
        # 2i 부분을 arange(0, d_model, 2)로 표현 (0부터 512까지 2씩 뛰니까)
        pe[:, 0::2] = torch.sin(position * div_term) # 짝수차원이니까 sin
        pe[:, 1::2] = torch.cos(position * div_term) # 홀수차우너이니까 cos
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        # resgister_buffer란?
        # Optimizer가 step 즉, update를 안함
        # state_dict에는 저장됨(나중에 사용 가능)
        # Gpu에서 작동

        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False) # 김승기 피셜 Variable 대신 토치 써도 된다
        # 확인해볼것
        # Variable은 옛날 포지션이지만 
        #임베딩된 값에 포지셔널 인코딩 값을 더해 위치 정보 포함
        return self.dropout(x) # 더 좋은 성능을 위한 dropout

# 다중 헤드 어텐션(Multi-Head Attention)
# 합성곱과 순환 제거하고 과감히 어텐션만 사용
# 어텐션레이어에서 Query와 Key,Value 쌍 간의 맵핑을 학습
# query는 입력의 임베딩, key와 value는 타깃(일반적으로 key와 value는 동일)
# Attention(Q,K,V) = softmax(QK'/d_k^1/2)V

class Attention:
    def __init__(self, dropout: float = 0.):
        super(Attention, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1) #QK'/d_k^1/2가 통과할 softmax함수

    def forward(self, query, key, value, mask=None):
        d_k = query.size(-1) # d_k는 query의 사이즈로 d_model / num_heads, 논문에선 각각 512와 8을 사용
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k) #attention score 공식을 표현한 것
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9) #mask가 0인 부분을 아주 작은 값으로, -inf 써도 된다~
        #마스크는 모델의 부정행위 예방(다음 토큰 예측할 때 이전 위치의 단어에 집중하도록)
        p_attn = self.dropout(self.softmax(scores)) #softmax함수에 QK'/d_k^1/2 통과시킴
        return torch.matmul(p_attn, value) # 최종적으로 V와 곱해서 attention matrix 획득
    
    def __call__(self, query, key, value, mask=None): #call은 함수를 호출하는 것처럼 클래스의 객체도 호출하게 해줌!
        return self.forward(query, key, value, mask)


# multihead
# 한 번의 어텐션 보다 여러 번의 병렬적인 어텐션이 더 효과적
# 각 어텐션 헤드가 다른 시각에서 단어 간의 연관성을 파악
# 차원을 축소시켜 num_heads 만큼의 병렬 연산
class MultiHeadAttention(nn.Module):
    def __init__(self, h: int = 8, d_model: int = 512, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_k = d_model // h #앞에서와 마찬가지로 d_k를 512/8로 정의
        self.h = h #num_heads=8 을 h로 정의
        self.attn = Attention(dropout) # 작성했던 attention클래스를 attn으로 정의 (우리가 train할때 model = class 했던거랑 같음)
        self.lindim = (d_model, d_model) # 512*512 행렬 정의 (트랜스포머 설명에서 봤던 정사각형 네모 행렬 생각하면 된다)
        self.linears = nn.ModuleList([deepcopy(nn.Linear(*self.lindim)) for _ in range(4)])
        # nn.ModuleList는 모듈의 존재를 pytorch에게 돌려줌
        # 그냥 리스트로 받아버리면 pytorch가 몰라서 hyperparameter가 모듈에 존재하지 않는다는 오류를 뱉음
        # 여러 종류의 모델이 받는 인풋이 서로 다르지만 반복해서 정의해야할 때 사용
        # attention에선 512*512와 query, key, value 4개를 정의해야하니까 for문으로 4번 반복
        # 서로 길이가 다르므로 *를 사용해 가변적으로 만들어줬어용
        self.final_linear = nn.Linear(*self.lindim, bias=False) # 마찬가지로 여러 개 받으려구 *를 사용해 정의
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1) #unsqueeze로 차원을 맞춰주자
        
        # 가중치에 곱해질 query, key, value를 for문으로 생성하고 곱하기 위해 전치
        query, key, value = [l(x).view(query.size(0), -1, self.h, self.d_k).transpose(1, 2) \
                             for l, x in zip(self.linears, (query, key, value))]
        nbatches = query.size(0)
        x = self.attn(query, key, value, mask=mask) #만들어낸 query, key, value로 어텐션해서
        #x에 어텐션 매트릭스를 리턴받음
        
        # Concatenate and multiply by W^O
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.final_linear(x)
#transpose는 기본 메모리 저장소를 원래의 tensor와 공유하기 때문에 .contiguous 방법은 .transpose 다음에 추가됩니다.
#이 후 .view를 호출하려면 인접한(contiguous) tensor가 필요합니다 (문서). .view 방법은 효율적인 변환(reshaping), 슬라이싱(slicing) 및 요소별 작업을 수행할 수 있습니다



# 레지듀얼 및 레이어 정규화
# 레지듀얼 커넥션 및 정규화가 퍼포먼스 향상 및 훈련 시간 단축에 영향을 줌
# 더 나은 일반화를 위해 각 레이어에 드롭아웃 추가

# 정규화
# 현대 딥러닝 기반 컴퓨터 비전 모델은 배치 정규화 포함
# 이런 정규화 유형은 순환에 적합하지 않음
# 레이어 정규화 연산을 위해 미니배치의 각 샘플의 평균과 표준편차를 별도로 계산
class LayerNorm(nn.Module):
    def __init__(self, features: int, eps: float = 1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps
# 정규화 공식에 필요한 감마와 베타를 미리 ones와 zeros로 생성
# beta는 bias이므로 0으로 시작하는게 적절
# eps는 표준편차가 0일 경우 수치 안정성을 

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta #주석으로 식 적진 못하지만 이거 그냥 계산만 한거!

#레지듀얼
#레지듀얼 커넥션은 네트워크에서 이전 레이어의 출력을 현재 레이어의 출력에 추가하는 것을 의미
#특정 레이어 건너뛰기가 가능해져 깊은 네트워크 허용
#각 레이어의 최종출력 = x + dropout(subLayer(LayerNorm(x)))
class ResidualConnection(nn.Module):
    def __init__(self, size: int = 512, dropout: float = .1):
        super(ResidualConnection,  self).__init__()
        self.norm = LayerNorm(size)
        # 필요한 드롭아웃 정의
        self.dropout = nn.Dropout(dropout)
# 역시 포워드에서 각 레이어의 최종출력 공식을 적용해줌
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

#피드 포워드
# 모든 어텐션 레이어의 위에 피드 포워드 네트워크 추가
# Relu활성화와 내부 레이어 드롭아웃을 통해 fully-connected레이어로 구성
# 입력 레이어 차원 : 512 vs 내부 레이어 차원 : 2048
class FeedForward(nn.Module):
    def __init__(self, d_model: int = 512, d_ff: int = 2048, dropout: float = .1):
        super(FeedForward, self).__init__()
        self.l1 = nn.Linear(d_model, d_ff)
        self.l2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU() #렐루 활성화에 필요
        self.dropout = nn.Dropout(dropout) # 내부 레이어 드롭아웃

    #feed-forward 구현
    def forward(self, x):
        return self.l2(self.dropout(self.relu(self.l1(x))))
    

#인코더 - 디코더
#앞에서 구현했던 각 층에서의 class를 활용하여
#인코더와 디코더의 역할에 맞게 순서를 조정해준다.
#이 과정이 헷갈릴 경우 전체 도식도를 파악하자
#self-attention과 attention이 어느 위치에서 발생하는지 확실히 인지하자!!!

#인코딩
# 인코더 레이어는 피드 포워드 네트워크가 뒤따르는 다중-헤드 어텐션 레이어
# 레지듀얼 커넥션 및 레이어 정규화 또한 포함
class EncoderLayer(nn.Module):
    def __init__(self, size: int, self_attn: MultiHeadAttention, feed_forward: FeedForward, dropout: float = .1):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sub1 = ResidualConnection(size, dropout)
        self.sub2 = ResidualConnection(size, dropout)
        self.size = size

    def forward(self, x, mask):
        x = self.sub1(x, lambda x: self.self_attn(x, x, x, mask))
        return self.sub2(x, self.feed_forward)

class Encoder(nn.Module):
    def __init__(self, layer, n: int = 6):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([deepcopy(layer) for _ in range(n)]) #n개만큼, 앞에서 사용한 테크닉
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

# 디코딩
# 디코딩 레이어는 메모리를 포함한 다중 헤드 어텐션 레이어가 뒤따르는 마스킹된 다중 헤드 어텐션 레이어
# 메모리는 인코더의 출력
# 마지막으로 피드 포워드 네트워크를 토오가
# 모든 구성요소는 앞에서 만들었던 레지듀얼 커넥션 및 레이어 정규화를 포함
from torch import nn
from copy import deepcopy
class DecoderLayer(nn.Module):
    def __init__(self, size: int, self_attn: MultiHeadAttention, src_attn: MultiHeadAttention, 
                 feed_forward: FeedForward, dropout: float = .1):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sub1 = ResidualConnection(size, dropout)
        self.sub2 = ResidualConnection(size, dropout)
        self.sub3 = ResidualConnection(size, dropout)
 
    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.sub1(x, lambda x: self.self_attn(x, x, x, tgt_mask)) #입력 이해
        x = self.sub2(x, lambda x: self.src_attn(x, memory, memory, src_mask)) # 입력 이해
        return self.sub3(x, self.feed_forward)

class Decoder(nn.Module):
    def __init__(self, layer: DecoderLayer, n: int = 6):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([deepcopy(layer) for _ in range(n)])
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

# 인코더-디코더
from torch import nn
class EncoderDecoder(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, 
                 src_embed: Embed, tgt_embed: Embed, final_layer: Output):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.final_layer = final_layer
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.final_layer(self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask))
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

# 최종 출력
class Output(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(Output, self).__init__()
        self.l1 = nn.Linear(input_dim, output_dim)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.l1(x)
        return self.log_softmax(logits)

# 모델 초기화
# 논문에서와 같이 동일한 차원으로 트랜스포머 모델을 구축
# 초기화 전략은 xavier/glorot 초기화이며 U[-1/n,-1/n] 범위의 균일 분포에서 선택
# 모든 bias는 0으로 초기화
def make_model(input_vocab: int, output_vocab: int, d_model: int = 512):
    encoder = Encoder(EncoderLayer(d_model, MultiHeadAttention(), FeedForward()))
    decoder = Decoder(DecoderLayer(d_model, MultiHeadAttention(), MultiHeadAttention(), FeedForward()))
    input_embed= nn.Sequential(Embed(vocab=input_vocab), PositionalEncoding())
    output_embed = nn.Sequential(Embed(vocab=output_vocab), PositionalEncoding())
    output = Output(input_dim=d_model, output_dim=output_vocab)
    model = EncoderDecoder(encoder, decoder, input_embed, output_embed, output)
    
    # Initialize parameters with Xavier uniform 
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

# 이함수는 seq2seq 문제에 대해 훈련할 수 있는 pytorch 모델 반환
# 토큰화된 입력 및 출력과 함께 사용하는 방법에 대한 예시 존재

# Tokenized symbols for source and target.
src = torch.tensor([[1, 2, 3, 4, 5]])
src_mask = torch.tensor([[1, 1, 1, 1, 1]])
tgt = torch.tensor([[6, 7, 8, 0, 0]])
tgt_mask = torch.tensor([[1, 1, 1, 0, 0]])

# Create PyTorch model
model = make_model(input_vocab=10, output_vocab=10)
# Do inference and take tokens with highest probability through argmax along the vocabulary axis (-1)
result = model(src, tgt, src_mask, tgt_mask)
result.argmax(dim=-1)

#결과 : tensor([[6, 6, 4, 3, 6]])

# 이 시점에서 모델은 균일학 ㅔ초기화된 가중치를 가지므로
# 출력은 타깃과의 거리는 상당히 멀다.
# 이러한 트랜스포머 모델을 처음부터 훈련하는 것은 몹시 오래 걸리지만
# 보통 사전 훈련된 트랜스포머 모델을 미세 조정하여 응용하낟.