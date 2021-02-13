import torch
from transformers import GPT2LMHeadModel, GPT2Config
from ratsnlp.nlpbook.tokenizers import KoGPT2Tokenizer

# initialize model and tokenizer
pretrained_model_config = GPT2Config.from_pretrained("taeminlee/kogpt2")
model = GPT2LMHeadModel(pretrained_model_config)
fine_tuned_model_ckpt = torch.load(
    "/Users/david/Downloads/epoch=5.ckpt",
    map_location=torch.device("cpu"),
)
model.load_state_dict({k.replace("model.", ""): v for k, v in fine_tuned_model_ckpt['state_dict'].items()})
tokenizer = KoGPT2Tokenizer.from_pretrained("taeminlee/kogpt2")
input_ids = tokenizer.encode("부정 와 정말", return_tensors="pt")

# greedy decoding
# best1 선택, 생성할 때마다 같은 문장이 나온다
generated_ids = model.generate(
    input_ids,
    do_sample=False,
    max_length=50,
)
# 안녕하세요</s><s> 나 지금 집에 가고 있어</s><s> 나 지금 집에 가고 있어</s>...
tokenizer.decode([el.item() for el in generated_ids[0]])

# greedy decoding + repetition penalty
# 반복을 방지하지만 greedy이기 때문에 생성할 때마다 같은 문장이 나온다
generated_ids = model.generate(
    input_ids,
    do_sample=False,
    max_length=50,
    repetition_penalty=1.2,
)
# 안녕하세요</s><s> 나 지금 집에 가고 있어.</s><s> 잘 자요 내 사랑</s><s> 사랑해</s>...
tokenizer.decode([el.item() for el in generated_ids[0]])

# beam search without sampling
# beam size만큼의 candidate 가운데 최대 확률을 내는 시퀀스를 찾는다 (deteministic)
# 생성할 때마다 같은 문장이 나온다
generated_ids = model.generate(
    input_ids,
    do_sample=False,
    max_length=50,
    num_beams=10,
)
# 안녕하세요"라는 글과 함께 한 장의 사진을 게재했다.</s><s> 한편 이날...
tokenizer.decode([el.item() for el in generated_ids[0]])

# beam search with sampling
generated_ids = model.generate(
    input_ids,
    do_sample=True,
    max_length=50,
    num_beams=10,
)
tokenizer.decode([el.item() for el in generated_ids[0]])

# beam search with sampling + length penalty
# length penalty는 beam search일 때 동작한다
generated_ids = model.generate(
    input_ids,
    do_sample=True,
    max_length=50,
    num_beams=10,
    length_penalty=1.5,
)
# 안녕하세요</s><s> 나 지금 집에 가고 있어.</s><s> 잘 자요 내 사랑</s><s> 사랑해</s>...
tokenizer.decode([el.item() for el in generated_ids[0]])

# top-p sampling
# p는 0~1 범위, sampling을 한다고는 하지만 p가 0에 가까울 수록 greedy decoding
# do_sample이 False일 경우 top_p가 작동하지 않는다
generated_ids = model.generate(
    input_ids,
    do_sample=True,
    top_p=0.0001,
    max_length=50,
)
# 안녕하세요</s><s> 나 지금 집에 가고 있어</s><s> 나 지금 집에 가고 있어</s>...
tokenizer.decode([el.item() for el in generated_ids[0]])

# top-p sampling
# p가 1에 가깝다면 모델 출력 분포를 가공 없이 그대로 사용
# do_sample이 False일 경우 top_p가 작동하지 않는다
generated_ids = model.generate(
    input_ids,
    do_sample=True,
    top_p=0.9999,
    max_length=50,
)
# 모델 출력 분포에서 다음 단어를 샘플하는 것이기 때문에 생성할 때마다 다른 문장이 나온다
tokenizer.decode([el.item() for el in generated_ids[0]])

# top-k sampling
# k가 1이라면 greedy decoding과 동일
generated_ids = model.generate(
    input_ids,
    do_sample=True,
    top_k=1,
    max_length=50,
)
tokenizer.decode([el.item() for el in generated_ids[0]])

# k가 커질수록 다양성이 커진다
generated_ids = model.generate(
    input_ids,
    do_sample=True,
    top_k=100,
    max_length=50,
)
tokenizer.decode([el.item() for el in generated_ids[0]])

# temperature sampling
# t는 0에서 inf 범위
# t가 0에 가까워질 수록 토큰 분포가 sharp해진다 > 1등 토큰이 뽑힐 확률이 그만큼 높아진다 > 사실상 greedy decoding이 된다
generated_ids = model.generate(
    input_ids,
    do_sample=True,
    temperature=0.0001,
    max_length=50,
)
# 안녕하세요</s><s> 나 지금 집에 가고 있어</s><s> 나 지금 집에 가고 있어</s>...
tokenizer.decode([el.item() for el in generated_ids[0]])

# temperature가 1이라면 모델 출력 분포를 가공 없이 그대로 사용
generated_ids = model.generate(
    input_ids,
    do_sample=True,
    temperature=1,
    max_length=50,
)
# 모델 출력 분포에서 다음 단어를 샘플하는 것이기 때문에 생성할 때마다 다른 문장이 나온다
tokenizer.decode([el.item() for el in generated_ids[0]])

# temperature가 클수록 uniform > 그만큼 다양한 문장이 생성된다 (정확도는 낮아짐)
generated_ids = model.generate(
    input_ids,
    do_sample=True,
    temperature=100,
    max_length=50,
)
# 생성할 때마다 다른 문장이 나온다
tokenizer.decode([el.item() for el in generated_ids[0]])
