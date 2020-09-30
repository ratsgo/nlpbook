import torch
from transformers import BertConfig, BertTokenizer, BertModel

# define the tokenizer
tokenizer = BertTokenizer.from_pretrained(
    "beomi/kcbert-base",
    do_lower_case=False,
)

# define the pretrained model
pretrained_model_config = BertConfig.from_pretrained(
    "beomi/kcbert-base"
)
model = BertModel.from_pretrained(
    "beomi/kcbert-base",
    config=pretrained_model_config,
)

# make features
sentences = ["안녕하세요", "하이!"]
features = tokenizer(
    sentences,
    max_length=10,
    padding="max_length",
    truncation=True,
)
features = {k: torch.tensor(v) for k, v in features.items()}

# compute BERT embddings
outputs = model(**features)

# outputs[0] : word-level representation [2, 10, 768]
# tensor([[[-0.6969, -0.8248,  1.7512,  ..., -0.3732,  0.7399,  1.1907],
#          [-1.4803, -0.4398,  0.9444,  ..., -0.7405, -0.0211,  1.3064],
#          [-1.4299, -0.5033, -0.2069,  ...,  0.1285, -0.2611,  1.6057],
#          ...,
#          [-1.4406,  0.3431,  1.4043,  ..., -0.0565,  0.8450, -0.2170],
#          [-1.3625, -0.2404,  1.1757,  ...,  0.8876, -0.1054,  0.0734],
#          [-1.4244,  0.1518,  1.2920,  ...,  0.0245,  0.7572,  0.0080]],
#         [[ 0.9371, -1.4749,  1.7351,  ..., -0.3426,  0.8050,  0.4031],
#          [ 1.6095, -1.7269,  2.7936,  ...,  0.3100, -0.4787, -1.2491],
#          [ 0.4861, -0.4569,  0.5712,  ..., -0.1769,  1.1253, -0.2756],
#          ...,
#          [ 1.2362, -0.6181,  2.0906,  ...,  1.3677,  0.8132, -0.2742],
#          [ 0.5409, -0.9652,  1.6237,  ...,  1.2395,  0.9185,  0.1782],
#          [ 1.9001, -0.5859,  3.0156,  ...,  1.4967,  0.1924, -0.4448]]],
#        grad_fn=<NativeLayerNormBackward>)

# outputs[1] : sentence-level representation [2, 768]
# tensor([[-0.1594,  0.0547,  0.1101,  ...,  0.2684,  0.1596, -0.9828],
#         [-0.9221,  0.2969, -0.0110,  ...,  0.4291,  0.0311, -0.9955]],
#        grad_fn=<TanhBackward>)
