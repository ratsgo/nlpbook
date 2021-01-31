import sys
import torch
from ratsnlp.nlpbook import load_arguments
from ratsnlp.nlpbook.ner import NERDeployArguments, get_web_service_app
from transformers import BertConfig, BertTokenizer, BertForTokenClassification


if __name__ == "__main__":
    # case1 : python deploy_local.py
    if len(sys.argv) == 1:
        args = NERDeployArguments(
            pretrained_model_name="beomi/kcbert-base",
            downstream_model_checkpoint_path="checkpoint/ner/epoch=2.ckpt",
            downstream_model_labelmap_path="checkpoint/ner/label_map.txt",
            max_seq_length=64,
        )
    # case2 : python deploy_local.py deploy_config.json
    elif len(sys.argv) == 2 and sys.argv[-1].endswith(".json"):
        args = load_arguments(NERDeployArguments, json_file_path=sys.argv[-1])
    # case3 : python deploy_local.py --pretrained_model_name beomi/kcbert-base
    else:
        args = load_arguments(NERDeployArguments)
    fine_tuned_model_ckpt = torch.load(
        args.downstream_model_checkpoint_path,
        map_location=torch.device("cpu")
    )
    labels = [label.strip() for label in open(args.downstream_model_labelmap_path, "r").readlines()]
    id_to_label = {}
    for idx, label in enumerate(labels):
        if "PER" in label:
            label = "인명"
        elif "LOC" in label:
            label = "지명"
        elif "ORG" in label:
            label = "기관명"
        elif "DAT" in label:
            label = "날짜"
        elif "TIM" in label:
            label = "시간"
        elif "DUR" in label:
            label = "기간"
        elif "MNY" in label:
            label = "통화"
        elif "PNT" in label:
            label = "비율"
        elif "NOH" in label:
            label = "기타 수량표현"
        elif "POH" in label:
            label = "기타"
        else:
            label = label
        id_to_label[idx] = label
    pretrained_model_config = BertConfig.from_pretrained(
        args.pretrained_model_name,
        num_labels=fine_tuned_model_ckpt['state_dict']['model.classifier.bias'].shape.numel(),
    )
    model = BertForTokenClassification(pretrained_model_config)
    model.load_state_dict({k.replace("model.", ""): v for k, v in fine_tuned_model_ckpt['state_dict'].items()})
    model.eval()
    tokenizer = BertTokenizer.from_pretrained(
        args.pretrained_model_name,
        do_lower_case=False,
    )

    def inference_fn(sentence):
        inputs = tokenizer(
            [sentence],
            max_length=args.max_seq_length,
            padding="max_length",
            truncation=True,
        )
        with torch.no_grad():
            logits, = model(**{k: torch.tensor(v) for k, v in inputs.items()})
            probs = logits[0].softmax(dim=1)
            top_probs, preds = torch.topk(probs, dim=1, k=1)
            tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            predicted_tags = [id_to_label[pred.item()] for pred in preds]
            result = []
            for token, predicted_tag, top_prob in zip(tokens, predicted_tags, top_probs):
                if token not in [tokenizer.pad_token, tokenizer.cls_token, tokenizer.sep_token]:
                    token_result = {
                        "token": token,
                        "predicted_tag": predicted_tag,
                        "top_prob": str(round(top_prob[0].item(), 4)),
                    }
                    result.append(token_result)
        return {
            "sentence": sentence,
            "result": result,
        }

    app = get_web_service_app(inference_fn, is_colab=False)
    app.run()
