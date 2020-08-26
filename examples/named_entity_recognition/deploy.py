import torch
from ratsnlp import nlpbook
from transformers import BertConfig, BertTokenizer
from ratsnlp.nlpbook.ner import ModelForNER, get_web_service_app


if __name__ == "__main__":

    # 학습이 완료된 모델 준비
    args = nlpbook.DeployArguments(
        pretrained_model_cache_dir="/Users/david/works/cache/kcbert-base",
        downstream_model_checkpoint_path="/Users/david/works/cache/checkpoint-ner/epoch=3.ckpt",
        downstream_model_labelmap_path="/Users/david/works/cache/checkpoint-ner/label_map.txt",
        downstream_task_name="named-entity-recognition",
        max_seq_length=128,
    )
    fine_tuned_model_ckpt = torch.load(
        args.downstream_model_checkpoint_path,
        map_location=torch.device("cpu")
    )
    labels = [label.strip() for label in open(args.downstream_model_labelmap_path, "r").readlines()]
    id_to_label = {}
    for idx, label in enumerate(labels):
        if "PER" in label:
            label = f"인명({label})"
        elif "LOC" in label:
            label = f"지명({label})"
        elif "ORG" in label:
            label = f"기관명({label})"
        elif "DAT" in label:
            label = f"날짜({label})"
        elif "TIM" in label:
            label = f"시간({label})"
        elif "DUR" in label:
            label = f"기간({label})"
        elif "MNY" in label:
            label = f"통화({label})"
        elif "PNT" in label:
            label = f"비율({label})"
        elif "NOH" in label:
            label = f"기타 수량표현({label})"
        elif "POH" in label:
            label = f"기타({label})"
        else:
            label = label
        id_to_label[idx] = label
    # 계산 그래프를 학습 때처럼 그려놓고,
    pretrained_model_config = BertConfig.from_pretrained(
        args.pretrained_model_cache_dir,
        num_labels=fine_tuned_model_ckpt['state_dict']['model.classifier.bias'].shape.numel(),
    )
    model = ModelForNER(pretrained_model_config)
    # 학습된 모델의 체크포인트를 해당 그래프에 부어넣는다
    model.load_state_dict({k.replace("model.", ""): v for k, v in fine_tuned_model_ckpt['state_dict'].items()})
    model.eval()
    tokenizer = BertTokenizer.from_pretrained(
        args.pretrained_model_cache_dir,
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

    app = get_web_service_app(inference_fn)
    app.run(host='0.0.0.0', port=5000)
