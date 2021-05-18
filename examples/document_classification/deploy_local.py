import sys
import torch
from ratsnlp.nlpbook import load_arguments
from ratsnlp.nlpbook.classification import ClassificationDeployArguments, get_web_service_app
from transformers import BertConfig, BertTokenizer, BertForSequenceClassification


if __name__ == "__main__":
    # case1 : python deploy_local.py
    if len(sys.argv) == 1:
        args = ClassificationDeployArguments(
            pretrained_model_name="beomi/kcbert-base",
            downstream_model_checkpoint_path="checkpoint/document-classification/epoch=10.ckpt",
            max_seq_length=128,
        )
    # case2 : python deploy_local.py deploy_config.json
    elif len(sys.argv) == 2 and sys.argv[-1].endswith(".json"):
        args = load_arguments(ClassificationDeployArguments, json_file_path=sys.argv[-1])
    # case3 : python deploy_local.py --pretrained_model_name beomi/kcbert-base --downstream_model_checkpoint_path checkpoint/document-classification/epoch=10.ckpt --downstream_task_name document-classification --max_seq_length 128
    else:
        args = load_arguments(ClassificationDeployArguments)

    fine_tuned_model_ckpt = torch.load(
        args.downstream_model_checkpoint_path,
        map_location=torch.device("cpu")
    )
    pretrained_model_config = BertConfig.from_pretrained(
        args.pretrained_model_name,
        num_labels=fine_tuned_model_ckpt['state_dict']['model.classifier.bias'].shape.numel(),
    )
    model = BertForSequenceClassification(pretrained_model_config)
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
            outputs = model(**{k: torch.tensor(v) for k, v in inputs.items()})
            prob = outputs.logits.softmax(dim=1)
            positive_prob = round(prob[0][1].item(), 4)
            negative_prob = round(prob[0][0].item(), 4)
            pred = "긍정 (positive)" if torch.argmax(prob) == 1 else "부정 (negative)"
        return {
            'sentence': sentence,
            'prediction': pred,
            'positive_data': f"긍정 {positive_prob}",
            'negative_data': f"부정 {negative_prob}",
            'positive_width': f"{positive_prob * 100}%",
            'negative_width': f"{negative_prob * 100}%",
        }

    app = get_web_service_app(inference_fn, is_colab=False)
    app.run()
