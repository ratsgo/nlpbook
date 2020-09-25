import torch
from ratsnlp import nlpbook
from google.colab import drive
from ratsnlp.nlpbook.classification import get_web_service_app
from transformers import BertConfig, BertTokenizer, BertForSequenceClassification


if __name__ == "__main__":
    drive.mount('/gdrive', force_remount=True)
    args = nlpbook.DeployArguments(
        pretrained_model_name="beomi/kcbert-base",
        downstream_model_checkpoint_path="/gdrive/My Drive/nlpbook/checkpoint-cls/_ckpt_epoch_0.ckpt",
        downstream_task_name="document-classification",
        max_seq_length=128,
    )
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
            logits, = model(**{k: torch.tensor(v) for k, v in inputs.items()})
            prob = logits.softmax(dim=1)
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

    app = get_web_service_app(inference_fn)
    app.run(host='0.0.0.0', port=5000)
