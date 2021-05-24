import sys
import torch
from ratsnlp.nlpbook import load_arguments
from ratsnlp.nlpbook.qa import QADeployArguments, get_web_service_app
from transformers import BertConfig, BertTokenizer, BertForQuestionAnswering


if __name__ == "__main__":
    # case1 : python deploy_local.py
    if len(sys.argv) == 1:
        args = QADeployArguments(
            pretrained_model_name="beomi/kcbert-base",
            downstream_model_dir="checkpoint/question-answering/epoch=0.ckpt",
            max_seq_length=128,
            max_query_length=32,
        )
    # case2 : python deploy_local.py deploy_config.json
    elif len(sys.argv) == 2 and sys.argv[-1].endswith(".json"):
        args = load_arguments(QADeployArguments, json_file_path=sys.argv[-1])
    # case3 : python deploy_local.py --pretrained_model_name beomi/kcbert-base --downstream_model_checkpoint_path checkpoint/document-classification/epoch=10.ckpt --downstream_task_name document-classification --max_seq_length 128
    else:
        args = load_arguments(QADeployArguments)

    fine_tuned_model_ckpt = torch.load(
        args.downstream_model_checkpoint_fpath,
        map_location=torch.device("cpu")
    )
    pretrained_model_config = BertConfig.from_pretrained(args.pretrained_model_name)
    model = BertForQuestionAnswering(pretrained_model_config)
    model.load_state_dict({k.replace("model.", ""): v for k, v in fine_tuned_model_ckpt['state_dict'].items()})
    model.eval()
    tokenizer = BertTokenizer.from_pretrained(
        args.pretrained_model_name,
        do_lower_case=False,
    )

    def inference_fn(question, context):
        if question and context:
            truncated_query = tokenizer.encode(
                question,
                add_special_tokens=False,
                truncation=True,
                max_length=args.max_query_length
            )
            inputs = tokenizer.encode_plus(
                text=truncated_query,
                text_pair=context,
                truncation="only_second",
                padding="max_length",
                max_length=args.max_seq_length,
                return_token_type_ids=True,
            )
            with torch.no_grad():
                outputs = model(**{k: torch.tensor([v]) for k, v in inputs.items()})
                start_pred = outputs.start_logits.argmax(dim=-1).item()
                end_pred = outputs.end_logits.argmax(dim=-1).item()
                pred_text = tokenizer.decode(inputs['input_ids'][start_pred:end_pred+1])
        else:
            pred_text = ""
        return {
            'question': question,
            'context': context,
            'answer': pred_text,
        }

    app = get_web_service_app(inference_fn, is_colab=False)
    app.run()
