import sys
import torch
from ratsnlp import nlpbook
from transformers import GPT2LMHeadModel, GPT2Config, PreTrainedTokenizerFast
from ratsnlp.nlpbook.generation import GenerationDeployArguments, get_web_service_app

if __name__ == "__main__":
    # case1 : python deploy_local.py
    if len(sys.argv) == 1:
        args = GenerationDeployArguments(
            pretrained_model_name="skt/kogpt2-base-v2",
            downstream_model_checkpoint_path="checkpoint/sentence-generation/epoch=0.ckpt",
        )
    # case2 : python deploy_local.py deploy_config.json
    elif len(sys.argv) == 2 and sys.argv[-1].endswith(".json"):
        args = nlpbook.load_arguments(GenerationDeployArguments, json_file_path=sys.argv[-1])
    # case3 : python deploy_local.py --pretrained_model_name beomi/kcbert-base --downstream_model_checkpoint_path checkpoint/document-classification/epoch=10.ckpt --downstream_task_name document-classification --max_seq_length 128
    else:
        args = nlpbook.load_arguments(GenerationDeployArguments)

    if args.downstream_model_checkpoint_path is None:
        model = GPT2LMHeadModel.from_pretrained(
            args.pretrained_model_name,
        )
    else:
        pretrained_model_config = GPT2Config.from_pretrained(
            args.pretrained_model_name,
        )
        model = GPT2LMHeadModel(pretrained_model_config)
        fine_tuned_model_ckpt = torch.load(
            args.downstream_model_checkpoint_path,
            map_location=torch.device("cpu")
        )
        model.load_state_dict({k.replace("model.", ""): v for k, v in fine_tuned_model_ckpt['state_dict'].items()})
    model.eval()
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        args.pretrained_model_name,
        eos_token='</s>',
    )

    def inference_fn(
            prompt,
            min_length=10,
            max_length=20,
            top_p=1.0,
            top_k=50,
            repetition_penalty=1.0,
            no_repeat_ngram_size=3,
            temperature=1.0,
    ):
        try:
            input_ids = tokenizer.encode(prompt, return_tensors="pt")
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids,
                    do_sample=True,
                    top_p=float(top_p),
                    top_k=int(top_k),
                    min_length=int(min_length),
                    max_length=int(max_length),
                    repetition_penalty=float(repetition_penalty),
                    no_repeat_ngram_size=int(no_repeat_ngram_size),
                    temperature=float(temperature),
                )
            generated_sentence = tokenizer.decode([el.item() for el in generated_ids[0]])
        except:
            generated_sentence = """처리 중 오류가 발생했습니다. <br>
                변수의 입력 범위를 확인하세요. <br><br> 
                생성 길이: 1 이상의 정수 <br>
                top-p: 0 이상 1 이하의 실수 <br>
                top-k: 1 이상의 정수 <br>
                repetition penalty: 1 이상의 실수 <br>
                temperature: 0 이상의 실수
                """
        return {
            'result': generated_sentence,
        }

    app = get_web_service_app(inference_fn, is_colab=False)
    app.run()
