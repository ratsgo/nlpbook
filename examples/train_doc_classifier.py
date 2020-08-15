from ratsnlp import nlpbook
from ratsnlp.nlpbook.classification import NsmcCorpus, Runner
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification


if __name__ == "__main__":
    # 파이썬 3.7 이상 (정확히는 python 3.7.3)
    # pip install wheel
    # pip install ratsnlp
    # 원하는 arguments로 변경 가능
    # dataclass만 정의하면 됨
    # colab에서 args 읽어들이기
    args = nlpbook.TrainArguments(
        pretrained_model_name="kcbert-base",
        pretrained_model_cache_dir="/mnt/sdb1/david/ratsnlp/content/kcbert-base",
        downstream_corpus_name="nsmc",
        downstream_corpus_dir="/mnt/sdb1/david/ratsnlp/content/nsmc",
        downstream_task_name="document-classification",
        downstream_model_dir="/mnt/sdb1/david/ratsnlp/content/checkpoint",
        do_train=True,
        do_eval=True,
        do_predict=False,
        batch_size=32,
    )
    # json 파일로부터 args 읽어들이기
    # args = load_arguments(Arguments, json_file_path="examples/document_classification.json")
    # python train_doc_classifier.py --pretrained_model_name kcbert-base .. 이렇게 읽어들이기
    # args = load_arguments(Arguments)
    nlpbook.set_logger(args)
    # 이미 데이터가 준비되어 있다면 생략 가능
    nlpbook.download_downstream_dataset(args)
    # 이미 모델이 준비되어 있다면 생략 가능
    nlpbook.download_pretrained_model(args)
    nlpbook.check_exist_checkpoints(args)
    nlpbook.seed_setting(args)
    # huggingface PretrainedTokenizer이기만 하면 됨
    # 원하는 토크나이저로 교체해 사용 가능 (vocab 교체만 설명하자)
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_cache_dir,
        do_lower_case=False,
    )
    # 데이터, 트레이너 커스터마이즈는 고급 지식으로 적어두자
    # from ratsnlp.nlpbook.dataset import Corpus
    # 위의 클래스를 상속받아 커스텀하게 사용 가능
    corpus = NsmcCorpus()
    # text > input_ids, attention_mask, token_type_ids 형태 말고 추가, 삭제, 변경하려면...
    # Dataset 정의 : 핵심은 init에 전체 데이터셋을 ids로 변환해 불러오고, getitem으로 인스턴스 하나씩만 넘겨주면 됨
    # Dataset, collate_fn만 사용자 정의하면 됨
    # 이제 나머지는 datasampler 등과 엮어서 DataLoader로 묶어주기
    train_dataloader, val_dataloader, test_dataloader = nlpbook.get_dataloaders(args, tokenizer, corpus)
    # huggingface PretrainedModel이기만 하면 됨, 원하는 모델로 교체해 사용 가능
    # 트랜스포머 말고 CNN 같은 모델 사용하려면 torch.nn.module로 만들기, 단 체크포인트가 사전에 읽혀져야 한다(torch.load)
    pretrained_model_config = AutoConfig.from_pretrained(
        args.pretrained_model_cache_dir,
        num_labels=corpus.num_labels,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
            args.pretrained_model_cache_dir,
            config=pretrained_model_config,
    )
    # 파이토치 라이트닝 모듈 상속받아 커스텀하게 사용 가능
    # 책 본문에는 풀 텍스트로 설명하자
    runner = Runner(model, args)
    checkpoint_callback, trainer = nlpbook.get_trainer(args)
    if args.do_train:
        trainer.fit(
            runner,
            train_dataloader=train_dataloader,
            val_dataloaders=val_dataloader,
        )
    if args.do_predict:
        trainer.test(
            runner,
            test_dataloaders=test_dataloader,
            ckpt_path=checkpoint_callback.best_model_path,
        )
