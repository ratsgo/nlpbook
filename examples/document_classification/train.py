from ratsnlp import nlpbook
from ratsnlp.nlpbook.classification import NsmcCorpus, ClassificationDataset, ClassificationTask
from transformers import BertConfig, BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler


if __name__ == "__main__":
    # 파이썬 3.7 이상 (정확히는 python 3.7.3)
    # pip install wheel
    # pip install ratsnlp
    # 원하는 arguments로 변경 가능
    # dataclass만 정의하면 됨
    # colab에서 args 읽어들이기
    args = nlpbook.TrainArguments(
        pretrained_model_name="kcbert-base",
        pretrained_model_cache_dir="/Users/david/works/cache/kcbert-base",
        downstream_corpus_name="nsmc",
        downstream_corpus_dir="/Users/david/works/cache/nsmc",
        data_cache_dir="/Users/david/works/cache/nsmc",
        downstream_task_name="document-classification",
        downstream_model_dir="/Users/david/works/cache/checkpoint-cls",
        do_train=True,
        do_eval=True,
        batch_size=32,
    )
    # json 파일로부터 args 읽어들이기
    # args = load_arguments(Arguments, json_file_path="examples/document_classification.json")
    # python train.py --pretrained_model_name kcbert-base .. 이렇게 읽어들이기
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
    tokenizer = BertTokenizer.from_pretrained(
        args.pretrained_model_cache_dir,
        do_lower_case=False,
    )
    # 데이터, 트레이너 커스터마이즈는 고급 지식으로 적어두자
    # 코퍼스 커스텀하게 변경 가능
    corpus = NsmcCorpus()
    # text > input_ids, attention_mask, token_type_ids 형태 말고 추가, 삭제, 변경하려면 convert_fn을 별도로 짜서 넘기기
    train_dataset = ClassificationDataset(
        args=args,
        corpus=corpus,
        tokenizer=tokenizer,
        mode="train",
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=RandomSampler(train_dataset, replacement=False),
        collate_fn=nlpbook.data_collator,
        drop_last=False,
        num_workers=args.cpu_workers,
    )
    # 여기는 초반 장이므로 val 개념만 설명하고 test 데이터셋은 나중에 설명
    if args.do_eval:
        val_dataset = ClassificationDataset(
            args=args,
            corpus=corpus,
            tokenizer=tokenizer,
            mode="val",
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            sampler=SequentialSampler(val_dataset),
            collate_fn=nlpbook.data_collator,
            drop_last=False,
            num_workers=args.cpu_workers,
        )
    else:
        val_dataloader = None
    # huggingface PretrainedModel이기만 하면 됨, 원하는 모델로 교체해 사용 가능
    # 트랜스포머 말고 CNN 같은 모델 사용하려면 torch.nn.module로 만들기, 단 체크포인트가 사전에 읽혀져야 한다(torch.load)
    pretrained_model_config = BertConfig.from_pretrained(
        args.pretrained_model_cache_dir,
        num_labels=corpus.num_labels,
    )
    model = BertForSequenceClassification.from_pretrained(
            args.pretrained_model_cache_dir,
            config=pretrained_model_config,
    )
    # 파이토치 라이트닝 모듈 상속받아 커스텀하게 사용 가능
    # 책 본문에는 풀 텍스트로 설명하자
    task = ClassificationTask(model, args)
    # 여기는 초반 장이므로 checkpoint_callback은 존재만 간단히 언급하고 trainer만 설명
    _, trainer = nlpbook.get_trainer(args)
    if args.do_train:
        trainer.fit(
            task,
            train_dataloader=train_dataloader,
            val_dataloaders=val_dataloader,
        )
