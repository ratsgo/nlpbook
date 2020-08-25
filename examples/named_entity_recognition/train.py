from ratsnlp import nlpbook
from transformers import BertConfig, BertTokenizer
from ratsnlp.nlpbook.ner import NERCorpus, NERDataset, NERModel, NERTask
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
        downstream_corpus_name="ner",
        downstream_corpus_dir="/Users/david/works/cache/ner",
        data_cache_dir="/Users/david/works/cache/ner",
        downstream_task_name="named-entity-recognition",
        downstream_model_dir="/Users/david/works/cache/checkpoint",
        do_train=True,
        do_eval=True,
        do_predict=True,
        batch_size=32,
        epochs=10,
        learning_rate=1e-6,
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
    corpus = NERCorpus(args)
    # text > input_ids, attention_mask, token_type_ids 형태 말고 추가, 삭제, 변경하려면 convert_fn을 별도로 짜서 넘기기
    train_dataset = NERDataset(
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
    val_dataset = NERDataset(
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
    test_dataset = NERDataset(
        args=args,
        corpus=corpus,
        tokenizer=tokenizer,
        mode="test",
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        sampler=SequentialSampler(test_dataset),
        collate_fn=nlpbook.data_collator,
        drop_last=False,
        num_workers=args.cpu_workers,
    )
    # huggingface PretrainedModel이기만 하면 됨, 원하는 모델로 교체해 사용 가능
    # 트랜스포머 말고 CNN 같은 모델 사용하려면 torch.nn.module로 만들기, 단 체크포인트가 사전에 읽혀져야 한다(torch.load)
    pretrained_model_config = BertConfig.from_pretrained(
        args.pretrained_model_cache_dir,
        num_labels=corpus.num_labels,
    )
    model = NERModel.from_pretrained(
            args.pretrained_model_cache_dir,
            config=pretrained_model_config,
    )
    # 파이토치 라이트닝 모듈 상속받아 커스텀하게 사용 가능
    # 책 본문에는 풀 텍스트로 설명하자
    task = NERTask(model, args)
    checkpoint_callback, trainer = nlpbook.get_trainer(args)
    if args.do_train:
        trainer.fit(
            task,
            train_dataloader=train_dataloader,
            val_dataloaders=val_dataloader,
        )
    if args.do_predict:
        trainer.test(
            task,
            test_dataloaders=test_dataloader,
            ckpt_path=checkpoint_callback.best_model_path,
        )
