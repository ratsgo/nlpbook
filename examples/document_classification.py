from ratsnlp.nlpbook import *
from ratsnlp.nlpbook.classification import NsmcCorpus, Runner


if __name__ == "__main__":
    # 원하는 arguments로 변경 가능
    # dataclass만 정의하면 됨
    args = load_arguments(Arguments, json_file_path="examples/document_classification.json")
    # args = load_arguments(Arguments)
    set_logger(args)
    # 이미 데이터가 준비되어 있다면 생략 가능
    download_downstream_dataset(args)
    # 이미 모델이 준비되어 있다면 생략 가능
    download_pretrained_model(args)
    check_exist_checkpoints(args)
    seed_setting(args)
    # huggingface PretrainedTokenizer이기만 하면 됨
    # 원하는 토크나이저로 교체해 사용 가능
    tokenizer = get_tokenizer(args)
    # from ratsnlp.nlpbook.dataset import Corpus
    # 위의 클래스를 상속받아 커스텀하게 사용 가능
    corpus = NsmcCorpus()
    # text > input_ids, attention_mask, token_type_ids 형태 말고 추가, 삭제, 변경하려면...
    # Dataset 정의 : 핵심은 init에 전체 데이터셋을 ids로 변환해 불러오고, getitem으로 인스턴스 하나씩만 넘겨주면 됨
    # Dataset, collate_fn만 사용자 정의하면 됨
    # 이제 나머지는 datasampler 등과 엮어서 DataLoader로 묶어주기
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(args, tokenizer, corpus)
    # huggingface PretrainedModel이기만 하면 됨
    # 원하는 모델로 교체해 사용 가능, 단 체크포인트가 사전에 읽혀져야 한다(torch.load)
    model = get_pretrained_model(args, num_labels=corpus.num_labels)
    # 파이토치 라이트닝 모듈 상속받아 커스텀하게 사용 가능
    runner = Runner(model, args)
    checkpoint_callback, trainer = get_trainer(args)
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
