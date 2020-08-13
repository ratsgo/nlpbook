from ratsnlp.nlpbook import *
from ratsnlp.nlpbook.classification import NsmcCorpus, Runner


if __name__ == "__main__":
    args = load_arguments(json_file_path="examples/document_classification.json")
    # args = load_arguments()
    set_logger(args)
    download_downstream_dataset(
        args.downstream_corpus_name,
        cache_dir=args.downstream_corpus_dir,
        force_download=False
    )
    download_pretrained_model(
        args.pretrained_model_name,
        cache_dir=args.pretrained_model_cache_dir,
        force_download=False
    )
    check_exist_checkpoints(args)
    seed_setting(args)
    tokenizer = get_tokenizer(args)
    corpus = NsmcCorpus()
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(corpus, tokenizer, args)
    model = get_pretrained_model(args, num_labels=2)
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
