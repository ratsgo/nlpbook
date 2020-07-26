from ratsnlp import *


if __name__ == "__main__":
    args = load_arguments(json_file_path="examples/document_classification.json")
    # args = load_arguments()
    pretrained_model_args, downstream_data_args, fine_tuning_args = args
    set_logger(fine_tuning_args)
    download_downstream_dataset(downstream_data_args.corpus_name,
                                cache_dir=downstream_data_args.corpus_dir,
                                force_download=False)
    download_pretrained_model(pretrained_model_args.pretrained_model_name,
                              cache_dir=pretrained_model_args.pretrained_model_cache_dir,
                              force_download=False)
    check_exist_checkpoints(fine_tuning_args)
    seed_setting(fine_tuning_args.seed)
    tokenizer = get_tokenizer(pretrained_model_args, output_dir=fine_tuning_args.output_dir)
    corpus_class = CORPUS_FACTORY[downstream_data_args.corpus_name]()
    # corpus_class = NsmcCorpus() # Custom Class 제작 가능
    train_dataset, eval_dataset, test_dataset = get_datasets(corpus_class, tokenizer, args)
    pretrained_model_config = get_pretrained_model_config(pretrained_model_args,
                                                          task_name=downstream_data_args.task_name,
                                                          num_labels=2,
                                                          output_dir=fine_tuning_args.output_dir)
    model = get_pretrained_model(pretrained_model_args.pretrained_model_cache_dir,
                                 pretrained_model_config,
                                 output_dir=fine_tuning_args.output_dir)
    trainer = Trainer(
        model=model,
        args=fine_tuning_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=build_compute_metrics_fn(downstream_data_args.task_name),
    )
    trainer.fit()