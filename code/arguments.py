import argparse
import pprint

# ----------------
import config.model_config

def load_args():

	parser = argparse.ArgumentParser(
		prog="MAML_CP_VAE", 
		description="Adversarial-Autoencoder-Text-Style-Transfer with MAML"
	)

	parser.add_argument(
		"--config-path", type=str, default='',
		help="path for model configuration"
	)
	parser.add_argument(
		"--corpus", type=str, default="s1",
		help="training corpus name"
	)
	parser.add_argument(
		"--load-data", action="store_true", 
		help="whether to load processed data"
	)
	parser.add_argument(
		"--load-model", action="store_true", 
		help="whether to load model from last checkpoint"
	)

	parser.add_argument(
		"--maml-batch-size", type=int, default=16,
		help="batch size for meta-learning"
	)
	parser.add_argument(
		"--sub-batch-size", type=int, default=64,
		help="batch size for sub-task training"
	)
	parser.add_argument(
		"--train-batch-size", type=int, default=64,
		help="batch size for fine-tuning"
	)

	parser.add_argument(
		"--maml-epochs", type=int, default=0,
		help="meta-training epochs"
	)
	parser.add_argument(
		"--transfer-epochs", type=int, default=0,
		help="fine-tuning epochs"
	)
	parser.add_argument(
		"--epochs-per-val", type=int, default=1,
		help="epochs per validation and checkpointing"
	)
	parser.add_argument(
		"--dump-embeddings", action="store_true",
		help="whether to dump content and style embeddings in validation"
	)
	parser.add_argument(
		"--task-id", type=str, default='',
		help="task for inference or extract embeddings, to disable training, set maml and transfer epochs to be 0"
	)
	parser.add_argument(
		"--extract-embeddings", action="store_true", 
		help="extract embeddings from saved checkpoint"
	)
	parser.add_argument(
		"--sample-size", type=int, default=1000,
		help="size of extracting embeddings"
	)
	parser.add_argument(
		"--from-pretrain", action="store_true",
		help="whether to reload ckpt from pretrain model path"
	)

	parser.add_argument(
		"--online-inference", action="store_true",
		help="whether to do online inference, suppressing other arguments"
	)
	parser.add_argument(
		"--ckpt", type=str, default="final",
		help="checkpoint to recover model, should be provided in online inference mode"
	)
	parser.add_argument(
		"--tgt-file", type=str, default="./tgt",
		help="file with sentences in target style for online inference"
	)

	# device
	parser.add_argument(
		"--disable-gpu", action="store_true",
		help="whether to disable GPU usage"
	)
	parser.add_argument(
		"--device-index", type=int, default=0,
		help="GPU device index to use"
	)


	args = parser.parse_args()

	return args


def build_mconf_from_args(args):

	mconf = config.model_config.MAMLModelConfig()

	for attr in vars(mconf).keys():
		if hasattr(args, attr):
			setattr(mconf, attr, getattr(args, attr))

	mconf.update_corpus()

	return mconf


if __name__ == "__main__":

	args = load_args()
	mconf = build_mconf_from_args(args)

	printer = pprint.PrettyPrinter(indent=4)
	printer.pprint(vars(args))
	printer.pprint(vars(mconf))

	
