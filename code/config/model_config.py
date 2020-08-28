class ModelConfig(object):

    def __init__(self):

        self.num_labels = 2

        self.betas = [0.5, 0.999]
        self.grad_clip = 30.

        self.adv_loss_tolerance = 7.

        self.gamma_init = 0.001
        self.gamma_min = 0.0001
        self.gamma_decay = 0.9

        # not including pad
        self.min_seq_length = 5
        self.max_seq_length = 15

        self.vocab_size = 5000
        self.bow_size = 5000
        self.vocab_cutoff = 3
        self.bow_cutoff = 3

        self.pad_id = 0
        self.eos_id = 1
        self.sos_id = 2
        self.unk_id = 3

        self.embedding_size = 300

        self.sequence_word_dropout = 0.1
        self.rnn_dropout = 0.1
        self.fc_dropout = 0.1

        self.kl_anneal_iterations = 20000
        self.eps = 1e-8

        # for cp-vae
        self.log_interval = 2000
        self.num_epochs = 100
        self.enc_lr = 1e-3
        self.dec_lr = 1.0
        self.warm_up = 10
        self.kl_start = 0.1
        self.beta1 = 0.35
        self.beta2 = 0.2
        self.srec_weight = 1.0
        self.reg_weight = 1.0
        self.ic_weight = 0.0
        self.aggressive = False
        self.lstm_ni = 256
        self.lstm_nh = 1024
        self.lstm_nz = 64
        self.mlp_ni = 300
        self.mlp_nz = 16
        self.dec_ni = 128
        self.dec_nh = 1024
        self.dec_dropout_in = 0.5
        self.dec_dropout_out = 0.5
        self.n_vars = 3

        # fine-tuning lr
        self.style_kl_weight = 0.03
        self.content_kl_weight = 0.03

        self.corpus = "s1"

        self.last_ckpt = "final"

        self.wordvec_path = None

        # --------------------------------------------------------------
        # requires corpus update

        self.model_save_dir_prefix = "../ckpt/{}/".format(self.corpus)
        self.emb_save_dir_prefix = "../emb/{}/".format(self.corpus)
        self.processed_data_save_dir_prefix = "../data/{}/processed/".format(
            self.corpus)
        self.data_dir_prefix = "../data/{}/".format(self.corpus)

        self.output_dir_prefix = "../output/{}/".format(self.corpus)

    def init_from_dict(self, config):

        for key in config:
            if hasattr(self, key):
                setattr(self, key, config[key])
        self.update_corpus()

    def update_corpus(self):

        self.model_save_dir_prefix = "../ckpt/{}/".format(self.corpus)
        self.emb_save_dir_prefix = "../emb/{}/".format(self.corpus)
        self.processed_data_save_dir_prefix = "../data/{}/processed/".format(
            self.corpus)
        self.data_dir_prefix = "../data/{}/".format(self.corpus)

        self.output_dir_prefix = "../output/{}/".format(self.corpus)


class MAMLModelConfig(ModelConfig):

    def __init__(self):

        super(MAMLModelConfig, self).__init__()
        # for maml_cp_vae
        self.meta_enc_lr = 1e-3
        self.meta_dec_lr = 1.0
        self.sub_enc_lr = 1e-3
        self.sub_dec_lr = 1.0

        self.num_updates = 2

        self.num_tasks = 7

        self.tsf_tasks = list(range(1, self.num_tasks+1))

        self.last_maml_ckpt = "maml"
        self.last_tsf_ckpts = dict(
            ("t{}".format(task_id), "tsf_t{}".format(task_id)) for task_id in self.tsf_tasks
        )

        # self.processed_data_save_dir_prefix = "../data/{}/processed/{}t/".format(self.corpus, self.num_tasks)

    '''
	def update_corpus(self):

		super(MAMLModelConfig, self).update_corpus()
		self.processed_data_save_dir_prefix = "../data/{}/processed/{}t/".format(self.corpus, self.num_tasks)
	'''


default_mconf = ModelConfig()
default_maml_mconf = MAMLModelConfig()
