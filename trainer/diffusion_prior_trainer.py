import torch 




class Trainer1D(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        *,
        device=torch.device("cpu"),
        train_batch_size = 16,
        gradient_accumulate_every = 1,
        train_lr = 1e-4,
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 1000,
        num_samples = 25,
        results_folder = './results/sketch_cond',
        max_grad_norm = 1.,  
        amp=False  
    ):
        super().__init__()


        self.gt_data_path = 'data/cad_embeddings.h5'
        with h5py.File(self.gt_data_path, 'r') as f:
            self.gt_data = f["zs"][:]
        self.gt_latent = self._get_data(self.gt_data, gt=True)

        # model

        self.device = device
        self.model = diffusion_model.to(self.device)

        # sampling and training hyperparameters

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.max_grad_norm = max_grad_norm

        self.train_num_steps = train_num_steps

        self.train_tb = SummaryWriter(os.path.join(results_folder, 'train.events'))

        # dataset and dataloader        
        dl = DataLoader(dataset, batch_size = train_batch_size, shuffle = True, pin_memory = False, num_workers = cpu_count())
        self.dl = cycle(dl)


        # optimizer

        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)

        # for logging results in a folder periodically

        self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
        self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        # step counter state

        self.step = 0


        # Mixed Precision Setup
        self.amp = amp
        self.scaler = torch.amp.GradScaler("cuda", enabled = self.amp) if self.amp else None


    def _get_data(self, latent_data, gt=False):        
        pca = PCA(n_components=2)
        pca.fit(latent_data)
        latent_reduced = pca.transform(latent_data)

        return latent_reduced

    def save(self, milestone):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict()
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        device = self.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device)

        model = self.model
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        self.ema.load_state_dict(data["ema"])

    def _record_loss(self, loss):
        self.train_tb.add_scalar('loss', loss.item(), self.step)

    def run_validation(self, samples):

        latent_diff = self._get_data(samples)

        plt.figure()
        plt.scatter(self.gt_latent[:, 0], self.gt_latent[:, 1], s=0.5, color='gray', alpha=0.25, label='ground truth')
        plt.scatter(latent_diff[:, 0], latent_diff[:, 1], s=0.5, color='blue', alpha=0.75, label='generated')
        plt.legend(fontsize=14)
        plt.xlim(-2.5, 3)
        plt.ylim(-1.5, 2)
        plt.savefig(f'{self.results_folder}/samples.png')


    def train(self):

        with tqdm(initial = self.step, total = self.train_num_steps) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.


                for _ in range(self.gradient_accumulate_every):
                    batch = next(self.dl)

                    cad_emb, image_emb = batch[0].to(self.device), batch[1].to(self.device)

                    if self.amp:
                        with torch.amp.autocast("cuda", enabled = self.amp):
                            loss = self.model(cad_emb, cond=image_emb)
                            loss = loss / self.gradient_accumulate_every
                            total_loss += loss.item()

                    self.scaler.scale(loss).backward()
                    
                else: 
                    loss = self.model(cad_emb, cond=image_emb) / self.gradient_accumulate_every
                    total_loss += loss.item()
                    loss.backward()

                self._record_loss(loss)

                pbar.set_description(f'loss: {total_loss:.4f}')

                if self.amp:
                    self.scaler.unscale_(self.opt)

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                if self.amp:
                    self.scaler.step(self.opt)
                    self.scaler.update()
                else: 
                    self.opt.step()
    
                self.opt.zero_grad()

                self.step += 1

                if self.step != 0 and self.step % self.save_and_sample_every == 0:
                    self.ema.update()
                    self.ema.ema_model.eval()

                    batch = next(self.dl)

                    cad_emb, image_emb = batch[0].to(self.device), batch[1].to(self.device)

                    with torch.no_grad():
                        milestone = self.step // self.save_and_sample_every
                        sampled = self.ema.ema_model.sample(cond=image_emb)
                        with h5py.File(f'{self.results_folder}/davinci_samples_{milestone}.h5', 'w') as f:
                            f.create_dataset('zs', data=sampled.cpu().numpy())

                    print(sampled.size())

                    self.run_validation(sampled.cpu().numpy())

                    self.save(milestone)

                pbar.update(1)

        print('# # training complete # #')
