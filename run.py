import os, sys
from pytorch_lightning.cli import LightningCLI
from dataset.toy_2d import DATA_NAME


class Lightning_Run(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--mode", type=str, default="train", choices=["train", "test",])
        parser.add_argument("--data_name", type=str,
                            choices=DATA_NAME+["mnist"])
        parser.add_argument("--diffusion_type", type=str, choices=["ddpm", "sde", "discrete"])
        parser.add_argument("--discrete", type=bool, default=False)
        parser.add_argument("--load_train", type=bool)
        parser.add_argument("--ckpt_path", type=str)

    def before_instantiate_classes(self):
        if self.config['load_train'] or self.config['mode'] == "test":
            assert self.config['ckpt_path'] is not None, "ckpt_path must be specified"


if __name__ == "__main__":
    sys.path.append(os.getcwd()+"/Toy-Diffusion-Models")
    cli = Lightning_Run(save_config_overwrite=True, run=False,)

    if cli.config["mode"] == "train":
        if cli.config['load_train']:
            cli.trainer.fit(cli.model, cli.datamodule, ckpt_path=cli.config['ckpt_path'])
        else:
            cli.trainer.fit(cli.model, cli.datamodule)
    
    if cli.config["mode"] == "test":
        cli.trainer.test(cli.model, cli.datamodule, ckpt_path=cli.config['ckpt_path'])
