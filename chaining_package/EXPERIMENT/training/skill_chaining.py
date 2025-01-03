import hydra
from chaining_package.EXPERIMENT.training.trainer.my_trainer.trainers.sc_trainer import SkillChainingTrainer


@hydra.main(version_base=None, config_path="trainer/my_trainer/configs", config_name="skill_chaining")
def main(cfg):
    exp = SkillChainingTrainer(cfg)
    exp.train()

if __name__ == "__main__":
    main()