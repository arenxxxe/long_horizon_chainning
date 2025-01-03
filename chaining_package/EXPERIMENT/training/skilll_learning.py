import hydra
from chaining_package.EXPERIMENT.training.trainer.my_trainer.trainers.sl_trainer import SkillLearningTrainer


@hydra.main(version_base=None, config_path="trainer/my_trainer/configs", config_name="skill_learning")
def main(cfg):
    exp = SkillLearningTrainer(cfg)

    exp.train()

#Nihao
if __name__ == "__main__":
    main()