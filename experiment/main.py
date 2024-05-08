import hydra
from trainer.skill_trainer import SkillLearningTrainer


@hydra.main(version_base=None, config_path="./configs", config_name="skill_learning")
def main(cfg):
    #写一个trainer
    exp = SkillLearningTrainer(cfg)
    exp.train()

if __name__ == "__main__":
    main()