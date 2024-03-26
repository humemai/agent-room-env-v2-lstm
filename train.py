import matplotlib

matplotlib.use("Agg")
import argparse
import logging

logger = logging.getLogger()
logger.disabled = True

from humemai.utils import read_yaml

from agent.dqn import DQNExploreAgent, DQNLSTMBaselineAgent, DQNMMAgent

parser = argparse.ArgumentParser(description="train")
parser.add_argument(
    "-c",
    "--config",
    default="./train.yaml",
    type=str,
    help="config file path (default: ./train.yaml)",
)
args = parser.parse_args()
hparams = read_yaml(args.config)
print("Arguments:")
for k, v in hparams.items():
    print(f"  {k:>21} : {v}")


agent = DQNLSTMBaselineAgent(**hparams)
agent.train()
