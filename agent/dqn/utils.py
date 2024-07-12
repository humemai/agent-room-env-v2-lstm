"""Utility functions for DQN."""

import os
import shutil
from typing import Literal

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from humemai.utils import argmax, is_running_notebook, list_duplicates_of, write_yaml
from IPython.display import clear_output
from tqdm.auto import tqdm

from ..utils import write_pickle


class ReplayBuffer:
    """A simple numpy replay buffer.

    numpy replay buffer is faster than deque or list.
    copied from https://github.com/Curt-Park/rainbow-is-all-you-need

    """

    def __init__(
        self,
        observation_type: Literal["dict", "tensor"],
        size: int,
        batch_size: int = 32,
    ):
        """Initialize replay buffer.

        Args:
            observation_type: "dict" or "tensor"
            size: size of the buffer
            batch_size: batch size to sample

        """
        if batch_size > size:
            raise ValueError("batch_size must be smaller than size")
        if observation_type == "dict":
            self.obs_buf = np.array([{}] * size)
            self.next_obs_buf = np.array([{}] * size)
        else:
            raise ValueError("At the moment, observation_type must be 'dict'")

        self.acts_buf = np.zeros([size], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        (
            self.ptr,
            self.size,
        ) = (
            0,
            0,
        )

    def store(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: float,
        next_obs: np.ndarray,
        done: bool,
    ):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self) -> dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(
            obs=self.obs_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            done=self.done_buf[idxs],
        )

    def __len__(self) -> int:
        return self.size


def plot_results(
    policy: Literal["mm", "explore"],
    scores: dict,
    training_loss: list,
    epsilons: list,
    q_values: dict,
    iteration_idx: int,
    number_of_actions: int,
    num_iterations: int,
    total_maximum_episode_rewards: int,
    default_root_dir: str,
    to_plot: str = "all",
    save_fig: bool = False,
) -> None:
    """Plot things for DQN training.

    Args:
        to_plot: what to plot:
            training_td_loss
            epsilons
            scores
            q_values_train
            q_values_val
            q_values_test

    """
    is_notebook = is_running_notebook()

    if is_notebook:
        clear_output(True)

    if to_plot == "all":
        plt.figure(figsize=(20, 13))

        plt.subplot(233)
        if scores["train"]:
            plt.title(
                f"iteration {iteration_idx} out of {num_iterations}. "
                f"training score: {scores['train'][-1]} out of {total_maximum_episode_rewards}"
            )
            plt.plot(scores["train"], label="Training score")
            plt.xlabel("episode")

        if scores["val"]:
            val_means = [round(np.mean(scores).item()) for scores in scores["val"]]
            plt.title(
                f"validation score: {val_means[-1]} out of {total_maximum_episode_rewards}"
            )
            plt.plot(val_means, label="Validation score")
            plt.xlabel("episode")

        if scores["test"]:
            plt.title(
                f"test score: {np.mean(scores['test'])} out of {total_maximum_episode_rewards}"
            )
            plt.plot(
                [round(np.mean(scores["test"]).item(), 2)] * len(scores["train"]),
                label="Test score",
            )
            plt.xlabel("episode")
        plt.legend(loc="upper left")

        plt.subplot(231)
        plt.title("training td loss")
        plt.plot(training_loss)
        plt.xlabel("update counts")

        plt.subplot(232)
        plt.title("epsilons")
        plt.plot(epsilons)
        plt.xlabel("update counts")

        plt.subplot(234)
        plt.title("Q-values, train")
        for action_number in range(number_of_actions):
            plt.plot(
                [q_values_[action_number] for q_values_ in q_values["train"]],
                label=f"action {action_number}",
            )
        plt.legend(loc="upper left")
        plt.xlabel("number of actions taken")

        plt.subplot(235)
        plt.title("Q-values, val")
        for action_number in range(number_of_actions):
            plt.plot(
                [q_values_[action_number] for q_values_ in q_values["val"]],
                label=f"action {action_number}",
            )
        plt.legend(loc="upper left")
        plt.xlabel("number of actions taken")

        plt.subplot(236)
        plt.title("Q-values, test")
        for action_number in range(number_of_actions):
            plt.plot(
                [q_values_[action_number] for q_values_ in q_values["test"]],
                label=f"action {action_number}",
            )
        plt.legend(loc="upper left")
        plt.xlabel("number of actions taken")

        plt.subplots_adjust(hspace=0.5)
        if save_fig:

            subdir = ""
            if policy is not None:
                subdir = policy

            plt.savefig(os.path.join(default_root_dir, subdir, "plot.pdf"))

        if is_notebook:
            plt.show()
        else:
            console(**locals())
            plt.close("all")

    elif to_plot == "training_td_loss":
        plt.figure()
        plt.title("training td loss")
        plt.plot(training_loss)
        plt.xlabel("update counts")

    elif to_plot == "epsilons":
        plt.figure()
        plt.title("epsilons")
        plt.plot(epsilons)
        plt.xlabel("update counts")

    elif to_plot == "scores":
        plt.figure()

        if scores["train"]:
            plt.title(
                f"iteration {iteration_idx} out of {num_iterations}. "
                f"training score: {scores['train'][-1]} out of {total_maximum_episode_rewards}"
            )
            plt.plot(scores["train"], label="Training score")
            plt.xlabel("episode")

        if scores["val"]:
            val_means = [round(np.mean(scores).item()) for scores in scores["val"]]
            plt.title(
                f"validation score: {val_means[-1]} out of {total_maximum_episode_rewards}"
            )
            plt.plot(val_means, label="Validation score")
            plt.xlabel("episode")

        if scores["test"]:
            plt.title(
                f"test score: {np.mean(scores['test'])} out of {total_maximum_episode_rewards}"
            )
            plt.plot(
                [round(np.mean(scores["test"]).item(), 2)] * len(scores["train"]),
                label="Test score",
            )
            plt.xlabel("episode")
        plt.legend(loc="upper left")

    elif to_plot == "q_values_train":
        plt.figure()
        plt.title("Q-values, train")
        for action_number in range(number_of_actions):
            plt.plot(
                [q_values_[action_number] for q_values_ in q_values["train"]],
                label=f"action {action_number}",
            )
        plt.legend(loc="upper left")
        plt.xlabel("number of actions taken")

    elif to_plot == "q_values_val":
        plt.figure()
        plt.title("Q-values, val")
        for action_number in range(number_of_actions):
            plt.plot(
                [q_values_[action_number] for q_values_ in q_values["val"]],
                label=f"action {action_number}",
            )
        plt.legend(loc="upper left")
        plt.xlabel("number of actions taken")

    elif to_plot == "q_values_test":
        plt.figure()
        plt.title("Q-values, test")
        for action_number in range(number_of_actions):
            plt.plot(
                [q_values_[action_number] for q_values_ in q_values["test"]],
                label=f"action {action_number}",
            )
        plt.legend(loc="upper left")
        plt.xlabel("number of actions taken")
    else:
        raise ValueError(f"to_plot={to_plot} is not valid.")


def console(
    scores: dict,
    training_loss: list,
    iteration_idx: int,
    num_iterations: int,
    total_maximum_episode_rewards: int,
    **kwargs,
) -> None:
    """Print the dqn training to the console."""
    if scores["train"]:
        tqdm.write(
            f"iteration {iteration_idx} out of {num_iterations}.\n"
            f"training score: "
            f"{scores['train'][-1]} out of {total_maximum_episode_rewards}"
        )

    if scores["val"]:
        val_means = [round(np.mean(scores).item()) for scores in scores["val"]]
        tqdm.write(
            f"validation score: {val_means[-1]} "
            f"out of {total_maximum_episode_rewards}"
        )

    if scores["test"]:
        tqdm.write(
            f"test score: {np.mean(scores['test'])} out of {total_maximum_episode_rewards}"
        )

    tqdm.write(f"training loss: {training_loss[-1]}\n")
    print()


def save_final_results(
    scores: dict,
    training_loss: list,
    default_root_dir: str,
    q_values: dict,
    self: object,
    policy: Literal["mm", "explore", None],
    save_the_agent: bool = False,
) -> None:
    """Save dqn train / val / test results.

    Args:
        scores: a dictionary of scores for train, validation, and test.
        training_loss: a list of training loss values.
        default_root_dir: the root directory where the results are saved.
        q_values: a dictionary of q_values for train, validation, and test.
        self: the agent object.
        policy: "mm" or "explore"
        save_the_agent: whether to save the agent or not.

    """
    results = {
        "train_score": scores["train"],
        "validation_score": [
            {
                "mean": round(np.mean(scores).item(), 2),
                "std": round(np.std(scores).item(), 2),
            }
            for scores in scores["val"]
        ],
        "test_score": {
            "mean": round(np.mean(scores["test"]).item(), 2),
            "std": round(np.std(scores["test"]).item(), 2),
        },
        "training_loss": training_loss,
    }
    subdir = policy if policy is not None else ""

    write_yaml(results, os.path.join(default_root_dir, subdir, "results.yaml"))
    write_yaml(q_values, os.path.join(default_root_dir, subdir, "q_values.yaml"))

    if save_the_agent:
        write_pickle(self, os.path.join(default_root_dir, "agent.pkl"))


def compute_loss(
    memory_types: list[str],
    samples: dict[str, np.ndarray],
    device: str,
    lstm: torch.nn.Module,
    lstm_target: torch.nn.Module,
    mlp: torch.nn.Module,
    mlp_target: torch.nn.Module,
    ddqn: str,
    gamma: float,
) -> torch.Tensor:
    """Return td loss.

    Args:
        memory_types: memory_types
        samples: A dictionary of samples from the replay buffer.
            obs: np.ndarray,
            act: np.ndarray,
            rew: float,
            next_obs: np.ndarray,
            done: bool,
        device: cpu or cuda
        lstm: lstm model
        lstm_target: lstm target model
        mlp: mlp model
        mlp_target: mlp target model
        ddqn: whether to use double dqn or not
        gamma: discount factor

    Returns:
        loss: torch.Tensor

    """
    state = samples["obs"]
    next_state = samples["next_obs"]
    action = torch.LongTensor(samples["acts"].reshape(-1, 1)).to(device)
    reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
    done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

    # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
    #       = r                       otherwise
    curr_q_value = mlp(lstm(state, memory_types)[0]).gather(1, action)
    if ddqn:
        next_q_value = (
            mlp_target(lstm_target(next_state, memory_types)[0])
            .gather(
                1, mlp(lstm(next_state, memory_types)[0]).argmax(dim=1, keepdim=True)
            )
            .detach()
        )
    else:
        next_q_value = (
            mlp_target(lstm_target(next_state, memory_types)[0])
            .max(dim=1, keepdim=True)[0]
            .detach()
        )
    mask = 1 - done
    target = (reward + gamma * next_q_value * mask).to(device)

    # calculate dqn loss
    loss = F.smooth_l1_loss(curr_q_value, target)

    return loss


def select_action(
    memory_types: list[str] | None,
    state: dict,
    greedy: bool,
    lstm: torch.nn.Module,
    mlp: torch.nn.Module,
    epsilon: float,
    action_space: gym.spaces.Discrete,
) -> tuple[int, list]:
    """Select an action from the input state, with epsilon-greedy policy.

    Args:
        memory_types: memory_types
        state: The current state of the memory systems. This is NOT what the gym env
            gives you. This is made by the agent.
        greedy: always pick greedy action if True
        lstm: lstm model
        mlp: mlp model, which is the dqn model
        epsilon: epsilon value
        action_space: gym action space

    Returns:
        selected_action: an action to take.
        q_values: a list of q values for each action.

    """
    q_values = mlp(lstm(np.array([state]), memory_types)[0]).detach().cpu().tolist()[0]

    if greedy or epsilon < np.random.random():
        selected_action = argmax(q_values)
    else:
        selected_action = action_space.sample().item()

    return selected_action, q_values


def update_model(
    memory_types: list[str],
    replay_buffer: ReplayBuffer,
    optimizer: torch.optim.Adam,
    device: str,
    lstm: torch.nn.Module,
    lstm_target: torch.nn.Module,
    mlp: torch.nn.Module,
    mlp_target: torch.nn.Module,
    ddqn: str,
    gamma: float,
) -> torch.Tensor:
    """Update the model by gradient descent.

    Args:
        memory_types: memory_types
        replay_buffer: replay buffer
        optimizer: optimizer
        device: cpu or cuda
        lstm: lstm model
        lstm_target: lstm target model
        mlp: mlp model
        mlp_target: mlp target model
        ddqn: whether to use double dqn or not
        gamma: discount factor

    Returns:
        loss: temporal difference loss value
    """
    samples = replay_buffer.sample_batch()

    loss = compute_loss(
        memory_types, samples, device, lstm, lstm_target, mlp, mlp_target, ddqn, gamma
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def save_validation(
    policy: Literal["mm", "explore", None],
    scores_temp: list,
    scores: dict,
    default_root_dir: str,
    num_episodes: int,
    validation_interval: int,
    val_dir_names: list,
    lstm: torch.nn.Module,
    mlp: torch.nn.Module,
) -> None:
    """Keep the best validation model.

    Args:
        policy: "mm", "explore", or None.
        scores_temp: a list of validation scores for the current validation episode.
        scores: a dictionary of scores for train, validation, and test.
        default_root_dir: the root directory where the results are saved.
        num_episodes: number of episodes run so far
        validation_interval: the interval to validate the model.
        val_dir_names: a list of dirnames for the validation models.
        lstm: the lstm model.
        mlp: the mlp model.

    """
    mean_score = round(np.mean(scores_temp).item())

    subdir = "checkpoint"
    if policy is not None:
        subdir = os.path.join(policy, subdir)

    dir_name = os.path.join(
        default_root_dir,
        subdir,
        f"episode={num_episodes}_val-score={mean_score}",
    )

    os.makedirs(dir_name, exist_ok=True)
    torch.save(lstm.state_dict(), os.path.join(dir_name, "lstm.pt"))
    torch.save(mlp.state_dict(), os.path.join(dir_name, "mlp.pt"))

    val_dir_names.append(dir_name)

    for _ in range(validation_interval):
        scores["val"].append(scores_temp)

    scores_to_compare = []
    for dir_name in val_dir_names:
        score = int(dir_name.split("val-score=")[-1].split(".pt")[0].split("/")[-1])
        scores_to_compare.append(score)

    indexes = list_duplicates_of(scores_to_compare, max(scores_to_compare))
    dir_to_keep = val_dir_names[indexes[-1]]

    for dir_name in val_dir_names:
        if dir_name != dir_to_keep:
            shutil.rmtree(dir_name, ignore_errors=True)
            val_dir_names.remove(dir_name)


def save_states_q_values_actions(
    policy: Literal["mm", "explore", None],
    states: list,
    q_values: list,
    actions: list,
    default_root_dir: str,
    val_or_test: str,
    num_episodes: int | None = None,
) -> None:
    """Save states, q_values, and actions.

    Args:
        policy: "mm", "explore", or None.
        states: a list of states.
        q_values: a list of q_values.
        actions: a list of actions.
        default_root_dir: the root directory where the results are saved.
        val_or_test: "val" or "test"
        num_episodes: the number of episodes run so far.

    """

    subdir = policy if policy is not None else ""
    filename_template = (
        f"states_q_values_actions_val_episode={num_episodes}.yaml"
        if val_or_test.lower() == "val"
        else "states_q_values_actions_test.yaml"
    )

    filename = os.path.join(default_root_dir, subdir, filename_template)

    assert len(states) == len(q_values) == len(actions)
    to_save = [
        {"state": s, "q_values": q, "action": a}
        for s, q, a in zip(states, q_values, actions)
    ]
    write_yaml(to_save, filename)


def target_hard_update(dqn: torch.nn.Module, dqn_target: torch.nn.Module) -> None:
    """Hard update: target <- local.

    Args:
        dqn: dqn model
        dqn_target: dqn target model
    """
    dqn_target.load_state_dict(dqn.state_dict())


def update_epsilon(
    epsilon: float, max_epsilon: float, min_epsilon: float, epsilon_decay_until: int
) -> float:
    """Linearly decrease epsilon

    Args:
        epsilon: current epsilon
        max_epsilon: initial epsilon
        min_epsilon: minimum epsilon
        epsilon_decay_until: the last iteration index to decay epsilon

    Returns:
        epsilon: updated epsilon

    """
    epsilon = max(
        min_epsilon, epsilon - (max_epsilon - min_epsilon) / epsilon_decay_until
    )

    return epsilon
