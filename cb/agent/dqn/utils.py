"""Utility functions for DQN."""

import os
import random
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from humemai_research.utils import (argmax, is_running_notebook, list_duplicates_of,
                           write_pickle, write_yaml)
from IPython.display import clear_output
from tqdm.auto import tqdm


class ReplayBuffer:
    r"""A simple numpy replay buffer.

    numpy replay buffer is faster than deque or list.
    copied from https://github.com/Curt-Park/rainbow-is-all-you-need

    Attributes:
        states_buf (np.ndarray): Buffer for observations, initialized as an array of None
            values of dtype=object.
        questions_buf (np.ndarray): Buffer for questions, initialized similarly to
            states_buf.
        rewards_buf (np.ndarray): Buffer for rewards, initialized similarly to
            rews_explore_buf.
        max_size (int): Maximum size of the buffer.
        batch_size (int): Batch size for sampling from the buffer.
        ptr (int): Pointer to the current position in the buffer.
        size (int): Current size of the buffer.

    """

    def __init__(self, size: int, batch_size: int, num_questions_step: int):
        """Initialize replay buffer.

        Args:
            size: size of the buffer
            batch_size: batch size to sample
            num_questions_step: number of questions to ask at each step

        Raises:
            ValueError: If batch_size is greater than size.

        """
        if batch_size > size:
            raise ValueError("batch_size must be smaller than size")

        self.num_questions_step = num_questions_step

        self.states_buf = np.array([None] * size, dtype=object)
        self.actions_buf = np.array([None] * size, dtype=object)
        self.questions_buf = np.array([None] * size, dtype=object)
        self.rewards_buf = np.zeros([size, self.num_questions_step], dtype=np.float32)
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
        state: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        question: list[list],
    ) -> None:
        r"""Store the data in the buffer.

        Args:
            state: state
            action: action
            reward: reward
            question: question

        """

        self.states_buf[self.ptr] = state
        self.actions_buf[self.ptr] = action
        self.questions_buf[self.ptr] = question
        self.rewards_buf[self.ptr] = reward
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self) -> dict[str, np.ndarray]:
        r"""Sample a batch of data from the buffer.

        Returns:
            A dictionary of samples from the replay buffer.

        """
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(
            states=self.states_buf[idxs],
            actions=self.actions_buf[idxs],
            questions=self.questions_buf[idxs],
            rewards=self.rewards_buf[idxs],
        )

    def __len__(self) -> int:
        return self.size


def plot_results(
    scores: dict[str, list[float]],
    training_loss: list[float],
    epsilons: list[float],
    iteration_idx: int,
    num_iterations: int,
    total_maximum_episode_rewards: int,
    default_root_dir: str,
) -> None:
    r"""Plot things for DQN training.

    Args:
        scores: a dictionary of scores for train, validation, and test.
        training_loss: a dict of training losses for all, mm, and explore.
        epsilons: a list of epsilons.
        iteration_idx: the current iteration index.
        num_iterations: the total number of iterations.
        total_maximum_episode_rewards: the total maximum episode rewards.
        default_root_dir: the root directory where the results are saved.

    """
    is_notebook = is_running_notebook()

    if is_notebook:
        clear_output(True)

    plt.figure(figsize=(20, 6))

    plt.subplot(133)
    if scores["train"]:
        plt.title(
            f"iteration {iteration_idx} out of {num_iterations}. "
            f"training score: {scores['train'][-1]} out of "
            f"{total_maximum_episode_rewards}"
        )
        plt.plot(scores["train"], label="Training score")
        plt.xlabel("episode")

    if scores["val"]:
        val_means = [round(np.mean(scores).item()) for scores in scores["val"]]
        plt.title(
            f"validation score: {val_means[-1]} out of "
            f"{total_maximum_episode_rewards}"
        )
        plt.plot(val_means, label="Validation score")
        plt.xlabel("episode")

    if scores["test"]:
        plt.title(
            f"test score: {np.mean(scores['test'])} out of "
            f"{total_maximum_episode_rewards}"
        )
        plt.plot(
            [round(np.mean(scores["test"]).item(), 2)] * len(scores["train"]),
            label="Test score",
        )
        plt.xlabel("episode")
    plt.legend(loc="best")

    plt.subplot(131)
    plt.title("training BCE loss (log scale)")
    plt.plot(training_loss)
    plt.yscale("log")  # Set y-axis to log scale
    plt.xlabel("update counts")

    plt.subplot(132)
    plt.title("epsilons")
    plt.plot(epsilons)
    plt.xlabel("update counts")

    plt.subplots_adjust(hspace=0.5)
    plt.savefig(os.path.join(default_root_dir, "plot.pdf"))

    if is_notebook:
        plt.show()
    else:
        plt.close("all")


def save_final_results(
    scores: dict[str, list[float]],
    training_loss: list[float],
    default_root_dir: str,
) -> None:
    r"""Save dqn train / val / test results.

    Args:
        scores: a dictionary of scores for train, validation, and test.
        training_loss: BCE loss
        default_root_dir: the root directory where the results are saved.

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
    write_yaml(results, os.path.join(default_root_dir, "results.yaml"))


def update_model(
    replay_buffer: ReplayBuffer,
    optimizer: torch.optim.Adam,
    device: str,
    dqn: torch.nn.Module,
) -> tuple[float, float, float]:
    r"""Update the model by gradient descent.

    Args:
        replay_buffer: replay buffer
        optimizer: optimizer
        device: cpu or cuda
        dqn: dqn model

    Returns:
        loss_qa

    """

    batch = replay_buffer.sample_batch()

    states = batch["states"]
    actions = batch["actions"]
    questions = batch["questions"]
    rewards = batch["rewards"]

    q_values, qa_triples = dqn(states, policy_type="qa", questions=questions)

    q_values_chosen = []

    for actions_, q_values_ in zip(actions, q_values):
        q_values_chosen_ = q_values_[
            torch.arange(q_values_.size(0)), actions_.astype(int)
        ]
        q_values_chosen.append(q_values_chosen_)

    q_values_chosen = torch.cat(q_values_chosen).reshape(-1, 1)
    rewards = torch.tensor(rewards).reshape(-1, 1).type(torch.float32).to(device)

    loss = F.binary_cross_entropy(q_values_chosen, rewards)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss = loss.detach().cpu().numpy().item()

    return loss


def select_action(
    state: list[list],
    greedy: bool,
    dqn: torch.nn.Module,
    epsilon: float,
    questions: list | None = None,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    r"""Select action(s) from the input state, with epsilon-greedy policy.

    Args:
        state: This is the input to the neural network. Make sure that it's compatible
            with the input shape of the neural network. It's very likely that this
            looks like a list of quadruples.
        greedy: always pick greedy action if True
        dqn: dqn model
        epsilon: epsilon
        policy_type: "mm", "explore", or "qa"
        questions: a list of questions. This is only used when policy_type is "qa".

    Returns:
        answers: dimension is [num_questions]
        q_values: list of np.ndarray, where each np.ndarray has dimension
            [num_actions]. num_action (arms) varies by question.

    """

    # Since dqn requires a batch dimension, we need to encapsulate the state in a list
    q_values, qa_triples = dqn(
        np.array([state], dtype=object),
        policy_type="qa",
        questions=[questions],
    )
    q_values = q_values[0]  # remove the dummy batch dimension
    qa_triples = qa_triples[0]  # remove the dummy batch dimension

    q_values = q_values.detach().cpu().numpy()

    if greedy or epsilon < np.random.random():
        a_qa = q_values.argmax(axis=1)
    else:
        a_qa = np.random.randint(0, q_values.shape[1], q_values.shape[0])

    answers = []
    for action, triples in zip(a_qa, qa_triples):
        answers.append(triples[action][2])

    return answers, a_qa, q_values


def save_validation(
    scores_temp: list,
    default_root_dir: str,
    num_episodes: int,
    val_file_names: list,
    dqn: torch.nn.Module,
) -> None:
    r"""Keep the best validation model.

    Args:
        scores_temp: a list of validation scores for the current validation episode.
        default_root_dir: the root directory where the results are saved.
        num_episodes: number of episodes run so far
        val_file_names: a list of dirnames for the validation models.
        dqn: the dqn model.

    """
    mean_score = round(np.mean(scores_temp).item())

    filename = os.path.join(
        default_root_dir, f"episode={num_episodes}_val-score={mean_score}.pt"
    )
    torch.save(dqn.state_dict(), filename)

    val_file_names.append(filename)

    scores_to_compare = []
    for fn in val_file_names:
        score = int(fn.split("val-score=")[-1].split(".pt")[0])
        scores_to_compare.append(score)

    indexes = list_duplicates_of(scores_to_compare, max(scores_to_compare))
    file_to_keep = val_file_names[indexes[-1]]

    for fn in val_file_names:
        if fn != file_to_keep:
            os.remove(fn)
            val_file_names.remove(fn)


def target_hard_update(
    dqn: torch.nn.Module,
    dqn_target: torch.nn.Module,
) -> None:
    r"""Hard update: update target with local.

    Args:
        dqn: dqn model
        dqn_target: dqn target model
    """
    dqn_target.load_state_dict(dqn.state_dict())


def update_epsilon(
    epsilon: float, max_epsilon: float, min_epsilon: float, epsilon_decay_until: int
) -> float:
    r"""Linearly decrease epsilon

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
