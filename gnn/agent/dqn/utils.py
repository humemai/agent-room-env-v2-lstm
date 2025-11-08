"""Utility functions for DQN."""

import os
import random
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from humemai_research.utils import (
    argmax,
    is_running_notebook,
    list_duplicates_of,
    write_pickle,
    write_yaml,
)
from IPython.display import clear_output
from tqdm.auto import tqdm


class ReplayBuffer:
    r"""A simple numpy replay buffer.

    numpy replay buffer is faster than deque or list.
    copied from https://github.com/Curt-Park/rainbow-is-all-you-need

    Attributes:
        obs_buf (np.ndarray): Buffer for observations, initialized as an array of None
            values of dtype=object.
        next_obs_buf (np.ndarray): Buffer for next observations, initialized similarly
            to obs_buf.
        acts_explore_buf (np.ndarray): Buffer for explore actions, initialized as an
            array of None values of dtype=object.
        acts_mm_buf (np.ndarray): Buffer for mm actions, initialized as an array of None
            values of dtype=object.
        rews_explore_buf (np.ndarray): Buffer for rewards, initialized as an array of
            zeros of dtype=np.float32.
        rews_mm_buf (np.ndarray): Buffer for rewards, initialized similarly to
            rews_explore_buf.
        done_buf (np.ndarray): Buffer for done flags, initialized as an array of zeros
            of dtype=np.float32.
        max_size (int): Maximum size of the buffer.
        batch_size (int): Batch size for sampling from the buffer.
        ptr (int): Pointer to the current position in the buffer.
        size (int): Current size of the buffer.


    Example:
    ```
    from agent.dqn.utils import ReplayBuffer
    import random

    buffer = ReplayBuffer(8, 4)

    for _ in range(6):

        obs = {str(i): str(random.randint(0, 10)) for i in range(3)}
        next_obs = {str(i): str(random.randint(0, 10)) for i in range(3)}
        action_explore = random.randint(0, 4)
        action_mm = [random.randint(0, 3) for _ in range(random.randint(1, 10))]
        reward_explore = random.randint(0, 1)
        reward_mm = random.randint(0, 1)
        done = random.choice([False, True])
        buffer.store(
            *[
                obs,
                action_explore,
                action_mm,
                reward_explore,
                reward_mm,
                next_obs,
                done,
            ]
        )

    sample = buffer.sample_batch()
    ```
    >>> sample
    {'obs': array([{'0': '5', '1': '10', '2': '2'}, {'0': '9', '1': '2', '2': '5'},
            {'0': '6', '1': '10', '2': '9'}, {'0': '6', '1': '0', '2': '6'}],
        dtype=object),
    'next_obs': array([{'0': '10', '1': '0', '2': '2'}, {'0': '9', '1': '2', '2': '4'},
            {'0': '1', '1': '7', '2': '1'}, {'0': '8', '1': '9', '2': '0'}],
        dtype=object),
    'acts_explore': array([4., 3., 2., 2.], dtype=float32),
    'acts_mm': array([np.array([2, 0, 2, 1, 3]), np.array([0, 1, 2]),
            np.array([0, 1, 3, 1, 1, 2, 2]), np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])],
        dtype=object),
    'rews_explore': array([0., 1., 1., 0.], dtype=float32),
    'rews_mm': array([0., 1., 1., 0.], dtype=float32),
    'done': array([1., 0., 1., 1.], dtype=float32)}

    >>> sample["acts_mm"].shape
    (4,)

    """

    def __init__(
        self,
        size: int,
        batch_size: int = 32,
    ):
        """Initialize replay buffer.

        Args:
            size: size of the buffer
            batch_size: batch size to sample

        Raises:
            ValueError: If batch_size is greater than size.

        Note:
            The obs_buf, next_obs_buf, and acts_mm_buf are initialized with `None`
            values and have `dtype=object` to accommodate arbitrary Python objects,
            ensuring flexibility in storing different types of data.

        """

        if batch_size > size:
            raise ValueError("batch_size must be smaller than size")

        self.obs_buf = np.array([None] * size, dtype=object)
        self.next_obs_buf = np.array([None] * size, dtype=object)
        self.acts_explore_buf = np.zeros([size], dtype=int)
        self.acts_mm_buf = np.array([None] * size, dtype=object)
        self.rews_explore_buf = np.zeros([size], dtype=np.float32)
        self.rews_mm_buf = np.zeros([size], dtype=np.float32)
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
        act_explore: np.ndarray,
        act_mm: np.ndarray,
        rew_explore: float,
        rew_mm: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        r"""Store the data in the buffer.

        Args:
            obs: observation
            act_explore: explore action
            act_mm: memory management action
            rew_explore: reward for explore
            rew_mm: reward for memory management
            next_obs: next observation
            done: done

        """
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_explore_buf[self.ptr] = act_explore
        self.acts_mm_buf[self.ptr] = act_mm
        self.rews_explore_buf[self.ptr] = rew_explore
        self.rews_mm_buf[self.ptr] = rew_mm
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self) -> dict[str, np.ndarray]:
        r"""Sample a batch of data from the buffer.

        Returns:
            A dictionary of samples from the replay buffer.
                obs: np.ndarray,
                next_obs: np.ndarray,
                acts_explore: np.ndarray,
                acts_mm: np.ndarray,
                rews_explore: np.ndarray,
                rews_mm: np.ndarray,
                done: np.ndarray

        """
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(
            obs=self.obs_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            acts_explore=self.acts_explore_buf[idxs],
            acts_mm=self.acts_mm_buf[idxs],
            rews_explore=self.rews_explore_buf[idxs],
            rews_mm=self.rews_mm_buf[idxs],
            done=self.done_buf[idxs],
        )

    def __len__(self) -> int:
        return self.size


def plot_results(
    scores: dict[str, list[float]],
    training_loss: dict[str, list[float]],
    epsilons: list[float],
    q_values: dict[str, dict[str, list[float]]],
    iteration_idx: int,
    num_iterations: int,
    total_maximum_episode_rewards: int,
    default_root_dir: str,
    action_mm2str,
    action_explore2str,
    to_plot: str = "all",
    save_fig: bool = False,
) -> None:
    r"""Plot things for DQN training.

    Args:
        scores: a dictionary of scores for train, validation, and test.
        training_loss: a dict of training losses for all, mm, and explore.
        epsilons: a list of epsilons.
        q_values: a dictionary of q_values for train, validation, and test.
        iteration_idx: the current iteration index.
        num_iterations: the total number of iterations.
        total_maximum_episode_rewards: the total maximum episode rewards.
        default_root_dir: the root directory where the results are saved.
        action_mm2str: a dictionary to convert mm actions to strings.
        action_explore2str: a dictionary to convert explore actions to strings.
        to_plot: what to plot:
            "all": plot everything
            "training_td_loss": plot training td loss
            "epsilons": plot epsilons
            "scores": plot scores
            "q_value_train": plot q_values for training
            "q_value_val": plot q_values for validation
            "q_value_test": plot q_values for test
        save_fig: whether to save the figure or not

    """
    is_notebook = is_running_notebook()

    if is_notebook:
        clear_output(True)

    if to_plot == "all":
        plt.figure(figsize=(20, 20))

        plt.subplot(333)
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

        plt.subplot(331)
        plt.title("training td loss (log scale)")
        plt.plot(training_loss["total"], label="total")
        plt.plot(training_loss["mm"], label="mm")
        plt.plot(training_loss["explore"], label="explore")
        plt.yscale("log")  # Set y-axis to log scale
        plt.xlabel("update counts")
        plt.legend(loc="best")

        plt.subplot(332)
        plt.title("epsilons")
        plt.plot(epsilons)
        plt.xlabel("update counts")

        for subplot_num, split in zip([334, 335, 336], ["train", "val", "test"]):
            plt.subplot(subplot_num)
            plt.title(f"Q-values (mm), {split}")
            for action_number in range(3):
                plt.plot(
                    [
                        q[action_number]
                        for q_value_ in q_values[split]["mm"]
                        for q in q_value_
                    ],
                    label=action_mm2str[action_number],
                )
            plt.legend(loc="best")
            plt.xlabel("number of actions")

        for subplot_num, split in zip([337, 338, 339], ["train", "val", "test"]):
            plt.subplot(subplot_num)
            plt.title(f"Q-values (explore), {split}")
            for action_number in range(5):
                plt.plot(
                    [
                        q_value_[0][action_number]
                        for q_value_ in q_values[split]["explore"]
                    ],
                    label=action_explore2str[action_number],
                )
            plt.legend(loc="best")
            plt.xlabel("number of actions")

        plt.subplots_adjust(hspace=0.5)
        if save_fig:
            plt.savefig(os.path.join(default_root_dir, "plot.pdf"))

        if is_notebook:
            plt.show()
        else:
            console(**locals())
            plt.close("all")

    elif to_plot == "training_td_loss":
        plt.figure()
        plt.title("training td loss")
        plt.plot(training_loss["total"], label="total")
        plt.plot(training_loss["mm"], label="mm")
        plt.plot(training_loss["explore"], label="explore")
        plt.xlabel("update counts")
        plt.legend(loc="best")
        plt.subplots_adjust(hspace=0.5)

    elif to_plot == "epsilons":
        plt.figure()
        plt.title("epsilons")
        plt.plot(epsilons)
        plt.xlabel("update counts")
        plt.subplots_adjust(hspace=0.5)

    elif to_plot == "scores":
        plt.figure()

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
        plt.subplots_adjust(hspace=0.5)

    else:
        plt.figure(figsize=(20, 13))

        for subplot_num, split in zip([231, 232, 233], ["train", "val", "test"]):
            plt.subplot(subplot_num)
            plt.title(f"Q-values (mm), {split}")
            for action_number in range(3):
                plt.plot(
                    [q_value_[action_number] for q_value_ in q_values[split]["mm"]],
                    label=f"action {action_number}",
                )
            plt.legend(loc="best")
            plt.xlabel("number of actions")

        for subplot_num, split in zip([234, 235, 236], ["train", "val", "test"]):
            plt.subplot(subplot_num)
            plt.title(f"Q-values (explore), {split}")
            for action_number in range(5):
                plt.plot(
                    [
                        q_value_[action_number]
                        for q_value_ in q_values[split]["explore"]
                    ],
                    label=f"action {action_number}",
                )
            plt.legend(loc="best")
            plt.xlabel("number of actions")

        plt.subplots_adjust(hspace=0.5)


def console(
    scores: dict,
    training_loss: list,
    iteration_idx: int,
    num_iterations: int,
    total_maximum_episode_rewards: int,
    **kwargs,
) -> None:
    r"""Print the dqn training to the console."""
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
            f"test score: {np.mean(scores['test'])} out of "
            f"{total_maximum_episode_rewards}"
        )

    tqdm.write(f"training loss (all): {training_loss['total'][-1]}\n")
    print()


def save_final_results(
    scores: dict[str, list[float]],
    training_loss: dict[str, list[float]],
    default_root_dir: str,
    q_values: dict[str, dict[str, list[float]]],
    self: object,
    save_the_agent: bool = False,
) -> None:
    r"""Save dqn train / val / test results.

    Args:
        scores: a dictionary of scores for train, validation, and test.
        training_loss: a dict of training losses for all, mm, and explore.
        training_loss_explore: a list of training losses for explore.
        default_root_dir: the root directory where the results are saved.
        q_values: a dictionary of q_values for train, validation, and test.
        self: the agent object.
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
    write_yaml(results, os.path.join(default_root_dir, "results.yaml"))

    q_values_list = {
        key: {key_: [val__.tolist() for val__ in val_] for key_, val_ in val.items()}
        for key, val in q_values.items()
    }

    write_yaml(q_values_list, os.path.join(default_root_dir, "q_values.yaml"))
    if save_the_agent:
        write_pickle(self, os.path.join(default_root_dir, "agent.pkl"))


def find_non_masked_rows(mask: torch.Tensor) -> list[torch.Tensor]:
    r"""
    Identifies rows that are not completely masked in each batch of a 3D tensor.

    This function considers a row as non-masked if it contains at least one non-zero
    element. It processes each batch independently and returns the indices of non-masked
    rows for each batch.

    Args:
        mask (torch.Tensor): A 3D tensor of shape [batch_size, num_items, num_features],
            containing binary values (0s and 1s).
            - batch_size: number of batches
            - num_items: number of items (rows) in each batch
            - num_features: number of features for each item

    Returns:
        list[torch.Tensor]: A list of 1D tensors, where each tensor contains the indices
        of non-masked rows for the corresponding batch. The length of the list is equal
        to the batch_size. Each tensor in the list has shape [num_non_masked_rows] and
        dtype torch.long.

    Example:
        >>> mask = torch.tensor([
        ...     [[1, 0, 1], [0, 0, 0], [0, 1, 0]],
        ...     [[0, 0, 0], [1, 1, 1], [0, 1, 0]]
        ... ])
        >>> result = find_non_masked_rows(mask)
        >>> print(result)
        [tensor([0, 2]), tensor([1, 2])]

    Note:
        - A row is considered non-masked if it contains at least one non-zero element.
        - If a batch has all rows masked, the corresponding tensor in the output list
          will be empty.
        - This function does not modify the input tensor.

    Raises:
        ValueError: If the input tensor is not 3-dimensional.
    """
    # Assuming mask values are 0 for masked and 1 for unmasked
    # Sum across the last dimension (features), we want rows where the sum is 0
    row_sums = mask.sum(dim=2)

    # Identify rows where the sum is 0 (all features are masked)
    masked_rows = row_sums != 0

    # Optionally, get the indices of the masked rows
    masked_indices = [
        torch.nonzero(masked_batch, as_tuple=False).squeeze(1)
        for masked_batch in masked_rows
    ]

    return masked_indices


def compute_loss_mm(
    batch: dict,
    device: str,
    dqn: torch.nn.Module,
    dqn_target: torch.nn.Module,
    ddqn: str,
    gamma: float,
) -> torch.Tensor:
    r"""Return the DQN td loss for the memory management policy.

    G_t   = r + gamma * v(s_{t+1})  if state != Terminal
          = r                       otherwise

    Args:
        batch: A dictionary of samples from the replay buffer.
            obs: np.ndarray,
            act: np.ndarray,
            rew: float,
            next_obs: np.ndarray,
            done: bool,
        device: cpu or cuda
        dqn: dqn model
        dqn_target: dqn target model
        ddqn: whether to use double dqn or not
        gamma: discount factor

    Returns:
        loss_mm: TD loss for memory management

    """
    state = batch["obs"]
    state_next = batch["next_obs"]
    action = [torch.LongTensor(s).reshape(-1, 1).to(device) for s in batch["acts"]]
    reward = torch.FloatTensor(batch["rews"]).to(device)
    done = torch.FloatTensor(batch["done"]).to(device)

    # Forward pass on current state to get Q-values
    q_value_current = dqn(state, policy_type="mm")

    # Forward pass on next state to get Q-values
    q_value_next = dqn_target(state_next, policy_type="mm")

    if ddqn:
        q_value_for_action = dqn(state_next, policy_type="mm")

    q_value_current_batch = []
    q_value_target_batch = []

    min_lens = [min(len(i), len(j)) for i, j in zip(q_value_current, q_value_next)]

    for idx in range(len(min_lens)):
        min_len = min_lens[idx]

        action_ = action[idx][:min_len]
        q_value_current_ = q_value_current[idx][:min_len]
        q_value_current_chosen = q_value_current_.gather(1, action_)
        q_value_next_ = q_value_next[idx][:min_len]

        if ddqn:
            # Double DQN: Use current DQN to select actions, target DQN to evaluate
            # those actions
            q_value_for_action_ = q_value_for_action[idx][:min_len]
            action_next = q_value_for_action_.argmax(dim=1, keepdim=True)
            q_value_next_chosen = q_value_next_.gather(1, action_next).detach()
        else:
            # Vanilla DQN: Use target DQN to get max Q-value for next state
            q_value_next_chosen = q_value_next_.max(dim=1, keepdim=True)[0].detach()

        # Compute the target Q-values considering whether the state is terminal
        q_value_target_ = reward[idx] + gamma * q_value_next_chosen * (1 - done[idx])

        q_value_current_batch.append(q_value_current_chosen)
        q_value_target_batch.append(q_value_target_)

    q_value_current_batch = torch.concat(q_value_current_batch, dim=0)
    q_value_target_batch = torch.concat(q_value_target_batch, dim=0)

    assert q_value_current_batch.shape == q_value_target_batch.shape

    # Calculate loss
    loss = F.smooth_l1_loss(q_value_current_batch, q_value_target_batch)

    return loss


def compute_loss_explore(
    batch: dict,
    device: str,
    dqn: torch.nn.Module,
    dqn_target: torch.nn.Module,
    ddqn: str,
    gamma: float,
) -> torch.Tensor:
    r"""Return the DQN td loss for explore policy.

    G_t   = r + gamma * v(s_{t+1})  if state != Terminal
          = r                       otherwise

    Args:
        batch: A dictionary of samples from the replay buffer.
            obs: np.ndarray,
            acts: np.ndarray,
            rews: float,
            next_obs: np.ndarray,
            done: bool,
        device: cpu or cuda
        dqn: dqn model
        dqn_target: dqn target model
        ddqn: whether to use double dqn or not
        gamma: discount factor

    Returns:
        loss: TD loss for the explore policy

    """

    state = batch["obs"]
    state_next = batch["next_obs"]
    action = torch.LongTensor(batch["acts"].reshape(-1, 1)).to(device)
    reward = torch.FloatTensor(batch["rews"]).reshape(-1, 1).to(device)
    done = torch.FloatTensor(batch["done"]).reshape(-1, 1).to(device)

    # Forward pass on current state to get Q-values
    q_value_current = dqn(state, policy_type="explore")

    q_value_current = torch.concat(q_value_current)
    q_value_current = q_value_current.gather(1, action)

    q_value_next = dqn_target(state_next, policy_type="explore")
    q_value_next = torch.concat(q_value_next)

    if ddqn:
        # Double DQN: Use current DQN to select actions, target DQN to evaluate those
        # actions
        q_value_for_action = dqn(state_next, policy_type="explore")
        q_value_for_action = torch.concat(q_value_for_action)
        action_next = q_value_for_action.argmax(dim=1, keepdim=True)
        q_value_next = q_value_next.gather(1, action_next).detach()
    else:
        # Vanilla DQN: Use target DQN to get max Q-value for next state
        q_value_next = q_value_next.max(dim=1, keepdim=True)[0].detach()

    # Compute the target Q-values considering whether the state is terminal
    q_value_target = reward + gamma * q_value_next * (1 - done)

    assert q_value_current.shape == q_value_target.shape

    # Calculate loss
    loss = F.smooth_l1_loss(q_value_current, q_value_target)

    return loss


def update_model(
    replay_buffer: ReplayBuffer,
    optimizer: torch.optim.Adam,
    device: str,
    dqn: torch.nn.Module,
    dqn_target: torch.nn.Module,
    ddqn: str,
    gamma: dict[str, float],
) -> tuple[float, float, float]:
    r"""Update the model by gradient descent.

    Args:
        replay_buffer: replay buffer
        optimizer: optimizer
        device: cpu or cuda
        dqn: dqn model
        dqn_target: dqn target model
        ddqn: whether to use double dqn or not
        gamma: discount factor

    Returns:
        loss_mm, loss_explore, loss_combined: TD losses for memory management,
            explore and combined

    """
    batch = replay_buffer.sample_batch()
    batch_mm = {
        "obs": batch["obs"],
        "acts": batch["acts_mm"],
        "rews": batch["rews_mm"],
        "next_obs": batch["next_obs"],
        "done": batch["done"],
    }
    batch_explore = {
        "obs": batch["obs"],
        "acts": batch["acts_explore"],
        "rews": batch["rews_explore"],
        "next_obs": batch["next_obs"],
        "done": batch["done"],
    }

    loss_mm = compute_loss_mm(batch_mm, device, dqn, dqn_target, ddqn, gamma["mm"])
    loss_explore = compute_loss_explore(
        batch_explore,
        device,
        dqn,
        dqn_target,
        ddqn,
        gamma["explore"],
    )

    loss = loss_mm + loss_explore

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_mm = loss_mm.detach().cpu().numpy().item()
    loss_explore = loss_explore.detach().cpu().numpy().item()
    loss = loss.detach().cpu().numpy().item()

    return loss_mm, loss_explore, loss


def select_action(
    state: list[list],
    greedy: bool,
    dqn: torch.nn.Module,
    epsilon: float,
    policy_type: Literal["mm", "explore"],
) -> tuple[np.ndarray, np.ndarray]:
    r"""Select action(s) from the input state, with epsilon-greedy policy.

    Args:
        state: This is the input to the neural network. Make sure that it's compatible
            with the input shape of the neural network. It's very likely that this
            looks like a list of quadruples.
        greedy: always pick greedy action if True
        dqn: dqn model
        epsilon: epsilon
        policy_type: "mm" or "explore"

    Returns:
        selected_actions: dimension is [num_actions_taken]
        q_values: dimension is [num_actions_taken, action_space_dim]

    """
    # Since dqn requires a batch dimension, we need to encapsulate the state in a list
    q_values = dqn(np.array([state], dtype=object), policy_type=policy_type)

    q_values = q_values[0]  # remove the dummy batch dimension
    q_values = q_values.detach().cpu().numpy()

    action_space_dim = q_values.shape[1]

    if greedy or epsilon < np.random.random():
        selected_actions = q_values.argmax(axis=1)
    else:
        selected_actions = np.random.randint(0, action_space_dim, size=len(q_values))

    return selected_actions, q_values


def save_validation(
    scores_temp: list,
    scores: dict,
    default_root_dir: str,
    num_episodes: int,
    validation_interval: int,
    val_file_names: list,
    dqn: torch.nn.Module,
) -> None:
    r"""Keep the best validation model.

    Args:
        policy: "mm", "explore", or None.
        scores_temp: a list of validation scores for the current validation episode.
        scores: a dictionary of scores for train, validation, and test.
        default_root_dir: the root directory where the results are saved.
        num_episodes: number of episodes run so far
        validation_interval: the interval to validate the model.
        val_file_names: a list of dirnames for the validation models.
        dqn: the dqn model.

    """
    mean_score = round(np.mean(scores_temp).item())

    filename = os.path.join(
        default_root_dir, f"episode={num_episodes}_val-score={mean_score}.pt"
    )
    torch.save(dqn.state_dict(), filename)

    val_file_names.append(filename)

    for _ in range(validation_interval):
        scores["val"].append(scores_temp)

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


def save_states_q_values_actions(
    states: list[list[list]],
    q_values: list[dict],
    actions: list[dict],
    default_root_dir: str,
    val_or_test: str,
    num_episodes: int | None = None,
) -> None:
    r"""Save states, q_values, and actions.

    Args:
        states: a list of states.
        q_values: a list of q_values.
        actions: a list of actions.
        default_root_dir: the root directory where the results are saved.
        val_or_test: "val" or "test"
        num_episodes: the current validation episode.

    """
    filename_template = (
        f"states_q_values_actions_val_episode={num_episodes}.yaml"
        if val_or_test.lower() == "val"
        else "states_q_values_actions_test.yaml"
    )

    filename = os.path.join(default_root_dir, filename_template)

    assert len(states) == len(q_values) == len(actions)
    to_save = [
        {
            "state": s,
            "q_values_explore": q["explore"].tolist(),
            "action_explore": a["explore"].tolist(),
            "q_values_mm": q["mm"].tolist(),
            "action_mm": a["mm"].tolist(),
        }
        for s, q, a in zip(states, q_values, actions)
    ]
    write_yaml(to_save, filename)


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
