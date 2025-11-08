"""Handcrafted / trained policies.

The trained neural network policies are not implemented yet.
"""

import random
import re
from typing import Literal

import numpy as np
import torch
from humemai_research.memory import MemorySystems, ShortMemory


def encode_observation(memory_systems: MemorySystems, obs: list[str | int]) -> None:
    """Non RL policy of encoding an observation into a short-term memory.

    At the moment, observation is the same as short-term memory, since this is made for
    RoomEnv-v2.

    Args:
        MemorySystems
        obs: observation as a quadruple: [head, relation, tail, int]

    """
    mem_short = ShortMemory.ob2short(obs)

    check, error_msg = memory_systems.short.can_be_added(mem_short)
    if check:
        memory_systems.short.add(mem_short)
    else:
        raise ValueError(error_msg)


def encode_all_observations(
    memory_systems: MemorySystems, obs_multiple: list[list[str | int]]
) -> None:
    """Non RL policy of encoding all observations into short-term memories.

    Args:
        MemorySystems
        obs_multiple: a list of observations

    """
    assert isinstance(obs_multiple, list), "`obs_multi1ple` should be a list."
    assert memory_systems.short.is_empty
    for obs in obs_multiple:
        encode_observation(memory_systems, obs)


def find_agent_location(memory_systems: MemorySystems) -> str | None:
    """Find the current location of the agent. This uses the agent's short-term memory.

    Args:
        MemorySystems

    Returns:
        agent_current_location: str | None

    """
    for mem in memory_systems.get_working_memory():
        if mem[0] == "agent" and mem[1] == "atlocation" and "current_time" in mem[3]:
            return mem[2]

    raise ValueError(
        "Make sure that the agent's short-term memory has the current location."
    )


def explore(
    memory_systems: MemorySystems,
    policy: Literal["random", "avoid_walls", "neural"],
    action_explore2int: dict = {
        "north": 0,
        "east": 1,
        "south": 2,
        "west": 3,
        "stay": 4,
    },
    nn: None | torch.nn.Module = None,
) -> np.ndarray:
    """Explore the room (sub-graph).

    "random" policy: randomly choose an action.
    "avoid_walls" policy: avoid the walls. This takes into account the agent's current
        location (from episodic) and the memories about the map (from semantic and
        episodic)
    "neural" policy: neural network policy.

    Args:
        memory_systems: MemorySystems
        policy: "random" "avoid_walls", or "neural"
        action_explore2int: a dictionary that maps actions to integers
        nn: neural network

    Returns:
        action: The exploration action to take.

    """
    if policy == "random":
        action = random.choice(["north", "east", "south", "west", "stay"])
    elif policy == "avoid_walls":
        agent_current_location = find_agent_location(memory_systems)

        # Get all the memories related to the current location
        mems_map = []

        for mem in memory_systems.get_working_memory():
            if mem[0] == agent_current_location and mem[1] in [
                "north",
                "east",
                "south",
                "west",
            ]:
                mems_map.append(mem)

        assert len(mems_map) > 0, (
            "No memories about the map. Make sure that the agent's short-term memory "
            "has the map."
        )

        to_take = ["north", "east", "south", "west"]

        for mem in mems_map:
            if mem[2] == "wall" and mem[1] in to_take:
                to_take.remove(mem[1])

        if len(to_take) == 0:  # This is a very special case where there only one room.
            action = "stay"
        else:
            action = random.choice(to_take)

    elif policy == "neural":
        state = memory_systems.get_working_memory().to_list()
        q_values = nn(np.array([state], dtype=object), policy_type="explore")
        q_values = q_values[0]  # remove the dummy batch dimension
        q_values = q_values.detach().cpu().numpy()
        action = q_values.argmax(axis=1)

    else:
        raise ValueError("Unknown exploration policy.")

    if isinstance(action, str):
        action = np.array(action_explore2int[action])

    return action


def manage_memory(
    memory_systems: MemorySystems,
    policy: Literal["episodic", "semantic", "forget"],
    mem_short: list,
) -> None:
    """Non RL memory management policy. This function directly manages the long-term
    memory. It tries to balance the number of episodic and semantic memories. Balancing
    is done by trying to keep the equal number of timestamps and strengths.

    Args:
        MemorySystems
        policy: "episodic", "semantic", "random", "forget", or "handcrafted"
        mem_short: a short-term memory to be moved into a long-term memory.

    """
    assert policy.lower() in [
        "episodic",
        "semantic",
        "forget",
    ]
    assert memory_systems.short.has_memory(mem_short)
    assert not memory_systems.short.is_empty

    def forget_long_when_full(memory_systems: MemorySystems):
        num_timestamps, num_strengths = memory_systems.long.count_memories()

        if num_timestamps > num_strengths:
            memory_systems.long.forget_by_selection("oldest")

        elif num_timestamps < num_strengths:
            memory_systems.long.forget_by_selection("weakest")

        else:
            if "oldest" == random.choice(["oldest", "weakest"]):
                memory_systems.long.forget_by_selection("oldest")
            else:
                memory_systems.long.forget_by_selection("weakest")

    def move_to_episodic(memory_systems: MemorySystems, mem_short: list) -> None:
        mem_epi = ShortMemory.short2epi(mem_short)
        check, error_msg = memory_systems.long.can_be_added(mem_epi)
        if check:
            memory_systems.long.add(mem_epi)
        else:
            if error_msg == "The memory system is full!":
                forget_long_when_full(memory_systems)
                memory_systems.long.add(mem_epi)
            else:
                raise ValueError(error_msg)

    def move_to_semantic(memory_systems: MemorySystems, mem_short: list) -> None:
        mem_sem = ShortMemory.short2sem(mem_short)
        check, error_msg = memory_systems.long.can_be_added(mem_sem)
        if check:
            memory_systems.long.add(mem_sem)
        else:
            if error_msg == "The memory system is full!":
                forget_long_when_full(memory_systems)
                memory_systems.long.add(mem_sem)
            else:
                raise ValueError(error_msg)

    if policy.lower() == "episodic":
        move_to_episodic(memory_systems, mem_short)

    elif policy.lower() == "semantic":
        move_to_semantic(memory_systems, mem_short)

    elif policy.lower() == "forget":
        pass

    else:
        raise ValueError

    memory_systems.short.forget(mem_short)  # don't forget to forget the short memory!


def manage_short(
    memory_systems: MemorySystems,
    policy: Literal[
        "random", "episodic", "semantic", "forget", "handcrafted", "neural"
    ],
    action_mm2int: dict = {"episodic": 0, "semantic": 1, "forget": 2},
    nn: None | torch.nn.Module = None,
) -> np.ndarray:
    """Non RL memory management policy. This function directly manages the long-term
    memory. It tries to balance the number of episodic and semantic memories. Balancing
    is done by trying to keep the equal number of timestamps and strengths.

    Args:
        memory_systems: MemorySystems
        policy: "episodic", "semantic", "random", "forget", or "handcrafted"
        action_mm2int: a dictionary that maps actions to integers
        nn: neural network

    Returns:
        a_mm: The actions to take.

    """
    assert not memory_systems.short.is_empty

    a_mm = []

    if policy == "random":
        for mem_short in memory_systems.short:
            action = random.choice(list(action_mm2int.keys()))
            a_mm.append(action_mm2int[action])
    elif policy == "episodic":
        for mem_short in memory_systems.short:
            action = "episodic"
            a_mm.append(action_mm2int[action])
    elif policy == "semantic":
        for mem_short in memory_systems.short:
            action = "semantic"
            a_mm.append(action_mm2int[action])
    elif policy == "forget":
        for mem_short in memory_systems.short:
            action = "forget"
            a_mm.append(action_mm2int[action])
    elif policy == "handcrafted":
        for mem_short in memory_systems.short:
            if mem_short[0] == "agent":
                action = "episodic"
            elif "ind" in mem_short[0] or "dep" in mem_short[0]:
                action = "episodic"
            elif "sta" in mem_short[0]:
                action = "semantic"
            elif "room" in mem_short[0] and "room" in mem_short[2]:
                action = "semantic"
            elif "wall" in mem_short[2]:
                action = "forget"
            else:
                raise ValueError("something is wrong")

            a_mm.append(action_mm2int[action])

    elif policy == "neural":
        state = memory_systems.get_working_memory().to_list()
        q_values = nn(np.array([state], dtype=object), policy_type="mm")
        q_values = q_values[0]  # remove the dummy batch dimension
        q_values = q_values.detach().cpu().numpy()
        a_mm = q_values.argmax(axis=1)

    else:
        raise ValueError("Unknown policy.")

    if isinstance(a_mm, list):
        a_mm = np.array(a_mm)

    return a_mm


def answer_question(
    memory_systems: MemorySystems,
    qa_function: Literal["latest_strongest", "latest", "strongest", "random", "llm"],
    question: list[str | int],
    llm: None | torch.nn.Module = None,
) -> None | str | int:
    """Use the memory systems to answer a question from RoomEnv-v2. It assumes that
    the question is a one-hop query. As we closely follow the cognitive science,
    we only use the working memory answer the questions. The working memory is
    short-term + (partial) long-term memory. The long-term memory used is partial,
    since it can be too big.

    Short-term memory is first used to look for the latest memory and then the episodic
    memory.

    Args:
        MemorySystems: MemorySystems
        qa_function: "latest_strongest", "latest", "strongest", "random", or "llm"
        question: A quadruple given by RoomEnv-v2, e.g., [laptop, atlocation, ?,
            current_time]
        llm: The LLM model

    Returns:
        pred: prediction

    """
    if qa_function.lower() == "llm":
        llm.reset()  # reset the conversation to reduce the computation.
        llm.add_system_instruction(
            "You are an memory AI. You answer questions based on your memory. "
            "Your memories have a type, including short-term, episodic, "
            "or semantic, and can even combine multiple types. "
            "short-term meomories are the most recent ones, and they have a 'current_time' value "
            "episodic memories are the ones that have a 'timestamp' value, indicating the time when the memory was stored. "
            "semantic memories are the ones that have a 'strength' value, indicating the number of times this memory was stored. The higher it is, the stronger and more generalizable the memory is."
            "For example, given memory: "
            "[['ind_001', 'atlocation', 'room_021', {'current_time': 11}], "
            "['room_021', 'east', 'wall', {'current_time': 11}], "
            "['sta_005', 'atlocation', 'room_021', {'current_time': 11, 'timestamp': [10]}], "
            "['agent', 'atlocation', 'room_021', {'current_time': 11, 'timestamp': [9, 10]}], "
            "['room_021', 'west', 'wall', {'current_time': 11}], "
            "['room_021', 'north', 'room_017', {'current_time': 11, 'timestamp': [9, 10]}], "
            "['room_021', 'south', 'room_027', {'current_time': 11, 'timestamp': [9, 10]}], "
            "['room_000', 'north', 'wall', {'strength': 1}], "
            "['agent', 'atlocation', 'room_000', {'strength': 1}], "
            "['dep_007', 'atlocation', 'room_000', {'strength': 1}], "
            "['room_000', 'south', 'room_004', {'timestamp': [0]}], "
            "['dep_001', 'atlocation', 'room_000', {'strength': 1}], "
            "['room_000', 'east', 'room_001', {'timestamp': [0]}], "
            "['room_001', 'north', 'wall', {'strength': 1}], "
            "['agent', 'atlocation', 'room_001', {'strength': 1}], "
            "['room_001', 'south', 'room_005', {'timestamp': [1]}], "
            "['room_001', 'west', 'room_000', {'strength': 1}], "
            "['room_001', 'east', 'wall', {'strength': 1}], "
            "['room_005', 'east', 'room_006', {'timestamp': [2]}], "
            "['room_005', 'south', 'wall', {'timestamp': [2]}], "
            "['agent', 'atlocation', 'room_005', {'timestamp': [2]}], "
            "['room_005', 'west', 'room_004', {'timestamp': [2]}], "
            "['room_006', 'north', 'wall', {'timestamp': [3]}], "
            "['sta_004', 'atlocation', 'room_006', {'timestamp': [3]}], "
            "['room_006', 'south', 'room_010', {'timestamp': [3]}], "
            "['room_006', 'west', 'room_005', {'timestamp': [3]}], "
            "['room_006', 'east', 'room_007', {'timestamp': [3]}], "
            "['room_007', 'south', 'wall', {'strength': 1}], "
            "['agent', 'atlocation', 'room_007', {'timestamp': [4]}], "
            "['room_007', 'west', 'room_006', {'strength': 1}], "
            "['sta_001', 'atlocation', 'room_007', {'timestamp': [4]}], "
            "['room_007', 'east', 'room_008', {'strength': 1}], "
            "['room_007', 'north', 'wall', {'timestamp': [4]}], "
            "['agent', 'atlocation', 'room_008', {'timestamp': [5]}], "
            "['ind_006', 'atlocation', 'room_008', {'timestamp': [5]}], "
            "['room_008', 'south', 'room_011', {'strength': 1}], "
            "['room_008', 'east', 'room_009', {'strength': 1}], "
            "['sta_002', 'atlocation', 'room_008', {'timestamp': [5]}], "
            "['room_008', 'north', 'room_002', {'strength': 1}], "
            "['dep_002', 'atlocation', 'room_008', {'timestamp': [5]}], "
            "['room_009', 'south', 'room_012', {'timestamp': [6]}], "
            "['room_009', 'north', 'room_003', {'timestamp': [6]}], "
            "['agent', 'atlocation', 'room_009', {'strength': 1}], "
            "['room_009', 'east', 'wall', {'strength': 1}], "
            "['agent', 'atlocation', 'room_012', {'timestamp': [7]}], "
            "['room_012', 'west', 'room_011', {'strength': 1}], "
            "['room_012', 'east', 'wall', {'strength': 1}], "
            "['room_012', 'south', 'room_017', {'strength': 1}], "
            "['room_017', 'west', 'room_016', {'timestamp': [8]}], "
            "['agent', 'atlocation', 'room_017', {'timestamp': [8]}], "
            "['room_017', 'east', 'wall', {'strength': 1}], "
            "['room_017', 'south', 'room_021', {'timestamp': [8]}], "
            "['room_017', 'north', 'room_012', {'timestamp': [8]}], "
            "['sta_005', 'atlocation', 'room_030', {'strength': 3}]] "
            "and question ['sta_005', 'atlocation', '?', {'current_time': 11}]], you can either use the memory "
            "['sta_005', 'atlocation', 'room_021', {'current_time': 11, 'timestamp': [10]}] or "
            "['sta_005', 'atlocation', 'room_030', {'strength': 3}] to answer the question. So your answer can be either "
            "'room_021' or 'room_030'. It's up to you to decide which one to use. "
            "You can even choose any other memory, if you think it's relevant and it can better answer the question! "
            "It's all up to you. "
            "Sometimes, you don't have a relevant memory to answer the question. In that case, you should take a good guess."
        )

        working_memory_str = str(memory_systems.get_working_memory().to_list())

        question_str = str(question[:-1] + [{"current_time": question[-1]}])

        prompt = (
            f"Now the is memory {working_memory_str}, and the question "
            f"is {question_str}. What is the answer? "
            f"Sometimes, you don't have a relevant memory to answer the question. "
            f"In that case, you should take a good guess."
            f"Only output the answer. Don't give me explanation. "
            f"Always give me the answer as 'The answer is XXX'."
        )

        answer = llm.talk_to_llm(prompt)
        try:
            answer = re.search(r"room_\S+(?=\.)", answer).group(0)
        except AttributeError:
            answer = None

        return answer

    else:  # this is symbolic reasoning
        query_idx = question.index("?")
        memory_object = memory_systems.get_working_memory().query(question[:-1] + ["?"])

        if len(memory_object) == 0:
            return None

        if qa_function.lower() == "random":
            return memory_object.retrieve_random_memory()[query_idx]

        mem_latest = memory_object.retrieve_memory_by_qualifier(
            "current_time", "int", "max"
        )

        if mem_latest is None:
            mem_latest = memory_object.retrieve_memory_by_qualifier(
                "timestamp", "list", "max", "max"
            )

        mem_strongest = memory_object.retrieve_memory_by_qualifier(
            "strength", "int", "max"
        )

        if qa_function.lower() == "latest_strongest":

            if mem_latest is not None:
                return mem_latest[query_idx]
            elif mem_strongest is not None:
                return mem_strongest[query_idx]
            else:
                return None

        elif qa_function.lower() == "strongest_latest":

            if mem_strongest is not None:
                return mem_strongest[query_idx]
            elif mem_latest is not None:
                return mem_latest[query_idx]
            else:
                return None

        elif qa_function.lower() == "latest":
            if mem_latest is not None:
                return mem_latest[query_idx]
            else:
                return None

        elif qa_function.lower() == "strongest":
            if mem_strongest is not None:
                return mem_strongest[query_idx]
            else:
                return None

        else:
            raise ValueError("Unknown policy.")
