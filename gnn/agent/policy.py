"""Handcrafted / trained policies.

The trained neural network policies are not implemented yet.
"""

import random
from typing import Literal

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
    policy: Literal["random", "avoid_walls"],
) -> str:
    """Explore the room (sub-graph).

    "random" policy: randomly choose an action.
    "avoid_walls" policy: avoid the walls. This takes into account the agent's current
        location (from episodic) and the memories about the map (from semantic and
        episodic)

    Args:
        memory_systems: MemorySystems
        policy: "random" or "avoid_walls"

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

    else:
        raise ValueError("Unknown exploration policy.")

    assert action in ["north", "east", "south", "west", "stay"]

    return action


def manage_memory(
    memory_systems: MemorySystems,
    policy: Literal["episodic", "semantic", "random", "forget", "handcrafted"],
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
    assert policy.lower() in ["episodic", "semantic", "random", "forget", "handcrafted"]
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

    elif policy.lower() == "random":
        if random.choice(["episodic", "semantic"]) == "episodic":
            move_to_episodic(memory_systems, mem_short)
        else:
            move_to_semantic(memory_systems, mem_short)

    elif policy.lower() == "handcrafted":
        if mem_short[0] == "agent":
            move_to_episodic(memory_systems, mem_short)
        elif "ind" in mem_short[0] or "dep" in mem_short[0]:
            move_to_episodic(memory_systems, mem_short)
        elif "sta" in mem_short[0]:
            move_to_semantic(memory_systems, mem_short)
        elif "room" in mem_short[0] and "room" in mem_short[2]:
            move_to_semantic(memory_systems, mem_short)
        elif "wall" in mem_short[2]:
            pass
        else:
            raise ValueError("something is wrong")

    elif policy.lower() == "forget":
        pass

    else:
        raise ValueError

    memory_systems.short.forget(mem_short)  # don't forget to forget the short memory!


def answer_question(
    memory_systems: MemorySystems,
    qa_function: Literal["latest_strongest", "latest", "strongest", "random"],
    question: list[str | int],
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
        qa_function: "latest_strongest", "latest", "strongest", or "random"
        question: A quadruple given by RoomEnv-v2, e.g., [laptop, atlocation, ?,
            current_time]

    Returns:
        pred: prediction

    """
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

    mem_strongest = memory_object.retrieve_memory_by_qualifier("strength", "int", "max")

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
