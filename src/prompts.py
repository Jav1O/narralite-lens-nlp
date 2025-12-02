from dataclasses import dataclass
from typing import List


@dataclass
class Character:
    name: str
    traits: List[str]


@dataclass
class Scenario:
    name: str
    setting: str
    characters: List[Character]
    instruction: str = (
        "Write the beginning of a short story that includes all the characters."
    )


def build_prompt(scenario: Scenario) -> str:
    """
    Build a textual prompt from a Scenario:
    - list of characters with traits
    - setting description
    - task instruction
    """
    lines = ["Characters:"]
    for ch in scenario.characters:
        traits_str = ", ".join(ch.traits)
        lines.append(f"- {ch.name}: {traits_str}")

    lines.append("\nSetting:")
    lines.append(scenario.setting)

    lines.append("\nTask:")
    lines.append(scenario.instruction)

    return "\n".join(lines)


def example_scenarios() -> List[Scenario]:
    """
    A few example scenarios to start with.
    You can edit or add more later.
    """
    return [
        Scenario(
            name="enchanted_forest",
            setting="An enchanted forest at night.",
            characters=[
                Character("Luna", ["a timid and curious explorer"]),
                Character("Orion", ["a brave and impulsive warrior"]),
            ],
        ),
        Scenario(
            name="space_station",
            setting="A quiet space station orbiting a distant planet.",
            characters=[
                Character("Mira", ["a wise and mysterious scientist"]),
                Character("Kai", ["a reckless and impatient pilot"]),
            ],
        ),
    ]
