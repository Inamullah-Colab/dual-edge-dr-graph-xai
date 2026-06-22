from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LesionClass:
    """Named lesion evidence class used by XAI/contrastive modules.

    These labels define model-derived lesion evidence channels. They are not assumed to be
    ophthalmologist-confirmed lesion masks unless the user provides expert annotations.
    """

    name: str
    channel: int
    display_name: str
    color_rgb: tuple[int, int, int]


LESION_CLASSES: tuple[LesionClass, ...] = (
    LesionClass("microaneurysm", 0, "Microaneurysm", (220, 40, 50)),
    LesionClass("hemorrhage", 1, "Hemorrhage", (150, 25, 40)),
    LesionClass("hard_exudate", 2, "Hard exudate", (245, 190, 40)),
    LesionClass("cotton_wool_spot", 3, "Cotton-wool spot", (235, 235, 235)),
    LesionClass("neovascularization", 4, "Neovascularization", (110, 50, 180)),
)

LESION_TO_CHANNEL = {x.name: x.channel for x in LESION_CLASSES}
CHANNEL_TO_LESION = {x.channel: x.name for x in LESION_CLASSES}


def lesion_names() -> list[str]:
    return [x.name for x in LESION_CLASSES]
