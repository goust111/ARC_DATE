import random
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7
AZURE = 8
MAROON = 9

ARM_COLORS = (BLUE, RED, GREEN, YELLOW, ORANGE, AZURE)
ARM_STEPS = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}
DIAGONALS = [(-1, -1), (-1, 1), (1, 1), (1, -1)]
POINTER_TO_ARM = {
    (-1, -1): "up",
    (-1, 1): "right",
    (1, 1): "down",
    (1, -1): "left",
}
MAX_GENERATION_ATTEMPTS = 400


def _find_spinners(grid: np.ndarray) -> list[dict[str, object]]:
    h, w = map(int, grid.shape)
    spinners: list[dict[str, object]] = []
    forbidden_arm_colors = {BLACK, GREY, MAROON}

    for r in range(2, h - 2):
        for c in range(2, w - 2):
            if int(grid[r, c]) != GREY:
                continue

            arms: dict[str, int] = {}
            valid = True
            for name, (dr, dc) in ARM_STEPS.items():
                first = int(grid[r + dr, c + dc])
                second = int(grid[r + 2 * dr, c + 2 * dc])
                if first != second or first in forbidden_arm_colors:
                    valid = False
                    break
                arms[name] = first
            if not valid or len(set(arms.values())) != 4:
                continue

            pointer = None
            for dr, dc in DIAGONALS:
                value = int(grid[r + dr, c + dc])
                if value == MAROON:
                    if pointer is not None:
                        valid = False
                        break
                    pointer = (dr, dc)
                elif value != BLACK:
                    valid = False
                    break
            if not valid or pointer is None:
                continue

            spinners.append(
                {
                    "center": (r, c),
                    "arms": arms,
                    "pointer": pointer,
                    "read_color": arms[POINTER_TO_ARM[pointer]],
                }
            )

    assert spinners, "at least one spinner is required"
    return spinners


def _template_pattern_matches(subgrid: np.ndarray) -> bool:
    if subgrid.shape[0] != 3 or subgrid.shape[1] % 2 != 1:
        return False
    if not np.all(subgrid[0] == GREY):
        return False
    if not np.all(subgrid[2] == GREY):
        return False
    if not np.all(subgrid[1, 0::2] == GREY):
        return False
    if not np.all(subgrid[1, 1::2] == BLACK):
        return False
    return True


def _find_template(grid: np.ndarray, count: int) -> tuple[int, int]:
    h, w = map(int, grid.shape)
    width = 2 * count + 1
    candidates: list[tuple[int, int]] = []

    for top in range(h - 2):
        for left in range(w - width + 1):
            subgrid = grid[top : top + 3, left : left + width]
            if not _template_pattern_matches(subgrid):
                continue
            touches_corner = (
                (top == 0 and left == 0)
                or (top == 0 and left + width == w)
                or (top + 3 == h and left == 0)
                or (top + 3 == h and left + width == w)
            )
            if touches_corner:
                candidates.append((top, left))

    assert len(candidates) == 1, "template must be unique and corner-anchored"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read each spinner's clockwise-next arm color and write the sequence into the corner template.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    spinners = _find_spinners(g)
    # The prompt states "from left to right, from top to bottom", so sort by column, then row.
    spinners.sort(key=lambda item: (int(item["center"][1]), int(item["center"][0])))
    top, left = _find_template(g, len(spinners))

    out = g.copy()
    for index, spinner in enumerate(spinners):
        out[top + 1, left + 1 + 2 * index] = int(spinner["read_color"])
    return out


def _template_origin(height: int, width: int, count: int, corner: str) -> tuple[int, int]:
    template_width = 2 * count + 1
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, width - template_width
    if corner == "bl":
        return height - 3, 0
    if corner == "br":
        return height - 3, width - template_width
    raise ValueError(f"unknown corner {corner}")


def _paint_template(grid: np.ndarray, top: int, left: int, count: int) -> None:
    width = 2 * count + 1
    grid[top, left : left + width] = GREY
    grid[top + 2, left : left + width] = GREY
    grid[top + 1, left : left + width : 2] = GREY


def _reserve(mask: np.ndarray, top: int, left: int, bottom: int, right: int, margin: int = 1) -> None:
    h, w = mask.shape
    r0 = max(0, top - margin)
    c0 = max(0, left - margin)
    r1 = min(h, bottom + margin)
    c1 = min(w, right + margin)
    mask[r0:r1, c0:c1] = True


def _place_spinner(
    grid: np.ndarray,
    center: tuple[int, int],
    arms: dict[str, int],
    pointer: tuple[int, int],
) -> None:
    r, c = center
    grid[r, c] = GREY
    for name, (dr, dc) in ARM_STEPS.items():
        color = arms[name]
        grid[r + dr, c + dc] = color
        grid[r + 2 * dr, c + 2 * dc] = color
    pr, pc = pointer
    grid[r + pr, c + pc] = MAROON


def _changed_cells(before: np.ndarray, after: np.ndarray) -> set[tuple[int, int]]:
    coords = np.argwhere(before != after)
    return {(int(r), int(c)) for r, c in coords}


def _template_fill_cells(top: int, left: int, count: int) -> set[tuple[int, int]]:
    return {(top + 1, left + 1 + 2 * idx) for idx in range(count)}


def _candidate_centers(height: int, width: int) -> list[tuple[int, int]]:
    centers = [(r, c) for r in range(2, height - 2) for c in range(2, width - 2)]
    random.shuffle(centers)
    return centers


def _sample_colors() -> dict[str, int]:
    palette = random.sample(ARM_COLORS, 4)
    return dict(zip(("up", "right", "down", "left"), palette))


def _gen_attempt() -> np.ndarray | None:
    count = random.randint(4, 6)
    template_width = 2 * count + 1

    min_height = 13
    min_width = max(15, template_width + 4)
    height = random.randint(min_height, 24)
    width = random.randint(min_width, 28)

    grid = np.full((height, width), BLACK, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)

    corner = random.choice(("tl", "tr", "bl", "br"))
    template_top, template_left = _template_origin(height, width, count, corner)
    _paint_template(grid, template_top, template_left, count)
    _reserve(occupied, template_top, template_left, template_top + 3, template_left + template_width)

    pointers = DIAGONALS.copy()
    while len(pointers) < count:
        pointers.append(random.choice(DIAGONALS))
    random.shuffle(pointers)

    centers: list[tuple[int, int]] = []
    for center in _candidate_centers(height, width):
        r, c = center
        if occupied[r, c]:
            continue
        top = r - 2
        left = c - 2
        bottom = r + 3
        right = c + 3
        if occupied[top:bottom, left:right].any():
            continue
        centers.append(center)
        _reserve(occupied, top, left, bottom, right, margin=1)
        if len(centers) == count:
            break

    if len(centers) != count:
        return None

    for center, pointer in zip(centers, pointers):
        _place_spinner(grid, center, _sample_colors(), pointer)

    try:
        solved = main(grid)
    except AssertionError:
        return None

    changed = _changed_cells(grid, solved)
    expected_changed = _template_fill_cells(template_top, template_left, count)
    if changed != expected_changed:
        return None
    if any(int(solved[r, c]) in {BLACK, GREY} for r, c in expected_changed):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid spinner-template input")
