from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import numpy as np


_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
_ARC_UTILS_SPEC = importlib.util.spec_from_file_location(
    "_arc_utils_fable_032_2", _ARC_UTILS_PATH
)
assert _ARC_UTILS_SPEC is not None and _ARC_UTILS_SPEC.loader is not None
_ARC_UTILS = importlib.util.module_from_spec(_ARC_UTILS_SPEC)
_ARC_UTILS_SPEC.loader.exec_module(_ARC_UTILS)

find_components = _ARC_UTILS.find_components
get_bg = _ARC_UTILS.get_bg
paste = _ARC_UTILS.paste
random_free_location = _ARC_UTILS.random_free_location


_DIRECTIONS = ("up", "down", "left", "right")
_GEN_CALL_COUNT = 0
_GEN_ATTEMPTS = 300
_GEN_SIZE_RANGE = (18, 29)


def _u_shape_coords(height: int, width: int, direction: str) -> set[tuple[int, int]]:
    h = int(height)
    w = int(width)
    if h < 3 or w < 3:
        return set()
    if direction == "up":
        return {(r, 0) for r in range(h)} | {(r, w - 1) for r in range(h)} | {
            (h - 1, c) for c in range(w)
        }
    if direction == "down":
        return {(r, 0) for r in range(h)} | {(r, w - 1) for r in range(h)} | {
            (0, c) for c in range(w)
        }
    if direction == "left":
        return {(0, c) for c in range(w)} | {(h - 1, c) for c in range(w)} | {
            (r, w - 1) for r in range(h)
        }
    if direction == "right":
        return {(0, c) for c in range(w)} | {(h - 1, c) for c in range(w)} | {
            (r, 0) for r in range(h)
        }
    raise ValueError(f"unknown direction: {direction}")


def _match_u_bracket(component: dict) -> str | None:
    r0, c0, r1, c1 = map(int, component["bbox"])
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return None
    coords = {(int(r) - r0, int(c) - c0) for r, c in component["coords"]}
    for direction in _DIRECTIONS:
        if coords == _u_shape_coords(height, width, direction):
            return direction
    return None


def _parse_brackets(grid: np.ndarray) -> tuple[int, list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    components = find_components(g, bg=bg, connectivity=4, monochromatic=True)

    brackets: list[dict] = []
    for component in components:
        direction = _match_u_bracket(component)
        if direction is None:
            continue
        bracket_coords = {(int(r), int(c)) for r, c in component["coords"]}
        r0, c0, r1, c1 = map(int, component["bbox"])
        markers = [
            (r, c)
            for r in range(r0, r1 + 1)
            for c in range(c0, c1 + 1)
            if (r, c) not in bracket_coords and int(g[r, c]) != bg
        ]
        if len(markers) != 1:
            continue
        marker = markers[0]
        brackets.append(
            {
                "direction": direction,
                "marker": marker,
                "marker_color": int(g[marker]),
                "coords": bracket_coords,
                "bbox": (r0, c0, r1, c1),
                "color": int(component["color"]),
            }
        )

    brackets.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    return bg, components, brackets


def _direction_selector(component: dict, direction: str) -> np.ndarray:
    coords = component["coords"]
    rows = coords[:, 0].astype(int)
    cols = coords[:, 1].astype(int)
    r0, c0, r1, c1 = map(int, component["bbox"])
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if direction == "up":
        return rows < r0 + height // 2
    if direction == "down":
        return rows >= r0 + height // 2
    if direction == "left":
        return cols < c0 + width // 2
    if direction == "right":
        return cols >= c0 + width // 2
    raise ValueError(f"unknown direction: {direction}")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase the direction-facing half of every object whose color is selected by a marker inside a U-shaped bracket.

    Input structure:
    - The background is the most common color.
    - There are one or two monochromatic 4-connected U-shaped brackets, each open in one cardinal direction and containing exactly one non-background marker pixel in its mouth.
    - Other monochromatic 4-connected objects are scattered elsewhere; some share a color with one of the markers.

    Rule:
    - For each bracket, read the marker color K and the bracket opening direction D.
    - For every object of color K except the protected marker pixel itself, erase that object's own bbox half on the D side by recoloring those object pixels to the background.
    - Preserve the brackets, the markers, and all non-selected objects.

    Key insight:
    - Each bracket encodes two independent parameters at once: the enclosed marker selects the target color, while the U opening selects which half of each matching object's own bbox is removed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg, components, brackets = _parse_brackets(g)
    assert 1 <= len(brackets) <= 2, "expected one or two valid U-brackets"

    protected = {tuple(map(int, bracket["marker"])) for bracket in brackets}
    for bracket in brackets:
        protected.update(bracket["coords"])

    out = g.copy()
    for bracket in brackets:
        marker_color = int(bracket["marker_color"])
        direction = bracket["direction"]
        for component in components:
            if int(component["color"]) != marker_color:
                continue
            comp_coords = {(int(r), int(c)) for r, c in component["coords"]}
            if comp_coords & protected:
                continue
            selector = _direction_selector(component, direction)
            coords = component["coords"][selector]
            if coords.size:
                out[coords[:, 0], coords[:, 1]] = bg

    return out


def _make_bracket_patch(
    height: int,
    width: int,
    direction: str,
    bracket_color: int,
    marker_color: int,
    bg: int,
) -> np.ndarray:
    patch = np.full((int(height), int(width)), int(bg), dtype=int)
    for r, c in _u_shape_coords(height, width, direction):
        patch[int(r), int(c)] = int(bracket_color)
    marker_r = random.randint(1, int(height) - 2)
    marker_c = random.randint(1, int(width) - 2)
    patch[marker_r, marker_c] = int(marker_color)
    return patch


def _row_profile_mask(height: int, width: int) -> np.ndarray:
    h = int(height)
    w = int(width)
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    max_inset = min(2, w - 1)
    for r in range(1, h - 1):
        left = random.randint(0, max_inset)
        right = random.randint(0, max_inset)
        if left + right >= w:
            right = max(0, w - left - 1)
        mask[r, left : w - right] = True
    return mask


def _random_object_patch(color: int, bg: int, direction: str, *, targeted: bool) -> np.ndarray:
    if direction in ("up", "down"):
        height = random.choice((4, 6, 8)) if targeted else random.randint(3, 7)
        width = random.randint(3, 6)
    else:
        height = random.randint(3, 6)
        width = random.choice((4, 6, 8)) if targeted else random.randint(3, 7)

    style = random.choice(("rect", "rows", "cols"))
    if style == "rect":
        mask = np.ones((height, width), dtype=bool)
    elif style == "rows":
        mask = _row_profile_mask(height, width)
    else:
        mask = _row_profile_mask(width, height).T

    patch = np.full(mask.shape, int(bg), dtype=int)
    patch[mask] = int(color)
    return patch


def _pick_directions(count: int) -> list[str]:
    global _GEN_CALL_COUNT
    primary = _DIRECTIONS[_GEN_CALL_COUNT % len(_DIRECTIONS)]
    _GEN_CALL_COUNT += 1
    if count == 1:
        return [primary]
    others = [direction for direction in _DIRECTIONS if direction != primary]
    picked = [primary, random.choice(others)]
    random.shuffle(picked)
    return picked


def _place_patch(grid: np.ndarray, patch: np.ndarray, bg: int, *, border: int = 1, padding: int = 1) -> None:
    r0, c0 = random_free_location(
        grid,
        patch,
        transparent=int(bg),
        border=int(border),
        padding=int(padding),
        padding_connectivity=8,
    )
    paste(grid, patch, int(r0), int(c0), transparent=int(bg))


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*_GEN_SIZE_RANGE)
    width = random.randint(*_GEN_SIZE_RANGE)
    bg = random.randint(0, 9)
    grid = np.full((height, width), bg, dtype=int)

    bracket_count = 1 if random.random() < 0.42 else 2
    directions = _pick_directions(bracket_count)

    colors = [color for color in range(10) if color != bg]
    random.shuffle(colors)
    marker_colors = colors[:bracket_count]
    bracket_colors = colors[bracket_count : 2 * bracket_count]
    remaining = colors[2 * bracket_count :]

    try:
        for direction, bracket_color, marker_color in zip(
            directions, bracket_colors, marker_colors
        ):
            bracket_h = random.randint(4, 7)
            bracket_w = random.randint(4, 7)
            patch = _make_bracket_patch(
                bracket_h, bracket_w, direction, bracket_color, marker_color, bg
            )
            _place_patch(grid, patch, bg, border=1, padding=1)

        for index, (direction, marker_color) in enumerate(zip(directions, marker_colors)):
            object_count = random.randint(2, 3) if bracket_count == 1 else random.randint(1, 2)
            if bracket_count == 2 and index == 0 and object_count == 1:
                object_count = 2
            for _ in range(object_count):
                patch = _random_object_patch(
                    marker_color, bg, direction, targeted=True
                )
                _place_patch(grid, patch, bg, border=1, padding=1)

        distractor_count = random.randint(2, min(4, len(remaining)))
        for distractor_color in remaining[:distractor_count]:
            direction = random.choice(_DIRECTIONS)
            patch = _random_object_patch(
                distractor_color, bg, direction, targeted=False
            )
            _place_patch(grid, patch, bg, border=1, padding=1)
    except ValueError:
        return None

    if int(np.count_nonzero(grid != bg)) >= grid.size // 2:
        return None
    if int(get_bg(grid)) != bg:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    changed = int(np.count_nonzero(out != grid))
    if changed < 4:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 凹形括号双参数选择器 input")
