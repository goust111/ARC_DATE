from __future__ import annotations

from collections import Counter
import random

import numpy as np


RING_SIZE = 5
INNER_SLICE = (slice(1, 4), slice(1, 4))
CORNER_POS = ((0, 0), (0, 4), (4, 4), (4, 0))
SIDE_POS = ((0, 2), (2, 4), (4, 2), (2, 0))  # top, right, bottom, left
RING_POS = CORNER_POS + SIDE_POS
RING_POS_SET = set(RING_POS)
NON_RING_MASK = np.ones((RING_SIZE, RING_SIZE), dtype=bool)
for _r, _c in RING_POS:
    NON_RING_MASK[_r, _c] = False

GAP_NAMES = ("top", "right", "bottom", "left")
GRID_BOUNDS = (18, 25)
RING_COUNT_CHOICES = (3, 4)
MIN_BOX_GAP = 2
MAX_GEN_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    """Mirror the shared ARC toolbox convention: background is the most common color."""
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _cycle(values: list[int], step: int) -> list[int]:
    n = len(values)
    return [int(values[(i - step) % n]) for i in range(n)]


def _bbox_for(origin: tuple[int, int]) -> tuple[int, int, int, int]:
    r0, c0 = origin
    return r0, c0, r0 + RING_SIZE - 1, c0 + RING_SIZE - 1


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return (
        ar1 + MIN_BOX_GAP < br0
        or br1 + MIN_BOX_GAP < ar0
        or ac1 + MIN_BOX_GAP < bc0
        or bc1 + MIN_BOX_GAP < ac0
    )


def _ring_from_patch(patch: np.ndarray, r0: int, c0: int) -> dict | None:
    if patch.shape != (RING_SIZE, RING_SIZE):
        return None
    if np.any(patch[NON_RING_MASK] != 0):
        return None

    corners = [int(patch[r, c]) for r, c in CORNER_POS]
    if any(color == 0 for color in corners):
        return None

    side_vals = [int(patch[r, c]) for r, c in SIDE_POS]
    gap_indices = [i for i, color in enumerate(side_vals) if color == 0]
    if len(gap_indices) != 1:
        return None
    gap_idx = int(gap_indices[0])

    visible_side_vals = [color for color in side_vals if color != 0]
    if len(set(visible_side_vals)) != 1:
        return None
    side_color = int(visible_side_vals[0])
    if any(color == side_color for color in corners):
        return None

    coords = {(r0 + r, c0 + c) for r, c in CORNER_POS}
    coords |= {
        (r0 + r, c0 + c)
        for i, (r, c) in enumerate(SIDE_POS)
        if i != gap_idx
    }
    return {
        "origin": (r0, c0),
        "gap_idx": gap_idx,
        "gap_name": GAP_NAMES[gap_idx],
        "corner_colors": corners,
        "side_color": side_color,
        "coords": coords,
    }


def _detect_rings(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    rings: list[dict] = []
    covered: set[tuple[int, int]] = set()

    for r0 in range(H - RING_SIZE + 1):
        for c0 in range(W - RING_SIZE + 1):
            ring = _ring_from_patch(g[r0 : r0 + RING_SIZE, c0 : c0 + RING_SIZE], r0, c0)
            if ring is None:
                continue
            if ring["coords"] & covered:
                raise AssertionError("ring pixels must belong to exactly one sparse ring")
            covered |= ring["coords"]
            rings.append(ring)

    assert rings, "expected at least one sparse 5x5 ring"
    foreground = {tuple(map(int, rc)) for rc in np.argwhere(g != 0)}
    assert covered == foreground, "all non-black pixels must be explained by rings"
    return rings


def _transform_ring(
    out: np.ndarray,
    origin: tuple[int, int],
    corner_colors: list[int],
    side_color: int,
    gap_idx: int,
) -> None:
    r0, c0 = origin
    corner_step = 1 if gap_idx in (0, 1) else -1
    side_step = -corner_step

    rotated_corners = _cycle(corner_colors, corner_step)
    rotated_sides = _cycle([side_color] * 4, side_step)

    for (dr, dc), color in zip(CORNER_POS, rotated_corners):
        out[r0 + dr, c0 + dc] = int(color)
    for i, (dr, dc) in enumerate(SIDE_POS):
        out[r0 + dr, c0 + dc] = 0 if i == gap_idx else int(rotated_sides[i])

    out[r0 + 1 : r0 + 4, c0 + 1 : c0 + 4] = int(side_color)


def _transform_ring_with_step(
    out: np.ndarray,
    origin: tuple[int, int],
    corner_colors: list[int],
    side_color: int,
    gap_idx: int,
    corner_step: int,
) -> None:
    r0, c0 = origin
    side_step = -corner_step

    rotated_corners = _cycle(corner_colors, corner_step)
    rotated_sides = _cycle([side_color] * 4, side_step)

    for (dr, dc), color in zip(CORNER_POS, rotated_corners):
        out[r0 + dr, c0 + dc] = int(color)
    for i, (dr, dc) in enumerate(SIDE_POS):
        out[r0 + dr, c0 + dc] = 0 if i == gap_idx else int(rotated_sides[i])

    out[r0 + 1 : r0 + 4, c0 + 1 : c0 + 4] = int(side_color)


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30
    assert get_bg(g) == 0, "expected black background"

    out = g.copy()
    for ring in _detect_rings(g):
        _transform_ring(
            out,
            ring["origin"],
            ring["corner_colors"],
            ring["side_color"],
            ring["gap_idx"],
        )
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate each one-gap sparse ring according to its gap direction, then flood its interior with the hidden side color.

    Input structure:
    - The grid has a black background and several separated sparse 5x5 rings.
    - Each ring uses the four 5x5 corners plus the four side-center positions, with exactly one side-center missing.
    - The three visible side-centers of a ring share one non-black color; the four corners are non-black corner pixels.

    Rule:
    - Find the missing side-center of each ring; top/right gaps mean the corners move one step clockwise, while bottom/left gaps mean the corners move one step counterclockwise.
    - The four side-center slots participate in the opposite one-step cycle, treating the missing slot as a virtual side pixel whose color is the shared side color, but the gap remains black in the output.
    - Fill the enclosed 3x3 interior with the side color that occupied the virtual gap slot before the cycle.

    Key insight:
    - The missing side-center simultaneously reveals the rotation direction and the hidden fill color, so solving the task requires reading the sparse 5x5 ring in its own coordinate system.
    """
    return _solve(grid)


def _expected_from_specs(grid: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for spec in specs:
        _transform_ring(
            out,
            spec["origin"],
            spec["corner_colors"],
            spec["side_color"],
            spec["gap_idx"],
        )
    return out


def _fill_only_from_specs(grid: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for spec in specs:
        r0, c0 = spec["origin"]
        out[r0 + 1 : r0 + 4, c0 + 1 : c0 + 4] = int(spec["side_color"])
    return out


def _wrong_direction_from_specs(grid: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for spec in specs:
        correct_step = 1 if spec["gap_idx"] in (0, 1) else -1
        _transform_ring_with_step(
            out,
            spec["origin"],
            spec["corner_colors"],
            spec["side_color"],
            spec["gap_idx"],
            -correct_step,
        )
    return out


def _make_ring_spec(origin: tuple[int, int], gap_idx: int, side_color: int) -> dict:
    corner_palette = [color for color in range(1, 10) if color != side_color]
    corner_colors = random.sample(corner_palette, 4)
    return {
        "origin": origin,
        "gap_idx": int(gap_idx),
        "corner_colors": [int(color) for color in corner_colors],
        "side_color": int(side_color),
    }


def _render_specs(shape: tuple[int, int], specs: list[dict]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for spec in specs:
        r0, c0 = spec["origin"]
        for (dr, dc), color in zip(CORNER_POS, spec["corner_colors"]):
            grid[r0 + dr, c0 + dc] = int(color)
        for i, (dr, dc) in enumerate(SIDE_POS):
            if i != spec["gap_idx"]:
                grid[r0 + dr, c0 + dc] = int(spec["side_color"])
    return grid


def _random_positions(height: int, width: int, count: int) -> list[tuple[int, int]] | None:
    positions = [(r, c) for r in range(height - RING_SIZE + 1) for c in range(width - RING_SIZE + 1)]
    random.shuffle(positions)
    chosen: list[tuple[int, int]] = []
    boxes: list[tuple[int, int, int, int]] = []

    for pos in positions:
        box = _bbox_for(pos)
        if all(_boxes_separated(box, other) for other in boxes):
            chosen.append(pos)
            boxes.append(box)
            if len(chosen) == count:
                return chosen
    return None


def _semantic_checks(inp: np.ndarray, out: np.ndarray, specs: list[dict]) -> bool:
    if np.array_equal(inp, out):
        return False
    if np.array_equal(out, _fill_only_from_specs(inp, specs)):
        return False
    if np.array_equal(out, _wrong_direction_from_specs(inp, specs)):
        return False

    detected = _detect_rings(inp)
    by_origin = {ring["origin"]: ring for ring in detected}
    for spec in specs:
        ring = by_origin.get(spec["origin"])
        if ring is None:
            return False
        if ring["gap_idx"] != spec["gap_idx"] or ring["side_color"] != spec["side_color"]:
            return False
        if ring["corner_colors"] == _cycle(ring["corner_colors"], 1):
            return False

        r0, c0 = spec["origin"]
        expected_corners = _cycle(spec["corner_colors"], 1 if spec["gap_idx"] in (0, 1) else -1)
        actual_corners = [int(out[r0 + dr, c0 + dc]) for dr, dc in CORNER_POS]
        if actual_corners != expected_corners:
            return False
        if not np.all(out[r0 + 1 : r0 + 4, c0 + 1 : c0 + 4] == spec["side_color"]):
            return False
        gap_r, gap_c = SIDE_POS[spec["gap_idx"]]
        if int(out[r0 + gap_r, c0 + gap_c]) != 0:
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    count = random.choice(RING_COUNT_CHOICES)
    height = random.randint(*GRID_BOUNDS)
    width = random.randint(*GRID_BOUNDS)
    positions = _random_positions(height, width, count)
    if positions is None:
        return None

    gap_indices = random.sample(range(4), count) if count == 4 else random.sample(range(4), 3)
    random.shuffle(gap_indices)
    side_colors = random.sample(range(1, 10), count)
    specs = [
        _make_ring_spec(origin, gap_idx, side_color)
        for origin, gap_idx, side_color in zip(positions, gap_indices, side_colors)
    ]

    inp = _render_specs((height, width), specs)
    expected = _expected_from_specs(inp, specs)

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    if not _semantic_checks(inp, expected, specs):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sparse-ring ARC input")
