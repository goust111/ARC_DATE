from __future__ import annotations

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg, paste
import random


def _frame_interior_area(g: np.ndarray, bg: int, obj: dict) -> tuple[int, int]:
    color = int(obj["color"])
    r0, c0, r1, c1 = obj["bbox"]
    height = bbox_h(obj["bbox"])
    width = bbox_w(obj["bbox"])
    assert height >= 3 and width >= 3, "expected a frame with a nonempty interior"

    sub = g[r0 : r1 + 1, c0 : c1 + 1]
    border = np.zeros((height, width), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    assert np.all(sub[border] == color), "expected a complete monochrome frame border"
    assert np.all(sub[~border] == bg), "expected frame interior to be background"
    assert int(obj["area"]) == int(border.sum()), "expected a one-cell-thick rectangular frame"

    return int((height - 2) * (width - 2)), color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Output a 2x2 block colored like the hollow rectangular frame with the largest interior.

    Input structure
    - The background is the most common color.
    - The foreground consists of exactly two 4-connected monochrome hollow rectangular frames.
    - Each frame is one cell thick and has a background-colored interior.

    Rule
    - Measure each frame by the area strictly inside its border.
    - Select the unique frame with the largest interior area.
    - Return a 2x2 grid filled with that frame's color.

    Key insight
    - The task compares the size of the empty regions enclosed by the frames; the
      output is only the winning frame color, normalized to a small solid block.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert min(g.shape) > 0, "expected a nonempty grid"

    bg = get_bg(g)
    objects = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    assert len(objects) == 2, "expected exactly two foreground frames"

    frames = [_frame_interior_area(g, bg, obj) for obj in objects]
    largest_area = max(area for area, _color in frames)
    largest = [(area, color) for area, color in frames if area == largest_area]
    assert len(largest) == 1, "expected a unique largest frame interior"

    _area, color = largest[0]
    return np.full((2, 2), color, dtype=int)

# --------------- generator (o5x) ---------------

# Canvas and color bounds.
GRID_SIZE, BACKGROUND_COLOR, OUTPUT_SIZE = 10, 0, 2
NONZERO_COLORS = range(1, 10)

# Frame size and separation constraints.
FRAME_MIN_SIZE, FRAME_MAX_SIZE = 3, GRID_SIZE
FRAME_DIMENSIONS = range(FRAME_MIN_SIZE, FRAME_MAX_SIZE + 1)
MIN_FRAME_GAP = 1
MAX_FRAME_PAIR_SPAN = GRID_SIZE - MIN_FRAME_GAP

# Layout choices and retry budget.
SIDE_BY_SIDE, STACKED = "side_by_side", "stacked"
LAYOUTS = (SIDE_BY_SIDE, STACKED)
GENERATION_RETRIES = 300


def _gen_interior_area(h: int, w: int) -> int:
    return (h - 2) * (w - 2)


def _gen_dimension_candidates(layout: str) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    candidates = []
    for h1 in FRAME_DIMENSIONS:
        for w1 in FRAME_DIMENSIONS:
            for h2 in FRAME_DIMENSIONS:
                for w2 in FRAME_DIMENSIONS:
                    span = w1 + w2 if layout == SIDE_BY_SIDE else h1 + h2
                    if span <= MAX_FRAME_PAIR_SPAN and _gen_interior_area(h1, w1) != _gen_interior_area(h2, w2):
                        candidates.append(((h1, w1), (h2, w2)))
    return candidates


DIMENSION_CANDIDATES = {layout: _gen_dimension_candidates(layout) for layout in LAYOUTS}


def _gen_frame_patch(h: int, w: int, color: int) -> np.ndarray:
    patch = np.zeros((h, w), dtype=int)
    patch[[0, -1], :] = color
    patch[:, [0, -1]] = color
    return patch


def _gen_choose_frames() -> tuple[list[dict], str]:
    layout = random.choice(LAYOUTS)
    colors = random.sample(NONZERO_COLORS, 2)
    (h1, w1), (h2, w2) = random.choice(DIMENSION_CANDIDATES[layout])
    frames = [
        {"h": h1, "w": w1, "color": colors[0], "area": _gen_interior_area(h1, w1)},
        {"h": h2, "w": w2, "color": colors[1], "area": _gen_interior_area(h2, w2)},
    ]
    random.shuffle(frames)
    return frames, layout


def _gen_place_frames(grid: np.ndarray, frames: list[dict], layout: str) -> None:
    first, second = frames
    if layout == SIDE_BY_SIDE:
        gap = random.randint(MIN_FRAME_GAP, GRID_SIZE - first["w"] - second["w"])
        c0 = random.randint(0, GRID_SIZE - first["w"] - gap - second["w"])
        first["r"] = random.randint(0, GRID_SIZE - first["h"])
        first["c"] = c0
        second["r"] = random.randint(0, GRID_SIZE - second["h"])
        second["c"] = c0 + first["w"] + gap
    else:
        gap = random.randint(MIN_FRAME_GAP, GRID_SIZE - first["h"] - second["h"])
        r0 = random.randint(0, GRID_SIZE - first["h"] - gap - second["h"])
        first["r"] = r0
        first["c"] = random.randint(0, GRID_SIZE - first["w"])
        second["r"] = r0 + first["h"] + gap
        second["c"] = random.randint(0, GRID_SIZE - second["w"])

    for frame in frames:
        patch = _gen_frame_patch(frame["h"], frame["w"], frame["color"])
        paste(grid, patch, frame["r"], frame["c"], transparent=BACKGROUND_COLOR)


def _gen_attempt() -> np.ndarray | None:
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    frames, layout = _gen_choose_frames()
    _gen_place_frames(grid, frames, layout)

    winner = max(frames, key=lambda frame: frame["area"])
    expected = np.full((OUTPUT_SIZE, OUTPUT_SIZE), winner["color"], dtype=int)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if np.array_equal(actual, expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for 445eab21")
