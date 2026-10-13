import random

import numpy as np


AZURE = 8
YELLOW = 4
GREY = 5
OBJECT_COLORS = (0, 1, 2, 3, 6, 7, 9)
MAX_GENERATION_ATTEMPTS = 400
MAX_GRID_SIDE = 30


def _extract_objects(grid: np.ndarray) -> list[dict[str, int]]:
    g = np.asarray(grid, dtype=int)
    ground_r = g.shape[0] - 1
    objects: list[dict[str, int]] = []

    for color in sorted(int(c) for c in np.unique(g[:-1])):
        if color in {AZURE, GREY, YELLOW}:
            continue
        coords = np.argwhere(g == color)
        if coords.size == 0:
            continue
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
        if r1 != ground_r - 1:
            raise AssertionError("objects must stand on the ground row")
        if not np.all(g[r0 : r1 + 1, c0 : c1 + 1] == color):
            raise AssertionError("each object must be a solid rectangle")
        objects.append(
            {
                "color": int(color),
                "top": r0,
                "left": c0,
                "right": c1,
                "height": r1 - r0 + 1,
            }
        )

    objects.sort(key=lambda obj: (obj["left"], obj["top"]))
    return objects


def _shadow_lengths(objects: list[dict[str, int]], width: int) -> list[int]:
    lengths: list[int] = []
    for idx, obj in enumerate(objects):
        start = obj["right"]
        end = min(start + obj["height"] - 1, width - 1)
        if idx + 1 < len(objects):
            end = min(end, objects[idx + 1]["left"] - 1)
        lengths.append(max(0, end - start + 1))
    return lengths


def _render_shadow_output(grid: np.ndarray, objects: list[dict[str, int]]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    ground_r = out.shape[0] - 1
    for obj, length in zip(objects, _shadow_lengths(objects, out.shape[1])):
        if length <= 0:
            continue
        start = obj["right"]
        out[ground_r, start : start + length] = YELLOW
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each standing rectangle's shadow to the right along the ground.

    The input uses azure background, a single grey ground row at the bottom, and
    solid colored rectangles standing on that ground. Each rectangle casts a
    yellow shadow whose horizontal length equals the rectangle height. Shadows
    live only on the ground row and are truncated when they would reach the
    footprint of the next object to the right.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or g.shape[0] < 2 or g.shape[1] < 1:
        raise AssertionError("expected a non-empty 2D grid with a ground row")

    objects = _extract_objects(g)
    return _render_shadow_output(g, objects)


def _choose_gap_kinds(mode: str, n_objects: int) -> list[str]:
    if n_objects < 2:
        return []
    if mode == "all_open":
        return ["open"] * (n_objects - 1)
    if mode == "all_blocked":
        return ["blocked"] * (n_objects - 1)
    while True:
        gap_kinds = [random.choice(("open", "blocked")) for _ in range(n_objects - 1)]
        if "open" in gap_kinds and "blocked" in gap_kinds:
            return gap_kinds


def _sample_heights(gap_kinds: list[str], n_objects: int) -> list[int]:
    max_height = random.randint(4, 6)
    heights: list[int] = []
    for idx in range(n_objects):
        min_height = 3 if idx < len(gap_kinds) and gap_kinds[idx] == "blocked" else 1
        heights.append(random.randint(min_height, max_height))

    if n_objects > 1 and len(set(heights)) == 1:
        idx = random.randrange(n_objects)
        lower = 3 if idx < len(gap_kinds) and gap_kinds[idx] == "blocked" else 1
        choices = [h for h in range(lower, max_height + 1) if h != heights[idx]]
        if choices:
            heights[idx] = random.choice(choices)
    return heights


def _sample_gaps(heights: list[int], gap_kinds: list[str]) -> list[int]:
    gaps: list[int] = []
    for height, kind in zip(heights[:-1], gap_kinds):
        if kind == "blocked":
            gaps.append(random.randint(1, height - 2))
            continue
        min_gap = max(1, height - 1)
        gaps.append(random.randint(min_gap, min_gap + 3))
    return gaps


def _build_input_from_layout(
    heights: list[int],
    widths: list[int],
    gaps: list[int],
    colors: list[int],
    top_margin: int,
    left_margin: int,
    right_margin: int,
) -> tuple[np.ndarray, list[dict[str, int]]]:
    height = top_margin + max(heights) + 1
    width = left_margin + sum(widths) + sum(gaps) + right_margin
    if height > MAX_GRID_SIDE or width > MAX_GRID_SIDE:
        raise AssertionError("grid too large")

    grid = np.full((height, width), AZURE, dtype=int)
    grid[-1, :] = GREY

    ground_r = height - 1
    objects: list[dict[str, int]] = []
    col = left_margin
    for idx, (obj_h, obj_w, color) in enumerate(zip(heights, widths, colors)):
        grid[ground_r - obj_h : ground_r, col : col + obj_w] = color
        objects.append(
            {
                "color": int(color),
                "top": ground_r - obj_h,
                "left": col,
                "right": col + obj_w - 1,
                "height": obj_h,
            }
        )
        col += obj_w
        if idx < len(gaps):
            col += gaps[idx]

    return grid, objects


def _mode_is_valid(mode: str, objects: list[dict[str, int]], width: int) -> bool:
    if len(objects) < 2:
        return False
    lengths = _shadow_lengths(objects, width)
    blocked = [length < obj["height"] for obj, length in zip(objects[:-1], lengths[:-1])]
    if mode == "all_open":
        return not any(blocked)
    if mode == "all_blocked":
        return bool(blocked) and all(blocked)
    return any(blocked) and not all(blocked)


def _gen_attempt() -> np.ndarray | None:
    mode = random.choice(("all_open", "all_blocked", "mixed"))
    if mode == "mixed":
        n_objects = random.randint(3, 5)
    else:
        n_objects = random.randint(2, 5)

    gap_kinds = _choose_gap_kinds(mode, n_objects)
    heights = _sample_heights(gap_kinds, n_objects)
    widths = random.sample(range(1, 7), n_objects)
    random.shuffle(widths)
    colors = random.sample(list(OBJECT_COLORS), n_objects)

    gaps = _sample_gaps(heights, gap_kinds)
    top_margin = random.randint(1, 3)
    left_margin = random.randint(1, 3)
    min_right_margin = max(1, heights[-1] - 1)
    right_margin = random.randint(min_right_margin, min_right_margin + 3)

    try:
        grid, objects = _build_input_from_layout(
            heights=heights,
            widths=widths,
            gaps=gaps,
            colors=colors,
            top_margin=top_margin,
            left_margin=left_margin,
            right_margin=right_margin,
        )
    except AssertionError:
        return None

    if not _mode_is_valid(mode, objects, grid.shape[1]):
        return None

    expected = _render_shadow_output(grid, objects)
    try:
        actual = main(grid)
    except AssertionError:
        return None

    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(actual, grid):
        return None
    if len(_extract_objects(grid)) != n_objects:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid shadow-projection grid")
