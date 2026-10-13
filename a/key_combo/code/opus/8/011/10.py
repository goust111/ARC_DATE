from __future__ import annotations

import random
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np


_ARC_UTILS = SourceFileLoader(
    "_arc_utils_shared",
    str(Path("/home/lixinyu/snap/new/solvers/arc_utils.py")),
).load_module()

Color = _ARC_UTILS.Color
find_components = _ARC_UTILS.find_components


COOL_GROUP = {Color.BLUE, Color.TEAL}
WARM_GROUP = {Color.YELLOW, Color.ORANGE}
SMALL_COLOR = Color.GREY
LARGE_COLOR = Color.MAROON

COOL_PAIRS = (
    (Color.BLUE, Color.TEAL),
    (Color.TEAL, Color.BLUE),
)
WARM_PAIRS = (
    (Color.YELLOW, Color.ORANGE),
    (Color.ORANGE, Color.YELLOW),
)
CROSS_FAMILY_DISTRACTORS = (
    (Color.BLUE, Color.YELLOW),
    (Color.YELLOW, Color.BLUE),
    (Color.BLUE, Color.ORANGE),
    (Color.ORANGE, Color.BLUE),
    (Color.TEAL, Color.YELLOW),
    (Color.YELLOW, Color.TEAL),
    (Color.TEAL, Color.ORANGE),
    (Color.ORANGE, Color.TEAL),
)
OUTSIDER_DISTRACTORS = (
    (Color.RED, Color.GREEN),
    (Color.GREEN, Color.RED),
    (Color.RED, Color.PINK),
    (Color.PINK, Color.RED),
    (Color.GREEN, Color.PINK),
    (Color.PINK, Color.GREEN),
)

GEN_ATTEMPTS = 400


def _row_kind(colors: tuple[int, int]) -> str | None:
    a, b = map(int, colors)
    pair = {a, b}
    if pair <= COOL_GROUP:
        return "cool"
    if pair <= WARM_GROUP:
        return "warm"
    return None


def _non_bg_columns(grid: np.ndarray, row: int) -> np.ndarray:
    return np.flatnonzero(grid[int(row)] != Color.BLACK)


def _slide_pairs(grid: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    height, _width = out.shape
    for r in range(height):
        cols = _non_bg_columns(out, r)
        if cols.size == 0:
            continue
        assert cols.size == 2, "each non-empty row must contain exactly two pixels"
        left, right = map(int, cols)
        colors = (int(out[r, left]), int(out[r, right]))
        if _row_kind(colors) is None:
            continue
        right_color = int(out[r, right])
        target = left + 1
        out[r, right] = Color.BLACK
        out[r, target] = right_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each same-family pixel pair together, then recolor every resulting component by an area threshold.

    Input structure:
    - The background is black.
    - Some rows are empty; each non-empty row contains exactly two non-background single pixels.
    - A row is movable iff its two colors are both in the blue/teal family or both in the yellow/orange family.
    - Other two-pixel rows are distractors and stay fixed.

    Rule:
    - For every movable row, keep the left pixel fixed and slide the right pixel left until it occupies the cell immediately to the right of the left pixel.
    - Erase the right pixel's original location; distractor rows are unchanged.
    - After all row-wise slides, compute 4-connected components over all non-background pixels regardless of color.
    - Recolor each component grey if its area is at most three, otherwise recolor it maroon.

    Key insight:
    - The horizontal slide changes which rows become vertically connected, and distractor pixels can bridge those moved pairs, so the final component areas must be measured only after the slide on the full multicolor foreground.
    """
    shifted = _slide_pairs(np.asarray(grid, dtype=int))
    out = np.full_like(shifted, Color.BLACK)
    for comp in find_components(
        shifted,
        bg=Color.BLACK,
        connectivity=4,
        monochromatic=False,
    ):
        fill = SMALL_COLOR if int(comp["area"]) <= 3 else LARGE_COLOR
        coords = comp["coords"]
        out[coords[:, 0], coords[:, 1]] = int(fill)
    return out


def _pair_for_family(family: str) -> tuple[int, int]:
    if family == "cool":
        return random.choice(COOL_PAIRS)
    assert family == "warm"
    return random.choice(WARM_PAIRS)


def _distractor_pair(kind: str) -> tuple[int, int]:
    if kind == "cross":
        return random.choice(CROSS_FAMILY_DISTRACTORS)
    assert kind == "outsider"
    return random.choice(OUTSIDER_DISTRACTORS)


def _movable_input_cols(width: int, left: int, prefer_min_gap: int = 3) -> tuple[int, int]:
    min_right = left + prefer_min_gap
    if min_right >= width:
        min_right = left + 2
    assert min_right < width
    right = random.randint(min_right, width - 1)
    return int(left), int(right)


def _place_pair(
    grid: np.ndarray,
    row: int,
    cols: tuple[int, int],
    colors: tuple[int, int],
) -> None:
    left, right = map(int, cols)
    assert left < right
    assert int(grid[row, left]) == Color.BLACK
    assert int(grid[row, right]) == Color.BLACK
    grid[row, left] = int(colors[0])
    grid[row, right] = int(colors[1])


def _component_areas(grid: np.ndarray) -> list[int]:
    comps = find_components(grid, bg=Color.BLACK, connectivity=4, monochromatic=False)
    return sorted(int(comp["area"]) for comp in comps)


def _component_summaries(grid: np.ndarray) -> list[dict]:
    row_roles: dict[int, str] = {}
    for r in range(grid.shape[0]):
        cols = _non_bg_columns(grid, r)
        if cols.size == 0:
            continue
        left, right = map(int, cols)
        role = _row_kind((int(grid[r, left]), int(grid[r, right])))
        row_roles[int(r)] = "distractor" if role is None else role

    summaries = []
    shifted = _slide_pairs(grid)
    for comp in find_components(
        shifted,
        bg=Color.BLACK,
        connectivity=4,
        monochromatic=False,
    ):
        coords = comp["coords"]
        rows = {int(r) for r in coords[:, 0]}
        summaries.append(
            {
                "area": int(comp["area"]),
                "rows": rows,
                "has_distractor": any(row_roles.get(r) == "distractor" for r in rows),
                "has_movable": any(row_roles.get(r) in {"cool", "warm"} for r in rows),
            }
        )
    return summaries


def _rows_have_two_pixels(grid: np.ndarray) -> bool:
    for r in range(grid.shape[0]):
        cols = _non_bg_columns(grid, r)
        if cols.size not in {0, 2}:
            return False
    return True


def _semantic_checks(grid: np.ndarray) -> None:
    assert grid.ndim == 2
    assert _rows_have_two_pixels(grid)

    movable_families = set()
    has_distractor = False
    has_cross_family_distractor = False
    for r in range(grid.shape[0]):
        cols = _non_bg_columns(grid, r)
        if cols.size == 0:
            continue
        left, right = map(int, cols)
        colors = (int(grid[r, left]), int(grid[r, right]))
        kind = _row_kind(colors)
        if kind is None:
            has_distractor = True
            if len({colors[0], colors[1]} & COOL_GROUP) == 1 and len(
                {colors[0], colors[1]} & WARM_GROUP
            ) == 1:
                has_cross_family_distractor = True
        else:
            movable_families.add(kind)
            assert right - left >= 2

    assert movable_families == {"cool", "warm"}
    assert has_distractor
    assert has_cross_family_distractor

    shifted = _slide_pairs(grid)
    areas_after = _component_areas(shifted)
    assert any(area <= 3 for area in areas_after)
    assert any(area > 3 for area in areas_after)

    summaries = _component_summaries(grid)
    assert any(
        summary["area"] > 3 and summary["has_movable"] and summary["has_distractor"]
        for summary in summaries
    )

    areas_before = _component_areas(grid)
    assert max(areas_before, default=0) <= 3
    assert max(areas_after, default=0) > max(areas_before, default=0)

    shifted_without_distractors = shifted.copy()
    for r in range(grid.shape[0]):
        cols = _non_bg_columns(grid, r)
        if cols.size == 0:
            continue
        left, right = map(int, cols)
        if _row_kind((int(grid[r, left]), int(grid[r, right]))) is None:
            shifted_without_distractors[r, :] = Color.BLACK
    assert max(_component_areas(shifted_without_distractors), default=0) <= 3

    out = main(grid)
    assert set(map(int, np.unique(out))) <= {Color.BLACK, SMALL_COLOR, LARGE_COLOR}
    assert SMALL_COLOR in out
    assert LARGE_COLOR in out


def _build_large_bridge_specs(width: int, start_row: int) -> list[dict]:
    left = random.randint(1, width - 6)
    return [
        {"row": start_row, "mode": "movable", "family": "cool", "left": left},
        {
            "row": start_row + 1,
            "mode": "distractor",
            "kind": "cross",
            "cols": (left + 1, left + 2),
        },
        {
            "row": start_row + 2,
            "mode": "movable",
            "family": "warm",
            "left": left + 2,
        },
    ]


def _far_column(width: int, avoid: set[int]) -> int:
    candidates = [c for c in range(1, width - 1) if c not in avoid]
    assert candidates
    scores = [(min(abs(c - a) for a in avoid) if avoid else width, c) for c in candidates]
    best_dist = max(score for score, _c in scores)
    best = [c for score, c in scores if score == best_dist]
    return int(random.choice(best))


def _build_small_specs(width: int, start_row: int) -> list[dict]:
    left = random.randint(1, width - 4)
    family = random.choice(("cool", "warm"))
    if random.random() < 0.5:
        return [{"row": start_row, "mode": "movable", "family": family, "left": left}]

    attach_col = left + random.randint(0, 1)
    far = _far_column(width, {attach_col, left, left + 1})
    cols = tuple(sorted((attach_col, far)))
    return [
        {"row": start_row, "mode": "movable", "family": family, "left": left},
        {
            "row": start_row + 1,
            "mode": "distractor",
            "kind": "outsider",
            "cols": cols,
        },
    ]


def _extra_specs(height: int, width: int, occupied_rows: set[int]) -> list[dict]:
    specs: list[dict] = []
    candidate_rows = [
        r
        for r in range(1, height - 1)
        if all(abs(r - used) > 1 for used in occupied_rows)
    ]
    random.shuffle(candidate_rows)
    count = min(len(candidate_rows), random.randint(0, 2))
    for row in candidate_rows[:count]:
        if random.random() < 0.5:
            left = random.randint(1, width - 4)
            specs.append(
                {
                    "row": row,
                    "mode": "movable",
                    "family": random.choice(("cool", "warm")),
                    "left": left,
                }
            )
        else:
            cols = sorted(random.sample(range(1, width - 1), 2))
            if cols[1] - cols[0] == 1 and random.random() < 0.5:
                cols[1] = min(width - 2, cols[0] + 2)
            specs.append(
                {
                    "row": row,
                    "mode": "distractor",
                    "kind": random.choice(("cross", "outsider")),
                    "cols": tuple(cols),
                }
            )
    return specs


def _render_specs(height: int, width: int, specs: list[dict]) -> np.ndarray:
    grid = np.full((height, width), Color.BLACK, dtype=int)
    for spec in specs:
        row = int(spec["row"])
        if spec["mode"] == "movable":
            cols = _movable_input_cols(width, int(spec["left"]))
            colors = _pair_for_family(str(spec["family"]))
        else:
            cols = tuple(map(int, spec["cols"]))
            colors = _distractor_pair(str(spec["kind"]))
        _place_pair(grid, row, cols, colors)
    return grid


def _attempt_input() -> np.ndarray | None:
    height = random.randint(10, 16)
    width = random.randint(13, 20)

    large_start = random.randint(1, 3)
    small_start = random.randint(large_start + 5, height - 2)

    specs = _build_large_bridge_specs(width, large_start)
    specs.extend(_build_small_specs(width, small_start))

    used_rows = {int(spec["row"]) for spec in specs}
    specs.extend(_extra_specs(height, width, used_rows))

    try:
        grid = _render_specs(height, width, specs)
        _semantic_checks(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_input()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 10")
