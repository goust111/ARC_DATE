from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
ORANGE = 7

GEN_HEIGHT_RANGE = (10, 15)
GEN_WIDTH_RANGE = (10, 15)
GEN_DOMINO_COUNT_RANGE = (2, 4)
GEN_BLUE_SEGMENTS_RANGE = (1, 3)
GEN_SINGLE_RG_RANGE = (0, 2)
GEN_MAX_ATTEMPTS = 500


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_from_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in neighbors4(cr, cc):
                    if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _baseline_color(grid: np.ndarray) -> int:
    bottom = list(map(int, grid[-1]))
    return Counter(bottom).most_common(1)[0][0]


def _settle_grid(grid: np.ndarray, bg: int, baseline: int) -> np.ndarray:
    h, w = grid.shape
    out = np.full((h, w), bg, dtype=int)
    out[-1, :] = baseline
    for c in range(w):
        falling = [int(grid[r, c]) for r in range(h) if int(grid[r, c]) not in (bg, baseline)]
        for depth, color in enumerate(reversed(falling)):
            out[h - 1 - depth, c] = color
    return out


def _find_dominoes(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    comps = _components_from_mask((grid == RED) | (grid == GREEN))
    dominoes: list[list[tuple[int, int]]] = []
    for comp in comps:
        if len(comp) != 2:
            continue
        colors = {int(grid[r, c]) for r, c in comp}
        if colors == {RED, GREEN}:
            dominoes.append(comp)
    return dominoes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop scattered pixels to the baseline, erase every red-green domino that forms, and trace the hole's outer contour in orange.

    Input structure:
    - The grid has a dominant black background and a uniform grey baseline on the bottom row.
    - Red, green, and a few blue single-pixel distractors are scattered above the baseline, with at most two non-baseline pixels per column.
    - Before gravity, the red and green pixels are isolated and do not already form 4-neighbor dominoes.

    Rule:
    - Identify the background as the most frequent color and the baseline from the bottom row.
    - Let every non-background, non-baseline pixel fall to the lowest available cell in its own column, treating the baseline as passable landing space.
    - In the settled grid, find every 4-connected component made of exactly two cells, one red and one green; erase both cells to background.
    - For the union of erased cells, paint every 4-neighbor background or baseline cell orange, while leaving blue and any unpaired red/green distractors unchanged.

    Key insight:
    - The important red-green pairings do not exist in the input layout; gravity creates them, and the final marking traces the negative-space holes left behind after those paired cells disappear.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    baseline = _baseline_color(g)
    settled = _settle_grid(g, bg, baseline)

    dominoes = _find_dominoes(settled)
    out = settled.copy()
    holes = {cell for domino in dominoes for cell in domino}
    for r, c in holes:
        out[r, c] = bg

    for r, c in holes:
        for nr, nc in neighbors4(r, c):
            if not in_bounds(out.shape, nr, nc) or (nr, nc) in holes:
                continue
            if int(out[nr, nc]) in (bg, baseline):
                out[nr, nc] = ORANGE
    return out


def _make_vertical_segment() -> dict:
    top, bottom = random.choice([(RED, GREEN), (GREEN, RED)])
    return {"kind": "vertical", "width": 1, "stacks": [[top, bottom]]}


def _make_horizontal_segment() -> dict:
    left_color, right_color = random.choice([(RED, GREEN), (GREEN, RED)])
    left_stack = [left_color]
    right_stack = [right_color]
    if random.random() < 0.4:
        left_stack = [BLUE, left_color]
    if random.random() < 0.4:
        right_stack = [BLUE, right_color]
    return {"kind": "horizontal", "width": 2, "stacks": [left_stack, right_stack]}


def _make_blue_segment() -> dict:
    stack = [BLUE]
    if random.random() < 0.2:
        stack = [BLUE, BLUE]
    return {"kind": "blue", "width": 1, "stacks": [stack]}


def _make_single_rg_segment() -> dict:
    color = random.choice([RED, GREEN])
    stack = [color]
    if random.random() < 0.3:
        stack = [BLUE, color]
    return {"kind": "single_rg", "width": 1, "stacks": [stack]}


def _random_gaps(extra: int, count: int) -> list[int]:
    if count == 1:
        return [extra]
    bins = [0] * count
    for _ in range(extra):
        bins[random.randrange(count)] += 1
    return bins


def _layout_segments(width: int, segments: list[dict]) -> tuple[list[list[int]], list[tuple[int, dict]]] | None:
    min_width = sum(seg["width"] for seg in segments) + (len(segments) - 1)
    if min_width > width:
        return None

    extra = width - min_width
    gaps = _random_gaps(extra, len(segments) + 1)
    stacks: list[list[int]] = [[] for _ in range(width)]
    placed: list[tuple[int, dict]] = []
    col = gaps[0]
    for idx, seg in enumerate(segments):
        placed.append((col, seg))
        for offset, stack in enumerate(seg["stacks"]):
            stacks[col + offset] = list(stack)
        col += seg["width"]
        if idx + 1 < len(segments):
            col += 1 + gaps[idx + 1]
    return stacks, placed


def _render_settled(stacks: list[list[int]], height: int) -> np.ndarray:
    width = len(stacks)
    grid = np.full((height, width), BLACK, dtype=int)
    grid[-1, :] = GREY
    for c, stack in enumerate(stacks):
        for depth, color in enumerate(reversed(stack)):
            grid[height - 1 - depth, c] = int(color)
    return grid


def _row_choices_for_stack(height: int, stack: list[int]) -> list[int] | None:
    max_row = height - 2
    if len(stack) == 1:
        return [random.randint(0, max_row)]
    if stack[0] in (RED, GREEN) and stack[1] in (RED, GREEN):
        top = random.randint(0, max_row - 2)
        bottom = random.randint(top + 2, max_row)
        return [top, bottom]
    bottom = random.randint(1, max_row)
    top = random.randint(0, bottom - 1)
    return [top, bottom]


def _lift_segments(height: int, width: int, placed: list[tuple[int, dict]]) -> np.ndarray | None:
    grid = np.full((height, width), BLACK, dtype=int)
    grid[-1, :] = GREY

    for start, seg in placed:
        if seg["kind"] == "horizontal":
            stacks = seg["stacks"]
            left_need = 1 if len(stacks[0]) == 2 else 0
            right_need = 1 if len(stacks[1]) == 2 else 0
            left_rg = random.randint(left_need, height - 2)
            right_rg_options = [r for r in range(right_need, height - 1) if r != left_rg]
            if not right_rg_options:
                return None
            right_rg = random.choice(right_rg_options)

            for offset, (stack, rg_row) in enumerate(zip(stacks, [left_rg, right_rg])):
                col = start + offset
                if len(stack) == 1:
                    grid[rg_row, col] = stack[0]
                else:
                    blue_row = random.randint(0, rg_row - 1)
                    grid[blue_row, col] = stack[0]
                    grid[rg_row, col] = stack[1]
            continue

        col = start
        stack = seg["stacks"][0]
        rows = _row_choices_for_stack(height, stack)
        if rows is None:
            return None
        for row, color in zip(rows, stack):
            grid[row, col] = color

    rg_mask = (grid == RED) | (grid == GREEN)
    if any(len(comp) != 1 for comp in _components_from_mask(rg_mask)):
        return None
    if any(np.count_nonzero((grid[:, c] != BLACK) & (grid[:, c] != GREY)) > 2 for c in range(width)):
        return None
    return grid


def _domino_orientations(grid: np.ndarray) -> list[str]:
    out = []
    for domino in _find_dominoes(grid):
        rows = {r for r, _ in domino}
        cols = {c for _, c in domino}
        out.append("vertical" if len(cols) == 1 else "horizontal")
        assert len(rows) == 1 or len(cols) == 1
    return out


def _generate_candidate() -> np.ndarray | None:
    height = random.randint(*GEN_HEIGHT_RANGE)
    domino_count = random.randint(*GEN_DOMINO_COUNT_RANGE)

    if domino_count == 1:
        orientations = [random.choice(["horizontal", "vertical"])]
    else:
        mode = random.random()
        if mode < 0.65:
            horizontal_count = random.randint(1, domino_count - 1)
        elif mode < 0.825:
            horizontal_count = domino_count
        else:
            horizontal_count = 0
        orientations = ["horizontal"] * horizontal_count + ["vertical"] * (domino_count - horizontal_count)
        random.shuffle(orientations)

    segments = [
        _make_horizontal_segment() if orient == "horizontal" else _make_vertical_segment()
        for orient in orientations
    ]
    segments.extend(_make_blue_segment() for _ in range(random.randint(*GEN_BLUE_SEGMENTS_RANGE)))
    segments.extend(_make_single_rg_segment() for _ in range(random.randint(*GEN_SINGLE_RG_RANGE)))
    random.shuffle(segments)

    min_width = sum(seg["width"] for seg in segments) + (len(segments) - 1)
    if min_width > GEN_WIDTH_RANGE[1]:
        return None
    width = random.randint(max(GEN_WIDTH_RANGE[0], min_width), GEN_WIDTH_RANGE[1])

    layout = _layout_segments(width, segments)
    if layout is None:
        return None
    stacks, placed = layout
    settled = _render_settled(stacks, height)

    dominoes = _find_dominoes(settled)
    if len(dominoes) != domino_count:
        return None
    if _domino_orientations(settled).count("horizontal") != orientations.count("horizontal"):
        return None
    if np.count_nonzero(settled == BLUE) == 0:
        return None

    candidate = _lift_segments(height, width, placed)
    if candidate is None:
        return None
    if not np.array_equal(_settle_grid(candidate, BLACK, GREY), settled):
        return None
    if np.any(candidate[-1, :] != GREY):
        return None

    produced = main(candidate)
    if produced.shape != candidate.shape:
        return None
    if np.count_nonzero(produced == ORANGE) == 0:
        return None
    return candidate


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _generate_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
