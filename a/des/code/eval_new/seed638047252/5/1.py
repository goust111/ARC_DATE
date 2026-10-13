from __future__ import annotations

from collections import Counter
import random

import numpy as np


FRAME_COLOR = 5
GRID_SIDE_MIN = 12
GRID_SIDE_MAX = 30
RECT_COUNT_CHOICES = (1, 2, 2, 3, 3, 4)
INTERIOR_SIDE_MIN = 3
INTERIOR_SIDE_MAX_SMALL = 6
INTERIOR_SIDE_MAX_LARGE = 8
PLACEMENT_GAP = 1
GENERATION_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 120
COLOR_IDS = tuple(range(10))


def _most_common_color(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _rect_frame_bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return None

    expected = {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if r in (r0, r1) or c in (c0, c1)
    }
    if len(coords) != len(expected) or set(coords) != expected:
        return None
    return int(r0), int(c0), int(r1), int(c1)


def _find_frames(grid: np.ndarray, bg: int) -> tuple[int, list[tuple[int, int, int, int]]]:
    candidates: list[tuple[int, list[tuple[int, int, int, int]]]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        bboxes: list[tuple[int, int, int, int]] = []
        for comp in _components(grid == color):
            bbox = _rect_frame_bbox(comp)
            if bbox is None:
                break
            bboxes.append(bbox)
        else:
            if bboxes:
                candidates.append((color, sorted(bboxes)))

    assert len(candidates) == 1, "expected a unique frame color"
    return candidates[0]


def _majority_vote(interior: np.ndarray, bg: int, frame_color: int) -> int:
    counts = Counter(
        int(value)
        for value in interior.ravel()
        if int(value) != bg and int(value) != frame_color
    )
    assert counts, "each frame interior must contain at least one vote pixel"
    ranking = counts.most_common()
    assert len(ranking) == 1 or ranking[0][1] > ranking[1][1], "vote winner must be unique"
    return int(ranking[0][0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each framed region with its majority interior vote color and fill its blank interior.

    Input structure:
    - The background is the most common color.
    - One non-background color forms several disjoint one-pixel-thick rectangular frames.
    - Inside each frame are scattered non-background vote pixels on an otherwise background interior.

    Rule:
    - For each rectangular frame, count the interior pixels of every non-background, non-frame color.
    - Select the unique most frequent interior color C for that frame.
    - Repaint the whole frame border to C and replace every interior background pixel with C, keeping existing non-background interior pixels unchanged.

    Key insight:
    - The interior colored pixels are votes tied to their enclosing frame, so the task is solve-per-rectangle: detect frames, tally votes inside each one, then recolor the border and blank interior with the winner.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    frame_color, frames = _find_frames(g, bg)

    out = g.copy()
    for r0, c0, r1, c1 in frames:
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        winner = _majority_vote(interior, bg, frame_color)

        out[r0 : r1 + 1, c0] = winner
        out[r0 : r1 + 1, c1] = winner
        out[r0, c0 : c1 + 1] = winner
        out[r1, c0 : c1 + 1] = winner

        fill_view = out[r0 + 1 : r1, c0 + 1 : c1]
        fill_view[fill_view == bg] = winner

    return out


def _can_place(occupied: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> bool:
    rr0 = max(0, r0 - PLACEMENT_GAP)
    cc0 = max(0, c0 - PLACEMENT_GAP)
    rr1 = min(occupied.shape[0] - 1, r1 + PLACEMENT_GAP)
    cc1 = min(occupied.shape[1] - 1, c1 + PLACEMENT_GAP)
    return not bool(occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _mark_occupied(occupied: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    rr0 = max(0, r0 - PLACEMENT_GAP)
    cc0 = max(0, c0 - PLACEMENT_GAP)
    rr1 = min(occupied.shape[0] - 1, r1 + PLACEMENT_GAP)
    cc1 = min(occupied.shape[1] - 1, c1 + PLACEMENT_GAP)
    occupied[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _sample_vote_specs(area: int, palette: list[int]) -> list[tuple[int, int]] | None:
    max_colors = min(4, len(palette), area - 1)
    if max_colors < 1:
        return None

    for _ in range(80):
        color_count = random.randint(1, max_colors)
        chosen = random.sample(palette, color_count)
        winner = random.choice(chosen)
        losers = [color for color in chosen if color != winner]

        if not losers:
            winner_count = random.randint(1, min(area - 1, 6))
            return [(winner, winner_count)]

        loser_counts = [random.randint(1, min(3, area - 2)) for _ in losers]
        winner_lo = max(loser_counts) + 1
        winner_hi = min(area - 1 - sum(loser_counts), winner_lo + 4)
        if winner_lo > winner_hi:
            continue

        winner_count = random.randint(winner_lo, winner_hi)
        specs = [(winner, winner_count)]
        specs.extend((color, count) for color, count in zip(losers, loser_counts))
        return specs

    return None


def _apply_region(
    grid: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    vote_specs: list[tuple[int, int]],
) -> None:
    grid[r0 : r1 + 1, c0] = FRAME_COLOR
    grid[r0 : r1 + 1, c1] = FRAME_COLOR
    grid[r0, c0 : c1 + 1] = FRAME_COLOR
    grid[r1, c0 : c1 + 1] = FRAME_COLOR

    cells = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
    assert len(cells) > sum(count for _, count in vote_specs)
    random.shuffle(cells)

    cursor = 0
    for color, count in vote_specs:
        for _ in range(count):
            r, c = cells[cursor]
            grid[r, c] = int(color)
            cursor += 1


def _expected_output(grid: np.ndarray, frames: list[tuple[int, int, int, int]]) -> np.ndarray:
    out = grid.copy()
    bg = _most_common_color(grid)
    frame_color, _ = _find_frames(grid, bg)
    for r0, c0, r1, c1 in frames:
        winner = _majority_vote(grid[r0 + 1 : r1, c0 + 1 : c1], bg, frame_color)
        out[r0 : r1 + 1, c0] = winner
        out[r0 : r1 + 1, c1] = winner
        out[r0, c0 : c1 + 1] = winner
        out[r1, c0 : c1 + 1] = winner
        fill_view = out[r0 + 1 : r1, c0 + 1 : c1]
        fill_view[fill_view == bg] = winner
    return out


def _gen_attempt() -> np.ndarray | None:
    rect_count = random.choice(RECT_COUNT_CHOICES)
    min_side = min(GRID_SIDE_MAX, 8 + 4 * rect_count)
    height = random.randint(min_side, GRID_SIDE_MAX)
    width = random.randint(min_side, GRID_SIDE_MAX)

    bg = random.choice([color for color in COLOR_IDS if color != FRAME_COLOR])
    palette = [color for color in COLOR_IDS if color not in {bg, FRAME_COLOR}]

    interior_cap = INTERIOR_SIDE_MAX_SMALL if rect_count >= 3 else INTERIOR_SIDE_MAX_LARGE
    max_inner_h = min(interior_cap, height - 4)
    max_inner_w = min(interior_cap, width - 4)
    if max_inner_h < INTERIOR_SIDE_MIN or max_inner_w < INTERIOR_SIDE_MIN:
        return None

    dims = [
        (
            random.randint(INTERIOR_SIDE_MIN, max_inner_h),
            random.randint(INTERIOR_SIDE_MIN, max_inner_w),
        )
        for _ in range(rect_count)
    ]
    dims.sort(key=lambda hw: hw[0] * hw[1], reverse=True)

    occupied = np.zeros((height, width), dtype=bool)
    frames: list[tuple[int, int, int, int]] = []
    for inner_h, inner_w in dims:
        outer_h = inner_h + 2
        outer_w = inner_w + 2
        if height - outer_h - 1 < 1 or width - outer_w - 1 < 1:
            return None

        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            r0 = random.randint(1, height - outer_h - 1)
            c0 = random.randint(1, width - outer_w - 1)
            r1 = r0 + outer_h - 1
            c1 = c0 + outer_w - 1
            if not _can_place(occupied, r0, c0, r1, c1):
                continue
            _mark_occupied(occupied, r0, c0, r1, c1)
            frames.append((r0, c0, r1, c1))
            placed = True
            break
        if not placed:
            return None

    grid = np.full((height, width), bg, dtype=int)
    for r0, c0, r1, c1 in frames:
        area = (r1 - r0 - 1) * (c1 - c0 - 1)
        vote_specs = _sample_vote_specs(area, palette)
        if vote_specs is None:
            return None
        _apply_region(grid, r0, c0, r1, c1, vote_specs)

    colors, counts = np.unique(grid, return_counts=True)
    bg_count = int(counts[colors == bg][0])
    other_counts = [int(count) for color, count in zip(colors, counts) if int(color) != bg]
    if other_counts and bg_count <= max(other_counts):
        return None

    expected = _expected_output(grid, sorted(frames))
    try:
        produced = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(produced, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid majority-frame ARC input")
