from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _draw_frame(grid: np.ndarray, box: tuple[int, int, int, int], border: int, interior: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0 : r1 + 1, c0 : c1 + 1] = interior
    grid[r0, c0 : c1 + 1] = border
    grid[r1, c0 : c1 + 1] = border
    grid[r0 : r1 + 1, c0] = border
    grid[r0 : r1 + 1, c1] = border


def _find_rectangle(grid: np.ndarray, bg: int) -> tuple[tuple[int, int, int, int], int, int]:
    candidates: list[tuple[tuple[int, int, int, int], int, int]] = []
    for coords in _components(grid != bg):
        if len(coords) < 9:
            continue
        r0, c0, r1, c1 = _bbox(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 3 or w < 3 or len(coords) != h * w:
            continue
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        colors = sorted(set(map(int, patch.ravel())))
        if len(colors) != 2:
            continue
        border = int(patch[0, 0])
        interior = int(patch[1, 1])
        if border == interior:
            continue
        expected = np.full((h, w), interior, dtype=int)
        expected[0, :] = border
        expected[-1, :] = border
        expected[:, 0] = border
        expected[:, -1] = border
        if np.array_equal(patch, expected):
            candidates.append(((r0, c0, r1, c1), border, interior))
    assert len(candidates) == 1, "expected one two-color solid rectangle"
    return candidates[0]


def _cascade_box(
    box: tuple[int, int, int, int], markers: set[tuple[int, int]]
) -> tuple[tuple[int, int, int, int], set[tuple[int, int]]]:
    r0, c0, r1, c1 = box
    remaining = set(markers)
    while True:
        nr0, nc0, nr1, nc1 = r0, c0, r1, c1
        consumed: set[tuple[int, int]] = set()
        for r, c in remaining:
            if c0 <= c <= c1:
                if r < r0:
                    nr0 = min(nr0, r)
                    consumed.add((r, c))
                elif r > r1:
                    nr1 = max(nr1, r)
                    consumed.add((r, c))
            if r0 <= r <= r1:
                if c < c0:
                    nc0 = min(nc0, c)
                    consumed.add((r, c))
                elif c > c1:
                    nc1 = max(nc1, c)
                    consumed.add((r, c))
        if not consumed:
            break
        r0, c0, r1, c1 = nr0, nc0, nr1, nc1
        remaining -= consumed
    return (r0, c0, r1, c1), remaining


def main(grid: np.ndarray) -> np.ndarray:
    """Cascade a two-color rectangle toward aligned marker pixels until no new marker is unlocked.

    Input structure:
    - The grid has a dominant background, one filled rectangle with a border color and an interior color, and several isolated single-pixel markers.
    - The rectangle is axis-aligned and occupies its whole bounding box; markers lie outside the rectangle.
    - At least one marker shares the rectangle's current row span or column span.

    Rule:
    - Find the rectangle and treat every non-background pixel outside it as a marker.
    - Whenever a marker is above, below, left, or right of the rectangle while sharing its column span or row span, stretch the corresponding side until that marker lies on the new border, consuming it.
    - Repeat until no marker is aligned; redraw the final rectangle and keep all never-aligned markers unchanged.

    Key insight:
    - Each stretch enlarges the rectangle's row or column coverage, which can unlock later markers that were not aligned at the start.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    box, border, interior = _find_rectangle(g, bg)
    r0, c0, r1, c1 = box

    markers = {
        (int(r), int(c))
        for r, c in np.argwhere(g != bg)
        if not (r0 <= int(r) <= r1 and c0 <= int(c) <= c1)
    }
    final_box, survivors = _cascade_box(box, markers)

    out = np.full_like(g, bg)
    _draw_frame(out, final_box, border, interior)
    for r, c in survivors:
        out[r, c] = int(g[r, c])
    return out


STEP_COUNT_CHOICES = [1, 2, 3, 4]
STEP_COUNT_WEIGHTS = [1, 4, 4, 3]
INITIAL_SIDE_RANGE = (3, 6)
STEP_DISTANCE_RANGE = (2, 5)
PADDING_RANGE = (2, 6)
DISTRACTOR_COUNT_CHOICES = [0, 1, 2, 3]
DISTRACTOR_COUNT_WEIGHTS = [1, 3, 3, 2]
MAX_GENERATION_ATTEMPTS = 400


def _shift_box(box: tuple[int, int, int, int], dr: int, dc: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    return r0 + dr, c0 + dc, r1 + dr, c1 + dc


def _aligned_markers(box: tuple[int, int, int, int], markers: set[tuple[int, int]]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = box
    out = set()
    for r, c in markers:
        if c0 <= c <= c1 and (r < r0 or r > r1):
            out.add((r, c))
        elif r0 <= r <= r1 and (c < c0 or c > c1):
            out.add((r, c))
    return out


def _build_chain() -> tuple[tuple[int, int, int, int], tuple[int, int, int, int], list[tuple[int, int]]]:
    init_h = random.randint(*INITIAL_SIDE_RANGE)
    init_w = random.randint(*INITIAL_SIDE_RANGE)
    initial_box = (0, 0, init_h - 1, init_w - 1)
    r0, c0, r1, c1 = initial_box
    markers: list[tuple[int, int]] = []
    last_rows: list[int] = []
    last_cols: list[int] = []

    step_count = random.choices(STEP_COUNT_CHOICES, weights=STEP_COUNT_WEIGHTS, k=1)[0]
    prev_axis = random.choice(("vertical", "horizontal"))
    for step_idx in range(step_count):
        axis = prev_axis if step_idx == 0 else ("horizontal" if prev_axis == "vertical" else "vertical")
        distance = random.randint(*STEP_DISTANCE_RANGE)
        if axis == "vertical":
            direction = random.choice(("up", "down"))
            col = random.randint(c0, c1) if step_idx == 0 else random.choice(last_cols)
            if direction == "up":
                marker = (r0 - distance, col)
                last_rows = list(range(r0 - distance, r0))
                r0 -= distance
            else:
                marker = (r1 + distance, col)
                last_rows = list(range(r1 + 1, r1 + distance + 1))
                r1 += distance
        else:
            direction = random.choice(("left", "right"))
            row = random.randint(r0, r1) if step_idx == 0 else random.choice(last_rows)
            if direction == "left":
                marker = (row, c0 - distance)
                last_cols = list(range(c0 - distance, c0))
                c0 -= distance
            else:
                marker = (row, c1 + distance)
                last_cols = list(range(c1 + 1, c1 + distance + 1))
                c1 += distance
        markers.append(marker)
        prev_axis = axis

    return initial_box, (r0, c0, r1, c1), markers


def _well_separated(cell: tuple[int, int], occupied: set[tuple[int, int]], min_chebyshev: int = 2) -> bool:
    r, c = cell
    return all(max(abs(r - rr), abs(c - cc)) >= min_chebyshev for rr, cc in occupied)


def _place_distractors(
    final_box: tuple[int, int, int, int],
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = final_box
    h, w = shape
    candidates = [
        (r, c)
        for r in range(h)
        for c in range(w)
        if (r < r0 or r > r1) and (c < c0 or c > c1) and _well_separated((r, c), occupied)
    ]
    random.shuffle(candidates)
    distractor_count = random.choices(
        DISTRACTOR_COUNT_CHOICES, weights=DISTRACTOR_COUNT_WEIGHTS, k=1
    )[0]
    placed: list[tuple[int, int]] = []
    for cell in candidates:
        if len(placed) >= distractor_count:
            break
        if _well_separated(cell, occupied | set(placed)):
            placed.append(cell)
    return placed


def _palette() -> tuple[int, int, int, int]:
    bg, border, interior, marker = random.sample(range(10), 4)
    return bg, border, interior, marker


def _gen_attempt() -> np.ndarray | None:
    initial_box, final_box, chain_markers = _build_chain()
    final_h = final_box[2] - final_box[0] + 1
    final_w = final_box[3] - final_box[1] + 1

    top = random.randint(*PADDING_RANGE)
    left = random.randint(*PADDING_RANGE)
    bottom = random.randint(*PADDING_RANGE)
    right = random.randint(*PADDING_RANGE)
    height = final_h + top + bottom
    width = final_w + left + right
    if height > 30 or width > 30:
        return None

    row_shift = top - final_box[0]
    col_shift = left - final_box[1]
    shifted_initial = _shift_box(initial_box, row_shift, col_shift)
    shifted_final = _shift_box(final_box, row_shift, col_shift)
    shifted_chain = [(r + row_shift, c + col_shift) for r, c in chain_markers]

    bg, border, interior, marker_color = _palette()
    inp = np.full((height, width), bg, dtype=int)
    out = np.full((height, width), bg, dtype=int)
    _draw_frame(inp, shifted_initial, border, interior)
    _draw_frame(out, shifted_final, border, interior)

    occupied = {(int(r), int(c)) for r in range(shifted_initial[0], shifted_initial[2] + 1) for c in range(shifted_initial[1], shifted_initial[3] + 1)}
    for r, c in shifted_chain:
        if not _well_separated((r, c), occupied):
            return None
        inp[r, c] = marker_color
        occupied.add((r, c))

    distractors = _place_distractors(shifted_final, inp.shape, occupied)
    for r, c in distractors:
        inp[r, c] = marker_color
        out[r, c] = marker_color

    chain_set = set(shifted_chain)
    distractor_set = set(distractors)
    if len(_aligned_markers(shifted_initial, chain_set | distractor_set)) != 1:
        return None

    box = shifted_initial
    remaining = set(shifted_chain) | distractor_set
    for _ in shifted_chain:
        aligned = _aligned_markers(box, remaining)
        if len(aligned) != 1:
            return None
        box, remaining = _cascade_box(box, remaining)
    if box != shifted_final or remaining != distractor_set:
        return None

    try:
        if not np.array_equal(main(inp), out):
            return None
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cascade-stretch input")
