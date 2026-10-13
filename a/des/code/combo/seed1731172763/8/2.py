from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
YELLOW = 4
GREY = 5
INPUT_COLORS = (1, 2, 3, 6, 7, 8, 9)


def _content_bands(separators: list[int], limit: int) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    start = 0
    for sep in [*separators, limit]:
        assert sep > start, "separator columns must leave non-empty panels"
        bands.append((start, sep))
        start = sep + 1
    return bands


def _majority_color(values: np.ndarray) -> int:
    counts = Counter(map(int, values.reshape(-1)))
    color, count = counts.most_common(1)[0]
    return int(color) if count > values.size / 2 else YELLOW


def main(grid: np.ndarray) -> np.ndarray:
    """
    Merge repeated side-by-side panels by local majority vote.

    Input structure:
    - Black background.
    - 3-5 equal-width panels separated by full-height grey columns.
    - Each panel is a noisy copy of the same colored pattern.

    Rule:
    - For each local coordinate, count the colors across all panels.
    - If one color has a strict majority, output that color.
    - Otherwise output yellow.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep_cols = [c for c in range(g.shape[1]) if bool(np.all(g[:, c] == GREY))]
    assert sep_cols, "expected grey separator columns"

    bands = _content_bands(sep_cols, g.shape[1])
    panel_count = len(bands)
    assert 3 <= panel_count <= 5, "expected 3-5 panels"

    widths = {c1 - c0 for c0, c1 in bands}
    assert len(widths) == 1, "panels must have equal width"
    panel_w = widths.pop()
    assert panel_w > 0, "panel width must be positive"

    panels = np.empty((panel_count, g.shape[0], panel_w), dtype=int)
    for idx, (c0, c1) in enumerate(bands):
        panels[idx] = g[:, c0:c1]

    out = np.empty((g.shape[0], panel_w), dtype=int)
    for r in range(g.shape[0]):
        for c in range(panel_w):
            out[r, c] = _majority_color(panels[:, r, c])
    return out


# --------------- generator ---------------

PANEL_H_RANGE = (5, 9)
PANEL_W_RANGE = (5, 9)
PANEL_COUNT_RANGE = (3, 5)
MOTIF_COLOR_COUNT_RANGE = (2, 4)
SHAPE_COUNT_RANGE = (2, 5)
MAJ_NOISE_RANGE = (2, 6)
NO_MAJORITY_RANGE = (1, 4)
MAX_FILL_RATIO = 0.55
MIN_NON_BLACK = 6
MAX_SHAPE_ATTEMPTS = 60
GEN_RETRIES = 400


def _shape_cells(kind: str, max_h: int, max_w: int) -> tuple[list[tuple[int, int]], int, int] | None:
    if kind == "rect":
        h = random.randint(1, min(3, max_h))
        w = random.randint(1, min(3, max_w))
        return [(r, c) for r in range(h) for c in range(w)], h, w

    if kind == "hline":
        if max_w < 2:
            return None
        w = random.randint(2, min(4, max_w))
        return [(0, c) for c in range(w)], 1, w

    if kind == "vline":
        if max_h < 2:
            return None
        h = random.randint(2, min(4, max_h))
        return [(r, 0) for r in range(h)], h, 1

    if kind == "lshape":
        if max_h < 2 or max_w < 2:
            return None
        h = random.randint(2, min(4, max_h))
        w = random.randint(2, min(4, max_w))
        orient = random.choice(("tl", "tr", "bl", "br"))
        cells: set[tuple[int, int]] = set()
        if orient == "tl":
            cells.update((r, 0) for r in range(h))
            cells.update((0, c) for c in range(w))
        elif orient == "tr":
            cells.update((r, w - 1) for r in range(h))
            cells.update((0, c) for c in range(w))
        elif orient == "bl":
            cells.update((r, 0) for r in range(h))
            cells.update((h - 1, c) for c in range(w))
        else:
            cells.update((r, w - 1) for r in range(h))
            cells.update((h - 1, c) for c in range(w))
        return sorted(cells), h, w

    if kind == "plus":
        if max_h < 3 or max_w < 3:
            return None
        return [(1, 0), (1, 1), (1, 2), (0, 1), (2, 1)], 3, 3

    return None


def _can_place(occupied: np.ndarray, top: int, left: int, h: int, w: int) -> bool:
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(occupied.shape[0], top + h + 1)
    c1 = min(occupied.shape[1], left + w + 1)
    return not bool(occupied[r0:r1, c0:c1].any())


def _build_motif(height: int, width: int) -> np.ndarray | None:
    motif = np.full((height, width), BLACK, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)

    palette_size = random.randint(*MOTIF_COLOR_COUNT_RANGE)
    palette = random.sample(INPUT_COLORS, palette_size)
    shape_count = random.randint(*SHAPE_COUNT_RANGE)
    kinds = ["rect", "hline", "vline", "lshape", "plus"]

    for _ in range(shape_count):
        placed = False
        for _ in range(MAX_SHAPE_ATTEMPTS):
            kind = random.choice(kinds)
            spec = _shape_cells(kind, height, width)
            if spec is None:
                continue
            cells, shape_h, shape_w = spec
            top = random.randint(0, height - shape_h)
            left = random.randint(0, width - shape_w)
            if not _can_place(occupied, top, left, shape_h, shape_w):
                continue

            color = random.choice(palette)
            for dr, dc in cells:
                motif[top + dr, left + dc] = color
                occupied[top + dr, left + dc] = True
            placed = True
            break
        if not placed:
            break

    non_black = int(np.count_nonzero(motif != BLACK))
    used_colors = set(map(int, motif[motif != BLACK].reshape(-1)))
    if non_black < MIN_NON_BLACK or len(used_colors) < 2:
        return None
    if non_black > int(height * width * MAX_FILL_RATIO):
        return None
    return motif


def _sample_cells(height: int, width: int, count: int) -> list[tuple[int, int]]:
    all_cells = [(r, c) for r in range(height) for c in range(width)]
    random.shuffle(all_cells)
    return all_cells[:count]


def _pick_alt_colors(base: int, needed: int) -> list[int]:
    pool = [color for color in INPUT_COLORS if color != base]
    assert needed <= len(pool)
    return random.sample(pool, needed)


def _apply_majority_noise(panels: np.ndarray, cells: list[tuple[int, int]]) -> None:
    panel_count = panels.shape[0]
    max_mutations = (panel_count - 1) // 2
    assert max_mutations >= 1

    for r, c in cells:
        base = int(panels[0, r, c])
        mutated = random.randint(1, max_mutations)
        targets = random.sample(range(panel_count), mutated)
        if mutated == 1 or random.random() < 0.5:
            alt = _pick_alt_colors(base, 1)[0]
            for idx in targets:
                panels[idx, r, c] = alt
        else:
            alts = _pick_alt_colors(base, mutated)
            for idx, alt in zip(targets, alts):
                panels[idx, r, c] = alt


def _no_majority_assignment(base: int, panel_count: int) -> list[int]:
    if panel_count == 3:
        colors = [base, *_pick_alt_colors(base, 2)]
        random.shuffle(colors)
        return colors

    if panel_count == 4:
        if random.random() < 0.5:
            alt = _pick_alt_colors(base, 1)[0]
            colors = [base, base, alt, alt]
        else:
            alt1, alt2 = _pick_alt_colors(base, 2)
            colors = [base, base, alt1, alt2]
        random.shuffle(colors)
        return colors

    if random.random() < 0.5:
        alt1, alt2 = _pick_alt_colors(base, 2)
        colors = [base, base, alt1, alt1, alt2]
    else:
        alt1, alt2, alt3 = _pick_alt_colors(base, 3)
        colors = [base, base, alt1, alt2, alt3]
    random.shuffle(colors)
    return colors


def _apply_no_majority_noise(panels: np.ndarray, cells: list[tuple[int, int]]) -> None:
    panel_count = panels.shape[0]
    for r, c in cells:
        base = int(panels[0, r, c])
        for idx, color in enumerate(_no_majority_assignment(base, panel_count)):
            panels[idx, r, c] = color


def _assemble_input(panels: np.ndarray) -> np.ndarray:
    panel_count, height, width = panels.shape
    grid = np.full((height, panel_count * width + panel_count - 1), GREY, dtype=int)
    for idx in range(panel_count):
        c0 = idx * (width + 1)
        grid[:, c0:c0 + width] = panels[idx]
    return grid


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*PANEL_H_RANGE)
    width = random.randint(*PANEL_W_RANGE)
    panel_count = random.randint(*PANEL_COUNT_RANGE)
    motif = _build_motif(height, width)
    if motif is None:
        return None

    area = height * width
    no_majority_cap = min(NO_MAJORITY_RANGE[1], area - MAJ_NOISE_RANGE[0])
    majority_cap_limit = min(MAJ_NOISE_RANGE[1], area - NO_MAJORITY_RANGE[0])
    if no_majority_cap < NO_MAJORITY_RANGE[0] or majority_cap_limit < MAJ_NOISE_RANGE[0]:
        return None

    no_majority_count = random.randint(NO_MAJORITY_RANGE[0], no_majority_cap)
    majority_cap = min(MAJ_NOISE_RANGE[1], area - no_majority_count)
    if majority_cap < MAJ_NOISE_RANGE[0]:
        return None
    majority_count = random.randint(MAJ_NOISE_RANGE[0], majority_cap)

    chosen = _sample_cells(height, width, no_majority_count + majority_count)
    no_majority_cells = chosen[:no_majority_count]
    majority_cells = chosen[no_majority_count:]

    panels = np.repeat(motif[None, :, :], panel_count, axis=0)
    _apply_majority_noise(panels, majority_cells)
    _apply_no_majority_noise(panels, no_majority_cells)
    inp = _assemble_input(panels)

    try:
        out = main(inp)
    except Exception:
        return None

    expected = motif.copy()
    for r, c in no_majority_cells:
        expected[r, c] = YELLOW
    if not np.array_equal(out, expected):
        return None
    if not np.any(out == YELLOW):
        return None
    if not np.any(np.isin(out, INPUT_COLORS)):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
