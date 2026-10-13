from __future__ import annotations

from collections import deque

import numpy as np


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _turn_count(mask: np.ndarray) -> int:
    padded = np.pad(mask.astype(int), 1)
    windows = (
        padded[:-1, :-1]
        + padded[:-1, 1:]
        + padded[1:, :-1]
        + padded[1:, 1:]
    )
    return int(np.count_nonzero((windows == 1) | (windows == 3)))


def _legend_colors(last_row: np.ndarray, bg: int) -> list[int]:
    colors: list[int] = []
    c = 0
    while c < last_row.shape[0]:
        if int(last_row[c]) == bg:
            c += 1
            continue
        color = int(last_row[c])
        while c < last_row.shape[0] and int(last_row[c]) == color:
            c += 1
        colors.append(color)
    return colors


def main(grid: np.ndarray) -> np.ndarray:
    """Classify grey objects by boundary turn count and draw a colored outer 8-neighbor ring."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 4 and g.shape[1] >= 6

    bg = _most_common_color(g)
    legend = _legend_colors(g[-1], bg)
    assert len(legend) == 3
    turn_to_color = {4: int(legend[0]), 6: int(legend[1]), 8: int(legend[2])}

    body = g[:-1]
    fg_colors = sorted(set(map(int, body.ravel())) - {bg})
    assert len(fg_colors) == 1
    grey = int(fg_colors[0])

    out = g.copy()
    out[-1, :] = bg
    comps = _find_components(body == grey)
    assert comps

    for coords in comps:
        r0, c0, r1, c1 = _bbox(coords)
        local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        local[coords[:, 0] - r0, coords[:, 1] - c0] = True
        turns = _turn_count(local)
        assert turns in turn_to_color
        color = int(turn_to_color[turns])

        ring = np.zeros_like(body, dtype=bool)
        for r, c in coords:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = int(r + dr), int(c + dc)
                    if 0 <= nr < body.shape[0] and 0 <= nc < body.shape[1] and int(body[nr, nc]) == bg:
                        ring[nr, nc] = True
        out[:-1][ring] = color

    return out


# --------------- generator ---------------

GREY = 5
TURN_CLASSES = (4, 6, 8)
GRID_H_RANGE = (16, 25)
GRID_W_RANGE = (18, 27)
OBJECT_COUNT_RANGE = (3, 7)
LEGEND_RUN_WIDTHS = (2, 5)
LEGEND_GAPS = (1, 4)
PLACEMENT_PADDING = 2
MAX_ATTEMPTS = 400


def _reserve(mask: np.ndarray, r: int, c: int, h: int, w: int, pad: int) -> None:
    r0 = max(0, r - pad)
    c0 = max(0, c - pad)
    r1 = min(mask.shape[0], r + h + pad)
    c1 = min(mask.shape[1], c + w + pad)
    mask[r0:r1, c0:c1] = True


def _random_transform(shape: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.array(shape, dtype=bool)
    out = np.rot90(out, int(rng.integers(0, 4)))
    if bool(rng.integers(0, 2)):
        out = np.fliplr(out)
    rows = np.any(out, axis=1)
    cols = np.any(out, axis=0)
    return out[np.ix_(rows, cols)]


def _rect_shape(rng: np.random.Generator) -> np.ndarray:
    h = int(rng.integers(2, 6))
    w = int(rng.integers(2, 7))
    return np.ones((h, w), dtype=bool)


def _l_shape(rng: np.random.Generator) -> np.ndarray:
    h = int(rng.integers(3, 7))
    w = int(rng.integers(3, 7))
    thick = int(rng.integers(1, min(3, h, w)))
    thick = min(thick, h - 1, w - 1)
    shape = np.zeros((h, w), dtype=bool)
    shape[:, :thick] = True
    shape[-thick:, :] = True
    return _random_transform(shape, rng)


def _t_shape(rng: np.random.Generator) -> np.ndarray:
    h = int(rng.integers(4, 8))
    w = int(rng.integers(4, 8))
    bar_h = int(rng.integers(1, min(3, h - 1)))
    stem_w = int(rng.integers(1, min(4, w - 1)))
    stem_w = min(stem_w, w - 2)
    stem_c = int(rng.integers(1, w - stem_w))
    shape = np.zeros((h, w), dtype=bool)
    shape[:bar_h, :] = True
    shape[bar_h:, stem_c : stem_c + stem_w] = True
    return _random_transform(shape, rng)


def _make_shape(rng: np.random.Generator, turns: int) -> np.ndarray:
    for _ in range(100):
        if turns == 4:
            shape = _rect_shape(rng)
        elif turns == 6:
            shape = _l_shape(rng)
        else:
            shape = _t_shape(rng)
        if _turn_count(shape) == turns:
            return shape
    raise RuntimeError(f"failed to build shape with {turns} turns")


def _place_shape(
    grid: np.ndarray,
    occupied: np.ndarray,
    shape: np.ndarray,
    color: int,
    rng: np.random.Generator,
) -> bool:
    h, w = shape.shape
    max_r = grid.shape[0] - h - 2
    max_c = grid.shape[1] - w - 1
    if max_r < 1 or max_c < 1:
        return False
    spots = [(r, c) for r in range(1, max_r + 1) for c in range(1, max_c + 1)]
    rng.shuffle(spots)
    for r, c in spots:
        if occupied[r : r + h, c : c + w].any():
            continue
        patch = grid[r : r + h, c : c + w]
        if np.any(shape & (patch != grid[0, 0])):
            continue
        patch[shape] = color
        _reserve(occupied, r, c, h, w, PLACEMENT_PADDING)
        return True
    return False


def _draw_legend(row: np.ndarray, colors: list[int], rng: np.random.Generator, bg: int) -> bool:
    widths = [int(rng.integers(*LEGEND_RUN_WIDTHS)) for _ in colors]
    gaps = [int(rng.integers(*LEGEND_GAPS)) for _ in range(len(colors) + 1)]
    total = sum(widths) + sum(gaps)
    if total > row.shape[0]:
        return False
    slack = int(row.shape[0] - total)
    gaps[int(rng.integers(0, len(gaps)))] += slack
    row[:] = bg
    c = gaps[0]
    for idx, color in enumerate(colors):
        row[c : c + widths[idx]] = int(color)
        c += widths[idx] + gaps[idx + 1]
    return True


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    h = int(rng.integers(*GRID_H_RANGE))
    w = int(rng.integers(*GRID_W_RANGE))

    bg_choices = [c for c in range(10) if c != GREY]
    bg = int(rng.choice(bg_choices))
    legend_colors = [int(x) for x in rng.choice([c for c in range(10) if c not in (bg, GREY)], size=3, replace=False)]

    grid = np.full((h, w), bg, dtype=int)
    occupied = np.zeros((h - 1, w), dtype=bool)

    extra = int(rng.integers(0, OBJECT_COUNT_RANGE[1] - OBJECT_COUNT_RANGE[0] + 1))
    targets = [4, 6, 8] + [int(rng.choice(TURN_CLASSES)) for _ in range(extra)]
    rng.shuffle(targets)

    for turns in targets:
        shape = _make_shape(rng, turns)
        if not _place_shape(grid[:-1], occupied, shape, GREY, rng):
            return None

    if not _draw_legend(grid[-1], legend_colors, rng, bg):
        return None

    values, counts = np.unique(grid, return_counts=True)
    if int(values[np.argmax(counts)]) != bg:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
