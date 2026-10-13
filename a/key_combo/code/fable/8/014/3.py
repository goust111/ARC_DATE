from __future__ import annotations

import math
import random
from typing import Iterable

import numpy as np


Coord = tuple[int, int]


def _neighbors8(r: int, c: int) -> Iterable[Coord]:
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors8(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_deviations(grid: np.ndarray, mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected at least one deviating cell"
    r0, c0, r1, c1 = _bbox(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    subgrid = grid[r0 : r1 + 1, c0 : c1 + 1]
    submask = mask[r0 : r1 + 1, c0 : c1 + 1]
    out[submask] = subgrid[submask]
    return out


def _evaluate_model(grid: np.ndarray, period: int, step: int) -> dict:
    h, w = grid.shape
    counts = [dict() for _ in range(period)]
    first_seen = [dict() for _ in range(period)]

    for r in range(h):
        row_shift = (r * step) % period
        for c in range(w):
            k = (c - row_shift) % period
            color = int(grid[r, c])
            counts[k][color] = counts[k].get(color, 0) + 1
            first_seen[k].setdefault(color, r * w + c)

    pattern = np.zeros(period, dtype=int)
    margin = 0
    for k in range(period):
        items = list(counts[k].items())
        items.sort(key=lambda item: (-item[1], first_seen[k][item[0]]))
        color, best_count = items[0]
        second_count = items[1][1] if len(items) > 1 else 0
        pattern[k] = color
        margin += best_count - second_count

    mismatch = np.zeros((h, w), dtype=bool)
    for r in range(h):
        row_shift = (r * step) % period
        idx = (np.arange(w) - row_shift) % period
        mismatch[r] = pattern[idx] != grid[r]

    parts = _component_coords(mismatch)
    total_mismatches = int(mismatch.sum())
    component_penalty = 0 if len(parts) == 1 else 1
    step_penalty = 0 if step != 0 else 1
    score = (component_penalty, total_mismatches, period, -margin, step_penalty, step)
    return {
        "score": score,
        "mask": mismatch,
        "period": period,
        "step": step,
    }


def _best_deviation_mask(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    best: dict | None = None
    for period in range(1, w + 1):
        for step in range(period):
            candidate = _evaluate_model(grid, period, step)
            if best is None or candidate["score"] < best["score"]:
                best = candidate

    assert best is not None
    mask = best["mask"]
    parts = _component_coords(mask)
    assert len(parts) == 1, "failed to isolate one connected deviation sprite"
    return mask


def main(grid: np.ndarray) -> np.ndarray:
    """Reveal the connected anomaly sprite by reconstructing the slanted periodic field.

    Input structure:
    - The whole grid uses only non-black colors.
    - Ignoring a small set of cells, every row is a cyclic repetition of one base sequence,
      and each row is shifted horizontally by the same fixed step relative to the previous row.
    - The deviating cells form one 8-connected sprite, but their colors also belong to the
      periodic palette, so color alone does not identify them.

    Rule:
    - Infer the row period and the constant row-to-row shift that best explain the grid as a
      diagonal periodic field.
    - Reconstruct that field by taking the majority color on each slanted period class.
    - Mark every cell whose actual color differs from the reconstructed field.
    - Crop the bounding box of those deviating cells, keep their original colors, and set all
      other cells inside the crop to black.

    Key insight:
    - The hidden object is only visible as disagreement with a slanted repetition rule, so the
      periodic background must be rebuilt before the sprite can be extracted.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must stay within 30x30"
    assert not np.any(g == 0), "input is expected to avoid black"

    deviation_mask = _best_deviation_mask(g)
    return _crop_deviations(g, deviation_mask)


GEN_ATTEMPTS = 400
GRID_HEIGHT_RANGE = (8, 18)
GRID_WIDTH_RANGE = (11, 24)
PERIOD_RANGE = (3, 7)
SPRITE_AREA_RANGE = (4, 12)
SPRITE_BOX_RANGE = (2, 6)
PALETTE = tuple(range(1, 10))


def _build_field(height: int, width: int, pattern: list[int], step: int) -> np.ndarray:
    period = len(pattern)
    row_pattern = np.array(pattern, dtype=int)
    grid = np.empty((height, width), dtype=int)
    for r in range(height):
        shift = (r * step) % period
        idx = (np.arange(width) - shift) % period
        grid[r] = row_pattern[idx]
    return grid


def _random_connected_mask(height: int, width: int, area: int) -> np.ndarray:
    while True:
        start = (random.randrange(height), random.randrange(width))
        cells = {start}
        frontier = set()
        for cell in _neighbors8(*start):
            rr, cc = cell
            if 0 <= rr < height and 0 <= cc < width:
                frontier.add(cell)

        while len(cells) < area and frontier:
            nxt = random.choice(tuple(frontier))
            frontier.remove(nxt)
            cells.add(nxt)
            for cell in _neighbors8(*nxt):
                rr, cc = cell
                if 0 <= rr < height and 0 <= cc < width and cell not in cells:
                    frontier.add(cell)

        if len(cells) != area:
            continue

        mask = np.zeros((height, width), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        coords = np.argwhere(mask)
        if len(np.unique(coords[:, 0])) < 2 or len(np.unique(coords[:, 1])) < 2:
            continue
        if mask.all():
            continue

        r0, c0, r1, c1 = _bbox(coords)
        return mask[r0 : r1 + 1, c0 : c1 + 1]


def _expected_output(grid: np.ndarray, sprite_mask: np.ndarray, top: int, left: int) -> np.ndarray:
    full_mask = np.zeros_like(grid, dtype=bool)
    h, w = sprite_mask.shape
    full_mask[top : top + h, left : left + w] = sprite_mask
    return _crop_deviations(grid, full_mask)


def _primitive_pattern(period: int) -> list[int]:
    # Distinct colors make the intended period easy to read and prevent smaller repeats.
    return random.sample(PALETTE, period)


def _gen_attempt() -> np.ndarray | None:
    period = random.randint(*PERIOD_RANGE)
    step_choices = [s for s in range(1, period) if math.gcd(s, period) == 1]
    if not step_choices:
        return None
    step = random.choice(step_choices)

    min_width = max(GRID_WIDTH_RANGE[0], 2 * period + 1)
    width = random.randint(min_width, GRID_WIDTH_RANGE[1])
    height = random.randint(*GRID_HEIGHT_RANGE)
    field = _build_field(height, width, _primitive_pattern(period), step)

    box_h = random.randint(*SPRITE_BOX_RANGE)
    box_w = random.randint(*SPRITE_BOX_RANGE)
    if box_h > height or box_w > width:
        return None

    max_area = min(SPRITE_AREA_RANGE[1], box_h * box_w - 1)
    if max_area < SPRITE_AREA_RANGE[0]:
        return None
    area = random.randint(SPRITE_AREA_RANGE[0], max_area)
    sprite_mask = _random_connected_mask(box_h, box_w, area)
    sprite_h, sprite_w = sprite_mask.shape

    top = random.randint(0, height - sprite_h)
    left = random.randint(0, width - sprite_w)
    inp = field.copy()

    coords = np.argwhere(sprite_mask)
    palette = list(np.unique(field))
    for dr, dc in coords:
        r = top + int(dr)
        c = left + int(dc)
        current = int(inp[r, c])
        choices = [color for color in palette if color != current]
        inp[r, c] = random.choice(choices)

    expected = _expected_output(inp, sprite_mask, top, left)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid periodic-residue input")
