from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


MAROON = 9
BLUE = 1
GREEN = 3
YELLOW = 4
OUTPUT_COLORS = (MAROON, BLUE, GREEN, YELLOW)
INPUT_COLOR_POOL = tuple(color for color in range(10) if color not in OUTPUT_COLORS)

DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
GENERATION_ATTEMPTS = 300


def _border_color(grid: np.ndarray) -> int:
    h, w = grid.shape
    border = [int(v) for v in grid[0, :]]
    border.extend(int(v) for v in grid[h - 1, :])
    border.extend(int(v) for v in grid[1:h - 1, 0])
    border.extend(int(v) for v in grid[1:h - 1, w - 1])
    return Counter(border).most_common(1)[0][0]


def _background_neighbor_counts(grid: np.ndarray, bg: int) -> np.ndarray:
    bg_mask = (grid == int(bg)).astype(int)
    counts = np.zeros(grid.shape, dtype=int)
    counts[1:, :] += bg_mask[:-1, :]
    counts[:-1, :] += bg_mask[1:, :]
    counts[:, 1:] += bg_mask[:, :-1]
    counts[:, :-1] += bg_mask[:, 1:]
    return counts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each foreground pixel by how exposed it is to the background.

    Rule:
    - Count background-colored 4-neighbors for every non-background pixel.
    - 0 background neighbors -> maroon.
    - 1 background neighbor -> blue.
    - 2 background neighbors -> green.
    - 3 or 4 background neighbors -> yellow.
    - Background pixels stay unchanged.
    """
    g = np.asarray(grid, dtype=int)
    bg = _border_color(g)
    fg_mask = g != int(bg)
    counts = _background_neighbor_counts(g, bg)

    out = g.copy()
    out[fg_mask & (counts == 0)] = MAROON
    out[fg_mask & (counts == 1)] = BLUE
    out[fg_mask & (counts == 2)] = GREEN
    out[fg_mask & (counts >= 3)] = YELLOW
    return out


def _add_tab(
    mask: np.ndarray,
    top: int,
    bottom: int,
    left: int,
    right: int,
    side: str,
    offset: int,
    span: int,
    length: int,
) -> None:
    if side == "top":
        c0 = left + offset
        mask[top - length:top, c0:c0 + span] = True
    elif side == "bottom":
        c0 = left + offset
        mask[bottom + 1:bottom + 1 + length, c0:c0 + span] = True
    elif side == "left":
        r0 = top + offset
        mask[r0:r0 + span, left - length:left] = True
    elif side == "right":
        r0 = top + offset
        mask[r0:r0 + span, right + 1:right + 1 + length] = True
    else:
        raise ValueError(f"unknown side: {side}")


def _component_sizes(mask: np.ndarray) -> list[int]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    sizes: list[int] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            size = 0
            while q:
                rr, cc = q.popleft()
                size += 1
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            sizes.append(size)
    return sizes


def _bbox_area(mask: np.ndarray) -> int:
    rows, cols = np.nonzero(mask)
    return (int(rows.max()) - int(rows.min()) + 1) * (int(cols.max()) - int(cols.min()) + 1)


def _tab_params(
    rng: random.Random,
    side: str,
    base_h: int,
    base_w: int,
    spaces: dict[str, int],
    force_finger: bool,
) -> tuple[int, int, int]:
    max_length = min(4, spaces[side] - 1)
    side_len = base_w if side in {"top", "bottom"} else base_h
    max_span = min(4, side_len - 2)

    if force_finger:
        span = 1
        length = rng.randint(1, max_length)
    else:
        span = rng.randint(1, max_span)
        length = rng.randint(1, max_length)

    offset = rng.randint(1, side_len - span - 1)
    return offset, span, length


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(15, 23)
    w = rng.randint(15, 23)
    bg, fg = rng.sample(INPUT_COLOR_POOL, 2)

    min_base_h = max(9, int(round(0.6 * h)))
    min_base_w = max(9, int(round(0.6 * w)))
    base_h = rng.randint(min_base_h, h - 4)
    base_w = rng.randint(min_base_w, w - 4)

    top = rng.randint(2, h - base_h - 2)
    left = rng.randint(2, w - base_w - 2)
    bottom = top + base_h - 1
    right = left + base_w - 1

    mask = np.zeros((h, w), dtype=bool)
    mask[top:bottom + 1, left:right + 1] = True

    spaces = {
        "top": top,
        "bottom": h - 1 - bottom,
        "left": left,
        "right": w - 1 - right,
    }
    sides = ["top", "bottom", "left", "right"]

    finger_side = rng.choice(sides)
    offset, span, length = _tab_params(rng, finger_side, base_h, base_w, spaces, force_finger=True)
    _add_tab(mask, top, bottom, left, right, finger_side, offset, span, length)

    for _ in range(rng.randint(2, 5)):
        side = rng.choice(sides)
        offset, span, length = _tab_params(rng, side, base_h, base_w, spaces, force_finger=False)
        _add_tab(mask, top, bottom, left, right, side, offset, span, length)

    if any(mask[edge].any() for edge in (0, -1)) or mask[:, 0].any() or mask[:, -1].any():
        return None

    sizes = _component_sizes(mask)
    if sizes != [int(mask.sum())]:
        return None

    area_ratio = float(mask.mean())
    if not (0.38 <= area_ratio <= 0.72):
        return None

    if int(mask.sum()) >= _bbox_area(mask):
        return None

    grid = np.full((h, w), bg, dtype=int)
    grid[mask] = fg

    counts = _background_neighbor_counts(grid, bg)
    if not np.any(mask & (counts == 0)):
        return None
    if not np.any(mask & (counts == 1)):
        return None
    if not np.any(mask & (counts == 2)):
        return None
    if not np.any(mask & (counts >= 3)):
        return None

    out_colors = {int(v) for v in np.unique(main(grid)[mask])}
    if out_colors != set(OUTPUT_COLORS):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
