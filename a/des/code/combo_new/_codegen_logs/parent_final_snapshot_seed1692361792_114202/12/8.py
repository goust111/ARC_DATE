from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_ATTEMPTS = 400


def _count_holes(mask: np.ndarray) -> int:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    holes = 0
    for r in range(h):
        for c in range(w):
            if mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            touches_edge = False
            while q:
                cr, cc = q.popleft()
                comp.append((int(cr), int(cc)))
                if cr in (0, h - 1) or cc in (0, w - 1):
                    touches_edge = True
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            if not touches_edge:
                holes += 1
    return holes


def _panel_object(sub: np.ndarray) -> tuple[np.ndarray, int] | None:
    colors = [int(v) for v in np.unique(sub) if int(v) not in (BG, GREY)]
    if not colors:
        return None
    color = colors[0]
    coords = np.argwhere(sub == color)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    mask = sub[r0 : r1 + 1, c0 : c1 + 1] == color
    full_mask = np.zeros(sub.shape, dtype=bool)
    full_mask[r0 : r1 + 1, c0 : c1 + 1] = mask
    return full_mask, color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Learn the hole-count-to-color mapping from sample panels and apply it to the query panel.

    Input structure:
    - The background is black.
    - Full-width grey separator rows split the scene into stacked panel bands, and a full-height grey column divides sample panels on the left from answer panels on the right.
    - On the left, each consecutive pair of panel bands contains a top object and its recolored bottom example.
    - On the right, exactly one panel pair has a top object and an empty black bottom panel.

    Rule:
    - For every sample pair, count the number of enclosed black holes in the top object's cropped mask and read the bottom object's color.
    - Build a mapping from hole count to output color from those sample pairs.
    - Find the right-side query pair whose top panel is non-empty and whose bottom panel is empty.
    - Count the query object's holes and redraw the same shape in the blank lower panel using the learned color.

    Key insight:
    - The left side is a lookup lesson rather than part of the target scene: the important latent feature is hole count, and the output color must be inferred from examples before it can be applied to the query.
    """
    g = np.asarray(grid, dtype=int)
    divider = [c for c in range(g.shape[1]) if np.all(g[:, c] == GREY)][0]
    separator_rows = [r for r in range(g.shape[0]) if np.all(g[r, :] == GREY)]
    bounds = [-1] + separator_rows + [g.shape[0]]
    panels = [(bounds[i] + 1, bounds[i + 1]) for i in range(len(bounds) - 1) if bounds[i + 1] - bounds[i] > 1]
    assert len(panels) % 2 == 0

    left_map: dict[int, int] = {}
    query: tuple[int, np.ndarray] | None = None
    out = g.copy()

    for pair_idx in range(0, len(panels), 2):
        top0, top1 = panels[pair_idx]
        bot0, bot1 = panels[pair_idx + 1]
        left_top = g[top0:top1, :divider]
        left_bot = g[bot0:bot1, :divider]
        right_top = g[top0:top1, divider + 1 :]
        right_bot = g[bot0:bot1, divider + 1 :]

        left_top_obj = _panel_object(left_top)
        left_bot_obj = _panel_object(left_bot)
        if left_top_obj is not None and left_bot_obj is not None:
            top_mask, _top_color = left_top_obj
            _bot_mask, bot_color = left_bot_obj
            rows, cols = np.where(top_mask)
            crop = top_mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]
            left_map[_count_holes(crop)] = bot_color

        right_top_obj = _panel_object(right_top)
        right_bot_obj = _panel_object(right_bot)
        if right_top_obj is not None and right_bot_obj is None:
            query = (pair_idx, right_top_obj[0])

    assert query is not None and set(left_map) >= {0, 1, 2}
    pair_idx, query_mask = query
    rows, cols = np.where(query_mask)
    crop = query_mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]
    target_color = left_map[_count_holes(crop)]

    bot0, bot1 = panels[pair_idx + 1]
    dest = out[bot0:bot1, divider + 1 :]
    dest[query_mask] = target_color
    return out


def _make_shape(hole_count: int, rng: random.Random) -> np.ndarray:
    if hole_count == 0:
        bases = [
            np.array([[1, 1, 1], [1, 1, 1]], dtype=bool),
            np.array([[1, 1], [1, 1], [1, 1]], dtype=bool),
            np.array([[1, 1, 1], [1, 1, 0], [0, 1, 0]], dtype=bool),
        ]
    elif hole_count == 1:
        bases = [
            np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=bool),
            np.array([[1, 1, 1, 1], [1, 0, 0, 1], [1, 1, 1, 1]], dtype=bool),
        ]
    else:
        bases = [
            np.array([[1, 1, 1, 1, 1], [1, 0, 1, 0, 1], [1, 1, 1, 1, 1]], dtype=bool),
            np.array([[1, 1, 1, 1, 1, 1], [1, 0, 1, 1, 0, 1], [1, 1, 1, 1, 1, 1]], dtype=bool),
        ]
    shape = bases[rng.randrange(len(bases))].copy()
    if rng.random() < 0.5:
        shape = np.flipud(shape)
    if rng.random() < 0.5:
        shape = np.fliplr(shape)
    return np.asarray(shape, dtype=bool)


def generate_input() -> np.ndarray:
    rng = random.Random()
    panel_h = 4
    panel_w = rng.randint(7, 10)
    slot_count = 3
    total_h = slot_count * 2 * panel_h + (slot_count * 2 - 1)
    if total_h > 30:
        raise RuntimeError("panel layout exceeds ARC bound")

    for _ in range(GEN_ATTEMPTS):
        total_w = panel_w * 2 + 1
        grid = np.zeros((total_h, total_w), dtype=int)
        divider = panel_w
        grid[:, divider] = GREY

        panel_bounds: list[tuple[int, int]] = []
        row = 0
        for idx in range(slot_count * 2):
            panel_bounds.append((row, row + panel_h))
            row += panel_h
            if idx != slot_count * 2 - 1:
                grid[row, :] = GREY
                row += 1

        mapping_colors = rng.sample(COLOR_POOL, 3)
        input_colors = rng.sample([c for c in COLOR_POOL if c not in mapping_colors], 4)
        query_slot = rng.randrange(slot_count)
        query_holes = rng.randrange(3)
        perm = [0, 1, 2]
        rng.shuffle(perm)

        for slot in range(slot_count):
            top0, top1 = panel_bounds[2 * slot]
            bot0, bot1 = panel_bounds[2 * slot + 1]
            top_panel = grid[top0:top1, :divider]
            bot_panel = grid[bot0:bot1, :divider]
            hole_count = perm[slot]
            shape = _make_shape(hole_count, rng)
            sh, sw = map(int, shape.shape)
            top = rng.randint(0, panel_h - sh)
            left = rng.randint(0, panel_w - sw)
            top_panel[top : top + sh, left : left + sw][shape] = input_colors[slot]
            bot_panel[top : top + sh, left : left + sw][shape] = mapping_colors[hole_count]

        top0, top1 = panel_bounds[2 * query_slot]
        bot0, bot1 = panel_bounds[2 * query_slot + 1]
        right_top = grid[top0:top1, divider + 1 :]
        right_bot = grid[bot0:bot1, divider + 1 :]
        shape = _make_shape(query_holes, rng)
        sh, sw = map(int, shape.shape)
        top = rng.randint(0, panel_h - sh)
        left = rng.randint(0, panel_w - sw)
        right_top[top : top + sh, left : left + sw][shape] = input_colors[3]
        right_bot[:, :] = BG
        return grid

    raise RuntimeError("failed to generate input")
