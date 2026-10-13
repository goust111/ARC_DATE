from __future__ import annotations

from collections import deque
import random

import numpy as np


def _mode(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
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
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _is_rectangular_frame(mask: np.ndarray, comp: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None
    sub = mask[r0 : r1 + 1, c0 : c1 + 1]
    if sub[1:-1, 1:-1].any():
        return None
    if not sub[0, :].all() or not sub[-1, :].all() or not sub[:, 0].all() or not sub[:, -1].all():
        return None
    return r0, r1, c0, c1


def _frame_components(grid: np.ndarray, bg: int) -> tuple[int | None, list[tuple[int, int, int, int]]]:
    frame_info: list[tuple[int, list[tuple[int, int, int, int]], int]] = []
    colors = [int(v) for v in np.unique(grid) if int(v) != bg]
    for color in colors:
        mask = grid == color
        boxes: list[tuple[int, int, int, int]] = []
        total = 0
        for comp in _components4(mask):
            box = _is_rectangular_frame(mask, comp)
            if box is None:
                continue
            boxes.append(box)
            total += len(comp)
        if boxes:
            frame_info.append((color, boxes, total))
    if not frame_info:
        return None, []
    frame_info.sort(key=lambda item: (-item[2], min(item[1])))
    border_color, boxes, _ = frame_info[0]
    boxes.sort()
    return border_color, boxes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each hollow rectangle from its internal seed color and thicken it by the seed count.

    Input structure:
    - The grid has a dominant background color.
    - Several 1-pixel-thick closed rectangular frames appear in a shared border color.
    - Inside each frame are one or more same-colored seed pixels; different frames use different seed colors.

    Rule:
    - For each rectangular frame, inspect the cells strictly inside it.
    - The unique non-background interior color is the frame's output color, and its pixel count N is the number of concentric border layers to draw.
    - Replace the original frame with that seed color and keep drawing inward for N total layers, stopping early if that fills the whole rectangle; all remaining interior cells revert to background.

    Key insight:
    - The seed pixels encode two visible attributes at once: their color chooses the repaint color, and their count chooses how many inward rectangular rings survive.
    """
    g = np.asarray(grid, dtype=int)
    bg = _mode(g)
    out = g.copy()

    border_color, boxes = _frame_components(g, bg)
    if border_color is None:
        return out

    for r0, r1, c0, c1 in boxes:
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        non_bg = [int(v) for v in np.unique(inner) if int(v) != bg]
        if len(non_bg) != 1:
            continue
        seed_color = non_bg[0]
        if seed_color == border_color:
            continue

        seed_count = int(np.count_nonzero(inner == seed_color))
        if seed_count <= 0:
            continue

        max_layers = (min(r1 - r0 + 1, c1 - c0 + 1) + 1) // 2
        layers = min(seed_count, max_layers)
        for rr in range(r0, r1 + 1):
            for cc in range(c0, c1 + 1):
                depth = min(rr - r0, r1 - rr, cc - c0, c1 - cc)
                out[rr, cc] = seed_color if depth < layers else bg

    return out


GRID_H_RANGE = (10, 30)
GRID_W_RANGE = (10, 30)
RECT_COUNT_CHOICES = [1, 2, 2, 3, 3, 4]
RECT_SIDE_CAP = 11
MAX_SEEDS = 8
PADDING = 1
GEN_RETRIES = 500
PLACEMENT_TRIES = 80
SEED_TRIES = 80


def _has_frame_component(mask: np.ndarray) -> bool:
    for comp in _components4(mask):
        if _is_rectangular_frame(mask, comp) is not None:
            return True
    return False


def _sample_seeds(inner_h: int, inner_w: int, count: int, rng: random.Random) -> list[tuple[int, int]] | None:
    cells = [(r, c) for r in range(inner_h) for c in range(inner_w)]
    if count > len(cells):
        return None
    for _ in range(SEED_TRIES):
        picks = rng.sample(cells, count)
        mask = np.zeros((inner_h, inner_w), dtype=bool)
        for r, c in picks:
            mask[r, c] = True
        if not _has_frame_component(mask):
            return picks
    return None


def _can_place(occ: np.ndarray, r0: int, c0: int, box_h: int, box_w: int) -> bool:
    h, w = occ.shape
    return not occ[
        max(0, r0 - PADDING) : min(h, r0 + box_h + PADDING),
        max(0, c0 - PADDING) : min(w, c0 + box_w + PADDING),
    ].any()


def _mark_occupied(occ: np.ndarray, r0: int, c0: int, box_h: int, box_w: int) -> None:
    h, w = occ.shape
    occ[
        max(0, r0 - PADDING) : min(h, r0 + box_h + PADDING),
        max(0, c0 - PADDING) : min(w, c0 + box_w + PADDING),
    ] = True


def _choose_rectangles(h: int, w: int, n_rects: int, rng: random.Random):
    modes = [rng.choice(["fit", "saturate"])] if n_rects == 1 else ["fit", "saturate"] + [
        rng.choice(["fit", "saturate"]) for _ in range(n_rects - 2)
    ]
    rng.shuffle(modes)

    occ = np.zeros((h, w), dtype=bool)
    rects = []
    for mode in modes:
        placed = False
        for _ in range(PLACEMENT_TRIES):
            box_h = rng.randint(3, min(RECT_SIDE_CAP, h - 2))
            box_w = rng.randint(3, min(RECT_SIDE_CAP, w - 2))
            inner_h, inner_w = box_h - 2, box_w - 2
            interior_area = inner_h * inner_w
            max_layers = (min(box_h, box_w) + 1) // 2
            max_seed_count = min(MAX_SEEDS, interior_area)
            if mode == "saturate":
                if max_seed_count <= max_layers:
                    continue
                seed_count = rng.randint(max_layers + 1, max_seed_count)
            else:
                if max_seed_count < 1:
                    continue
                seed_count = rng.randint(1, min(max_layers, max_seed_count))

            seeds = _sample_seeds(inner_h, inner_w, seed_count, rng)
            if seeds is None:
                continue

            if h - box_h - 1 < 1 or w - box_w - 1 < 1:
                continue
            for _ in range(PLACEMENT_TRIES):
                r0 = rng.randint(1, h - box_h - 1)
                c0 = rng.randint(1, w - box_w - 1)
                if not _can_place(occ, r0, c0, box_h, box_w):
                    continue
                _mark_occupied(occ, r0, c0, box_h, box_w)
                rects.append((r0, c0, box_h, box_w, seeds))
                placed = True
                break
            if placed:
                break
        if not placed:
            return None
    return rects


def _paint_input(
    h: int,
    w: int,
    bg: int,
    border_color: int,
    seed_colors: list[int],
    rects: list[tuple[int, int, int, int, list[tuple[int, int]]]],
) -> np.ndarray:
    grid = np.full((h, w), bg, dtype=int)
    for seed_color, (r0, c0, box_h, box_w, seeds) in zip(seed_colors, rects):
        r1 = r0 + box_h - 1
        c1 = c0 + box_w - 1
        grid[r0, c0 : c1 + 1] = border_color
        grid[r1, c0 : c1 + 1] = border_color
        grid[r0 : r1 + 1, c0] = border_color
        grid[r0 : r1 + 1, c1] = border_color
        for sr, sc in seeds:
            grid[r0 + 1 + sr, c0 + 1 + sc] = seed_color
    return grid


def _background_is_dominant(grid: np.ndarray, bg: int) -> bool:
    values, counts = np.unique(grid, return_counts=True)
    order = np.argsort(-counts)
    return int(values[order[0]]) == bg


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_rects = rng.choice(RECT_COUNT_CHOICES)
    side_floor = {1: 10, 2: 12, 3: 15, 4: 18}[n_rects]
    h = rng.randint(side_floor, GRID_H_RANGE[1])
    w = rng.randint(side_floor, GRID_W_RANGE[1])

    rects = _choose_rectangles(h, w, n_rects, rng)
    if rects is None:
        return None

    colors = list(range(10))
    bg = rng.choice(colors)
    colors.remove(bg)
    border_color = rng.choice(colors)
    colors.remove(border_color)
    if n_rects > len(colors):
        return None
    seed_colors = rng.sample(colors, n_rects)

    inp = _paint_input(h, w, bg, border_color, seed_colors, rects)
    if not _background_is_dominant(inp, bg):
        return None

    out = main(inp)
    if np.array_equal(inp, out):
        return None
    if max(inp.shape) > 30 or max(out.shape) > 30:
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
