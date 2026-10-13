from __future__ import annotations

import random

import numpy as np


# --------------- solver helpers ---------------

MAX_PERIOD = 6


def _neighbors4(r: int, c: int):
    if r > 0:
        yield r - 1, c
    if c > 0:
        yield r, c - 1
    yield r + 1, c
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0] or seen[r0, c0]:
                continue
            seen[r0, c0] = True
            stack = [(int(r0), int(c0))]
            comp: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                comp.append((r, c))
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _count_holes(mask: np.ndarray) -> int:
    h, w = mask.shape
    bg = ~mask
    seen = np.zeros((h, w), dtype=bool)
    holes = 0
    for r0 in range(h):
        for c0 in range(w):
            if not bg[r0, c0] or seen[r0, c0]:
                continue
            seen[r0, c0] = True
            stack = [(int(r0), int(c0))]
            touches_border = False
            while stack:
                r, c = stack.pop()
                if r == 0 or c == 0 or r == h - 1 or c == w - 1:
                    touches_border = True
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and bg[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            if not touches_border:
                holes += 1
    return holes


def _supports_period(tile: np.ndarray, ph: int, pw: int) -> bool:
    h, w = tile.shape
    for r in range(h):
        for c in range(w):
            if int(tile[r, c]) != int(tile[r % ph, c % pw]):
                return False
    return True


def _minimal_period(tile: np.ndarray) -> tuple[int, int]:
    h, w = tile.shape
    best = None
    for ph in range(1, h + 1):
        if h % ph:
            continue
        for pw in range(1, w + 1):
            if w % pw:
                continue
            if not _supports_period(tile, ph, pw):
                continue
            cand = (ph * pw, ph, pw)
            if best is None or cand < best:
                best = cand
    assert best is not None
    return best[1], best[2]


def _repeat_tile(tile: np.ndarray, h: int, w: int) -> np.ndarray:
    ph, pw = tile.shape
    rows = np.arange(h) % int(ph)
    cols = np.arange(w) % int(pw)
    return tile[rows[:, None], cols[None, :]]


def _infer_tile(grid: np.ndarray, blocked: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    best_score = None
    best_tile = None

    for ph in range(1, min(MAX_PERIOD, h) + 1):
        for pw in range(1, min(MAX_PERIOD, w) + 1):
            tile = np.zeros((ph, pw), dtype=int)
            mismatches = 0
            exact_groups = 0
            valid = True

            for rr in range(ph):
                for cc in range(pw):
                    vals = [
                        int(grid[r, c])
                        for r in range(rr, h, ph)
                        for c in range(cc, w, pw)
                        if not blocked[r, c]
                    ]
                    if not vals:
                        valid = False
                        break
                    uniq, cnts = np.unique(vals, return_counts=True)
                    idx = int(cnts.argmax())
                    tile[rr, cc] = int(uniq[idx])
                    mismatches += len(vals) - int(cnts[idx])
                    if int(cnts[idx]) == len(vals):
                        exact_groups += 1
                if not valid:
                    break
            if not valid:
                continue

            score = (mismatches, ph * pw, ph + pw, -exact_groups)
            if best_score is None or score < best_score:
                best_score = score
                best_tile = tile.copy()

    assert best_tile is not None
    return best_tile


def _recover_background_and_boxes(
    grid: np.ndarray,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]], np.ndarray]:
    h, w = grid.shape
    legend_blocked = np.zeros((h, w), dtype=bool)
    legend_blocked[:, 0] = True
    infer_blocked = legend_blocked.copy()
    previous = None
    recon = None
    diff = None
    boxes: list[tuple[int, int, int, int]] = []

    for _ in range(3):
        tile = _infer_tile(grid, infer_blocked)
        recon = _repeat_tile(tile, h, w)
        diff = (grid != recon) & (~legend_blocked)
        boxes = sorted(_bbox(comp) for comp in _components(diff))
        signature = tuple(boxes)
        if signature == previous:
            break
        previous = signature
        infer_blocked = legend_blocked.copy()
        for r0, c0, r1, c1 in boxes:
            infer_blocked[r0 : r1 + 1, c0 : c1 + 1] = True

    assert recon is not None and diff is not None
    return recon, boxes, diff


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, _w = g.shape
    recon, boxes, diff = _recover_background_and_boxes(g)
    out = recon.copy()

    for r0, c0, r1, c1 in boxes:
        obj_mask = diff[r0 : r1 + 1, c0 : c1 + 1]
        hole_count = _count_holes(obj_mask)
        assert 0 <= hole_count < h
        border_color = int(g[hole_count, 0])
        out[r0 : r1 + 1, c0] = border_color
        out[r0 : r1 + 1, c1] = border_color
        out[r0, c0 : c1 + 1] = border_color
        out[r1, c0 : c1 + 1] = border_color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct a periodic background hidden by rectangular occluders, then redraw each
    occluder as a 1-pixel border whose color is chosen from the left-edge legend by hole count.
    """
    return _solve(grid)


# --------------- generator helpers ---------------

COLORS = list(range(10))
GRID_H_RANGE = (12, 18)
GRID_W_RANGE = (14, 22)
PERIOD_H_RANGE = (1, 4)
PERIOD_W_RANGE = (2, 5)
BACKGROUND_COLOR_COUNT = (2, 3)
OBJECT_COUNT_RANGE = (1, 3)
MAX_HOLES = 3
RECT_H_RANGE = (4, 8)
RECT_W_RANGE = (4, 8)
PLACEMENT_GAP = 1
INSTANCE_RETRIES = 600
HOLE_RETRIES = 40
OBJECT_RETRIES = 160


def _sample_tile(ph: int, pw: int, palette: list[int]) -> np.ndarray | None:
    for _ in range(200):
        color_count = random.randint(2, min(3, len(palette)))
        colors = random.sample(palette, color_count)
        tile = np.array(
            [[random.choice(colors) for _ in range(pw)] for _ in range(ph)],
            dtype=int,
        )
        if len(set(map(int, tile.flatten()))) < 2:
            continue
        if _minimal_period(tile) != (ph, pw):
            continue
        return tile
    return None


def _sample_holes(rect_h: int, rect_w: int, hole_count: int):
    if hole_count == 0:
        return []

    candidates = [(r, c) for r in range(1, rect_h - 1) for c in range(1, rect_w - 1)]
    for _ in range(HOLE_RETRIES):
        random.shuffle(candidates)
        holes = []
        for r, c in candidates:
            if any(max(abs(r - hr), abs(c - hc)) <= 1 for hr, hc, _hh, _ww in holes):
                continue
            holes.append((r, c, 1, 1))
            if len(holes) == hole_count:
                return holes

    return None


def _expanded_overlap(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _sample_object_specs(
    grid_h: int, grid_w: int, hole_counts: list[int], object_colors: list[int]
):
    specs = []
    boxes: list[tuple[int, int, int, int]] = []

    for hole_count, color in zip(hole_counts, object_colors, strict=False):
        placed = False
        for _ in range(OBJECT_RETRIES):
            min_h = max(RECT_H_RANGE[0], hole_count + 3)
            min_w = max(RECT_W_RANGE[0], hole_count + 3)
            rect_h = random.randint(min_h, RECT_H_RANGE[1])
            rect_w = random.randint(min_w, RECT_W_RANGE[1])
            holes = _sample_holes(rect_h, rect_w, hole_count)
            if holes is None:
                continue
            if rect_h > grid_h - 2 or rect_w > grid_w - 4:
                continue
            r0 = random.randint(1, grid_h - rect_h - 1)
            c0 = random.randint(2, grid_w - rect_w - 2)
            box = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)
            if any(_expanded_overlap(box, old, PLACEMENT_GAP) for old in boxes):
                continue
            specs.append(
                {
                    "box": box,
                    "color": int(color),
                    "hole_count": int(hole_count),
                    "holes": holes,
                }
            )
            boxes.append(box)
            placed = True
            break
        if not placed:
            return None

    return specs


def _foreground_mask(shape: tuple[int, int], specs) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    for spec in specs:
        r0, c0, r1, c1 = spec["box"]
        mask[r0 : r1 + 1, c0 : c1 + 1] = True
        for hr, hc, hh, ww in spec["holes"]:
            mask[r0 + hr : r0 + hr + hh, c0 + hc : c0 + hc + ww] = False
    return mask


def _visible_residue_counts(mask: np.ndarray, ph: int, pw: int) -> bool:
    h, w = mask.shape
    for rr in range(ph):
        for cc in range(pw):
            visible = 0
            for r in range(rr, h, ph):
                for c in range(cc, w, pw):
                    if c == 0 or mask[r, c]:
                        continue
                    visible += 1
            if visible < 2:
                return False
    return True


def _compose_input(
    base: np.ndarray, specs, legend_colors: list[int]
) -> np.ndarray:
    grid = base.copy()
    grid[:, 0] = np.array(legend_colors, dtype=int)
    for spec in specs:
        r0, c0, r1, c1 = spec["box"]
        color = int(spec["color"])
        grid[r0 : r1 + 1, c0 : c1 + 1] = color
        for hr, hc, hh, ww in spec["holes"]:
            grid[r0 + hr : r0 + hr + hh, c0 + hc : c0 + hc + ww] = base[
                r0 + hr : r0 + hr + hh, c0 + hc : c0 + hc + ww
            ]
    return grid


def _expected_output(
    base: np.ndarray, specs, border_colors: list[int]
) -> np.ndarray:
    out = base.copy()
    for spec in specs:
        r0, c0, r1, c1 = spec["box"]
        color = int(border_colors[int(spec["hole_count"])])
        out[r0 : r1 + 1, c0] = color
        out[r0 : r1 + 1, c1] = color
        out[r0, c0 : c1 + 1] = color
        out[r1, c0 : c1 + 1] = color
    return out


def _legend_column(height: int, border_colors: list[int]) -> list[int]:
    legend = []
    for r in range(height):
        if r < len(border_colors):
            legend.append(int(border_colors[r]))
        else:
            legend.append(int(border_colors[(r - len(border_colors)) % len(border_colors)]))
    return legend


def _generate_attempt() -> np.ndarray | None:
    ph = random.randint(*PERIOD_H_RANGE)
    pw = random.randint(*PERIOD_W_RANGE)
    grid_h = random.randint(max(GRID_H_RANGE[0], ph * 3), GRID_H_RANGE[1])
    grid_w = random.randint(max(GRID_W_RANGE[0], pw * 3 + 1), GRID_W_RANGE[1])

    bg_count = random.randint(*BACKGROUND_COLOR_COUNT)
    bg_palette = random.sample(COLORS, bg_count)
    tile = _sample_tile(ph, pw, bg_palette)
    if tile is None:
        return None
    base = _repeat_tile(tile, grid_h, grid_w)

    remaining = [c for c in COLORS if c not in bg_palette]
    border_colors = random.sample(remaining, MAX_HOLES + 1)
    remaining = [c for c in remaining if c not in border_colors]
    obj_count = random.randint(*OBJECT_COUNT_RANGE)
    if len(remaining) < obj_count:
        return None
    object_colors = random.sample(remaining, obj_count)
    hole_counts = [random.randint(0, MAX_HOLES) for _ in range(obj_count)]

    specs = _sample_object_specs(grid_h, grid_w, hole_counts, object_colors)
    if specs is None:
        return None

    fg_mask = _foreground_mask((grid_h, grid_w), specs)
    if not _visible_residue_counts(fg_mask, ph, pw):
        return None

    legend_colors = _legend_column(grid_h, border_colors)
    inp = _compose_input(base, specs, legend_colors)
    expected = _expected_output(base, specs, border_colors)

    if np.array_equal(inp, expected):
        return None
    if not np.array_equal(main(inp), expected):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(INSTANCE_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC instance")
