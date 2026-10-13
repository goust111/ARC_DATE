from __future__ import annotations

from collections import deque
import random

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = DIR4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]

GEN_ATTEMPTS = 300
OBJECT_ATTEMPTS = 160
LAYOUT_ATTEMPTS = 120
MAX_OBJECT_SIDE = 12
MAX_OBJECT_AREA = 80


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    dirs = DIR4 if connectivity == 4 else DIR8
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for dr, dc in dirs:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _component_submask(coords: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([r0, c0], dtype=int)
    mask[rel[:, 0], rel[:, 1]] = True
    return (r0, c0, r1, c1), mask


def _background_partition(mask: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    bg = ~mask
    exterior = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if bg[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if bg[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and bg[nr, nc] and not exterior[nr, nc]:
                exterior[nr, nc] = True
                q.append((nr, nc))

    holes = _components(bg & ~exterior, connectivity=4)
    return exterior, holes


def _outer_contour(mask: np.ndarray) -> tuple[np.ndarray, int]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    exterior, holes = _background_partition(mask)
    contour = np.zeros((h, w), dtype=bool)

    for r, c in np.argwhere(mask):
        for dr, dc in DIR4:
            nr = int(r + dr)
            nc = int(c + dc)
            if not (0 <= nr < h and 0 <= nc < w):
                contour[r, c] = True
                break
            if exterior[nr, nc]:
                contour[r, c] = True
                break

    return contour, len(holes)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repaint each monochrome closed shape's outer contour using the bottom legend color keyed by hole count.

    Input structure:
    - The grid has a dominant background color and a bottom legend row.
    - Above the legend row, several 8-connected monochrome foreground shapes appear on background.
    - The legend row contains four distinct non-background pixels in left-to-right order, separated by single background gaps, encoding colors for hole counts 0, 1, 2, 3.

    Rule:
    - Remove the bottom legend row in the output.
    - For each foreground shape, count its enclosed 4-connected background holes.
    - Use that hole count to select the corresponding legend color, then keep only the shape's outer contour pixels in that color; all non-contour shape pixels become background and holes stay background.

    Key insight:
    - Hole count determines a color lookup, but only pixels bordering exterior background survive, so the solver must separate outer background from enclosed holes before recoloring.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2

    bg = _most_common_color(g)
    legend_row = g[-1]
    legend_cols = [int(c) for c in np.flatnonzero(legend_row != bg)]
    assert len(legend_cols) == 4, "expected four legend colors on the last row"
    legend = [int(legend_row[c]) for c in legend_cols]

    upper = g[:-1]
    fg_colors = [int(v) for v in np.unique(upper) if int(v) != bg]
    assert len(fg_colors) == 1, "expected one foreground object color above the legend"
    fg = int(fg_colors[0])

    out = np.full_like(upper, bg)
    for coords in _components(upper == fg, connectivity=8):
        (r0, c0, _r1, _c1), mask = _component_submask(coords)
        contour, hole_count = _outer_contour(mask)
        assert 0 <= hole_count < len(legend), "hole count must map into the legend"
        color = int(legend[hole_count])
        rr, cc = np.argwhere(contour).T
        out[r0 + rr, c0 + cc] = color

    return out


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _hole_count(mask: np.ndarray) -> int:
    return _outer_contour(mask)[1]


def _valid_object(mask: np.ndarray, hole_count: int) -> bool:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return False
    mask = _crop_mask(mask)
    if len(_components(mask, connectivity=8)) != 1:
        return False
    if _hole_count(mask) != hole_count:
        return False
    h, w = mask.shape
    area = int(mask.sum())
    if h > MAX_OBJECT_SIDE or w > MAX_OBJECT_SIDE or area > MAX_OBJECT_AREA:
        return False
    if h < 4 or w < 4 or area < 12:
        return False
    return True


def _rand1(rng: random.Random, lo: int, hi: int) -> int:
    return rng.randint(lo, hi)


def _base_pattern(hole_count: int, rng: random.Random) -> np.ndarray:
    if hole_count == 0:
        h = _rand1(rng, 4, 7)
        w = _rand1(rng, 4, 8)
        return np.ones((h, w), dtype=bool)

    patterns = {
        1: [[(0, 0)]],
        2: [[(0, 0), (0, 3)], [(0, 0), (3, 0)], [(0, 0), (3, 3)]],
        3: [
            [(0, 0), (0, 3), (0, 6)],
            [(0, 0), (3, 0), (6, 0)],
            [(0, 0), (0, 3), (3, 0)],
            [(0, 0), (0, 3), (3, 3)],
        ],
    }

    slots = list(rng.choice(patterns[hole_count]))
    hole_sizes = []
    for _ in slots:
        hh = _rand1(rng, 1, 2)
        ww = _rand1(rng, 1, 2)
        hole_sizes.append((hh, ww))

    top = _rand1(rng, 2, 3)
    left = _rand1(rng, 2, 3)
    bottom = _rand1(rng, 2, 3)
    right = _rand1(rng, 2, 3)

    max_r = max(sr + hh for (sr, _sc), (hh, _ww) in zip(slots, hole_sizes))
    max_c = max(sc + ww for (_sr, sc), (_hh, ww) in zip(slots, hole_sizes))
    mask = np.ones((top + max_r + bottom, left + max_c + right), dtype=bool)

    for (sr, sc), (hh, ww) in zip(slots, hole_sizes):
        jr = 0 if hh == 2 else _rand1(rng, 0, 1)
        jc = 0 if ww == 2 else _rand1(rng, 0, 1)
        r0 = top + sr + jr
        c0 = left + sc + jc
        mask[r0 : r0 + hh, c0 : c0 + ww] = False

    return mask


def _add_attachment(mask: np.ndarray, rng: random.Random) -> np.ndarray:
    out = mask.copy()
    r0, c0, r1, c1 = _bbox(np.argwhere(mask))
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    side = rng.choice(["top", "bottom", "left", "right"])

    if side in {"top", "bottom"}:
        aw = _rand1(rng, 1, min(3, w))
        depth = _rand1(rng, 1, 2)
        start = _rand1(rng, c0, c1 - aw + 1)
        if side == "top":
            if r0 - depth < 0:
                return mask
            out[r0 - depth : r0, start : start + aw] = True
        else:
            if r1 + depth >= mask.shape[0]:
                return mask
            out[r1 + 1 : r1 + depth + 1, start : start + aw] = True
    else:
        ah = _rand1(rng, 1, min(3, h))
        depth = _rand1(rng, 1, 2)
        start = _rand1(rng, r0, r1 - ah + 1)
        if side == "left":
            if c0 - depth < 0:
                return mask
            out[start : start + ah, c0 - depth : c0] = True
        else:
            if c1 + depth >= mask.shape[1]:
                return mask
            out[start : start + ah, c1 + 1 : c1 + depth + 1] = True

    return out


def _cut_notch(mask: np.ndarray, rng: random.Random) -> np.ndarray:
    out = mask.copy()
    r0, c0, r1, c1 = _bbox(np.argwhere(mask))
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    side = rng.choice(["top", "bottom", "left", "right"])

    if side in {"top", "bottom"}:
        if w < 3 or h < 4:
            return mask
        nw = _rand1(rng, 1, min(3, w - 2))
        depth = _rand1(rng, 1, min(2, h - 2))
        start = _rand1(rng, c0 + 1, c1 - nw)
        if side == "top":
            out[r0 : r0 + depth, start : start + nw] = False
        else:
            out[r1 - depth + 1 : r1 + 1, start : start + nw] = False
    else:
        if h < 3 or w < 4:
            return mask
        nh = _rand1(rng, 1, min(3, h - 2))
        depth = _rand1(rng, 1, min(2, w - 2))
        start = _rand1(rng, r0 + 1, r1 - nh)
        if side == "left":
            out[start : start + nh, c0 : c0 + depth] = False
        else:
            out[start : start + nh, c1 - depth + 1 : c1 + 1] = False

    return out


def _clip_corner(mask: np.ndarray, rng: random.Random) -> np.ndarray:
    out = mask.copy()
    r0, c0, r1, c1 = _bbox(np.argwhere(mask))
    corner = rng.choice(["tl", "tr", "bl", "br"])
    size = _rand1(rng, 1, 2)

    if corner == "tl":
        out[r0 : min(r0 + size, r1 + 1), c0 : min(c0 + size, c1 + 1)] = False
    elif corner == "tr":
        out[r0 : min(r0 + size, r1 + 1), max(c1 - size + 1, c0) : c1 + 1] = False
    elif corner == "bl":
        out[max(r1 - size + 1, r0) : r1 + 1, c0 : min(c0 + size, c1 + 1)] = False
    else:
        out[max(r1 - size + 1, r0) : r1 + 1, max(c1 - size + 1, c0) : c1 + 1] = False

    return out


def _random_orientation(mask: np.ndarray, rng: random.Random) -> np.ndarray:
    out = np.array(mask, copy=True)
    out = np.rot90(out, k=_rand1(rng, 0, 3))
    if rng.random() < 0.5:
        out = np.fliplr(out)
    if rng.random() < 0.5:
        out = np.flipud(out)
    return np.array(out, copy=True)


def _generate_object(hole_count: int, rng: random.Random) -> np.ndarray:
    for _ in range(OBJECT_ATTEMPTS):
        mask = _base_pattern(hole_count, rng)
        mask = np.pad(mask, 2, constant_values=False)

        for _ in range(_rand1(rng, 2, 6)):
            op = rng.choice([_add_attachment, _cut_notch, _clip_corner])
            candidate = op(mask, rng)
            if _valid_object(candidate, hole_count):
                mask = candidate

        mask = _random_orientation(_crop_mask(mask), rng)
        if _valid_object(mask, hole_count):
            return _crop_mask(mask)

    raise RuntimeError(f"failed to generate object with {hole_count} holes")


def _layout_objects(masks: list[np.ndarray], rng: random.Random) -> tuple[int, int, list[tuple[int, int]]] | None:
    n = len(masks)
    row_options = [r for r in (1, 2, 3) if r <= n]
    if n >= 5:
        row_options = [r for r in row_options if r >= 2]

    for _ in range(LAYOUT_ATTEMPTS):
        row_count = rng.choice(row_options)
        order = list(range(n))
        rng.shuffle(order)
        order.sort(key=lambda i: (masks[i].shape[1], masks[i].shape[0]), reverse=True)

        rows: list[list[int]] = [[] for _ in range(row_count)]
        row_loads = [0] * row_count
        row_heights = [0] * row_count
        for idx in order:
            h, w = masks[idx].shape
            ranked = sorted(
                range(row_count),
                key=lambda r: (row_loads[r] + 3 * len(rows[r]), row_heights[r], len(rows[r])),
            )
            target = ranked[0]
            rows[target].append(idx)
            row_loads[target] += w
            row_heights[target] = max(row_heights[target], h)

        if any(len(row) == 0 for row in rows):
            continue

        row_specs = []
        content_w = 0
        content_h = 0
        for row in rows:
            rng.shuffle(row)
            gaps = [_rand1(rng, 2, 4) for _ in range(max(0, len(row) - 1))]
            width = sum(int(masks[idx].shape[1]) for idx in row) + sum(gaps)
            height = max(int(masks[idx].shape[0]) for idx in row)
            row_specs.append((row, gaps, width, height))
            content_w = max(content_w, width)
            content_h += height

        row_gaps = [_rand1(rng, 2, 4) for _ in range(max(0, row_count - 1))]
        content_h += sum(row_gaps)

        base_left = _rand1(rng, 1, 3)
        base_right = _rand1(rng, 1, 3)
        base_top = _rand1(rng, 1, 3)
        base_bottom = _rand1(rng, 1, 3)
        upper_h = base_top + content_h + base_bottom
        width = base_left + content_w + base_right
        if upper_h > 29 or width > 30:
            continue

        placements: list[tuple[int, int] | None] = [None] * n
        y = base_top
        for row_idx, (row, gaps, row_width, row_height) in enumerate(row_specs):
            x = base_left + _rand1(rng, 0, content_w - row_width)
            for j, idx in enumerate(row):
                obj_h, obj_w = masks[idx].shape
                dy = _rand1(rng, 0, row_height - obj_h)
                placements[idx] = (y + dy, x)
                x += obj_w
                if j < len(gaps):
                    x += gaps[j]
            y += row_height
            if row_idx < len(row_gaps):
                y += row_gaps[row_idx]

        placements = [p for p in placements if p is not None]
        if len(placements) != n:
            continue
        return upper_h, width, placements

    return None


def generate_input() -> np.ndarray:
    rng = random.Random()

    for _ in range(GEN_ATTEMPTS):
        hole_counts = [0, 1, 2, 3]
        for _ in range(_rand1(rng, 0, 2)):
            hole_counts.append(_rand1(rng, 0, 3))
        rng.shuffle(hole_counts)

        try:
            masks = [_generate_object(hc, rng) for hc in hole_counts]
        except RuntimeError:
            continue

        layout = _layout_objects(masks, rng)
        if layout is None:
            continue
        upper_h, width, placements = layout

        total_fg = sum(int(mask.sum()) for mask in masks)
        total_area = (upper_h + 1) * width
        if total_area - total_fg - 4 <= total_fg:
            extra_h_cap = 29 - upper_h
            extra_w_cap = 30 - width
            grown = False
            for extra_h in range(extra_h_cap, -1, -1):
                for extra_w in range(extra_w_cap, -1, -1):
                    new_upper_h = upper_h + extra_h
                    new_width = width + extra_w
                    if (new_upper_h + 1) * new_width - total_fg - 4 > total_fg:
                        shift_r = _rand1(rng, 0, extra_h)
                        shift_c = _rand1(rng, 0, extra_w)
                        placements = [(r + shift_r, c + shift_c) for r, c in placements]
                        upper_h = new_upper_h
                        width = new_width
                        grown = True
                        break
                if grown:
                    break
            if not grown:
                continue

        colors = list(range(10))
        rng.shuffle(colors)
        bg = int(colors[0])
        fg = int(colors[1])
        legend = [int(c) for c in colors[2:6]]

        grid = np.full((upper_h + 1, width), bg, dtype=int)
        for mask, (top, left) in zip(masks, placements):
            h, w = mask.shape
            grid[top : top + h, left : left + w][mask] = fg

        legend_start = _rand1(rng, 0, width - 7)
        for idx, color in enumerate(legend):
            grid[-1, legend_start + 2 * idx] = color

        if len(np.unique(grid[:-1][grid[:-1] != bg])) != 1:
            continue
        if _most_common_color(grid) != bg:
            continue
        if max(grid.shape) > 30:
            continue

        out = main(grid)
        if out.shape != (upper_h, width):
            continue

        return grid

    raise RuntimeError("failed to generate a valid input grid")
