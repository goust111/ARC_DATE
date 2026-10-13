from collections import deque
import random

import numpy as np


BG = 0
YELLOW = 4
SIDES = ("top", "bottom", "left", "right")

# Canonical top-gap key shapes in a 3x3 window around the entry cell.
# Coordinates are (u, v) where u grows inward from the gap and v is lateral.
KEY_SHAPES = (
    frozenset({(0, -1), (0, 1), (1, -1), (1, 0), (1, 1)}),
    frozenset({(0, -1), (1, -1), (1, 0), (1, 1), (2, -1)}),
    frozenset({(0, -1), (1, -1), (1, 0), (1, 1), (2, 0)}),
    frozenset({(0, -1), (1, -1), (1, 0), (1, 1), (2, 1)}),
    frozenset({(0, -1), (1, -1), (1, 0), (2, 0), (2, 1)}),
    frozenset({(0, -1), (1, -1), (2, -1), (2, 0), (2, 1)}),
    frozenset({(0, 1), (1, -1), (1, 0), (1, 1), (2, -1)}),
    frozenset({(0, 1), (1, -1), (1, 0), (1, 1), (2, 0)}),
)

GEN_RETRIES = 400
FRAME_RETRIES = 120
PLACEMENT_RETRIES = 250


def _neighbors4(r, c):
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components_mask(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points):
    rs = [r for r, _ in points]
    cs = [c for _, c in points]
    return min(rs), min(cs), max(rs), max(cs)


def _normalize_shape(coords):
    pts = [(int(r), int(c)) for r, c in coords]
    assert pts, "shape must be non-empty"
    variants = []
    cur = pts
    for _ in range(4):
        min_r = min(r for r, _ in cur)
        min_c = min(c for _, c in cur)
        variants.append(tuple(sorted((r - min_r, c - min_c) for r, c in cur)))
        cur = [(c, -r) for r, c in cur]
    return min(variants)


def _orient_offset(side, u, v):
    if side == "top":
        return u, v
    if side == "bottom":
        return -u, -v
    if side == "left":
        return -v, u
    if side == "right":
        return v, -u
    raise ValueError(side)


def _rotate_shape(shape, turns):
    pts = list(shape)
    for _ in range(turns % 4):
        pts = [(c, -r) for r, c in pts]
    min_r = min(r for r, _ in pts)
    min_c = min(c for _, c in pts)
    return [(r - min_r, c - min_c) for r, c in pts]


def _label_mask(mask):
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    cells = {}
    nxt = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            nxt += 1
            q = deque([(r, c)])
            labels[r, c] = nxt
            pts = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = nxt
                        q.append((nr, nc))
            cells[nxt] = pts
    return labels, cells


def _scan_wall_links(labels, wall_mask):
    h, w = labels.shape
    links = {}

    for r in range(h):
        c = 0
        while c < w:
            left = int(labels[r, c])
            if left <= 0:
                c += 1
                continue
            c2 = c + 1
            while c2 < w and int(labels[r, c2]) == left:
                c2 += 1
            s = c2
            if s >= w or not wall_mask[r, s]:
                c = c2
                continue
            while s < w and wall_mask[r, s]:
                s += 1
            thickness = s - c2
            if s < w:
                right = int(labels[r, s])
                if right > 0 and right != left:
                    key = (left, right) if left < right else (right, left)
                    links[key] = min(links.get(key, thickness), thickness)
            c = c2

    for c in range(w):
        r = 0
        while r < h:
            top = int(labels[r, c])
            if top <= 0:
                r += 1
                continue
            r2 = r + 1
            while r2 < h and int(labels[r2, c]) == top:
                r2 += 1
            s = r2
            if s >= h or not wall_mask[s, c]:
                r = r2
                continue
            while s < h and wall_mask[s, c]:
                s += 1
            thickness = s - r2
            if s < h:
                bottom = int(labels[s, c])
                if bottom > 0 and bottom != top:
                    key = (top, bottom) if top < bottom else (bottom, top)
                    links[key] = min(links.get(key, thickness), thickness)
            r = r2

    return links


def _analyze_frame(grid, frame):
    color = int(frame["color"])
    yellow = frame["yellow"]
    r0, c0, r1, c1 = frame["bbox"]
    crop = np.array(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)
    h, w = crop.shape

    perimeter = []
    for cc in range(w):
        perimeter.append((0, cc))
        perimeter.append((h - 1, cc))
    for rr in range(1, h - 1):
        perimeter.append((rr, 0))
        perimeter.append((rr, w - 1))
    gaps = [(rr, cc) for rr, cc in perimeter if int(crop[rr, cc]) == BG]
    assert len(gaps) == 1, "frame must have exactly one border gap"
    gr, gc = gaps[0]
    if gr == 0:
        side = "top"
        entry = (1, gc)
    elif gr == h - 1:
        side = "bottom"
        entry = (h - 2, gc)
    elif gc == 0:
        side = "left"
        entry = (gr, 1)
    else:
        side = "right"
        entry = (gr, w - 2)

    key_offsets = []
    er, ec = entry
    for u in range(3):
        for v in (-1, 0, 1):
            dr, dc = _orient_offset(side, u, v)
            rr, cc = er + dr, ec + dc
            if 0 <= rr < h and 0 <= cc < w and int(crop[rr, cc]) == color:
                key_offsets.append((dr, dc))
    key_shape = _normalize_shape(key_offsets)

    walk = crop != color
    walk[gr, gc] = False
    labels, cells = _label_mask(walk)

    ly, lx = int(yellow[0] - r0), int(yellow[1] - c0)
    start = int(labels[ly, lx])
    assert start > 0, "yellow marker must be inside a chamber"

    wall_mask = crop == color
    wall_mask[gr, gc] = True
    links = _scan_wall_links(labels, wall_mask)

    graph = {lab: set() for lab in cells}
    for (a, b), thickness in links.items():
        if thickness == 1:
            graph[a].add(b)
            graph[b].add(a)

    reachable = {start}
    q = deque([start])
    while q:
        cur = q.popleft()
        for nxt in graph[cur]:
            if nxt not in reachable:
                reachable.add(nxt)
                q.append(nxt)

    return {
        "bbox": frame["bbox"],
        "key_shape": key_shape,
        "cells": cells,
        "reachable": reachable,
    }


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    yellow_cells = [tuple(map(int, p)) for p in np.argwhere(g == YELLOW)]
    assert yellow_cells, "at least one yellow marker is required"

    frames = []
    fragments = []
    colors = sorted(set(int(v) for v in g.ravel()) - {BG, YELLOW})
    for color in colors:
        for pts in _components_mask(g == color):
            r0, c0, r1, c1 = _bbox(pts)
            inside_yellow = [
                (yr, yc)
                for yr, yc in yellow_cells
                if r0 < yr < r1 and c0 < yc < c1
            ]
            comp = {"color": color, "pts": pts, "bbox": (r0, c0, r1, c1)}
            if inside_yellow:
                assert len(inside_yellow) == 1, "each frame must contain exactly one yellow marker"
                comp["yellow"] = inside_yellow[0]
                frames.append(comp)
            else:
                fragments.append(comp)

    assert frames, "no frame detected"

    fragments_by_shape = {}
    for frag in fragments:
        shape = _normalize_shape(frag["pts"])
        assert shape not in fragments_by_shape, "fragment shapes must be unique up to rotation"
        fragments_by_shape[shape] = frag

    out = g.copy()
    for frag in fragments:
        for r, c in frag["pts"]:
            out[r, c] = BG

    for frame in frames:
        info = _analyze_frame(g, frame)
        assert info["key_shape"] in fragments_by_shape, "missing matching fragment"
        fill_color = int(fragments_by_shape[info["key_shape"]]["color"])
        r0, c0, _, _ = info["bbox"]
        for lab in info["reachable"]:
            for rr, cc in info["cells"][lab]:
                out[r0 + rr, c0 + cc] = fill_color

    return out


def _reachable_chambers(nrows, ncols, h_walls, v_walls, start):
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dc in (-1, 1):
            nc = c + dc
            if 0 <= nc < ncols:
                wall_idx = min(c, nc)
                if v_walls[wall_idx] == 1 and (r, nc) not in seen:
                    seen.add((r, nc))
                    q.append((r, nc))
        for dr in (-1, 1):
            nr = r + dr
            if 0 <= nr < nrows:
                wall_idx = min(r, nr)
                if h_walls[wall_idx] == 1 and (nr, c) not in seen:
                    seen.add((nr, c))
                    q.append((nr, c))
    return seen


def _build_frame_patch(rng, frame_color, fill_color, key_shape):
    for _ in range(FRAME_RETRIES):
        nrows = rng.choice((2, 3))
        ncols = rng.choice((2, 3))
        row_heights = [rng.randint(3, 6) for _ in range(nrows)]
        col_widths = [rng.randint(3, 6) for _ in range(ncols)]
        h_walls = [rng.randint(1, 2) for _ in range(nrows - 1)]
        v_walls = [rng.randint(1, 2) for _ in range(ncols - 1)]
        wall_thicknesses = h_walls + v_walls
        if 1 not in wall_thicknesses or 2 not in wall_thicknesses:
            continue

        total = nrows * ncols
        start = (rng.randrange(nrows), rng.randrange(ncols))
        reachable = _reachable_chambers(nrows, ncols, h_walls, v_walls, start)
        if not (1 < len(reachable) < total):
            continue

        h = 2 + sum(row_heights) + sum(h_walls)
        w = 2 + sum(col_widths) + sum(v_walls)
        patch = np.zeros((h, w), dtype=int)
        patch[0, :] = frame_color
        patch[-1, :] = frame_color
        patch[:, 0] = frame_color
        patch[:, -1] = frame_color

        row_ranges = []
        cur_r = 1
        for idx, span in enumerate(row_heights):
            row_ranges.append((cur_r, cur_r + span - 1))
            cur_r += span
            if idx < len(h_walls):
                patch[cur_r : cur_r + h_walls[idx], 1:-1] = frame_color
                cur_r += h_walls[idx]

        col_ranges = []
        cur_c = 1
        for idx, span in enumerate(col_widths):
            col_ranges.append((cur_c, cur_c + span - 1))
            cur_c += span
            if idx < len(v_walls):
                patch[1:-1, cur_c : cur_c + v_walls[idx]] = frame_color
                cur_c += v_walls[idx]

        side = rng.choice(SIDES)
        if side == "top":
            band = rng.randrange(ncols)
            lo, hi = col_ranges[band]
            choices = list(range(lo + 1, hi))
            if not choices:
                continue
            gap = (0, rng.choice(choices))
            entry = (1, gap[1])
        elif side == "bottom":
            band = rng.randrange(ncols)
            lo, hi = col_ranges[band]
            choices = list(range(lo + 1, hi))
            if not choices:
                continue
            gap = (h - 1, rng.choice(choices))
            entry = (h - 2, gap[1])
        elif side == "left":
            band = rng.randrange(nrows)
            lo, hi = row_ranges[band]
            choices = list(range(lo + 1, hi))
            if not choices:
                continue
            gap = (rng.choice(choices), 0)
            entry = (gap[0], 1)
        else:
            band = rng.randrange(nrows)
            lo, hi = row_ranges[band]
            choices = list(range(lo + 1, hi))
            if not choices:
                continue
            gap = (rng.choice(choices), w - 1)
            entry = (gap[0], w - 2)

        ok = True
        for u, v in key_shape:
            dr, dc = _orient_offset(side, u, v)
            rr, cc = entry[0] + dr, entry[1] + dc
            if not (0 <= rr < h and 0 <= cc < w) or patch[rr, cc] != BG:
                ok = False
                break
            patch[rr, cc] = frame_color
        if not ok:
            continue

        patch[gap] = BG
        sr, sc = start
        rr0, rr1 = row_ranges[sr]
        cc0, cc1 = col_ranges[sc]
        candidates = [
            (rr, cc)
            for rr in range(rr0, rr1 + 1)
            for cc in range(cc0, cc1 + 1)
            if patch[rr, cc] == BG
        ]
        if not candidates:
            continue
        yr, yc = rng.choice(candidates)
        patch[yr, yc] = YELLOW

        return {
            "patch": patch,
            "fill_color": fill_color,
            "row_ranges": row_ranges,
            "col_ranges": col_ranges,
            "reachable": reachable,
        }

    return None


def _build_fragment_patch(rng, fill_color, key_shape):
    turns = rng.randrange(4)
    pts = _rotate_shape(key_shape, turns)
    h = 1 + max(r for r, _ in pts)
    w = 1 + max(c for _, c in pts)
    patch = np.zeros((h, w), dtype=int)
    for r, c in pts:
        patch[r, c] = fill_color
    return patch


def _place_patch(rng, canvas, blocked, patch, margin):
    h, w = canvas.shape
    ph, pw = patch.shape
    for _ in range(PLACEMENT_RETRIES):
        r0 = rng.randint(0, h - ph)
        c0 = rng.randint(0, w - pw)
        rr0 = max(0, r0 - margin)
        cc0 = max(0, c0 - margin)
        rr1 = min(h, r0 + ph + margin)
        cc1 = min(w, c0 + pw + margin)
        if np.any(blocked[rr0:rr1, cc0:cc1]):
            continue
        view = canvas[r0 : r0 + ph, c0 : c0 + pw]
        mask = patch != BG
        view[mask] = patch[mask]
        blocked[r0 : r0 + ph, c0 : c0 + pw] = True
        return r0, c0
    return None


def _render_expected(grid, frame_metas, fragment_metas):
    expected = np.array(grid, dtype=int, copy=True)

    for meta in fragment_metas:
        r0, c0 = meta["origin"]
        patch = meta["patch"]
        mask = patch != BG
        expected[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]][mask] = BG

    for meta in frame_metas:
        r0, c0 = meta["origin"]
        patch = meta["patch"]
        fill_color = meta["fill_color"]
        for chamber_r, chamber_c in meta["reachable"]:
            rr0, rr1 = meta["row_ranges"][chamber_r]
            cc0, cc1 = meta["col_ranges"][chamber_c]
            local = patch[rr0 : rr1 + 1, cc0 : cc1 + 1]
            mask = (local == BG) | (local == YELLOW)
            expected[r0 + rr0 : r0 + rr1 + 1, c0 + cc0 : c0 + cc1 + 1][mask] = fill_color

    return expected


def _generate_scene(rng):
    count = rng.choice((1, 2, 3))
    colors = [c for c in range(1, 10) if c != YELLOW]
    rng.shuffle(colors)
    frame_colors = colors[:count]
    fill_colors = colors[count : 2 * count]
    key_shapes = rng.sample(KEY_SHAPES, count)

    frame_specs = []
    fragment_specs = []
    for idx in range(count):
        frame_meta = _build_frame_patch(rng, frame_colors[idx], fill_colors[idx], key_shapes[idx])
        if frame_meta is None:
            return None
        frame_specs.append(frame_meta)
        fragment_specs.append({
            "patch": _build_fragment_patch(rng, fill_colors[idx], key_shapes[idx]),
        })

    all_patches = [m["patch"] for m in frame_specs] + [m["patch"] for m in fragment_specs]
    total_rect = sum((p.shape[0] + 2) * (p.shape[1] + 2) for p in all_patches)
    max_h = max(p.shape[0] for p in all_patches)
    max_w = max(p.shape[1] for p in all_patches)
    base = int(np.ceil(np.sqrt(total_rect))) + 4
    grid_h = rng.randint(max(base, max_h + 4), max(base, max_h + 4) + 7)
    grid_w = rng.randint(max(base, max_w + 4), max(base, max_w + 4) + 7)

    grid = np.zeros((grid_h, grid_w), dtype=int)
    blocked = np.zeros((grid_h, grid_w), dtype=bool)

    frame_metas = []
    fragment_metas = []

    for spec in sorted(frame_specs, key=lambda x: x["patch"].shape[0] * x["patch"].shape[1], reverse=True):
        origin = _place_patch(rng, grid, blocked, spec["patch"], margin=1)
        if origin is None:
            return None
        placed = dict(spec)
        placed["origin"] = origin
        frame_metas.append(placed)

    for spec in fragment_specs:
        origin = _place_patch(rng, grid, blocked, spec["patch"], margin=1)
        if origin is None:
            return None
        placed = dict(spec)
        placed["origin"] = origin
        fragment_metas.append(placed)

    expected = _render_expected(grid, frame_metas, fragment_metas)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _generate_scene(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
