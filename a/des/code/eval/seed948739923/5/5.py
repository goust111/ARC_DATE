import random
from collections import defaultdict, deque

import numpy as np


BG = 0
GREY = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_RETRIES = 300


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _mask_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _color_components(grid: np.ndarray, blocked: np.ndarray | None = None) -> list[dict]:
    h, w = grid.shape
    if blocked is None:
        blocked = np.zeros((h, w), dtype=bool)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or blocked[r, c] or color in (BG, GREY):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or blocked[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            comps.append({"color": color, "cells": cells})
    return comps


def _parse_panel_interior(interior: np.ndarray) -> dict[frozenset[int], int] | None:
    mapping: dict[frozenset[int], int] = {}
    nonblank_rows = 0
    for row in interior:
        vals = [int(v) for v in row if int(v) != BG]
        if not vals:
            continue
        if len(vals) < 2:
            return None
        new_color = vals[0]
        signature = frozenset(vals[1:])
        if not signature or new_color in (BG, GREY) or any(v in (BG, GREY) for v in signature):
            return None
        if signature in mapping and mapping[signature] != new_color:
            return None
        mapping[signature] = new_color
        nonblank_rows += 1
    if nonblank_rows < 2:
        return None
    return mapping


def _find_panel(grid: np.ndarray) -> tuple[tuple[int, int, int, int], dict[frozenset[int], int]]:
    candidates: list[tuple[tuple[int, int, int, int], dict[frozenset[int], int]]] = []
    for comp in _mask_components(grid == GREY):
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        border = set()
        for c in range(c0, c1 + 1):
            border.add((r0, c))
            border.add((r1, c))
        for r in range(r0, r1 + 1):
            border.add((r, c0))
            border.add((r, c1))
        if set(comp) != border:
            continue
        mapping = _parse_panel_interior(grid[r0 + 1 : r1, c0 + 1 : c1])
        if mapping:
            candidates.append(((r0, c0, r1, c1), mapping))
    assert candidates, "main: could not locate grey-framed legend panel"
    candidates.sort(key=lambda item: (len(item[1]), (item[0][2] - item[0][0]) * (item[0][3] - item[0][1])), reverse=True)
    return candidates[0]


def _region_data(grid: np.ndarray, blocked: np.ndarray) -> list[dict]:
    h, w = grid.shape
    regions = []
    for comp in _color_components(grid, blocked):
        color = int(comp["color"])
        neighbors: set[int] = set()
        for r, c in comp["cells"]:
            for dr, dc in NEIGHBORS4:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w) or blocked[nr, nc]:
                    continue
                other = int(grid[nr, nc])
                if other in (BG, GREY, color):
                    continue
                neighbors.add(other)
        regions.append({"color": color, "cells": comp["cells"], "signature": frozenset(neighbors)})
    return regions


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    (r0, c0, r1, c1), mapping = _find_panel(g)

    panel_mask = np.zeros_like(g, dtype=bool)
    panel_mask[r0 : r1 + 1, c0 : c1 + 1] = True

    out = g.copy()
    out[panel_mask] = BG

    for region in _region_data(g, panel_mask):
        signature = region["signature"]
        assert signature in mapping, f"main: missing legend entry for signature {sorted(signature)}"
        new_color = int(mapping[signature])
        for r, c in region["cells"]:
            out[r, c] = new_color

    return out


def _gap_segments(width: int, intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    segments = []
    x = 0
    for start, end in sorted(intervals):
        if x < start:
            segments.append((x, start))
        x = max(x, end)
    if x < width:
        segments.append((x, width))
    return segments


def _add_teeth(
    patch: np.ndarray,
    start_row: int,
    direction: int,
    segments: list[tuple[int, int]],
    color: int,
    rng: random.Random,
    max_depth: int,
) -> None:
    h, w = patch.shape
    for seg0, seg1 in segments:
        span = seg1 - seg0
        if span <= 0 or rng.random() < 0.45:
            continue
        tooth_w = rng.randint(1, min(2, span))
        tooth_c0 = rng.randint(seg0, seg1 - tooth_w)
        depth = rng.randint(1, max_depth)
        for step in range(depth):
            rr = start_row + direction * step
            if not (0 <= rr < h):
                break
            for cc in range(tooth_c0, tooth_c0 + tooth_w):
                if 0 <= cc < w and patch[rr, cc] == BG:
                    patch[rr, cc] = color


def _place_non_touching_intervals(
    width: int,
    count: int,
    rng: random.Random,
    w_lo: int = 1,
    w_hi: int = 2,
) -> list[tuple[int, int]] | None:
    intervals: list[tuple[int, int]] = []
    for _ in range(200):
        intervals.clear()
        ok = True
        for _ in range(count):
            placed = False
            for _ in range(80):
                span = rng.randint(w_lo, min(w_hi, width))
                x0 = rng.randint(0, width - span)
                x1 = x0 + span
                if any(not (x1 + 1 <= a or b + 1 <= x0) for a, b in intervals):
                    continue
                intervals.append((x0, x1))
                placed = True
                break
            if not placed:
                ok = False
                break
        if ok:
            intervals.sort()
            return intervals
    return None


def _build_template_a(rng: random.Random) -> np.ndarray:
    colors = rng.sample(COLOR_POOL, 4)
    pillar_colors = colors[:2]
    bar_colors = colors[2:]

    pillar_count = rng.randint(3, 6)
    widths = [rng.randint(2, 4) for _ in range(pillar_count)]
    gaps = [rng.randint(1, 3) for _ in range(pillar_count - 1)]
    left_pad = rng.randint(1, 3)
    right_pad = rng.randint(1, 3)
    top_t = rng.randint(2, 3)
    bottom_t = rng.randint(2, 3)
    middle_h = rng.randint(5, 8)

    patch_h = top_t + middle_h + bottom_t
    patch_w = left_pad + sum(widths) + sum(gaps) + right_pad
    patch = np.zeros((patch_h, patch_w), dtype=int)
    patch[:top_t, :] = bar_colors[0]
    patch[-bottom_t:, :] = bar_colors[1]

    pillar_palette = pillar_colors * ((pillar_count + 1) // 2)
    pillar_palette = pillar_palette[:pillar_count]
    rng.shuffle(pillar_palette)
    if len(set(pillar_palette)) < 2:
        pillar_palette[0] = pillar_colors[0]
        pillar_palette[-1] = pillar_colors[1]

    intervals: list[tuple[int, int]] = []
    x = left_pad
    for idx, span in enumerate(widths):
        color = pillar_palette[idx]
        patch[top_t : patch_h - bottom_t, x : x + span] = color
        intervals.append((x, x + span))
        if idx < len(gaps):
            x += span + gaps[idx]

    gap_segments = _gap_segments(patch_w, intervals)
    max_depth = max(1, middle_h // 3)
    _add_teeth(patch, top_t, 1, gap_segments, bar_colors[0], rng, max_depth)
    _add_teeth(patch, patch_h - bottom_t - 1, -1, gap_segments, bar_colors[1], rng, max_depth)

    return np.rot90(patch, rng.randint(0, 3)).copy()


def _build_template_c(rng: random.Random) -> np.ndarray:
    colors = rng.sample(COLOR_POOL, 5)
    bridge_colors = colors[:3]
    top_color, bottom_color = colors[3], colors[4]

    bridge_count = rng.randint(3, 5)
    widths = [rng.randint(2, 4) for _ in range(bridge_count)]
    gaps = [rng.randint(1, 3) for _ in range(bridge_count - 1)]
    left_pad = rng.randint(1, 3)
    right_pad = rng.randint(1, 3)
    sat_count = rng.randint(2, 4)
    sat_heights = [rng.randint(2, 4) for _ in range(sat_count)]
    sat_top = max(sat_heights)
    top_t = rng.randint(2, 3)
    bottom_t = rng.randint(2, 3)
    middle_h = rng.randint(5, 8)

    patch_h = sat_top + top_t + middle_h + bottom_t
    patch_w = left_pad + sum(widths) + sum(gaps) + right_pad
    patch = np.zeros((patch_h, patch_w), dtype=int)

    top_start = sat_top
    top_end = top_start + top_t
    bottom_start = top_end + middle_h

    patch[top_start:top_end, :] = top_color
    patch[bottom_start:, :] = bottom_color

    bridge_palette = bridge_colors[:]
    while len(bridge_palette) < bridge_count:
        bridge_palette.append(rng.choice(bridge_colors))
    rng.shuffle(bridge_palette)
    for base_color in bridge_colors:
        if base_color not in bridge_palette:
            bridge_palette[rng.randrange(bridge_count)] = base_color

    intervals: list[tuple[int, int]] = []
    x = left_pad
    for idx, span in enumerate(widths):
        color = bridge_palette[idx]
        patch[top_end:bottom_start, x : x + span] = color
        intervals.append((x, x + span))
        if idx < len(gaps):
            x += span + gaps[idx]

    sat_intervals = _place_non_touching_intervals(patch_w, sat_count, rng)
    assert sat_intervals is not None, "template_c: failed to place satellites"
    sat_colors = bridge_colors[:2]
    while len(sat_colors) < sat_count:
        sat_colors.append(rng.choice(bridge_colors))
    rng.shuffle(sat_colors)
    for idx, (x0, x1) in enumerate(sat_intervals):
        sat_h = sat_heights[idx]
        color = sat_colors[idx]
        patch[top_start - sat_h : top_start, x0:x1] = color

    gap_segments = _gap_segments(patch_w, intervals)
    max_depth = max(1, middle_h // 3)
    _add_teeth(patch, top_end, 1, gap_segments, top_color, rng, max_depth)
    _add_teeth(patch, bottom_start - 1, -1, gap_segments, bottom_color, rng, max_depth)

    return np.rot90(patch, rng.randint(0, 3)).copy()


def _summarize_signatures(patch: np.ndarray) -> tuple[list[dict], dict[frozenset[int], set[int]]]:
    blocked = np.zeros_like(patch, dtype=bool)
    regions = _region_data(patch, blocked)
    sig_to_colors: dict[frozenset[int], set[int]] = defaultdict(set)
    for region in regions:
        sig_to_colors[region["signature"]].add(int(region["color"]))
    return regions, sig_to_colors


def _validate_patch(patch: np.ndarray, template: str) -> dict[frozenset[int], set[int]] | None:
    regions, sig_to_colors = _summarize_signatures(patch)
    input_colors = {int(region["color"]) for region in regions}
    if len(regions) < 5 or len(input_colors) not in (4, 5):
        return None
    if not (2 <= len(sig_to_colors) <= 3):
        return None
    if any(len(sig) == 0 or len(sig) > 3 for sig in sig_to_colors):
        return None
    if not any(len(colors) >= 2 for colors in sig_to_colors.values()):
        return None

    size_profile = {len(sig) for sig in sig_to_colors}
    if template == "a":
        if len(sig_to_colors) != 2 or size_profile != {2}:
            return None
    else:
        if len(sig_to_colors) != 3 or size_profile != {1, 2, 3}:
            return None
    return sig_to_colors


def _build_panel(
    sig_to_colors: dict[frozenset[int], set[int]],
    input_colors: set[int],
    rng: random.Random,
) -> tuple[np.ndarray, dict[frozenset[int], int]]:
    signatures = list(sig_to_colors)
    out_colors = [c for c in COLOR_POOL if c not in input_colors]
    assert len(out_colors) >= len(signatures), "generator: insufficient output colors"
    rng.shuffle(out_colors)
    mapping = {sig: out_colors[idx] for idx, sig in enumerate(signatures)}

    rows = signatures[:]
    rng.shuffle(rows)
    max_sig = max(len(sig) for sig in rows)
    extra_pad = rng.randint(0, 1)
    interior = np.zeros((len(rows), 1 + max_sig + extra_pad), dtype=int)
    for rr, sig in enumerate(rows):
        interior[rr, 0] = mapping[sig]
        colors = list(sig)
        rng.shuffle(colors)
        interior[rr, 1 : 1 + len(colors)] = colors

    panel = np.zeros((interior.shape[0] + 2, interior.shape[1] + 2), dtype=int)
    panel[0, :] = GREY
    panel[-1, :] = GREY
    panel[:, 0] = GREY
    panel[:, -1] = GREY
    panel[1:-1, 1:-1] = interior
    return panel, mapping


def _compose_scene(map_patch: np.ndarray, panel: np.ndarray, rng: random.Random) -> np.ndarray:
    map_h, map_w = map_patch.shape
    panel_h, panel_w = panel.shape
    orientation = rng.choice(["right", "bottom"])
    gap = rng.randint(1, 3)
    top_pad = rng.randint(1, 2)
    left_pad = rng.randint(1, 2)
    bottom_pad = rng.randint(1, 2)
    right_pad = rng.randint(1, 2)

    if orientation == "right":
        extra_h = rng.randint(0, 2)
        h = max(map_h, panel_h) + top_pad + bottom_pad + extra_h
        w = map_w + gap + panel_w + left_pad + right_pad + rng.randint(0, 2)
        grid = np.zeros((h, w), dtype=int)
        map_r = rng.randint(top_pad, h - bottom_pad - map_h)
        panel_r = rng.randint(top_pad, h - bottom_pad - panel_h)
        map_c = left_pad
        panel_c = w - right_pad - panel_w
    else:
        extra_w = rng.randint(0, 2)
        h = map_h + gap + panel_h + top_pad + bottom_pad + rng.randint(0, 2)
        w = max(map_w, panel_w) + left_pad + right_pad + extra_w
        grid = np.zeros((h, w), dtype=int)
        map_r = top_pad
        panel_r = h - bottom_pad - panel_h
        map_c = rng.randint(left_pad, w - right_pad - map_w)
        panel_c = rng.randint(left_pad, w - right_pad - panel_w)

    grid[map_r : map_r + map_h, map_c : map_c + map_w] = map_patch
    panel_target = grid[panel_r : panel_r + panel_h, panel_c : panel_c + panel_w]
    panel_target[panel != BG] = panel[panel != BG]
    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    template = rng.choice(["a", "c"])
    map_patch = _build_template_a(rng) if template == "a" else _build_template_c(rng)

    sig_to_colors = _validate_patch(map_patch, template)
    if sig_to_colors is None:
        return None

    input_colors = {int(region["color"]) for region in _region_data(map_patch, np.zeros_like(map_patch, dtype=bool))}
    panel, mapping = _build_panel(sig_to_colors, input_colors, rng)
    scene = _compose_scene(map_patch, panel, rng)

    try:
        _, parsed_mapping = _find_panel(scene)
        if parsed_mapping != mapping:
            return None
        out = main(scene)
    except Exception:
        return None

    if not np.any(out != scene):
        return None
    if np.any(out == GREY):
        return None
    return scene


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
