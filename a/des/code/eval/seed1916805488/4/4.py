from __future__ import annotations

import random
from collections import Counter

import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PALETTE = list(range(10))
BG = 0
GEN_RETRIES = 400


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _find_palette_row(g: np.ndarray) -> tuple[int, list[int]]:
    h, w = g.shape
    for r in range(h - 1, -1, -1):
        nz = [c for c in range(w) if int(g[r, c]) != BG]
        if len(nz) < 2:
            continue
        if any(nz[i + 1] - nz[i] < 2 for i in range(len(nz) - 1)):
            continue
        if any(int(g[r, c]) != BG for c in range(nz[0], nz[-1] + 1) if c not in nz):
            continue
        return r, [int(g[r, c]) for c in nz]
    raise AssertionError("palette row not found")


def _detect_frame(comp: list[tuple[int, int]], color: int, g: np.ndarray) -> dict:
    r0, c0, r1, c1 = _bbox(comp)
    if r1 - r0 < 2 or c1 - c0 < 2:
        raise AssertionError("frame too small")
    border_cells = []
    missing = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if r in (r0, r1) or c in (c0, c1):
                border_cells.append((r, c))
                if int(g[r, c]) != color:
                    missing.append((r, c))
    if len(missing) != 1:
        raise AssertionError("expected one gap")
    gap = missing[0]
    expected = set(border_cells) - {gap}
    if set(comp) != expected:
        raise AssertionError("component is not a rectangle border with one gap")

    gr, gc = gap
    if gr == r0:
        side = "top"
        step = (1, 0)
        if gc != (c0 + c1) // 2:
            raise AssertionError("top gap must be centered")
    elif gr == r1:
        side = "bottom"
        step = (-1, 0)
        if gc != (c0 + c1) // 2:
            raise AssertionError("bottom gap must be centered")
    elif gc == c0:
        side = "left"
        step = (0, 1)
        if gr != (r0 + r1) // 2:
            raise AssertionError("left gap must be centered")
    elif gc == c1:
        side = "right"
        step = (0, -1)
        if gr != (r0 + r1) // 2:
            raise AssertionError("right gap must be centered")
    else:
        raise AssertionError("gap must lie on a side")

    return {
        "color": color,
        "bbox": (r0, c0, r1, c1),
        "gap": gap,
        "side": side,
        "ray_step": step,
    }


def _group_frames(g: np.ndarray) -> list[list[dict]]:
    groups: list[list[dict]] = []
    for color in sorted({int(v) for v in np.unique(g) if int(v) != BG}):
        comps = _components(g == color)
        if not comps:
            continue
        frames = [_detect_frame(comp, color, g) for comp in comps]
        frames.sort(key=lambda fr: (fr["bbox"][0], fr["bbox"][1], -fr["bbox"][2], -fr["bbox"][3]))
        for i in range(len(frames) - 1):
            r0, c0, r1, c1 = frames[i]["bbox"]
            ir0, ic0, ir1, ic1 = frames[i + 1]["bbox"]
            if not (r0 < ir0 and c0 < ic0 and ir1 < r1 and ic1 < c1):
                raise AssertionError("frames of a group must be nested")
        groups.append(sorted(frames, key=lambda fr: (fr["bbox"][2] - fr["bbox"][0], fr["bbox"][3] - fr["bbox"][1]), reverse=True))
    return groups


def _ring_mask(shape: tuple[int, int], frames: list[dict], idx: int) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    r0, c0, r1, c1 = frames[idx]["bbox"]
    mask[r0 + 1 : r1, c0 + 1 : c1] = True
    if idx + 1 < len(frames):
        ir0, ic0, ir1, ic1 = frames[idx + 1]["bbox"]
        mask[ir0 : ir1 + 1, ic0 : ic1 + 1] = False
    return mask


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    palette_row, seq = _find_palette_row(g)
    work = g[:palette_row, :].copy()
    out = work.copy()
    groups = _group_frames(work)
    assert groups, "expected at least one nested-frame group"

    max_depth = max(len(group) for group in groups)
    assert len(seq) == max_depth, "palette length must equal maximum nesting depth"

    for group in groups:
        for idx, frame in enumerate(group):
            fill_color = seq[idx]
            region = _ring_mask(work.shape, group, idx)
            out[region] = fill_color

        for idx, frame in enumerate(group):
            ray_color = seq[(idx + 1) % len(seq)]
            region = _ring_mask(work.shape, group, idx)
            gr, gc = frame["gap"]
            dr, dc = frame["ray_step"]
            rr, cc = gr + dr, gc + dc
            while 0 <= rr < h and 0 <= cc < w and region[rr, cc]:
                out[rr, cc] = ray_color
                rr += dr
                cc += dc

    full = np.zeros_like(g)
    full[:palette_row, :] = out
    return full


def _rand_odd(lo: int, hi: int) -> int:
    vals = [v for v in range(lo, hi + 1) if v % 2 == 1]
    return random.choice(vals)


def _draw_group(
    grid: np.ndarray,
    top: int,
    left: int,
    layers: int,
    outer_h: int,
    outer_w: int,
    border_color: int,
    directions: list[str],
) -> None:
    for k in range(layers):
        r0 = top + 2 * k
        c0 = left + 2 * k
        r1 = top + outer_h - 1 - 2 * k
        c1 = left + outer_w - 1 - 2 * k
        grid[r0, c0 : c1 + 1] = border_color
        grid[r1, c0 : c1 + 1] = border_color
        grid[r0 : r1 + 1, c0] = border_color
        grid[r0 : r1 + 1, c1] = border_color

        side = directions[k]
        if side == "top":
            grid[r0, (c0 + c1) // 2] = BG
        elif side == "bottom":
            grid[r1, (c0 + c1) // 2] = BG
        elif side == "left":
            grid[(r0 + r1) // 2, c0] = BG
        else:
            grid[(r0 + r1) // 2, c1] = BG


def _group_bbox(layers: int, outer_h: int, outer_w: int) -> tuple[int, int]:
    min_needed = 4 * (layers - 1) + 3
    assert outer_h >= min_needed and outer_w >= min_needed
    return outer_h, outer_w


def _try_place_groups(
    h: int, w: int, palette_row: int, group_specs: list[dict]
) -> list[tuple[int, int]] | None:
    order = list(range(len(group_specs)))
    random.shuffle(order)
    temp: list[tuple[int, int] | None] = [None] * len(group_specs)
    widths = [group_specs[idx]["outer_w"] for idx in order]
    total_width = sum(widths)
    min_gaps = len(order) + 1
    if total_width + min_gaps > w:
        return None
    slack = w - total_width - min_gaps
    gaps = [1] * (len(order) + 1)
    for _ in range(slack):
        gaps[random.randrange(len(gaps))] += 1

    x = gaps[0]
    for pos, idx in enumerate(order):
        gh = group_specs[idx]["outer_h"]
        gw = group_specs[idx]["outer_w"]
        if palette_row - gh - 2 < 1:
            return None
        top = random.randint(1, palette_row - gh - 2)
        temp[idx] = (top, x)
        x += gw + gaps[pos + 1]
    return [p for p in temp if p is not None]


def _make_palette_row(w: int, seq: list[int]) -> np.ndarray:
    row = np.zeros(w, dtype=int)
    span = 2 * len(seq) - 1
    start = random.randint(1, w - span - 1)
    for i, color in enumerate(seq):
        row[start + 2 * i] = color
    return row


def _generate_attempt() -> np.ndarray | None:
    max_depth = random.randint(1, 4)
    n_groups = random.randint(1, 3)

    seq_colors = random.sample([c for c in PALETTE if c != BG], max_depth)
    border_choices = [c for c in PALETTE if c not in seq_colors and c != BG]
    if len(border_choices) < n_groups:
        return None
    border_colors = random.sample(border_choices, n_groups)

    group_specs: list[dict] = []
    for gi in range(n_groups):
        layers = max_depth if gi == 0 else random.randint(1, max_depth)
        min_side = 4 * (layers - 1) + 3
        outer_h = _rand_odd(max(7, min_side), max(7, min_side) + 6)
        outer_w = _rand_odd(max(7, min_side), max(7, min_side) + 6)
        directions = [random.choice(["top", "bottom", "left", "right"]) for _ in range(layers)]
        group_specs.append(
            {
                "layers": layers,
                "outer_h": outer_h,
                "outer_w": outer_w,
                "color": border_colors[gi],
                "directions": directions,
            }
        )

    min_h = max(spec["outer_h"] for spec in group_specs) + 5
    min_w = sum(spec["outer_w"] for spec in group_specs) + 2 * n_groups + 4
    h = random.randint(max(14, min_h), max(22, min_h + 4))
    w = random.randint(max(16, min_w), max(28, min_w + 6))
    palette_row = h - 1

    placements = _try_place_groups(h, w, palette_row, group_specs)
    if placements is None:
        return None

    grid = np.zeros((h, w), dtype=int)
    for spec, (top, left) in zip(group_specs, placements, strict=False):
        _draw_group(
            grid,
            top,
            left,
            spec["layers"],
            spec["outer_h"],
            spec["outer_w"],
            spec["color"],
            spec["directions"],
        )

    grid[palette_row, :] = _make_palette_row(w, seq_colors)

    try:
        out = main(grid)
    except Exception:
        return None

    if out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None
    if np.count_nonzero(out[:-1]) < 8:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
