from __future__ import annotations

import math
import random
from collections import Counter

import numpy as np


BG = 0
LINE = 5
FILL_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_SIDE = 30
MAX_THUMB = 5
GEN_RETRIES = 5000
SAFE_K4_STARTS = (
    (8, 8, 9, 9),
    (8, 8, 8, 9),
    (7, 8, 8, 9),
    (7, 8, 9, 9),
)
SAFE_K3_STARTS = (
    (8, 8, 9),
    (8, 9, 9),
    (8, 8, 8),
)
SAFE_K3_TEMPLATES_WITH_4 = (
    (2, 2, 2, 2, 2, 3, 2, 2, 4),
    (2, 2, 2, 2, 2, 4, 2, 2, 3),
    (2, 2, 2, 2, 3, 2, 2, 4, 2),
)


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    return [
        (r - 1, c - 1),
        (r - 1, c),
        (r - 1, c + 1),
        (r, c - 1),
        (r, c + 1),
        (r + 1, c - 1),
        (r + 1, c),
        (r + 1, c + 1),
    ]


def _components_mask(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                if connectivity == 8:
                    neigh = _neighbors8(cr, cc)
                else:
                    neigh = [(cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)]
                for nr, nc in neigh:
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _extract_thumbnail(grid: np.ndarray, k: int) -> np.ndarray:
    assert 2 <= k <= 5, f"unexpected thumbnail size {k}"
    thumb = np.asarray(grid[:k, :k], dtype=int).copy()
    assert np.any(thumb != BG), "missing top-left thumbnail"
    assert not np.any(thumb == LINE), "thumbnail must not contain the grey segment color"
    return thumb


def _segment_info(grid: np.ndarray) -> list[dict]:
    segments: list[dict] = []
    by_sum: dict[int, list[int]] = {}
    for r, c in np.argwhere(grid == LINE):
        by_sum.setdefault(int(r + c), []).append(int(r - c))
    for k, bs_unsorted in by_sum.items():
        bs = sorted(bs_unsorted)
        run = [bs[0]]
        for b in bs[1:]:
            if b == run[-1] + 2:
                run.append(b)
                continue
            segments.append({"k": int(k), "bmin": int(run[0]), "bmax": int(run[-1]), "cells": set(_segment_cells(k, run[0], run[-1]))})
            run = [b]
        segments.append({"k": int(k), "bmin": int(run[0]), "bmax": int(run[-1]), "cells": set(_segment_cells(k, run[0], run[-1]))})
    return segments


def _pair_infos(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    grouped: dict[tuple[int, int], list[dict]] = {}
    for seg in _segment_info(grid):
        grouped.setdefault((seg["bmin"], seg["bmax"]), []).append(seg)

    pairs: list[dict] = []
    for (bmin, bmax), segs in grouped.items():
        assert len(segs) == 2, "each b-range must identify exactly one segment pair"
        segs = sorted(segs, key=lambda s: s["k"])
        k1, k2 = int(segs[0]["k"]), int(segs[1]["k"])
        region: set[tuple[int, int]] = set()
        for r in range(h):
            for c in range(w):
                b = r - c
                k = r + c
                if bmin <= b <= bmax and k1 < k < k2:
                    region.add((r, c))
        assert region, "segment pair must bound a visible region"
        center_r = ((k1 + k2) / 2.0 + (bmin + bmax) / 2.0) / 2.0
        center_c = ((k1 + k2) / 2.0 - (bmin + bmax) / 2.0) / 2.0
        pairs.append(
            {
                "bmin": bmin,
                "bmax": bmax,
                "k1": k1,
                "k2": k2,
                "region": region,
                "segments": segs,
                "center_r": center_r,
                "center_c": center_c,
            }
        )
    pairs.sort(key=lambda p: (p["center_r"], p["center_c"]))
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill selected diagonal-strip regions according to the top-left thumbnail.

    Input structure:
    - Background cells are black.
    - The top-left corner contains a small kxk thumbnail whose cells are either black or non-grey colors.
    - Elsewhere, grey anti-diagonal segments appear in matched pairs; some pair-bounded regions may already be filled.

    Rule:
    - Sort the segment pairs by the midpoint of the region between them, using row-major order.
    - Read the thumbnail in row-major order; the i-th thumbnail cell controls the i-th segment pair.
    - If a pair-region is already prefilled, its color must match the corresponding thumbnail cell.
    - For every non-black thumbnail cell, fill all still-black cells in its corresponding region with that color.
    - Thumbnail black cells leave their regions empty, and the grey boundary segments stay unchanged.

    Key insight:
    - The thumbnail is not a geometric scale copy of the main scene; it is a row-major control table whose entries map onto scattered anti-diagonal region pairs.
    """
    g = np.asarray(grid, dtype=int)
    pairs = _pair_infos(g)
    assert pairs, "grid must contain paired diagonal segments"
    k = math.isqrt(len(pairs))
    assert k * k == len(pairs), "segment-pair count must be a perfect square"
    thumb = _extract_thumbnail(g, k)

    out = g.copy()
    for idx, pair in enumerate(pairs):
        color = int(thumb.flat[idx])
        existing = {int(g[r, c]) for r, c in pair["region"] if int(g[r, c]) not in (BG, LINE)}
        if existing:
            assert len(existing) == 1, "a prefilled region must have a single color"
            anchor_color = next(iter(existing))
            assert color != BG and anchor_color == color, "anchor color must agree with the thumbnail"
        if color == BG:
            continue
        for r, c in pair["region"]:
            if out[r, c] == BG:
                out[r, c] = color
    return out


def _segment_cells(k: int, bmin: int, bmax: int) -> list[tuple[int, int]]:
    return [((k + b) // 2, (k - b) // 2) for b in range(bmin, bmax + 1, 2)]


def _pair_from_center(center_r: int, center_c: int, length: int, span: int) -> dict | None:
    kmid = center_r + center_c
    bmid = center_r - center_c
    bmin = bmid - (length - 1)
    bmax = bmid + (length - 1)
    k1 = kmid - span // 2
    k2 = kmid + span // 2
    if (k1 - bmin) % 2 != 0:
        return None

    seg1 = _segment_cells(k1, bmin, bmax)
    seg2 = _segment_cells(k2, bmin, bmax)
    if len(set(seg1) | set(seg2)) != 2 * length:
        return None

    region: set[tuple[int, int]] = set()
    for r in range(MAX_SIDE):
        for c in range(MAX_SIDE):
            b = r - c
            k = r + c
            if bmin <= b <= bmax and k1 < k < k2:
                region.add((r, c))

    if not region:
        return None
    full = set(seg1) | set(seg2) | region
    if any(r < 0 or c < 0 or r >= MAX_SIDE or c >= MAX_SIDE for r, c in full):
        return None
    if any((r < MAX_THUMB and c < MAX_THUMB) for r, c in full):
        return None

    return {
        "bmin": bmin,
        "bmax": bmax,
        "k1": k1,
        "k2": k2,
        "seg1": seg1,
        "seg2": seg2,
        "region": region,
        "full": full,
        "center_r": float(center_r),
        "center_c": float(center_c),
    }


def _build_pairs(k: int) -> list[dict] | None:
    if k == 4:
        row_shift = random.choice([0, 1])
        row_centers = [8 + row_shift + 5 * i for i in range(4)]
        starts = random.choice(SAFE_K4_STARTS)
        lengths = [random.choice([2, 3]) for _ in range(16)]
    else:
        row_shift = random.choice([0, 1, 2])
        row_centers = [8 + row_shift + 5 * i for i in range(3)]
        starts = random.choice(SAFE_K3_STARTS)
        if random.random() < 0.45:
            lengths = list(random.choice(SAFE_K3_TEMPLATES_WITH_4))
        else:
            lengths = [random.choice([2, 3]) for _ in range(9)]

    pairs: list[dict] = []
    used_ranges: set[tuple[int, int]] = set()
    line_runs: dict[int, list[tuple[int, int]]] = {}
    occupied: set[tuple[int, int]] = set()
    idx = 0
    for rr in range(k):
        cols = [starts[rr] + 6 * i for i in range(k)]
        for cc in range(k):
            length = int(lengths[idx])
            idx += 1
            span = 4 if length == 3 else 6
            pair = _pair_from_center(row_centers[rr], cols[cc], length, span)
            if pair is None:
                return None
            key = (pair["bmin"], pair["bmax"])
            if key in used_ranges or occupied & pair["full"]:
                return None
            for seg in (pair["seg1"], pair["seg2"]):
                k_line = int(seg[0][0] + seg[0][1])
                b0 = min(r - c for r, c in seg)
                b1 = max(r - c for r, c in seg)
                if any(b0 <= old_b1 + 2 and b1 >= old_b0 - 2 for old_b0, old_b1 in line_runs.get(k_line, [])):
                    return None
            used_ranges.add(key)
            for seg in (pair["seg1"], pair["seg2"]):
                k_line = int(seg[0][0] + seg[0][1])
                b0 = min(r - c for r, c in seg)
                b1 = max(r - c for r, c in seg)
                line_runs.setdefault(k_line, []).append((b0, b1))
            occupied |= pair["full"]
            pairs.append(pair)

    max_r = max(r for pair in pairs for r, _ in pair["full"])
    max_c = max(c for pair in pairs for _, c in pair["full"])
    if max_r >= MAX_SIDE or max_c >= MAX_SIDE:
        return None
    return pairs


def _make_thumbnail(k: int, anchor_count: int) -> np.ndarray | None:
    total = k * k
    max_non_bg = total - 1 if total > 2 else total
    min_non_bg = max(anchor_count + 1, total // 2)
    if min_non_bg > max_non_bg:
        return None
    non_bg_count = random.randint(min_non_bg, max_non_bg)
    positions = [(r, c) for r in range(k) for c in range(k)]
    required = {(k - 1, random.randrange(k)), (random.randrange(k), k - 1)}
    chosen = set(required)
    others = [pos for pos in positions if pos not in chosen]
    random.shuffle(others)
    chosen.update(others[: non_bg_count - len(chosen)])
    if len(chosen) != non_bg_count:
        return None

    palette_size = random.randint(2, min(5, non_bg_count))
    palette = random.sample(FILL_COLORS, palette_size)
    thumb = np.full((k, k), BG, dtype=int)

    # Force at least two distinct visible colors so the filled output cannot collapse
    # to a single-colored trivial case.
    first_two = list(chosen)[:2]
    thumb[first_two[0]] = palette[0]
    thumb[first_two[1]] = palette[1 % len(palette)]
    remaining_colors = [random.choice(palette) for _ in range(non_bg_count - 2)]
    random.shuffle(remaining_colors)
    remaining_positions = [pos for pos in chosen if pos not in first_two]
    for pos, color in zip(remaining_positions, remaining_colors):
        thumb[pos] = color

    if len(set(map(int, thumb[thumb != BG].ravel()))) < 2:
        return None
    return thumb


def _render_output(k: int, thumb: np.ndarray, pairs: list[dict]) -> np.ndarray:
    max_r = max(r for pair in pairs for r, _ in pair["full"])
    max_c = max(c for pair in pairs for _, c in pair["full"])
    h = min(MAX_SIDE, max(k + 2, max_r + 1 + random.randint(1, 2)))
    w = min(MAX_SIDE, max(k + 2, max_c + 1 + random.randint(1, 2)))
    grid = np.full((h, w), BG, dtype=int)
    grid[:k, :k] = thumb
    for pair in pairs:
        for r, c in pair["seg1"] + pair["seg2"]:
            grid[r, c] = LINE
    for idx, pair in enumerate(sorted(pairs, key=lambda p: (p["center_r"], p["center_c"]))):
        color = int(thumb.flat[idx])
        if color == BG:
            continue
        for r, c in pair["region"]:
            if r < h and c < w and grid[r, c] == BG:
                grid[r, c] = color
    return grid


def _apply_anchor_mask(
    output_grid: np.ndarray, thumb: np.ndarray, ordered_pairs: list[dict], anchor_indices: set[int]
) -> np.ndarray:
    inp = output_grid.copy()
    for idx, pair in enumerate(ordered_pairs):
        if idx in anchor_indices or int(thumb.flat[idx]) == BG:
            continue
        for r, c in pair["region"]:
            if inp[r, c] != LINE:
                inp[r, c] = BG
    return inp


def _anchor_orders_ok(inp: np.ndarray) -> bool:
    pairs = _pair_infos(inp)
    if not pairs:
        return False
    k = math.isqrt(len(pairs))
    if k * k != len(pairs):
        return False
    thumb = _extract_thumbnail(inp, k)

    anchor_cells = []
    for pair in pairs:
        colors = {int(inp[r, c]) for r, c in pair["region"] if int(inp[r, c]) not in (BG, LINE)}
        if colors:
            if len(colors) != 1:
                return False
            anchor_cells.append(next(iter(colors)))
    if not (1 <= len(anchor_cells) <= 2):
        return False

    candidates = {
        "tlbr": sorted(pairs, key=lambda p: (p["center_r"], p["center_c"])),
        "trbl": sorted(pairs, key=lambda p: (p["center_r"], -p["center_c"])),
        "bltr": sorted(pairs, key=lambda p: (-p["center_r"], p["center_c"])),
        "reverse": list(reversed(sorted(pairs, key=lambda p: (p["center_r"], p["center_c"])))),
    }
    matches = []
    for name, order in candidates.items():
        ok = True
        for idx, pair in enumerate(order):
            colors = {int(inp[r, c]) for r, c in pair["region"] if int(inp[r, c]) not in (BG, LINE)}
            if not colors:
                continue
            if int(thumb.flat[idx]) == BG or next(iter(colors)) != int(thumb.flat[idx]):
                ok = False
                break
        if ok:
            matches.append(name)
    return matches == ["tlbr"]


def _semantic_check(inp: np.ndarray, out: np.ndarray) -> bool:
    try:
        pairs = _pair_infos(inp)
        if not pairs:
            return False
        k = math.isqrt(len(pairs))
        if k * k != len(pairs):
            return False
        thumb = _extract_thumbnail(inp, k)

        non_bg_count = int(np.sum(thumb != BG))
        anchor_count = 0
        for idx, pair in enumerate(pairs):
            colors = {int(inp[r, c]) for r, c in pair["region"] if int(inp[r, c]) not in (BG, LINE)}
            if colors:
                anchor_count += 1
                if len(colors) != 1 or int(thumb.flat[idx]) != next(iter(colors)):
                    return False
            if int(thumb.flat[idx]) == BG:
                if any(int(out[r, c]) != BG for r, c in pair["region"]):
                    return False
            else:
                if not all(int(out[r, c]) == int(thumb.flat[idx]) for r, c in pair["region"]):
                    return False
        if not (1 <= anchor_count <= 2):
            return False
        if non_bg_count <= anchor_count:
            return False
        if len(set(map(int, out.ravel()))) < 4:
            return False
        return _anchor_orders_ok(inp)
    except Exception:
        return False


def _gen_attempt() -> np.ndarray | None:
    k = random.choice([3, 4])
    anchor_count = random.choice([1, 2, 2])
    pairs = _build_pairs(k)
    if pairs is None:
        return None

    ordered_pairs = sorted(pairs, key=lambda p: (p["center_r"], p["center_c"]))
    thumb = _make_thumbnail(k, anchor_count)
    if thumb is None:
        return None

    non_bg_indices = [idx for idx, value in enumerate(thumb.flat) if int(value) != BG]
    if len(non_bg_indices) <= anchor_count:
        return None
    random.shuffle(non_bg_indices)
    anchor_indices = set(non_bg_indices[:anchor_count])

    output_grid = _render_output(k, thumb, pairs)
    inp = _apply_anchor_mask(output_grid, thumb, ordered_pairs, anchor_indices)

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, output_grid):
        return None
    if not _semantic_check(inp, output_grid):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
