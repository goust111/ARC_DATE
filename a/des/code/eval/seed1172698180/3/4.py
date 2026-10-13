from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8
MAROON = 9

OUTPUT_COLOR_BY_MARKERS = {
    1: RED,
    2: GREEN,
    3: YELLOW,
    4: BLUE,
    5: MAGENTA,
    6: AZURE,
}

PIECE_COLORS = [BLUE, RED, GREEN, YELLOW, GRAY, MAGENTA, AZURE, MAROON]
SIDES = ("left", "right", "top", "bottom")
OPPOSITE = {"left": "right", "right": "left", "top": "bottom", "bottom": "top"}
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _mode(values: list[int]) -> int:
    return Counter(values).most_common(1)[0][0]


def _component_coords(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in DIR4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _piece_cells(base_h: int, base_w: int, side: str, pattern: tuple[int, ...]) -> set[tuple[int, int]]:
    cells = {(r, c) for r in range(base_h) for c in range(base_w)}
    if side == "right":
        for r, delta in enumerate(pattern):
            if delta == -1:
                cells.remove((r, base_w - 1))
            elif delta == 1:
                cells.add((r, base_w))
    elif side == "left":
        for r, delta in enumerate(pattern):
            if delta == -1:
                cells.remove((r, 0))
            elif delta == 1:
                cells.add((r, -1))
    elif side == "bottom":
        for c, delta in enumerate(pattern):
            if delta == -1:
                cells.remove((base_h - 1, c))
            elif delta == 1:
                cells.add((base_h, c))
    else:
        assert side == "top"
        for c, delta in enumerate(pattern):
            if delta == -1:
                cells.remove((0, c))
            elif delta == 1:
                cells.add((-1, c))
    return cells


def _mask_from_piece(base_h: int, base_w: int, side: str, pattern: tuple[int, ...], core_top: int, core_left: int, out_h: int, out_w: int) -> np.ndarray:
    mask = np.zeros((out_h, out_w), dtype=bool)
    for r, c in _piece_cells(base_h, base_w, side, pattern):
        rr = core_top + r
        cc = core_left + c
        if not (0 <= rr < out_h and 0 <= cc < out_w):
            raise ValueError("piece reconstruction out of bounds")
        mask[rr, cc] = True
    return mask


def _fit_piece(mask: np.ndarray, side: str) -> dict | None:
    h, w = mask.shape
    if h < 3 or w < 3:
        return None

    if side in ("left", "right"):
        starts: list[int] = []
        ends: list[int] = []
        lengths: list[int] = []
        for r in range(h):
            cols = np.flatnonzero(mask[r])
            if cols.size == 0 or int(cols[-1]) - int(cols[0]) + 1 != int(cols.size):
                return None
            starts.append(int(cols[0]))
            ends.append(int(cols[-1]))
            lengths.append(int(cols.size))
        base_w = _mode(lengths)
        if base_w < 3:
            return None
        if side == "right":
            if len(set(starts)) != 1 or starts[0] != 0:
                return None
            pattern = tuple(length - base_w for length in lengths)
            if any(delta not in (-1, 0, 1) for delta in pattern) or all(delta == 0 for delta in pattern):
                return None
            core_left = 0
        else:
            if len(set(ends)) != 1 or ends[0] != w - 1:
                return None
            core_left = _mode(starts)
            pattern = tuple(core_left - start for start in starts)
            if any(delta not in (-1, 0, 1) for delta in pattern) or all(delta == 0 for delta in pattern):
                return None
        try:
            rebuilt = _mask_from_piece(h, base_w, side, pattern, 0, core_left, h, w)
        except ValueError:
            return None
        if not np.array_equal(rebuilt, mask):
            return None
        return {
            "side": side,
            "base_h": h,
            "base_w": base_w,
            "pattern": pattern,
            "core_top": 0,
            "core_left": core_left,
        }

    tops: list[int] = []
    bottoms: list[int] = []
    heights: list[int] = []
    for c in range(w):
        rows = np.flatnonzero(mask[:, c])
        if rows.size == 0 or int(rows[-1]) - int(rows[0]) + 1 != int(rows.size):
            return None
        tops.append(int(rows[0]))
        bottoms.append(int(rows[-1]))
        heights.append(int(rows.size))
    base_h = _mode(heights)
    if base_h < 3:
        return None
    if side == "bottom":
        if len(set(tops)) != 1 or tops[0] != 0:
            return None
        pattern = tuple(height - base_h for height in heights)
        if any(delta not in (-1, 0, 1) for delta in pattern) or all(delta == 0 for delta in pattern):
            return None
        core_top = 0
    else:
        assert side == "top"
        if len(set(bottoms)) != 1 or bottoms[0] != h - 1:
            return None
        core_top = _mode(tops)
        pattern = tuple(core_top - top for top in tops)
        if any(delta not in (-1, 0, 1) for delta in pattern) or all(delta == 0 for delta in pattern):
            return None
    try:
        rebuilt = _mask_from_piece(base_h, w, side, pattern, core_top, 0, h, w)
    except ValueError:
        return None
    if not np.array_equal(rebuilt, mask):
        return None
    return {
        "side": side,
        "base_h": base_h,
        "base_w": w,
        "pattern": pattern,
        "core_top": core_top,
        "core_left": 0,
    }


def _analyze_piece(grid: np.ndarray, coords: list[tuple[int, int]]) -> dict:
    r0, c0, r1, c1 = _bbox(coords)
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    mask = sub != BLACK
    fits = []
    for side in SIDES:
        info = _fit_piece(mask, side)
        if info is not None:
            fits.append(info)
    assert len(fits) == 1, "piece shape must encode exactly one matching side"
    info = fits[0]
    rel_cells = sorted(_piece_cells(info["base_h"], info["base_w"], info["side"], info["pattern"]))
    info.update(
        {
            "bbox": (r0, c0, r1, c1),
            "area": len(coords),
            "marker_count": int(np.count_nonzero(sub == ORANGE)),
            "core_abs": (r0 + info["core_top"], c0 + info["core_left"]),
            "cells": rel_cells,
        }
    )
    return info


def _negate(pattern: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(-x for x in pattern)


def _pair_axis(side: str) -> str:
    return "horizontal" if side in ("left", "right") else "vertical"


def _piece_bbox_left(piece: dict) -> int:
    return piece["bbox"][1]


def _piece_bbox_top(piece: dict) -> int:
    return piece["bbox"][0]


def _choose_anchor(a: dict, b: dict) -> tuple[dict, dict]:
    if a["area"] > b["area"]:
        return a, b
    if b["area"] > a["area"]:
        return b, a
    axis = _pair_axis(a["side"])
    if axis == "horizontal":
        key_a = (_piece_bbox_left(a), _piece_bbox_top(a))
        key_b = (_piece_bbox_left(b), _piece_bbox_top(b))
    else:
        key_a = (_piece_bbox_top(a), _piece_bbox_left(a))
        key_b = (_piece_bbox_top(b), _piece_bbox_left(b))
    return (a, b) if key_a <= key_b else (b, a)


def _placed_cells(piece: dict, core_pos: tuple[int, int]) -> list[tuple[int, int]]:
    cr, cc = core_pos
    return [(cr + r, cc + c) for r, c in piece["cells"]]


def _mover_core(anchor: dict, mover: dict, anchor_core: tuple[int, int]) -> tuple[int, int]:
    ar, ac = anchor_core
    if anchor["side"] == "right":
        return ar, ac + anchor["base_w"]
    if anchor["side"] == "left":
        return ar, ac - mover["base_w"]
    if anchor["side"] == "bottom":
        return ar + anchor["base_h"], ac
    assert anchor["side"] == "top"
    return ar - mover["base_h"], ac


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair complementary jigsaw fragments, assemble them along the matching side,
    and recolor each merged pair from the total number of internal orange markers.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    components = _component_coords(g != BLACK)
    pieces = [_analyze_piece(g, coords) for coords in components]
    assert pieces and len(pieces) % 2 == 0

    by_signature: dict[tuple[str, tuple[int, ...]], list[int]] = {}
    for i, piece in enumerate(pieces):
        by_signature.setdefault((piece["side"], piece["pattern"]), []).append(i)

    used: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for i, piece in enumerate(pieces):
        if i in used:
            continue
        wanted = (OPPOSITE[piece["side"]], _negate(piece["pattern"]))
        candidates = [j for j in by_signature.get(wanted, []) if j != i and j not in used]
        assert len(candidates) == 1, "each piece must have exactly one complementary partner"
        j = candidates[0]
        used.add(i)
        used.add(j)
        pairs.append((i, j))
    assert len(used) == len(pieces)

    out = np.full((h, w), BLACK, dtype=int)
    for ia, ib in pairs:
        a = pieces[ia]
        b = pieces[ib]
        anchor, mover = _choose_anchor(a, b)
        anchor_core = anchor["core_abs"]
        mover_core = _mover_core(anchor, mover, anchor_core)
        color = OUTPUT_COLOR_BY_MARKERS[anchor["marker_count"] + mover["marker_count"]]

        for piece, core in ((anchor, anchor_core), (mover, mover_core)):
            for r, c in _placed_cells(piece, core):
                assert 0 <= r < h and 0 <= c < w, "assembled piece left the canvas"
                assert out[r, c] == BLACK, "assembled pairs must not overlap"
                out[r, c] = color
    return out


GRID_H_RANGE = (20, 28)
GRID_W_RANGE = (20, 28)
PAIR_COUNT_CHOICES = [1, 2, 3]
SHARED_RANGE = (5, 8)
OTHER_RANGE = (5, 9)
GEN_RETRIES = 400


def _canonical_pattern(axis: str, pattern: tuple[int, ...]) -> tuple[str, tuple[int, ...]]:
    neg = _negate(pattern)
    return axis, pattern if pattern <= neg else neg


def _random_pattern(length: int, balanced: bool, used_keys: set[tuple[str, tuple[int, ...]]], axis: str, rng: random.Random) -> tuple[int, ...]:
    interior = list(range(1, length - 1))
    for _ in range(200):
        pattern = [0] * length
        if balanced:
            k_choices = [k for k in range(1, min(2, len(interior) // 2) + 1) if length - 2 * k > k]
            if not k_choices:
                continue
            k = rng.choice(k_choices)
            pos = rng.sample(interior, 2 * k)
            rng.shuffle(pos)
            for i in pos[:k]:
                pattern[i] = 1
            for i in pos[k:]:
                pattern[i] = -1
        else:
            choices = []
            for plus_n in range(1, min(2, len(interior)) + 1):
                for minus_n in range(1, min(2, len(interior) - plus_n) + 1):
                    if plus_n == minus_n:
                        continue
                    zero_n = length - plus_n - minus_n
                    if zero_n > max(plus_n, minus_n):
                        choices.append((plus_n, minus_n))
            if not choices:
                continue
            plus_n, minus_n = rng.choice(choices)
            pos = rng.sample(interior, plus_n + minus_n)
            for i in pos[:plus_n]:
                pattern[i] = 1
            for i in pos[plus_n:]:
                pattern[i] = -1
        pat = tuple(pattern)
        if all(v == 0 for v in pat):
            continue
        if _canonical_pattern(axis, pat) in used_keys:
            continue
        used_keys.add(_canonical_pattern(axis, pat))
        return pat
    raise RuntimeError("failed to create a unique connector pattern")


def _interior_cells(cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    return sorted((r, c) for r, c in cells if all((r + dr, c + dc) in cells for dr, dc in DIR4))


def _piece_spec(side: str, pattern: tuple[int, ...], base_h: int, base_w: int, marker_count: int, color: int, rng: random.Random) -> dict | None:
    cells = _piece_cells(base_h, base_w, side, pattern)
    interior = _interior_cells(cells)
    if len(interior) < marker_count:
        return None
    markers = tuple(rng.sample(interior, marker_count))
    rel_rs = [r for r, _ in cells]
    rel_cs = [c for _, c in cells]
    return {
        "side": side,
        "pattern": pattern,
        "base_h": base_h,
        "base_w": base_w,
        "cells": sorted(cells),
        "area": len(cells),
        "marker_count": marker_count,
        "marker_cells": markers,
        "color": color,
        "bbox_rel": (min(rel_rs), min(rel_cs), max(rel_rs), max(rel_cs)),
    }


def _split_marker_total(total: int, rng: random.Random) -> tuple[int, int]:
    splits = [(a, total - a) for a in range(4) if 0 <= total - a <= 3]
    return rng.choice(splits)


def _pair_spec(used_patterns: set[tuple[str, tuple[int, ...]]], colors: list[int], rng: random.Random) -> dict | None:
    horizontal = rng.random() < 0.5
    axis = "horizontal" if horizontal else "vertical"
    balanced = rng.random() < 0.5
    shared = rng.randint(*SHARED_RANGE)
    other_a = rng.randint(*OTHER_RANGE)
    other_b = other_a if balanced else rng.randint(*OTHER_RANGE)
    if not balanced and other_b == other_a:
        other_b = OTHER_RANGE[0] if other_a != OTHER_RANGE[0] else OTHER_RANGE[1]

    try:
        pattern = _random_pattern(shared, balanced=balanced, used_keys=used_patterns, axis=axis, rng=rng)
    except RuntimeError:
        return None

    if horizontal:
        base_h = shared
        base_w_a = other_a
        base_w_b = other_b
        side_a = rng.choice(["left", "right"])
        side_b = OPPOSITE[side_a]
        dims_a = (base_h, base_w_a)
        dims_b = (base_h, base_w_b)
    else:
        base_w = shared
        base_h_a = other_a
        base_h_b = other_b
        side_a = rng.choice(["top", "bottom"])
        side_b = OPPOSITE[side_a]
        dims_a = (base_h_a, base_w)
        dims_b = (base_h_b, base_w)

    total_markers = rng.randint(1, 6)
    markers_a, markers_b = _split_marker_total(total_markers, rng)

    piece_a = _piece_spec(side_a, pattern, dims_a[0], dims_a[1], markers_a, colors[0], rng)
    piece_b = _piece_spec(side_b, _negate(pattern), dims_b[0], dims_b[1], markers_b, colors[1], rng)
    if piece_a is None or piece_b is None:
        return None

    if balanced:
        if piece_a["area"] != piece_b["area"]:
            return None
        anchor_index = rng.randint(0, 1)
    else:
        if piece_a["area"] == piece_b["area"]:
            return None
        anchor_index = 0 if piece_a["area"] > piece_b["area"] else 1

    pieces = [piece_a, piece_b]
    anchor = pieces[anchor_index]
    mover = pieces[1 - anchor_index]
    offsets = [(0, 0), (0, 0)]
    offsets[anchor_index] = (0, 0)
    offsets[1 - anchor_index] = _mover_core(anchor, mover, (0, 0))

    union_rel: set[tuple[int, int]] = set()
    for idx, piece in enumerate(pieces):
        dr, dc = offsets[idx]
        for r, c in piece["cells"]:
            union_rel.add((r + dr, c + dc))

    return {
        "axis": axis,
        "balanced": balanced,
        "pieces": pieces,
        "anchor_index": anchor_index,
        "final_offsets": offsets,
        "union_rel": union_rel,
        "total_markers": total_markers,
    }


def _shift_cells(cells: list[tuple[int, int]] | set[tuple[int, int]], dr: int, dc: int) -> set[tuple[int, int]]:
    return {(r + dr, c + dc) for r, c in cells}


def _halo(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    result: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < h and 0 <= cc < w:
                    result.add((rr, cc))
    return result


def _place_final_pairs(pairs: list[dict], h: int, w: int, rng: random.Random) -> bool:
    occupied: set[tuple[int, int]] = set()
    blocked: set[tuple[int, int]] = set()

    for pair in sorted(pairs, key=lambda p: -len(p["union_rel"])):
        rel_rs = [r for r, _ in pair["union_rel"]]
        rel_cs = [c for _, c in pair["union_rel"]]
        min_r, max_r = min(rel_rs), max(rel_rs)
        min_c, max_c = min(rel_cs), max(rel_cs)
        row_lo = -min_r
        row_hi = h - 1 - max_r
        col_lo = -min_c
        col_hi = w - 1 - max_c
        if row_lo > row_hi or col_lo > col_hi:
            return False

        for _ in range(200):
            anchor_core = (rng.randint(row_lo, row_hi), rng.randint(col_lo, col_hi))
            union_abs = _shift_cells(pair["union_rel"], anchor_core[0], anchor_core[1])
            if union_abs & blocked:
                continue
            pair["final_cores"] = [
                (anchor_core[0] + dr, anchor_core[1] + dc)
                for dr, dc in pair["final_offsets"]
            ]
            occupied |= union_abs
            blocked |= _halo(union_abs, h, w)
            break
        else:
            return False
    return True


def _piece_abs_bbox(piece: dict, core: tuple[int, int]) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = piece["bbox_rel"]
    return core[0] + r0, core[1] + c0, core[0] + r1, core[1] + c1


def _place_scattered_movers(pairs: list[dict], h: int, w: int, rng: random.Random) -> bool:
    occupied: set[tuple[int, int]] = set()
    blocked: set[tuple[int, int]] = set()

    for pair in pairs:
        pair["input_cores"] = list(pair["final_cores"])
        anchor_piece = pair["pieces"][pair["anchor_index"]]
        anchor_core = pair["input_cores"][pair["anchor_index"]]
        anchor_cells = _shift_cells(anchor_piece["cells"], anchor_core[0], anchor_core[1])
        occupied |= anchor_cells
        blocked |= _halo(anchor_cells, h, w)

    for pair in pairs:
        mover_index = 1 - pair["anchor_index"]
        mover = pair["pieces"][mover_index]
        anchor = pair["pieces"][pair["anchor_index"]]
        anchor_core = pair["input_cores"][pair["anchor_index"]]
        anchor_bbox = _piece_abs_bbox(anchor, anchor_core)
        rel_r0, rel_c0, rel_r1, rel_c1 = mover["bbox_rel"]
        row_lo = -rel_r0
        row_hi = h - 1 - rel_r1
        col_lo = -rel_c0
        col_hi = w - 1 - rel_c1
        if row_lo > row_hi or col_lo > col_hi:
            return False

        for _ in range(400):
            core = (rng.randint(row_lo, row_hi), rng.randint(col_lo, col_hi))
            cells = _shift_cells(mover["cells"], core[0], core[1])
            if cells & blocked:
                continue
            if core == pair["final_cores"][mover_index]:
                continue
            mover_bbox = _piece_abs_bbox(mover, core)
            if pair["balanced"]:
                if pair["axis"] == "horizontal":
                    if not (anchor_bbox[1] < mover_bbox[1]):
                        continue
                else:
                    if not (anchor_bbox[0] < mover_bbox[0]):
                        continue
            pair["input_cores"][mover_index] = core
            occupied |= cells
            blocked |= _halo(cells, h, w)
            break
        else:
            return False
    return True


def _render_input(pairs: list[dict], h: int, w: int) -> np.ndarray:
    grid = np.full((h, w), BLACK, dtype=int)
    for pair in pairs:
        for idx, piece in enumerate(pair["pieces"]):
            core = pair["input_cores"][idx]
            marker_abs = {(core[0] + r, core[1] + c) for r, c in piece["marker_cells"]}
            for r, c in _placed_cells(piece, core):
                grid[r, c] = ORANGE if (r, c) in marker_abs else piece["color"]
    return grid


def _render_output(pairs: list[dict], h: int, w: int) -> np.ndarray:
    grid = np.full((h, w), BLACK, dtype=int)
    for pair in pairs:
        color = OUTPUT_COLOR_BY_MARKERS[pair["total_markers"]]
        for idx, piece in enumerate(pair["pieces"]):
            core = pair["final_cores"][idx]
            for r, c in _placed_cells(piece, core):
                grid[r, c] = color
    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    pair_count = rng.choice(PAIR_COUNT_CHOICES)
    color_pool = rng.sample(PIECE_COLORS, 2 * pair_count)
    used_patterns: set[tuple[str, tuple[int, ...]]] = set()

    pairs: list[dict] = []
    for i in range(pair_count):
        pair = _pair_spec(used_patterns, color_pool[2 * i : 2 * i + 2], rng)
        if pair is None:
            return None
        pairs.append(pair)

    if not _place_final_pairs(pairs, h, w, rng):
        return None
    if not _place_scattered_movers(pairs, h, w, rng):
        return None

    inp = _render_input(pairs, h, w)
    expected = _render_output(pairs, h, w)
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
