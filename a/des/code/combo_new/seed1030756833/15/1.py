from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not bool(mask[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and bool(mask[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return np.zeros((0, 0), dtype=bool)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return np.array(mask[int(r0): int(r1) + 1, int(c0): int(c1) + 1], dtype=bool)


def _mask_from_cells(cells: list[tuple[int, int]]) -> np.ndarray:
    rs = [r for r, _c in cells]
    cs = [c for _r, c in cells]
    r0, c0 = min(rs), min(cs)
    out = np.zeros((max(rs) - r0 + 1, max(cs) - c0 + 1), dtype=bool)
    for r, c in cells:
        out[r - r0, c - c0] = True
    return out


def _same_shape(a: np.ndarray, b: np.ndarray) -> bool:
    aa = _trim_mask(a)
    bb = _trim_mask(b)
    return aa.shape == bb.shape and bool(np.array_equal(aa, bb))


def _frame_color(g: np.ndarray) -> int:
    vals = [int(v) for v in g.ravel() if int(v) != BG]
    assert vals, "grid must contain a frame"
    return int(Counter(vals).most_common(1)[0][0])


def _ring_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int], thick: int) -> np.ndarray:
    h, w = shape
    r0, c0, r1, c1 = bbox
    mask = np.zeros((h, w), dtype=bool)
    mask[r0: r1 + 1, c0: c1 + 1] = True
    mask[r0 + thick: r1 - thick + 1, c0 + thick: c1 - thick + 1] = False
    return mask


def _infer_thickness(g: np.ndarray, bbox: tuple[int, int, int, int], frame_color: int) -> tuple[int, int] | None:
    r0, c0, r1, c1 = bbox
    bh, bw = r1 - r0 + 1, c1 - c0 + 1
    max_t = (min(bh, bw) - 2) // 2
    for thick in range(1, max_t + 1):
        inner = g[r0 + thick: r1 - thick + 1, c0 + thick: c1 - thick + 1]
        if inner.size == 0 or not np.all(inner == BG):
            continue
        ring = _ring_mask(g.shape, bbox, thick)
        ring_vals = g[ring]
        if np.any((ring_vals != BG) & (ring_vals != frame_color)):
            continue
        frame_count = int(np.sum(ring_vals == frame_color))
        if frame_count >= max(12, int(0.45 * int(ring.sum()))):
            return thick, frame_count
    return None


def _find_frame(g: np.ndarray, frame_color: int) -> tuple[tuple[int, int, int, int], int]:
    h, w = g.shape
    color_mask = g == frame_color
    row_candidates = [int(i) for i in np.where(color_mask.sum(axis=1) >= 3)[0]]
    col_candidates = [int(i) for i in np.where(color_mask.sum(axis=0) >= 3)[0]]
    best: tuple[int, int, int, tuple[int, int, int, int], int] | None = None

    for ai, r0 in enumerate(row_candidates):
        for r1 in row_candidates[ai + 1:]:
            if r1 - r0 + 1 < 7:
                continue
            for bi, c0 in enumerate(col_candidates):
                for c1 in col_candidates[bi + 1:]:
                    if c1 - c0 + 1 < 7:
                        continue
                    if not (
                        g[r0, c0] == frame_color
                        and g[r0, c1] == frame_color
                        and g[r1, c0] == frame_color
                        and g[r1, c1] == frame_color
                    ):
                        continue
                    top = int(np.sum(g[r0, c0: c1 + 1] == frame_color))
                    bottom = int(np.sum(g[r1, c0: c1 + 1] == frame_color))
                    left = int(np.sum(g[r0: r1 + 1, c0] == frame_color))
                    right = int(np.sum(g[r0: r1 + 1, c1] == frame_color))
                    if min(top, bottom) < max(4, (c1 - c0 + 1) // 3):
                        continue
                    if min(left, right) < max(4, (r1 - r0 + 1) // 3):
                        continue
                    inferred = _infer_thickness(g, (r0, c0, r1, c1), frame_color)
                    if inferred is None:
                        continue
                    thick, frame_count = inferred
                    area = (r1 - r0 + 1) * (c1 - c0 + 1)
                    score = (frame_count, area, -thick, (r0, c0, r1, c1), thick)
                    if best is None or score[:3] > best[:3]:
                        best = score

    assert best is not None, "rectangular frame was not found"
    return best[3], int(best[4])


def _find_gaps(g: np.ndarray, bbox: tuple[int, int, int, int], thick: int) -> list[list[tuple[int, int]]]:
    r0, c0, r1, c1 = bbox
    ring = _ring_mask(g.shape, bbox, thick)
    gap_mask = ring & (g == BG)
    gaps = []
    for comp in _components(gap_mask):
        touches_outer = any(r in (r0, r1) or c in (c0, c1) for r, c in comp)
        if touches_outer:
            gaps.append(comp)
    gaps.sort(key=lambda pts: (min(r for r, _c in pts), min(c for _r, c in pts)))
    return gaps


def _outside_bbox_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    h, w = shape
    r0, c0, r1, c1 = bbox
    out = np.ones((h, w), dtype=bool)
    out[r0: r1 + 1, c0: c1 + 1] = False
    return out


def _find_fragments(
    g: np.ndarray, frame_color: int, bbox: tuple[int, int, int, int]
) -> list[dict]:
    mask = (g != BG) & _outside_bbox_mask(g.shape, bbox)
    fragments = []
    for comp in _components(mask):
        vals = [int(g[r, c]) for r, c in comp]
        if frame_color not in vals:
            continue
        marker_cells = [(r, c) for r, c in comp if int(g[r, c]) not in (BG, frame_color)]
        if len(marker_cells) != 1:
            continue
        rs = [r for r, _c in comp]
        cs = [c for _r, c in comp]
        r0, c0, r1, c1 = min(rs), min(cs), max(rs), max(cs)
        patch = g[r0: r1 + 1, c0: c1 + 1]
        outline = patch == frame_color
        fragments.append(
            {
                "cells": comp,
                "outline": _trim_mask(outline),
                "marker": int(g[marker_cells[0]]),
                "bbox": (r0, c0, r1, c1),
            }
        )
    fragments.sort(key=lambda d: (d["bbox"][0], d["bbox"][1]))
    return fragments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock matching colored fragments into a notched frame and use their marker colors to fill the frame interior.

    Input structure:
    - The background is black.
    - A thick monochrome rectangular frame has connected background-colored gaps cut into its outer edge.
    - Outside the frame are small two-color fragments: their outline color matches the frame, and each has exactly one different marker pixel inside the fragment's bounding box.

    Rule:
    - Detect every frame-edge gap and every outside fragment outline.
    - Match each gap to the fragment whose outline has the same binary shape under a 0/90/180/270-degree rotation.
    - Restore the frame by filling the gap cells with the frame color, erase the matched fragments from their original locations, and read the matched marker colors.
    - Fill the frame interior with horizontal bands ordered by the gaps from top to bottom and then left to right; one band is used per gap.

    Key insight:
    - The fragment outline is the geometric key for docking, while the single hidden marker pixel is the color key for filling the frame interior.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    frame_color = _frame_color(g)
    bbox, thick = _find_frame(g, frame_color)
    gaps = _find_gaps(g, bbox, thick)
    fragments = _find_fragments(g, frame_color, bbox)
    assert gaps, "frame must contain at least one gap"

    used: set[int] = set()
    matched_markers: list[int] = []
    matched_fragments: list[int] = []

    for gap in gaps:
        gap_shape = _mask_from_cells(gap)
        match_idx = None
        for i, frag in enumerate(fragments):
            if i in used:
                continue
            outline = frag["outline"]
            if any(_same_shape(np.rot90(outline, k), gap_shape) for k in range(4)):
                match_idx = i
                break
        assert match_idx is not None, "every gap must have a matching fragment"
        used.add(match_idx)
        matched_fragments.append(match_idx)
        matched_markers.append(int(fragments[match_idx]["marker"]))
        for r, c in gap:
            out[r, c] = frame_color

    for i in matched_fragments:
        for r, c in fragments[i]["cells"]:
            out[r, c] = BG

    r0, c0, r1, c1 = bbox
    inner_rows = np.arange(r0 + thick, r1 - thick + 1)
    inner_cols = np.arange(c0 + thick, c1 - thick + 1)
    for rows, color in zip(np.array_split(inner_rows, len(matched_markers)), matched_markers):
        if len(rows) == 0:
            continue
        region = out[np.ix_(rows, inner_cols)]
        region[region == BG] = int(color)
        out[np.ix_(rows, inner_cols)] = region
    return out


_SHAPES = [
    ("###", "#M#", "###"),
    ("###", "#M.", "#.."),
    ("####", "#M..", "###."),
    ("###.", "#M##", "..#."),
    ("##.", "#M#", "###"),
]


def _shape_array(lines: tuple[str, ...], frame_color: int, marker_color: int) -> np.ndarray:
    h, w = len(lines), len(lines[0])
    arr = np.full((h, w), BG, dtype=int)
    for r, row in enumerate(lines):
        for c, ch in enumerate(row):
            if ch == "#":
                arr[r, c] = frame_color
            elif ch == "M":
                arr[r, c] = marker_color
    return arr


def _side_rotation(side: str) -> int:
    return {"top": 0, "left": 1, "bottom": 2, "right": 3}[side]


def _non_bg_cells(patch: np.ndarray) -> list[tuple[int, int]]:
    return [(int(r), int(c)) for r, c in np.argwhere(patch != BG)]


def _dilated_hits(occupied: np.ndarray, cells: list[tuple[int, int]], top: int, left: int) -> bool:
    h, w = occupied.shape
    for r, c in cells:
        gr, gc = top + r, left + c
        for nr in range(max(0, gr - 1), min(h, gr + 2)):
            for nc in range(max(0, gc - 1), min(w, gc + 2)):
                if occupied[nr, nc]:
                    return True
    return False


def _place_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    for r, c in _non_bg_cells(patch):
        grid[top + r, left + c] = int(patch[r, c])


def _fragment_positions(
    grid: np.ndarray,
    occupied: np.ndarray,
    patch: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
) -> list[tuple[int, int]]:
    h, w = grid.shape
    ph, pw = patch.shape
    r0, c0, r1, c1 = frame_bbox
    cells = _non_bg_cells(patch)
    positions = []
    for top in range(h - ph + 1):
        for left in range(w - pw + 1):
            ok = True
            for rr, cc in cells:
                gr, gc = top + rr, left + cc
                if r0 - 1 <= gr <= r1 + 1 and c0 - 1 <= gc <= c1 + 1:
                    ok = False
                    break
            if not ok or _dilated_hits(occupied, cells, top, left):
                continue
            positions.append((top, left))
    return positions


def _gap_cells(
    mask: np.ndarray, side: str, bbox: tuple[int, int, int, int], thick: int
) -> list[tuple[int, int, int, int]]:
    r0, c0, r1, c1 = bbox
    mh, mw = mask.shape
    if side == "top":
        rows = range(r0, r0 + 1)
        col_min, col_max = c0 + 1, c1 - mw
        fixed = [("row", r0)]
    elif side == "bottom":
        rows = range(r1 - mh + 1, r1 - mh + 2)
        col_min, col_max = c0 + 1, c1 - mw
        fixed = [("row", r1 - mh + 1)]
    elif side == "left":
        row_min, row_max = r0 + 1, r1 - mh
        cols = range(c0, c0 + 1)
        fixed = [("col", c0)]
    else:
        row_min, row_max = r0 + 1, r1 - mh
        cols = range(c1 - mw + 1, c1 - mw + 2)
        fixed = [("col", c1 - mw + 1)]

    del thick
    candidates: list[tuple[int, int, int, int]] = []
    if side in ("top", "bottom"):
        if col_min > col_max:
            return []
        top = fixed[0][1]
        for left in range(col_min, col_max + 1):
            candidates.append((top, left, mh, mw))
    else:
        if row_min > row_max:
            return []
        left = fixed[0][1]
        for top in range(row_min, row_max + 1):
            candidates.append((top, left, mh, mw))
    return candidates


def _try_place_gaps(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    thick: int,
    frame_color: int,
    marker_colors: list[int],
    shape_ids: list[int],
) -> list[dict] | None:
    gap_pad = np.zeros_like(grid, dtype=bool)
    infos: list[dict] = []
    sides = ["top", "bottom", "left", "right"]
    random.shuffle(sides)

    for i, shape_id in enumerate(shape_ids):
        side_order = sides[:]
        random.shuffle(side_order)
        placed = False
        for side in side_order:
            solved_patch = np.rot90(_shape_array(_SHAPES[shape_id], frame_color, marker_colors[i]), _side_rotation(side))
            outline = solved_patch == frame_color
            if side in ("top", "bottom") and outline.shape[0] > thick:
                continue
            if side in ("left", "right") and outline.shape[1] > thick:
                continue
            candidates = _gap_cells(outline, side, bbox, thick)
            random.shuffle(candidates)
            true_cells = _non_bg_cells(np.where(outline, frame_color, BG))
            for top, left, _mh, _mw in candidates:
                global_cells = [(top + r, left + c) for r, c in true_cells]
                blocked = False
                for gr, gc in global_cells:
                    if gap_pad[gr, gc]:
                        blocked = True
                        break
                    for nr in range(max(0, gr - 1), min(grid.shape[0], gr + 2)):
                        for nc in range(max(0, gc - 1), min(grid.shape[1], gc + 2)):
                            if gap_pad[nr, nc]:
                                blocked = True
                                break
                        if blocked:
                            break
                    if blocked:
                        break
                if blocked:
                    continue
                for gr, gc in global_cells:
                    grid[gr, gc] = BG
                    gap_pad[gr, gc] = True
                infos.append(
                    {
                        "side": side,
                        "top": top,
                        "left": left,
                        "patch": solved_patch,
                        "cells": global_cells,
                        "marker": int(marker_colors[i]),
                    }
                )
                placed = True
                break
            if placed:
                break
        if not placed:
            return None
    return infos


def _expected_output(
    shape: tuple[int, int],
    bbox: tuple[int, int, int, int],
    thick: int,
    frame_color: int,
    gap_infos: list[dict],
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    r0, c0, r1, c1 = bbox
    out[r0: r1 + 1, c0: c1 + 1] = frame_color
    out[r0 + thick: r1 - thick + 1, c0 + thick: c1 - thick + 1] = BG
    ordered = sorted(gap_infos, key=lambda d: (min(r for r, _c in d["cells"]), min(c for _r, c in d["cells"])))
    rows = np.arange(r0 + thick, r1 - thick + 1)
    cols = np.arange(c0 + thick, c1 - thick + 1)
    for band_rows, info in zip(np.array_split(rows, len(ordered)), ordered):
        out[np.ix_(band_rows, cols)] = int(info["marker"])
    return out


def _gen_attempt() -> np.ndarray | None:
    frame_color = random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9])
    gap_count = random.choice([1, 2, 3, 4])
    thick = random.choice([3, 4])
    max_band = max(2, min(5, (22 - 2 * thick) // gap_count))
    if max_band < 2:
        return None
    band_h = random.randint(2, max_band)
    inner_h = gap_count * band_h
    inner_w = random.randint(7, 14)
    outer_h = inner_h + 2 * thick
    outer_w = inner_w + 2 * thick
    if outer_h + 6 > 30 or outer_w + 6 > 30:
        return None

    h = random.randint(outer_h + 6, 30)
    w = random.randint(outer_w + 6, 30)
    r0 = random.randint(2, h - outer_h - 2)
    c0 = random.randint(2, w - outer_w - 2)
    r1, c1 = r0 + outer_h - 1, c0 + outer_w - 1
    bbox = (r0, c0, r1, c1)

    grid = np.full((h, w), BG, dtype=int)
    grid[r0: r1 + 1, c0: c1 + 1] = frame_color
    grid[r0 + thick: r1 - thick + 1, c0 + thick: c1 - thick + 1] = BG

    colors = [c for c in range(1, 10) if c != frame_color]
    random.shuffle(colors)
    marker_colors = colors[:gap_count]
    shape_ids = random.sample(range(len(_SHAPES)), gap_count)
    gap_infos = _try_place_gaps(grid, bbox, thick, frame_color, marker_colors, shape_ids)
    if gap_infos is None:
        return None

    occupied = grid != BG
    random.shuffle(gap_infos)
    for info in gap_infos:
        frag = np.rot90(info["patch"], random.randint(0, 3))
        positions = _fragment_positions(grid, occupied, frag, bbox)
        if not positions:
            return None
        top, left = random.choice(positions)
        _place_patch(grid, frag, top, left)
        for rr, cc in _non_bg_cells(frag):
            occupied[top + rr, left + cc] = True

    expected = _expected_output(grid.shape, bbox, thick, frame_color, gap_infos)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    vals, counts = np.unique(grid, return_counts=True)
    if int(vals[int(np.argmax(counts))]) != BG:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
