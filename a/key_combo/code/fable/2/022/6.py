from __future__ import annotations

import importlib.util
import random
import sys
from collections import Counter, deque
from pathlib import Path

import numpy as np


_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")

try:
    from arc_utils import Color, get_bg, neighbors4
except Exception:
    _old_dont_write = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        _spec = importlib.util.spec_from_file_location("_shared_arc_utils", _ARC_UTILS_PATH)
        if _spec is None or _spec.loader is None:
            raise ImportError(f"cannot load arc_utils from {_ARC_UTILS_PATH}")
        _arc_utils = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_arc_utils)
        Color = _arc_utils.Color
        get_bg = _arc_utils.get_bg
        neighbors4 = _arc_utils.neighbors4
    finally:
        sys.dont_write_bytecode = _old_dont_write


Coord = tuple[int, int]

PALETTE = list(range(1, 10))
GEN_RETRIES = 400


def _label_mask(mask: np.ndarray) -> tuple[np.ndarray, int]:
    labels = np.zeros(mask.shape, dtype=int)
    next_id = 0
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            next_id += 1
            queue = deque([(r, c)])
            labels[r, c] = next_id
            while queue:
                rr, cc = queue.popleft()
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = next_id
                        queue.append((nr, nc))
    return labels, next_id


def _foreground_mode(grid: np.ndarray, bg: int) -> int:
    counts = Counter(int(v) for v in grid.flatten() if int(v) != bg)
    if not counts:
        raise AssertionError("expected foreground pixels")
    return counts.most_common(1)[0][0]


def _detect_gate_pixels(grid: np.ndarray, wall_color: int, bg: int) -> dict[Coord, str]:
    h, w = grid.shape
    gates: dict[Coord, str] = {}
    for r in range(h):
        for c in range(w):
            val = int(grid[r, c])
            if val in (bg, wall_color):
                continue
            horizontal = (
                0 < c < w - 1
                and int(grid[r, c - 1]) == wall_color
                and int(grid[r, c + 1]) == wall_color
            )
            vertical = (
                0 < r < h - 1
                and int(grid[r - 1, c]) == wall_color
                and int(grid[r + 1, c]) == wall_color
            )
            if horizontal and vertical:
                raise AssertionError("gate cannot lie at a crossing")
            if horizontal:
                gates[(r, c)] = "h"
            elif vertical:
                gates[(r, c)] = "v"
    return gates


def _region_adjacency(region_labels: np.ndarray, barrier_labels: np.ndarray) -> tuple[dict[int, set[int]], set[int]]:
    h, w = region_labels.shape
    adjacency: dict[int, set[int]] = {}
    border_regions: set[int] = set()
    for r in range(h):
        for c in range(w):
            rid = int(region_labels[r, c])
            if rid == 0:
                continue
            adjacency.setdefault(rid, set())
            if r in (0, h - 1) or c in (0, w - 1):
                border_regions.add(rid)
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w:
                    bid = int(barrier_labels[nr, nc])
                    if bid != 0:
                        adjacency[rid].add(bid)
    return adjacency, border_regions


def _analyze_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = int(get_bg(g))
    wall_color = int(_foreground_mode(g, bg))
    gates = _detect_gate_pixels(g, wall_color, bg)

    barrier_mask = g == wall_color
    for r, c in gates:
        barrier_mask[r, c] = True

    barrier_labels, barrier_count = _label_mask(barrier_mask)
    assert barrier_count == 2, f"expected two wall components, got {barrier_count}"

    region_labels, region_count = _label_mask(~barrier_mask)
    assert region_count == 3, f"expected three regions, got {region_count}"

    adjacency, border_regions = _region_adjacency(region_labels, barrier_labels)
    assert len(border_regions) == 1, f"expected one outer region, got {sorted(border_regions)}"
    outer_region = next(iter(border_regions))
    assert len(adjacency[outer_region]) == 1, "outer region should touch only the outer wall"
    outer_wall = next(iter(adjacency[outer_region]))

    ring_candidates = [rid for rid, walls in adjacency.items() if rid != outer_region and len(walls) == 2]
    assert len(ring_candidates) == 1, "expected one ring region"
    ring_region = ring_candidates[0]

    inner_candidates = [rid for rid in adjacency if rid not in (outer_region, ring_region)]
    assert len(inner_candidates) == 1, "expected one inner region"
    inner_region = inner_candidates[0]

    inner_wall_candidates = adjacency[ring_region] - {outer_wall}
    assert len(inner_wall_candidates) == 1, "expected one inner wall"
    inner_wall = next(iter(inner_wall_candidates))
    assert adjacency[inner_region] == {inner_wall}, "inner region must touch only the inner wall"

    region_level = {
        inner_region: 0,
        ring_region: 1,
        outer_region: 2,
    }

    transitions: dict[tuple[int, int], dict] = {}
    gate_records: list[dict] = []
    for (r, c), axis in gates.items():
        passable_neighbors = []
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1] and not barrier_mask[nr, nc]:
                passable_neighbors.append((nr, nc))
        assert len(passable_neighbors) == 2, f"gate at {(r, c)} should have two passable sides"

        side_regions = {int(region_labels[nr, nc]) for nr, nc in passable_neighbors}
        assert len(side_regions) == 2, f"gate at {(r, c)} should separate two regions"
        reg_a, reg_b = sorted(side_regions, key=lambda rid: region_level[rid])
        assert region_level[reg_b] == region_level[reg_a] + 1
        outward_cells = [(nr, nc) for nr, nc in passable_neighbors if int(region_labels[nr, nc]) == reg_b]
        assert len(outward_cells) == 1

        key = (reg_a, int(g[r, c]))
        assert key not in transitions, f"duplicate gate color {key[1]} on one boundary"
        transitions[key] = {
            "next_region": reg_b,
            "dest": outward_cells[0],
            "gate": (r, c),
            "axis": axis,
            "wall": int(barrier_labels[r, c]),
        }
        gate_records.append(
            {
                "pos": (r, c),
                "color": int(g[r, c]),
                "axis": axis,
                "from_region": reg_a,
                "to_region": reg_b,
                "wall": int(barrier_labels[r, c]),
                "dest": outward_cells[0],
            }
        )

    markers = []
    marker_mask = (g != bg) & ~barrier_mask
    for r, c in np.argwhere(marker_mask):
        markers.append(
            {
                "pos": (int(r), int(c)),
                "color": int(g[r, c]),
                "region": int(region_labels[r, c]),
                "level": region_level[int(region_labels[r, c])],
            }
        )

    return {
        "bg": bg,
        "wall_color": wall_color,
        "gates": gate_records,
        "markers": markers,
        "barrier_mask": barrier_mask,
        "barrier_labels": barrier_labels,
        "region_labels": region_labels,
        "inner_region": inner_region,
        "ring_region": ring_region,
        "outer_region": outer_region,
        "inner_wall": inner_wall,
        "outer_wall": outer_wall,
        "region_level": region_level,
        "transitions": transitions,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each marker outward through same-color gates on the current wall until it can no longer escape.

    Input structure:
    - A black background contains two same-color one-pixel-thick closed walls, one nested inside the other.
    - Each wall may contain a few single-pixel gates of other colors embedded on straight wall segments.
    - Colored single-pixel markers lie in the inner region, the ring between the walls, and the outer region.

    Rule:
    - Treat wall pixels and gate pixels as barriers, then flood-fill the remaining cells to identify the inner, ring, and outer regions.
    - For each marker, look only at the outer boundary of its current region; if that boundary has a gate of the same color, erase the marker and redraw it on the cell immediately outside that gate.
    - Repeat from the new region so an inner marker can chain through the inner wall and then the outer wall; stop once no same-color gate exists on the current outer boundary, or once the marker is already in the outer region.
    - Keep both walls and every gate unchanged.

    Key insight:
    - The same gate color can matter on different layers, so the solution must recover the three flood-filled regions and apply color-matched escapes layer by layer.
    """
    info = _analyze_grid(np.asarray(grid, dtype=int))
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    bg = info["bg"]

    for marker in info["markers"]:
        r, c = marker["pos"]
        out[r, c] = bg

    for marker in info["markers"]:
        color = marker["color"]
        region = marker["region"]
        dest = marker["pos"]
        steps = 0
        while (region, color) in info["transitions"]:
            move = info["transitions"][(region, color)]
            dest = move["dest"]
            region = move["next_region"]
            steps += 1
            assert steps <= 2, "two nested walls allow at most two escapes"
        rr, cc = dest
        assert out[rr, cc] in (bg, color), "moved markers must not collide"
        out[rr, cc] = color
    return out


def _boundary_from_fill(fill: np.ndarray) -> np.ndarray:
    h, w = fill.shape
    boundary = np.zeros_like(fill, dtype=bool)
    for r in range(h):
        for c in range(w):
            if not fill[r, c]:
                continue
            for nr, nc in neighbors4(r, c):
                if not (0 <= nr < h and 0 <= nc < w) or not fill[nr, nc]:
                    boundary[r, c] = True
                    break
    return boundary


def _random_walk_pairs(width: int, row_lo: int, row_hi: int, min_span: int, max_span: int) -> tuple[list[int], list[int]] | None:
    max_top = row_hi - min_span + 1
    if row_lo > max_top:
        return None
    tops = [random.randint(row_lo, max_top)]
    span_hi0 = min(max_span, row_hi - tops[0] + 1)
    if span_hi0 < min_span:
        return None
    spans = [random.randint(min_span, span_hi0)]

    for _ in range(1, width):
        candidates = []
        for top in range(max(row_lo, tops[-1] - 1), min(max_top, tops[-1] + 1) + 1):
            span_lo = max(min_span, spans[-1] - 1)
            span_hi = min(max_span, spans[-1] + 1, row_hi - top + 1)
            for span in range(span_lo, span_hi + 1):
                candidates.append((top, span))
        if not candidates:
            return None
        top, span = random.choice(candidates)
        tops.append(top)
        spans.append(span)

    if len(set(tops)) == 1 and len(set(spans)) == 1 and random.random() < 0.85:
        return None
    bottoms = [top + span - 1 for top, span in zip(tops, spans)]
    return tops, bottoms


def _fill_from_profiles(shape: tuple[int, int], c0: int, tops: list[int], bottoms: list[int]) -> np.ndarray:
    h, w = shape
    fill = np.zeros((h, w), dtype=bool)
    for i, (top, bottom) in enumerate(zip(tops, bottoms)):
        fill[top : bottom + 1, c0 + i] = True
    return fill


def _paint_rect(fill: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    fill[r0 : r1 + 1, c0 : c1 + 1] = True


def _build_rectilinear_fill(
    shape: tuple[int, int],
    base_box: tuple[int, int, int, int],
    *,
    outer_limits: tuple[int, int, int, int] | None = None,
    max_bulges: int = 2,
) -> np.ndarray:
    fill = np.zeros(shape, dtype=bool)
    top, left, bottom, right = base_box
    _paint_rect(fill, top, left, bottom, right)
    height = bottom - top + 1
    width = right - left + 1

    candidate_sides = []
    if top >= 3 and width >= 8:
        candidate_sides.append("top")
    if bottom <= shape[0] - 4 and width >= 8:
        candidate_sides.append("bottom")
    if left >= 3 and height >= 8:
        candidate_sides.append("left")
    if right <= shape[1] - 4 and height >= 8:
        candidate_sides.append("right")

    random.shuffle(candidate_sides)
    bulge_count = random.randint(1, min(max_bulges, len(candidate_sides))) if candidate_sides else 0
    chosen_sides = candidate_sides[:bulge_count]

    for side in chosen_sides:
        if side in {"top", "bottom"}:
            span = right - left + 1
            seg_len = random.randint(4, span - 4)
            seg_left = random.randint(left + 2, right - seg_len - 1)
            seg_right = seg_left + seg_len - 1
            if side == "top":
                limit_top = 0 if outer_limits is None else outer_limits[0]
                ext_hi = min(3, top - limit_top)
                if ext_hi < 2:
                    continue
                ext = random.randint(2, ext_hi)
                _paint_rect(fill, top - ext, seg_left, top - 1, seg_right)
            else:
                limit_bottom = shape[0] - 1 if outer_limits is None else outer_limits[2]
                ext_hi = min(3, limit_bottom - bottom)
                if ext_hi < 2:
                    continue
                ext = random.randint(2, ext_hi)
                _paint_rect(fill, bottom + 1, seg_left, bottom + ext, seg_right)
        else:
            span = bottom - top + 1
            seg_len = random.randint(4, span - 4)
            seg_top = random.randint(top + 2, bottom - seg_len - 1)
            seg_bottom = seg_top + seg_len - 1
            if side == "left":
                limit_left = 0 if outer_limits is None else outer_limits[1]
                ext_hi = min(3, left - limit_left)
                if ext_hi < 2:
                    continue
                ext = random.randint(2, ext_hi)
                _paint_rect(fill, seg_top, left - ext, seg_bottom, left - 1)
            else:
                limit_right = shape[1] - 1 if outer_limits is None else outer_limits[3]
                ext_hi = min(3, limit_right - right)
                if ext_hi < 2:
                    continue
                ext = random.randint(2, ext_hi)
                _paint_rect(fill, seg_top, right + 1, seg_bottom, right + ext)
    return fill


def _generate_rectilinear_nested_fills(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    h, w = shape
    outer_margin = 3
    gap = 4
    min_outer_h = 14
    min_outer_w = 14
    max_outer_h = h - 2 * outer_margin
    max_outer_w = w - 2 * outer_margin
    if max_outer_h < min_outer_h or max_outer_w < min_outer_w:
        return None

    outer_h = random.randint(min_outer_h, max_outer_h)
    outer_w = random.randint(min_outer_w, max_outer_w)
    outer_top = random.randint(outer_margin, h - outer_margin - outer_h)
    outer_left = random.randint(outer_margin, w - outer_margin - outer_w)
    outer_bottom = outer_top + outer_h - 1
    outer_right = outer_left + outer_w - 1

    inner_top_lo = outer_top + gap
    inner_left_lo = outer_left + gap
    inner_bottom_hi = outer_bottom - gap
    inner_right_hi = outer_right - gap
    min_inner_h = 7
    min_inner_w = 7
    if inner_bottom_hi - inner_top_lo + 1 < min_inner_h or inner_right_hi - inner_left_lo + 1 < min_inner_w:
        return None

    inner_h = random.randint(min_inner_h, inner_bottom_hi - inner_top_lo + 1)
    inner_w = random.randint(min_inner_w, inner_right_hi - inner_left_lo + 1)
    inner_top = random.randint(inner_top_lo, inner_bottom_hi - inner_h + 1)
    inner_left = random.randint(inner_left_lo, inner_right_hi - inner_w + 1)
    inner_bottom = inner_top + inner_h - 1
    inner_right = inner_left + inner_w - 1

    outer_fill = np.zeros(shape, dtype=bool)
    inner_fill = np.zeros(shape, dtype=bool)
    _paint_rect(outer_fill, outer_top, outer_left, outer_bottom, outer_right)
    _paint_rect(inner_fill, inner_top, inner_left, inner_bottom, inner_right)

    outer_boundary = np.zeros(shape, dtype=bool)
    inner_boundary = np.zeros(shape, dtype=bool)
    outer_boundary[outer_top, outer_left : outer_right + 1] = True
    outer_boundary[outer_bottom, outer_left : outer_right + 1] = True
    outer_boundary[outer_top : outer_bottom + 1, outer_left] = True
    outer_boundary[outer_top : outer_bottom + 1, outer_right] = True
    inner_boundary[inner_top, inner_left : inner_right + 1] = True
    inner_boundary[inner_bottom, inner_left : inner_right + 1] = True
    inner_boundary[inner_top : inner_bottom + 1, inner_left] = True
    inner_boundary[inner_top : inner_bottom + 1, inner_right] = True
    return outer_fill, inner_fill, outer_boundary, inner_boundary


def _contour_from_profiles(shape: tuple[int, int], c0: int, tops: list[int], bottoms: list[int]) -> np.ndarray:
    wall = np.zeros(shape, dtype=bool)
    width = len(tops)
    c1 = c0 + width - 1

    wall[tops[0] : bottoms[0] + 1, c0] = True
    wall[tops[-1] : bottoms[-1] + 1, c1] = True

    wall[tops[0], c0] = True
    for i in range(width - 1):
        c = c0 + i
        t1, t2 = tops[i], tops[i + 1]
        if t2 > t1:
            wall[t1 : t2 + 1, c] = True
            wall[t2, c + 1] = True
        elif t2 < t1:
            wall[t1, c + 1] = True
            wall[t2 : t1 + 1, c + 1] = True
        else:
            wall[t1, c + 1] = True

    wall[bottoms[0], c0] = True
    for i in range(width - 1):
        c = c0 + i
        b1, b2 = bottoms[i], bottoms[i + 1]
        if b2 > b1:
            wall[b1, c + 1] = True
            wall[b1 : b2 + 1, c + 1] = True
        elif b2 < b1:
            wall[b2 : b1 + 1, c] = True
            wall[b2, c + 1] = True
        else:
            wall[b1, c + 1] = True

    return wall


def _generate_outer_fill(shape: tuple[int, int]) -> tuple[np.ndarray, dict] | None:
    h, w = shape
    if h < 12 or w < 12:
        return None

    margin = 2
    min_w = max(8, w // 3)
    max_w = max(min_w, w - 2 * margin - 2)
    width = random.randint(min_w, max_w)
    c0 = random.randint(margin, w - margin - width)
    min_span = max(8, h // 3)
    max_span = min(h - 2 * margin - 1, min_span + 6)
    profiles = _random_walk_pairs(width, margin, h - 1 - margin, min_span, max_span)
    if profiles is None:
        return None
    tops, bottoms = profiles
    fill = _fill_from_profiles(shape, c0, tops, bottoms)
    return fill, {"c0": c0, "tops": tops, "bottoms": bottoms}


def _generate_inner_fill(shape: tuple[int, int], outer_meta: dict) -> tuple[np.ndarray, dict] | None:
    h, w = shape
    c0 = outer_meta["c0"]
    tops_o = outer_meta["tops"]
    bottoms_o = outer_meta["bottoms"]
    outer_width = len(tops_o)

    clear_cols = 3
    clear_rows = 3
    min_w = 6
    min_span = 5

    intervals = []
    for start in range(clear_cols, outer_width - clear_cols - min_w + 1):
        for end in range(start + min_w - 1, outer_width - clear_cols - 1 + 1):
            room_ok = True
            for idx in range(start, end + 1):
                if bottoms_o[idx] - tops_o[idx] + 1 < min_span + 2 * clear_rows:
                    room_ok = False
                    break
            if room_ok:
                intervals.append((start, end))
    if not intervals:
        return None

    start, end = random.choice(intervals)
    width = end - start + 1
    local_tops: list[int] = []
    local_bottoms: list[int] = []

    for local_idx, outer_idx in enumerate(range(start, end + 1)):
        row_lo = tops_o[outer_idx] + clear_rows
        row_hi = bottoms_o[outer_idx] - clear_rows
        if local_idx == 0:
            max_top = row_hi - min_span + 1
            if row_lo > max_top:
                return None
            top = random.randint(row_lo, max_top)
            max_span = min(8, row_hi - top + 1)
            if max_span < min_span:
                return None
            span = random.randint(min_span, max_span)
        else:
            candidates = []
            prev_top = local_tops[-1]
            prev_span = local_bottoms[-1] - local_tops[-1] + 1
            max_top = row_hi - min_span + 1
            for top in range(max(row_lo, prev_top - 1), min(max_top, prev_top + 1) + 1):
                span_lo = max(min_span, prev_span - 1)
                span_hi = min(8, prev_span + 1, row_hi - top + 1)
                for span in range(span_lo, span_hi + 1):
                    candidates.append((top, span))
            if not candidates:
                return None
            top, span = random.choice(candidates)
        local_tops.append(top)
        local_bottoms.append(top + span - 1)

    if len(set(local_tops)) == 1 and len(set(local_bottoms)) == 1 and random.random() < 0.85:
        return None

    inner_c0 = c0 + start
    fill = _fill_from_profiles(shape, inner_c0, local_tops, local_bottoms)
    return fill, {"c0": inner_c0, "tops": local_tops, "bottoms": local_bottoms}


def _generate_nested_fills(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    h, w = shape
    transpose = random.random() < 0.5
    local_shape = (w, h) if transpose else (h, w)
    outer = _generate_outer_fill(local_shape)
    if outer is None:
        return None
    outer_fill, outer_meta = outer
    inner = _generate_inner_fill(local_shape, outer_meta)
    if inner is None:
        return None
    inner_fill, inner_meta = inner

    outer_boundary = _contour_from_profiles(local_shape, outer_meta["c0"], outer_meta["tops"], outer_meta["bottoms"])
    inner_boundary = _contour_from_profiles(local_shape, inner_meta["c0"], inner_meta["tops"], inner_meta["bottoms"])
    if np.any(outer_boundary & inner_boundary):
        return None

    if transpose:
        outer_fill = outer_fill.T
        inner_fill = inner_fill.T
        outer_boundary = outer_boundary.T
        inner_boundary = inner_boundary.T
    return outer_fill, inner_fill, outer_boundary, inner_boundary


def _straight_gate_candidates(boundary: np.ndarray, fill: np.ndarray) -> list[dict]:
    h, w = boundary.shape
    candidates = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if not boundary[r, c]:
                continue
            horiz = boundary[r, c - 1] and boundary[r, c + 1] and not boundary[r - 1, c] and not boundary[r + 1, c]
            vert = boundary[r - 1, c] and boundary[r + 1, c] and not boundary[r, c - 1] and not boundary[r, c + 1]
            if horiz == vert:
                continue

            if horiz:
                side_a = (r - 1, c)
                side_b = (r + 1, c)
            else:
                side_a = (r, c - 1)
                side_b = (r, c + 1)

            in_a = bool(fill[side_a])
            in_b = bool(fill[side_b])
            if in_a == in_b:
                continue
            outward = side_a if not in_a else side_b
            inward = side_b if not in_a else side_a
            candidates.append(
                {
                    "pos": (r, c),
                    "axis": "h" if horiz else "v",
                    "outward": outward,
                    "inward": inward,
                }
            )
    random.shuffle(candidates)
    return candidates


def _chebyshev(a: Coord, b: Coord) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _pick_separated(candidates: list[dict], count: int, blocked: set[Coord], min_sep: int = 3) -> list[dict] | None:
    chosen: list[dict] = []
    for cand in candidates:
        pos = cand["pos"]
        if pos in blocked:
            continue
        too_close = False
        for other in chosen:
            if _chebyshev(pos, other["pos"]) < min_sep:
                too_close = True
                break
        if too_close:
            continue
        chosen.append(cand)
        if len(chosen) == count:
            return chosen
    return None


def _safe_region_cells(region_mask: np.ndarray, barrier_mask: np.ndarray) -> list[Coord]:
    h, w = region_mask.shape
    safe = []
    for r, c in np.argwhere(region_mask):
        rr = int(r)
        cc = int(c)
        if rr in (0, h - 1) or cc in (0, w - 1):
            continue
        near_barrier = False
        for nr, nc in neighbors4(rr, cc):
            if barrier_mask[nr, nc]:
                near_barrier = True
                break
        if not near_barrier:
            safe.append((rr, cc))
    if safe:
        random.shuffle(safe)
        return safe
    fallback = [(int(r), int(c)) for r, c in np.argwhere(region_mask)]
    random.shuffle(fallback)
    return fallback


def _pick_cells(candidates: list[Coord], count: int, forbidden: set[Coord], min_sep: int = 2) -> list[Coord] | None:
    chosen: list[Coord] = []
    for cell in candidates:
        if cell in forbidden:
            continue
        if any(_chebyshev(cell, bad) < min_sep for bad in forbidden):
            continue
        if any(_chebyshev(cell, other) < min_sep for other in chosen):
            continue
        chosen.append(cell)
        if len(chosen) == count:
            return chosen
    return None


def _build_expected(shape: tuple[int, int], wall_color: int, outer_boundary: np.ndarray, inner_boundary: np.ndarray,
                    gate_plan: dict[str, dict], marker_plan: list[dict]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    grid[outer_boundary | inner_boundary] = wall_color
    for gate in gate_plan.values():
        r, c = gate["pos"]
        grid[r, c] = gate["color"]
    for marker in marker_plan:
        r, c = marker["final"]
        grid[r, c] = marker["color"]
    return grid


def _validate_generated(inp: np.ndarray, expected: np.ndarray, meta: dict) -> bool:
    try:
        analyzed = _analyze_grid(inp)
        solved = main(inp)
    except Exception:
        return False

    if not np.array_equal(solved, expected):
        return False

    inner_colors = {color for (region, color), move in analyzed["transitions"].items() if region == analyzed["inner_region"]}
    ring_colors = {color for (region, color), move in analyzed["transitions"].items() if region == analyzed["ring_region"]}
    if inner_colors != meta["expected_inner_gate_colors"]:
        return False
    if ring_colors != meta["expected_outer_gate_colors"]:
        return False

    markers_by_pos = {tuple(m["pos"]): m for m in analyzed["markers"]}
    if len(markers_by_pos) != len(meta["markers"]):
        return False
    for marker in meta["markers"]:
        actual = markers_by_pos.get(marker["start"])
        if actual is None or actual["color"] != marker["color"] or actual["level"] != marker["start_level"]:
            return False

    final_positions = [marker["final"] for marker in meta["markers"]]
    if len(final_positions) != len(set(final_positions)):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    shape = (random.randint(20, 28), random.randint(20, 28))
    fills = _generate_rectilinear_nested_fills(shape)
    if fills is None:
        return None
    outer_fill, inner_fill, outer_boundary, inner_boundary = fills

    wall_color = random.choice(PALETTE)
    colors = [c for c in PALETTE if c != wall_color]
    shared_color, inner_move_color, inner_wrong_color, outer_move_color, outer_decoy_color = random.sample(colors, 5)

    outer_gate_candidates = _straight_gate_candidates(outer_boundary, outer_fill)
    inner_gate_candidates = _straight_gate_candidates(inner_boundary, inner_fill)
    outer_gates = _pick_separated(outer_gate_candidates, 2, blocked=set())
    inner_gates = _pick_separated(inner_gate_candidates, 3, blocked=set())
    if outer_gates is None or inner_gates is None:
        return None

    gate_plan = {
        "outer_shared": {**outer_gates[0], "color": shared_color},
        "outer_move": {**outer_gates[1], "color": outer_move_color},
        "inner_shared": {**inner_gates[0], "color": shared_color},
        "inner_move": {**inner_gates[1], "color": inner_move_color},
        "inner_wrong": {**inner_gates[2], "color": inner_wrong_color},
    }

    all_gate_positions = {gate["pos"] for gate in gate_plan.values()}
    all_gate_dests = {gate["outward"] for gate in gate_plan.values()}
    if len(all_gate_dests) != len(gate_plan):
        return None

    barrier_mask = outer_boundary | inner_boundary
    inner_region_mask = inner_fill & ~inner_boundary
    ring_region_mask = outer_fill & ~inner_fill & ~outer_boundary
    outer_region_mask = ~outer_fill

    inner_candidates = _safe_region_cells(inner_region_mask, barrier_mask)
    ring_candidates = _safe_region_cells(ring_region_mask, barrier_mask)
    outer_candidates = _safe_region_cells(outer_region_mask, barrier_mask)

    forbidden = set(all_gate_positions) | set(all_gate_dests)
    inner_cells = _pick_cells(inner_candidates, 3, forbidden)
    if inner_cells is None:
        return None
    forbidden |= set(inner_cells)
    ring_cells = _pick_cells(ring_candidates, 2, forbidden)
    if ring_cells is None:
        return None
    forbidden |= set(ring_cells)
    outer_cells = _pick_cells(outer_candidates, 1, forbidden, min_sep=1)
    if outer_cells is None:
        return None

    marker_plan = [
        {
            "start": inner_cells[0],
            "final": gate_plan["outer_shared"]["outward"],
            "color": shared_color,
            "start_level": 0,
        },
        {
            "start": inner_cells[1],
            "final": gate_plan["inner_move"]["outward"],
            "color": inner_move_color,
            "start_level": 0,
        },
        {
            "start": inner_cells[2],
            "final": inner_cells[2],
            "color": outer_move_color,
            "start_level": 0,
        },
        {
            "start": ring_cells[0],
            "final": gate_plan["outer_move"]["outward"],
            "color": outer_move_color,
            "start_level": 1,
        },
        {
            "start": ring_cells[1],
            "final": ring_cells[1],
            "color": inner_wrong_color,
            "start_level": 1,
        },
        {
            "start": outer_cells[0],
            "final": outer_cells[0],
            "color": outer_decoy_color,
            "start_level": 2,
        },
    ]

    if len({marker["start"] for marker in marker_plan}) != len(marker_plan):
        return None
    if len({marker["final"] for marker in marker_plan}) != len(marker_plan):
        return None

    inp = np.zeros(shape, dtype=int)
    inp[outer_boundary | inner_boundary] = wall_color
    for gate in gate_plan.values():
        r, c = gate["pos"]
        inp[r, c] = gate["color"]
    for marker in marker_plan:
        r, c = marker["start"]
        inp[r, c] = marker["color"]

    expected = _build_expected(shape, wall_color, outer_boundary, inner_boundary, gate_plan, marker_plan)
    meta = {
        "markers": marker_plan,
        "expected_inner_gate_colors": {shared_color, inner_move_color, inner_wrong_color},
        "expected_outer_gate_colors": {shared_color, outer_move_color},
    }
    if not _validate_generated(inp, expected, meta):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid nested-gate escape input")
