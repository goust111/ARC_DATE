from __future__ import annotations

from collections import Counter, deque
from fractions import Fraction
import itertools
import math
import random

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GREY = 5
COLORS = tuple(range(10))
GEN_ATTEMPTS = 400


def _round_half_up(value: Fraction) -> int:
    return math.floor(value + Fraction(1, 2))


def _get_bg(grid: np.ndarray) -> int:
    counts = Counter(int(v) for v in grid.ravel())
    ranked = counts.most_common()
    assert ranked, "grid must be non-empty"
    if len(ranked) > 1:
        assert ranked[0][1] > ranked[1][1], "background must be uniquely most common"
    return int(ranked[0][0])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((int(cr), int(cc)))
                for dr, dc in DIR4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _normalize(coords: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0, c0, _r1, _c1 = _bbox(coords)
    return tuple(sorted((r - r0, c - c0) for r, c in coords))


def _is_rect_frame(coords: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    expected = {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if r in (r0, r1) or c in (c0, c1)
    }
    return set(coords) == expected


def _ordered_path(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    pts = set(coords)
    adj: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r, c in coords:
        nbrs: list[tuple[int, int]] = []
        for dr, dc in DIR4:
            nxt = (r + dr, c + dc)
            if nxt in pts:
                nbrs.append(nxt)
        adj[(r, c)] = nbrs
    if any(len(nbrs) > 2 for nbrs in adj.values()):
        raise AssertionError("segment must be a simple path")
    ends = sorted(p for p, nbrs in adj.items() if len(nbrs) == 1)
    assert len(ends) == 2, "segment must have exactly two endpoints"
    start = ends[0]
    path = [start]
    prev: tuple[int, int] | None = None
    cur = start
    while True:
        choices = [nxt for nxt in adj[cur] if nxt != prev]
        if not choices:
            break
        assert len(choices) == 1, "segment path must not branch"
        prev, cur = cur, choices[0]
        path.append(cur)
    assert len(path) == len(coords), "segment path must cover every pixel exactly once"
    return path


def _turn_count(path: list[tuple[int, int]]) -> int:
    dirs = [
        (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        for i in range(len(path) - 1)
    ]
    return sum(dirs[i] != dirs[i - 1] for i in range(1, len(dirs)))


def _assert_segment(coords: list[tuple[int, int]]) -> None:
    assert 3 <= len(coords) <= 5, "each path segment must contain 3-5 pixels"
    path = _ordered_path(coords)
    assert _turn_count(path) <= 1, "each segment may have at most one right-angle bend"


def _component_center(coords: list[tuple[int, int]]) -> tuple[Fraction, Fraction]:
    n = len(coords)
    sr = sum(r for r, _ in coords)
    sc = sum(c for _, c in coords)
    return Fraction(sr, n), Fraction(sc, n)


def _average_segment_center(segments: list[list[tuple[int, int]]]) -> tuple[int, int]:
    rs = sum((_component_center(seg)[0] for seg in segments), start=Fraction(0, 1))
    cs = sum((_component_center(seg)[1] for seg in segments), start=Fraction(0, 1))
    return _round_half_up(rs / len(segments)), _round_half_up(cs / len(segments))


def _ring_boundary_local(h: int, w: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        (r, c)
        for r in range(h)
        for c in range(w)
        if r in (0, h - 1) or c in (0, w - 1)
    )


def _candidate_ring_shapes(
    segments: list[tuple[tuple[int, int], ...]]
) -> list[tuple[int, int]]:
    total = sum(len(seg) for seg in segments)
    if (total + 4) % 2 != 0:
        return []
    sum_hw = (total + 4) // 2
    shapes: list[tuple[int, int]] = []
    lengths = [len(seg) for seg in segments]

    for h in range(3, sum_hw - 2):
        w = sum_hw - h
        if w < 3:
            continue
        boundary = frozenset(_ring_boundary_local(h, w))
        options: list[list[frozenset[tuple[int, int]]]] = []
        feasible = True
        for seg in segments:
            max_r = max(r for r, _ in seg)
            max_c = max(c for _, c in seg)
            placements: set[frozenset[tuple[int, int]]] = set()
            for top in range(h - max_r):
                for left in range(w - max_c):
                    placed = frozenset((r + top, c + left) for r, c in seg)
                    if placed <= boundary:
                        placements.add(placed)
            if not placements:
                feasible = False
                break
            options.append(sorted(placements, key=lambda cells: tuple(sorted(cells))))
        if not feasible:
            continue

        order = sorted(range(len(segments)), key=lambda idx: len(options[idx]))
        remaining = [0] * (len(order) + 1)
        for i in range(len(order) - 1, -1, -1):
            remaining[i] = remaining[i + 1] + lengths[order[i]]

        def dfs(pos: int, used: frozenset[tuple[int, int]]) -> bool:
            if len(used) + remaining[pos] < len(boundary):
                return False
            if pos == len(order):
                return len(used) == len(boundary)
            seg_idx = order[pos]
            for placed in options[seg_idx]:
                if used & placed:
                    continue
                if dfs(pos + 1, used | placed):
                    return True
            return False

        if dfs(0, frozenset()):
            shapes.append((h, w))
    return shapes


def _ring_top_left(center_r: int, center_c: int, h: int, w: int) -> tuple[int, int]:
    top = _round_half_up(Fraction(center_r, 1) - Fraction(h - 1, 2))
    left = _round_half_up(Fraction(center_c, 1) - Fraction(w - 1, 2))
    return top, left


def _find_sample_bbox(grid: np.ndarray, bg: int) -> tuple[int, int, int, int] | None:
    h, w = grid.shape
    candidates: list[tuple[int, int, int, int]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for comp in _components(grid == color):
            if not _is_rect_frame(comp):
                continue
            r0, c0, r1, c1 = _bbox(comp)
            if (r0 in (0, h - 1) or r1 in (0, h - 1)) and (c0 in (0, w - 1) or c1 in (0, w - 1)):
                candidates.append((r0, c0, r1, c1))
    if not candidates:
        return None
    candidates.sort(key=lambda box: ((box[2] - box[0] + 1) * (box[3] - box[1] + 1), box))
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)
    sample_bbox = _find_sample_bbox(g, bg)

    protected = np.zeros_like(g, dtype=bool)
    out = np.full_like(g, int(bg))
    if sample_bbox is not None:
        sr0, sc0, sr1, sc1 = sample_bbox
        protected[sr0 : sr1 + 1, sc0 : sc1 + 1] = True
        out[sr0 : sr1 + 1, sc0 : sc1 + 1] = g[sr0 : sr1 + 1, sc0 : sc1 + 1]

    color_candidates: list[tuple[int, list[tuple[int, int, int, int, tuple[tuple[int, int], ...]]]]] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) != bg):
        mask = (g == color) & ~protected
        segments = _components(mask)
        if not segments:
            continue
        assert len(segments) == 4, "each non-sample color must have exactly four scattered segments"
        for seg in segments:
            _assert_segment(seg)
        center_r, center_c = _average_segment_center(segments)
        segment_shapes = [_normalize(seg) for seg in segments]
        placements = []
        for ring_h, ring_w in _candidate_ring_shapes(segment_shapes):
            top, left = _ring_top_left(center_r, center_c, ring_h, ring_w)
            cells = tuple((top + r, left + c) for r, c in _ring_boundary_local(ring_h, ring_w))
            if any(not (0 <= r < h and 0 <= c < w) for r, c in cells):
                continue
            if any(bool(protected[r, c]) for r, c in cells):
                continue
            placements.append((ring_h, ring_w, top, left, cells))
        assert placements, "every color must admit at least one ring placement"
        color_candidates.append((color, placements))

    occupied = {(r, c) for r, c in zip(*np.nonzero(protected))}
    chosen: dict[int, tuple[int, int, int, int, tuple[tuple[int, int], ...]]] = {}
    order = sorted(range(len(color_candidates)), key=lambda idx: len(color_candidates[idx][1]))

    def backtrack(pos: int) -> bool:
        if pos == len(order):
            return True
        idx = order[pos]
        color, placements = color_candidates[idx]
        for plan in placements:
            cells = plan[4]
            if any(cell in occupied for cell in cells):
                continue
            chosen[idx] = plan
            for cell in cells:
                occupied.add(cell)
            if backtrack(pos + 1):
                return True
            for cell in cells:
                occupied.remove(cell)
            del chosen[idx]
        return False

    assert backtrack(0), "ring placements must be mutually non-overlapping"
    for idx, (color, _placements) in enumerate(color_candidates):
        cells = chosen[idx][4]
        for r, c in cells:
            out[r, c] = int(color)
    return out


def _ring_cycle(h: int, w: int) -> list[tuple[int, int]]:
    cycle: list[tuple[int, int]] = []
    cycle.extend((0, c) for c in range(w))
    cycle.extend((r, w - 1) for r in range(1, h - 1))
    cycle.extend((h - 1, c) for c in range(w - 1, -1, -1))
    cycle.extend((r, 0) for r in range(h - 2, 0, -1))
    return cycle


def _turns_for_fragment(fragment: list[tuple[int, int]]) -> int:
    return _turn_count(fragment)


def _profile_ok(turns: list[int], profile: str) -> bool:
    bent = sum(t == 1 for t in turns)
    straight = sum(t == 0 for t in turns)
    if profile == "mixed":
        return bent >= 1 and straight >= 1
    if profile == "bent_heavy":
        return bent >= 3
    if profile == "straight_heavy":
        return straight >= 2
    raise ValueError(profile)


def _random_ring_plan() -> dict[str, object]:
    dims = [
        (h, w)
        for h in range(3, 8)
        for w in range(3, 8)
        if 12 <= 2 * (h + w) - 4 <= 20
    ]
    profile = random.choice(["mixed", "bent_heavy", "straight_heavy"])
    for _ in range(300):
        h, w = random.choice(dims)
        perimeter = 2 * (h + w) - 4
        cycle = _ring_cycle(h, w)
        lengths = None
        for _ in range(80):
            first_three = [random.randint(3, 5) for _ in range(3)]
            last = perimeter - sum(first_three)
            if 3 <= last <= 5:
                lengths = tuple(first_three + [last])
                break
        if lengths is None:
            continue
        start = random.randrange(perimeter)
        fragments: list[list[tuple[int, int]]] = []
        turns: list[int] = []
        pos = start
        ok = True
        for length in lengths:
            fragment = [cycle[(pos + step) % perimeter] for step in range(length)]
            turn_count = _turns_for_fragment(fragment)
            if turn_count > 1:
                ok = False
                break
            fragments.append(fragment)
            turns.append(turn_count)
            pos += length
        if not ok or not _profile_ok(turns, profile):
            continue
        return {
            "h": h,
            "w": w,
            "ring_cells": tuple(_ring_boundary_local(h, w)),
            "segments": [tuple(_normalize(fragment)) for fragment in fragments],
        }
    raise RuntimeError("failed to create a valid ring plan")


def _coords_center_local(coords: tuple[tuple[int, int], ...]) -> tuple[Fraction, Fraction]:
    n = len(coords)
    return (
        Fraction(sum(r for r, _ in coords), n),
        Fraction(sum(c for _, c in coords), n),
    )


def _translate_coords(
    coords: tuple[tuple[int, int], ...],
    top: int,
    left: int,
) -> list[tuple[int, int]]:
    return [(top + r, left + c) for r, c in coords]


def _mark_rect(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    mask[r0 : r1 + 1, c0 : c1 + 1] = True


def _paint_coords(grid: np.ndarray, coords: list[tuple[int, int]], color: int) -> None:
    for r, c in coords:
        grid[r, c] = int(color)


def _sample_bbox_for_corner(corner: str, frame_h: int, frame_w: int, h: int, w: int) -> tuple[int, int, int, int]:
    if corner == "tl":
        return 0, 0, frame_h - 1, frame_w - 1
    if corner == "tr":
        return 0, w - frame_w, frame_h - 1, w - 1
    if corner == "bl":
        return h - frame_h, 0, h - 1, frame_w - 1
    if corner == "br":
        return h - frame_h, w - frame_w, h - 1, w - 1
    raise ValueError(corner)


def _bbox_distance(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    row_gap = max(br0 - ar1 - 1, ar0 - br1 - 1, 0)
    col_gap = max(bc0 - ac1 - 1, ac0 - bc1 - 1, 0)
    return max(row_gap, col_gap)


def _choose_anchor(
    grid_h: int,
    grid_w: int,
    ring_h: int,
    ring_w: int,
    forbidden_bbox: tuple[int, int, int, int],
    existing: list[tuple[int, int, int, int]],
) -> tuple[int, int] | None:
    margin_r = ring_h + 3
    margin_c = ring_w + 3
    candidates: list[tuple[int, int]] = []
    for center_r in range(margin_r, grid_h - margin_r):
        for center_c in range(margin_c, grid_w - margin_c):
            top, left = _ring_top_left(center_r, center_c, ring_h, ring_w)
            box = (top, left, top + ring_h - 1, left + ring_w - 1)
            if _bbox_distance(box, forbidden_bbox) < 1:
                continue
            if any(_bbox_distance(box, other) < 2 for other in existing):
                continue
            candidates.append((center_r, center_c))
    if not candidates:
        return None
    return random.choice(candidates)


def _can_place_cells(
    cells: list[tuple[int, int]],
    grid_h: int,
    grid_w: int,
    occupied: np.ndarray,
    forbidden: np.ndarray,
    pad: int,
) -> bool:
    for r, c in cells:
        if not (0 <= r < grid_h and 0 <= c < grid_w):
            return False
        if bool(forbidden[r, c]):
            return False
        r0 = max(0, r - pad)
        r1 = min(grid_h, r + pad + 1)
        c0 = max(0, c - pad)
        c1 = min(grid_w, c + pad + 1)
        if bool(np.any(occupied[r0:r1, c0:c1])):
            return False
    return True


def _place_segments_for_color(
    plan: dict[str, object],
    anchor: tuple[int, int],
    grid_h: int,
    grid_w: int,
    occupied: np.ndarray,
    forbidden: np.ndarray,
) -> list[list[tuple[int, int]]] | None:
    segments = list(plan["segments"])
    random.shuffle(segments)
    spread = random.randint(4, 6)
    sector_offsets = [(-spread, -spread), (-spread, spread), (spread, -spread), (spread, spread)]
    random.shuffle(sector_offsets)
    placed: list[list[tuple[int, int]]] = []
    local_marks: list[list[tuple[int, int]]] = []

    for seg, (off_r, off_c) in zip(segments, sector_offsets):
        local_center_r, local_center_c = _coords_center_local(seg)
        target_center_r = anchor[0] + off_r
        target_center_c = anchor[1] + off_c
        base_top = _round_half_up(Fraction(target_center_r, 1) - local_center_r)
        base_left = _round_half_up(Fraction(target_center_c, 1) - local_center_c)
        deltas = [(dr, dc) for dr in range(-3, 4) for dc in range(-3, 4)]
        random.shuffle(deltas)
        picked: list[tuple[int, int]] | None = None
        for dr, dc in deltas:
            cells = _translate_coords(seg, base_top + dr, base_left + dc)
            if _can_place_cells(cells, grid_h, grid_w, occupied, forbidden, pad=1):
                picked = cells
                break
        if picked is None:
            return None
        placed.append(picked)
        local_marks.append(picked)
        for r, c in picked:
            occupied[r, c] = True

    return placed


def _build_expected_output(
    inp: np.ndarray,
    bg: int,
    sample_bbox: tuple[int, int, int, int],
    plans: dict[int, dict[str, object]],
    placed_segments: dict[int, list[list[tuple[int, int]]]],
) -> np.ndarray | None:
    h, w = inp.shape
    out = np.full((h, w), int(bg), dtype=int)
    protected = np.zeros((h, w), dtype=bool)
    sr0, sc0, sr1, sc1 = sample_bbox
    out[sr0 : sr1 + 1, sc0 : sc1 + 1] = inp[sr0 : sr1 + 1, sc0 : sc1 + 1]
    protected[sr0 : sr1 + 1, sc0 : sc1 + 1] = True

    occupied = {(r, c) for r, c in zip(*np.nonzero(protected))}
    for color, plan in plans.items():
        ring_h = int(plan["h"])
        ring_w = int(plan["w"])
        center_r, center_c = _average_segment_center(placed_segments[color])
        top, left = _ring_top_left(center_r, center_c, ring_h, ring_w)
        cells = [(top + r, left + c) for r, c in plan["ring_cells"]]
        if any(not (0 <= r < h and 0 <= c < w) for r, c in cells):
            return None
        if any((r, c) in occupied for r, c in cells):
            return None
        for r, c in cells:
            occupied.add((r, c))
            out[r, c] = int(color)
    return out


def _generate_case() -> np.ndarray | None:
    color_count = random.choice([1, 2, 3])
    bg = random.choice([color for color in COLORS if color != GREY])
    usable_colors = [color for color in COLORS if color not in {bg, GREY}]
    colors = random.sample(usable_colors, color_count)

    plans = {color: _random_ring_plan() for color in colors}
    sample_color = random.choice(colors)
    sample_plan = plans[sample_color]
    frame_h = int(sample_plan["h"]) + 4
    frame_w = int(sample_plan["w"]) + 4

    min_h = max(22, frame_h + 8, 12 + 4 * color_count)
    min_w = max(22, frame_w + 8, 12 + 4 * color_count)
    grid_h = random.randint(min_h, 30)
    grid_w = random.randint(min_w, 30)
    corner = random.choice(["tl", "tr", "bl", "br"])
    sample_bbox = _sample_bbox_for_corner(corner, frame_h, frame_w, grid_h, grid_w)
    sr0, sc0, sr1, sc1 = sample_bbox

    inp = np.full((grid_h, grid_w), int(bg), dtype=int)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    forbidden = np.zeros((grid_h, grid_w), dtype=bool)
    pad_r0 = max(0, sr0 - 1)
    pad_c0 = max(0, sc0 - 1)
    pad_r1 = min(grid_h - 1, sr1 + 1)
    pad_c1 = min(grid_w - 1, sc1 + 1)
    _mark_rect(forbidden, pad_r0, pad_c0, pad_r1, pad_c1)

    frame_cells = [
        (r, c)
        for r in range(sr0, sr1 + 1)
        for c in range(sc0, sc1 + 1)
        if r in (sr0, sr1) or c in (sc0, sc1)
    ]
    _paint_coords(inp, frame_cells, GREY)
    for r, c in frame_cells:
        occupied[r, c] = True

    sample_ring_top = sr0 + 2
    sample_ring_left = sc0 + 2
    sample_ring = [(sample_ring_top + r, sample_ring_left + c) for r, c in sample_plan["ring_cells"]]
    _paint_coords(inp, sample_ring, sample_color)
    for r, c in sample_ring:
        occupied[r, c] = True

    color_boxes: list[tuple[int, int, int, int]] = []
    anchors: dict[int, tuple[int, int]] = {}
    for color in random.sample(colors, len(colors)):
        plan = plans[color]
        anchor = _choose_anchor(
            grid_h,
            grid_w,
            int(plan["h"]),
            int(plan["w"]),
            sample_bbox,
            color_boxes,
        )
        if anchor is None:
            return None
        anchors[color] = anchor
        top, left = _ring_top_left(anchor[0], anchor[1], int(plan["h"]), int(plan["w"]))
        color_boxes.append((top, left, top + int(plan["h"]) - 1, left + int(plan["w"]) - 1))

    placed_segments: dict[int, list[list[tuple[int, int]]]] = {}
    for color in colors:
        placed = _place_segments_for_color(plans[color], anchors[color], grid_h, grid_w, occupied, forbidden)
        if placed is None:
            return None
        placed_segments[color] = placed
        for seg in placed:
            _paint_coords(inp, seg, color)

    expected = _build_expected_output(inp, bg, sample_bbox, plans, placed_segments)
    if expected is None:
        return None
    try:
        if np.array_equal(main(inp), expected):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        candidate = _generate_case()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid input for puzzle 2")
