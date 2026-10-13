from __future__ import annotations

import itertools
import random

import numpy as np


BG = 0
FRAME = 5
RAY = 6
PAYLOAD_COLORS = (1, 2, 3, 4, 8, 9)

_GEN_COUNTER = 0
_SIDE_CACHE: dict[tuple[str, tuple[int, int], tuple[int, int, int, int]], dict[tuple[int, int], list[tuple[tuple[tuple[int, int], ...], ...]]]] = {}
_SIDE_TARGETS = ((1, 2), (2, 2), (2, 3), (2, 4))
_GEOMETRY_PRESETS = (
    (4, 5, 5, 6, 6, 5),
    (5, 4, 6, 5, 7, 5),
    (5, 6, 6, 7, 5, 4),
    (6, 5, 7, 6, 6, 4),
)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _components_from_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    if connectivity == 4:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        steps = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]

    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in steps:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(sorted(pts), dtype=int))
    return comps


def _count_zero_regions(mask: np.ndarray) -> int:
    return len(_components_from_mask(mask, connectivity=4))


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = FRAME
    grid[r1, c0 : c1 + 1] = FRAME
    grid[r0 : r1 + 1, c0] = FRAME
    grid[r0 : r1 + 1, c1] = FRAME


def _find_frame_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    mask = grid == FRAME
    assert mask.any(), "expected one gray frame"
    r0, c0, r1, c1 = _bbox_from_mask(mask)
    assert r1 - r0 >= 2 and c1 - c0 >= 2, "frame too small"

    border = np.zeros_like(mask, dtype=bool)
    border[r0, c0 : c1 + 1] = True
    border[r1, c0 : c1 + 1] = True
    border[r0 : r1 + 1, c0] = True
    border[r0 : r1 + 1, c1] = True

    assert np.all(grid[border] == FRAME), "incomplete frame border"
    assert not np.any(grid[r0 + 1 : r1, c0 + 1 : c1] == FRAME), "frame interior must not contain frame color"
    assert np.count_nonzero(mask) == np.count_nonzero(border), "frame color must be exactly the rectangle border"
    return r0, c0, r1, c1


def _segment_endpoints(comp: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    pts = [tuple(map(int, p)) for p in comp]
    pset = set(pts)
    ends = []
    for r, c in pts:
        neighbors = 0
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (dr, dc) == (0, 0):
                    continue
                if (r + dr, c + dc) in pset:
                    neighbors += 1
        if neighbors == 1:
            ends.append((r, c))
    assert len(ends) == 2, f"segment must have exactly two endpoints, got {ends}"
    (r0, c0), (r1, c1) = ends
    assert abs(r1 - r0) == abs(c1 - c0) and r0 != r1, "segment must be diagonal"
    return ends[0], ends[1]


def _extend_from_segments(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    blocked = grid != BG
    comps = _components_from_mask(grid == RAY, connectivity=8)
    for comp in comps:
        p0, p1 = _segment_endpoints(comp)
        for src, other in ((p0, p1), (p1, p0)):
            dr = int(np.sign(src[0] - other[0]))
            dc = int(np.sign(src[1] - other[1]))
            rr, cc = src[0] + dr, src[1] + dc
            while _in_bounds(out.shape, rr, cc) and not blocked[rr, cc]:
                out[rr, cc] = RAY
                rr += dr
                cc += dc
    return out


def _solve_counts(grid: np.ndarray) -> tuple[np.ndarray, int, int, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    r0, c0, r1, c1 = _find_frame_bbox(g)
    painted = _extend_from_segments(g)
    v = _count_zero_regions(painted[:, :c0] == BG)
    h = _count_zero_regions(painted[:r0, :] == BG)
    payload = g[r0 + 1 : r1, c0 + 1 : c1]
    assert payload.size > 0, "payload missing"
    rolled = np.roll(payload, shift=v % payload.shape[0], axis=0)
    rolled = np.roll(rolled, shift=h % payload.shape[1], axis=1)
    return rolled, v, h, painted


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend endpoint rays to split the frame's outer space, then use the left/top region counts as cyclic shifts for the payload.

    Input structure:
    - Black background with one gray hollow rectangular frame around a fully filled multicolor payload.
    - Outside the frame are 1-2 short magenta diagonal segments on the left side and 1-2 on the upper side.
    - The segments do not touch the frame; the frame interior contains only the payload.

    Rule:
    - For every magenta segment, extend a magenta diagonal ray from each endpoint outward along the segment's own direction until the next non-background cell or the grid boundary.
    - Count 4-connected background regions in the full strip strictly left of the frame to get the vertical shift V, and in the full strip strictly above the frame to get the horizontal shift H.
    - Crop the payload inside the gray frame, cyclically roll it down by V and right by H, and output only that rolled payload.

    Key insight:
    - The magenta marks do not encode offsets directly; their endpoint rays geometrically partition outer background space, and those region counts are the two shift parameters.
    """
    rolled, _v, _h, _painted = _solve_counts(grid)
    return rolled


def _random_payload(h: int, w: int) -> np.ndarray:
    while True:
        payload = np.array(
            [[random.choice(PAYLOAD_COLORS) for _ in range(w)] for _ in range(h)],
            dtype=int,
        )
        if len(set(map(int, payload.ravel()))) < 4:
            continue
        periodic = False
        for dr in range(h):
            for dc in range(w):
                if (dr, dc) == (0, 0):
                    continue
                if np.array_equal(payload, np.roll(np.roll(payload, dr, axis=0), dc, axis=1)):
                    periodic = True
                    break
            if periodic:
                break
        if not periodic:
            return payload


def _baseline_grid(shape: tuple[int, int], frame_bbox: tuple[int, int, int, int]) -> np.ndarray:
    g = np.zeros(shape, dtype=int)
    _draw_frame(g, frame_bbox)
    return g


def _add_segments(grid: np.ndarray, segments: tuple[tuple[tuple[int, int], ...], ...]) -> np.ndarray:
    out = np.array(grid, copy=True)
    for seg in segments:
        for r, c in seg:
            out[r, c] = RAY
    return out


def _classify_side_counts(
    painted: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
) -> tuple[int, int]:
    r0, c0, _r1, _c1 = frame_bbox
    left = _count_zero_regions(painted[:, :c0] == BG)
    top = _count_zero_regions(painted[:r0, :] == BG)
    return left, top


def _segments_disjoint(segments: tuple[tuple[tuple[int, int], ...], ...]) -> bool:
    cells = []
    for idx, seg in enumerate(segments):
        for cell in seg:
            cells.append((idx, cell))
    for (ia, a), (ib, b) in itertools.combinations(cells, 2):
        if ia == ib:
            continue
        if max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= 1:
            return False
    return True


def _expanded_frame_contains(bbox: tuple[int, int, int, int], r: int, c: int) -> bool:
    r0, c0, r1, c1 = bbox
    return r0 - 1 <= r <= r1 + 1 and c0 - 1 <= c <= c1 + 1


def _candidate_segments(
    kind: str,
    shape: tuple[int, int],
    frame_bbox: tuple[int, int, int, int],
) -> list[tuple[tuple[int, int], ...]]:
    h, w = shape
    r0, c0, r1, c1 = frame_bbox
    candidates: list[tuple[tuple[int, int], ...]] = []

    if kind == "left":
        row_lo = r0 + 1
        row_hi = min(h - 1, r0 + 8)
        col_lo = 0
        col_hi = min(c0 - 1, 4)
    else:
        row_lo = 0
        row_hi = min(r0 - 1, 4)
        col_lo = max(c1 + 1, w - 8)
        col_hi = w - 1

    if row_lo > row_hi or col_lo > col_hi:
        return candidates

    for length in (2, 3):
        for dr, dc in ((1, 1), (1, -1)):
            for sr in range(row_lo, row_hi + 1):
                for sc in range(col_lo, col_hi + 1):
                    pts = tuple((sr + i * dr, sc + i * dc) for i in range(length))
                    if not all(_in_bounds(shape, rr, cc) for rr, cc in pts):
                        continue
                    if not all(row_lo <= rr <= row_hi and col_lo <= cc <= col_hi for rr, cc in pts):
                        continue
                    if any(_expanded_frame_contains(frame_bbox, rr, cc) for rr, cc in pts):
                        continue
                    candidates.append(pts)

    unique = []
    seen = set()
    for seg in candidates:
        if seg not in seen:
            seen.add(seg)
            unique.append(seg)
    return unique


def _build_side_catalog(
    kind: str,
    shape: tuple[int, int],
    frame_bbox: tuple[int, int, int, int],
) -> dict[tuple[int, int], list[tuple[tuple[tuple[int, int], ...], ...]]]:
    key = (kind, shape, frame_bbox)
    if key in _SIDE_CACHE:
        return _SIDE_CACHE[key]

    base = _baseline_grid(shape, frame_bbox)
    candidates = _candidate_segments(kind, shape, frame_bbox)
    catalog: dict[tuple[int, int], list[tuple[tuple[tuple[int, int], ...], ...]]] = {}

    for nseg in (1, 2):
        for combo in itertools.combinations(candidates, nseg):
            if not _segments_disjoint(combo):
                continue
            painted = _extend_from_segments(_add_segments(base, combo))
            left_count, top_count = _classify_side_counts(painted, frame_bbox)
            if kind == "left":
                if top_count != 1:
                    continue
                key2 = (nseg, left_count)
            else:
                if left_count != 1:
                    continue
                key2 = (nseg, top_count)
            catalog.setdefault(key2, []).append(combo)

    _SIDE_CACHE[key] = catalog
    return catalog


def _pick_segments_for_target(
    kind: str,
    shape: tuple[int, int],
    frame_bbox: tuple[int, int, int, int],
    target: tuple[int, int],
) -> tuple[tuple[int, int], ...] | tuple[tuple[tuple[int, int], ...], ...]:
    catalog = _build_side_catalog(kind, shape, frame_bbox)
    options = catalog.get(target, [])
    if not options:
        raise ValueError(f"no {kind} configuration for target {target}")
    return random.choice(options)


def _plan_generation() -> tuple[int, int, int, int, int, int, tuple[int, int], tuple[int, int]]:
    global _GEN_COUNTER
    idx = _GEN_COUNTER
    _GEN_COUNTER += 1
    payload_h, payload_w, top_margin, left_margin, right_margin, bottom_margin = _GEOMETRY_PRESETS[idx % len(_GEOMETRY_PRESETS)]
    left_target = _SIDE_TARGETS[idx % len(_SIDE_TARGETS)]
    top_target = _SIDE_TARGETS[(idx + 2) % len(_SIDE_TARGETS)]
    if random.random() < 0.5:
        left_target, top_target = top_target, left_target
    return payload_h, payload_w, top_margin, left_margin, right_margin, bottom_margin, left_target, top_target


def _segment_components_by_strip(
    grid: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
) -> tuple[int, int]:
    r0, c0, _r1, _c1 = frame_bbox
    left_mask = (grid == RAY).copy()
    left_mask[:r0, :] = False
    left_mask[:, c0:] = False

    top_mask = (grid == RAY).copy()
    top_mask[r0:, :] = False
    top_mask[:, :c0] = False
    return len(_components_from_mask(left_mask, connectivity=8)), len(_components_from_mask(top_mask, connectivity=8))


def _gen_attempt() -> np.ndarray | None:
    payload_h, payload_w, top_margin, left_margin, right_margin, bottom_margin, left_target, top_target = _plan_generation()
    frame_h = payload_h + 2
    frame_w = payload_w + 2

    h = top_margin + frame_h + bottom_margin
    w = left_margin + frame_w + right_margin
    if max(h, w) > 30:
        return None

    frame_bbox = (top_margin, left_margin, top_margin + frame_h - 1, left_margin + frame_w - 1)

    try:
        left_segments = _pick_segments_for_target("left", (h, w), frame_bbox, left_target)
        top_segments = _pick_segments_for_target("top", (h, w), frame_bbox, top_target)
    except ValueError:
        return None

    grid = _baseline_grid((h, w), frame_bbox)
    for seg in itertools.chain(left_segments, top_segments):
        for r, c in seg:
            grid[r, c] = RAY

    payload = _random_payload(payload_h, payload_w)
    r0, c0, r1, c1 = frame_bbox
    grid[r0 + 1 : r1, c0 + 1 : c1] = payload

    try:
        out, v, hshift, painted = _solve_counts(grid)
    except AssertionError:
        return None

    if not (2 <= v <= payload_h and 2 <= hshift <= payload_w):
        return None

    left_seg_count, top_seg_count = _segment_components_by_strip(grid, frame_bbox)
    if left_seg_count not in (1, 2) or top_seg_count not in (1, 2):
        return None

    if left_seg_count != left_target[0] or top_seg_count != top_target[0]:
        return None
    if v != left_target[1] or hshift != top_target[1]:
        return None

    if np.array_equal(out, payload):
        return None
    if np.count_nonzero(painted == RAY) <= np.count_nonzero(grid == RAY):
        return None
    if not np.array_equal(out, main(grid)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
