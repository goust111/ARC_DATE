import random
from collections import Counter, deque

import numpy as np


BG = 0
FRAME_COLOR = 3
STAMP_COLORS = (1, 2, 7, 9)
MAX_RETRIES = 400

_BASE_SHAPES = [
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1], [1, 0], [1, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 1, 0], [0, 1, 1, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 1], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 1, 1], [0, 1, 0, 0], [0, 1, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1, 1], [1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 0, 0]], dtype=bool),
]

_SIDE_PATTERNS = {
    2: [
        ("top", "bottom"),
        ("left", "right"),
        ("top", "right"),
        ("top", "left"),
        ("bottom", "right"),
        ("bottom", "left"),
        ("top", "top"),
        ("left", "left"),
        ("right", "right"),
        ("bottom", "bottom"),
    ],
    3: [
        ("top", "bottom", "left"),
        ("top", "bottom", "right"),
        ("left", "right", "top"),
        ("left", "right", "bottom"),
        ("top", "right", "bottom"),
        ("top", "left", "bottom"),
        ("top", "top", "right"),
        ("left", "left", "bottom"),
        ("top", "right", "right"),
        ("bottom", "left", "left"),
    ],
    4: [
        ("top", "bottom", "left", "right"),
        ("top", "top", "left", "right"),
        ("bottom", "bottom", "left", "right"),
        ("top", "bottom", "right", "right"),
        ("top", "bottom", "left", "left"),
    ],
}


def _build_shape_variants() -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[bytes] = set()
    for base in _BASE_SHAPES:
        forms = [base, np.flipud(base), np.fliplr(base), np.flipud(np.fliplr(base))]
        if base.shape[0] != base.shape[1]:
            t = base.T
            forms.extend([t, np.flipud(t), np.fliplr(t), np.flipud(np.fliplr(t))])
        for form in forms:
            key = form.tobytes() + bytes(form.shape)
            if key not in seen:
                seen.add(key)
                variants.append(form.copy())
    return variants


SHAPE_VARIANTS = _build_shape_variants()


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    return _components_mask(grid == color)


def _draw_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    h, w = mask.shape
    sub = grid[r0 : r0 + h, c0 : c0 + w]
    sub[mask] = color


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _border_positions(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    pts = set()
    for c in range(c0, c1 + 1):
        pts.add((r0, c))
        pts.add((r1, c))
    for r in range(r0, r1 + 1):
        pts.add((r, c0))
        pts.add((r, c1))
    return pts


def _side_scan_positions(bbox: tuple[int, int, int, int], side: str) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    if side == "top":
        return [(r0, c) for c in range(c0 + 1, c1)]
    if side == "bottom":
        return [(r1, c) for c in range(c0 + 1, c1)]
    if side == "left":
        return [(r, c0) for r in range(r0 + 1, r1)]
    return [(r, c1) for r in range(r0 + 1, r1)]


def _parse_gaps(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    frame_color: int,
    bg: int,
) -> list[dict]:
    gaps: list[dict] = []
    side_rank = {"top": 0, "right": 1, "bottom": 2, "left": 3}
    for side in ("top", "right", "bottom", "left"):
        positions = _side_scan_positions(bbox, side)
        values = [int(grid[r, c]) for r, c in positions]
        i = 0
        while i < len(values):
            v = values[i]
            if v == bg:
                raise AssertionError("canvas border must remain non-background")
            if v == frame_color:
                i += 1
                continue
            j = i
            while j < len(values) and values[j] == v:
                j += 1
            if j < len(values) and values[j] not in (frame_color,):
                raise AssertionError("gaps must be separated by frame cells")
            cells = positions[i:j]
            if not 1 <= len(cells) <= 3:
                raise AssertionError("gap length must be 1..3")
            if side in ("top", "bottom"):
                start = cells[0][1]
                end = cells[-1][1]
                center2 = start + end
                within = center2 if side == "top" else -center2
            else:
                start = cells[0][0]
                end = cells[-1][0]
                center2 = start + end
                within = center2 if side == "right" else -center2
            gaps.append(
                {
                    "side": side,
                    "color": int(v),
                    "length": len(cells),
                    "start": int(start),
                    "end": int(end),
                    "clockwise": (side_rank[side], within),
                }
            )
            i = j
    colors = [gap["color"] for gap in gaps]
    if not (2 <= len(gaps) <= 4):
        raise AssertionError("expected 2..4 gaps")
    if len(set(colors)) != len(colors):
        raise AssertionError("gap colors must be unique")
    return gaps


def _find_canvas(grid: np.ndarray, bg: int) -> tuple[tuple[int, int, int, int], int, list[dict]]:
    candidates = _components_mask(grid != bg)
    candidates.sort(key=len, reverse=True)

    for comp in candidates:
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 < 4 or c1 - c0 < 4:
            continue
        corner_colors = {
            int(grid[r0, c0]),
            int(grid[r0, c1]),
            int(grid[r1, c0]),
            int(grid[r1, c1]),
        }
        if len(corner_colors) != 1 or bg in corner_colors:
            continue
        color = next(iter(corner_colors))
        if any(not (r in (r0, r1) or c in (c0, c1)) for r, c in comp):
            continue
        if np.any(grid[r0 + 1 : r1, c0 + 1 : c1] != bg):
            continue
        border = [int(grid[r, c]) for r, c in _border_positions((r0, c0, r1, c1))]
        if any(v == bg for v in border):
            continue
        try:
            gaps = _parse_gaps(grid, (r0, c0, r1, c1), color, bg)
        except AssertionError:
            continue
        return (r0, c0, r1, c1), int(color), gaps

    raise AssertionError("failed to locate canvas")


def _stamp_mask(coords: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask


def _find_stamps(
    grid: np.ndarray,
    canvas_bbox: tuple[int, int, int, int],
    gaps: list[dict],
) -> dict[int, dict]:
    border_pts = _border_positions(canvas_bbox)
    stamps: dict[int, dict] = {}
    for gap in gaps:
        color = gap["color"]
        stamp_comps = []
        for comp in _components_of_color(grid, color):
            if all((r, c) in border_pts for r, c in comp):
                continue
            stamp_comps.append(comp)
        if len(stamp_comps) != 1:
            raise AssertionError("each gap color must have one outside stamp")
        coords = stamp_comps[0]
        r0, c0, r1, c1 = _bbox(coords)
        cr0, cc0, cr1, cc1 = canvas_bbox
        if not all(r < cr0 or r > cr1 or c < cc0 or c > cc1 for r, c in coords):
            raise AssertionError("stamps must stay outside canvas")
        stamps[color] = {"mask": _stamp_mask(coords)}
    return stamps


def _placement_origin(
    canvas_bbox: tuple[int, int, int, int],
    gap: dict,
    mask: np.ndarray,
) -> tuple[int, int]:
    r0, c0, r1, c1 = canvas_bbox
    h, w = mask.shape
    side = gap["side"]
    if side in ("top", "bottom"):
        numer = gap["start"] + gap["end"] - (w - 1)
        if numer % 2 != 0:
            raise AssertionError("stamp center cannot align to gap center")
        rr = r0 + 1 if side == "top" else r1 - h
        cc = numer // 2
    else:
        numer = gap["start"] + gap["end"] - (h - 1)
        if numer % 2 != 0:
            raise AssertionError("stamp center cannot align to gap center")
        rr = numer // 2
        cc = c0 + 1 if side == "left" else c1 - w
    return int(rr), int(cc)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify the canvas frame and the outside stamps, then stamp each shape
    into the canvas from its matching border gap. Larger gaps are stamped later;
    equal-sized gaps are ordered clockwise from the top side.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(Counter(map(int, g.flat)).most_common(1)[0][0])
    canvas_bbox, frame_color, gaps = _find_canvas(g, bg)
    stamps = _find_stamps(g, canvas_bbox, gaps)

    out = np.full_like(g, bg)
    _draw_frame(out, canvas_bbox, frame_color)

    for gap in sorted(gaps, key=lambda item: (item["length"], item["clockwise"])):
        mask = stamps[gap["color"]]["mask"]
        r0, c0 = _placement_origin(canvas_bbox, gap, mask)
        cr0, cc0, cr1, cc1 = canvas_bbox
        h, w = mask.shape
        if not (cr0 < r0 <= cr1 - h and cc0 < c0 <= cc1 - w):
            raise AssertionError("stamped mask must fit inside canvas interior")
        _draw_mask(out, mask, r0, c0, gap["color"])

    return out


def _pick_tie_indices(rng: random.Random, sides: list[str], n: int) -> tuple[int, int] | None:
    opposite = {("top", "bottom"), ("bottom", "top"), ("left", "right"), ("right", "left")}
    pairs = [
        (i, j)
        for i in range(n)
        for j in range(i + 1, n)
        if (sides[i], sides[j]) in opposite
    ]
    if pairs:
        return rng.choice(pairs)
    if n >= 2:
        return tuple(rng.sample(range(n), 2))
    return None


def _sample_lengths(
    rng: random.Random,
    sides: list[str],
) -> tuple[list[int], tuple[int, int] | None]:
    n = len(sides)
    lengths = [rng.randint(1, 3) for _ in range(n)]
    force_tie = rng.random() < 0.6 or n == 4
    tie_pair = _pick_tie_indices(rng, sides, n) if force_tie else None
    if tie_pair is not None:
        tie_len = rng.randint(1, 3)
        lengths[tie_pair[0]] = tie_len
        lengths[tie_pair[1]] = tie_len
    return lengths, tie_pair


def _shape_candidates(side: str, gap_len: int, inner_h: int, inner_w: int) -> list[np.ndarray]:
    want_odd = gap_len % 2
    result = []
    for mask in SHAPE_VARIANTS:
        h, w = mask.shape
        if h > inner_h or w > inner_w:
            continue
        parallel = w if side in ("top", "bottom") else h
        if parallel % 2 != want_odd:
            continue
        if mask.sum() < 3 or mask.sum() > 8:
            continue
        result.append(mask)
    return result


def _choose_shapes(
    rng: random.Random,
    specs: list[dict],
    inner_h: int,
    inner_w: int,
    tie_pair: tuple[int, int] | None,
) -> bool:
    if tie_pair is not None:
        i, j = tie_pair
        side_i = specs[i]["side"]
        side_j = specs[j]["side"]
        if {side_i, side_j} == {"top", "bottom"}:
            perp_limit = inner_h
            candidates_i = _shape_candidates(side_i, specs[i]["length"], inner_h, inner_w)
            candidates_j = _shape_candidates(side_j, specs[j]["length"], inner_h, inner_w)
            pairs = [
                (a, b)
                for a in candidates_i
                for b in candidates_j
                if a.shape[0] + b.shape[0] > perp_limit
            ]
            if pairs and rng.random() < 0.7:
                a, b = rng.choice(pairs)
                specs[i]["mask"] = a
                specs[j]["mask"] = b
        elif {side_i, side_j} == {"left", "right"}:
            perp_limit = inner_w
            candidates_i = _shape_candidates(side_i, specs[i]["length"], inner_h, inner_w)
            candidates_j = _shape_candidates(side_j, specs[j]["length"], inner_h, inner_w)
            pairs = [
                (a, b)
                for a in candidates_i
                for b in candidates_j
                if a.shape[1] + b.shape[1] > perp_limit
            ]
            if pairs and rng.random() < 0.7:
                a, b = rng.choice(pairs)
                specs[i]["mask"] = a
                specs[j]["mask"] = b

    for spec in specs:
        if "mask" in spec:
            continue
        candidates = _shape_candidates(spec["side"], spec["length"], inner_h, inner_w)
        if not candidates:
            return False
        spec["mask"] = rng.choice(candidates)
    return True


def _common_p0_candidates(spec_a: dict, spec_b: dict, inner_len: int) -> list[int]:
    dim_a = spec_a["mask"].shape[1] if spec_a["side"] in ("top", "bottom") else spec_a["mask"].shape[0]
    dim_b = spec_b["mask"].shape[1] if spec_b["side"] in ("top", "bottom") else spec_b["mask"].shape[0]
    return list(range(0, min(inner_len - dim_a, inner_len - dim_b) + 1))


def _gap_interval(spec: dict, p0: int) -> tuple[int, int]:
    mask = spec["mask"]
    side = spec["side"]
    gap_len = spec["length"]
    parallel = mask.shape[1] if side in ("top", "bottom") else mask.shape[0]
    start = (2 * p0 + parallel - gap_len) // 2
    return start, start + gap_len - 1


def _assign_parallel_offsets(
    rng: random.Random,
    specs: list[dict],
    inner_h: int,
    inner_w: int,
    tie_pair: tuple[int, int] | None,
) -> bool:
    forced: dict[int, list[int]] = {}
    if tie_pair is not None:
        i, j = tie_pair
        if {specs[i]["side"], specs[j]["side"]} in ({"top", "bottom"}, {"left", "right"}) and rng.random() < 0.7:
            inner_len = inner_w if specs[i]["side"] in ("top", "bottom") else inner_h
            common = _common_p0_candidates(specs[i], specs[j], inner_len)
            if common:
                p0 = rng.choice(common)
                forced[i] = [p0]
                forced[j] = [p0]

    candidates: dict[int, list[int]] = {}
    for idx, spec in enumerate(specs):
        side = spec["side"]
        h, w = spec["mask"].shape
        inner_len = inner_w if side in ("top", "bottom") else inner_h
        parallel = w if side in ("top", "bottom") else h
        values = list(range(0, inner_len - parallel + 1))
        if idx in forced:
            values = forced[idx]
        rng.shuffle(values)
        candidates[idx] = values

    side_to_indices: dict[str, list[int]] = {}
    for idx, spec in enumerate(specs):
        side_to_indices.setdefault(spec["side"], []).append(idx)

    order = sorted(range(len(specs)), key=lambda idx: len(candidates[idx]))
    assigned: dict[int, int] = {}
    intervals_by_side: dict[str, list[tuple[int, int]]] = {side: [] for side in side_to_indices}

    def ok(side: str, interval: tuple[int, int]) -> bool:
        for lo, hi in intervals_by_side[side]:
            if not (interval[1] + 1 < lo or hi + 1 < interval[0]):
                return False
        return True

    def rec(pos: int) -> bool:
        if pos == len(order):
            return True
        idx = order[pos]
        side = specs[idx]["side"]
        for p0 in candidates[idx]:
            interval = _gap_interval(specs[idx], p0)
            if not ok(side, interval):
                continue
            assigned[idx] = p0
            intervals_by_side[side].append(interval)
            if rec(pos + 1):
                return True
            intervals_by_side[side].pop()
            del assigned[idx]
        return False

    if not rec(0):
        return False

    for idx, p0 in assigned.items():
        specs[idx]["p0"] = p0
        specs[idx]["gap_start_rel"], specs[idx]["gap_end_rel"] = _gap_interval(specs[idx], p0)
    return True


def _inside_origin(canvas_bbox: tuple[int, int, int, int], spec: dict) -> tuple[int, int]:
    r0, c0, r1, c1 = canvas_bbox
    h, w = spec["mask"].shape
    if spec["side"] == "top":
        return r0 + 1, c0 + 1 + spec["p0"]
    if spec["side"] == "bottom":
        return r1 - h, c0 + 1 + spec["p0"]
    if spec["side"] == "left":
        return r0 + 1 + spec["p0"], c0 + 1
    return r0 + 1 + spec["p0"], c1 - w


def _clockwise_key_from_spec(spec: dict, canvas_bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = canvas_bbox
    if spec["side"] in ("top", "bottom"):
        start = c0 + 1 + spec["gap_start_rel"]
        end = c0 + 1 + spec["gap_end_rel"]
        center2 = start + end
        if spec["side"] == "top":
            return 0, center2
        return 2, -center2
    start = r0 + 1 + spec["gap_start_rel"]
    end = r0 + 1 + spec["gap_end_rel"]
    center2 = start + end
    if spec["side"] == "right":
        return 1, center2
    return 3, -center2


def _render_expected(
    shape: tuple[int, int],
    canvas_bbox: tuple[int, int, int, int],
    frame_color: int,
    specs: list[dict],
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    _draw_frame(out, canvas_bbox, frame_color)
    ordered = sorted(specs, key=lambda spec: (spec["length"], _clockwise_key_from_spec(spec, canvas_bbox)))
    for spec in ordered:
        r0, c0 = _inside_origin(canvas_bbox, spec)
        _draw_mask(out, spec["mask"], r0, c0, spec["color"])
    return out


def _paint_input_canvas(grid: np.ndarray, canvas_bbox: tuple[int, int, int, int], specs: list[dict]) -> None:
    _draw_frame(grid, canvas_bbox, FRAME_COLOR)
    r0, c0, _r1, _c1 = canvas_bbox
    for spec in specs:
        if spec["side"] in ("top", "bottom"):
            rr = r0 if spec["side"] == "top" else canvas_bbox[2]
            c_start = c0 + 1 + spec["gap_start_rel"]
            c_end = c0 + 1 + spec["gap_end_rel"]
            grid[rr, c_start : c_end + 1] = spec["color"]
        else:
            cc = c0 if spec["side"] == "left" else canvas_bbox[3]
            r_start = r0 + 1 + spec["gap_start_rel"]
            r_end = r0 + 1 + spec["gap_end_rel"]
            grid[r_start : r_end + 1, cc] = spec["color"]


def _reserve_rect(blocked: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    h_total, w_total = blocked.shape
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(h_total, r0 + h + 1)
    cc1 = min(w_total, c0 + w + 1)
    blocked[rr0:rr1, cc0:cc1] = True


def _place_outside_stamps(
    rng: random.Random,
    grid: np.ndarray,
    canvas_bbox: tuple[int, int, int, int],
    specs: list[dict],
) -> bool:
    h_total, w_total = grid.shape
    blocked = np.zeros((h_total, w_total), dtype=bool)
    r0, c0, r1, c1 = canvas_bbox
    blocked[max(0, r0 - 1) : min(h_total, r1 + 2), max(0, c0 - 1) : min(w_total, c1 + 2)] = True

    for spec in sorted(specs, key=lambda item: item["mask"].size, reverse=True):
        h, w = spec["mask"].shape
        positions = []
        for rr in range(h_total - h + 1):
            for cc in range(w_total - w + 1):
                if blocked[rr : rr + h, cc : cc + w].any():
                    continue
                positions.append((rr, cc))
        if not positions:
            return False
        rr, cc = rng.choice(positions)
        _draw_mask(grid, spec["mask"], rr, cc, spec["color"])
        _reserve_rect(blocked, rr, cc, h, w)
    return True


def _has_overlap(spec_a: dict, spec_b: dict, canvas_bbox: tuple[int, int, int, int]) -> bool:
    ra, ca = _inside_origin(canvas_bbox, spec_a)
    rb, cb = _inside_origin(canvas_bbox, spec_b)
    mask_a = spec_a["mask"]
    mask_b = spec_b["mask"]
    pts_a = {(ra + r, ca + c) for r, c in zip(*np.where(mask_a))}
    pts_b = {(rb + r, cb + c) for r, c in zip(*np.where(mask_b))}
    return bool(pts_a & pts_b)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n = rng.randint(2, 4)
    sides = list(rng.choice(_SIDE_PATTERNS[n]))
    lengths, tie_pair = _sample_lengths(rng, sides)

    force_overlap = tie_pair is not None and rng.random() < 0.7
    if force_overlap and {sides[tie_pair[0]], sides[tie_pair[1]]} == {"top", "bottom"}:
        inner_h = rng.randint(5, 7)
    else:
        inner_h = rng.randint(5, 9)
    if force_overlap and {sides[tie_pair[0]], sides[tie_pair[1]]} == {"left", "right"}:
        inner_w = rng.randint(5, 7)
    else:
        inner_w = rng.randint(5, 9)

    canvas_h = inner_h + 2
    canvas_w = inner_w + 2

    grid_h = rng.randint(max(18, canvas_h + 8), max(18, canvas_h + 14))
    grid_w = rng.randint(max(18, canvas_w + 8), max(18, canvas_w + 14))

    top = rng.randint(3, grid_h - canvas_h - 3)
    left = rng.randint(3, grid_w - canvas_w - 3)
    canvas_bbox = (top, left, top + canvas_h - 1, left + canvas_w - 1)

    colors = list(rng.sample(STAMP_COLORS, n))
    specs = [
        {"side": side, "length": length, "color": color}
        for side, length, color in zip(sides, lengths, colors)
    ]

    if not _choose_shapes(rng, specs, inner_h, inner_w, tie_pair):
        return None
    if not _assign_parallel_offsets(rng, specs, inner_h, inner_w, tie_pair):
        return None

    if force_overlap and tie_pair is not None:
        if not _has_overlap(specs[tie_pair[0]], specs[tie_pair[1]], canvas_bbox):
            return None

    inp = np.full((grid_h, grid_w), BG, dtype=int)
    _paint_input_canvas(inp, canvas_bbox, specs)
    if not _place_outside_stamps(rng, inp, canvas_bbox, specs):
        return None

    expected = _render_expected(inp.shape, canvas_bbox, FRAME_COLOR, specs)

    if Counter(map(int, inp.flat)).most_common(1)[0][0] != BG:
        return None
    try:
        got = main(inp)
    except Exception:
        return None
    if not np.array_equal(got, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_RETRIES} retries")
