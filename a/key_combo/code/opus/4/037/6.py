import random
from collections import Counter

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
BODY_SHAPES = [(2, 2), (2, 3), (3, 2)]
MOVE_DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
MAX_ATTEMPTS = 500


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    nbrs = DIRS4 if connectivity == 4 else DIRS8
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in nbrs:
                    nr, nc = cr + dr, cc + dc
                    if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _coords_set(coords: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, rc)) for rc in coords}


def _largest_solid_rectangle(mask: np.ndarray) -> tuple[int, int, int, int]:
    h, w = mask.shape
    rects: list[tuple[int, int, int, int, int, int, int]] = []
    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0]:
                continue
            for r1 in range(r0, h):
                for c1 in range(c0, w):
                    block = mask[r0 : r1 + 1, c0 : c1 + 1]
                    if not block.all():
                        continue
                    hh = r1 - r0 + 1
                    ww = c1 - c0 + 1
                    area = hh * ww
                    rects.append((area, hh, ww, r0, c0, r1, c1))
    assert rects, "blue object must contain a solid body"
    non_line = [rect for rect in rects if rect[1] > 1 and rect[2] > 1]
    pool = non_line if non_line else rects
    best_area = max(rect[0] for rect in pool)
    best = [rect for rect in pool if rect[0] == best_area]
    assert len(best) == 1, "blue object must have a unique largest solid rectangle"
    _area, _hh, _ww, r0, c0, r1, c1 = best[0]
    return r0, c0, r1, c1


def _stem_shift(stem_mask: np.ndarray, body_box: tuple[int, int, int, int]) -> tuple[int, int, int]:
    r0, c0, r1, c1 = body_box
    rows, cols = np.where(stem_mask)
    assert rows.size > 0, "blue object must include a stem"

    unique_rows = sorted(set(map(int, rows)))
    unique_cols = sorted(set(map(int, cols)))

    if len(unique_rows) == 1:
        row = unique_rows[0]
        cs = sorted(map(int, cols))
        assert cs == list(range(cs[0], cs[-1] + 1)), "horizontal stem must be contiguous"
        assert r0 <= row <= r1, "horizontal stem must attach to body side"
        if cs[-1] < c0:
            return 0, -1, c0 - cs[0]
        if c1 < cs[0]:
            return 0, 1, cs[-1] - c1
        raise AssertionError("horizontal stem must lie outside the body")

    if len(unique_cols) == 1:
        col = unique_cols[0]
        rs = sorted(map(int, rows))
        assert rs == list(range(rs[0], rs[-1] + 1)), "vertical stem must be contiguous"
        assert c0 <= col <= c1, "vertical stem must attach to body side"
        if rs[-1] < r0:
            return -1, 0, r0 - rs[0]
        if r1 < rs[0]:
            return 1, 0, rs[-1] - r1
        raise AssertionError("vertical stem must lie outside the body")

    raise AssertionError("stem must be a straight one-pixel line")


def _extract_segments(grid: np.ndarray) -> list[dict]:
    segs = []
    for coords in components_mask(grid == GREEN, connectivity=8):
        pts = _coords_set(coords)
        ks = {r + c for r, c in pts}
        assert len(ks) == 1, "green component must lie on one anti-diagonal"
        k = int(next(iter(ks)))
        bs = sorted(r - c for r, c in pts)
        assert bs == list(range(bs[0], bs[-1] + 1, 2)), "green segment must be contiguous on its anti-diagonal"
        segs.append(
            {
                "coords": pts,
                "k": k,
                "b0": int(bs[0]),
                "b1": int(bs[-1]),
                "length": len(pts),
            }
        )
    return sorted(segs, key=lambda s: (s["b0"], s["k"]))


def _blue_body_cells(component: np.ndarray, body_box: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = body_box
    body = set()
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if component[r, c]:
                body.add((r, c))
    assert len(body) == (r1 - r0 + 1) * (c1 - c0 + 1)
    return body


def _move_blue_bodies(grid: np.ndarray) -> tuple[np.ndarray, list[set[tuple[int, int]]]]:
    src = np.asarray(grid, dtype=int)
    blue_comps = components_mask(src == BLUE, connectivity=4)
    out = src.copy()
    out[src == BLUE] = BG
    final_bodies: list[set[tuple[int, int]]] = []
    for comp in blue_comps:
        comp_mask = np.zeros_like(out, dtype=bool)
        comp_mask[comp[:, 0], comp[:, 1]] = True
        body_box = _largest_solid_rectangle(comp_mask)
        body_cells = _blue_body_cells(comp_mask, body_box)
        stem_mask = comp_mask.copy()
        for r, c in body_cells:
            stem_mask[r, c] = False
        dr, dc, distance = _stem_shift(stem_mask, body_box)

        moved = {(r + dr * distance, c + dc * distance) for r, c in body_cells}
        for r, c in moved:
            assert in_bounds(out.shape, r, c), "moved body leaves grid"
            assert out[r, c] == BG, "moved body must land on empty background"
        for r, c in moved:
            out[r, c] = BLUE
        final_bodies.append(moved)
    return out, final_bodies


def _fill_between_segments(grid: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    h, w = out.shape
    segs = _extract_segments(out)
    for b in range(-(w - 1), h):
        active = [seg for seg in segs if seg["b0"] <= b <= seg["b1"]]
        if len(active) < 2:
            continue
        active.sort(key=lambda seg: seg["k"])
        for left, right in zip(active, active[1:]):
            for k in range(left["k"] + 1, right["k"]):
                if (k + b) % 2 != 0:
                    continue
                r = (k + b) // 2
                c = (k - b) // 2
                if in_bounds(out.shape, r, c) and out[r, c] == BG:
                    out[r, c] = RED
    return out


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Slide each blue rectangular body along its straight stem, then fill the region between each parallel green anti-diagonal pair while the slid bodies block the fill.

    Input structure:
    - A black background contains four short green segments, each a contiguous run on one anti-diagonal `r+c=const`.
    - The green segments form two parallel pairs whose active spans on the main-diagonal axis `r-c` do not mix.
    - Two separate blue connected objects each consist of one solid rectangular body plus a one-pixel-wide horizontal or vertical stem.

    Rule:
    - For each blue object, identify the unique largest solid rectangle as the body and the remaining straight run as the stem.
    - Erase the whole blue object, then translate only the body by exactly the stem length in the stem direction and redraw that rectangle in blue.
    - After both bodies slide, scan each main-diagonal line `r-c=b`; whenever two green segments cover that same `b`, fill every background cell strictly between them with red.
    - Blue rectangles drawn after sliding are blockers, so those occupied cells stay blue instead of turning red.

    Key insight:
    - The stem is only a displacement cue: moving the blue rectangles first determines the obstacle layout, and only then can the diagonal-pair fill be computed correctly.
    """
    grid = np.asarray(input_grid, dtype=int)
    assert grid.ndim == 2
    moved_grid, _ = _move_blue_bodies(grid)
    return _fill_between_segments(moved_grid)


def _segment_cells(k: int, b0: int, length: int) -> set[tuple[int, int]]:
    cells = set()
    for t in range(length):
        b = b0 + 2 * t
        r = (k + b) // 2
        c = (k - b) // 2
        cells.add((r, c))
    return cells


def _pair_fill_cells(k0: int, gap: int, b0: int, length: int) -> set[tuple[int, int]]:
    fill = set()
    for t in range(length):
        b = b0 + 2 * t
        for k in range(k0 + 2, k0 + gap, 2):
            r = (k + b) // 2
            c = (k - b) // 2
            fill.add((r, c))
    return fill


def _fill_cells_from_segments(shape: tuple[int, int], seg_a: set[tuple[int, int]], seg_b: set[tuple[int, int]]) -> set[tuple[int, int]]:
    grid = np.full(shape, BG, dtype=int)
    _paint_cells(grid, seg_a, GREEN)
    _paint_cells(grid, seg_b, GREEN)
    filled = _fill_between_segments(grid)
    coords = np.argwhere(filled == RED)
    return {tuple(map(int, rc)) for rc in coords}


def _paint_cells(grid: np.ndarray, cells: set[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _bbox_from_cells(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    coords = np.asarray(sorted(cells), dtype=int)
    return bbox_coords(coords)


def _expand_bbox(box: tuple[int, int, int, int], shape: tuple[int, int], margin: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    h, w = shape
    return max(0, r0 - margin), max(0, c0 - margin), min(h - 1, r1 + margin), min(w - 1, c1 + margin)


def _body_cells(top: int, left: int, height: int, width: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(top, top + height) for c in range(left, left + width)}


def _min_manhattan(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> int:
    return min(abs(r1 - r2) + abs(c1 - c2) for r1, c1 in a for r2, c2 in b)


def _touches_4(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> bool:
    for r, c in a:
        for dr, dc in DIRS4:
            if (r + dr, c + dc) in b:
                return True
    return False


def _stem_cells_for_move(final_top: int, final_left: int, body_h: int, body_w: int, dr: int, dc: int, distance: int, lane: int) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    input_top = final_top - dr * distance
    input_left = final_left - dc * distance
    stem = set()
    if dr == 0 and dc == 1:
        row = input_top + lane
        for c in range(input_left + body_w, input_left + body_w + distance):
            stem.add((row, c))
    elif dr == 0 and dc == -1:
        row = input_top + lane
        for c in range(final_left + body_w, input_left):
            stem.add((row, c))
    elif dr == 1 and dc == 0:
        col = input_left + lane
        for r in range(input_top + body_h, input_top + body_h + distance):
            stem.add((r, col))
    elif dr == -1 and dc == 0:
        col = input_left + lane
        for r in range(final_top + body_h, input_top):
            stem.add((r, col))
    else:
        raise AssertionError("invalid move direction")
    return stem, (input_top, input_left)


def _build_expected(shape: tuple[int, int], pair_specs: list[dict], body_specs: list[dict]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for pair in pair_specs:
        _paint_cells(out, pair["seg_a"], GREEN)
        _paint_cells(out, pair["seg_b"], GREEN)
    for body in body_specs:
        _paint_cells(out, body["final_body"], BLUE)
    return _fill_between_segments(out)


def _build_no_slide_variant(inp: np.ndarray) -> np.ndarray:
    return _fill_between_segments(inp)


def _build_no_block_variant(shape: tuple[int, int], pair_specs: list[dict]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for pair in pair_specs:
        _paint_cells(out, pair["seg_a"], GREEN)
        _paint_cells(out, pair["seg_b"], GREEN)
    return _fill_between_segments(out)


def _intervals_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return not (a[1] < b[0] or b[1] < a[0])


def _make_pair_spec(shape: tuple[int, int], reserved: np.ndarray, used_b_ranges: list[tuple[int, int]]) -> dict | None:
    h, w = shape
    length = random.choice([3, 4])
    gap = random.choice([4, 6])
    k0 = random.randint(2, h + w - 2 - gap - 1)
    b0 = random.randint(-(w - 1), h - 1)
    b_range = (b0, b0 + 2 * (length - 1))
    if any(_intervals_overlap(b_range, used) for used in used_b_ranges):
        return None
    if (k0 + b0) % 2 != 0:
        return None

    seg_a = _segment_cells(k0, b0, length)
    seg_b = _segment_cells(k0 + gap, b0, length)
    if len(seg_a) != length or len(seg_b) != length:
        return None
    if not all(in_bounds(shape, r, c) for r, c in (seg_a | seg_b)):
        return None
    fill = _fill_cells_from_segments(shape, seg_a, seg_b)
    all_cells = seg_a | seg_b | fill
    if not fill:
        return None
    if any(reserved[r, c] for r, c in all_cells):
        return None
    box = _bbox_from_cells(all_cells)
    ex = _expand_bbox(box, shape, margin=1)
    if reserved[ex[0] : ex[2] + 1, ex[1] : ex[3] + 1].any():
        return None
    return {
        "k0": k0,
        "gap": gap,
        "b0": b0,
        "b_range": b_range,
        "length": length,
        "seg_a": seg_a,
        "seg_b": seg_b,
        "fill": fill,
    }


def _find_body_for_pair(shape: tuple[int, int], reserved: np.ndarray, pair: dict) -> dict | None:
    h, w = shape
    pair_cells = pair["seg_a"] | pair["seg_b"] | pair["fill"]
    pair_box = _expand_bbox(_bbox_from_cells(pair_cells), shape, margin=4)
    all_segments = [pair["seg_a"], pair["seg_b"]]

    body_shapes = BODY_SHAPES[:]
    random.shuffle(body_shapes)
    for body_h, body_w in body_shapes:
        top_values = list(range(pair_box[0], pair_box[2] - body_h + 2))
        left_values = list(range(pair_box[1], pair_box[3] - body_w + 2))
        random.shuffle(top_values)
        random.shuffle(left_values)
        for top in top_values:
            for left in left_values:
                final_body = _body_cells(top, left, body_h, body_w)
                if any(reserved[r, c] for r, c in final_body):
                    continue
                if final_body & (pair["seg_a"] | pair["seg_b"]):
                    continue
                overlap = len(final_body & pair["fill"])
                if overlap == 0 or overlap >= len(pair["fill"]):
                    continue
                for seg_idx, target_seg in enumerate(all_segments):
                    other_seg = all_segments[1 - seg_idx]
                    if not _touches_4(final_body, target_seg):
                        continue
                    if _min_manhattan(final_body, other_seg) <= 1:
                        continue
                    for dr, dc in random.sample(MOVE_DIRS, len(MOVE_DIRS)):
                        axis = body_w if dr == 0 else body_h
                        distances = [d for d in [2, 3, 4] if d >= axis]
                        random.shuffle(distances)
                        for distance in distances:
                            lane_values = list(range(body_h if dr == 0 else body_w))
                            random.shuffle(lane_values)
                            for lane in lane_values:
                                stem, _ = _stem_cells_for_move(top, left, body_h, body_w, dr, dc, distance, lane)
                                input_top = top - dr * distance
                                input_left = left - dc * distance
                                input_body = _body_cells(input_top, input_left, body_h, body_w)
                                input_object = input_body | stem
                                if not all(in_bounds(shape, r, c) for r, c in input_object):
                                    continue
                                if len(stem) != distance:
                                    continue
                                if input_body & final_body:
                                    continue
                                if input_body & pair_cells:
                                    continue
                                if stem & (pair["seg_a"] | pair["seg_b"]):
                                    continue
                                if any(reserved[r, c] for r, c in input_object):
                                    continue
                                if _min_manhattan(input_body, target_seg) <= 1:
                                    continue
                                blocked = len(final_body & pair["fill"])
                                if blocked < 1 or blocked >= len(pair["fill"]):
                                    continue
                                return {
                                    "input_body": input_body,
                                    "stem": stem,
                                    "final_body": final_body,
                                    "target_segment": seg_idx,
                                }
    return None


def _choose_pair_and_body(shape: tuple[int, int], reserved: np.ndarray, used_b_ranges: list[tuple[int, int]]) -> tuple[dict, dict, np.ndarray] | None:
    for _ in range(180):
        pair = _make_pair_spec(shape, reserved, used_b_ranges)
        if pair is None:
            continue
        body = _find_body_for_pair(shape, reserved, pair)
        if body is None:
            continue
        pair_cells = pair["seg_a"] | pair["seg_b"] | pair["fill"]
        reserve_box = _expand_bbox(
            _bbox_from_cells(pair_cells | body["input_body"] | body["stem"] | body["final_body"]),
            shape,
            margin=1,
        )
        reserve_mask = np.zeros(shape, dtype=bool)
        reserve_mask[reserve_box[0] : reserve_box[2] + 1, reserve_box[1] : reserve_box[3] + 1] = True
        return pair, body, reserve_mask
    return None


def _validate_semantics(inp: np.ndarray, expected: np.ndarray, pair_specs: list[dict], body_specs: list[dict]) -> bool:
    try:
        solved = main(inp)
    except Exception:
        return False
    if not np.array_equal(solved, expected):
        return False

    green_segments = _extract_segments(inp)
    if len(green_segments) != 4:
        return False
    if sorted(seg["length"] for seg in green_segments) not in ([3, 3, 3, 3], [3, 3, 3, 4], [3, 3, 4, 4], [3, 4, 4, 4], [4, 4, 4, 4]):
        return False
    if len(components_mask(inp == BLUE, connectivity=4)) != 2:
        return False

    for pair, body in zip(pair_specs, body_specs):
        blocked = len(body["final_body"] & pair["fill"])
        if blocked < 1 or blocked >= len(pair["fill"]):
            return False
        if len(pair["fill"] - body["final_body"]) < 1:
            return False

    no_slide = _build_no_slide_variant(inp)
    if np.array_equal(no_slide, expected):
        return False
    no_block = _build_no_block_variant(inp.shape, pair_specs)
    if np.array_equal(no_block, expected):
        return False

    if int(np.sum(expected == RED)) <= 0 or int(np.sum(expected == BLUE)) <= 0:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(16, 22)
    w = random.randint(16, 22)
    reserved = np.zeros((h, w), dtype=bool)
    pair_specs: list[dict] = []
    body_specs: list[dict] = []
    used_b_ranges: list[tuple[int, int]] = []

    for _ in range(2):
        choice = _choose_pair_and_body((h, w), reserved, used_b_ranges)
        if choice is None:
            return None
        pair, body, reserve_mask = choice
        pair_specs.append(pair)
        body_specs.append(body)
        used_b_ranges.append(pair["b_range"])
        reserved |= reserve_mask

    inp = np.full((h, w), BG, dtype=int)
    for pair in pair_specs:
        _paint_cells(inp, pair["seg_a"], GREEN)
        _paint_cells(inp, pair["seg_b"], GREEN)
    for body in body_specs:
        _paint_cells(inp, body["input_body"], BLUE)
        _paint_cells(inp, body["stem"], BLUE)

    expected = _build_expected((h, w), pair_specs, body_specs)
    if not _validate_semantics(inp, expected, pair_specs, body_specs):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for task 6")
