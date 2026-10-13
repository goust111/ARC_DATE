from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np


BBox = tuple[int, int, int, int]
Coord = tuple[int, int]

PALETTE = (1, 2, 4, 6, 7)
SCENARIOS = ("length", "column", "row")
GRID_MIN = 20
GRID_MAX = 26
MAX_ATTEMPTS = 400

_GEN_CALL_COUNT = 0


@dataclass(frozen=True)
class ObjSpec:
    color: int
    body_bbox: BBox
    direction: Coord
    tail_len: int
    attach: int


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _box_h(box: BBox) -> int:
    return int(box[2] - box[0] + 1)


def _box_w(box: BBox) -> int:
    return int(box[3] - box[1] + 1)


def _boxes_overlap(a: BBox, b: BBox) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _is_full(mask: np.ndarray, box: BBox) -> bool:
    r0, c0, r1, c1 = box
    return bool(np.all(mask[r0 : r1 + 1, c0 : c1 + 1]))


def _single_component_coords(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a nonempty object"
    cells = {tuple(map(int, rc)) for rc in coords}
    start = next(iter(cells))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (rr, cc) in cells and (rr, cc) not in seen:
                seen.add((rr, cc))
                stack.append((rr, cc))
    assert len(seen) == len(cells), "same-color object must be 4-connected"
    return coords


def _body_cells(spec: ObjSpec) -> set[Coord]:
    r0, c0, r1, c1 = spec.body_bbox
    return {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
    }


def _tail_cells(spec: ObjSpec) -> list[Coord]:
    r0, c0, r1, c1 = spec.body_bbox
    dr, dc = spec.direction
    L = spec.tail_len
    if dr == 0 and dc > 0:
        return [(spec.attach, c) for c in range(c1 + 1, c1 + L + 1)]
    if dr == 0 and dc < 0:
        return [(spec.attach, c) for c in range(c0 - L, c0)]
    if dr > 0 and dc == 0:
        return [(r, spec.attach) for r in range(r1 + 1, r1 + L + 1)]
    assert dr < 0 and dc == 0
    return [(r, spec.attach) for r in range(r0 - L, r0)]


def _input_cells(spec: ObjSpec) -> set[Coord]:
    return _body_cells(spec) | set(_tail_cells(spec))


def _envelope_cells(spec: ObjSpec, shape: tuple[int, int]) -> set[Coord]:
    H, W = map(int, shape)
    r0, c0, r1, c1 = spec.body_bbox
    cells: set[Coord] = set()
    for offset in range(1, spec.tail_len + 1):
        left = c0 + offset
        right = c1 - offset
        if left <= right:
            for rr in (r0 - offset, r1 + offset):
                if 0 <= rr < H:
                    for cc in range(left, right + 1):
                        cells.add((rr, cc))

        top = r0 + offset
        bottom = r1 - offset
        if top <= bottom:
            for cc in (c0 - offset, c1 + offset):
                if 0 <= cc < W:
                    for rr in range(top, bottom + 1):
                        cells.add((rr, cc))
    return cells


def _output_footprint(spec: ObjSpec, shape: tuple[int, int]) -> set[Coord]:
    return _body_cells(spec) | _envelope_cells(spec, shape)


def _priority_key(spec: ObjSpec) -> tuple[int, int, int]:
    r0, c0, _r1, _c1 = spec.body_bbox
    return int(spec.tail_len), int(-c0), int(-r0)


def _spec_fits(spec: ObjSpec, shape: tuple[int, int]) -> bool:
    H, W = map(int, shape)
    for r, c in _input_cells(spec):
        if not (0 <= r < H and 0 <= c < W):
            return False
    for r, c in _output_footprint(spec, shape):
        if not (0 <= r < H and 0 <= c < W):
            return False
    return True


def _detect_object(mask: np.ndarray, color: int) -> ObjSpec:
    coords = _single_component_coords(mask)
    r0, c0, r1, c1 = _bbox_coords(coords)
    candidates: set[tuple[BBox, Coord, int, int]] = set()

    def add_candidate(body_box: BBox, direction: Coord, outside: np.ndarray) -> None:
        if outside.size == 0:
            return
        br0, bc0, br1, bc1 = body_box
        h = int(br1 - br0 + 1)
        w = int(bc1 - bc0 + 1)
        if min(h, w) < 2 or max(h, w) < 3 or h == w:
            return
        if direction[0] == 0 and w <= h:
            return
        if direction[1] == 0 and h <= w:
            return
        if not _is_full(mask, body_box):
            return
        length = int(outside.shape[0])
        if not 2 <= length <= 4:
            return

        rows = sorted({int(v) for v in outside[:, 0]})
        cols = sorted({int(v) for v in outside[:, 1]})
        dr, dc = direction
        if dr == 0:
            if len(rows) != 1:
                return
            attach = int(rows[0])
            if not (br0 <= attach <= br1):
                return
            if dc > 0:
                expected = list(range(bc1 + 1, bc1 + length + 1))
            else:
                expected = list(range(bc0 - length, bc0))
            if cols != expected:
                return
        else:
            if len(cols) != 1:
                return
            attach = int(cols[0])
            if not (bc0 <= attach <= bc1):
                return
            if dr > 0:
                expected = list(range(br1 + 1, br1 + length + 1))
            else:
                expected = list(range(br0 - length, br0))
            if rows != expected:
                return

        candidates.add((body_box, direction, length, attach))

    for body_c1 in range(c0, c1):
        body = (r0, c0, r1, body_c1)
        add_candidate(body, (0, 1), coords[coords[:, 1] > body_c1])
    for body_c0 in range(c0 + 1, c1 + 1):
        body = (r0, body_c0, r1, c1)
        add_candidate(body, (0, -1), coords[coords[:, 1] < body_c0])
    for body_r1 in range(r0, r1):
        body = (r0, c0, body_r1, c1)
        add_candidate(body, (1, 0), coords[coords[:, 0] > body_r1])
    for body_r0 in range(r0 + 1, r1 + 1):
        body = (body_r0, c0, r1, c1)
        add_candidate(body, (-1, 0), coords[coords[:, 0] < body_r0])

    assert len(candidates) == 1, "bar body and tail decomposition must be unique"
    body_box, direction, length, attach = next(iter(candidates))
    return ObjSpec(int(color), body_box, direction, int(length), int(attach))


def _extract_objects(grid: np.ndarray) -> list[ObjSpec]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    assert bg == 0, "expected black background"

    colors = sorted(int(v) for v in np.unique(g) if int(v) != bg)
    assert 3 <= len(colors) <= 5, "expected 3-5 colored objects"
    assert len(set(colors)) == len(colors)
    assert set(colors) <= set(PALETTE), "unexpected foreground colors"

    objects: list[ObjSpec] = []
    for color in colors:
        spec = _detect_object(g == color, color)
        assert _spec_fits(spec, g.shape), "detected object must stay in bounds"
        for other in objects:
            assert not _boxes_overlap(spec.body_bbox, other.body_bbox), "body bboxes must be disjoint"
        objects.append(spec)
    return objects


def _render_from_specs(shape: tuple[int, int], objects: list[ObjSpec]) -> np.ndarray:
    H, W = map(int, shape)
    out = np.zeros((H, W), dtype=int)
    body_owner = np.full((H, W), -1, dtype=int)
    best_owner = np.full((H, W), -1, dtype=int)

    for idx, spec in enumerate(objects):
        r0, c0, r1, c1 = spec.body_bbox
        assert not np.any(body_owner[r0 : r1 + 1, c0 : c1 + 1] != -1), "body cells must be unique"
        body_owner[r0 : r1 + 1, c0 : c1 + 1] = idx
        out[r0 : r1 + 1, c0 : c1 + 1] = int(spec.color)

    for idx, spec in enumerate(objects):
        key = _priority_key(spec)
        for r, c in _envelope_cells(spec, shape):
            if body_owner[r, c] != -1:
                continue
            prev = int(best_owner[r, c])
            if prev == -1 or key > _priority_key(objects[prev]):
                best_owner[r, c] = idx

    for r in range(H):
        for c in range(W):
            idx = int(best_owner[r, c])
            if idx != -1 and body_owner[r, c] == -1:
                out[r, c] = int(objects[idx].color)
    return out


def _build_input(shape: tuple[int, int], objects: list[ObjSpec]) -> np.ndarray:
    H, W = map(int, shape)
    grid = np.zeros((H, W), dtype=int)
    for spec in objects:
        for r, c in _input_cells(spec):
            assert grid[r, c] == 0, "input objects must not overlap"
            grid[r, c] = int(spec.color)
    return grid


def _mark_input_buffer(blocked: np.ndarray, cells: set[Coord]) -> None:
    H, W = blocked.shape
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < H and 0 <= cc < W:
                    blocked[rr, cc] = True


def _make_horizontal(color: int, r0: int, c0: int, h: int, w: int, tail_len: int, direction: int, attach: int) -> ObjSpec:
    return ObjSpec(color, (r0, c0, r0 + h - 1, c0 + w - 1), (0, direction), tail_len, attach)


def _make_vertical(color: int, r0: int, c0: int, h: int, w: int, tail_len: int, direction: int, attach: int) -> ObjSpec:
    return ObjSpec(color, (r0, c0, r0 + h - 1, c0 + w - 1), (direction, 0), tail_len, attach)


def _sample_grid_shape(total_count: int) -> tuple[int, int]:
    lo = GRID_MIN + (2 if total_count == 5 else 0)
    return random.randint(lo, GRID_MAX), random.randint(lo, GRID_MAX)


def _focus_pair(shape: tuple[int, int], colors: list[int], scenario: str) -> list[ObjSpec] | None:
    H, W = map(int, shape)
    color_a, color_b = colors[:2]

    if scenario == "row":
        L = random.randint(2, 4)
        h = random.randint(2, 3)
        w_hi = min(9, W - 2 * L)
        if w_hi < 2 * L + 1:
            return None
        w = random.randint(2 * L + 1, w_hi)
        gap = L
        total_h = 2 * h + gap
        if H < total_h + 2 * L + 1:
            return None
        r0 = random.randint(L, H - total_h - L)
        c0 = random.randint(L, W - w - L)
        spec_a = _make_horizontal(
            color_a,
            r0,
            c0,
            h,
            w,
            L,
            random.choice((-1, 1)),
            r0 + random.randint(0, h - 1),
        )
        spec_b = _make_horizontal(
            color_b,
            r0 + h + gap,
            c0,
            h,
            w,
            L,
            random.choice((-1, 1)),
            r0 + h + gap + random.randint(0, h - 1),
        )
        return [spec_a, spec_b] if _spec_fits(spec_a, shape) and _spec_fits(spec_b, shape) else None

    if scenario == "column":
        L = random.randint(2, 4)
        w = random.randint(2, 3)
        h_hi = min(9, H - 2 * L)
        if h_hi < 2 * L + 1:
            return None
        h = random.randint(2 * L + 1, h_hi)
        gap = L
        total_w = 2 * w + gap
        if W < total_w + 2 * L + 1:
            return None
        r0 = random.randint(L, H - h - L)
        c0 = random.randint(L, W - total_w - L)
        spec_a = _make_vertical(
            color_a,
            r0,
            c0,
            h,
            w,
            L,
            random.choice((-1, 1)),
            c0 + random.randint(0, w - 1),
        )
        spec_b = _make_vertical(
            color_b,
            r0,
            c0 + w + gap,
            h,
            w,
            L,
            random.choice((-1, 1)),
            c0 + w + gap + random.randint(0, w - 1),
        )
        return [spec_a, spec_b] if _spec_fits(spec_a, shape) and _spec_fits(spec_b, shape) else None

    left_L = random.randint(2, 3)
    right_L = random.randint(left_L + 1, 4)
    w = random.randint(2, 3)
    h_hi = min(9, H - 2 * right_L)
    if h_hi < 2 * right_L + 1:
        return None
    h = random.randint(2 * right_L + 1, h_hi)
    gap = right_L
    total_w = 2 * w + gap
    if W < total_w + 2 * right_L + 1:
        return None
    r0 = random.randint(right_L, H - h - right_L)
    c0 = random.randint(right_L, W - total_w - right_L)
    spec_a = _make_vertical(
        color_a,
        r0,
        c0,
        h,
        w,
        left_L,
        random.choice((-1, 1)),
        c0 + random.randint(0, w - 1),
    )
    spec_b = _make_vertical(
        color_b,
        r0,
        c0 + w + gap,
        h,
        w,
        right_L,
        random.choice((-1, 1)),
        c0 + w + gap + random.randint(0, w - 1),
    )
    return [spec_a, spec_b] if _spec_fits(spec_a, shape) and _spec_fits(spec_b, shape) else None


def _random_extra_object(shape: tuple[int, int], color: int) -> ObjSpec | None:
    H, W = map(int, shape)
    L = random.randint(2, 4)
    if random.random() < 0.5:
        h = random.randint(2, 3)
        w_hi = min(9, W - 2 * L)
        if w_hi < 2 * L + 1 or H - h - 2 * L < 0:
            return None
        w = random.randint(2 * L + 1, w_hi)
        r0 = random.randint(L, H - h - L)
        c0 = random.randint(L, W - w - L)
        return _make_horizontal(
            color,
            r0,
            c0,
            h,
            w,
            L,
            random.choice((-1, 1)),
            r0 + random.randint(0, h - 1),
        )

    w = random.randint(2, 3)
    h_hi = min(9, H - 2 * L)
    if h_hi < 2 * L + 1 or W - w - 2 * L < 0:
        return None
    h = random.randint(2 * L + 1, h_hi)
    r0 = random.randint(L, H - h - L)
    c0 = random.randint(L, W - w - L)
    return _make_vertical(
        color,
        r0,
        c0,
        h,
        w,
        L,
        random.choice((-1, 1)),
        c0 + random.randint(0, w - 1),
    )


def _focus_overlap_cells(shape: tuple[int, int], spec_a: ObjSpec, spec_b: ObjSpec) -> set[Coord]:
    body_mask = _body_cells(spec_a) | _body_cells(spec_b)
    return (_envelope_cells(spec_a, shape) & _envelope_cells(spec_b, shape)) - body_mask


def _validate_focus_branch(shape: tuple[int, int], specs: list[ObjSpec], scenario: str) -> None:
    spec_a, spec_b = specs[0], specs[1]
    overlap = _focus_overlap_cells(shape, spec_a, spec_b)
    assert overlap, "focus pair must have non-body envelope overlap"

    a_r0, a_c0, _a_r1, _a_c1 = spec_a.body_bbox
    b_r0, b_c0, _b_r1, _b_c1 = spec_b.body_bbox
    winner = spec_a if _priority_key(spec_a) > _priority_key(spec_b) else spec_b

    if scenario == "length":
        assert spec_a.tail_len < spec_b.tail_len
        assert a_r0 == b_r0 and a_c0 < b_c0, "length branch should contradict the column tiebreak shortcut"
        assert winner is spec_b, "larger tail must win"
    elif scenario == "column":
        assert spec_a.tail_len == spec_b.tail_len
        assert a_r0 == b_r0 and a_c0 < b_c0
        assert winner is spec_a, "smaller left-column body must win the tie"
    else:
        assert spec_a.tail_len == spec_b.tail_len
        assert a_c0 == b_c0 and a_r0 < b_r0
        assert winner is spec_a, "smaller top-row body must win the tie"


def _validate_specs(shape: tuple[int, int], specs: list[ObjSpec], scenario: str) -> None:
    H, W = map(int, shape)
    assert H <= 30 and W <= 30
    assert 3 <= len(specs) <= 5
    assert len({spec.color for spec in specs}) == len(specs)
    assert {spec.color for spec in specs} <= set(PALETTE)

    body_boxes: list[BBox] = []
    occupied_input: set[Coord] = set()
    non_focus_output: set[Coord] = set()
    for idx, spec in enumerate(specs):
        assert _spec_fits(spec, shape)
        h = _box_h(spec.body_bbox)
        w = _box_w(spec.body_bbox)
        assert min(h, w) >= 2 and max(h, w) >= 3 and h != w
        assert 2 <= spec.tail_len <= 4
        assert max(h, w) >= 2 * spec.tail_len + 1, "radius must visibly affect the envelope"
        for other_box in body_boxes:
            assert not _boxes_overlap(spec.body_bbox, other_box)
        body_boxes.append(spec.body_bbox)

        cells = _input_cells(spec)
        assert occupied_input.isdisjoint(cells), "input objects must not overlap"
        occupied_input |= cells

        if idx >= 2:
            footprint = _output_footprint(spec, shape)
            assert non_focus_output.isdisjoint(footprint), "extra objects should stay visually isolated"
            non_focus_output |= footprint

    _validate_focus_branch(shape, specs, scenario)


def _generate_specs(shape: tuple[int, int], scenario: str, total_count: int) -> list[ObjSpec] | None:
    colors = random.sample(list(PALETTE), total_count)
    focus = _focus_pair(shape, colors, scenario)
    if focus is None:
        return None

    H, W = map(int, shape)
    blocked_input = np.zeros((H, W), dtype=bool)
    protected_output: set[Coord] = set()
    specs = focus[:]
    for spec in specs:
        _mark_input_buffer(blocked_input, _input_cells(spec))
        protected_output |= _output_footprint(spec, shape)

    for color in colors[2:]:
        placed = False
        for _ in range(200):
            spec = _random_extra_object(shape, color)
            if spec is None or not _spec_fits(spec, shape):
                continue
            cells = _input_cells(spec)
            if any(blocked_input[r, c] for r, c in cells):
                continue
            footprint = _output_footprint(spec, shape)
            if protected_output & footprint:
                continue
            specs.append(spec)
            _mark_input_buffer(blocked_input, cells)
            protected_output |= footprint
            placed = True
            break
        if not placed:
            return None
    return specs


def _attempt_generate(scenario: str) -> np.ndarray | None:
    total_count = random.choices([3, 4, 5], weights=[4, 4, 2], k=1)[0]
    shape = _sample_grid_shape(total_count)
    specs = _generate_specs(shape, scenario, total_count)
    if specs is None:
        return None
    _validate_specs(shape, specs, scenario)

    inp = _build_input(shape, specs)
    expected = _render_from_specs(shape, specs)
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return None

    recovered = _extract_objects(inp)
    assert sorted((s.color, s.body_bbox, s.direction, s.tail_len, s.attach) for s in recovered) == sorted(
        (s.color, s.body_bbox, s.direction, s.tail_len, s.attach) for s in specs
    )
    assert not np.array_equal(inp, expected), "task must visibly transform the grid"
    return inp


def main(grid: np.ndarray) -> np.ndarray:
    """Erase each bar tail and repaint a tail-length shrinking diamond envelope around the bar body.

    Input structure
    - The background is black.
    - There are 3-5 uniquely colored filled horizontal or vertical bars, and each bar has a same-colored one-pixel-wide straight tail attached to one end.
    - Tail lengths are between 2 and 4, body bounding boxes do not overlap, and different colors identify different objects.

    Rule
    - For each color, uniquely decompose the object into a solid rectangular body and its one-pixel-wide tail, then measure the tail length.
    - Erase every tail, keep every body pixel unchanged, and generate the shrinking diamond envelope around the body bbox using that tail length as the radius.
    - When envelope pixels from multiple objects overlap, keep the color with larger tail length; if tied, keep the one whose body bbox has the smaller top-left column, then the smaller top-left row; body pixels override every envelope pixel.

    Key insight
    - The tail no longer means displacement: its length is reinterpreted as the envelope radius, and contested cells can only be colored correctly after all objects and all envelopes are compared globally.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    objects = _extract_objects(g)
    return _render_from_specs(g.shape, objects)


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT

    for _ in range(MAX_ATTEMPTS):
        scenario = SCENARIOS[_GEN_CALL_COUNT % len(SCENARIOS)]
        _GEN_CALL_COUNT += 1
        grid = _attempt_generate(scenario)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid tail-radius diamond task")
