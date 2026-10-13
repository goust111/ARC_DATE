from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


BBox = tuple[int, int, int, int]
Direction = tuple[int, int]

BG = 0
DIRS: tuple[Direction, ...] = ((0, 1), (0, -1), (1, 0), (-1, 0))

GRID_H_RANGE = (14, 18)
GRID_W_RANGE = (14, 18)
BODY_H_RANGE = (3, 6)
BODY_W_RANGE = (3, 6)
TAIL_LEN_RANGE = (2, 4)
MAX_GEN_ATTEMPTS = 500


@dataclass(frozen=True)
class Tail:
    direction: Direction
    length: int
    anchor: int


@dataclass(frozen=True)
class ObjectSpec:
    color: int
    body: BBox
    tail: Tail | None
    hidden_tail: bool = False

    @property
    def area(self) -> int:
        return _bbox_area(self.body)


def _bbox_area(box: BBox) -> int:
    r0, c0, r1, c1 = box
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _bbox_dims(box: BBox) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    return r1 - r0 + 1, c1 - c0 + 1


def _in_bbox(r: int, c: int, box: BBox) -> bool:
    r0, c0, r1, c1 = box
    return r0 <= r <= r1 and c0 <= c <= c1


def _shift_box(box: BBox, direction: Direction, length: int) -> BBox:
    dr, dc = direction
    r0, c0, r1, c1 = box
    return r0 + dr * length, c0 + dc * length, r1 + dr * length, c1 + dc * length


def _box_mask(shape: tuple[int, int], box: BBox) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    r0, c0, r1, c1 = box
    out[r0 : r1 + 1, c0 : c1 + 1] = True
    return out


def _tail_cells(box: BBox, tail: Tail | None) -> list[tuple[int, int]]:
    if tail is None:
        return []
    r0, c0, r1, c1 = box
    dr, dc = tail.direction
    if dc > 0:
        return [(tail.anchor, c1 + k) for k in range(1, tail.length + 1)]
    if dc < 0:
        return [(tail.anchor, c0 - k) for k in range(1, tail.length + 1)]
    if dr > 0:
        return [(r1 + k, tail.anchor) for k in range(1, tail.length + 1)]
    return [(r0 - k, tail.anchor) for k in range(1, tail.length + 1)]


def _object_cells(spec: ObjectSpec) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = spec.body
    cells = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
    cells.update(_tail_cells(spec.body, spec.tail))
    return cells


def _visible_side_cells(grid: np.ndarray, color: int, box: BBox) -> dict[str, list[tuple[int, int]]]:
    r0, c0, r1, c1 = box
    sides = {
        "top": [(r0, c) for c in range(c0, c1 + 1)],
        "bottom": [(r1, c) for c in range(c0, c1 + 1)],
        "left": [(r, c0) for r in range(r0, r1 + 1)],
        "right": [(r, c1) for r in range(r0, r1 + 1)],
    }
    return {
        name: [(r, c) for r, c in cells if int(grid[r, c]) == color]
        for name, cells in sides.items()
    }


def _tail_signature_from_coords(coords: np.ndarray, box: BBox) -> tuple[Direction, int, int] | None:
    r0, c0, r1, c1 = box
    outside = [(int(r), int(c)) for r, c in coords if not _in_bbox(int(r), int(c), box)]
    if not outside:
        return None
    candidates: list[tuple[Direction, int, int]] = []

    def add(direction: Direction, anchor: int, line: list[int], expected: list[int]) -> None:
        if len(expected) >= 2 and line == expected:
            candidates.append((direction, len(expected), anchor))

    if all(r0 <= r <= r1 and c > c1 for r, c in outside):
        rows = sorted({r for r, _ in outside})
        if len(rows) == 1:
            cols = sorted(c for _, c in outside)
            add((0, 1), rows[0], cols, list(range(c1 + 1, max(cols) + 1)))

    if all(r0 <= r <= r1 and c < c0 for r, c in outside):
        rows = sorted({r for r, _ in outside})
        if len(rows) == 1:
            cols = sorted(c for _, c in outside)
            add((0, -1), rows[0], cols, list(range(min(cols), c0)))

    if all(c0 <= c <= c1 and r > r1 for r, c in outside):
        cols = sorted({c for _, c in outside})
        if len(cols) == 1:
            rows = sorted(r for r, _ in outside)
            add((1, 0), cols[0], rows, list(range(r1 + 1, max(rows) + 1)))

    if all(c0 <= c <= c1 and r < r0 for r, c in outside):
        cols = sorted({c for _, c in outside})
        if len(cols) == 1:
            rows = sorted(r for r, _ in outside)
            add((-1, 0), cols[0], rows, list(range(min(rows), r0)))

    if len(candidates) != 1:
        return None
    return candidates[0]


def _infer_body_bbox(grid: np.ndarray, color: int) -> BBox:
    coords = np.argwhere(grid == color)
    assert coords.size, "expected color pixels"
    rows = sorted({int(r) for r in coords[:, 0]})
    cols = sorted({int(c) for c in coords[:, 1]})

    candidates: list[tuple[int, BBox]] = []
    for r0 in rows:
        for r1 in rows:
            if r1 <= r0:
                continue
            for c0 in cols:
                for c1 in cols:
                    if c1 <= c0:
                        continue
                    region = grid[r0 : r1 + 1, c0 : c1 + 1]
                    if np.any(region == BG):
                        continue
                    touches = (
                        np.any((coords[:, 0] == r0) & (coords[:, 1] >= c0) & (coords[:, 1] <= c1))
                        and np.any((coords[:, 0] == r1) & (coords[:, 1] >= c0) & (coords[:, 1] <= c1))
                        and np.any((coords[:, 1] == c0) & (coords[:, 0] >= r0) & (coords[:, 0] <= r1))
                        and np.any((coords[:, 1] == c1) & (coords[:, 0] >= r0) & (coords[:, 0] <= r1))
                    )
                    if not touches:
                        continue
                    outside_sig = _tail_signature_from_coords(coords, (r0, c0, r1, c1))
                    outside_count = int(np.sum(~np.array([_in_bbox(int(r), int(c), (r0, c0, r1, c1)) for r, c in coords])))
                    if outside_sig is None and outside_count:
                        continue
                    candidates.append((_bbox_area((r0, c0, r1, c1)), (r0, c0, r1, c1)))

    assert candidates, f"no valid body bbox for color {color}"
    max_area = max(area for area, _ in candidates)
    best_boxes = sorted({box for area, box in candidates if area == max_area})
    assert len(best_boxes) == 1, f"body bbox must be unique for color {color}"
    return best_boxes[0]


def _completed_mask(grid: np.ndarray, color: int, body: BBox) -> np.ndarray:
    mask = grid == color
    r0, c0, r1, c1 = body
    mask = mask.copy()
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _detect_tail(mask: np.ndarray, body: BBox) -> Tail | None:
    coords = np.argwhere(mask)
    sig = _tail_signature_from_coords(coords, body)
    if sig is None:
        return None
    direction, length, anchor = sig
    return Tail(direction=direction, length=length, anchor=anchor)


def _draw_body(grid: np.ndarray, box: BBox, color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0 : r1 + 1, c0 : c1 + 1] = color


def _draw_spec(grid: np.ndarray, spec: ObjectSpec) -> None:
    _draw_body(grid, spec.body, spec.color)
    for r, c in _tail_cells(spec.body, spec.tail):
        grid[r, c] = spec.color


def _render_input(shape: tuple[int, int], specs: list[ObjectSpec], z_order: list[int]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for idx in z_order:
        _draw_spec(grid, specs[idx])
    return grid


def _render_output(shape: tuple[int, int], specs: list[ObjectSpec]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    statics = [spec for spec in specs if spec.tail is None]
    sliders = [spec for spec in specs if spec.tail is not None]
    statics.sort(key=lambda spec: (-spec.area, spec.color))
    sliders.sort(key=lambda spec: (-spec.area, spec.color))

    for spec in statics:
        _draw_body(out, spec.body, spec.color)
    for spec in sliders:
        moved = _shift_box(spec.body, spec.tail.direction, spec.tail.length)
        _draw_body(out, moved, spec.color)
    return out


def _sample_unique_boxes(
    rows: int,
    cols: int,
    occupied: np.ndarray,
    height: int,
    width: int,
    forbid_adjacent: bool = False,
) -> BBox | None:
    choices: list[BBox] = []
    for r0 in range(rows - height + 1):
        for c0 in range(cols - width + 1):
            box = (r0, c0, r0 + height - 1, c0 + width - 1)
            mask = _box_mask((rows, cols), box)
            if np.any(mask & occupied):
                continue
            if forbid_adjacent:
                rlo = max(0, r0 - 1)
                rhi = min(rows, r0 + height + 1)
                clo = max(0, c0 - 1)
                chi = min(cols, c0 + width + 1)
                if np.any(occupied[rlo:rhi, clo:chi]):
                    continue
            choices.append(box)
    return random.choice(choices) if choices else None


def _can_place_slider(
    shape: tuple[int, int],
    body: BBox,
    tail: Tail,
    occupied: np.ndarray,
    slider_dest_boxes: list[BBox],
) -> bool:
    rows, cols = shape
    full = np.zeros(shape, dtype=bool)
    full |= _box_mask(shape, body)
    for r, c in _tail_cells(body, tail):
        if not (0 <= r < rows and 0 <= c < cols):
            return False
        full[r, c] = True
    if np.any(full & occupied):
        return False
    dest = _shift_box(body, tail.direction, tail.length)
    dr0, dc0, dr1, dc1 = dest
    if not (0 <= dr0 <= dr1 < rows and 0 <= dc0 <= dc1 < cols):
        return False
    for other in slider_dest_boxes:
        if _boxes_overlap(dest, other):
            return False
    return True


def _boxes_overlap(a: BBox, b: BBox) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _edge_overlap_slice(hidden: ObjectSpec, occ_h: int, occ_w: int) -> BBox | None:
    r0, c0, r1, c1 = hidden.body
    tail = hidden.tail
    assert tail is not None

    if tail.direction == (0, 1):
        if occ_w > c1 - c0 + 1 or occ_h > (r1 - r0):
            return None
        rows = list(range(r0 + 1, r1))
        if len(rows) < 2:
            return None
        start = random.randint(max(r0, tail.anchor - 1), min(r1 - 1, tail.anchor))
        end = min(start + random.choice([1, 2]), r1 - 1)
        if start == r0 and end == r1:
            return None
        ov_r0, ov_r1 = start, end
        top_pad = random.randint(0, occ_h - (ov_r1 - ov_r0 + 1))
        box_r0 = ov_r0 - top_pad
        box_r1 = box_r0 + occ_h - 1
        box_c1 = c1
        box_c0 = box_c1 - occ_w + 1
        if box_r0 < 0 or box_r1 > r1 + 1 or box_c0 < 0:
            return None
        return (box_r0, box_c0, box_r1, box_c1)

    if tail.direction == (0, -1):
        if occ_w > c1 - c0 + 1 or occ_h > (r1 - r0):
            return None
        start = random.randint(max(r0, tail.anchor - 1), min(r1 - 1, tail.anchor))
        end = min(start + random.choice([1, 2]), r1 - 1)
        ov_r0, ov_r1 = start, end
        top_pad = random.randint(0, occ_h - (ov_r1 - ov_r0 + 1))
        box_r0 = ov_r0 - top_pad
        box_r1 = box_r0 + occ_h - 1
        box_c0 = c0
        box_c1 = box_c0 + occ_w - 1
        if box_r0 < 0 or box_r1 > r1 + 1 or box_c1 >= c1 + 2:
            return None
        return (box_r0, box_c0, box_r1, box_c1)

    if tail.direction == (1, 0):
        if occ_h > r1 - r0 + 1 or occ_w > (c1 - c0):
            return None
        start = random.randint(max(c0, tail.anchor - 1), min(c1 - 1, tail.anchor))
        end = min(start + random.choice([1, 2]), c1 - 1)
        ov_c0, ov_c1 = start, end
        left_pad = random.randint(0, occ_w - (ov_c1 - ov_c0 + 1))
        box_c0 = ov_c0 - left_pad
        box_c1 = box_c0 + occ_w - 1
        box_r1 = r1
        box_r0 = box_r1 - occ_h + 1
        if box_c0 < 0 or box_c1 > c1 + 1 or box_r0 < 0:
            return None
        return (box_r0, box_c0, box_r1, box_c1)

    if occ_h > r1 - r0 + 1 or occ_w > (c1 - c0):
        return None
    start = random.randint(max(c0, tail.anchor - 1), min(c1 - 1, tail.anchor))
    end = min(start + random.choice([1, 2]), c1 - 1)
    ov_c0, ov_c1 = start, end
    left_pad = random.randint(0, occ_w - (ov_c1 - ov_c0 + 1))
    box_c0 = ov_c0 - left_pad
    box_c1 = box_c0 + occ_w - 1
    box_r0 = r0
    box_r1 = box_r0 + occ_h - 1
    if box_c0 < 0 or box_c1 > c1 + 1 or box_r1 >= r1 + 2:
        return None
    return (box_r0, box_c0, box_r1, box_c1)


def _sample_hidden_slider(shape: tuple[int, int], used_areas: set[int]) -> ObjectSpec | None:
    rows, cols = shape
    for _ in range(200):
        height = random.randint(*BODY_H_RANGE)
        width = random.randint(*BODY_W_RANGE)
        area = height * width
        if area in used_areas:
            continue
        tail_len = random.randint(*TAIL_LEN_RANGE)
        direction = random.choice(DIRS)

        if direction[0] == 0 and height < 3:
            continue
        if direction[1] == 0 and width < 3:
            continue

        if direction == (0, 1):
            c0 = random.randint(1, cols - width - tail_len - 1)
            r0 = random.randint(1, rows - height - 1)
            anchor = random.randint(r0 + 1, r0 + height - 2)
        elif direction == (0, -1):
            c0 = random.randint(tail_len + 1, cols - width - 2)
            r0 = random.randint(1, rows - height - 1)
            anchor = random.randint(r0 + 1, r0 + height - 2)
        elif direction == (1, 0):
            r0 = random.randint(1, rows - height - tail_len - 1)
            c0 = random.randint(1, cols - width - 1)
            anchor = random.randint(c0 + 1, c0 + width - 2)
        else:
            r0 = random.randint(tail_len + 1, rows - height - 2)
            c0 = random.randint(1, cols - width - 1)
            anchor = random.randint(c0 + 1, c0 + width - 2)

        body = (r0, c0, r0 + height - 1, c0 + width - 1)
        tail = Tail(direction=direction, length=tail_len, anchor=anchor)
        moved = _shift_box(body, direction, tail_len)
        mr0, mc0, mr1, mc1 = moved
        if not (0 <= mr0 <= mr1 < rows and 0 <= mc0 <= mc1 < cols):
            continue
        return ObjectSpec(color=-1, body=body, tail=tail, hidden_tail=True)
    return None


def _sample_occluder(shape: tuple[int, int], hidden: ObjectSpec, used_areas: set[int]) -> ObjectSpec | None:
    rows, cols = shape
    for _ in range(250):
        height = random.randint(*BODY_H_RANGE)
        width = random.randint(*BODY_W_RANGE)
        area = height * width
        if area in used_areas:
            continue
        box = _edge_overlap_slice(hidden, height, width)
        if box is None:
            continue
        r0, c0, r1, c1 = box
        if not (0 <= r0 <= r1 < rows and 0 <= c0 <= c1 < cols):
            continue
        if set(_tail_cells(hidden.body, hidden.tail)) & {
            (r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
        }:
            continue
        if _box_inside(box, hidden.body):
            continue
        return ObjectSpec(color=-1, body=box, tail=None)
    return None


def _box_inside(inner: BBox, outer: BBox) -> bool:
    ir0, ic0, ir1, ic1 = inner
    or0, oc0, or1, oc1 = outer
    return or0 <= ir0 and ic0 >= oc0 and ir1 <= or1 and ic1 <= oc1


def _sample_free_object(
    shape: tuple[int, int],
    occupied_input: np.ndarray,
    slider_dest_boxes: list[BBox],
    used_areas: set[int],
    role: str,
) -> ObjectSpec | None:
    rows, cols = shape
    for _ in range(250):
        height = random.randint(*BODY_H_RANGE)
        width = random.randint(*BODY_W_RANGE)
        area = height * width
        if area in used_areas:
            continue
        box = _sample_unique_boxes(rows, cols, occupied_input, height, width, forbid_adjacent=False)
        if box is None:
            continue
        if role == "static":
            return ObjectSpec(color=-1, body=box, tail=None)

        tail_len = random.randint(*TAIL_LEN_RANGE)
        direction = random.choice(DIRS)
        if direction[0] == 0 and height < 3:
            continue
        if direction[1] == 0 and width < 3:
            continue
        r0, c0, r1, c1 = box
        if direction[0] == 0:
            anchor = random.randint(r0 + 1, r1 - 1)
        else:
            anchor = random.randint(c0 + 1, c1 - 1)
        tail = Tail(direction=direction, length=tail_len, anchor=anchor)
        if not _can_place_slider(shape, box, tail, occupied_input, slider_dest_boxes):
            continue
        return ObjectSpec(color=-1, body=box, tail=tail)
    return None


def _assign_colors(specs: list[ObjectSpec]) -> list[ObjectSpec]:
    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], len(specs))
    return [
        ObjectSpec(color=color, body=spec.body, tail=spec.tail, hidden_tail=spec.hidden_tail)
        for color, spec in zip(colors, specs)
    ]


def _validate_specs(shape: tuple[int, int], specs: list[ObjectSpec], z_order: list[int]) -> tuple[np.ndarray, np.ndarray] | None:
    inp = _render_input(shape, specs, z_order)
    out = _render_output(shape, specs)

    body_overlaps = 0
    for i in range(len(specs)):
        for j in range(i + 1, len(specs)):
            if _boxes_overlap(specs[i].body, specs[j].body):
                body_overlaps += 1
    if body_overlaps == 0:
        return None

    for spec in specs:
        sides = _visible_side_cells(inp, spec.color, spec.body)
        if not all(sides[name] for name in ("top", "bottom", "left", "right")):
            return None
        inferred = _infer_body_bbox(inp, spec.color)
        if inferred != spec.body:
            return None

        completed = _completed_mask(inp, spec.color, inferred)
        detected = _detect_tail(completed, inferred)
        if spec.tail is None:
            if detected is not None:
                return None
        else:
            if detected != spec.tail:
                return None
            if spec.hidden_tail:
                tail_cells = _tail_cells(spec.body, spec.tail)
                visible_tail = [(r, c) for r, c in tail_cells if int(inp[r, c]) == spec.color]
                if len(visible_tail) < 2:
                    return None
                dr, dc = spec.tail.direction
                if dc > 0:
                    attach = (spec.tail.anchor, spec.body[3])
                elif dc < 0:
                    attach = (spec.tail.anchor, spec.body[1])
                elif dr > 0:
                    attach = (spec.body[2], spec.tail.anchor)
                else:
                    attach = (spec.body[0], spec.tail.anchor)
                if int(inp[attach]) == spec.color:
                    return None

    if not np.array_equal(main(inp), out):
        return None
    return inp, out


def _generate_scene() -> np.ndarray | None:
    rows = random.randint(*GRID_H_RANGE)
    cols = random.randint(*GRID_W_RANGE)
    shape = (rows, cols)

    hidden = _sample_hidden_slider(shape, used_areas=set())
    if hidden is None:
        return None
    used_areas = {hidden.area}

    occluder = _sample_occluder(shape, hidden, used_areas)
    if occluder is None:
        return None
    used_areas.add(occluder.area)

    specs = [hidden, occluder]
    slider_dest_boxes = [_shift_box(hidden.body, hidden.tail.direction, hidden.tail.length)]
    occupied_input = np.zeros(shape, dtype=bool)
    for spec in specs:
        for r, c in _object_cells(spec):
            occupied_input[r, c] = True

    total_objects = random.choice([3, 4])
    if total_objects == 3:
        extra_roles = [random.choice(["static", "slider"])]
    elif random.random() < 0.55:
        extra_roles = ["slider", "static"]
    else:
        extra_roles = ["static", "static"]
    random.shuffle(extra_roles)

    for role in extra_roles:
        spec = _sample_free_object(shape, occupied_input, slider_dest_boxes, used_areas, role)
        if spec is None:
            return None
        specs.append(spec)
        used_areas.add(spec.area)
        for r, c in _object_cells(spec):
            occupied_input[r, c] = True
        if spec.tail is not None:
            slider_dest_boxes.append(_shift_box(spec.body, spec.tail.direction, spec.tail.length))

    specs = _assign_colors(specs)
    z_order = list(range(len(specs)))
    hidden_idx = next(i for i, spec in enumerate(specs) if spec.hidden_tail)
    occ_idx = next(i for i, spec in enumerate(specs) if spec.tail is None and i != hidden_idx)
    z_order.remove(hidden_idx)
    z_order.remove(occ_idx)
    random.shuffle(z_order)
    z_order = [hidden_idx, *z_order, occ_idx]

    validated = _validate_specs(shape, specs, z_order)
    if validated is None:
        return None
    inp, _ = validated
    return inp


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each foreground-filled rectangle body, then slide any body whose newly exposed one-pixel tail gives a direction and distance.

    Input structure:
    - The grid is black background with 3-4 foreground colors, one object per color.
    - Each object has an axis-aligned solid rectangular body whose bbox contains no background, though parts of that body may be covered by other colors.
    - Some objects also have a one-pixel-wide straight tail extending from one side; the attachment can be hidden by overlap, so the tail may only become recognizable after body completion.

    Rule:
    - For each color, find the unique largest bbox with height and width both greater than 1 whose cells are all foreground and whose visible pixels touch all four sides; that bbox is the object's body.
    - Complete the object by treating every cell in that body bbox as its color, then check whether the completed object has a single straight one-pixel tail of length at least 2 extending from one side.
    - Objects with no tail stay in place; objects with a tail erase the tail and translate only the rectangular body by the tail's direction and length.
    - Draw all static bodies first from larger area to smaller area, then draw all moved bodies from larger area to smaller area, letting later draws overwrite earlier ones.

    Key insight:
    - A protrusion can only be identified as a tail after the occluded rectangular body is restored, because completion fixes the true body bbox and reconnects tails whose attachment cell was hidden.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert int(np.min(g)) >= 0

    colors = sorted(int(v) for v in np.unique(g) if int(v) != BG)
    objects: list[ObjectSpec] = []
    for color in colors:
        body = _infer_body_bbox(g, color)
        tail = _detect_tail(_completed_mask(g, color, body), body)
        objects.append(ObjectSpec(color=color, body=body, tail=tail))

    out = np.full_like(g, BG)
    statics = sorted((spec for spec in objects if spec.tail is None), key=lambda spec: (-spec.area, spec.color))
    sliders = sorted((spec for spec in objects if spec.tail is not None), key=lambda spec: (-spec.area, spec.color))

    for spec in statics:
        _draw_body(out, spec.body, spec.color)
    for spec in sliders:
        moved = _shift_box(spec.body, spec.tail.direction, spec.tail.length)
        _draw_body(out, moved, spec.color)
    return out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _generate_scene()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hidden-tail sliding scene")
