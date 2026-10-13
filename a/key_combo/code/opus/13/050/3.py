from __future__ import annotations

import random

import numpy as np


BG = 4
FG = 1
LEFT_COLOR = 2
RIGHT_COLOR = 3

HEIGHT_RANGE = (4, 6)
WIDTH_RANGE = (3, 5)
OBJECT_COUNT_RANGE = (3, 4)
GEN_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def find_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(pts, dtype=int))
    return components


def is_contiguous(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    return len(find_components(mask)) == 1


def paste_mask(dst: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int, bg: int) -> None:
    h, w = mask.shape
    patch = dst[r0 : r0 + h, c0 : c0 + w]
    assert patch.shape == mask.shape
    assert np.all(patch[mask] == bg)
    patch[mask] = int(color)


def has_flanked_gap(line: np.ndarray) -> bool:
    for idx, value in enumerate(line):
        if not value and line[:idx].any() and line[idx + 1 :].any():
            return True
    return False


def has_border_notch(mask: np.ndarray) -> bool:
    return any(
        (
            has_flanked_gap(mask[0, :]),
            has_flanked_gap(mask[-1, :]),
            has_flanked_gap(mask[:, 0]),
            has_flanked_gap(mask[:, -1]),
        )
    )


def shift_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    shifted = np.zeros((h, w + 1), dtype=bool)
    for r in range(h):
        delta = 1 if (h - 1 - r) % 2 == 1 else 0
        cols = np.flatnonzero(mask[r])
        shifted[r, cols + delta] = True
    coords = np.argwhere(shifted)
    r0, c0, r1, c1 = bbox_coords(coords)
    return shifted[r0 : r1 + 1, c0 : c1 + 1]


def mask_from_intervals(intervals: list[tuple[int, int]]) -> np.ndarray:
    width = max(c1 for _, c1 in intervals) + 1
    mask = np.zeros((len(intervals), width), dtype=bool)
    for r, (c0, c1) in enumerate(intervals):
        mask[r, c0 : c1 + 1] = True
    return mask


def analyze_mask(mask: np.ndarray) -> dict:
    shifted = shift_mask(mask)
    return {
        "mask": mask,
        "bbox_h": int(mask.shape[0]),
        "bbox_w": int(mask.shape[1]),
        "shifted_mask": shifted,
        "shifted_w": int(shifted.shape[1]),
        "has_notch_before": bool(has_border_notch(mask)),
        "has_notch_after": bool(has_border_notch(shifted)),
    }


def analyze_component(grid: np.ndarray, coords: np.ndarray) -> dict:
    r0, c0, r1, c1 = bbox_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    info = analyze_mask(mask)
    info["r0"] = int(r0)
    info["c0"] = int(c0)
    info["bbox"] = (int(r0), int(c0), int(r1), int(c1))
    return info


def render_output_from_objects(shape: tuple[int, int], objects: list[dict], bg: int, use_shifted_notch: bool) -> np.ndarray:
    h, w = shape
    out = np.full((h, w), bg, dtype=int)

    annotated = []
    for obj in objects:
        go_right = obj["has_notch_after"] if use_shifted_notch else obj["has_notch_before"]
        annotated.append({**obj, "go_right": bool(go_right)})

    left_group = sorted(
        [obj for obj in annotated if not obj["go_right"]],
        key=lambda item: (int(item["r0"]), int(item["c0"])),
    )
    right_group = sorted(
        [obj for obj in annotated if obj["go_right"]],
        key=lambda item: (int(item["r0"]), int(item["c0"])),
    )

    left_width = sum(int(obj["shifted_mask"].shape[1]) for obj in left_group)
    right_width = sum(int(obj["shifted_mask"].shape[1]) for obj in right_group)
    assert left_width + right_width <= w

    cursor = 0
    for obj in left_group:
        paste_mask(out, obj["shifted_mask"], int(obj["r0"]), cursor, LEFT_COLOR, bg)
        cursor += int(obj["shifted_mask"].shape[1])

    cursor = w
    for obj in right_group:
        width = int(obj["shifted_mask"].shape[1])
        cursor -= width
        paste_mask(out, obj["shifted_mask"], int(obj["r0"]), cursor, RIGHT_COLOR, bg)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Alternatingly shift each object by row, then use the shifted shape's border notch to decide slide direction and recolor.

    Input structure:
    - The grid has a yellow background and one foreground color forming 3-4 separated 4-connected objects.
    - Each object is an irregular blue blob whose bounding box stays away from the grid edges.
    - Objects may share row ranges, so output packing depends on their original top-row order.

    Rule:
    - For each object, keep the bbox bottom row fixed and shift every row at odd distance above it one cell right.
    - Recompute the shifted object's bbox and test its four bbox borders for a flanked background gap.
    - Shifted objects with a border notch slide to the right side and become green; shifted objects without one slide to the left side and become red.
    - Rebuild the output on a blank yellow grid, packing same-side objects from the edge inward in ascending order of original top row.

    Key insight:
    - The decisive notch is measured after the per-row alternating shift, because that shift can both create and remove bbox-border notches.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    fg_colors = sorted(set(map(int, g.flatten())) - {bg})
    assert len(fg_colors) == 1

    components = find_components(g != bg)
    assert 3 <= len(components) <= 4
    objects = [analyze_component(g, coords) for coords in components]
    return render_output_from_objects(g.shape, objects, bg, use_shifted_notch=True)


def make_flip_to_notch_mask(h: int, w: int) -> np.ndarray:
    k = random.randint(1, min(w - 1, h - 1))
    start_width = w - k
    inc_rows = sorted(random.sample(range(1, h), k))
    widths = []
    cur = start_width
    inc_idx = 0
    for r in range(h):
        while inc_idx < len(inc_rows) and inc_rows[inc_idx] == r:
            cur += 1
            inc_idx += 1
        widths.append(cur)
    intervals = [(0, width - 1) for width in widths]
    mask = mask_from_intervals(intervals)
    info = analyze_mask(mask)
    assert not info["has_notch_before"] and info["has_notch_after"]
    return mask


def make_flip_to_closed_mask(h: int, w: int) -> np.ndarray:
    intervals = []
    run_width = w - 1
    for r in range(h):
        dist = h - 1 - r
        start = 0 if dist % 2 == 1 else 1
        intervals.append((start, start + run_width - 1))
    mask = mask_from_intervals(intervals)
    info = analyze_mask(mask)
    assert info["has_notch_before"] and not info["has_notch_after"]
    return mask


def sample_object_mask(kind: str, n_objects: int) -> np.ndarray:
    for _ in range(80):
        h = random.randint(*HEIGHT_RANGE)
        w_hi = WIDTH_RANGE[1] if n_objects == 3 else 4
        w = random.randint(WIDTH_RANGE[0], w_hi)
        try:
            mask = make_flip_to_notch_mask(h, w) if kind == "flip_to_notch" else make_flip_to_closed_mask(h, w)
        except AssertionError:
            continue
        if is_contiguous(mask):
            return mask
    raise RuntimeError(f"failed to sample object mask for {kind}")


def choose_slot_layout(objects: list[dict]) -> tuple[int, int, list[tuple[int, int]]]:
    slots = random.sample(range(4), len(objects))
    pairs = list(zip(objects, slots))
    slot_to_obj = {slot: obj for obj, slot in pairs}

    top_slots = [slot for slot in slots if slot < 2]
    bottom_slots = [slot for slot in slots if slot >= 2]
    left_slots = [slot for slot in slots if slot % 2 == 0]
    right_slots = [slot for slot in slots if slot % 2 == 1]

    top_band_h = max(int(slot_to_obj[slot]["bbox_h"]) for slot in top_slots)
    bottom_band_h = max(int(slot_to_obj[slot]["bbox_h"]) for slot in bottom_slots)
    left_band_w = max(int(slot_to_obj[slot]["bbox_w"]) for slot in left_slots)
    right_band_w = max(int(slot_to_obj[slot]["bbox_w"]) for slot in right_slots)

    band_slack = 1
    margin = 1
    row_gap = random.randint(1, 2)
    top_start = margin
    bottom_start = margin + top_band_h + band_slack + row_gap
    h = bottom_start + bottom_band_h + band_slack + margin

    output_width = sum(int(obj["shifted_w"]) for obj in objects)
    input_width = left_band_w + right_band_w + 2 * margin + 2 * band_slack + 2
    w = max(15, output_width + random.randint(0, 2), input_width)

    left_start = margin
    right_start = w - margin - right_band_w - band_slack
    assert right_start - (left_start + left_band_w + band_slack) >= 1

    top_offsets = [0, 1][: len(top_slots)]
    bottom_offsets = [0, 1][: len(bottom_slots)]
    left_offsets = [0, 1][: len(left_slots)]
    right_offsets = [0, 1][: len(right_slots)]
    random.shuffle(top_offsets)
    random.shuffle(bottom_offsets)
    random.shuffle(left_offsets)
    random.shuffle(right_offsets)

    row_offset = {}
    for slot in sorted(top_slots):
        row_offset[slot] = top_offsets.pop()
    for slot in sorted(bottom_slots):
        row_offset[slot] = bottom_offsets.pop()

    col_offset = {}
    for slot in sorted(left_slots):
        col_offset[slot] = left_offsets.pop()
    for slot in sorted(right_slots):
        col_offset[slot] = right_offsets.pop()

    placements_by_slot = {}
    for slot in slots:
        r0 = top_start if slot < 2 else bottom_start
        c0 = left_start if slot % 2 == 0 else right_start
        placements_by_slot[slot] = (r0 + row_offset[slot], c0 + col_offset[slot])

    placements = [placements_by_slot[slot] for _obj, slot in pairs]
    return h, w, placements


def bboxes_separate(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0


def build_input(objects: list[dict]) -> np.ndarray:
    h, w, placements = choose_slot_layout(objects)
    grid = np.full((h, w), BG, dtype=int)

    placed_boxes = []
    for obj, (r0, c0) in zip(objects, placements):
        mask = obj["mask"]
        box = (r0, c0, r0 + mask.shape[0] - 1, c0 + mask.shape[1] - 1)
        assert box[0] >= 1 and box[1] >= 1 and box[2] < h - 1 and box[3] < w - 1
        for other in placed_boxes:
            assert bboxes_separate(box, other, pad=1)
        grid[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]][mask] = FG
        placed_boxes.append(box)
        obj["r0"] = int(r0)
        obj["c0"] = int(c0)
        obj["bbox"] = box

    return grid


def validate_semantics(inp: np.ndarray) -> None:
    comps = find_components(inp == FG)
    assert 3 <= len(comps) <= 4
    objects = [analyze_component(inp, coords) for coords in comps]
    assert all(WIDTH_RANGE[0] <= obj["bbox_w"] <= WIDTH_RANGE[1] for obj in objects)
    assert all(HEIGHT_RANGE[0] <= obj["bbox_h"] <= HEIGHT_RANGE[1] for obj in objects)
    assert all(np.any(obj["mask"].sum(axis=1) < obj["bbox_w"]) for obj in objects)

    flip_to_notch = sum((not obj["has_notch_before"]) and obj["has_notch_after"] for obj in objects)
    flip_to_closed = sum(obj["has_notch_before"] and (not obj["has_notch_after"]) for obj in objects)
    assert flip_to_notch >= 1
    assert flip_to_closed >= 1

    boxes = [obj["bbox"] for obj in objects]
    for i, box in enumerate(boxes):
        assert box[0] >= 1 and box[1] >= 1 and box[2] < inp.shape[0] - 1 and box[3] < inp.shape[1] - 1
        for other in boxes[i + 1 :]:
            assert bboxes_separate(box, other, pad=1)

    out = main(inp)
    assert np.any(out == LEFT_COLOR)
    assert np.any(out == RIGHT_COLOR)

    naive = render_output_from_objects(inp.shape, objects, BG, use_shifted_notch=False)
    assert not np.array_equal(out, naive)


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        n_objects = random.randint(*OBJECT_COUNT_RANGE)
        kinds = ["flip_to_notch", "flip_to_closed"]
        while len(kinds) < n_objects:
            kinds.append(random.choice(["flip_to_notch", "flip_to_closed"]))
        random.shuffle(kinds)

        objects = []
        for kind in kinds:
            mask = sample_object_mask(kind, n_objects)
            info = analyze_mask(mask)
            objects.append(info)

        try:
            inp = build_input(objects)
            validate_semantics(inp)
        except AssertionError:
            continue
        return inp
    raise RuntimeError("failed to generate valid input")
