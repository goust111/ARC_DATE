import random
import numpy as np


BLACK = 0
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
AZURE = 8

VALUE_COLORS = (RED, GREEN, YELLOW)
GEN_RETRIES = 300
PLACEMENT_ATTEMPTS = 250


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _component_mask(points: list[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(points)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in points:
        mask[r - r0, c - c0] = True
    return mask, (r0, c0, r1, c1)


def _draw_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    sub = grid[top : top + h, left : left + w]
    sub[mask] = color


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    pts = np.argwhere(mask)
    return tuple((int(r), int(c)) for r, c in pts)


def _rotation_signature(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    return min(_mask_signature(np.rot90(mask, k)) for k in range(4))


def _find_legend_frame(grid: np.ndarray) -> tuple[int, int, int, int]:
    candidates = []
    for pts in _components(grid == AZURE):
        r0, c0, r1, c1 = _bbox(pts)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        border = {
            (r, c)
            for r in range(r0, r1 + 1)
            for c in range(c0, c1 + 1)
            if r in (r0, r1) or c in (c0, c1)
        }
        if len(pts) == len(border) and set(pts) == border:
            candidates.append((r0, c0, r1, c1))
    assert candidates, "legend frame not found"
    return min(candidates, key=lambda box: (box[0] + box[1], box[0], box[1]))


def _extract_dictionary(grid: np.ndarray) -> tuple[tuple[int, int, int, int], list[dict[str, object]]]:
    fr0, fc0, fr1, fc1 = _find_legend_frame(grid)
    inner = grid[fr0 + 1 : fr1, fc0 + 1 : fc1]
    items = []
    for pts in _components(inner != BLACK):
        mask, bbox = _component_mask(pts)
        color = int(inner[pts[0][0], pts[0][1]])
        items.append({"mask": mask, "bbox": bbox, "color": color})
    assert items, "legend entries not found"

    rows = []
    for item in sorted(items, key=lambda part: (part["bbox"][0], part["bbox"][1])):
        if not rows or item["bbox"][0] > rows[-1]["bottom"] + 1:
            rows.append({"parts": [item], "bottom": item["bbox"][2]})
        else:
            rows[-1]["parts"].append(item)
            rows[-1]["bottom"] = max(rows[-1]["bottom"], item["bbox"][2])

    entries = []
    for row in rows:
        parts = sorted(row["parts"], key=lambda part: part["bbox"][1])
        assert len(parts) == 2, "each legend row must have exactly two shapes"
        left, right = parts
        assert int(left["color"]) == GREY, "legend key must be grey"
        assert int(right["color"]) in VALUE_COLORS, "legend value must be red/green/yellow"
        assert left["bbox"][3] + 1 < right["bbox"][1], "legend shapes must be separated"
        entries.append(
            {
                "key_mask": np.asarray(left["mask"], dtype=bool),
                "value_mask": np.asarray(right["mask"], dtype=bool),
                "value_color": int(right["color"]),
            }
        )
    assert 2 <= len(entries) <= 3, "legend must contain 2-3 entries"
    return (fr0, fc0, fr1, fc1), entries


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each grey main-area shape that matches a legend key up to rotation
    with the equally rotated legend value shape.
    """
    g = np.asarray(grid, dtype=int)
    (fr0, fc0, fr1, fc1), entries = _extract_dictionary(g)

    outside = g == GREY
    outside[fr0 : fr1 + 1, fc0 : fc1 + 1] = False
    comps = _components(outside)

    out = g.copy()
    for pts in comps:
        mask, (r0, c0, _r1, _c1) = _component_mask(pts)
        match = None
        for entry in entries:
            key_mask = entry["key_mask"]
            for rot in range(4):
                if np.array_equal(mask, np.rot90(key_mask, rot)):
                    assert match is None, "shape matched multiple legend entries"
                    match = (entry, rot)
        if match is None:
            continue

        for r, c in pts:
            out[r, c] = BLACK

        entry, rot = match
        value_mask = np.rot90(entry["value_mask"], rot)
        vh, vw = value_mask.shape
        assert r0 + vh <= g.shape[0] and c0 + vw <= g.shape[1], "replacement would go out of bounds"
        target = out[r0 : r0 + vh, c0 : c0 + vw]
        assert np.all(target[value_mask] == BLACK), "replacement overlaps another object"
        target[value_mask] = int(entry["value_color"])

    return out


SHAPE_COORDS = (
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (0, 2), (1, 0)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 2)),
    ((0, 0), (0, 1), (0, 2), (0, 3), (1, 0)),
    ((0, 0), (0, 1), (0, 2), (0, 3), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 0), (1, 2)),
    ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1)),
    ((0, 0), (1, 0), (1, 1), (1, 2), (2, 1)),
    ((0, 1), (0, 2), (1, 0), (1, 1), (2, 0)),
)

SHAPE_MASKS = tuple(_component_mask(list(coords))[0] for coords in SHAPE_COORDS)
SHAPE_CLASSES = tuple(_rotation_signature(mask) for mask in SHAPE_MASKS)


def _claim_rect(occupied: np.ndarray, top: int, left: int, height: int, width: int, pad: int) -> None:
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(occupied.shape[0], top + height + pad)
    c1 = min(occupied.shape[1], left + width + pad)
    occupied[r0:r1, c0:c1] = True


def _can_place_rect(occupied: np.ndarray, top: int, left: int, height: int, width: int, pad: int) -> bool:
    if top < pad or left < pad:
        return False
    if top + height + pad > occupied.shape[0] or left + width + pad > occupied.shape[1]:
        return False
    r0 = top - pad
    c0 = left - pad
    r1 = top + height + pad
    c1 = left + width + pad
    return not occupied[r0:r1, c0:c1].any()


def _build_entries() -> list[dict[str, object]]:
    entry_count = random.choice((2, 3))
    left_ids = random.sample(range(len(SHAPE_MASKS)), entry_count)
    right_ids = []
    used_right = set()
    for left_id in left_ids:
        options = [
            idx
            for idx in range(len(SHAPE_MASKS))
            if SHAPE_CLASSES[idx] != SHAPE_CLASSES[left_id] and idx not in used_right
        ]
        if not options:
            options = [idx for idx in range(len(SHAPE_MASKS)) if SHAPE_CLASSES[idx] != SHAPE_CLASSES[left_id]]
        right_id = random.choice(options)
        right_ids.append(right_id)
        used_right.add(right_id)

    colors = random.sample(VALUE_COLORS, entry_count)
    entries = []
    for left_id, right_id, color in zip(left_ids, right_ids, colors):
        key_mask = np.rot90(SHAPE_MASKS[left_id], random.randrange(4))
        value_mask = np.rot90(SHAPE_MASKS[right_id], random.randrange(4))
        entries.append(
            {
                "key_mask": np.asarray(key_mask, dtype=bool),
                "value_mask": np.asarray(value_mask, dtype=bool),
                "value_color": int(color),
                "key_class": SHAPE_CLASSES[left_id],
            }
        )
    return entries


def _build_legend_patch(entries: list[dict[str, object]]) -> np.ndarray:
    left_pad = random.randint(1, 2)
    right_pad = random.randint(1, 2)
    top_pad = random.randint(1, 2)
    bottom_pad = random.randint(1, 2)
    middle_gap = random.randint(1, 3)
    row_gaps = [random.randint(1, 2) for _ in range(len(entries) - 1)]

    left_width = max(entry["key_mask"].shape[1] for entry in entries)
    right_width = max(entry["value_mask"].shape[1] for entry in entries)
    row_heights = [max(entry["key_mask"].shape[0], entry["value_mask"].shape[0]) for entry in entries]

    inner_h = top_pad + sum(row_heights) + sum(row_gaps) + bottom_pad
    inner_w = left_pad + left_width + middle_gap + right_width + right_pad
    patch = np.full((inner_h + 2, inner_w + 2), BLACK, dtype=int)
    patch[0, :] = AZURE
    patch[-1, :] = AZURE
    patch[:, 0] = AZURE
    patch[:, -1] = AZURE

    top = 1 + top_pad
    left_x = 1 + left_pad
    right_x = 1 + left_pad + left_width + middle_gap
    for idx, entry in enumerate(entries):
        _draw_mask(patch, entry["key_mask"], top, left_x, GREY)
        _draw_mask(patch, entry["value_mask"], top, right_x, int(entry["value_color"]))
        top += row_heights[idx]
        if idx < len(row_gaps):
            top += row_gaps[idx]
    return patch


def _place_objects(
    grid: np.ndarray,
    occupied: np.ndarray,
    entries: list[dict[str, object]],
) -> bool:
    specs = []
    for entry_idx, entry in enumerate(entries):
        rot = random.randrange(4)
        key_mask = np.rot90(entry["key_mask"], rot)
        value_mask = np.rot90(entry["value_mask"], rot)
        specs.append(
            {
                "kind": "match",
                "entry_idx": entry_idx,
                "mask": key_mask,
                "value_mask": value_mask,
                "reserve_h": max(key_mask.shape[0], value_mask.shape[0]),
                "reserve_w": max(key_mask.shape[1], value_mask.shape[1]),
            }
        )

    extra_matches = random.randint(0, 2)
    for _ in range(extra_matches):
        entry_idx = random.randrange(len(entries))
        entry = entries[entry_idx]
        rot = random.randrange(4)
        key_mask = np.rot90(entry["key_mask"], rot)
        value_mask = np.rot90(entry["value_mask"], rot)
        specs.append(
            {
                "kind": "match",
                "entry_idx": entry_idx,
                "mask": key_mask,
                "value_mask": value_mask,
                "reserve_h": max(key_mask.shape[0], value_mask.shape[0]),
                "reserve_w": max(key_mask.shape[1], value_mask.shape[1]),
            }
        )

    distractor_count = random.randint(1, 3)
    used_classes = {entry["key_class"] for entry in entries}
    distractor_ids = [idx for idx, cls in enumerate(SHAPE_CLASSES) if cls not in used_classes]
    assert distractor_ids, "need distractor shapes distinct from legend keys"
    random.shuffle(distractor_ids)
    while len(distractor_ids) < distractor_count:
        distractor_ids.append(random.choice(distractor_ids))
    for idx in distractor_ids[:distractor_count]:
        mask = np.rot90(SHAPE_MASKS[idx], random.randrange(4))
        specs.append(
            {
                "kind": "distractor",
                "mask": mask,
                "reserve_h": mask.shape[0],
                "reserve_w": mask.shape[1],
            }
        )

    if all(spec["kind"] != "match" or np.array_equal(spec["mask"], entries[spec["entry_idx"]]["key_mask"]) for spec in specs):
        match_specs = [spec for spec in specs if spec["kind"] == "match"]
        if match_specs:
            chosen = random.choice(match_specs)
            extra_rot = random.randint(1, 3)
            chosen["mask"] = np.rot90(entries[chosen["entry_idx"]]["key_mask"], extra_rot)
            chosen["value_mask"] = np.rot90(entries[chosen["entry_idx"]]["value_mask"], extra_rot)
            chosen["reserve_h"] = max(chosen["mask"].shape[0], chosen["value_mask"].shape[0])
            chosen["reserve_w"] = max(chosen["mask"].shape[1], chosen["value_mask"].shape[1])

    random.shuffle(specs)
    specs.sort(key=lambda spec: spec["reserve_h"] * spec["reserve_w"], reverse=True)

    for spec in specs:
        placed = False
        mask = spec["mask"]
        for _ in range(PLACEMENT_ATTEMPTS):
            top = random.randint(1, grid.shape[0] - spec["reserve_h"] - 1)
            left = random.randint(1, grid.shape[1] - spec["reserve_w"] - 1)
            if not _can_place_rect(occupied, top, left, spec["reserve_h"], spec["reserve_w"], pad=1):
                continue
            _claim_rect(occupied, top, left, spec["reserve_h"], spec["reserve_w"], pad=1)
            _draw_mask(grid, mask, top, left, GREY)
            placed = True
            break
        if not placed:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    entries = _build_entries()
    legend = _build_legend_patch(entries)
    min_h = max(20, legend.shape[0] + 8)
    min_w = max(22, legend.shape[1] + 12)
    max_h = max(min_h, 28)
    max_w = max(min_w, 30)
    if min_h > max_h or min_w > max_w:
        return None

    h = random.randint(min_h, max_h)
    w = random.randint(min_w, max_w)
    grid = np.full((h, w), BLACK, dtype=int)

    legend_top = random.randint(1, 2)
    legend_left = random.randint(1, 2)
    patch = grid[legend_top : legend_top + legend.shape[0], legend_left : legend_left + legend.shape[1]]
    patch[:] = legend

    occupied = np.zeros_like(grid, dtype=bool)
    _claim_rect(occupied, legend_top, legend_left, legend.shape[0], legend.shape[1], pad=2)
    if not _place_objects(grid, occupied, entries):
        return None

    try:
        frame_bbox, parsed_entries = _extract_dictionary(grid)
    except AssertionError:
        return None
    if len(parsed_entries) != len(entries):
        return None

    out = main(grid)
    fr0, fc0, fr1, fc1 = frame_bbox
    outside = np.ones_like(grid, dtype=bool)
    outside[fr0 : fr1 + 1, fc0 : fc1 + 1] = False
    if not np.any((grid != out) & outside):
        return None
    if not np.any((grid == GREY) & (out == GREY) & outside):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
