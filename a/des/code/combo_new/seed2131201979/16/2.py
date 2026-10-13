import random
from collections import Counter

import numpy as np


NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRECTION_VECTORS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
PALETTE = list(range(10))

GRID_H_RANGE = (18, 28)
GRID_W_RANGE = (18, 28)
OBJECT_COUNT_CHOICES = [1, 2, 3]
OBJECT_COUNT_WEIGHTS = [2, 3, 1]
HOLE_COUNT_WEIGHTS = {1: 3, 2: 2}
TOTAL_HOLES_MAX = 4
OBJECT_BOX_SINGLE = (7, 11)
OBJECT_BOX_MULTI = (9, 13)
MAX_GENERATE_ATTEMPTS = 400


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each enclosed hole to an external stamp shape and spray that stamp color outward from the facing side.

    Input structure:
    - The background is the dominant color.
    - Some monochrome foreground components are larger hollow objects whose enclosed
      background holes are fully surrounded by the object pixels.
    - Separate smaller monochrome foreground blobs act as stamps; each stamp's
      normalized shape matches exactly one hole shape.

    Rule:
    - For every hole inside every hollow object, find the stamp component with the
      same normalized shape and read that stamp's color.
    - Compare the hole center to the hollow object's bounding-box center and keep
      the main offset direction: left, right, up, or down.
    - Project that hole onto the corresponding outer side of the object and emit
      straight rays of the matched stamp color from those side pixels outward until
      the grid edge or another original foreground pixel is reached.

    Key insight:
    - The hole shape selects the ray color, while the hole position inside the
      object selects which side becomes the nozzle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _background_color(g)
    objects, stamps = _scene_parts(g, bg)
    assert objects, "expected at least one hollow object"
    assert stamps, "expected at least one stamp component"

    stamp_colors: dict[tuple[tuple[int, int], ...], int] = {}
    for stamp in stamps:
        key = _normalize_points(stamp["coords"])
        assert key not in stamp_colors, "stamp shapes must be unique"
        stamp_colors[key] = int(stamp["color"])

    occupied = g != bg
    out = g.copy()
    for obj in objects:
        for hole in obj["holes"]:
            key = _normalize_points(hole)
            assert key in stamp_colors, "every hole must match exactly one stamp"
            color = int(stamp_colors[key])
            direction = _hole_direction(hole, obj["bbox"])
            for sr, sc in _ray_sources(obj, hole, direction):
                dr, dc = DIRECTION_VECTORS[direction]
                rr = int(sr + dr)
                cc = int(sc + dc)
                while 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1] and not bool(occupied[rr, cc]):
                    out[rr, cc] = color
                    rr += dr
                    cc += dc

    return out


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for dr, dc in NEIGHBORS4:
                    nr = int(cr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize_points(points: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0, c0, _r1, _c1 = _bbox(points)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in points))


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    return mask[np.ix_(rows, cols)]


def _mask_to_points(mask: np.ndarray, top: int = 0, left: int = 0) -> list[tuple[int, int]]:
    rr, cc = np.where(mask)
    return [(int(top + r), int(left + c)) for r, c in zip(rr, cc, strict=False)]


def _scene_parts(grid: np.ndarray, bg: int) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    objects: list[dict[str, object]] = []
    stamps: list[dict[str, object]] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != int(bg)):
        for coords in _components4(grid == color):
            holes = _holes_in_component(grid, bg, coords)
            r0, c0, r1, c1 = _bbox(coords)
            row_to_cols: dict[int, list[int]] = {}
            col_to_rows: dict[int, list[int]] = {}
            for r, c in coords:
                row_to_cols.setdefault(int(r), []).append(int(c))
                col_to_rows.setdefault(int(c), []).append(int(r))
            for vals in row_to_cols.values():
                vals.sort()
            for vals in col_to_rows.values():
                vals.sort()
            record = {
                "color": int(color),
                "coords": [(int(r), int(c)) for r, c in coords],
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "row_to_cols": row_to_cols,
                "col_to_rows": col_to_rows,
                "holes": holes,
            }
            if holes:
                objects.append(record)
            else:
                stamps.append(record)
    return objects, stamps


def _holes_in_component(grid: np.ndarray, bg: int, coords: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    r0, c0, r1, c1 = _bbox(coords)
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    obj_mask = np.zeros_like(patch, dtype=bool)
    for r, c in coords:
        obj_mask[int(r - r0), int(c - c0)] = True

    holes: list[list[tuple[int, int]]] = []
    for comp in _components4(patch == bg):
        if any(r == 0 or c == 0 or r == patch.shape[0] - 1 or c == patch.shape[1] - 1 for r, c in comp):
            continue
        touches_object = False
        touches_foreign = False
        for r, c in comp:
            for dr, dc in NEIGHBORS4:
                nr = int(r + dr)
                nc = int(c + dc)
                if not (0 <= nr < patch.shape[0] and 0 <= nc < patch.shape[1]):
                    continue
                if bool(obj_mask[nr, nc]):
                    touches_object = True
                elif int(patch[nr, nc]) != int(bg):
                    touches_foreign = True
        if touches_object and not touches_foreign:
            holes.append([(int(r0 + r), int(c0 + c)) for r, c in comp])
    return holes


def _hole_direction(
    hole_points: list[tuple[int, int]], obj_bbox: tuple[int, int, int, int]
) -> str:
    r0, c0, r1, c1 = obj_bbox
    obj_center_r = (r0 + r1) / 2.0
    obj_center_c = (c0 + c1) / 2.0
    hole_center_r = float(sum(r for r, _ in hole_points)) / len(hole_points)
    hole_center_c = float(sum(c for _, c in hole_points)) / len(hole_points)
    dr = hole_center_r - obj_center_r
    dc = hole_center_c - obj_center_c
    if abs(dc) > abs(dr):
        return "left" if dc < 0 else "right"
    if abs(dr) > abs(dc):
        return "up" if dr < 0 else "down"
    if dc != 0:
        return "left" if dc < 0 else "right"
    return "up" if dr < 0 else "down"


def _ray_sources(
    obj: dict[str, object], hole_points: list[tuple[int, int]], direction: str
) -> list[tuple[int, int]]:
    row_to_cols = obj["row_to_cols"]
    col_to_rows = obj["col_to_rows"]
    if direction in {"left", "right"}:
        rows = sorted({int(r) for r, _ in hole_points})
        if direction == "left":
            return [(int(r), int(row_to_cols[int(r)][0])) for r in rows]
        return [(int(r), int(row_to_cols[int(r)][-1])) for r in rows]
    cols = sorted({int(c) for _, c in hole_points})
    if direction == "up":
        return [(int(col_to_rows[int(c)][0]), int(c)) for c in cols]
    return [(int(col_to_rows[int(c)][-1]), int(c)) for c in cols]


def _orientations(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for k in range(4):
        rotated = np.rot90(mask, k)
        for candidate in (rotated, np.fliplr(rotated)):
            trimmed = _trim_mask(np.asarray(candidate, dtype=bool))
            key = _normalize_points(_mask_to_points(trimmed))
            if key in seen:
                continue
            seen.add(key)
            variants.append(trimmed.copy())
    return variants


def _shape_library() -> list[np.ndarray]:
    raw_shapes = [
        [[1]],
        [[1, 1]],
        [[1], [1]],
        [[1, 1, 1]],
        [[1], [1], [1]],
        [[1, 0], [1, 1]],
        [[0, 1], [1, 1]],
        [[1, 1], [1, 0]],
        [[1, 1], [0, 1]],
        [[1, 1], [1, 1]],
        [[1, 1, 1], [0, 1, 0]],
        [[1, 0], [1, 1], [1, 0]],
        [[1, 1, 0], [0, 1, 1]],
        [[0, 1, 1], [1, 1, 0]],
        [[1, 1, 1], [1, 0, 0]],
        [[1, 1, 1], [0, 0, 1]],
        [[1, 0, 0], [1, 1, 1]],
        [[0, 0, 1], [1, 1, 1]],
        [[1, 1, 1], [1, 1, 0]],
        [[1, 1, 1], [0, 1, 1]],
        [[1, 1, 0], [1, 0, 0], [1, 1, 1]],
        [[0, 1, 1], [0, 0, 1], [1, 1, 1]],
    ]

    seen: set[tuple[tuple[int, int], ...]] = set()
    shapes: list[np.ndarray] = []
    for raw in raw_shapes:
        mask = np.array(raw, dtype=bool)
        if len(_components4(mask)) != 1:
            continue
        for oriented in _orientations(mask):
            key = _normalize_points(_mask_to_points(oriented))
            if key in seen:
                continue
            seen.add(key)
            shapes.append(oriented.copy())
    return shapes


SHAPE_LIBRARY = _shape_library()


def _sample_hole_counts(object_count: int) -> list[int]:
    counts: list[int] = []
    used = 0
    for index in range(object_count):
        remaining_objects = object_count - index - 1
        max_for_this = min(2, TOTAL_HOLES_MAX - used - remaining_objects)
        choices = [count for count in (1, 2) if count <= max_for_this]
        weights = [HOLE_COUNT_WEIGHTS[count] for count in choices]
        picked = random.choices(choices, weights=weights, k=1)[0]
        counts.append(int(picked))
        used += int(picked)
    return counts


def _build_object_patch(
    bg: int, obj_color: int, hole_specs: list[dict[str, object]]
) -> dict[str, object] | None:
    size_range = OBJECT_BOX_SINGLE if len(hole_specs) == 1 else OBJECT_BOX_MULTI
    for _ in range(120):
        outer_h = random.randint(size_range[0], size_range[1])
        outer_w = random.randint(size_range[0], size_range[1])
        if any(spec["shape"].shape[0] > outer_h - 2 or spec["shape"].shape[1] > outer_w - 2 for spec in hole_specs):
            continue

        blocked = np.zeros((outer_h, outer_w), dtype=bool)
        placements: list[list[tuple[int, int]]] = []
        order = list(range(len(hole_specs)))
        random.shuffle(order)
        ok = True

        for spec_index in order:
            spec = hole_specs[spec_index]
            shape = spec["shape"]
            mh, mw = shape.shape
            rr, cc = np.where(shape)
            candidates: list[list[tuple[int, int]]] = []
            for top in range(1, outer_h - mh):
                for left in range(1, outer_w - mw):
                    cells = [(int(top + r), int(left + c)) for r, c in zip(rr, cc, strict=False)]
                    if any(bool(blocked[r, c]) for r, c in cells):
                        continue
                    if _hole_direction(cells, (0, 0, outer_h - 1, outer_w - 1)) != spec["direction"]:
                        continue
                    candidates.append(cells)
            if not candidates:
                ok = False
                break

            chosen = random.choice(candidates)
            placements.append(chosen)
            for r, c in chosen:
                for nr in range(max(0, r - 1), min(outer_h, r + 2)):
                    for nc in range(max(0, c - 1), min(outer_w, c + 2)):
                        blocked[nr, nc] = True

        if not ok:
            continue

        # Reorder placements back to the original spec order.
        ordered_placements: list[list[tuple[int, int]]] = [None] * len(hole_specs)  # type: ignore[list-item]
        for idx, spec_index in enumerate(order):
            ordered_placements[spec_index] = placements[idx]

        object_mask = np.ones((outer_h, outer_w), dtype=bool)
        for cells in ordered_placements:
            for r, c in cells:
                object_mask[r, c] = False
        if len(_components4(object_mask)) != 1:
            continue

        recovered = _closed_holes_in_mask(object_mask)
        if Counter(_normalize_points(hole) for hole in recovered) != Counter(
            _normalize_points(cells) for cells in ordered_placements
        ):
            continue

        patch = np.full((outer_h, outer_w), int(bg), dtype=int)
        patch[object_mask] = int(obj_color)
        holes = []
        for spec, cells in zip(hole_specs, ordered_placements, strict=False):
            holes.append(
                {
                    "coords": [(int(r), int(c)) for r, c in cells],
                    "direction": str(spec["direction"]),
                    "shape_key": _normalize_points(cells),
                    "stamp_color": int(spec["stamp_color"]),
                }
            )
        return {
            "patch": patch,
            "object_mask": object_mask,
            "holes": holes,
            "color": int(obj_color),
        }
    return None


def _closed_holes_in_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    holes: list[list[tuple[int, int]]] = []
    for comp in _components4(~mask):
        if any(r == 0 or c == 0 or r == mask.shape[0] - 1 or c == mask.shape[1] - 1 for r, c in comp):
            continue
        holes.append([(int(r), int(c)) for r, c in comp])
    return holes


def _region_clear(occupied: np.ndarray, top: int, left: int, height: int, width: int, pad: int = 1) -> bool:
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(occupied.shape[0], top + height + pad)
    c1 = min(occupied.shape[1], left + width + pad)
    return not bool(occupied[r0:r1, c0:c1].any())


def _place_bboxes(grid_shape: tuple[int, int], records: list[dict[str, object]]) -> list[tuple[int, int]] | None:
    occupied = np.zeros(grid_shape, dtype=bool)
    placements: list[tuple[int, int]] = []
    order = sorted(range(len(records)), key=lambda i: -int(records[i]["patch"].shape[0] * records[i]["patch"].shape[1]))
    slots: list[tuple[int, int] | None] = [None] * len(records)
    for index in order:
        patch = records[index]["patch"]
        ph, pw = patch.shape
        candidates: list[tuple[int, int]] = []
        for top in range(1, grid_shape[0] - ph):
            for left in range(1, grid_shape[1] - pw):
                if _region_clear(occupied, top, left, ph, pw, pad=1):
                    candidates.append((int(top), int(left)))
        if not candidates:
            return None
        top, left = random.choice(candidates)
        occupied[top : top + ph, left : left + pw] = True
        slots[index] = (int(top), int(left))
    return [slot for slot in slots if slot is not None]


def _object_record_from_patch(
    patch_record: dict[str, object], top: int, left: int
) -> dict[str, object]:
    patch = patch_record["patch"]
    object_mask = patch_record["object_mask"]
    coords = _mask_to_points(object_mask, top=top, left=left)
    row_to_cols: dict[int, list[int]] = {}
    col_to_rows: dict[int, list[int]] = {}
    for r, c in coords:
        row_to_cols.setdefault(int(r), []).append(int(c))
        col_to_rows.setdefault(int(c), []).append(int(r))
    for vals in row_to_cols.values():
        vals.sort()
    for vals in col_to_rows.values():
        vals.sort()
    holes = []
    for hole in patch_record["holes"]:
        holes.append(
            {
                "coords": [(int(top + r), int(left + c)) for r, c in hole["coords"]],
                "direction": str(hole["direction"]),
                "shape_key": hole["shape_key"],
                "stamp_color": int(hole["stamp_color"]),
            }
        )
    return {
        "patch": patch,
        "top": int(top),
        "left": int(left),
        "coords": coords,
        "bbox": (int(top), int(left), int(top + patch.shape[0] - 1), int(left + patch.shape[1] - 1)),
        "row_to_cols": row_to_cols,
        "col_to_rows": col_to_rows,
        "holes": holes,
        "color": int(patch_record["color"]),
    }


def _trace_ray_cells(
    obj: dict[str, object], hole: dict[str, object], occupied: np.ndarray
) -> list[tuple[int, int]] | None:
    direction = str(hole["direction"])
    dr, dc = DIRECTION_VECTORS[direction]
    cells: list[tuple[int, int]] = []
    for sr, sc in _ray_sources(obj, hole["coords"], direction):
        rr = int(sr + dr)
        cc = int(sc + dc)
        started = False
        while 0 <= rr < occupied.shape[0] and 0 <= cc < occupied.shape[1]:
            if bool(occupied[rr, cc]):
                return None
            cells.append((int(rr), int(cc)))
            started = True
            rr += dr
            cc += dc
        if not started:
            return None
    return cells


def _place_stamp(
    occupied: np.ndarray, forbidden: np.ndarray, shape: np.ndarray
) -> tuple[int, int] | None:
    sh, sw = shape.shape
    candidates: list[tuple[int, int]] = []
    rr, cc = np.where(shape)
    for top in range(1, occupied.shape[0] - sh):
        for left in range(1, occupied.shape[1] - sw):
            if not _region_clear(occupied, top, left, sh, sw, pad=1):
                continue
            if any(bool(forbidden[top + r, left + c]) for r, c in zip(rr, cc, strict=False)):
                continue
            candidates.append((int(top), int(left)))
    if not candidates:
        return None
    return random.choice(candidates)


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(GRID_H_RANGE[0], GRID_H_RANGE[1])
    width = random.randint(GRID_W_RANGE[0], GRID_W_RANGE[1])
    bg = random.choice([0, 1, 7])
    object_count = random.choices(OBJECT_COUNT_CHOICES, weights=OBJECT_COUNT_WEIGHTS, k=1)[0]
    hole_counts = _sample_hole_counts(object_count)
    total_holes = sum(hole_counts)

    available_colors = [color for color in PALETTE if color != bg]
    if len(available_colors) < object_count + total_holes:
        return None
    chosen_colors = random.sample(available_colors, object_count + total_holes)
    object_colors = chosen_colors[:object_count]
    stamp_colors = chosen_colors[object_count:]
    hole_shapes = random.sample(SHAPE_LIBRARY, total_holes)

    patch_records: list[dict[str, object]] = []
    hole_specs_flat: list[dict[str, object]] = []
    hole_index = 0
    for obj_color, hole_count in zip(object_colors, hole_counts, strict=False):
        dirs = random.sample(list(DIRECTION_VECTORS), hole_count)
        hole_specs: list[dict[str, object]] = []
        for direction in dirs:
            spec = {
                "shape": hole_shapes[hole_index].copy(),
                "direction": direction,
                "stamp_color": int(stamp_colors[hole_index]),
            }
            hole_specs.append(spec)
            hole_specs_flat.append(spec)
            hole_index += 1
        patch_record = _build_object_patch(bg, int(obj_color), hole_specs)
        if patch_record is None:
            return None
        patch_records.append(patch_record)

    placements = _place_bboxes((height, width), patch_records)
    if placements is None:
        return None

    grid = np.full((height, width), int(bg), dtype=int)
    occupied = np.zeros((height, width), dtype=bool)
    objects: list[dict[str, object]] = []
    for patch_record, (top, left) in zip(patch_records, placements, strict=False):
        patch = patch_record["patch"]
        ph, pw = patch.shape
        grid[top : top + ph, left : left + pw] = patch
        occupied[top : top + ph, left : left + pw] |= patch != bg
        objects.append(_object_record_from_patch(patch_record, top, left))

    all_ray_cells: set[tuple[int, int]] = set()
    expected = grid.copy()
    for obj in objects:
        for hole in obj["holes"]:
            cells = _trace_ray_cells(obj, hole, occupied)
            if cells is None:
                return None
            for cell in cells:
                if cell in all_ray_cells:
                    return None
                all_ray_cells.add(cell)
                expected[cell] = int(hole["stamp_color"])

    forbidden = np.zeros((height, width), dtype=bool)
    for r, c in all_ray_cells:
        forbidden[r, c] = True

    stamp_occupied = occupied.copy()
    for spec in hole_specs_flat:
        shape = spec["shape"]
        slot = _place_stamp(stamp_occupied, forbidden, shape)
        if slot is None:
            return None
        top, left = slot
        rr, cc = np.where(shape)
        for r, c in zip(rr, cc, strict=False):
            grid[top + r, left + c] = int(spec["stamp_color"])
            stamp_occupied[top + r, left + c] = True
            expected[top + r, left + c] = int(spec["stamp_color"])

    # Re-parse the final grid and make sure the solver reproduces the intended output.
    parsed_objects, parsed_stamps = _scene_parts(grid, bg)
    if sum(len(obj["holes"]) for obj in parsed_objects) != total_holes:
        return None
    if len(parsed_stamps) != total_holes:
        return None
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATE_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for seed2131201979/16/2")
