from __future__ import annotations

import random

import numpy as np


DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIRS8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[int(np.argmax(counts))])


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a non-empty mask"
    (r0, c0), (r1, c1) = coords.min(axis=0), coords.max(axis=0)
    return int(r0), int(c0), int(r1) + 1, int(c1) + 1


def _crop_to_mask(mask: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_from_mask(mask)
    return mask[r0:r1, c0:c1].copy()


def _neighbors(shape: tuple[int, int], r: int, c: int, dirs: tuple[tuple[int, int], ...]):
    height, width = shape
    for dr, dc in dirs:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < height and 0 <= cc < width:
            yield rr, cc


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for rr, cc in _neighbors(mask.shape, r, c, DIRS4):
            if mask[rr, cc] and (rr, cc) not in seen:
                seen.add((rr, cc))
                stack.append((rr, cc))
    return len(seen) == int(mask.sum())


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    scene = np.asarray(grid, dtype=int)
    height, width = scene.shape
    seen = np.zeros((height, width), dtype=bool)
    comps: list[dict] = []

    for r in range(height):
        for c in range(width):
            color = int(scene[r, c])
            if color == bg or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors(scene.shape, rr, cc, DIRS8):
                    if seen[nr, nc] or int(scene[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            comps.append(
                {
                    "color": color,
                    "coords": {tuple(map(int, rc)) for rc in coords},
                    "bbox": (min(rows), min(cols), max(rows) + 1, max(cols) + 1),
                }
            )

    return comps


def _reflect_coords(
    coords: set[tuple[int, int]], axis_kind: str, two_axis: int
) -> set[tuple[int, int]]:
    if axis_kind == "h":
        return {(int(two_axis - r), int(c)) for r, c in coords}
    assert axis_kind == "v"
    return {(int(r), int(two_axis - c)) for r, c in coords}


def _rank_axes(coords: set[tuple[int, int]]) -> list[dict]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)

    ranked: list[dict] = []
    for two_axis in range(2 * min_r, 2 * max_r + 1):
        reflected = _reflect_coords(coords, "h", two_axis)
        ranked.append(
            {
                "axis_kind": "h",
                "two_axis": int(two_axis),
                "diff": int(len(coords ^ reflected)),
                "repair": coords | reflected,
            }
        )
    for two_axis in range(2 * min_c, 2 * max_c + 1):
        reflected = _reflect_coords(coords, "v", two_axis)
        ranked.append(
            {
                "axis_kind": "v",
                "two_axis": int(two_axis),
                "diff": int(len(coords ^ reflected)),
                "repair": coords | reflected,
            }
        )

    ranked.sort(
        key=lambda item: (
            int(item["diff"]),
            0 if item["axis_kind"] == "h" else 1,
            int(item["two_axis"]),
        )
    )
    return ranked


def _repair_component(coords: set[tuple[int, int]]) -> tuple[set[tuple[int, int]], int]:
    ranked = _rank_axes(coords)
    best = ranked[0]
    return set(best["repair"]), int(best["diff"])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair each nearly symmetric object and recolor it by the legend position
    indexed by its symmetry-defect count.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2 and g.shape[1] >= 1

    bg = _background_color(g)
    legend = [int(color) for color in g[-1] if int(color) != bg]
    assert legend, "expected a bottom legend row"

    comps = _find_components(g[:-1], bg)
    assert comps, "expected at least one foreground object above the legend"

    out = np.full_like(g, bg)
    out[-1] = g[-1]

    for comp in comps:
        repaired, defect_count = _repair_component(comp["coords"])
        assert defect_count > 0, "objects must contain visible symmetry defects"
        assert defect_count <= len(legend), "legend must cover every defect count"
        new_color = legend[defect_count - 1]
        for r, c in repaired:
            assert 0 <= r < g.shape[0] - 1 and 0 <= c < g.shape[1]
            out[r, c] = int(new_color)

    return out


# --------------- generator ---------------

GEN_RETRIES = 400
GRID_H_RANGE = (17, 26)
GRID_W_RANGE = (20, 31)
LEGEND_LEN = 6
OBJECT_COUNT_CHOICES = (2, 3)
LOCAL_H_RANGE = (7, 11)
LOCAL_W_RANGE = (7, 11)
MIN_OBJECT_AREA = 7
MAX_OBJECT_FILL = 0.42
STOP_PROBABILITY = 0.28
PLACEMENT_GAP = 1


def _mirror_local(shape: tuple[int, int], axis_kind: str, r: int, c: int) -> tuple[int, int]:
    height, width = shape
    if axis_kind == "h":
        return int(height - 1 - r), int(c)
    assert axis_kind == "v"
    return int(r), int(width - 1 - c)


def _orbit(shape: tuple[int, int], axis_kind: str, r: int, c: int) -> tuple[tuple[int, int], ...]:
    other = _mirror_local(shape, axis_kind, r, c)
    if other == (r, c):
        return ((int(r), int(c)),)
    a = (int(r), int(c))
    b = (int(other[0]), int(other[1]))
    return tuple(sorted((a, b)))


def _has_neighbor(mask: np.ndarray, cell: tuple[int, int]) -> bool:
    r, c = cell
    return any(mask[rr, cc] for rr, cc in _neighbors(mask.shape, r, c, DIRS4))


def _build_symmetric_mask(height: int, width: int, axis_kind: str) -> np.ndarray | None:
    max_area = max(MIN_OBJECT_AREA + 2, int(height * width * MAX_OBJECT_FILL))
    if max_area < MIN_OBJECT_AREA:
        return None

    mask = np.zeros((height, width), dtype=bool)
    if axis_kind == "h":
        base_row = height // 2 if height % 2 else height // 2 - 1
        seed = (int(base_row), int(random.randrange(width)))
    else:
        base_col = width // 2 if width % 2 else width // 2 - 1
        seed = (int(random.randrange(height)), int(base_col))

    for r, c in _orbit(mask.shape, axis_kind, *seed):
        mask[r, c] = True

    while True:
        area = int(mask.sum())
        if area >= MIN_OBJECT_AREA and (area >= max_area or random.random() < STOP_PROBABILITY):
            break

        candidates: list[tuple[tuple[int, int], ...]] = []
        seen_orbits: set[tuple[tuple[int, int], ...]] = set()
        for r in range(height):
            for c in range(width):
                orb = _orbit(mask.shape, axis_kind, r, c)
                if orb in seen_orbits:
                    continue
                seen_orbits.add(orb)
                add_cells = [cell for cell in orb if not mask[cell]]
                if not add_cells:
                    continue
                if area + len(add_cells) > max_area:
                    continue
                if any(_has_neighbor(mask, cell) for cell in add_cells):
                    candidates.append(orb)

        if not candidates:
            return None

        for r, c in random.choice(candidates):
            mask[r, c] = True

    if not _is_connected(mask):
        return None

    cropped = _crop_to_mask(mask)
    if min(cropped.shape) < 3 or int(cropped.sum()) < MIN_OBJECT_AREA:
        return None
    return mask


def _remove_if_safe(mask: np.ndarray, cell: tuple[int, int]) -> bool:
    trial = mask.copy()
    trial[cell] = False
    if not _is_connected(trial):
        return False
    mask[cell] = False
    return True


def _defect_ops(defect_pairs: int, mode: str) -> list[str]:
    if mode == "add":
        return ["add"] * defect_pairs
    if mode == "remove":
        return ["remove"] * defect_pairs
    if defect_pairs == 1:
        return [random.choice(["add", "remove"])]
    ops = ["add", "remove"]
    while len(ops) < defect_pairs:
        ops.append(random.choice(["add", "remove"]))
    random.shuffle(ops)
    return ops


def _apply_defects(
    base_mask: np.ndarray, axis_kind: str, defect_pairs: int, mode: str
) -> np.ndarray | None:
    mask = base_mask.copy()
    used_orbits: set[tuple[tuple[int, int], ...]] = set()

    for op in _defect_ops(defect_pairs, mode):
        candidates: list[tuple[tuple[tuple[int, int], ...], tuple[int, int]]] = []
        seen_orbits: set[tuple[tuple[int, int], ...]] = set()
        for r in range(mask.shape[0]):
            for c in range(mask.shape[1]):
                orb = _orbit(mask.shape, axis_kind, r, c)
                if len(orb) != 2 or orb in used_orbits or orb in seen_orbits:
                    continue
                seen_orbits.add(orb)
                a, b = orb

                if op == "add":
                    if mask[a] or mask[b]:
                        continue
                    for cell in (a, b):
                        if _has_neighbor(mask, cell):
                            candidates.append((orb, cell))
                else:
                    if not (mask[a] and mask[b]):
                        continue
                    for cell in (a, b):
                        trial = mask.copy()
                        trial[cell] = False
                        if _is_connected(trial):
                            candidates.append((orb, cell))

        if not candidates:
            return None

        orb, cell = random.choice(candidates)
        if op == "add":
            mask[cell] = True
        else:
            if not _remove_if_safe(mask, cell):
                return None
        used_orbits.add(orb)

    return mask


def _mask_to_coords(mask: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, rc)) for rc in np.argwhere(mask)}


def _validate_local_object(mask: np.ndarray, axis_kind: str, expected_diff: int) -> bool:
    cropped = _crop_to_mask(mask)
    coords = _mask_to_coords(cropped)
    ranked = _rank_axes(coords)
    best = ranked[0]
    second = ranked[1] if len(ranked) > 1 else None
    if int(best["diff"]) != int(expected_diff):
        return False
    if best["axis_kind"] != axis_kind:
        return False
    if second is not None and int(second["diff"]) == int(best["diff"]):
        return False
    return True


def _pair_masks_from_coords(
    input_coords: set[tuple[int, int]], repair_coords: set[tuple[int, int]]
) -> tuple[np.ndarray, np.ndarray]:
    all_coords = input_coords | repair_coords
    rows = [r for r, _ in all_coords]
    cols = [c for _, c in all_coords]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    shape = (max_r - min_r + 1, max_c - min_c + 1)

    input_mask = np.zeros(shape, dtype=bool)
    repair_mask = np.zeros(shape, dtype=bool)
    for r, c in input_coords:
        input_mask[r - min_r, c - min_c] = True
    for r, c in repair_coords:
        repair_mask[r - min_r, c - min_c] = True
    return input_mask, repair_mask


def _generate_object() -> dict | None:
    axis_kind = random.choice(["h", "v"])
    height = random.randint(*LOCAL_H_RANGE)
    width = random.randint(*LOCAL_W_RANGE)

    base = _build_symmetric_mask(height, width, axis_kind)
    if base is None:
        return None

    defect_pairs = random.randint(1, LEGEND_LEN // 2)
    mode = random.choice(["add", "remove", "mixed"])
    damaged = _apply_defects(base, axis_kind, defect_pairs, mode)
    if damaged is None:
        return None
    damaged_crop = _crop_to_mask(damaged)
    if not _validate_local_object(damaged_crop, axis_kind, 2 * defect_pairs):
        return None

    input_coords = _mask_to_coords(damaged_crop)
    repair_coords, diff = _repair_component(input_coords)
    if diff != 2 * defect_pairs:
        return None

    damaged_crop, repaired_crop = _pair_masks_from_coords(input_coords, repair_coords)
    if not _is_connected(damaged_crop):
        return None
    if min(repaired_crop.shape) < 3:
        return None

    return {
        "axis_kind": axis_kind,
        "input_mask": damaged_crop,
        "repair_mask": repaired_crop,
        "defect_count": diff,
    }


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _place_objects(
    objects: list[dict], scene_h: int, scene_w: int
) -> list[tuple[int, int]] | None:
    occupied = np.zeros((scene_h, scene_w), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(objects)
    order = sorted(range(len(objects)), key=lambda i: int(objects[i]["repair_mask"].sum()), reverse=True)

    for idx in order:
        repair = objects[idx]["repair_mask"]
        h, w = repair.shape
        if h > scene_h or w > scene_w:
            return None

        blocked = _dilate8(occupied)
        valid = []
        for r0 in range(scene_h - h + 1):
            for c0 in range(scene_w - w + 1):
                if r0 + h > scene_h - PLACEMENT_GAP:
                    continue
                if np.any(blocked[r0 : r0 + h, c0 : c0 + w] & repair):
                    continue
                valid.append((r0, c0))

        if not valid:
            return None

        r0, c0 = random.choice(valid)
        placements[idx] = (r0, c0)
        occupied[r0 : r0 + h, c0 : c0 + w] |= repair

    assert all(p is not None for p in placements)
    return [tuple(map(int, p)) for p in placements]


def _paint_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    h, w = mask.shape
    window = grid[r0 : r0 + h, c0 : c0 + w]
    window[mask] = int(color)


def _expected_output(
    shape: tuple[int, int],
    bg: int,
    legend_start: int,
    legend: list[int],
    objects: list[dict],
    placements: list[tuple[int, int]],
) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    out[-1, legend_start : legend_start + len(legend)] = legend
    for obj, (r0, c0) in zip(objects, placements):
        new_color = legend[int(obj["defect_count"]) - 1]
        _paint_mask(out, obj["repair_mask"], r0, c0, new_color)
    return out


def _attempt_generate() -> np.ndarray | None:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    scene_h = height - 1
    if scene_h < 6 or width <= LEGEND_LEN:
        return None

    bg = random.randrange(10)
    palette = [color for color in range(10) if color != bg]
    legend = random.sample(palette, LEGEND_LEN)
    object_count = random.choice(OBJECT_COUNT_CHOICES)
    object_colors = [color for color in palette if color not in legend]
    if len(object_colors) < object_count:
        return None
    object_colors = random.sample(object_colors, object_count)

    objects = []
    for color in object_colors:
        obj = _generate_object()
        if obj is None:
            return None
        obj["input_color"] = color
        objects.append(obj)

    placements = _place_objects(objects, scene_h, width)
    if placements is None:
        return None

    legend_start = random.randint(0, width - LEGEND_LEN)
    grid = np.full((height, width), bg, dtype=int)
    grid[-1, legend_start : legend_start + LEGEND_LEN] = legend
    for obj, (r0, c0) in zip(objects, placements):
        _paint_mask(grid, obj["input_mask"], r0, c0, int(obj["input_color"]))

    expected = _expected_output(grid.shape, bg, legend_start, legend, objects, placements)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
