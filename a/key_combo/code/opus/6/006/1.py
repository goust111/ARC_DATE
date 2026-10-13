from __future__ import annotations

import itertools
import random

import numpy as np


BG = 0
BASELINE = 5
LANE_MARKERS = (1, 3, 2)  # left-to-right marker colors: blue, green, red
SYM_TO_LANE = {"horizontal": 1, "vertical": 3, "none": 2}
SPRITE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _mask_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _classify_symmetry(mask: np.ndarray) -> str | None:
    horizontal = bool(np.array_equal(mask, np.flipud(mask)))
    vertical = bool(np.array_equal(mask, np.fliplr(mask)))
    if horizontal and not vertical:
        return "horizontal"
    if vertical and not horizontal:
        return "vertical"
    if not horizontal and not vertical:
        return "none"
    return None


def _shape_sig(mask: np.ndarray) -> tuple[int, int, bytes]:
    return int(mask.shape[0]), int(mask.shape[1]), mask.astype(np.uint8).tobytes()


def _canonical_flip_sig(mask: np.ndarray) -> tuple[int, int, bytes]:
    variants = [
        _trim_mask(mask),
        _trim_mask(np.flipud(mask)),
        _trim_mask(np.fliplr(mask)),
        _trim_mask(np.flipud(np.fliplr(mask))),
    ]
    return min(_shape_sig(variant) for variant in variants)


def _canonical_d4_sig(mask: np.ndarray) -> tuple[int, int, bytes]:
    variants = []
    for k in range(4):
        rot = np.rot90(mask, k)
        variants.append(_trim_mask(rot))
        variants.append(_trim_mask(np.fliplr(rot)))
    return min(_shape_sig(variant) for variant in variants)


def _is_irregular(mask: np.ndarray) -> bool:
    if bool(mask.all()):
        return False
    if int(np.count_nonzero(mask.sum(axis=0))) == 1:
        return False
    if int(np.count_nonzero(mask.sum(axis=1))) == 1:
        return False
    return True


def _build_shape_library() -> dict[str, list[np.ndarray]]:
    library: dict[str, list[np.ndarray]] = {"horizontal": [], "vertical": [], "none": []}
    seen: dict[str, set[tuple[int, int, bytes]]] = {key: set() for key in library}

    for h in range(2, 5):
        for w in range(2, 5):
            cells = [(r, c) for r in range(h) for c in range(w)]
            for area in range(3, min(7, h * w) + 1):
                for picks in itertools.combinations(cells, area):
                    mask = np.zeros((h, w), dtype=bool)
                    for r, c in picks:
                        mask[r, c] = True
                    mask = _trim_mask(mask)
                    if int(mask.sum()) != area or not _mask_connected(mask) or not _is_irregular(mask):
                        continue
                    category = _classify_symmetry(mask)
                    if category is None:
                        continue
                    sig = _canonical_flip_sig(mask)
                    if sig in seen[category]:
                        continue
                    seen[category].add(sig)
                    library[category].append(mask.copy())

    for category in library:
        library[category].sort(key=lambda mask: (int(mask.sum()), int(mask.shape[0]), int(mask.shape[1]), _shape_sig(mask)))
        assert len(library[category]) >= 8, f"shape library too small for {category}"
    return library


SHAPE_LIBRARY = _build_shape_library()


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _extract_objects(grid: np.ndarray) -> tuple[int, dict[int, int], list[dict]]:
    g = np.asarray(grid, dtype=int)
    baseline_rows = [r for r in range(g.shape[0] - 1) if bool(np.all(g[r, :] == BASELINE))]
    assert baseline_rows, "expected a grey baseline"

    baseline_row = None
    lane_cols = None
    for r in baseline_rows:
        marker_row = r + 1
        cols = {int(g[marker_row, c]): c for c in range(g.shape[1]) if int(g[marker_row, c]) in set(LANE_MARKERS)}
        if set(cols) == set(LANE_MARKERS):
            ordered = [cols[color] for color in LANE_MARKERS]
            if ordered[1] - ordered[0] == ordered[2] - ordered[1]:
                baseline_row = r
                lane_cols = {color: col for color, col in zip(LANE_MARKERS, ordered)}
                break
    assert baseline_row is not None and lane_cols is not None, "could not identify lane markers"

    fg = np.zeros_like(g, dtype=bool)
    fg[:baseline_row, :] = (g[:baseline_row, :] != BG) & (g[:baseline_row, :] != BASELINE)
    components = _find_components(fg)

    objects: list[dict] = []
    for coords in components:
        bbox = _bbox_coords(coords)
        r0, c0, r1, c1 = bbox
        color = int(g[coords[0, 0], coords[0, 1]])
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
        objects.append(
            {
                "coords": coords,
                "bbox": bbox,
                "mask": mask,
                "color": color,
                "area": int(coords.shape[0]),
            }
        )

    return int(baseline_row), lane_cols, objects


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    baseline_row, lane_cols, objects = _extract_objects(g)

    out = np.zeros_like(g)
    out[baseline_row, :] = BASELINE
    marker_row = baseline_row + 1
    for color in LANE_MARKERS:
        out[marker_row, lane_cols[color]] = color

    by_lane: dict[int, list[dict]] = {1: [], 2: [], 3: []}
    for obj in objects:
        if obj["area"] == 1:
            continue
        category = _classify_symmetry(obj["mask"])
        assert category is not None, "non-noise sprite must have exactly one category"
        lane_color = SYM_TO_LANE[category]
        by_lane[lane_color].append(obj)

    for lane_color, lane_col in lane_cols.items():
        stack = sorted(
            by_lane[lane_color],
            key=lambda obj: (int(obj["area"]), int(obj["bbox"][0]), int(obj["bbox"][1])),
        )
        bottom = baseline_row - 1
        for obj in stack:
            mask = obj["mask"]
            h, w = mask.shape
            left = int(lane_col) - ((w - 1) // 2)
            top = bottom - h + 1
            assert 0 <= top and 0 <= left and left + w <= g.shape[1], "settled sprite must fit the canvas"
            out[top : top + h, left : left + w][mask] = lane_color
            bottom = top - 1

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route scattered monochrome sprites into three marker lanes by symmetry class, recolor them, and stack them upward from the baseline.

    Input structure:
    - Black background contains one full-width grey baseline near the bottom and one marker row below it.
    - The marker row has exactly three equally spaced lane markers: blue on the left, green in the middle, and red on the right.
    - Above the baseline are several separated monochrome 4-connected sprites plus a few isolated single-pixel noise dots.

    Rule:
    - Ignore singleton components.
    - Classify every larger sprite by reflection symmetry inside its own bounding box: horizontal-axis symmetric, vertical-axis symmetric, or asymmetric.
    - In this task instance, horizontally symmetric sprites go to the blue lane, vertically symmetric sprites go to the green lane, and asymmetric sprites go to the red lane.
    - Within each lane, sort sprites by pixel count ascending; break equal-size ties by smaller input bbox top row.
    - Repaint each sprite to its lane color and stack the lane upward from the baseline so each sprite's bounding box bottom edge touches the baseline or the previous settled bounding box top edge.
    - Erase the original sprites and noise, while keeping the baseline and lane markers unchanged.

    Key insight:
    - Symmetry is the routing signal: it decides both the destination lane and the recolor, while pixel-count ordering decides the final gravity stack within each lane.
    """
    return _solve(np.asarray(grid, dtype=int))


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    coords = np.argwhere(mask)
    grid[coords[:, 0] + int(top), coords[:, 1] + int(left)] = int(color)


def _coords_from_mask(mask: np.ndarray, top: int, left: int) -> np.ndarray:
    coords = np.argwhere(mask).copy()
    coords[:, 0] += int(top)
    coords[:, 1] += int(left)
    return coords


def _has_buffer(coords: np.ndarray, occupied: np.ndarray) -> bool:
    h, w = occupied.shape
    for r, c in coords:
        for rr in range(int(r) - 1, int(r) + 2):
            for cc in range(int(c) - 1, int(c) + 2):
                if 0 <= rr < h and 0 <= cc < w and occupied[rr, cc]:
                    return False
    return True


def _sample_counts(total: int) -> dict[str, int]:
    counts = {"horizontal": 2, "vertical": 2, "none": 2}
    for _ in range(total - 6):
        counts[random.choice(tuple(counts))] += 1
    return counts


def _choose_shapes(counts: dict[str, int]) -> dict[str, list[np.ndarray]]:
    chosen: dict[str, list[np.ndarray]] = {key: [] for key in counts}
    used_d4: set[tuple[int, int, bytes]] = set()

    tie_cat = random.choice(tuple(counts))
    tie_area_options = [
        area
        for area in range(3, 8)
        if sum(1 for mask in SHAPE_LIBRARY[tie_cat] if int(mask.sum()) == area) >= 2
    ]
    assert tie_area_options, "expected at least one repeated area in every library"
    tie_area = random.choice(tie_area_options)

    tie_candidates = [mask for mask in SHAPE_LIBRARY[tie_cat] if int(mask.sum()) == tie_area]
    random.shuffle(tie_candidates)
    for mask in tie_candidates:
        sig = _canonical_d4_sig(mask)
        if sig not in used_d4:
            chosen[tie_cat].append(mask.copy())
            used_d4.add(sig)
            break
    for mask in tie_candidates:
        sig = _canonical_d4_sig(mask)
        if sig not in used_d4:
            chosen[tie_cat].append(mask.copy())
            used_d4.add(sig)
            break
    assert len(chosen[tie_cat]) >= 2, "failed to pick a same-area pair"

    for category, count in counts.items():
        candidates = SHAPE_LIBRARY[category][:]
        random.shuffle(candidates)
        for mask in candidates:
            if len(chosen[category]) >= count:
                break
            sig = _canonical_d4_sig(mask)
            if sig in used_d4:
                continue
            chosen[category].append(mask.copy())
            used_d4.add(sig)
        assert len(chosen[category]) == count, f"not enough {category} shapes"

    for category in chosen:
        random.shuffle(chosen[category])
    return chosen


def _build_expected(shape: tuple[int, int], baseline_row: int, lane_cols: dict[int, int], objects: list[dict]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    out[baseline_row, :] = BASELINE
    marker_row = baseline_row + 1
    for color in LANE_MARKERS:
        out[marker_row, lane_cols[color]] = color

    grouped: dict[int, list[dict]] = {1: [], 2: [], 3: []}
    for obj in objects:
        grouped[SYM_TO_LANE[obj["category"]]].append(obj)

    for lane_color, lane_col in lane_cols.items():
        stack = sorted(grouped[lane_color], key=lambda obj: (obj["area"], obj["bbox"][0], obj["bbox"][1]))
        bottom = baseline_row - 1
        for obj in stack:
            mask = obj["mask"]
            h, w = mask.shape
            left = lane_col - ((w - 1) // 2)
            top = bottom - h + 1
            _paint_mask(out, mask, top, left, lane_color)
            bottom = top - 1

    return out


def _validate_generated(inp: np.ndarray, objects: list[dict], noises: list[tuple[int, int, int]], lane_cols: dict[int, int]) -> bool:
    baseline_row, parsed_lane_cols, parsed_objects = _extract_objects(inp)
    if parsed_lane_cols != lane_cols:
        return False
    if baseline_row <= 0 or baseline_row + 1 >= inp.shape[0]:
        return False
    if len(noises) not in {2, 3}:
        return False

    by_sig = sorted((obj["category"], obj["area"], _shape_sig(obj["mask"])) for obj in objects)
    parsed_non_noise = [obj for obj in parsed_objects if obj["area"] >= 2]
    parsed_sig = sorted(
        (_classify_symmetry(obj["mask"]), obj["area"], _shape_sig(obj["mask"]))
        for obj in parsed_non_noise
    )
    if parsed_sig != by_sig:
        return False

    counts = {"horizontal": 0, "vertical": 0, "none": 0}
    for obj in objects:
        counts[obj["category"]] += 1
    if any(counts[key] < 2 for key in counts):
        return False

    tie_ok = False
    misorder_ok = False
    wrong_lane_ok = False
    for category, lane_color in SYM_TO_LANE.items():
        lane_objects = [obj for obj in objects if obj["category"] == category]
        areas = [obj["area"] for obj in lane_objects]
        if len(areas) != len(set((area, obj["bbox"][0]) for area, obj in zip(areas, lane_objects))):
            return False
        if len(areas) != len(set((_canonical_d4_sig(obj["mask"]) for obj in lane_objects))):
            return False
        if len(areas) != len(set(areas)):
            tie_ok = True
        sorted_by_row = sorted(lane_objects, key=lambda obj: obj["bbox"][0])
        sorted_by_rule = sorted(lane_objects, key=lambda obj: (obj["area"], obj["bbox"][0]))
        if [obj["area"] for obj in sorted_by_row] != [obj["area"] for obj in sorted_by_rule]:
            misorder_ok = True
        lane_col = lane_cols[lane_color]
        for obj in lane_objects:
            _, c0, _, c1 = obj["bbox"]
            center = (c0 + c1) / 2.0
            nearest = min(lane_cols, key=lambda color: (abs(lane_cols[color] - center), lane_cols[color]))
            if nearest != lane_color:
                wrong_lane_ok = True
    if not (tie_ok and misorder_ok and wrong_lane_ok):
        return False

    expected = _build_expected(inp.shape, baseline_row, lane_cols, objects)
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return False

    for lane_color in LANE_MARKERS:
        lane_objects = [obj for obj in objects if SYM_TO_LANE[obj["category"]] == lane_color]
        lane_stack = sorted(lane_objects, key=lambda obj: (obj["area"], obj["bbox"][0], obj["bbox"][1]))
        bottom = baseline_row - 1
        for obj in lane_stack:
            h, w = obj["mask"].shape
            left = lane_cols[lane_color] - ((w - 1) // 2)
            top = bottom - h + 1
            patch = actual[top : top + h, left : left + w]
            if not np.array_equal(patch == lane_color, obj["mask"]):
                return False
            bottom = top - 1

    return True


def _attempt_generate() -> np.ndarray | None:
    height = random.randint(20, 26)
    width = random.randint(22, 29)
    baseline_row = random.randint(height - 5, height - 4)
    marker_row = baseline_row + 1

    spacing = random.randint(5, 7)
    start_min = 2
    start_max = width - 3 - 2 * spacing
    if start_max < start_min:
        return None
    start = random.randint(start_min, start_max)
    lane_cols = {1: start, 3: start + spacing, 2: start + 2 * spacing}

    total = random.randint(6, 8)
    counts = _sample_counts(total)
    shape_groups = _choose_shapes(counts)
    colors = random.sample(SPRITE_COLORS, total)

    inp = np.zeros((height, width), dtype=int)
    inp[baseline_row, :] = BASELINE
    inp[marker_row, lane_cols[1]] = 1
    inp[marker_row, lane_cols[3]] = 3
    inp[marker_row, lane_cols[2]] = 2

    occupied = np.zeros((height, width), dtype=bool)
    occupied[baseline_row, :] = True

    objects: list[dict] = []
    all_masks: list[tuple[str, np.ndarray]] = []
    for category, masks in shape_groups.items():
        for mask in masks:
            variant = mask.copy()
            if category in {"horizontal", "none"} and random.random() < 0.5:
                variant = np.fliplr(variant)
            if category in {"vertical", "none"} and random.random() < 0.5:
                variant = np.flipud(variant)
            all_masks.append((category, variant.copy()))
    random.shuffle(all_masks)

    for (category, mask), color in zip(all_masks, colors):
        h, w = mask.shape
        row_candidates = list(range(0, baseline_row - h - 0))
        col_candidates = list(range(0, width - w + 1))
        random.shuffle(row_candidates)
        random.shuffle(col_candidates)
        placed = False
        for top in row_candidates:
            if top + h - 1 >= baseline_row - 1:
                continue
            for left in col_candidates:
                coords = _coords_from_mask(mask, top, left)
                if not _has_buffer(coords, occupied):
                    continue
                _paint_mask(inp, mask, top, left, color)
                occupied[coords[:, 0], coords[:, 1]] = True
                bbox = (top, left, top + h - 1, left + w - 1)
                objects.append(
                    {
                        "mask": mask.copy(),
                        "color": int(color),
                        "category": category,
                        "area": int(mask.sum()),
                        "bbox": bbox,
                    }
                )
                placed = True
                break
            if placed:
                break
        if not placed:
            return None

    noises: list[tuple[int, int, int]] = []
    for _ in range(random.randint(2, 3)):
        candidates = [(r, c) for r in range(0, baseline_row - 1) for c in range(width) if inp[r, c] == BG]
        random.shuffle(candidates)
        placed = False
        for r, c in candidates:
            coords = np.array([[r, c]], dtype=int)
            if not _has_buffer(coords, occupied):
                continue
            color = random.choice([obj["color"] for obj in objects])
            inp[r, c] = color
            occupied[r, c] = True
            noises.append((r, c, int(color)))
            placed = True
            break
        if not placed:
            return None

    stack_heights = {1: 0, 2: 0, 3: 0}
    for obj in objects:
        stack_heights[SYM_TO_LANE[obj["category"]]] += int(obj["mask"].shape[0])
    if any(total_height > baseline_row for total_height in stack_heights.values()):
        return None

    if not _validate_generated(inp, objects, noises, lane_cols):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
