from __future__ import annotations

from collections import Counter
import hashlib
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7

INPUT_COLORS = (BLUE, RED, PINK, ORANGE)
FOUR_NEIGHBORS = ((1, 0), (-1, 0), (0, 1), (0, -1))

GRID_ROWS = (16, 19)
GRID_COLS = (16, 19)
THRESHOLD_AREAS = (6, 7)
LARGE_AREAS = (10, 11, 12)
SMALL_AREAS = (3, 4)
MAX_GENERATION_ATTEMPTS = 500
RECENT_OUTPUT_LIMIT = 192

_RECENT_OUTPUT_SIGNATURES: set[str] = set()


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []

    for r0 in range(h):
        for c0 in range(w):
            color = int(g[r0, c0])
            if color == BLACK or seen[r0, c0]:
                continue

            stack = [(r0, c0)]
            seen[r0, c0] = True
            coords: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                for dr, dc in FOUR_NEIGHBORS:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            arr = np.array(coords, dtype=int)
            objects.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": _bbox_from_coords(arr),
                    "area": int(arr.shape[0]),
                }
            )
    return objects


def _mode_area_threshold(areas: list[int]) -> int:
    counts = Counter(int(area) for area in areas)
    best_count = max(counts.values())
    return max(area for area, count in counts.items() if count == best_count)


def _lower_median(values: list[int]) -> int:
    ordered = sorted(int(v) for v in values)
    return ordered[(len(ordered) - 1) // 2]


def _expanded_perimeter_cells(
    bbox: tuple[int, int, int, int], shape: tuple[int, int]
) -> list[tuple[int, int]]:
    h, w = map(int, shape)
    r0, c0, r1, c1 = bbox
    er0 = max(0, int(r0) - 1)
    ec0 = max(0, int(c0) - 1)
    er1 = min(h - 1, int(r1) + 1)
    ec1 = min(w - 1, int(c1) + 1)

    cells: list[tuple[int, int]] = []
    for r in range(er0, er1 + 1):
        for c in range(ec0, ec1 + 1):
            if r in (er0, er1) or c in (ec0, ec1):
                cells.append((r, c))
    return cells


def _render_output_from_objects(
    shape: tuple[int, int],
    objects: list[dict],
    *,
    keep_mode_equals_only: bool,
    border_mode: str,
) -> np.ndarray:
    areas = [int(obj["area"]) for obj in objects]
    threshold = _mode_area_threshold(areas)
    if keep_mode_equals_only:
        survivors = [obj for obj in objects if int(obj["area"]) == threshold]
    else:
        survivors = [obj for obj in objects if int(obj["area"]) >= threshold]
    assert survivors, "expected at least one surviving object"

    median_area = _lower_median([int(obj["area"]) for obj in survivors])
    out = np.full(shape, BLACK, dtype=int)
    body_mask = np.zeros(shape, dtype=bool)

    for obj in survivors:
        rr, cc = obj["coords"].T
        out[rr, cc] = GREY
        body_mask[rr, cc] = True

    ordered = sorted(survivors, key=lambda obj: (int(obj["bbox"][0]), int(obj["bbox"][1])))
    for obj in ordered:
        if border_mode == "classified":
            border_color = GREEN if int(obj["area"]) > median_area else YELLOW
        elif border_mode == "all_yellow":
            border_color = YELLOW
        else:
            raise ValueError(f"unknown border mode: {border_mode}")

        for r, c in _expanded_perimeter_cells(obj["bbox"], shape):
            if body_mask[r, c]:
                continue
            out[r, c] = border_color
    return out


def _render_output_black_only_border(shape: tuple[int, int], objects: list[dict]) -> np.ndarray:
    areas = [int(obj["area"]) for obj in objects]
    threshold = _mode_area_threshold(areas)
    survivors = [obj for obj in objects if int(obj["area"]) >= threshold]
    assert survivors, "expected at least one surviving object"

    median_area = _lower_median([int(obj["area"]) for obj in survivors])
    out = np.full(shape, BLACK, dtype=int)
    body_mask = np.zeros(shape, dtype=bool)

    for obj in survivors:
        rr, cc = obj["coords"].T
        out[rr, cc] = GREY
        body_mask[rr, cc] = True

    ordered = sorted(survivors, key=lambda obj: (int(obj["bbox"][0]), int(obj["bbox"][1])))
    for obj in ordered:
        border_color = GREEN if int(obj["area"]) > median_area else YELLOW
        for r, c in _expanded_perimeter_cells(obj["bbox"], shape):
            if body_mask[r, c]:
                continue
            if out[r, c] == BLACK:
                out[r, c] = border_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only objects at or above the modal area threshold, then outline surviving size classes and gray-fill their bodies.

    Input structure:
    - The grid has a black background and several single-color 4-connected foreground objects.
    - Each object is an isolated irregular blob; different objects may reuse colors.
    - Object areas span small, medium, and large bands, with at least one repeated medium area and at least two repeated large maxima.

    Rule:
    - Find every 4-connected foreground object and compute its pixel area.
    - Let T be the most frequent area value; if multiple area values tie for that frequency, choose the largest tied area.
    - Erase every object whose area is smaller than T.
    - Among the survivors, compute the lower median area M; recolor every surviving object body to gray.
    - For each surviving object, expand its bounding box by one cell in every direction, clip to the grid, and paint that perimeter while never covering a gray body cell.
    - Survivors with area greater than M get a green border; survivors with area less than or equal to M get a yellow border; objects are processed by bbox top-left order, and later borders overwrite earlier ones.

    Key insight:
    - Area is reused three times: first as the threshold that decides survival, then as the statistic that splits survivors into two border classes, and finally through the ordered border overwrites when expanded boxes overlap.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert int(np.min(g)) >= 0, "colors must be non-negative"

    objects = _find_objects(g)
    assert objects, "expected at least one foreground object"
    return _render_output_from_objects(g.shape, objects, keep_mode_equals_only=False, border_mode="classified")


def _random_blob_patch(area: int, color: int) -> np.ndarray:
    dim_options = [
        (h, w)
        for h in range(2, 7)
        for w in range(2, 7)
        if h * w > area and h * w <= area + max(4, area // 2 + 1)
    ]
    assert dim_options, f"no feasible dimensions for area {area}"

    for _ in range(200):
        h, w = random.choice(dim_options)
        cells = {(random.randrange(h), random.randrange(w))}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for dr, dc in FOUR_NEIGHBORS:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            cells.add(random.choice(frontier))

        coords = np.array(sorted(cells), dtype=int)
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        cropped = coords - np.array([r0, c0], dtype=int)
        patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
        patch[cropped[:, 0], cropped[:, 1]] = int(color)
        if patch.shape[0] > 1 and patch.shape[1] > 1 and int(np.count_nonzero(patch)) < int(patch.size):
            return patch

    raise RuntimeError(f"failed to create irregular blob of area {area}")


def _sample_area_plan() -> tuple[int, list[int]]:
    threshold_area = random.choice(THRESHOLD_AREAS)
    threshold_count = random.choice((3, 4))
    max_large_area = random.choice(LARGE_AREAS)
    large_count = 2 if threshold_count == 4 else random.choice((2, 3))

    small_areas = [random.choice(SMALL_AREAS), random.choice(SMALL_AREAS)]
    threshold_areas = [threshold_area] * threshold_count
    if large_count == 2:
        large_areas = [max_large_area, max_large_area]
    else:
        extra_large = random.randint(threshold_area + 1, max_large_area - 1)
        large_areas = [max_large_area, max_large_area, extra_large]

    areas = small_areas + threshold_areas + large_areas
    random.shuffle(areas)
    return threshold_area, areas


def _bbox_has_gap(
    bbox: tuple[int, int, int, int], others: list[tuple[int, int, int, int]]
) -> bool:
    r0, c0, r1, c1 = bbox
    for or0, oc0, or1, oc1 in others:
        if not (r1 < or0 - 1 or or1 < r0 - 1 or c1 < oc0 - 1 or oc1 < c0 - 1):
            return False
    return True


def _candidate_positions(
    shape: tuple[int, int],
    patch_shape: tuple[int, int],
    placed_bboxes: list[tuple[int, int, int, int]],
) -> list[tuple[int, int]]:
    h, w = map(int, shape)
    ph, pw = map(int, patch_shape)
    candidates = []
    for r in range(h - ph + 1):
        for c in range(w - pw + 1):
            bbox = (r, c, r + ph - 1, c + pw - 1)
            if _bbox_has_gap(bbox, placed_bboxes):
                candidates.append((r, c))
    random.shuffle(candidates)
    return candidates


def _place_overlap_pair(
    shape: tuple[int, int],
    small_patch: np.ndarray,
    large_patch: np.ndarray,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    h, w = map(int, shape)
    sh, sw = small_patch.shape
    lh, lw = large_patch.shape

    layouts = []
    if sw + 1 + lw <= w:
        for r in range(h - max(sh, lh) + 1):
            for c in range(w - (sw + 1 + lw) + 1):
                layouts.append(((r, c), (r, c + sw + 1)))
    if sh + 1 + lh <= h:
        for r in range(h - (sh + 1 + lh) + 1):
            for c in range(w - max(sw, lw) + 1):
                layouts.append(((r, c), (r + sh + 1, c)))

    random.shuffle(layouts)
    for small_pos, large_pos in layouts:
        sr, sc = small_pos
        lr, lc = large_pos
        small_bbox = (sr, sc, sr + sh - 1, sc + sw - 1)
        large_bbox = (lr, lc, lr + lh - 1, lc + lw - 1)
        if not _bbox_has_gap(small_bbox, [large_bbox]):
            continue

        small_body = {
            (int(r), int(c))
            for r, c in (np.argwhere(small_patch != BLACK) + np.array([sr, sc], dtype=int))
        }
        large_body = {
            (int(r), int(c))
            for r, c in (np.argwhere(large_patch != BLACK) + np.array([lr, lc], dtype=int))
        }
        small_perim = set(_expanded_perimeter_cells(small_bbox, shape))
        large_perim = set(_expanded_perimeter_cells(large_bbox, shape))
        shared = (small_perim & large_perim) - small_body - large_body
        if shared:
            return small_pos, large_pos
    return None


def _build_specs(shape: tuple[int, int]) -> list[dict] | None:
    threshold_area, areas = _sample_area_plan()
    colors = [random.choice(INPUT_COLORS) for _ in areas]
    patches = [_random_blob_patch(area, color) for area, color in zip(areas, colors)]

    threshold_indices = [i for i, area in enumerate(areas) if area == threshold_area]
    large_indices = [i for i, area in enumerate(areas) if area > threshold_area]
    if not threshold_indices or not large_indices:
        return None

    overlap_small_i = random.choice(threshold_indices)
    overlap_large_i = random.choice(large_indices)
    pair = _place_overlap_pair(shape, patches[overlap_small_i], patches[overlap_large_i])
    if pair is None:
        return None

    specs: list[dict | None] = [None] * len(areas)
    placed_bboxes: list[tuple[int, int, int, int]] = []

    for index, pos in ((overlap_small_i, pair[0]), (overlap_large_i, pair[1])):
        patch = patches[index]
        r, c = pos
        ph, pw = patch.shape
        coords = np.argwhere(patch != BLACK) + np.array([r, c], dtype=int)
        bbox = (r, c, r + ph - 1, c + pw - 1)
        specs[index] = {
            "color": int(colors[index]),
            "area": int(areas[index]),
            "coords": coords,
            "bbox": bbox,
            "patch": patch,
        }
        placed_bboxes.append(bbox)

    remaining = [i for i in range(len(areas)) if specs[i] is None]
    remaining.sort(key=lambda i: patches[i].shape[0] * patches[i].shape[1], reverse=True)
    for index in remaining:
        patch = patches[index]
        positions = _candidate_positions(shape, patch.shape, placed_bboxes)
        if not positions:
            return None
        r, c = random.choice(positions)
        ph, pw = patch.shape
        coords = np.argwhere(patch != BLACK) + np.array([r, c], dtype=int)
        bbox = (r, c, r + ph - 1, c + pw - 1)
        specs[index] = {
            "color": int(colors[index]),
            "area": int(areas[index]),
            "coords": coords,
            "bbox": bbox,
            "patch": patch,
        }
        placed_bboxes.append(bbox)

    return [spec for spec in specs if spec is not None]


def _render_input(shape: tuple[int, int], specs: list[dict]) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)
    for spec in specs:
        patch = spec["patch"]
        r0, c0, _, _ = spec["bbox"]
        mask = patch != BLACK
        grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]][mask] = patch[mask]
    return grid


def _spec_signature(grid: np.ndarray) -> str:
    return hashlib.md5(grid.tobytes() + str(grid.shape).encode()).hexdigest()


def _validate_semantics(inp: np.ndarray, specs: list[dict]) -> bool:
    objects = _find_objects(inp)
    if len(objects) != len(specs):
        return False

    areas = sorted(int(spec["area"]) for spec in specs)
    found_areas = sorted(int(obj["area"]) for obj in objects)
    if areas != found_areas:
        return False

    counts = Counter(int(spec["area"]) for spec in specs)
    threshold = _mode_area_threshold([int(spec["area"]) for spec in specs])
    if counts[threshold] < 3:
        return False
    if sum(1 for spec in specs if int(spec["area"]) > threshold) < 2:
        return False
    max_area = max(int(spec["area"]) for spec in specs)
    if counts[max_area] < 2:
        return False

    expected = _render_output_from_objects(inp.shape, specs, keep_mode_equals_only=False, border_mode="classified")
    if not np.array_equal(main(inp), expected):
        return False

    shortcut_equal_only = _render_output_from_objects(
        inp.shape, specs, keep_mode_equals_only=True, border_mode="classified"
    )
    if np.array_equal(shortcut_equal_only, expected):
        return False

    shortcut_no_class = _render_output_from_objects(
        inp.shape, specs, keep_mode_equals_only=False, border_mode="all_yellow"
    )
    if np.array_equal(shortcut_no_class, expected):
        return False

    shortcut_black_only = _render_output_black_only_border(inp.shape, specs)
    if np.array_equal(shortcut_black_only, expected):
        return False

    if int(np.count_nonzero(expected == GREEN)) == 0 or int(np.count_nonzero(expected == YELLOW)) == 0:
        return False
    if int(np.count_nonzero(expected == GREY)) == 0:
        return False

    output_sig = _spec_signature(expected)
    if output_sig in _RECENT_OUTPUT_SIGNATURES:
        return False
    _RECENT_OUTPUT_SIGNATURES.add(output_sig)
    if len(_RECENT_OUTPUT_SIGNATURES) > RECENT_OUTPUT_LIMIT:
        _RECENT_OUTPUT_SIGNATURES.clear()
        _RECENT_OUTPUT_SIGNATURES.add(output_sig)
    return True


def _generate_once() -> np.ndarray | None:
    rows = random.randint(*GRID_ROWS)
    cols = random.randint(*GRID_COLS)
    specs = _build_specs((rows, cols))
    if specs is None:
        return None

    inp = _render_input((rows, cols), specs)
    if not _validate_semantics(inp, specs):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_once()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
