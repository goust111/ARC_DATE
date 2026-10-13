from __future__ import annotations

from collections import deque
from importlib.machinery import SourceFileLoader
from pathlib import Path
import random

import numpy as np


def _load_arc_utils():
    candidates = [Path("/home/lixinyu/snap/new/solvers/arc_utils.py")]
    here = Path(__file__).resolve()
    candidates.extend(parent / "new/solvers/arc_utils.py" for parent in here.parents)
    for candidate in candidates:
        if candidate.exists():
            return SourceFileLoader("_arc_utils_task_10", str(candidate)).load_module()
    return None


_ARC_UTILS = _load_arc_utils()
if _ARC_UTILS is not None:
    Color = _ARC_UTILS.Color
    bbox_coords = _ARC_UTILS.bbox_coords
    components_mask = _ARC_UTILS.components_mask
    crop_bbox = _ARC_UTILS.crop_bbox
    neighbors4 = _ARC_UTILS.neighbors4
else:
    class Color:
        BLACK = 0
        RED = 2
        GREEN = 3
        YELLOW = 4
        TEAL = 8
        MAROON = 9

    def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        return int(r0), int(c0), int(r1), int(c1)

    def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
        r0, c0, r1, c1 = map(int, bbox)
        return grid[r0 : r1 + 1, c0 : c1 + 1].copy()

    def neighbors4(r: int, c: int):
        yield r - 1, c
        yield r + 1, c
        yield r, c - 1
        yield r, c + 1

    def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
        mask = np.asarray(mask, dtype=bool)
        seen = np.zeros_like(mask, dtype=bool)
        if int(connectivity) == 4:
            deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        else:
            deltas = [
                (-1, -1), (-1, 0), (-1, 1),
                (0, -1),           (0, 1),
                (1, -1),  (1, 0),  (1, 1),
            ]

        comps: list[np.ndarray] = []
        h, w = mask.shape
        for r, c in np.argwhere(mask):
            if seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in deltas:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
        return comps


Coord = tuple[int, int]
Placement = tuple[np.ndarray, tuple[int, int, int, int], np.ndarray]

BG = Color.BLACK
MATCH_COLOR = Color.GREEN
KEEP_COLOR = Color.TEAL
ODD_COLOR = Color.RED
EVEN_COLOR = Color.YELLOW
NOISE_COLOR = Color.MAROON

TEMPLATE_BOX = 5
CORE_MASK = np.array(
    [
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
    ],
    dtype=bool,
)

GRID_MIN = 18
GRID_MAX = 24
MATCH_COUNTS = (2, 3)
DISTRACTOR_COUNTS = (1, 2)
NOISE_RANGE = (2, 5)
TARGET_AREA_RANGE = (13, 20)
PLACEMENT_PADDING = 1
MAX_GENERATION_ATTEMPTS = 500


def _coords_to_mask(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = bbox_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    shifted = coords - np.array([[r0, c0]])
    mask[shifted[:, 0], shifted[:, 1]] = True
    return mask, (int(r0), int(c0), int(r1), int(c1))


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    coords = np.argwhere(mask)
    assert len(coords) > 0
    b = bbox_coords(coords)
    cropped = crop_bbox(mask.astype(int), b)
    return tuple(tuple(int(v) for v in row) for row in cropped)


def _dihedral_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, ...], ...]] = set()
    for reflect in (False, True):
        base = np.fliplr(mask) if reflect else mask
        for turns in range(4):
            candidate = np.rot90(base, turns)
            key = _mask_key(candidate)
            if key not in seen:
                seen.add(key)
                variants.append(np.asarray(candidate, dtype=bool))
    return variants


def _layer_depths(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    depth = np.full((h, w), -1, dtype=int)
    queue: deque[Coord] = deque()

    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        for nr, nc in neighbors4(rr, cc):
            if not (0 <= nr < h and 0 <= nc < w) or not mask[nr, nc]:
                depth[rr, cc] = 0
                queue.append((rr, cc))
                break

    assert queue, "nonempty object must have a boundary"
    while queue:
        r, c = queue.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and depth[nr, nc] == -1:
                depth[nr, nc] = depth[r, c] + 1
                queue.append((nr, nc))

    return depth


def _paint_layers(mask: np.ndarray) -> np.ndarray:
    depth = _layer_depths(mask)
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask & (depth == 0)] = KEEP_COLOR
    patch[mask & (depth > 0) & (depth % 2 == 1)] = ODD_COLOR
    patch[mask & (depth > 0) & (depth % 2 == 0)] = EVEN_COLOR
    return patch


def _classify_green_components(grid: np.ndarray):
    teal_components = components_mask(grid == KEEP_COLOR, connectivity=4)
    assert len(teal_components) == 1, "expected exactly one teal reference component"
    template_mask, _ = _coords_to_mask(teal_components[0])
    variant_keys = {_mask_key(variant) for variant in _dihedral_variants(template_mask)}

    matched = []
    unmatched = []
    for coords in components_mask(grid == MATCH_COLOR, connectivity=4):
        mask, bbox = _coords_to_mask(coords)
        info = (coords, bbox, mask)
        if _mask_key(mask) in variant_keys:
            matched.append(info)
        else:
            unmatched.append(info)
    return template_mask, matched, unmatched


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find every rotated/reflected green copy of the teal template and recolor only those copies by concentric interior depth.

    Input structure:
    - A black background contains one 4-connected teal reference object in a corner.
    - Elsewhere there are several separated 4-connected green objects, some exact rotated/reflected copies of the teal shape and some near-miss distractors.
    - A few isolated maroon noise pixels may also appear on the background.

    Rule:
    - Extract the teal object's bbox-normalized mask as the template.
    - Enumerate all unique rotations and reflections of that template.
    - For each green connected component, keep it only if its bbox-normalized mask matches one template variant exactly.
    - Recolor each kept component by 4-neighbor depth from its own boundary: depth 0 stays teal, odd positive depths become red, and even positive depths become yellow.
    - Remove every nonmatching green object and every maroon noise pixel to black, while leaving the original teal reference unchanged.

    Key insight:
    The task combines shape matching under dihedral symmetry with concentric-layer coloring on an irregular object, so identifying the right components and measuring their internal depth are both necessary.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = np.zeros_like(g)
    out[g == KEEP_COLOR] = KEEP_COLOR

    _template_mask, matched, _unmatched = _classify_green_components(g)
    for _coords, bbox, mask in matched:
        r0, c0, r1, c1 = bbox
        patch = _paint_layers(mask)
        region = out[r0 : r1 + 1, c0 : c1 + 1]
        region[patch != BG] = patch[patch != BG]

    return out


def _is_single_component(mask: np.ndarray) -> bool:
    return len(components_mask(mask.astype(bool), connectivity=4)) == 1


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _template_is_valid(mask: np.ndarray) -> bool:
    area = int(mask.sum())
    if not (TARGET_AREA_RANGE[0] <= area <= TARGET_AREA_RANGE[1]):
        return False
    if not _touches_all_sides(mask):
        return False
    if not _is_single_component(mask):
        return False
    if int(_layer_depths(mask).max()) < 2:
        return False
    if len(_dihedral_variants(mask)) < 4:
        return False
    return True


def _frontier_cells(mask: np.ndarray) -> list[Coord]:
    h, w = mask.shape
    cells: list[Coord] = []
    for r in range(h):
        for c in range(w):
            if mask[r, c]:
                continue
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                    cells.append((r, c))
                    break
    return cells


def _generate_template_mask() -> np.ndarray:
    for _ in range(200):
        target_area = random.randint(*TARGET_AREA_RANGE)
        mask = CORE_MASK.copy()
        while int(mask.sum()) < target_area:
            frontier = _frontier_cells(mask)
            if not frontier:
                break
            rr, cc = random.choice(frontier)
            mask[rr, cc] = True
        if _template_is_valid(mask):
            return mask
    raise RuntimeError("failed to generate a valid template mask")


def _candidate_remove_masks(mask: np.ndarray) -> list[np.ndarray]:
    candidates: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        trial = mask.copy()
        trial[int(r), int(c)] = False
        if int(trial.sum()) < TARGET_AREA_RANGE[0]:
            continue
        if _touches_all_sides(trial) and _is_single_component(trial):
            candidates.append(trial)
    return candidates


def _candidate_add_masks(mask: np.ndarray) -> list[np.ndarray]:
    if int(mask.sum()) >= TARGET_AREA_RANGE[1]:
        return []
    candidates: list[np.ndarray] = []
    for r, c in _frontier_cells(mask):
        trial = mask.copy()
        trial[r, c] = True
        if _touches_all_sides(trial):
            candidates.append(trial)
    return candidates


def _candidate_move_masks(mask: np.ndarray) -> list[np.ndarray]:
    candidates: list[np.ndarray] = []
    for removed in _candidate_remove_masks(mask):
        for r, c in _frontier_cells(removed):
            if removed[r, c]:
                continue
            trial = removed.copy()
            trial[r, c] = True
            if _touches_all_sides(trial) and _is_single_component(trial):
                candidates.append(trial)
    return candidates


def _generate_distractor_mask(
    template_mask: np.ndarray,
    template_keys: set[tuple[tuple[int, ...], ...]],
    used_keys: set[tuple[tuple[int, ...], ...]],
    preferred: str | None = None,
) -> np.ndarray:
    strategies = ["move", "remove", "add"]
    if preferred in strategies:
        strategies.remove(preferred)
        strategies.insert(0, preferred)

    for strategy in strategies:
        if strategy == "move":
            candidates = _candidate_move_masks(template_mask)
        elif strategy == "remove":
            candidates = _candidate_remove_masks(template_mask)
        else:
            candidates = _candidate_add_masks(template_mask)
        random.shuffle(candidates)
        for candidate in candidates:
            key = _mask_key(candidate)
            if key in template_keys or key in used_keys:
                continue
            used_keys.add(key)
            variants = _dihedral_variants(candidate)
            return random.choice(variants)
    raise RuntimeError("failed to generate a distractor mask")


def _color_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = int(color)
    return patch


def _can_place(blocked: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = PLACEMENT_PADDING) -> bool:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(blocked.shape[0], r0 + h + pad)
    cc1 = min(blocked.shape[1], c0 + w + pad)
    return not bool(blocked[rr0:rr1, cc0:cc1].any())


def _reserve(blocked: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = PLACEMENT_PADDING) -> None:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(blocked.shape[0], r0 + h + pad)
    cc1 = min(blocked.shape[1], c0 + w + pad)
    blocked[rr0:rr1, cc0:cc1] = True


def _place_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> set[Coord]:
    h, w = patch.shape
    region = grid[r0 : r0 + h, c0 : c0 + w]
    mask = patch != BG
    assert not np.any(region[mask] != BG)
    region[mask] = patch[mask]
    cells = {(r0 + int(r), c0 + int(c)) for r, c in np.argwhere(mask)}
    return cells


def _corner_positions(height: int, width: int, h: int, w: int) -> list[Coord]:
    return [(0, 0), (0, width - w), (height - h, 0), (height - h, width - w)]


def _random_position(blocked: np.ndarray, h: int, w: int) -> Coord | None:
    candidates = [
        (r, c)
        for r in range(blocked.shape[0] - h + 1)
        for c in range(blocked.shape[1] - w + 1)
        if _can_place(blocked, r, c, h, w)
    ]
    if not candidates:
        return None
    return random.choice(candidates)


def _random_noise_cell(grid: np.ndarray, blocked: np.ndarray) -> Coord | None:
    candidates = [
        (r, c)
        for r in range(grid.shape[0])
        for c in range(grid.shape[1])
        if grid[r, c] == BG and _can_place(blocked, r, c, 1, 1)
    ]
    if not candidates:
        return None
    return random.choice(candidates)


def _expected_output(
    shape: tuple[int, int],
    ref_patch: np.ndarray,
    ref_pos: Coord,
    match_specs: list[tuple[np.ndarray, Coord]],
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    _place_patch(out, ref_patch, ref_pos[0], ref_pos[1])
    for mask, (r0, c0) in match_specs:
        patch = _paint_layers(mask)
        _place_patch(out, patch, r0, c0)
    return out


def _attempt_example() -> np.ndarray | None:
    height = random.randint(GRID_MIN, GRID_MAX)
    width = random.randint(GRID_MIN, GRID_MAX)
    template_mask = _generate_template_mask()
    variants = _dihedral_variants(template_mask)
    template_keys = {_mask_key(variant) for variant in variants}

    reference_mask = random.choice(variants)
    ref_patch = _color_patch(reference_mask, KEEP_COLOR)
    ref_h, ref_w = ref_patch.shape

    grid = np.zeros((height, width), dtype=int)
    blocked = np.zeros((height, width), dtype=bool)

    ref_pos = random.choice(_corner_positions(height, width, ref_h, ref_w))
    ref_cells = _place_patch(grid, ref_patch, ref_pos[0], ref_pos[1])
    _reserve(blocked, ref_pos[0], ref_pos[1], ref_h, ref_w)

    match_count = random.choice(MATCH_COUNTS)
    distractor_count = random.choice(DISTRACTOR_COUNTS)

    match_specs: list[tuple[np.ndarray, Coord]] = []
    match_cells: set[Coord] = set()
    variant_pool = variants[:]
    random.shuffle(variant_pool)
    while len(variant_pool) < match_count:
        variant_pool.append(random.choice(variants))
    if len(variants) > 1 and all(_mask_key(v) == _mask_key(reference_mask) for v in variant_pool[:match_count]):
        variant_pool[0] = random.choice([v for v in variants if _mask_key(v) != _mask_key(reference_mask)])

    for variant in variant_pool[:match_count]:
        patch = _color_patch(variant, MATCH_COLOR)
        pos = _random_position(blocked, patch.shape[0], patch.shape[1])
        if pos is None:
            return None
        cells = _place_patch(grid, patch, pos[0], pos[1])
        _reserve(blocked, pos[0], pos[1], patch.shape[0], patch.shape[1])
        match_specs.append((variant.copy(), pos))
        match_cells |= cells

    distractor_specs: list[tuple[np.ndarray, Coord]] = []
    distractor_cells: set[Coord] = set()
    used_distractor_keys: set[tuple[tuple[int, ...], ...]] = set()
    distractor_preferences = ["move"]
    if distractor_count == 2:
        distractor_preferences.append(random.choice(["remove", "add"]))
    else:
        distractor_preferences[0] = random.choice(["move", "remove", "add"])

    for idx in range(distractor_count):
        preferred = distractor_preferences[min(idx, len(distractor_preferences) - 1)]
        distractor_mask = _generate_distractor_mask(template_mask, template_keys, used_distractor_keys, preferred)
        patch = _color_patch(distractor_mask, MATCH_COLOR)
        pos = _random_position(blocked, patch.shape[0], patch.shape[1])
        if pos is None:
            return None
        cells = _place_patch(grid, patch, pos[0], pos[1])
        _reserve(blocked, pos[0], pos[1], patch.shape[0], patch.shape[1])
        distractor_specs.append((distractor_mask.copy(), pos))
        distractor_cells |= cells

    noise_count = random.randint(*NOISE_RANGE)
    noise_cells: set[Coord] = set()
    for _ in range(noise_count):
        pos = _random_noise_cell(grid, blocked)
        if pos is None:
            return None
        r, c = pos
        grid[r, c] = NOISE_COLOR
        _reserve(blocked, r, c, 1, 1, pad=1)
        noise_cells.add((r, c))

    expected = _expected_output(grid.shape, ref_patch, ref_pos, match_specs)

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    try:
        found_template, found_matches, found_unmatched = _classify_green_components(grid)
    except Exception:
        return None
    if _mask_key(found_template) not in template_keys:
        return None
    if len(found_matches) != match_count or len(found_unmatched) != distractor_count:
        return None

    found_match_cells = set()
    for coords, _bbox, _mask in found_matches:
        for r, c in coords:
            found_match_cells.add((int(r), int(c)))
    found_unmatched_cells = set()
    for coords, _bbox, _mask in found_unmatched:
        for r, c in coords:
            found_unmatched_cells.add((int(r), int(c)))

    if found_match_cells != match_cells:
        return None
    if found_unmatched_cells != distractor_cells:
        return None
    if any(actual[r, c] != BG for r, c in distractor_cells | noise_cells):
        return None
    if any(actual[r, c] != KEEP_COLOR for r, c in ref_cells):
        return None

    output_colors = set(map(int, np.unique(actual)))
    if not {BG, KEEP_COLOR, ODD_COLOR, EVEN_COLOR}.issubset(output_colors):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_example()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle-10 input")
