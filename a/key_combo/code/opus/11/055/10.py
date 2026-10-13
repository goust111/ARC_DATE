from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
FG = 5
OUT_FG = 2
NEIGHBORS8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]

# Generator configuration.
GEN_ATTEMPTS = 500
GEN_COUNTER = 0
GRID_SIDE_RANGE = (15, 20)  # inclusive low, exclusive high for randint
OBJECT_COUNT_RANGE = (5, 8)
AREA_RANGE = (3, 8)
BBOX_SIDE_RANGE = (2, 5)
PLACEMENT_TRIES = 250
SHAPE_TRIES = 400


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 8-connected components from a boolean mask as (N,2) coord arrays."""
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS8:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _shape_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    m = np.asarray(mask, dtype=bool)
    return tuple(map(int, m.shape)), m.tobytes()


def _dihedral_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    variants = []
    base = np.asarray(mask, dtype=bool)
    for k in range(4):
        rot = np.rot90(base, k)
        variants.append(_shape_key(rot))
        variants.append(_shape_key(np.fliplr(rot)))
    return min(variants)


def _analyze(grid: np.ndarray) -> tuple[list[dict], Counter, int, dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert max(g.shape) <= 30, "ARC grids must fit within 30x30"
    assert np.all((g == BG) | (g == FG)), "expected only black background and gray objects"

    comps = _find_components(g == FG)
    assert comps, "expected at least one foreground object"

    items: list[dict] = []
    for coords in comps:
        bbox = _bbox_from_coords(coords)
        patch = _crop_bbox(g, bbox)
        mask = patch == FG
        items.append(
            {
                "coords": coords,
                "bbox": bbox,
                "patch": patch,
                "mask": mask,
                "shape_key": _shape_key(mask),
                "area": int(mask.sum()),
            }
        )

    freq = Counter(item["shape_key"] for item in items)
    singletons = [key for key, count in freq.items() if count == 1]
    assert len(singletons) == 1, "expected exactly one singleton shape"
    scale = len(freq)
    assert scale in (2, 3), "expected exactly 2 or 3 shape kinds"

    unique_item = next(item for item in items if freq[item["shape_key"]] == 1)
    return items, freq, scale, unique_item


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the uniquely occurring shape and enlarge it by the number of distinct shape kinds.

    Input structure:
    - The grid has a black background and several gray 8-connected foreground objects.
    - Objects are compared by their normalized binary bbox masks, up to translation only.
    - Exactly one normalized shape occurs once, while every other shape occurs at least twice.

    Rule:
    - Split the gray pixels into 8-connected objects and crop each object to its bbox.
    - Group objects by the cropped binary mask and count how many distinct shape kinds exist.
    - Select the object whose shape kind occurs exactly once.
    - Recolor that cropped object from gray to red, keep cropped background cells black,
      then scale the cropped patch by S where S is the number of distinct shape kinds.

    Key insight:
    - The same shape-frequency analysis drives two different decisions: which object to crop
      and the scaling factor used on that crop.
    """
    _items, _freq, scale, unique_item = _analyze(grid)
    cropped = np.where(unique_item["patch"] == FG, OUT_FG, BG)
    return np.kron(cropped, np.ones((scale, scale), dtype=int))


def _shape_bbox(mask: np.ndarray) -> tuple[int, int]:
    return int(mask.shape[0]), int(mask.shape[1])


def _has_internal_background(mask: np.ndarray) -> bool:
    return bool((~mask).any())


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


def _random_path_mask(h: int, w: int, area: int) -> np.ndarray:
    min_path_len = max(h, w)
    max_path_len = min(area, h + w - 1)
    if max_path_len < min_path_len:
        raise ValueError("infeasible bbox/area pair")

    path_len = random.randint(min_path_len, max_path_len)
    diag_count = h + w - 1 - path_len
    steps = (
        [(1, 1)] * diag_count
        + [(1, 0)] * (h - 1 - diag_count)
        + [(0, 1)] * (w - 1 - diag_count)
    )
    random.shuffle(steps)

    mask = np.zeros((h, w), dtype=bool)
    r = c = 0
    mask[r, c] = True
    for dr, dc in steps:
        r += dr
        c += dc
        mask[r, c] = True

    while int(mask.sum()) < area:
        frontier: set[tuple[int, int]] = set()
        for rr, cc in np.argwhere(mask):
            for dr, dc in NEIGHBORS8:
                nr = int(rr) + dr
                nc = int(cc) + dc
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.add((nr, nc))
        if not frontier:
            break
        nr, nc = random.choice(tuple(frontier))
        mask[nr, nc] = True

    if random.random() < 0.5:
        mask = np.flipud(mask)
    if random.random() < 0.5:
        mask = np.fliplr(mask)
    return mask


def _make_shape(
    *,
    forbidden_keys: set[tuple[tuple[int, int], bytes]],
    forbidden_dihedrals: set[tuple[tuple[int, int], bytes]],
    forced_area: int | None = None,
    forced_bbox: tuple[int, int] | None = None,
    require_hole: bool = False,
) -> np.ndarray | None:
    for _ in range(SHAPE_TRIES):
        if forced_bbox is None:
            h = random.randint(*BBOX_SIDE_RANGE)
            w = random.randint(*BBOX_SIDE_RANGE)
        else:
            h, w = forced_bbox

        min_area = max(AREA_RANGE[0], max(h, w))
        max_area = min(AREA_RANGE[1], h * w)
        if require_hole:
            max_area = min(max_area, h * w - 1)
        if forced_area is not None:
            area = forced_area
            if not (min_area <= area <= max_area):
                continue
        else:
            if min_area > max_area:
                continue
            area = random.randint(min_area, max_area)

        try:
            mask = _random_path_mask(h, w, area)
        except ValueError:
            continue
        if require_hole and not _has_internal_background(mask):
            continue

        key = _shape_key(mask)
        dihedral = _dihedral_key(mask)
        if key in forbidden_keys or dihedral in forbidden_dihedrals:
            continue
        return mask
    return None


def _choose_shape_counts(scale: int, n_objects: int) -> list[int]:
    if scale == 2:
        return [1, n_objects - 1]
    pairs = [(a, n_objects - 1 - a) for a in range(2, n_objects - 2) if n_objects - 1 - a >= 2]
    a, b = random.choice(pairs)
    return [1, a, b]


def _build_shape_templates(case_id: int, scale: int) -> list[np.ndarray] | None:
    forbidden_keys: set[tuple[tuple[int, int], bytes]] = set()
    forbidden_dihedrals: set[tuple[tuple[int, int], bytes]] = set()

    need_hole = case_id in (0, 2)
    unique_mask = _make_shape(
        forbidden_keys=forbidden_keys,
        forbidden_dihedrals=forbidden_dihedrals,
        require_hole=need_hole,
    )
    if unique_mask is None:
        return None

    templates = [unique_mask]
    forbidden_keys.add(_shape_key(unique_mask))
    forbidden_dihedrals.add(_dihedral_key(unique_mask))
    unique_area = int(unique_mask.sum())
    unique_bbox = _shape_bbox(unique_mask)

    for idx in range(1, scale):
        force_area = unique_area if (idx == 1 and case_id in (1, 3)) else None
        force_bbox = unique_bbox if (idx == 1 and case_id in (0, 2)) else None
        mask = _make_shape(
            forbidden_keys=forbidden_keys,
            forbidden_dihedrals=forbidden_dihedrals,
            forced_area=force_area,
            forced_bbox=force_bbox,
            require_hole=False,
        )
        if mask is None:
            return None
        templates.append(mask)
        forbidden_keys.add(_shape_key(mask))
        forbidden_dihedrals.add(_dihedral_key(mask))
    return templates


def _render_patches(templates: list[np.ndarray], counts: list[int]) -> list[np.ndarray]:
    patches: list[np.ndarray] = []
    for mask, count in zip(templates, counts):
        patch = np.where(mask, FG, BG).astype(int)
        for _ in range(count):
            patches.append(patch)
    random.shuffle(patches)
    return patches


def _place_patches(patches: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray | None:
    h, w = shape
    grid = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    order = sorted(range(len(patches)), key=lambda i: int((patches[i] == FG).sum()), reverse=True)
    for idx in order:
        patch = patches[idx]
        ph, pw = patch.shape
        positions = [(r, c) for r in range(h - ph + 1) for c in range(w - pw + 1)]
        random.shuffle(positions)
        placed = False
        dilated = _dilate8(occupied)
        for r0, c0 in positions[:PLACEMENT_TRIES]:
            region = dilated[r0 : r0 + ph, c0 : c0 + pw]
            mask = patch == FG
            if bool((region & mask).any()):
                continue
            grid[r0 : r0 + ph, c0 : c0 + pw][mask] = FG
            occupied[r0 : r0 + ph, c0 : c0 + pw] |= mask
            placed = True
            break
        if not placed:
            return None
    return grid


def _case_requirements_ok(case_id: int, scale: int, items: list[dict], freq: Counter, unique_item: dict) -> bool:
    if not (5 <= len(items) <= 8):
        return False
    if scale not in (2, 3):
        return False
    repeated_counts = sorted(count for count in freq.values() if count > 1)
    if not repeated_counts or min(repeated_counts) < 2:
        return False

    unique_bbox = _shape_bbox(unique_item["mask"])
    if max(unique_bbox) > 5:
        return False
    if not all(AREA_RANGE[0] <= item["area"] <= AREA_RANGE[1] for item in items):
        return False

    repeated_items = [item for item in items if freq[item["shape_key"]] > 1]
    same_area = any(item["area"] == unique_item["area"] for item in repeated_items)
    same_bbox = any(_shape_bbox(item["mask"]) == unique_bbox for item in repeated_items)
    has_hole = _has_internal_background(unique_item["mask"])

    if case_id in (0, 2) and not same_bbox:
        return False
    if case_id in (1, 3) and not same_area:
        return False
    if case_id in (0, 2) and not has_hole:
        return False
    return True


def _generate_candidate(case_id: int) -> np.ndarray | None:
    scale = 2 if case_id in (0, 3) else 3
    n_objects = random.randint(*OBJECT_COUNT_RANGE)
    if scale == 3:
        n_objects = max(n_objects, 5)
    counts = _choose_shape_counts(scale, n_objects)
    templates = _build_shape_templates(case_id, scale)
    if templates is None:
        return None

    grid_shape = (
        random.randint(*GRID_SIDE_RANGE),
        random.randint(*GRID_SIDE_RANGE),
    )
    grid = _place_patches(_render_patches(templates, counts), grid_shape)
    if grid is None:
        return None

    try:
        items, freq, actual_scale, unique_item = _analyze(grid)
        out = main(grid)
    except AssertionError:
        return None

    if actual_scale != scale:
        return None
    if not _case_requirements_ok(case_id, actual_scale, items, freq, unique_item):
        return None

    uh, uw = unique_item["patch"].shape
    if out.shape != (uh * actual_scale, uw * actual_scale):
        return None
    if max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    global GEN_COUNTER
    case_id = GEN_COUNTER % 4
    GEN_COUNTER += 1

    for _ in range(GEN_ATTEMPTS):
        grid = _generate_candidate(case_id)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid puzzle instance for case {case_id}")
