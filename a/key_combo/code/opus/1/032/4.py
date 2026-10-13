from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
INPUT_COLORS = (3, 4, 5, 6, 7, 8, 9)
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_ATTEMPTS = 500
LOCAL_ATTEMPTS = 200


def _find_components(mask: np.ndarray) -> list[dict[str, object]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            arr = np.asarray(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            c0 = int(arr[:, 1].min())
            r1 = int(arr[:, 0].max())
            c1 = int(arr[:, 1].max())
            components.append({"coords": arr, "bbox": (r0, c0, r1, c1)})
    return components


def _normalize_coords(coords: np.ndarray | list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> frozenset[tuple[int, int]]:
    arr = np.asarray(coords, dtype=int)
    if arr.size == 0:
        return frozenset()
    r0 = int(arr[:, 0].min())
    c0 = int(arr[:, 1].min())
    return frozenset((int(r - r0), int(c - c0)) for r, c in arr)


def _mask_dims(mask: frozenset[tuple[int, int]]) -> tuple[int, int]:
    return max(r for r, _ in mask) + 1, max(c for _, c in mask) + 1


def _rotate_cw(mask: frozenset[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    return _normalize_coords([(c, -r) for r, c in mask])


def _rotation_sequence(mask: frozenset[tuple[int, int]]) -> list[frozenset[tuple[int, int]]]:
    seq = []
    cur = mask
    for _ in range(4):
        seq.append(cur)
        cur = _rotate_cw(cur)
    return seq


def _rotation_set(mask: frozenset[tuple[int, int]]) -> set[frozenset[tuple[int, int]]]:
    return set(_rotation_sequence(mask))


def _shape_occupies_both_halves(mask: frozenset[tuple[int, int]]) -> bool:
    h, _ = _mask_dims(mask)
    split = (h + 1) // 2
    has_upper = any(r < split for r, _ in mask)
    has_lower = any(r >= split for r, _ in mask)
    return has_upper and has_lower


def _canonical_shape_key(mask: frozenset[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    variants = [tuple(sorted(rot)) for rot in _rotation_set(mask)]
    return min(variants)


def _random_connected_mask(
    rng: random.Random,
    bbox_h: int,
    bbox_w: int,
    *,
    size_min: int,
    size_max: int,
    exact_size: int | None = None,
    require_full_bbox: bool = True,
) -> frozenset[tuple[int, int]] | None:
    for _ in range(LOCAL_ATTEMPTS):
        size = exact_size if exact_size is not None else rng.randint(size_min, size_max)
        cells = {(rng.randrange(bbox_h), rng.randrange(bbox_w))}
        while len(cells) < size:
            frontier = []
            for r, c in cells:
                for dr, dc in DIRS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < bbox_h and 0 <= nc < bbox_w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(rng.choice(frontier))
        if len(cells) != size:
            continue
        mask = _normalize_coords(tuple(cells))
        h, w = _mask_dims(mask)
        if require_full_bbox and (h != bbox_h or w != bbox_w):
            continue
        if h < 2 or w < 2:
            continue
        if len(mask) == h * w:
            continue
        if not _shape_occupies_both_halves(mask):
            continue
        return mask
    return None


def _random_template_mask(rng: random.Random) -> frozenset[tuple[int, int]] | None:
    bbox_options = [(3, 3), (3, 4), (4, 3), (4, 4)]
    for _ in range(LOCAL_ATTEMPTS):
        bbox_h, bbox_w = rng.choice(bbox_options)
        mask = _random_connected_mask(
            rng,
            bbox_h,
            bbox_w,
            size_min=4,
            size_max=6,
            require_full_bbox=True,
        )
        if mask is None:
            continue
        if len(_rotation_set(mask)) != 4:
            continue
        return mask
    return None


def _random_nonmatching_mask(
    rng: random.Random,
    template_mask: frozenset[tuple[int, int]],
    mode: str,
    used_keys: set[tuple[tuple[int, int], ...]],
) -> frozenset[tuple[int, int]] | None:
    template_area = len(template_mask)
    template_rots = tuple(_rotation_set(template_mask))
    template_rot_set = set(template_rots)
    rot_dims = [_mask_dims(rot) for rot in template_rots]
    for _ in range(LOCAL_ATTEMPTS):
        if mode == "same_bbox_area":
            bbox_h, bbox_w = rng.choice(rot_dims)
            mask = _random_connected_mask(
                rng,
                bbox_h,
                bbox_w,
                size_min=template_area,
                size_max=template_area,
                exact_size=template_area,
                require_full_bbox=True,
            )
        elif mode == "same_area":
            bbox_h = rng.randint(2, 4)
            bbox_w = rng.randint(2, 4)
            if bbox_h * bbox_w < template_area:
                continue
            mask = _random_connected_mask(
                rng,
                bbox_h,
                bbox_w,
                size_min=template_area,
                size_max=template_area,
                exact_size=template_area,
                require_full_bbox=True,
            )
        else:
            bbox_h = rng.randint(2, 4)
            bbox_w = rng.randint(2, 4)
            max_size = min(8, bbox_h * bbox_w - 1)
            if max_size < 3:
                continue
            mask = _random_connected_mask(
                rng,
                bbox_h,
                bbox_w,
                size_min=3,
                size_max=max_size,
                require_full_bbox=True,
            )
        if mask is None:
            continue
        if mask in template_rot_set:
            continue
        key = _canonical_shape_key(mask)
        if key in used_keys:
            continue
        used_keys.add(key)
        return mask
    return None


def _reserve_bbox(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> None:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(occupied.shape[0], r0 + h + pad)
    cc1 = min(occupied.shape[1], c0 + w + pad)
    occupied[rr0:rr1, cc0:cc1] = True


def _bbox_is_free(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> bool:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(occupied.shape[0], r0 + h + pad)
    cc1 = min(occupied.shape[1], c0 + w + pad)
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _paint_shape(
    grid: np.ndarray,
    occupied: np.ndarray,
    mask: frozenset[tuple[int, int]],
    color: int,
    rng: random.Random,
) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    h, w = _mask_dims(mask)
    candidates = []
    for r0 in range(grid.shape[0] - h + 1):
        for c0 in range(grid.shape[1] - w + 1):
            if _bbox_is_free(occupied, r0, c0, h, w, pad=1):
                candidates.append((r0, c0))
    if not candidates:
        return None
    r0, c0 = rng.choice(candidates)
    coords = np.asarray([(r0 + r, c0 + c) for r, c in sorted(mask)], dtype=int)
    for rr, cc in coords:
        grid[rr, cc] = int(color)
    _reserve_bbox(occupied, r0, c0, h, w, pad=1)
    return coords, (r0, c0, r0 + h - 1, c0 + w - 1)


def _render_expected_from_metadata(grid: np.ndarray, metadata: dict[str, object]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for r, c in metadata["template_coords"]:
        out[r, c] = BG
    for obj in metadata["objects"]:
        r0, _c0, r1, _c1 = obj["bbox"]
        split = r0 + (r1 - r0 + 2) // 2
        for r, c in obj["coords"]:
            if obj["matched"]:
                if r >= split:
                    out[r, c] = RED
            else:
                if r < split:
                    out[r, c] = BLUE
    return out


def _validate_generated_case(grid: np.ndarray, metadata: dict[str, object]) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    components = _find_components(grid != BG)
    if len(components) != 1 + len(metadata["objects"]):
        return False

    template_candidates = [comp for comp in components if comp["bbox"][0] == 0 and comp["bbox"][1] == 0]
    if len(template_candidates) != 1:
        return False
    template_coords = set(map(tuple, template_candidates[0]["coords"]))
    if template_coords != set(metadata["template_coords"]):
        return False

    template_colors = {int(grid[r, c]) for r, c in template_coords}
    if not (2 <= len(template_colors) <= 3):
        return False

    template_mask = _normalize_coords(template_candidates[0]["coords"])
    template_rotations = _rotation_set(template_mask)
    if len(template_rotations) != 4:
        return False

    match_count = 0
    nonmatch_count = 0
    match_colors: set[int] = set()
    nonmatch_colors: set[int] = set()
    nonmatch_same_area = False
    rotated_match_present = False
    covered_cells = set(template_coords)

    for obj in metadata["objects"]:
        coords = [tuple(cell) for cell in obj["coords"]]
        if any(cell in covered_cells for cell in coords):
            return False
        covered_cells.update(coords)
        colors = {int(grid[r, c]) for r, c in coords}
        if len(colors) != 1:
            return False
        color = next(iter(colors))
        if color not in INPUT_COLORS:
            return False

        bbox = obj["bbox"]
        r0, c0, r1, c1 = bbox
        if r1 - r0 + 1 < 2:
            return False
        mask = _normalize_coords(np.asarray(coords, dtype=int))
        if not _shape_occupies_both_halves(mask):
            return False
        is_match = mask in template_rotations
        if is_match != bool(obj["matched"]):
            return False
        if obj["matched"]:
            match_count += 1
            match_colors.add(color)
            if int(obj["rotation"]) % 4 != 0:
                rotated_match_present = True
        else:
            nonmatch_count += 1
            nonmatch_colors.add(color)
            if len(mask) == len(template_mask):
                nonmatch_same_area = True

    if match_count < 2 or nonmatch_count < 2:
        return False
    if not rotated_match_present:
        return False
    if not nonmatch_same_area:
        return False
    if not (match_colors & nonmatch_colors):
        return False

    expected = _render_expected_from_metadata(grid, metadata)
    if not np.any(expected == BLUE) or not np.any(expected == RED):
        return False
    return True


def _build_case_once(rng: random.Random, *, return_metadata: bool = False):
    grid_h = rng.randint(15, 21)
    grid_w = rng.randint(18, 25)
    grid = np.zeros((grid_h, grid_w), dtype=int)
    occupied = np.zeros_like(grid, dtype=bool)

    template_mask = _random_template_mask(rng)
    if template_mask is None:
        return None
    template_h, template_w = _mask_dims(template_mask)
    template_palette = rng.sample(INPUT_COLORS, k=rng.randint(2, 3))
    template_cells = sorted(template_mask)
    rng.shuffle(template_cells)
    for idx, (r, c) in enumerate(template_cells):
        grid[r, c] = template_palette[idx % len(template_palette)]
    _reserve_bbox(occupied, 0, 0, template_h, template_w, pad=1)

    n_objects = rng.randint(4, 6)
    n_match = rng.randint(2, n_objects - 2)
    n_nonmatch = n_objects - n_match

    template_rotations = _rotation_sequence(template_mask)
    forced_rotations = [rng.choice((1, 2, 3))]
    forced_rotations.extend(rng.randrange(4) for _ in range(n_match - 1))

    used_negative_keys: set[tuple[tuple[int, int], ...]] = set()
    negative_modes = ["same_bbox_area", "same_area"]
    negative_modes.extend(["general"] * max(0, n_nonmatch - len(negative_modes)))
    rng.shuffle(negative_modes)

    specs: list[dict[str, object]] = []
    for rot in forced_rotations:
        specs.append({"matched": True, "mask": template_rotations[rot], "rotation": rot})
    for mode in negative_modes:
        mask = _random_nonmatching_mask(rng, template_mask, mode, used_negative_keys)
        if mask is None:
            return None
        specs.append({"matched": False, "mask": mask, "rotation": -1})

    match_indices = [idx for idx, spec in enumerate(specs) if spec["matched"]]
    nonmatch_indices = [idx for idx, spec in enumerate(specs) if not spec["matched"]]
    shared_color = rng.choice(INPUT_COLORS)
    colors = [rng.choice(INPUT_COLORS) for _ in specs]
    colors[rng.choice(match_indices)] = shared_color
    colors[rng.choice(nonmatch_indices)] = shared_color
    for spec, color in zip(specs, colors):
        spec["color"] = int(color)

    specs.sort(key=lambda spec: _mask_dims(spec["mask"])[0] * _mask_dims(spec["mask"])[1], reverse=True)

    objects: list[dict[str, object]] = []
    for spec in specs:
        placed = _paint_shape(grid, occupied, spec["mask"], spec["color"], rng)
        if placed is None:
            return None
        coords, bbox = placed
        objects.append(
            {
                "coords": tuple(map(tuple, coords.tolist())),
                "bbox": bbox,
                "matched": bool(spec["matched"]),
                "rotation": int(spec["rotation"]),
                "color": int(spec["color"]),
            }
        )

    metadata = {
        "template_coords": tuple((int(r), int(c)) for r, c in sorted(template_mask)),
        "objects": objects,
    }
    if not _validate_generated_case(grid, metadata):
        return None

    if return_metadata:
        return grid, metadata
    return grid


def _build_case(rng: random.Random, *, return_metadata: bool = False):
    for _ in range(40):
        case = _build_case_once(rng, return_metadata=return_metadata)
        if case is not None:
            return case
    return None


def generate_input() -> np.ndarray:
    for seed in range(GEN_ATTEMPTS):
        rng = random.Random(random.randrange(1 << 30) ^ seed)
        case = _build_case(rng, return_metadata=False)
        if case is not None:
            return case
    raise RuntimeError("failed to generate a valid input for puzzle 4")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the top-left template shape as a rotation-aware classifier, then recolor each object's upper or lower half by class.

    Input structure:
    - The background is black.
    - A multicolor 4-connected template object has its bounding box anchored at the top-left corner.
    - The remaining foreground objects are separated 4-connected shapes whose colors are irrelevant to the rule.

    Rule:
    - Extract the template shape mask from its bounding box, ignoring template colors, and enumerate its 0/90/180/270 degree rotations.
    - For each non-template object, extract its normalized shape mask and test whether it exactly matches one rotated template mask.
    - If an object matches, recolor only the foreground pixels in the lower half of its bounding box red.
    - If an object does not match, recolor only the foreground pixels in the upper half of its bounding box blue.
    - Erase the template object itself to background.

    Key insight:
    - The template is not copied anywhere; it acts as a binary shape classifier, and the class is encoded by which half of each object changes color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    components = _find_components(g != BG)
    template_candidates = [comp for comp in components if comp["bbox"][0] == 0 and comp["bbox"][1] == 0]
    assert len(template_candidates) == 1, "expected exactly one template anchored at the top-left corner"

    template = template_candidates[0]
    template_coords = set(map(tuple, template["coords"]))
    template_mask = _normalize_coords(template["coords"])
    template_rotations = _rotation_set(template_mask)

    out = g.copy()
    for r, c in template_coords:
        out[r, c] = BG

    for comp in components:
        coords = [tuple(cell) for cell in comp["coords"]]
        if set(coords) == template_coords:
            continue
        mask = _normalize_coords(comp["coords"])
        r0, _c0, r1, _c1 = comp["bbox"]
        split = r0 + (r1 - r0 + 2) // 2
        if mask in template_rotations:
            for r, c in coords:
                if r >= split:
                    out[r, c] = RED
        else:
            for r, c in coords:
                if r < split:
                    out[r, c] = BLUE
    return out
