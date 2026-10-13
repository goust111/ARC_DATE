from __future__ import annotations

import random

import numpy as np


BG = 0
CANDIDATE = 1
PROTOTYPE = 2
COMPLETION = 3

TRANSFORM_ORDER = (
    "rot0",
    "rot90",
    "rot180",
    "rot270",
    "flip_ud",
    "flip_lr",
    "transpose",
    "anti_transpose",
)

INVERSE_TRANSFORM = {
    "rot0": "rot0",
    "rot90": "rot270",
    "rot180": "rot180",
    "rot270": "rot90",
    "flip_ud": "flip_ud",
    "flip_lr": "flip_lr",
    "transpose": "transpose",
    "anti_transpose": "anti_transpose",
}

MATCH_TO_DIRECTION = {
    "rot0": "down",
    "rot180": "down",
    "rot90": "right",
    "rot270": "right",
    "flip_ud": "left",
    "flip_lr": "up",
    # The puzzle examples are generated without diagonal reflections; these
    # deterministic fallbacks keep the solver total on arbitrary valid masks.
    "transpose": "left",
    "anti_transpose": "up",
}

BRANCH_TRANSFORMS = {
    "down": ("rot0", "rot180"),
    "right": ("rot90", "rot270"),
    "left": ("flip_ud",),
    "up": ("flip_lr",),
}

MAX_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_component(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask, (r0, c0, r1, c1)


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in mask.astype(int))


def _crop_binary(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _apply_transform(mask: np.ndarray, name: str) -> np.ndarray:
    if name == "rot0":
        return mask.copy()
    if name == "rot90":
        return np.rot90(mask, 1)
    if name == "rot180":
        return np.rot90(mask, 2)
    if name == "rot270":
        return np.rot90(mask, 3)
    if name == "flip_ud":
        return np.flipud(mask)
    if name == "flip_lr":
        return np.fliplr(mask)
    if name == "transpose":
        return mask.T.copy()
    if name == "anti_transpose":
        return np.fliplr(np.flipud(mask)).T.copy()
    raise ValueError(f"unknown transform: {name}")


def _match_transform(candidate_mask: np.ndarray, prototype_mask: np.ndarray) -> str | None:
    for name in TRANSFORM_ORDER:
        transformed = _apply_transform(candidate_mask, name)
        if transformed.shape == prototype_mask.shape and np.array_equal(transformed, prototype_mask):
            return name
    return None


def _paint_completion(
    out: np.ndarray,
    bbox: tuple[int, int, int, int],
    candidate_mask: np.ndarray,
    direction: str,
) -> None:
    r0, c0, r1, c1 = bbox
    h, w = candidate_mask.shape
    if direction == "down":
        mirror = np.flipud(candidate_mask)
        tr, tc = r1 + 1, c0
    elif direction == "up":
        mirror = np.flipud(candidate_mask)
        tr, tc = r0 - h, c0
    elif direction == "right":
        mirror = np.fliplr(candidate_mask)
        tr, tc = r0, c1 + 1
    elif direction == "left":
        mirror = np.fliplr(candidate_mask)
        tr, tc = r0, c0 - w
    else:
        raise ValueError(f"unknown completion direction: {direction}")

    for rr, cc in np.argwhere(mirror):
        gr = tr + int(rr)
        gc = tc + int(cc)
        if 0 <= gr < out.shape[0] and 0 <= gc < out.shape[1] and out[gr, gc] == BG:
            out[gr, gc] = COMPLETION


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match candidate shapes to the red prototype, then mirror-complete them in a direction encoded by the matching transform.

    Input structure:
    - The grid has a black background, one small red 4-connected prototype near the upper-left area, and several blue 4-connected candidate objects elsewhere.
    - Each object is read through its tight bounding box; 1-2 blue candidates are rotation/reflection-equivalent to the prototype and the rest are distractors.

    Rule:
    - Extract the red prototype mask and compare every blue candidate mask against it under the eight rotation/reflection transforms.
    - Leave unmatched blue objects unchanged.
    - For each matched candidate, keep the original blue pixels and add a green mirrored copy adjacent to it:
      0/180-degree matches complete downward, 90/270-degree matches complete to the right,
      horizontal-reflection matches complete to the left, and vertical-reflection matches complete upward.
    - Clip any added green cells that would fall outside the grid. The red prototype stays unchanged.

    Key insight:
    - The useful signal is not only which candidate matches the prototype, but which transform makes it match; that intermediate transform choice directly selects the completion direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    prototype_components = _components(g == PROTOTYPE)
    assert len(prototype_components) == 1, "expected exactly one prototype object"
    prototype_mask, _ = _crop_component(prototype_components[0])

    out = g.copy()
    for coords in _components(g == CANDIDATE):
        candidate_mask, bbox = _crop_component(coords)
        match_name = _match_transform(candidate_mask, prototype_mask)
        if match_name is None:
            continue
        _paint_completion(out, bbox, candidate_mask, MATCH_TO_DIRECTION[match_name])
    return out


def _expanded_rect(rect: tuple[int, int, int, int], pad: int, h: int, w: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = rect
    return (
        max(0, r0 - pad),
        max(0, c0 - pad),
        min(h - 1, r1 + pad),
        min(w - 1, c1 + pad),
    )


def _reserve(blocked: np.ndarray, rect: tuple[int, int, int, int], pad: int) -> None:
    rr = _expanded_rect(rect, pad, *blocked.shape)
    blocked[rr[0] : rr[2] + 1, rr[1] : rr[3] + 1] = True


def _random_connected_mask(min_cells: int = 3, max_cells: int = 6) -> np.ndarray:
    for _ in range(200):
        h = random.randint(2, 4)
        w = random.randint(2, 4)
        area = h * w
        if area < min_cells:
            continue
        target = random.randint(min_cells, min(max_cells, area))
        mask = np.zeros((h, w), dtype=bool)
        seed = (random.randrange(h), random.randrange(w))
        mask[seed] = True
        frontier = set()
        frontier.update((nr, nc) for nr, nc in _neighbors4(*seed) if 0 <= nr < h and 0 <= nc < w)
        while int(mask.sum()) < target and frontier:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            mask[cell] = True
            for nr, nc in _neighbors4(*cell):
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.add((nr, nc))
        if int(mask.sum()) != target:
            continue
        cropped = _crop_binary(mask)
        if 2 <= cropped.shape[0] <= 4 and 2 <= cropped.shape[1] <= 4:
            return cropped
    raise RuntimeError("failed to generate connected mask")


def _all_transform_keys(mask: np.ndarray) -> list[tuple[tuple[int, ...], ...]]:
    return [_mask_key(_apply_transform(mask, name)) for name in TRANSFORM_ORDER]


def _random_prototype() -> np.ndarray:
    for _ in range(300):
        mask = _random_connected_mask(4, 6)
        keys = _all_transform_keys(mask)
        if len(set(keys)) == len(TRANSFORM_ORDER):
            return mask
    raise RuntimeError("failed to generate asymmetric prototype")


def _random_distractor(prototype_signatures: set[tuple[tuple[int, ...], ...]]) -> np.ndarray:
    for _ in range(300):
        mask = _random_connected_mask(3, 6)
        if all(key not in prototype_signatures for key in _all_transform_keys(mask)):
            return mask
    raise RuntimeError("failed to generate distractor")


def _extension_rect(
    top: tuple[int, int],
    mask: np.ndarray,
    direction: str,
) -> tuple[int, int, int, int]:
    r0, c0 = top
    h, w = mask.shape
    if direction == "down":
        return r0 + h, c0, r0 + 2 * h - 1, c0 + w - 1
    if direction == "up":
        return r0 - h, c0, r0 - 1, c0 + w - 1
    if direction == "right":
        return r0, c0 + w, r0 + h - 1, c0 + 2 * w - 1
    if direction == "left":
        return r0, c0 - w, r0 + h - 1, c0 - 1
    raise ValueError(f"unknown direction: {direction}")


def _fits_rect(rect: tuple[int, int, int, int], h: int, w: int) -> bool:
    r0, c0, r1, c1 = rect
    return 0 <= r0 <= r1 < h and 0 <= c0 <= c1 < w


def _place_object(
    canvas: np.ndarray,
    blocked: np.ndarray,
    mask: np.ndarray,
    color: int,
    *,
    direction: str | None = None,
    row_range: range | None = None,
    col_range: range | None = None,
) -> tuple[int, int] | None:
    h, w = canvas.shape
    mh, mw = mask.shape
    rows = list(row_range if row_range is not None else range(0, h - mh + 1))
    cols = list(col_range if col_range is not None else range(0, w - mw + 1))
    candidates = [(r0, c0) for r0 in rows for c0 in cols if r0 + mh <= h and c0 + mw <= w]
    random.shuffle(candidates)
    for r0, c0 in candidates:
        rect = (r0, c0, r0 + mh - 1, c0 + mw - 1)
        if blocked[rect[0] : rect[2] + 1, rect[1] : rect[3] + 1].any():
            continue
        if direction is not None:
            ext = _extension_rect((r0, c0), mask, direction)
            if not _fits_rect(ext, h, w):
                continue
            if blocked[ext[0] : ext[2] + 1, ext[1] : ext[3] + 1].any():
                continue
        canvas[r0 : r0 + mh, c0 : c0 + mw][mask] = color
        _reserve(blocked, rect, 3)
        return r0, c0
    return None


def _analyze_scene(grid: np.ndarray) -> tuple[np.ndarray, list[str], int]:
    prototype_components = _components(grid == PROTOTYPE)
    if len(prototype_components) != 1:
        raise ValueError("scene has invalid prototype count")
    prototype_mask, _ = _crop_component(prototype_components[0])
    transforms: list[str] = []
    blue_count = 0
    for coords in _components(grid == CANDIDATE):
        blue_count += 1
        candidate_mask, _ = _crop_component(coords)
        match = _match_transform(candidate_mask, prototype_mask)
        if match is not None:
            transforms.append(match)
    return prototype_mask, transforms, blue_count


def _build_attempt() -> np.ndarray | None:
    h = random.randint(18, 22)
    w = random.randint(18, 22)
    canvas = np.zeros((h, w), dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    prototype_mask = _random_prototype()
    proto_h, proto_w = prototype_mask.shape
    prototype_pos = _place_object(
        canvas,
        blocked,
        prototype_mask,
        PROTOTYPE,
        row_range=range(1, min(5, h - proto_h + 1)),
        col_range=range(1, min(5, w - proto_w + 1)),
    )
    if prototype_pos is None:
        return None

    match_count = random.choice([1, 2, 2])
    candidate_count = random.choice([3, 4])
    if candidate_count < match_count + 1:
        candidate_count = match_count + 1
    distractor_count = candidate_count - match_count

    chosen_branches = random.sample(list(BRANCH_TRANSFORMS), k=match_count)
    chosen_transforms = [random.choice(BRANCH_TRANSFORMS[branch]) for branch in chosen_branches]
    random.shuffle(chosen_transforms)

    for match_name in chosen_transforms:
        candidate_mask = _apply_transform(prototype_mask, INVERSE_TRANSFORM[match_name])
        direction = MATCH_TO_DIRECTION[match_name]
        pos = _place_object(canvas, blocked, candidate_mask, CANDIDATE, direction=direction)
        if pos is None:
            return None

    prototype_signatures = set(_all_transform_keys(prototype_mask))
    for _ in range(distractor_count):
        distractor = _random_distractor(prototype_signatures)
        pos = _place_object(canvas, blocked, distractor, CANDIDATE)
        if pos is None:
            return None

    _, transforms, blue_count = _analyze_scene(canvas)
    if blue_count != candidate_count:
        return None
    if len(transforms) != match_count:
        return None
    if len(set(transforms)) != len(transforms):
        return None
    if set(transforms) != set(chosen_transforms):
        return None

    out = main(canvas)
    if not np.any(out == COMPLETION):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
