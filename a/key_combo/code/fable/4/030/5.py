import random
from collections import Counter, deque

import numpy as np


BG = 0
GREY = 5
CORNER_ORDER = ("tl", "tr", "bl", "br")


def _signature(arr: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(map(int, arr.shape)), arr.tobytes()


def _orient_contract(arr: np.ndarray, corner: str) -> np.ndarray:
    if corner == "tl":
        return arr.copy()
    if corner == "tr":
        return np.fliplr(arr).copy()
    if corner == "bl":
        return np.flipud(arr).copy()
    if corner == "br":
        return np.flipud(np.fliplr(arr)).copy()
    raise ValueError(f"unknown corner: {corner}")


def _all_orientations(arr: np.ndarray) -> list[np.ndarray]:
    rots = [np.rot90(arr, k).copy() for k in range(4)]
    flips = [np.fliplr(r).copy() for r in rots]
    return rots + flips


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _frame_boxes(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    boxes = []
    for comp in _components(grid == GREY):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros_like(sub, dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if np.array_equal(sub == GREY, border):
            boxes.append((r0, c0, r1, c1))
    boxes.sort()
    assert boxes, "expected at least one compartment frame"
    return boxes


def _marker_corner(grid: np.ndarray, box: tuple[int, int, int, int], bg: int) -> str:
    r0, c0, r1, c1 = box
    h, w = grid.shape
    probes = {
        "tl": (r0 - 1, c0 - 1),
        "tr": (r0 - 1, c1 + 1),
        "bl": (r1 + 1, c0 - 1),
        "br": (r1 + 1, c1 + 1),
    }
    hits = []
    for corner, (r, c) in probes.items():
        if 0 <= r < h and 0 <= c < w and grid[r, c] not in (bg, GREY):
            hits.append(corner)
    assert len(hits) == 1, "expected one private corner marker per compartment"
    return hits[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Return the corrected single compartment whose sprite pose violates the corner-marker contract.

    Input structure:
    - The grid is a black canvas holding several equal-size gray rectangular compartments.
    - Each compartment contains the same asymmetric two-color sprite in one rotated or mirrored pose.
    - Exactly one colored marker pixel sits diagonally outside one corner of each compartment.

    Rule:
    - Treat the outside corner marker as that compartment's required pose label.
    - Normalize every sprite back to a shared canonical pose according to its marker corner.
    - Exactly one compartment fails to normalize with the majority; reapply the contract pose for its marker and output only that corrected compartment interior.

    Key insight:
    - Most compartments agree on one hidden canonical sprite, so the unique violator is the only crop whose marker-based deorientation does not match the majority prototype.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = int(Counter(map(int, g.ravel())).most_common(1)[0][0])
    boxes = _frame_boxes(g)

    interior_shapes = {(r1 - r0 - 1, c1 - c0 - 1) for r0, c0, r1, c1 in boxes}
    assert len(interior_shapes) == 1, "compartments must have equal interior sizes"

    compartments: list[tuple[str, np.ndarray]] = []
    for box in boxes:
        r0, c0, r1, c1 = box
        corner = _marker_corner(g, box, bg)
        crop = g[r0 + 1 : r1, c0 + 1 : c1].copy()
        compartments.append((corner, crop))

    canonical_crops = [_orient_contract(crop, corner) for corner, crop in compartments]
    counts = Counter(_signature(crop) for crop in canonical_crops)
    majority_sig, majority_count = counts.most_common(1)[0]
    assert majority_count == len(compartments) - 1, "expected exactly one violating compartment"

    base = next(crop for crop in canonical_crops if _signature(crop) == majority_sig)
    violators = [idx for idx, crop in enumerate(canonical_crops) if _signature(crop) != majority_sig]
    assert len(violators) == 1
    violator_idx = violators[0]
    target_corner, _bad_crop = compartments[violator_idx]
    return _orient_contract(base, target_corner)


def _random_sprite(size: int, body_color: int, feature_color: int) -> np.ndarray:
    for _ in range(200):
        target_area = random.randint(size + 2, size + 5)
        cells = {(0, 0), (0, 1), (1, 0)}
        while len(cells) < target_area:
            frontier = set()
            for r, c in cells:
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < size and 0 <= nc < size and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(tuple(frontier)))

        mask = np.zeros((size, size), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        if mask[size - 1, 0] or mask[0, size - 1] or mask[size - 1, size - 1]:
            continue
        if mask.sum() < size + 1:
            continue

        sprite = np.zeros((size, size), dtype=int)
        sprite[mask] = body_color
        sprite[0, 0] = feature_color

        if any(np.array_equal(sprite, other) for other in _all_orientations(sprite)[1:]):
            continue
        contract_sigs = {_signature(_orient_contract(sprite, corner)) for corner in CORNER_ORDER}
        if len(contract_sigs) != 4:
            continue
        return sprite
    raise RuntimeError("failed to build an asymmetric sprite")


def _sample_layout() -> tuple[int, int]:
    return random.choice([(2, 2), (2, 3), (3, 2)])


def _marker_coord(box: tuple[int, int, int, int], corner: str) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    return {
        "tl": (r0 - 1, c0 - 1),
        "tr": (r0 - 1, c1 + 1),
        "bl": (r1 + 1, c0 - 1),
        "br": (r1 + 1, c1 + 1),
    }[corner]


def _generate_case() -> tuple[np.ndarray, np.ndarray]:
    rows, cols = _sample_layout()
    inner = random.randint(5, 6)
    outer = inner + 2
    gap = 2
    margin = 1
    height = rows * outer + (rows - 1) * gap + 2 * margin
    width = cols * outer + (cols - 1) * gap + 2 * margin
    assert max(height, width) <= 30

    palette = [1, 2, 3, 4, 6, 7, 8, 9]
    body_color, feature_color, marker_color = random.sample(palette, 3)
    sprite = _random_sprite(inner, body_color, feature_color)

    grid = np.full((height, width), BG, dtype=int)
    boxes = []
    for rr in range(rows):
        for cc in range(cols):
            r0 = margin + rr * (outer + gap)
            c0 = margin + cc * (outer + gap)
            r1 = r0 + outer - 1
            c1 = c0 + outer - 1
            boxes.append((r0, c0, r1, c1))
            grid[r0 : r1 + 1, c0] = GREY
            grid[r0 : r1 + 1, c1] = GREY
            grid[r0, c0 : c1 + 1] = GREY
            grid[r1, c0 : c1 + 1] = GREY

    corners = list(CORNER_ORDER)
    while len(corners) < len(boxes):
        corners.append(random.choice(CORNER_ORDER))
    random.shuffle(corners)

    violator_idx = random.randrange(len(boxes))
    pose_corners = list(corners)
    pose_corners[violator_idx] = random.choice([corner for corner in CORNER_ORDER if corner != corners[violator_idx]])

    for idx, box in enumerate(boxes):
        r0, c0, r1, c1 = box
        crop = _orient_contract(sprite, pose_corners[idx])
        grid[r0 + 1 : r1, c0 + 1 : c1] = crop
        mr, mc = _marker_coord(box, corners[idx])
        grid[mr, mc] = marker_color

    expected = _orient_contract(sprite, corners[violator_idx])
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid, expected = _generate_case()
        if np.array_equal(main(grid), expected):
            return grid
    raise RuntimeError("failed to generate a valid input")
