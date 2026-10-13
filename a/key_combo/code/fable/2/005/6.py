from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BLACK = 0
PALETTE = tuple(range(1, 10))
QUADRANTS = ("tl", "tr", "bl", "br")
GEN_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _orbit_positions(r: int, c: int, height: int, width: int) -> list[tuple[int, int]]:
    cells = {
        (int(r), int(c)),
        (height - 1 - int(r), int(c)),
        (int(r), width - 1 - int(c)),
        (height - 1 - int(r), width - 1 - int(c)),
    }
    return sorted(cells)


def _canonical_key(r: int, c: int, height: int, width: int) -> tuple[int, int]:
    return min(int(r), height - 1 - int(r)), min(int(c), width - 1 - int(c))


def _component_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False

    start = tuple(map(int, coords[0]))
    seen = {start}
    queue = deque([start])
    height, width = mask.shape
    while queue:
        r, c = queue.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))
    return len(seen) == int(mask.sum())


def _crop_nonblack(grid: np.ndarray) -> np.ndarray:
    box = bbox(grid != BLACK)
    assert box is not None
    return crop_bbox(grid, box)


def _paint_orbit(patch: np.ndarray, r: int, c: int, color: int) -> None:
    height, width = patch.shape
    for rr, cc in _orbit_positions(r, c, height, width):
        patch[rr, cc] = int(color)


def _quadrant_position(r: int, c: int, height: int, width: int, quadrant: str) -> tuple[int, int]:
    if quadrant == "tl":
        return int(r), int(c)
    if quadrant == "tr":
        return int(r), width - 1 - int(c)
    if quadrant == "bl":
        return height - 1 - int(r), int(c)
    assert quadrant == "br"
    return height - 1 - int(r), width - 1 - int(c)


def _random_connected_mask(max_h: int, max_w: int) -> np.ndarray | None:
    for _ in range(200):
        box_h = random.randint(2, min(5, max_h))
        box_w = random.randint(2, min(5, max_w))
        max_area = min(8, box_h * box_w)
        target_area = random.randint(4, max_area)

        start = (random.randrange(box_h), random.randrange(box_w))
        cells = {start}
        while len(cells) < target_area:
            frontier = []
            for r, c in cells:
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))

        if len(cells) != target_area:
            continue

        arr = np.zeros((box_h, box_w), dtype=int)
        for r, c in cells:
            arr[r, c] = 1
        arr = _crop_nonblack(arr)
        if min(arr.shape) < 2:
            continue
        if not _component_connected(arr.astype(bool)):
            continue
        return arr
    return None


def _build_symmetric_patch(height: int, width: int) -> np.ndarray:
    fill = random.choice(PALETTE)
    patch = np.full((height, width), fill, dtype=int)
    quad_h = (height + 1) // 2
    quad_w = (width + 1) // 2

    feature_count = random.randint(6, 11)
    for _ in range(feature_count):
        feature_h = random.randint(1, min(3, quad_h))
        feature_w = random.randint(1, min(4, quad_w))
        r0 = random.randint(0, quad_h - feature_h)
        c0 = random.randint(0, quad_w - feature_w)
        color = random.choice(PALETTE)
        for r in range(r0, r0 + feature_h):
            for c in range(c0, c0 + feature_w):
                _paint_orbit(patch, r, c, color)

    if len(set(map(int, patch.ravel()))) < 3:
        accent = random.choice([c for c in PALETTE if c != fill])
        for c in range(0, quad_w, 2):
            _paint_orbit(patch, 0, c, accent)

    assert np.array_equal(patch, patch[::-1, :])
    assert np.array_equal(patch, patch[:, ::-1])
    return patch


def _sample_dimensions() -> tuple[int, int, int, int] | None:
    for _ in range(120):
        grid_h = random.randint(18, 30)
        grid_w = random.randint(18, 30)
        max_patch_h = min(18, grid_h - 2)
        max_patch_w = min(18, grid_w - 2)
        if max_patch_h < 8 or max_patch_w < 8:
            continue
        patch_h = random.randint(8, max_patch_h)
        patch_w = random.randint(8, max_patch_w)
        if patch_h * patch_w >= (grid_h * grid_w) // 2:
            continue
        if patch_h // 2 < 2 or patch_w // 2 < 2:
            continue
        return grid_h, grid_w, patch_h, patch_w
    return None


def _make_hidden_shape(base_patch: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int, int]], list[str]] | None:
    patch_h, patch_w = base_patch.shape
    fold_h = patch_h // 2
    fold_w = patch_w // 2
    mask = _random_connected_mask(fold_h, fold_w)
    if mask is None:
        return None

    shape_h, shape_w = mask.shape
    offset_r = random.randint(0, fold_h - shape_h)
    offset_c = random.randint(0, fold_w - shape_w)
    cells = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    random.shuffle(cells)

    target = np.zeros((shape_h, shape_w), dtype=int)
    quadrants = list(QUADRANTS)
    random.shuffle(quadrants)
    assigned_quadrants = []
    placements: list[tuple[int, int, int]] = []
    used_colors: list[int] = []

    for idx, (local_r, local_c) in enumerate(cells):
        canon_r = offset_r + local_r
        canon_c = offset_c + local_c
        base_color = int(base_patch[canon_r, canon_c])
        choices = [color for color in PALETTE if color != base_color]
        if idx == 0:
            color = random.choice(choices)
        elif idx == 1:
            diff_choices = [choice for choice in choices if choice != used_colors[0]]
            color = random.choice(diff_choices if diff_choices else choices)
        else:
            color = random.choice(choices)

        quadrant = quadrants[idx] if idx < 4 else random.choice(QUADRANTS)
        target[local_r, local_c] = int(color)
        placements.append((canon_r, canon_c, int(color)))
        used_colors.append(int(color))
        assigned_quadrants.append(quadrant)

    if len(set(used_colors)) < 2:
        return None
    return target, placements, assigned_quadrants


def _embed_instance() -> tuple[np.ndarray, np.ndarray, dict] | None:
    dims = _sample_dimensions()
    if dims is None:
        return None
    grid_h, grid_w, patch_h, patch_w = dims
    base_patch = _build_symmetric_patch(patch_h, patch_w)
    hidden = _make_hidden_shape(base_patch)
    if hidden is None:
        return None
    target, placements, assigned_quadrants = hidden

    row_sides = set()
    col_sides = set()
    grid = np.full((grid_h, grid_w), BLACK, dtype=int)
    top = random.randint(1, grid_h - patch_h - 1)
    left = random.randint(1, grid_w - patch_w - 1)
    grid[top : top + patch_h, left : left + patch_w] = base_patch

    for idx, (canon_r, canon_c, color) in enumerate(placements):
        quadrant = assigned_quadrants[idx]
        rr, cc = _quadrant_position(canon_r, canon_c, patch_h, patch_w, quadrant)
        row_sides.add("top" if rr < patch_h // 2 else "bottom")
        col_sides.add("left" if cc < patch_w // 2 else "right")
        grid[top + rr, left + cc] = int(color)

    info = {
        "base_patch": base_patch,
        "target": target,
        "patch_box": (top, left, top + patch_h - 1, left + patch_w - 1),
        "row_sides": row_sides,
        "col_sides": col_sides,
        "quadrants": set(assigned_quadrants),
    }
    return grid, target, info


def _validate_instance(grid: np.ndarray, target: np.ndarray, info: dict) -> bool:
    if grid.ndim != 2 or target.ndim != 2 or max(grid.shape) > 30 or max(target.shape) > 30:
        return False
    if get_bg(grid) != BLACK:
        return False
    if not np.array_equal(main(grid), target):
        return False
    if len(info["quadrants"]) != 4:
        return False
    if info["row_sides"] != {"top", "bottom"} or info["col_sides"] != {"left", "right"}:
        return False
    if len(set(map(int, target[target != BLACK].ravel()))) < 2:
        return False
    if not _component_connected(target != BLACK):
        return False

    top, left, bottom, right = info["patch_box"]
    patch = grid[top : bottom + 1, left : right + 1]
    patch_h, patch_w = patch.shape
    for r in range(patch_h):
        for c in range(patch_w):
            key = _canonical_key(r, c, patch_h, patch_w)
            orbit = _orbit_positions(r, c, patch_h, patch_w)
            if (r, c) != orbit[0]:
                continue
            colors = [int(patch[rr, cc]) for rr, cc in orbit]
            counts = Counter(colors).most_common()
            if len(counts) == 1:
                continue
            if counts[0][1] <= counts[1][1]:
                return False
            major = counts[0][0]
            impurities = [(rr, cc) for rr, cc in orbit if int(patch[rr, cc]) != major]
            if len(impurities) != 1:
                return False
            if key[0] >= patch_h // 2 or key[1] >= patch_w // 2:
                return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the hidden multicolor figure by folding the symmetry-breaking impurity pixels into one quadrant.

    Input structure:
    - The grid has a dominant black background and one off-center rectangular motif made of non-black colors.
    - Inside that motif, the intended picture is symmetric about one horizontal and one vertical mirror axis.
    - A small number of off-axis pixels have been recolored; each affected two-axis orbit contains exactly one such impurity.

    Rule:
    - Crop the non-background motif and use the center lines of that tight crop as the two mirror axes.
    - For every orbit formed by horizontal and vertical reflection, treat the orbit's majority color as the true motif color.
    - Any lone minority pixel in that orbit is an impurity; reflect it into the common upper-left folded quadrant and overlay all folded impurities there.
    - Return the tight crop of that overlay on a black background.

    Key insight:
    - The symmetric motif is only a carrier: majority vote within each two-axis orbit isolates the noisy pixel, and folding those noisy pixels is what reassembles the hidden shape.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    motif_box = bbox(g != bg)
    assert motif_box is not None, "expected a non-background motif"
    patch = crop_bbox(g, motif_box)
    patch_h, patch_w = patch.shape

    folded = np.full((patch_h, patch_w), BLACK, dtype=int)
    seen: set[tuple[int, int]] = set()
    impurity_count = 0

    for r in range(patch_h):
        for c in range(patch_w):
            key = _canonical_key(r, c, patch_h, patch_w)
            if key in seen:
                continue
            seen.add(key)

            orbit = _orbit_positions(r, c, patch_h, patch_w)
            colors = [int(patch[rr, cc]) for rr, cc in orbit]
            counts = Counter(colors).most_common()
            if len(counts) == 1:
                continue
            assert len(counts) == 1 or counts[0][1] > counts[1][1], "expected a unique orbit majority"
            major = counts[0][0]
            impurities = [(rr, cc) for rr, cc in orbit if int(patch[rr, cc]) != major]
            assert len(impurities) == 1, "expected exactly one impurity in each noisy orbit"

            impurity_r, impurity_c = impurities[0]
            color = int(patch[impurity_r, impurity_c])
            assert color != BLACK, "impurity color must remain visible after folding"
            if folded[key] not in (BLACK, color):
                raise AssertionError("folded impurities must not collide with different colors")
            folded[key] = color
            impurity_count += 1

    assert impurity_count > 0, "expected at least one impurity orbit"
    out_box = bbox(folded != BLACK)
    assert out_box is not None, "expected a nonempty folded result"
    return crop_bbox(folded, out_box)


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        built = _embed_instance()
        if built is None:
            continue
        grid, target, info = built
        if _validate_instance(grid, target, info):
            return grid
    raise RuntimeError("failed to generate a valid double-axis impurity-folding instance")
