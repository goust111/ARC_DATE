import random
from collections import deque

import numpy as np


BLACK = 0
RED = 2
YELLOW = 4
ORANGE = 7


BASE_TEMPLATES = (
    (
        np.array(
            [
                [0, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1],
                [1, 0, 1, 0, 0, 1],
            ],
            dtype=bool,
        ),
        "top",
    ),
    (
        np.array(
            [
                [1, 0, 1, 0, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
            ],
            dtype=bool,
        ),
        "bottom",
    ),
    (
        np.array(
            [
                [1, 0, 1, 0, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
            ],
            dtype=bool,
        ),
        "bottom",
    ),
    (
        np.array(
            [
                [1, 1, 1],
                [0, 1, 1],
                [1, 1, 1],
                [0, 1, 1],
                [0, 1, 1],
                [1, 1, 1],
            ],
            dtype=bool,
        ),
        "right",
    ),
    (
        np.array(
            [
                [0, 1, 1, 0, 0, 0],
                [0, 0, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1],
                [0, 1, 0, 1, 0, 1],
            ],
            dtype=bool,
        ),
        "top",
    ),
    (
        np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1],
                [1, 0, 1, 0, 1, 0, 1],
            ],
            dtype=bool,
        ),
        "top",
    ),
    (
        np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1],
                [1, 0, 1, 0, 1, 0, 1],
            ],
            dtype=bool,
        ),
        "top",
    ),
    (
        np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1],
                [1, 0, 1, 0, 1, 0, 1],
            ],
            dtype=bool,
        ),
        "top",
    ),
)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """4-connected components, following the shared ARC `(r, c)` convention."""
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _paint(grid: np.ndarray, coords: np.ndarray, color: int) -> None:
    grid[coords[:, 0], coords[:, 1]] = int(color)


def _extract_single_component(grid: np.ndarray, color: int) -> np.ndarray:
    comps = _components(grid == int(color))
    assert len(comps) == 1, f"expected exactly one {color}-component"
    return comps[0]


def _detect_marker_side(object_coords: np.ndarray, marker_coords: np.ndarray) -> str:
    object_set = {tuple(x) for x in object_coords.tolist()}
    sides = set()
    for r, c in marker_coords.tolist():
        if (r, c + 1) in object_set:
            sides.add("left")
        if (r, c - 1) in object_set:
            sides.add("right")
        if (r + 1, c) in object_set:
            sides.add("top")
        if (r - 1, c) in object_set:
            sides.add("bottom")
    assert len(sides) == 1, "red markers must indicate exactly one side"
    return next(iter(sides))


def _mirror_coords(object_coords: np.ndarray, side: str) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_from_coords(object_coords)
    mirrored = []
    if side == "left":
        axis = c0 - 1
        mirrored = [(r, 2 * axis - c) for r, c in object_coords.tolist()]
    elif side == "right":
        axis = c1 + 1
        mirrored = [(r, 2 * axis - c) for r, c in object_coords.tolist()]
    elif side == "top":
        axis = r0 - 1
        mirrored = [(2 * axis - r, c) for r, c in object_coords.tolist()]
    elif side == "bottom":
        axis = r1 + 1
        mirrored = [(2 * axis - r, c) for r, c in object_coords.tolist()]
    else:
        raise ValueError(f"unknown side: {side}")
    return np.asarray(mirrored, dtype=int)


def _background_regions(grid: np.ndarray) -> list[np.ndarray]:
    return _components(grid == BLACK)


def _sorted_regions(grid: np.ndarray) -> list[np.ndarray]:
    regions = _background_regions(grid)
    return sorted(regions, key=lambda coords: _bbox_from_coords(coords)[:2])


def _mirrored_object_grid(grid: np.ndarray) -> tuple[np.ndarray, str]:
    g = np.asarray(grid, dtype=int)
    object_coords = _extract_single_component(g, ORANGE)
    marker_coords = _extract_single_component(g, RED)
    side = _detect_marker_side(object_coords, marker_coords)
    mirrored_coords = _mirror_coords(object_coords, side)

    out = np.zeros_like(g)
    _paint(out, object_coords, ORANGE)
    _paint(out, mirrored_coords, ORANGE)
    return out, side


def _solve(grid: np.ndarray) -> np.ndarray:
    out, _side = _mirrored_object_grid(grid)
    regions = _sorted_regions(out)
    assert len(regions) >= 3, "mirroring must split the background into at least three regions"
    for i, region in enumerate(regions):
        if i % 3 == 0:
            _paint(out, region, YELLOW)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror the orange object across the red-indicated side, then periodically fill the new background regions.

    Input structure:
    - The grid is black except for one 4-connected orange object and one 4-connected red marker component.
    - The red markers lie immediately outside exactly one side of the orange object's bounding box.
    - The orange object does not touch the grid border, and the pre-mirror black background is one 4-connected region.

    Rule:
    - Infer which side of the orange object the red markers touch.
    - Place a mirror axis one cell outside that bounding-box side and reflect the full orange object across it.
    - Remove the red markers, keep the original and mirrored orange pixels, and find all 4-connected black background regions.
    - Sort those regions by their bounding box upper-left coordinate `(row, col)` and fill every region whose index satisfies `i mod 3 == 0` with yellow.

    Key insight:
    - The red markers do not directly mark the fill target; they choose the mirror direction, and that mirror changes the negative-space topology that determines which regions are filled.
    """
    return _solve(np.asarray(grid, dtype=int))


def _rotate_side(side: str, k: int) -> str:
    order = ["top", "left", "bottom", "right"]
    idx = order.index(side)
    return order[(idx + k) % 4]


def _oriented_template() -> tuple[np.ndarray, str]:
    template, side = random.choice(BASE_TEMPLATES)
    k = random.randrange(4)
    return np.rot90(template, k=k), _rotate_side(side, k)


def _base_sprite(template: np.ndarray, side: str) -> np.ndarray:
    h, w = template.shape
    sprite = np.zeros((h + 2, w + 2), dtype=int)
    sprite[1 : h + 1, 1 : w + 1][template] = ORANGE
    object_coords = _extract_single_component(sprite, ORANGE)
    r0, c0, r1, c1 = _bbox_from_coords(object_coords)
    if side == "right":
        edge = object_coords[object_coords[:, 1] == c1]
        marker_coords = edge + np.array([0, 1], dtype=int)
    elif side == "left":
        edge = object_coords[object_coords[:, 1] == c0]
        marker_coords = edge + np.array([0, -1], dtype=int)
    elif side == "top":
        edge = object_coords[object_coords[:, 0] == r0]
        marker_coords = edge + np.array([-1, 0], dtype=int)
    elif side == "bottom":
        edge = object_coords[object_coords[:, 0] == r1]
        marker_coords = edge + np.array([1, 0], dtype=int)
    else:
        raise ValueError(f"unknown side: {side}")
    marker_mask = np.zeros_like(sprite, dtype=bool)
    marker_mask[marker_coords[:, 0], marker_coords[:, 1]] = True
    assert len(_components(marker_mask)) == 1
    _paint(sprite, marker_coords, RED)
    return sprite


def _target_bbox_ranges(
    grid_h: int, grid_w: int, obj_h: int, obj_w: int, side: str
) -> tuple[tuple[int, int], tuple[int, int]]:
    if side == "right":
        r_range = (1, grid_h - obj_h - 1)
        c_range = (1, grid_w - 2 * obj_w - 1)
    elif side == "left":
        r_range = (1, grid_h - obj_h - 1)
        c_range = (obj_w + 1, grid_w - obj_w - 1)
    elif side == "bottom":
        r_range = (1, grid_h - 2 * obj_h - 1)
        c_range = (1, grid_w - obj_w - 1)
    elif side == "top":
        r_range = (obj_h + 1, grid_h - obj_h - 1)
        c_range = (1, grid_w - obj_w - 1)
    else:
        raise ValueError(f"unknown side: {side}")
    assert r_range[0] <= r_range[1] and c_range[0] <= c_range[1]
    return r_range, c_range


def _gen_attempt() -> np.ndarray | None:
    template, side = _oriented_template()
    sprite = _base_sprite(template, side)
    object_coords = _extract_single_component(sprite, ORANGE)
    marker_coords = _extract_single_component(sprite, RED)
    side = _detect_marker_side(object_coords, marker_coords)

    local_r0, local_c0, local_r1, local_c1 = _bbox_from_coords(object_coords)
    obj_h = local_r1 - local_r0 + 1
    obj_w = local_c1 - local_c0 + 1

    min_grid_h = max(9, 2 * obj_h + 2 if side in {"top", "bottom"} else obj_h + 2)
    min_grid_w = max(9, 2 * obj_w + 2 if side in {"left", "right"} else obj_w + 2)
    max_grid_h = min(18, max(min_grid_h, min_grid_h + random.randint(2, 6)))
    max_grid_w = min(18, max(min_grid_w, min_grid_w + random.randint(2, 6)))
    if min_grid_h > max_grid_h or min_grid_w > max_grid_w:
        return None

    grid_h = random.randint(min_grid_h, max_grid_h)
    grid_w = random.randint(min_grid_w, max_grid_w)
    r_range, c_range = _target_bbox_ranges(grid_h, grid_w, obj_h, obj_w, side)
    if side == "top":
        target_r0 = random.randint(r_range[0], min(r_range[1], r_range[0] + 1))
        target_c0 = random.randint(*c_range)
    elif side == "bottom":
        target_r0 = random.randint(max(r_range[0], r_range[1] - 1), r_range[1])
        target_c0 = random.randint(*c_range)
    elif side == "left":
        target_r0 = random.randint(*r_range)
        target_c0 = random.randint(c_range[0], min(c_range[1], c_range[0] + 1))
    else:
        target_r0 = random.randint(*r_range)
        target_c0 = random.randint(max(c_range[0], c_range[1] - 1), c_range[1])

    dr = target_r0 - local_r0
    dc = target_c0 - local_c0
    shifted_object = object_coords + np.array([dr, dc], dtype=int)
    shifted_marker = marker_coords + np.array([dr, dc], dtype=int)

    grid = np.zeros((grid_h, grid_w), dtype=int)
    _paint(grid, shifted_object, ORANGE)
    _paint(grid, shifted_marker, RED)

    try:
        _validate_instance(grid)
    except AssertionError:
        return None
    return grid


def _validate_instance(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30
    assert set(np.unique(g)).issubset({BLACK, RED, ORANGE})

    object_coords = _extract_single_component(g, ORANGE)
    marker_coords = _extract_single_component(g, RED)
    side = _detect_marker_side(object_coords, marker_coords)
    r0, c0, r1, c1 = _bbox_from_coords(object_coords)
    assert r0 > 0 and c0 > 0 and r1 < g.shape[0] - 1 and c1 < g.shape[1] - 1
    assert len(_background_regions(g)) == 1

    mirrored_only, _ = _mirrored_object_grid(g)
    assert np.all((mirrored_only == BLACK) | (mirrored_only == ORANGE))
    regions = _sorted_regions(mirrored_only)
    assert len(regions) >= 3
    assert any(i % 3 == 0 for i in range(len(regions)))
    assert any(i % 3 != 0 for i in range(len(regions)))

    solved = main(g)
    assert solved.shape == g.shape
    assert np.any(solved == YELLOW)
    assert not np.any(solved == RED)

    # Mirror is semantically necessary: filling the pre-mirror background should not match.
    no_mirror = np.zeros_like(g)
    _paint(no_mirror, object_coords, ORANGE)
    original_regions = _sorted_regions(no_mirror)
    for i, region in enumerate(original_regions):
        if i % 3 == 0:
            _paint(no_mirror, region, YELLOW)
    assert not np.array_equal(no_mirror, solved)

    # Direction matters: any feasible wrong side should yield a different geometry.
    feasible_wrong = 0
    for wrong_side in ("left", "right", "top", "bottom"):
        if wrong_side == side:
            continue
        wrong_coords = _mirror_coords(object_coords, wrong_side)
        if (
            wrong_coords[:, 0].min() < 0
            or wrong_coords[:, 1].min() < 0
            or wrong_coords[:, 0].max() >= g.shape[0]
            or wrong_coords[:, 1].max() >= g.shape[1]
        ):
            continue
        feasible_wrong += 1
        wrong = np.zeros_like(g)
        _paint(wrong, object_coords, ORANGE)
        _paint(wrong, wrong_coords, ORANGE)
        wrong_regions = _sorted_regions(wrong)
        for i, region in enumerate(wrong_regions):
            if i % 3 == 0:
                _paint(wrong, region, YELLOW)
        assert not np.array_equal(wrong, solved)
    if feasible_wrong == 0:
        return


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirror-negative-space ARC input")
