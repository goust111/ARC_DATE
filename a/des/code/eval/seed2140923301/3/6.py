from __future__ import annotations

import random

import numpy as np


BLACK = 0

# Shape vocabulary and geometry.
SHAPES = ("cross", "square", "diamond")
SIDE_CHOICES = (5, 7, 9, 11)

# Generator sampling.
GROUP_COUNT_RANGE = (3, 5)
SAMPLED_POINT_RANGE = (5, 18)
GRID_HEIGHT_RANGE = (24, 36)
GRID_WIDTH_RANGE = (24, 36)
PLACEMENT_PADDING = 2
MAX_GENERATION_ATTEMPTS = 300
PLACEMENT_ATTEMPTS = 200


def _shape_points(shape: str, side: int) -> list[tuple[int, int]]:
    mid = side // 2
    points: list[tuple[int, int]] = []
    if shape == "square":
        for r in range(side):
            for c in range(side):
                if r in {0, side - 1} or c in {0, side - 1}:
                    points.append((r, c))
        return points
    if shape == "cross":
        for r in range(side):
            points.append((r, mid))
        for c in range(side):
            if c != mid:
                points.append((mid, c))
        return points
    if shape == "diamond":
        for r in range(side):
            for c in range(side):
                if abs(r - mid) + abs(c - mid) == mid:
                    points.append((r, c))
        return points
    raise ValueError(f"unknown shape: {shape}")


def _render_shape(shape: str, side: int, color: int) -> np.ndarray:
    patch = np.full((side, side), BLACK, dtype=int)
    for r, c in _shape_points(shape, side):
        patch[r, c] = int(color)
    return patch


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _detect_shape(rel_points: set[tuple[int, int]], side: int) -> str:
    mid = side // 2
    if all(r == mid or c == mid for r, c in rel_points):
        return "cross"
    corners = {(0, 0), (0, side - 1), (side - 1, 0), (side - 1, side - 1)}
    if rel_points & corners:
        return "square"
    return "diamond"


def _extract_groups(grid: np.ndarray) -> list[dict[str, int | str]]:
    groups: list[dict[str, int | str]] = []
    for color in sorted(int(c) for c in np.unique(grid) if int(c) != BLACK):
        coords = np.argwhere(grid == int(color))
        assert coords.size > 0
        r0, c0, r1, c1 = _bbox(coords)
        side_h = r1 - r0 + 1
        side_w = c1 - c0 + 1
        assert side_h == side_w and side_h % 2 == 1 and side_h >= 3
        rel_points = {(int(r - r0), int(c - c0)) for r, c in coords}
        shape = _detect_shape(rel_points, side_h)
        groups.append(
            {
                "color": int(color),
                "count": int(coords.shape[0]),
                "shape": shape,
            }
        )
    assert 3 <= len(groups) <= 5
    return groups


def _nested_output(groups: list[dict[str, int | str]]) -> np.ndarray:
    ordered = sorted(groups, key=lambda group: (int(group["count"]), int(group["color"])))
    out_side = 4 * len(ordered) - 1
    center = out_side // 2
    out = np.full((out_side, out_side), BLACK, dtype=int)

    for layer_idx, group in enumerate(ordered):
        layer_side = 4 * layer_idx + 3
        radius = layer_side // 2
        patch = _render_shape(str(group["shape"]), layer_side, int(group["color"]))
        r0 = center - radius
        c0 = center - radius
        mask = patch != BLACK
        out_view = out[r0 : r0 + layer_side, c0 : c0 + layer_side]
        out_view[mask] = patch[mask]

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify sparse shape outlines by color, complete each outline, then redraw
    the shapes as concentric nested layers ordered by sampled-point count.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    groups = _extract_groups(g)
    return _nested_output(groups)


def _required_points(shape: str, side: int) -> list[tuple[int, int]]:
    mid = side // 2
    if shape == "square":
        return [(0, 0), (0, side - 1), (side - 1, 0), (side - 1, side - 1)]
    if shape == "cross":
        return [(0, mid), (mid, 0), (mid, mid), (mid, side - 1), (side - 1, mid)]
    if shape == "diamond":
        diagonal = [(r, c) for r, c in _shape_points(shape, side) if r != mid and c != mid]
        return [(0, mid), (mid, 0), (mid, side - 1), (side - 1, mid), random.choice(diagonal)]
    raise ValueError(f"unknown shape: {shape}")


def _sample_outline(shape: str, side: int, count: int, color: int) -> np.ndarray:
    full = _shape_points(shape, side)
    required = _required_points(shape, side)
    required_set = set(required)
    assert len(required_set) <= count <= len(full)

    remaining = [point for point in full if point not in required_set]
    chosen = list(required_set)
    if count > len(chosen):
        chosen.extend(random.sample(remaining, count - len(chosen)))

    patch = np.full((side, side), BLACK, dtype=int)
    for r, c in chosen:
        patch[r, c] = int(color)
    return patch


def _shape_capacity(shape: str, side: int) -> int:
    return len(_shape_points(shape, side))


def _pick_side(shape: str, count: int) -> int:
    valid = [side for side in SIDE_CHOICES if _shape_capacity(shape, side) >= count]
    if not valid:
        raise ValueError(f"no side supports shape={shape} count={count}")
    return random.choice(valid)


def _sample_group_specs() -> list[dict[str, int | str]]:
    group_count = random.randint(*GROUP_COUNT_RANGE)
    counts = sorted(random.sample(range(SAMPLED_POINT_RANGE[0], SAMPLED_POINT_RANGE[1] + 1), group_count))

    non_cross_shapes = ["square", "diamond"]
    if group_count > 3:
        non_cross_shapes.extend(random.choice(("square", "diamond")) for _ in range(group_count - 3))
    random.shuffle(non_cross_shapes)

    shapes = ["cross"] + non_cross_shapes
    assigned_counts = [counts[0]] + random.sample(counts[1:], len(counts) - 1)
    specs: list[dict[str, int | str]] = []
    colors = random.sample(range(1, 10), group_count)
    for shape, color, count in zip(shapes, colors, assigned_counts):
        specs.append({"shape": shape, "count": int(count), "color": int(color)})

    for spec in specs:
        spec["side"] = _pick_side(str(spec["shape"]), int(spec["count"]))
    return specs


def _can_place(occupied: np.ndarray, top: int, left: int, side: int) -> bool:
    h, w = occupied.shape
    r0 = max(0, top - PLACEMENT_PADDING)
    c0 = max(0, left - PLACEMENT_PADDING)
    r1 = min(h, top + side + PLACEMENT_PADDING)
    c1 = min(w, left + side + PLACEMENT_PADDING)
    return not occupied[r0:r1, c0:c1].any()


def _mark_occupied(occupied: np.ndarray, top: int, left: int, side: int) -> None:
    h, w = occupied.shape
    r0 = max(0, top - PLACEMENT_PADDING)
    c0 = max(0, left - PLACEMENT_PADDING)
    r1 = min(h, top + side + PLACEMENT_PADDING)
    c1 = min(w, left + side + PLACEMENT_PADDING)
    occupied[r0:r1, c0:c1] = True


def _place_patch(grid: np.ndarray, occupied: np.ndarray, patch: np.ndarray) -> bool:
    side = patch.shape[0]
    h, w = grid.shape
    if side > h or side > w:
        return False
    for _ in range(PLACEMENT_ATTEMPTS):
        top = random.randint(0, h - side)
        left = random.randint(0, w - side)
        if not _can_place(occupied, top, left, side):
            continue
        mask = patch != BLACK
        view = grid[top : top + side, left : left + side]
        view[mask] = patch[mask]
        _mark_occupied(occupied, top, left, side)
        return True
    return False


def _build_input(specs: list[dict[str, int | str]]) -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    grid = np.full((height, width), BLACK, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)

    for spec in sorted(specs, key=lambda item: int(item["side"]), reverse=True):
        patch = _sample_outline(str(spec["shape"]), int(spec["side"]), int(spec["count"]), int(spec["color"]))
        if not _place_patch(grid, occupied, patch):
            return None
    return grid


def _expected_output(specs: list[dict[str, int | str]]) -> np.ndarray:
    groups = [{"shape": spec["shape"], "count": spec["count"], "color": spec["color"]} for spec in specs]
    return _nested_output(groups)


def _gen_attempt() -> np.ndarray | None:
    specs = _sample_group_specs()
    grid = _build_input(specs)
    if grid is None:
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    expected = _expected_output(specs)
    if np.array_equal(actual, expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
