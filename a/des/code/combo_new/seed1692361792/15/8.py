from __future__ import annotations

import itertools
import random

import numpy as np


BG_COLOR = 0
LINE_COLOR = 2
OBJECT_COLORS = [1, 3, 4, 5, 6, 7, 8, 9]
GENERATION_RETRY_LIMIT = 600
ORTHOGONAL_STEPS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _full_line_candidates(grid: np.ndarray, bg: int) -> list[tuple[str, int, int]]:
    height, width = grid.shape
    candidates: list[tuple[str, int, int]] = []
    for r in range(height):
        color = int(grid[r, 0])
        if color != bg and np.all(grid[r, :] == color):
            candidates.append(("horizontal", r, color))
    for c in range(width):
        color = int(grid[0, c])
        if color != bg and np.all(grid[:, c] == color):
            candidates.append(("vertical", c, color))
    return candidates


def _build_component(color: int, coords: list[tuple[int, int]]) -> dict[str, object]:
    points = np.asarray(coords, dtype=int)
    r0 = int(points[:, 0].min())
    c0 = int(points[:, 1].min())
    r1 = int(points[:, 0].max())
    c1 = int(points[:, 1].max())
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[points[:, 0] - r0, points[:, 1] - c0] = True
    return {
        "color": int(color),
        "coords": points,
        "bbox": (r0, c0, r1, c1),
        "mask": mask,
        "area": int(points.shape[0]),
    }


def _find_components(grid: np.ndarray, bg: int, blocked: np.ndarray) -> list[dict[str, object]]:
    height, width = grid.shape
    visited = np.zeros((height, width), dtype=bool)
    components: list[dict[str, object]] = []
    for r in range(height):
        for c in range(width):
            if visited[r, c] or blocked[r, c]:
                continue
            color = int(grid[r, c])
            if color == bg:
                visited[r, c] = True
                continue
            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in ORTHOGONAL_STEPS:
                    nr = cr + dr
                    nc = cc + dc
                    if not (0 <= nr < height and 0 <= nc < width):
                        continue
                    if visited[nr, nc] or blocked[nr, nc]:
                        continue
                    if int(grid[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    stack.append((nr, nc))
            components.append(_build_component(color, coords))
    return components


def _split_components(
    components: list[dict[str, object]],
    orientation: str,
    axis: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    first: list[dict[str, object]] = []
    second: list[dict[str, object]] = []
    for component in components:
        r0, c0, r1, c1 = component["bbox"]
        if orientation == "horizontal":
            if r1 < axis:
                first.append(component)
            elif r0 > axis:
                second.append(component)
            else:
                raise AssertionError("foreground component touches or crosses the divider row")
        else:
            if c1 < axis:
                first.append(component)
            elif c0 > axis:
                second.append(component)
            else:
                raise AssertionError("foreground component touches or crosses the divider column")
    return first, second


def _mirror_bbox(
    bbox: tuple[int, int, int, int],
    orientation: str,
    axis: int,
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    if orientation == "horizontal":
        return (2 * axis - r1, c0, 2 * axis - r0, c1)
    return (r0, 2 * axis - c1, r1, 2 * axis - c0)


def _bbox_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    r0 = max(a[0], b[0])
    c0 = max(a[1], b[1])
    r1 = min(a[2], b[2])
    c1 = min(a[3], b[3])
    if r0 > r1 or c0 > c1:
        return 0
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _bbox_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    r0, c0, r1, c1 = bbox
    return ((r0 + r1) / 2.0, (c0 + c1) / 2.0)


def _pair_components(
    first: list[dict[str, object]],
    second: list[dict[str, object]],
    orientation: str,
    axis: int,
) -> list[tuple[dict[str, object], dict[str, object]]]:
    assert 2 <= len(first) <= 3
    assert len(first) == len(second)

    overlap = np.zeros((len(first), len(second)), dtype=int)
    distance = np.zeros((len(first), len(second)), dtype=float)
    for i, comp_a in enumerate(first):
        mirrored_box = _mirror_bbox(comp_a["bbox"], orientation, axis)
        center_a = _bbox_center(mirrored_box)
        for j, comp_b in enumerate(second):
            overlap[i, j] = _bbox_overlap(mirrored_box, comp_b["bbox"])
            center_b = _bbox_center(comp_b["bbox"])
            distance[i, j] = abs(center_a[0] - center_b[0]) + abs(center_a[1] - center_b[1])

    best_key: tuple[int, float] | None = None
    best_perm: tuple[int, ...] | None = None
    tie_count = 0
    for perm in itertools.permutations(range(len(second))):
        total_overlap = int(sum(overlap[i, perm[i]] for i in range(len(first))))
        total_distance = float(sum(distance[i, perm[i]] for i in range(len(first))))
        key = (total_overlap, -total_distance)
        if best_key is None or key > best_key:
            best_key = key
            best_perm = perm
            tie_count = 1
        elif key == best_key:
            tie_count += 1

    assert best_key is not None and best_perm is not None
    assert tie_count == 1, "mirror-position pairing is ambiguous"
    assert all(overlap[i, best_perm[i]] > 0 for i in range(len(first)))
    return [(first[i], second[best_perm[i]]) for i in range(len(first))]


def _detect_structure(
    grid: np.ndarray,
) -> tuple[int, str, int, list[dict[str, object]], list[dict[str, object]], list[tuple[dict[str, object], dict[str, object]]]]:
    bg = _dominant_color(grid)
    valid: list[
        tuple[
            int,
            str,
            int,
            list[dict[str, object]],
            list[dict[str, object]],
            list[tuple[dict[str, object], dict[str, object]]],
        ]
    ] = []

    for orientation, axis, _color in _full_line_candidates(grid, bg):
        blocked = np.zeros_like(grid, dtype=bool)
        if orientation == "horizontal":
            blocked[axis, :] = True
        else:
            blocked[:, axis] = True
        components = _find_components(grid, bg=bg, blocked=blocked)
        if not components:
            continue
        try:
            first, second = _split_components(components, orientation, axis)
            if not (2 <= len(first) <= 3 and len(first) == len(second)):
                continue
            pairs = _pair_components(first, second, orientation, axis)
        except AssertionError:
            continue
        valid.append((bg, orientation, axis, first, second, pairs))

    assert len(valid) == 1, "expected exactly one valid divider line"
    return valid[0]


def _reflect_mask(mask: np.ndarray, orientation: str) -> np.ndarray:
    if orientation == "horizontal":
        return np.flipud(mask)
    return np.fliplr(mask)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trim each mirrored larger region down to the smaller region's reflected shape across the divider line.

    Input structure:
    - The background is black, and one one-pixel-wide colored line spans the whole grid horizontally or vertically.
    - On each side of that divider line there are 2-3 monochrome 4-connected foreground regions.
    - Regions are paired across the divider by mirror position rather than by color.

    Rule:
    - Detect the full-length divider line and its orientation.
    - Pair regions across the divider by mirroring one side and matching the bounding boxes with maximum overlap.
    - In each pair, identify the smaller and larger region by area.
    - Reflect the smaller region's shape across the divider, align its bounding box to the larger region's bounding box, and keep only the overlapping pixels of the larger region in the larger region's own color; turn the rest to black.

    Key insight:
    - The divider is a symmetry axis used only for cross-side matching, and then the smaller shape acts as a stencil that trims the larger partner.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg, orientation, _axis, _first, _second, pairs = _detect_structure(g)
    out = g.copy()
    for comp_a, comp_b in pairs:
        if int(comp_a["area"]) < int(comp_b["area"]):
            small = comp_a
            large = comp_b
        else:
            small = comp_b
            large = comp_a
        assert int(small["area"]) != int(large["area"])

        template = _reflect_mask(np.asarray(small["mask"], dtype=bool), orientation)
        lr0, lc0, _lr1, _lc1 = large["bbox"]
        large_coords = np.asarray(large["coords"], dtype=int)
        rel_r = large_coords[:, 0] - lr0
        rel_c = large_coords[:, 1] - lc0
        keep = np.zeros(large_coords.shape[0], dtype=bool)
        in_bounds = (
            (rel_r >= 0)
            & (rel_r < template.shape[0])
            & (rel_c >= 0)
            & (rel_c < template.shape[1])
        )
        keep[in_bounds] = template[rel_r[in_bounds], rel_c[in_bounds]]
        removed = large_coords[~keep]
        if removed.size:
            out[removed[:, 0], removed[:, 1]] = bg
    return out


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(int(x) for x in coords[0])
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in ORTHOGONAL_STEPS:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc]:
                nxt = (nr, nc)
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
    return len(seen) == int(mask.sum())


def _touches_full_bbox(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def _carve_full_bbox_mask(height: int, width: int, target_area: int) -> np.ndarray | None:
    mask = np.ones((height, width), dtype=bool)
    while int(mask.sum()) > target_area:
        removable: list[tuple[int, int]] = []
        for r, c in np.argwhere(mask):
            trial = mask.copy()
            trial[int(r), int(c)] = False
            if _touches_full_bbox(trial) and _is_connected(trial):
                removable.append((int(r), int(c)))
        if not removable:
            return None
        rr, cc = random.choice(removable)
        mask[rr, cc] = False
    return mask if _touches_full_bbox(mask) and _is_connected(mask) else None


def _grow_connected_superset(mask: np.ndarray, target_area: int) -> np.ndarray | None:
    out = mask.copy()
    while int(out.sum()) < target_area:
        frontier: list[tuple[int, int]] = []
        for r in range(out.shape[0]):
            for c in range(out.shape[1]):
                if out[r, c]:
                    continue
                if any(
                    0 <= r + dr < out.shape[0]
                    and 0 <= c + dc < out.shape[1]
                    and out[r + dr, c + dc]
                    for dr, dc in ORTHOGONAL_STEPS
                ):
                    frontier.append((r, c))
        if not frontier:
            return None
        rr, cc = random.choice(frontier)
        out[rr, cc] = True
    return out if _touches_full_bbox(out) and _is_connected(out) else None


def _separated(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int], gap: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = box_a
    br0, bc0, br1, bc1 = box_b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _place_boxes(
    height: int,
    width: int,
    orientation: str,
    pair_count: int,
) -> list[tuple[int, int, int, int]] | None:
    if orientation == "horizontal":
        axis = height // 2
    else:
        axis = width // 2

    boxes: list[tuple[int, int, int, int]] = []
    for _ in range(pair_count):
        placed = False
        for _ in range(200):
            box_h = random.randint(3, 5)
            box_w = random.randint(3, 5)
            if orientation == "horizontal":
                max_r0 = axis - box_h
                if max_r0 < 1 or width - box_w - 1 < 1:
                    return None
                r0 = random.randint(1, max_r0)
                c0 = random.randint(1, width - box_w - 1)
            else:
                max_c0 = axis - box_w
                if max_c0 < 1 or height - box_h - 1 < 1:
                    return None
                r0 = random.randint(1, height - box_h - 1)
                c0 = random.randint(1, max_c0)
            box = (r0, c0, r0 + box_h - 1, c0 + box_w - 1)
            if all(_separated(box, prev, gap=1) for prev in boxes):
                boxes.append(box)
                placed = True
                break
        if not placed:
            return None
    return boxes


def _draw_mask(grid: np.ndarray, box: tuple[int, int, int, int], mask: np.ndarray, color: int) -> None:
    r0, c0, r1, c1 = box
    assert mask.shape == (r1 - r0 + 1, c1 - c0 + 1)
    grid[r0 : r1 + 1, c0 : c1 + 1][mask] = int(color)


def _generate_pair_masks(box_h: int, box_w: int, orientation: str) -> tuple[np.ndarray, np.ndarray] | None:
    box_area = box_h * box_w
    small_low = max(5, int(np.ceil(box_area * 0.4)))
    small_high = min(box_area - 2, int(np.floor(box_area * 0.65)))
    if small_low > small_high:
        return None
    for _ in range(120):
        small_area = random.randint(small_low, small_high)
        small = _carve_full_bbox_mask(box_h, box_w, small_area)
        if small is None:
            continue
        reflected_small = _reflect_mask(small, orientation)
        capacity = box_area - int(reflected_small.sum())
        if capacity < 1:
            continue
        extra = random.randint(1, min(capacity, max(2, box_area // 3)))
        large = _grow_connected_superset(reflected_small, int(reflected_small.sum()) + extra)
        if large is None:
            continue
        if np.array_equal(large, reflected_small):
            continue
        return small, large
    return None


def _mirror_box(
    box: tuple[int, int, int, int],
    orientation: str,
    axis: int,
) -> tuple[int, int, int, int]:
    return _mirror_bbox(box, orientation, axis)


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    orientation = random.choice(["horizontal", "vertical"])
    pair_count = random.choice([2, 3])
    if orientation == "horizontal":
        height = random.choice([17, 19, 21, 23])
        width = random.choice([17, 19, 21, 23, 25])
        axis = height // 2
    else:
        height = random.choice([17, 19, 21, 23, 25])
        width = random.choice([17, 19, 21, 23])
        axis = width // 2

    boxes = _place_boxes(height, width, orientation, pair_count)
    if boxes is None:
        return None

    colors = random.sample(OBJECT_COLORS, 2 * pair_count)
    grid = np.full((height, width), BG_COLOR, dtype=int)
    expected = np.full((height, width), BG_COLOR, dtype=int)
    if orientation == "horizontal":
        grid[axis, :] = LINE_COLOR
        expected[axis, :] = LINE_COLOR
    else:
        grid[:, axis] = LINE_COLOR
        expected[:, axis] = LINE_COLOR

    changed_pixels = 0
    for pair_index, primary_box in enumerate(boxes):
        box_h = primary_box[2] - primary_box[0] + 1
        box_w = primary_box[3] - primary_box[1] + 1
        masks = _generate_pair_masks(box_h, box_w, orientation)
        if masks is None:
            return None
        small_mask, large_mask = masks
        secondary_box = _mirror_box(primary_box, orientation, axis)
        primary_color = colors[2 * pair_index]
        secondary_color = colors[2 * pair_index + 1]
        primary_is_small = bool(random.getrandbits(1))

        if primary_is_small:
            primary_mask = small_mask
            secondary_template = _reflect_mask(small_mask, orientation)
            secondary_mask = large_mask

            _draw_mask(grid, primary_box, primary_mask, primary_color)
            _draw_mask(grid, secondary_box, secondary_mask, secondary_color)
            _draw_mask(expected, primary_box, primary_mask, primary_color)
            _draw_mask(expected, secondary_box, secondary_template, secondary_color)
            changed_pixels += int(secondary_mask.sum() - secondary_template.sum())
        else:
            secondary_mask = small_mask
            primary_template = _reflect_mask(small_mask, orientation)
            primary_mask = large_mask

            _draw_mask(grid, primary_box, primary_mask, primary_color)
            _draw_mask(grid, secondary_box, secondary_mask, secondary_color)
            _draw_mask(expected, primary_box, primary_template, primary_color)
            _draw_mask(expected, secondary_box, secondary_mask, secondary_color)
            changed_pixels += int(primary_mask.sum() - primary_template.sum())

    if changed_pixels <= 0:
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRY_LIMIT):
        built = _build_case()
        if built is None:
            continue
        grid, expected = built
        try:
            actual = main(grid)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return grid
    raise RuntimeError("failed to generate a valid divider-trimming ARC input")
