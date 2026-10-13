import random
from collections import Counter, deque

import numpy as np


BG = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
GEN_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 300


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _find_components(grid: np.ndarray, bg: int = BG) -> list[tuple[int, int, int, int]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    boxes = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) != bg:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            rs = [p[0] for p in cells]
            cs = [p[1] for p in cells]
            boxes.append((min(rs), min(cs), max(rs), max(cs)))
    boxes.sort()
    return boxes


def _complete_vertical(half: np.ndarray) -> np.ndarray:
    return np.concatenate([half, np.fliplr(half[:, :-1])], axis=1)


def _complete_horizontal(half: np.ndarray) -> np.ndarray:
    return np.concatenate([half, np.flipud(half[:-1, :])], axis=0)


def _is_valid_full_square(square: np.ndarray) -> bool:
    if square.ndim != 2 or square.shape[0] != square.shape[1]:
        return False
    side = int(square.shape[0])
    if side % 2 == 0 or side < 5 or side > 7:
        return False
    center = side // 2
    marker = int(square[center, center])
    if marker == BG or int(np.count_nonzero(square == marker)) != 1:
        return False

    shell_colors = []
    yy, xx = np.indices(square.shape)
    dist = np.maximum(np.abs(yy - center), np.abs(xx - center))
    for radius in range(center, 0, -1):
        shell = square[dist == radius]
        if shell.size == 0:
            return False
        color = int(shell[0])
        if color == BG or not np.all(shell == color):
            return False
        shell_colors.append(color)
    return len(set(shell_colors)) == len(shell_colors) and marker not in shell_colors


def _analyze_half(half: np.ndarray) -> tuple[str, np.ndarray]:
    candidates: list[tuple[str, np.ndarray]] = []
    vertical = _complete_vertical(half)
    if _is_valid_full_square(vertical):
        candidates.append(("vertical", vertical))
    horizontal = _complete_horizontal(half)
    if _is_valid_full_square(horizontal):
        candidates.append(("horizontal", horizontal))
    assert len(candidates) == 1, "half-square must map to one valid symmetry axis"
    return candidates[0]


def _paint_completion(out: np.ndarray, full: np.ndarray, r0: int, c0: int) -> None:
    patch = out[r0 : r0 + full.shape[0], c0 : c0 + full.shape[1]]
    mask = full != BG
    conflict = mask & (patch != BG) & (patch != full)
    assert not np.any(conflict), "completion conflicts with existing colors"
    patch[mask] = full[mask]


def _grow_marker_band(out: np.ndarray, axis: str, r0: int, c0: int, side: int, color: int) -> None:
    if axis == "vertical":
        for cc in (c0 - 1, c0 + side):
            if 0 <= cc < out.shape[1]:
                band = out[r0 : r0 + side, cc]
                band[band == BG] = color
    else:
        for rr in (r0 - 1, r0 + side):
            if 0 <= rr < out.shape[0]:
                band = out[rr, c0 : c0 + side]
                band[band == BG] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror-complete each half concentric square, then grow a marker-colored band along the axis-controlled direction.

    Input structure:
    - The grid is mostly black background and contains 2-3 separated multicolor half-squares.
    - Each object is the left half or the top half of a concentric odd square with 2-3 colored shells.
    - The geometric center is visible as a unique single-pixel marker color on the cut axis.

    Rule:
    - For each object, test vertical and horizontal mirror completion and keep the one that yields a valid full concentric square.
    - Paste that completed square back into the scene in place of the visible half.
    - If the detected symmetry axis is vertical, add one marker-colored column on both left and right sides of the square; if horizontal, add one marker-colored row on both top and bottom sides.
    - Growth only fills black background cells and never overwrites existing colors.

    Key insight:
    - The same recovered symmetry axis determines both how to restore the hidden half and which perpendicular direction should receive the outward growth.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for box in _find_components(g, bg=BG):
        r0, c0, _r1, _c1 = box
        half = _crop(g, box)
        axis, full = _analyze_half(half)
        side = full.shape[0]
        center = side // 2
        marker_color = int(full[center, center])
        _paint_completion(out, full, r0, c0)
        _grow_marker_band(out, axis, r0, c0, side, marker_color)

    return out


def _build_full_square(shell_colors: list[int], marker_color: int) -> np.ndarray:
    layer_count = len(shell_colors)
    side = layer_count * 2 + 1
    center = side // 2
    yy, xx = np.indices((side, side))
    dist = np.maximum(np.abs(yy - center), np.abs(xx - center))
    square = np.zeros((side, side), dtype=int)
    for idx, radius in enumerate(range(center, 0, -1)):
        square[dist == radius] = int(shell_colors[idx])
    square[center, center] = int(marker_color)
    return square


def _make_object_spec(existing_markers: list[int]) -> dict:
    orientation = random.choice(["vertical", "horizontal"])
    layer_count = random.choice([2, 3])
    shell_colors = random.sample(PALETTE, layer_count)

    if existing_markers and random.random() < 0.35:
        marker_color = random.choice(existing_markers)
        if marker_color in shell_colors:
            marker_color = random.choice([c for c in PALETTE if c not in shell_colors])
    else:
        marker_color = random.choice([c for c in PALETTE if c not in shell_colors])

    full = _build_full_square(shell_colors, marker_color)
    side = full.shape[0]
    half_span = side // 2 + 1
    if orientation == "vertical":
        visible = full[:, :half_span]
        reserve_h, reserve_w = side + 4, side + 6
        top_offset, left_offset = 2, 3
    else:
        visible = full[:half_span, :]
        reserve_h, reserve_w = side + 6, side + 4
        top_offset, left_offset = 3, 2

    return {
        "orientation": orientation,
        "visible": visible,
        "full": full,
        "side": side,
        "reserve_shape": (reserve_h, reserve_w),
        "visible_offset": (top_offset, left_offset),
        "marker": marker_color,
    }


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _place_specs(specs: list[dict], h: int, w: int) -> np.ndarray | None:
    grid = np.zeros((h, w), dtype=int)
    placed_boxes: list[tuple[int, int, int, int]] = []

    order = list(range(len(specs)))
    random.shuffle(order)
    for idx in order:
        spec = specs[idx]
        rh, rw = spec["reserve_shape"]
        if rh > h or rw > w:
            return None
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            r0 = random.randint(0, h - rh)
            c0 = random.randint(0, w - rw)
            box = (r0, c0, r0 + rh - 1, c0 + rw - 1)
            if any(_boxes_overlap(box, other) for other in placed_boxes):
                continue
            vr, vc = spec["visible_offset"]
            patch = spec["visible"]
            grid[r0 + vr : r0 + vr + patch.shape[0], c0 + vc : c0 + vc + patch.shape[1]] = patch
            placed_boxes.append(box)
            placed = True
            break
        if not placed:
            return None
    return grid


def _choose_orientations(count: int) -> list[str]:
    if count == 2:
        orientations = ["vertical", "horizontal"]
    else:
        orientations = ["vertical", "horizontal", random.choice(["vertical", "horizontal"])]
    random.shuffle(orientations)
    return orientations


def _gen_attempt() -> np.ndarray | None:
    obj_count = random.choice([2, 3])
    orientations = _choose_orientations(obj_count)

    markers: list[int] = []
    specs = []
    for orientation in orientations:
        spec = _make_object_spec(markers)
        spec["orientation"] = orientation
        full = spec["full"]
        side = spec["side"]
        half_span = side // 2 + 1
        spec["visible"] = full[:, :half_span] if orientation == "vertical" else full[:half_span, :]
        if orientation == "vertical":
            spec["reserve_shape"] = (side + 4, side + 6)
            spec["visible_offset"] = (2, 3)
        else:
            spec["reserve_shape"] = (side + 6, side + 4)
            spec["visible_offset"] = (3, 2)
        specs.append(spec)
        markers.append(spec["marker"])

    min_h = max(15, max(spec["reserve_shape"][0] for spec in specs))
    min_w = max(15, max(spec["reserve_shape"][1] for spec in specs))
    max_h = min(30, max(min_h, 18 + 4 * (obj_count - 2)) + 6)
    max_w = min(30, max(min_w, 18 + 4 * (obj_count - 2)) + 6)

    for _ in range(PLACEMENT_ATTEMPTS):
        h = random.randint(min_h, max_h)
        w = random.randint(min_w, max_w)
        grid = _place_specs(specs, h, w)
        if grid is None:
            continue

        boxes = _find_components(grid, bg=BG)
        if len(boxes) != obj_count:
            continue
        try:
            out = main(grid)
        except AssertionError:
            continue
        if out.shape != grid.shape or np.array_equal(out, grid):
            continue
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid half-concentric-square input")
