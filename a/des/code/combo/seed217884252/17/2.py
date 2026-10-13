from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


GEN_BG = 7
GEN_FRAME = 5
GEN_COLORS = [1, 2, 3, 4, 6, 8, 9]
GEN_RETRIES = 400

_UNDO_ROTATIONS = {"tl": 0, "tr": 1, "br": 2, "bl": 3}


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _detect_frames(grid: np.ndarray, bg: int) -> list[dict]:
    candidates: list[dict] = []

    for raw_color in np.unique(grid):
        color = int(raw_color)
        if color == bg:
            continue
        for coords in _components(grid == color):
            rows = coords[:, 0]
            cols = coords[:, 1]
            r0, r1 = int(rows.min()), int(rows.max())
            c0, c1 = int(cols.min()), int(cols.max())
            height = int(r1 - r0 + 1)
            width = int(c1 - c0 + 1)
            if height < 4 or width < 4:
                continue

            coord_set = {(int(r), int(c)) for r, c in coords}
            if any(r not in (r0, r1) and c not in (c0, c1) for r, c in coord_set):
                continue

            corners = {
                "tl": (r0, c0),
                "tr": (r0, c1),
                "br": (r1, c1),
                "bl": (r1, c0),
            }
            missing = [name for name, pos in corners.items() if pos not in coord_set]
            if len(missing) != 1:
                continue
            missing_name = missing[0]
            marker_r, marker_c = corners[missing_name]

            expected_cells = 2 * height + 2 * width - 5
            if len(coord_set) != expected_cells:
                continue
            if int(grid[marker_r, marker_c]) in (bg, color):
                continue

            ok = True
            for c in range(c0, c1 + 1):
                if (r0, c) == (marker_r, marker_c):
                    continue
                if int(grid[r0, c]) != color:
                    ok = False
                if (r1, c) == (marker_r, marker_c):
                    continue
                if int(grid[r1, c]) != color:
                    ok = False
            for r in range(r0, r1 + 1):
                if (r, c0) == (marker_r, marker_c):
                    continue
                if int(grid[r, c0]) != color:
                    ok = False
                if (r, c1) == (marker_r, marker_c):
                    continue
                if int(grid[r, c1]) != color:
                    ok = False
            if not ok:
                continue

            candidates.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "frame_color": color,
                    "marker_corner": missing_name,
                    "outer_shape": (height, width),
                }
            )

    groups: defaultdict[tuple[int, int, int], list[dict]] = defaultdict(list)
    for item in candidates:
        height, width = item["outer_shape"]
        groups[(int(item["frame_color"]), int(height), int(width))].append(item)

    assert groups, "expected at least one framed object"
    best_group = max(groups.values(), key=lambda items: (len(items), items[0]["outer_shape"][0] * items[0]["outer_shape"][1]))
    return best_group


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore each framed scatter pattern to its standard orientation, then overlay them.

    Marker corner -> frame rotation:
    - top-left: 0 degrees
    - top-right: 90 degrees
    - bottom-right: 180 degrees
    - bottom-left: 270 degrees

    The inverse rotation is applied to the frame interior before all restored
    interiors are overlaid on a shared canvas in reading order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _background_color(g)
    frames = _detect_frames(g, bg)
    frames.sort(key=lambda item: (int(item["bbox"][0]), int(item["bbox"][1])))

    r0, c0, r1, c1 = frames[0]["bbox"]
    inner_h = int(r1 - r0 - 1)
    inner_w = int(c1 - c0 - 1)
    assert inner_h > 0 and inner_w > 0
    out = np.full((inner_h, inner_w), bg, dtype=int)

    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        assert inner.shape == out.shape, "all frame interiors must have the same size"
        k = _UNDO_ROTATIONS[str(frame["marker_corner"])]
        restored = np.rot90(inner, k=k)
        mask = restored != bg
        out[mask] = restored[mask]

    return out


def _interesting_canvas(canvas: np.ndarray, bg: int) -> bool:
    occupied = np.argwhere(canvas != bg)
    if occupied.shape[0] < 4:
        return False
    if len({(int(r), int(c)) for r, c in occupied}) != occupied.shape[0]:
        return False
    if len(np.unique(canvas[canvas != bg])) < 2:
        return False
    if len({int(r) for r, _ in occupied}) == 1:
        return False
    if len({int(c) for _, c in occupied}) == 1:
        return False
    for k in (1, 2, 3):
        if np.array_equal(canvas, np.rot90(canvas, k=k)):
            return False
    return True


def _make_target(rng: random.Random, side: int, frame_count: int) -> np.ndarray | None:
    cells = [(r, c) for r in range(side) for c in range(side)]
    min_points = max(2 * frame_count, side + 1)
    max_points = min(len(cells) // 2, side + 7)
    if min_points > max_points:
        return None

    for _ in range(200):
        point_count = rng.randint(min_points, max_points)
        colors = rng.sample(GEN_COLORS, k=rng.randint(2, min(4, len(GEN_COLORS))))
        target = np.full((side, side), GEN_BG, dtype=int)
        for r, c in rng.sample(cells, k=point_count):
            target[r, c] = rng.choice(colors)
        if _interesting_canvas(target, GEN_BG):
            return target
    return None


def _split_target(target: np.ndarray, frame_count: int, rng: random.Random) -> list[np.ndarray] | None:
    occupied = [(int(r), int(c)) for r, c in np.argwhere(target != GEN_BG)]
    if len(occupied) < 2 * frame_count:
        return None

    rng.shuffle(occupied)
    layers = [np.full_like(target, GEN_BG) for _ in range(frame_count)]
    idx = 0
    for layer_id in range(frame_count):
        for _ in range(2):
            r, c = occupied[idx]
            layers[layer_id][r, c] = int(target[r, c])
            idx += 1

    for r, c in occupied[idx:]:
        copies = rng.randint(1, min(3, frame_count))
        for layer_id in rng.sample(range(frame_count), k=copies):
            layers[layer_id][r, c] = int(target[r, c])

    union = np.full_like(target, GEN_BG)
    for layer in layers:
        mask = layer != GEN_BG
        union[mask] = layer[mask]
    if not np.array_equal(union, target):
        return None
    if not any(not np.array_equal(layer, target) for layer in layers):
        return None
    if len({layer.tobytes() for layer in layers}) == 1:
        return None
    return layers


def _rotation_schedule(rng: random.Random, frame_count: int) -> list[int]:
    base = [0, 1, 2, 3]
    rng.shuffle(base)
    rotations: list[int] = []
    while len(rotations) < frame_count:
        rotations.extend(base)
        rng.shuffle(base)
    return rotations[:frame_count]


def _choose_layout(rng: random.Random, frame_count: int, outer: int) -> tuple[int, int, list[tuple[int, int]], int, int]:
    shape_options = []
    for rows in range(1, 4):
        for cols in range(1, 5):
            if rows * cols >= frame_count:
                shape_options.append((rows, cols))
    rows, cols = rng.choice(shape_options)

    row_gaps = [rng.randint(1, 4) for _ in range(max(0, rows - 1))]
    col_gaps = [rng.randint(1, 4) for _ in range(max(0, cols - 1))]
    top = rng.randint(1, 4)
    left = rng.randint(1, 4)
    bottom = rng.randint(1, 4)
    right = rng.randint(1, 4)

    row_starts = []
    cursor = top
    for row in range(rows):
        row_starts.append(cursor)
        if row < rows - 1:
            cursor += outer + row_gaps[row]
    total_h = cursor + outer + bottom

    col_starts = []
    cursor = left
    for col in range(cols):
        col_starts.append(cursor)
        if col < cols - 1:
            cursor += outer + col_gaps[col]
    total_w = cursor + outer + right

    slots = [(r0, c0) for r0 in row_starts for c0 in col_starts]
    chosen = rng.sample(slots, k=frame_count)
    return total_h, total_w, chosen, rows, cols


def _stamp_frame(
    grid: np.ndarray,
    top: int,
    left: int,
    inner: np.ndarray,
    rotation: int,
    marker_color: int,
) -> None:
    side = int(inner.shape[0])
    outer = side + 2
    rotated = np.rot90(inner, k=(-rotation) % 4)

    grid[top : top + outer, left : left + outer] = GEN_BG
    grid[top, left : left + outer] = GEN_FRAME
    grid[top + outer - 1, left : left + outer] = GEN_FRAME
    grid[top : top + outer, left] = GEN_FRAME
    grid[top : top + outer, left + outer - 1] = GEN_FRAME

    corner_name = {0: "tl", 1: "tr", 2: "br", 3: "bl"}[rotation]
    corner_pos = {
        "tl": (top, left),
        "tr": (top, left + outer - 1),
        "br": (top + outer - 1, left + outer - 1),
        "bl": (top + outer - 1, left),
    }[corner_name]
    grid[corner_pos] = marker_color
    grid[top + 1 : top + outer - 1, left + 1 : left + outer - 1] = rotated


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    side = rng.randint(4, 8)
    max_points = min((side * side) // 2, side + 7)
    max_frames = min(6, max_points // 2)
    if max_frames < 2:
        return None
    frame_count = rng.randint(2, max_frames)
    target = _make_target(rng, side, frame_count)
    if target is None:
        return None

    layers = _split_target(target, frame_count, rng)
    if layers is None:
        return None

    rotations = _rotation_schedule(rng, frame_count)
    for layer, rotation in zip(layers, rotations):
        if rotation != 0 and np.array_equal(layer, np.rot90(layer, k=(-rotation) % 4)):
            return None

    outer = side + 2
    total_h, total_w, positions, _rows, _cols = _choose_layout(rng, frame_count, outer)
    grid = np.full((total_h, total_w), GEN_BG, dtype=int)

    marker_colors = [rng.choice(GEN_COLORS) for _ in range(frame_count)]
    for (top, left), layer, rotation, marker_color in zip(positions, layers, rotations, marker_colors):
        _stamp_frame(grid, int(top), int(left), layer, int(rotation), int(marker_color))

    try:
        output = main(grid)
    except Exception:
        return None
    if not np.array_equal(output, target):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
