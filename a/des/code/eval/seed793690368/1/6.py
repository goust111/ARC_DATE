from __future__ import annotations

from collections import deque
import random

import numpy as np


BG_COLOR = 0
LINE_COLOR = 5
OBJECT_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
SQUARE_SIZES = (3, 4)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

GEN_ATTEMPTS = 400
PLACE_ATTEMPTS = 300
BLOCK_ATTEMPTS = 150


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[dict]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue

            queue: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            coord_arr = np.array(coords, dtype=int)
            r0, c0 = coord_arr.min(axis=0)
            r1, c1 = coord_arr.max(axis=0)
            comps.append(
                {
                    "coords": coord_arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                }
            )

    return comps


def _object_components(grid: np.ndarray, bg: int) -> tuple[list[dict], np.ndarray]:
    mask = (grid != bg) & (grid != LINE_COLOR)
    comps = _components(mask)
    labels = np.full(grid.shape, -1, dtype=int)
    objects: list[dict] = []

    for idx, comp in enumerate(comps):
        r0, c0, r1, c1 = comp["bbox"]
        patch = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
        side = int(r1 - r0 + 1)

        assert patch.shape[0] == patch.shape[1] and side in SQUARE_SIZES
        assert int(comp["coords"].shape[0]) == side * side
        assert np.all(patch != bg) and np.all(patch != LINE_COLOR)

        for r, c in comp["coords"]:
            labels[int(r), int(c)] = idx

        objects.append(
            {
                "coords": comp["coords"],
                "bbox": comp["bbox"],
                "patch": patch,
                "side": side,
            }
        )

    return objects, labels


def _ordered_path(coords: np.ndarray) -> list[tuple[int, int]]:
    cells = {tuple(map(int, rc)) for rc in coords}
    assert len(cells) >= 2
    neighbors = {
        cell: [(cell[0] + dr, cell[1] + dc) for dr, dc in DIRS4 if (cell[0] + dr, cell[1] + dc) in cells]
        for cell in cells
    }

    assert all(len(adj) <= 2 for adj in neighbors.values())
    endpoints = [cell for cell, adj in neighbors.items() if len(adj) == 1]
    assert len(endpoints) == 2

    start = min(endpoints)
    path = [start]
    prev: tuple[int, int] | None = None
    cur = start

    while True:
        nxts = [cell for cell in neighbors[cur] if cell != prev]
        if not nxts:
            break
        assert len(nxts) == 1
        prev, cur = cur, nxts[0]
        path.append(cur)

    assert len(path) == len(cells)
    return path


def _segment_dirs(path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    dirs: list[tuple[int, int]] = []
    for (r0, c0), (r1, c1) in zip(path, path[1:]):
        dr, dc = r1 - r0, c1 - c0
        assert abs(dr) + abs(dc) == 1
        step = (int(dr), int(dc))
        if not dirs or dirs[-1] != step:
            dirs.append(step)
    return dirs


def _axis(step: tuple[int, int]) -> str:
    return "H" if step[0] == 0 else "V"


def _turn_sense(step1: tuple[int, int], step2: tuple[int, int]) -> str:
    cross = step1[0] * step2[1] - step1[1] * step2[0]
    assert cross != 0
    return "cw" if cross < 0 else "ccw"


def _classify_line(coords: np.ndarray) -> dict:
    path = _ordered_path(coords)
    dirs = _segment_dirs(path)

    if len(dirs) == 1:
        return {"kind": "straight", "main_axis": _axis(dirs[0])}

    if len(dirs) == 2:
        assert _axis(dirs[0]) != _axis(dirs[1])
        return {"kind": "L", "turn": _turn_sense(dirs[0], dirs[1])}

    assert len(dirs) == 3
    axes = [_axis(step) for step in dirs]
    assert axes[0] == axes[2] and axes[0] != axes[1]
    return {"kind": "Z", "main_axis": axes[0]}


def _analyze_pairs(grid: np.ndarray) -> tuple[int, list[dict], list[dict]]:
    bg = _dominant_color(grid)
    objects, labels = _object_components(grid, bg)
    line_comps = _components(grid == LINE_COLOR)

    assert line_comps
    assert len(objects) == 2 * len(line_comps)

    usage = [0] * len(objects)
    pairs: list[dict] = []
    h, w = grid.shape

    for comp in line_comps:
        info = _classify_line(comp["coords"])
        neighbors: set[int] = set()

        for r, c in comp["coords"]:
            for dr, dc in DIRS4:
                nr, nc = int(r + dr), int(c + dc)
                if 0 <= nr < h and 0 <= nc < w and labels[nr, nc] >= 0:
                    neighbors.add(int(labels[nr, nc]))

        assert len(neighbors) == 2
        a, b = sorted(neighbors)
        usage[a] += 1
        usage[b] += 1
        info["objects"] = (a, b)
        pairs.append(info)

    assert all(count == 1 for count in usage)
    return bg, objects, pairs


def _color_order(patch: np.ndarray, bg: int) -> list[int]:
    order: list[int] = []
    seen: set[int] = set()
    for value in patch.ravel():
        color = int(value)
        if color == bg or color in seen:
            continue
        seen.add(color)
        order.append(color)
    return order


def _swap_colors(patch_a: np.ndarray, patch_b: np.ndarray, bg: int) -> tuple[np.ndarray, np.ndarray]:
    colors_a = _color_order(patch_a, bg)
    colors_b = _color_order(patch_b, bg)
    assert len(colors_a) == len(colors_b) >= 1

    out_a = patch_a.copy()
    out_b = patch_b.copy()
    for src, dst in zip(colors_a, colors_b):
        out_a[patch_a == src] = dst
    for src, dst in zip(colors_b, colors_a):
        out_b[patch_b == src] = dst
    return out_a, out_b


def _write_patch(grid: np.ndarray, bbox: tuple[int, int, int, int], patch: np.ndarray) -> None:
    r0, c0, r1, c1 = bbox
    assert patch.shape == (r1 - r0 + 1, c1 - c0 + 1)
    grid[r0 : r1 + 1, c0 : c1 + 1] = patch


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the grey connector shape to choose a per-pair object transform.

    Input structure
    - Background is black.
    - Each pair consists of two 3x3 or 4x4 colored square blocks.
    - A 1-cell-wide grey path connects the pair and is either straight, L-shaped,
      or Z/S-shaped.

    Rule
    - Straight path: swap the two blocks' color palettes.
    - L path: rotate both blocks 90 degrees in place. The turn direction is read
      from the line path using the lexicographically smaller grey endpoint as the
      traversal start.
    - Z/S path: mirror both blocks in place along the path's main axis.
    - Keep every grey path unchanged.
    """
    g = np.asarray(grid, dtype=int)
    bg, objects, pairs = _analyze_pairs(g)
    out = g.copy()

    for pair in pairs:
        ia, ib = pair["objects"]
        obj_a = objects[ia]
        obj_b = objects[ib]
        patch_a = obj_a["patch"]
        patch_b = obj_b["patch"]

        if pair["kind"] == "straight":
            new_a, new_b = _swap_colors(patch_a, patch_b, bg)
        elif pair["kind"] == "L":
            k = -1 if pair["turn"] == "cw" else 1
            new_a = np.rot90(patch_a, k).copy()
            new_b = np.rot90(patch_b, k).copy()
        else:
            if pair["main_axis"] == "H":
                new_a = np.fliplr(patch_a).copy()
                new_b = np.fliplr(patch_b).copy()
            else:
                new_a = np.flipud(patch_a).copy()
                new_b = np.flipud(patch_b).copy()

        _write_patch(out, obj_a["bbox"], new_a)
        _write_patch(out, obj_b["bbox"], new_b)

    return out


def _paste(canvas: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    assert np.all(canvas[top : top + h, left : left + w] == BG_COLOR)
    canvas[top : top + h, left : left + w] = patch


def _draw_segment(canvas: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> None:
    r0, c0 = start
    r1, c1 = end
    if r0 == r1:
        lo, hi = sorted((c0, c1))
        canvas[r0, lo : hi + 1] = LINE_COLOR
    else:
        assert c0 == c1
        lo, hi = sorted((r0, r1))
        canvas[lo : hi + 1, c0] = LINE_COLOR


def _two_color_block(size: int, colors: tuple[int, int]) -> np.ndarray:
    cells = [(r, c) for r in range(size) for c in range(size)]
    for _ in range(BLOCK_ATTEMPTS):
        patch = np.full((size, size), colors[0], dtype=int)
        k = random.randint(1, size * size - 1)
        for r, c in random.sample(cells, k):
            patch[r, c] = colors[1]
        if len({int(v) for v in patch.ravel()}) == 2:
            return patch
    raise RuntimeError("failed to build two-color block")


def _block_for_rotation(size: int) -> np.ndarray:
    for _ in range(BLOCK_ATTEMPTS):
        colors = tuple(random.sample(OBJECT_COLORS, 2))
        patch = _two_color_block(size, colors)
        if not np.array_equal(patch, np.rot90(patch, 1)) and not np.array_equal(patch, np.rot90(patch, -1)):
            return patch
    raise RuntimeError("failed to build non-rotational block")


def _block_for_mirror(size: int) -> np.ndarray:
    for _ in range(BLOCK_ATTEMPTS):
        colors = tuple(random.sample(OBJECT_COLORS, 2))
        patch = _two_color_block(size, colors)
        if not np.array_equal(patch, np.fliplr(patch)) and not np.array_equal(patch, np.flipud(patch)):
            return patch
    raise RuntimeError("failed to build non-mirror block")


def _block_for_swap(size: int, color_count: int) -> np.ndarray:
    if color_count == 1:
        color = random.choice(OBJECT_COLORS)
        return np.full((size, size), color, dtype=int)
    colors = tuple(random.sample(OBJECT_COLORS, 2))
    return _two_color_block(size, colors)


def _pair_straight() -> np.ndarray:
    size_a = random.choice(SQUARE_SIZES)
    size_b = random.choice(SQUARE_SIZES)
    color_count = random.choice((1, 2))

    patch_a = _block_for_swap(size_a, color_count)
    while True:
        patch_b = _block_for_swap(size_b, color_count)
        if not np.array_equal(_swap_colors(patch_a, patch_b, BG_COLOR)[0], patch_a):
            break

    row_a = size_a // 2
    row_b = size_b // 2
    line_row = max(row_a, row_b)
    top_a = line_row - row_a
    top_b = line_row - row_b
    left_a = 0
    gap = random.randint(2, 5)
    left_b = size_a + gap

    height = max(top_a + size_a, top_b + size_b)
    width = left_b + size_b
    canvas = np.full((height, width), BG_COLOR, dtype=int)
    _paste(canvas, patch_a, top_a, left_a)
    _paste(canvas, patch_b, top_b, left_b)
    _draw_segment(canvas, (line_row, size_a), (line_row, left_b - 1))
    return canvas


def _pair_l() -> np.ndarray:
    size_a = random.choice(SQUARE_SIZES)
    size_b = random.choice(SQUARE_SIZES)
    patch_a = _block_for_rotation(size_a)
    patch_b = _block_for_rotation(size_b)

    top_a, left_a = 0, 0
    top_b = size_a + random.randint(2, 4)
    left_b = size_a + random.randint(2, 4)

    start = (size_a // 2, size_a)
    end = (top_b - 1, left_b + size_b // 2)
    turn = (start[0], end[1])

    height = max(size_a, top_b + size_b)
    width = max(size_a, left_b + size_b)
    canvas = np.full((height, width), BG_COLOR, dtype=int)
    _paste(canvas, patch_a, top_a, left_a)
    _paste(canvas, patch_b, top_b, left_b)
    _draw_segment(canvas, start, turn)
    _draw_segment(canvas, turn, end)
    return canvas


def _pair_z() -> np.ndarray:
    size_a = random.choice(SQUARE_SIZES)
    size_b = random.choice(SQUARE_SIZES)
    patch_a = _block_for_mirror(size_a)
    patch_b = _block_for_mirror(size_b)

    top_a, left_a = 0, 0
    top_b = size_a + random.randint(1, 3)
    left_b = size_a + random.randint(5, 7)

    start = (size_a // 2, size_a)
    end = (top_b + size_b // 2, left_b - 1)
    mid_col = random.randint(size_a + 1, left_b - 2)
    turn1 = (start[0], mid_col)
    turn2 = (end[0], mid_col)

    height = max(size_a, top_b + size_b)
    width = left_b + size_b
    canvas = np.full((height, width), BG_COLOR, dtype=int)
    _paste(canvas, patch_a, top_a, left_a)
    _paste(canvas, patch_b, top_b, left_b)
    _draw_segment(canvas, start, turn1)
    _draw_segment(canvas, turn1, turn2)
    _draw_segment(canvas, turn2, end)
    return canvas


def _random_symmetry(grid: np.ndarray) -> np.ndarray:
    out = np.rot90(grid, random.randrange(4)).copy()
    if random.random() < 0.5:
        out = np.fliplr(out).copy()
    return out


def _place_pairs(patches: list[np.ndarray]) -> np.ndarray | None:
    max_h = max(patch.shape[0] for patch in patches)
    max_w = max(patch.shape[1] for patch in patches)

    for _ in range(PLACE_ATTEMPTS):
        height = random.randint(max(26, max_h + 4), 36)
        width = random.randint(max(26, max_w + 4), 36)
        grid = np.full((height, width), BG_COLOR, dtype=int)
        occupied = np.zeros((height, width), dtype=bool)
        order = list(range(len(patches)))
        random.shuffle(order)
        placed: list[tuple[int, int, int]] = []

        ok = True
        for idx in order:
            patch = patches[idx]
            ph, pw = patch.shape
            if ph > height or pw > width:
                ok = False
                break

            found = False
            for _ in range(PLACE_ATTEMPTS):
                top = random.randint(0, height - ph)
                left = random.randint(0, width - pw)
                r0 = max(0, top - 1)
                c0 = max(0, left - 1)
                r1 = min(height, top + ph + 1)
                c1 = min(width, left + pw + 1)
                if occupied[r0:r1, c0:c1].any():
                    continue
                placed.append((idx, top, left))
                occupied[r0:r1, c0:c1] = True
                found = True
                break

            if not found:
                ok = False
                break

        if not ok:
            continue

        for idx, top, left in placed:
            _paste(grid, patches[idx], top, left)
        return grid

    return None


def _generated_pair(kind: str) -> np.ndarray:
    if kind == "straight":
        return _pair_straight()
    if kind == "L":
        return _pair_l()
    return _pair_z()


def _gen_attempt() -> np.ndarray | None:
    kinds = ["straight", "L", "Z"]
    if random.random() < 0.5:
        kinds.append(random.choice(kinds))
    random.shuffle(kinds)

    patches = [_generated_pair(kind) for kind in kinds]
    grid = _place_pairs(patches)
    if grid is None:
        return None

    grid = _random_symmetry(grid)

    try:
        bg, objects, pairs = _analyze_pairs(grid)
        assert bg == BG_COLOR
        assert len(objects) == 2 * len(kinds)
        assert {pair["kind"] for pair in pairs} == {"straight", "L", "Z"}
        out = main(grid)
        assert not np.array_equal(out, grid)
        for obj in objects:
            r0, c0, r1, c1 = obj["bbox"]
            assert not np.array_equal(grid[r0 : r1 + 1, c0 : c1 + 1], out[r0 : r1 + 1, c0 : c1 + 1])
    except AssertionError:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
