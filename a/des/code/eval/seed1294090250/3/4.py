from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
TARGET = 5
PATH = 9
SOURCE_COLORS = (1, 2, 3, 4, 6, 7, 8)
TURN_CHOICES = (1, 2, 3, 4)
MAX_GEN_ATTEMPTS = 400

DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
RIGHT = (0, 1)
LEFT = (0, -1)
UP = (-1, 0)
DOWN = (1, 0)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _shape_signature_from_mask(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    coords = np.argwhere(mask)
    assert len(coords) > 0
    coords = coords - coords.min(axis=0)
    return tuple(sorted((int(r), int(c)) for r, c in coords))


def _boundary(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, constant_values=False)
    up = padded[:-2, 1:-1]
    down = padded[2:, 1:-1]
    left = padded[1:-1, :-2]
    right = padded[1:-1, 2:]
    return mask & ~(up & down & left & right)


def _shape_depth(mask: np.ndarray) -> int:
    remaining = np.asarray(mask, dtype=bool).copy()
    depth = 0
    while remaining.any():
        layer = _boundary(remaining)
        remaining &= ~layer
        depth += 1
    return depth


def _path_turns(path_mask: np.ndarray) -> int:
    pts = [tuple(map(int, rc)) for rc in np.argwhere(path_mask)]
    assert pts, "path must be non-empty"
    if len(pts) == 1:
        return 0

    pt_set = set(pts)
    nbrs: dict[tuple[int, int], list[tuple[int, int]]] = {}
    endpoints: list[tuple[int, int]] = []
    for cell in pts:
        cell_nbrs = []
        for dr, dc in DIRS4:
            nxt = (cell[0] + dr, cell[1] + dc)
            if nxt in pt_set:
                cell_nbrs.append(nxt)
        assert len(cell_nbrs) in (1, 2), "path must be a simple chain"
        nbrs[cell] = cell_nbrs
        if len(cell_nbrs) == 1:
            endpoints.append(cell)

    assert len(endpoints) == 2, "path must have exactly two endpoints"

    ordered = [endpoints[0]]
    prev: tuple[int, int] | None = None
    cur = endpoints[0]
    while True:
        options = [nxt for nxt in nbrs[cur] if nxt != prev]
        if not options:
            break
        assert len(options) == 1
        prev, cur = cur, options[0]
        ordered.append(cur)

    assert len(ordered) == len(pts), "path traversal must visit every path cell"

    step_dirs = [
        (b[0] - a[0], b[1] - a[1])
        for a, b in zip(ordered, ordered[1:], strict=False)
    ]
    return sum(d0 != d1 for d0, d1 in zip(step_dirs, step_dirs[1:], strict=False))


def _paint_target_layers(
    out: np.ndarray,
    target_mask: np.ndarray,
    source_color: int,
    turns: int,
    bg: int,
) -> None:
    remaining = target_mask.copy()
    for layer_idx in range(turns):
        if not remaining.any():
            break
        layer = _boundary(remaining)
        fill = source_color if layer_idx % 2 == 0 else bg
        out[layer] = fill
        remaining &= ~layer


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair each colored source shape with its grey target through the maroon path,
    count the path turns, and apply that many alternating concentric layers to the target.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(np.bincount(g.ravel()).argmax())
    out = g.copy()

    for comp in _components(g != bg):
        comp_mask = np.zeros_like(g, dtype=bool)
        for r, c in comp:
            comp_mask[r, c] = True

        colors = {int(g[r, c]) for r, c in comp}
        assert PATH in colors and TARGET in colors, "each object pair must include the path and grey target"
        src_colors = sorted(colors - {bg, PATH, TARGET})
        assert len(src_colors) == 1, "each pair must have exactly one source color"
        source_color = int(src_colors[0])

        path_mask = comp_mask & (g == PATH)
        target_mask = comp_mask & (g == TARGET)
        source_mask = comp_mask & (g == source_color)

        assert len(_components(path_mask)) == 1
        assert len(_components(target_mask)) == 1
        assert len(_components(source_mask)) == 1
        assert _shape_signature_from_mask(source_mask) == _shape_signature_from_mask(target_mask)

        turns = _path_turns(path_mask)
        _paint_target_layers(out, target_mask, source_color, turns, bg)

    return out


def _random_shape(rng: random.Random) -> np.ndarray:
    kind = rng.choice(("rect", "plus", "diamond"))
    if kind == "rect":
        h = rng.randint(3, 5)
        w = rng.randint(3, 5)
        return np.ones((h, w), dtype=bool)
    if kind == "plus":
        arm = rng.randint(1, 2)
        size = 2 * arm + 1
        mask = np.zeros((size, size), dtype=bool)
        mid = arm
        mask[mid, :] = True
        mask[:, mid] = True
        return mask

    radius = rng.randint(1, 2)
    size = 2 * radius + 1
    rows, cols = np.indices((size, size))
    return (np.abs(rows - radius) + np.abs(cols - radius)) <= radius


def _side_from_dir(direction: tuple[int, int]) -> str:
    mapping = {
        RIGHT: "right",
        LEFT: "left",
        UP: "top",
        DOWN: "bottom",
    }
    return mapping[direction]


def _anchor_for_dir(mask: np.ndarray, direction: tuple[int, int]) -> tuple[int, int]:
    side = _side_from_dir(direction)
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    if side == "right":
        edge = max(c for _, c in coords)
        candidates = sorted((r, c) for r, c in coords if c == edge)
    elif side == "left":
        edge = min(c for _, c in coords)
        candidates = sorted((r, c) for r, c in coords if c == edge)
    elif side == "top":
        edge = min(r for r, _ in coords)
        candidates = sorted((r, c) for r, c in coords if r == edge)
    else:
        edge = max(r for r, _ in coords)
        candidates = sorted((r, c) for r, c in coords if r == edge)
    return candidates[len(candidates) // 2]


def _build_path(rng: random.Random, turns: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    dirs: list[tuple[int, int]] = [RIGHT]
    for idx in range(turns):
        if idx % 2 == 0:
            dirs.append((rng.choice((-1, 1)), 0))
        else:
            dirs.append(RIGHT)

    coords = [(0, 0)]
    cur_r, cur_c = 0, 0
    for dr, dc in dirs:
        if dc != 0:
            seg_len = rng.randint(2, 4)
        else:
            seg_len = rng.randint(2, 3)
        for _ in range(seg_len - 1):
            cur_r += dr
            cur_c += dc
            coords.append((cur_r, cur_c))

    return coords, dirs


def _shift_cells(cells: set[tuple[int, int]], dr: int, dc: int) -> set[tuple[int, int]]:
    return {(r + dr, c + dc) for r, c in cells}


def _touching_path_cells(shape_cells: set[tuple[int, int]], path_cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    touching = set()
    for r, c in path_cells:
        for dr, dc in DIRS4:
            if (r + dr, c + dc) in shape_cells:
                touching.add((r, c))
                break
    return touching


def _touches(shape_a: set[tuple[int, int]], shape_b: set[tuple[int, int]]) -> bool:
    for r, c in shape_a:
        for dr, dc in DIRS4:
            if (r + dr, c + dc) in shape_b:
                return True
    return False


def _trim_patch(patch: np.ndarray) -> np.ndarray:
    coords = np.argwhere(patch != BG)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return patch[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _paste(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    view = canvas[r0 : r0 + h, c0 : c0 + w]
    mask = patch != BG
    view[mask] = patch[mask]


def _build_pair_patch(
    rng: random.Random,
    source_color: int,
    relation: str,
) -> tuple[np.ndarray, int, int]:
    for _ in range(200):
        shape = _random_shape(rng)
        depth = _shape_depth(shape)

        if relation == "fit":
            turn_options = [t for t in TURN_CHOICES if t <= depth]
        elif relation == "saturate":
            turn_options = [t for t in TURN_CHOICES if t > depth]
        else:
            turn_options = list(TURN_CHOICES)
        if not turn_options:
            continue

        turns = rng.choice(turn_options)
        path_coords, dirs = _build_path(rng, turns)
        start = path_coords[0]
        end = path_coords[-1]
        source_dir = dirs[0]
        target_dir = (-dirs[-1][0], -dirs[-1][1])

        source_anchor = _anchor_for_dir(shape, source_dir)
        target_anchor = _anchor_for_dir(shape, target_dir)
        shape_coords = {tuple(map(int, rc)) for rc in np.argwhere(shape)}

        src_off_r = start[0] - source_dir[0] - source_anchor[0]
        src_off_c = start[1] - source_dir[1] - source_anchor[1]
        tgt_off_r = end[0] - target_dir[0] - target_anchor[0]
        tgt_off_c = end[1] - target_dir[1] - target_anchor[1]

        source_cells = _shift_cells(shape_coords, src_off_r, src_off_c)
        target_cells = _shift_cells(shape_coords, tgt_off_r, tgt_off_c)
        path_cells = set(path_coords)

        if source_cells & target_cells:
            continue
        if source_cells & path_cells:
            continue
        if target_cells & path_cells:
            continue
        if _touches(source_cells, target_cells):
            continue

        if _touching_path_cells(source_cells, path_cells) != {start}:
            continue
        if _touching_path_cells(target_cells, path_cells) != {end}:
            continue

        all_cells = source_cells | target_cells | path_cells
        min_r, min_c, max_r, max_c = _bbox(all_cells)
        shift_r = -min_r
        shift_c = -min_c
        source_cells = _shift_cells(source_cells, shift_r, shift_c)
        target_cells = _shift_cells(target_cells, shift_r, shift_c)
        path_cells = _shift_cells(path_cells, shift_r, shift_c)
        max_r += shift_r
        max_c += shift_c

        patch = np.full((max_r + 1, max_c + 1), BG, dtype=int)
        for r, c in source_cells:
            patch[r, c] = source_color
        for r, c in target_cells:
            patch[r, c] = TARGET
        for r, c in path_cells:
            assert patch[r, c] == BG
            patch[r, c] = PATH

        rot_k = rng.randrange(4)
        patch = np.rot90(patch, rot_k).copy()
        if rng.random() < 0.5:
            patch = np.fliplr(patch).copy()
        return _trim_patch(patch), turns, depth

    raise RuntimeError("failed to build a valid pair patch")


def _row_sizes(rng: random.Random, n: int) -> list[int]:
    if n == 2:
        return [2] if rng.random() < 0.5 else [1, 1]
    if n == 3:
        roll = rng.random()
        if roll < 0.45:
            return [2, 1]
        if roll < 0.9:
            return [1, 2]
        return [1, 1, 1]
    raise AssertionError("unsupported patch count")


def _pack_patches(rng: random.Random, patches: list[np.ndarray]) -> np.ndarray:
    patches = patches[:]
    rng.shuffle(patches)
    sizes = _row_sizes(rng, len(patches))

    rows: list[list[np.ndarray]] = []
    idx = 0
    for size in sizes:
        rows.append(patches[idx : idx + size])
        idx += size

    row_heights: list[int] = []
    row_widths: list[int] = []
    row_hgaps: list[list[int]] = []
    for row in rows:
        gaps = [rng.randint(2, 4) for _ in range(max(0, len(row) - 1))]
        row_hgaps.append(gaps)
        row_heights.append(max(patch.shape[0] for patch in row))
        row_widths.append(sum(patch.shape[1] for patch in row) + sum(gaps))

    vgaps = [rng.randint(2, 4) for _ in range(max(0, len(rows) - 1))]
    layout_h = sum(row_heights) + sum(vgaps)
    layout_w = max(row_widths)

    top = rng.randint(1, 4)
    bottom = rng.randint(1, 4)
    left = rng.randint(1, 4)
    right = rng.randint(1, 4)
    extra_h = rng.randint(0, 4)
    extra_w = rng.randint(0, 4)

    canvas = np.full((layout_h + top + bottom + extra_h, layout_w + left + right + extra_w), BG, dtype=int)

    cur_y = top + rng.randint(0, extra_h)
    for row_idx, row in enumerate(rows):
        row_h = row_heights[row_idx]
        row_w = row_widths[row_idx]
        row_x = left + rng.randint(0, canvas.shape[1] - left - right - row_w)
        cur_x = row_x
        for patch_idx, patch in enumerate(row):
            patch_y = cur_y + rng.randint(0, row_h - patch.shape[0])
            _paste(canvas, patch, patch_y, cur_x)
            cur_x += patch.shape[1]
            if patch_idx < len(row_hgaps[row_idx]):
                cur_x += row_hgaps[row_idx][patch_idx]
        cur_y += row_h
        if row_idx < len(vgaps):
            cur_y += vgaps[row_idx]

    return canvas


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    pair_count = rng.choice((2, 3))
    relations = ["fit", "saturate"]
    if pair_count == 3:
        relations.append("free")
    rng.shuffle(relations)

    source_colors = list(SOURCE_COLORS)
    rng.shuffle(source_colors)

    patches: list[np.ndarray] = []
    turns_seen: list[int] = []
    has_fit = False
    has_saturate = False
    for idx in range(pair_count):
        patch, turns, depth = _build_pair_patch(rng, source_colors[idx], relations[idx])
        patches.append(patch)
        turns_seen.append(turns)
        has_fit |= turns <= depth
        has_saturate |= turns > depth

    if len(set(turns_seen)) < 2 or not has_fit or not has_saturate:
        return None

    grid = _pack_patches(rng, patches)
    if int(np.bincount(grid.ravel()).argmax()) != BG:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {MAX_GEN_ATTEMPTS} attempts")
