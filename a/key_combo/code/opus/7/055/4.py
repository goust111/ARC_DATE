from __future__ import annotations

from collections import deque
import random

import numpy as np


SEP = 7
BACKGROUND = 0
ACTIVE_COLORS = (1, 2, 3, 4)
DUMMY_COLORS = (5, 6, 8, 9)
OUTER_MARGIN = 2
MAX_GEN_ATTEMPTS = 400

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SHAPES_BY_AREA: dict[int, list[np.ndarray]] = {
    2: [
        np.array([(0, 0), (0, 1)], dtype=int),
        np.array([(0, 0), (1, 0)], dtype=int),
    ],
    3: [
        np.array([(0, 0), (0, 1), (1, 0)], dtype=int),
        np.array([(0, 0), (0, 1), (1, 1)], dtype=int),
        np.array([(0, 0), (1, 0), (1, 1)], dtype=int),
        np.array([(0, 1), (1, 0), (1, 1)], dtype=int),
    ],
    4: [
        np.array([(0, 0), (0, 1), (1, 0), (1, 1)], dtype=int),
    ],
}


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bands_from_separators(separators: list[int]) -> list[tuple[int, int]]:
    return [(a + 1, b - 1) for a, b in zip(separators, separators[1:]) if b - a > 1]


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_lattice(grid: np.ndarray) -> tuple[list[int], list[int], list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    sep_rows = [r for r in range(g.shape[0]) if np.all(g[r, :] == SEP)]
    sep_cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == SEP)]
    assert len(sep_rows) >= 3 and len(sep_cols) >= 3, "expected a separator lattice with outer bounds"
    row_bands = _bands_from_separators(sep_rows)
    col_bands = _bands_from_separators(sep_cols)
    assert row_bands and col_bands, "separator lattice must contain cell interiors"
    return sep_rows, sep_cols, row_bands, col_bands


def _pure_cells(
    grid: np.ndarray, row_bands: list[tuple[int, int]], col_bands: list[tuple[int, int]]
) -> dict[int, dict]:
    g = np.asarray(grid, dtype=int)
    pure: dict[int, dict] = {}
    for ri, (r0, r1) in enumerate(row_bands):
        for ci, (c0, c1) in enumerate(col_bands):
            cell = g[r0 : r1 + 1, c0 : c1 + 1]
            if np.all(cell == cell[0, 0]) and int(cell[0, 0]) != SEP:
                color = int(cell[0, 0])
                assert color not in pure, "pure-cell colors must be unique"
                pure[color] = {
                    "cell_rc": (ri, ci),
                    "bbox": (r0, c0, r1, c1),
                }
    assert pure, "expected at least one solid cell"
    return pure


def _sprite_side(coords: np.ndarray, sep_rows: list[int], sep_cols: list[int]) -> str:
    r0, c0, r1, c1 = _bbox(coords)
    sides = []
    if r1 < sep_rows[0]:
        sides.append("top")
    if r0 > sep_rows[-1]:
        sides.append("bottom")
    if c1 < sep_cols[0]:
        sides.append("left")
    if c0 > sep_cols[-1]:
        sides.append("right")
    assert len(sides) == 1, f"sprite must live in exactly one outer band: {sides}"
    return sides[0]


def _target_distance(side: str, coords: np.ndarray, target_bbox: tuple[int, int, int, int]) -> tuple[np.ndarray, int]:
    r0, c0, r1, c1 = _bbox(coords)
    tr0, tc0, tr1, tc1 = target_bbox
    if side == "left":
        return np.array([0, 1], dtype=int), int(tc0 - 1 - c1)
    if side == "right":
        return np.array([0, -1], dtype=int), int(c0 - (tc1 + 1))
    if side == "top":
        return np.array([1, 0], dtype=int), int(tr0 - 1 - r1)
    assert side == "bottom"
    return np.array([-1, 0], dtype=int), int(r0 - (tr1 + 1))


def _extract_sprites(
    grid: np.ndarray, pure: dict[int, dict], sep_rows: list[int], sep_cols: list[int]
) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    outer = np.zeros((h, w), dtype=bool)
    outer[: sep_rows[0], :] = True
    outer[sep_rows[-1] + 1 :, :] = True
    outer[:, : sep_cols[0]] = True
    outer[:, sep_cols[-1] + 1 :] = True
    mask = outer & (g != BACKGROUND) & (g != SEP)

    sprites: list[dict] = []
    for coords in _components(mask):
        colors = {int(g[r, c]) for r, c in coords}
        assert len(colors) == 1, "each outer sprite must be monochrome"
        color = next(iter(colors))
        assert color in pure, f"sprite color {color} must match a pure target cell"
        side = _sprite_side(coords, sep_rows, sep_cols)
        delta, distance = _target_distance(side, coords, pure[color]["bbox"])
        assert distance >= 1, "sprite must start away from its target stop line"
        sprites.append(
            {
                "color": color,
                "coords": coords,
                "side": side,
                "delta": delta,
                "distance": int(distance),
                "area": int(len(coords)),
            }
        )

    assert sprites, "expected at least one outer sprite"
    seen_colors = {sprite["color"] for sprite in sprites}
    assert len(seen_colors) == len(sprites), "each active color should have exactly one sprite"
    return sprites


def _settle_sprites(grid_shape: tuple[int, int], sprites: list[dict]) -> tuple[list[np.ndarray], list[int]]:
    h, w = map(int, grid_shape)
    blocked_trace = np.zeros((h, w), dtype=bool)
    finals: list[np.ndarray | None] = [None] * len(sprites)
    shifts = [0] * len(sprites)

    order = sorted(
        range(len(sprites)),
        key=lambda i: (
            -int(sprites[i]["area"]),
            int(sprites[i]["distance"]),
            int(sprites[i]["color"]),
            int(sprites[i]["coords"][:, 0].min()),
            int(sprites[i]["coords"][:, 1].min()),
        ),
    )

    for idx in order:
        sprite = sprites[idx]
        coords = sprite["coords"]
        delta = sprite["delta"]
        max_steps = int(sprite["distance"])
        steps = 0
        while steps < max_steps:
            candidate = coords + delta * (steps + 1)
            rows = candidate[:, 0]
            cols = candidate[:, 1]
            assert rows.min() >= 0 and rows.max() < h and cols.min() >= 0 and cols.max() < w
            if np.any(blocked_trace[rows, cols]):
                break
            steps += 1

        final_coords = coords + delta * steps
        finals[idx] = final_coords
        shifts[idx] = steps
        for step in range(steps + 1):
            trace = coords + delta * step
            blocked_trace[trace[:, 0], trace[:, 1]] = True

    return [coords for coords in finals if coords is not None], shifts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep the monochrome lattice cells and slide each matching outer sprite straight to the separator beside its target cell.

    Input structure:
    - Orange full-span separator rows and columns form a rectangular lattice, with a thin outer band around the lattice.
    - One contiguous rectangle of lattice cells is monochrome; its cells use distinct colors, while all other cells contain multicolor noise.
    - Small monochrome outer sprites sit in the top/bottom/left/right outer bands, each sharing a color with one monochrome cell.

    Rule:
    - Detect the lattice from the orange separator rows and columns, and identify every monochrome cell interior.
    - Erase every non-monochrome cell interior to black while keeping the separator lattice and the monochrome cells unchanged.
    - For each outer sprite, find the same-colored monochrome cell and slide the sprite horizontally or vertically toward that cell.
    - The sprite stops with its nearest edge on the separator line adjacent to the target cell; if multiple sprites' swept paths conflict, larger sprites continue first, and equal-sized conflicts favor the nearer sprite.

    Key insight:
    - The separator lattice is both the parser and the motion geometry: solid cells identify color-coded destinations, and the outer sprites are resolved by straight-line sliding constrained only by same-color matching and path-priority collisions.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    sep_rows, sep_cols, row_bands, col_bands = _find_lattice(g)
    pure = _pure_cells(g, row_bands, col_bands)
    sprites = _extract_sprites(g, pure, sep_rows, sep_cols)
    final_coords, _shifts = _settle_sprites(g.shape, sprites)

    out = np.full_like(g, BACKGROUND)
    out[sep_rows, :] = SEP
    out[:, sep_cols] = SEP
    for info in pure.values():
        r0, c0, r1, c1 = info["bbox"]
        color = next(color for color, cell in pure.items() if cell is info)
        out[r0 : r1 + 1, c0 : c1 + 1] = int(color)
    for sprite, coords in zip(sprites, final_coords):
        out[coords[:, 0], coords[:, 1]] = int(sprite["color"])
    return out


def _sample_band_sizes(count_range: tuple[int, int], size_range: tuple[int, int], cap: int) -> list[int]:
    for _ in range(200):
        count = random.randint(*count_range)
        bands = [random.randint(*size_range) for _ in range(count)]
        total = 2 * OUTER_MARGIN + sum(bands) + (count + 1)
        if total <= cap:
            return bands
    raise RuntimeError("failed to sample valid lattice dimensions")


def _build_lattice_positions(lengths: list[int]) -> tuple[list[int], list[tuple[int, int]], int]:
    separators = [OUTER_MARGIN]
    bands: list[tuple[int, int]] = []
    cur = OUTER_MARGIN
    for size in lengths:
        bands.append((cur + 1, cur + size))
        cur = cur + size + 1
        separators.append(cur)
    total = cur + OUTER_MARGIN + 1
    return separators, bands, total


def _noise_cell(h: int, w: int, palette: tuple[int, ...], seed: int) -> np.ndarray:
    cell = np.empty((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            idx = (2 * r + 3 * c + seed + (r ^ c)) % len(palette)
            cell[r, c] = palette[idx]
    if len(set(map(int, cell.flatten()))) == 1:
        cell[0, 0] = palette[0]
        cell[-1, -1] = palette[1]
    return cell


def _choose_block_shape() -> tuple[int, int]:
    return 2, random.choice([2, 3])


def _sample_active_positions(block_h: int, block_w: int, active_count: int, mode: str) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(block_h) for c in range(block_w)]
    for _ in range(200):
        active = random.sample(cells, active_count)
        rows = {r: sum(1 for rr, _ in active if rr == r) for r in range(block_h)}
        cols = {c: sum(1 for _, cc in active if cc == c) for c in range(block_w)}
        if mode.startswith("row") and max(rows.values()) < 2:
            continue
        if mode.startswith("col") and max(cols.values()) < 2:
            continue
        return active
    raise RuntimeError("failed to sample active positions")


def _assign_sides(active: list[tuple[int, int]], mode: str) -> tuple[dict[tuple[int, int], str], tuple[tuple[int, int], tuple[int, int]] | None]:
    side_by_pos: dict[tuple[int, int], str] = {}
    used_lr: set[tuple[str, int]] = set()
    used_tb: set[tuple[str, int]] = set()
    collision_pair = None

    if mode.startswith("row"):
        rows = sorted({r for r, _ in active if sum(1 for rr, _ in active if rr == r) >= 2})
        row = random.choice(rows)
        row_cells = sorted([pos for pos in active if pos[0] == row], key=lambda pos: pos[1])
        left_target = row_cells[0]
        right_target = row_cells[-1]
        side_by_pos[left_target] = "right"
        side_by_pos[right_target] = "left"
        used_lr.add(("right", row))
        used_lr.add(("left", row))
        collision_pair = (left_target, right_target)
    elif mode.startswith("col"):
        cols = sorted({c for _, c in active if sum(1 for _, cc in active if cc == c) >= 2})
        col = random.choice(cols)
        col_cells = sorted([pos for pos in active if pos[1] == col], key=lambda pos: pos[0])
        top_target = col_cells[0]
        bottom_target = col_cells[-1]
        side_by_pos[top_target] = "bottom"
        side_by_pos[bottom_target] = "top"
        used_tb.add(("bottom", col))
        used_tb.add(("top", col))
        collision_pair = (top_target, bottom_target)

    for pos in random.sample(active, len(active)):
        if pos in side_by_pos:
            continue
        r, c = pos
        if mode.startswith("row"):
            prefs = ["top", "bottom", "left", "right"]
        elif mode.startswith("col"):
            prefs = ["left", "right", "top", "bottom"]
        else:
            prefs = random.sample(["left", "right", "top", "bottom"], 4)
        for side in prefs:
            if side in {"left", "right"} and (side, r) in used_lr:
                continue
            if side in {"top", "bottom"} and (side, c) in used_tb:
                continue
            side_by_pos[pos] = side
            if side in {"left", "right"}:
                used_lr.add((side, r))
            else:
                used_tb.add((side, c))
            break
        else:
            raise RuntimeError("failed to assign a non-overlapping side")

    return side_by_pos, collision_pair


def _shape_dims(coords: np.ndarray) -> tuple[int, int]:
    r0, c0, r1, c1 = _bbox(coords)
    return r1 - r0 + 1, c1 - c0 + 1


def _place_sprite(
    grid_shape: tuple[int, int],
    row_band: tuple[int, int],
    col_band: tuple[int, int],
    side: str,
    shape: np.ndarray,
    occupied: set[tuple[int, int]],
) -> np.ndarray | None:
    h, w = map(int, grid_shape)
    shape_h, shape_w = _shape_dims(shape)
    row_options: list[int] = []
    col_options: list[int] = []

    if side in {"left", "right"}:
        r0_min, r0_max = row_band[0], row_band[1] - shape_h + 1
        if r0_min > r0_max:
            return None
        row_options = list(range(r0_min, r0_max + 1))
        if side == "left":
            c0_min, c0_max = 0, OUTER_MARGIN - shape_w
        else:
            c0_min, c0_max = w - OUTER_MARGIN, w - shape_w
        col_options = list(range(c0_min, c0_max + 1))
    else:
        c0_min, c0_max = col_band[0], col_band[1] - shape_w + 1
        if c0_min > c0_max:
            return None
        col_options = list(range(c0_min, c0_max + 1))
        if side == "top":
            r0_min, r0_max = 0, OUTER_MARGIN - shape_h
        else:
            r0_min, r0_max = h - OUTER_MARGIN, h - shape_h
        row_options = list(range(r0_min, r0_max + 1))

    random.shuffle(row_options)
    random.shuffle(col_options)
    for r0 in row_options:
        for c0 in col_options:
            placed = shape + np.array([r0, c0], dtype=int)
            cells = {tuple(map(int, pt)) for pt in placed}
            if cells & occupied:
                continue
            return placed
    return None


def _trace_mask(grid_shape: tuple[int, int], sprite: dict) -> np.ndarray:
    mask = np.zeros(grid_shape, dtype=bool)
    coords = sprite["coords"]
    delta = sprite["delta"]
    for step in range(int(sprite["distance"]) + 1):
        trace = coords + delta * step
        mask[trace[:, 0], trace[:, 1]] = True
    return mask


def _gen_attempt() -> np.ndarray | None:
    row_sizes = _sample_band_sizes((4, 5), (2, 4), 30)
    col_sizes = _sample_band_sizes((4, 5), (2, 4), 30)
    sep_rows, row_bands, h = _build_lattice_positions(row_sizes)
    sep_cols, col_bands, w = _build_lattice_positions(col_sizes)
    if h > 30 or w > 30:
        return None

    block_h, block_w = _choose_block_shape()
    if len(row_bands) < block_h or len(col_bands) < block_w:
        return None
    block_r0 = random.randint(0, len(row_bands) - block_h)
    block_c0 = random.randint(0, len(col_bands) - block_w)

    mode = random.choice(["none", "row_size", "row_tie", "col_size", "col_tie"])
    active_count = random.randint(2, min(4, block_h * block_w))
    active_local = _sample_active_positions(block_h, block_w, active_count, mode if mode != "none" else "")
    side_by_local, collision_pair = _assign_sides(active_local, mode if mode != "none" else "")

    block_positions = [(r, c) for r in range(block_h) for c in range(block_w)]
    active_colors = random.sample(list(ACTIVE_COLORS), active_count)
    dummy_colors = random.sample(list(DUMMY_COLORS), len(block_positions) - active_count)
    color_by_local: dict[tuple[int, int], int] = {}
    for pos, color in zip(active_local, active_colors):
        color_by_local[pos] = color
    for pos, color in zip([p for p in block_positions if p not in color_by_local], dummy_colors):
        color_by_local[pos] = color

    grid = np.full((h, w), BACKGROUND, dtype=int)
    grid[sep_rows, :] = SEP
    grid[:, sep_cols] = SEP

    noise_palette = tuple(random.sample(list(DUMMY_COLORS + ACTIVE_COLORS), random.randint(3, 5)))
    for ri, (r0, r1) in enumerate(row_bands):
        for ci, (c0, c1) in enumerate(col_bands):
            local = (ri - block_r0, ci - block_c0)
            if 0 <= local[0] < block_h and 0 <= local[1] < block_w:
                color = color_by_local[local]
                grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)
            else:
                cell = _noise_cell(r1 - r0 + 1, c1 - c0 + 1, noise_palette, ri * 11 + ci * 17 + random.randint(0, 6))
                if len(set(map(int, cell.flatten()))) < 2:
                    return None
                grid[r0 : r1 + 1, c0 : c1 + 1] = cell

    occupied: set[tuple[int, int]] = set()
    sprites: list[dict] = []
    sprite_by_local: dict[tuple[int, int], dict] = {}

    local_collision_high = None
    local_collision_low = None
    if collision_pair is not None:
        first, second = collision_pair
        if mode.endswith("size"):
            if random.choice([False, True]):
                local_collision_high, local_collision_low = first, second
            else:
                local_collision_high, local_collision_low = second, first

    for local in active_local:
        global_rc = (block_r0 + local[0], block_c0 + local[1])
        row_band = row_bands[global_rc[0]]
        col_band = col_bands[global_rc[1]]
        side = side_by_local[local]

        if local == local_collision_high:
            area = 4
        elif local == local_collision_low:
            area = 2
        elif collision_pair is not None and mode.endswith("tie") and local in collision_pair:
            area = 3
        else:
            area = random.randint(2, 4)

        shapes = random.sample(SHAPES_BY_AREA[area], len(SHAPES_BY_AREA[area]))
        placed = None
        for shape in shapes:
            placed = _place_sprite(grid.shape, row_band, col_band, side, shape, occupied)
            if placed is not None:
                break
        if placed is None:
            return None

        cells = {tuple(map(int, pt)) for pt in placed}
        occupied |= cells
        color = color_by_local[local]
        grid[placed[:, 0], placed[:, 1]] = int(color)
        sprite = {
            "color": int(color),
            "coords": placed,
            "side": side,
            "delta": _target_distance(
                side,
                placed,
                (
                    row_bands[global_rc[0]][0],
                    col_bands[global_rc[1]][0],
                    row_bands[global_rc[0]][1],
                    col_bands[global_rc[1]][1],
                ),
            )[0],
            "distance": _target_distance(
                side,
                placed,
                (
                    row_bands[global_rc[0]][0],
                    col_bands[global_rc[1]][0],
                    row_bands[global_rc[0]][1],
                    col_bands[global_rc[1]][1],
                ),
            )[1],
            "area": int(len(placed)),
        }
        if sprite["distance"] < 1:
            return None
        sprites.append(sprite)
        sprite_by_local[local] = sprite

    if collision_pair is not None:
        a = sprite_by_local[collision_pair[0]]
        b = sprite_by_local[collision_pair[1]]
        if not np.any(_trace_mask(grid.shape, a) & _trace_mask(grid.shape, b)):
            return None
        if mode.endswith("tie") and a["area"] == b["area"] and a["distance"] == b["distance"]:
            return None

    try:
        final_coords, shifts = _settle_sprites(grid.shape, sprites)
    except AssertionError:
        return None

    if collision_pair is not None:
        pair_indices = [sprites.index(sprite_by_local[pos]) for pos in collision_pair]
        if all(shifts[idx] == sprites[idx]["distance"] for idx in pair_indices):
            return None

    out = main(grid)
    if out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None
    if len(np.unique(grid)) < 4:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
