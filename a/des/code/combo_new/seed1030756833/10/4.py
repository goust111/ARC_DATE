from __future__ import annotations

import random

import numpy as np


RED = 2
BLUE = 1
GREEN = 3
GEN_ATTEMPTS = 600


def _background(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _marker_to_rotation(marker: int, bg: int) -> int | None:
    if int(marker) == int(bg):
        return 0
    if int(marker) == RED:
        return 1
    if int(marker) == BLUE:
        return 2
    if int(marker) == GREEN:
        return 3
    return None


def _rotation_to_marker(rotation: int, bg: int) -> int:
    return {0: int(bg), 1: RED, 2: BLUE, 3: GREEN}[int(rotation) % 4]


def _rot_cw(block: np.ndarray, rotation: int) -> np.ndarray:
    return np.rot90(block, -int(rotation) % 4).copy()


def _rotated_block(raw: np.ndarray, rotation: int, bg: int) -> np.ndarray:
    block = np.array(raw, dtype=int, copy=True)
    block[0, 0] = _rotation_to_marker(rotation, bg)
    return _rot_cw(block, rotation)


def _edge_arrays(a: np.ndarray, b: np.ndarray, dr: int, dc: int) -> tuple[np.ndarray, np.ndarray]:
    if dr == -1:
        return a[0, :], b[-1, :]
    if dr == 1:
        return a[-1, :], b[0, :]
    if dc == -1:
        return a[:, 0], b[:, -1]
    return a[:, -1], b[:, 0]


def _neighbors(r: int, c: int, rows: int, cols: int):
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < rows and 0 <= cc < cols:
            yield rr, cc, dr, dc


def _block_score(block: np.ndarray, placed: list[list[np.ndarray]], r: int, c: int) -> tuple[int, int]:
    rows, cols = len(placed), len(placed[0])
    pixels = 0
    full_edges = 0
    for rr, cc, dr, dc in _neighbors(r, c, rows, cols):
        edge, other = _edge_arrays(block, placed[rr][cc], dr, dc)
        eq = edge == other
        pixels += int(np.count_nonzero(eq))
        full_edges += int(bool(np.all(eq)))
    return pixels, full_edges


def _all_neighbor_edges_bad(placed: list[list[np.ndarray]], r: int, c: int) -> bool:
    rows, cols = len(placed), len(placed[0])
    bad = []
    for rr, cc, dr, dc in _neighbors(r, c, rows, cols):
        edge, other = _edge_arrays(placed[r][c], placed[rr][cc], dr, dc)
        bad.append(not bool(np.array_equal(edge, other)))
    return bool(bad) and all(bad)


def _correct_blocks(
    raw_blocks: list[list[np.ndarray]],
    rotations: list[list[int]],
    bg: int,
) -> list[list[np.ndarray]]:
    rows, cols = len(raw_blocks), len(raw_blocks[0])
    placed = [
        [_rotated_block(raw_blocks[r][c], rotations[r][c], bg) for c in range(cols)]
        for r in range(rows)
    ]

    for _ in range(rows * cols):
        changed = False
        for r in range(rows):
            for c in range(cols):
                if not _all_neighbor_edges_bad(placed, r, c):
                    continue

                current_pixels, current_full = _block_score(placed[r][c], placed, r, c)
                best = (current_pixels, current_full, rotations[r][c], placed[r][c])
                for rotation in range(4):
                    candidate = _rotated_block(raw_blocks[r][c], rotation, bg)
                    pixels, full_edges = _block_score(candidate, placed, r, c)
                    if (pixels, full_edges) > (best[0], best[1]):
                        best = (pixels, full_edges, rotation, candidate)

                if int(best[2]) != int(rotations[r][c]) and best[0] > current_pixels:
                    rotations[r][c] = int(best[2])
                    placed[r][c] = best[3]
                    changed = True
        if not changed:
            break

    return placed


def _total_boundary_score(placed: list[list[np.ndarray]]) -> tuple[int, int, int]:
    rows, cols = len(placed), len(placed[0])
    full_edges = 0
    pixels = 0
    edge_count = 0

    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                edge_count += 1
                eq = placed[r][c][:, -1] == placed[r][c + 1][:, 0]
                pixels += int(np.count_nonzero(eq))
                full_edges += int(bool(np.all(eq)))
            if r + 1 < rows:
                edge_count += 1
                eq = placed[r][c][-1, :] == placed[r + 1][c][0, :]
                pixels += int(np.count_nonzero(eq))
                full_edges += int(bool(np.all(eq)))

    return full_edges, pixels, edge_count


def _assemble(placed: list[list[np.ndarray]], block_size: int) -> np.ndarray:
    rows, cols = len(placed), len(placed[0])
    out = np.zeros((rows * block_size, cols * block_size), dtype=int)
    for r in range(rows):
        for c in range(cols):
            out[
                r * block_size : (r + 1) * block_size,
                c * block_size : (c + 1) * block_size,
            ] = placed[r][c]
    return out


def _solve_with_block_size(grid: np.ndarray, block_size: int, bg: int):
    h, w = grid.shape
    if h % block_size or w % block_size:
        return None
    rows, cols = h // block_size, w // block_size
    if rows < 2 or cols < 2:
        return None

    raw_blocks: list[list[np.ndarray]] = []
    rotations: list[list[int]] = []
    for r in range(rows):
        raw_row = []
        rot_row = []
        for c in range(cols):
            block = grid[
                r * block_size : (r + 1) * block_size,
                c * block_size : (c + 1) * block_size,
            ].copy()
            rotation = _marker_to_rotation(int(block[0, 0]), bg)
            if rotation is None:
                return None
            raw_row.append(block)
            rot_row.append(rotation)
        raw_blocks.append(raw_row)
        rotations.append(rot_row)

    placed = _correct_blocks(raw_blocks, rotations, bg)
    full_edges, pixels, edge_count = _total_boundary_score(placed)
    if edge_count == 0:
        return None

    pixel_total = edge_count * block_size
    score = (
        int(full_edges == edge_count),
        full_edges / edge_count,
        pixels / pixel_total,
        -block_size,
    )
    return score, _assemble(placed, block_size)


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate marked square blocks and repair the marker whose borders do not fit.

    Input structure:
    - The grid is a tight rectangular array of equal-size square blocks with no separator lines.
    - Each block contains a multicolor asymmetric pattern in its unrotated form.
    - The top-left cell of each block is a rotation marker: red means clockwise 90 degrees, blue means 180 degrees, green means clockwise 270 degrees, and the background color means no rotation.
    - A small number of markers are deliberately changed to another valid marker color.

    Rule:
    - Infer the square block size from the marker lattice and the boundary agreement produced by candidate partitions.
    - Rotate every block, including its marker cell, by the amount encoded by its top-left marker.
    - Any block whose rotated edges all fail to match its neighboring blocks is treated as having a wrong marker.
    - For such a block, try all four marker colors and keep the rotation whose shared-boundary pixels match the neighbors best.

    Key insight:
    - The corner color gives the first rotation guess, but the real answer is the block puzzle whose adjacent border pixels become identical.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = _background(g)

    best = None
    for block_size in range(3, min(h, w) + 1):
        solved = _solve_with_block_size(g, block_size, bg)
        if solved is None:
            continue
        score, out = solved
        if best is None or score > best[0]:
            best = (score, out)

    assert best is not None, "expected a valid square-block partition"
    return best[1]


# --------------- generator ---------------

GEN_BG = 0
GEN_PAYLOAD = (0, 4, 5, 6, 7, 8, 9)
GEN_BLOCK_SIZES = (5, 6, 7)


def _coord_key(x: int, limit: int, block_size: int) -> tuple[str, int]:
    if x > 0 and x % block_size == 0:
        return ("line", x)
    if x < limit - 1 and (x + 1) % block_size == 0:
        return ("line", x + 1)
    return ("cell", x)


def _group_key(r: int, c: int, height: int, width: int, block_size: int):
    return (_coord_key(r, height, block_size), _coord_key(c, width, block_size))


def _group_cells(row_key, col_key) -> list[tuple[int, int]]:
    r_kind, r_val = row_key
    c_kind, c_val = col_key
    rows = [r_val - 1, r_val] if r_kind == "line" else [r_val]
    cols = [c_val - 1, c_val] if c_kind == "line" else [c_val]
    return [(r, c) for r in rows for c in cols]


def _marker_corner(br: int, bc: int, block_size: int, rotation: int) -> tuple[int, int]:
    r0, c0 = br * block_size, bc * block_size
    if rotation == 0:
        return r0, c0
    if rotation == 1:
        return r0, c0 + block_size - 1
    if rotation == 2:
        return r0 + block_size - 1, c0 + block_size - 1
    return r0 + block_size - 1, c0


def _sample_rotations(rows: int, cols: int, block_size: int, height: int, width: int):
    count = rows * cols
    for _ in range(300):
        values = [0, 1, 2, 3] + [random.randrange(4) for _ in range(count - 4)]
        random.shuffle(values)
        rotations = [values[r * cols : (r + 1) * cols] for r in range(rows)]
        constraints: dict[tuple, int] = {}
        ok = True
        for br in range(rows):
            for bc in range(cols):
                rotation = rotations[br][bc]
                rr, cc = _marker_corner(br, bc, block_size, rotation)
                key = _group_key(rr, cc, height, width, block_size)
                color = _rotation_to_marker(rotation, GEN_BG)
                if key in constraints and constraints[key] != color:
                    ok = False
                    break
                constraints[key] = color
            if not ok:
                break
        if ok and {0, 1, 2, 3}.issubset(set(values)):
            return rotations, constraints
    return None


def _random_final_grid(height: int, width: int) -> np.ndarray:
    weights = [0.28, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12]
    flat = random.choices(GEN_PAYLOAD, weights=weights, k=height * width)
    return np.array(flat, dtype=int).reshape(height, width)


def _enforce_boundaries(grid: np.ndarray, block_size: int) -> None:
    height, width = grid.shape
    for c in range(block_size, width, block_size):
        grid[:, c] = grid[:, c - 1]
    for r in range(block_size, height, block_size):
        grid[r, :] = grid[r - 1, :]


def _with_marker_constraints(
    final: np.ndarray,
    rotations: list[list[int]],
    constraints: dict[tuple, int],
    block_size: int,
) -> None:
    height, width = final.shape
    for key, color in constraints.items():
        for r, c in _group_cells(*key):
            final[r, c] = int(color)

    _enforce_boundaries(final, block_size)

    for br, row in enumerate(rotations):
        for bc, rotation in enumerate(row):
            rr, cc = _marker_corner(br, bc, block_size, rotation)
            assert final[rr, cc] == _rotation_to_marker(rotation, GEN_BG)
            key = _group_key(rr, cc, height, width, block_size)
            assert constraints[key] == _rotation_to_marker(rotation, GEN_BG)


def _non_symmetric(block: np.ndarray) -> bool:
    variants = [_rot_cw(block, rotation) for rotation in range(4)]
    keys = {tuple(map(tuple, variant.tolist())) for variant in variants}
    return len(keys) == 4


def _build_input_from_final(final: np.ndarray, rotations: list[list[int]], block_size: int) -> np.ndarray | None:
    rows, cols = len(rotations), len(rotations[0])
    inp = np.zeros_like(final)
    for br in range(rows):
        for bc in range(cols):
            r0, c0 = br * block_size, bc * block_size
            tile = final[r0 : r0 + block_size, c0 : c0 + block_size]
            raw = np.rot90(tile, int(rotations[br][bc])).copy()
            if int(raw[0, 0]) != _rotation_to_marker(rotations[br][bc], GEN_BG):
                return None
            if not _non_symmetric(raw):
                return None
            inp[r0 : r0 + block_size, c0 : c0 + block_size] = raw
    return inp


def _initial_wrong_is_visible(
    inp: np.ndarray,
    rotations: list[list[int]],
    wrong_cells: list[tuple[int, int]],
    block_size: int,
) -> bool:
    rows, cols = len(rotations), len(rotations[0])
    raw_blocks = [
        [
            inp[
                r * block_size : (r + 1) * block_size,
                c * block_size : (c + 1) * block_size,
            ].copy()
            for c in range(cols)
        ]
        for r in range(rows)
    ]
    marker_rots = [
        [_marker_to_rotation(int(raw_blocks[r][c][0, 0]), GEN_BG) for c in range(cols)]
        for r in range(rows)
    ]
    if any(marker_rots[r][c] is None for r in range(rows) for c in range(cols)):
        return False
    placed = [
        [_rotated_block(raw_blocks[r][c], int(marker_rots[r][c]), GEN_BG) for c in range(cols)]
        for r in range(rows)
    ]
    return all(_all_neighbor_edges_bad(placed, r, c) for r, c in wrong_cells)


def _choose_wrong_cells(rows: int, cols: int) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(rows) for c in range(cols)]
    random.shuffle(cells)
    if rows * cols >= 9 and random.random() < 0.35:
        first = cells[0]
        options = [
            cell
            for cell in cells[1:]
            if abs(cell[0] - first[0]) + abs(cell[1] - first[1]) > 1
        ]
        if options:
            return [first, random.choice(options)]
    return [cells[0]]


def _gen_attempt() -> np.ndarray | None:
    block_size = random.choice(GEN_BLOCK_SIZES)
    possible_counts = [n for n in range(2, 6) if n * block_size <= 30]
    rows = random.choice(possible_counts)
    cols = random.choice(possible_counts)
    if rows * cols < 6:
        return None

    height, width = rows * block_size, cols * block_size
    sampled = _sample_rotations(rows, cols, block_size, height, width)
    if sampled is None:
        return None
    rotations, constraints = sampled

    final = _random_final_grid(height, width)
    _with_marker_constraints(final, rotations, constraints, block_size)
    inp = _build_input_from_final(final, rotations, block_size)
    if inp is None:
        return None

    wrong_cells = _choose_wrong_cells(rows, cols)
    for r, c in wrong_cells:
        correct = int(rotations[r][c])
        choices = [rotation for rotation in range(4) if rotation != correct]
        random.shuffle(choices)
        changed = False
        for wrong in choices:
            trial = inp.copy()
            trial[r * block_size, c * block_size] = _rotation_to_marker(wrong, GEN_BG)
            if _initial_wrong_is_visible(trial, rotations, wrong_cells, block_size):
                inp = trial
                changed = True
                break
        if not changed:
            return None

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, final):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for task 4")
