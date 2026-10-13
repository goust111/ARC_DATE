from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))
DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]
LAYOUTS = [(2, 2), (2, 3), (2, 4), (3, 2), (4, 2)]
PIECE_SIZE_RANGE = (5, 7)
GRID_H_RANGE = (30, 38)
GRID_W_RANGE = (30, 40)
MAX_RETRIES = 300


def _mode_color(grid: np.ndarray) -> int:
    vals = np.asarray(grid, dtype=int).ravel()
    return int(np.bincount(vals).argmax())


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _edge_positions(size: int):
    for c in range(size):
        yield 0, c
    for r in range(1, size - 1):
        yield r, size - 1
    for c in range(size - 1, -1, -1):
        yield size - 1, c
    for r in range(size - 2, 0, -1):
        yield r, 0


def _rotate_coord(coord: tuple[int, int], size: int, k: int) -> tuple[int, int]:
    r, c = coord
    for _ in range(k % 4):
        r, c = size - 1 - c, r
    return r, c


def _gap_side(coord: tuple[int, int], size: int) -> str:
    r, c = coord
    if r == 0:
        return "U"
    if c == size - 1:
        return "R"
    if r == size - 1:
        return "D"
    if c == 0:
        return "L"
    raise ValueError("gap must lie on the boundary")


def _corner_distance(corner: str, coord: tuple[int, int], size: int) -> int:
    r, c = coord
    side = _gap_side(coord, size)
    if corner == "TL":
        return c if side == "U" else r
    if corner == "TR":
        return (size - 1 - c) if side == "U" else r
    if corner == "BL":
        return c if side == "D" else (size - 1 - r)
    if corner == "BR":
        return (size - 1 - c) if side == "D" else (size - 1 - r)
    raise ValueError(corner)


def _piece_signature(patch: np.ndarray, bg: int):
    size = patch.shape[0]
    if patch.shape[0] != patch.shape[1] or size < 4:
        return None
    edge_vals = [int(patch[r, c]) for r, c in _edge_positions(size)]
    gap_cells = [(r, c) for r, c in _edge_positions(size) if int(patch[r, c]) == bg]
    if len(gap_cells) != 1:
        return None
    border_vals = [v for v in edge_vals if v != bg]
    if not border_vals or len(set(border_vals)) != 1:
        return None
    return {
        "size": size,
        "border_color": border_vals[0],
        "gap": gap_cells[0],
        "patch": patch.copy(),
    }


def _locate_graph(components: list[dict], pieces: list[dict], grid: np.ndarray, bg: int) -> dict:
    piece_colors = {int(piece["border_color"]) for piece in pieces}
    candidates = []
    for comp in components:
        if comp.get("kind") == "piece":
            continue
        r0, c0, r1, c1 = comp["bbox"]
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        vals = [int(v) for v in patch.ravel()]
        if any(v == bg for v in vals):
            continue
        if len(vals) != len(set(vals)):
            continue
        if set(vals) != piece_colors:
            continue
        if patch.size != len(pieces):
            continue
        candidates.append(
            {
                "bbox": comp["bbox"],
                "patch": patch.copy(),
                "anchor_score": (r0 + c0, r0, c0),
            }
        )
    if not candidates:
        raise ValueError("failed to locate topology graph")
    candidates.sort(key=lambda item: item["anchor_score"], reverse=True)
    graph = candidates[0]
    graph["rows"], graph["cols"] = graph["patch"].shape
    return graph


def _rotation_for_position(
    piece: dict,
    row: int,
    col: int,
    total_rows: int,
    total_cols: int,
) -> int:
    size = int(piece["size"])
    gap = piece["gap"]
    outsides = []
    if row == 0:
        outsides.append("U")
    if row == total_rows - 1:
        outsides.append("D")
    if col == 0:
        outsides.append("L")
    if col == total_cols - 1:
        outsides.append("R")
    if not outsides:
        raise ValueError("interior pieces cannot have outward gaps")

    scores = []
    for k in range(4):
        rotated_gap = _rotate_coord(gap, size, k)
        side = _gap_side(rotated_gap, size)
        if side not in outsides:
            continue
        if len(outsides) == 1:
            scores.append((0, k))
            continue
        if row == 0 and col == 0:
            corner = "TL"
        elif row == 0 and col == total_cols - 1:
            corner = "TR"
        elif row == total_rows - 1 and col == 0:
            corner = "BL"
        else:
            corner = "BR"
        scores.append((_corner_distance(corner, rotated_gap, size), k))
    if not scores:
        raise ValueError("no valid rotation for piece")
    scores.sort()
    if len(scores) > 1 and scores[0][0] == scores[1][0]:
        raise ValueError("ambiguous corner rotation")
    return int(scores[0][1])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble square fragments according to the colored topology graph, rotating each
    fragment so its one-cell border gap faces the outside of the final rectangle.
    """
    g = np.asarray(grid, dtype=int)
    bg = _mode_color(g)

    components = []
    for coords in _components(g != bg):
        r0, c0, r1, c1 = _bbox(coords)
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        info = {"bbox": (r0, c0, r1, c1), "coords": coords}
        sig = _piece_signature(patch, bg)
        if sig is not None:
            info.update(sig)
            info["kind"] = "piece"
        components.append(info)

    pieces = [comp for comp in components if comp.get("kind") == "piece"]
    if not pieces:
        raise ValueError("no fragments found")
    piece_sizes = {int(piece["size"]) for piece in pieces}
    if len(piece_sizes) != 1:
        raise ValueError("all fragments must share the same square size")
    graph = _locate_graph(components, pieces, g, bg)

    color_to_piece = {}
    for piece in pieces:
        color = int(piece["border_color"])
        if color in color_to_piece:
            raise ValueError("border colors must be unique")
        color_to_piece[color] = piece

    graph_patch = graph["patch"]
    rows, cols = graph_patch.shape
    size = int(next(iter(piece_sizes)))
    out = np.full((rows * size, cols * size), bg, dtype=int)

    for gr in range(rows):
        for gc in range(cols):
            color = int(graph_patch[gr, gc])
            piece = color_to_piece[color]
            k = _rotation_for_position(piece, gr, gc, rows, cols)
            tile = np.rot90(piece["patch"], k)
            out[gr * size : (gr + 1) * size, gc * size : (gc + 1) * size] = tile

    return out


def _offset_buckets(size: int) -> tuple[list[int], list[int], list[int]]:
    all_offsets = [i for i in range(1, size - 1)]
    small = [i for i in all_offsets if 2 * i < size - 1]
    large = [i for i in all_offsets if 2 * i > size - 1]
    return all_offsets, small, large


def _make_piece(
    size: int,
    border_color: int,
    fill_color: int,
    side: str,
    offset: int,
) -> np.ndarray:
    patch = np.full((size, size), fill_color, dtype=int)
    patch[0, :] = border_color
    patch[-1, :] = border_color
    patch[:, 0] = border_color
    patch[:, -1] = border_color
    if side == "U":
        patch[0, offset] = BG
    elif side == "R":
        patch[offset, -1] = BG
    elif side == "D":
        patch[-1, offset] = BG
    elif side == "L":
        patch[offset, 0] = BG
    else:
        raise ValueError(side)
    return patch


def _target_side_and_offset(rng: random.Random, size: int, row: int, col: int, rows: int, cols: int):
    all_offsets, small, large = _offset_buckets(size)
    if row == 0 and row != rows - 1 and 0 < col < cols - 1:
        return "U", rng.choice(all_offsets)
    if row == rows - 1 and row != 0 and 0 < col < cols - 1:
        return "D", rng.choice(all_offsets)
    if col == 0 and col != cols - 1 and 0 < row < rows - 1:
        return "L", rng.choice(all_offsets)
    if col == cols - 1 and col != 0 and 0 < row < rows - 1:
        return "R", rng.choice(all_offsets)

    if row == 0 and col == 0:
        side = rng.choice(["U", "L"])
        return side, rng.choice(small)
    if row == 0 and col == cols - 1:
        side = rng.choice(["U", "R"])
        return ("U", rng.choice(large)) if side == "U" else ("R", rng.choice(small))
    if row == rows - 1 and col == 0:
        side = rng.choice(["D", "L"])
        return ("D", rng.choice(small)) if side == "D" else ("L", rng.choice(large))
    if row == rows - 1 and col == cols - 1:
        side = rng.choice(["D", "R"])
        return ("D", rng.choice(large)) if side == "D" else ("R", rng.choice(large))
    raise ValueError("unexpected piece position")


def _can_place(occ: np.ndarray, r0: int, c0: int, size: int) -> bool:
    h, w = occ.shape
    if r0 < 1 or c0 < 1 or r0 + size >= h - 1 or c0 + size >= w - 1:
        return False
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(h, r0 + size + 1)
    cc1 = min(w, c0 + size + 1)
    return not occ[rr0:rr1, cc0:cc1].any()


def _mark_occ(occ: np.ndarray, r0: int, c0: int, patch: np.ndarray):
    size = patch.shape[0]
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(occ.shape[0], r0 + size + 1)
    cc1 = min(occ.shape[1], c0 + size + 1)
    occ[rr0:rr1, cc0:cc1] = True


def _assemble_expected(patches: list[list[np.ndarray]], rows: int, cols: int, size: int) -> np.ndarray:
    out = np.full((rows * size, cols * size), BG, dtype=int)
    for r in range(rows):
        for c in range(cols):
            out[r * size : (r + 1) * size, c * size : (c + 1) * size] = patches[r][c]
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    rows, cols = rng.choice(LAYOUTS)
    size = rng.randint(PIECE_SIZE_RANGE[0], PIECE_SIZE_RANGE[1])
    count = rows * cols
    border_colors = rng.sample(COLORS, count)
    fill_pool = [c for c in COLORS if c not in border_colors]
    h = rng.randint(GRID_H_RANGE[0], GRID_H_RANGE[1])
    w = rng.randint(GRID_W_RANGE[0], GRID_W_RANGE[1])

    if h <= rows + 3 or w <= cols + 3:
        return None

    grid = np.full((h, w), BG, dtype=int)
    occ = np.zeros((h, w), dtype=bool)

    graph_r0 = h - rows - 1
    graph_c0 = w - cols - 1
    if graph_r0 < 2 or graph_c0 < 2:
        return None
    occ[max(0, graph_r0 - 2) : h, max(0, graph_c0 - 2) : w] = True

    color_grid = np.array(border_colors, dtype=int).reshape(rows, cols)
    canonical: list[list[np.ndarray]] = []
    placed_inputs: list[np.ndarray] = []

    for r in range(rows):
        row_tiles = []
        for c in range(cols):
            border_color = int(color_grid[r, c])
            fill_color = int(rng.choice(fill_pool))
            side, offset = _target_side_and_offset(rng, size, r, c, rows, cols)
            patch = _make_piece(size, border_color, fill_color, side, offset)
            row_tiles.append(patch)
            placed_inputs.append(np.rot90(patch, rng.randrange(4)))
        canonical.append(row_tiles)

    rng.shuffle(placed_inputs)
    for patch in placed_inputs:
        placed = False
        for _ in range(250):
            r0 = rng.randint(1, h - size - 2)
            c0 = rng.randint(1, w - size - 2)
            if not _can_place(occ, r0, c0, size):
                continue
            grid[r0 : r0 + size, c0 : c0 + size] = patch
            _mark_occ(occ, r0, c0, patch)
            placed = True
            break
        if not placed:
            return None

    grid[graph_r0 : graph_r0 + rows, graph_c0 : graph_c0 + cols] = color_grid

    if np.count_nonzero(grid) * 2 >= grid.size:
        return None

    expected = _assemble_expected(canonical, rows, cols, size)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_RETRIES} attempts")
