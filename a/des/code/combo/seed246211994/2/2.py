from __future__ import annotations

from collections import Counter
import random

import numpy as np


DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ALL_COLORS = tuple(range(10))

# Generator configuration.
GEN_RETRIES = 400
BODY_H_RANGE = (10, 16)
BODY_W_RANGE = (12, 20)
PATH_COUNT_RANGE = (2, 4)
PATH_LENGTH_MIN = 4
PATH_LENGTH_MAX = 12
PATH_BUILD_RETRIES = 120
PLACEMENT_RETRIES = 80
PALETTE_BLOCK_WIDTHS = (1, 2, 3)
PALETTE_GAP_RANGE = (0, 2)


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _color_runs(row: np.ndarray, bg: int) -> list[tuple[int, int, int]]:
    runs: list[tuple[int, int, int]] = []
    width = int(row.shape[0])
    c = 0
    while c < width:
        color = int(row[c])
        if color == bg:
            c += 1
            continue
        start = c
        c += 1
        while c < width and int(row[c]) == color:
            c += 1
        runs.append((start, c - 1, color))
    return runs


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIRS:
                    nr, nc = int(rr + dr), int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(coords)
    return components


def _is_simple_path(coords: list[tuple[int, int]]) -> bool:
    if len(coords) < 2:
        return False
    cells = set(coords)
    degree_counts: list[int] = []
    for r, c in coords:
        degree = 0
        for dr, dc in DIRS:
            if (r + dr, c + dc) in cells:
                degree += 1
        if degree not in (1, 2):
            return False
        degree_counts.append(degree)
    return degree_counts.count(1) == 2 and degree_counts.count(2) == len(coords) - 2


def _candidate_from_side(grid: np.ndarray, bg: int, side: str) -> dict[str, object] | None:
    row = grid[0] if side == "top" else grid[-1]
    runs = _color_runs(row, bg)
    if len(runs) < 2:
        return None

    body = grid[1:, :] if side == "top" else grid[:-1, :]
    mask = body != bg
    if not bool(mask.any()):
        return None

    components = _connected_components(mask)
    if len(components) != len(runs):
        return None
    if not all(_is_simple_path(component) for component in components):
        return None

    return {
        "side": side,
        "body": body,
        "components": components,
        "palette_colors": [color for _start, _end, color in runs],
        "non_bg": int(np.count_nonzero(row != bg)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort snake-like paths by pixel length, then recolor them using the palette row order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 3 and g.shape[1] >= 4

    bg = _background_color(g)
    candidates = []
    for side in ("top", "bottom"):
        candidate = _candidate_from_side(g, bg, side)
        if candidate is not None:
            candidates.append(candidate)
    assert candidates, "failed to identify the palette row"

    if len(candidates) > 1:
        candidates.sort(key=lambda item: (int(item["non_bg"]), len(item["palette_colors"])), reverse=True)
    chosen = candidates[0]

    body = np.array(chosen["body"], dtype=int, copy=True)
    components = list(chosen["components"])
    palette_colors = list(chosen["palette_colors"])
    lengths = [len(component) for component in components]

    assert len(components) == len(palette_colors)
    assert len(set(lengths)) == len(lengths), "path lengths must be unique"

    order = sorted(range(len(components)), key=lambda idx: lengths[idx])
    for rank, comp_idx in enumerate(order):
        color = int(palette_colors[rank])
        for r, c in components[comp_idx]:
            body[r, c] = color
    return body


def _count_turns(path: list[tuple[int, int]]) -> int:
    turns = 0
    directions = []
    for (r0, c0), (r1, c1) in zip(path, path[1:]):
        directions.append((r1 - r0, c1 - c0))
    for d0, d1 in zip(directions, directions[1:]):
        if d0 != d1:
            turns += 1
    return turns


def _choose_local_dims(length: int) -> tuple[int, int]:
    dims: list[tuple[int, int]] = []
    max_side = min(7, length + 1)
    for h in range(3, max_side + 1):
        for w in range(3, max_side + 1):
            area = h * w
            if area < length + 2:
                continue
            if area > length + 20:
                continue
            dims.append((h, w))
    assert dims
    return random.choice(dims)


def _can_extend_to(
    nr: int,
    nc: int,
    prev: tuple[int, int],
    occupied: set[tuple[int, int]],
    h: int,
    w: int,
) -> bool:
    if not (0 <= nr < h and 0 <= nc < w):
        return False
    if (nr, nc) in occupied:
        return False

    neighbors = []
    for dr, dc in DIRS:
        cell = (nr + dr, nc + dc)
        if cell in occupied:
            neighbors.append(cell)
    return neighbors == [prev] or (len(neighbors) == 1 and neighbors[0] == prev)


def _count_future_moves(
    cell: tuple[int, int],
    occupied: set[tuple[int, int]],
    h: int,
    w: int,
) -> int:
    nr, nc = cell
    extended = set(occupied)
    extended.add(cell)
    count = 0
    for dr, dc in DIRS:
        rr, cc = nr + dr, nc + dc
        if _can_extend_to(rr, cc, cell, extended, h, w):
            count += 1
    return count


def _dfs_build_path(
    path: list[tuple[int, int]],
    occupied: set[tuple[int, int]],
    target_len: int,
    h: int,
    w: int,
    prefer_turns: bool,
    last_dir: tuple[int, int] | None,
) -> bool:
    if len(path) == target_len:
        return True

    tail = path[-1]
    moves: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
    for dr, dc in DIRS:
        nr, nc = tail[0] + dr, tail[1] + dc
        if not _can_extend_to(nr, nc, tail, occupied, h, w):
            continue
        if len(path) + 1 < target_len and _count_future_moves((nr, nc), occupied, h, w) == 0:
            continue
        is_turn = 1 if last_dir is not None and (dr, dc) != last_dir else 0
        moves.append((is_turn, (nr, nc), (dr, dc)))

    random.shuffle(moves)
    if prefer_turns:
        moves.sort(key=lambda item: item[0], reverse=True)
    else:
        moves.sort(key=lambda item: item[0])

    for _turn_flag, cell, direction in moves:
        path.append(cell)
        occupied.add(cell)
        if _dfs_build_path(path, occupied, target_len, h, w, prefer_turns, direction):
            return True
        occupied.remove(cell)
        path.pop()
    return False


def _build_path_coords(length: int) -> list[tuple[int, int]] | None:
    min_turns = 0 if length <= 4 else 1 + int(length >= 8 and random.random() < 0.5)
    for _ in range(PATH_BUILD_RETRIES):
        h, w = _choose_local_dims(length)
        start = (random.randrange(h), random.randrange(w))
        path = [start]
        occupied = {start}
        prefer_turns = random.random() < 0.7
        if not _dfs_build_path(path, occupied, length, h, w, prefer_turns, None):
            continue
        if _count_turns(path) < min_turns:
            continue
        return path
    return None


def _path_to_patch(path: list[tuple[int, int]], color: int, bg: int) -> np.ndarray:
    rows = [r for r, _c in path]
    cols = [c for _r, c in path]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for r, c in path:
        patch[r - r0, c - c0] = color
    return patch


def _can_place_patch(
    occupied: np.ndarray,
    patch_cells: list[tuple[int, int]],
    top: int,
    left: int,
) -> bool:
    h, w = occupied.shape
    for pr, pc in patch_cells:
        rr, cc = top + pr, left + pc
        if bool(occupied[rr, cc]):
            return False
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and bool(occupied[nr, nc]):
                    return False
    return True


def _paste_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> None:
    mask = patch != bg
    block = grid[top : top + patch.shape[0], left : left + patch.shape[1]]
    block[mask] = patch[mask]


def _place_patches(
    body_h: int,
    body_w: int,
    infos: list[dict[str, object]],
    bg: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(PLACEMENT_RETRIES):
        body_in = np.full((body_h, body_w), bg, dtype=int)
        body_out = np.full((body_h, body_w), bg, dtype=int)
        occupied = np.zeros((body_h, body_w), dtype=bool)
        success = True

        for info in sorted(infos, key=lambda item: int(np.count_nonzero(item["input_patch"] != bg)), reverse=True):
            input_patch = np.asarray(info["input_patch"], dtype=int)
            output_patch = np.asarray(info["output_patch"], dtype=int)
            ph, pw = input_patch.shape
            if ph + 2 > body_h or pw + 2 > body_w:
                success = False
                break

            patch_cells = [tuple(map(int, cell)) for cell in np.argwhere(input_patch != bg)]
            candidates = []
            for top in range(1, body_h - ph):
                for left in range(1, body_w - pw):
                    if _can_place_patch(occupied, patch_cells, top, left):
                        candidates.append((top, left))
            if not candidates:
                success = False
                break

            top, left = random.choice(candidates)
            _paste_patch(body_in, input_patch, top, left, bg)
            _paste_patch(body_out, output_patch, top, left, bg)
            for pr, pc in patch_cells:
                occupied[top + pr, left + pc] = True

        if success:
            return body_in, body_out
    return None


def _build_palette_row(width: int, colors: list[int], bg: int) -> np.ndarray | None:
    row = np.full(width, bg, dtype=int)
    widths = [random.choice(PALETTE_BLOCK_WIDTHS) for _ in colors]
    gaps = [random.randint(*PALETTE_GAP_RANGE) for _ in range(max(0, len(colors) - 1))]
    span = sum(widths) + sum(gaps)
    if span > width:
        return None
    start = random.randint(0, width - span)
    cursor = start
    for idx, color in enumerate(colors):
        block_w = widths[idx]
        row[cursor : cursor + block_w] = int(color)
        cursor += block_w
        if idx < len(gaps):
            cursor += gaps[idx]
    return row


def _sample_lengths(body_h: int, body_w: int, n_paths: int) -> list[int] | None:
    max_length = min(PATH_LENGTH_MAX, (body_h * body_w) // (n_paths * 3))
    if max_length < PATH_LENGTH_MIN + n_paths - 1:
        return None
    pool = list(range(PATH_LENGTH_MIN, max_length + 1))
    for _ in range(40):
        lengths = sorted(random.sample(pool, n_paths))
        if sum(lengths) <= (body_h * body_w) // 4 + body_h:
            return lengths
    return None


def _generate_case() -> tuple[np.ndarray, np.ndarray] | None:
    body_h = random.randint(*BODY_H_RANGE)
    body_w = random.randint(*BODY_W_RANGE)
    n_paths = random.randint(*PATH_COUNT_RANGE)
    lengths = _sample_lengths(body_h, body_w, n_paths)
    if lengths is None:
        return None

    bg = random.choice(ALL_COLORS)
    usable_colors = [color for color in ALL_COLORS if color != bg]
    sampled = random.sample(usable_colors, 2 * n_paths)
    path_colors = sampled[:n_paths]
    palette_colors = sampled[n_paths:]

    infos: list[dict[str, object]] = []
    for length, path_color, target_color in zip(lengths, path_colors, palette_colors):
        path = _build_path_coords(length)
        if path is None:
            return None
        infos.append(
            {
                "length": length,
                "input_patch": _path_to_patch(path, path_color, bg),
                "output_patch": _path_to_patch(path, target_color, bg),
            }
        )

    random.shuffle(infos)
    placed = _place_patches(body_h, body_w, infos, bg)
    if placed is None:
        return None
    body_in, body_out = placed

    palette_row = _build_palette_row(body_w, palette_colors, bg)
    if palette_row is None:
        return None

    if random.random() < 0.5:
        inp = np.vstack([palette_row[np.newaxis, :], body_in])
    else:
        inp = np.vstack([body_in, palette_row[np.newaxis, :]])

    return inp, body_out


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        generated = _generate_case()
        if generated is None:
            continue
        inp, expected = generated
        try:
            if np.array_equal(main(inp), expected):
                return inp
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid input")
