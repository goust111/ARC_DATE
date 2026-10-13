from __future__ import annotations

import random
from collections import Counter

import numpy as np


START_COLOR = 9
CCW_COLOR = 1
CW_COLOR = 2
HFLIP_COLOR = 3
VFLIP_COLOR = 4
OP_COLORS = (CCW_COLOR, CW_COLOR, HFLIP_COLOR, VFLIP_COLOR)
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
ALL_COLORS = tuple(range(10))


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in DIRS:
                    nr = rr + dr
                    nc = cc + dc
                    if _in_bounds(mask.shape, nr, nc) and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(comp)
    return out


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _apply_transform(
    sprite: np.ndarray, anchor: tuple[int, int], marker_color: int
) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = sprite.shape
    ar, ac = anchor
    if marker_color == CW_COLOR:
        return np.rot90(sprite, -1), (int(ac), int(h - 1 - ar))
    if marker_color == CCW_COLOR:
        return np.rot90(sprite, 1), (int(w - 1 - ac), int(ar))
    if marker_color == HFLIP_COLOR:
        return np.fliplr(sprite), (int(ar), int(w - 1 - ac))
    if marker_color == VFLIP_COLOR:
        return np.flipud(sprite), (int(h - 1 - ar), int(ac))
    raise AssertionError(f"unknown marker color: {marker_color}")


def _place_sprite(
    canvas: np.ndarray,
    sprite: np.ndarray,
    anchor: tuple[int, int],
    target: tuple[int, int],
    bg: int,
) -> None:
    ar, ac = anchor
    tr, tc = target
    h, w = sprite.shape
    top = int(tr - ar)
    left = int(tc - ac)
    assert top >= 0 and left >= 0 and top + h <= canvas.shape[0] and left + w <= canvas.shape[1]
    patch = sprite != bg
    sub = canvas[top : top + h, left : left + w]
    sub[patch] = sprite[patch]


def _extract_structure(
    grid: np.ndarray,
) -> tuple[int, np.ndarray, tuple[int, int], tuple[int, int], dict[tuple[int, int], int]]:
    bg = _most_common_color(grid)
    components = _components(grid != bg)
    sprite_components = [comp for comp in components if len(comp) > 1]
    assert len(sprite_components) == 1, "expected exactly one non-singleton sprite"
    sprite_cells = sprite_components[0]
    sprite_set = set(sprite_cells)

    start_positions = np.argwhere(grid == START_COLOR)
    assert len(start_positions) == 1, "expected exactly one start pixel"
    start_pos = tuple(map(int, start_positions[0]))
    assert start_pos in sprite_set, "start pixel must belong to the sprite"

    r0, c0, r1, c1 = _bbox(sprite_cells)
    sprite = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
    anchor = (int(start_pos[0] - r0), int(start_pos[1] - c0))
    corners = {
        (0, 0),
        (0, sprite.shape[1] - 1),
        (sprite.shape[0] - 1, 0),
        (sprite.shape[0] - 1, sprite.shape[1] - 1),
    }
    assert anchor in corners, "start pixel must sit on a sprite corner"

    markers: dict[tuple[int, int], int] = {}
    for comp in components:
        if len(comp) != 1:
            continue
        pos = comp[0]
        color = int(grid[pos])
        assert color in OP_COLORS, "all singleton components must be path markers"
        markers[pos] = color
    return bg, sprite, start_pos, anchor, markers


def _trace_path(
    shape: tuple[int, int], start_pos: tuple[int, int], markers: dict[tuple[int, int], int]
) -> list[tuple[int, int]]:
    visited: set[tuple[int, int]] = set()
    current = start_pos
    path: list[tuple[int, int]] = []
    while True:
        candidates: list[tuple[int, int]] = []
        for dr, dc in DIRS:
            rr, cc = current
            while True:
                rr += dr
                cc += dc
                if not _in_bounds(shape, rr, cc):
                    break
                pos = (rr, cc)
                if pos in markers:
                    if pos not in visited:
                        candidates.append(pos)
                    break
        assert len(candidates) <= 1, "path must be uniquely directed at every step"
        if not candidates:
            break
        nxt = candidates[0]
        visited.add(nxt)
        path.append(nxt)
        current = nxt
    assert len(visited) == len(markers), "all markers must lie on the unique path"
    return path


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct a unique path of isolated colored markers and propagate a sprite along it
    while cumulatively rotating/flipping the sprite at each station.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) >= 2

    bg, sprite, start_pos, anchor, markers = _extract_structure(g)
    path = _trace_path(g.shape, start_pos, markers)

    out = g.copy()
    current_sprite = sprite.copy()
    current_anchor = anchor
    for pos in path:
        current_sprite, current_anchor = _apply_transform(current_sprite, current_anchor, markers[pos])
        _place_sprite(out, current_sprite, current_anchor, pos, bg)
    return out


def _anchor_from_corner(shape: tuple[int, int], corner: str) -> tuple[int, int]:
    h, w = shape
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, w - 1
    if corner == "bl":
        return h - 1, 0
    if corner == "br":
        return h - 1, w - 1
    raise AssertionError(corner)


def _spaced_values(count: int, low: int, high: int, min_gap: int) -> list[int] | None:
    candidates = list(range(int(low), int(high) + 1))
    if count == 0:
        return []
    if not candidates or len(candidates) < count:
        return None
    for _ in range(200):
        values = sorted(random.sample(candidates, count))
        if all(b - a >= min_gap for a, b in zip(values, values[1:])):
            return values
    return None


def _build_points(
    rows: list[int], cols: list[int], horizontal_first: bool, n_points: int
) -> list[tuple[int, int]]:
    points = [(int(rows[0]), int(cols[0]))]
    row_idx = 0
    col_idx = 0
    for step in range(1, n_points):
        if horizontal_first:
            if step % 2 == 1:
                col_idx += 1
            else:
                row_idx += 1
        else:
            if step % 2 == 1:
                row_idx += 1
            else:
                col_idx += 1
        points.append((int(rows[row_idx]), int(cols[col_idx])))
    return points


def _sprite_is_clear(sprite: np.ndarray) -> bool:
    variants = [
        np.rot90(sprite, -1),
        np.rot90(sprite, 1),
        np.fliplr(sprite),
        np.flipud(sprite),
    ]
    return all(not np.array_equal(sprite, variant) for variant in variants)


def _random_sprite(bg: int) -> tuple[np.ndarray, tuple[int, int]]:
    body_colors = [c for c in ALL_COLORS if c not in {bg, START_COLOR, *OP_COLORS}]
    for _ in range(200):
        h = random.randint(2, 4)
        w = random.randint(2, 4)
        palette = random.sample(body_colors, random.randint(2, min(3, len(body_colors))))
        sprite = np.empty((h, w), dtype=int)
        for r in range(h):
            for c in range(w):
                sprite[r, c] = random.choice(palette)
        for color in palette:
            if int(np.sum(sprite == color)) == 0:
                rr = random.randrange(h)
                cc = random.randrange(w)
                sprite[rr, cc] = color

        corner = random.choice(("tl", "tr", "bl", "br"))
        anchor = _anchor_from_corner(sprite.shape, corner)
        sprite[anchor] = START_COLOR

        if len(set(map(int, sprite.ravel()))) < 3:
            continue
        if not _sprite_is_clear(sprite):
            continue
        return sprite, anchor
    raise AssertionError("failed to sample a clear sprite")


def _footprint_cells(
    sprite: np.ndarray, anchor: tuple[int, int], target: tuple[int, int], bg: int
) -> set[tuple[int, int]]:
    ar, ac = anchor
    tr, tc = target
    cells: set[tuple[int, int]] = set()
    for r in range(sprite.shape[0]):
        for c in range(sprite.shape[1]):
            if int(sprite[r, c]) == bg:
                continue
            cells.add((int(tr - ar + r), int(tc - ac + c)))
    return cells


def _states_for_ops(
    sprite: np.ndarray, anchor: tuple[int, int], op_colors: list[int]
) -> list[tuple[np.ndarray, tuple[int, int]]]:
    states: list[tuple[np.ndarray, tuple[int, int]]] = []
    current_sprite = sprite.copy()
    current_anchor = anchor
    for color in op_colors:
        current_sprite, current_anchor = _apply_transform(current_sprite, current_anchor, color)
        states.append((current_sprite.copy(), current_anchor))
    return states


def _path_counts(n_points: int, horizontal_first: bool) -> tuple[int, int]:
    if horizontal_first:
        return (n_points + 1) // 2, n_points // 2 + 1
    return n_points // 2 + 1, (n_points + 1) // 2


def _sample_points(sprite: np.ndarray, path_len: int) -> list[tuple[int, int]] | None:
    n_points = int(path_len + 1)
    m = max(sprite.shape) - 1
    margin = m + 1
    gap = 2 * m + 2
    horizontal_first = random.choice((False, True))
    row_count, col_count = _path_counts(n_points, horizontal_first)
    min_h = 2 * margin + 1 + (row_count - 1) * gap
    min_w = 2 * margin + 1 + (col_count - 1) * gap
    if min_h > 32 or min_w > 32:
        return None

    h = random.randint(min_h, min(32, min_h + 6))
    w = random.randint(min_w, min(32, min_w + 6))
    row_values = _spaced_values(row_count, margin, h - 1 - margin, gap)
    col_values = _spaced_values(col_count, margin, w - 1 - margin, gap)
    if row_values is None or col_values is None:
        return None
    random.shuffle(row_values)
    random.shuffle(col_values)
    return _build_points(row_values, col_values, horizontal_first, n_points)


def _ops_for_path(path_len: int) -> list[int]:
    ops = list(OP_COLORS)
    random.shuffle(ops)
    if path_len > 4:
        ops.extend(random.choice(OP_COLORS) for _ in range(path_len - 4))
        random.shuffle(ops)
    return [int(color) for color in ops]


def _placements_are_clear(
    bg: int,
    sprite: np.ndarray,
    anchor: tuple[int, int],
    points: list[tuple[int, int]],
    op_colors: list[int],
) -> bool:
    occupied = _footprint_cells(sprite, anchor, points[0], bg)
    states = _states_for_ops(sprite, anchor, op_colors)
    for (state_sprite, state_anchor), point in zip(states, points[1:]):
        cells = _footprint_cells(state_sprite, state_anchor, point, bg)
        if occupied & cells:
            return False
        occupied |= cells
    return True


def _render_input(
    shape: tuple[int, int],
    bg: int,
    sprite: np.ndarray,
    anchor: tuple[int, int],
    points: list[tuple[int, int]],
    op_colors: list[int],
) -> np.ndarray:
    grid = np.full(shape, int(bg), dtype=int)
    _place_sprite(grid, sprite, anchor, points[0], bg)
    for point, color in zip(points[1:], op_colors):
        grid[point] = int(color)
    return grid


def generate_input() -> np.ndarray:
    """
    Randomly generate a valid path-and-transform ARC-style input.
    """
    for _ in range(500):
        bg = random.choice([0, 5, 6, 7, 8])
        sprite, anchor = _random_sprite(bg)
        feasible_lengths = []
        for path_len in range(4, 8):
            points = _sample_points(sprite, path_len)
            if points is not None:
                feasible_lengths.append((path_len, points))
        if not feasible_lengths:
            continue

        path_len, points = random.choice(feasible_lengths)
        op_colors = _ops_for_path(path_len)
        if not _placements_are_clear(bg, sprite, anchor, points, op_colors):
            continue

        shape = (
            max(r for r, _ in points) + max(sprite.shape),
            max(c for _, c in points) + max(sprite.shape),
        )
        shape = (
            max(shape[0], max(r for r, _ in points) + 1),
            max(shape[1], max(c for _, c in points) + 1),
        )
        grid = _render_input(shape, bg, sprite, anchor, points, op_colors)

        try:
            out = main(grid)
        except Exception:
            continue

        if out.shape != grid.shape:
            continue
        if not np.any(out != grid):
            continue
        if len(set(op_colors)) != 4:
            continue
        return grid
    raise AssertionError("failed to generate a valid input")
