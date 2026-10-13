from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
PIPE_COLORS = (BLUE, RED)
FRAME_COLORS = (3, 4, 5, 6, 7, 8, 9)

GRID_SIZE_RANGE = (30, 38)
OUTER_SIZE_RANGE = (4, 6)
PIPE_LEN_RANGE = (2, 5)
PAIR_COUNT_OPTIONS = (4, 5)
MAX_GENERATION_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 200


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[dict[str, object]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[dict[str, object]] = []
    for r, c in np.argwhere(mask & ~seen):
        if seen[int(r), int(c)]:
            continue
        stack = [(int(r), int(c))]
        seen[int(r), int(c)] = True
        cells: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            cells.append((cr, cc))
            for nr, nc in _neighbors4(cr, cc):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        coords = np.array(cells, dtype=int)
        r0 = int(coords[:, 0].min())
        c0 = int(coords[:, 1].min())
        r1 = int(coords[:, 0].max())
        c1 = int(coords[:, 1].max())
        comps.append(
            {
                "coords": coords,
                "bbox": (r0, c0, r1, c1),
                "size": int(coords.shape[0]),
            }
        )
    return comps


def _frame_border_cells(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        cells.append((r, c1))
    return cells


def _is_rect_frame(grid: np.ndarray, color: int, comp: dict[str, object]) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    if int(comp["size"]) != 2 * (h + w) - 4:
        return False
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    border = np.ones((h, w), dtype=bool)
    border[1:-1, 1:-1] = False
    return bool(np.all(patch[border] == color) and np.all(patch[~border] != color))


def _find_containers(grid: np.ndarray) -> list[dict[str, object]]:
    containers: list[dict[str, object]] = []
    for color in sorted({int(v) for v in grid.flat if int(v) != BG}):
        for comp in _components(grid == color):
            if not _is_rect_frame(grid, color, comp):
                continue
            r0, c0, r1, c1 = comp["bbox"]
            containers.append(
                {
                    "color": int(color),
                    "bbox": (r0, c0, r1, c1),
                    "interior_bbox": (r0 + 1, c0 + 1, r1 - 1, c1 - 1),
                    "border_cells": _frame_border_cells((r0, c0, r1, c1)),
                }
            )
    return containers


def _find_pipes(grid: np.ndarray) -> list[dict[str, object]]:
    pipes: list[dict[str, object]] = []
    for color in PIPE_COLORS:
        for comp in _components(grid == color):
            r0, c0, r1, c1 = comp["bbox"]
            if r0 == r1:
                assert int(comp["size"]) == c1 - c0 + 1, "horizontal pipe must be solid"
                orient = "horizontal"
            elif c0 == c1:
                assert int(comp["size"]) == r1 - r0 + 1, "vertical pipe must be solid"
                orient = "vertical"
            else:
                raise AssertionError("pipe components must be straight lines")
            pipes.append({"color": int(color), "bbox": (r0, c0, r1, c1), "orient": orient})
    return pipes


def _nonempty(patch: np.ndarray) -> bool:
    return bool(np.any(patch != BG))


def _transform_patch(patch: np.ndarray, pipe_color: int, orient: str) -> np.ndarray:
    if pipe_color == RED:
        return patch.copy()
    if orient == "horizontal":
        return np.fliplr(patch)
    return np.flipud(patch)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy or mirror a source container's interior into the empty container linked by a pipe.

    Red pipes copy the pattern unchanged. Blue pipes mirror the pattern across
    the pipe axis: horizontal pipes flip left-right, vertical pipes flip up-down.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    containers = _find_containers(g)
    pipes = _find_pipes(g)
    assert pipes, "expected at least one pipe"

    border_to_container: dict[tuple[int, int], int] = {}
    for idx, container in enumerate(containers):
        for cell in container["border_cells"]:
            assert cell not in border_to_container, "containers may not overlap"
            border_to_container[cell] = idx

    out = g.copy()
    used: set[int] = set()

    for pipe in pipes:
        r0, c0, r1, c1 = pipe["bbox"]
        if pipe["orient"] == "horizontal":
            left_idx = border_to_container.get((r0, c0 - 1))
            right_idx = border_to_container.get((r0, c1 + 1))
        else:
            left_idx = border_to_container.get((r0 - 1, c0))
            right_idx = border_to_container.get((r1 + 1, c0))

        assert left_idx is not None and right_idx is not None and left_idx != right_idx
        assert left_idx not in used and right_idx not in used, "containers may only belong to one pair"
        used.add(left_idx)
        used.add(right_idx)

        a = containers[left_idx]
        b = containers[right_idx]
        ar0, ac0, ar1, ac1 = a["interior_bbox"]
        br0, bc0, br1, bc1 = b["interior_bbox"]
        patch_a = g[ar0 : ar1 + 1, ac0 : ac1 + 1]
        patch_b = g[br0 : br1 + 1, bc0 : bc1 + 1]

        has_a = _nonempty(patch_a)
        has_b = _nonempty(patch_b)
        assert has_a ^ has_b, "exactly one container in a pair must contain a pattern"
        assert patch_a.shape == patch_b.shape, "paired containers must share the same interior size"

        if has_a:
            src_patch = patch_a
            dst_bbox = (br0, bc0, br1, bc1)
        else:
            src_patch = patch_b
            dst_bbox = (ar0, ac0, ar1, ac1)

        dr0, dc0, dr1, dc1 = dst_bbox
        assert np.all(out[dr0 : dr1 + 1, dc0 : dc1 + 1] == BG), "target container must start empty"
        out[dr0 : dr1 + 1, dc0 : dc1 + 1] = _transform_patch(src_patch, int(pipe["color"]), str(pipe["orient"]))

    assert len(used) == 2 * len(pipes), "each pipe must connect two distinct containers"
    return out


def _draw_frame(grid: np.ndarray, top: int, left: int, height: int, width: int, color: int) -> None:
    grid[top, left : left + width] = color
    grid[top + height - 1, left : left + width] = color
    grid[top : top + height, left] = color
    grid[top : top + height, left + width - 1] = color


def _random_pattern(height: int, width: int, rng: random.Random, blocked: set[int]) -> np.ndarray:
    palette = [c for c in FRAME_COLORS if c not in blocked]
    assert palette
    min_cells = max(2, (height * width) // 3)
    max_cells = max(min_cells, (2 * height * width) // 3)
    for _ in range(200):
        patch = np.zeros((height, width), dtype=int)
        cell_count = rng.randint(min_cells, max_cells)
        palette_size = rng.randint(1, min(3, len(palette)))
        colors = rng.sample(palette, palette_size)
        for idx in rng.sample(range(height * width), cell_count):
            patch.flat[idx] = rng.choice(colors)
        if np.count_nonzero(patch) < 2:
            continue
        if np.array_equal(patch, np.fliplr(patch)):
            continue
        if np.array_equal(patch, np.flipud(patch)):
            continue
        return patch
    raise RuntimeError("failed to sample a non-symmetric pattern")


def _sample_group_dims(
    grid_shape: tuple[int, int], orient: str, rng: random.Random
) -> tuple[int, int, int, int, int]:
    h_lim, w_lim = grid_shape
    options: list[tuple[int, int, int, int, int]] = []
    for outer_h in range(OUTER_SIZE_RANGE[0], OUTER_SIZE_RANGE[1] + 1):
        for outer_w in range(OUTER_SIZE_RANGE[0], OUTER_SIZE_RANGE[1] + 1):
            for pipe_len in range(PIPE_LEN_RANGE[0], PIPE_LEN_RANGE[1] + 1):
                if orient == "horizontal":
                    group_h = outer_h
                    group_w = 2 * outer_w + pipe_len
                else:
                    group_h = 2 * outer_h + pipe_len
                    group_w = outer_w
                if group_h + 2 <= h_lim and group_w + 2 <= w_lim:
                    options.append((outer_h, outer_w, pipe_len, group_h, group_w))
    assert options, "grid must be large enough for at least one pair"
    return rng.choice(options)


def _paint_group(
    grid: np.ndarray,
    out: np.ndarray,
    occupied: np.ndarray,
    *,
    pipe_color: int,
    orient: str,
    frame_color: int,
    rng: random.Random,
) -> bool:
    outer_h, outer_w, pipe_len, group_h, group_w = _sample_group_dims(grid.shape, orient, rng)
    h, w = grid.shape
    for _ in range(PLACEMENT_ATTEMPTS):
        top = rng.randint(1, h - group_h - 1)
        left = rng.randint(1, w - group_w - 1)
        halo_r0 = top - 1
        halo_c0 = left - 1
        halo_r1 = top + group_h + 1
        halo_c1 = left + group_w + 1
        if np.any(occupied[halo_r0:halo_r1, halo_c0:halo_c1]):
            continue

        patch = _random_pattern(outer_h - 2, outer_w - 2, rng, {frame_color, BLUE, RED})
        source_first = bool(rng.getrandbits(1))

        if orient == "horizontal":
            first_top, first_left = top, left
            second_top, second_left = top, left + outer_w + pipe_len
            pipe_row = top + rng.randint(1, outer_h - 2)
            grid[pipe_row, first_left + outer_w : second_left] = pipe_color
            out[pipe_row, first_left + outer_w : second_left] = pipe_color
        else:
            first_top, first_left = top, left
            second_top, second_left = top + outer_h + pipe_len, left
            pipe_col = left + rng.randint(1, outer_w - 2)
            grid[first_top + outer_h : second_top, pipe_col] = pipe_color
            out[first_top + outer_h : second_top, pipe_col] = pipe_color

        _draw_frame(grid, first_top, first_left, outer_h, outer_w, frame_color)
        _draw_frame(grid, second_top, second_left, outer_h, outer_w, frame_color)
        _draw_frame(out, first_top, first_left, outer_h, outer_w, frame_color)
        _draw_frame(out, second_top, second_left, outer_h, outer_w, frame_color)

        if source_first:
            src_top, src_left = first_top, first_left
            dst_top, dst_left = second_top, second_left
        else:
            src_top, src_left = second_top, second_left
            dst_top, dst_left = first_top, first_left

        grid[src_top + 1 : src_top + outer_h - 1, src_left + 1 : src_left + outer_w - 1] = patch
        out[src_top + 1 : src_top + outer_h - 1, src_left + 1 : src_left + outer_w - 1] = patch
        transformed = _transform_patch(patch, pipe_color, orient)
        out[dst_top + 1 : dst_top + outer_h - 1, dst_left + 1 : dst_left + outer_w - 1] = transformed

        occupied[halo_r0:halo_r1, halo_c0:halo_c1] = True
        return True
    return False


def _apply_transform(grid: np.ndarray, op: int) -> np.ndarray:
    if op < 4:
        return np.rot90(grid, op)
    return np.fliplr(np.rot90(grid, op - 4))


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    pair_count = rng.choice(PAIR_COUNT_OPTIONS)
    height = rng.randint(GRID_SIZE_RANGE[0], GRID_SIZE_RANGE[1])
    width = rng.randint(GRID_SIZE_RANGE[0], GRID_SIZE_RANGE[1])
    if pair_count == 5:
        height = max(height, 34)
        width = max(width, 34)

    grid = np.full((height, width), BG, dtype=int)
    expected = grid.copy()
    occupied = np.zeros_like(grid, dtype=bool)

    combos = [
        (RED, "horizontal"),
        (RED, "vertical"),
        (BLUE, "horizontal"),
        (BLUE, "vertical"),
    ]
    if pair_count > 4:
        combos.extend(rng.choices(combos, k=pair_count - 4))
    rng.shuffle(combos)

    frame_colors = rng.sample(FRAME_COLORS, len(combos))
    for frame_color, (pipe_color, orient) in zip(frame_colors, combos):
        if not _paint_group(
            grid,
            expected,
            occupied,
            pipe_color=pipe_color,
            orient=orient,
            frame_color=frame_color,
            rng=rng,
        ):
            return None

    op = rng.randrange(8)
    grid = _apply_transform(grid, op)
    expected = _apply_transform(expected, op)

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if np.array_equal(actual, grid):
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
