from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import numpy as np


def _load_arc_utils():
    """Load arc_utils from the repository even when this file is loaded directly."""
    try:
        import arc_utils as mod  # type: ignore
        return mod
    except Exception:
        candidates = [
            Path(__file__).resolve().parents[6] / "new" / "solvers" / "arc_utils.py",
            Path("/home/lixinyu/snap/new/solvers/arc_utils.py"),
        ]
        for path in candidates:
            if path.exists():
                spec = importlib.util.spec_from_file_location("_arc_utils", path)
                if spec is None or spec.loader is None:
                    continue
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                return mod
        raise


_UTILS = _load_arc_utils()
Color = _UTILS.Color
get_bg = _UTILS.get_bg
in_bounds = _UTILS.in_bounds
neighbors4 = _UTILS.neighbors4

BG = Color.BLACK
GRAY = Color.GRAY if hasattr(Color, "GRAY") else Color.GREY
BLUE = Color.BLUE


def _discover_pointers(grid: np.ndarray, bg: int):
    H, W = grid.shape
    grays = np.argwhere(grid == GRAY)
    blue_cells = set(map(tuple, np.argwhere(grid == BLUE).tolist()))

    pointers: list[dict[str, object]] = []
    pointer_by_cell: dict[tuple[int, int], int] = {}
    used_gray: set[tuple[int, int]] = set()
    used_blue: set[tuple[int, int]] = set()

    for gr in grays.tolist():
        gr = int(gr[0]), int(gr[1])
        if not _interior(gr, H, W):
            continue
        if gr in used_gray:
            continue
        neighbors: list[tuple[int, int]] = []
        for nr, nc in neighbors4(*gr):
            if in_bounds((H, W), int(nr), int(nc)) and int(grid[nr, nc]) == BLUE:
                neighbors.append((int(nr), int(nc)))

        if len(neighbors) != 1:
            raise AssertionError("each gray pixel must form a 1x2 pointer with exactly one blue neighbor")

        tail = neighbors[0]
        if tail in used_blue:
            raise AssertionError("a blue pixel cannot belong to multiple pointers")

        used_gray.add(gr)
        used_blue.add(tail)

        dr = gr[0] - tail[0]
        dc = gr[1] - tail[1]
        if abs(dr) + abs(dc) != 1:
            raise AssertionError("pointer cells must be orthogonally adjacent")

        pid = len(pointers)
        pointers.append(
            {
                "head": gr,
                "tail": tail,
                "dir": (dr, dc),
                "cells": (gr, tail),
            }
        )
        pointer_by_cell[gr] = pid
        pointer_by_cell[tail] = pid

    if len(blue_cells) != len(pointers):
        raise AssertionError("each blue pixel must pair with exactly one gray pixel")

    return pointers, pointer_by_cell


def _first_hit(
    grid: np.ndarray,
    pointer_by_cell: dict[tuple[int, int], int],
    bg: int,
    head: tuple[int, int],
    direction: tuple[int, int],
) -> tuple[str, int]:
    """Trace along a ray from the head and return ('frame', color) or ('pointer', idx)."""
    H, W = grid.shape
    r, c = head
    dr, dc = direction
    r += dr
    c += dc

    while in_bounds((H, W), r, c):
        cell = (int(r), int(c))
        owner = pointer_by_cell.get(cell)
        if owner is not None:
            return "pointer", owner

        v = int(grid[r, c])
        if v != bg:
            return "frame", v

        r += dr
        c += dc

    raise AssertionError("sightline must hit frame or another pointer")


def _in_front(grid: np.ndarray, bg: int) -> tuple[np.ndarray, dict[tuple[int, int], int]]:
    pointers, pointer_by_cell = _discover_pointers(grid, bg)
    assert pointers, "expected at least one pointer"

    memo: dict[int, int] = {}
    stack: set[int] = set()
    out = np.array(grid, dtype=int).copy()

    def resolve(pid: int) -> int:
        if pid in memo:
            return memo[pid]
        if pid in stack:
            raise AssertionError("pointer dependency cycle detected")

        stack.add(pid)
        pointer = pointers[pid]
        head = pointer["head"]  # type: ignore[assignment]
        direction = pointer["dir"]  # type: ignore[assignment]
        kind, target = _first_hit(grid, pointer_by_cell, bg, head, direction)

        if kind == "frame":
            color = target
            if color == GRAY:
                raise AssertionError("ray hitting a frame corner is not valid")
        else:
            color = resolve(target)

        stack.remove(pid)
        memo[pid] = color
        return color

    for i, pointer in enumerate(pointers):
        color = resolve(i)
        hr, hc = pointer["head"]  # type: ignore[assignment]
        tr, tc = pointer["tail"]  # type: ignore[assignment]
        out[hr, hc] = color
        out[tr, tc] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate each domino pointer's final color along its ray, following pointer chains back to the frame.

    Input structure:
    - A one-cell-thick rectangular frame occupies the grid boundary; side colors are distinct, non-blue, non-gray colors.
    - Corners are neutral gray.
    - Interior background is black.
    - One or more 1x2 domino pointers are present, each composed of one gray head and one blue tail.
    - A pointer faces from its blue tail toward its gray head.

    Rule:
    - For each pointer, cast a ray from the head in its facing direction.
    - The first non-background cell on the ray determines the target.
    - If the target is a frame side pixel, the pointer final color is that frame color.
    - If the target is another pointer, the color is that pointed pointer's final color (resolved recursively).
    - Repaint both cells of each pointer with its final color, leaving frame, corners, and background unchanged.

    Key insight:
    - Colors propagate from the frame through dependency chains, so pointers must be resolved in topological order of ray targets instead of by local appearance.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = int(get_bg(g))
    return _in_front(g, bg)


def _interior(point: tuple[int, int], H: int, W: int) -> bool:
    r, c = point
    return 0 < r < H - 1 and 0 < c < W - 1


def _local_pointer_free(grid: np.ndarray, head: tuple[int, int], tail: tuple[int, int]) -> bool:
    """Return True if placing gray/blue there creates an isolated 1x2 pointer."""
    H, W = grid.shape
    for cell in (head, tail):
        r, c = cell
        for nr, nc in neighbors4(r, c):
            if not in_bounds((H, W), int(nr), int(nc)):
                continue
            if int(grid[nr, nc]) == BLUE:
                if cell == head and (nr, nc) == tail:
                    continue
                if cell == tail and (nr, nc) == head:
                    continue
                return False
    return True


def _side_colors(rng: random.Random) -> list[int]:
    palette = [c for c in Color.NOT_BLACK if c not in {Color.BLUE, Color.GREY}]
    return rng.sample(palette, 4)


def _build_frame(H: int, W: int, sides: list[int]) -> np.ndarray:
    top, bottom, left, right = sides
    grid = np.full((H, W), BG, dtype=int)
    grid[0, :] = top
    grid[H - 1, :] = bottom
    grid[:, 0] = left
    grid[:, W - 1] = right
    grid[0, 0] = GRAY
    grid[0, W - 1] = GRAY
    grid[H - 1, 0] = GRAY
    grid[H - 1, W - 1] = GRAY
    return grid


def _place_terminal_pointer(
    rng: random.Random,
    grid: np.ndarray,
    pointer_by_cell: dict[tuple[int, int], int],
    bg: int,
    attempts: int = 600,
) -> dict[str, tuple[int, int] | tuple[int, int, int, int] | tuple[int, int] | tuple[int, int] | tuple[int, int]] | None:
    H, W = grid.shape
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for _ in range(attempts):
        dr, dc = rng.choice(directions)
        head_r = rng.randint(1, H - 2)
        head_c = rng.randint(1, W - 2)
        head = (head_r, head_c)
        tail = (head_r - dr, head_c - dc)

        if not _interior(tail, H, W):
            continue
        if grid[head] != bg or grid[tail] != bg:
            continue
        if head in pointer_by_cell or tail in pointer_by_cell:
            continue
        if not _local_pointer_free(grid, head, tail):
            continue

        kind, target = _first_hit(grid, pointer_by_cell, bg, head, (dr, dc))
        if kind == "frame" and target != GRAY:
            return {"head": head, "tail": tail, "dir": (dr, dc), "cells": (head, tail)}

    return None


def _max_chain_distance(
    target: tuple[int, int],
    direction: tuple[int, int],
    H: int,
    W: int,
) -> int:
    tr, tc = target
    dr, dc = direction
    max_d = 0
    d = 2
    while True:
        head = (tr - d * dr, tc - d * dc)
        tail = (tr - (d + 1) * dr, tc - (d + 1) * dc)
        if _interior(head, H, W) and _interior(tail, H, W):
            max_d = d
            d += 1
            continue
        break
    return max_d


def _place_inherit_pointer(
    rng: random.Random,
    grid: np.ndarray,
    pointer_by_cell: dict[tuple[int, int], int],
    pointers: list[dict[str, object]],
    bg: int,
    attempts: int = 1200,
) -> dict[str, tuple[int, int] | tuple[int, int, int, int] | tuple[int, int] | tuple[int, int] | tuple[int, int]] | None:
    H, W = grid.shape
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    if not pointers:
        return None

    for _ in range(attempts):
        target_idx = rng.randint(0, len(pointers) - 1)
        target_cells = pointers[target_idx]["cells"]  # type: ignore[assignment]
        target = rng.choice(target_cells)  # type: ignore[arg-type]
        dr, dc = rng.choice(directions)
        max_d = _max_chain_distance(target, (dr, dc), H, W)
        if max_d < 2:
            continue

        d = rng.randint(2, max_d)
        head = (target[0] - d * dr, target[1] - d * dc)
        tail = (head[0] - dr, head[1] - dc)

        if head in pointer_by_cell or tail in pointer_by_cell:
            continue
        if not _interior(head, H, W) or not _interior(tail, H, W):
            continue
        if grid[head] != bg or grid[tail] != bg:
            continue

        kind, hit = _first_hit(grid, pointer_by_cell, bg, head, (dr, dc))
        if kind == "pointer" and hit == target_idx:
            if not _local_pointer_free(grid, head, tail):
                continue
            return {"head": head, "tail": tail, "dir": (dr, dc), "cells": (head, tail)}

    return None


def _build_pointer_grid(rng: random.Random) -> np.ndarray | None:
    H = rng.randint(12, 24)
    W = rng.randint(12, 24)
    bg = BG
    side_colors = _side_colors(rng)
    grid = _build_frame(H, W, side_colors)

    n_pointers = rng.randint(2, 6)
    terminal_count = rng.randint(1, n_pointers - 1)

    pointers: list[dict[str, object]] = []
    pointer_by_cell: dict[tuple[int, int], int] = {}

    for i in range(n_pointers):
        if i < terminal_count:
            spec = _place_terminal_pointer(rng, grid, pointer_by_cell, bg)
            if spec is None:
                return None
        else:
            spec = _place_inherit_pointer(rng, grid, pointer_by_cell, pointers, bg)
            if spec is None:
                return None

        pid = len(pointers)
        head = spec["head"]  # type: ignore[assignment]
        tail = spec["tail"]  # type: ignore[assignment]
        grid[head[0], head[1]] = GRAY
        grid[tail[0], tail[1]] = BLUE

        for cell in (head, tail):
            pointer_by_cell[cell] = pid

        pointers.append(spec)

    return grid


def generate_input() -> np.ndarray:
    """Generate random inputs covering direct hits and pointer-chain inherited hits."""
    rng = random.Random()

    for _ in range(500):
        grid = _build_pointer_grid(rng)
        if grid is None:
            continue

        bg = int(get_bg(grid))

        try:
            pointers, pointer_by_cell = _discover_pointers(grid, bg)
            kind_count = {"frame": 0, "pointer": 0}
            for p in pointers:
                kind, val = _first_hit(grid, pointer_by_cell, bg, p["head"], p["dir"])  # type: ignore[index]
                kind_count[kind] += 1
                if kind == "frame" and val == GRAY:
                    raise AssertionError("invalid sample: corner hit")

            if kind_count["frame"] < 1 or kind_count["pointer"] < 1:
                continue

            _ = main(grid)
        except Exception:
            continue

        return grid

    raise RuntimeError("failed to generate a valid input sample")
