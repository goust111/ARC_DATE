from __future__ import annotations

import numpy as np


BG = 0
GREY = 5
MAROON = 9
EMITTER_PALETTE = [1, 2, 3, 4, 6, 7, 8]

DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

GRID_H_RANGE = (14, 27)
GRID_W_RANGE = (14, 27)
EMITTER_COUNT_RANGE = (2, 5)
OUTER_RETRIES = 400
PLACEMENT_TRIES = 220


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(shape: tuple[int, int], r: int, c: int):
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr = r + dr
        cc = c + dc
        if _in_bounds(shape, rr, cc):
            yield rr, cc


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        coords: list[tuple[int, int]] = []
        while stack:
            rr, cc = stack.pop()
            coords.append((rr, cc))
            for nr, nc in _neighbors4(mask.shape, rr, cc):
                if mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(np.array(coords, dtype=int))
    return comps


def _emitter_cells(top: int, left: int) -> list[tuple[int, int]]:
    return [
        (top, left),
        (top, left + 1),
        (top + 1, left),
        (top + 1, left + 1),
    ]


def _grey_cell(top: int, left: int, direction: str, lane: int) -> tuple[int, int]:
    if direction == "up":
        return top - 1, left + lane
    if direction == "down":
        return top + 2, left + lane
    if direction == "left":
        return top + lane, left - 1
    if direction == "right":
        return top + lane, left + 2
    raise ValueError(direction)


def _find_marker_info(grid: np.ndarray, top: int, left: int) -> tuple[str, tuple[int, int]]:
    candidates = [
        ("up", [(top - 1, left), (top - 1, left + 1)]),
        ("down", [(top + 2, left), (top + 2, left + 1)]),
        ("left", [(top, left - 1), (top + 1, left - 1)]),
        ("right", [(top, left + 2), (top + 1, left + 2)]),
    ]
    found: list[tuple[str, tuple[int, int]]] = []
    for direction, cells in candidates:
        for r, c in cells:
            if _in_bounds(grid.shape, r, c) and int(grid[r, c]) == GREY:
                found.append((direction, (r, c)))
    assert len(found) == 1, "each emitter must have exactly one adjacent grey marker"
    return found[0]


def _find_emitters(grid: np.ndarray) -> list[dict[str, object]]:
    emitters: list[dict[str, object]] = []
    for color in sorted(int(x) for x in np.unique(grid)):
        if color in (BG, GREY, MAROON):
            continue
        coords = np.argwhere(grid == color)
        assert coords.shape == (4, 2), "each emitter color must occupy a 2x2 block"
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        r0 = int(r0)
        c0 = int(c0)
        r1 = int(r1)
        c1 = int(c1)
        assert (r1 - r0, c1 - c0) == (1, 1), "emitter bbox must be 2x2"
        cell_set = {tuple(map(int, rc)) for rc in coords.tolist()}
        assert cell_set == set(_emitter_cells(r0, c0)), "emitter must be a solid 2x2 square"
        direction, marker = _find_marker_info(grid, r0, c0)
        emitters.append(
            {
                "color": color,
                "top": r0,
                "left": c0,
                "cells": cell_set,
                "direction": direction,
                "marker": marker,
            }
        )
    return sorted(emitters, key=lambda item: (int(item["top"]), int(item["left"])))


def _build_mirror_map(grid: np.ndarray) -> dict[tuple[int, int], tuple[str, int, int]]:
    mirror_map: dict[tuple[int, int], tuple[str, int, int]] = {}
    for comp in _components(grid == MAROON):
        assert comp.shape[0] == 2, "each mirror must be exactly two maroon pixels"
        rows = sorted(int(x) for x in comp[:, 0].tolist())
        cols = sorted(int(x) for x in comp[:, 1].tolist())
        if rows[0] == rows[1] and cols[1] == cols[0] + 1:
            row = rows[0]
            left = cols[0]
            desc = ("h", row, left)
            mirror_map[(row, left)] = desc
            mirror_map[(row, left + 1)] = desc
        elif cols[0] == cols[1] and rows[1] == rows[0] + 1:
            top = rows[0]
            col = cols[0]
            desc = ("v", top, col)
            mirror_map[(top, col)] = desc
            mirror_map[(top + 1, col)] = desc
        else:
            raise AssertionError("maroon components must be horizontal or vertical pairs")
    return mirror_map


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace one-cell beams from 2x2 emitters, bending them with maroon mirrors.

    Input structure:
    - Black is the background.
    - Each emitter is a unique-colored solid 2x2 block with one adjacent grey pixel; the side of that grey pixel gives the launch direction, and its row/column alignment picks the beam lane.
    - Maroon appears only as isolated 1x2 or 2x1 mirror pieces; emitters, mirrors, and grey markers stay fixed.

    Rule:
    - For every emitter, start from its grey marker and move one cell at a time in the indicated cardinal direction.
    - Paint only the black cells visited by the beam with the emitter color.
    - When a vertical beam reaches a horizontal maroon pair, redirect it to the right from that mirror; when a horizontal beam reaches a vertical maroon pair, redirect it downward.
    - Stop when the next step would leave the grid or enter an emitter cell.

    Key insight:
    - The puzzle combines two spatial cues: grey adjacency chooses the source side and the exact one-pixel beam lane, while mirror orientation determines the only legal bend, so each emitter follows a deterministic polyline.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    emitters = _find_emitters(g)
    emitter_cells = {cell for emitter in emitters for cell in emitter["cells"]}
    mirror_map = _build_mirror_map(g)

    for emitter in emitters:
        color = int(emitter["color"])
        direction = str(emitter["direction"])
        dr, dc = DIRS[direction]
        r, c = map(int, emitter["marker"])
        while True:
            nr = r + dr
            nc = c + dc
            if not _in_bounds(g.shape, nr, nc):
                break
            if (nr, nc) in emitter_cells:
                break
            cell = int(g[nr, nc])
            if cell == MAROON:
                mirror = mirror_map.get((nr, nc))
                if mirror is not None and dr != 0 and mirror[0] == "h":
                    r, c = mirror[1], mirror[2] + 1
                    dr, dc = DIRS["right"]
                    continue
                if mirror is not None and dc != 0 and mirror[0] == "v":
                    r, c = mirror[1] + 1, mirror[2]
                    dr, dc = DIRS["down"]
                    continue
            r, c = nr, nc
            if cell == BG and int(out[nr, nc]) == BG:
                out[nr, nc] = color
    return out


def _mirror_ok(grid: np.ndarray, path_taken: np.ndarray, mirror_mask: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    proposed = set(cells)
    for r, c in cells:
        if not _in_bounds(grid.shape, r, c):
            return False
        if int(grid[r, c]) != BG or bool(path_taken[r, c]):
            return False
    for r, c in cells:
        for nr, nc in _neighbors4(grid.shape, r, c):
            if bool(mirror_mask[nr, nc]) and (nr, nc) not in proposed:
                return False
    return True


def _path_cells_clear(grid: np.ndarray, path_taken: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    for r, c in cells:
        if not _in_bounds(grid.shape, r, c):
            return False
        if int(grid[r, c]) != BG or bool(path_taken[r, c]):
            return False
    return True


def _finish_route(
    grid: np.ndarray,
    path_taken: np.ndarray,
    current: tuple[int, int],
    direction: str,
    targets: list[tuple[int, int]] | None,
) -> list[tuple[int, int]] | None:
    dr, dc = DIRS[direction]
    r, c = current
    if targets is None:
        cells: list[tuple[int, int]] = []
        while True:
            r += dr
            c += dc
            if not _in_bounds(grid.shape, r, c):
                break
            if int(grid[r, c]) != BG or bool(path_taken[r, c]):
                return None
            cells.append((r, c))
        return cells if cells else None

    for tr, tc in targets:
        if dr != 0 and tc == c and (tr - r) * dr > 1:
            cells = [(rr, c) for rr in range(r + dr, tr, dr)]
        elif dc != 0 and tr == r and (tc - c) * dc > 1:
            cells = [(r, cc) for cc in range(c + dc, tc, dc)]
        else:
            continue
        if _path_cells_clear(grid, path_taken, cells) and int(grid[tr, tc]) not in (BG, GREY, MAROON):
            return cells
    return None


def _plan_route(
    grid: np.ndarray,
    path_taken: np.ndarray,
    current: tuple[int, int],
    direction: str,
    turns_left: int,
    targets: list[tuple[int, int]] | None,
    rng: np.random.Generator,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    if turns_left == 0:
        tail = _finish_route(grid, path_taken, current, direction, targets)
        if tail is None:
            return None
        return [], tail

    mirror_mask = grid == MAROON
    h, w = grid.shape
    r, c = current
    candidates: list[tuple[list[tuple[int, int]], list[tuple[int, int]], tuple[int, int], str]] = []

    if direction in ("up", "down"):
        if c + 1 >= w:
            return None
        rows = list(range(r + 2, h)) if direction == "down" else list(range(0, r - 1))
        rng.shuffle(rows)
        step = 1 if direction == "down" else -1
        for mr in rows:
            seg = [(rr, c) for rr in range(r + step, mr, step)]
            mirror = [(mr, c), (mr, c + 1)]
            if not seg or not _path_cells_clear(grid, path_taken, seg):
                continue
            if not _mirror_ok(grid, path_taken, mirror_mask, mirror):
                continue
            candidates.append((mirror, seg, (mr, c + 1), "right"))
    else:
        if r + 1 >= h:
            return None
        cols = list(range(c + 2, w)) if direction == "right" else list(range(0, c - 1))
        rng.shuffle(cols)
        step = 1 if direction == "right" else -1
        for mc in cols:
            seg = [(r, cc) for cc in range(c + step, mc, step)]
            mirror = [(r, mc), (r + 1, mc)]
            if not seg or not _path_cells_clear(grid, path_taken, seg):
                continue
            if not _mirror_ok(grid, path_taken, mirror_mask, mirror):
                continue
            candidates.append((mirror, seg, (r + 1, mc), "down"))

    rng.shuffle(candidates)
    for mirror, seg, nxt, ndir in candidates:
        g2 = grid.copy()
        p2 = path_taken.copy()
        for rr, cc in mirror:
            g2[rr, cc] = MAROON
        for rr, cc in seg:
            p2[rr, cc] = True
        rest = _plan_route(g2, p2, nxt, ndir, turns_left - 1, targets, rng)
        if rest is None:
            continue
        more_mirrors, tail = rest
        return mirror + more_mirrors, seg + tail
    return None


def _site_clear(
    grid: np.ndarray,
    path_taken: np.ndarray,
    emitter_cells: list[tuple[int, int]],
    marker: tuple[int, int],
) -> bool:
    all_cells = emitter_cells + [marker]
    for r, c in all_cells:
        if not _in_bounds(grid.shape, r, c):
            return False
        if int(grid[r, c]) != BG or bool(path_taken[r, c]):
            return False
    rows = [r for r, _ in all_cells]
    cols = [c for _, c in all_cells]
    r0 = max(0, min(rows) - 1)
    r1 = min(grid.shape[0] - 1, max(rows) + 1)
    c0 = max(0, min(cols) - 1)
    c1 = min(grid.shape[1] - 1, max(cols) + 1)
    if np.any(grid[r0 : r1 + 1, c0 : c1 + 1] != BG):
        return False
    if np.any(path_taken[r0 : r1 + 1, c0 : c1 + 1]):
        return False
    return True


def _top_range(height: int, direction: str, top_min: int | None, top_max: int | None) -> tuple[int, int] | None:
    lo = 1 if direction == "up" else 0
    hi = height - 3 if direction == "down" else height - 2
    if top_min is not None:
        lo = max(lo, top_min)
    if top_max is not None:
        hi = min(hi, top_max)
    return (lo, hi) if lo <= hi else None


def _left_range(width: int, direction: str, left_min: int | None, left_max: int | None) -> tuple[int, int] | None:
    lo = 1 if direction == "left" else 0
    hi = width - 3 if direction == "right" else width - 2
    if left_min is not None:
        lo = max(lo, left_min)
    if left_max is not None:
        hi = min(hi, left_max)
    return (lo, hi) if lo <= hi else None


def _place_emitter(
    grid: np.ndarray,
    path_taken: np.ndarray,
    color: int,
    direction_choices: list[str],
    turns_choices: list[int],
    rng: np.random.Generator,
    targets: list[tuple[int, int]] | None = None,
    top_min: int | None = None,
    top_max: int | None = None,
    left_min: int | None = None,
    left_max: int | None = None,
) -> dict[str, object] | None:
    h, w = grid.shape
    for _ in range(PLACEMENT_TRIES):
        direction = str(rng.choice(direction_choices))
        turns = int(rng.choice(turns_choices))
        lane = int(rng.integers(0, 2))
        top_bounds = _top_range(h, direction, top_min, top_max)
        left_bounds = _left_range(w, direction, left_min, left_max)
        if top_bounds is None or left_bounds is None:
            continue
        top = int(rng.integers(top_bounds[0], top_bounds[1] + 1))
        left = int(rng.integers(left_bounds[0], left_bounds[1] + 1))
        emitter_cells = _emitter_cells(top, left)
        marker = _grey_cell(top, left, direction, lane)
        if not _site_clear(grid, path_taken, emitter_cells, marker):
            continue

        g2 = grid.copy()
        p2 = path_taken.copy()
        for r, c in emitter_cells:
            g2[r, c] = color
        g2[marker] = GREY

        route = _plan_route(g2, p2, marker, direction, turns, targets, rng)
        if route is None:
            continue

        mirrors, path_cells = route
        for r, c in emitter_cells:
            grid[r, c] = color
        grid[marker] = GREY
        for r, c in mirrors:
            grid[r, c] = MAROON
        for r, c in path_cells:
            path_taken[r, c] = True
        return {
            "color": color,
            "top": top,
            "left": left,
            "direction": direction,
            "marker": marker,
            "turns": turns,
        }
    return None


def _count_mirror_components(grid: np.ndarray) -> int:
    return len(_components(grid == MAROON))


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(*GRID_H_RANGE))
    w = int(rng.integers(*GRID_W_RANGE))
    grid = np.full((h, w), BG, dtype=int)
    path_taken = np.zeros((h, w), dtype=bool)

    scenario = int(rng.integers(0, 4))
    use_target = scenario >= 2
    require_mirror = (scenario % 2) == 1

    emitter_count = int(rng.integers(*EMITTER_COUNT_RANGE))
    if use_target:
        emitter_count = max(emitter_count, 2)
    if emitter_count > len(EMITTER_PALETTE):
        return None
    colors = list(rng.choice(EMITTER_PALETTE, size=emitter_count, replace=False))

    next_color = 0
    receiver_targets: list[tuple[int, int]] | None = None

    if use_target:
        receiver = _place_emitter(
            grid,
            path_taken,
            colors[next_color],
            direction_choices=["right"],
            turns_choices=[0],
            rng=rng,
            left_min=max(0, w // 2 - 1),
            left_max=w - 4,
        )
        if receiver is None:
            return None
        receiver_targets = [
            (int(receiver["top"]), int(receiver["left"])),
            (int(receiver["top"]) + 1, int(receiver["left"])),
        ]
        next_color += 1

        if require_mirror:
            if bool(rng.integers(0, 2)):
                source_dirs = ["up", "down"]
                source_turns = [1]
            else:
                source_dirs = ["left", "right"]
                source_turns = [2]
        else:
            source_dirs = ["right"]
            source_turns = [0]

        target_source = _place_emitter(
            grid,
            path_taken,
            colors[next_color],
            direction_choices=source_dirs,
            turns_choices=source_turns,
            rng=rng,
            targets=receiver_targets,
            left_max=int(receiver["left"]) - 4,
        )
        if target_source is None:
            return None
        next_color += 1

    boundary_dirs = ["up", "down", "left", "right"]
    mirrored_placed = require_mirror and use_target
    while next_color < emitter_count:
        if require_mirror and not mirrored_placed:
            turns_choices = [1, 2]
        elif require_mirror:
            turns_choices = [0, 1, 2]
        else:
            turns_choices = [0]
        placed = _place_emitter(
            grid,
            path_taken,
            colors[next_color],
            direction_choices=boundary_dirs,
            turns_choices=turns_choices,
            rng=rng,
        )
        if placed is None:
            return None
        mirrored_placed = mirrored_placed or int(placed["turns"]) > 0
        next_color += 1

    if np.count_nonzero(grid == GREY) != emitter_count:
        return None
    mirror_components = _count_mirror_components(grid)
    if require_mirror and mirror_components == 0:
        return None
    if not require_mirror and mirror_components != 0:
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(OUTER_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid emitter-mirror puzzle")
