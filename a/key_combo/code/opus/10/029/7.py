from __future__ import annotations

from collections import deque
from itertools import combinations
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3

TEMPLATE_SIZE = 3
GRID_SIZE = TEMPLATE_SIZE * TEMPLATE_SIZE
MAX_RETRIES = 300

_CANONICAL_SILENT = (0, 0)
_CANONICAL_BLOCKERS = ((0, 1), (1, 0))


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    dst[r0 : r0 + h, c0 : c0 + w] = patch


def _flood_mask(template: np.ndarray, seed: tuple[int, int], *, pass_black: bool) -> np.ndarray:
    h, w = template.shape
    seen = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque([seed])
    seen[seed] = True

    while q:
        r, c = q.popleft()
        for rr, cc in neighbors4(r, c):
            if not in_bounds((h, w), rr, cc) or seen[rr, cc]:
                continue
            value = int(template[rr, cc])
            if value == RED:
                continue
            if not pass_black and value == BLACK:
                continue
            seen[rr, cc] = True
            q.append((rr, cc))
    return seen


def _active_stamp(template: np.ndarray) -> np.ndarray:
    seeds = np.argwhere(template == BLUE)
    assert len(seeds) == 1, "expected exactly one blue seed inside the template"
    seed = tuple(map(int, seeds[0]))
    reachable = _flood_mask(template, seed, pass_black=True)
    active = reachable & ((template == GREEN) | (template == BLUE))
    return np.where(active, GREEN, BLACK).astype(int)


def _expand_stamp(stamp: np.ndarray) -> np.ndarray:
    k = int(stamp.shape[0])
    out = np.zeros((k * k, k * k), dtype=int)
    for br, bc in np.argwhere(stamp == GREEN):
        paste(out, stamp, int(br) * k, int(bc) * k)
    return out


def _connected(coords: set[tuple[int, int]]) -> bool:
    start = next(iter(coords))
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for rr, cc in neighbors4(r, c):
            if (rr, cc) in coords and (rr, cc) not in seen:
                seen.add((rr, cc))
                q.append((rr, cc))
    return seen == coords


def _rotate_coord(coord: tuple[int, int], k: int, turns: int) -> tuple[int, int]:
    r, c = coord
    for _ in range(turns % 4):
        r, c = c, k - 1 - r
    return r, c


def _make_template(option: dict[str, object], turns: int) -> np.ndarray:
    template = np.zeros((TEMPLATE_SIZE, TEMPLATE_SIZE), dtype=int)
    greens = option["greens"]
    seed = option["seed"]

    for coord in greens:
        rr, cc = _rotate_coord(coord, TEMPLATE_SIZE, turns)
        template[rr, cc] = GREEN
    for coord in _CANONICAL_BLOCKERS:
        rr, cc = _rotate_coord(coord, TEMPLATE_SIZE, turns)
        template[rr, cc] = RED

    sr, sc = _rotate_coord(seed, TEMPLATE_SIZE, turns)
    template[sr, sc] = BLUE
    return template


def _naive_no_flood_output(grid: np.ndarray) -> np.ndarray:
    box = bbox(grid != BLACK)
    assert box is not None
    template = crop_bbox(grid, box)
    stamp = np.where((template == GREEN) | (template == BLUE), GREEN, BLACK).astype(int)
    return _expand_stamp(stamp)


def _green_only_output(grid: np.ndarray) -> np.ndarray:
    box = bbox(grid != BLACK)
    assert box is not None
    template = crop_bbox(grid, box)
    seed = tuple(map(int, np.argwhere(template == BLUE)[0]))
    reachable = _flood_mask(template, seed, pass_black=False)
    active = reachable & ((template == GREEN) | (template == BLUE))
    stamp = np.where(active, GREEN, BLACK).astype(int)
    return _expand_stamp(stamp)


def _build_canonical_options() -> list[dict[str, object]]:
    cells = [(r, c) for r in range(TEMPLATE_SIZE) for c in range(TEMPLATE_SIZE)]
    core = {_CANONICAL_SILENT, *_CANONICAL_BLOCKERS}
    options: list[dict[str, object]] = []

    for extra in combinations([p for p in cells if p not in core], 3):
        greens = core | set(extra)
        rows = {r for r, _ in greens}
        cols = {c for _, c in greens}
        if rows != {0, 1, 2} or cols != {0, 1, 2}:
            continue
        if not _connected(greens):
            continue

        for seed in sorted(greens - core):
            template = np.zeros((TEMPLATE_SIZE, TEMPLATE_SIZE), dtype=int)
            for r, c in greens:
                template[r, c] = GREEN
            for r, c in _CANONICAL_BLOCKERS:
                template[r, c] = RED
            template[seed] = BLUE

            stamp = _active_stamp(template)
            active_coords = {tuple(map(int, rc)) for rc in np.argwhere(stamp == GREEN)}
            remaining_green = greens - set(_CANONICAL_BLOCKERS) - {seed}
            if _CANONICAL_SILENT not in remaining_green:
                continue
            if _CANONICAL_SILENT in active_coords:
                continue
            if not any(nb in remaining_green for nb in neighbors4(*seed)):
                continue
            if len(active_coords) < 3:
                continue

            green_only_reach = _flood_mask(template, seed, pass_black=False)
            uses_black = any(
                coord in active_coords and not green_only_reach[coord]
                for coord in remaining_green
            )

            options.append(
                {
                    "greens": tuple(sorted(greens)),
                    "seed": seed,
                    "uses_black": uses_black,
                }
            )

    assert options, "failed to build generator options"
    assert any(option["uses_black"] for option in options)
    assert any(not option["uses_black"] for option in options)
    return options


_CANONICAL_OPTIONS = _build_canonical_options()
_OPTIONS_BY_BLACK_PATH = {
    False: [option for option in _CANONICAL_OPTIONS if not option["uses_black"]],
    True: [option for option in _CANONICAL_OPTIONS if option["uses_black"]],
}


def main(grid: np.ndarray) -> np.ndarray:
    """Run template-local reachability, then self-expand only the surviving foreground pixels.

    Input structure:
    - A square N x N grid (with N = k^2) has black background and a single non-black k x k motif bbox.
    - Inside that bbox, green cells are the original foreground, one blue cell is the unique seed, and red cells are blockers replacing original foreground positions.

    Rule:
    - Crop the k x k bbox, then flood-fill inside it from the blue seed using 4-neighbor moves through green and black cells while red cells block movement.
    - Build a filtered stamp where the seed and every reachable green become green, while unreachable greens and all red blockers become black.
    - Create an N x N black output and paste that filtered stamp into each k x k block whose local template coordinate is green in the filtered stamp.

    Key insight:
    - Reachability is computed inside the template itself, so blockers simultaneously erase stamp content and erase the block placements that those silent foreground cells would have triggered.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] == g.shape[1], "expected a square 2D grid"

    box = bbox(g != BLACK)
    assert box is not None, "expected a non-black template"
    template = crop_bbox(g, box)
    k = int(template.shape[0])
    assert template.shape == (k, k), "expected a square template bbox"
    assert g.shape == (k * k, k * k), "expected full grid size to equal template_size^2"

    stamp = _active_stamp(template)
    return _expand_stamp(stamp)


def _validate_generated_input(grid: np.ndarray, *, expect_black_path: bool) -> bool:
    if not isinstance(grid, np.ndarray) or grid.shape != (GRID_SIZE, GRID_SIZE):
        return False
    if set(map(int, np.unique(grid))) - {BLACK, BLUE, RED, GREEN}:
        return False
    if int(np.sum(grid == BLUE)) != 1 or int(np.sum(grid == RED)) != 2:
        return False

    box = bbox(grid != BLACK)
    if box is None or box[2] - box[0] != TEMPLATE_SIZE - 1 or box[3] - box[1] != TEMPLATE_SIZE - 1:
        return False

    template = crop_bbox(grid, box)
    if template.shape != (TEMPLATE_SIZE, TEMPLATE_SIZE):
        return False

    stamp = _active_stamp(template)
    active_count = int(np.sum(stamp == GREEN))
    if active_count < 3:
        return False
    visible_foreground = int(np.sum((template == GREEN) | (template == BLUE)))
    if visible_foreground <= active_count:
        return False

    out = main(grid)
    if not np.array_equal(out, _expand_stamp(stamp)):
        return False
    if np.array_equal(out, _naive_no_flood_output(grid)):
        return False
    if expect_black_path and np.array_equal(out, _green_only_output(grid)):
        return False

    nonempty_blocks = 0
    for br in range(TEMPLATE_SIZE):
        for bc in range(TEMPLATE_SIZE):
            block = out[br * TEMPLATE_SIZE : (br + 1) * TEMPLATE_SIZE, bc * TEMPLATE_SIZE : (bc + 1) * TEMPLATE_SIZE]
            if np.any(block != BLACK):
                nonempty_blocks += 1
                if not np.array_equal(block, stamp):
                    return False
            elif stamp[br, bc] == GREEN:
                return False
    return nonempty_blocks == active_count


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        expect_black_path = bool(random.getrandbits(1))
        option = random.choice(_OPTIONS_BY_BLACK_PATH[expect_black_path])
        turns = random.randrange(4)
        template = _make_template(option, turns)

        grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        r0 = random.randint(0, GRID_SIZE - TEMPLATE_SIZE)
        c0 = random.randint(0, GRID_SIZE - TEMPLATE_SIZE)
        paste(grid, template, r0, c0)

        if _validate_generated_input(grid, expect_black_path=expect_black_path):
            return grid

    raise RuntimeError("failed to generate a valid input")
