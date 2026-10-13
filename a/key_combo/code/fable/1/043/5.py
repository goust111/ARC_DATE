from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


DIRS8 = tuple(
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if not (dr == 0 and dc == 0)
)
PAIR_DIRS = ((0, 1), (1, 0), (1, 1), (1, -1))
BACKGROUND = 0
COLOR_POOL = tuple(range(1, 10))
MIN_SIZE = 14
MAX_SIZE = 28
MIN_CENTER = 2
MAX_GEN_ATTEMPTS = 250
SHORT_CHAIN_DIM = (12, 15)
LONG_CHAIN_DIM = (22, 28)


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _neighbors8(r: int, c: int):
    for dr, dc in DIRS8:
        yield int(r + dr), int(c + dc)


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_blocks(grid: np.ndarray) -> tuple[int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    blocks: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue
            color = int(g[r, c])
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((int(rr), int(cc)))
                for nr, nc in _neighbors8(rr, cc):
                    if not _in_bounds(g.shape, nr, nc):
                        continue
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    queue.append((nr, nc))

            pts = np.array(coords, dtype=int)
            r0, c0, r1, c1 = _bbox_coords(pts)
            assert pts.shape[0] == 9, "each object must contain exactly nine cells"
            assert r1 - r0 == 2 and c1 - c0 == 2, "each object must have a 3x3 bbox"
            assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == color), "each object must be a filled 3x3 square"
            blocks.append(
                {
                    "color": color,
                    "center": (int(r0 + 1), int(c0 + 1)),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                }
            )

    assert len(blocks) >= 2, "expected at least two filled 3x3 blocks"
    colors = [int(block["color"]) for block in blocks]
    assert len(colors) == len(set(colors)), "block colors must be distinct"
    blocks.sort(key=lambda block: (block["center"][0], block["center"][1], block["color"]))
    return bg, blocks


def _build_emitters(blocks: list[dict]) -> tuple[dict, list[dict], dict]:
    occupied: dict[tuple[int, int], int] = {}
    emitters: list[dict] = []
    seed_lookup: dict[tuple[int, int], dict] = {}

    for block_id, block in enumerate(blocks):
        cr, cc = block["center"]
        for rr in range(cr - 1, cr + 2):
            for col in range(cc - 1, cc + 2):
                coord = (int(rr), int(col))
                assert coord not in occupied, "blocks must not overlap"
                occupied[coord] = int(block_id)
        for dr, dc in DIRS8:
            seed = (int(cr + dr), int(cc + dc))
            emitter = {
                "id": len(emitters),
                "block": int(block_id),
                "color": int(block["color"]),
                "seed": seed,
                "dir": (int(dr), int(dc)),
            }
            assert seed not in seed_lookup, "emitter seeds must be unique"
            seed_lookup[seed] = emitter
            emitters.append(emitter)
    return occupied, emitters, seed_lookup


def _analyze_blocks(shape: tuple[int, int], blocks: list[dict]) -> dict:
    occupied, emitters, seed_lookup = _build_emitters(blocks)

    for emitter in emitters:
        sr, sc = emitter["seed"]
        dr, dc = emitter["dir"]
        opponent_id = None
        distance = None
        step = 1

        while True:
            rr = int(sr + step * dr)
            cc = int(sc + step * dc)
            if not _in_bounds(shape, rr, cc):
                break
            if (rr, cc) not in occupied:
                step += 1
                continue

            opponent = seed_lookup.get((rr, cc))
            assert opponent is not None, "a ray hit a block cell that is not a matching emitter"
            assert int(opponent["block"]) != int(emitter["block"]), "a ray cannot hit its own block"
            assert tuple(opponent["dir"]) == (-int(dr), -int(dc)), "rays can only stop at a facing emitter"
            opponent_id = int(opponent["id"])
            distance = int(step)
            break

        if opponent_id is None:
            paint: list[tuple[int, int]] = []
            step = 1
            while True:
                rr = int(sr + step * dr)
                cc = int(sc + step * dc)
                if not _in_bounds(shape, rr, cc):
                    break
                assert (rr, cc) not in occupied, "unopposed rays may only paint background"
                paint.append((rr, cc))
                step += 1
        else:
            gap = int(distance - 1)
            assert gap % 2 == 0, "facing emitters must have an even number of empty cells between them"
            half = gap // 2
            paint = [(int(sr + k * dr), int(sc + k * dc)) for k in range(1, half + 1)]

        emitter["opponent"] = opponent_id
        emitter["paint"] = tuple(paint)

    pair_ids: set[tuple[int, int]] = set()
    painted_cells: dict[tuple[int, int], int] = {}
    for emitter in emitters:
        opponent_id = emitter["opponent"]
        if opponent_id is not None:
            assert int(emitters[opponent_id]["opponent"]) == int(emitter["id"]), "facing relation must be symmetric"
            pair_ids.add(tuple(sorted((int(emitter["id"]), int(opponent_id)))))
        for cell in emitter["paint"]:
            assert cell not in occupied, "rays may not paint over block cells"
            prev = painted_cells.get(cell)
            assert prev is None, "distinct ray paths must not intersect"
            painted_cells[cell] = int(emitter["id"])

    return {"emitters": emitters, "pairs": pair_ids}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend the eight edge cells of every filled 3x3 block as rays, splitting any direct facing gap evenly.

    Input structure:
    - The grid has a dominant background color and at least two solid 3x3 monochrome blocks.
    - Different blocks use different colors, and each block's eight non-center cells act as ray seeds.
    - Any possible blocker for a ray is another seed on the same row, column, or diagonal with the opposite direction.

    Rule:
    - Keep every original block unchanged.
    - For each non-center cell, continue its color away from the block center along that compass direction.
    - If the first non-background cell in front is a facing seed from another block, paint only the near half of the background gap and stop; otherwise paint to the grid edge.

    Key insight:
    - A ray only cares about the first occupied cell on its own line: a true opposite seed causes a midpoint split, and every other valid ray is unopposed all the way to the border.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg, blocks = _find_blocks(g)
    analysis = _analyze_blocks(g.shape, blocks)

    out = g.copy()
    for emitter in analysis["emitters"]:
        color = int(emitter["color"])
        for r, c in emitter["paint"]:
            assert int(out[r, c]) == bg, "rays may only add color onto background cells"
            out[r, c] = color
    return out


def _render_input(shape: tuple[int, int], blocks: list[dict]) -> np.ndarray:
    grid = np.full(shape, BACKGROUND, dtype=int)
    for block in blocks:
        cr, cc = block["center"]
        color = int(block["color"])
        grid[cr - 1 : cr + 2, cc - 1 : cc + 2] = color
    return grid


def _sample_pair(shape: tuple[int, int]) -> list[tuple[int, int]] | None:
    h, w = map(int, shape)
    directions = list(PAIR_DIRS)
    random.shuffle(directions)

    for dr, dc in directions:
        span_r = (h - 1 - MIN_CENTER) - MIN_CENTER if dr != 0 else 10**9
        span_c = (w - 1 - MIN_CENTER) - MIN_CENTER if dc != 0 else 10**9
        max_total = min(span_r, span_c)
        distance_options = [d for d in range(5, int(max_total) + 1) if d % 2 == 1]
        if not distance_options:
            continue
        total = random.choice(distance_options)
        r_min = MIN_CENTER + max(0, -dr * total)
        r_max = (h - 1 - MIN_CENTER) - max(0, dr * total)
        c_min = MIN_CENTER + max(0, -dc * total)
        c_max = (w - 1 - MIN_CENTER) - max(0, dc * total)
        if r_min > r_max or c_min > c_max:
            continue
        start = (random.randint(r_min, r_max), random.randint(c_min, c_max))
        return [start, (int(start[0] + dr * total), int(start[1] + dc * total))]
    return None


def _sample_three_chain(shape: tuple[int, int], horizontal: bool) -> list[tuple[int, int]] | None:
    h, w = map(int, shape)
    if horizontal:
        max_span = w - 1 - 2 * MIN_CENTER
        total_options = [t for t in range(max(10, h + 1), max_span + 1) if t % 2 == 0]
        if not total_options:
            return None
        total = random.choice(total_options)
        step1_options = [d for d in range(5, total - 4) if d % 2 == 1]
        if not step1_options:
            return None
        step1 = random.choice(step1_options)
        cross_step = (total - 2) // 2
        row_min = max(MIN_CENTER, h - 1 - cross_step)
        row_max = min(h - 1 - MIN_CENTER, cross_step)
        if row_min > row_max:
            return None
        row = random.randint(row_min, row_max)
        start_col = random.randint(MIN_CENTER, w - 1 - MIN_CENTER - total)
        return [
            (int(row), int(start_col)),
            (int(row), int(start_col + step1)),
            (int(row), int(start_col + total)),
        ]

    max_span = h - 1 - 2 * MIN_CENTER
    total_options = [t for t in range(max(10, w + 1), max_span + 1) if t % 2 == 0]
    if not total_options:
        return None
    total = random.choice(total_options)
    step1_options = [d for d in range(5, total - 4) if d % 2 == 1]
    if not step1_options:
        return None
    step1 = random.choice(step1_options)
    cross_step = (total - 2) // 2
    col_min = max(MIN_CENTER, w - 1 - cross_step)
    col_max = min(w - 1 - MIN_CENTER, cross_step)
    if col_min > col_max:
        return None
    col = random.randint(col_min, col_max)
    start_row = random.randint(MIN_CENTER, h - 1 - MIN_CENTER - total)
    return [
        (int(start_row), int(col)),
        (int(start_row + step1), int(col)),
        (int(start_row + total), int(col)),
    ]


def _gen_attempt() -> np.ndarray | None:
    block_count = random.choice((2, 3))
    if block_count == 2:
        h = random.randint(MIN_SIZE, MAX_SIZE)
        w = random.randint(MIN_SIZE, MAX_SIZE)
        centers = _sample_pair((h, w))
        if centers is None:
            return None
    else:
        horizontal = random.random() < 0.5
        if horizontal:
            h = random.randint(*SHORT_CHAIN_DIM)
            w = random.randint(*LONG_CHAIN_DIM)
        else:
            h = random.randint(*LONG_CHAIN_DIM)
            w = random.randint(*SHORT_CHAIN_DIM)
        centers = _sample_three_chain((h, w), horizontal)
        if centers is None:
            return None

    colors = random.sample(COLOR_POOL, block_count)
    blocks = [{"center": center, "color": int(color)} for center, color in zip(centers, colors)]

    try:
        analysis = _analyze_blocks((h, w), blocks)
    except AssertionError:
        return None
    if not analysis["pairs"]:
        return None

    if not any(emitter["opponent"] is None for emitter in analysis["emitters"]):
        return None

    grid = _render_input((h, w), blocks)
    assert grid.ndim == 2 and max(grid.shape) <= 30
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GEN_ATTEMPTS} attempts")
