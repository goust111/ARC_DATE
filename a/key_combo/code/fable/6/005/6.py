from __future__ import annotations

import random
from typing import Dict, Tuple

import numpy as np


Coord = Tuple[int, int]


def _settle_particle(grid: np.ndarray, start: Coord, lookahead: int = 2) -> Coord:
    """Drop one water pixel through empty space, sliding toward the deepest reachable opening."""
    g = grid
    H, W = g.shape
    start_c = int(start[1])
    memo: dict[tuple[int, int, int], Coord] = {}

    def descend(r: int, c: int, depth: int) -> Coord:
        key = (int(r), int(c), int(depth))
        if key in memo:
            return memo[key]

        rr, cc = int(r), int(c)
        for _ in range(H * W * 4):
            if rr + 1 < H and int(g[rr + 1, cc]) == 0:
                rr += 1
                continue

            diagonal: list[Coord] = []
            if rr + 1 < H and cc - 1 >= 0 and int(g[rr, cc - 1]) == 0 and int(g[rr + 1, cc - 1]) == 0:
                diagonal.append((rr + 1, cc - 1))
            if rr + 1 < H and cc + 1 < W and int(g[rr, cc + 1]) == 0 and int(g[rr + 1, cc + 1]) == 0:
                diagonal.append((rr + 1, cc + 1))
            if diagonal:
                if len(diagonal) == 1 or depth <= 0:
                    rr, cc = min(diagonal, key=lambda p: (abs(int(p[1]) - cc), abs(int(p[1]) - start_c), int(p[1])))
                else:
                    scored = []
                    for nr, nc in diagonal:
                        tr, tc = descend(int(nr), int(nc), int(depth - 1))
                        scored.append(((-int(tr), abs(int(nc) - cc), abs(int(nc) - start_c), int(nc)), (int(nr), int(nc))))
                    rr, cc = min(scored)[1]
                continue

            if rr + 1 < H and int(g[rr + 1, cc]) != 0:
                left = int(cc)
                while left - 1 >= 0 and int(g[rr, left - 1]) == 0:
                    left -= 1
                right = int(cc)
                while right + 1 < W and int(g[rr, right + 1]) == 0:
                    right += 1

                drops = [x for x in range(left, right + 1) if int(g[rr + 1, x]) == 0]
                if not drops:
                    break

                def drop_score(x: int) -> tuple[int, int, int, int]:
                    if depth <= 0:
                        tr = rr
                    else:
                        tr, _tc = descend(rr, int(x), int(depth - 1))
                    return (-int(tr), abs(int(x) - cc), abs(int(x) - start_c), int(x))

                best_x = min(drops, key=drop_score)
                if best_x == cc:
                    break
                cc += -1 if best_x < cc else 1
                continue

            break

        memo[key] = (int(rr), int(cc))
        return memo[key]

    return descend(int(start[0]), int(start[1]), int(lookahead))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Simulate suspended blue water settling into orthogonally sealed basins while diagonal wall joints leak.

    Input structure:
    - Background is black 0 and water is blue 1, initially suspended in vertical source stacks.
    - Every other visible color is a static solid obstacle: floors, walls, steps, and basin rims.
    - Some candidate basins are truly sealed by orthogonally connected walls, while others contain a diagonal-only wall joint.

    Rule:
    - Remove the initial water from the canvas and drop each water pixel through background cells.
    - A falling pixel moves straight down while possible; when blocked below, it may slip diagonally through a corner gap or slide along a supported surface toward a reachable drop opening.
    - Orthogonally sealed basins retain water layer by layer; once such a basin is full, surplus water spills over the rim and continues downward.
    - A wall joint made from diagonal-only obstacle contact does not seal: water reaching that seam escapes through it and falls onward.

    Key insight:
    - The decisive question is not just where gravity points, but whether an apparent container is water-tight under 4-connected wall contact; diagonal joints turn a seeming basin into a leak.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    sources = sorted([(int(r), int(c)) for r, c in np.argwhere(out == 1)], reverse=True)
    out[out == 1] = 0

    for r, c in sources:
        rr, cc = _settle_particle(out, (int(r), int(c)))
        out[int(rr), int(cc)] = 1

    return out


def _build_candidate() -> tuple[np.ndarray, Dict[str, int]]:
    H = random.randint(17, 20)
    W = random.randint(24, 30)
    color = random.choice([2, 3, 4, 5, 6, 7, 8, 9])

    top = H - 8
    floor = H - 3
    left_x = random.randint(2, 4)
    right_x = random.randint(max(left_x + 11, W - 10), W - 9)
    leak_n = random.randint(4, min(7, top))
    spill_n = random.randint(9, min(11, top))

    g = np.zeros((H, W), dtype=int)
    g[H - 1, :] = color

    # Left basin: one diagonal joint on the left wall, so it leaks completely.
    g[top, left_x + 1] = color
    g[top + 1 : floor + 1, left_x] = color
    g[top : floor + 1, left_x + 5] = color
    g[floor, left_x : left_x + 6] = color
    for r in range(leak_n):
        g[r, left_x + 1] = 1

    # Right basin: fully sealed on the left wall, but its right rim is lower, so it can overflow once full.
    g[top : floor + 1, right_x] = color
    g[top + 3 : floor + 1, right_x + 5] = color
    g[floor, right_x : right_x + 6] = color
    for r in range(spill_n):
        g[r, right_x + 1] = 1

    meta = {
        "H": H,
        "W": W,
        "top": top,
        "floor": floor,
        "left_x": left_x,
        "right_x": right_x,
        "leak_n": leak_n,
        "spill_n": spill_n,
        "color": color,
    }
    return g, meta


def _seal_left_basin(grid: np.ndarray, meta: Dict[str, int]) -> np.ndarray:
    """Counterfactual: replace the diagonal joint by a continuous orthogonal wall."""
    g = np.array(grid, dtype=int, copy=True)
    top = int(meta["top"])
    floor = int(meta["floor"])
    left_x = int(meta["left_x"])
    color = int(meta["color"])
    g[top : floor + 1, left_x] = color
    g[top, left_x + 1] = 0
    return g


def _remove_one_spill_source(grid: np.ndarray, meta: Dict[str, int]) -> np.ndarray:
    """Counterfactual: lower the right source count by one to verify that overflow really depends on extra water."""
    g = np.array(grid, dtype=int, copy=True)
    source_c = int(meta["right_x"]) + 1
    rows = [int(r) for r in np.where(g[:, source_c] == 1)[0]]
    assert rows
    g[max(rows), source_c] = 0
    return g


def _semantic_checks(inp: np.ndarray, meta: Dict[str, int]) -> bool:
    top = int(meta["top"])
    floor = int(meta["floor"])
    left_x = int(meta["left_x"])
    right_x = int(meta["right_x"])
    leak_n = int(meta["leak_n"])
    spill_n = int(meta["spill_n"])

    out = main(inp)
    if int(np.count_nonzero(inp == 1)) != int(np.count_nonzero(out == 1)):
        return False

    # Left leaky basin should not retain water in the would-be interior.
    leaky_interior = int(np.count_nonzero(out[floor - 2 : floor, left_x + 1 : left_x + 5] == 1))
    if leaky_interior != 0:
        return False

    # Right spill basin should fill its 2x4 core completely and spill the rest.
    spill_interior = int(np.count_nonzero(out[floor - 2 : floor, right_x + 1 : right_x + 5] == 1))
    spill_outside = int(np.count_nonzero(out[floor + 1, right_x + 6 :] == 1))
    if spill_interior != 8:
        return False
    if spill_outside != spill_n - 8:
        return False

    # Counterfactual 1: sealing the diagonal seam must make the left basin retain more water.
    sealed_left = main(_seal_left_basin(inp, meta))
    sealed_left_interior = int(np.count_nonzero(sealed_left[floor - 2 : floor, left_x + 1 : left_x + 5] == 1))
    if sealed_left_interior < min(2, leak_n):
        return False
    if sealed_left_interior <= leaky_interior:
        return False

    # Counterfactual 2: removing one right-source pixel must reduce right-side overflow by exactly one.
    reduced = main(_remove_one_spill_source(inp, meta))
    reduced_spill_outside = int(np.count_nonzero(reduced[floor + 1, right_x + 6 :] == 1))
    if spill_outside - reduced_spill_outside != 1:
        return False

    # The two basins must produce visibly different fates.
    left_outside = int(np.count_nonzero(out[floor - 1 : floor + 2, : left_x + 1] == 1))
    if left_outside < leak_n:
        return False
    if not np.any(out != inp):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid, meta = _build_candidate()
        if not _semantic_checks(grid, meta):
            continue
        if random.random() < 0.5:
            grid = np.fliplr(grid)
        return grid
    raise RuntimeError("failed to generate a valid diagonal-leak ARC input")
