import random

import numpy as np


BG_COLORS = (4, 7)
ANCHOR_COLORS = [0, 1, 2, 3, 5, 6, 8, 9]
DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

GRID_H_RANGE = (9, 18)
GRID_W_RANGE = (9, 18)
GENERATE_RETRIES = 400
GEN_MODES = ("open", "chain", "cross")


def _checkerboard(shape: tuple[int, int], even_color: int) -> np.ndarray:
    odd_color = BG_COLORS[1] if even_color == BG_COLORS[0] else BG_COLORS[0]
    rows, cols = np.indices(shape)
    return np.where(((rows + cols) & 1) == 0, even_color, odd_color).astype(int)


def _infer_checkerboard(grid: np.ndarray) -> np.ndarray:
    colors, counts = np.unique(grid, return_counts=True)
    if len(colors) < 2:
        raise ValueError("expected at least two colors")

    order = np.argsort(counts)[::-1]
    even_color = int(colors[order[0]])
    odd_color = int(colors[order[1]])

    rows, cols = np.indices(grid.shape)
    parity = (rows + cols) & 1
    score0 = int(
        np.count_nonzero(grid[parity == 0] == even_color)
        + np.count_nonzero(grid[parity == 1] == odd_color)
    )
    score1 = int(
        np.count_nonzero(grid[parity == 0] == odd_color)
        + np.count_nonzero(grid[parity == 1] == even_color)
    )
    if score1 > score0:
        even_color, odd_color = odd_color, even_color
    return np.where(parity == 0, even_color, odd_color).astype(int)


def _solve(grid: np.ndarray, with_stats: bool = False):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    expected_bg = _infer_checkerboard(g)

    anchor_cells = [tuple(map(int, coord)) for coord in np.argwhere(g != expected_bg)]
    anchor_cells.sort()
    out = g.copy()

    stats = {
        "anchor_count": len(anchor_cells),
        "blocked_rays": 0,
        "boundary_rays": 0,
        "zero_blocked_rays": 0,
        "positive_blocked_rays": 0,
        "painted_cells": 0,
        "max_ray_len": 0,
    }

    for r0, c0 in anchor_cells:
        color = int(g[r0, c0])
        for dr, dc in DIAG_DIRS:
            r = r0 + dr
            c = c0 + dc
            length = 0
            while 0 <= r < h and 0 <= c < w:
                if int(out[r, c]) != int(expected_bg[r, c]):
                    stats["blocked_rays"] += 1
                    if length == 0:
                        stats["zero_blocked_rays"] += 1
                    else:
                        stats["positive_blocked_rays"] += 1
                    break
                out[r, c] = color
                length += 1
                stats["painted_cells"] += 1
                stats["max_ray_len"] = max(stats["max_ray_len"], length)
                r += dr
                c += dc
            else:
                stats["boundary_rays"] += 1

    if with_stats:
        stats["non_bg_cells"] = stats["anchor_count"] + stats["painted_cells"]
        return out, stats
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Cast diagonal rays from each colored anchor across the checkerboard until earlier strokes block them.

    Input structure:
    - The background is a two-color checkerboard.
    - Several singleton colored anchors are scattered on cells of one checkerboard parity.
    - Every non-checkerboard pixel is an anchor; there are no multi-cell foreground objects.

    Rule:
    - Recover the checkerboard colors, then scan anchors from top to bottom and left to right.
    - For each anchor, extend four rays in the diagonal directions NW, NE, SW, and SE using the anchor color.
    - A ray may repaint only untouched checkerboard cells and stops before the first anchor or previously painted ray cell, or when it leaves the grid.
    - Keep each anchor cell itself unchanged.

    Key insight:
    - The network is order-sensitive: once an earlier anchor claims a diagonal cell, that occupied cell becomes a hard blocker for later anchors.
    """
    return _solve(grid, with_stats=False)


def _is_well_spaced(cell: tuple[int, int], anchors: list[tuple[int, int]]) -> bool:
    return all(
        max(abs(cell[0] - other[0]), abs(cell[1] - other[1])) >= 2 for other in anchors
    )


def _build_input(
    h: int, w: int, even_color: int, anchors: list[tuple[int, int]]
) -> np.ndarray:
    grid = _checkerboard((h, w), even_color)
    for (r, c), color in zip(anchors, random.sample(ANCHOR_COLORS, len(anchors))):
        grid[r, c] = color
    return grid


def _add_open_extras(
    h: int,
    w: int,
    parity: int,
    anchors: list[tuple[int, int]],
    total_count: int,
) -> list[tuple[int, int]] | None:
    cells = [(r, c) for r in range(h) for c in range(w) if ((r + c) & 1) == parity]
    random.shuffle(cells)

    out = list(anchors)
    used_diff = {r - c for r, c in anchors}
    used_sum = {r + c for r, c in anchors}

    for cell in cells:
        if len(out) == total_count:
            break
        diff = cell[0] - cell[1]
        diag_sum = cell[0] + cell[1]
        if diff in used_diff or diag_sum in used_sum:
            continue
        if not _is_well_spaced(cell, out):
            continue
        out.append(cell)
        used_diff.add(diff)
        used_sum.add(diag_sum)

    if len(out) == total_count:
        return out
    return None


def _generate_open_positions(h: int, w: int, parity: int) -> list[tuple[int, int]] | None:
    target = random.randint(2, 3)
    cells = [(r, c) for r in range(h) for c in range(w) if ((r + c) & 1) == parity]
    random.shuffle(cells)

    anchors: list[tuple[int, int]] = []
    used_diff: set[int] = set()
    used_sum: set[int] = set()

    for cell in cells:
        if not (1 <= cell[0] < h - 1 and 1 <= cell[1] < w - 1):
            continue
        diff = cell[0] - cell[1]
        diag_sum = cell[0] + cell[1]
        if diff in used_diff or diag_sum in used_sum:
            continue
        if not _is_well_spaced(cell, anchors):
            continue
        anchors.append(cell)
        used_diff.add(diff)
        used_sum.add(diag_sum)
        if len(anchors) == target:
            return anchors
    return None


def _line_cells(
    h: int, w: int, parity: int, kind: str, const: int
) -> list[tuple[int, int]]:
    cells = []
    for r in range(h):
        for c in range(w):
            if ((r + c) & 1) != parity:
                continue
            if kind == "diff" and r - c == const:
                cells.append((r, c))
            if kind == "sum" and r + c == const:
                cells.append((r, c))
    cells.sort()
    return cells


def _choose_spaced_on_line(
    cells: list[tuple[int, int]], count: int
) -> list[tuple[int, int]] | None:
    for _ in range(80):
        picks = sorted(random.sample(range(len(cells)), count))
        if all(right - left >= 2 for left, right in zip(picks, picks[1:])):
            return [cells[idx] for idx in picks]
    return None


def _generate_chain_positions(h: int, w: int, parity: int) -> list[tuple[int, int]] | None:
    total_count = random.randint(2, 4)
    chain_count = random.randint(2, min(3, total_count))
    kind = random.choice(("diff", "sum"))

    consts = list(range(-(w - 1), h)) if kind == "diff" else list(range(h + w - 1))
    random.shuffle(consts)

    chosen = None
    for const in consts:
        cells = [
            cell
            for cell in _line_cells(h, w, parity, kind, const)
            if 0 < cell[0] < h - 1 and 0 < cell[1] < w - 1
        ]
        if len(cells) < chain_count:
            continue
        chosen = _choose_spaced_on_line(cells, chain_count)
        if chosen is not None:
            break

    if chosen is None:
        return None
    return _add_open_extras(h, w, parity, chosen, total_count)


def _generate_cross_positions(h: int, w: int, parity: int) -> list[tuple[int, int]] | None:
    side = random.choice(("left", "right"))
    candidates = []
    for r in range(2, h - 2):
        for c in range(2, w - 2):
            if ((r + c) & 1) != parity:
                continue
            if side == "left":
                max_a = min(r, c)
                max_b = min(h - 1 - r, c)
            else:
                max_a = min(r, w - 1 - c)
                max_b = min(h - 1 - r, w - 1 - c)
            if max_a >= 2 and max_b >= 2:
                candidates.append((r, c, max_a, max_b))

    if not candidates:
        return None

    pr, pc, max_a, max_b = random.choice(candidates)
    dist_a = random.randint(2, min(max_a, 6))
    dist_b = random.randint(2, min(max_b, 6))

    if side == "left":
        anchors = [(pr - dist_a, pc - dist_a), (pr + dist_b, pc - dist_b)]
    else:
        anchors = [(pr - dist_a, pc + dist_a), (pr + dist_b, pc + dist_b)]

    if random.random() < 0.5:
        extra = _add_open_extras(h, w, parity, anchors, 3)
        if extra is not None:
            anchors = extra
    return anchors


def _accept_mode(mode: str, stats: dict[str, int], shape: tuple[int, int]) -> bool:
    h, w = shape
    density = stats["non_bg_cells"] / float(h * w)
    if stats["painted_cells"] < max(6, stats["anchor_count"] * 2):
        return False
    if density < 0.08 or density > 0.45:
        return False

    if mode == "open":
        return (
            stats["blocked_rays"] <= 2
            and stats["boundary_rays"] >= stats["blocked_rays"] + 4
        )
    if mode == "chain":
        return (
            stats["blocked_rays"] >= 2
            and stats["zero_blocked_rays"] >= 1
            and stats["boundary_rays"] >= 4
        )
    return stats["positive_blocked_rays"] >= 1 and stats["boundary_rays"] >= 4


def _generate_attempt() -> np.ndarray | None:
    mode = random.choice(GEN_MODES)
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    even_color = random.choice(BG_COLORS)
    parity = random.randint(0, 1)

    if mode == "open":
        anchors = _generate_open_positions(h, w, parity)
    elif mode == "chain":
        anchors = _generate_chain_positions(h, w, parity)
    else:
        anchors = _generate_cross_positions(h, w, parity)

    if anchors is None:
        return None

    grid = _build_input(h, w, even_color, anchors)
    _, stats = _solve(grid, with_stats=True)
    if not _accept_mode(mode, stats, (h, w)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid checkerboard-ray task")
