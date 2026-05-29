# Variant of 1818057f
# Dimension: D9 — 扫描方向和处理顺序 (multi_step)
# Change: 从上到下、从左到右扫描内部像素(r in 1..H-2, c in 1..W-2)，基于原始输入判断'+'，一次性完成所有标记
#         → 逐个'+'依次处理，后续检测基于已修改网格

from __future__ import annotations

import numpy as np

from arc_utils import get_bg


# Design analysis for the D9 variant
# ----------------------------------
# a) Concretization and behavior trace
#    - Keep the original palette and local pattern test: bg is the most common color,
#      there is one fg color, and a plus center is a fg cell with 4 orthogonal fg
#      neighbors.
#    - The new behavior is not "find all centers first". Instead the scan itself is
#      the process: when the first visible plus center is encountered, its 5 cells are
#      recolored to teal immediately, so later tests see teal rather than fg.
#    - Concrete example: a solid 3x4 yellow rectangle contains two candidate centers
#      on its middle row. Original rule colors both crosses because both centers are
#      tested against the untouched input. New rule colors only the left one: after it
#      becomes teal, the right candidate loses a yellow neighbor and no longer matches.
# b) Complexity calibration
#    - If the generator kept isolated non-overlapping 5-cell pluses, the new rule would
#      look identical to the original. The generator therefore must create components
#      with multiple nearby candidate centers, but keep the repertoire small enough for
#      humans to infer: mostly 3xL and Lx3 fg corridors plus a few isolated distractors.
# c) Hidden assumptions in the original main()
#    - `out = g.copy()` is still fine, but the old code read membership and neighbors
#      from `g`. That relied on simultaneity. Under D9 this is wrong: both the center
#      check and neighbor check must read from the evolving grid.
#    - The old one-pass scan order was observationally irrelevant. Under D9 it becomes
#      the rule itself, so we must preserve a fixed top-to-bottom / left-to-right order
#      and avoid precomputing centers.
# d) Placement legality
#    - Order-sensitive components are full 3xL or Lx3 rectangles of fg. Their candidate
#      centers stay one cell away from the component boundary, so every tested plus has
#      its full 5-cell footprint inside the grid.
#    - Components and distractors are separated by a 1-cell moat so they do not merge
#      and accidentally create extra plus centers across shapes.
# e) Visual observability
#    - A human can see that a longer corridor contains several local plus opportunities,
#      yet only the topmost or leftmost one turns teal. Including both horizontal and
#      vertical corridors exposes both parts of the scan order.
# f) Visual cleanliness
#    - Distractors are small isolated shapes that never touch the order-sensitive blobs.
#      This keeps the "why this one turned teal first" relation legible.
# g) Color inferability
#    - Colors remain exactly the original trio: red bg, yellow fg, teal output marker.
#      The only new signal is positional/temporal, which is visible through which yellow
#      candidate survives inside a corridor.
# h) Generator-side non-degeneracy metric
#    - For every generated input we compare the number of executed sequential plus events
#      with the number of simultaneous plus centers on the same input. We require the
#      sequential count to be strictly smaller so the sample cannot collapse back to the
#      original task behavior.
# i) New solver assumptions and generator guarantees
#    - The solver assumes the order-sensitive structures are encoded purely by fg
#      adjacency; the generator guarantees this by using only one fg color and isolated
#      components. The solver also assumes teal is not considered fg for later matches;
#      the generator depends on exactly that inhibition effect.

PLUS5 = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))
ORTHO4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _is_plus_center(grid: np.ndarray, r: int, c: int, fg: int) -> bool:
    return int(grid[r, c]) == fg and all(int(grid[r + dr, c + dc]) == fg for dr, dc in ORTHO4)


def _sequential_recolor(grid: np.ndarray, fg: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    out = np.array(grid, dtype=int).copy()
    H, W = out.shape
    centers: list[tuple[int, int]] = []
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if _is_plus_center(out, r, c, fg):
                centers.append((r, c))
                for dr, dc in PLUS5:
                    out[r + dr, c + dc] = 8
    return out, centers


def _simultaneous_recolor(grid: np.ndarray, fg: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    g = np.array(grid, dtype=int)
    out = g.copy()
    H, W = g.shape
    centers: list[tuple[int, int]] = []
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if _is_plus_center(g, r, c, fg):
                centers.append((r, c))
                for dr, dc in PLUS5:
                    out[r + dr, c + dc] = 8
    return out, centers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor fg '+' motifs, but consume them in scan order so earlier hits can suppress later ones.

    Input structure:
    - bg is the most common color.
    - One foreground color forms mixed yellow components on the red background; some components contain several nearby local '+' opportunities.

    Rule:
    - Scan interior cells from top to bottom and left to right.
    - Whenever the current cell and its 4 orthogonal neighbors are still foreground in the current working grid, recolor that 5-cell '+' to teal immediately.
    - Continue scanning on the already modified grid; cells turned teal are no longer foreground for later checks.

    Key insight:
    - This is an in-place process rather than a simultaneous one. In corridor-like blobs with multiple overlapping '+' candidates, only the candidates that appear earliest in scan order survive to be recolored.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    colors = sorted(set(map(int, g.flatten())) - {bg})
    if not colors:
        return out
    fg = int(colors[0])

    out, _ = _sequential_recolor(out, fg)
    return out


def _mark_reserved(reserved: np.ndarray, cells: list[tuple[int, int]]) -> None:
    H, W = reserved.shape
    for r, c in cells:
        for nr in range(max(0, r - 1), min(H, r + 2)):
            for nc in range(max(0, c - 1), min(W, c + 2)):
                reserved[nr, nc] = True


def _place_shape(
    grid: np.ndarray,
    reserved: np.ndarray,
    shape: list[tuple[int, int]],
    fg: int,
    rng: np.random.Generator,
    *,
    attempts: int = 80,
) -> bool:
    H, W = grid.shape
    h = 1 + max(r for r, _ in shape)
    w = 1 + max(c for _, c in shape)
    if h > H or w > W:
        return False

    for _ in range(attempts):
        r0 = int(rng.integers(0, H - h + 1))
        c0 = int(rng.integers(0, W - w + 1))
        cells = [(r0 + dr, c0 + dc) for dr, dc in shape]
        if any(reserved[r, c] for r, c in cells):
            continue
        for r, c in cells:
            grid[r, c] = fg
        _mark_reserved(reserved, cells)
        return True
    return False


def _rect_shape(h: int, w: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(h) for c in range(w)]


def _random_order_shape(H: int, W: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    if rng.random() < 0.5:
        length = int(rng.integers(4, min(8, W) + 1))
        return _rect_shape(3, length)
    length = int(rng.integers(4, min(8, H) + 1))
    return _rect_shape(length, 3)


DISTRACTOR_SHAPES = [
    [(0, 0)],
    [(0, 0), (0, 1)],
    [(0, 0), (1, 0)],
    [(0, 0), (0, 1), (0, 2)],
    [(0, 0), (1, 0), (2, 0)],
    [(0, 0), (0, 1), (1, 0)],
    [(0, 0), (1, 0), (1, 1)],
    [(0, 0), (0, 1), (1, 0), (1, 1)],
    [(0, 0), (0, 1), (0, 2), (0, 3)],
    [(0, 0), (1, 0), (2, 0), (3, 0)],
]


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    H = int(rng.integers(10, 23))
    W = int(rng.integers(10, 23))
    bg, fg = 2, 4

    grid = np.full((H, W), bg, dtype=int)
    reserved = np.zeros((H, W), dtype=bool)
    total = H * W
    target_density = float(rng.uniform(0.26, 0.41))
    target_fg = int(total * target_density)

    mandatory_shapes = [
        _rect_shape(3, int(rng.integers(4, min(8, W) + 1))),
        _rect_shape(int(rng.integers(4, min(8, H) + 1)), 3),
    ]
    for shape in mandatory_shapes:
        if not _place_shape(grid, reserved, shape, fg, rng):
            return None

    extra_order = 1 + int(total > 220) + int(rng.random() < 0.5)
    for _ in range(extra_order):
        if rng.random() < 0.75:
            _place_shape(grid, reserved, _random_order_shape(H, W, rng), fg, rng, attempts=40)

    current_fg = int(np.sum(grid == fg))
    fill_attempts = 0
    while current_fg < target_fg and fill_attempts < 500:
        fill_attempts += 1
        if target_fg - current_fg >= 10 and rng.random() < 0.25:
            placed = _place_shape(grid, reserved, _random_order_shape(H, W, rng), fg, rng, attempts=30)
        else:
            shape = DISTRACTOR_SHAPES[int(rng.integers(0, len(DISTRACTOR_SHAPES)))]
            placed = _place_shape(grid, reserved, shape, fg, rng, attempts=30)
        if placed:
            current_fg = int(np.sum(grid == fg))

    density = current_fg / total
    if density < 0.26 or density > 0.41:
        return None
    if current_fg >= total // 2:
        return None

    seq_out, seq_centers = _sequential_recolor(grid, fg)
    sim_out, sim_centers = _simultaneous_recolor(grid, fg)
    if len(seq_centers) < 2:
        return None
    if len(sim_centers) <= len(seq_centers):
        return None
    if np.array_equal(seq_out, sim_out):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 400 attempts")
