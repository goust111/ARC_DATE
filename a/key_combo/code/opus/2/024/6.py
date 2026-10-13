from __future__ import annotations

import itertools
import random

import numpy as np


BG = 0
SHADOW = 2
RECT_COLORS = (1, 4, 6, 7)
GRID_MIN = 7
GRID_MAX = 10
MAX_GENERATION_ATTEMPTS = 400


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _render_input(
    shape: tuple[int, int],
    rects: dict[int, tuple[int, int, int, int]],
    order: tuple[int, ...],
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for color in order:
        r0, c0, r1, c1 = rects[color]
        out[r0 : r1 + 1, c0 : c1 + 1] = int(color)
    return out


def _single_tile_with_shadow(
    visible: np.ndarray,
    rects: dict[int, tuple[int, int, int, int]],
) -> np.ndarray:
    out = np.array(visible, dtype=int, copy=True)
    H, W = out.shape
    for r0, c0, r1, c1 in rects.values():
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                rr = r - 1
                cc = c - 1
                if 0 <= rr < H and 0 <= cc < W and int(out[rr, cc]) == BG:
                    out[rr, cc] = SHADOW
    return out


def _tiled_output(
    visible: np.ndarray,
    rects: dict[int, tuple[int, int, int, int]],
) -> np.ndarray:
    H, W = map(int, visible.shape)
    out = np.zeros((3 * H, 3 * W), dtype=int)
    for ty in range(3):
        for tx in range(3):
            out[ty * H : (ty + 1) * H, tx * W : (tx + 1) * W] = visible

    for ty in range(3):
        for tx in range(3):
            roff = ty * H
            coff = tx * W
            for r0, c0, r1, c1 in rects.values():
                for r in range(r0, r1 + 1):
                    for c in range(c0, c1 + 1):
                        rr = roff + r - 1
                        cc = coff + c - 1
                        if 0 <= rr < 3 * H and 0 <= cc < 3 * W and int(out[rr, cc]) == BG:
                            out[rr, cc] = SHADOW
    return out


def _candidate_rectangles(grid: np.ndarray, color: int) -> list[tuple[int, int, int, int]]:
    coords = np.argwhere(grid == int(color))
    if coords.size == 0:
        return []
    H, W = map(int, grid.shape)
    vr0, vc0, vr1, vc1 = _bbox_coords(coords)
    candidates: set[tuple[int, int, int, int]] = set()

    for r0 in range(vr0 + 1):
        for c0 in range(vc0 + 1):
            for r1 in range(vr1, H):
                if r1 - r0 + 1 < 2:
                    continue
                for c1 in range(vc1, W):
                    if c1 - c0 + 1 < 2:
                        continue

                    view = grid[r0 : r1 + 1, c0 : c1 + 1]
                    if np.any(view == BG):
                        continue

                    # A recoverable rectangle must expose at least one full row or one full column.
                    if not (
                        np.any(np.all(view == int(color), axis=1))
                        or np.any(np.all(view == int(color), axis=0))
                    ):
                        continue

                    candidates.add((r0, c0, r1, c1))

    ranked = sorted(
        candidates,
        key=lambda b: (
            (b[2] - b[0] + 1) * (b[3] - b[1] + 1),
            b[0],
            b[1],
            b[2],
            b[3],
        ),
    )
    return ranked


def _valid_rect_sets(
    grid: np.ndarray,
) -> list[tuple[dict[int, tuple[int, int, int, int]], tuple[int, ...]]]:
    colors = sorted(int(c) for c in np.unique(grid) if int(c) != BG)
    assert colors and SHADOW not in colors
    candidates = {color: _candidate_rectangles(grid, color) for color in colors}
    if any(not cands for cands in candidates.values()):
        return []

    solutions: list[tuple[dict[int, tuple[int, int, int, int]], tuple[int, ...]]] = []
    seen_rect_sets: set[tuple[tuple[int, tuple[int, int, int, int]], ...]] = set()

    for choice in itertools.product(*(candidates[color] for color in colors)):
        rects = {color: rect for color, rect in zip(colors, choice)}
        rect_key = tuple(sorted(rects.items()))
        if rect_key in seen_rect_sets:
            continue

        for order in itertools.permutations(colors):
            if np.array_equal(_render_input(grid.shape, rects, order), grid):
                solutions.append((rects, order))
                seen_rect_sets.add(rect_key)
                break

    return solutions


def _recover_rectangles(grid: np.ndarray) -> dict[int, tuple[int, int, int, int]]:
    solutions = _valid_rect_sets(grid)
    rect_keys = {tuple(sorted(rects.items())) for rects, _order in solutions}
    assert len(rect_keys) == 1, "rectangle recovery must be unique"
    return dict(next(iter(rect_keys)))


def _visible_full_line(
    grid: np.ndarray,
    color: int,
    rect: tuple[int, int, int, int],
) -> bool:
    r0, c0, r1, c1 = rect
    for r in range(r0, r1 + 1):
        if np.all(grid[r, c0 : c1 + 1] == int(color)):
            return True
    for c in range(c0, c1 + 1):
        if np.all(grid[r0 : r1 + 1, c] == int(color)):
            return True
    return False


def _has_true_occlusion(
    rects: dict[int, tuple[int, int, int, int]],
    order: tuple[int, ...],
) -> bool:
    z_index = {color: idx for idx, color in enumerate(order)}
    colors = list(rects)
    for a, b in itertools.combinations(colors, 2):
        ar0, ac0, ar1, ac1 = rects[a]
        br0, bc0, br1, bc1 = rects[b]
        ir0 = max(ar0, br0)
        ic0 = max(ac0, bc0)
        ir1 = min(ar1, br1)
        ic1 = min(ac1, bc1)
        if ir0 <= ir1 and ic0 <= ic1 and z_index[a] != z_index[b]:
            return True
    return False


def _has_hidden_cells(
    rects: dict[int, tuple[int, int, int, int]],
    order: tuple[int, ...],
) -> bool:
    H = max(rect[2] for rect in rects.values()) + 1
    W = max(rect[3] for rect in rects.values()) + 1
    coverage = [[[] for _ in range(W)] for _ in range(H)]
    for color, rect in rects.items():
        r0, c0, r1, c1 = rect
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                coverage[r][c].append(color)

    z_index = {color: idx for idx, color in enumerate(order)}
    for r in range(H):
        for c in range(W):
            if len(coverage[r][c]) >= 2:
                top = max(coverage[r][c], key=lambda color: z_index[color])
                if any(color != top for color in coverage[r][c]):
                    return True
    return False


def _has_cross_tile_difference(
    visible: np.ndarray,
    rects: dict[int, tuple[int, int, int, int]],
) -> bool:
    naive = np.tile(_single_tile_with_shadow(visible, rects), (3, 3))
    full = _tiled_output(visible, rects)
    return not np.array_equal(naive, full)


def _has_cross_tile_conflict(
    visible: np.ndarray,
    rects: dict[int, tuple[int, int, int, int]],
) -> bool:
    H, W = map(int, visible.shape)
    rect_mask = visible != BG
    for r0, c0, r1, c1 in rects.values():
        for r in range(r0, r1 + 1):
            if c0 == 0 and r > 0 and rect_mask[r - 1, W - 1]:
                return True
        for c in range(c0, c1 + 1):
            if r0 == 0 and c > 0 and rect_mask[H - 1, c - 1]:
                return True
        if r0 == 0 and c0 == 0 and rect_mask[H - 1, W - 1]:
            return True
    return False


def _sample_rect(
    H: int,
    W: int,
    *,
    big: bool = False,
    edge_bias: bool = False,
) -> tuple[int, int, int, int]:
    max_h = min(H, 6 if big else 5)
    max_w = min(W, 6 if big else 5)
    min_h = 3 if big else 2
    min_w = 3 if big else 2
    h = random.randint(min_h, max_h)
    w = random.randint(min_w, max_w)
    r0 = random.randint(0, H - h)
    c0 = random.randint(0, W - w)

    if edge_bias:
        choices: list[tuple[int | None, int | None]] = [
            (0, None),
            (None, 0),
            (0, 0),
            (H - h, None),
            (None, W - w),
        ]
        rr, cc = random.choice(choices)
        if rr is not None:
            r0 = int(rr)
        if cc is not None:
            c0 = int(cc)

    return r0, c0, r0 + h - 1, c0 + w - 1


def _overlap_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    r0 = max(a[0], b[0])
    c0 = max(a[1], b[1])
    r1 = min(a[2], b[2])
    c1 = min(a[3], b[3])
    if r0 > r1 or c0 > c1:
        return 0
    return int((r1 - r0 + 1) * (c1 - c0 + 1))


def _visible_counts(grid: np.ndarray, color: int) -> tuple[int, int]:
    row_best = 0
    col_best = 0
    for r in range(grid.shape[0]):
        row_best = max(row_best, int(np.count_nonzero(grid[r] == int(color))))
    for c in range(grid.shape[1]):
        col_best = max(col_best, int(np.count_nonzero(grid[:, c] == int(color))))
    return row_best, col_best


def _semantic_ok(
    visible: np.ndarray,
    rects: dict[int, tuple[int, int, int, int]],
    order: tuple[int, ...],
) -> bool:
    if np.any(visible == SHADOW):
        return False
    if not _has_true_occlusion(rects, order):
        return False
    if not _has_hidden_cells(rects, order):
        return False
    if not _has_cross_tile_difference(visible, rects):
        return False

    solutions = _valid_rect_sets(visible)
    rect_keys = {tuple(sorted(found.items())) for found, _ in solutions}
    if len(rect_keys) != 1:
        return False
    recovered = dict(next(iter(rect_keys)))
    if recovered != rects:
        return False

    for color, rect in rects.items():
        if not _visible_full_line(visible, color, rect):
            return False

    occluded_colors = set()
    z_index = {color: idx for idx, color in enumerate(order)}
    for a, b in itertools.combinations(rects, 2):
        if _overlap_area(rects[a], rects[b]) > 0:
            if z_index[a] < z_index[b]:
                occluded_colors.add(a)
            else:
                occluded_colors.add(b)

    if not occluded_colors:
        return False
    for color in occluded_colors:
        row_best, col_best = _visible_counts(visible, color)
        h = rects[color][2] - rects[color][0] + 1
        w = rects[color][3] - rects[color][1] + 1
        if row_best > w or col_best > h:
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(GRID_MIN, GRID_MAX)
    W = random.randint(GRID_MIN, GRID_MAX)
    n_rects = random.choice((2, 3, 3))
    colors = random.sample(RECT_COLORS, n_rects)
    order = tuple(colors)

    rects: dict[int, tuple[int, int, int, int]] = {}
    base_h = random.randint(4, min(5, H - (n_rects - 1)))
    base_w = random.randint(4, min(5, W - (n_rects - 1)))
    anchor = random.choice(("corner", "corner", "top", "left"))
    if anchor == "corner":
        base_r0 = 0
        base_c0 = 0
    elif anchor == "top":
        base_r0 = 0
        base_c0 = random.randint(0, max(0, W - base_w - 2))
    else:
        base_r0 = random.randint(0, max(0, H - base_h - 2))
        base_c0 = 0
    rects[colors[0]] = (base_r0, base_c0, base_r0 + base_h - 1, base_c0 + base_w - 1)

    prev = rects[colors[0]]
    for idx, color in enumerate(colors[1:], start=1):
        placed = False
        for _ in range(120):
            pr0, pc0, pr1, pc1 = prev
            r0_min = pr0 + 1
            c0_min = pc0 + 1
            r0_max = min(pr1, H - 2)
            c0_max = min(pc1, W - 2)
            if r0_min > r0_max or c0_min > c0_max:
                return None

            r0 = random.randint(r0_min, r0_max)
            c0 = random.randint(c0_min, c0_max)
            extend_down = random.choice((True, False))
            extend_right = random.choice((True, False))
            if not extend_down and not extend_right:
                extend_down = True

            if idx == n_rects - 1 and random.random() < 0.65:
                extend_down = True
                extend_right = True

            if extend_down:
                r1_min = max(r0 + 1, pr1 + 1)
                r1_max = min(H - 1, r0 + 4)
            else:
                r1_min = r0 + 1
                r1_max = min(pr1, r0 + 4)
            if extend_right:
                c1_min = max(c0 + 1, pc1 + 1)
                c1_max = min(W - 1, c0 + 4)
            else:
                c1_min = c0 + 1
                c1_max = min(pc1, c0 + 4)

            if idx == n_rects - 1 and random.random() < 0.5:
                r1_max = H - 1
            if idx == n_rects - 1 and random.random() < 0.5:
                c1_max = W - 1

            if r1_min > r1_max or c1_min > c1_max:
                continue

            r1 = random.randint(r1_min, r1_max)
            c1 = random.randint(c1_min, c1_max)
            candidate = (r0, c0, r1, c1)

            overlap = _overlap_area(candidate, prev)
            if overlap == 0:
                continue
            candidate_area = (r1 - r0 + 1) * (c1 - c0 + 1)
            prev_area = (pr1 - pr0 + 1) * (pc1 - pc0 + 1)
            if overlap >= 0.75 * min(candidate_area, prev_area):
                continue

            rects[color] = candidate
            prev = candidate
            placed = True
            break
        if not placed:
            return None

    visible = _render_input((H, W), rects, order)

    # Keep scenes readable.
    colored = int(np.count_nonzero(visible))
    if colored > int(0.65 * H * W):
        return None

    if not _semantic_ok(visible, rects, order):
        return None

    return visible


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover each partially occluded solid rectangle, cast an up-left shadow from the recovered rectangles, then tile the result 3x3.

    Input structure:
    - The background is black.
    - The foreground consists of 2-3 differently colored solid rectangles chosen from blue, yellow, pink, and orange.
    - Rectangles may overlap, so a lower rectangle can be split into several visible same-color fragments.

    Rule:
    - Infer one complete axis-aligned rectangle for each visible non-black color.
    - Keep the input's visible rectangle colors as the tile's rectangle layer, but use the recovered full rectangles as shadow sources.
    - Every recovered rectangle pixel casts a red shadow to its up-left neighbor when that global tiled cell is still black.
    - Repeat the visible tile in a 3x3 grid; shadows may spill across tile boundaries, and rectangle pixels always override shadows.

    Key insight:
    - Same-color pieces belong to one underlying rectangle, so the solver must reconstruct full bboxes before it can place the shadow pattern correctly across tile boundaries.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 10
    colors = sorted(int(c) for c in np.unique(g) if int(c) != BG)
    assert 2 <= len(colors) <= 3
    assert SHADOW not in colors

    rects = _recover_rectangles(g)
    return _tiled_output(g, rects)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid occluded-rectangle scene")
