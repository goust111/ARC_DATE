from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5

SCALE = 3
FG_COLORS = (RED, GRAY)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIRS8 = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)
GENERATION_RETRIES = 500

_BG_BLOCK = np.zeros((SCALE, SCALE), dtype=int)
_GRAY_BLOCK = np.full((SCALE, SCALE), GRAY, dtype=int)
_GRAY_BLOCK[1, 1] = GREEN
_RED_BLOCK = np.full((SCALE, SCALE), RED, dtype=int)
_RED_BLOCK[1, 1] = BLUE


def _base_block(color: int) -> np.ndarray:
    if color == BG:
        return _BG_BLOCK
    if color == GRAY:
        return _GRAY_BLOCK
    if color == RED:
        return _RED_BLOCK
    raise AssertionError(f"unsupported color {color}")


def main(grid: np.ndarray) -> np.ndarray:
    """Scale each foreground pixel into a 3x3 ring block and repaint same-color shared borders yellow.

    Input structure:
    - The input is a small black-background grid whose cells use only black, red, and gray.
    - Red and gray pixels form sparse foreground patterns; some same-color pixels are 4-neighbor adjacent and some are isolated.

    Rule:
    - Replace every input pixel by a 3x3 output block: black stays all black, gray becomes a gray ring with a green center, and red becomes a red ring with a blue center.
    - For every horizontal pair of same-color 4-neighbor foreground pixels, repaint the left block's rightmost column and the right block's leftmost column yellow.
    - For every vertical pair of same-color 4-neighbor foreground pixels, repaint the upper block's bottom row and the lower block's top row yellow.
    - Yellow bridge repainting overrides the base ring pattern, while different-color adjacencies do nothing.

    Key insight:
    - The 3x3 block for a pixel is not fixed by color alone: same-color 4-neighbor links in the original grid overwrite the enlarged blocks' shared borders and reveal the input adjacency graph.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert set(map(int, np.unique(g))) <= {BG, RED, GRAY}, "input colors must be black, red, or gray"

    h, w = g.shape
    out = np.zeros((h * SCALE, w * SCALE), dtype=int)

    for r in range(h):
        for c in range(w):
            block = _base_block(int(g[r, c]))
            r0 = r * SCALE
            c0 = c * SCALE
            out[r0 : r0 + SCALE, c0 : c0 + SCALE] = block

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color not in FG_COLORS:
                continue

            r0 = r * SCALE
            c0 = c * SCALE

            if c > 0 and int(g[r, c - 1]) == color:
                out[r0 : r0 + SCALE, c0] = YELLOW
            if c + 1 < w and int(g[r, c + 1]) == color:
                out[r0 : r0 + SCALE, c0 + SCALE - 1] = YELLOW
            if r > 0 and int(g[r - 1, c]) == color:
                out[r0, c0 : c0 + SCALE] = YELLOW
            if r + 1 < h and int(g[r + 1, c]) == color:
                out[r0 + SCALE - 1, c0 : c0 + SCALE] = YELLOW

    return out


def _line_patch(length: int, orientation: str, color: int) -> np.ndarray:
    if orientation == "h":
        return np.full((1, length), color, dtype=int)
    if orientation == "v":
        return np.full((length, 1), color, dtype=int)
    raise AssertionError(f"unsupported orientation {orientation}")


def _pair_patch(orientation: str, first: int, second: int) -> np.ndarray:
    if orientation == "h":
        return np.array([[first, second]], dtype=int)
    if orientation == "v":
        return np.array([[first], [second]], dtype=int)
    raise AssertionError(f"unsupported orientation {orientation}")


def _single_patch(color: int) -> np.ndarray:
    return np.array([[color]], dtype=int)


def _l_patch(color: int, rotation: int) -> np.ndarray:
    patch = np.array([[color, color], [color, BG]], dtype=int)
    return np.rot90(patch, k=rotation)


def _plus_patch(color: int) -> np.ndarray:
    patch = np.zeros((3, 3), dtype=int)
    patch[1, :] = color
    patch[:, 1] = color
    return patch


def _random_patch_of_size(size: int) -> np.ndarray:
    if size == 1:
        return _single_patch(random.choice(FG_COLORS))
    if size == 2:
        if random.random() < 0.35:
            first, second = random.sample(FG_COLORS, 2)
            return _pair_patch(random.choice(("h", "v")), first, second)
        return _line_patch(2, random.choice(("h", "v")), random.choice(FG_COLORS))
    if size == 3:
        if random.random() < 0.55:
            return _line_patch(3, random.choice(("h", "v")), random.choice(FG_COLORS))
        return _l_patch(random.choice(FG_COLORS), random.randrange(4))
    if size == 5:
        return _plus_patch(random.choice(FG_COLORS))
    raise AssertionError(f"unsupported patch size {size}")


def _patch_cells(patch: np.ndarray) -> list[tuple[int, int]]:
    return [tuple(map(int, coord)) for coord in np.argwhere(patch != BG)]


def _placement_score(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> int:
    patch_cells = {(r0 + pr, c0 + pc) for pr, pc in _patch_cells(patch)}
    score = 0
    for rr, cc in patch_cells:
        for dr, dc in DIRS8:
            nr = rr + dr
            nc = cc + dc
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                if (nr, nc) not in patch_cells and int(grid[nr, nc]) != BG:
                    score += 1
    return score


def _place_patch(grid: np.ndarray, patch: np.ndarray) -> bool:
    h, w = grid.shape
    ph, pw = patch.shape
    cells = _patch_cells(patch)
    candidates: list[tuple[int, int, int]] = []

    for r0 in range(h - ph + 1):
        for c0 in range(w - pw + 1):
            ok = True
            for pr, pc in cells:
                if int(grid[r0 + pr, c0 + pc]) != BG:
                    ok = False
                    break
            if not ok:
                continue
            score = _placement_score(grid, patch, r0, c0)
            candidates.append((score, r0, c0))

    if not candidates:
        return False

    candidates.sort()
    best_score = candidates[0][0]
    good = [(r0, c0) for score, r0, c0 in candidates if score <= best_score + 1]
    r0, c0 = random.choice(good)
    mask = patch != BG
    region = grid[r0 : r0 + ph, c0 : c0 + pw]
    region[mask] = patch[mask]
    return True


def _place_isolated_single(grid: np.ndarray, color: int) -> bool:
    h, w = grid.shape
    candidates: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) != BG:
                continue
            ok = True
            for dr, dc in DIRS8:
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) != BG:
                    ok = False
                    break
            if ok:
                candidates.append((r, c))

    if not candidates:
        return False

    r, c = random.choice(candidates)
    grid[r, c] = color
    return True


def _component_sizes(grid: np.ndarray) -> list[int]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    sizes: list[int] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or visited[r, c]:
                continue

            stack = [(r, c)]
            visited[r, c] = True
            size = 0
            while stack:
                rr, cc = stack.pop()
                size += 1
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            sizes.append(size)

    return sizes


def _same_color_neighbor_flags(grid: np.ndarray, r: int, c: int) -> tuple[bool, bool, bool, bool]:
    color = int(grid[r, c])
    if color == BG:
        return False, False, False, False
    h, w = grid.shape
    up = r > 0 and int(grid[r - 1, c]) == color
    down = r + 1 < h and int(grid[r + 1, c]) == color
    left = c > 0 and int(grid[r, c - 1]) == color
    right = c + 1 < w and int(grid[r, c + 1]) == color
    return up, down, left, right


def _has_any_foreground_neighbor(grid: np.ndarray, r: int, c: int) -> bool:
    h, w = grid.shape
    for dr, dc in DIRS4:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) != BG:
            return True
    return False


def _reference_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = np.zeros((h * SCALE, w * SCALE), dtype=int)

    for r in range(h):
        for c in range(w):
            out[r * SCALE : (r + 1) * SCALE, c * SCALE : (c + 1) * SCALE] = _base_block(int(g[r, c]))

    for r in range(h):
        for c in range(w - 1):
            color = int(g[r, c])
            if color in FG_COLORS and int(g[r, c + 1]) == color:
                c0 = c * SCALE
                out[r * SCALE : (r + 1) * SCALE, c0 + SCALE - 1] = YELLOW
                out[r * SCALE : (r + 1) * SCALE, c0 + SCALE] = YELLOW

    for r in range(h - 1):
        for c in range(w):
            color = int(g[r, c])
            if color in FG_COLORS and int(g[r + 1, c]) == color:
                r0 = r * SCALE
                out[r0 + SCALE - 1, c * SCALE : (c + 1) * SCALE] = YELLOW
                out[r0 + SCALE, c * SCALE : (c + 1) * SCALE] = YELLOW

    return out


def _semantic_ok(grid: np.ndarray) -> bool:
    if grid.ndim != 2:
        return False
    if set(map(int, np.unique(grid))) - {BG, RED, GRAY}:
        return False

    h, w = grid.shape
    area = h * w
    fg_count = int(np.count_nonzero(grid))
    if not (0.30 <= fg_count / area <= 0.50):
        return False
    if int(np.count_nonzero(grid == RED)) == 0 or int(np.count_nonzero(grid == GRAY)) == 0:
        return False

    same_h = False
    same_v = False
    diff_adj = False
    isolated = False
    opposite_bridge = False

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG:
                continue
            up, down, left, right = _same_color_neighbor_flags(grid, r, c)
            if left and right:
                opposite_bridge = True
            if up and down:
                opposite_bridge = True
            if not _has_any_foreground_neighbor(grid, r, c):
                isolated = True
            if c + 1 < w and int(grid[r, c + 1]) != BG:
                if int(grid[r, c + 1]) == color:
                    same_h = True
                else:
                    diff_adj = True
            if r + 1 < h and int(grid[r + 1, c]) != BG:
                if int(grid[r + 1, c]) == color:
                    same_v = True
                else:
                    diff_adj = True

    if not (same_h and same_v and diff_adj and isolated and opposite_bridge):
        return False

    if max(_component_sizes(grid), default=0) > 6:
        return False

    out = main(grid)
    if not np.array_equal(out, _reference_output(grid)):
        return False

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            block = out[r * SCALE : (r + 1) * SCALE, c * SCALE : (c + 1) * SCALE]

            if color == BG and not np.all(block == BG):
                return False

            up, down, left, right = _same_color_neighbor_flags(grid, r, c)
            if color in FG_COLORS and not any((up, down, left, right)):
                if not np.array_equal(block, _base_block(color)):
                    return False

    for r in range(h):
        for c in range(w - 1):
            left_color = int(grid[r, c])
            right_color = int(grid[r, c + 1])
            left_edge = out[r * SCALE : (r + 1) * SCALE, c * SCALE + SCALE - 1]
            right_edge = out[r * SCALE : (r + 1) * SCALE, (c + 1) * SCALE]
            if left_color in FG_COLORS and right_color == left_color:
                if not (np.all(left_edge == YELLOW) and np.all(right_edge == YELLOW)):
                    return False
            if left_color in FG_COLORS and right_color in FG_COLORS and left_color != right_color:
                if np.all(left_edge == YELLOW) and np.all(right_edge == YELLOW):
                    return False

    for r in range(h - 1):
        for c in range(w):
            top_color = int(grid[r, c])
            bottom_color = int(grid[r + 1, c])
            top_edge = out[r * SCALE + SCALE - 1, c * SCALE : (c + 1) * SCALE]
            bottom_edge = out[(r + 1) * SCALE, c * SCALE : (c + 1) * SCALE]
            if top_color in FG_COLORS and bottom_color == top_color:
                if not (np.all(top_edge == YELLOW) and np.all(bottom_edge == YELLOW)):
                    return False
            if top_color in FG_COLORS and bottom_color in FG_COLORS and top_color != bottom_color:
                if np.all(top_edge == YELLOW) and np.all(bottom_edge == YELLOW):
                    return False

    return True


def _target_foreground_count(area: int) -> int:
    min_fg = max(8, int(np.ceil(0.30 * area)))
    max_fg = min(int(np.floor(0.50 * area)), min_fg + 4)
    return random.randint(min_fg, max_fg)


def _build_patch_list(target_fg: int) -> list[np.ndarray]:
    colors = list(FG_COLORS)
    random.shuffle(colors)
    patches = [
        _line_patch(3, "h", colors[0]),
        _line_patch(3, "v", colors[1]),
        _pair_patch(random.choice(("h", "v")), *random.sample(FG_COLORS, 2)),
    ]

    current = sum(int(np.count_nonzero(patch)) for patch in patches)
    target_without_reserved_single = target_fg - 1

    while current < target_without_reserved_single:
        remaining = target_without_reserved_single - current
        size_choices = [size for size in (5, 3, 2, 1) if size <= remaining]
        weights = []
        for size in size_choices:
            if size == 5:
                weights.append(1)
            elif size == 3:
                weights.append(4)
            elif size == 2:
                weights.append(3)
            else:
                weights.append(2)
        size = random.choices(size_choices, weights=weights, k=1)[0]
        patches.append(_random_patch_of_size(size))
        current += size

    patches.sort(key=lambda patch: (int(np.count_nonzero(patch)), patch.shape[0] * patch.shape[1]), reverse=True)
    return patches


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(5, 7)
    w = random.randint(5, 7)
    grid = np.zeros((h, w), dtype=int)

    target_fg = _target_foreground_count(h * w)
    patches = _build_patch_list(target_fg)

    for patch in patches:
        if not _place_patch(grid, patch):
            return None

    if not _place_isolated_single(grid, random.choice(FG_COLORS)):
        return None

    return grid if _semantic_ok(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
