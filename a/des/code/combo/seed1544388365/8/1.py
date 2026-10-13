from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
ARC_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIR4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _decode_legend_block(block: np.ndarray) -> tuple[int, np.ndarray] | None:
    if block.shape != (3, 3):
        return None
    key = int(block[0, 0])
    if key in (BLACK, GREY):
        return None
    colors = {int(x) for x in np.unique(block)}
    if not colors.issubset({BLACK, key}):
        return None
    pattern = (block[1:3, 1:3] == key)
    return key, pattern


def _legend_blocks_on_side(grid: np.ndarray, col0: int) -> list[tuple[int, int, int, np.ndarray]]:
    h, _w = grid.shape
    candidates: list[tuple[int, int, int, np.ndarray]] = []
    for r0 in range(h - 2):
        block = grid[r0 : r0 + 3, col0 : col0 + 3]
        decoded = _decode_legend_block(block)
        if decoded is None:
            continue
        if r0 > 0 and np.any(grid[r0 - 1, col0 : col0 + 3] != BLACK):
            continue
        if r0 + 3 < h and np.any(grid[r0 + 3, col0 : col0 + 3] != BLACK):
            continue
        key, pattern = decoded
        candidates.append((r0, col0, key, pattern))

    result: list[tuple[int, int, int, np.ndarray]] = []
    used_keys: set[int] = set()
    last_end = -10
    for r0, c0, key, pattern in sorted(candidates, key=lambda x: x[0]):
        if r0 < last_end or key in used_keys:
            continue
        result.append((r0, c0, key, pattern))
        used_keys.add(key)
        last_end = r0 + 3
    return result


def _find_legend(grid: np.ndarray) -> tuple[dict[int, np.ndarray], np.ndarray]:
    h, w = grid.shape
    assert w >= 3 and h >= 3

    options = []
    for col0 in (0, w - 3):
        blocks = _legend_blocks_on_side(grid, col0)
        score = (
            len(blocks),
            sum(int(np.sum(grid[r0 : r0 + 3, c0 : c0 + 3] != BLACK)) for r0, c0, _key, _pattern in blocks),
        )
        options.append((score, blocks))

    score, blocks = max(options, key=lambda x: x[0])
    assert score[0] > 0, "expected a legend on the left or right edge"

    mapping: dict[int, np.ndarray] = {}
    legend_mask = np.zeros_like(grid, dtype=bool)
    for r0, c0, key, pattern in blocks:
        mapping[key] = pattern.copy()
        legend_mask[r0 : r0 + 3, c0 : c0 + 3] = True
    return mapping, legend_mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each grey-bounded interior by looking up its marker color in the edge legend.

    The legend consists of 3x3 monochrome blocks on the left or right edge. The block's
    top-left pixel is the key color; the block's lower-right 2x2 area is the tiling
    texture. Each enclosed interior contains one marker pixel of a legend color.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    mapping, legend_mask = _find_legend(g)

    interior_mask = (g != GREY) & (~legend_mask)
    out = g.copy()
    for comp in _components(interior_mask):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue
        marker_colors = {
            int(g[r, c])
            for r, c in comp
            if int(g[r, c]) not in (BLACK, GREY)
        }
        if len(marker_colors) != 1:
            continue
        color = next(iter(marker_colors))
        if color not in mapping:
            continue
        pattern = mapping[color]
        r_anchor = min(r for r, _c in comp)
        c_anchor = min(c for _r, c in comp)
        for r, c in comp:
            out[r, c] = color if pattern[(r - r_anchor) % 2, (c - c_anchor) % 2] else BLACK
    return out


PATTERN_BANK = [
    np.array([[bool((bits >> 0) & 1), bool((bits >> 1) & 1)],
              [bool((bits >> 2) & 1), bool((bits >> 3) & 1)]], dtype=bool)
    for bits in range(1, 16)
]


def _has_hole(mask: np.ndarray) -> bool:
    empty = ~mask
    for comp in _components(empty):
        if any(
            r == 0 or c == 0 or r == mask.shape[0] - 1 or c == mask.shape[1] - 1
            for r, c in comp
        ):
            continue
        return True
    return False


def _grow_shape() -> np.ndarray | None:
    for _ in range(80):
        h = random.randint(3, 7)
        w = random.randint(3, 7)
        target = random.randint(max(4, (h * w) // 3), h * w - 1)
        cells = {(random.randrange(h), random.randrange(w))}
        while len(cells) < target:
            frontier: list[tuple[tuple[int, int], int]] = []
            for r, c in cells:
                for dr, dc in DIR4:
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        adj = 0
                        for dr2, dc2 in DIR4:
                            if (nr + dr2, nc + dc2) in cells:
                                adj += 1
                        frontier.append(((nr, nc), adj))
            if not frontier:
                break
            coords = [pt for pt, _adj in frontier]
            weights = [adj + 1 for _pt, adj in frontier]
            cells.add(random.choices(coords, weights=weights, k=1)[0])

        rs = [r for r, _c in cells]
        cs = [c for _r, c in cells]
        r0 = min(rs)
        c0 = min(cs)
        r1 = max(rs)
        c1 = max(cs)
        hh = r1 - r0 + 1
        ww = c1 - c0 + 1
        if hh < 2 or ww < 2:
            continue
        mask = np.zeros((hh, ww), dtype=bool)
        for r, c in cells:
            mask[r - r0, c - c0] = True
        if _has_hole(mask):
            continue
        if np.all(mask):
            continue
        return mask
    return None


def _build_region_patch(inner_mask: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]], tuple[int, int]]:
    h, w = inner_mask.shape
    patch = np.zeros((h + 2, w + 2), dtype=bool)
    patch[1:-1, 1:-1] = inner_mask
    border = np.zeros_like(patch, dtype=bool)
    for r, c in np.argwhere(patch):
        rr = int(r)
        cc = int(c)
        for dr, dc in DIR4:
            nr = rr + dr
            nc = cc + dc
            if not patch[nr, nc]:
                border[nr, nc] = True

    interior_coords = [(int(r), int(c)) for r, c in np.argwhere(patch)]
    border_coords = [(int(r), int(c)) for r, c in np.argwhere(border)]
    return interior_coords, border_coords, patch.shape


def _legend_block(color: int, pattern: np.ndarray) -> np.ndarray:
    block = np.zeros((3, 3), dtype=int)
    block[0, 0] = color
    for r in range(3):
        for c in range(3):
            if r == 0 and c == 0:
                continue
            if pattern[(r - 1) % 2, (c - 1) % 2]:
                block[r, c] = color
    return block


def _render_expected(
    grid: np.ndarray,
    regions: list[tuple[list[tuple[int, int]], int, np.ndarray]],
) -> np.ndarray:
    out = grid.copy()
    for pts, color, pattern in regions:
        r_anchor = min(r for r, _c in pts)
        c_anchor = min(c for _r, c in pts)
        for r, c in pts:
            out[r, c] = color if pattern[(r - r_anchor) % 2, (c - c_anchor) % 2] else BLACK
    return out


def _candidate_input() -> tuple[np.ndarray, np.ndarray] | None:
    height = random.randint(17, 28)
    width = random.randint(18, 30)
    max_legend = min(len(ARC_COLORS), (height + 1) // 4)
    if max_legend < 2:
        return None

    object_count = random.randint(2, min(5, max_legend))
    extra_count = random.randint(0, min(2, max_legend - object_count))
    legend_count = object_count + extra_count

    legend_colors = random.sample(ARC_COLORS, legend_count)
    used_colors = random.sample(legend_colors, object_count)
    patterns = random.sample(PATTERN_BANK, legend_count)
    pattern_by_color = {color: pattern.copy() for color, pattern in zip(legend_colors, patterns)}

    grid = np.zeros((height, width), dtype=int)
    side = random.choice(["left", "right"])
    legend_rows = 4 * legend_count - 1
    legend_top = random.randint(0, height - legend_rows)
    legend_col = 0 if side == "left" else width - 3
    for i, color in enumerate(legend_colors):
        r0 = legend_top + 4 * i
        grid[r0 : r0 + 3, legend_col : legend_col + 3] = _legend_block(color, pattern_by_color[color])

    play_r0 = 1
    play_r1 = height - 1
    if side == "left":
        play_c0 = 5
        play_c1 = width - 1
    else:
        play_c0 = 1
        play_c1 = width - 5
    if play_c1 - play_c0 < 8:
        return None

    occupied = np.zeros_like(grid, dtype=bool)
    occupied[:, legend_col : legend_col + 3] = True

    regions_for_expected: list[tuple[list[tuple[int, int]], int, np.ndarray]] = []
    random.shuffle(used_colors)

    for color in used_colors:
        placed = False
        pattern = pattern_by_color[color]
        for _ in range(120):
            inner_mask = _grow_shape()
            if inner_mask is None:
                continue
            interior_local, border_local, patch_shape = _build_region_patch(inner_mask)
            ph, pw = patch_shape
            if ph >= play_r1 - play_r0 or pw >= play_c1 - play_c0:
                continue

            r_base = random.randint(play_r0, play_r1 - ph)
            c_base = random.randint(play_c0, play_c1 - pw)
            border_global = [(r_base + r, c_base + c) for r, c in border_local]
            interior_global = [(r_base + r, c_base + c) for r, c in interior_local]
            all_cells = border_global + interior_global

            moat: set[tuple[int, int]] = set()
            fits = True
            for r, c in all_cells:
                if not (play_r0 <= r < play_r1 and play_c0 <= c < play_c1):
                    fits = False
                    break
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        rr = r + dr
                        cc = c + dc
                        if 0 <= rr < height and 0 <= cc < width:
                            moat.add((rr, cc))
            if not fits:
                continue
            if any(occupied[r, c] for r, c in moat):
                continue

            for r, c in border_global:
                grid[r, c] = GREY
            marker_r, marker_c = random.choice(interior_global)
            for r, c in interior_global:
                grid[r, c] = BLACK
            grid[marker_r, marker_c] = color
            for r, c in moat:
                occupied[r, c] = True
            regions_for_expected.append((interior_global, color, pattern))
            placed = True
            break
        if not placed:
            return None

    expected = _render_expected(grid, regions_for_expected)
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(300):
        built = _candidate_input()
        if built is None:
            continue
        grid, expected = built
        if np.array_equal(main(grid), expected):
            return grid
    raise RuntimeError("failed to generate a valid input")
