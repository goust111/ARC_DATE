from __future__ import annotations

from collections import Counter

import numpy as np


def _divisors_with_repeats(n: int) -> list[int]:
    return [d for d in range(2, n) if n % d == 0 and n // d >= 2]


def _infer_template(mask: np.ndarray, tile_h: int, tile_w: int) -> np.ndarray:
    h, w = mask.shape
    tiles_r = h // tile_h
    tiles_c = w // tile_w
    blocks = mask.reshape(tiles_r, tile_h, tiles_c, tile_w).transpose(1, 3, 0, 2)
    votes = blocks.reshape(tile_h, tile_w, tiles_r * tiles_c).sum(axis=2)
    return votes * 2 > (tiles_r * tiles_c)


def _dominant_color(tile: np.ndarray) -> int:
    vals = tile[tile != 0]
    assert vals.size > 0, "tile must expose its own color"
    counts = Counter(map(int, vals.tolist()))
    return counts.most_common(1)[0][0]


def _evaluate_candidate(grid: np.ndarray, template: np.ndarray, tile_h: int, tile_w: int) -> tuple[int, int, int, int, int]:
    h, w = grid.shape
    tiles_r = h // tile_h
    tiles_c = w // tile_w
    area = tile_h * tile_w
    occ = grid != 0
    tiled_template = np.tile(template, (tiles_r, tiles_c))
    occ_mismatch = int(np.sum(occ != tiled_template))

    if not template.any() or template.all():
        return (10**9, occ_mismatch, area, tile_h, tile_w)

    bases = np.zeros((tiles_r, tiles_c), dtype=int)
    tie_penalty = 0
    structure_penalty = 0
    mirror_penalty = 0
    neighbor_penalty = 0
    color_mismatch = 0
    anomaly_count = 0
    false_pos_store: list[tuple[int, int, list[tuple[int, int, int]]]] = []

    expected_shape = np.where(template, 1, 0)

    for tr in range(tiles_r):
        for tc in range(tiles_c):
            r0 = tr * tile_h
            c0 = tc * tile_w
            tile = grid[r0 : r0 + tile_h, c0 : c0 + tile_w]

            vals = tile[tile != 0]
            if vals.size == 0:
                return (10**9, occ_mismatch, area, tile_h, tile_w)

            counts = Counter(map(int, vals.tolist()))
            common = counts.most_common()
            if len(common) >= 2 and common[0][1] == common[1][1]:
                tie_penalty += 3
            base = common[0][0]
            bases[tr, tc] = base

            expected = np.where(expected_shape, base, 0)
            color_mismatch += int(np.sum(tile != expected))

            false_pos = []
            false_neg = set()
            for rr in range(tile_h):
                for cc in range(tile_w):
                    if template[rr, cc]:
                        if tile[rr, cc] == 0:
                            false_neg.add((rr, cc))
                        elif int(tile[rr, cc]) != int(base):
                            color_mismatch += 2
                    elif tile[rr, cc] != 0:
                        false_pos.append((rr, cc, int(tile[rr, cc])))

            anomaly_count += len(false_pos) + len(false_neg)
            structure_penalty += abs(len(false_pos) - len(false_neg)) * 4

            for rr, cc, foreign in false_pos:
                mr = tile_h - 1 - rr
                mc = tile_w - 1 - cc
                if (mr, mc) not in false_neg:
                    mirror_penalty += 8
                if foreign == base:
                    mirror_penalty += 3
            false_pos_store.append((tr, tc, false_pos))

    for tr, tc, false_pos in false_pos_store:
        for rr, cc, foreign in false_pos:
            valid = False
            if rr < tile_h - 1 - rr and tr > 0 and int(bases[tr - 1, tc]) == foreign:
                valid = True
            if rr > tile_h - 1 - rr and tr + 1 < tiles_r and int(bases[tr + 1, tc]) == foreign:
                valid = True
            if cc < tile_w - 1 - cc and tc > 0 and int(bases[tr, tc - 1]) == foreign:
                valid = True
            if cc > tile_w - 1 - cc and tc + 1 < tiles_c and int(bases[tr, tc + 1]) == foreign:
                valid = True
            if not valid:
                neighbor_penalty += 2

    return (
        structure_penalty + mirror_penalty + neighbor_penalty + tie_penalty + color_mismatch,
        occ_mismatch,
        anomaly_count,
        area,
        tile_h * 100 + tile_w,
    )


def _infer_tile_template(grid: np.ndarray) -> tuple[int, int, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    row_divs = _divisors_with_repeats(h)
    col_divs = _divisors_with_repeats(w)
    assert row_divs and col_divs, "grid must contain repeated tiles on both axes"

    best: tuple[tuple[int, int, int, int, int], int, int, np.ndarray] | None = None
    occ = g != 0
    for tile_h in row_divs:
        for tile_w in col_divs:
            template = _infer_template(occ, tile_h, tile_w)
            key = _evaluate_candidate(g, template, tile_h, tile_w)
            if best is None or key < best[0]:
                best = (key, tile_h, tile_w, template)

    assert best is not None
    _, tile_h, tile_w, template = best
    return tile_h, tile_w, template


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover a hidden rectangular tile period, remove each foreign-color intrusion,
    and repaint every tile from the shared shape template using that tile's own
    dominant color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert 0 in set(map(int, g.ravel())), "background must be black"

    tile_h, tile_w, template = _infer_tile_template(g)
    h, w = g.shape
    tiles_r = h // tile_h
    tiles_c = w // tile_w

    out = np.zeros_like(g)
    for tr in range(tiles_r):
        for tc in range(tiles_c):
            r0 = tr * tile_h
            c0 = tc * tile_w
            tile = g[r0 : r0 + tile_h, c0 : c0 + tile_w]
            base = _dominant_color(tile)
            out[r0 : r0 + tile_h, c0 : c0 + tile_w] = np.where(template, base, 0)
    return out


PALETTE = np.arange(1, 10, dtype=int)
GEN_RETRIES = 600
TILE_H_RANGE = (5, 9)
TILE_W_RANGE = (5, 9)
TILE_ROWS_RANGE = (3, 6)
TILE_COLS_RANGE = (3, 6)
CORRUPT_TILE_RATIO = 3


def _axis_has_smaller_period(mask: np.ndarray, axis: int) -> bool:
    n = mask.shape[axis]
    for step in range(1, n):
        if n % step != 0:
            continue
        if step == n:
            continue
        periodic = True
        for idx in range(step, n):
            if axis == 0:
                if not np.array_equal(mask[idx], mask[idx % step]):
                    periodic = False
                    break
            else:
                if not np.array_equal(mask[:, idx], mask[:, idx % step]):
                    periodic = False
                    break
        if periodic:
            return True
    return False


def _random_connected_mask(tile_h: int, tile_w: int) -> np.ndarray | None:
    inner_h = tile_h - 2
    inner_w = tile_w - 2
    area = inner_h * inner_w
    target = int(np.random.randint(max(5, area // 4), max(6, (area * 3) // 5) + 1))

    start = (int(np.random.randint(0, inner_h)), int(np.random.randint(0, inner_w)))
    pts = {start}
    frontier: set[tuple[int, int]] = set()

    def add_neighbors(cell: tuple[int, int]) -> None:
        r, c = cell
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr = r + dr
            nc = c + dc
            if 0 <= nr < inner_h and 0 <= nc < inner_w and (nr, nc) not in pts:
                frontier.add((nr, nc))

    add_neighbors(start)
    while len(pts) < target and frontier:
        choices = list(frontier)
        rr, cc = choices[int(np.random.randint(0, len(choices)))]
        frontier.remove((rr, cc))
        pts.add((rr, cc))
        add_neighbors((rr, cc))

    if len(pts) < target:
        return None

    mask = np.zeros((tile_h, tile_w), dtype=bool)
    for rr, cc in pts:
        mask[rr + 1, cc + 1] = True

    extra_tries = int(np.random.randint(1, 4))
    for _ in range(extra_tries):
        filled = np.argwhere(mask[1:-1, 1:-1])
        if filled.size == 0:
            break
        rr, cc = filled[int(np.random.randint(0, len(filled)))]
        dr, dc = ((-1, 0), (1, 0), (0, -1), (0, 1))[int(np.random.randint(0, 4))]
        nr = rr + dr + 1
        nc = cc + dc + 1
        if 1 <= nr < tile_h - 1 and 1 <= nc < tile_w - 1:
            mask[nr, nc] = True

    return mask


def _candidate_intrusions(mask: np.ndarray) -> list[tuple[int, int, tuple[tuple[int, int], ...]]]:
    tile_h, tile_w = mask.shape
    items: list[tuple[int, int, tuple[tuple[int, int], ...]]] = []
    for rr in range(tile_h):
        for cc in range(tile_w):
            mr = tile_h - 1 - rr
            mc = tile_w - 1 - cc
            if mask[rr, cc] or not mask[mr, mc]:
                continue
            dirs: list[tuple[int, int]] = []
            if rr < mr:
                dirs.append((-1, 0))
            elif rr > mr:
                dirs.append((1, 0))
            if cc < mc:
                dirs.append((0, -1))
            elif cc > mc:
                dirs.append((0, 1))
            if dirs:
                items.append((rr, cc, tuple(dirs)))
    return items


def _mask_ok(mask: np.ndarray) -> bool:
    if not mask.any() or mask.all():
        return False
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if rows.size < 3 or cols.size < 3:
        return False
    if np.array_equal(mask, mask[::-1, ::-1]):
        return False
    if _axis_has_smaller_period(mask, axis=0) or _axis_has_smaller_period(mask, axis=1):
        return False

    cand = _candidate_intrusions(mask)
    if len(cand) < 4:
        return False

    direction_kinds = set()
    for _rr, _cc, dirs in cand:
        direction_kinds.update(dirs)
    return len(direction_kinds) >= 2


def _random_template(tile_h: int, tile_w: int) -> np.ndarray | None:
    for _ in range(120):
        mask = _random_connected_mask(tile_h, tile_w)
        if mask is not None and _mask_ok(mask):
            return mask
    return None


def _random_color_grid(rows: int, cols: int, palette: np.ndarray) -> np.ndarray:
    grid = np.zeros((rows, cols), dtype=int)
    for rr in range(rows):
        for cc in range(cols):
            blocked = set()
            if rr > 0:
                blocked.add(int(grid[rr - 1, cc]))
            if cc > 0:
                blocked.add(int(grid[rr, cc - 1]))
            choices = [int(color) for color in palette.tolist() if int(color) not in blocked]
            assert choices, "palette must support proper coloring"
            grid[rr, cc] = int(choices[int(np.random.randint(0, len(choices)))])
    return grid


def _render(template: np.ndarray, colors: np.ndarray) -> np.ndarray:
    tile_h, tile_w = template.shape
    rows, cols = colors.shape
    out = np.zeros((rows * tile_h, cols * tile_w), dtype=int)
    for rr in range(rows):
        for cc in range(cols):
            r0 = rr * tile_h
            c0 = cc * tile_w
            out[r0 : r0 + tile_h, c0 : c0 + tile_w] = np.where(template, int(colors[rr, cc]), 0)
    return out


def _valid_dirs_for_tile(rr: int, cc: int, rows: int, cols: int) -> set[tuple[int, int]]:
    dirs = set()
    if rr > 0:
        dirs.add((-1, 0))
    if rr + 1 < rows:
        dirs.add((1, 0))
    if cc > 0:
        dirs.add((0, -1))
    if cc + 1 < cols:
        dirs.add((0, 1))
    return dirs


def _neighbor_color(colors: np.ndarray, rr: int, cc: int, direction: tuple[int, int]) -> int:
    dr, dc = direction
    return int(colors[rr + dr, cc + dc])


def _gen_attempt() -> np.ndarray | None:
    tile_h = int(np.random.randint(*TILE_H_RANGE))
    tile_w = int(np.random.randint(*TILE_W_RANGE))
    rows = int(np.random.randint(*TILE_ROWS_RANGE))
    cols = int(np.random.randint(*TILE_COLS_RANGE))

    template = _random_template(tile_h, tile_w)
    if template is None:
        return None

    cand = _candidate_intrusions(template)
    if not cand:
        return None

    palette_size = int(np.random.randint(3, min(7, len(PALETTE)) + 1))
    palette = np.random.choice(PALETTE, size=palette_size, replace=False)
    colors = _random_color_grid(rows, cols, palette)
    clean = _render(template, colors)
    noisy = clean.copy()

    tile_count = rows * cols
    max_corrupt_tiles = max(1, tile_count // CORRUPT_TILE_RATIO)
    corrupt_tiles = int(np.random.randint(1, max_corrupt_tiles + 1))
    tile_order = np.random.permutation(tile_count)[:corrupt_tiles]

    changed = 0
    for idx in tile_order.tolist():
        rr = idx // cols
        cc = idx % cols
        allowed_dirs = _valid_dirs_for_tile(rr, cc, rows, cols)
        local_candidates = [(r, c, dirs) for r, c, dirs in cand if allowed_dirs.intersection(dirs)]
        if not local_candidates:
            continue

        max_intrusions = min(3, len(local_candidates))
        if max_intrusions == 1:
            count = 1
        else:
            weights = np.array([5, 3, 2][:max_intrusions], dtype=float)
            weights /= weights.sum()
            count = int(np.random.choice(np.arange(1, max_intrusions + 1), p=weights))

        chosen_idx = np.random.permutation(len(local_candidates))[:count]
        used: set[tuple[int, int]] = set()
        for choice in chosen_idx.tolist():
            r, c, dirs = local_candidates[choice]
            mr = tile_h - 1 - r
            mc = tile_w - 1 - c
            if (r, c) in used or (mr, mc) in used:
                continue

            valid = [direction for direction in dirs if direction in allowed_dirs]
            if not valid:
                continue

            direction = valid[int(np.random.randint(0, len(valid)))]
            foreign = _neighbor_color(colors, rr, cc, direction)
            r0 = rr * tile_h
            c0 = cc * tile_w
            noisy[r0 + r, c0 + c] = foreign
            noisy[r0 + mr, c0 + mc] = 0
            used.add((r, c))
            used.add((mr, mc))
            changed += 1

    if changed == 0 or np.array_equal(noisy, clean):
        return None

    repaired = main(noisy)
    if not np.array_equal(repaired, clean):
        return None
    return noisy


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
