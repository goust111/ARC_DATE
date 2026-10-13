from collections import deque
import random

import numpy as np


_BG = 0
_FENCE = 5
_ENTRY_INNER_W = 4
_PATTERN_H = 2
_PATTERN_W = 3
_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
_GEN_ATTEMPTS = 400

_PATTERN_LIBRARY = (
    np.array([[1, 1, 1], [0, 0, 0]], dtype=bool),
    np.array([[1, 0, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 0], [0, 1, 1]], dtype=bool),
    np.array([[0, 1, 0], [1, 0, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 0, 1], [1, 0, 1]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 0]], dtype=bool),
    np.array([[0, 1, 1], [1, 1, 0]], dtype=bool),
)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(points: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(points[:, 0].min()),
        int(points[:, 1].min()),
        int(points[:, 0].max()),
        int(points[:, 1].max()),
    )


def _touches_boundary(points: np.ndarray, h: int, w: int) -> bool:
    return bool(
        np.any(points[:, 0] == 0)
        or np.any(points[:, 1] == 0)
        or np.any(points[:, 0] == h - 1)
        or np.any(points[:, 1] == w - 1)
    )


def _is_rectangular_ring(points: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(points)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    ring = np.zeros((h, w), dtype=bool)
    ring[0, :] = True
    ring[-1, :] = True
    ring[:, 0] = True
    ring[:, -1] = True
    sub = np.zeros((h, w), dtype=bool)
    sub[points[:, 0] - r0, points[:, 1] - c0] = True
    return bool(np.array_equal(sub, ring))


def _find_dictionary(grid: np.ndarray) -> tuple[dict[int, np.ndarray], np.ndarray]:
    grey = _components(grid == _FENCE)
    candidates: list[tuple[int, int, dict[int, np.ndarray], np.ndarray]] = []
    for comp in grey:
        if not _is_rectangular_ring(comp):
            continue
        r0, c0, r1, c1 = _bbox(comp)
        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        if inner.shape[1] != _ENTRY_INNER_W or inner.shape[0] < 4 or inner.shape[0] % 2 != 0:
            continue

        mapping: dict[int, np.ndarray] = {}
        ok = True
        for band_r in range(0, inner.shape[0], 2):
            band = inner[band_r : band_r + 2, :]
            colors = set(map(int, np.unique(band))) - {_BG}
            if len(colors) != 1:
                ok = False
                break
            color = colors.pop()
            if int(band[0, 0]) != color:
                ok = False
                break
            mapping[color] = (band[:, 1:] == color)
        if not ok or len(mapping) < 2:
            continue

        inner_mask = np.zeros_like(grid, dtype=bool)
        inner_mask[r0 + 1 : r1, c0 + 1 : c1] = True
        candidates.append((r0, c0, mapping, inner_mask))

    assert candidates, "expected one parseable texture dictionary"
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2], candidates[0][3]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed region by tiling the texture whose key color matches the region's seed.

    Input structure:
    - The background is black. Several one-pixel-wide grey fences enclose irregular regions.
    - One corner contains a small grey rectangular dictionary box. Inside it, each 2-row entry band has a colored key pixel in the left column and a 3x2 same-colored texture pattern to its right.
    - Every non-dictionary enclosed region contains exactly one colored seed pixel on an otherwise black interior.

    Rule:
    - Parse the dictionary to build a mapping from seed color to a 3x2 texture mask.
    - For each enclosed region outside the dictionary, find its unique seed color and select the matching texture.
    - Repaint the entire region, including the seed cell, by tiling that 3x2 texture from the region's top-left interior coordinate; texture-off cells stay black, while grey fences and the dictionary remain unchanged.

    Key insight:
    - The task combines two independent steps: lookup of a color-keyed texture template, then coordinate-anchored tiling of that template over an enclosed spatial region.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    mapping, dict_inner_mask = _find_dictionary(g)

    out = g.copy()
    for comp in _components(g != _FENCE):
        if _touches_boundary(comp, h, w):
            continue
        if np.any(dict_inner_mask[comp[:, 0], comp[:, 1]]):
            continue

        values = g[comp[:, 0], comp[:, 1]]
        colors = set(map(int, np.unique(values))) - {_BG}
        if len(colors) != 1:
            continue
        color = colors.pop()
        if int(np.sum(values == color)) != 1 or color not in mapping:
            continue

        pattern = mapping[color]
        r0 = int(comp[:, 0].min())
        c0 = int(comp[:, 1].min())
        for r, c in comp:
            out[r, c] = color if pattern[(int(r) - r0) % _PATTERN_H, (int(c) - c0) % _PATTERN_W] else _BG

    return out


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    q = deque()
    for r in range(h):
        for c in (0, w - 1):
            if not mask[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not mask[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))

    return mask | (~mask & ~seen)


def _random_blob(rng: random.Random, h: int, w: int, target_area: int) -> np.ndarray:
    cells = {(rng.randrange(h), rng.randrange(w))}
    while len(cells) < target_area:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            break
        cells.add(rng.choice(frontier))

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return _fill_holes(mask)


def _dilate4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[:-1, :] |= mask[1:, :]
    out[1:, :] |= mask[:-1, :]
    out[:, :-1] |= mask[:, 1:]
    out[:, 1:] |= mask[:, :-1]
    return out


def _make_region_patch(rng: random.Random, color: int) -> np.ndarray | None:
    for _ in range(120):
        inner_h = rng.randint(4, 7)
        inner_w = rng.randint(5, 9)
        max_area = min(inner_h * inner_w - 2, 18)
        min_area = max(8, inner_h + inner_w - 3)
        if min_area > max_area:
            continue
        target_area = rng.randint(min_area, max_area)
        mask = _random_blob(rng, inner_h, inner_w, target_area)
        if not np.any(mask) or bool(mask.all()):
            continue
        rows, cols = np.where(mask)
        if int(rows.max() - rows.min()) < 2 or int(cols.max() - cols.min()) < 3:
            continue

        inner = np.zeros((inner_h + 2, inner_w + 2), dtype=bool)
        inner[1:-1, 1:-1] = mask
        border = _dilate4(inner) & ~inner
        if not np.any(border):
            continue

        patch = np.full(inner.shape, _BG, dtype=int)
        patch[border] = _FENCE
        region_cells = list(zip(*np.where(inner)))
        seed_r, seed_c = rng.choice(region_cells)
        patch[seed_r, seed_c] = color
        return patch
    return None


def _render_dictionary(colors: list[int], patterns: list[np.ndarray]) -> np.ndarray:
    n = len(colors)
    h = 2 * n + 2
    w = _ENTRY_INNER_W + 2
    box = np.full((h, w), _BG, dtype=int)
    box[0, :] = _FENCE
    box[-1, :] = _FENCE
    box[:, 0] = _FENCE
    box[:, -1] = _FENCE
    for i, (color, pattern) in enumerate(zip(colors, patterns)):
        r = 1 + 2 * i
        box[r, 1] = color
        sub = box[r : r + 2, 2:5]
        sub[pattern] = color
    return box


def _can_place(reserved: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(reserved.shape[0], r0 + h + 1)
    cc1 = min(reserved.shape[1], c0 + w + 1)
    return not bool(np.any(reserved[rr0:rr1, cc0:cc1]))


def _mark_reserved(reserved: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(reserved.shape[0], r0 + h + 1)
    cc1 = min(reserved.shape[1], c0 + w + 1)
    reserved[rr0:rr1, cc0:cc1] = True


def _corner_position(corner: str, grid_h: int, grid_w: int, box_h: int, box_w: int) -> tuple[int, int]:
    if corner == "tl":
        return 1, 1
    if corner == "tr":
        return 1, grid_w - box_w - 1
    if corner == "bl":
        return grid_h - box_h - 1, 1
    return grid_h - box_h - 1, grid_w - box_w - 1


def _region_colors(rng: random.Random, entry_colors: list[int], n_regions: int) -> list[int]:
    colors = entry_colors.copy()
    rng.shuffle(colors)
    assigned = colors[: min(len(colors), n_regions)]
    while len(assigned) < n_regions:
        assigned.append(rng.choice(entry_colors))
    rng.shuffle(assigned)
    if len(set(assigned)) == 1 and len(entry_colors) > 1:
        assigned[-1] = entry_colors[1]
        rng.shuffle(assigned)
    return assigned


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_entries = rng.randint(2, 4)
    n_regions = rng.randint(2, 4)
    colors = rng.sample(list(_COLORS), n_entries)
    patterns = [p.copy() for p in rng.sample(list(_PATTERN_LIBRARY), n_entries)]

    dict_box = _render_dictionary(colors, patterns)
    dict_h, dict_w = dict_box.shape

    min_h = max(18, dict_h + 4, 10 + 3 * n_regions)
    min_w = max(18, dict_w + 4, 12 + 3 * n_regions)
    if min_h > 30 or min_w > 30:
        return None
    grid_h = rng.randint(min_h, 30)
    grid_w = rng.randint(min_w, 30)

    grid = np.full((grid_h, grid_w), _BG, dtype=int)
    reserved = np.zeros((grid_h, grid_w), dtype=bool)

    corner = rng.choice(("tl", "tr", "bl", "br"))
    dr, dc = _corner_position(corner, grid_h, grid_w, dict_h, dict_w)
    if not _can_place(reserved, dr, dc, dict_h, dict_w):
        return None
    grid[dr : dr + dict_h, dc : dc + dict_w] = dict_box
    _mark_reserved(reserved, dr, dc, dict_h, dict_w)

    for color in _region_colors(rng, colors, n_regions):
        patch = _make_region_patch(rng, color)
        if patch is None:
            return None
        ph, pw = patch.shape
        placed = False
        for _ in range(160):
            r0 = rng.randint(1, grid_h - ph - 1)
            c0 = rng.randint(1, grid_w - pw - 1)
            if not _can_place(reserved, r0, c0, ph, pw):
                continue
            grid[r0 : r0 + ph, c0 : c0 + pw] = np.where(patch != _BG, patch, grid[r0 : r0 + ph, c0 : c0 + pw])
            _mark_reserved(reserved, r0, c0, ph, pw)
            placed = True
            break
        if not placed:
            return None

    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != grid.shape or not np.any(out != grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid texture-tiling instance")
