import random
from collections import deque

import numpy as np


BG = 0


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _try_parse_horizontal_legend(sub: np.ndarray, wall: int):
    h, w = sub.shape
    if h != 5 or w < 9 or (w - 1) % 4 != 0:
        return None
    sep_cols = set(range(0, w, 4))
    if any(not np.all(sub[:, c] == wall) for c in sep_cols):
        return None
    non_sep = [c for c in range(w) if c not in sep_cols]
    if np.any(sub[1:4, :][:, non_sep] == wall):
        return None

    mapping: dict[int, np.ndarray] = {}
    count = (w - 1) // 4
    for i in range(count):
        tile = sub[1:4, 1 + 4 * i : 4 + 4 * i]
        vals = [int(v) for v in np.unique(tile) if int(v) not in (BG, wall)]
        if len(vals) != 1:
            return None
        color = vals[0]
        if np.any((tile != BG) & (tile != color)):
            return None
        if color in mapping:
            return None
        mapping[color] = tile == color
    return mapping


def _try_parse_vertical_legend(sub: np.ndarray, wall: int):
    h, w = sub.shape
    if w != 5 or h < 9 or (h - 1) % 4 != 0:
        return None
    sep_rows = set(range(0, h, 4))
    if any(not np.all(sub[r, :] == wall) for r in sep_rows):
        return None
    non_sep = [r for r in range(h) if r not in sep_rows]
    if np.any(sub[:, 1:4][non_sep, :] == wall):
        return None

    mapping: dict[int, np.ndarray] = {}
    count = (h - 1) // 4
    for i in range(count):
        tile = sub[1 + 4 * i : 4 + 4 * i, 1:4]
        vals = [int(v) for v in np.unique(tile) if int(v) not in (BG, wall)]
        if len(vals) != 1:
            return None
        color = vals[0]
        if np.any((tile != BG) & (tile != color)):
            return None
        if color in mapping:
            return None
        mapping[color] = tile == color
    return mapping


def _find_legend(grid: np.ndarray):
    h, w = grid.shape
    candidates = []
    for color in [int(v) for v in np.unique(grid) if int(v) != BG]:
        for comp in _components(grid == color):
            if not any(r in (0, h - 1) or c in (0, w - 1) for r, c in comp):
                continue
            r0, c0, r1, c1 = _bbox(comp)
            sub = grid[r0 : r1 + 1, c0 : c1 + 1]
            if not (
                np.all(sub[0, :] == color)
                and np.all(sub[-1, :] == color)
                and np.all(sub[:, 0] == color)
                and np.all(sub[:, -1] == color)
            ):
                continue

            mapping = _try_parse_horizontal_legend(sub, color)
            if mapping is None:
                mapping = _try_parse_vertical_legend(sub, color)
            if mapping is None:
                continue

            mask = np.zeros_like(grid, dtype=bool)
            mask[r0 : r1 + 1, c0 : c1 + 1] = True
            candidates.append((len(mapping), color, mapping, mask))

    assert len(candidates) == 1, "expected exactly one legend"
    _, wall, mapping, mask = candidates[0]
    return wall, mapping, mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the edge legend, map seed colors to 3x3 binary templates,
    and tile each enclosed region with the mapped template anchored
    at the seed pixel.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    wall, mapping, legend_mask = _find_legend(g)
    usable = (g != wall) & ~legend_mask

    out = g.copy()
    for comp in _components(usable):
        if any(
            r == 0 or c == 0 or r == g.shape[0] - 1 or c == g.shape[1] - 1
            for r, c in comp
        ):
            continue

        colored = [(r, c) for r, c in comp if g[r, c] not in (BG, wall)]
        if len(colored) != 1:
            continue

        seed_r, seed_c = colored[0]
        seed_color = int(g[seed_r, seed_c])
        assert seed_color in mapping, "seed color missing from legend"
        pattern = mapping[seed_color]

        for r, c in comp:
            out[r, c] = seed_color if pattern[(r - seed_r) % 3, (c - seed_c) % 3] else BG

    return out


_BASE_PATTERNS = [
    np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
    np.array([[1, 0, 1], [1, 0, 1], [1, 0, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=bool),
    np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[0, 0, 1], [0, 0, 1], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 0], [1, 0, 0]], dtype=bool),
]


def _all_patterns() -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen = set()
    for base in _BASE_PATTERNS:
        for k in range(4):
            arr = np.rot90(base, k)
            key = tuple(int(v) for v in arr.flatten())
            if key not in seen:
                seen.add(key)
                variants.append(arr.copy())
            flipped = np.fliplr(arr)
            key = tuple(int(v) for v in flipped.flatten())
            if key not in seen:
                seen.add(key)
                variants.append(flipped.copy())
    return variants


_PATTERN_BANK = _all_patterns()


def _dilate4(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros_like(mask, dtype=bool)
    pts = np.argwhere(mask)
    for r0, c0 in pts:
        r = int(r0)
        c = int(c0)
        out[r, c] = True
        if r > 0:
            out[r - 1, c] = True
        if r + 1 < h:
            out[r + 1, c] = True
        if c > 0:
            out[r, c - 1] = True
        if c + 1 < w:
            out[r, c + 1] = True
    return out


def _random_polyomino(rng: random.Random, h: int, w: int) -> np.ndarray:
    target = rng.randint(max(7, (h * w) // 3), max(8, (2 * h * w) // 3))
    mask = np.zeros((h, w), dtype=bool)
    start = (rng.randrange(h), rng.randrange(w))
    mask[start] = True
    frontier: set[tuple[int, int]] = set()
    for nr, nc in _neighbors4(*start):
        if 0 <= nr < h and 0 <= nc < w:
            frontier.add((nr, nc))

    while int(mask.sum()) < target and frontier:
        choices = list(frontier)
        weights = []
        for r, c in choices:
            weight = 1
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                    weight += 3
            weights.append(weight)
        cell = rng.choices(choices, weights=weights, k=1)[0]
        frontier.remove(cell)
        r, c = cell
        mask[r, c] = True
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                frontier.add((nr, nc))

    if h >= 5 and w >= 5 and rng.random() < 0.35:
        carved = False
        for _ in range(10):
            r = rng.randrange(1, h - 1)
            c = rng.randrange(1, w - 1)
            if not mask[r, c]:
                continue
            mask[r, c] = False
            if len(_components(mask)) == 1 and int(mask.sum()) >= 7:
                carved = True
                break
            mask[r, c] = True
        if not carved:
            pass

    return mask


def _build_region_patch(rng: random.Random) -> tuple[np.ndarray, tuple[int, int]]:
    inner_h = rng.randint(4, 8)
    inner_w = rng.randint(4, 8)
    inside = _random_polyomino(rng, inner_h, inner_w)

    patch = np.zeros((inner_h + 2, inner_w + 2), dtype=int)
    inset = np.zeros_like(patch, dtype=bool)
    inset[1:-1, 1:-1] = inside
    boundary = _dilate4(inset) & ~inset
    patch[boundary] = 1

    cells = [(int(r), int(c)) for r, c in np.argwhere(inset)]
    seed_r, seed_c = rng.choice(cells)
    return patch, (seed_r, seed_c)


def _reserve(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int):
    rr0 = max(0, r0)
    cc0 = max(0, c0)
    rr1 = min(mask.shape[0] - 1, r1)
    cc1 = min(mask.shape[1] - 1, c1)
    mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _place_legend(
    grid: np.ndarray,
    rng: random.Random,
    wall: int,
    color_to_pattern: dict[int, np.ndarray],
) -> np.ndarray:
    h, w = grid.shape
    k = len(color_to_pattern)
    horizontal = rng.random() < 0.5
    legend_mask = np.zeros_like(grid, dtype=bool)

    if horizontal:
        legend_h, legend_w = 5, 4 * k + 1
        side = rng.choice(["top", "bottom"])
        r0 = 0 if side == "top" else h - legend_h
        c0 = rng.randint(0, w - legend_w)
        grid[r0 : r0 + legend_h, c0 : c0 + legend_w] = wall
        for i, (color, pattern) in enumerate(color_to_pattern.items()):
            rr = r0 + 1
            cc = c0 + 1 + 4 * i
            grid[rr : rr + 3, cc : cc + 3] = np.where(pattern, color, BG)
        legend_mask[r0 : r0 + legend_h, c0 : c0 + legend_w] = True
    else:
        legend_h, legend_w = 4 * k + 1, 5
        side = rng.choice(["left", "right"])
        r0 = rng.randint(0, h - legend_h)
        c0 = 0 if side == "left" else w - legend_w
        grid[r0 : r0 + legend_h, c0 : c0 + legend_w] = wall
        for i, (color, pattern) in enumerate(color_to_pattern.items()):
            rr = r0 + 1 + 4 * i
            cc = c0 + 1
            grid[rr : rr + 3, cc : cc + 3] = np.where(pattern, color, BG)
        legend_mask[r0 : r0 + legend_h, c0 : c0 + legend_w] = True

    return legend_mask


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    wall = 1
    h = rng.randint(22, 30)
    w = rng.randint(22, 30)
    grid = np.zeros((h, w), dtype=int)

    palette = [c for c in range(2, 10) if c != wall]
    color_count = rng.randint(2, 4)
    colors = rng.sample(palette, color_count)
    patterns = rng.sample(_PATTERN_BANK, color_count)
    color_to_pattern = {color: pattern.copy() for color, pattern in zip(colors, patterns)}

    legend_mask = _place_legend(grid, rng, wall, color_to_pattern)
    reserved = np.zeros((h, w), dtype=bool)
    for r, c in np.argwhere(legend_mask):
        _reserve(reserved, int(r) - 1, int(c) - 1, int(r) + 1, int(c) + 1)

    region_count = rng.randint(color_count, min(color_count + 2, 5))
    region_colors = colors.copy()
    while len(region_colors) < region_count:
        region_colors.append(rng.choice(colors))
    rng.shuffle(region_colors)

    for seed_color in region_colors:
        placed = False
        for _ in range(80):
            patch, (seed_r, seed_c) = _build_region_patch(rng)
            ph, pw = patch.shape
            if h - ph - 2 < 1 or w - pw - 2 < 1:
                return None
            top = rng.randint(1, h - ph - 1)
            left = rng.randint(1, w - pw - 1)
            bottom = top + ph - 1
            right = left + pw - 1
            if np.any(reserved[top - 1 : bottom + 2, left - 1 : right + 2]):
                continue

            target = grid[top : bottom + 1, left : right + 1]
            if np.any(target != BG):
                continue

            target[patch == wall] = wall
            grid[top + seed_r, left + seed_c] = seed_color
            _reserve(reserved, top - 1, left - 1, bottom + 1, right + 1)
            placed = True
            break
        if not placed:
            return None

    wall_mask = grid == wall
    if len(_components(wall_mask)) < 2:
        return None

    usable = (grid != wall) & ~legend_mask
    enclosed = [
        comp
        for comp in _components(usable)
        if not any(r in (0, h - 1) or c in (0, w - 1) for r, c in comp)
    ]
    if len(enclosed) != region_count:
        return None
    if any(sum(grid[r, c] not in (BG, wall) for r, c in comp) != 1 for comp in enclosed):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
