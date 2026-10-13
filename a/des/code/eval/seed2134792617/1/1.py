from __future__ import annotations

from collections import Counter, deque

import numpy as np


PATTERN_TEMPLATES = {
    2: [
        np.array([[0, 1], [1, 0]], dtype=int),
        np.array([[0, 1], [0, 1]], dtype=int),
        np.array([[0, 0], [1, 1]], dtype=int),
    ],
    3: [
        np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=int),
        np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=int),
        np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=int),
        np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=int),
        np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=int),
    ],
}

COLOR_POOL = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 400


def _most_common_color(grid: np.ndarray) -> int:
    counts = Counter(map(int, grid.reshape(-1)))
    return int(counts.most_common(1)[0][0])


def _segments(mask: np.ndarray) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = None
    for idx, active in enumerate(mask.tolist()):
        if active and start is None:
            start = idx
        elif not active and start is not None:
            spans.append((int(start), int(idx)))
            start = None
    if start is not None:
        spans.append((int(start), int(mask.shape[0])))
    return spans


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(int(r), int(c))])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _parse_legend(grid: np.ndarray, bg: int) -> tuple[dict[int, np.ndarray], np.ndarray]:
    h, w = grid.shape
    best: tuple[int, dict[int, np.ndarray], np.ndarray] | None = None

    for tile_size in (2, 3):
        legend_h = tile_size + 1
        if h < legend_h:
            continue
        top = h - legend_h
        region = grid[top:, :]
        active_cols = np.any(region != bg, axis=0)
        spans = _segments(active_cols)
        if len(spans) < 2:
            continue

        mapping: dict[int, np.ndarray] = {}
        legend_mask = np.zeros_like(grid, dtype=bool)
        valid = True
        for c0, c1 in spans:
            if c1 - c0 != tile_size:
                valid = False
                break
            block = region[:, c0:c1]
            key_row = block[0]
            if np.any(key_row == bg) or len(np.unique(key_row)) != 1:
                valid = False
                break
            key = int(key_row[0])
            if key in mapping:
                valid = False
                break
            pattern = np.asarray(block[1:, :], dtype=int)
            vals = {int(v) for v in np.unique(pattern)}
            if len(vals) != 2 or bg in vals:
                valid = False
                break
            mapping[key] = pattern.copy()
            legend_mask[top:, c0:c1] = True

        if valid:
            score = len(mapping)
            if best is None or score > best[0]:
                best = (score, mapping, legend_mask)

    assert best is not None, "expected a bottom legend made of 2x2 or 3x3 pattern blocks"
    return best[1], best[2]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each bordered interior with the periodic pattern selected by its seed color.

    The bottom legend consists of horizontal blocks. Each block has a solid top row whose
    color matches one seed, and a 2x2 or 3x3 pattern below it. The pattern is tiled over
    the entire enclosed interior of the matching border, and the whole legend is erased.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = _most_common_color(g)
    mapping, legend_mask = _parse_legend(g, bg)

    out = g.copy()
    out[legend_mask] = bg

    seed_colors = np.array(sorted(mapping), dtype=int)
    passable = (out == bg) | np.isin(out, seed_colors)

    for comp in _components(passable):
        if any(r == 0 or c == 0 or r == out.shape[0] - 1 or c == out.shape[1] - 1 for r, c in comp):
            continue

        seeds = [(r, c) for r, c in comp if int(out[r, c]) in mapping]
        if not seeds:
            continue
        assert len(seeds) == 1, "each enclosed region must contain exactly one seed"

        seed_color = int(out[seeds[0]])
        pattern = mapping[seed_color]
        ph, pw = pattern.shape
        r0, c0, _r1, _c1 = _bbox(comp)
        for r, c in comp:
            out[r, c] = int(pattern[(r - r0) % ph, (c - c0) % pw])

    return out


def _rand_int(rng: np.random.Generator, lo: int, hi: int) -> int:
    assert lo <= hi
    return lo if lo == hi else int(rng.integers(lo, hi + 1))


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _random_blob_mask(rng: np.random.Generator) -> np.ndarray:
    for _ in range(120):
        target_area = _rand_int(rng, 5, 10)
        cells = {(0, 0)}
        while len(cells) < target_area:
            frontier: list[tuple[int, int]] = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if (nr, nc) not in cells:
                        frontier.append((nr, nc))
            pick = frontier[int(rng.integers(len(frontier)))]
            cells.add(pick)

        rows = [r for r, _c in cells]
        cols = [c for _r, c in cells]
        r_min, c_min = min(rows), min(cols)
        normalized = {(r - r_min, c - c_min) for r, c in cells}
        h = max(r for r, _c in normalized) + 1
        w = max(c for _r, c in normalized) + 1
        if h < 3 or w < 3 or h > 6 or w > 6:
            continue
        mask = np.zeros((h, w), dtype=bool)
        for r, c in normalized:
            mask[r, c] = True
        density = float(mask.mean())
        if density < 0.35 or density > 0.8:
            continue
        if int(mask.sum()) == mask.size:
            continue
        return mask
    raise RuntimeError("failed to create blob mask")


def _rectangle_mask(rng: np.random.Generator) -> np.ndarray:
    h = _rand_int(rng, 3, 6)
    w = _rand_int(rng, 3, 6)
    return np.ones((h, w), dtype=bool)


def _l_mask(rng: np.random.Generator) -> np.ndarray:
    h = _rand_int(rng, 4, 6)
    w = _rand_int(rng, 4, 6)
    t = _rand_int(rng, 1, min(2, h - 2, w - 2))
    mask = np.zeros((h, w), dtype=bool)
    mask[:, :t] = True
    mask[-t:, :] = True
    mask = np.rot90(mask, int(rng.integers(4)))
    if bool(rng.integers(2)):
        mask = np.fliplr(mask)
    if bool(rng.integers(2)):
        mask = np.flipud(mask)
    return _normalize_mask(mask)


def _sample_shape_mask(rng: np.random.Generator) -> np.ndarray:
    family = int(rng.integers(3))
    if family == 0:
        mask = _rectangle_mask(rng)
    elif family == 1:
        mask = _l_mask(rng)
    else:
        mask = _random_blob_mask(rng)

    if bool(rng.integers(2)):
        mask = np.fliplr(mask)
    if bool(rng.integers(2)):
        mask = np.flipud(mask)
    return _normalize_mask(mask)


def _pattern_from_template(
    rng: np.random.Generator,
    tile_size: int,
    colors: tuple[int, int],
) -> np.ndarray:
    templates = PATTERN_TEMPLATES[tile_size]
    binary = templates[int(rng.integers(len(templates)))].copy()
    binary = np.rot90(binary, int(rng.integers(4)))
    if bool(rng.integers(2)):
        binary = 1 - binary
    a, b = colors
    return np.where(binary == 0, a, b).astype(int)


def _shell_border(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    shell = np.zeros((h + 2, w + 2), dtype=bool)
    padded = np.zeros((h + 2, w + 2), dtype=bool)
    padded[1:-1, 1:-1] = mask
    for r, c in np.argwhere(mask):
        rr = int(r) + 1
        cc = int(c) + 1
        shell[rr - 1 : rr + 2, cc - 1 : cc + 2] = True
    shell[padded] = False
    return shell


def _render_object_patch(
    mask: np.ndarray,
    bg: int,
    border_color: int,
    seed_color: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, tuple[int, int]]:
    patch = np.full((mask.shape[0] + 2, mask.shape[1] + 2), bg, dtype=int)
    patch[_shell_border(mask)] = border_color
    coords = np.argwhere(mask)
    seed_r, seed_c = coords[int(rng.integers(len(coords)))]
    patch[int(seed_r) + 1, int(seed_c) + 1] = seed_color
    return patch, (int(seed_r), int(seed_c))


def _place_objects(
    rng: np.random.Generator,
    grid: np.ndarray,
    objects: list[dict[str, object]],
    usable_bottom: int,
) -> bool:
    h, w = grid.shape
    occupied = np.zeros((h, w), dtype=bool)

    for obj in sorted(objects, key=lambda item: int(np.prod(item["patch"].shape)), reverse=True):
        patch = obj["patch"]
        ph, pw = patch.shape
        max_r0 = usable_bottom - ph
        max_c0 = w - pw - 1
        if max_r0 < 1 or max_c0 < 1:
            return False

        placed = False
        for _ in range(160):
            r0 = _rand_int(rng, 1, max_r0)
            c0 = _rand_int(rng, 1, max_c0)
            r1 = r0 + ph
            c1 = c0 + pw
            rr0 = r0 - 1
            cc0 = c0 - 1
            rr1 = min(h, r1 + 1)
            cc1 = min(w, c1 + 1)
            if np.any(occupied[rr0:rr1, cc0:cc1]):
                continue

            region = grid[r0:r1, c0:c1]
            region[patch != 0] = patch[patch != 0]
            occupied[r0:r1, c0:c1] = True
            obj["top"] = int(r0)
            obj["left"] = int(c0)
            placed = True
            break

        if not placed:
            return False

    return True


def _build_expected(
    inp: np.ndarray,
    objects: list[dict[str, object]],
    legend_top: int,
) -> np.ndarray:
    out = inp.copy()
    out[legend_top:, :] = 0
    for obj in objects:
        mask = obj["mask"]
        top = int(obj["top"])
        left = int(obj["left"])
        pattern = obj["pattern"]
        ph, pw = pattern.shape
        for r, c in np.argwhere(mask):
            rr = top + 1 + int(r)
            cc = left + 1 + int(c)
            out[rr, cc] = int(pattern[int(r) % ph, int(c) % pw])
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    bg = 0
    tile_size = 2 if bool(rng.integers(2)) else 3
    legend_h = tile_size + 1
    object_count = _rand_int(rng, 2, 4)

    border_color = int(rng.choice(COLOR_POOL))
    remaining = [c for c in COLOR_POOL if c != border_color]
    seed_colors = [int(c) for c in rng.choice(remaining, size=object_count, replace=False)]
    pattern_pool = [c for c in remaining if c not in seed_colors]
    if len(pattern_pool) < 2:
        return None

    objects: list[dict[str, object]] = []
    for seed_color in seed_colors:
        mask = _sample_shape_mask(rng)
        pair = tuple(int(c) for c in rng.choice(pattern_pool, size=2, replace=False))
        pattern = _pattern_from_template(rng, tile_size, pair)
        patch, _seed_pos = _render_object_patch(mask, bg, border_color, seed_color, rng)
        objects.append(
            {
                "seed": int(seed_color),
                "mask": mask,
                "pattern": pattern,
                "patch": patch,
            }
        )

    gaps = [_rand_int(rng, 1, 2) for _ in range(object_count - 1)]
    legend_w = object_count * tile_size + sum(gaps)
    max_patch_h = max(int(obj["patch"].shape[0]) for obj in objects)
    max_patch_w = max(int(obj["patch"].shape[1]) for obj in objects)

    min_h = max(18, max_patch_h + legend_h + 4)
    min_w = max(20, max_patch_w + 3, legend_w + 3)
    if min_h > 28 or min_w > 30:
        return None

    h = _rand_int(rng, min_h, 28)
    w = _rand_int(rng, min_w, 30)
    grid = np.full((h, w), bg, dtype=int)

    legend_top = h - legend_h
    usable_bottom = legend_top - 1
    if not _place_objects(rng, grid, objects, usable_bottom):
        return None

    legend_left = _rand_int(rng, 1, w - legend_w - 1)
    order = objects.copy()
    rng.shuffle(order)
    col = legend_left
    for idx, obj in enumerate(order):
        seed = int(obj["seed"])
        pattern = np.asarray(obj["pattern"], dtype=int)
        grid[legend_top, col : col + tile_size] = seed
        grid[legend_top + 1 : legend_top + 1 + tile_size, col : col + tile_size] = pattern
        if idx < len(gaps):
            col += tile_size + gaps[idx]

    expected = _build_expected(grid, objects, legend_top)

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
