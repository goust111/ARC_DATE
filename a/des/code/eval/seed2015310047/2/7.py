import random
from collections import Counter

import numpy as np


WALL_COLOR = 0
SEPARATOR_COLOR = 5
TEXTURE_SIZES = (3, 4)
GEN_RETRIES = 400


def _most_common_color(values: np.ndarray) -> int:
    counts = Counter(map(int, np.asarray(values).reshape(-1)))
    assert counts
    return int(max(counts.items(), key=lambda kv: kv[1])[0])


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _touches_border(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = map(int, shape)
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 0] == h - 1)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 1] == w - 1)
    )


def _find_separator_row(grid: np.ndarray, bg: int) -> int:
    candidates = []
    h = int(grid.shape[0])
    for r in range(h):
        if int(grid[r, 0]) == bg:
            continue
        if np.all(grid[r, :] == grid[r, 0]):
            if r > 0 and r + 1 < h and np.any(grid[:r, :] != bg) and np.any(grid[r + 1 :, :] != bg):
                candidates.append(r)
    assert len(candidates) == 1, "expected exactly one full-width separator row"
    return int(candidates[0])


def _extract_palette(top: np.ndarray, bg: int) -> dict[int, np.ndarray]:
    comps = _components(top != bg)
    palette: dict[int, np.ndarray] = {}
    assert comps, "palette area must contain samples"
    for comp in comps:
        r0, c0, r1, c1 = _bbox(comp)
        sub = top[r0 : r1 + 1, c0 : c1 + 1]
        h, w = map(int, sub.shape)
        assert h in TEXTURE_SIZES and w - 1 in TEXTURE_SIZES
        assert h == w - 1, "texture sample must be square with one extra key column"
        left = sub[:, 0]
        key_rows = np.flatnonzero(left != bg)
        assert len(key_rows) == 1, "expected exactly one key pixel"
        assert np.all(sub[:, 1:] != bg), "texture sample must be dense"
        texture = sub[:, 1:].copy()
        assert texture.shape[0] == texture.shape[1] and texture.shape[0] in TEXTURE_SIZES
        assert len(np.unique(texture)) >= 2, "texture should be visibly multi-color"
        key_color = int(left[int(key_rows[0])])
        assert key_color not in palette, "key colors must be unique"
        palette[key_color] = texture
    return palette


def _infer_wall_color(main_area: np.ndarray, bg: int) -> int:
    vals = main_area[main_area != bg]
    assert vals.size > 0, "main area must contain walls and seeds"
    colors, counts = np.unique(vals, return_counts=True)
    return int(colors[int(np.argmax(counts))])


def _fill_region(
    out: np.ndarray,
    coords: np.ndarray,
    seed_color: int,
    bg: int,
    texture: np.ndarray,
) -> None:
    bg_cells = [(int(r), int(c)) for r, c in coords if int(out[int(r), int(c)]) == bg]
    assert bg_cells, "each region must contain background cells to fill"
    anchor_r, anchor_c = min(bg_cells)
    th, tw = map(int, texture.shape)
    for r, c in bg_cells:
        out[r, c] = int(texture[(r - anchor_r) % th, (c - anchor_c) % tw])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the top palette, match each seed color to a texture tile, and tile-fill
    every enclosed region from its top-left background cell anchor.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"

    bg = _most_common_color(g)
    sep_row = _find_separator_row(g, bg)
    palette = _extract_palette(g[:sep_row, :], bg)

    main_area = g[sep_row + 1 :, :].copy()
    wall = _infer_wall_color(main_area, bg)
    passable = main_area != wall
    out = main_area.copy()

    for comp in _components(passable):
        if _touches_border(comp, main_area.shape):
            continue
        seed_coords = [(int(r), int(c)) for r, c in comp if int(main_area[int(r), int(c)]) != bg]
        assert len(seed_coords) == 1, "each enclosed region must contain exactly one seed"
        seed_r, seed_c = seed_coords[0]
        seed_color = int(main_area[seed_r, seed_c])
        assert seed_color in palette, "seed color must match a palette key"
        _fill_region(out, comp, seed_color, bg, palette[seed_color])

    return out


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    assert len(pts) > 0
    r0, c0, r1, c1 = _bbox(pts)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _has_hole(mask: np.ndarray) -> bool:
    comps = _components(~mask)
    for comp in comps:
        if not _touches_border(comp, mask.shape):
            return True
    return False


def _dilate8(mask: np.ndarray) -> np.ndarray:
    h, w = map(int, mask.shape)
    out = np.zeros((h, w), dtype=bool)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            r0 = max(0, -dr)
            r1 = min(h, h - dr)
            c0 = max(0, -dc)
            c1 = min(w, w - dc)
            out[r0 + dr : r1 + dr, c0 + dc : c1 + dc] |= mask[r0:r1, c0:c1]
    return out


def _build_region_patch(interior: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    interior = _crop_mask(interior)
    padded = np.zeros((interior.shape[0] + 2, interior.shape[1] + 2), dtype=bool)
    padded[1:-1, 1:-1] = interior
    wall = _dilate8(padded) & ~padded
    comps = _components(~wall)
    target = None
    start = tuple(map(int, np.argwhere(padded)[0]))
    for comp in comps:
        comp_set = {tuple(map(int, xy)) for xy in comp}
        if start in comp_set:
            target = comp
            break
    assert target is not None
    if _touches_border(target, wall.shape):
        return None
    return wall, padded


def _random_texture(size: int, pool: list[int]) -> np.ndarray:
    n_colors = random.randint(2, min(3, len(pool)))
    colors = random.sample(pool, n_colors)
    for _ in range(100):
        tex = np.array([[random.choice(colors) for _ in range(size)] for _ in range(size)], dtype=int)
        if len(np.unique(tex)) >= 2:
            return tex
    raise RuntimeError("failed to generate a nontrivial texture")


def _random_blob() -> np.ndarray | None:
    for _ in range(80):
        h = random.randint(4, 8)
        w = random.randint(4, 10)
        area = random.randint(max(6, (h * w) // 2), h * w - 1)
        mask = np.zeros((h, w), dtype=bool)
        sr = random.randrange(h)
        sc = random.randrange(w)
        mask[sr, sc] = True
        while int(mask.sum()) < area:
            frontier = set()
            for r, c in np.argwhere(mask):
                for nr, nc in _neighbors4(int(r), int(c), h, w):
                    if not mask[nr, nc]:
                        frontier.add((nr, nc))
            if not frontier:
                break
            rr, cc = random.choice(tuple(frontier))
            mask[rr, cc] = True
        mask = _crop_mask(mask)
        if mask.shape[0] < 3 or mask.shape[1] < 3:
            continue
        if mask.all() or _has_hole(mask):
            continue
        return mask
    return None


def _random_region_interior() -> np.ndarray | None:
    kind = random.choice(("rect", "blob"))
    if kind == "rect":
        h = random.randint(3, 7)
        w = random.randint(4, 9)
        return np.ones((h, w), dtype=bool)
    return _random_blob()


def _choose_seed(cells: list[tuple[int, int]]) -> tuple[int, int]:
    ordered = sorted(cells)
    if len(ordered) > 1 and random.random() < 0.7:
        return random.choice(ordered[1:])
    return random.choice(ordered)


def _expected_output(main_area: np.ndarray, specs: list[dict], bg: int) -> np.ndarray:
    out = main_area.copy()
    for spec in specs:
        seed = spec["seed"]
        bg_cells = [cell for cell in spec["cells"] if cell != seed]
        anchor_r, anchor_c = min(bg_cells)
        tex = spec["texture"]
        th, tw = map(int, tex.shape)
        for r, c in bg_cells:
            out[r, c] = int(tex[(r - anchor_r) % th, (c - anchor_c) % tw])
    return out


def _place_regions(
    main_h: int,
    main_w: int,
    assignments: list[int],
    palette: dict[int, np.ndarray],
    bg: int,
) -> tuple[np.ndarray, list[dict]] | None:
    main_area = np.full((main_h, main_w), bg, dtype=int)
    reserved = np.zeros((main_h, main_w), dtype=bool)
    specs = []

    for key_color in assignments:
        interior = _random_region_interior()
        if interior is None:
            return None
        built = _build_region_patch(interior)
        if built is None:
            return None
        wall_mask, interior_mask = built
        ph, pw = map(int, wall_mask.shape)
        if ph + 2 > main_h or pw + 2 > main_w:
            return None

        placed = False
        for _ in range(120):
            top = random.randint(1, main_h - ph - 1)
            left = random.randint(1, main_w - pw - 1)
            r0 = max(0, top - 1)
            c0 = max(0, left - 1)
            r1 = min(main_h, top + ph + 1)
            c1 = min(main_w, left + pw + 1)
            if np.any(reserved[r0:r1, c0:c1]):
                continue

            patch = main_area[top : top + ph, left : left + pw]
            patch[wall_mask] = WALL_COLOR

            cells = [(top + int(r), left + int(c)) for r, c in np.argwhere(interior_mask)]
            seed = _choose_seed(cells)
            main_area[seed[0], seed[1]] = key_color
            specs.append({"cells": cells, "seed": seed, "texture": palette[key_color]})
            reserved[top : top + ph, left : left + pw] = True
            placed = True
            break

        if not placed:
            return None

    return main_area, specs


def _assemble_grid(
    palette_entries: list[tuple[int, np.ndarray, int, int, int]],
    width: int,
    palette_h: int,
    main_area: np.ndarray,
    bg: int,
) -> np.ndarray:
    grid = np.full((palette_h + 1 + main_area.shape[0], width), bg, dtype=int)
    for key_color, texture, row0, left0, key_row in palette_entries:
        size = int(texture.shape[0])
        grid[row0 : row0 + size, left0 + 1 : left0 + 1 + size] = texture
        grid[row0 + key_row, left0] = key_color
    grid[palette_h, :] = SEPARATOR_COLOR
    grid[palette_h + 1 :, :] = main_area
    return grid


def _gen_attempt() -> np.ndarray | None:
    bg_choices = [c for c in range(10) if c not in {WALL_COLOR, SEPARATOR_COLOR}]
    bg = random.choice(bg_choices)
    free_colors = [c for c in range(10) if c not in {bg, WALL_COLOR, SEPARATOR_COLOR}]

    n_palette = random.randint(2, 4)
    key_colors = random.sample(free_colors, n_palette)

    textures: dict[int, np.ndarray] = {}
    signatures = set()
    for key in key_colors:
        for _ in range(50):
            size = random.choice(TEXTURE_SIZES)
            tex = _random_texture(size, free_colors)
            sig = (size, tuple(map(int, tex.reshape(-1))))
            if sig not in signatures:
                signatures.add(sig)
                textures[key] = tex
                break
        else:
            return None

    sizes = [int(textures[key].shape[0]) for key in key_colors]
    palette_h = max(sizes) + random.randint(1, 2)
    left_margin = random.randint(1, 3)
    right_margin = random.randint(1, 3)
    gaps = [random.randint(1, 3) for _ in range(n_palette - 1)]
    base_palette_w = left_margin + right_margin + sum(size + 1 for size in sizes) + sum(gaps)

    region_count = n_palette + random.randint(0, 1)
    assignments = key_colors.copy()
    while len(assignments) < region_count:
        assignments.append(random.choice(key_colors))
    random.shuffle(assignments)

    min_main_h = 16 + 2 * max(0, region_count - 2)
    min_main_w = max(base_palette_w, 20 + 2 * max(0, region_count - 2))
    max_main_h = max(min_main_h, 26)
    max_main_w = max(min_main_w, 30)
    if min_main_h > max_main_h or min_main_w > max_main_w:
        return None
    main_h = random.randint(min_main_h, max_main_h)
    main_w = random.randint(min_main_w, max_main_w)

    placed = _place_regions(main_h, main_w, assignments, textures, bg)
    if placed is None:
        return None
    main_area, specs = placed

    extra_shift = random.randint(0, main_w - base_palette_w)
    x = left_margin + extra_shift
    palette_entries = []
    for idx, key in enumerate(key_colors):
        tex = textures[key]
        size = int(tex.shape[0])
        row0 = random.randint(0, palette_h - size)
        key_row = random.randint(0, size - 1)
        palette_entries.append((key, tex, row0, x, key_row))
        x += size + 1
        if idx + 1 < n_palette:
            x += gaps[idx]

    grid = _assemble_grid(palette_entries, main_w, palette_h, main_area, bg)
    expected = _expected_output(main_area, specs, bg)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
