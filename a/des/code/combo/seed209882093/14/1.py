import numpy as np

from collections import Counter


BLACK = 0
GREY = 5
MAROON = 9

NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
PATTERN_LAYOUTS = (
    ((0, 1), (1, 0)),
    ((0, 0), (1, 1)),
    ((0, 1), (0, 1)),
    ((0, 0), (0, 1)),
    ((0, 1), (1, 1)),
    ((0, 1), (0, 0)),
    ((0, 0), (1, 0)),
)
PALETTE_COLORS = (1, 2, 3, 4, 6, 7, 8)
GEN_RETRIES = 300

_RNG = np.random.default_rng()


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
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _touches_border(comp: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp)


def _split_input(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    grey_rows = [r for r in range(h) if np.all(g[r] == GREY)]
    grey_cols = [c for c in range(w) if np.all(g[:, c] == GREY)]

    if grey_rows:
        start = grey_rows[0]
        end = start
        while end + 1 < h and np.all(g[end + 1] == GREY):
            end += 1
        legend = g[:start, :]
        main = g[end + 1 :, :]
        assert legend.size > 0 and main.size > 0
        return legend, main

    if grey_cols:
        start = grey_cols[0]
        end = start
        while end + 1 < w and np.all(g[:, end + 1] == GREY):
            end += 1
        legend = g[:, :start]
        main = g[:, end + 1 :]
        assert legend.size > 0 and main.size > 0
        return legend, main

    raise AssertionError("expected a full grey divider row or column")


def _most_common(values: list[int]) -> int:
    return Counter(values).most_common(1)[0][0]


def _extract_palette(legend: np.ndarray) -> dict[int, np.ndarray]:
    mask = (legend != BLACK) & (legend != GREY)
    comps = _components(mask)
    mapping: dict[int, np.ndarray] = {}

    for comp in comps:
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        key = int(legend[r0, c0])
        assert key not in (BLACK, GREY, MAROON)

        pattern = np.zeros((2, 2), dtype=int)
        by_parity: dict[tuple[int, int], list[int]] = {(0, 0): [], (0, 1): [], (1, 0): [], (1, 1): []}
        for r, c in comp:
            by_parity[((r - r0) % 2, (c - c0) % 2)].append(int(legend[r, c]))

        for pr in range(2):
            for pc in range(2):
                vals = by_parity[(pr, pc)]
                assert vals
                pattern[pr, pc] = _most_common(vals)

        pattern[0, 0] = key
        mapping[key] = pattern

    assert mapping, "legend must contain at least one swatch"
    return mapping


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    legend, main = _split_input(g)
    mapping = _extract_palette(legend)

    h, w = main.shape
    out = main.copy()
    passable = main != MAROON

    for comp in _components(passable):
        if _touches_border(comp, h, w):
            continue

        seeds = [(r, c, int(main[r, c])) for r, c in comp if int(main[r, c]) != BLACK]
        assert len(seeds) == 1, "each enclosed region must contain exactly one seed"
        sr, sc, seed_color = seeds[0]
        assert seed_color in mapping, "seed color must exist in the legend"
        pattern = mapping[seed_color]

        for r, c in comp:
            out[r, c] = int(pattern[(r - sr) % 2, (c - sc) % 2])

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove the legend, then fill each maroon-enclosed region using the 2x2
    texture keyed by its seed color and anchored at the seed position.
    """
    return _solve(grid)


def _randint(rng: np.random.Generator, lo: int, hi: int) -> int:
    return int(rng.integers(lo, hi + 1))


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _has_hole(mask: np.ndarray) -> bool:
    bg = ~mask
    for comp in _components(bg):
        if not _touches_border(comp, *mask.shape):
            return True
    return False


def _make_rectilinear_mask(rng: np.random.Generator) -> np.ndarray:
    for _ in range(80):
        if rng.random() < 0.4:
            h = _randint(rng, 2, 5)
            w = _randint(rng, 2, 6)
            mask = np.ones((h, w), dtype=bool)
            if h >= 3 and w >= 3 and rng.random() < 0.7:
                cut_h = _randint(rng, 1, h - 1)
                cut_w = _randint(rng, 1, w - 1)
                corner = _randint(rng, 0, 3)
                if corner == 0:
                    mask[:cut_h, :cut_w] = False
                elif corner == 1:
                    mask[:cut_h, w - cut_w :] = False
                elif corner == 2:
                    mask[h - cut_h :, :cut_w] = False
                else:
                    mask[h - cut_h :, w - cut_w :] = False
            if mask.sum() >= 4:
                return mask

        h = _randint(rng, 2, 6)
        w = _randint(rng, 2, 7)
        target = _randint(rng, 4, min(h * w, 14))
        cells = {(int(rng.integers(h)), int(rng.integers(w)))}
        while len(cells) < target:
            frontier: dict[tuple[int, int], int] = {}
            for r, c in cells:
                for dr, dc in NEIGHBORS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier[(nr, nc)] = frontier.get((nr, nc), 0) + 1
            if not frontier:
                break
            opts = list(frontier.items())
            weights = np.array([score for _, score in opts], dtype=float)
            weights /= weights.sum()
            choice = int(rng.choice(len(opts), p=weights))
            cells.add(opts[choice][0])

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _trim_mask(mask)
        if mask.shape[0] < 2 or mask.shape[1] < 2:
            continue
        if _has_hole(mask):
            continue
        if int(mask.sum()) < 4:
            continue
        return mask

    raise RuntimeError("failed to generate a valid interior mask")


def _region_blueprint(mask: np.ndarray) -> dict[str, object]:
    ih, iw = mask.shape
    interior = {(r + 1, c + 1) for r, c in np.argwhere(mask)}
    boundary: set[tuple[int, int]] = set()
    for r, c in interior:
        for dr, dc in NEIGHBORS4:
            nbr = (r + dr, c + dc)
            if nbr not in interior:
                boundary.add(nbr)
    return {
        "h": ih + 2,
        "w": iw + 2,
        "interior": sorted(interior),
        "boundary": sorted(boundary),
        "occupied": sorted(interior | boundary),
    }


def _can_place(
    busy: np.ndarray, occupied: list[tuple[int, int]], top: int, left: int
) -> bool:
    h, w = busy.shape
    for r, c in occupied:
        rr, cc = top + r, left + c
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = rr + dr, cc + dc
                if not (0 <= nr < h and 0 <= nc < w):
                    return False
                if busy[nr, nc]:
                    return False
    return True


def _mark_busy(
    busy: np.ndarray, occupied: list[tuple[int, int]], top: int, left: int
) -> None:
    for r, c in occupied:
        rr, cc = top + r, left + c
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                busy[rr + dr, cc + dc] = True


def _make_pattern(rng: np.random.Generator, key: int) -> np.ndarray:
    accent_choices = [c for c in PALETTE_COLORS if c != key]
    accent = int(rng.choice(accent_choices))
    layout = np.array(PATTERN_LAYOUTS[_randint(rng, 0, len(PATTERN_LAYOUTS) - 1)], dtype=int)
    pattern = np.where(layout == 0, key, accent).astype(int)
    pattern[0, 0] = key
    return pattern


def _make_swatch(pattern: np.ndarray, size: int) -> np.ndarray:
    patch = np.zeros((size, size), dtype=int)
    for r in range(size):
        for c in range(size):
            patch[r, c] = int(pattern[r % 2, c % 2])
    return patch


def _place_regions(
    rng: np.random.Generator, main_h: int, main_w: int, region_keys: list[int]
) -> np.ndarray | None:
    main = np.zeros((main_h, main_w), dtype=int)
    busy = np.zeros((main_h, main_w), dtype=bool)

    for key in region_keys:
        placed = False
        for _ in range(120):
            blueprint = _region_blueprint(_make_rectilinear_mask(rng))
            bh = int(blueprint["h"])
            bw = int(blueprint["w"])
            if bh + 2 > main_h or bw + 2 > main_w:
                continue

            top = _randint(rng, 0, main_h - bh)
            left = _randint(rng, 0, main_w - bw)
            occupied = blueprint["occupied"]
            if not _can_place(busy, occupied, top, left):
                continue

            for r, c in blueprint["boundary"]:
                main[top + r, left + c] = MAROON

            interior = [(top + r, left + c) for r, c in blueprint["interior"]]
            seed_r, seed_c = interior[_randint(rng, 0, len(interior) - 1)]
            main[seed_r, seed_c] = key
            _mark_busy(busy, occupied, top, left)
            placed = True
            break

        if not placed:
            return None

    return main


def _build_top_layout(
    rng: np.random.Generator, swatches: list[np.ndarray], main: np.ndarray
) -> np.ndarray:
    size = swatches[0].shape[0]
    gap = _randint(rng, 1, 2)
    pad = _randint(rng, 1, 2)
    needed_w = pad + len(swatches) * size + (len(swatches) - 1) * gap + pad
    legend_h = size + 2 * pad
    total_w = max(main.shape[1], needed_w + _randint(rng, 0, 3))

    if main.shape[1] != total_w:
        extra = total_w - main.shape[1]
        main = np.pad(main, ((0, 0), (0, extra)), constant_values=BLACK)

    legend = np.zeros((legend_h, total_w), dtype=int)
    start_c = _randint(rng, 0, total_w - needed_w)
    c = start_c + pad
    for swatch in swatches:
        legend[pad : pad + size, c : c + size] = swatch
        c += size + gap

    grid = np.zeros((legend_h + 1 + main.shape[0], total_w), dtype=int)
    grid[:legend_h] = legend
    grid[legend_h, :] = GREY
    grid[legend_h + 1 :] = main
    return grid


def _build_left_layout(
    rng: np.random.Generator, swatches: list[np.ndarray], main: np.ndarray
) -> np.ndarray:
    size = swatches[0].shape[0]
    gap = _randint(rng, 1, 2)
    pad = _randint(rng, 1, 2)
    needed_h = pad + len(swatches) * size + (len(swatches) - 1) * gap + pad
    legend_w = size + 2 * pad
    total_h = max(main.shape[0], needed_h + _randint(rng, 0, 3))

    if main.shape[0] != total_h:
        extra = total_h - main.shape[0]
        main = np.pad(main, ((0, extra), (0, 0)), constant_values=BLACK)

    legend = np.zeros((total_h, legend_w), dtype=int)
    start_r = _randint(rng, 0, total_h - needed_h)
    r = start_r + pad
    for swatch in swatches:
        legend[r : r + size, pad : pad + size] = swatch
        r += size + gap

    grid = np.zeros((total_h, legend_w + 1 + main.shape[1]), dtype=int)
    grid[:, :legend_w] = legend
    grid[:, legend_w] = GREY
    grid[:, legend_w + 1 :] = main
    return grid


def _count_enclosed_regions(main: np.ndarray) -> int:
    count = 0
    for comp in _components(main != MAROON):
        if not _touches_border(comp, *main.shape):
            count += 1
    return count


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    orientation_top = bool(_randint(rng, 0, 1))
    swatch_size = _randint(rng, 3, 5)
    legend_count = _randint(rng, 2, 5)
    keys = [int(x) for x in rng.choice(PALETTE_COLORS, size=legend_count, replace=False)]
    patterns = {key: _make_pattern(rng, key) for key in keys}
    swatches = [_make_swatch(patterns[key], swatch_size) for key in keys]

    region_count = _randint(rng, 1, 4)
    used_key_count = _randint(rng, 1, min(region_count, legend_count))
    used_keys = [int(x) for x in rng.choice(keys, size=used_key_count, replace=False)]
    region_keys = used_keys[:]
    while len(region_keys) < region_count:
        region_keys.append(int(rng.choice(used_keys)))
    rng.shuffle(region_keys)

    if orientation_top:
        gap = 2
        pad = 2
        needed_w = pad + legend_count * swatch_size + (legend_count - 1) * gap + pad
        main_h = _randint(rng, 10, 20)
        main_w = _randint(rng, max(needed_w, 14), max(needed_w, 30))
    else:
        gap = 2
        pad = 2
        needed_h = pad + legend_count * swatch_size + (legend_count - 1) * gap + pad
        main_h = _randint(rng, max(needed_h, 12), max(needed_h, 30))
        main_w = _randint(rng, 12, 22)

    main = _place_regions(rng, main_h, main_w, region_keys)
    if main is None:
        return None

    grid = _build_top_layout(rng, swatches, main) if orientation_top else _build_left_layout(rng, swatches, main)

    try:
        out = _solve(grid)
    except AssertionError:
        return None

    _, cropped_main = _split_input(grid)
    if out.shape != cropped_main.shape:
        return None
    if _count_enclosed_regions(cropped_main) != region_count:
        return None
    if np.array_equal(out, cropped_main):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(_RNG)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
