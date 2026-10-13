import random
from collections import Counter, defaultdict

import numpy as np


BG = 0
SEPARATOR = 5
LEGEND_INK = 9
PLAY_COLORS = (1, 2, 3, 4, 6, 7, 8)
GEN_RETRIES = 400

TEMPLATE_LIBRARY = (
    ("plus", np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)),
    ("diag", np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool)),
    ("full", np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=bool)),
    ("tee_up", np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool)),
    ("hline", np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=bool)),
    ("vline", np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=bool)),
)


def _spans(size: int, cuts: list[int]) -> list[tuple[int, int]] | None:
    cuts = sorted(int(x) for x in cuts)
    spans: list[tuple[int, int]] = []
    start = 0
    for cut in cuts:
        if cut <= start or cut >= size:
            return None
        spans.append((start, cut - 1))
        start = cut + 1
    if start >= size:
        return None
    spans.append((start, size - 1))
    if any(a > b for a, b in spans):
        return None
    return spans


def _partition_structure(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates = []
    for color in map(int, np.unique(g)):
        rows = [r for r in range(h) if np.all(g[r, :] == color)]
        cols = [c for c in range(w) if np.all(g[:, c] == color)]
        if not rows or not cols:
            continue
        row_spans = _spans(h, rows)
        col_spans = _spans(w, cols)
        if row_spans is None or col_spans is None:
            continue
        row_sizes = {r1 - r0 + 1 for r0, r1 in row_spans}
        col_sizes = {c1 - c0 + 1 for c0, c1 in col_spans}
        if len(row_sizes) != 1 or len(col_sizes) != 1:
            continue
        if len(row_spans) < 2 or len(col_spans) < 2:
            continue
        candidates.append((color, row_spans, col_spans, len(rows) + len(cols), int(np.count_nonzero(g == color))))
    assert candidates, "expected a separator color forming full partition lines"
    candidates.sort(key=lambda item: (item[3], item[4]), reverse=True)
    color, row_spans, col_spans, _, _ = candidates[0]
    return color, row_spans, col_spans


def _tile_offsets(template: np.ndarray) -> tuple[tuple[int, int], ...]:
    offsets = []
    for rr in range(3):
        for cc in range(3):
            if bool(template[rr, cc]):
                offsets.append((rr - 1, cc - 1))
    assert (0, 0) in offsets
    return tuple(offsets)


def _grow_partition(
    shape: tuple[int, int],
    seeds: list[tuple[int, int, int]],
    templates: dict[int, np.ndarray],
) -> tuple[np.ndarray, dict[int, int], int]:
    h, w = shape
    owners = np.full((h, w), -1, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)
    frontier: dict[int, set[tuple[int, int]]] = defaultdict(set)
    offsets = {color: _tile_offsets(template) for color, template in templates.items()}

    for r, c, color in seeds:
        assert 0 <= r < h and 0 <= c < w
        if owners[r, c] == -1:
            owners[r, c] = color
            frontier[color].add((r, c))
        else:
            assert owners[r, c] == color, "multiple colors cannot share the same seed cell"

    conflicts = 0
    while True:
        claims: dict[tuple[int, int], int | None] = {}
        for color, cells in frontier.items():
            for r, c in cells:
                for dr, dc in offsets[color]:
                    nr = r + dr
                    nc = c + dc
                    if nr < 0 or nr >= h or nc < 0 or nc >= w:
                        continue
                    if owners[nr, nc] != -1 or blocked[nr, nc]:
                        continue
                    key = (nr, nc)
                    prev = claims.get(key)
                    if prev is None:
                        claims[key] = color
                    elif prev != color:
                        claims[key] = -1

        if not claims:
            break

        new_frontier: dict[int, set[tuple[int, int]]] = defaultdict(set)
        any_new = False
        for (r, c), claimant in claims.items():
            if claimant == -1:
                blocked[r, c] = True
                conflicts += 1
                continue
            owners[r, c] = claimant
            new_frontier[claimant].add((r, c))
            any_new = True

        if not any_new:
            break
        frontier = new_frontier

    out = np.full((h, w), BG, dtype=int)
    areas: dict[int, int] = {}
    for color in sorted(templates):
        count = int(np.count_nonzero(owners == color))
        if count > 0:
            out[owners == color] = color
            areas[color] = count
    return out, areas, conflicts


def _legend_from_tile(tile: np.ndarray, bg: int) -> tuple[int, int, np.ndarray] | None:
    colors = [int(color) for color in np.unique(tile) if int(color) != bg]
    for ink in colors:
        ink_count = int(np.count_nonzero(tile == ink))
        if ink_count < 2:
            continue
        others = np.argwhere((tile != bg) & (tile != ink))
        if others.shape[0] != 1:
            continue
        key_r, key_c = map(int, others[0])
        for r0 in range(tile.shape[0] - 2):
            for c0 in range(tile.shape[1] - 2):
                if key_c >= c0:
                    continue
                window = tile[r0 : r0 + 3, c0 : c0 + 3]
                if not set(map(int, np.unique(window))).issubset({bg, ink}):
                    continue
                if int(np.count_nonzero(window == ink)) != ink_count:
                    continue
                template = window == ink
                if not bool(template[1, 1]):
                    continue
                return ink, int(tile[key_r, key_c]), template
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow color-coded crystal templates inside equal partition cells while grey lines block expansion.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    separator, row_spans, col_spans = _partition_structure(g)
    bg = int(Counter(map(int, g[g != separator].ravel())).most_common(1)[0][0])

    legend_entries = []
    for tr, (r0, r1) in enumerate(row_spans):
        for tc, (c0, c1) in enumerate(col_spans):
            tile = g[r0 : r1 + 1, c0 : c1 + 1]
            info = _legend_from_tile(tile, bg)
            if info is not None:
                legend_entries.append((tr, tc, info))

    assert legend_entries, "expected at least one legend tile"
    ink = Counter(info[0] for _, _, info in legend_entries).most_common(1)[0][0]

    templates: dict[int, np.ndarray] = {}
    legend_tiles: set[tuple[int, int]] = set()
    for tr, tc, (entry_ink, color, template) in legend_entries:
        if entry_ink != ink:
            continue
        assert color not in templates, "legend colors must be unique"
        templates[color] = template.copy()
        legend_tiles.add((tr, tc))

    assert len(templates) >= 2, "legend must define at least two color-template pairs"

    out = g.copy()
    for tr, (r0, r1) in enumerate(row_spans):
        for tc, (c0, c1) in enumerate(col_spans):
            if (tr, tc) in legend_tiles:
                continue
            tile = g[r0 : r1 + 1, c0 : c1 + 1]
            seed_cells = np.argwhere(tile != bg)
            if seed_cells.size == 0:
                continue
            seeds: list[tuple[int, int, int]] = []
            for rr, cc in seed_cells:
                value = int(tile[int(rr), int(cc)])
                assert value in templates, "non-legend tiles may only contain mapped seed colors"
                seeds.append((int(rr), int(cc), value))
            grown, _, _ = _grow_partition(tile.shape, seeds, templates)
            out[r0 : r1 + 1, c0 : c1 + 1] = grown

    return out


def _tile_origin(tile_r: int, tile_c: int, tile_h: int, tile_w: int) -> tuple[int, int]:
    return tile_r * (tile_h + 1), tile_c * (tile_w + 1)


def _make_canvas(tile_rows: int, tile_cols: int, tile_h: int, tile_w: int) -> np.ndarray:
    h = tile_rows * tile_h + (tile_rows - 1)
    w = tile_cols * tile_w + (tile_cols - 1)
    grid = np.full((h, w), BG, dtype=int)
    for r in range(tile_h, h, tile_h + 1):
        grid[r, :] = SEPARATOR
    for c in range(tile_w, w, tile_w + 1):
        grid[:, c] = SEPARATOR
    return grid


def _legend_positions(
    corner: str,
    legend_rows: int,
    legend_cols: int,
    tile_rows: int,
    tile_cols: int,
) -> list[tuple[int, int]]:
    row_base = 0 if "T" in corner else tile_rows - legend_rows
    col_base = 0 if "L" in corner else tile_cols - legend_cols
    positions = []
    for rr in range(legend_rows):
        for cc in range(legend_cols):
            positions.append((row_base + rr, col_base + cc))
    return positions


def _paint_legend_tile(
    grid: np.ndarray,
    tile_r: int,
    tile_c: int,
    tile_h: int,
    tile_w: int,
    key_color: int,
    template: np.ndarray,
) -> None:
    r0, c0 = _tile_origin(tile_r, tile_c, tile_h, tile_w)
    tile = grid[r0 : r0 + tile_h, c0 : c0 + tile_w]
    center_r = tile_h // 2
    box_r0 = center_r - 1
    box_c0 = tile_w - 3
    tile[center_r, 0] = key_color
    patch = tile[box_r0 : box_r0 + 3, box_c0 : box_c0 + 3]
    patch[template] = LEGEND_INK


def _random_single_seed(
    rng: random.Random,
    tile_shape: tuple[int, int],
    color: int,
    templates: dict[int, np.ndarray],
) -> tuple[int, int]:
    h, w = tile_shape
    for _ in range(80):
        r = rng.randrange(h)
        c = rng.randrange(w)
        _, areas, _ = _grow_partition(tile_shape, [(r, c, color)], templates)
        if areas.get(color, 0) >= max(3, min(h, w) - 1):
            return r, c
    return h // 2, w // 2


def _competitive_colors(
    mapping: dict[int, tuple[str, np.ndarray]]
) -> list[int]:
    preferred = {
        color
        for color, (name, _) in mapping.items()
        if name not in {"hline", "vline"}
    }
    return sorted(preferred or mapping.keys())


def _random_dual_seeds(
    rng: random.Random,
    tile_shape: tuple[int, int],
    color_a: int,
    color_b: int,
    templates: dict[int, np.ndarray],
) -> list[tuple[int, int, int]] | None:
    h, w = tile_shape
    min_total = max(6, min(h, w) + 1)
    for _ in range(200):
        r1 = rng.randrange(h)
        c1 = rng.randrange(w)
        r2 = rng.randrange(h)
        c2 = rng.randrange(w)
        if (r1, c1) == (r2, c2):
            continue
        if abs(r1 - r2) + abs(c1 - c2) < 2:
            continue
        _, isolated_a, _ = _grow_partition(tile_shape, [(r1, c1, color_a)], templates)
        _, isolated_b, _ = _grow_partition(tile_shape, [(r2, c2, color_b)], templates)
        _, areas, conflicts = _grow_partition(
            tile_shape,
            [(r1, c1, color_a), (r2, c2, color_b)],
            templates,
        )
        if conflicts != 0:
            continue
        area_a = areas.get(color_a, 0)
        area_b = areas.get(color_b, 0)
        if area_a < 2 or area_b < 2:
            continue
        if area_a + area_b < min_total:
            continue
        if area_a + area_b >= isolated_a.get(color_a, 0) + isolated_b.get(color_b, 0):
            continue
        return [(r1, c1, color_a), (r2, c2, color_b)]
    return None


def _paint_seeds(
    grid: np.ndarray,
    tile_r: int,
    tile_c: int,
    tile_h: int,
    tile_w: int,
    seeds: list[tuple[int, int, int]],
) -> None:
    r0, c0 = _tile_origin(tile_r, tile_c, tile_h, tile_w)
    tile = grid[r0 : r0 + tile_h, c0 : c0 + tile_w]
    for rr, cc, color in seeds:
        tile[rr, cc] = color


def _candidate_layouts(tile_h: int, tile_w: int, n_colors: int) -> list[tuple[int, int, int, int]]:
    max_rows = 31 // (tile_h + 1)
    max_cols = 31 // (tile_w + 1)
    shapes = [(1, n_colors), (n_colors, 1)]
    if n_colors == 4:
        shapes.append((2, 2))

    layouts = []
    for legend_rows, legend_cols in shapes:
        if legend_rows > max_rows or legend_cols > max_cols:
            continue
        min_rows = max(3, legend_rows + 1)
        min_cols = max(3, legend_cols + 1)
        for tile_rows in range(min_rows, max_rows + 1):
            for tile_cols in range(min_cols, max_cols + 1):
                play_tiles = tile_rows * tile_cols - n_colors
                if play_tiles >= n_colors + 1:
                    layouts.append((legend_rows, legend_cols, tile_rows, tile_cols))
    return layouts


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    tile_h = rng.choice((5, 6, 7))
    tile_w = rng.choice((5, 6))
    n_colors = rng.choice((3, 4))

    layouts = _candidate_layouts(tile_h, tile_w, n_colors)
    if not layouts:
        return None
    legend_rows, legend_cols, tile_rows, tile_cols = rng.choice(layouts)

    chosen_colors = rng.sample(PLAY_COLORS, n_colors)
    chosen_templates = rng.sample(TEMPLATE_LIBRARY, n_colors)
    mapping = {color: (name, template.copy()) for color, (name, template) in zip(chosen_colors, chosen_templates)}
    templates = {color: template for color, (_, template) in mapping.items()}

    corner = rng.choice(("TL", "TR", "BL", "BR"))
    legend_tiles = _legend_positions(corner, legend_rows, legend_cols, tile_rows, tile_cols)
    legend_tile_set = set(legend_tiles)

    play_tiles = [(r, c) for r in range(tile_rows) for c in range(tile_cols) if (r, c) not in legend_tile_set]
    rng.shuffle(play_tiles)
    if len(play_tiles) < n_colors + 1:
        return None

    grid = _make_canvas(tile_rows, tile_cols, tile_h, tile_w)

    for (tile_r, tile_c), color in zip(legend_tiles, chosen_colors):
        _paint_legend_tile(grid, tile_r, tile_c, tile_h, tile_w, color, templates[color])

    used_tiles = 0
    for color in chosen_colors:
        tile_r, tile_c = play_tiles[used_tiles]
        used_tiles += 1
        seed_r, seed_c = _random_single_seed(rng, (tile_h, tile_w), color, templates)
        _paint_seeds(grid, tile_r, tile_c, tile_h, tile_w, [(seed_r, seed_c, color)])

    dual_candidates = _competitive_colors(mapping)
    rng.shuffle(dual_candidates)
    dual_seeds = None
    for idx_a in range(len(dual_candidates)):
        for idx_b in range(idx_a + 1, len(dual_candidates)):
            dual_seeds = _random_dual_seeds(
                rng,
                (tile_h, tile_w),
                dual_candidates[idx_a],
                dual_candidates[idx_b],
                templates,
            )
            if dual_seeds is not None:
                break
        if dual_seeds is not None:
            break
    if dual_seeds is None:
        return None

    dual_tile_r, dual_tile_c = play_tiles[used_tiles]
    used_tiles += 1
    _paint_seeds(grid, dual_tile_r, dual_tile_c, tile_h, tile_w, dual_seeds)

    extra_tiles = play_tiles[used_tiles:]
    rng.shuffle(extra_tiles)
    extra_count = rng.randint(0, min(3, len(extra_tiles)))
    for tile_r, tile_c in extra_tiles[:extra_count]:
        if rng.random() < 0.5:
            color = rng.choice(chosen_colors)
            seed_r, seed_c = _random_single_seed(rng, (tile_h, tile_w), color, templates)
            _paint_seeds(grid, tile_r, tile_c, tile_h, tile_w, [(seed_r, seed_c, color)])
        else:
            pair = rng.sample(chosen_colors, 2)
            seeds = _random_dual_seeds(rng, (tile_h, tile_w), pair[0], pair[1], templates)
            if seeds is None:
                color = rng.choice(chosen_colors)
                seed_r, seed_c = _random_single_seed(rng, (tile_h, tile_w), color, templates)
                seeds = [(seed_r, seed_c, color)]
            _paint_seeds(grid, tile_r, tile_c, tile_h, tile_w, seeds)

    # Reject accidental parsing ambiguities before returning the sample.
    if len(np.unique(main(grid))) < 3:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {GEN_RETRIES} attempts")
