from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
FRAME = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
PERIODIC_TILE_SPECS = ((2, 2), (3, 2), (2, 3))
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _find_frame_color(grid: np.ndarray) -> int:
    h, w = grid.shape
    best: tuple[int, int] | None = None
    for color in map(int, np.unique(grid)):
        full_rows = sum(int(np.all(grid[r, :] == color)) for r in range(h))
        full_cols = sum(int(np.all(grid[:, c] == color)) for c in range(w))
        if full_rows >= 2 and full_cols >= 2:
            score = full_rows + full_cols
            if best is None or score > best[0]:
                best = (score, color)
    assert best is not None, "expected a frame color forming complete rows and columns"
    return best[1]


def _split_spans(indices: list[int], limit: int) -> list[tuple[int, int]]:
    cuts = sorted(set(int(v) for v in indices))
    if not cuts or cuts[0] != 0:
        cuts = [0] + cuts
    if cuts[-1] != limit - 1:
        cuts.append(limit - 1)

    spans: list[tuple[int, int]] = []
    for a, b in zip(cuts, cuts[1:]):
        if b - a > 1:
            spans.append((a + 1, b - 1))
    assert spans, "expected at least one non-frame region"
    return spans


def _extract_regions(grid: np.ndarray, frame: int) -> list[dict]:
    h, w = grid.shape
    frame_rows = [r for r in range(h) if np.all(grid[r, :] == frame)]
    frame_cols = [c for c in range(w) if np.all(grid[:, c] == frame)]
    row_spans = _split_spans(frame_rows, h)
    col_spans = _split_spans(frame_cols, w)

    regions: list[dict] = []
    for ri, (r0, r1) in enumerate(row_spans):
        for ci, (c0, c1) in enumerate(col_spans):
            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            regions.append(
                {
                    "key": (ri, ci),
                    "bbox": (r0, c0, r1, c1),
                    "patch": patch.copy(),
                }
            )
    return regions


def _majority_color(values: np.ndarray) -> int:
    counts = Counter(map(int, values.flatten()))
    assert counts, "expected at least one value"
    return int(counts.most_common(1)[0][0])


def _repair_periodic_region(patch: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = patch.shape
    best: tuple[tuple[int, int, int, int], np.ndarray, tuple[int, int]] | None = None

    for tile_h, tile_w in PERIODIC_TILE_SPECS:
        if h < tile_h * 2 or w < tile_w * 2:
            continue

        tile = np.zeros((tile_h, tile_w), dtype=int)
        for rr in range(tile_h):
            for cc in range(tile_w):
                tile[rr, cc] = _majority_color(patch[rr::tile_h, cc::tile_w])

        repaired = np.tile(tile, ((h + tile_h - 1) // tile_h, (w + tile_w - 1) // tile_w))[:h, :w]
        colors = sorted(set(map(int, np.unique(repaired))) - {BG, FRAME})
        if len(colors) != 2:
            continue

        mismatches = int(np.sum(repaired != patch))
        key = (mismatches, tile_h * tile_w, tile_h, tile_w)
        if best is None or key < best[0]:
            best = (key, repaired, (colors[0], colors[1]))

    assert best is not None, "expected a periodic region with a valid 2-color tile"
    assert best[0][0] <= 3, "periodic region should contain only small noise"
    return best[1], best[2]


def _swap_colors(patch: np.ndarray, pair: tuple[int, int]) -> np.ndarray:
    x, y = pair
    out = patch.copy()
    x_mask = patch == x
    y_mask = patch == y
    out[x_mask] = y
    out[y_mask] = x
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair noisy periodic tiles and use each repaired tile's color pair to swap colors in its neighboring object region.

    Input structure:
    - The grid has a black background plus grey full-row/full-column frame lines that partition the canvas into rectangular regions.
    - Some regions are fully filled by a 2-color periodic tile with 1 to 3 noisy pixels; the remaining regions contain 1 or 2 small monochrome objects on black background.
    - Every object region shares one frame edge with exactly one periodic region, and its object colors are drawn from that periodic region's two tile colors.

    Rule:
    - Split the grid by the grey frame lines into rectangular regions.
    - For each periodic region, infer the smallest repeating 2x2, 3x2, or 2x3 tile by majority vote on residue classes, then replace noisy pixels with the repaired tile and record its two colors (X, Y).
    - For each object region, find the unique adjacent periodic region and swap X with Y on every foreground pixel inside the object region; keep frame lines and black background unchanged.

    Key insight:
    - The repaired tile is not only the visual target for the periodic region: its recovered 2-color palette is the local recoloring rule for the neighboring non-periodic region.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frame = _find_frame_color(g)
    regions = _extract_regions(g, frame)
    region_by_key = {region["key"]: region for region in regions}

    out = g.copy()
    periodic_pairs: dict[tuple[int, int], tuple[int, int]] = {}
    periodic_flags: dict[tuple[int, int], bool] = {}

    for region in regions:
        r0, c0, r1, c1 = region["bbox"]
        patch = region["patch"]
        is_periodic = not np.any(patch == BG)
        periodic_flags[region["key"]] = is_periodic
        if is_periodic:
            repaired, pair = _repair_periodic_region(patch)
            out[r0 : r1 + 1, c0 : c1 + 1] = repaired
            periodic_pairs[region["key"]] = pair

    for region in regions:
        key = region["key"]
        if periodic_flags[key]:
            continue

        ri, ci = key
        periodic_neighbors = []
        for nbr in ((ri - 1, ci), (ri + 1, ci), (ri, ci - 1), (ri, ci + 1)):
            if nbr in region_by_key and periodic_flags.get(nbr, False):
                periodic_neighbors.append(nbr)
        assert len(periodic_neighbors) == 1, "each object region must have exactly one periodic neighbor"

        r0, c0, r1, c1 = region["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = _swap_colors(region["patch"], periodic_pairs[periodic_neighbors[0]])

    return out


def _linear_types(count: int) -> str:
    templates = {
        4: ("POOP", "OPPO"),
        5: ("POOPP", "PPOOP"),
        6: ("OPPOOP", "POOPPO"),
    }
    return random.choice(templates[count])


def _allowed_cross_periods(cross: int) -> tuple[int, ...]:
    if cross == 4:
        return (2,)
    if cross == 6:
        return (2, 3)
    raise AssertionError("unexpected cross-axis size")


def _random_tile(tile_h: int, tile_w: int, pair: tuple[int, int]) -> np.ndarray:
    x, y = pair
    for _ in range(100):
        tile = np.random.choice([x, y], size=(tile_h, tile_w))
        if len(set(map(int, np.unique(tile)))) != 2:
            continue
        return tile.astype(int)
    raise RuntimeError("failed to sample a non-degenerate tile")


def _build_periodic_patch(h: int, w: int) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    pair = tuple(random.sample(PALETTE, 2))
    for _ in range(200):
        tile_h, tile_w = random.choice(PERIODIC_TILE_SPECS)
        if h < tile_h * 2 or w < tile_w * 2:
            continue
        if h % tile_h != 0 or w % tile_w != 0:
            continue

        tile = _random_tile(tile_h, tile_w, pair)
        clean = np.tile(tile, (h // tile_h, w // tile_w)).astype(int)
        repaired, solved_pair = _repair_periodic_region(clean)
        if not np.array_equal(repaired, clean) or set(solved_pair) != set(pair):
            continue

        noisy = clean.copy()
        noise_count = random.randint(1, 3)
        noise_positions = random.sample([(r, c) for r in range(h) for c in range(w)], noise_count)
        noise_choices = [c for c in PALETTE if c not in pair]
        for r, c in noise_positions:
            noisy[r, c] = random.choice(noise_choices)

        repaired_noisy, solved_pair_noisy = _repair_periodic_region(noisy)
        if np.array_equal(repaired_noisy, clean) and set(solved_pair_noisy) == set(pair):
            return noisy, clean, pair

    raise RuntimeError("failed to build a valid periodic region")


def _grow_object(h: int, w: int, occupied: np.ndarray, area: int) -> list[tuple[int, int]] | None:
    free = [
        (r, c)
        for r in range(h)
        for c in range(w)
        if not occupied[r, c] and all(not occupied[nr, nc] for nr, nc in _neighbors4(r, c) if 0 <= nr < h and 0 <= nc < w)
    ]
    if not free:
        return None

    for _ in range(60):
        seed = random.choice(free)
        cells = {seed}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if occupied[nr, nc] or (nr, nc) in cells:
                        continue
                    if any(
                        occupied[ar, ac]
                        for ar, ac in _neighbors4(nr, nc)
                        if 0 <= ar < h and 0 <= ac < w and (ar, ac) not in cells
                    ):
                        continue
                    frontier.append((nr, nc))
            frontier = sorted(set(frontier))
            if not frontier:
                break
            cells.add(random.choice(frontier))

        if len(cells) == area:
            rows = [r for r, _ in cells]
            cols = [c for _, c in cells]
            if max(rows) - min(rows) <= h - 1 and max(cols) - min(cols) <= w - 1:
                return sorted(cells)
    return None


def _build_object_patch(h: int, w: int, pair: tuple[int, int]) -> np.ndarray:
    for _ in range(200):
        patch = np.zeros((h, w), dtype=int)
        occupied = np.zeros((h, w), dtype=bool)
        target_objects = 2 if h * w >= 15 and random.random() < 0.55 else 1

        if target_objects == 2 and random.random() < 0.7:
            colors = [pair[0], pair[1]]
        else:
            colors = [random.choice(pair) for _ in range(target_objects)]

        ok = True
        for color in colors:
            max_area = min(6, max(3, h * w // (target_objects + 3)))
            area = random.randint(3, max_area)
            cells = _grow_object(h, w, occupied, area)
            if cells is None:
                ok = False
                break
            for r, c in cells:
                patch[r, c] = color
                occupied[r, c] = True

        if ok and np.count_nonzero(patch) >= 3:
            return patch

    raise RuntimeError("failed to build an object region")


def _sample_layout() -> tuple[str, str, list[tuple[int, int]], int, tuple[int, int]]:
    orientation = random.choice(("horizontal", "vertical"))
    count = random.randint(4, 6)
    types = _linear_types(count)
    cross = random.choice((4, 6))

    periodic_sizes: list[int] = []
    object_sizes: list[int] = []
    for kind in types:
        if kind == "P":
            if cross == 4:
                periodic_sizes.append(4)
            else:
                periodic_sizes.append(random.choice((4, 6)))
        else:
            object_sizes.append(random.choice((3, 4)))

    sizes: list[int] = []
    p_i = 0
    o_i = 0
    for kind in types:
        if kind == "P":
            sizes.append(periodic_sizes[p_i])
            p_i += 1
        else:
            sizes.append(object_sizes[o_i])
            o_i += 1

    total = sum(sizes) + count + 1
    if total > 30:
        raise RuntimeError("layout too large")

    if orientation == "horizontal":
        shape = (cross + 2, total)
    else:
        shape = (total, cross + 2)
    assert max(shape) <= 30
    return orientation, types, list(enumerate(sizes)), cross, shape


def _neighbor_periodic_index(types: str, idx: int) -> int:
    hits = [j for j in (idx - 1, idx + 1) if 0 <= j < len(types) and types[j] == "P"]
    assert len(hits) == 1
    return hits[0]


def _assemble_case() -> tuple[np.ndarray, np.ndarray]:
    orientation, types, indexed_sizes, cross, shape = _sample_layout()
    grid = np.full(shape, FRAME, dtype=int)
    expected = np.full(shape, FRAME, dtype=int)

    periodic_pairs: dict[int, tuple[int, int]] = {}
    periodic_clean: dict[int, np.ndarray] = {}
    periodic_noisy: dict[int, np.ndarray] = {}

    pos = 1
    for idx, size in indexed_sizes:
        if orientation == "horizontal":
            r0, c0 = 1, pos
            h, w = cross, size
        else:
            r0, c0 = pos, 1
            h, w = size, cross

        if types[idx] == "P":
            noisy, clean, pair = _build_periodic_patch(h, w)
            periodic_pairs[idx] = pair
            periodic_clean[idx] = clean
            periodic_noisy[idx] = noisy
            grid[r0 : r0 + h, c0 : c0 + w] = noisy
            expected[r0 : r0 + h, c0 : c0 + w] = clean
        pos += size + 1

    pos = 1
    for idx, size in indexed_sizes:
        if orientation == "horizontal":
            r0, c0 = 1, pos
            h, w = cross, size
        else:
            r0, c0 = pos, 1
            h, w = size, cross

        if types[idx] == "O":
            pair = periodic_pairs[_neighbor_periodic_index(types, idx)]
            patch = _build_object_patch(h, w, pair)
            grid[r0 : r0 + h, c0 : c0 + w] = patch
            expected[r0 : r0 + h, c0 : c0 + w] = _swap_colors(patch, pair)
        pos += size + 1

    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            grid, expected = _assemble_case()
        except Exception:
            continue

        try:
            actual = main(grid)
        except Exception:
            continue

        if not np.array_equal(actual, expected):
            continue
        if np.array_equal(actual, grid):
            continue
        return grid

    raise RuntimeError("failed to generate a valid puzzle instance")
