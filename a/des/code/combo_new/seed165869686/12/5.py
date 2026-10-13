import random
from collections import defaultdict

import numpy as np


ARC_GRID_MAX = 30
BG_COLOR = 0
SEP_COLOR = 5
SEED_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]


def _mode_color(values: np.ndarray) -> int:
    vals, counts = np.unique(values, return_counts=True)
    return int(vals[np.argmax(counts)])


def _partition_slices(separators: list[int], limit: int) -> list[slice]:
    parts: list[slice] = []
    start = 0
    for sep in separators + [limit]:
        assert sep >= start, "separator indices must be nondecreasing"
        if sep > start:
            parts.append(slice(start, sep))
        start = sep + 1
    assert parts, "expected at least one partition"
    sizes = [part.stop - part.start for part in parts]
    assert len(set(sizes)) == 1, "partitions must have equal sizes"
    return parts


def _parse_partitions(grid: np.ndarray) -> tuple[int, int, list[slice], list[slice]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    solid_rows = [r for r in range(h) if np.all(g[r, :] == g[r, 0])]
    solid_cols = [c for c in range(w) if np.all(g[:, c] == g[0, c])]

    row_colors = {int(g[r, 0]) for r in solid_rows}
    col_colors = {int(g[0, c]) for c in solid_cols}
    sep_candidates = row_colors & col_colors
    assert len(sep_candidates) == 1, "expected one separator color"
    sep = sep_candidates.pop()

    row_parts = _partition_slices([r for r in solid_rows if int(g[r, 0]) == sep], h)
    col_parts = _partition_slices([c for c in solid_cols if int(g[0, c]) == sep], w)
    return sep, _mode_color(g[g != sep]), row_parts, col_parts


def _seed_patch(region: np.ndarray, bg: int) -> tuple[int, int, np.ndarray] | None:
    mask = region != bg
    if not np.any(mask):
        return None
    rows, cols = np.argwhere(mask).T
    r0, c0 = int(rows.min()), int(cols.min())
    r1, c1 = int(rows.max()), int(cols.max())
    return r0, c0, region[r0 : r1 + 1, c0 : c1 + 1].copy()


def _neighbor_counts(seed_mask: np.ndarray) -> np.ndarray:
    rows, cols = seed_mask.shape
    out = np.zeros((rows, cols), dtype=int)
    for r in range(rows):
        for c in range(cols):
            if not seed_mask[r, c]:
                continue
            count = 0
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and seed_mask[nr, nc]:
                    count += 1
            out[r, c] = count
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repeat each partition's seed horizontally according to how many orthogonal neighboring partitions also contain seeds.

    Input structure:
    - One-cell-wide Grey separator rows and columns split the grid into equal rectangular partitions arranged as a 2x2, 3x3, or similar small lattice.
    - Each partition has a Black background; some partitions contain one small colored foreground seed pattern anchored at the partition's top-left corner.
    - Seeded partitions are separated from each other by the Grey lines, while empty partitions are pure Black.

    Rule:
    - Detect which partitions contain any non-background seed pixels.
    - For each seeded partition, count how many edge-sharing neighboring partitions also contain a seed.
    - If that count is 0, erase the seed. Otherwise repeat the original seed patch that many times horizontally inside the same partition, clipping any overflow at the partition's right edge. Do not repeat vertically, and keep the separator lines unchanged.

    Key insight:
    - The seed's own shape and color stay the same; the only driver of the transformation is the partition's degree in the seed-presence adjacency graph.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    _sep, bg, row_parts, col_parts = _parse_partitions(g)
    n_rows, n_cols = len(row_parts), len(col_parts)

    seeded = np.zeros((n_rows, n_cols), dtype=bool)
    patches: list[list[tuple[int, int, np.ndarray] | None]] = [
        [None for _ in range(n_cols)] for _ in range(n_rows)
    ]

    for r, rs in enumerate(row_parts):
        for c, cs in enumerate(col_parts):
            patch = _seed_patch(g[rs, cs], bg)
            patches[r][c] = patch
            seeded[r, c] = patch is not None

    counts = _neighbor_counts(seeded)
    out = g.copy()

    for r, rs in enumerate(row_parts):
        for c, cs in enumerate(col_parts):
            region = out[rs, cs]
            region[:, :] = bg
            patch = patches[r][c]
            copies = int(counts[r, c])
            if patch is None or copies == 0:
                continue

            r0, c0, seed = patch
            ph, pw = seed.shape
            region_h, region_w = region.shape
            assert r0 + ph <= region_h and c0 + pw <= region_w

            for i in range(copies):
                left = c0 + i * pw
                if left >= region_w:
                    break
                take_w = min(pw, region_w - left)
                region[r0 : r0 + ph, left : left + take_w] = seed[:, :take_w]

    return out


def _catalog_masks(n_parts: int) -> dict[str, list[np.ndarray]]:
    catalog: dict[str, list[np.ndarray]] = defaultdict(list)
    total = n_parts * n_parts
    for bits in range(1, 1 << total):
        if bits.bit_count() > len(SEED_COLORS):
            continue
        mask = np.zeros((n_parts, n_parts), dtype=bool)
        for idx in range(total):
            if bits & (1 << idx):
                mask[idx // n_parts, idx % n_parts] = True
        degrees = _neighbor_counts(mask)[mask]
        has_zero = bool(np.any(degrees == 0))
        has_repeat = bool(np.any(degrees >= 2))
        has_clip = bool(np.any(degrees >= 3))

        if has_zero and not has_repeat:
            catalog["delete"].append(mask)
        if has_repeat and not has_zero:
            catalog["repeat_fit"].append(mask)
        if has_repeat and has_zero:
            catalog["mixed_fit"].append(mask)
        if has_clip and not has_zero:
            catalog["clip"].append(mask)
        if has_clip and has_zero:
            catalog["mixed_clip"].append(mask)
    return catalog


MASK_CATALOG = {2: _catalog_masks(2), 3: _catalog_masks(3)}


def _build_seed_shape(height: int, width: int, color: int) -> np.ndarray:
    cells: set[tuple[int, int]]
    if height == 1:
        cells = {(0, c) for c in range(width)}
    elif width == 1:
        cells = {(r, 0) for r in range(height)}
    elif random.random() < 0.5:
        cells = {(0, c) for c in range(width)} | {(r, width - 1) for r in range(height)}
    else:
        cells = {(r, 0) for r in range(height)} | {(height - 1, c) for c in range(width)}

    target_area = random.randint(len(cells), height * width)
    while len(cells) < target_area:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        cells.add(random.choice(frontier))

    patch = np.full((height, width), BG_COLOR, dtype=int)
    for r, c in cells:
        patch[r, c] = color
    return patch


def _partition_starts(n_parts: int, part_size: int) -> list[int]:
    return [i * (part_size + 1) for i in range(n_parts)]


def _scenario_options(n_parts: int) -> list[str]:
    if n_parts == 2:
        return ["delete", "repeat_fit"]
    return ["delete", "repeat_fit", "mixed_fit", "clip", "mixed_clip"]


def _branch_stats(seed_mask: np.ndarray, widths: dict[tuple[int, int], int], part_w: int) -> tuple[bool, bool, bool]:
    degrees = _neighbor_counts(seed_mask)
    has_delete = False
    has_repeat = False
    has_clip = False
    for r, c in np.argwhere(seed_mask):
        degree = int(degrees[r, c])
        if degree == 0:
            has_delete = True
        if degree >= 2:
            has_repeat = True
        if degree >= 1 and widths[(int(r), int(c))] * degree > part_w:
            has_clip = True
    return has_delete, has_repeat, has_clip


def _generate_case() -> np.ndarray | None:
    n_parts = random.choices([2, 3], weights=[2, 3])[0]
    scenario = random.choice(_scenario_options(n_parts))
    masks = MASK_CATALOG[n_parts][scenario]
    if not masks:
        return None
    seed_mask = random.choice(masks)
    degrees = _neighbor_counts(seed_mask)

    max_part = (ARC_GRID_MAX - (n_parts - 1)) // n_parts
    part_h = random.randint(4, max_part)
    part_w = random.randint(4, max_part)

    h_starts = _partition_starts(n_parts, part_h)
    w_starts = _partition_starts(n_parts, part_w)
    grid = np.full((n_parts * part_h + (n_parts - 1), n_parts * part_w + (n_parts - 1)), BG_COLOR, dtype=int)
    for r in range(1, n_parts):
        grid[r * part_h + (r - 1), :] = SEP_COLOR
    for c in range(1, n_parts):
        grid[:, c * part_w + (c - 1)] = SEP_COLOR

    seed_positions = [(int(r), int(c)) for r, c in np.argwhere(seed_mask)]
    colors = random.sample(SEED_COLORS, len(seed_positions))
    widths: dict[tuple[int, int], int] = {}

    special_pos: tuple[int, int] | None = None
    if scenario in {"repeat_fit", "mixed_fit"}:
        repeaters = [pos for pos in seed_positions if int(degrees[pos]) >= 2]
        special_pos = random.choice(repeaters)
    elif scenario in {"clip", "mixed_clip"}:
        clippers = [pos for pos in seed_positions if int(degrees[pos]) >= 3]
        special_pos = random.choice(clippers)

    for color, (pr, pc) in zip(colors, seed_positions):
        degree = int(degrees[pr, pc])
        max_h = part_h // 2
        max_w = part_w // 2
        if (pr, pc) == special_pos and scenario in {"repeat_fit", "mixed_fit"}:
            width = random.randint(1, min(max_w, part_w // degree))
        elif (pr, pc) == special_pos and scenario in {"clip", "mixed_clip"}:
            min_clip_w = part_w // degree + 1
            if min_clip_w > max_w:
                return None
            width = random.randint(min_clip_w, max_w)
        else:
            width = random.randint(1, max_w)
        height = random.randint(1, max_h)
        patch = _build_seed_shape(height, width, color)
        widths[(pr, pc)] = width
        r0, c0 = h_starts[pr], w_starts[pc]
        grid[r0 : r0 + height, c0 : c0 + width] = patch

    has_delete, has_repeat, has_clip = _branch_stats(seed_mask, widths, part_w)
    if scenario == "delete" and not has_delete:
        return None
    if scenario in {"repeat_fit", "mixed_fit", "clip", "mixed_clip"} and not has_repeat:
        return None
    if scenario in {"clip", "mixed_clip"} and not has_clip:
        return None
    if scenario in {"mixed_fit", "mixed_clip"} and not has_delete:
        return None

    out = main(grid)
    if np.array_equal(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _generate_case()
        if grid is None:
            continue
        return grid
    raise RuntimeError("failed to generate a valid input")
