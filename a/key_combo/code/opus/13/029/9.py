import numpy as np


BG = 0
MARKER_TO_TURNS = {1: 0, 2: 1, 3: 2, 4: 3}
TEMPLATE_COLORS = (7, 8)
SEP = 5
GEN_RETRIES = 400


def _bands_from_separators(separators: list[int], length: int) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    start = 0
    for idx in sorted(separators):
        if start < idx:
            bands.append((start, idx))
        start = idx + 1
    if start < length:
        bands.append((start, length))
    return bands


def _paste_transparent(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    view = dst[r0:r0 + h, c0:c0 + w]
    mask = patch != BG
    view[mask] = patch[mask]


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _consensus_tile(tiles: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(tiles, axis=0)
    h, w = stack.shape[1:]
    out = np.zeros((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            counts = np.bincount(stack[:, r, c], minlength=10)
            out[r, c] = int(np.argmax(counts))
    return out


def _all_full_rows(grid: np.ndarray, color: int) -> list[int]:
    return [r for r in range(grid.shape[0]) if np.all(grid[r] == color)]


def _all_full_cols(grid: np.ndarray, color: int) -> list[int]:
    return [c for c in range(grid.shape[1]) if np.all(grid[:, c] == color)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover a consensus motif from repeated damaged tiles, then stamp its rotations at colored marker positions.

    Input structure:
    - A black background grid is split by one full grey row into an upper tile-lattice region and a lower marker region.
    - In the upper region, a 2x3 or 3x2 lattice of equal tiles is separated by single grey rows and columns.
    - Every tile is a copy of one motif drawn with orange and cyan on black, except one tile where 1-2 motif pixels were erased to black.
    - In the lower region, 2-4 isolated marker pixels use colors blue, red, green, and yellow; each color appears at most once.

    Rule:
    - Extract the upper tile lattice by its grey separators and crop away any surrounding black padding.
    - Reconstruct the motif by taking the per-coordinate majority color across all tile copies.
    - Interpret marker colors as rotation codes: blue=0 degrees, red=90 degrees clockwise, green=180 degrees, yellow=270 degrees clockwise.
    - Produce a black output the same size as the lower region, then stamp the appropriately rotated consensus motif with the marker as the rotated motif's top-left corner; black motif cells are transparent.
    - Process markers in row-major order so later stamps overwrite earlier ones where their non-black pixels overlap.

    Key insight:
    - The upper half does not directly contain the stamp template; it must first be repaired by majority voting across redundant damaged copies, and only then can the lower-half markers be decoded as rotation instructions.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    divider_rows = [r for r in range(g.shape[0]) if np.all(g[r] == SEP)]
    assert divider_rows, "expected a full grey divider row"
    divider = int(divider_rows[-1])

    upper = g[:divider]
    lower = g[divider + 1 :]
    box = _bbox(upper != BG)
    assert box is not None, "upper region must contain a non-black tile lattice"
    r0, c0, r1, c1 = box
    crop = upper[r0 : r1 + 1, c0 : c1 + 1]

    sep_rows = _all_full_rows(crop, SEP)
    sep_cols = _all_full_cols(crop, SEP)
    row_bands = _bands_from_separators(sep_rows, crop.shape[0])
    col_bands = _bands_from_separators(sep_cols, crop.shape[1])
    assert len(row_bands) in (2, 3) and len(col_bands) in (2, 3), "expected a 2x3 or 3x2 tile lattice"
    assert len(row_bands) * len(col_bands) == 6, "expected exactly six tile copies"

    heights = {r1b - r0b for r0b, r1b in row_bands}
    widths = {c1b - c0b for c0b, c1b in col_bands}
    assert len(heights) == 1 and len(widths) == 1, "all tiles must have equal size"

    tiles: list[np.ndarray] = []
    for br0, br1 in row_bands:
        for bc0, bc1 in col_bands:
            tile = crop[br0:br1, bc0:bc1]
            tiles.append(tile.copy())

    motif = _consensus_tile(tiles)

    out = np.zeros_like(lower)
    markers = []
    for r, c in np.argwhere(np.isin(lower, list(MARKER_TO_TURNS))):
        color = int(lower[r, c])
        markers.append((int(r), int(c), color))
    markers.sort()

    for r, c, color in markers:
        rotated = np.rot90(motif, k=-MARKER_TO_TURNS[color])
        h, w = rotated.shape
        assert r + h <= out.shape[0] and c + w <= out.shape[1], "stamp must fit inside the lower region"
        _paste_transparent(out, rotated, r, c)
    return out


def _neighbors8(r: int, c: int, h: int, w: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w:
                yield nr, nc


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    h, w = mask.shape
    seen = {tuple(map(int, coords[0]))}
    stack = [tuple(map(int, coords[0]))]
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors8(r, c, h, w):
            if mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _random_connected_mask(h: int, w: int, target: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    start = (int(np.random.randint(h)), int(np.random.randint(w)))
    mask[start] = True
    frontier = {start}
    while int(mask.sum()) < target:
        base = list(frontier)[int(np.random.randint(len(frontier)))]
        neighbors = [p for p in _neighbors8(base[0], base[1], h, w) if not mask[p]]
        if not neighbors:
            frontier.discard(base)
            if not frontier:
                frontier = {tuple(map(int, p)) for p in np.argwhere(mask)}
            continue
        nr, nc = neighbors[int(np.random.randint(len(neighbors)))]
        mask[nr, nc] = True
        frontier.add((nr, nc))
    return mask


def _rotation_signatures(tile: np.ndarray) -> set[tuple[tuple[int, int], bytes]]:
    sigs = set()
    cur = tile
    for _ in range(4):
        sigs.add((cur.shape, cur.tobytes()))
        cur = np.rot90(cur, k=-1)
    return sigs


def _make_template(h: int, w: int) -> np.ndarray:
    area = h * w
    min_cells = max(4, area // 2)
    max_cells = area - 1
    for _ in range(200):
        target = int(np.random.randint(min_cells, max_cells + 1))
        mask = _random_connected_mask(h, w, target)
        if not _is_connected(mask):
            continue
        tile = np.zeros((h, w), dtype=int)
        coords = np.argwhere(mask)
        colors = np.random.choice(TEMPLATE_COLORS, size=len(coords))
        if len(set(map(int, colors))) < 2:
            continue
        for (r, c), color in zip(coords, colors):
            tile[int(r), int(c)] = int(color)
        if not tile[:, 0].any() or not tile[:, -1].any():
            continue
        if not tile[0, :].any() or not tile[-1, :].any():
            continue
        if len(_rotation_signatures(tile)) < 4:
            continue
        return tile
    raise RuntimeError("failed to generate a sufficiently asymmetric template")


def _build_upper_region(tile: np.ndarray, tile_rows: int, tile_cols: int) -> tuple[np.ndarray, np.ndarray]:
    h, w = tile.shape
    upper_h = tile_rows * h + (tile_rows - 1)
    upper_w = tile_cols * w + (tile_cols - 1)
    clean = np.zeros((upper_h, upper_w), dtype=int)
    damaged = np.zeros((upper_h, upper_w), dtype=int)

    for rr in range(1, tile_rows):
        sep_r = rr * (h + 1) - 1
        clean[sep_r, :] = SEP
        damaged[sep_r, :] = SEP
    for cc in range(1, tile_cols):
        sep_c = cc * (w + 1) - 1
        clean[:, sep_c] = SEP
        damaged[:, sep_c] = SEP

    tile_origins: list[tuple[int, int]] = []
    for rr in range(tile_rows):
        for cc in range(tile_cols):
            r0 = rr * (h + 1)
            c0 = cc * (w + 1)
            clean[r0 : r0 + h, c0 : c0 + w] = tile
            damaged[r0 : r0 + h, c0 : c0 + w] = tile
            tile_origins.append((r0, c0))

    occupied = np.argwhere(tile != BG)
    damage_count = int(np.random.choice([1, 2]))
    if len(occupied) <= damage_count:
        raise RuntimeError("template too sparse for damage")
    chosen = occupied[np.random.choice(len(occupied), size=damage_count, replace=False)]
    damaged_idx = int(np.random.randint(len(tile_origins)))
    tr, tc = tile_origins[damaged_idx]
    for lr, lc in chosen:
        damaged[tr + int(lr), tc + int(lc)] = BG
    return clean, damaged


def _box_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 <= br0 or br1 <= ar0 or ac1 <= bc0 or bc1 <= ac0)


def _pairwise_overlap_counts(
    placements: list[tuple[int, int]],
    patches: list[np.ndarray],
    shape: tuple[int, int],
) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    h, w = shape
    stamp_masks = []
    boxes = []
    padded = []
    for (r0, c0), patch in zip(placements, patches):
        ph, pw = patch.shape
        mask = np.zeros((h, w), dtype=bool)
        mask[r0 : r0 + ph, c0 : c0 + pw] = patch != BG
        stamp_masks.append(mask)
        boxes.append((r0, c0, r0 + ph, c0 + pw))
        padded.append((max(0, r0 - 1), max(0, c0 - 1), min(h, r0 + ph + 1), min(w, c0 + pw + 1)))

    actual = []
    box_only = []
    padded_hits = []
    for i in range(len(placements)):
        for j in range(i + 1, len(placements)):
            overlap = int(np.logical_and(stamp_masks[i], stamp_masks[j]).sum())
            if overlap > 0:
                actual.append((i, j, overlap))
            elif _box_overlap(boxes[i], boxes[j]):
                box_only.append((i, j, 0))
            if _box_overlap(padded[i], padded[j]):
                padded_hits.append((i, j, overlap))
    return actual, box_only, padded_hits


def _sample_marker_layout(patches: list[np.ndarray], lower_h: int, lower_w: int, want_overlap: bool) -> list[tuple[int, int]] | None:
    candidates: list[list[tuple[int, int]]] = []
    for patch in patches:
        ph, pw = patch.shape
        coords = [(r, c) for r in range(lower_h - ph + 1) for c in range(lower_w - pw + 1)]
        np.random.shuffle(coords)
        candidates.append(coords)

    for _ in range(500):
        placements = []
        for coords in candidates:
            placements.append(coords[int(np.random.randint(len(coords)))])
        actual, box_only, padded_hits = _pairwise_overlap_counts(placements, patches, (lower_h, lower_w))
        if want_overlap:
            if len(actual) != 1:
                continue
            if any(overlap > 0 and (i, j, overlap) not in actual for i, j, overlap in padded_hits):
                continue
            actual_pair = {actual[0][:2], actual[0][:2][::-1]}
            bad = False
            for i, j, overlap in padded_hits:
                if overlap == 0 and (i, j) not in actual_pair:
                    bad = True
                    break
            if bad or box_only:
                continue
            overlap = actual[0][2]
            min_nonzero = min(int((patches[actual[0][0]] != BG).sum()), int((patches[actual[0][1]] != BG).sum()))
            if overlap > max(1, min_nonzero // 3):
                continue
            return placements
        else:
            if actual or box_only or padded_hits:
                continue
            return placements
    return None


def _render_lower_region(lower_h: int, lower_w: int, markers: list[tuple[int, int, int]], motif: np.ndarray) -> np.ndarray:
    out = np.zeros((lower_h, lower_w), dtype=int)
    for r, c, color in sorted(markers):
        patch = np.rot90(motif, k=-MARKER_TO_TURNS[color])
        _paste_transparent(out, patch, r, c)
    return out


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            tile_h = int(np.random.choice([3, 4]))
            tile_w = int(np.random.choice([3, 4]))
            tile_rows, tile_cols = ((2, 3) if np.random.rand() < 0.5 else (3, 2))
            motif = _make_template(tile_h, tile_w)
            _, upper = _build_upper_region(motif, tile_rows, tile_cols)

            upper_h, upper_w = upper.shape
            max_dim = max(tile_h, tile_w)
            lower_h = int(np.random.randint(2 * max_dim + 4, 2 * max_dim + 8))
            overall_w = max(upper_w + int(np.random.randint(0, 3)), int(np.random.randint(2 * max_dim + 4, 2 * max_dim + 8)))

            marker_count = int(np.random.choice([2, 3, 4], p=[0.15, 0.35, 0.50]))
            marker_colors = list(np.random.choice([1, 2, 3, 4], size=marker_count, replace=False))
            patches = [np.rot90(motif, k=-MARKER_TO_TURNS[color]) for color in marker_colors]
            want_overlap = marker_count >= 2 and np.random.rand() < 0.4
            placements = _sample_marker_layout(patches, lower_h, overall_w, want_overlap)
            if placements is None:
                continue

            lower = np.zeros((lower_h, overall_w), dtype=int)
            markers: list[tuple[int, int, int]] = []
            for color, (r, c) in zip(marker_colors, placements):
                lower[r, c] = color
                markers.append((int(r), int(c), int(color)))

            grid = np.zeros((upper_h + 1 + lower_h, overall_w), dtype=int)
            left_pad = int(np.random.randint(0, overall_w - upper_w + 1))
            grid[:upper_h, left_pad : left_pad + upper_w] = upper
            grid[upper_h, :] = SEP
            grid[upper_h + 1 :, :] = lower

            expected = _render_lower_region(lower_h, overall_w, markers, motif)
            produced = main(grid)
            if not np.array_equal(produced, expected):
                continue

            if want_overlap:
                nonzero_masks = []
                for r, c, color in markers:
                    patch = np.rot90(motif, k=-MARKER_TO_TURNS[color])
                    mask = np.zeros_like(expected, dtype=bool)
                    ph, pw = patch.shape
                    mask[r : r + ph, c : c + pw] = patch != BG
                    nonzero_masks.append(mask)
                overlap_pairs = 0
                for i in range(len(nonzero_masks)):
                    for j in range(i + 1, len(nonzero_masks)):
                        if np.logical_and(nonzero_masks[i], nonzero_masks[j]).any():
                            overlap_pairs += 1
                if overlap_pairs != 1:
                    continue

            return grid
        except RuntimeError:
            continue
    raise RuntimeError("failed to generate a valid input")
