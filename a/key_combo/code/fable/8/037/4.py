from __future__ import annotations

from collections import Counter, deque

import numpy as np


_RNG = np.random.default_rng()
_DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_DIRS8 = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)
_COLORS = tuple(range(10))
_GEN_RETRY_LIMIT = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    dirs = _DIRS4 if connectivity == 4 else _DIRS8
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in dirs:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.asarray(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _dilate(mask: np.ndarray, connectivity: int = 8) -> np.ndarray:
    h, w = mask.shape
    dirs = _DIRS4 if connectivity == 4 else _DIRS8
    out = mask.copy()
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        for dr, dc in dirs:
            nr, nc = rr + dr, cc + dc
            if _in_bounds(mask.shape, nr, nc):
                out[nr, nc] = True
    return out


def _checkerboard(shape: tuple[int, int], even_color: int, odd_color: int) -> np.ndarray:
    rows, cols = np.indices(shape)
    return np.where((rows + cols) % 2 == 0, int(even_color), int(odd_color)).astype(int)


def _top_two_colors(grid: np.ndarray) -> tuple[int, int]:
    counts = Counter(map(int, grid.ravel()))
    (c0, _), (c1, _) = counts.most_common(2)
    return int(c0), int(c1)


def _infer_reference_checkerboard(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    c0, c1 = _top_two_colors(grid)
    pat0 = _checkerboard(grid.shape, c0, c1)
    pat1 = _checkerboard(grid.shape, c1, c0)
    corners = ((0, 0), (0, grid.shape[1] - 1), (grid.shape[0] - 1, 0), (grid.shape[0] - 1, grid.shape[1] - 1))
    score0 = sum(int(grid[r, c] == pat0[r, c]) for r, c in corners)
    score1 = sum(int(grid[r, c] == pat1[r, c]) for r, c in corners)
    assert score0 != score1, "corner phase must determine a unique checkerboard reference"
    expected = pat0 if score0 > score1 else pat1
    even_color = int(expected[0, 0])
    odd_color = c1 if even_color == c0 else c0
    return expected, (even_color, odd_color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each hidden checkerboard phase-flip region with the color of its internal marker pixel.

    Input structure:
    - The whole canvas is a two-color checkerboard, except for a few third-party single-pixel markers.
    - The four corners stay on the normal checkerboard phase and determine the global reference phase.
    - Some 4-connected regions have their checkerboard colors globally swapped, and each such region contains exactly one marker.
    - Additional marker pixels may lie on normal-phase checkerboard cells as decoys.

    Rule:
    - Infer the two checkerboard colors and the unique global phase consistent with the four corners.
    - Find all pixels whose checkerboard color disagrees with that reference phase; together with any marker embedded in them, these form the hidden flipped regions.
    - For every flipped region, fill the entire region, including its marker, with that region's unique internal marker color.
    - Keep the normal-phase checkerboard and all decoy markers unchanged.

    Key insight:
    - The target regions are invisible in shape and are recoverable only by comparing each cell's checkerboard phase against the corner-defined reference; markers provide colors, not boundaries.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    expected, base_colors = _infer_reference_checkerboard(g)
    c0, c1 = base_colors
    marker_mask = (g != c0) & (g != c1)
    checker_mask = ~marker_mask
    assert np.all((g[checker_mask] == c0) | (g[checker_mask] == c1))

    phase_flipped = checker_mask & (g != expected)
    candidate_mask = phase_flipped | marker_mask

    out = g.copy()
    for comp in _components(candidate_mask, connectivity=4):
        rr = comp[:, 0]
        cc = comp[:, 1]
        has_phase = phase_flipped[rr, cc]
        if not np.any(has_phase):
            continue
        has_marker = marker_mask[rr, cc]
        assert int(has_marker.sum()) == 1, "every flipped region must contain exactly one marker"
        marker_r, marker_c = map(int, comp[np.flatnonzero(has_marker)[0]])
        out[rr, cc] = int(g[marker_r, marker_c])

    return out


def _grow_blob(h: int, w: int, target: int) -> np.ndarray:
    for _ in range(80):
        mask = np.zeros((h, w), dtype=bool)
        start = (int(_RNG.integers(h)), int(_RNG.integers(w)))
        mask[start] = True
        while int(mask.sum()) < target:
            frontier = []
            for r, c in np.argwhere(mask):
                rr = int(r)
                cc = int(c)
                for dr, dc in _DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and not mask[nr, nc]:
                        frontier.append((nr, nc))
            if not frontier:
                break
            pick = frontier[int(_RNG.integers(len(frontier)))]
            mask[pick] = True
        coords = np.argwhere(mask)
        if coords.size == 0:
            continue
        r0, c0, r1, c1 = _bbox(coords)
        if (r1 - r0 + 1) >= 2 and (c1 - c0 + 1) >= 2:
            return _crop_mask(mask)
    return np.ones((max(2, h), max(2, w)), dtype=bool)


def _make_region_shape(force_non_rect: bool = False) -> np.ndarray:
    kind_pool = ["rectangle", "l_shape", "cross", "blob"]
    if force_non_rect:
        kind_pool = ["l_shape", "cross", "blob"]
    kind = kind_pool[int(_RNG.integers(len(kind_pool)))]

    h = int(_RNG.integers(3, 8))
    w = int(_RNG.integers(3, 8))

    if kind == "rectangle":
        return np.ones((h, w), dtype=bool)

    if kind == "l_shape":
        mask = np.ones((h, w), dtype=bool)
        cut_h = int(_RNG.integers(1, h))
        cut_w = int(_RNG.integers(1, w))
        corner = int(_RNG.integers(4))
        if corner == 0:
            mask[:cut_h, :cut_w] = False
        elif corner == 1:
            mask[:cut_h, w - cut_w :] = False
        elif corner == 2:
            mask[h - cut_h :, :cut_w] = False
        else:
            mask[h - cut_h :, w - cut_w :] = False
        return _crop_mask(mask)

    if kind == "cross":
        mask = np.zeros((h, w), dtype=bool)
        bar_h = int(_RNG.integers(1, max(2, h)))
        bar_w = int(_RNG.integers(1, max(2, w)))
        if h > 1:
            bar_h = min(bar_h, h - 1)
        if w > 1:
            bar_w = min(bar_w, w - 1)
        r0 = int(_RNG.integers(0, h - bar_h + 1))
        c0 = int(_RNG.integers(0, w - bar_w + 1))
        mask[r0 : r0 + bar_h, :] = True
        mask[:, c0 : c0 + bar_w] = True
        return _crop_mask(mask)

    target = int(_RNG.integers(max(5, (h * w) // 2), h * w + 1))
    return _grow_blob(h, w, target)


def _place_region(existing: np.ndarray, local_mask: np.ndarray) -> np.ndarray | None:
    h, w = existing.shape
    rh, rw = local_mask.shape
    if rh > h - 2 or rw > w - 2:
        return None
    row_choices = list(range(1, h - rh))
    col_choices = list(range(1, w - rw))
    _RNG.shuffle(row_choices)
    _RNG.shuffle(col_choices)

    for r0 in row_choices:
        for c0 in col_choices:
            global_mask = np.zeros_like(existing, dtype=bool)
            global_mask[r0 : r0 + rh, c0 : c0 + rw] = local_mask
            if np.any(existing & global_mask):
                continue
            if np.any(_dilate(existing, connectivity=8) & global_mask):
                continue
            return global_mask
    return None


def _pick_marker_cell(region_coords: np.ndarray) -> tuple[int, int]:
    idx = int(_RNG.integers(region_coords.shape[0]))
    return tuple(map(int, region_coords[idx]))


def _sample_decoys(
    grid_shape: tuple[int, int],
    region_mask: np.ndarray,
    true_marker_positions: list[tuple[int, int]],
    true_colors: list[int],
    base_colors: tuple[int, int],
) -> tuple[list[tuple[int, int]], list[int]] | None:
    blocked = _dilate(region_mask, connectivity=4)
    for r, c in true_marker_positions:
        blocked[r, c] = True
    blocked = _dilate(blocked, connectivity=8)
    blocked[0, 0] = blocked[0, -1] = blocked[-1, 0] = blocked[-1, -1] = True

    decoy_count = int(_RNG.integers(1, len(true_colors) + 3))
    positions = []
    colors = []
    free = blocked.copy()

    true_colors_arr = np.asarray(true_colors, dtype=int)
    allowed = np.asarray([c for c in _COLORS if c not in set(map(int, base_colors))], dtype=int)

    for i in range(decoy_count):
        candidates = np.argwhere(~free)
        if candidates.shape[0] == 0:
            return None
        rr, cc = map(int, candidates[int(_RNG.integers(candidates.shape[0]))])
        positions.append((rr, cc))
        if i == 0:
            colors.append(int(true_colors_arr[int(_RNG.integers(true_colors_arr.size))]))
        else:
            source = np.concatenate([true_colors_arr, allowed])
            colors.append(int(source[int(_RNG.integers(source.size))]))
        free[max(0, rr - 1) : rr + 2, max(0, cc - 1) : cc + 2] = True

    return positions, colors


def _build_instance() -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[int], list[tuple[int, int]], list[tuple[int, int]], list[int]] | None:
    h = int(_RNG.integers(12, 25))
    w = int(_RNG.integers(12, 25))
    base_colors = [int(c) for c in _RNG.choice(_COLORS, size=2, replace=False)]
    even_color = base_colors[int(_RNG.integers(2))]
    odd_color = base_colors[1] if base_colors[0] == even_color else base_colors[0]
    base = _checkerboard((h, w), even_color, odd_color)

    n_regions = int(_RNG.choice(np.asarray([1, 2, 3]), p=np.asarray([0.2, 0.5, 0.3])))
    placed_regions: list[np.ndarray] = []
    occupancy = np.zeros((h, w), dtype=bool)
    non_rect_seen = False

    for idx in range(n_regions):
        region = None
        for _ in range(120):
            force_non_rect = idx == n_regions - 1 and not non_rect_seen
            local = _make_region_shape(force_non_rect=force_non_rect)
            coords = np.argwhere(local)
            r0, c0, r1, c1 = _bbox(coords)
            if (r1 - r0 + 1) < 2 or (c1 - c0 + 1) < 2 or coords.shape[0] < 5:
                continue
            region = _place_region(occupancy, local)
            if region is None:
                continue
            non_rect_seen |= int(coords.shape[0]) != local.shape[0] * local.shape[1]
            break
        if region is None:
            return None
        occupancy |= region
        placed_regions.append(region)

    if not non_rect_seen:
        return None

    inp = base.copy()
    inp[occupancy] = np.where(base[occupancy] == even_color, odd_color, even_color)

    available_marker_colors = [c for c in _COLORS if c not in {even_color, odd_color}]
    true_colors = [int(available_marker_colors[int(_RNG.integers(len(available_marker_colors)))]) for _ in placed_regions]
    true_positions: list[tuple[int, int]] = []
    for region, color in zip(placed_regions, true_colors):
        pos = _pick_marker_cell(np.argwhere(region))
        true_positions.append(pos)
        inp[pos] = color

    decoys = _sample_decoys(inp.shape, occupancy, true_positions, true_colors, (even_color, odd_color))
    if decoys is None:
        return None
    decoy_positions, decoy_colors = decoys
    for pos, color in zip(decoy_positions, decoy_colors):
        inp[pos] = color

    out = inp.copy()
    for region, color in zip(placed_regions, true_colors):
        out[region] = color

    return inp, out, placed_regions, true_colors, true_positions, decoy_positions, decoy_colors


def _validate_instance(
    inp: np.ndarray,
    out: np.ndarray,
    regions: list[np.ndarray],
    true_colors: list[int],
    true_positions: list[tuple[int, int]],
    decoy_positions: list[tuple[int, int]],
    decoy_colors: list[int],
) -> bool:
    if inp.ndim != 2 or max(inp.shape) > 30 or max(out.shape) > 30:
        return False
    if len(regions) == 0 or len(decoy_positions) == 0:
        return False
    if not np.array_equal(main(inp), out):
        return False

    expected, base_colors = _infer_reference_checkerboard(inp)
    c0, c1 = base_colors
    marker_mask = (inp != c0) & (inp != c1)
    phase_flipped = (~marker_mask) & (inp != expected)

    for region, color, pos in zip(regions, true_colors, true_positions):
        coords = np.argwhere(region)
        r0, c0r, r1, c1r = _bbox(coords)
        if coords.shape[0] == (r1 - r0 + 1) * (c1r - c0r + 1):
            continue
        break
    else:
        return False

    for region, color, pos in zip(regions, true_colors, true_positions):
        rr, cc = np.argwhere(region).T
        if int(inp[pos]) != color or not region[pos]:
            return False
        if not np.all(out[rr, cc] == color):
            return False
        region_phase = phase_flipped.copy()
        region_phase[pos] = True
        if not np.array_equal(region_phase & region, region):
            return False

    if not any(decoy_color in true_colors for decoy_color in decoy_colors):
        return False
    for pos in decoy_positions:
        if any(region[pos] for region in regions):
            return False
        if int(out[pos]) != int(inp[pos]):
            return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        built = _build_instance()
        if built is None:
            continue
        inp, out, regions, true_colors, true_positions, decoy_positions, decoy_colors = built
        if _validate_instance(inp, out, regions, true_colors, true_positions, decoy_positions, decoy_colors):
            return inp
    raise RuntimeError("failed to generate a valid phase-defect ARC input")
