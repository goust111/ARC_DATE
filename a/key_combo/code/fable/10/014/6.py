from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


def get_bg(grid: np.ndarray) -> int:
    """Return the most common color in the grid."""
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def neighbors4(r: int, c: int):
    """Yield 4-neighbors in (row, col) coordinates."""
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components of a boolean mask as coordinate arrays."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _dilate8(mask: np.ndarray) -> np.ndarray:
    """One-step 8-neighbor dilation."""
    h, w = mask.shape
    padded = np.pad(mask.astype(bool), 1, constant_values=False)
    out = np.zeros((h, w), dtype=bool)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            out |= padded[1 + dr : 1 + dr + h, 1 + dc : 1 + dc + w]
    return out


def _erosion_depths(mask: np.ndarray) -> np.ndarray:
    """Assign 4-neighbor onion depth to every True cell of mask."""
    remaining = mask.astype(bool).copy()
    depths = np.zeros(mask.shape, dtype=int)
    layer = 1
    while np.any(remaining):
        padded = np.pad(remaining, 1, constant_values=False)
        interior = (
            remaining
            & padded[:-2, 1:-1]
            & padded[2:, 1:-1]
            & padded[1:-1, :-2]
            & padded[1:-1, 2:]
        )
        boundary = remaining & ~interior
        depths[boundary] = layer
        remaining[boundary] = False
        layer += 1
    return depths


def _find_legend(grid: np.ndarray, bg: int) -> tuple[np.ndarray, list[int]]:
    """Locate the unique corner-anchored single-column legend and return its mask/colors."""
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates: list[tuple[np.ndarray, list[int]]] = []
    for coords in _components(g != bg):
        rows = sorted(int(r) for r in coords[:, 0])
        cols = sorted(set(int(c) for c in coords[:, 1]))
        if len(cols) != 1:
            continue
        if rows != list(range(rows[0], rows[-1] + 1)):
            continue
        if len(rows) != len(coords):
            continue
        c = cols[0]
        if c not in (0, w - 1):
            continue
        if rows[0] != 0 and rows[-1] != h - 1:
            continue
        colors = [int(g[r, c]) for r in rows]
        if len(set(colors)) != len(colors):
            continue
        mask = np.zeros_like(g, dtype=bool)
        mask[coords[:, 0], coords[:, 1]] = True
        candidates.append((mask, colors))
    assert len(candidates) == 1, "expected a unique corner legend bar"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor each solid blob by 4-neighbor erosion depth, cycling through a corner legend bar.

    Input structure:
    - The background is a single dominant color.
    - One corner contains a vertical one-column legend bar whose cells are all different colors.
    - Every other non-background pixel belongs to one of several solid monochrome blobs.

    Rule:
    - Leave the background and the legend bar unchanged.
    - For each blob pixel, compute its onion depth inside that blob: depth 1 touches background by a 4-neighbor step, and each inward erosion layer increases the depth by 1.
    - Recolor depth d with the legend color in row ((d-1) mod k) from the top, where k is the legend height.

    Key insight:
    - The legend is an ordered depth palette rather than a direct color-substitution key, and thick blobs force the palette to wrap and repeat.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = get_bg(g)
    legend_mask, legend_colors = _find_legend(g, bg)
    blob_mask = (g != bg) & (~legend_mask)
    assert np.any(blob_mask), "expected at least one blob"

    depths = _erosion_depths(blob_mask)
    out = g.copy()
    k = len(legend_colors)
    for depth in range(1, int(depths.max()) + 1):
        out[depths == depth] = legend_colors[(depth - 1) % k]
    return out


GEN_ATTEMPTS = 400
GEN_GRID_MIN = 18
GEN_GRID_MAX = 30
GEN_LEGEND_K_RANGE = (3, 5)
GEN_BLOB_COUNT_RANGE = (2, 4)


def _randint(rng: random.Random, lo: int, hi: int) -> int:
    return int(rng.randint(int(lo), int(hi)))


def _make_blob(rng: random.Random, min_depth: int, *, irregular_bias: float = 0.8) -> np.ndarray:
    """Build a connected solid blob from a deep core rectangle plus optional side protrusions."""
    base_h = 2 * int(min_depth) - 1 + _randint(rng, 0, 3)
    base_w = 2 * int(min_depth) - 1 + _randint(rng, 0, 4)
    max_ext = max(2, int(min_depth) + 1)
    canvas = np.zeros((base_h + 2 * max_ext, base_w + 2 * max_ext), dtype=bool)
    r0 = max_ext
    c0 = max_ext
    canvas[r0 : r0 + base_h, c0 : c0 + base_w] = True

    if rng.random() < irregular_bias:
        sides = ["top", "bottom", "left", "right"]
        rng.shuffle(sides)
        num_ext = _randint(rng, 1, min(3, len(sides)))
        for side in sides[:num_ext]:
            if side in ("top", "bottom"):
                ext_h = _randint(rng, 1, max(1, int(min_depth)))
                span_lo = min(base_w, max(1, int(min_depth)))
                ext_w = _randint(rng, span_lo, base_w)
                start_c = c0 + _randint(rng, 0, base_w - ext_w)
                if side == "top":
                    canvas[r0 - ext_h : r0, start_c : start_c + ext_w] = True
                else:
                    canvas[r0 + base_h : r0 + base_h + ext_h, start_c : start_c + ext_w] = True
            else:
                ext_w = _randint(rng, 1, max(1, int(min_depth)))
                span_lo = min(base_h, max(1, int(min_depth)))
                ext_h = _randint(rng, span_lo, base_h)
                start_r = r0 + _randint(rng, 0, base_h - ext_h)
                if side == "left":
                    canvas[start_r : start_r + ext_h, c0 - ext_w : c0] = True
                else:
                    canvas[start_r : start_r + ext_h, c0 + base_w : c0 + base_w + ext_w] = True

    coords = np.argwhere(canvas)
    r1, c1 = coords.min(axis=0)
    r2, c2 = coords.max(axis=0)
    blob = canvas[r1 : r2 + 1, c1 : c2 + 1]
    assert int(_erosion_depths(blob).max()) >= int(min_depth)
    return blob


def _place_patch(
    rng: random.Random,
    forbidden: np.ndarray,
    patch: np.ndarray,
    *,
    border: int = 1,
    max_tries: int = 300,
) -> tuple[int, int] | None:
    """Place a boolean patch where its True cells avoid forbidden cells."""
    h, w = forbidden.shape
    ph, pw = patch.shape
    if h - ph - border < border or w - pw - border < border:
        return None
    for _ in range(max_tries):
        r0 = _randint(rng, border, h - ph - border)
        c0 = _randint(rng, border, w - pw - border)
        if not np.any(forbidden[r0 : r0 + ph, c0 : c0 + pw] & patch):
            return r0, c0
    return None


def _legend_layout(h: int, w: int, k: int, corner: str) -> tuple[np.ndarray, list[tuple[int, int]]]:
    mask = np.zeros((h, w), dtype=bool)
    if "top" in corner:
        rows = list(range(k))
    else:
        rows = list(range(h - k, h))
    col = 0 if "left" in corner else w - 1
    cells = [(r, col) for r in rows]
    for r, c in cells:
        mask[r, c] = True
    return mask, cells


def _oracle_expected(grid: np.ndarray) -> np.ndarray:
    """Independent semantic oracle used during generation-time validation."""
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    legend_mask, legend_colors = _find_legend(g, bg)
    blob_mask = (g != bg) & (~legend_mask)
    depths = _erosion_depths(blob_mask)
    out = g.copy()
    k = len(legend_colors)
    coords = np.argwhere(blob_mask)
    for r, c in coords:
        d = int(depths[r, c])
        out[r, c] = legend_colors[(d - 1) % k]
    return out


def _validate_grid(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or max(g.shape) > 30:
        return False

    bg = get_bg(g)
    if np.count_nonzero(g == bg) <= (g.size // 2):
        return False

    try:
        legend_mask, legend_colors = _find_legend(g, bg)
    except AssertionError:
        return False

    blob_mask = (g != bg) & (~legend_mask)
    if not np.any(blob_mask):
        return False

    comps = _components(blob_mask)
    if len(comps) < 2:
        return False

    blob_colors = sorted(set(map(int, g[blob_mask])))
    if len(blob_colors) != 1:
        return False

    depths = _erosion_depths(blob_mask)
    max_depth = int(depths.max())
    k = len(legend_colors)
    if max_depth <= k:
        return False

    # Require the palette-wrap color to appear at two different depths.
    if not np.any(depths == 1) or not np.any(depths == k + 1):
        return False

    # Ensure blobs are visually separated and away from the legend.
    if np.any(_dilate8(blob_mask) & legend_mask):
        return False

    expected = _oracle_expected(g)
    try:
        solved = main(g)
    except AssertionError:
        return False
    if not np.array_equal(solved, expected):
        return False

    # Semantic checks on the output, not just round-trip equality.
    if np.array_equal(solved, g):
        return False
    if np.count_nonzero(solved == legend_colors[0]) <= np.count_nonzero(legend_mask):
        return False
    return True


def _attempt_generate(rng: random.Random) -> np.ndarray | None:
    k = _randint(rng, *GEN_LEGEND_K_RANGE)
    deep_depth = _randint(rng, k + 1, min(6, k + 2))
    min_side = 2 * deep_depth - 1
    h = _randint(rng, max(GEN_GRID_MIN, min_side + 5), GEN_GRID_MAX)
    w = _randint(rng, max(GEN_GRID_MIN, min_side + 5), GEN_GRID_MAX)

    palette = list(range(10))
    rng.shuffle(palette)
    bg = palette[0]
    blob_color = palette[1]
    legend_colors = palette[2 : 2 + k]

    grid = np.full((h, w), bg, dtype=int)

    corner = rng.choice(("top_left", "top_right", "bottom_left", "bottom_right"))
    legend_mask, legend_cells = _legend_layout(h, w, k, corner)
    for color, (r, c) in zip(legend_colors, legend_cells):
        grid[r, c] = color

    occupied = legend_mask.copy()
    forbidden = _dilate8(occupied)
    forbidden[0, :] = True
    forbidden[-1, :] = True
    forbidden[:, 0] = True
    forbidden[:, -1] = True

    patches: list[tuple[np.ndarray, int, int]] = []
    deep_patch = _make_blob(rng, deep_depth, irregular_bias=0.9)
    loc = _place_patch(rng, forbidden, deep_patch)
    if loc is None:
        return None
    r0, c0 = loc
    patches.append((deep_patch, r0, c0))
    occupied[r0 : r0 + deep_patch.shape[0], c0 : c0 + deep_patch.shape[1]] |= deep_patch
    forbidden = _dilate8(occupied)
    forbidden[0, :] = True
    forbidden[-1, :] = True
    forbidden[:, 0] = True
    forbidden[:, -1] = True

    blob_count = _randint(rng, *GEN_BLOB_COUNT_RANGE)
    for _ in range(blob_count - 1):
        shallow_depth = rng.choice([1, 1, 2, 2, 3, min(4, k)])
        patch = _make_blob(rng, shallow_depth, irregular_bias=0.75)
        loc = _place_patch(rng, forbidden, patch)
        if loc is None:
            continue
        r0, c0 = loc
        patches.append((patch, r0, c0))
        occupied[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]] |= patch
        forbidden = _dilate8(occupied)
        forbidden[0, :] = True
        forbidden[-1, :] = True
        forbidden[:, 0] = True
        forbidden[:, -1] = True

    if len(patches) < 2:
        return None

    for patch, r0, c0 in patches:
        ph, pw = patch.shape
        grid[r0 : r0 + ph, c0 : c0 + pw][patch] = blob_color

    if not _validate_grid(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
