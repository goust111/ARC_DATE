from collections import deque

import numpy as np


BG = 0
WALL = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

GEN_ATTEMPTS = 400
SHAPE_ATTEMPTS = 120
MIN_REGIONS = 1
MAX_REGIONS = 4


def _components(mask: np.ndarray) -> list[tuple[list[tuple[int, int]], bool]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[tuple[list[tuple[int, int]], bool]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            touches_border = False
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                if rr == 0 or rr == h - 1 or cc == 0 or cc == w - 1:
                    touches_border = True
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append((coords, touches_border))
    return comps


def _peel_layers(mask: np.ndarray) -> list[np.ndarray]:
    current = mask.astype(bool).copy()
    layers: list[np.ndarray] = []
    while np.any(current):
        padded = np.pad(current, 1, constant_values=False)
        interior = (
            current
            & padded[:-2, 1:-1]
            & padded[2:, 1:-1]
            & padded[1:-1, :-2]
            & padded[1:-1, 2:]
        )
        boundary = current & ~interior
        layers.append(boundary)
        current = interior
    return layers


def _layer_count(mask: np.ndarray) -> int:
    return len(_peel_layers(mask))


def _is_connected(mask: np.ndarray) -> bool:
    if not np.any(mask):
        return False
    comps = _components(mask)
    return len(comps) == 1


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow seed-colored inward contour rings inside each grey-walled enclosed region.

    Input structure:
    - The background is black.
    - One or more one-pixel-thick grey outlines form closed regions on the grid.
    - Each closed region contains one or more seed pixels of a single non-grey color.
    - Outside the grey walls there is only black background, and the grey walls remain fixed.

    Rule:
    - Keep every grey wall pixel unchanged.
    - For each enclosed region, count its seed pixels to get N and read their shared color.
    - Clear the original seeds, then color the first N inward contour layers of that region,
      where each layer is the current region boundary under 4-neighbor erosion.
    - If the region runs out of interior layers before N is exhausted, fill all remaining cells.

    Key insight:
    - Seed positions do not control the geometry; they only choose the ring color and how many
      inward boundary layers of the enclosed shape should be drawn.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    out = g.copy()
    for coords, touches_border in _components(g != WALL):
        if touches_border:
            continue

        values = [int(g[r, c]) for r, c in coords]
        seed_colors = {value for value in values if value not in (BG, WALL)}
        assert len(seed_colors) <= 1, "each closed region must use a single seed color"

        for r, c in coords:
            out[r, c] = BG

        if not seed_colors:
            continue

        color = next(iter(seed_colors))
        n_layers = sum(1 for value in values if value == color)
        r0, c0, r1, c1 = _bbox(coords)
        local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in coords:
            local[r - r0, c - c0] = True

        for layer in _peel_layers(local)[:n_layers]:
            rr, cc = np.where(layer)
            out[r0 + rr, c0 + cc] = color

    return out


def _carve_notch(rng: np.random.Generator, mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    trial = mask.copy()
    side = rng.choice(("top", "bottom", "left", "right"))
    if side in ("top", "bottom"):
        if h < 3 or w < 3:
            return mask
        depth = int(rng.integers(1, min(h - 1, max(2, h // 2 + 1))))
        span = int(rng.integers(1, w))
        start = int(rng.integers(0, w - span + 1))
        rows = slice(0, depth) if side == "top" else slice(h - depth, h)
        trial[rows, start : start + span] = False
    else:
        if h < 3 or w < 3:
            return mask
        depth = int(rng.integers(1, min(w - 1, max(2, w // 2 + 1))))
        span = int(rng.integers(1, h))
        start = int(rng.integers(0, h - span + 1))
        cols = slice(0, depth) if side == "left" else slice(w - depth, w)
        trial[start : start + span, cols] = False

    if trial.sum() < 6 or not _is_connected(trial):
        return mask
    return trial


def _build_interior_mask(rng: np.random.Generator, want_partial: bool) -> tuple[np.ndarray, int] | None:
    for _ in range(SHAPE_ATTEMPTS):
        h = int(rng.integers(3, 9))
        w = int(rng.integers(3, 9))
        mask = np.ones((h, w), dtype=bool)

        notch_count = int(rng.integers(0, 4))
        for _ in range(notch_count):
            mask = _carve_notch(rng, mask)

        if int(mask.sum()) < 6 or not _is_connected(mask):
            continue

        depth = _layer_count(mask)
        if want_partial and depth < 2:
            continue
        return mask, depth
    return None


def _seed_count(rng: np.random.Generator, depth: int, area: int, want_partial: bool) -> int:
    if want_partial and depth >= 2:
        return int(rng.integers(1, depth))
    extra = min(4, area - depth)
    return depth + int(rng.integers(0, extra + 1))


def _build_patch(
    rng: np.random.Generator, color: int, want_partial: bool
) -> np.ndarray | None:
    built = _build_interior_mask(rng, want_partial)
    if built is None:
        return None
    interior, depth = built
    area = int(interior.sum())
    n_seeds = _seed_count(rng, depth, area, want_partial)

    h, w = interior.shape
    inner = np.zeros((h + 2, w + 2), dtype=bool)
    inner[1:-1, 1:-1] = interior
    wall = np.zeros_like(inner)
    for r, c in np.argwhere(inner):
        rr = int(r)
        cc = int(c)
        for dr, dc in DIR4:
            nr, nc = rr + dr, cc + dc
            if not inner[nr, nc]:
                wall[nr, nc] = True

    patch = np.zeros_like(inner, dtype=int)
    patch[wall] = WALL

    interior_coords = np.argwhere(inner)
    chosen = rng.choice(len(interior_coords), size=n_seeds, replace=False)
    for idx in np.atleast_1d(chosen):
        r, c = interior_coords[int(idx)]
        patch[int(r), int(c)] = color
    return patch


def _layout_rows(
    rng: np.random.Generator, patches: list[np.ndarray]
) -> np.ndarray | None:
    order = list(range(len(patches)))
    rng.shuffle(order)
    ordered = [patches[idx] for idx in order]

    if len(ordered) == 1 or rng.random() < 0.5:
        rows = [ordered]
    else:
        split = int(rng.integers(1, len(ordered)))
        rows = [ordered[:split], ordered[split:]]

    row_specs = []
    total_width = 0
    top_margin = int(rng.integers(1, 4))
    bottom_margin = int(rng.integers(1, 4))
    extra_width = int(rng.integers(0, 4))
    extra_height = int(rng.integers(0, 4))
    total_height = top_margin
    vertical_gap = int(rng.integers(2, 5)) if len(rows) == 2 else 0

    for row in rows:
        left_margin = int(rng.integers(1, 4))
        right_margin = int(rng.integers(1, 4))
        gaps = [int(rng.integers(2, 5)) for _ in range(max(0, len(row) - 1))]
        row_height = max(int(patch.shape[0]) for patch in row)
        row_width = left_margin + right_margin + sum(int(patch.shape[1]) for patch in row) + sum(gaps)
        row_specs.append((row, row_height, row_width, left_margin, gaps))
        total_width = max(total_width, row_width)
        total_height += row_height

    total_height += bottom_margin
    if len(rows) == 2:
        total_height += vertical_gap

    total_width += extra_width
    total_height += extra_height
    if total_width > 30 or total_height > 30:
        return None

    grid = np.zeros((total_height, total_width), dtype=int)
    y = top_margin
    for row_index, (row, row_height, row_width, left_margin, gaps) in enumerate(row_specs):
        row_offset = int(rng.integers(0, total_width - row_width + 1))
        x = row_offset + left_margin
        for idx, patch in enumerate(row):
            ph, pw = patch.shape
            dy = int(rng.integers(0, row_height - ph + 1))
            sub = grid[y + dy : y + dy + ph, x : x + pw]
            if np.any((sub != BG) & (patch != BG)):
                return None
            sub[...] = np.where(patch != BG, patch, sub)
            if idx < len(gaps):
                x += pw + gaps[idx]
            else:
                x += pw
        y += row_height
        if row_index == 0 and len(rows) == 2:
            y += vertical_gap

    return grid


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    region_count = int(rng.integers(MIN_REGIONS, MAX_REGIONS + 1))
    colors = rng.choice(SEED_COLORS, size=region_count, replace=False)

    patches: list[np.ndarray] = []
    for color in colors:
        want_partial = bool(rng.integers(0, 2))
        patch = _build_patch(rng, int(color), want_partial)
        if patch is None:
            return None
        patches.append(patch)

    grid = _layout_rows(rng, patches)
    if grid is None:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
