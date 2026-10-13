from collections import deque

import numpy as np


BG = 0
MAX_GENERATION_ATTEMPTS = 300
LAYER_CHOICES = {
    0: np.array([3, 5, 7], dtype=int),
    1: np.array([2, 4, 6], dtype=int),
}
COLOR_POOL = np.arange(1, 10, dtype=int)
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _largest_nonblack_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    mask = grid != BG
    assert bool(mask.any()), "expected a non-black nested structure"

    h, w = map(int, grid.shape)
    seen = np.zeros((h, w), dtype=bool)
    best_area = -1
    best_bbox = None

    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if bool(seen[r, c]):
            continue

        q: deque[tuple[int, int]] = deque([(r, c)])
        seen[r, c] = True
        area = 0
        r0 = r1 = r
        c0 = c1 = c

        while q:
            cr, cc = q.popleft()
            area += 1
            r0 = min(r0, cr)
            r1 = max(r1, cr)
            c0 = min(c0, cc)
            c1 = max(c1, cc)

            for dr, dc in NEIGHBORS4:
                nr = int(cr + dr)
                nc = int(cc + dc)
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if bool(seen[nr, nc]) or not bool(mask[nr, nc]):
                    continue
                seen[nr, nc] = True
                q.append((nr, nc))

        if area > best_area:
            best_area = int(area)
            best_bbox = (int(r0), int(c0), int(r1), int(c1))

    assert best_bbox is not None
    return best_bbox


def _layer_depths(height: int, width: int) -> np.ndarray:
    rows = np.arange(height, dtype=int)[:, None]
    cols = np.arange(width, dtype=int)[None, :]
    return np.minimum(
        np.minimum(rows, height - 1 - rows),
        np.minimum(cols, width - 1 - cols),
    )


def _majority_color(values: np.ndarray) -> int:
    counts = np.bincount(values.astype(int).ravel(), minlength=10)
    return int(np.argmax(counts))


def _repair_core(core: np.ndarray) -> np.ndarray:
    h, w = map(int, core.shape)
    depths = _layer_depths(h, w)
    n_layers = int(depths.max()) + 1

    layer_colors = []
    for layer in range(n_layers):
        layer_colors.append(_majority_color(core[depths == layer]))

    if n_layers % 2 == 0:
        layer_colors = layer_colors[::-1]

    repaired = np.zeros_like(core)
    for layer, color in enumerate(layer_colors):
        repaired[depths == layer] = int(color)
    return repaired


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore a noisy stack of concentric rectangular frames and reverse the layer colors when the stack depth is even.

    Input structure:
    - The background is black.
    - One large non-black component is a tightly nested set of one-pixel-thick concentric rectangular frames, so its bounding box is fully filled.
    - Some pixels inside the structure are corrupted to colors belonging to other layers, and extra isolated colored noise pixels may appear in the outer black area.

    Rule:
    - Find the largest non-black connected component and treat its bounding box as the full nested rectangle; clear every colored pixel outside that box back to black.
    - Within the box, group cells by rectangular layer using their minimum distance to the box boundary, then repaint each layer with its majority color.
    - If the total number of layers is odd, keep that outside-to-inside color order; if it is even, reverse the recovered layer-color order before repainting.

    Key insight:
    - Geometry identifies the layers exactly: the ring index of a cell is its depth from the outer bounding box, so denoising is a per-layer majority vote followed by a parity-controlled color-order reversal.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    if not bool((g != BG).any()):
        return g.copy()

    r0, c0, r1, c1 = _largest_nonblack_bbox(g)
    core = g[r0 : r1 + 1, c0 : c1 + 1]
    repaired = _repair_core(core)

    out = np.zeros_like(g)
    out[r0 : r1 + 1, c0 : c1 + 1] = repaired
    return out


def _clean_core(layer_colors: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = map(int, shape)
    depths = _layer_depths(h, w)
    core = np.zeros((h, w), dtype=int)
    for layer, color in enumerate(layer_colors):
        core[depths == layer] = int(color)
    return core


def _expected_output(
    grid_shape: tuple[int, int],
    top: int,
    left: int,
    layer_colors: np.ndarray,
    core_shape: tuple[int, int],
) -> np.ndarray:
    colors = layer_colors[::-1] if len(layer_colors) % 2 == 0 else layer_colors
    out = np.zeros(grid_shape, dtype=int)
    core = _clean_core(colors, core_shape)
    h, w = map(int, core_shape)
    out[top : top + h, left : left + w] = core
    return out


def _sample_core_shape(rng: np.random.Generator, n_layers: int) -> tuple[int, int]:
    min_dim = int(rng.choice(np.array([2 * n_layers - 1, 2 * n_layers], dtype=int)))
    long_dim = int(rng.integers(min_dim, min(24, min_dim + 8) + 1))
    if bool(rng.integers(0, 2)):
        return min_dim, long_dim
    return long_dim, min_dim


def _corrupt_core(rng: np.random.Generator, clean: np.ndarray, layer_colors: np.ndarray) -> np.ndarray:
    depths = _layer_depths(*clean.shape)
    corrupted = clean.copy()
    candidates = []
    total_noise = 0

    for layer, color in enumerate(layer_colors):
        positions = np.argwhere(depths == layer)
        count = int(len(positions))
        max_noise = int(count // 3)
        if max_noise > 0:
            candidates.append(layer)

        noise_count = int(rng.integers(0, max_noise + 1))
        if noise_count == 0:
            continue

        chosen = positions[rng.choice(count, size=noise_count, replace=False)]
        others = layer_colors[layer_colors != int(color)]
        replacements = rng.choice(others, size=noise_count, replace=True)
        for (rr, cc), repl in zip(chosen, replacements):
            corrupted[int(rr), int(cc)] = int(repl)
        total_noise += noise_count

    if total_noise == 0 and candidates:
        layer = int(rng.choice(np.array(candidates, dtype=int)))
        positions = np.argwhere(depths == layer)
        rr, cc = positions[int(rng.integers(len(positions)))]
        others = layer_colors[layer_colors != int(layer_colors[layer])]
        corrupted[int(rr), int(cc)] = int(rng.choice(others))

    return corrupted


def _can_place_isolated_noise(grid: np.ndarray, r: int, c: int) -> bool:
    h, w = map(int, grid.shape)
    if int(grid[r, c]) != BG:
        return False
    for nr in range(max(0, r - 1), min(h, r + 2)):
        for nc in range(max(0, c - 1), min(w, c + 2)):
            if int(grid[nr, nc]) != BG:
                return False
    return True


def _add_external_noise(rng: np.random.Generator, grid: np.ndarray) -> None:
    target = 0 if bool(rng.integers(0, 4) == 0) else int(rng.integers(1, 5))
    for _ in range(target):
        candidates = []
        h, w = map(int, grid.shape)
        for r in range(h):
            for c in range(w):
                if _can_place_isolated_noise(grid, r, c):
                    candidates.append((r, c))
        if not candidates:
            break
        rr, cc = candidates[int(rng.integers(len(candidates)))]
        grid[int(rr), int(cc)] = int(rng.choice(COLOR_POOL))


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    parity = int(rng.integers(0, 2))
    n_layers = int(rng.choice(LAYER_CHOICES[parity]))

    core_h, core_w = _sample_core_shape(rng, n_layers)
    assert (min(core_h, core_w) + 1) // 2 == n_layers

    extra_h = int(rng.integers(4, 30 - core_h + 1))
    extra_w = int(rng.integers(4, 30 - core_w + 1))
    grid_h = int(core_h + extra_h)
    grid_w = int(core_w + extra_w)

    top = int(rng.integers(1, extra_h))
    left = int(rng.integers(1, extra_w))

    layer_colors = rng.choice(COLOR_POOL, size=n_layers, replace=False)
    clean = _clean_core(layer_colors, (core_h, core_w))
    corrupted = _corrupt_core(rng, clean, layer_colors)

    grid = np.zeros((grid_h, grid_w), dtype=int)
    grid[top : top + core_h, left : left + core_w] = corrupted
    _add_external_noise(rng, grid)

    expected = _expected_output(grid.shape, top, left, layer_colors, (core_h, core_w))
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid concentric-frame sample")
