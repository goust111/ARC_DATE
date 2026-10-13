from __future__ import annotations

import numpy as np


BLUE = 1
BLACK = 0
DIRS8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]
RNG = np.random.default_rng()
MAX_GENERATION_ATTEMPTS = 500


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 8-connected components of a boolean mask as coord arrays."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in DIRS8:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _shadow_mask(fg: np.ndarray) -> np.ndarray:
    """Shadow cells created by foreground pixels, excluding cells already occupied by foreground."""
    h, w = fg.shape
    coords = np.argwhere(fg)
    assert np.all(coords[:, 0] > 0), "foreground pixels must not occupy the top row"
    shadow = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        sr = int(r) - 1
        sc = (int(c) - 1) % w
        if not fg[sr, sc]:
            shadow[sr, sc] = True
    return shadow


def _survivor_mask(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    fg = g == BLUE
    shadow = _shadow_mask(fg)
    enhanced = fg | shadow

    keep = np.zeros_like(fg, dtype=bool)
    for comp in _components(enhanced):
        blue_cells = fg[comp[:, 0], comp[:, 1]]
        if int(blue_cells.sum()) > 1:
            blue_coords = comp[blue_cells]
            keep[blue_coords[:, 0], blue_coords[:, 1]] = True
    return keep


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove only those blue pixels that remain singleton foreground components even after shadow-based bridge augmentation.

    Input structure:
    - The grid is a black background with blue foreground pixels.
    - Blue pixels may appear as isolated points or as small 8-connected clusters.
    - No blue pixel lies on the top row, so every blue pixel has a valid up-left shadow row.

    Rule:
    - For each blue pixel at (r, c), form a virtual shadow at (r-1, (c-1) mod W).
    - Build an enhanced 8-connected mask from all blue pixels plus only those shadow cells that fall on black background.
    - For each enhanced connected component, count how many true blue pixels it contains and erase every blue pixel whose component contains exactly one blue pixel.
    - Discard all shadow cells from the final result and keep only the surviving blue pixels.

    Key insight:
    - Connectivity is judged on a hidden shadow-augmented layer, so a visually isolated blue pixel can survive if shadows bridge it into a multi-blue 8-connected component.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    keep = _survivor_mask(g)
    out = np.zeros_like(g, dtype=int)
    out[keep] = BLUE
    return out


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _enhanced_cells(coords: list[tuple[int, int]], width: int) -> set[tuple[int, int]]:
    cells = set(coords)
    for r, c in coords:
        cells.add((r - 1, (c - 1) % width))
    return cells


def _can_place(
    coords: list[tuple[int, int]],
    shape: tuple[int, int],
    used_blue: set[tuple[int, int]],
    used_enhanced: set[tuple[int, int]],
) -> bool:
    h, w = shape
    if len(set(coords)) != len(coords):
        return False
    for r, c in coords:
        if not (1 <= r < h and 0 <= c < w):
            return False
        if (r, c) in used_blue:
            return False
    candidate_enhanced = _enhanced_cells(coords, w)
    for cell in candidate_enhanced:
        if cell in used_enhanced:
            return False
        for other in used_enhanced:
            if _chebyshev(cell, other) <= 1:
                return False
    return True


def _add_coords(
    coords: list[tuple[int, int]],
    shape: tuple[int, int],
    used_blue: set[tuple[int, int]],
    used_enhanced: set[tuple[int, int]],
) -> None:
    used_blue.update(coords)
    used_enhanced.update(_enhanced_cells(coords, shape[1]))


def _normalize(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    return sorted((r - min_r, c - min_c) for r, c in coords)


def _random_cluster_shape(size: int) -> list[tuple[int, int]]:
    assert 2 <= size <= 4
    while True:
        cells = {(1, 1)}
        while len(cells) < size:
            frontier: list[tuple[int, int]] = []
            for r, c in cells:
                for dr, dc in DIRS8:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < 3 and 0 <= nc < 3 and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            pick = frontier[int(RNG.integers(len(frontier)))]
            cells.add(pick)
        coords = _normalize(list(cells))
        if len(set(coords)) == size:
            return coords


def _place_cluster(
    size: int,
    shape: tuple[int, int],
    used_blue: set[tuple[int, int]],
    used_enhanced: set[tuple[int, int]],
) -> bool:
    h, w = shape
    for _ in range(120):
        rel = _random_cluster_shape(size)
        max_r = max(r for r, _ in rel)
        max_c = max(c for _, c in rel)
        r0 = int(RNG.integers(1, h - max_r))
        c0 = int(RNG.integers(0, w - max_c))
        coords = [(r0 + r, c0 + c) for r, c in rel]
        if _can_place(coords, shape, used_blue, used_enhanced):
            _add_coords(coords, shape, used_blue, used_enhanced)
            return True
    return False


def _place_singleton(
    shape: tuple[int, int],
    used_blue: set[tuple[int, int]],
    used_enhanced: set[tuple[int, int]],
) -> bool:
    h, w = shape
    for _ in range(200):
        coord = (int(RNG.integers(1, h)), int(RNG.integers(0, w)))
        if _can_place([coord], shape, used_blue, used_enhanced):
            _add_coords([coord], shape, used_blue, used_enhanced)
            return True
    return False


def _bridge_coords(kind: str, shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    if kind == "diag_adjacent":
        r = int(RNG.integers(1, h - 2))
        c = int(RNG.integers(0, w - 2))
        return [(r, c), (r + 2, c + 2)]
    if kind == "ortho_adjacent":
        r = int(RNG.integers(1, h - 2))
        c = int(RNG.integers(0, w - 1))
        return [(r, c), (r + 2, c + 1)]
    if kind == "wrap_overlap":
        r = int(RNG.integers(2, h))
        return [(r - 1, w - 1), (r, 0)]
    if kind == "wrap_shadow_chain":
        r = int(RNG.integers(1, h - 1))
        return [(r, 0), (r + 1, w - 1)]
    raise ValueError(f"unknown bridge kind: {kind}")


def _place_bridge(
    kind: str,
    shape: tuple[int, int],
    used_blue: set[tuple[int, int]],
    used_enhanced: set[tuple[int, int]],
) -> bool:
    for _ in range(160):
        coords = _bridge_coords(kind, shape)
        if _can_place(coords, shape, used_blue, used_enhanced):
            _add_coords(coords, shape, used_blue, used_enhanced)
            return True
    return False


def _semantic_ok(grid: np.ndarray) -> bool:
    fg = grid == BLUE
    if fg.sum() < 8 or fg.sum() > 15:
        return False
    if np.any(fg[0]):
        return False

    keep = _survivor_mask(grid)
    out = main(grid)
    if not np.array_equal(out, keep.astype(int)):
        return False
    if np.array_equal(out, grid):
        return False
    if not np.any(out == BLUE):
        return False

    original_components = _components(fg)
    kept_singletons = 0
    removed_singletons = 0
    kept_cluster_pixels = 0
    naive = np.zeros_like(fg, dtype=bool)
    for comp in original_components:
        size = len(comp)
        if size > 1:
            naive[comp[:, 0], comp[:, 1]] = True
            kept_cluster_pixels += int(keep[comp[:, 0], comp[:, 1]].sum())
        else:
            r, c = map(int, comp[0])
            if keep[r, c]:
                kept_singletons += 1
            else:
                removed_singletons += 1

    if kept_singletons < 1 or removed_singletons < 1:
        return False
    if kept_cluster_pixels < 2:
        return False
    if np.array_equal(naive, keep):
        return False
    return True


def _attempt_generate() -> np.ndarray | None:
    h = int(RNG.integers(9, 14))
    w = int(RNG.integers(9, 14))
    shape = (h, w)
    target_pixels = int(RNG.integers(8, 16))

    used_blue: set[tuple[int, int]] = set()
    used_enhanced: set[tuple[int, int]] = set()

    bridge_kind = str(
        RNG.choice(
            ["diag_adjacent", "ortho_adjacent", "wrap_overlap", "wrap_shadow_chain"],
            p=[0.25, 0.25, 0.25, 0.25],
        )
    )

    mandatory_cluster_size = int(RNG.integers(2, 5))
    mandatory_total = 2 + 1 + mandatory_cluster_size
    if mandatory_total > target_pixels:
        return None

    if not _place_bridge(bridge_kind, shape, used_blue, used_enhanced):
        return None
    if not _place_singleton(shape, used_blue, used_enhanced):
        return None
    if not _place_cluster(mandatory_cluster_size, shape, used_blue, used_enhanced):
        return None

    while len(used_blue) < target_pixels:
        remaining = target_pixels - len(used_blue)
        if remaining == 1:
            if not _place_singleton(shape, used_blue, used_enhanced):
                return None
            continue

        choices: list[tuple[str, int]] = [("singleton", 1), ("bridge", 2), ("cluster", min(4, remaining))]
        kind, size = choices[int(RNG.integers(len(choices)))]
        if kind == "singleton":
            if not _place_singleton(shape, used_blue, used_enhanced):
                return None
        elif kind == "bridge":
            extra_kind = str(
                RNG.choice(["diag_adjacent", "ortho_adjacent", "wrap_overlap", "wrap_shadow_chain"])
            )
            if remaining < 2 or not _place_bridge(extra_kind, shape, used_blue, used_enhanced):
                return None
        else:
            cluster_size = int(RNG.integers(2, size + 1))
            if not _place_cluster(cluster_size, shape, used_blue, used_enhanced):
                return None

    grid = np.zeros(shape, dtype=int)
    for r, c in used_blue:
        grid[r, c] = BLUE

    if not _semantic_ok(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a non-degenerate shadow-bridge example")
