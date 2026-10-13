from __future__ import annotations

from collections import deque
import random

import numpy as np


_RNG = np.random.default_rng()
_MAX_GENERATION_ATTEMPTS = 400
_NEIGHBORS8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]
_NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = map(int, mask.shape)
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in _NEIGHBORS8:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _local_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    shifted = coords - np.array([r0, c0])
    mask[shifted[:, 0], shifted[:, 1]] = True
    return mask


def _reflect_coords(coords: np.ndarray, row_sum: int, col_sum: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    refl_v = coords.copy()
    refl_v[:, 1] = col_sum - coords[:, 1]
    refl_h = coords.copy()
    refl_h[:, 0] = row_sum - coords[:, 0]
    refl_both = coords.copy()
    refl_both[:, 0] = row_sum - coords[:, 0]
    refl_both[:, 1] = col_sum - coords[:, 1]
    return refl_v, refl_h, refl_both


def _is_axis_symmetric(mask: np.ndarray) -> bool:
    return bool(np.array_equal(mask, np.flipud(mask)) and np.array_equal(mask, np.fliplr(mask)))


def _is_self_asymmetric(sprite: np.ndarray, bg: int) -> bool:
    if np.array_equal(sprite, np.flipud(sprite)):
        return False
    if np.array_equal(sprite, np.fliplr(sprite)):
        return False
    if np.array_equal(sprite, np.rot90(sprite, 2)):
        return False
    colors = [int(c) for c in np.unique(sprite) if int(c) != bg]
    return len(colors) == 2


def _marker_mask(h: int, w: int, kind: str) -> np.ndarray:
    rows = np.arange(h)[:, None]
    cols = np.arange(w)[None, :]
    row_sum = h - 1
    col_sum = w - 1

    if kind == "ring":
        mask = np.zeros((h, w), dtype=bool)
        mask[0, :] = True
        mask[-1, :] = True
        mask[:, 0] = True
        mask[:, -1] = True
        return mask

    if kind == "cross":
        row_dist = np.abs(2 * rows - row_sum)
        col_dist = np.abs(2 * cols - col_sum)
        return (row_dist <= 1) | (col_dist <= 1)

    if kind == "diamond":
        row_dist = np.abs(2 * rows - row_sum)
        col_dist = np.abs(2 * cols - col_sum)
        radius = max(2, min(row_sum, col_sum))
        mask = (row_dist + col_dist) <= radius
        if not np.any(mask[0, :]) or not np.any(mask[-1, :]) or not np.any(mask[:, 0]) or not np.any(mask[:, -1]):
            mask[0, w // 2] = True
            mask[-1, w // 2] = True
            mask[h // 2, 0] = True
            mask[h // 2, -1] = True
        return mask

    raise ValueError(f"unknown marker kind: {kind}")


def _grow_connected_mask(h: int, w: int, area: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    start = (int(_RNG.integers(h)), int(_RNG.integers(w)))
    mask[start] = True
    while int(mask.sum()) < area:
        frontier: list[tuple[int, int]] = []
        for r, c in zip(*np.nonzero(mask)):
            for dr, dc in _NEIGHBORS4:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.append((nr, nc))
        if not frontier:
            break
        nr, nc = random.choice(frontier)
        mask[nr, nc] = True
    return mask


def _color_source(mask: np.ndarray, bg: int, colors: tuple[int, int]) -> np.ndarray:
    h, w = mask.shape
    coords = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    random.shuffle(coords)
    sprite = np.full((h, w), bg, dtype=int)
    sprite[mask] = colors[1]

    if len(coords) <= 2:
        first_group = coords[:1]
    elif random.random() < 0.5:
        ordered = sorted(coords, key=lambda rc: (rc[random.randrange(2)], rc[1 - random.randrange(2)]))
        split = random.randint(1, len(ordered) - 1)
        first_group = ordered[:split]
    else:
        size = random.randint(1, len(coords) - 1)
        first_group = random.sample(coords, size)

    for r, c in first_group:
        sprite[r, c] = colors[0]
    return sprite


def _generate_source_sprite(h: int, w: int, bg: int, colors: tuple[int, int]) -> np.ndarray | None:
    max_area = h * w - 1
    min_area = max(5, (h * w) // 2)
    if max_area < min_area:
        return None

    for _ in range(120):
        area = int(_RNG.integers(min_area, max_area + 1))
        mask = _grow_connected_mask(h, w, area)
        if int(mask.sum()) != area:
            continue
        if np.all(mask):
            continue
        sprite = _color_source(mask, bg, colors)
        if not _is_self_asymmetric(sprite, bg):
            continue
        return sprite
    return None


def _swap_map(colors: list[int]) -> dict[int, int]:
    assert len(colors) == 2
    return {colors[0]: colors[1], colors[1]: colors[0]}


def _extract_scene(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    fg_mask = g != bg
    components = _find_components(fg_mask)
    assert len(components) >= 2, "expected a marker and a source pattern"

    marker_candidates: list[dict[str, object]] = []
    for coords in components:
        colors = sorted(int(c) for c in np.unique(g[coords[:, 0], coords[:, 1]]))
        local_mask = _local_mask(coords)
        if len(colors) == 1 and _is_axis_symmetric(local_mask):
            r0, c0, r1, c1 = _bbox(coords)
            marker_candidates.append(
                {
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "row_sum": r0 + r1,
                    "col_sum": c0 + c1,
                    "color": colors[0],
                }
            )

    assert len(marker_candidates) == 1, "expected exactly one symmetric monochrome marker"
    marker = marker_candidates[0]

    marker_mask = np.zeros_like(g, dtype=bool)
    mc = marker["coords"]
    marker_mask[mc[:, 0], mc[:, 1]] = True
    source_mask = fg_mask & ~marker_mask
    source_coords = np.argwhere(source_mask)
    assert len(source_coords) > 0, "expected source pixels outside the marker"
    source_colors = sorted(int(c) for c in np.unique(g[source_mask]))
    assert len(source_colors) == 2, "source pattern must use exactly two colors"

    row_sum = int(marker["row_sum"])
    col_sum = int(marker["col_sum"])
    row_signs = set()
    col_signs = set()
    for r, c in source_coords:
        row_delta = 2 * int(r) - row_sum
        col_delta = 2 * int(c) - col_sum
        assert row_delta != 0 and col_delta != 0, "source pixels cannot lie on a borrowed axis"
        row_signs.add(1 if row_delta > 0 else -1)
        col_signs.add(1 if col_delta > 0 else -1)

    assert len(row_signs) == 1 and len(col_signs) == 1, "source must lie in exactly one quadrant"
    return {
        "grid": g,
        "bg": bg,
        "marker": marker,
        "marker_mask": marker_mask,
        "source_mask": source_mask,
        "source_coords": source_coords.astype(int),
        "source_colors": source_colors,
        "row_sign": next(iter(row_signs)),
        "col_sign": next(iter(col_signs)),
    }


def _apply_rule(scene: dict[str, object]) -> np.ndarray:
    g = np.array(scene["grid"], copy=True)
    source_coords = np.asarray(scene["source_coords"], dtype=int)
    row_sum = int(scene["marker"]["row_sum"])
    col_sum = int(scene["marker"]["col_sum"])
    colors = list(scene["source_colors"])
    swapped = _swap_map(colors)
    src_values = g[source_coords[:, 0], source_coords[:, 1]].copy()
    refl_v, refl_h, refl_both = _reflect_coords(source_coords, row_sum, col_sum)

    for arr in (refl_v, refl_h, refl_both):
        assert arr[:, 0].min() >= 0 and arr[:, 1].min() >= 0
        assert arr[:, 0].max() < g.shape[0] and arr[:, 1].max() < g.shape[1]

    g[refl_v[:, 0], refl_v[:, 1]] = src_values
    g[refl_h[:, 0], refl_h[:, 1]] = src_values
    g[refl_both[:, 0], refl_both[:, 1]] = np.array([swapped[int(v)] for v in src_values], dtype=int)
    return g


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reflect a two-color source through the symmetric marker's borrowed axes, swapping colors only after the double reflection.

    Input structure:
    - A dominant background contains one monochrome marker object with both horizontal and vertical mirror symmetry.
    - All remaining foreground pixels form a two-color source pattern located entirely inside one quadrant defined by the marker's unseen axes.
    - The three reflected target regions are background in the input.

    Rule:
    - Detect the marker and recover its horizontal and vertical symmetry axes from the marker's bounding box center.
    - Reflect every source pixel once across the vertical axis and once across the horizontal axis, keeping its color in both single-reflection copies.
    - Reflect every source pixel across both axes to the diagonal quadrant, but swap the source pattern's two colors there; keep the original source and the marker unchanged.

    Key insight:
    - The axes are borrowed from a different object, and the diagonal copy is distinguished only by the parity of reflections: two reflections preserve position parity but force a two-color swap.
    """
    scene = _extract_scene(np.asarray(grid, dtype=int))
    return _apply_rule(scene)


def _quadrant_ranges(
    shape: tuple[int, int],
    marker_bbox: tuple[int, int, int, int],
    quadrant: tuple[int, int],
    gap: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    h, w = map(int, shape)
    r0, c0, r1, c1 = marker_bbox
    row_sum = r0 + r1
    col_sum = c0 + c1

    if quadrant[0] < 0:
        row_lo = max(0, row_sum - (h - 1))
        row_hi = r0 - gap - 1
    else:
        row_lo = r1 + gap + 1
        row_hi = min(h - 1, row_sum)

    if quadrant[1] < 0:
        col_lo = max(0, col_sum - (w - 1))
        col_hi = c0 - gap - 1
    else:
        col_lo = c1 + gap + 1
        col_hi = min(w - 1, col_sum)

    return (int(row_lo), int(row_hi)), (int(col_lo), int(col_hi))


def _place_marker(grid: np.ndarray, color: int) -> tuple[np.ndarray, tuple[int, int, int, int]] | tuple[None, None]:
    h, w = map(int, grid.shape)
    marker_h = int(_RNG.integers(3, min(8, h - 3)))
    marker_w = int(_RNG.integers(3, min(8, w - 3)))
    kind = random.choice(["ring", "cross", "diamond"])
    mask = _marker_mask(marker_h, marker_w, kind)
    if int(mask.sum()) < 5:
        return None, None

    for _ in range(80):
        r0 = int(_RNG.integers(1, h - marker_h))
        c0 = int(_RNG.integers(1, w - marker_w))
        r1 = r0 + marker_h - 1
        c1 = c0 + marker_w - 1
        row_sum = r0 + r1
        col_sum = c0 + c1
        feasible = False
        for quadrant in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            (row_lo, row_hi), (col_lo, col_hi) = _quadrant_ranges((h, w), (r0, c0, r1, c1), quadrant, gap=1)
            if row_hi - row_lo + 1 >= 3 and col_hi - col_lo + 1 >= 3:
                feasible = True
                break
        if not feasible:
            continue
        out = grid.copy()
        out[r0 : r1 + 1, c0 : c1 + 1][mask] = color
        return out, (r0, c0, r1, c1)
    return None, None


def _stamp_sprite(grid: np.ndarray, sprite: np.ndarray, top: int, left: int, bg: int) -> None:
    mask = sprite != bg
    sub = grid[top : top + sprite.shape[0], left : left + sprite.shape[1]]
    sub[mask] = sprite[mask]


def _validate_generated_input(grid: np.ndarray) -> bool:
    try:
        scene = _extract_scene(grid)
    except Exception:
        return False

    marker_mask = np.asarray(scene["marker_mask"], dtype=bool)
    source_mask = np.asarray(scene["source_mask"], dtype=bool)
    source_coords = np.asarray(scene["source_coords"], dtype=int)
    bg = int(scene["bg"])
    base = np.array(grid, copy=True)
    row_sum = int(scene["marker"]["row_sum"])
    col_sum = int(scene["marker"]["col_sum"])

    refl_v, refl_h, refl_both = _reflect_coords(source_coords, row_sum, col_sum)
    for arr in (refl_v, refl_h, refl_both):
        if not (arr[:, 0].min() >= 0 and arr[:, 1].min() >= 0 and arr[:, 0].max() < grid.shape[0] and arr[:, 1].max() < grid.shape[1]):
            return False
        if np.any(marker_mask[arr[:, 0], arr[:, 1]]):
            return False
        if np.any(source_mask[arr[:, 0], arr[:, 1]]):
            return False
        if np.any(base[arr[:, 0], arr[:, 1]] != bg):
            return False

    source_crop_bbox = _bbox(source_coords)
    sr0, sc0, sr1, sc1 = source_crop_bbox
    source_crop = base[sr0 : sr1 + 1, sc0 : sc1 + 1]
    if not _is_self_asymmetric(source_crop, bg):
        return False

    actual = main(grid)
    expected = _apply_rule(scene)
    if not np.array_equal(actual, expected):
        return False

    colors = list(scene["source_colors"])
    swapped = _swap_map(colors)
    src_values = base[source_coords[:, 0], source_coords[:, 1]]
    diag_values = actual[refl_both[:, 0], refl_both[:, 1]]
    if not np.array_equal(diag_values, np.array([swapped[int(v)] for v in src_values], dtype=int)):
        return False
    if np.array_equal(diag_values, src_values):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = int(_RNG.integers(14, 27))
    w = int(_RNG.integers(14, 27))
    bg = int(_RNG.integers(0, 10))
    palette = [c for c in range(10) if c != bg]
    marker_color, src_a, src_b = random.sample(palette, 3)
    grid = np.full((h, w), bg, dtype=int)

    placed, marker_bbox = _place_marker(grid, marker_color)
    if placed is None or marker_bbox is None:
        return None
    grid = placed

    feasible_quadrants: list[tuple[int, int, tuple[int, int], tuple[int, int]]] = []
    for quadrant in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        row_range, col_range = _quadrant_ranges((h, w), marker_bbox, quadrant, gap=1)
        if row_range[1] - row_range[0] + 1 >= 3 and col_range[1] - col_range[0] + 1 >= 3:
            feasible_quadrants.append((quadrant[0], quadrant[1], row_range, col_range))
    if not feasible_quadrants:
        return None

    q_row, q_col, row_range, col_range = random.choice(feasible_quadrants)
    max_sh = min(6, row_range[1] - row_range[0] + 1)
    max_sw = min(6, col_range[1] - col_range[0] + 1)
    if max_sh < 3 or max_sw < 3:
        return None

    for _ in range(120):
        sh = int(_RNG.integers(3, max_sh + 1))
        sw = int(_RNG.integers(3, max_sw + 1))
        sprite = _generate_source_sprite(sh, sw, bg, (src_a, src_b))
        if sprite is None:
            continue

        row_lo, row_hi = row_range
        col_lo, col_hi = col_range
        start_r_hi = row_hi - sh + 1
        start_c_hi = col_hi - sw + 1
        if start_r_hi < row_lo or start_c_hi < col_lo:
            continue
        top = int(_RNG.integers(row_lo, start_r_hi + 1))
        left = int(_RNG.integers(col_lo, start_c_hi + 1))

        candidate = grid.copy()
        _stamp_sprite(candidate, sprite, top, left, bg)
        if not _validate_generated_input(candidate):
            continue

        scene = _extract_scene(candidate)
        if int(scene["row_sign"]) != q_row or int(scene["col_sign"]) != q_col:
            continue
        return candidate
    return None


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid reflection-parity input")
