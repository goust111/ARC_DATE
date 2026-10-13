from __future__ import annotations

import numpy as np


BG = 0
PALETTE = list(range(1, 10))
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

MIN_GRID_H = 18
MAX_GRID_H = 30
MIN_GRID_W = 18
MAX_GRID_W = 30
RING_COUNT_RANGE = (2, 5)
INNER_H_RANGE = (3, 6)
INNER_W_CHOICES = (4, 6, 8)
MAX_GEN_ATTEMPTS = 300


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _border_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _classify_ring(grid: np.ndarray, coords: list[tuple[int, int]]) -> dict[str, object] | None:
    r0, c0, r1, c1 = _bbox(coords)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    if h < 3 or w < 3:
        return None

    mask = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    border = _border_mask(h, w)
    if not np.array_equal(mask, border):
        return None

    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    values = patch[mask]
    colors, counts = np.unique(values, return_counts=True)
    if len(colors) != 2:
        return None
    major_idx = int(np.argmax(counts))
    minor_idx = 1 - major_idx
    if int(counts[major_idx]) != values.size - 1 or int(counts[minor_idx]) != 1:
        return None

    base_color = int(colors[major_idx])
    marker_color = int(colors[minor_idx])
    marker_rc = np.argwhere(mask & (patch == marker_color))
    if marker_rc.shape != (1, 2):
        return None
    mr, mc = marker_rc[0]
    return {
        "kind": "ring",
        "bbox": (r0, c0, r1, c1),
        "area": h * w,
        "base_color": base_color,
        "marker_color": marker_color,
        "marker_pos": (r0 + int(mr), c0 + int(mc)),
    }


def _classify_core(grid: np.ndarray, coords: list[tuple[int, int]]) -> dict[str, object] | None:
    r0, c0, r1, c1 = _bbox(coords)
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    if np.any(patch == BG):
        return None

    mask = np.zeros_like(patch, dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    if not bool(mask.all()):
        return None

    colors = [int(color) for color in np.unique(patch)]
    if len(colors) != 2:
        return None

    mean_cols = {}
    for color in colors:
        cols = np.argwhere(patch == color)[:, 1]
        mean_cols[color] = float(cols.mean())
    left_color, right_color = sorted(colors, key=lambda color: mean_cols[color])
    if not mean_cols[left_color] < mean_cols[right_color]:
        return None

    return {
        "kind": "core",
        "bbox": (r0, c0, r1, c1),
        "patch": patch.copy(),
        "left_color": left_color,
        "right_color": right_color,
    }


def _parse_scene(grid: np.ndarray) -> tuple[list[dict[str, object]], dict[str, object]]:
    comps = _connected_components(grid != BG)
    rings: list[dict[str, object]] = []
    cores: list[dict[str, object]] = []
    for coords in comps:
        ring = _classify_ring(grid, coords)
        if ring is not None:
            rings.append(ring)
            continue
        core = _classify_core(grid, coords)
        if core is not None:
            cores.append(core)
            continue
        raise AssertionError("unexpected foreground component")

    assert rings, "expected at least one ring"
    assert len(cores) == 1, "expected exactly one inner solid pattern"

    rings.sort(key=lambda item: int(item["area"]), reverse=True)
    for outer, inner in zip(rings, rings[1:]):
        or0, oc0, or1, oc1 = outer["bbox"]
        ir0, ic0, ir1, ic1 = inner["bbox"]
        assert or0 < ir0 and oc0 < ic0 and ir1 < or1 and ic1 < oc1, "rings must be nested"

    core = cores[0]
    cr0, cc0, cr1, cc1 = core["bbox"]
    ir0, ic0, ir1, ic1 = rings[-1]["bbox"]
    assert ir0 < cr0 and ic0 < cc0 and cr1 < ir1 and cc1 < ic1, "core must be inside innermost ring"
    return rings, core


def _draw_ring(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _mirror_col(c0: int, c1: int, c: int) -> int:
    return c0 + c1 - c


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify nested rectangular rings with one marker pixel each, horizontally mirror
    the markers on even layers, and recolor odd/even markers from the inner pattern's
    right/left colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    rings, core = _parse_scene(g)
    left_color = int(core["left_color"])
    right_color = int(core["right_color"])

    out = np.full_like(g, BG)
    for layer_idx, ring in enumerate(rings, start=1):
        bbox = ring["bbox"]
        _draw_ring(out, bbox, int(ring["base_color"]))
        mr, mc = ring["marker_pos"]
        r0, c0, _r1, c1 = bbox
        if layer_idx % 2 == 0:
            mc = _mirror_col(c0, c1, int(mc))
            marker_color = left_color
        else:
            marker_color = right_color
        out[int(mr), int(mc)] = marker_color

    cr0, cc0, cr1, cc1 = core["bbox"]
    out[cr0 : cr1 + 1, cc0 : cc1 + 1] = np.asarray(core["patch"], dtype=int)
    return out


def _sample_side_position(
    rng: np.random.Generator, bbox: tuple[int, int, int, int]
) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    side = int(rng.integers(0, 4))
    if side == 0:
        return r0, int(rng.integers(c0, c1 + 1))
    if side == 1:
        return int(rng.integers(r0, r1 + 1)), c1
    if side == 2:
        return r1, int(rng.integers(c0, c1 + 1))
    return int(rng.integers(r0, r1 + 1)), c0


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    ring_count = int(rng.integers(RING_COUNT_RANGE[0], RING_COUNT_RANGE[1] + 1))
    inner_h = int(rng.integers(INNER_H_RANGE[0], INNER_H_RANGE[1] + 1))
    inner_w = int(rng.choice(INNER_W_CHOICES))

    outer_h = inner_h + 4 * ring_count
    outer_w = inner_w + 4 * ring_count

    min_extra_h = max(2, MIN_GRID_H - outer_h)
    max_extra_h = MAX_GRID_H - outer_h
    min_extra_w = max(2, MIN_GRID_W - outer_w)
    max_extra_w = MAX_GRID_W - outer_w
    if min_extra_h > max_extra_h or min_extra_w > max_extra_w:
        return None

    extra_h = int(rng.integers(min_extra_h, max_extra_h + 1))
    extra_w = int(rng.integers(min_extra_w, max_extra_w + 1))
    top = int(rng.integers(1, extra_h))
    left = int(rng.integers(1, extra_w))
    bottom = extra_h - top
    right = extra_w - left
    h = outer_h + top + bottom
    w = outer_w + left + right

    grid = np.full((h, w), BG, dtype=int)

    color_perm = [int(color) for color in rng.permutation(PALETTE)]
    base_colors = color_perm[:ring_count]
    left_color, right_color = color_perm[ring_count : ring_count + 2]
    marker_pool = [color for color in PALETTE if color not in set(base_colors) | {left_color, right_color}]
    if not marker_pool:
        return None

    for layer in range(ring_count):
        r0 = top + 2 * layer
        c0 = left + 2 * layer
        r1 = top + outer_h - 1 - 2 * layer
        c1 = left + outer_w - 1 - 2 * layer
        bbox = (r0, c0, r1, c1)
        _draw_ring(grid, bbox, int(base_colors[layer]))
        mr, mc = _sample_side_position(rng, bbox)
        marker_choices = [color for color in marker_pool if color != base_colors[layer]]
        if not marker_choices:
            return None
        grid[mr, mc] = int(rng.choice(marker_choices))

    core_r0 = top + 2 * ring_count
    core_c0 = left + 2 * ring_count
    core_r1 = core_r0 + inner_h
    core_c1 = core_c0 + inner_w // 2
    grid[core_r0 : core_r0 + inner_h, core_c0 : core_c0 + inner_w] = right_color
    grid[core_r0 : core_r0 + inner_h, core_c0:core_c1] = left_color

    try:
        rings, core = _parse_scene(grid)
        assert len(rings) == ring_count
        assert int(core["left_color"]) == left_color
        assert int(core["right_color"]) == right_color
        out = main(grid)
        assert out.shape == grid.shape
        assert not np.array_equal(out, grid), "expected a non-trivial transformation"
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GEN_ATTEMPTS} attempts")
