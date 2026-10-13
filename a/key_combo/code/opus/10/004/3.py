from __future__ import annotations

import itertools
import random
from collections import Counter

import numpy as np


BG = 0
RED = 2
GREY = 5
CONTROL_PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIRS8 = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0))
MAX_TRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    mask = np.asarray(grid == color, dtype=bool)
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        comp: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            comp.append((cr, cc))
            for nr, nc in _neighbors4(cr, cc):
                if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(sorted(comp))
    return comps


def _normalize(coords: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0 = min(r for r, _ in coords)
    c0 = min(c for _, c in coords)
    return tuple(sorted((r - r0, c - c0) for r, c in coords))


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _mask_from_coords(coords: list[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    box = _bbox(coords)
    r0, c0, r1, c1 = box
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask, box


def _scale_mask(mask: np.ndarray, factor: int) -> np.ndarray:
    return np.kron(mask.astype(int), np.ones((factor, factor), dtype=int)).astype(bool)


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> bool:
    h, w = mask.shape
    if top < 0 or left < 0 or top + h > grid.shape[0] or left + w > grid.shape[1]:
        return False
    grid[top : top + h, left : left + w][mask] = int(color)
    return True


def _transform_shape(coords: tuple[tuple[int, int], ...], rot: int, flip: bool) -> tuple[tuple[int, int], ...]:
    pts = list(coords)
    for _ in range(rot % 4):
        pts = [(-c, r) for r, c in pts]
    if flip:
        pts = [(r, -c) for r, c in pts]
    return _normalize(pts)


def _enumerate_shapes(area_lo: int, area_hi: int, *, require_min_bbox: bool) -> list[tuple[tuple[int, int], ...]]:
    out: set[tuple[tuple[int, int], ...]] = set()
    cells = [(r, c) for r in range(3) for c in range(3)]
    for area in range(area_lo, area_hi + 1):
        for picked in itertools.combinations(cells, area):
            coords = list(picked)
            pts = set(coords)
            stack = [coords[0]]
            seen = {coords[0]}
            while stack:
                r, c = stack.pop()
                for nr, nc in _neighbors4(r, c):
                    if (nr, nc) in pts and (nr, nc) not in seen:
                        seen.add((nr, nc))
                        stack.append((nr, nc))
            if len(seen) != area:
                continue
            shape = _normalize(coords)
            rs = [r for r, _ in shape]
            cs = [c for _, c in shape]
            h = max(rs) + 1
            w = max(cs) + 1
            if require_min_bbox and (h < 2 or w < 2):
                continue
            out.add(shape)
    return sorted(out)


TEMPLATE_SHAPES = _enumerate_shapes(3, 6, require_min_bbox=True)
SCATTER_SHAPES = _enumerate_shapes(2, 4, require_min_bbox=False)


def _extract_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    assert bg == BG

    grey_components = _components_of_color(g, GREY)
    assert 2 <= len(grey_components) <= 4
    for comp in grey_components:
        assert 2 <= len(comp) <= 4

    red_components = _components_of_color(g, RED)
    template_candidates = [comp for comp in red_components if 3 <= len(comp) <= 6]
    assert len(template_candidates) == 1
    template_coords = template_candidates[0]
    template_mask, template_bbox = _mask_from_coords(template_coords)
    template_r0, template_c0, template_r1, template_c1 = template_bbox

    controls: list[tuple[int, int, int]] = []
    for color in np.unique(g):
        color = int(color)
        if color in (bg, GREY):
            continue
        for comp in _components_of_color(g, color):
            if color == RED and comp == template_coords:
                continue
            assert len(comp) == 1
            r, c = comp[0]
            controls.append((r, c, color))

    assert len(controls) >= 2
    control_rows = {r for r, _, _ in controls}
    assert len(control_rows) == 1
    control_row = next(iter(control_rows))
    assert control_row > template_r1
    controls.sort(key=lambda item: item[1])
    control_colors = [color for _, _, color in controls]
    control_cols = [c for _, c, _ in controls]
    assert len(set(control_colors)) == len(control_colors)
    assert control_colors.count(RED) == 1
    for c0, c1 in zip(control_cols, control_cols[1:]):
        gap = c1 - c0 - 1
        assert 1 <= gap <= 3

    for r, c, _ in controls:
        assert all(
            not _in_bounds(g.shape, r + dr, c + dc) or (dr == 0 and dc == 0) or g[r + dr, c + dc] == bg
            for dr, dc in DIRS8
        )

    n = len(grey_components)
    scaled = _scale_mask(template_mask, n)
    scaled_h, scaled_w = scaled.shape
    anchor_index = control_colors.index(RED)
    anchor_control_col = control_cols[anchor_index]
    copy_top = control_row - scaled_h + 1
    assert copy_top == template_r0
    copy_starts = [template_c0 + (col - anchor_control_col) * n for col in control_cols]
    assert all(0 <= start <= g.shape[1] - scaled_w for start in copy_starts)
    for a, b in zip(copy_starts, copy_starts[1:]):
        assert b - a >= scaled_w

    total_grey_area = sum(len(comp) for comp in grey_components)
    assert total_grey_area != n

    return {
        "grid": g,
        "bg": bg,
        "n": n,
        "template_mask": template_mask,
        "template_bbox": template_bbox,
        "control_row": control_row,
        "controls": controls,
        "copy_top": copy_top,
        "copy_starts": copy_starts,
        "total_grey_area": total_grey_area,
    }


def _render_scene(
    scene: dict,
    *,
    scale_factor: int | None = None,
    position_factor: int | None = None,
    recolor: bool = True,
) -> np.ndarray | None:
    g = scene["grid"]
    factor = scene["n"] if scale_factor is None else int(scale_factor)
    if factor <= 0:
        return None

    scaled = _scale_mask(scene["template_mask"], factor)
    scaled_h, scaled_w = scaled.shape
    top = int(scene["control_row"]) - scaled_h + 1
    if top < 0:
        return None

    controls = scene["controls"]
    anchor_index = next(i for i, (_, _, color) in enumerate(controls) if color == RED)
    anchor_control_col = controls[anchor_index][1]
    pos_factor = factor if position_factor is None else int(position_factor)
    starts = [scene["template_bbox"][1] + (col - anchor_control_col) * pos_factor for _, col, _ in controls]
    for a, b in zip(starts, starts[1:]):
        if b - a < scaled_w:
            return None
    if any(start < 0 or start + scaled_w > g.shape[1] for start in starts):
        return None

    out = np.full(g.shape, scene["bg"], dtype=int)
    for start, (_, _, color) in zip(starts, controls):
        stamp_color = int(color if recolor else RED)
        if not _paint_mask(out, scaled, top, int(start), stamp_color):
            return None
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count grey scatter objects to choose the scaling factor, then stamp scaled template copies along the control row.

    Input structure:
    - The background is black.
    - One red 4-connected template sprite of 3-6 cells sits in the upper-left area.
    - One bottom control row contains isolated singleton pixels of distinct colors, with exactly one red control.
    - Two to four grey 4-connected scatter objects lie away from the template and control row.

    Rule:
    - Count the grey scatter objects to obtain N.
    - Scale the template by replacing each template cell with an N x N block.
    - Read the control pixels from left to right; the red control marks the anchor copy, and every other control recolors one same-shape scaled copy.
    - Keep the anchor copy aligned so its bottom edge meets the control row, and place every other copy so left-edge differences equal the corresponding control-column differences multiplied by N.
    - Remove the original template, scatter objects, and control pixels; output only the scaled copies on black background.

    Key insight:
    - The grey blobs are pure counting tokens: their object count, not their area or position, determines the magnification used before the control row expands into recolored template copies.
    """
    scene = _extract_scene(np.asarray(grid, dtype=int))
    out = _render_scene(scene)
    assert out is not None
    return out


def _control_layout_candidates(n: int, bbox_w: int) -> list[dict]:
    dist_choices = [dist for dist in range(2, 5) if dist >= bbox_w]
    scaled_w = bbox_w * n
    cands: list[dict] = []
    for count in range(2, 5):
        for anchor_index in range(count):
            for diffs in itertools.product(dist_choices, repeat=count - 1):
                positions = [0]
                for diff in diffs:
                    positions.append(positions[-1] + diff)
                deltas = [pos - positions[anchor_index] for pos in positions]
                span = (max(deltas) - min(deltas)) * n + scaled_w
                if span > 28:
                    continue
                cands.append(
                    {
                        "count": count,
                        "anchor_index": anchor_index,
                        "diffs": diffs,
                        "deltas": tuple(deltas),
                        "copy_span": span,
                        "control_span": positions[-1] - positions[0] + 1,
                    }
                )
    return cands


def _seed_from_params(*values: int) -> int:
    seed = 1469598103
    for value in values:
        seed = (seed * 16777619 + int(value) + 97) & 0xFFFFFFFF
    return seed


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _place_scatter_objects(
    grid: np.ndarray,
    *,
    count: int,
    template_cells: set[tuple[int, int]],
    control_pixels: set[tuple[int, int]],
    seed: int,
) -> bool:
    h, w = grid.shape
    blocked = set(template_cells) | set(control_pixels)
    halo = set()
    for r, c in blocked:
        for dr, dc in DIRS8 + ((0, 0),):
            nr, nc = r + dr, c + dc
            if _in_bounds(grid.shape, nr, nc):
                halo.add((nr, nc))

    rng = random.Random(seed)
    for index in range(count):
        base = SCATTER_SHAPES[(seed + 3 * index) % len(SCATTER_SHAPES)]
        shape = _transform_shape(base, rot=(seed + index) % 4, flip=((seed >> index) & 1) == 1)
        sh, sw = _shape_dims(shape)
        candidates: list[tuple[int, int]] = []
        row_hi = max(1, h - sh - 2)
        col_lo = max(w // 2, 1)
        for r0 in range(1, row_hi + 1):
            for c0 in range(col_lo, w - sw):
                cells = {(r0 + dr, c0 + dc) for dr, dc in shape}
                if any((r, c) in halo for r, c in cells):
                    continue
                expanded = set()
                ok = True
                for r, c in cells:
                    if not _in_bounds(grid.shape, r, c) or grid[r, c] != BG:
                        ok = False
                        break
                    for dr, dc in DIRS8 + ((0, 0),):
                        nr, nc = r + dr, c + dc
                        if _in_bounds(grid.shape, nr, nc):
                            expanded.add((nr, nc))
                if not ok:
                    continue
                if expanded & halo:
                    continue
                candidates.append((r0, c0))
        if not candidates:
            return False
        r0, c0 = candidates[rng.randrange(len(candidates))]
        cells = {(r0 + dr, c0 + dc) for dr, dc in shape}
        for r, c in cells:
            grid[r, c] = GREY
        for r, c in cells:
            for dr, dc in DIRS8 + ((0, 0),):
                nr, nc = r + dr, c + dc
                if _in_bounds(grid.shape, nr, nc):
                    halo.add((nr, nc))
    return True


def _semantic_checks(grid: np.ndarray) -> bool:
    scene = _extract_scene(grid)
    expected = _render_scene(scene)
    if expected is None or not np.array_equal(main(grid), expected):
        return False

    wrong_area = _render_scene(scene, scale_factor=scene["total_grey_area"])
    if wrong_area is not None and np.array_equal(wrong_area, expected):
        return False

    wrong_gap = _render_scene(scene, position_factor=1)
    if wrong_gap is not None and np.array_equal(wrong_gap, expected):
        return False

    wrong_color = _render_scene(scene, recolor=False)
    if wrong_color is not None and np.array_equal(wrong_color, expected):
        return False

    alt_scale = scene["n"] - 1 if scene["n"] > 2 else scene["n"] + 1
    wrong_count = _render_scene(scene, scale_factor=alt_scale)
    if wrong_count is not None and np.array_equal(wrong_count, expected):
        return False

    return True


def _build_attempt(rng: random.Random) -> np.ndarray | None:
    template_shape = rng.choice(TEMPLATE_SHAPES)
    template_h, template_w = _shape_dims(template_shape)
    n = rng.randint(2, 4)

    layout_candidates = _control_layout_candidates(n, template_w)
    if not layout_candidates:
        return None
    layout = rng.choice(layout_candidates)
    deltas = layout["deltas"]
    scaled_h = template_h * n
    scaled_w = template_w * n

    width_min = max(22, layout["copy_span"] + 3, layout["control_span"] + 3)
    if width_min > 30:
        return None
    width = rng.randint(width_min, 30)

    c0_lo = max(1, -min(deltas) * n + 1)
    c0_hi = min(width // 3, width - (max(deltas) * n + scaled_w) - 2)
    if c0_lo > c0_hi:
        return None
    template_c0 = rng.randint(c0_lo, c0_hi)

    row_max = 25 - scaled_h - 2
    if row_max < 1:
        return None
    template_r0 = rng.randint(1, min(8, row_max))
    bottom_margin = rng.randint(2, min(5, 25 - (template_r0 + scaled_h)))
    height = template_r0 + scaled_h + bottom_margin
    if height > 25:
        return None

    control_row = template_r0 + scaled_h - 1
    anchor_control_col = max(1, 1 - min(deltas))
    if anchor_control_col + max(deltas) > width - 2:
        return None
    control_cols = [anchor_control_col + delta for delta in deltas]

    count = layout["count"]
    anchor_index = layout["anchor_index"]
    control_colors = random.sample([color for color in CONTROL_PALETTE if color != RED], count - 1)
    control_colors.insert(anchor_index, RED)

    grid = np.full((height, width), BG, dtype=int)
    template_cells = {(template_r0 + r, template_c0 + c) for r, c in template_shape}
    for r, c in template_cells:
        grid[r, c] = RED

    control_pixels = {(control_row, col) for col in control_cols}
    for col, color in zip(control_cols, control_colors):
        grid[control_row, col] = color

    seed = _seed_from_params(
        n,
        template_r0,
        template_c0,
        height,
        width,
        anchor_index,
        sum(control_cols),
        sum(control_colors),
        len(template_shape),
    )
    if not _place_scatter_objects(
        grid,
        count=n,
        template_cells=template_cells,
        control_pixels=control_pixels,
        seed=seed,
    ):
        return None

    if max(grid.shape) > 30:
        return None
    if not _semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_TRIES):
        grid = _build_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
