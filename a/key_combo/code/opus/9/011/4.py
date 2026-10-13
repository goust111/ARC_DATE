from __future__ import annotations

import math
import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

GRID_H_RANGE = (18, 24)
GRID_W_RANGE = (18, 24)
MARKER_COUNT_RANGE = (4, 7)  # includes the green origin
STEP_RANGE = (5, 8)
MAX_GENERATION_ATTEMPTS = 400

TEMPLATE_LIBRARY = (
    np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[0, 0, 1], [0, 0, 1], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[0, 1, 1], [0, 1, 0], [1, 1, 0]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[0, 0, 1], [0, 1, 1], [1, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 0, 1], [0, 0, 1]], dtype=bool),
)


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
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
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_mask(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _infer_step(diffs: list[int]) -> int:
    nonzero = [abs(int(v)) for v in diffs if int(v) != 0]
    assert nonzero, "expected a nonzero lattice step on each axis"
    step = nonzero[0]
    for v in nonzero[1:]:
        step = math.gcd(step, v)
    assert step > 0
    return int(step)


def _extract_problem(
    grid: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int], list[tuple[int, int, bool]], tuple[int, int, int, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert _get_bg(g) == BG, "expected black background"

    blue_comps = [comp for comp in _components(g == BLUE) if int(comp.shape[0]) >= 4]
    assert len(blue_comps) == 1, "expected one non-singleton blue template"
    template_coords = blue_comps[0]
    template_box = _bbox_coords(template_coords)
    template = _crop_mask(g == BLUE, template_box)
    assert template.shape[0] % 2 == 1 and template.shape[1] % 2 == 1, "template bbox center must be integral"
    assert not np.array_equal(template, np.fliplr(template)), "template must be horizontally asymmetric"

    origin_pts = np.argwhere(g == GREEN)
    yellow_pts = np.argwhere(g == YELLOW)
    assert origin_pts.shape == (1, 2), "expected exactly one green origin marker"
    assert 3 <= yellow_pts.shape[0] <= 6, "expected 3..6 yellow replica markers"
    assert np.count_nonzero((g != BG) & (g != BLUE) & (g != GREEN) & (g != YELLOW)) == 0

    marker_pts = np.vstack([origin_pts, yellow_pts]).astype(int)
    orr, orc = map(int, origin_pts[0])
    dy = _infer_step([int(r) - orr for r, _c in marker_pts.tolist()])
    dx = _infer_step([int(c) - orc for _r, c in marker_pts.tolist()])

    marker_offsets: list[tuple[int, int, bool]] = []
    for r, c in marker_pts.tolist():
        dr = int(r) - orr
        dc = int(c) - orc
        assert dr % dy == 0 and dc % dx == 0, "all markers must lie on the same lattice"
        marker_offsets.append((dr // dy, dc // dx, (int(r) == orr and int(c) == orc)))

    row_offsets = {dr for dr, _dc, _is_origin in marker_offsets}
    col_offsets = {dc for _dr, dc, _is_origin in marker_offsets}
    assert len(row_offsets) >= 2 and len(col_offsets) >= 2, "lattice must reveal both row and column steps"
    assert any(abs(dc) % 2 == 1 for _dr, dc, _is_origin in marker_offsets), "need at least one odd column offset"

    pair_exists = any(
        dr1 == dr2 and abs(dc1 - dc2) == 1
        for i, (dr1, dc1, _o1) in enumerate(marker_offsets)
        for dr2, dc2, _o2 in marker_offsets[i + 1 :]
    )
    assert pair_exists, "need a horizontally adjacent lattice pair"

    r0, c0, r1, c1 = template_box
    base_center = ((r0 + r1) // 2, (c0 + c1) // 2)
    return template, base_center, marker_offsets, template_box


def _render(
    shape: tuple[int, int],
    template: np.ndarray,
    base_center: tuple[int, int],
    marker_offsets: list[tuple[int, int, bool]],
    *,
    use_mirror_parity: bool,
) -> np.ndarray:
    h, w = template.shape
    cr = h // 2
    cc = w // 2
    coverage = np.zeros(shape, dtype=int)
    colors = np.full(shape, BG, dtype=int)

    for dr, dc, is_origin in marker_offsets:
        sprite = np.fliplr(template) if use_mirror_parity and abs(dc) % 2 == 1 else template
        stamp_color = GREEN if is_origin else BLUE
        center_r = int(base_center[0]) + int(dr)
        center_c = int(base_center[1]) + int(dc)
        top = center_r - cr
        left = center_c - cc

        for rr, cc2 in np.argwhere(sprite):
            r = top + int(rr)
            c = left + int(cc2)
            assert 0 <= r < shape[0] and 0 <= c < shape[1], "stamp placement out of bounds"
            coverage[r, c] += 1
            if coverage[r, c] == 1:
                colors[r, c] = stamp_color

    out = np.full(shape, BG, dtype=int)
    out[coverage == 1] = colors[coverage == 1]
    out[coverage >= 2] = RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Translate a source template by lattice offsets, mirror odd columns, and recolor overlaps red.

    Input structure:
    - The background is black.
    - A single connected blue template sits near the upper-left and is not invariant under left-right mirroring.
    - Elsewhere, one green singleton and several yellow singletons lie on a uniform row/column lattice.

    Rule:
    - Treat the green singleton as lattice origin and infer the lattice row and column steps from all singleton coordinates.
    - Convert every singleton to an integer lattice offset relative to the origin, then translate the source template center by that offset.
    - Copies at odd column offsets are mirrored horizontally before stamping; the origin copy is recolored green and the others stay blue.
    - Any pixel covered by two or more stamped copies is recolored red, and the singleton markers themselves do not survive.

    Key insight:
    - The singletons do not mark final paint locations directly; they encode small lattice offsets, and column parity only becomes visible because mirrored odd-column copies overlap their neighbors differently.
    """
    template, base_center, marker_offsets, _template_box = _extract_problem(np.asarray(grid, dtype=int))
    return _render(grid.shape, template, base_center, marker_offsets, use_mirror_parity=True)


def _sample_template() -> np.ndarray:
    return random.choice(TEMPLATE_LIBRARY).copy()


def _sample_marker_layout() -> tuple[list[tuple[int, int]], tuple[int, int]]:
    for _ in range(200):
        n_rows = random.randint(2, 3)
        n_cols = random.randint(3, 4)
        all_cells = [(r, c) for r in range(n_rows) for c in range(n_cols)]
        target_count = random.randint(MARKER_COUNT_RANGE[0], min(MARKER_COUNT_RANGE[1], len(all_cells)))
        origin = random.choice(all_cells)
        chosen = set(random.sample(all_cells, target_count))
        chosen.add(origin)
        if len(chosen) > target_count:
            removable = list(chosen - {origin})
            random.shuffle(removable)
            while len(chosen) > target_count:
                chosen.remove(removable.pop())

        rows = {r for r, _c in chosen}
        cols = {c for _r, c in chosen}
        if len(rows) < 2 or len(cols) < 2:
            continue

        offsets = [(r - origin[0], c - origin[1]) for r, c in chosen]
        if not any(abs(dc) % 2 == 1 for _dr, dc in offsets):
            continue
        if not any(r1 == r2 and abs(c1 - c2) == 1 for r1, c1 in chosen for r2, c2 in chosen if (r1, c1) < (r2, c2)):
            continue
        return sorted(chosen), origin
    raise RuntimeError("failed to sample marker lattice")


def _separated(box1: tuple[int, int, int, int], box2: tuple[int, int, int, int], gap: int = 1) -> bool:
    r0a, c0a, r1a, c1a = box1
    r0b, c0b, r1b, c1b = box2
    return r1a + gap < r0b or r1b + gap < r0a or c1a + gap < c0b or c1b + gap < c0a


def _build_input(
    shape: tuple[int, int],
    template: np.ndarray,
    template_top_left: tuple[int, int],
    lattice_cells: list[tuple[int, int]],
    origin_cell: tuple[int, int],
    step_r: int,
    step_c: int,
    lattice_top_left: tuple[int, int],
) -> np.ndarray:
    h, w = template.shape
    grid = np.full(shape, BG, dtype=int)

    tr0, tc0 = template_top_left
    grid[tr0 : tr0 + h, tc0 : tc0 + w][template] = BLUE

    lr0, lc0 = lattice_top_left
    for cell_r, cell_c in lattice_cells:
        r = lr0 + cell_r * step_r
        c = lc0 + cell_c * step_c
        grid[r, c] = GREEN if (cell_r, cell_c) == origin_cell else YELLOW
    return grid


def _semantic_ok(grid: np.ndarray) -> bool:
    try:
        template, base_center, marker_offsets, template_box = _extract_problem(grid)
        actual = main(grid)
    except Exception:
        return False

    if actual.shape != grid.shape:
        return False
    if max(actual.shape) > 30 or max(grid.shape) > 30:
        return False
    if not np.any(actual == RED):
        return False
    if not np.any(actual == GREEN):
        return False
    if not np.any(actual == BLUE):
        return False

    no_mirror = _render(grid.shape, template, base_center, marker_offsets, use_mirror_parity=False)
    if np.array_equal(actual, no_mirror):
        return False

    all_mirrored = _render(
        grid.shape,
        np.fliplr(template),
        base_center,
        [(dr, dc, is_origin) for dr, dc, is_origin in marker_offsets],
        use_mirror_parity=False,
    )
    if np.array_equal(actual, all_mirrored):
        return False

    r0, c0, r1, c1 = template_box
    if not (r0 <= base_center[0] <= r1 and c0 <= base_center[1] <= c1):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    template = _sample_template()
    th, tw = template.shape

    lattice_cells, origin_cell = _sample_marker_layout()
    row_vals = [r for r, _c in lattice_cells]
    col_vals = [c for _r, c in lattice_cells]
    min_dr = min(r - origin_cell[0] for r, _c in lattice_cells)
    max_dr = max(r - origin_cell[0] for r, _c in lattice_cells)
    min_dc = min(c - origin_cell[1] for _r, c in lattice_cells)
    max_dc = max(c - origin_cell[1] for _r, c in lattice_cells)

    base_r_lo = th // 2 - min_dr
    base_r_hi = min(h - 1 - (th // 2) - max_dr, 7)
    base_c_lo = tw // 2 - min_dc
    base_c_hi = min(w - 1 - (tw // 2) - max_dc, 7)
    if base_r_lo > base_r_hi or base_c_lo > base_c_hi:
        return None

    center_r = random.randint(base_r_lo, base_r_hi)
    center_c = random.randint(base_c_lo, base_c_hi)
    template_top_left = (center_r - th // 2, center_c - tw // 2)
    template_box = (
        template_top_left[0],
        template_top_left[1],
        template_top_left[0] + th - 1,
        template_top_left[1] + tw - 1,
    )

    step_r = random.randint(*STEP_RANGE)
    step_c = random.randint(*STEP_RANGE)
    lattice_h = (max(row_vals) - min(row_vals)) * step_r + 1
    lattice_w = (max(col_vals) - min(col_vals)) * step_c + 1
    if lattice_h > h or lattice_w > w:
        return None

    for _ in range(120):
        lr0 = random.randint(0, h - lattice_h)
        lc0 = random.randint(0, w - lattice_w)
        lattice_box = (lr0, lc0, lr0 + lattice_h - 1, lc0 + lattice_w - 1)
        if not _separated(template_box, lattice_box, gap=1):
            continue
        grid = _build_input(
            (h, w),
            template,
            template_top_left,
            lattice_cells,
            origin_cell,
            step_r,
            step_c,
            (lr0 - min(row_vals) * step_r, lc0 - min(col_vals) * step_c),
        )
        if _semantic_ok(grid):
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
