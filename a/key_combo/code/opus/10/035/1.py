from __future__ import annotations

import random

import numpy as np


BLACK = 0
RED = 2
MAX_GENERATION_ATTEMPTS = 500

_CASE_ORDER = [
    ("odd", "left", "left", "right"),
    ("odd", "right", "right", "left"),
    ("even", "left", "right", "left"),
    ("even", "right", "left", "right"),
]
_CASE_CURSOR = 0


def _neighbors4(r: int, c: int, height: int, width: int):
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr = r + dr
        nc = c + dc
        if 0 <= nr < height and 0 <= nc < width:
            yield nr, nc


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, height, width):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.asarray(coords, dtype=int))
    return out


def _degree_map(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape
    deg = np.zeros(mask.shape, dtype=int)
    for r, c in np.argwhere(mask):
        deg[r, c] = sum(mask[nr, nc] for nr, nc in _neighbors4(int(r), int(c), height, width))
    return deg


def _denoise(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    out = g.copy()
    for r, c in np.argwhere(g == RED):
        support = sum(g[nr, nc] != BLACK for nr, nc in _neighbors4(int(r), int(c), height, width))
        if support < 2:
            out[r, c] = BLACK
    return out


def _mirror_complete(clean: np.ndarray) -> np.ndarray:
    out = np.asarray(clean, dtype=int).copy()
    mask = out == RED
    box = _bbox(mask)
    if box is None:
        return out
    _r0, left, _r1, right = box
    coords = np.argwhere(mask)
    for r, c in coords:
        mirror_c = left + right - int(c)
        out[int(r), mirror_c] = RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Remove isolated red noise, infer the cleaned vertical symmetry, and restore the missing side.

    Input structure:
    - The grid is mostly black background with red foreground pixels.
    - One 4-connected red main shape is almost left-right symmetric inside its bounding box, but 2-4 pixels are missing on exactly one side.
    - Additional red noise pixels are isolated and have too little 4-neighbor support.

    Rule:
    - Simultaneously erase every red pixel that has fewer than two non-black 4-neighbors.
    - Compute the bounding box of the remaining red shape and use the bbox's vertical center line as the symmetry axis.
    - For every surviving red pixel, mirror it across that axis and fill the mirrored cell with red.

    Key insight:
    - The red anomalies have opposite meanings: unsupported ones are noise to delete, while supported one-sided gaps reveal missing symmetric counterparts to fill.
    """
    clean = _denoise(np.asarray(grid, dtype=int))
    return _mirror_complete(clean)


def _side_of_col(col: int, width: int) -> str:
    doubled = 2 * int(col)
    axis_doubled = int(width) - 1
    if doubled < axis_doubled:
        return "left"
    if doubled > axis_doubled:
        return "right"
    return "axis"


def _make_full_mask(width_mode: str) -> np.ndarray | None:
    widths_pool = [3, 5, 7] if width_mode == "odd" else [2, 4, 6]
    for _ in range(200):
        height = random.randint(5, 8)
        widths = [random.choice(widths_pool[:2])]
        while len(widths) < height:
            prev = widths[-1]
            options = [w for w in widths_pool if abs(w - prev) <= 2]
            if prev not in options:
                options.append(prev)
            weights = [3 if w == prev else 1 for w in options]
            widths.append(random.choices(options, weights=weights, k=1)[0])
        if len(set(widths)) == 1 and random.random() < 0.7:
            continue
        bbox_w = max(widths)
        area = sum(widths)
        if not (15 <= area <= 30):
            continue
        if bbox_w < (5 if width_mode == "odd" else 4):
            continue

        mask = np.zeros((height, bbox_w), dtype=bool)
        for r, row_width in enumerate(widths):
            c0 = (bbox_w - row_width) // 2
            mask[r, c0 : c0 + row_width] = True

        deg = _degree_map(mask)
        if np.any(deg[mask] < 2):
            continue
        if widths.count(bbox_w) == 0:
            continue
        return mask
    return None


def _pick_missing(mask: np.ndarray, missing_side: str) -> tuple[np.ndarray, list[tuple[int, int]]] | None:
    height, width = mask.shape
    full_box = _bbox(mask)
    if full_box is None:
        return None

    candidates = []
    degrees = _degree_map(mask)
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if rr in (0, height - 1):
            continue
        if cc in (0, width - 1):
            continue
        if _side_of_col(cc, width) != missing_side:
            continue
        if degrees[rr, cc] < 3:
            continue
        candidates.append((rr, cc))

    if len(candidates) < 4:
        return None

    for _ in range(300):
        need = random.randint(2, min(4, len(candidates)))
        picked = random.sample(candidates, need)
        damaged = mask.copy()
        for r, c in picked:
            damaged[r, c] = False

        if _bbox(damaged) != full_box:
            continue
        comps = _components(damaged)
        if len(comps) != 1:
            continue
        deg = _degree_map(damaged)
        if np.any(deg[damaged] < 2):
            continue
        return damaged, picked
    return None


def _absolute_cells(mask: np.ndarray, top: int, left: int) -> set[tuple[int, int]]:
    return {
        (top + int(r), left + int(c))
        for r, c in np.argwhere(mask)
    }


def _choose_bbox_position(
    grid_h: int,
    grid_w: int,
    mask_h: int,
    mask_w: int,
    placement_side: str,
) -> tuple[int, int] | None:
    top_options = list(range(1, grid_h - mask_h))
    left_options = list(range(1, grid_w - mask_w))
    if not top_options or not left_options:
        return None

    grid_center = (grid_w - 1) / 2.0
    favored = []
    others = []
    for left in left_options:
        center = left + (mask_w - 1) / 2.0
        if placement_side == "left" and center < grid_center:
            favored.append(left)
        elif placement_side == "right" and center > grid_center:
            favored.append(left)
        else:
            others.append(left)
    pool = favored if favored else others
    return random.choice(top_options), random.choice(pool)


def _can_place_noise(
    cell: tuple[int, int],
    shape: tuple[int, int],
    blocked: set[tuple[int, int]],
    chosen: set[tuple[int, int]],
) -> bool:
    r, c = cell
    if cell in blocked or cell in chosen:
        return False
    for nb in _neighbors4(r, c, shape[0], shape[1]):
        if nb in blocked or nb in chosen:
            return False
    return True


def _place_noise(
    grid_shape: tuple[int, int],
    full_box_abs: tuple[int, int, int, int],
    blocked: set[tuple[int, int]],
    count: int,
    anchor_side: str,
) -> set[tuple[int, int]] | None:
    height, width = grid_shape
    _r0, c0, _r1, c1 = full_box_abs
    anchor_pool = []
    all_pool = []
    for r in range(height):
        for c in range(width):
            cell = (r, c)
            if c0 <= c <= c1:
                continue
            all_pool.append(cell)
            if anchor_side == "left" and c < c0:
                anchor_pool.append(cell)
            if anchor_side == "right" and c > c1:
                anchor_pool.append(cell)
    random.shuffle(anchor_pool)
    random.shuffle(all_pool)

    chosen: set[tuple[int, int]] = set()
    for cell in anchor_pool:
        if _can_place_noise(cell, grid_shape, blocked, chosen):
            chosen.add(cell)
            break
    if not chosen:
        return None

    for cell in all_pool:
        if len(chosen) >= count:
            break
        if _can_place_noise(cell, grid_shape, blocked, chosen):
            chosen.add(cell)
    if len(chosen) != count:
        return None
    return chosen


def _raw_mirror_without_denoise(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int).copy()
    mask = g == RED
    box = _bbox(mask)
    if box is None:
        return g
    _r0, left, _r1, right = box
    for r, c in np.argwhere(mask):
        g[int(r), left + right - int(c)] = RED
    return g


def _next_case() -> tuple[str, str, str, str]:
    global _CASE_CURSOR
    case = _CASE_ORDER[_CASE_CURSOR % len(_CASE_ORDER)]
    _CASE_CURSOR += 1
    return case


def _gen_attempt(case: tuple[str, str, str, str]) -> np.ndarray | None:
    width_mode, missing_side, placement_side, noise_anchor_side = case
    full_local = _make_full_mask(width_mode)
    if full_local is None:
        return None
    damaged_info = _pick_missing(full_local, missing_side)
    if damaged_info is None:
        return None
    damaged_local, removed_local = damaged_info

    mask_h, mask_w = full_local.shape
    grid_h = random.randint(max(mask_h + 3, 13), 18)
    grid_w = random.randint(max(mask_w + 5, 13), 18)
    pos = _choose_bbox_position(grid_h, grid_w, mask_h, mask_w, placement_side)
    if pos is None:
        return None
    top, left = pos

    out = np.zeros((grid_h, grid_w), dtype=int)
    inp = np.zeros((grid_h, grid_w), dtype=int)
    out_cells = _absolute_cells(full_local, top, left)
    damaged_cells = _absolute_cells(damaged_local, top, left)

    for r, c in out_cells:
        out[r, c] = RED
    for r, c in damaged_cells:
        inp[r, c] = RED

    full_box_abs = (top, left, top + mask_h - 1, left + mask_w - 1)
    blocked = set(out_cells)
    noise_count = random.randint(5, 8)
    noise_cells = _place_noise(inp.shape, full_box_abs, blocked, noise_count, noise_anchor_side)
    if noise_cells is None:
        return None
    for r, c in noise_cells:
        inp[r, c] = RED

    cleaned = _denoise(inp)
    clean_expected = np.zeros_like(inp)
    for r, c in damaged_cells:
        clean_expected[r, c] = RED
    if not np.array_equal(cleaned, clean_expected):
        return None

    expected = out
    if not np.array_equal(main(inp), expected):
        return None
    if np.array_equal(_denoise(inp), expected):
        return None
    if np.array_equal(_raw_mirror_without_denoise(inp), expected):
        return None

    cleaned_mask = _denoise(inp) == RED
    if len(_components(cleaned_mask)) != 1:
        return None
    if _bbox(cleaned_mask) != full_box_abs:
        return None

    output_box = _bbox(expected == RED)
    if output_box != full_box_abs:
        return None

    removed_abs = {(top + r, left + c) for r, c in removed_local}
    if not (2 <= len(removed_abs) <= 4):
        return None
    if any(_side_of_col(c - left, mask_w) != missing_side for _r, c in removed_abs):
        return None
    if any(cell in noise_cells for cell in removed_abs):
        return None

    for r, c in noise_cells:
        support = sum(inp[nr, nc] == RED for nr, nc in _neighbors4(r, c, grid_h, grid_w))
        if support != 0:
            return None

    clean_deg = _degree_map(cleaned_mask)
    if np.any(clean_deg[cleaned_mask] < 2):
        return None

    if np.array_equal(inp, expected):
        return None
    if np.array_equal(cleaned, expected):
        return None

    return inp


def generate_input() -> np.ndarray:
    case = _next_case()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(case)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {MAX_GENERATION_ATTEMPTS} attempts for case {case}")
