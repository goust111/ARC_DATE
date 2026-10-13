from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import bbox, colors_present, crop_bbox


def _swap_array(arr: np.ndarray, color_a: int, color_b: int) -> np.ndarray:
    out = np.array(arr, copy=True)
    mask_a = out == int(color_a)
    mask_b = out == int(color_b)
    out[mask_a] = int(color_b)
    out[mask_b] = int(color_a)
    return out


def _is_solid_rectangle(mask: np.ndarray) -> bool:
    box = bbox(mask)
    if box is None:
        return False
    r0, c0, r1, c1 = box
    area = (r1 - r0 + 1) * (c1 - c0 + 1)
    return int(mask.sum()) == int(area) and bool(crop_bbox(mask, box).all())


def _separator_info(grid: np.ndarray) -> tuple[int, int, int]:
    h, w = map(int, grid.shape)
    assert h % 2 == 1 and w % 2 == 1, "separator cross requires odd grid dimensions"
    sep_r = h // 2
    sep_c = w // 2
    sep_color = int(grid[sep_r, 0])
    assert np.all(grid[sep_r, :] == sep_color), "separator row must be solid"
    assert np.all(grid[:, sep_c] == sep_color), "separator column must be solid"
    return sep_r, sep_c, sep_color


def _quadrants(grid: np.ndarray, sep_r: int, sep_c: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tl = np.array(grid[:sep_r, :sep_c], copy=True)
    tr = np.array(grid[:sep_r, sep_c + 1 :], copy=True)
    bl = np.array(grid[sep_r + 1 :, :sep_c], copy=True)
    br = np.array(grid[sep_r + 1 :, sep_c + 1 :], copy=True)
    assert tl.shape == tr.shape == bl.shape == br.shape
    return tl, tr, bl, br


def _normalize_quadrant(local: np.ndarray, quadrant: int, color_a: int, color_b: int, *, swap_odd: bool) -> np.ndarray:
    if quadrant == 0:
        aligned = np.array(local, copy=True)
    elif quadrant == 1:
        aligned = np.fliplr(local)
    elif quadrant == 2:
        aligned = np.flipud(local)
    elif quadrant == 3:
        aligned = np.rot90(local, 2)
    else:
        raise ValueError("invalid quadrant index")

    if swap_odd and quadrant in (1, 2):
        aligned = _swap_array(aligned, color_a, color_b)
    return np.array(aligned, copy=True)


def _render_quadrant(canonical_tl: np.ndarray, quadrant: int, color_a: int, color_b: int, *, swap_odd: bool) -> np.ndarray:
    if quadrant == 0:
        local = np.array(canonical_tl, copy=True)
    elif quadrant == 1:
        local = np.fliplr(canonical_tl)
    elif quadrant == 2:
        local = np.flipud(canonical_tl)
    elif quadrant == 3:
        local = np.rot90(canonical_tl, 2)
    else:
        raise ValueError("invalid quadrant index")

    if swap_odd and quadrant in (1, 2):
        local = _swap_array(local, color_a, color_b)
    return np.array(local, copy=True)


def _recover_from_candidate(
    quads: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    palette: list[int],
    occluder: int,
    *,
    swap_odd: bool,
) -> tuple[np.ndarray, tuple[int, int]] | None:
    pattern_colors = [int(color) for color in palette if int(color) != int(occluder)]
    if len(pattern_colors) != 2:
        return None
    color_a, color_b = pattern_colors

    normalized = []
    for quadrant, quad in enumerate(quads):
        mask = quad == int(occluder)
        if not _is_solid_rectangle(mask):
            return None
        normalized.append(
            _normalize_quadrant(quad, quadrant, color_a, color_b, swap_odd=swap_odd)
        )

    h, w = quads[0].shape
    canonical = np.empty((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            values = {
                int(arr[r, c])
                for arr in normalized
                if int(arr[r, c]) != int(occluder)
            }
            if len(values) != 1:
                return None
            value = values.pop()
            if value not in (color_a, color_b):
                return None
            canonical[r, c] = int(value)
    return canonical, (color_a, color_b)


def _assemble_grid(canonical_tl: np.ndarray, sep_color: int, color_a: int, color_b: int, *, swap_odd: bool) -> np.ndarray:
    h, w = map(int, canonical_tl.shape)
    out = np.empty((2 * h + 1, 2 * w + 1), dtype=int)
    sep_r = h
    sep_c = w
    out[sep_r, :] = int(sep_color)
    out[:, sep_c] = int(sep_color)
    out[:sep_r, :sep_c] = _render_quadrant(canonical_tl, 0, color_a, color_b, swap_odd=swap_odd)
    out[:sep_r, sep_c + 1 :] = _render_quadrant(canonical_tl, 1, color_a, color_b, swap_odd=swap_odd)
    out[sep_r + 1 :, :sep_c] = _render_quadrant(canonical_tl, 2, color_a, color_b, swap_odd=swap_odd)
    out[sep_r + 1 :, sep_c + 1 :] = _render_quadrant(canonical_tl, 3, color_a, color_b, swap_odd=swap_odd)
    return out


def _solve(grid: np.ndarray, *, swap_odd: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    sep_r, sep_c, sep_color = _separator_info(g)
    quads = _quadrants(g, sep_r, sep_c)
    assert all(not np.any(quad == sep_color) for quad in quads), "separator color should stay on the cross only"

    palette = colors_present(g, exclude={sep_color})
    assert len(palette) == 3, "expected two pattern colors plus one occluder color"

    matches: list[tuple[int, np.ndarray, tuple[int, int]]] = []
    for occluder in palette:
        recovered = _recover_from_candidate(quads, palette, int(occluder), swap_odd=swap_odd)
        if recovered is not None:
            canonical, pattern_colors = recovered
            matches.append((int(occluder), canonical, pattern_colors))

    assert len(matches) == 1, "occluder color must be structurally unique"
    _occluder, canonical_tl, (color_a, color_b) = matches[0]
    return _assemble_grid(canonical_tl, sep_color, color_a, color_b, swap_odd=swap_odd)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair the occluded four-quadrant mirror hall by restoring the hidden bicolor cells with reflection-parity color swaps.

    Input structure:
    - One solid separator row and one solid separator column split the grid into four equal-size quadrants.
    - The top-left quadrant is a two-color source pattern; the top-right and bottom-left are its horizontal and vertical reflections with the two colors swapped, and the bottom-right is its 180-degree rotation with colors preserved.
    - Each quadrant contains one solid rectangular occluder of the same unique color, while the separator cross itself stays unchanged.

    Rule:
    - Identify the unique occluder color, ignore those cells, and map every quadrant back into the top-left coordinate frame.
    - At each aligned position, take the unique visible non-occluder value; top-right and bottom-left contribute through one reflection so their two pattern colors must be swapped, while bottom-right contributes through two reflections so its colors stay unchanged.
    - Reconstruct the full four-quadrant assembly from the recovered top-left pattern and keep the separator cross unchanged.

    Key insight:
    - Pure mirror copying is insufficient: odd numbers of reflections toggle the two pattern colors, so hidden cells sometimes must be recovered through a third quadrant after accumulating reflection parity.
    """
    return _solve(np.asarray(grid, dtype=int), swap_odd=True)


# --------------- generator (o5x) ---------------

QUAD_H_RANGE = (5, 12)
QUAD_W_RANGE = (5, 12)
PATTERN_OP_RANGE = (4, 7)
PATTERN_DENSITY_RANGE = (0.22, 0.78)
OCCLUDER_AREA_FRAC = (0.10, 0.32)
GENERATION_RETRIES = 400

DIAG1 = "tl_br"
DIAG2 = "tr_bl"
MODE_BOTH = "both"
OCCLUDER_MODES = (DIAG1, DIAG2, MODE_BOTH)


def _toggle_rectangle(mask: np.ndarray) -> None:
    h, w = mask.shape
    hh = random.randint(1, max(1, h - 1))
    ww = random.randint(1, max(1, w - 1))
    r0 = random.randint(0, h - hh)
    c0 = random.randint(0, w - ww)
    mask[r0 : r0 + hh, c0 : c0 + ww] = ~mask[r0 : r0 + hh, c0 : c0 + ww]


def _toggle_row_band(mask: np.ndarray) -> None:
    h, _w = mask.shape
    band = random.randint(1, min(2, h))
    r0 = random.randint(0, h - band)
    mask[r0 : r0 + band, :] = ~mask[r0 : r0 + band, :]


def _toggle_col_band(mask: np.ndarray) -> None:
    _h, w = mask.shape
    band = random.randint(1, min(2, w))
    c0 = random.randint(0, w - band)
    mask[:, c0 : c0 + band] = ~mask[:, c0 : c0 + band]


def _toggle_main_diagonal(mask: np.ndarray) -> None:
    h, w = mask.shape
    offset = random.randint(-(h - 1), w - 1)
    band = random.randint(1, 2)
    rr, cc = np.indices(mask.shape)
    sel = np.abs((cc - rr) - offset) < band
    mask[sel] = ~mask[sel]


def _toggle_anti_diagonal(mask: np.ndarray) -> None:
    h, w = mask.shape
    offset = random.randint(0, h + w - 2)
    band = random.randint(1, 2)
    rr, cc = np.indices(mask.shape)
    sel = np.abs((rr + cc) - offset) < band
    mask[sel] = ~mask[sel]


def _toggle_l_shape(mask: np.ndarray) -> None:
    h, w = mask.shape
    corner = random.choice(("tl", "tr", "bl", "br"))
    arm_h = random.randint(2, h)
    arm_w = random.randint(2, w)
    thickness = random.randint(1, min(2, h, w))

    if corner == "tl":
        mask[:arm_h, :thickness] = ~mask[:arm_h, :thickness]
        mask[:thickness, :arm_w] = ~mask[:thickness, :arm_w]
    elif corner == "tr":
        mask[:arm_h, w - thickness :] = ~mask[:arm_h, w - thickness :]
        mask[:thickness, w - arm_w :] = ~mask[:thickness, w - arm_w :]
    elif corner == "bl":
        mask[h - arm_h :, :thickness] = ~mask[h - arm_h :, :thickness]
        mask[h - thickness :, :arm_w] = ~mask[h - thickness :, :arm_w]
    else:
        mask[h - arm_h :, w - thickness :] = ~mask[h - arm_h :, w - thickness :]
        mask[h - thickness :, w - arm_w :] = ~mask[h - thickness :, w - arm_w :]


def _degenerate_pattern(mask: np.ndarray) -> bool:
    variants = (
        np.fliplr(mask),
        np.flipud(mask),
        np.rot90(mask, 2),
    )
    for variant in variants:
        if np.array_equal(mask, variant):
            return True
        if np.array_equal(mask, ~variant):
            return True
    return False


def _gen_pattern_mask(shape: tuple[int, int]) -> np.ndarray | None:
    h, w = shape
    mask = np.full((h, w), random.choice([False, True]), dtype=bool)
    ops = [
        _toggle_rectangle,
        _toggle_row_band,
        _toggle_col_band,
        _toggle_main_diagonal,
        _toggle_anti_diagonal,
        _toggle_l_shape,
    ]
    for _ in range(random.randint(*PATTERN_OP_RANGE)):
        random.choice(ops)(mask)

    density = float(mask.mean())
    if not (PATTERN_DENSITY_RANGE[0] <= density <= PATTERN_DENSITY_RANGE[1]):
        return None
    if not mask.any() or bool(mask.all()):
        return None
    if _degenerate_pattern(mask):
        return None
    return mask


def _rect_mask(shape: tuple[int, int], rect: tuple[int, int, int, int]) -> np.ndarray:
    h, w = shape
    r0, c0, hh, ww = rect
    mask = np.zeros((h, w), dtype=bool)
    mask[r0 : r0 + hh, c0 : c0 + ww] = True
    return mask


def _sample_rect(shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    area = h * w
    max_h = max(1, min(h - 1, max(2, (2 * h) // 3)))
    max_w = max(1, min(w - 1, max(2, (2 * w) // 3)))
    min_area = max(2, int(np.ceil(area * OCCLUDER_AREA_FRAC[0])))
    max_area = max(min_area, int(np.floor(area * OCCLUDER_AREA_FRAC[1])))

    for _ in range(100):
        hh = random.randint(1, max_h)
        ww = random.randint(1, max_w)
        rect_area = hh * ww
        if not (min_area <= rect_area <= max_area):
            continue
        r0 = random.randint(0, h - hh)
        c0 = random.randint(0, w - ww)
        return r0, c0, hh, ww
    hh = min(max_h, max(1, h // 2))
    ww = min(max_w, max(1, w // 2))
    r0 = random.randint(0, h - hh)
    c0 = random.randint(0, w - ww)
    return r0, c0, hh, ww


def _sample_core(shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    hh = random.randint(1, min(2, h))
    ww = random.randint(1, min(2, w))
    r0 = random.randint(0, h - hh)
    c0 = random.randint(0, w - ww)
    return r0, c0, hh, ww


def _sample_rect_covering(shape: tuple[int, int], core: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    core_r0, core_c0, core_h, core_w = core
    core_r1 = core_r0 + core_h
    core_c1 = core_c0 + core_w
    area = h * w
    min_area = max(2, int(np.ceil(area * OCCLUDER_AREA_FRAC[0])))
    max_area = max(min_area, int(np.floor(area * OCCLUDER_AREA_FRAC[1])))
    max_h = max(1, min(h - 1, max(2, (2 * h) // 3)))
    max_w = max(1, min(w - 1, max(2, (2 * w) // 3)))

    for _ in range(100):
        hh = random.randint(core_h, max_h)
        ww = random.randint(core_w, max_w)
        rect_area = hh * ww
        if not (min_area <= rect_area <= max_area):
            continue
        r_lo = max(0, core_r1 - hh)
        r_hi = min(core_r0, h - hh)
        c_lo = max(0, core_c1 - ww)
        c_hi = min(core_c0, w - ww)
        if r_lo > r_hi or c_lo > c_hi:
            continue
        r0 = random.randint(r_lo, r_hi)
        c0 = random.randint(c_lo, c_hi)
        return r0, c0, hh, ww
    return _sample_rect(shape)


def _quadrant_occluder_masks(shape: tuple[int, int], canonical_tl: np.ndarray) -> list[np.ndarray] | None:
    mode = random.choice(OCCLUDER_MODES)
    h, w = shape
    if h < 2 or w < 2:
        return None

    for _ in range(200):
        rects: list[tuple[int, int, int, int] | None] = [None, None, None, None]
        if mode in (DIAG1, MODE_BOTH):
            core = _sample_core(shape)
            rects[0] = _sample_rect_covering(shape, core)
            rects[3] = _sample_rect_covering(shape, core)
        if mode in (DIAG2, MODE_BOTH):
            core = _sample_core(shape)
            rects[1] = _sample_rect_covering(shape, core)
            rects[2] = _sample_rect_covering(shape, core)
        for i in range(4):
            if rects[i] is None:
                rects[i] = _sample_rect(shape)

        masks = [_rect_mask(shape, rect) for rect in rects]  # type: ignore[arg-type]
        coverage = sum(mask.astype(int) for mask in masks)
        if int(coverage.max()) >= 4:
            continue
        if not np.any(coverage == 0):
            continue
        if not (np.any(masks[0] & masks[3]) or np.any(masks[1] & masks[2])):
            continue
        if mode in (DIAG1, MODE_BOTH) and not np.any(masks[0] & masks[3]):
            continue
        if mode in (DIAG2, MODE_BOTH) and not np.any(masks[1] & masks[2]):
            continue

        ok = True
        for mask in masks:
            if not np.any(mask & (coverage == 1)):
                ok = False
                break
            values = canonical_tl[mask]
            if len(set(map(int, values))) != 2:
                ok = False
                break
        if ok:
            return masks
    return None


def _actual_occluder_mask(mask_tl_frame: np.ndarray, quadrant: int) -> np.ndarray:
    if quadrant == 0:
        return np.array(mask_tl_frame, copy=True)
    if quadrant == 1:
        return np.fliplr(mask_tl_frame)
    if quadrant == 2:
        return np.flipud(mask_tl_frame)
    if quadrant == 3:
        return np.rot90(mask_tl_frame, 2)
    raise ValueError("invalid quadrant index")


def _build_clean_grid(canonical_tl: np.ndarray, separator: int, color_a: int, color_b: int) -> np.ndarray:
    return _assemble_grid(canonical_tl, separator, color_a, color_b, swap_odd=True)


def _occlude_grid(clean: np.ndarray, masks_tl_frame: list[np.ndarray], occluder: int) -> np.ndarray:
    inp = np.array(clean, copy=True)
    sep_r, sep_c = clean.shape[0] // 2, clean.shape[1] // 2
    slices = (
        (slice(0, sep_r), slice(0, sep_c)),
        (slice(0, sep_r), slice(sep_c + 1, clean.shape[1])),
        (slice(sep_r + 1, clean.shape[0]), slice(0, sep_c)),
        (slice(sep_r + 1, clean.shape[0]), slice(sep_c + 1, clean.shape[1])),
    )
    for quadrant, mask in enumerate(masks_tl_frame):
        local_mask = _actual_occluder_mask(mask, quadrant)
        view = inp[slices[quadrant]]
        view[local_mask] = int(occluder)
    return inp


def _wrong_solver_differs(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        wrong = _solve(inp, swap_odd=False)
    except AssertionError:
        return True
    return not np.array_equal(wrong, expected)


def _gen_attempt() -> np.ndarray | None:
    quad_h = random.randint(*QUAD_H_RANGE)
    quad_w = random.randint(*QUAD_W_RANGE)

    pattern_mask = _gen_pattern_mask((quad_h, quad_w))
    if pattern_mask is None:
        return None

    color_pool = random.sample(range(10), 4)
    color_a, color_b, separator, occluder = map(int, color_pool)
    canonical_tl = np.where(pattern_mask, color_a, color_b).astype(int)

    masks_tl_frame = _quadrant_occluder_masks((quad_h, quad_w), canonical_tl)
    if masks_tl_frame is None:
        return None

    clean = _build_clean_grid(canonical_tl, separator, color_a, color_b)
    inp = _occlude_grid(clean, masks_tl_frame, occluder)

    try:
        solved = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(solved, clean):
        return None
    if not _wrong_solver_differs(inp, clean):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirror-hall repair input")
