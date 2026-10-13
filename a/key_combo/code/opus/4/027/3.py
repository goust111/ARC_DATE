import random

import numpy as np


BLACK = 0
BLUE = 1
GREEN = 3
PINK = 6


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _band_spans(grid: np.ndarray) -> list[tuple[int, int]]:
    sep_rows = np.all(grid == PINK, axis=1)
    spans: list[tuple[int, int]] = []
    start = None
    for r, is_sep in enumerate(sep_rows.tolist()):
        if not is_sep and start is None:
            start = int(r)
        elif is_sep and start is not None:
            spans.append((start, int(r)))
            start = None
    if start is not None:
        spans.append((start, int(grid.shape[0])))
    return spans


def _hole_components(band: np.ndarray) -> list[list[tuple[int, int]]]:
    holes: list[list[tuple[int, int]]] = []
    for comp in _components(band == BLACK):
        if any(
            r == 0 or c == 0 or r == band.shape[0] - 1 or c == band.shape[1] - 1
            for r, c in comp
        ):
            continue
        holes.append(comp)
    return holes


def _count_holes(band: np.ndarray) -> int:
    return len(_hole_components(band))


def _fill_holes(band: np.ndarray) -> np.ndarray:
    out = band.copy()
    for comp in _hole_components(out):
        for r, c in comp:
            out[r, c] = GREEN
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count each band's enclosed holes, use that count as a cyclic right-shift amount, then fill the shifted holes.

    Input structure:
    - The grid has a black background plus several full-width pink separator rows.
    - The separator rows split the grid into horizontal bands; each non-separator band contains one blue hollow sprite.
    - Each sprite has one or more enclosed black holes that are 4-connected black regions not touching the band's boundary.

    Rule:
    - For each horizontal band, detect the blue sprite's enclosed black holes and count them as K.
    - Cyclically shift the entire band K columns to the right, leaving the pink separator rows unchanged.
    - Recompute the enclosed holes in the shifted band and fill those hole cells green.

    Key insight:
    - The shift amount is not marked separately: it is encoded by the sprite's own topology, so the solver must infer hole count first and then apply the shift before filling.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    for r0, r1 in _band_spans(g):
        band = g[r0:r1, :]
        shift = _count_holes(band)
        shifted = np.roll(band, shift=shift, axis=1)
        out[r0:r1, :] = _fill_holes(shifted)
    return out


def _sprite_patch(hole_count: int, side: str, top_ext: int, bottom_ext: int) -> np.ndarray:
    base_w = 2 * hole_count + 1
    cells: set[tuple[int, int]] = set()

    for c in range(base_w):
        cells.add((0, c))
        cells.add((2, c))
    for c in range(0, base_w, 2):
        cells.add((1, c))

    if side == "left":
        for c in range(-top_ext, 0):
            cells.add((0, c))
        for c in range(-bottom_ext, 0):
            cells.add((2, c))
    else:
        for c in range(base_w, base_w + top_ext):
            cells.add((0, c))
        for c in range(base_w, base_w + bottom_ext):
            cells.add((2, c))

    min_c = min(c for _r, c in cells)
    max_c = max(c for _r, c in cells)
    patch = np.zeros((3, max_c - min_c + 1), dtype=int)
    for r, c in cells:
        patch[r, c - min_c] = BLUE
    return patch


def _extension_options(hole_count: int) -> list[tuple[int, int]]:
    budget = 20 - (5 * hole_count + 3)
    options = [(0, 0)]
    for top_ext in range(0, min(3, budget) + 1):
        for bottom_ext in range(0, min(3, budget - top_ext) + 1):
            if top_ext == 0 and bottom_ext == 0:
                continue
            options.append((top_ext, bottom_ext))
    return options


def _top_row_choices(anchor_mode: str, band_index: int) -> list[int]:
    if anchor_mode == "top" and band_index == 2:
        return [0]
    if anchor_mode == "bottom" and band_index == 0:
        return [1]
    return [0, 1]


def _separator_rows(anchor_mode: str) -> list[int]:
    return [0, 5, 10] if anchor_mode == "top" else [4, 9, 14]


def _is_single_blue_component(band: np.ndarray) -> bool:
    return len(_components(band == BLUE)) == 1


def _build_instance(rng: random.Random) -> np.ndarray | None:
    wrap_k = int(rng.choice([1, 2, 3]))
    specs: dict[int, dict] = {}

    for hole_count in (1, 2, 3):
        side_choices = ["left", "right"] if hole_count != wrap_k else ["right"]
        option_choices = _extension_options(hole_count)
        if hole_count == wrap_k:
            option_choices = [opt for opt in option_choices if max(opt) > 0]
        side = str(rng.choice(side_choices))
        top_ext, bottom_ext = rng.choice(option_choices)
        patch = _sprite_patch(hole_count, side, int(top_ext), int(bottom_ext))
        specs[hole_count] = {
            "patch": patch,
            "side": side,
            "top_ext": int(top_ext),
            "bottom_ext": int(bottom_ext),
        }

    widths_by_holes = [specs[k]["patch"].shape[1] for k in (1, 2, 3)]
    if widths_by_holes[0] < widths_by_holes[1] < widths_by_holes[2]:
        return None
    if sum(width != 2 * k + 1 for k, width in zip((1, 2, 3), widths_by_holes, strict=False)) < 2:
        return None

    width_min = max(13, max(specs[k]["patch"].shape[1] + k for k in (1, 2, 3)))
    if width_min > 18:
        return None
    width = int(rng.randint(width_min, 18))
    anchor_mode = str(rng.choice(["top", "bottom"]))
    row_seps = _separator_rows(anchor_mode)

    grid = np.zeros((15, width), dtype=int)
    for r in row_seps:
        grid[r, :] = PINK

    band_order = list(rng.sample([1, 2, 3], 3))
    spans = _band_spans(grid)
    wrapped_band_index = band_order.index(wrap_k)
    saw_wrap = False

    for band_index, ((r0, r1), hole_count) in enumerate(zip(spans, band_order, strict=False)):
        patch = specs[hole_count]["patch"]
        patch_w = int(patch.shape[1])
        base_w = 2 * hole_count + 1
        top_row = r0 + int(rng.choice(_top_row_choices(anchor_mode, band_index)))

        if band_index == wrapped_band_index:
            lo = width - hole_count - patch_w
            hi = min(width - patch_w, width - hole_count - base_w)
            if lo > hi:
                return None
            left = int(rng.randint(lo, hi))
        else:
            max_left = width - patch_w - hole_count
            if max_left < 0:
                return None
            left = int(rng.randint(0, max_left))

        grid[top_row : top_row + 3, left : left + patch_w] = np.maximum(
            grid[top_row : top_row + 3, left : left + patch_w],
            patch,
        )

        band = grid[r0:r1, :]
        shifted = np.roll(band, shift=hole_count, axis=1)
        if _count_holes(band) != hole_count or _count_holes(shifted) != hole_count:
            return None
        if not _is_single_blue_component(band):
            return None
        if band_index == wrapped_band_index:
            saw_wrap = bool(np.any(shifted[:, :hole_count] == BLUE))

    if not saw_wrap:
        return None
    return grid


def _validate_instance(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False
    if max(grid.shape) > 30:
        return False

    spans = _band_spans(grid)
    if len(spans) != 3:
        return False

    hole_counts = []
    widths = []
    saw_wrap = False

    for r0, r1 in spans:
        band = grid[r0:r1, :]
        if not _is_single_blue_component(band):
            return False
        if np.any((band != BLACK) & (band != BLUE)):
            return False
        comp = _components(band == BLUE)[0]
        min_c = min(c for _r, c in comp)
        max_c = max(c for _r, c in comp)
        widths.append(max_c - min_c + 1)

        hole_count = _count_holes(band)
        if hole_count not in (1, 2, 3):
            return False
        shifted = np.roll(band, shift=hole_count, axis=1)
        if _count_holes(shifted) != hole_count:
            return False
        if np.any(shifted[:, :hole_count] == BLUE):
            saw_wrap = True
        hole_counts.append(hole_count)

    sorted_widths = [width for _k, width in sorted(zip(hole_counts, widths, strict=False))]
    if sorted_widths[0] < sorted_widths[1] < sorted_widths[2]:
        return False
    if len(set(hole_counts)) != 3:
        return False
    if not saw_wrap:
        return False

    expected = main(grid)
    if not np.array_equal(expected[grid == PINK], grid[grid == PINK]):
        return False
    for r0, r1 in spans:
        before = grid[r0:r1, :]
        after = expected[r0:r1, :]
        if np.any((after != BLACK) & (after != BLUE) & (after != GREEN)):
            return False
        shift = _count_holes(before)
        rolled = np.roll(before, shift=shift, axis=1)
        if np.count_nonzero(after == GREEN) != sum(len(comp) for comp in _hole_components(rolled)):
            return False
        if not np.array_equal(_fill_holes(rolled), after):
            return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(400):
        rng = random.Random()
        grid = _build_instance(rng)
        if grid is not None and _validate_instance(grid):
            return grid
    raise RuntimeError("failed to generate a valid enclosed-hole roll puzzle")
