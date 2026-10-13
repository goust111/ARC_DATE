from __future__ import annotations

from collections import Counter, deque
import itertools
import random

import numpy as np


BG = 0
GREY = 5
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
PALETTE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 500


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, flag in enumerate(flags.tolist()):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            runs.append((start, idx - 1))
            start = None
    if start is not None:
        runs.append((start, len(flags) - 1))
    return runs


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((int(cr), int(cc)))
                for dr, dc in DIR4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    return int(mins[0]), int(mins[1]), int(maxs[0]), int(maxs[1])


def _normalize_coords(coords: np.ndarray | list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    arr = np.asarray(coords, dtype=int)
    assert arr.size > 0
    arr = arr.reshape(-1, 2)
    mins = arr.min(axis=0)
    norm = arr - mins
    return tuple(sorted((int(r), int(c)) for r, c in norm.tolist()))


def _signature_dims(signature: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in signature) + 1, max(c for _, c in signature) + 1


def _fit_offsets(signature: tuple[tuple[int, int], ...], box_h: int, box_w: int) -> list[tuple[int, int]]:
    sig_h, sig_w = _signature_dims(signature)
    offsets: list[tuple[int, int]] = []
    for off_r in range(box_h - sig_h + 1):
        for off_c in range(box_w - sig_w + 1):
            offsets.append((off_r, off_c))
    return offsets


def _valid_palette_offsets(signature: tuple[tuple[int, int], ...]) -> list[tuple[int, int]]:
    valid: list[tuple[int, int]] = []
    for off_r, off_c in _fit_offsets(signature, 3, 3):
        holes = {(r + off_r, c + off_c) for r, c in signature}
        ok_rows = all(any((r, c) not in holes for c in range(3)) for r in range(3))
        ok_cols = all(any((r, c) not in holes for r in range(3)) for c in range(3))
        if ok_rows and ok_cols:
            valid.append((off_r, off_c))
    return valid


def _build_pattern_library() -> list[tuple[tuple[int, int], ...]]:
    shapes: set[tuple[tuple[int, int], ...]] = set()
    cells = [(r, c) for r in range(3) for c in range(3)]
    for area in range(1, 5):
        for combo in itertools.combinations(cells, area):
            sig = _normalize_coords(combo)
            if _valid_palette_offsets(sig):
                shapes.add(sig)
    return sorted(shapes, key=lambda sig: (len(sig), _signature_dims(sig), sig))


PATTERN_LIBRARY = _build_pattern_library()


def _enclosed_hole_mask(component_mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(component_mask)
    assert coords.size > 0
    r0, c0, r1, c1 = _bbox(coords)
    sub = component_mask[r0 : r1 + 1, c0 : c1 + 1]
    empty = ~sub
    seen = np.zeros_like(empty, dtype=bool)
    q: deque[tuple[int, int]] = deque()
    sh, sw = empty.shape

    def push(r: int, c: int) -> None:
        if empty[r, c] and not seen[r, c]:
            seen[r, c] = True
            q.append((r, c))

    for r in range(sh):
        push(r, 0)
        push(r, sw - 1)
    for c in range(sw):
        push(0, c)
        push(sh - 1, c)

    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < sh and 0 <= nc < sw and empty[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))

    holes = empty & ~seen
    full = np.zeros_like(component_mask, dtype=bool)
    full[r0 : r1 + 1, c0 : c1 + 1] = holes
    return full


def _exterior_background_mask(grid: np.ndarray, component_mask: np.ndarray, bg: int) -> np.ndarray:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    def push(r: int, c: int) -> None:
        if grid[r, c] == bg and not component_mask[r, c] and not seen[r, c]:
            seen[r, c] = True
            q.append((r, c))

    for r in range(h):
        push(r, 0)
        push(r, w - 1)
    for c in range(w):
        push(0, c)
        push(h - 1, c)

    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w:
                push(nr, nc)
    return seen


def _extract_palette_mapping(grid: np.ndarray, bg: int) -> tuple[dict[tuple[tuple[int, int], ...], int], int]:
    row_runs = _runs(np.any(grid != bg, axis=1))
    assert row_runs, "expected a top palette strip"
    pr0, pr1 = row_runs[0]
    band = grid[pr0 : pr1 + 1]
    mapping: dict[tuple[tuple[int, int], ...], int] = {}
    for color in [int(x) for x in np.unique(band) if int(x) != bg]:
        coords = np.argwhere(band == color)
        r0, c0, r1, c1 = _bbox(coords)
        patch = band[r0 : r1 + 1, c0 : c1 + 1]
        holes = np.argwhere(patch == bg)
        signature = _normalize_coords(holes)
        assert signature not in mapping, "palette hole patterns must be unique"
        mapping[signature] = int(color)
    assert mapping, "palette mapping must be non-empty"
    return mapping, pr1 + 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each grey region's interior hole layout to the palette and paint a
    one-cell outer halo with the corresponding palette color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    bg = BG if np.any(g == BG) else int(Counter(map(int, g.ravel())).most_common(1)[0][0])

    palette, lower_start = _extract_palette_mapping(g, bg)
    lower = g[lower_start:]
    lower_non_bg = lower[lower != bg]
    assert lower_non_bg.size > 0, "expected grey regions below the palette"
    grey = int(Counter(map(int, lower_non_bg.tolist())).most_common(1)[0][0])

    out = g.copy()
    for coords in _components(g == grey):
        component_mask = np.zeros_like(g, dtype=bool)
        component_mask[coords[:, 0], coords[:, 1]] = True

        hole_mask = _enclosed_hole_mask(component_mask)
        hole_coords = np.argwhere(hole_mask)
        assert hole_coords.size > 0, "each grey region must contain enclosed holes"
        signature = _normalize_coords(hole_coords)
        assert signature in palette, "every region hole pattern must match the palette"
        halo_color = int(palette[signature])

        exterior = _exterior_background_mask(g, component_mask, bg)
        halo = np.zeros_like(component_mask, dtype=bool)
        for r, c in coords.tolist():
            for dr, dc in DIR4:
                nr = int(r + dr)
                nc = int(c + dc)
                if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1] and exterior[nr, nc]:
                    halo[nr, nc] = True
        out[halo] = halo_color

    return out


def _make_region(signature: tuple[tuple[int, int], ...]) -> dict[str, object]:
    pad_top = random.randint(1, 3)
    pad_bottom = random.randint(1, 3)
    pad_left = random.randint(1, 3)
    pad_right = random.randint(1, 3)
    base_h = pad_top + 3 + pad_bottom
    base_w = pad_left + 3 + pad_right

    cells = {(r, c) for r in range(base_h) for c in range(base_w)}

    protrusions = random.choices([0, 1, 2], weights=[3, 4, 2])[0]
    for _ in range(protrusions):
        edge = random.choice(["top", "bottom", "left", "right"])
        if edge in ("top", "bottom"):
            length = random.randint(2, base_w)
            start = random.randint(0, base_w - length)
            thickness = random.randint(1, 2)
            rows = range(-thickness, 0) if edge == "top" else range(base_h, base_h + thickness)
            for r in rows:
                for c in range(start, start + length):
                    cells.add((r, c))
        else:
            length = random.randint(2, base_h)
            start = random.randint(0, base_h - length)
            thickness = random.randint(1, 2)
            cols = range(-thickness, 0) if edge == "left" else range(base_w, base_w + thickness)
            for r in range(start, start + length):
                for c in cols:
                    cells.add((r, c))

    offset_r, offset_c = random.choice(_fit_offsets(signature, 3, 3))
    holes = {(pad_top + offset_r + r, pad_left + offset_c + c) for r, c in signature}
    cells.difference_update(holes)

    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    norm_cells = sorted((r - min_r, c - min_c) for r, c in cells)
    norm_holes = sorted((r - min_r, c - min_c) for r, c in holes)
    height = max(r for r, _ in norm_cells) + 1
    width = max(c for _, c in norm_cells) + 1

    return {
        "cells": norm_cells,
        "holes": norm_holes,
        "height": height,
        "width": width,
        "signature": signature,
    }


def _place_region(
    grid: np.ndarray,
    forbidden: np.ndarray,
    region: dict[str, object],
    color: int,
    min_row: int,
) -> dict[str, object] | None:
    h, w = grid.shape
    cells = region["cells"]
    holes = set(region["holes"])
    rh = int(region["height"])
    rw = int(region["width"])

    row_lo = max(min_row, 1)
    row_hi = h - rh - 1
    col_lo = 1
    col_hi = w - rw - 1
    if row_lo > row_hi or col_lo > col_hi:
        return None

    for _ in range(200):
        top = random.randint(row_lo, row_hi)
        left = random.randint(col_lo, col_hi)
        footprint: set[tuple[int, int]] = set()
        ok = True
        for r, c in cells:
            rr = top + int(r)
            cc = left + int(c)
            if forbidden[rr, cc]:
                ok = False
                break
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr = rr + dr
                    nc = cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        ok = False
                        break
                    footprint.add((nr, nc))
                if not ok:
                    break
            if not ok:
                break
        if not ok or any(forbidden[r, c] for r, c in footprint):
            continue

        abs_cells = {(top + int(r), left + int(c)) for r, c in cells}
        abs_holes = {(top + int(r), left + int(c)) for r, c in holes}
        for r, c in abs_cells:
            grid[r, c] = color
        for r, c in footprint:
            forbidden[r, c] = True
        return {
            "cells": abs_cells,
            "holes": abs_holes,
            "color": color,
            "signature": region["signature"],
        }
    return None


def _expected_output(grid: np.ndarray, regions: list[dict[str, object]]) -> np.ndarray:
    out = grid.copy()
    h, w = grid.shape
    for region in regions:
        cells = region["cells"]
        holes = region["holes"]
        halo: set[tuple[int, int]] = set()
        for r, c in cells:
            for dr, dc in DIR4:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == BG and (nr, nc) not in holes:
                    halo.add((nr, nc))
        for r, c in halo:
            out[r, c] = int(region["color"])
    return out


def _gen_attempt() -> np.ndarray | None:
    n_palette = random.randint(3, 5)
    patterns = random.sample(PATTERN_LIBRARY, n_palette)
    palette_colors = random.sample(PALETTE_COLORS, n_palette)

    gaps = [random.randint(1, 2) for _ in range(n_palette - 1)]
    left_margin = random.randint(1, 3)
    right_margin = random.randint(1, 3)
    palette_width = left_margin + 3 * n_palette + sum(gaps) + right_margin
    sep_rows = random.randint(2, 4)

    n_regions = random.randint(2, min(6, n_palette + 2))
    region_pattern_ids = random.sample(range(n_palette), min(n_regions, n_palette))
    while len(region_pattern_ids) < n_regions:
        region_pattern_ids.append(random.randrange(n_palette))
    random.shuffle(region_pattern_ids)

    region_templates = [_make_region(patterns[idx]) for idx in region_pattern_ids]
    max_region_h = max(int(region["height"]) for region in region_templates)
    max_region_w = max(int(region["width"]) for region in region_templates)

    min_h = max(18, 3 + sep_rows + max_region_h + 4 + (2 if n_regions >= 4 else 0))
    min_w = max(24, palette_width + 1, max_region_w + 4)
    max_h = max(min_h, 30)
    max_w = max(min_w, 38)
    if min_h > max_h or min_w > max_w:
        return None

    h = random.randint(min_h, max_h)
    w = random.randint(min_w, max_w)
    grid = np.zeros((h, w), dtype=int)

    col = left_margin
    for idx, (pattern, color) in enumerate(zip(patterns, palette_colors)):
        holes = {(r + off_r, c + off_c) for off_r, off_c in [random.choice(_valid_palette_offsets(pattern))] for r, c in pattern}
        block = np.full((3, 3), int(color), dtype=int)
        for r, c in holes:
            block[r, c] = BG
        grid[0:3, col : col + 3] = block
        col += 3
        if idx < len(gaps):
            col += gaps[idx]

    forbidden = np.zeros((h, w), dtype=bool)
    forbidden[: 3 + sep_rows, :] = True
    regions: list[dict[str, object]] = []
    for template, palette_idx in zip(region_templates, region_pattern_ids):
        placed = _place_region(grid, forbidden, template, GREY, 3 + sep_rows)
        if placed is None:
            return None
        placed["color"] = palette_colors[palette_idx]
        regions.append(placed)

    expected = _expected_output(grid, regions)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
