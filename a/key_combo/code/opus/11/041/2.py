from __future__ import annotations

import itertools
import random
from collections import Counter

import numpy as np


BG = 0
GREY = 5
NON_GREY_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
LEGEND_TOP_BAND_MAX = 4
GEN_RETRIES = 400


def _neighbors(connectivity: int) -> tuple[tuple[int, int], ...]:
    if connectivity == 4:
        return ((-1, 0), (1, 0), (0, -1), (0, 1))
    return (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    )


def _components(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    dirs = _neighbors(connectivity)
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
                for dr, dc in dirs:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _normalize(coords: np.ndarray | list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    arr = np.array(coords, dtype=int)
    r0, c0 = arr.min(axis=0)
    norm = sorted((int(r - r0), int(c - c0)) for r, c in arr)
    return tuple(norm)


def _mask_from_coords(coords: tuple[tuple[int, int], ...]) -> np.ndarray:
    h = 1 + max(r for r, _c in coords)
    w = 1 + max(c for _r, c in coords)
    mask = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        mask[r, c] = True
    return mask


def _hole_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    bg = ~mask
    seen = np.zeros_like(bg, dtype=bool)
    holes = 0
    for r in range(h):
        for c in range(w):
            if not bg[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            touches_border = False
            while stack:
                rr, cc = stack.pop()
                if rr == 0 or cc == 0 or rr == h - 1 or cc == w - 1:
                    touches_border = True
                for dr, dc in _neighbors(4):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bg[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            if not touches_border:
                holes += 1
    return holes


def _color_coords(grid: np.ndarray, color: int) -> np.ndarray:
    return np.argwhere(grid == int(color))


def _shift_cells(cells: tuple[tuple[int, int], ...], top: int, left: int) -> tuple[tuple[int, int], ...]:
    return tuple((int(top + r), int(left + c)) for r, c in cells)


def _paint(grid: np.ndarray, cells: tuple[tuple[int, int], ...], color: int) -> None:
    for r, c in cells:
        grid[r, c] = int(color)


def _template_from_grid(grid: np.ndarray) -> tuple[tuple[tuple[int, int], ...], tuple[int, int, int, int]]:
    comps = _components(grid == GREY, connectivity=8)
    assert len(comps) == 1, "expected one grey template component"
    coords = comps[0]
    return _normalize(coords), _bbox(coords)


def _classify_colors(
    grid: np.ndarray,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    legend: dict[int, np.ndarray] = {}
    fragments: dict[int, np.ndarray] = {}
    for color in sorted(set(map(int, grid.flatten())) - {BG, GREY}):
        coords = _color_coords(grid, color)
        r0, _c0, _r1, _c1 = _bbox(coords)
        if r0 <= LEGEND_TOP_BAND_MAX and coords.shape[0] >= 4:
            legend[color] = coords
        else:
            fragments[color] = coords
    return legend, fragments


def _legend_map(legend: dict[int, np.ndarray]) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for color, coords in legend.items():
        norm = _normalize(coords)
        holes = _hole_count(_mask_from_coords(norm))
        assert holes not in mapping, "legend hole counts must be unique"
        mapping[holes] = int(color)
    return mapping


def _candidate_placements(
    grid: np.ndarray,
    template: tuple[tuple[int, int], ...],
    frag_coords: np.ndarray,
    color: int,
) -> list[tuple[int, int]]:
    h, w = grid.shape
    template_list = list(template)
    frag_set = {tuple(map(int, p)) for p in frag_coords}
    candidates: set[tuple[int, int]] = set()
    for fr, fc in frag_set:
        for tr, tc in template_list:
            top = int(fr - tr)
            left = int(fc - tc)
            shifted = _shift_cells(template, top, left)
            if not all(0 <= r < h and 0 <= c < w for r, c in shifted):
                continue
            if not frag_set.issubset(set(shifted)):
                continue
            ok = True
            for r, c in shifted:
                value = int(grid[r, c])
                if value not in (BG, int(color)):
                    ok = False
                    break
            if ok:
                candidates.add((top, left))
    return sorted(candidates)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each colored fragment into a translated template copy, then recolor that copy by looking up its enclosed-hole count in the legend.

    Input structure:
    - A black background contains one complete grey template in the lower-left area.
    - The upper-right area contains 2-3 colored legend shapes, each with a distinct number of enclosed black holes.
    - The middle area contains 4-6 non-grey fragment groups; each group's pixels are a subset of one translated copy of the grey template.

    Rule:
    - Treat the grey component as the template mask.
    - For each fragment color, find the unique translation where the full template covers that fragment and does not overlap other non-background colors.
    - Draw the full translated copy, count its enclosed holes, read the legend color with the same hole count, and recolor the whole copy to that legend color.
    - Keep the original grey template and the legend unchanged.

    Key insight:
    - The fragment pixels anchor a unique template placement, but the final color is determined only after reconstructing the full shape and matching its topology against the legend.
    """
    g = np.asarray(grid, dtype=int)
    template, _template_bbox = _template_from_grid(g)
    legend, fragments = _classify_colors(g)
    hole_to_color = _legend_map(legend)

    out = g.copy()
    for color, frag_coords in fragments.items():
        candidates = _candidate_placements(g, template, frag_coords, color)
        assert len(candidates) == 1, f"fragment color {color} must determine exactly one placement"
        top, left = candidates[0]
        full_cells = _shift_cells(template, top, left)
        holes = _hole_count(_mask_from_coords(_normalize(full_cells)))
        assert holes in hole_to_color, f"missing legend entry for hole count {holes}"
        _paint(out, full_cells, hole_to_color[holes])
    return out


def _rows_to_mask(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


def _unique_variants(mask: np.ndarray) -> list[np.ndarray]:
    seen: set[bytes] = set()
    variants: list[np.ndarray] = []
    for base in (mask, np.fliplr(mask)):
        for k in range(4):
            cur = np.rot90(base, k)
            key = cur.astype(np.uint8).tobytes() + bytes(cur.shape)
            if key not in seen:
                seen.add(key)
                variants.append(cur.copy())
    return variants


def _template_mask_with_holes(holes: int) -> np.ndarray:
    if holes == 1:
        base = _rows_to_mask(["11110", "10110", "11111"])
    elif holes == 2:
        base = _rows_to_mask(["111110", "101010", "111111"])
    else:
        base = _rows_to_mask(["0111111", "1010101", "1111111"])
    return random.choice(_unique_variants(base))


def _legend_patch(holes: int) -> np.ndarray:
    if holes == 0:
        solid = random.choice([["11", "11"], ["111", "111", "111"]])
        return _rows_to_mask(solid)
    width = 2 * holes + 1
    rows = ["1" * width, "".join("0" if c % 2 else "1" for c in range(width)), "1" * width]
    return _rows_to_mask(rows)


def _pattern_occurrences(template_mask: np.ndarray, pattern: tuple[tuple[int, int], ...]) -> int:
    th, tw = template_mask.shape
    ph = 1 + max(r for r, _c in pattern)
    pw = 1 + max(c for _r, c in pattern)
    total = 0
    for top in range(th - ph + 1):
        for left in range(tw - pw + 1):
            if all(template_mask[top + r, left + c] for r, c in pattern):
                total += 1
    return total


def _fragment_pattern_library(template_mask: np.ndarray) -> dict[int, list[tuple[tuple[int, int], ...]]]:
    coords = tuple(sorted(map(tuple, np.argwhere(template_mask))))
    lib: dict[int, list[tuple[tuple[int, int], ...]]] = {2: [], 3: [], 4: []}
    for size in (2, 3, 4):
        seen: set[tuple[tuple[int, int], ...]] = set()
        for combo in itertools.combinations(coords, size):
            pattern = _normalize(combo)
            if pattern in seen:
                continue
            if _pattern_occurrences(template_mask, pattern) == 1:
                seen.add(pattern)
        lib[size] = list(seen)
        random.shuffle(lib[size])
    return lib


def _mark_with_padding(occupied: np.ndarray, cells: tuple[tuple[int, int], ...], pad: int = 1) -> None:
    h, w = occupied.shape
    for r, c in cells:
        for dr in range(-pad, pad + 1):
            for dc in range(-pad, pad + 1):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    occupied[nr, nc] = True


def _can_place(
    occupied: np.ndarray,
    cells: tuple[tuple[int, int], ...],
    h: int,
    w: int,
) -> bool:
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if occupied[r, c]:
            return False
    return True


def _generate_one() -> np.ndarray:
    h = random.randint(24, 30)
    w = random.randint(24, 30)
    grid = np.zeros((h, w), dtype=int)

    template_holes = random.randint(1, 3)
    template_mask = _template_mask_with_holes(template_holes)
    template_cells = tuple(sorted(map(tuple, np.argwhere(template_mask))))
    th, tw = template_mask.shape

    template_top = h - th - random.randint(1, 3)
    template_left = random.randint(1, 3)
    grey_cells = _shift_cells(template_cells, template_top, template_left)
    _paint(grid, grey_cells, GREY)

    legend_count = random.randint(2, 3)
    other_holes = [v for v in range(4) if v != template_holes]
    legend_holes = [template_holes] + random.sample(other_holes, legend_count - 1)
    random.shuffle(legend_holes)
    legend_colors = random.sample(NON_GREY_COLORS, legend_count)
    legend_cells_by_color: dict[int, tuple[tuple[int, int], ...]] = {}
    legend_widths: list[int] = []
    legend_masks: list[np.ndarray] = []
    for holes in legend_holes:
        patch = _legend_patch(holes)
        legend_masks.append(patch)
        legend_widths.append(patch.shape[1])
    total_legend_width = sum(legend_widths) + (legend_count - 1)
    legend_top = random.randint(1, 2)
    legend_left = w - total_legend_width - random.randint(1, 3)
    if legend_left < 1:
        raise AssertionError("legend does not fit")
    cursor = legend_left
    for holes, color, patch in zip(legend_holes, legend_colors, legend_masks):
        patch_cells = tuple(sorted(map(tuple, np.argwhere(patch))))
        shifted = _shift_cells(patch_cells, legend_top, cursor)
        legend_cells_by_color[color] = shifted
        _paint(grid, shifted, color)
        cursor += patch.shape[1] + 1

    free_colors = [c for c in NON_GREY_COLORS if c not in legend_colors]
    frag_max = min(6, len(free_colors))
    frag_count = random.randint(4, frag_max)
    frag_colors = random.sample(free_colors, frag_count)

    pattern_lib = _fragment_pattern_library(template_mask)
    available_sizes = [size for size, patterns in pattern_lib.items() if patterns]
    assert available_sizes, "template must provide unique fragment patterns"

    chosen_patterns: list[tuple[tuple[int, int], ...]] = []
    size_cycle = available_sizes[:]
    random.shuffle(size_cycle)
    for size in size_cycle[: min(len(size_cycle), frag_count)]:
        chosen_patterns.append(random.choice(pattern_lib[size]))
    while len(chosen_patterns) < frag_count:
        size = random.choice(available_sizes)
        chosen_patterns.append(random.choice(pattern_lib[size]))
    random.shuffle(chosen_patterns)

    occupied = np.zeros_like(grid, dtype=bool)
    _mark_with_padding(occupied, grey_cells, pad=1)
    for cells in legend_cells_by_color.values():
        _mark_with_padding(occupied, cells, pad=1)

    workspace_rows = list(range(7, h - th - 3))
    workspace_cols = list(range(4, w - tw - 4))
    if not workspace_rows or not workspace_cols:
        raise AssertionError("workspace too small")

    for color, pattern in zip(frag_colors, chosen_patterns):
        candidates = [(r, c) for r in workspace_rows for c in workspace_cols]
        random.shuffle(candidates)
        placed = False
        for top, left in candidates:
            full_cells = _shift_cells(template_cells, top, left)
            if not _can_place(occupied, full_cells, h, w):
                continue
            frag_cells = _shift_cells(pattern, top, left)
            _paint(grid, frag_cells, color)
            _mark_with_padding(occupied, full_cells, pad=1)
            placed = True
            break
        if not placed:
            raise AssertionError("failed to place all fragment groups")

    return grid


def _validate_generated(grid: np.ndarray) -> None:
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2
    assert max(grid.shape) <= 30

    template, _ = _template_from_grid(grid)
    template_holes = _hole_count(_mask_from_coords(template))
    assert 1 <= template_holes <= 3

    legend, fragments = _classify_colors(grid)
    assert 2 <= len(legend) <= 3
    assert 4 <= len(fragments) <= 6
    hole_to_color = _legend_map(legend)
    assert template_holes in hole_to_color

    fragment_sizes = []
    for color, coords in fragments.items():
        fragment_sizes.append(int(coords.shape[0]))
        assert 2 <= coords.shape[0] <= 4
        assert _hole_count(_mask_from_coords(_normalize(coords))) == 0
        candidates = _candidate_placements(grid, template, coords, color)
        assert len(candidates) == 1
        top, left = candidates[0]
        full_cells = _shift_cells(template, top, left)
        assert len(full_cells) > coords.shape[0]
        assert _hole_count(_mask_from_coords(_normalize(full_cells))) == template_holes

    assert any(size == 2 for size in fragment_sizes) or any(size == 4 for size in fragment_sizes)

    out = main(grid)
    assert out.shape == grid.shape
    assert np.array_equal(out[grid == GREY], grid[grid == GREY])
    for color, coords in legend.items():
        rr, cc = coords[:, 0], coords[:, 1]
        assert np.array_equal(out[rr, cc], grid[rr, cc])
    for color in fragments:
        assert not np.any(out == color)


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            grid = _generate_one()
            _validate_generated(grid)
            return grid
        except Exception:
            continue
    raise AssertionError("failed to generate a valid puzzle input")
