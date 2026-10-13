from __future__ import annotations

from itertools import combinations

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))
_RNG = np.random.default_rng()

# Generator configuration.
GEN_ATTEMPTS = 300
UPPER_H_RANGE = (13, 20)
UPPER_W_RANGE = (23, 31)
INNER_H_RANGE = (2, 7)
INNER_W_RANGE = (3, 8)
PLACEMENT_MARGIN = 2
LEGEND_GAP_MIN = 1
LEGEND_GAP_MAX = 2
EXTRA_LEGEND_MAX = 2


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _parse_legend(row: np.ndarray) -> dict[frozenset[int], int]:
    row = np.asarray(row, dtype=int)
    mapping: dict[frozenset[int], int] = {}
    c = 0
    w = int(row.shape[0])
    while c < w:
        while c < w and int(row[c]) == BG:
            c += 1
        if c >= w:
            break
        start = c
        while c < w and int(row[c]) != BG:
            c += 1
        group = row[start:c]
        assert int(group.shape[0]) == 3, "legend groups must be separated triples"
        a, b, fill = map(int, group.tolist())
        mapping[frozenset((a, b))] = fill
    assert mapping, "expected at least one legend mapping"
    return mapping


def _enclosed_regions(area: np.ndarray) -> list[tuple[list[tuple[int, int]], frozenset[int]]]:
    area = np.asarray(area, dtype=int)
    h, w = area.shape
    regions: list[tuple[list[tuple[int, int]], frozenset[int]]] = []
    for comp in _components(area == BG):
        touches_edge = any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp)
        if touches_edge:
            continue
        boundary: set[int] = set()
        for r, c in comp:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w:
                    color = int(area[nr, nc])
                    if color != BG:
                        boundary.add(color)
        regions.append((comp, frozenset(boundary)))
    return regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed black region whose boundary uses exactly two colors.

    The bottom row is a legend of separated triples (A, B, X). For every closed black
    region above that row, collect the distinct 4-neighbor boundary colors. If the
    boundary color set has size two, look up the unordered pair {A, B} in the legend
    and fill the region with X. All other regions stay black.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2

    mapping = _parse_legend(g[-1])
    out = g.copy()
    for comp, boundary in _enclosed_regions(g[:-1]):
        if len(boundary) != 2:
            continue
        fill = mapping.get(boundary)
        if fill is None:
            continue
        for r, c in comp:
            out[r, c] = int(fill)
    return out


def _box_one(color: int, inner_h: int, inner_w: int) -> np.ndarray:
    patch = np.zeros((inner_h + 2, inner_w + 2), dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    return patch


def _box_two(color_a: int, color_b: int, inner_h: int, inner_w: int) -> np.ndarray:
    patch = np.zeros((inner_h + 2, inner_w + 2), dtype=int)
    patch[0, :] = color_a
    patch[:, 0] = color_a
    patch[-1, :] = color_b
    patch[:, -1] = color_b
    return patch


def _box_three(color_a: int, color_b: int, color_c: int, inner_h: int, inner_w: int) -> np.ndarray:
    patch = np.zeros((inner_h + 2, inner_w + 2), dtype=int)
    patch[0, :] = color_a
    patch[:, 0] = color_a
    patch[-1, :] = color_b
    patch[:, -1] = color_c
    return patch


def _random_orient(patch: np.ndarray) -> np.ndarray:
    patch = np.rot90(patch, int(_RNG.integers(0, 4)))
    if bool(_RNG.integers(0, 2)):
        patch = np.fliplr(patch)
    return np.asarray(patch, dtype=int)


def _choose_fill_color(pair: frozenset[int], curve_colors: set[int], used_outputs: set[int]) -> int:
    pair_colors = set(pair)
    prefer = [c for c in COLORS if c not in pair_colors and c not in curve_colors and c not in used_outputs]
    if prefer:
        return int(prefer[int(_RNG.integers(len(prefer)))])
    prefer = [c for c in COLORS if c not in pair_colors and c not in used_outputs]
    if prefer:
        return int(prefer[int(_RNG.integers(len(prefer)))])
    fallback = [c for c in COLORS if c not in pair_colors]
    assert fallback
    return int(fallback[int(_RNG.integers(len(fallback)))])


def _sample_motifs() -> tuple[list[dict], list[tuple[int, int, int]]]:
    curve_colors = list(map(int, _RNG.choice(COLORS, size=8, replace=False).tolist()))
    cursor = 0
    motifs: list[dict] = []
    legend_entries: list[tuple[int, int, int]] = []
    all_curve_colors = set(curve_colors)
    used_outputs: set[int] = set()

    for kind in ("two", "two", "one", "three"):
        inner_h = int(_RNG.integers(INNER_H_RANGE[0], INNER_H_RANGE[1]))
        inner_w = int(_RNG.integers(INNER_W_RANGE[0], INNER_W_RANGE[1]))
        if kind == "two":
            a, b = curve_colors[cursor : cursor + 2]
            cursor += 2
            patch = _random_orient(_box_two(a, b, inner_h, inner_w))
            pair = frozenset((int(a), int(b)))
            fill = _choose_fill_color(pair, all_curve_colors, used_outputs)
            legend_entries.append((int(a), int(b), int(fill)))
            used_outputs.add(int(fill))
            motifs.append({"kind": kind, "boundary": pair, "patch": patch})
        elif kind == "one":
            (a,) = curve_colors[cursor : cursor + 1]
            cursor += 1
            patch = _random_orient(_box_one(a, inner_h, inner_w))
            motifs.append({"kind": kind, "boundary": frozenset((int(a),)), "patch": patch})
        else:
            a, b, c = curve_colors[cursor : cursor + 3]
            cursor += 3
            patch = _random_orient(_box_three(a, b, c, inner_h, inner_w))
            motifs.append({"kind": kind, "boundary": frozenset((int(a), int(b), int(c))), "patch": patch})

    used_pairs = {frozenset((int(a), int(b))) for a, b, _ in legend_entries}
    extra_candidates = [frozenset(pair) for pair in combinations(sorted(all_curve_colors), 2) if frozenset(pair) not in used_pairs]
    max_extra = min(EXTRA_LEGEND_MAX, len(extra_candidates))
    if max_extra > 0:
        extra_count = int(_RNG.integers(0, max_extra + 1))
        if extra_count:
            pick_idx = _RNG.choice(len(extra_candidates), size=extra_count, replace=False)
            for idx in np.atleast_1d(pick_idx).tolist():
                a, b = tuple(extra_candidates[int(idx)])
                fill = _choose_fill_color(frozenset((a, b)), all_curve_colors, used_outputs)
                legend_entries.append((int(a), int(b), int(fill)))
                used_outputs.add(int(fill))

    return motifs, legend_entries


def _can_place(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    hh, ww = occupied.shape
    if r0 < 1 or c0 < 1 or r0 + h > hh - 1 or c0 + w > ww - 1:
        return False
    rr0 = max(0, r0 - PLACEMENT_MARGIN)
    cc0 = max(0, c0 - PLACEMENT_MARGIN)
    rr1 = min(hh, r0 + h + PLACEMENT_MARGIN)
    cc1 = min(ww, c0 + w + PLACEMENT_MARGIN)
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _place_motifs(upper_h: int, upper_w: int, motifs: list[dict]) -> np.ndarray | None:
    grid = np.zeros((upper_h, upper_w), dtype=int)
    occupied = np.zeros((upper_h, upper_w), dtype=bool)
    order = sorted(range(len(motifs)), key=lambda i: int(np.prod(motifs[i]["patch"].shape)), reverse=True)

    for idx in order:
        patch = motifs[idx]["patch"]
        h, w = map(int, patch.shape)
        candidates: list[tuple[int, int]] = []
        for r0 in range(1, upper_h - h):
            for c0 in range(1, upper_w - w):
                if _can_place(occupied, r0, c0, h, w):
                    candidates.append((int(r0), int(c0)))
        if not candidates:
            return None
        r0, c0 = candidates[int(_RNG.integers(len(candidates)))]
        mask = patch != BG
        grid[r0 : r0 + h, c0 : c0 + w][mask] = patch[mask]
        occupied[r0 : r0 + h, c0 : c0 + w] |= mask
    return grid


def _render_legend(width: int, entries: list[tuple[int, int, int]]) -> np.ndarray | None:
    gaps = [int(_RNG.integers(LEGEND_GAP_MIN, LEGEND_GAP_MAX + 1)) for _ in range(max(0, len(entries) - 1))]
    needed = 3 * len(entries) + sum(gaps)
    if needed > width:
        return None
    order = np.asarray(_RNG.permutation(len(entries)), dtype=int)
    row = np.zeros(width, dtype=int)
    start = int(_RNG.integers(0, width - needed + 1))
    c = start
    for i, idx in enumerate(order.tolist()):
        a, b, fill = entries[int(idx)]
        if bool(_RNG.integers(0, 2)):
            a, b = b, a
        row[c : c + 3] = np.array([a, b, fill], dtype=int)
        c += 3
        if i < len(entries) - 1:
            c += gaps[i]
    return row


def _region_profile(grid: np.ndarray) -> tuple[int, int]:
    mapping = _parse_legend(grid[-1])
    fillable = 0
    blocked = 0
    for _comp, boundary in _enclosed_regions(grid[:-1]):
        if len(boundary) == 2 and boundary in mapping:
            fillable += 1
        else:
            blocked += 1
    return int(fillable), int(blocked)


def _gen_attempt() -> np.ndarray | None:
    motifs, legend_entries = _sample_motifs()
    min_legend_w = 3 * len(legend_entries) + max(0, len(legend_entries) - 1) * LEGEND_GAP_MIN
    upper_h = int(_RNG.integers(UPPER_H_RANGE[0], UPPER_H_RANGE[1]))
    upper_w = int(_RNG.integers(max(UPPER_W_RANGE[0], min_legend_w), UPPER_W_RANGE[1]))

    upper = _place_motifs(upper_h, upper_w, motifs)
    if upper is None:
        return None

    legend = _render_legend(upper_w, legend_entries)
    if legend is None:
        return None

    grid = np.vstack([upper, legend[None, :]])
    fillable, blocked = _region_profile(grid)
    if fillable != 2 or blocked != 2:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if not np.array_equal(out[-1], grid[-1]):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
