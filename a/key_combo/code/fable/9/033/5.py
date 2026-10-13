from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
SEP = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
ORTHO = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _components_by_color(region: np.ndarray, blocked: set[int] | None = None) -> list[dict]:
    blocked = set() if blocked is None else set(blocked)
    h, w = region.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(region[r, c])
            if seen[r, c] or color in blocked:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ORTHO:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(region[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append({"color": color, "coords": coords})
    return comps


def _components_any_foreground(region: np.ndarray, blocked: set[int] | None = None) -> list[list[tuple[int, int]]]:
    blocked = set() if blocked is None else set(blocked)
    h, w = region.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(region[r, c]) in blocked:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ORTHO:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(region[nr, nc]) not in blocked:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _is_exact_cross(coords: list[tuple[int, int]]) -> bool:
    if len(coords) != 5:
        return False
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, c0 = min(rows), min(cols)
    norm = {(r - r0, c - c0) for r, c in coords}
    return norm == {(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)}


def _has_plus_center(coords: list[tuple[int, int]]) -> bool:
    pts = set(coords)
    return any(all((r + dr, c + dc) in pts for dr, dc in ORTHO) for r, c in coords)


def _extract_domino_mapping(region: np.ndarray) -> dict[int, int] | None:
    comps = _components_any_foreground(region, blocked={BG, SEP})
    if len(comps) < 2:
        return None
    mapping: dict[int, int] = {}
    indegree: dict[int, int] = {}
    for coords in comps:
        if len(coords) != 2:
            return None
        (r1, c1), (r2, c2) = sorted(coords)
        if r1 != r2 or c2 != c1 + 1:
            return None
        left = int(region[r1, c1])
        right = int(region[r2, c2])
        if left in {BG, SEP} or right in {BG, SEP} or left == right:
            return None
        if left in mapping and mapping[left] != right:
            return None
        mapping[left] = right
        indegree[right] = indegree.get(right, 0) + 1
        if indegree[right] > 1:
            return None
    return mapping if mapping else None


def _candidate_layouts(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    candidates: list[dict] = []

    def try_split(axis: str, idx: int) -> None:
        if axis == "row":
            parts = [
                ("top", grid[:idx, :], (slice(0, idx), slice(0, w))),
                ("bottom", grid[idx + 1 :, :], (slice(idx + 1, h), slice(0, w))),
            ]
        else:
            parts = [
                ("left", grid[:, :idx], (slice(0, h), slice(0, idx))),
                ("right", grid[:, idx + 1 :], (slice(0, h), slice(idx + 1, w))),
            ]

        for legend_name, legend_region, legend_slice in parts:
            if legend_region.size == 0:
                continue
            other = parts[1] if legend_name == parts[0][0] else parts[0]
            _, main_region, main_slice = other
            mapping = _extract_domino_mapping(legend_region)
            if mapping is None:
                continue
            main_comps = _components_by_color(main_region, blocked={BG, SEP})
            exact_crosses = [comp for comp in main_comps if _is_exact_cross(comp["coords"])]
            if not exact_crosses:
                continue
            if not any(comp["color"] in mapping for comp in exact_crosses):
                continue
            if any(len(coords) == 2 for coords in _components_any_foreground(main_region, blocked={BG, SEP})):
                continue
            candidates.append(
                {
                    "axis": axis,
                    "sep": idx,
                    "legend_region": legend_region,
                    "legend_slice": legend_slice,
                    "main_region": main_region,
                    "main_slice": main_slice,
                    "mapping": mapping,
                    "main_components": main_comps,
                }
            )

    for r in range(1, h - 1):
        if np.all(grid[r, :] == SEP):
            try_split("row", r)
    for c in range(1, w - 1):
        if np.all(grid[:, c] == SEP):
            try_split("col", c)
    return candidates


def _analyze_grid(grid: np.ndarray) -> dict:
    candidates = _candidate_layouts(np.asarray(grid, dtype=int))
    assert len(candidates) == 1, "expected a unique separator/main/legend interpretation"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor only exact five-cell crosses in the main region to their directed successor colors from the legend domino chains.

    Input structure:
    - A full grey separator line splits the grid into a black-background main region and a legend region.
    - The legend region contains scattered horizontal 1x2 two-color dominoes; each domino points from its left color to its right color.
    - The main region contains several monochrome connected objects, including exact 5-cell '+' crosses and larger distractor shapes.

    Rule:
    - Read every legend domino as a directed successor relation X->Y.
    - In the main region, find monochrome connected components that are exactly a 5-cell cross: one center cell plus its four orthogonal neighbors.
    - Recolor each such entire cross of color X to Y when X has an outgoing domino X->Y.
    - Leave all other main-region objects unchanged, even if a larger object contains a local plus-like center; keep the legend region and separator unchanged.

    Key insight:
    - The two necessary filters are exact-object matching and directed color chaining: only whole 5-cell crosses qualify, and the dominoes encode one-way successors rather than symmetric swaps.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    info = _analyze_grid(g)
    main_region = g[info["main_slice"]].copy()

    for comp in info["main_components"]:
        if _is_exact_cross(comp["coords"]) and comp["color"] in info["mapping"]:
            for r, c in comp["coords"]:
                main_region[r, c] = info["mapping"][comp["color"]]

    out[info["main_slice"]] = main_region
    return out


def _shape_cross() -> list[tuple[int, int]]:
    return [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)]


def _shape_big_plus(rng: random.Random) -> list[tuple[int, int]]:
    cells = set(_shape_cross())
    arms = {
        "up": (0, 1, -1, 0),
        "down": (2, 1, 1, 0),
        "left": (1, 0, 0, -1),
        "right": (1, 2, 0, 1),
    }
    chosen = rng.sample(list(arms), k=rng.randint(1, 2))
    for name in chosen:
        r, c, dr, dc = arms[name]
        extra_len = rng.randint(1, 2)
        for step in range(1, extra_len + 1):
            cells.add((r + dr * step, c + dc * step))
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, c0 = min(rows), min(cols)
    return sorted((r - r0, c - c0) for r, c in cells)


def _shape_other(rng: random.Random) -> list[tuple[int, int]]:
    shapes = [
        [(0, 0), (0, 1), (0, 2)],
        [(0, 0), (1, 0), (2, 0)],
        [(0, 0), (0, 1), (1, 0)],
        [(0, 0), (1, 0), (1, 1), (2, 1)],
        [(0, 0), (0, 1), (1, 0), (1, 1)],
        [(0, 1), (1, 0), (1, 1), (2, 1)],
    ]
    return list(rng.choice(shapes))


def _can_place(occupied: np.ndarray, shape: list[tuple[int, int]], top: int, left: int) -> bool:
    h, w = occupied.shape
    for r, c in shape:
        rr, cc = top + r, left + c
        if not (0 <= rr < h and 0 <= cc < w):
            return False
        r_lo, r_hi = max(0, rr - 1), min(h, rr + 2)
        c_lo, c_hi = max(0, cc - 1), min(w, cc + 2)
        if occupied[r_lo:r_hi, c_lo:c_hi].any():
            return False
    return True


def _paint_shape(region: np.ndarray, occupied: np.ndarray, shape: list[tuple[int, int]], color: int, rng: random.Random) -> list[tuple[int, int]] | None:
    sh = max(r for r, _ in shape) + 1
    sw = max(c for _, c in shape) + 1
    starts = [(r, c) for r in range(region.shape[0] - sh + 1) for c in range(region.shape[1] - sw + 1)]
    rng.shuffle(starts)
    for top, left in starts:
        if not _can_place(occupied, shape, top, left):
            continue
        placed = []
        for r, c in shape:
            rr, cc = top + r, left + c
            region[rr, cc] = color
            occupied[rr, cc] = True
            placed.append((rr, cc))
        return placed
    return None


def _random_chains(rng: random.Random) -> list[list[int]]:
    chain_count = rng.choice([1, 2])
    if chain_count == 1:
        lengths = [rng.randint(3, 5)]
    else:
        lengths = [rng.randint(3, 4), rng.randint(2, 3)]
    total = sum(lengths)
    colors = rng.sample(PALETTE, total)
    chains = []
    idx = 0
    for length in lengths:
        chains.append(colors[idx : idx + length])
        idx += length
    return chains


def _build_mapping(chains: list[list[int]]) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for chain in chains:
        for a, b in zip(chain, chain[1:]):
            mapping[a] = b
    return mapping


def _compose_grid(main_region: np.ndarray, legend_region: np.ndarray, orientation: str) -> np.ndarray:
    if orientation == "vertical":
        grid = np.full((main_region.shape[0], main_region.shape[1] + 1 + legend_region.shape[1]), BG, dtype=int)
        grid[:, : main_region.shape[1]] = main_region
        grid[:, main_region.shape[1]] = SEP
        grid[:, main_region.shape[1] + 1 :] = legend_region
        return grid
    grid = np.full((main_region.shape[0] + 1 + legend_region.shape[0], main_region.shape[1]), BG, dtype=int)
    grid[: main_region.shape[0], :] = main_region
    grid[main_region.shape[0], :] = SEP
    grid[main_region.shape[0] + 1 :, :] = legend_region
    return grid


def _validate_semantics(inp: np.ndarray, expected: np.ndarray, planned: dict) -> bool:
    try:
        info = _analyze_grid(inp)
        actual = main(inp)
    except AssertionError:
        return False
    if not np.array_equal(actual, expected):
        return False
    if info["mapping"] != planned["mapping"]:
        return False
    exact_crosses = [comp for comp in info["main_components"] if _is_exact_cross(comp["coords"])]
    if len(exact_crosses) != planned["cross_count"]:
        return False
    mapped_big_plus = 0
    for comp in info["main_components"]:
        coords = comp["coords"]
        color = comp["color"]
        if len(coords) > 5 and _has_plus_center(coords) and color in planned["mapping"]:
            before = {tuple(p) for p in coords}
            after_same = all(actual[planned["main_slice"]][r, c] == color for r, c in before)
            if after_same:
                mapped_big_plus += 1
    if mapped_big_plus < 1:
        return False
    if planned["internal_source_color"] not in {comp["color"] for comp in exact_crosses}:
        return False
    if np.array_equal(inp, expected):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    orientation = rng.choice(["vertical", "horizontal"])
    chains = _random_chains(rng)
    mapping = _build_mapping(chains)
    internal_sources = [chain[i] for chain in chains for i in range(1, len(chain) - 1)]
    source_colors = list(mapping)
    internal_source = rng.choice(internal_sources)

    cross_colors = [internal_source, rng.choice(source_colors)]
    if len(source_colors) > 2 and rng.random() < 0.7:
        cross_colors.append(rng.choice(source_colors))
    if len(source_colors) > 2 and rng.random() < 0.35:
        cross_colors.append(rng.choice(source_colors))
    cross_count = len(cross_colors)

    if orientation == "vertical":
        main_h = rng.randint(13, 19)
        main_w = rng.randint(13, 18)
        legend_h = main_h
        legend_w = rng.randint(6, 9)
    else:
        main_h = rng.randint(13, 18)
        main_w = rng.randint(13, 19)
        legend_h = rng.randint(5, 8)
        legend_w = main_w

    main_region = np.full((main_h, main_w), BG, dtype=int)
    occupied_main = np.zeros((main_h, main_w), dtype=bool)
    expected_main = main_region.copy()

    planned_shapes = []
    for color in cross_colors:
        coords = _paint_shape(main_region, occupied_main, _shape_cross(), color, rng)
        if coords is None:
            return None
        for r, c in coords:
            expected_main[r, c] = mapping[color]
        planned_shapes.append(("cross", color, coords))

    big_plus_colors = [cross_colors[0]]
    if rng.random() < 0.6:
        big_plus_colors.append(rng.choice(source_colors))
    for color in big_plus_colors:
        shape = _shape_big_plus(rng)
        coords = _paint_shape(main_region, occupied_main, shape, color, rng)
        if coords is None or not _has_plus_center(coords):
            return None
        for r, c in coords:
            expected_main[r, c] = color
        planned_shapes.append(("big_plus", color, coords))

    other_count = rng.randint(1, 3)
    for _ in range(other_count):
        color = rng.choice(PALETTE)
        shape = _shape_other(rng)
        coords = _paint_shape(main_region, occupied_main, shape, color, rng)
        if coords is None:
            return None
        for r, c in coords:
            expected_main[r, c] = color
        planned_shapes.append(("other", color, coords))

    legend_region = np.full((legend_h, legend_w), BG, dtype=int)
    occupied_legend = np.zeros((legend_h, legend_w), dtype=bool)
    dominoes = list(mapping.items())
    rng.shuffle(dominoes)
    for left, right in dominoes:
        coords = _paint_shape(legend_region, occupied_legend, [(0, 0), (0, 1)], left, rng)
        if coords is None:
            return None
        (r1, c1), (r2, c2) = sorted(coords)
        legend_region[r1, c1] = left
        legend_region[r2, c2] = right

    inp = _compose_grid(main_region, legend_region, orientation)
    expected = _compose_grid(expected_main, legend_region, orientation)

    main_slice = (slice(0, main_h), slice(0, main_w))
    if orientation == "horizontal":
        main_slice = (slice(0, main_h), slice(0, main_w))

    planned = {
        "mapping": mapping,
        "cross_count": cross_count,
        "internal_source_color": internal_source,
        "main_slice": main_slice,
        "shapes": planned_shapes,
    }
    if not _validate_semantics(inp, expected, planned):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None and grid.ndim == 2 and max(grid.shape) <= 30:
            return grid
    raise RuntimeError("failed to generate a valid directed-domino-cross sample")
