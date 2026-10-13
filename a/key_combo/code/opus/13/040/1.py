from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BLACK = 0
YELLOW = 4
ORANGE = 7

INPUT_COLORS = [1, 2, 3, 5, 6, 8, 9]
ORTHO_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

GRID_MIN = 12
GRID_MAX = 16
GEN_RETRIES = 800

COUNT_OPTIONS = {
    2: [(1, 2), (1, 4), (2, 3), (3, 4)],
    3: [(1, 2, 3), (1, 2, 4), (1, 3, 4), (2, 3, 4)],
}

# All shapes are 4-connected, irregular, and have area 3-6.
SHAPE_POOL = [
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (2, 0)),
    ((0, 0), (1, 0), (1, 1), (1, 2)),
    ((0, 1), (0, 2), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2)),
    ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1)),
    ((0, 0), (0, 2), (1, 0), (1, 1), (1, 2)),
    ((0, 1), (0, 2), (1, 0), (1, 1), (2, 0)),
    ((0, 1), (1, 0), (1, 1), (2, 1), (2, 2)),
    ((0, 0), (1, 0), (2, 0), (3, 0), (3, 1), (3, 2)),
    ((0, 0), (0, 1), (0, 2), (1, 0), (2, 0), (2, 1)),
    ((0, 1), (0, 2), (1, 0), (1, 1), (2, 0), (3, 0)),
    ((0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 3)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 0), (2, 2)),
]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _normalize(cells: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _orientations(shape: tuple[tuple[int, int], ...]) -> list[tuple[tuple[int, int], ...]]:
    seen: set[tuple[tuple[int, int], ...]] = set()
    out: list[tuple[tuple[int, int], ...]] = []
    for variant in range(8):
        cells = []
        for r, c in shape:
            if variant == 0:
                rr, cc = r, c
            elif variant == 1:
                rr, cc = r, -c
            elif variant == 2:
                rr, cc = -r, c
            elif variant == 3:
                rr, cc = -r, -c
            elif variant == 4:
                rr, cc = c, r
            elif variant == 5:
                rr, cc = c, -r
            elif variant == 6:
                rr, cc = -c, r
            else:
                rr, cc = -c, -r
            cells.append((rr, cc))
        norm = _normalize(cells)
        if norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out


SHAPE_ORIENTATIONS = {shape: _orientations(shape) for shape in SHAPE_POOL}


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in ORTHO_DIRS:
                    nr = rr + dr
                    nc = cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _color_components(grid: np.ndarray) -> dict[int, list[np.ndarray]]:
    return {
        color: _components(grid == color)
        for color in sorted(int(v) for v in np.unique(grid) if int(v) != BLACK)
    }


def _geometry_for_count(count: int) -> str:
    return "diag" if count % 2 == 1 else "ortho"


def _dirs_for_geometry(geometry: str) -> list[tuple[int, int]]:
    return DIAG_DIRS if geometry == "diag" else ORTHO_DIRS


def _claim_data(
    grid: np.ndarray,
    geometry_override: dict[int, str] | None = None,
) -> tuple[dict[int, set[tuple[int, int]]], dict[int, int], dict[int, str], dict[int, int]]:
    g = np.asarray(grid, dtype=int)
    components = _color_components(g)
    claims: dict[int, set[tuple[int, int]]] = {}
    areas: dict[int, int] = {}
    geometry: dict[int, str] = {}
    counts: dict[int, int] = {}

    for color, comps in components.items():
        counts[color] = len(comps)
        areas[color] = int(np.count_nonzero(g == color))
        chosen = geometry_override.get(color) if geometry_override else None
        geometry[color] = chosen or _geometry_for_count(counts[color])
        dirs = _dirs_for_geometry(geometry[color])
        cells = set()
        for r, c in np.argwhere(g == color):
            rr = int(r)
            cc = int(c)
            for dr, dc in dirs:
                nr = rr + dr
                nc = cc + dc
                if _in_bounds(g.shape, nr, nc) and g[nr, nc] == BLACK:
                    cells.add((nr, nc))
        claims[color] = cells
    return claims, areas, geometry, counts


def _solve_with_options(
    grid: np.ndarray,
    geometry_override: dict[int, str] | None = None,
    reverse_priority: bool = False,
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    claims, areas, geometry, _counts = _claim_data(g, geometry_override)

    cell_claimers: dict[tuple[int, int], list[int]] = defaultdict(list)
    for color, cells in claims.items():
        for cell in cells:
            cell_claimers[cell].append(color)

    def priority(color: int) -> tuple[int, int]:
        if reverse_priority:
            return (-areas[color], color)
        return (areas[color], color)

    for (r, c), colors in cell_claimers.items():
        winner = min(colors, key=priority)
        out[r, c] = YELLOW if geometry[winner] == "diag" else ORANGE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each foreground color with geometry chosen by its component-count parity, then resolve overlaps by smaller total area.

    Input structure:
    - A black background contains 2-3 non-black colors.
    - Each non-black color forms one or more small 4-connected monochromatic objects.
    - Different colors may be near each other, but the original foreground pixels are the only non-background cells.

    Rule:
    - For each foreground color, count its 4-connected components.
    - If the count is odd, every pixel of that color claims its four diagonal background neighbors and writes yellow.
    - If the count is even, every pixel of that color claims its four orthogonal background neighbors and writes orange.
    - Original foreground pixels stay unchanged.
    - If multiple colors claim the same background cell, the color with the smaller total input area wins; ties fall back to the smaller color id.

    Key insight:
    - The expansion geometry is not fixed by the pixel color itself; it is determined only after a global same-color component count, so counting, parity, and conflict resolution are all necessary.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30
    return _solve_with_options(g)


def _component_shape_signature(component: np.ndarray) -> tuple[tuple[int, int], ...]:
    return _normalize([(int(r), int(c)) for r, c in component])


def _shape_is_irregular(shape: tuple[tuple[int, int], ...]) -> bool:
    rows = [r for r, _ in shape]
    cols = [c for _, c in shape]
    height = max(rows) - min(rows) + 1
    width = max(cols) - min(cols) + 1
    return len(shape) < height * width


def _min_chebyshev_distance(a: np.ndarray, b: np.ndarray) -> int:
    best = 10**9
    for r0, c0 in a:
        for r1, c1 in b:
            best = min(best, max(abs(int(r0) - int(r1)), abs(int(c0) - int(c1))))
    return best


def _translate(shape: tuple[tuple[int, int], ...], dr: int, dc: int) -> list[tuple[int, int]]:
    return [(dr + r, dc + c) for r, c in shape]


def _placeable(
    cells: list[tuple[int, int]],
    grid: np.ndarray,
    same_color_components: list[np.ndarray],
    forbidden: set[tuple[int, int]],
) -> bool:
    for r, c in cells:
        if not _in_bounds(grid.shape, r, c) or grid[r, c] != BLACK:
            return False
        if (r, c) in forbidden:
            return False
    arr = np.array(cells, dtype=int)
    for comp in same_color_components:
        if _min_chebyshev_distance(arr, comp) < 3:
            return False
    return True


def _choose_counts(num_colors: int) -> tuple[int, ...]:
    counts = list(random.choice(COUNT_OPTIONS[num_colors]))
    random.shuffle(counts)
    return tuple(counts)


def _pick_anchor_positions(
    hub: tuple[int, int],
    counts_by_color: dict[int, int],
) -> dict[int, tuple[int, int]]:
    hr, hc = hub
    diag_positions = [(hr - 1, hc - 1), (hr - 1, hc + 1), (hr + 1, hc - 1), (hr + 1, hc + 1)]
    ortho_positions = [(hr - 1, hc), (hr + 1, hc), (hr, hc - 1), (hr, hc + 1)]
    random.shuffle(diag_positions)
    random.shuffle(ortho_positions)
    anchors: dict[int, tuple[int, int]] = {}
    diag_colors = [color for color, count in counts_by_color.items() if count % 2 == 1]
    ortho_colors = [color for color, count in counts_by_color.items() if count % 2 == 0]
    for color, pos in zip(diag_colors, diag_positions):
        anchors[color] = pos
    for color, pos in zip(ortho_colors, ortho_positions):
        anchors[color] = pos
    return anchors


def _anchor_component_candidates(
    anchor: tuple[int, int],
    hub: tuple[int, int],
    used_signatures: set[tuple[tuple[int, int], ...]],
) -> list[list[tuple[int, int]]]:
    ar, ac = anchor
    hr, hc = hub
    anchor_dist = abs(ar - hr) + abs(ac - hc)
    outward = (ar - hr, ac - hc)
    candidates: list[list[tuple[int, int]]] = []
    for base in SHAPE_POOL:
        for oriented in SHAPE_ORIENTATIONS[base]:
            if oriented in used_signatures:
                continue
            for pivot_r, pivot_c in oriented:
                cells = [(ar + r - pivot_r, ac + c - pivot_c) for r, c in oriented]
                if hub in cells:
                    continue
                if any(abs(r - hr) + abs(c - hc) < anchor_dist for r, c in cells if (r, c) != anchor):
                    continue
                support = 0
                for r, c in cells:
                    if (r, c) == anchor:
                        continue
                    dr = r - ar
                    dc = c - ac
                    if dr * outward[0] + dc * outward[1] >= 0:
                        support += 1
                if support == 0:
                    continue
                candidates.append(cells)
    random.shuffle(candidates)
    return candidates


def _place_anchor_component(
    grid: np.ndarray,
    color: int,
    anchor: tuple[int, int],
    hub: tuple[int, int],
    components_by_color: dict[int, list[np.ndarray]],
    used_signatures: set[tuple[tuple[int, int], ...]],
    reserved: set[tuple[int, int]],
) -> bool:
    own_forbidden = reserved - {anchor}
    for cells in _anchor_component_candidates(anchor, hub, used_signatures):
        if not _placeable(cells, grid, components_by_color[color], own_forbidden):
            continue
        sig = _normalize(cells)
        grid[tuple(np.array(cells).T)] = color
        components_by_color[color].append(np.array(cells, dtype=int))
        used_signatures.add(sig)
        return True
    return False


def _random_component_candidate(
    grid: np.ndarray,
    protected_center: tuple[int, int],
    used_signatures: set[tuple[tuple[int, int], ...]],
) -> list[tuple[int, int]] | None:
    hr, hc = protected_center
    pool = SHAPE_POOL[:]
    random.shuffle(pool)
    for base in pool:
        oriented_choices = SHAPE_ORIENTATIONS[base][:]
        random.shuffle(oriented_choices)
        for oriented in oriented_choices:
            if oriented in used_signatures:
                continue
            rows = [r for r, _ in oriented]
            cols = [c for _, c in oriented]
            height = max(rows) + 1
            width = max(cols) + 1
            if height > grid.shape[0] or width > grid.shape[1]:
                continue
            for _ in range(40):
                r0 = random.randint(0, grid.shape[0] - height)
                c0 = random.randint(0, grid.shape[1] - width)
                cells = _translate(oriented, r0, c0)
                if any(max(abs(r - hr), abs(c - hc)) <= 2 for r, c in cells):
                    continue
                return cells
    return None


def _place_extra_component(
    grid: np.ndarray,
    color: int,
    components_by_color: dict[int, list[np.ndarray]],
    used_signatures: set[tuple[tuple[int, int], ...]],
    protected_center: tuple[int, int],
) -> bool:
    for _ in range(240):
        cells = _random_component_candidate(grid, protected_center, used_signatures)
        if cells is None:
            return False
        if not _placeable(cells, grid, components_by_color[color], set()):
            continue
        sig = _normalize(cells)
        grid[tuple(np.array(cells).T)] = color
        components_by_color[color].append(np.array(cells, dtype=int))
        used_signatures.add(sig)
        return True
    return False


def _winner_map(grid: np.ndarray) -> dict[tuple[int, int], int]:
    claims, areas, _geometry, _counts = _claim_data(grid)
    winners: dict[tuple[int, int], int] = {}
    cell_claimers: dict[tuple[int, int], list[int]] = defaultdict(list)
    for color, cells in claims.items():
        for cell in cells:
            cell_claimers[cell].append(color)
    for cell, colors in cell_claimers.items():
        winners[cell] = min(colors, key=lambda color: (areas[color], color))
    return winners


def _validate(grid: np.ndarray) -> bool:
    colors = sorted(int(v) for v in np.unique(grid) if int(v) != BLACK)
    if not (2 <= len(colors) <= 3):
        return False

    components = _color_components(grid)
    counts = {color: len(comps) for color, comps in components.items()}
    if len(set(counts.values())) != len(colors):
        return False
    if not any(count % 2 == 1 for count in counts.values()):
        return False
    if not any(count % 2 == 0 for count in counts.values()):
        return False

    areas = {color: int(np.count_nonzero(grid == color)) for color in colors}
    if len(set(areas.values())) != len(colors):
        return False

    seen_shapes: set[tuple[tuple[int, int], ...]] = set()
    for color, comps in components.items():
        for comp in comps:
            if not (3 <= len(comp) <= 6):
                return False
            sig = _component_shape_signature(comp)
            if sig in seen_shapes or not _shape_is_irregular(sig):
                return False
            seen_shapes.add(sig)
        for i in range(len(comps)):
            for j in range(i + 1, len(comps)):
                if _min_chebyshev_distance(comps[i], comps[j]) < 3:
                    return False

    claims, _areas, geometry, _counts = _claim_data(grid)
    for i, color_a in enumerate(colors):
        for color_b in colors[i + 1 :]:
            if not (claims[color_a] & claims[color_b]):
                return False

    out = main(grid)
    if YELLOW not in out or ORANGE not in out:
        return False

    winners = _winner_map(grid)
    if not winners:
        return False
    visible_by_color = {color: 0 for color in colors}
    for (r, c), winner in winners.items():
        if out[r, c] in (YELLOW, ORANGE):
            visible_by_color[winner] += 1
    if any(count == 0 for count in visible_by_color.values()):
        return False

    if np.array_equal(_solve_with_options(grid, reverse_priority=True), out):
        return False

    if YELLOW == ORANGE:
        return False

    smaller = min(colors, key=lambda color: (areas[color], color))
    larger = max(colors, key=lambda color: (areas[color], color))
    if geometry[smaller] == geometry[larger]:
        return False

    for color in colors:
        flipped = "ortho" if geometry[color] == "diag" else "diag"
        alt = _solve_with_options(grid, geometry_override={color: flipped})
        if np.array_equal(alt, out):
            return False

    if int(np.count_nonzero(out == YELLOW)) < 2 or int(np.count_nonzero(out == ORANGE)) < 2:
        return False

    return True


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    grid = np.zeros((h, w), dtype=int)

    num_colors = random.choice([2, 3])
    colors = random.sample(INPUT_COLORS, num_colors)
    counts = _choose_counts(num_colors)
    counts_by_color = {color: count for color, count in zip(colors, counts)}

    hub = (random.randint(4, h - 5), random.randint(4, w - 5))
    anchors = _pick_anchor_positions(hub, counts_by_color)
    if len(anchors) != num_colors:
        return None

    reserved = {hub, *anchors.values()}
    components_by_color: dict[int, list[np.ndarray]] = {color: [] for color in colors}
    used_signatures: set[tuple[tuple[int, int], ...]] = set()

    placement_order = colors[:]
    random.shuffle(placement_order)
    for color in placement_order:
        if not _place_anchor_component(
            grid,
            color,
            anchors[color],
            hub,
            components_by_color,
            used_signatures,
            reserved,
        ):
            return None

    extra_order = colors[:]
    random.shuffle(extra_order)
    for color in extra_order:
        for _ in range(counts_by_color[color] - 1):
            if not _place_extra_component(
                grid,
                color,
                components_by_color,
                used_signatures,
                hub,
            ):
                return None

    return grid if _validate(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid parity-expansion ARC input")
