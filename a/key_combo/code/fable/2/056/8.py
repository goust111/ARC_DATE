from __future__ import annotations

from collections import Counter
import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


BG = 0
PALETTE = tuple(range(1, 10))
GEN_RETRIES = 600
CUP_COUNT_OPTIONS = (2, 3, 4)
TOTAL_CAPACITY_BOUNDS = {
    2: (12, 26),
    3: (18, 34),
    4: (24, 40),
}
CAPACITY_TO_DIMS: dict[int, list[tuple[int, int]]] = {
    4: [(2, 2)],
    6: [(2, 3), (3, 2)],
    8: [(2, 4), (4, 2)],
    9: [(3, 3)],
    10: [(2, 5), (5, 2)],
    12: [(3, 4), (4, 3)],
    15: [(3, 5), (5, 3)],
    16: [(4, 4)],
}


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        dirs = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]

    comps: list[np.ndarray] = []
    for sr in range(h):
        for sc in range(w):
            if not mask[sr, sc] or seen[sr, sc]:
                continue
            stack = [(sr, sc)]
            seen[sr, sc] = True
            coords: list[Coord] = []
            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                for dr, dc in dirs:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _fill_order(cavity_bbox: BBox) -> list[Coord]:
    r0, c0, r1, c1 = cavity_bbox
    return [(r, c) for r in range(r1, r0 - 1, -1) for c in range(c0, c1 + 1)]


def _parse_cup(grid: np.ndarray, outline: int, bg: int, coords: np.ndarray) -> dict:
    r0, c0, r1, c1 = _bbox_coords(coords)
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    h, w = patch.shape
    assert h >= 3 and w >= 4, "cup must have a rectangular cavity"

    outline_mask = patch == outline
    expected = np.zeros((h, w), dtype=bool)
    expected[:, 0] = True
    expected[:, -1] = True
    expected[-1, :] = True
    assert np.array_equal(outline_mask, expected), "outline components must be open-top U cups"

    cavity = patch[:-1, 1:-1]
    assert cavity.size > 0 and bool(np.all(cavity == bg)), "cup cavity must be empty"
    cavity_bbox = (r0, c0 + 1, r1 - 1, c1 - 1)
    return {
        "bbox": (r0, c0, r1, c1),
        "cavity_bbox": cavity_bbox,
        "capacity": int(cavity.size),
        "fill_order": _fill_order(cavity_bbox),
        "center": ((cavity_bbox[0] + cavity_bbox[2]) / 2.0, (cavity_bbox[1] + cavity_bbox[3]) / 2.0),
        "cavity_wh": (int(cavity.shape[0]), int(cavity.shape[1])),
    }


def _extract_cups(grid: np.ndarray, bg: int | None = None) -> tuple[int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g) if bg is None else int(bg)
    colors = sorted(int(c) for c in np.unique(g) if int(c) != bg)

    candidates = []
    for color in colors:
        comps = _components(g == color, connectivity=4)
        if any(comp.shape[0] > 1 for comp in comps):
            candidates.append(color)
    assert len(candidates) == 1, "expected one repeated outline color"
    outline = candidates[0]

    cups = [_parse_cup(g, outline, bg, comp) for comp in _components(g == outline, connectivity=4)]
    capacities = [cup["capacity"] for cup in cups]
    assert len(capacities) >= 2, "need multiple cups"
    assert len(capacities) == len(set(capacities)), "cup capacities must be distinct"
    cups.sort(key=lambda cup: (cup["bbox"][0], cup["bbox"][1]))
    return outline, cups


def _drop_color_counts(grid: np.ndarray, bg: int, outline: int) -> dict[int, int]:
    g = np.asarray(grid, dtype=int)
    counts: dict[int, int] = {}
    for color in sorted(int(c) for c in np.unique(g) if int(c) not in {bg, outline}):
        comps = _components(g == color, connectivity=8)
        assert comps and all(comp.shape[0] == 1 for comp in comps), "drops must stay as isolated single cells"
        counts[color] = int(np.count_nonzero(g == color))
    return counts


def _candidate_pairs(color_counts: dict[int, int], capacities: list[int]) -> dict[int, list[tuple[int, int]]]:
    colors = sorted(color_counts)
    pairs: dict[int, list[tuple[int, int]]] = {int(cap): [] for cap in capacities}
    for i, a in enumerate(colors):
        for b in colors[i + 1 :]:
            s = color_counts[a] + color_counts[b]
            if s in pairs:
                pairs[s].append((a, b))
    return pairs


def _enumerate_pairings(
    color_counts: dict[int, int], capacities: list[int], limit: int = 2
) -> tuple[list[dict[int, tuple[int, int]]], dict[int, list[tuple[int, int]]]]:
    capacities = [int(cap) for cap in capacities]
    pair_options = _candidate_pairs(color_counts, capacities)
    solutions: list[dict[int, tuple[int, int]]] = []

    def backtrack(remaining_caps: list[int], unused: set[int], chosen: dict[int, tuple[int, int]]) -> None:
        if len(solutions) >= limit:
            return
        if not remaining_caps:
            solutions.append(dict(chosen))
            return

        best_cap = min(
            remaining_caps,
            key=lambda cap: sum(1 for a, b in pair_options[cap] if a in unused and b in unused),
        )
        viable = [(a, b) for a, b in pair_options[best_cap] if a in unused and b in unused]
        if not viable:
            return

        next_caps = [cap for cap in remaining_caps if cap != best_cap]
        for a, b in viable:
            chosen[best_cap] = (a, b)
            backtrack(next_caps, unused - {a, b}, chosen)
            del chosen[best_cap]

    backtrack(list(capacities), set(color_counts), {})
    return solutions, pair_options


def _solve_pairing(
    color_counts: dict[int, int], capacities: list[int]
) -> tuple[dict[int, tuple[int, int]], dict[int, list[tuple[int, int]]]]:
    solutions, pair_options = _enumerate_pairings(color_counts, capacities, limit=2)
    assert len(solutions) == 1, "drop-color pairing must be globally unique"
    return solutions[0], pair_options


def main(grid: np.ndarray) -> np.ndarray:
    """Pair scattered droplet colors into cups by exact summed capacity, then layer-fill each cup.

    Input structure:
    - A black background contains several same-outline-color open-top cups.
    - Each cup is a thin U outline whose empty rectangular cavity has a distinct capacity.
    - Elsewhere, every non-outline foreground pixel is a single-cell droplet; there are exactly two droplet colors per cup.

    Rule:
    - Count each droplet color and find the unique perfect pairing whose two-color sums match the cup capacities.
    - For each cup, pour its matched colors into the cavity in bottom-up, left-to-right cell order.
    - The color with more droplets sinks first and occupies the earlier cells; the smaller color fills the remaining cells. All original droplets are erased.

    Key insight:
    - Spatial proximity of droplets is irrelevant: only the globally unique subset-sum pairing of color counts identifies which two colors belong in each cup, and the visible boundary inside a cup follows exact cell-order counting rather than row-only layering.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    outline, cups = _extract_cups(g, bg)
    color_counts = _drop_color_counts(g, bg, outline)
    assert len(color_counts) == 2 * len(cups), "must have exactly two droplet colors per cup"

    capacities = [cup["capacity"] for cup in cups]
    pairing, _ = _solve_pairing(color_counts, capacities)

    out = g.copy()
    for color in color_counts:
        out[out == color] = bg

    for cup in cups:
        a, b = pairing[cup["capacity"]]
        ca, cb = color_counts[a], color_counts[b]
        assert ca + cb == cup["capacity"] and ca != cb
        major, minor = (a, b) if ca > cb else (b, a)
        split = color_counts[major]
        for idx, (r, c) in enumerate(cup["fill_order"]):
            out[r, c] = major if idx < split else minor

    return out


def _build_cup_patch(outline: int, cavity_h: int, cavity_w: int) -> np.ndarray:
    patch = np.zeros((cavity_h + 1, cavity_w + 2), dtype=int)
    patch[:, 0] = outline
    patch[:, -1] = outline
    patch[-1, :] = outline
    return patch


def _random_derangement(n: int) -> list[int]:
    order = list(range(n))
    while True:
        random.shuffle(order)
        if all(i != order[i] for i in range(n)):
            return order


def _random_composition(total: int, parts: int) -> list[int]:
    vals = [0] * parts
    for _ in range(total):
        vals[random.randrange(parts)] += 1
    return vals


def _capacity_splits(capacity: int) -> list[tuple[int, int]]:
    return [(a, capacity - a) for a in range(1, capacity) if a != capacity - a]


def _choose_capacities_and_pairs(n_cups: int) -> tuple[list[int], dict[int, tuple[int, int]], dict[int, list[tuple[int, int]]]] | None:
    caps = list(CAPACITY_TO_DIMS)
    lo, hi = TOTAL_CAPACITY_BOUNDS[n_cups]
    for _ in range(300):
        capacities = sorted(random.sample(caps, n_cups))
        total_capacity = sum(capacities)
        if not (lo <= total_capacity <= hi):
            continue

        pair_by_cap = {cap: random.choice(_capacity_splits(cap)) for cap in capacities}
        fake_counts = {i + 1: v for i, v in enumerate([n for cap in capacities for n in pair_by_cap[cap]])}
        solutions, pair_options = _enumerate_pairings(fake_counts, capacities, limit=2)
        if len(solutions) != 1:
            continue
        if not any(len(pair_options[cap]) > 1 for cap in capacities):
            continue
        return capacities, pair_by_cap, pair_options
    return None


def _choose_dims(
    capacities: list[int], pair_by_cap: dict[int, tuple[int, int]]
) -> dict[int, tuple[int, int]] | None:
    for _ in range(300):
        dims = {cap: random.choice(CAPACITY_TO_DIMS[cap]) for cap in capacities}
        outer_widths = [dims[cap][1] + 2 for cap in capacities]
        needed_width = sum(outer_widths) + (len(capacities) - 1) + 2
        if needed_width > 30:
            continue
        if not any(max(pair_by_cap[cap]) % dims[cap][1] != 0 for cap in capacities):
            continue
        return dims
    return None


def _place_cups(
    outline: int,
    capacities: list[int],
    dims_by_cap: dict[int, tuple[int, int]],
    pair_colors_by_cap: dict[int, tuple[int, int]],
) -> tuple[np.ndarray, list[dict]] | None:
    specs = []
    for cap in capacities:
        cavity_h, cavity_w = dims_by_cap[cap]
        patch = _build_cup_patch(outline, cavity_h, cavity_w)
        specs.append(
            {
                "capacity": cap,
                "dims": (cavity_h, cavity_w),
                "outer_shape": patch.shape,
                "patch": patch,
                "pair_colors": pair_colors_by_cap[cap],
            }
        )
    random.shuffle(specs)

    outer_widths = [spec["outer_shape"][1] for spec in specs]
    needed_width = sum(outer_widths) + (len(specs) - 1) + 2
    max_outer_h = max(spec["outer_shape"][0] for spec in specs)
    h_min = max(20, max_outer_h + 11)
    if h_min > 30:
        return None

    grid_h = random.randint(h_min, 30)
    grid_w = random.randint(max(18, needed_width), 30)
    extras = _random_composition(grid_w - needed_width, len(specs) + 1)
    margins = [1 + x for x in extras]
    baseline = grid_h - 2

    grid = np.zeros((grid_h, grid_w), dtype=int)
    cups: list[dict] = []
    c = margins[0]
    for idx, spec in enumerate(specs):
        outer_h, outer_w = spec["outer_shape"]
        r0 = baseline - outer_h + 1
        patch = spec["patch"]
        grid[r0 : r0 + outer_h, c : c + outer_w] = np.maximum(grid[r0 : r0 + outer_h, c : c + outer_w], patch)
        cavity_bbox = (r0, c + 1, r0 + outer_h - 2, c + outer_w - 2)
        cups.append(
            {
                **spec,
                "bbox": (r0, c, r0 + outer_h - 1, c + outer_w - 1),
                "cavity_bbox": cavity_bbox,
                "fill_order": _fill_order(cavity_bbox),
                "center": ((cavity_bbox[0] + cavity_bbox[2]) / 2.0, (cavity_bbox[1] + cavity_bbox[3]) / 2.0),
            }
        )
        c += outer_w
        if idx < len(specs) - 1:
            c += 1 + extras[idx + 1]
    return grid, cups


def _pair_centroid(assignments: dict[int, list[Coord]], colors: tuple[int, int]) -> tuple[float, float]:
    pts = assignments[colors[0]] + assignments[colors[1]]
    rs = [p[0] for p in pts]
    cs = [p[1] for p in pts]
    return (sum(rs) / len(rs), sum(cs) / len(cs))


def _nearest_center(point: tuple[float, float], centers: list[tuple[float, float]]) -> int:
    pr, pc = point
    return min(range(len(centers)), key=lambda i: abs(pr - centers[i][0]) + abs(pc - centers[i][1]))


def _drop_candidates(grid_h: int, grid_w: int, top_stop: int, phase_r: int, phase_c: int) -> list[Coord]:
    return [
        (r, c)
        for r in range(1, top_stop + 1)
        for c in range(1, grid_w - 1)
        if (r - phase_r) % 2 == 0 and (c - phase_c) % 2 == 0
    ]


def _choose_drop_positions(
    available: list[Coord],
    count: int,
    correct_idx: int,
    decoy_idx: int,
    centers: list[tuple[float, float]],
) -> list[Coord] | None:
    scored = []
    for pos in available:
        nearest = _nearest_center(pos, centers)
        priority = 0 if nearest == decoy_idx else 1 if nearest != correct_idx else 2
        dr = abs(pos[0] - centers[decoy_idx][0]) + abs(pos[1] - centers[decoy_idx][1])
        scored.append((priority, dr, random.random(), pos))
    scored.sort()
    chosen = [pos for *_rest, pos in scored[:count]]
    if len(chosen) < count:
        return None
    return chosen


def _build_expected_output(
    grid: np.ndarray, cups: list[dict], color_counts: dict[int, int]
) -> np.ndarray:
    out = grid.copy()
    used_colors = set(color_counts)
    for color in used_colors:
        out[out == color] = BG
    for cup in cups:
        a, b = cup["pair_colors"]
        ca, cb = color_counts[a], color_counts[b]
        major, minor = (a, b) if ca > cb else (b, a)
        split = color_counts[major]
        for idx, (r, c) in enumerate(cup["fill_order"]):
            out[r, c] = major if idx < split else minor
    return out


def _semantic_checks(
    grid: np.ndarray,
    cups: list[dict],
    color_counts: dict[int, int],
    assignments: dict[int, list[Coord]],
) -> bool:
    outline, parsed_cups = _extract_cups(grid, BG)
    if outline == BG or len(parsed_cups) != len(cups):
        return False

    parsed_caps = sorted(cup["capacity"] for cup in parsed_cups)
    true_caps = sorted(cup["capacity"] for cup in cups)
    if parsed_caps != true_caps:
        return False

    parsed_counts = _drop_color_counts(grid, BG, outline)
    if parsed_counts != color_counts:
        return False

    solutions, pair_options = _enumerate_pairings(color_counts, true_caps, limit=2)
    if len(solutions) != 1:
        return False
    if not any(len(pair_options[cap]) > 1 for cap in true_caps):
        return False

    if not any(max(color_counts[a], color_counts[b]) % cup["dims"][1] != 0 for cup in cups for a, b in [cup["pair_colors"]]):
        return False

    centers = [cup["center"] for cup in cups]
    indexed = {cup["capacity"]: i for i, cup in enumerate(cups)}
    if not any(
        _nearest_center(_pair_centroid(assignments, cup["pair_colors"]), centers) != indexed[cup["capacity"]]
        for cup in cups
    ):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    n_cups = random.choice(CUP_COUNT_OPTIONS)
    outline = random.choice(PALETTE)
    drop_palette = [color for color in PALETTE if color != outline]
    if len(drop_palette) < 2 * n_cups:
        return None

    cap_info = _choose_capacities_and_pairs(n_cups)
    if cap_info is None:
        return None
    capacities, pair_by_cap, _pair_options = cap_info

    dims_by_cap = _choose_dims(capacities, pair_by_cap)
    if dims_by_cap is None:
        return None

    chosen_colors = random.sample(drop_palette, 2 * n_cups)
    random.shuffle(chosen_colors)
    pair_colors_by_cap: dict[int, tuple[int, int]] = {}
    color_counts: dict[int, int] = {}
    cursor = 0
    for cap in capacities:
        a, b = chosen_colors[cursor], chosen_colors[cursor + 1]
        cursor += 2
        pair_colors_by_cap[cap] = (a, b)
        count_a, count_b = pair_by_cap[cap]
        color_counts[a] = int(count_a)
        color_counts[b] = int(count_b)

    placed = _place_cups(outline, capacities, dims_by_cap, pair_colors_by_cap)
    if placed is None:
        return None
    grid, cups = placed

    top_stop = min(cup["bbox"][0] for cup in cups) - 2
    if top_stop < 8:
        return None

    phase_r = random.randint(0, 1)
    phase_c = random.randint(0, 1)
    available = _drop_candidates(grid.shape[0], grid.shape[1], top_stop, phase_r, phase_c)
    if len(available) < sum(color_counts.values()):
        return None

    centers = [cup["center"] for cup in cups]
    decoys = _random_derangement(len(cups))
    assignments: dict[int, list[Coord]] = {}

    cup_order = sorted(range(len(cups)), key=lambda i: sum(color_counts[c] for c in cups[i]["pair_colors"]), reverse=True)
    for cup_idx in cup_order:
        colors = sorted(cups[cup_idx]["pair_colors"], key=lambda color: color_counts[color], reverse=True)
        for color in colors:
            positions = _choose_drop_positions(
                available,
                color_counts[color],
                correct_idx=cup_idx,
                decoy_idx=decoys[cup_idx],
                centers=centers,
            )
            if positions is None:
                return None
            assignments[color] = positions
            used = set(positions)
            available = [pos for pos in available if pos not in used]

    for color, coords in assignments.items():
        for r, c in coords:
            grid[r, c] = color

    expected = _build_expected_output(grid, cups, color_counts)
    if not np.array_equal(main(grid), expected):
        return None
    if not _semantic_checks(grid, cups, color_counts, assignments):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cocktail puzzle input")
