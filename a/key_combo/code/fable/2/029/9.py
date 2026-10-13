from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
GEN_ATTEMPTS = 500
GEN_COMPACT_MIN = 4
GEN_COMPACT_MAX = 9
GEN_REGION_MIN = 4
GEN_REGION_MAX = 8
GEN_PAIR_MIN = 3
GEN_PAIR_MAX = 7
GEN_NOISE_MIN = 3
GEN_NOISE_MAX = 8


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _pair_details(
    grid: np.ndarray,
    bg: int,
    *,
    allow_conflicts: bool = False,
) -> dict[tuple[int, int], dict[str, object]] | None:
    h, w = map(int, grid.shape)
    assert h % 2 == 0 and w % 2 == 0, "grid must have even height and even width"

    details: dict[tuple[int, int], dict[str, object]] = {}
    for rr in range(0, h, 2):
        for cc in range(0, w, 2):
            hits: list[tuple[str, int]] = []

            if rr > 0:
                color = int(grid[rr - 1, cc])
                if color != bg and color == int(grid[rr + 1, cc]):
                    hits.append(("V", color))

            if cc > 0:
                color = int(grid[rr, cc - 1])
                if color != bg and color == int(grid[rr, cc + 1]):
                    hits.append(("H", color))

            if not hits:
                continue

            colors = {color for _orient, color in hits}
            if len(colors) != 1:
                if allow_conflicts:
                    return None
                raise AssertionError("one picture cell cannot receive conflicting paired colors")
            details[(rr // 2, cc // 2)] = {
                "color": hits[0][1],
                "orients": frozenset(orient for orient, _color in hits),
            }

    return details


def _pair_signature(
    grid: np.ndarray,
    bg: int,
    *,
    allow_conflicts: bool = False,
) -> frozenset[tuple[int, int, int, tuple[str, ...]]] | None:
    details = _pair_details(grid, bg, allow_conflicts=allow_conflicts)
    if details is None:
        return None
    return frozenset(
        (
            int(r),
            int(c),
            int(info["color"]),
            tuple(sorted(str(orient) for orient in info["orients"])),
        )
        for (r, c), info in details.items()
    )


def _separator_coords_for_cell(cell: tuple[int, int], orient: str) -> tuple[tuple[int, int], tuple[int, int]]:
    r, c = cell
    rr = 2 * int(r)
    cc = 2 * int(c)
    if orient == "V":
        return (rr - 1, cc), (rr + 1, cc)
    if orient == "H":
        return (rr, cc - 1), (rr, cc + 1)
    raise ValueError(f"unknown orientation: {orient}")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the even-even picture cells, then recolor each cell flanked by a same-color separator pair.

    Input structure
    - The grid has even height and even width. Even-indexed rows and columns carry the main picture, while odd-indexed rows or columns are separator bands.
    - A separator band may contain isolated colored pixels or two same-color pixels placed exactly above and below one picture cell, or exactly left and right of one picture cell.
    - The separator-intersection cells are also part of the separator area and may contain irrelevant noise.

    Rule
    - Keep only the even-even cells to obtain the half-size output picture.
    - For every picture cell whose separator neighbors form a same-color vertical pair or horizontal pair, recolor that output cell to the pair color.
    - Ignore every separator pixel that is not part of such a flanking pair.

    Key insight
    - The separator lattice is an instruction channel: only two matching pixels that face each other across one picture cell are meaningful, while lone separator marks are pure distractions.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _get_bg(g)
    out = g[::2, ::2].copy()
    for (r, c), info in _pair_details(g, bg).items():
        out[r, c] = int(info["color"])
    return out


def _randint(lo: int, hi: int) -> int:
    return int(random.randint(int(lo), int(hi)))


def _grow_labels(h: int, w: int, n_regions: int) -> np.ndarray:
    labels = np.full((h, w), -1, dtype=int)
    cells = [(r, c) for r in range(h) for c in range(w)]
    random.shuffle(cells)
    for label_idx, (r, c) in enumerate(cells[:n_regions]):
        labels[r, c] = label_idx

    while np.any(labels < 0):
        frontier: list[tuple[int, int]] = []
        for r in range(h):
            for c in range(w):
                if labels[r, c] >= 0:
                    continue
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    rr = r + dr
                    cc = c + dc
                    if 0 <= rr < h and 0 <= cc < w and labels[rr, cc] >= 0:
                        frontier.append((r, c))
                        break

        r, c = random.choice(frontier)
        neighbor_labels = []
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w and labels[rr, cc] >= 0:
                neighbor_labels.append(int(labels[rr, cc]))
        labels[r, c] = random.choice(neighbor_labels)

    return labels


def _build_compact_picture(h: int, w: int) -> np.ndarray:
    n_regions = _randint(GEN_REGION_MIN, min(GEN_REGION_MAX, h * w))
    labels = _grow_labels(h, w, n_regions)

    n_colors = _randint(3, min(5, len(PALETTE), n_regions))
    palette = random.sample(PALETTE, n_colors)
    region_colors = palette.copy()
    while len(region_colors) < n_regions:
        region_colors.append(random.choice(palette))
    random.shuffle(region_colors)

    compact = np.zeros((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            compact[r, c] = int(region_colors[int(labels[r, c])])

    assert len(set(map(int, compact.ravel()))) >= 3
    return compact


def _all_candidates(h: int, w: int) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            if r > 0:
                out.append({"cell": (r, c), "orient": "V", "seps": _separator_coords_for_cell((r, c), "V")})
            if c > 0:
                out.append({"cell": (r, c), "orient": "H", "seps": _separator_coords_for_cell((r, c), "H")})
    return out


def _pick_candidate(
    pool: list[dict[str, object]],
    used_sep: set[tuple[int, int]],
    targeted_cells: set[tuple[int, int]],
) -> dict[str, object] | None:
    options = [
        cand
        for cand in pool
        if tuple(cand["cell"]) not in targeted_cells
        and all(tuple(pos) not in used_sep for pos in cand["seps"])
    ]
    if not options:
        return None
    return random.choice(options)


def _target_pairs(compact: np.ndarray) -> list[dict[str, object]] | None:
    h, w = compact.shape
    all_cands = _all_candidates(h, w)
    vertical_edge = [cand for cand in all_cands if cand["orient"] == "V" and int(cand["cell"][1]) in {0, w - 1}]
    horizontal_edge = [cand for cand in all_cands if cand["orient"] == "H" and int(cand["cell"][0]) in {0, h - 1}]

    used_sep: set[tuple[int, int]] = set()
    targeted_cells: set[tuple[int, int]] = set()
    chosen: list[dict[str, object]] = []

    for pool in (vertical_edge, horizontal_edge):
        cand = _pick_candidate(pool, used_sep, targeted_cells)
        if cand is None:
            return None
        chosen.append(cand)
        targeted_cells.add(tuple(cand["cell"]))
        used_sep.update(tuple(pos) for pos in cand["seps"])

    target_total = _randint(GEN_PAIR_MIN, min(GEN_PAIR_MAX, len(all_cands)))
    while len(chosen) < target_total:
        cand = _pick_candidate(all_cands, used_sep, targeted_cells)
        if cand is None:
            break
        chosen.append(cand)
        targeted_cells.add(tuple(cand["cell"]))
        used_sep.update(tuple(pos) for pos in cand["seps"])

    if len(chosen) < GEN_PAIR_MIN:
        return None

    random.shuffle(chosen)
    pair_palette = random.sample(PALETTE, _randint(2, min(4, len(PALETTE), len(chosen))))
    used_pair_colors: set[int] = set()
    for idx, cand in enumerate(chosen):
        r, c = map(int, cand["cell"])
        preferred = pair_palette[idx % len(pair_palette)]
        if preferred != int(compact[r, c]):
            color = preferred
        else:
            choices = [col for col in pair_palette if col != int(compact[r, c])]
            if not choices:
                choices = [col for col in PALETTE if col != int(compact[r, c])]
            color = random.choice(choices)
        cand["color"] = int(color)
        used_pair_colors.add(int(color))

    if len(used_pair_colors) < 2:
        return None
    return chosen


def _separator_positions(shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = map(int, shape)
    return [(r, c) for r in range(h) for c in range(w) if not (r % 2 == 0 and c % 2 == 0)]


def _try_place_noise(
    grid: np.ndarray,
    bg: int,
    base_signature: frozenset[tuple[int, int, int, tuple[str, ...]]],
    coords: list[tuple[int, int]],
    preferred_colors: list[int] | None = None,
) -> tuple[int, int] | None:
    coord_order = coords[:]
    random.shuffle(coord_order)
    colors = preferred_colors[:] if preferred_colors else list(PALETTE)
    random.shuffle(colors)

    for r, c in coord_order:
        if int(grid[r, c]) != bg:
            continue
        color_order = colors[:]
        random.shuffle(color_order)
        for color in color_order:
            grid[r, c] = int(color)
            signature = _pair_signature(grid, bg, allow_conflicts=True)
            if signature is not None and signature == base_signature:
                return int(r), int(c)
            grid[r, c] = bg
    return None


def _build_input_from_compact(compact: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]], list[tuple[int, int]]] | None:
    h, w = compact.shape
    grid = np.full((2 * h, 2 * w), BG, dtype=int)
    grid[::2, ::2] = compact

    pair_specs = _target_pairs(compact)
    if pair_specs is None:
        return None

    expected = compact.copy()
    for spec in pair_specs:
        cell = tuple(spec["cell"])
        color = int(spec["color"])
        sep_a, sep_b = spec["seps"]
        grid[sep_a] = color
        grid[sep_b] = color
        expected[cell] = color

    base_signature = _pair_signature(grid, BG)
    if len(base_signature) != len(pair_specs):
        return None

    sep_positions = _separator_positions(grid.shape)
    cross_positions = [(r, c) for r, c in sep_positions if r % 2 == 1 and c % 2 == 1]
    adjacent_positions = [(r, c) for r, c in sep_positions if (r + c) % 2 == 1]

    pair_colors = [int(spec["color"]) for spec in pair_specs]
    noise_cells: list[tuple[int, int]] = []
    first_noise = _try_place_noise(grid, BG, base_signature, cross_positions, preferred_colors=pair_colors)
    if first_noise is None:
        return None
    noise_cells.append(first_noise)

    second_noise = _try_place_noise(grid, BG, base_signature, adjacent_positions, preferred_colors=pair_colors)
    if second_noise is None:
        return None
    noise_cells.append(second_noise)

    target_noise = _randint(GEN_NOISE_MIN, GEN_NOISE_MAX)
    while len(noise_cells) < target_noise:
        noise = _try_place_noise(grid, BG, base_signature, sep_positions, preferred_colors=pair_colors)
        if noise is None:
            break
        noise_cells.append(noise)

    if len(noise_cells) < GEN_NOISE_MIN:
        return None

    return grid, expected, pair_specs, noise_cells


def _validate_instance(
    grid: np.ndarray,
    expected: np.ndarray,
    pair_specs: list[dict[str, object]],
    noise_cells: list[tuple[int, int]],
) -> bool:
    if grid.ndim != 2 or expected.ndim != 2:
        return False
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return False
    if grid.shape[0] % 2 != 0 or grid.shape[1] % 2 != 0:
        return False

    compact = grid[::2, ::2].copy()
    produced = main(grid)
    if not np.array_equal(produced, expected):
        return False
    if np.array_equal(compact, expected):
        return False

    details = _pair_details(grid, BG)
    if len(details) != len(pair_specs) or len(details) < GEN_PAIR_MIN:
        return False

    all_orients = set()
    edge_vertical = False
    edge_horizontal = False
    diff_count = 0
    for spec in pair_specs:
        cell = tuple(spec["cell"])
        orient = str(spec["orient"])
        color = int(spec["color"])
        if cell not in details:
            return False
        info = details[cell]
        if int(info["color"]) != color or orient not in info["orients"]:
            return False
        r, c = cell
        if expected[r, c] == compact[r, c]:
            return False
        diff_count += 1
        all_orients.add(orient)
        if orient == "V" and c in {0, expected.shape[1] - 1}:
            edge_vertical = True
        if orient == "H" and r in {0, expected.shape[0] - 1}:
            edge_horizontal = True

    if diff_count != len(pair_specs):
        return False
    if all_orients != {"V", "H"}:
        return False
    if not edge_vertical or not edge_horizontal:
        return False

    if len(set(map(int, compact.ravel()))) < 3 or len(set(map(int, expected.ravel()))) < 3:
        return False

    if not any(r % 2 == 1 and c % 2 == 1 for r, c in noise_cells):
        return False
    if not any((r + c) % 2 == 1 for r, c in noise_cells):
        return False

    grid_without_noise = grid.copy()
    for r, c in noise_cells:
        grid_without_noise[r, c] = BG
    if not np.array_equal(main(grid_without_noise), expected):
        return False

    grid_without_pairs = grid.copy()
    pair_positions = []
    for spec in pair_specs:
        pair_positions.extend(spec["seps"])
    for r, c in pair_positions:
        grid_without_pairs[r, c] = BG
    if np.array_equal(main(grid_without_pairs), expected):
        return False

    first_pair = pair_specs[0]
    sep_a, _sep_b = first_pair["seps"]
    probe = grid.copy()
    probe[sep_a] = BG
    probe_out = main(probe)
    probe_r, probe_c = first_pair["cell"]
    if int(probe_out[probe_r, probe_c]) != int(compact[probe_r, probe_c]):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        h = _randint(GEN_COMPACT_MIN, GEN_COMPACT_MAX)
        w = _randint(GEN_COMPACT_MIN, GEN_COMPACT_MAX)
        compact = _build_compact_picture(h, w)
        built = _build_input_from_compact(compact)
        if built is None:
            continue
        grid, expected, pair_specs, noise_cells = built
        if _validate_instance(grid, expected, pair_specs, noise_cells):
            return grid
    raise RuntimeError("failed to generate a valid input for ARC puzzle 9")
