import random
from itertools import combinations

import numpy as np


BG = 0
MIRROR = 5
SUPPORT = 9
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8]

CARDINALS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
SOURCE_DIRS = {
    "top": (1, 0),
    "bottom": (-1, 0),
    "left": (0, 1),
    "right": (0, -1),
}
REFLECT = {
    "/": {
        (1, 0): (0, 1),
        (0, -1): (-1, 0),
        (-1, 0): (0, -1),
        (0, 1): (1, 0),
    },
    "\\": {
        (1, 0): (0, -1),
        (0, 1): (-1, 0),
        (-1, 0): (0, 1),
        (0, -1): (1, 0),
    },
}


def _line_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r1, c1 = a
    r2, c2 = b
    if r1 == r2:
        step = 1 if c2 >= c1 else -1
        return [(r1, c) for c in range(c1, c2 + step, step)]
    if c1 == c2:
        step = 1 if r2 >= r1 else -1
        return [(r, c1) for r in range(r1, r2 + step, step)]
    raise ValueError("segment must be axis-aligned")


def _parse_legend(grid: np.ndarray) -> tuple[dict[frozenset[int], int], np.ndarray]:
    h, w = grid.shape
    candidates: list[tuple[int, dict[frozenset[int], int], np.ndarray]] = []

    for top in (True, False):
        pair_row = 0 if top else h - 2
        result_row = pair_row + 1
        for left in (True, False):
            groups: list[tuple[int, int, int, int]] = []
            used: set[tuple[int, int]] = set()
            j = 0
            while True:
                anchor = 3 * j if left else w - 2 - 3 * j
                if anchor < 0 or anchor + 1 >= w:
                    break
                cells = [(pair_row, anchor), (pair_row, anchor + 1), (result_row, anchor)]
                if any(int(grid[r, c]) in (BG, MIRROR, SUPPORT) for r, c in cells):
                    break
                groups.append(
                    (
                        anchor,
                        int(grid[pair_row, anchor]),
                        int(grid[pair_row, anchor + 1]),
                        int(grid[result_row, anchor]),
                    )
                )
                used.update(cells)
                j += 1

            if not groups:
                continue

            width = 3 * len(groups) - 1
            c0 = 0 if left else w - width
            block = grid[pair_row : result_row + 1, c0 : c0 + width]
            block_used = np.zeros_like(block, dtype=bool)
            for r, c in used:
                block_used[r - pair_row, c - c0] = True

            if np.any((block != BG) & ~block_used):
                continue

            mapping: dict[frozenset[int], int] | None = {}
            for _anchor, a, b, c in groups:
                key = frozenset((a, b))
                if a == b or key in mapping:
                    mapping = None
                    break
                mapping[key] = c

            if mapping:
                legend_mask = np.zeros_like(grid, dtype=bool)
                legend_mask[pair_row : result_row + 1, c0 : c0 + width] = block_used
                candidates.append((len(groups), mapping, legend_mask))

    assert candidates, "legend not found"
    candidates.sort(key=lambda item: item[0], reverse=True)
    best_count = candidates[0][0]
    best = [item for item in candidates if item[0] == best_count]
    assert len(best) == 1, "legend must be unique"
    return best[0][1], best[0][2]


def _mirror_map(grid: np.ndarray) -> dict[tuple[int, int], str]:
    h, w = grid.shape
    mirrors: dict[tuple[int, int], str] = {}
    for r, c in zip(*np.where(grid == MIRROR)):
        bg_dirs: set[str] = set()
        for name, (dr, dc) in (
            ("U", (-1, 0)),
            ("R", (0, 1)),
            ("D", (1, 0)),
            ("L", (0, -1)),
        ):
            rr, cc = int(r + dr), int(c + dc)
            if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) == BG:
                bg_dirs.add(name)
        assert len(bg_dirs) == 2, "mirror must expose exactly two bg neighbors"
        if bg_dirs in ({"U", "R"}, {"D", "L"}):
            mirrors[(int(r), int(c))] = "/"
        elif bg_dirs in ({"U", "L"}, {"D", "R"}):
            mirrors[(int(r), int(c))] = "\\"
        else:
            raise AssertionError(f"invalid mirror encoding: {bg_dirs}")
    assert mirrors, "expected at least one mirror"
    return mirrors


def _collect_sources(grid: np.ndarray, legend_mask: np.ndarray) -> list[tuple[int, int, int, tuple[int, int]]]:
    h, w = grid.shape
    seen: set[tuple[int, int]] = set()
    sources: list[tuple[int, int, int, tuple[int, int]]] = []

    edge_positions: list[tuple[int, int, str]] = []
    edge_positions.extend((0, c, "top") for c in range(w))
    edge_positions.extend((h - 1, c, "bottom") for c in range(w))
    edge_positions.extend((r, 0, "left") for r in range(h))
    edge_positions.extend((r, w - 1, "right") for r in range(h))

    for r, c, side in edge_positions:
        if (r, c) in seen:
            continue
        seen.add((r, c))
        if legend_mask[r, c]:
            continue
        value = int(grid[r, c])
        if value in (BG, MIRROR, SUPPORT):
            continue
        assert not (r in (0, h - 1) and c in (0, w - 1)), "corner sources are ambiguous"
        sources.append((r, c, value, SOURCE_DIRS[side]))

    assert len(sources) >= 2, "expected multiple edge sources"
    return sorted(sources)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace colored rays from edge sources, reflect them on grey mirrors, and
    mix colors at intersections using the corner legend.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 4 and g.shape[1] >= 4

    mix_map, legend_mask = _parse_legend(g)
    mirrors = _mirror_map(g)
    sources = _collect_sources(g, legend_mask)

    out = g.copy()
    painted: dict[tuple[int, int], int] = {}
    h, w = g.shape

    for sr, sc, color, direction in sources:
        r, c = sr, sc
        dr, dc = direction
        seen: set[tuple[int, int, int, int]] = set()

        for _ in range(h * w * 4):
            state = (r, c, dr, dc)
            if state in seen:
                break
            seen.add(state)

            nr, nc = int(r + dr), int(c + dc)
            if not (0 <= nr < h and 0 <= nc < w):
                break

            if (nr, nc) in mirrors:
                r, c = nr, nc
                dr, dc = REFLECT[mirrors[(nr, nc)]][(dr, dc)]
                continue

            if int(g[nr, nc]) != BG:
                break

            r, c = nr, nc
            previous = painted.get((r, c))
            if previous is None or previous == color:
                painted[(r, c)] = color
            else:
                key = frozenset((previous, color))
                assert key in mix_map, f"missing mix entry for colors {previous} and {color}"
                painted[(r, c)] = mix_map[key]
        else:
            raise AssertionError("ray tracing exceeded step budget")

    for (r, c), color in painted.items():
        out[r, c] = color
    return out


def _place_mirror(
    grid: np.ndarray,
    pos: tuple[int, int],
    incoming: tuple[int, int],
    outgoing: tuple[int, int],
    blocked: set[tuple[int, int]],
) -> None:
    h, w = grid.shape
    r, c = pos
    assert int(grid[r, c]) == BG
    grid[r, c] = MIRROR

    entry = (r - incoming[0], c - incoming[1])
    exit_ = (r + outgoing[0], c + outgoing[1])
    assert 0 <= entry[0] < h and 0 <= entry[1] < w and int(grid[entry]) == BG
    assert 0 <= exit_[0] < h and 0 <= exit_[1] < w and int(grid[exit_]) == BG

    for dr, dc in CARDINALS:
        nr, nc = r + dr, c + dc
        if (nr, nc) in (entry, exit_):
            continue
        assert 0 <= nr < h and 0 <= nc < w
        assert (nr, nc) not in blocked, "mirror support would block a ray path"
        assert int(grid[nr, nc]) == BG
        grid[nr, nc] = SUPPORT


def _legend_rect(shape: tuple[int, int], entry_count: int, top: bool, left: bool) -> tuple[int, int, int]:
    h, w = shape
    width = 3 * entry_count - 1
    r0 = 0 if top else h - 2
    c0 = 0 if left else w - width
    return r0, c0, width


def _place_legend(
    grid: np.ndarray,
    entries: list[tuple[tuple[int, int], int]],
    top: bool,
    left: bool,
) -> None:
    r0, c0, width = _legend_rect(grid.shape, len(entries), top, left)
    assert c0 >= 0 and np.all(grid[r0 : r0 + 2, c0 : c0 + width] == BG)

    for j, ((a, b), c) in enumerate(entries):
        anchor = c0 + 3 * j
        grid[r0, anchor] = a
        grid[r0, anchor + 1] = b
        grid[r0 + 1, anchor] = c


def _gen_attempt() -> np.ndarray | None:
    n_sources = random.choice([2, 3])
    h = random.randint(12, 18)
    w = random.randint(16 if n_sources == 3 else 12, 22)

    need_cols = 4 if n_sources == 3 else 3
    cols = sorted(random.sample(range(2, w - 2), need_cols))
    rows = sorted(random.sample(range(2, h - 2), 3))
    if any(b - a < 2 for a, b in zip(cols, cols[1:])):
        return None
    if any(b - a < 2 for a, b in zip(rows, rows[1:])):
        return None

    c_a, c_b_turn, c_mix = cols[:3]
    c_c = cols[3] if n_sources == 3 else None
    r_a_turn, r_mix, r_b_turn = rows

    travel_cells: list[tuple[int, int]] = []
    travel_cells += _line_cells((0, c_a), (r_a_turn, c_a))[1:]
    travel_cells += _line_cells((r_a_turn, c_a), (r_a_turn, c_mix))[1:]
    travel_cells += _line_cells((r_a_turn, c_mix), (h - 1, c_mix))[1:]

    travel_cells += _line_cells((r_b_turn, 0), (r_b_turn, c_b_turn))[1:]
    travel_cells += _line_cells((r_b_turn, c_b_turn), (r_mix, c_b_turn))[1:]
    travel_cells += _line_cells((r_mix, c_b_turn), (r_mix, w - 1))[1:]

    if n_sources == 3:
        assert c_c is not None
        travel_cells += _line_cells((h - 1, c_c), (0, c_c))[1:]

    mirror_specs = [
        ((r_a_turn, c_a), (1, 0), (0, 1)),
        ((r_a_turn, c_mix), (0, 1), (1, 0)),
        ((r_b_turn, c_b_turn), (0, 1), (-1, 0)),
        ((r_mix, c_b_turn), (-1, 0), (0, 1)),
    ]
    mirror_cells = {pos for pos, _, _ in mirror_specs}
    allowed_overlap = {(r_mix, c_mix)} | ({(r_mix, c_c)} if n_sources == 3 else set())

    counts: dict[tuple[int, int], int] = {}
    for cell in travel_cells:
        counts[cell] = counts.get(cell, 0) + 1
    for cell, count in counts.items():
        if cell in mirror_cells and count != 1:
            return None
        if cell not in mirror_cells and count > 1 and cell not in allowed_overlap:
            return None

    grid = np.full((h, w), BG, dtype=int)
    blocked = set(travel_cells) | mirror_cells | {(0, c_a), (r_b_turn, 0)}
    if n_sources == 3:
        blocked.add((h - 1, c_c))

    for pos, incoming, outgoing in mirror_specs:
        r, c = pos
        entry = (r - incoming[0], c - incoming[1])
        exit_ = (r + outgoing[0], c + outgoing[1])
        for dr, dc in CARDINALS:
            cell = (r + dr, c + dc)
            if cell in (entry, exit_):
                continue
            if cell in blocked:
                return None

    source_colors = random.sample(COLOR_POOL, n_sources)
    result_pool = [c for c in COLOR_POOL if c not in source_colors]
    pair_keys = list(combinations(source_colors, 2))
    result_colors = random.sample(result_pool, len(pair_keys))
    legend_entries = list(zip(pair_keys, result_colors))
    random.shuffle(legend_entries)

    grid[0, c_a] = source_colors[0]
    grid[r_b_turn, 0] = source_colors[1]
    if n_sources == 3:
        grid[h - 1, c_c] = source_colors[2]

    for spec in mirror_specs:
        _place_mirror(grid, *spec, blocked)

    if random.choice([False, True]):
        grid = np.flipud(grid)
    if random.choice([False, True]):
        grid = np.fliplr(grid)

    corners: list[tuple[bool, bool]] = []
    for top in (True, False):
        for left in (True, False):
            r0, c0, width = _legend_rect(grid.shape, len(legend_entries), top, left)
            if c0 < 0:
                continue
            if np.all(grid[r0 : r0 + 2, c0 : c0 + width] == BG):
                corners.append((top, left))
    if not corners:
        return None

    top, left = random.choice(corners)
    _place_legend(grid, legend_entries, top, left)

    out = main(grid)
    visible_results = sum(bool(np.any(out == color)) for color in result_colors)
    if n_sources == 2 and visible_results < 1:
        return None
    if n_sources == 3 and visible_results < 2:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed after 300 attempts")
