import random
from collections import Counter

import numpy as np


DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


GEN_BG = 0
GEN_SEPARATOR = 9
GEN_COLORS = tuple(color for color in range(10) if color not in {GEN_BG, GEN_SEPARATOR})
GEN_RETRY_LIMIT = 400
GEN_LAYOUT_ATTEMPTS = 80


def _dominant_color(grid: np.ndarray) -> int:
    counts = Counter(int(value) for value in grid.flat)
    assert counts, "expected a non-empty grid"
    return counts.most_common(1)[0][0]


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _find_separator(grid: np.ndarray) -> tuple[int, int]:
    h, _ = grid.shape
    color_counts = Counter(int(value) for value in grid.flat)
    candidates = []
    for r in range(1, h - 1):
        if not np.all(grid[r] == grid[r, 0]):
            continue
        color = int(grid[r, 0])
        if np.any(grid[:r] != color) and np.any(grid[r + 1 :] != color):
            candidates.append((color_counts[color], r, color))
    assert candidates, "expected a unique full-width separator row"
    min_count = min(item[0] for item in candidates)
    best = [(r, color) for count, r, color in candidates if count == min_count]
    assert len(best) == 1, "separator row must be unambiguous"
    return best[0]


def _components_any_color(grid: np.ndarray, bg: int) -> list[np.ndarray]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    components = []
    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(grid[r, c]) == bg:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) != bg:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _monochrome_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    components = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if visited[r, c] or color == bg:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if visited[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    stack.append((nr, nc))
            comp_coords = np.array(coords, dtype=int)
            components.append({"color": color, "coords": comp_coords, "bbox": _bbox(comp_coords)})
    return components


def _parse_arrow(top: np.ndarray, coords: np.ndarray) -> tuple[int, int]:
    assert len(coords) == 4, "each arrow must contain four pixels"
    coord_list = [tuple(map(int, rc)) for rc in coords]
    colors = [int(top[r, c]) for r, c in coord_list]
    color_counts = Counter(colors)
    assert sorted(color_counts.values()) == [1, 3], "arrow must contain one tip pixel and three body pixels"
    src_color = next(color for color, count in color_counts.items() if count == 3)
    dst_color = next(color for color, count in color_counts.items() if count == 1)
    assert src_color != dst_color

    tip = next(coord for coord, color in zip(coord_list, colors) if color == dst_color)
    for coord, color in zip(coord_list, colors):
        if coord != tip:
            assert color == src_color

    row_counts = Counter(r for r, _ in coord_list)
    col_counts = Counter(c for _, c in coord_list)

    if sorted(row_counts.values()) == [1, 3]:
        line_row = next(r for r, count in row_counts.items() if count == 3)
        line = sorted((coord for coord in coord_list if coord[0] == line_row), key=lambda rc: rc[1])
        cols = [c for _, c in line]
        assert cols == list(range(cols[0], cols[0] + 3)), "horizontal arrow line must be consecutive"
        assert tip in {line[0], line[-1]}, "tip must lie at one end of the horizontal line"
        wing = next(coord for coord in coord_list if coord[0] != line_row)
        assert wing[1] == tip[1] and abs(wing[0] - tip[0]) == 1, "wing must attach to the tip endpoint"
    elif sorted(col_counts.values()) == [1, 3]:
        line_col = next(c for c, count in col_counts.items() if count == 3)
        line = sorted((coord for coord in coord_list if coord[1] == line_col), key=lambda rc: rc[0])
        rows = [r for r, _ in line]
        assert rows == list(range(rows[0], rows[0] + 3)), "vertical arrow line must be consecutive"
        assert tip in {line[0], line[-1]}, "tip must lie at one end of the vertical line"
        wing = next(coord for coord in coord_list if coord[1] != line_col)
        assert wing[0] == tip[0] and abs(wing[1] - tip[1]) == 1, "wing must attach to the tip endpoint"
    else:
        raise AssertionError("component does not match the asymmetric arrow geometry")

    return src_color, dst_color


def _resolve_color(color: int, rules: dict[int, int]) -> int:
    current = color
    seen = set()
    while current in rules:
        assert current not in seen, "color rules must be acyclic"
        seen.add(current)
        current = rules[current]
    return current


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each lower rectangle by following the color chain encoded by the upper arrow legend.

    Input structure:
    - A single full-width separator row splits the grid into an upper legend panel and a lower scene panel.
    - Each upper-panel arrow is a 4-connected asymmetric T-like shape made from a 3-pixel horizontal or vertical line plus a 1-pixel perpendicular wing attached at one endpoint.
    - The arrow body and wing use color X, while the tip endpoint uses color Y; this encodes the mapping X -> Y.
    - The lower panel has a black background with several solid 4-connected monochrome rectangles.

    Rule:
    - Extract every upper-panel arrow, recover its source color X and destination color Y from the arrow geometry, and build the directed color-replacement rules.
    - For each lower-panel rectangle, repeatedly apply the rule for its current color until no outgoing rule remains.
    - Repaint the entire rectangle with that terminal color while leaving the upper panel and separator unchanged.

    Key insight:
    - The singleton tip color is not enough by itself; the asymmetric wing identifies which endpoint is the arrow tip, and the resulting rules must be chased through a full chain rather than applied only once.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and 3 <= g.shape[0] <= 30 and 1 <= g.shape[1] <= 30

    sep_row, _ = _find_separator(g)
    top = g[:sep_row, :]
    bottom = g[sep_row + 1 :, :]
    assert top.size > 0 and bottom.size > 0, "both panels must be non-empty"

    top_bg = _dominant_color(top)
    rules = {}
    for coords in _components_any_color(top, top_bg):
        src_color, dst_color = _parse_arrow(top, coords)
        if src_color in rules:
            assert rules[src_color] == dst_color, "source color cannot map to multiple targets"
        rules[src_color] = dst_color
    assert rules, "expected at least one arrow rule"

    bottom_bg = _dominant_color(bottom)
    out = g.copy()
    for comp in _monochrome_components(bottom, bottom_bg):
        r0, c0, r1, c1 = comp["bbox"]
        color = int(comp["color"])
        assert np.all(bottom[r0 : r1 + 1, c0 : c1 + 1] == color), "lower objects must be solid rectangles"
        final_color = _resolve_color(color, rules)
        out[sep_row + 1 + r0 : sep_row + 2 + r1, c0 : c1 + 1] = final_color
    return out


def _arrow_cells(row: int, col: int, orientation: str, tip_at_start: bool, wing_side: int):
    if orientation == "h":
        line = [(row, col + offset) for offset in range(3)]
        tip = line[0] if tip_at_start else line[-1]
        wing = (tip[0] + wing_side, tip[1])
    else:
        line = [(row + offset, col) for offset in range(3)]
        tip = line[0] if tip_at_start else line[-1]
        wing = (tip[0], tip[1] + wing_side)
    return line + [wing], tip


def _can_place_cells(occupied: np.ndarray, cells: list[tuple[int, int]], pad: int = 1) -> bool:
    h, w = occupied.shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
    for r, c in cells:
        r0 = max(0, r - pad)
        r1 = min(h, r + pad + 1)
        c0 = max(0, c - pad)
        c1 = min(w, c + pad + 1)
        if np.any(occupied[r0:r1, c0:c1]):
            return False
    return True


def _mark_cells(occupied: np.ndarray, cells: list[tuple[int, int]]) -> None:
    for r, c in cells:
        occupied[r, c] = True


def _can_place_rect(occupied: np.ndarray, r0: int, c0: int, height: int, width: int, pad: int = 1) -> bool:
    h, w = occupied.shape
    r1 = r0 + height - 1
    c1 = c0 + width - 1
    if r0 < 0 or c0 < 0 or r1 >= h or c1 >= w:
        return False
    rr0 = max(0, r0 - pad)
    rr1 = min(h, r1 + pad + 1)
    cc0 = max(0, c0 - pad)
    cc1 = min(w, c1 + pad + 1)
    return not np.any(occupied[rr0:rr1, cc0:cc1])


def _mark_rect(occupied: np.ndarray, r0: int, c0: int, height: int, width: int) -> None:
    occupied[r0 : r0 + height, c0 : c0 + width] = True


def _sample_chains(rng: random.Random) -> list[list[int]]:
    chain_count = rng.choice([1, 2])
    if chain_count == 1:
        lengths = [rng.randint(2, 4)]
    else:
        lengths = [rng.randint(2, 3), rng.randint(1, 2)]
        rng.shuffle(lengths)
    total_nodes = sum(lengths) + chain_count
    colors = rng.sample(list(GEN_COLORS), total_nodes)
    chains = []
    idx = 0
    for length in lengths:
        chains.append(colors[idx : idx + length + 1])
        idx += length + 1
    return chains


def _rules_from_chains(chains: list[list[int]]) -> dict[int, int]:
    rules = {}
    for chain in chains:
        for src, dst in zip(chain, chain[1:]):
            rules[int(src)] = int(dst)
    return rules


def _sample_block_colors(rng: random.Random, chains: list[list[int]], rules: dict[int, int], count: int) -> list[int]:
    all_chain_colors = {color for chain in chains for color in chain}
    terminals = [chain[-1] for chain in chains]
    deep_sources = [chain[i] for chain in chains for i in range(max(0, len(chain) - 2))]
    transformed = list(rules)
    stable = terminals + [color for color in GEN_COLORS if color not in all_chain_colors]
    assert transformed and stable

    colors = [
        rng.choice(deep_sources if deep_sources else transformed),
        rng.choice(transformed),
        rng.choice(stable),
    ]
    palette = transformed + terminals + stable
    while len(colors) < count:
        colors.append(rng.choice(palette))
    rng.shuffle(colors)
    return colors


def _random_rect_size(rng: random.Random, bottom_h: int, width: int) -> tuple[int, int]:
    rect_h = rng.randint(1, min(4, bottom_h - 2))
    rect_w = rng.randint(1, min(5, width - 2))
    if rect_h * rect_w == 1:
        if rect_w < min(5, width - 2):
            rect_w += 1
        else:
            rect_h += 1
    return rect_h, rect_w


def _expected_output(grid: np.ndarray, sep_row: int, placements: list[tuple[int, int, int, int, int]], rules: dict[int, int]) -> np.ndarray:
    out = grid.copy()
    for r0, c0, height, width, color in placements:
        out[sep_row + 1 + r0 : sep_row + 1 + r0 + height, c0 : c0 + width] = _resolve_color(color, rules)
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    chains = _sample_chains(rng)
    rules = _rules_from_chains(chains)

    rect_count = rng.randint(4, 7)
    width_min = max(12, 4 * len(rules) + 4, 2 * rect_count + 6)
    if width_min > 30:
        return None
    width = rng.randint(width_min, 30)

    top_h = rng.randint(6, 11)
    bottom_min = max(8, rect_count + 3)
    bottom_max = min(17, 29 - top_h)
    if bottom_min > bottom_max:
        return None
    bottom_h = rng.randint(bottom_min, bottom_max)

    grid = np.full((top_h + 1 + bottom_h, width), GEN_BG, dtype=int)
    grid[top_h, :] = GEN_SEPARATOR

    top = grid[:top_h, :]
    top_occupied = np.zeros_like(top, dtype=bool)
    rule_items = list(rules.items())
    rng.shuffle(rule_items)
    for src_color, dst_color in rule_items:
        placed = False
        for _ in range(GEN_LAYOUT_ATTEMPTS):
            orientation = rng.choice(["h", "v"])
            tip_at_start = rng.choice([True, False])
            wing_side = rng.choice([-1, 1])
            row = rng.randrange(top_h)
            col = rng.randrange(width)
            cells, tip = _arrow_cells(row, col, orientation, tip_at_start, wing_side)
            if not _can_place_cells(top_occupied, cells, pad=1):
                continue
            for rr, cc in cells:
                top[rr, cc] = src_color
            top[tip] = dst_color
            _mark_cells(top_occupied, cells)
            placed = True
            break
        if not placed:
            return None

    bottom = grid[top_h + 1 :, :]
    block_colors = _sample_block_colors(rng, chains, rules, rect_count)
    placements = []
    bottom_occupied = np.zeros_like(bottom, dtype=bool)
    for color in block_colors:
        placed = False
        for _ in range(GEN_LAYOUT_ATTEMPTS):
            rect_h, rect_w = _random_rect_size(rng, bottom_h, width)
            max_r0 = bottom_h - rect_h - 1
            max_c0 = width - rect_w - 1
            if max_r0 < 1 or max_c0 < 1:
                return None
            r0 = rng.randint(1, max_r0)
            c0 = rng.randint(1, max_c0)
            if not _can_place_rect(bottom_occupied, r0, c0, rect_h, rect_w, pad=1):
                continue
            bottom[r0 : r0 + rect_h, c0 : c0 + rect_w] = color
            _mark_rect(bottom_occupied, r0, c0, rect_h, rect_w)
            placements.append((r0, c0, rect_h, rect_w, color))
            placed = True
            break
        if not placed:
            return None

    resolved = [_resolve_color(color, rules) for _, _, _, _, color in placements]
    if not any(before != after for (_, _, _, _, before), after in zip(placements, resolved)):
        return None
    if not any(before == after for (_, _, _, _, before), after in zip(placements, resolved)):
        return None

    expected = _expected_output(grid, top_h, placements, rules)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for the color-chain arrow task")
