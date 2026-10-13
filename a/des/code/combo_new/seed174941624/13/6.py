import random
from typing import Optional

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
PALETTE = (1, 2, 3, 4, 6, 7, 8)

DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
LEFT_TURN = {0: 3, 1: 0, 2: 1, 3: 2}
RIGHT_TURN = {0: 1, 1: 2, 2: 3, 3: 0}

LEFT_L_MASK = np.array(
    [
        [1, 0, 0],
        [1, 0, 0],
        [1, 1, 1],
    ],
    dtype=bool,
)
RIGHT_L_MASK = np.array(
    [
        [0, 0, 1],
        [0, 0, 1],
        [1, 1, 1],
    ],
    dtype=bool,
)

LEGEND_WIDTH = 7
LEGEND_ROWS_RANGE = (2, 4)
GRID_RANGE = (19, 30)
MAX_GENERATION_ATTEMPTS = 400


def _other_side(side: str) -> str:
    return "right" if side == "left" else "left"


def _turn_dir(dir_idx: int, side: str) -> int:
    return LEFT_TURN[dir_idx] if side == "left" else RIGHT_TURN[dir_idx]


def _step(cell: tuple[int, int], dir_idx: int) -> tuple[int, int]:
    dr, dc = DIRS[dir_idx]
    return cell[0] + dr, cell[1] + dc


def _logical_to_full(cell: tuple[int, int]) -> tuple[int, int]:
    return 2 * cell[0] + 1, 2 * cell[1] + 1


def _midpoint_full(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    ar, ac = _logical_to_full(a)
    br, bc = _logical_to_full(b)
    return (ar + br) // 2, (ac + bc) // 2


def _legend_box(height: int, width: int, entries: int, corner: str) -> tuple[int, int, int, int]:
    box_h = 2 + 3 * entries
    box_w = LEGEND_WIDTH
    if corner == "tl":
        return 0, 0, box_h - 1, box_w - 1
    if corner == "tr":
        return 0, width - box_w, box_h - 1, width - 1
    if corner == "bl":
        return height - box_h, 0, height - 1, box_w - 1
    if corner == "br":
        return height - box_h, width - box_w, height - 1, width - 1
    raise ValueError(f"unknown corner {corner}")


def _expand_box(box: tuple[int, int, int, int], height: int, width: int, pad: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    return max(0, r0 - pad), max(0, c0 - pad), min(height - 1, r1 + pad), min(width - 1, c1 + pad)


def _box_mask(shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    r0, c0, r1, c1 = box
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _parse_legend_box(grid: np.ndarray, box: tuple[int, int, int, int]) -> Optional[dict[int, str]]:
    r0, c0, r1, c1 = box
    if r1 - r0 + 1 < 8 or c1 - c0 + 1 < 7:
        return None
    border = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
    border[1:-1, 1:-1] = GREY
    if not np.all(border == GREY):
        return None

    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    inner_h, inner_w = inner.shape
    if inner_h < 6 or inner_h % 3 != 0 or inner_w < 5:
        return None
    if not np.any(inner == MAROON):
        return None

    mapping: dict[int, str] = {}
    bands = inner_h // 3
    for band_idx in range(bands):
        band = inner[3 * band_idx : 3 * band_idx + 3]
        color_positions = np.argwhere((band != BLACK) & (band != GREY) & (band != MAROON))
        if len(color_positions) != 1:
            return None
        sample_r, sample_c = map(int, color_positions[0])
        maroon_positions = np.argwhere(band == MAROON)
        if len(maroon_positions) != 5:
            return None
        mr0 = int(maroon_positions[:, 0].min())
        mc0 = int(maroon_positions[:, 1].min())
        mr1 = int(maroon_positions[:, 0].max())
        mc1 = int(maroon_positions[:, 1].max())
        if (mr1 - mr0 + 1, mc1 - mc0 + 1) != (3, 3):
            return None
        if sample_c >= mc0:
            return None
        shape = band[mr0 : mr1 + 1, mc0 : mc1 + 1] == MAROON
        if np.array_equal(shape, LEFT_L_MASK):
            rule = "left"
        elif np.array_equal(shape, RIGHT_L_MASK):
            rule = "right"
        else:
            return None
        sample_color = int(band[sample_r, sample_c])
        if sample_color in mapping:
            return None
        mapping[sample_color] = rule

    return mapping if mapping else None


def _find_legend(grid: np.ndarray) -> tuple[tuple[int, int, int, int], dict[int, str]]:
    height, width = grid.shape
    best_box = None
    best_mapping = None
    best_score = None
    corners = ("tl", "tr", "bl", "br")

    for corner in corners:
        for box_h in range(8, height + 1):
            for box_w in range(7, width + 1):
                if corner == "tl":
                    box = (0, 0, box_h - 1, box_w - 1)
                elif corner == "tr":
                    box = (0, width - box_w, box_h - 1, width - 1)
                elif corner == "bl":
                    box = (height - box_h, 0, height - 1, box_w - 1)
                else:
                    box = (height - box_h, width - box_w, height - 1, width - 1)
                mapping = _parse_legend_box(grid, box)
                if mapping is None:
                    continue
                area = box_h * box_w
                score = (len(mapping), -area)
                if best_score is None or score > best_score:
                    best_box = box
                    best_mapping = mapping
                    best_score = score

    assert best_box is not None and best_mapping is not None, "failed to locate the corner legend"
    return best_box, best_mapping


def _find_start(grid: np.ndarray, legend_box: tuple[int, int, int, int]) -> tuple[tuple[int, int], int, int]:
    legend_mask = _box_mask(grid.shape, legend_box)
    candidates = np.argwhere(~legend_mask & (grid != BLACK) & (grid != GREY) & (grid != MAROON))
    assert len(candidates) == 1, "expected exactly one start marker outside the legend"
    r, c = map(int, candidates[0])
    start_color = int(grid[r, c])
    dirs = []
    for dir_idx, (dr, dc) in enumerate(DIRS):
        nr, nc = r + dr, c + dc
        if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and not legend_mask[nr, nc] and int(grid[nr, nc]) == BLACK:
            dirs.append(dir_idx)
    assert len(dirs) == 1, "the start marker must sit at a dead-end"
    return (r, c), start_color, dirs[0]


def _simulate(grid: np.ndarray, legend_box: tuple[int, int, int, int], rule: str) -> np.ndarray:
    start, start_color, dir_idx = _find_start(grid, legend_box)
    legend_mask = _box_mask(grid.shape, legend_box)
    available = (grid == BLACK) & ~legend_mask
    out = grid.copy()

    r, c = start
    out[r, c] = start_color
    seen_states = {(r, c, dir_idx)}

    while True:
        forward = dir_idx
        if rule == "left":
            turn_order = [forward, LEFT_TURN[dir_idx], RIGHT_TURN[dir_idx], (dir_idx + 2) % 4]
        else:
            turn_order = [forward, RIGHT_TURN[dir_idx], LEFT_TURN[dir_idx], (dir_idx + 2) % 4]

        next_dir = None
        for cand_dir in turn_order:
            nr, nc = r + DIRS[cand_dir][0], c + DIRS[cand_dir][1]
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and available[nr, nc]:
                next_dir = cand_dir
                break
        if next_dir is None:
            break

        r += DIRS[next_dir][0]
        c += DIRS[next_dir][1]
        dir_idx = next_dir
        out[r, c] = start_color
        available[r, c] = False
        state = (r, c, dir_idx)
        if state in seen_states:
            break
        seen_states.add(state)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow the start-colored corridor marker through a walled maze using the matching corner legend row to decide left-vs-right turns.

    Input structure:
    - Grey pixels form a 1-cell-wide orthogonal maze; black pixels are the open corridor cells.
    - A corner is occupied by a grey-bordered rectangular legend box whose rows each contain one color sample pixel and one maroon L-shaped turn symbol.
    - Exactly one matching color appears again outside the legend as the maze start marker at the end of a dead-end corridor.

    Rule:
    - Read the legend row whose sample color matches the start marker.
    - A left-corner maroon L means prefer left turns after a blockage; a right-corner maroon L means prefer right turns.
    - Start from the marker and move through black corridor cells, painting every visited corridor cell with the start color.
    - Keep moving straight when possible; when blocked, try the preferred side first, then the other side, and never re-enter already painted path cells.
    - Stop when no unpainted corridor neighbor remains; keep the legend box unchanged.

    Key insight:
    - The corner rectangle is a lookup table: color chooses the row, and the maroon L orientation on that row encodes the turn bias needed to trace the maze path.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30
    legend_box, mapping = _find_legend(g)
    _, start_color, _ = _find_start(g, legend_box)
    assert start_color in mapping, "start color must be present in the legend"
    return _simulate(g, legend_box, mapping[start_color])


def _draw_legend(grid: np.ndarray, box: tuple[int, int, int, int], entries: list[tuple[int, str]]) -> None:
    r0, c0, r1, c1 = box
    grid[r0 : r1 + 1, c0 : c1 + 1] = GREY
    grid[r0 + 1 : r1, c0 + 1 : c1] = BLACK
    for idx, (color, rule) in enumerate(entries):
        band_r0 = r0 + 1 + 3 * idx
        inner_c0 = c0 + 1
        grid[band_r0 + 1, inner_c0] = color
        mask = LEFT_L_MASK if rule == "left" else RIGHT_L_MASK
        patch = grid[band_r0 : band_r0 + 3, inner_c0 + 2 : inner_c0 + 5]
        patch[mask] = MAROON


def _balanced_events(rng: random.Random, turns: int) -> list[str]:
    events: list[str] = []
    base = ["preferred_only", "both", "fallback"]
    while len(events) < turns:
        chunk = base[:]
        rng.shuffle(chunk)
        events.extend(chunk)
    return events[:turns]


def _can_add_edge(
    a: tuple[int, int],
    b: tuple[int, int],
    mh: int,
    mw: int,
    reserved: np.ndarray,
    used_nodes: set[tuple[int, int]],
) -> bool:
    br, bc = b
    if not (0 <= br < mh and 0 <= bc < mw):
        return False
    if b in used_nodes:
        return False
    fr, fc = _logical_to_full(b)
    mr, mc = _midpoint_full(a, b)
    return not reserved[fr, fc] and not reserved[mr, mc]


def _add_edge(
    a: tuple[int, int],
    b: tuple[int, int],
    used_nodes: set[tuple[int, int]],
    used_edges: set[frozenset[tuple[int, int]]],
    degrees: dict[tuple[int, int], int],
) -> None:
    used_nodes.add(b)
    used_edges.add(frozenset((a, b)))
    degrees[a] = degrees.get(a, 0) + 1
    degrees[b] = degrees.get(b, 0) + 1


def _build_route(
    rng: random.Random,
    height: int,
    width: int,
    reserved: np.ndarray,
    rule: str,
) -> Optional[dict[str, object]]:
    mh = (height - 1) // 2
    mw = (width - 1) // 2
    open_nodes = []
    for r in range(mh):
        for c in range(mw):
            fr, fc = _logical_to_full((r, c))
            if not reserved[fr, fc]:
                open_nodes.append((r, c))
    if not open_nodes:
        return None

    for _ in range(240):
        turns = rng.randint(3, 5)
        events = _balanced_events(rng, turns)
        start = rng.choice(open_nodes)
        dir_idx = rng.randrange(4)

        used_nodes = {start}
        used_edges: set[frozenset[tuple[int, int]]] = set()
        degrees: dict[tuple[int, int], int] = {start: 0}
        cur = start
        heading = dir_idx
        ok = True

        for event in events:
            straight_len = rng.randint(1, 3)
            for _ in range(straight_len):
                nxt = _step(cur, heading)
                if not _can_add_edge(cur, nxt, mh, mw, reserved, used_nodes):
                    ok = False
                    break
                _add_edge(cur, nxt, used_nodes, used_edges, degrees)
                cur = nxt
            if not ok:
                break

            route_side = rule if event != "fallback" else _other_side(rule)
            new_heading = _turn_dir(heading, route_side)
            nxt = _step(cur, new_heading)
            if not _can_add_edge(cur, nxt, mh, mw, reserved, used_nodes):
                ok = False
                break

            if event == "both":
                branch_heading = _turn_dir(heading, _other_side(rule))
                branch_nxt = _step(cur, branch_heading)
                if not _can_add_edge(cur, branch_nxt, mh, mw, reserved, used_nodes):
                    ok = False
                    break
                _add_edge(cur, branch_nxt, used_nodes, used_edges, degrees)

            _add_edge(cur, nxt, used_nodes, used_edges, degrees)
            cur = nxt
            heading = new_heading

        if not ok:
            continue

        final_len = rng.randint(1, 3)
        for _ in range(final_len):
            nxt = _step(cur, heading)
            if not _can_add_edge(cur, nxt, mh, mw, reserved, used_nodes):
                ok = False
                break
            _add_edge(cur, nxt, used_nodes, used_edges, degrees)
            cur = nxt
        if not ok:
            continue

        if degrees.get(start, 0) != 1 or degrees.get(cur, 0) != 1:
            continue
        if len(used_edges) < 8:
            continue

        return {
            "start": start,
            "start_dir": dir_idx,
            "used_nodes": used_nodes,
            "used_edges": used_edges,
        }

    return None


def _render_maze(
    height: int,
    width: int,
    legend_box: tuple[int, int, int, int],
    entries: list[tuple[int, str]],
    route: dict[str, object],
    start_color: int,
) -> np.ndarray:
    grid = np.full((height, width), GREY, dtype=int)
    _draw_legend(grid, legend_box, entries)

    used_nodes = route["used_nodes"]
    used_edges = route["used_edges"]
    assert isinstance(used_nodes, set) and isinstance(used_edges, set)

    for node in used_nodes:
        fr, fc = _logical_to_full(node)
        grid[fr, fc] = BLACK
    for edge in used_edges:
        a, b = tuple(edge)
        mr, mc = _midpoint_full(a, b)
        grid[mr, mc] = BLACK

    start = route["start"]
    assert isinstance(start, tuple)
    sr, sc = _logical_to_full(start)
    grid[sr, sc] = start_color
    return grid


def _gen_attempt(rng: random.Random) -> Optional[np.ndarray]:
    entry_count = rng.randint(*LEGEND_ROWS_RANGE)
    start_rule = rng.choice(("left", "right"))

    colors = list(rng.sample(PALETTE, entry_count))
    start_color = rng.choice(colors)
    rules = [rng.choice(("left", "right")) for _ in range(entry_count)]
    rules[colors.index(start_color)] = start_rule
    if len(set(rules)) == 1 and entry_count >= 2:
        flip_idx = 1 if colors[0] == start_color else 0
        rules[flip_idx] = _other_side(start_rule)
    entries = list(zip(colors, rules))

    legend_h = 2 + 3 * entry_count
    min_h = max(GRID_RANGE[0], legend_h + 5)
    min_w = max(GRID_RANGE[0], LEGEND_WIDTH + 8)
    height = rng.choice([v for v in range(min_h, GRID_RANGE[1]) if v % 2 == 1])
    width = rng.choice([v for v in range(min_w, GRID_RANGE[1]) if v % 2 == 1])
    corner = rng.choice(("tl", "tr", "bl", "br"))
    legend_box = _legend_box(height, width, entry_count, corner)

    reserved = np.zeros((height, width), dtype=bool)
    rr0, rc0, rr1, rc1 = _expand_box(legend_box, height, width, 1)
    reserved[rr0 : rr1 + 1, rc0 : rc1 + 1] = True

    route = _build_route(rng, height, width, reserved, start_rule)
    if route is None:
        return None

    grid = _render_maze(height, width, legend_box, entries, route, start_color)

    out = main(grid)
    legend_mask = _box_mask(grid.shape, legend_box)
    changed = np.count_nonzero((grid != out) & ~legend_mask)
    if changed < 3:
        return None
    if not np.array_equal(grid[legend_mask], out[legend_mask]):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid maze-legend ARC sample")
