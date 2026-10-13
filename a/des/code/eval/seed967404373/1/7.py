import random
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
MAROON = 9

SOURCE_COLORS = (RED, BLUE, YELLOW)
EDGE_SIDES = ("top", "bottom", "left", "right")
HORIZONTAL_SIDES = ("left", "right")
VERTICAL_SIDES = ("top", "bottom")

COLOR_TO_BIT = {RED: 1, BLUE: 2, YELLOW: 4}
BITS_TO_COLOR = {
    1: RED,
    2: BLUE,
    4: YELLOW,
    3: MAGENTA,
    5: ORANGE,
    6: GREEN,
    7: MAROON,
}

GRID_SIZE_RANGE = (11, 18)
EXTRA_SOURCE_COUNT_RANGE = (0, 3)
MAX_GENERATION_ATTEMPTS = 400


def _direction_for_side(side: str) -> Tuple[int, int]:
    return {
        "top": (1, 0),
        "bottom": (-1, 0),
        "left": (0, 1),
        "right": (0, -1),
    }[side]


def _source_cell(side: str, index: int, H: int, W: int) -> Tuple[int, int]:
    if side == "top":
        return (0, index)
    if side == "bottom":
        return (H - 1, index)
    if side == "left":
        return (index, 0)
    if side == "right":
        return (index, W - 1)
    raise ValueError(f"unknown side: {side}")


def _forward_cells(
    start: Tuple[int, int], side: str, H: int, W: int, *, include_edge: bool
) -> List[Tuple[int, int]]:
    dr, dc = _direction_for_side(side)
    r, c = start
    out: List[Tuple[int, int]] = []
    r += dr
    c += dc
    while 0 <= r < H and 0 <= c < W:
        if not include_edge and (r == 0 or r == H - 1 or c == 0 or c == W - 1):
            break
        out.append((r, c))
        r += dr
        c += dc
    return out


def _domino_options_through(anchor: Tuple[int, int], H: int, W: int) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    r, c = anchor
    options: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
    if 0 < c < W - 2:
        options.append(((r, c), (r, c + 1)))
    if 1 < c < W - 1:
        options.append(((r, c - 1), (r, c)))
    if 0 < r < H - 2:
        options.append(((r, c), (r + 1, c)))
    if 1 < r < H - 1:
        options.append(((r - 1, c), (r, c)))
    return options


def _neighbors4(r: int, c: int) -> Iterable[Tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _edge_sources(grid: np.ndarray) -> List[Tuple[int, int, str, int]]:
    g = np.asarray(grid, dtype=int)
    H, W = map(int, g.shape)
    assert H >= 3 and W >= 3, "grid must be at least 3x3"
    allowed = {BG, GREY, *SOURCE_COLORS}
    assert set(map(int, np.unique(g))) <= allowed, "unexpected input colors"
    assert g[0, 0] == BG and g[0, W - 1] == BG and g[H - 1, 0] == BG and g[H - 1, W - 1] == BG, (
        "corner sources are ambiguous"
    )
    assert not np.any(g[1 : H - 1, 1 : W - 1] == np.array(SOURCE_COLORS)[:, None, None]), (
        "sources must lie on the border only"
    )
    assert not np.any(g[0, :] == GREY)
    assert not np.any(g[H - 1, :] == GREY)
    assert not np.any(g[:, 0] == GREY)
    assert not np.any(g[:, W - 1] == GREY)

    sources: List[Tuple[int, int, str, int]] = []
    for c in range(1, W - 1):
        if int(g[0, c]) in SOURCE_COLORS:
            sources.append((0, c, "top", int(g[0, c])))
        if int(g[H - 1, c]) in SOURCE_COLORS:
            sources.append((H - 1, c, "bottom", int(g[H - 1, c])))
    for r in range(1, H - 1):
        if int(g[r, 0]) in SOURCE_COLORS:
            sources.append((r, 0, "left", int(g[r, 0])))
        if int(g[r, W - 1]) in SOURCE_COLORS:
            sources.append((r, W - 1, "right", int(g[r, W - 1])))

    assert sources, "expected at least one source"
    return sources


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cast edge-origin rays, stop them at grey blockers, and mix the unique source
    colors present at each covered cell.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    H, W = map(int, g.shape)
    sources = _edge_sources(g)

    hits = np.zeros((H, W), dtype=np.uint8)
    source_positions = {(r, c) for r, c, _side, _color in sources}

    for r, c, side, color in sources:
        dr, dc = _direction_for_side(side)
        bit = COLOR_TO_BIT[color]
        rr, cc = r + dr, c + dc
        while 0 <= rr < H and 0 <= cc < W:
            if int(g[rr, cc]) == GREY:
                break
            hits[rr, cc] |= bit
            if rr == 0 or rr == H - 1 or cc == 0 or cc == W - 1:
                break
            rr += dr
            cc += dc

    out = g.copy()
    for bits, color in BITS_TO_COLOR.items():
        out[hits == bits] = color

    for r, c in source_positions:
        out[r, c] = g[r, c]
    out[g == GREY] = GREY
    return out


def _can_place_blocker(
    blocker_mask: np.ndarray,
    cells: Sequence[Tuple[int, int]],
    forbidden_cells: Set[Tuple[int, int]],
    forbidden_rows: Set[int],
    forbidden_cols: Set[int],
) -> bool:
    H, W = map(int, blocker_mask.shape)
    cell_set = set(cells)
    for r, c in cells:
        if r <= 0 or r >= H - 1 or c <= 0 or c >= W - 1:
            return False
        if blocker_mask[r, c]:
            return False
        if (r, c) in forbidden_cells or r in forbidden_rows or c in forbidden_cols:
            return False
    for r, c in cells:
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < H and 0 <= nc < W and blocker_mask[nr, nc] and (nr, nc) not in cell_set:
                return False
    return True


def _place_blocker_through_anchor(
    blocker_mask: np.ndarray,
    anchor: Tuple[int, int],
    *,
    forbidden_cells: Optional[Set[Tuple[int, int]]] = None,
    forbidden_rows: Optional[Set[int]] = None,
    forbidden_cols: Optional[Set[int]] = None,
) -> bool:
    H, W = map(int, blocker_mask.shape)
    forbidden_cells = forbidden_cells or set()
    forbidden_rows = forbidden_rows or set()
    forbidden_cols = forbidden_cols or set()
    options = _domino_options_through(anchor, H, W)
    random.shuffle(options)
    for cells in options:
        if _can_place_blocker(blocker_mask, cells, forbidden_cells, forbidden_rows, forbidden_cols):
            for r, c in cells:
                blocker_mask[r, c] = True
            return True
    return False


def _place_blocker_ahead(
    blocker_mask: np.ndarray,
    start: Tuple[int, int],
    side: str,
    *,
    min_distance: int,
    max_distance: Optional[int] = None,
    forbidden_cells: Optional[Set[Tuple[int, int]]] = None,
    forbidden_rows: Optional[Set[int]] = None,
    forbidden_cols: Optional[Set[int]] = None,
) -> bool:
    H, W = map(int, blocker_mask.shape)
    anchors = _forward_cells(start, side, H, W, include_edge=False)
    if max_distance is not None:
        anchors = anchors[:max_distance]
    anchors = anchors[min_distance - 1 :]
    if not anchors:
        return False
    sample = anchors[: min(4, len(anchors))]
    random.shuffle(sample)
    for anchor in sample:
        if _place_blocker_through_anchor(
            blocker_mask,
            anchor,
            forbidden_cells=forbidden_cells,
            forbidden_rows=forbidden_rows,
            forbidden_cols=forbidden_cols,
        ):
            return True
    return False


def _add_source(
    sources: Dict[Tuple[int, int], int],
    row_sides: Dict[int, Set[str]],
    col_sides: Dict[int, Set[str]],
    side: str,
    index: int,
    color: int,
    H: int,
    W: int,
) -> bool:
    pos = _source_cell(side, index, H, W)
    if pos in sources:
        return False
    if side in HORIZONTAL_SIDES:
        if not (1 <= index <= H - 2):
            return False
        used = row_sides.setdefault(index, set())
        if side in used:
            return False
        used.add(side)
    else:
        if not (1 <= index <= W - 2):
            return False
        used = col_sides.setdefault(index, set())
        if side in used:
            return False
        used.add(side)
    sources[pos] = color
    return True


def _build_grid(H: int, W: int, sources: Dict[Tuple[int, int], int], blocker_mask: np.ndarray) -> np.ndarray:
    grid = np.full((H, W), BG, dtype=int)
    grid[blocker_mask] = GREY
    for (r, c), color in sources.items():
        grid[r, c] = color
    return grid


def _random_target(H: int, W: int) -> Tuple[int, int]:
    return random.randint(3, H - 4), random.randint(3, W - 4)


def _add_pair_core(
    H: int,
    W: int,
    sources: Dict[Tuple[int, int], int],
    row_sides: Dict[int, Set[str]],
    col_sides: Dict[int, Set[str]],
    blocker_mask: np.ndarray,
    pair_colors: Tuple[int, int],
) -> Optional[Tuple[Tuple[int, int], int]]:
    target = _random_target(H, W)
    tr, tc = target
    h_side = random.choice(HORIZONTAL_SIDES)
    v_side = random.choice(VERTICAL_SIDES)
    h_color, v_color = pair_colors

    if not _add_source(sources, row_sides, col_sides, h_side, tr, h_color, H, W):
        return None
    if not _add_source(sources, row_sides, col_sides, v_side, tc, v_color, H, W):
        return None

    forbidden = {target}
    if not _place_blocker_ahead(
        blocker_mask,
        target,
        h_side,
        min_distance=1,
        max_distance=4,
        forbidden_cells=forbidden,
    ):
        return None
    if not _place_blocker_ahead(
        blocker_mask,
        target,
        v_side,
        min_distance=1,
        max_distance=4,
        forbidden_cells=forbidden,
    ):
        return None

    return target, BITS_TO_COLOR[COLOR_TO_BIT[h_color] | COLOR_TO_BIT[v_color]]


def _add_triple_h_core(
    H: int,
    W: int,
    sources: Dict[Tuple[int, int], int],
    row_sides: Dict[int, Set[str]],
    col_sides: Dict[int, Set[str]],
    blocker_mask: np.ndarray,
) -> Optional[Tuple[Tuple[int, int], int]]:
    target = _random_target(H, W)
    tr, tc = target
    left_color, right_color, vertical_color = random.sample(list(SOURCE_COLORS), 3)
    v_side = random.choice(VERTICAL_SIDES)

    if not _add_source(sources, row_sides, col_sides, "left", tr, left_color, H, W):
        return None
    if not _add_source(sources, row_sides, col_sides, "right", tr, right_color, H, W):
        return None
    if not _add_source(sources, row_sides, col_sides, v_side, tc, vertical_color, H, W):
        return None

    forbidden = {target}
    if not _place_blocker_through_anchor(blocker_mask, (tr, tc - 1), forbidden_cells=forbidden):
        return None
    if not _place_blocker_through_anchor(blocker_mask, (tr, tc + 1), forbidden_cells=forbidden):
        return None
    if not _place_blocker_ahead(
        blocker_mask,
        target,
        v_side,
        min_distance=1,
        max_distance=3,
        forbidden_cells=forbidden,
    ):
        return None

    return target, MAROON


def _add_triple_v_core(
    H: int,
    W: int,
    sources: Dict[Tuple[int, int], int],
    row_sides: Dict[int, Set[str]],
    col_sides: Dict[int, Set[str]],
    blocker_mask: np.ndarray,
) -> Optional[Tuple[Tuple[int, int], int]]:
    target = _random_target(H, W)
    tr, tc = target
    top_color, bottom_color, horizontal_color = random.sample(list(SOURCE_COLORS), 3)
    h_side = random.choice(HORIZONTAL_SIDES)

    if not _add_source(sources, row_sides, col_sides, "top", tc, top_color, H, W):
        return None
    if not _add_source(sources, row_sides, col_sides, "bottom", tc, bottom_color, H, W):
        return None
    if not _add_source(sources, row_sides, col_sides, h_side, tr, horizontal_color, H, W):
        return None

    forbidden = {target}
    if not _place_blocker_through_anchor(blocker_mask, (tr - 1, tc), forbidden_cells=forbidden):
        return None
    if not _place_blocker_through_anchor(blocker_mask, (tr + 1, tc), forbidden_cells=forbidden):
        return None
    if not _place_blocker_ahead(
        blocker_mask,
        target,
        h_side,
        min_distance=1,
        max_distance=3,
        forbidden_cells=forbidden,
    ):
        return None

    return target, MAROON


def _add_extra_source(
    H: int,
    W: int,
    sources: Dict[Tuple[int, int], int],
    row_sides: Dict[int, Set[str]],
    col_sides: Dict[int, Set[str]],
    blocker_mask: np.ndarray,
    protected_rows: Set[int],
    protected_cols: Set[int],
) -> bool:
    choose_horizontal = random.random() < 0.5
    if choose_horizontal:
        rows = [r for r in range(1, H - 1) if r not in row_sides]
        if not rows:
            return False
        side = random.choice(HORIZONTAL_SIDES)
        index = random.choice(rows)
    else:
        cols = [c for c in range(1, W - 1) if c not in col_sides]
        if not cols:
            return False
        side = random.choice(VERTICAL_SIDES)
        index = random.choice(cols)

    color = random.choice(SOURCE_COLORS)
    if not _add_source(sources, row_sides, col_sides, side, index, color, H, W):
        return False

    source = _source_cell(side, index, H, W)
    if random.random() < 0.8:
        _place_blocker_ahead(
            blocker_mask,
            source,
            side,
            min_distance=2,
            max_distance=6,
            forbidden_cells={source},
            forbidden_rows=protected_rows,
            forbidden_cols=protected_cols,
        )
    return True


def _count_sources(grid: np.ndarray) -> int:
    return len(_edge_sources(grid))


def _generator_accepts(grid: np.ndarray, target: Tuple[int, int], expected_color: int) -> bool:
    try:
        out = main(grid)
    except Exception:
        return False
    tr, tc = target
    if int(out[tr, tc]) != expected_color:
        return False
    if np.array_equal(out, grid):
        return False
    if _count_sources(grid) < 2:
        return False
    return True


def _gen_attempt() -> Optional[np.ndarray]:
    H = random.randint(*GRID_SIZE_RANGE)
    W = random.randint(*GRID_SIZE_RANGE)
    sources: Dict[Tuple[int, int], int] = {}
    row_sides: Dict[int, Set[str]] = {}
    col_sides: Dict[int, Set[str]] = {}
    blocker_mask = np.zeros((H, W), dtype=bool)

    if random.random() < 0.5:
        pair_colors = random.choice([(RED, BLUE), (RED, YELLOW), (BLUE, YELLOW)])
        core = _add_pair_core(H, W, sources, row_sides, col_sides, blocker_mask, pair_colors)
    else:
        if random.random() < 0.5:
            core = _add_triple_h_core(H, W, sources, row_sides, col_sides, blocker_mask)
        else:
            core = _add_triple_v_core(H, W, sources, row_sides, col_sides, blocker_mask)
    if core is None:
        return None

    target, expected_color = core
    protected_rows = {target[0]}
    protected_cols = {target[1]}

    for _ in range(random.randint(*EXTRA_SOURCE_COUNT_RANGE)):
        for _attempt in range(8):
            if _add_extra_source(
                H,
                W,
                sources,
                row_sides,
                col_sides,
                blocker_mask,
                protected_rows,
                protected_cols,
            ):
                break

    grid = _build_grid(H, W, sources, blocker_mask)
    return grid if _generator_accepts(grid, target, expected_color) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
