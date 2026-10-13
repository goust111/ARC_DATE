import random
from collections import defaultdict

import numpy as np


BG = 0
RED = 2
GREY = 5
PATTERN_COLORS = (1, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400
MAX_GRID_SIZE = 30


ARROW_TEMPLATES = {
    (-1, 0): frozenset({(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)}),
    (1, 0): frozenset({(0, 1), (1, 1), (2, 0), (2, 1), (2, 2)}),
    (0, -1): frozenset({(0, 0), (1, 0), (1, 1), (1, 2), (2, 0)}),
    (0, 1): frozenset({(0, 2), (1, 0), (1, 1), (1, 2), (2, 2)}),
}
TEMPLATE_TO_DIR = {shape: direction for direction, shape in ARROW_TEMPLATES.items()}

PATTERN_SHAPES = (
    frozenset({(0, 0)}),
    frozenset({(0, 0), (0, 1)}),
    frozenset({(0, 0), (1, 0)}),
    frozenset({(0, 0), (1, 0), (2, 0)}),
    frozenset({(0, 0), (0, 1), (1, 0)}),
    frozenset({(0, 0), (0, 1), (1, 1)}),
    frozenset({(0, 0), (1, 0), (1, 1)}),
    frozenset({(0, 1), (1, 0), (1, 1)}),
    frozenset({(0, 0), (0, 1), (1, 0), (1, 1)}),
    frozenset({(0, 0), (1, 0), (2, 0), (2, 1)}),
    frozenset({(0, 1), (1, 1), (2, 0), (2, 1)}),
    frozenset({(0, 0), (1, 0), (1, 1), (2, 1)}),
    frozenset({(0, 1), (1, 0), (1, 1), (2, 0)}),
)


def _split_bands(length: int, separators: list[int]) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    start = 0
    for sep in separators:
        if start <= sep - 1:
            bands.append((start, sep - 1))
        start = sep + 1
    if start <= length - 1:
        bands.append((start, length - 1))
    return bands


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    return [
        (r - 1, c - 1),
        (r - 1, c),
        (r - 1, c + 1),
        (r, c - 1),
        (r, c + 1),
        (r + 1, c - 1),
        (r + 1, c),
        (r + 1, c + 1),
    ]


def _mask_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) <= 1:
        return True
    seen = set()
    stack = [tuple(map(int, coords[0]))]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        if (r, c) in seen:
            continue
        seen.add((r, c))
        for rr, cc in _neighbors8(r, c):
            if 0 <= rr < h and 0 <= cc < w and bool(mask[rr, cc]) and (rr, cc) not in seen:
                stack.append((rr, cc))
    return len(seen) == int(mask.sum())


def _room_bands(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = grid.shape
    sep_rows = [r for r in range(h) if bool(np.all(grid[r, :] == GREY))]
    sep_cols = [c for c in range(w) if bool(np.all(grid[:, c] == GREY))]
    assert sep_rows and sep_cols, "grid must contain grey separator rows and columns"

    row_bands = _split_bands(h, sep_rows)
    col_bands = _split_bands(w, sep_cols)
    assert len(row_bands) >= 2 and len(col_bands) >= 2, "must have multiple rooms"

    room_heights = {r1 - r0 + 1 for r0, r1 in row_bands}
    room_widths = {c1 - c0 + 1 for c0, c1 in col_bands}
    assert len(room_heights) == 1 and len(room_widths) == 1, "rooms must be equally sized"

    sep_row_mask = np.zeros(h, dtype=bool)
    sep_col_mask = np.zeros(w, dtype=bool)
    sep_row_mask[sep_rows] = True
    sep_col_mask[sep_cols] = True
    grey_pixels = np.argwhere(grid == GREY)
    for r, c in grey_pixels:
        assert bool(sep_row_mask[int(r)] or sep_col_mask[int(c)]), "grey may only appear on separators"

    return row_bands, col_bands


def _extract_arrow(room: np.ndarray) -> tuple[int, int] | None:
    coords = np.argwhere(room == RED)
    if coords.size == 0:
        return None
    assert len(coords) == 5, "non-terminal rooms must contain exactly one 5-pixel arrow"
    r0, c0 = coords.min(axis=0)
    normalized = frozenset((int(r - r0), int(c - c0)) for r, c in coords)
    assert normalized in TEMPLATE_TO_DIR, "unrecognized arrow template"
    return TEMPLATE_TO_DIR[normalized]


def _extract_pattern(room: np.ndarray) -> dict[str, object] | None:
    mask = (room != BG) & (room != RED) & (room != GREY)
    if not bool(mask.any()):
        return None
    assert _mask_connected(mask), "pattern pixels inside a room must be connected"
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    patch = room[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()
    return {
        "patch": patch,
        "top": int(r0),
        "left": int(c0),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push each room's pattern along red arrow links until all patterns accumulate in the terminal room.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    row_bands, col_bands = _room_bands(g)
    room_h = row_bands[0][1] - row_bands[0][0] + 1
    room_w = col_bands[0][1] - col_bands[0][0] + 1

    rooms: dict[tuple[int, int], dict[str, object]] = {}
    terminal_room: tuple[int, int] | None = None

    for ri, (r0, r1) in enumerate(row_bands):
        for ci, (c0, c1) in enumerate(col_bands):
            room = g[r0 : r1 + 1, c0 : c1 + 1]
            direction = _extract_arrow(room)
            pattern = _extract_pattern(room)
            rooms[(ri, ci)] = {
                "slice": (slice(r0, r1 + 1), slice(c0, c1 + 1)),
                "dir": direction,
                "pattern": pattern,
            }
            if direction is None:
                assert terminal_room is None, "there must be exactly one terminal room"
                terminal_room = (ri, ci)

    assert terminal_room is not None, "missing terminal room"

    next_room: dict[tuple[int, int], tuple[int, int] | None] = {}
    for (ri, ci), info in rooms.items():
        direction = info["dir"]
        if direction is None:
            next_room[(ri, ci)] = None
            continue
        dr, dc = direction
        nr, nc = ri + dr, ci + dc
        assert 0 <= nr < len(row_bands) and 0 <= nc < len(col_bands), "arrow must point to an adjacent room"
        next_room[(ri, ci)] = (nr, nc)

    memo: dict[tuple[int, int], int] = {terminal_room: 0}
    visiting: set[tuple[int, int]] = set()

    def distance_to_terminal(room_idx: tuple[int, int]) -> int:
        if room_idx in memo:
            return memo[room_idx]
        assert room_idx not in visiting, "arrow graph must be acyclic"
        visiting.add(room_idx)
        nxt = next_room[room_idx]
        assert nxt is not None, "only the terminal room may omit an arrow"
        value = 1 + distance_to_terminal(nxt)
        visiting.remove(room_idx)
        memo[room_idx] = value
        return value

    pattern_items = []
    anchor_rows: list[int] = []
    anchor_cols: list[int] = []
    for (ri, ci), info in rooms.items():
        pattern = info["pattern"]
        if pattern is None:
            continue
        dist = distance_to_terminal((ri, ci))
        patch = np.asarray(pattern["patch"], dtype=int)
        pattern_items.append((dist, ri, ci, patch))
        anchor_rows.append(int(pattern["top"]))
        anchor_cols.append(int(pattern["left"]))

    out = np.full_like(g, BG)
    out[g == GREY] = GREY
    if not pattern_items:
        return out

    target_top = min(anchor_rows)
    target_left = min(anchor_cols)
    term_rows, term_cols = rooms[terminal_room]["slice"]

    cursor = 0
    for _, ri, ci, patch in sorted(pattern_items, key=lambda item: (item[0], item[1], item[2])):
        patch_h, patch_w = patch.shape
        place_r = int(term_rows.start + target_top)
        place_c = int(term_cols.start + target_left + cursor)
        assert target_top + patch_h <= room_h, "pattern does not fit vertically in the terminal room"
        assert target_left + cursor + patch_w <= room_w, "pattern does not fit horizontally in the terminal room"
        mask = patch != BG
        window = out[place_r : place_r + patch_h, place_c : place_c + patch_w]
        window[mask] = patch[mask]
        out[place_r : place_r + patch_h, place_c : place_c + patch_w] = window
        cursor += patch_w + 1

    return out


def _random_pattern() -> np.ndarray:
    cells = set(random.choice(PATTERN_SHAPES))
    h = max(r for r, _ in cells) + 1
    w = max(c for _, c in cells) + 1
    patch = np.full((h, w), BG, dtype=int)

    occupied = sorted(cells)
    if len(occupied) == 1 or random.random() < 0.55:
        color = random.choice(PATTERN_COLORS)
        for r, c in occupied:
            patch[r, c] = color
        return patch

    colors = random.sample(PATTERN_COLORS, 2)
    accent_count = random.randint(1, len(occupied) - 1)
    accent_cells = set(random.sample(occupied, accent_count))
    for r, c in occupied:
        patch[r, c] = colors[1] if (r, c) in accent_cells else colors[0]
    return patch


def _layout_choices() -> tuple[int, int, int, int]:
    room_rows = random.randint(2, 3)
    room_cols = random.randint(2, 3)

    valid_room_h = [h for h in range(6, 11) if room_rows * h + (room_rows - 1) <= MAX_GRID_SIZE]
    valid_room_w = [w for w in range(7, 13) if room_cols * w + (room_cols - 1) <= MAX_GRID_SIZE]
    room_h = random.choice(valid_room_h)
    room_w = random.choice(valid_room_w)
    return room_rows, room_cols, room_h, room_w


def _build_parent_map(room_rows: int, room_cols: int, terminal: tuple[int, int]) -> dict[tuple[int, int], tuple[int, int]]:
    tr, tc = terminal
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    for ri in range(room_rows):
        for ci in range(room_cols):
            if (ri, ci) == terminal:
                continue
            dist = abs(ri - tr) + abs(ci - tc)
            candidates = []
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = ri + dr, ci + dc
                if 0 <= nr < room_rows and 0 <= nc < room_cols:
                    if abs(nr - tr) + abs(nc - tc) == dist - 1:
                        candidates.append((nr, nc))
            assert candidates
            parent[(ri, ci)] = random.choice(candidates)
    return parent


def _room_origin(ri: int, ci: int, room_h: int, room_w: int) -> tuple[int, int]:
    return ri * (room_h + 1), ci * (room_w + 1)


def _draw_arrow(grid: np.ndarray, top: int, left: int, direction: tuple[int, int]) -> None:
    for dr, dc in ARROW_TEMPLATES[direction]:
        grid[top + dr, left + dc] = RED


def _gen_attempt() -> np.ndarray | None:
    room_rows, room_cols, room_h, room_w = _layout_choices()
    terminal = (random.randrange(room_rows), random.randrange(room_cols))
    parent = _build_parent_map(room_rows, room_cols, terminal)

    directions = {
        (ri, ci): (parent[(ri, ci)][0] - ri, parent[(ri, ci)][1] - ci)
        for ri in range(room_rows)
        for ci in range(room_cols)
        if (ri, ci) != terminal
    }
    if len(set(directions.values())) < 2:
        return None

    depth_buckets: dict[int, list[tuple[int, int]]] = defaultdict(list)
    tr, tc = terminal
    for ri in range(room_rows):
        for ci in range(room_cols):
            depth = abs(ri - tr) + abs(ci - tc)
            depth_buckets[depth].append((ri, ci))

    depths = sorted(depth_buckets)
    max_patterns = min(3 if room_w >= 9 else 2, len(depths))
    if max_patterns < 2:
        return None
    pattern_count = random.randint(2, max_patterns)

    positive_depths = depths[1:]
    if len(positive_depths) >= pattern_count and random.random() < 0.55:
        chosen_depths = sorted(random.sample(positive_depths, pattern_count))
    else:
        chosen_depths = sorted(random.sample(depths, pattern_count))
    pattern_rooms = [random.choice(depth_buckets[depth]) for depth in chosen_depths]

    pattern_top = 1
    pattern_left = 1
    patterns: dict[tuple[int, int], np.ndarray] = {}
    for _ in range(100):
        patterns = {room: _random_pattern() for room in pattern_rooms}
        widths = [patch.shape[1] for patch in patterns.values()]
        heights = [patch.shape[0] for patch in patterns.values()]
        total_width = sum(widths) + (len(widths) - 1)
        if pattern_left + total_width > room_w:
            continue
        if pattern_top + max(heights) > room_h:
            continue
        signatures = {patch.tobytes() for patch in patterns.values()}
        if len(signatures) != len(patterns):
            continue
        break
    else:
        return None

    total_h = room_rows * room_h + (room_rows - 1)
    total_w = room_cols * room_w + (room_cols - 1)
    grid = np.full((total_h, total_w), BG, dtype=int)

    for r in range(room_h, total_h, room_h + 1):
        grid[r, :] = GREY
    for c in range(room_w, total_w, room_w + 1):
        grid[:, c] = GREY

    arrow_top = room_h - 4
    arrow_left = room_w - 4

    for ri in range(room_rows):
        for ci in range(room_cols):
            base_r, base_c = _room_origin(ri, ci, room_h, room_w)
            room = (ri, ci)
            if room != terminal:
                _draw_arrow(grid, base_r + arrow_top, base_c + arrow_left, directions[room])
            if room in patterns:
                patch = patterns[room]
                h, w = patch.shape
                window = grid[
                    base_r + pattern_top : base_r + pattern_top + h,
                    base_c + pattern_left : base_c + pattern_left + w,
                ]
                mask = patch != BG
                window[mask] = patch[mask]
                grid[
                    base_r + pattern_top : base_r + pattern_top + h,
                    base_c + pattern_left : base_c + pattern_left + w,
                ] = window

    try:
        out = main(grid)
    except AssertionError:
        return None

    term_r0, term_c0 = _room_origin(terminal[0], terminal[1], room_h, room_w)
    terminal_view = out[term_r0 : term_r0 + room_h, term_c0 : term_c0 + room_w]
    if np.count_nonzero(terminal_view == RED) != 0:
        return None
    if np.count_nonzero(out != BG) <= np.count_nonzero(grid == GREY):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
