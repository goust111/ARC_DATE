import random
from collections import defaultdict

import numpy as np


BG = 0
BRIDGE = 5
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _extract_rectangles(
    grid: np.ndarray,
) -> tuple[list[dict], dict[tuple[int, int], int]]:
    rectangles: list[dict] = []
    rect_at: dict[tuple[int, int], int] = {}
    colors = [int(c) for c in np.unique(grid) if int(c) not in (BG, BRIDGE)]
    for color in sorted(colors):
        comps = _components(grid == color)
        assert len(comps) == 1, "each rectangle color must appear in one component"
        coords = comps[0]
        r0, c0, r1, c1 = _bbox(coords)
        assert len(coords) == (r1 - r0 + 1) * (c1 - c0 + 1), "component must be a solid rectangle"
        assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color), "rectangle bbox must be filled"
        rect_id = len(rectangles)
        rect = {
            "id": rect_id,
            "color": color,
            "bbox": (r0, c0, r1, c1),
            "area": len(coords),
            "coords": coords,
        }
        rectangles.append(rect)
        for coord in coords:
            rect_at[coord] = rect_id
    return rectangles, rect_at


def _adjacent_rect_ids(
    coords: list[tuple[int, int]],
    rect_at: dict[tuple[int, int], int],
    shape: tuple[int, int],
) -> set[int]:
    h, w = shape
    ids: set[int] = set()
    for r, c in coords:
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                rect_id = rect_at.get((nr, nc))
                if rect_id is not None:
                    ids.add(rect_id)
    return ids


def _extract_bridges(grid: np.ndarray, rect_at: dict[tuple[int, int], int]) -> list[dict]:
    bridges: list[dict] = []
    attached: dict[int, int] = {}
    for coords in _components(grid == BRIDGE):
        rows = sorted({r for r, _ in coords})
        cols = sorted({c for _, c in coords})
        if len(rows) == 1:
            r = rows[0]
            c0, c1 = cols[0], cols[-1]
            assert len(coords) == c1 - c0 + 1, "horizontal bridge must be contiguous"
            assert all(int(grid[r, c]) == BRIDGE for c in range(c0, c1 + 1))
            left_id = rect_at.get((r, c0 - 1))
            right_id = rect_at.get((r, c1 + 1))
            assert left_id is not None and right_id is not None and left_id != right_id
            touched = _adjacent_rect_ids(coords, rect_at, grid.shape)
            assert touched == {left_id, right_id}, "bridge must touch exactly its two rectangles"
            bridge = {
                "orientation": "horizontal",
                "length": len(coords),
                "left": left_id,
                "right": right_id,
            }
        else:
            assert len(cols) == 1, "bridge must be horizontal or vertical"
            c = cols[0]
            r0, r1 = rows[0], rows[-1]
            assert len(coords) == r1 - r0 + 1, "vertical bridge must be contiguous"
            assert all(int(grid[r, c]) == BRIDGE for r in range(r0, r1 + 1))
            top_id = rect_at.get((r0 - 1, c))
            bottom_id = rect_at.get((r1 + 1, c))
            assert top_id is not None and bottom_id is not None and top_id != bottom_id
            touched = _adjacent_rect_ids(coords, rect_at, grid.shape)
            assert touched == {top_id, bottom_id}, "bridge must touch exactly its two rectangles"
            bridge = {
                "orientation": "vertical",
                "length": len(coords),
                "top": top_id,
                "bottom": bottom_id,
            }

        for rect_id in touched:
            assert rect_id not in attached, "each rectangle may connect to at most one bridge"
            attached[rect_id] = len(bridges)
        bridges.append(bridge)
    return bridges


def _winner_loser(rect_a: dict, rect_b: dict) -> tuple[int, int]:
    area_a = int(rect_a["area"])
    area_b = int(rect_b["area"])
    if area_a != area_b:
        if area_a > area_b:
            return int(rect_a["id"]), int(rect_b["id"])
        return int(rect_b["id"]), int(rect_a["id"])

    key_a = (int(rect_a["bbox"][0]), int(rect_a["bbox"][1]))
    key_b = (int(rect_b["bbox"][0]), int(rect_b["bbox"][1]))
    if key_a <= key_b:
        return int(rect_a["id"]), int(rect_b["id"])
    return int(rect_b["id"]), int(rect_a["id"])


def _paint_extension(out: np.ndarray, rect: dict, direction: str, length: int) -> None:
    r0, c0, r1, c1 = map(int, rect["bbox"])
    color = int(rect["color"])
    if direction == "left":
        assert c0 - length >= 0
        out[r0 : r1 + 1, c0 - length : c0] = color
    elif direction == "right":
        assert c1 + length < out.shape[1]
        out[r0 : r1 + 1, c1 + 1 : c1 + 1 + length] = color
    elif direction == "up":
        assert r0 - length >= 0
        out[r0 - length : r0, c0 : c1 + 1] = color
    else:
        assert direction == "down"
        assert r1 + length < out.shape[0]
        out[r1 + 1 : r1 + 1 + length, c0 : c1 + 1] = color


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    rectangles, rect_at = _extract_rectangles(g)
    bridges = _extract_bridges(g, rect_at)

    out = g.copy()
    out[out == BRIDGE] = BG

    for bridge in bridges:
        if bridge["orientation"] == "horizontal":
            rect_a = rectangles[int(bridge["left"])]
            rect_b = rectangles[int(bridge["right"])]
            winner_id, loser_id = _winner_loser(rect_a, rect_b)
            direction = "right" if winner_id == int(bridge["left"]) else "left"
        else:
            rect_a = rectangles[int(bridge["top"])]
            rect_b = rectangles[int(bridge["bottom"])]
            winner_id, loser_id = _winner_loser(rect_a, rect_b)
            direction = "down" if winner_id == int(bridge["top"]) else "up"

        for r, c in rectangles[loser_id]["coords"]:
            out[r, c] = BG
        _paint_extension(out, rectangles[winner_id], direction, int(bridge["length"]))

    return out


GRID_H_RANGE = (16, 28)
GRID_W_RANGE = (16, 28)
PAIR_TRIES = 120
EXTRA_TRIES = 120
GEN_RETRIES = 400
BRIDGE_LEN_RANGE = (1, 5)
RECT_H_RANGE = (2, 5)
RECT_W_RANGE = (2, 5)
COLOR_POOL = tuple(color for color in range(10) if color not in (BG, BRIDGE))

RECT_DIMS = [
    (h, w)
    for h in range(RECT_H_RANGE[0], RECT_H_RANGE[1] + 1)
    for w in range(RECT_W_RANGE[0], RECT_W_RANGE[1] + 1)
]
AREA_TO_DIMS: dict[int, list[tuple[int, int]]] = defaultdict(list)
for dims in RECT_DIMS:
    AREA_TO_DIMS[dims[0] * dims[1]].append(dims)
EQUAL_AREAS = sorted(AREA_TO_DIMS)
UNEQUAL_DIM_PAIRS = [
    (big, small)
    for big in RECT_DIMS
    for small in RECT_DIMS
    if big[0] * big[1] > small[0] * small[1]
]


def _balanced_flags(rng: random.Random, count: int) -> list[bool]:
    true_count = count // 2
    if count % 2 == 1 and rng.random() < 0.5:
        true_count += 1
    flags = [True] * true_count + [False] * (count - true_count)
    rng.shuffle(flags)
    return flags


def _bbox_free(occupied: np.ndarray, r0: int, r1: int, c0: int, c1: int) -> bool:
    rr0 = max(0, r0 - 1)
    rr1 = min(occupied.shape[0] - 1, r1 + 1)
    cc0 = max(0, c0 - 1)
    cc1 = min(occupied.shape[1] - 1, c1 + 1)
    return not occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any()


def _mark_bbox(occupied: np.ndarray, r0: int, r1: int, c0: int, c1: int) -> None:
    rr0 = max(0, r0 - 1)
    rr1 = min(occupied.shape[0] - 1, r1 + 1)
    cc0 = max(0, c0 - 1)
    cc1 = min(occupied.shape[1] - 1, c1 + 1)
    occupied[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _choose_dims(
    rng: random.Random,
    tie: bool,
    winner_first: bool,
) -> tuple[tuple[int, int], tuple[int, int]]:
    if tie:
        area = rng.choice(EQUAL_AREAS)
        return rng.choice(AREA_TO_DIMS[area]), rng.choice(AREA_TO_DIMS[area])

    big, small = rng.choice(UNEQUAL_DIM_PAIRS)
    if winner_first:
        return big, small
    return small, big


def _horizontal_offsets(
    rng: random.Random,
    h_left: int,
    h_right: int,
    tie: bool,
    left_wins_tie: bool,
) -> tuple[int, int]:
    choices: list[tuple[int, int]] = []
    for off_left in range(h_left):
        for off_right in range(h_right):
            if not tie:
                choices.append((off_left, off_right))
            elif left_wins_tie and off_left >= off_right:
                choices.append((off_left, off_right))
            elif not left_wins_tie and off_right > off_left:
                choices.append((off_left, off_right))
    assert choices
    return rng.choice(choices)


def _draw_rect(grid: np.ndarray, top: int, left: int, height: int, width: int, color: int) -> None:
    grid[top : top + height, left : left + width] = color


def _place_horizontal_pair(
    grid: np.ndarray,
    occupied: np.ndarray,
    rng: random.Random,
    tie: bool,
    color_left: int,
    color_right: int,
) -> bool:
    left_wins = rng.random() < 0.5
    left_dims, right_dims = _choose_dims(rng, tie=tie, winner_first=left_wins)
    h_left, w_left = left_dims
    h_right, w_right = right_dims
    bridge_len = rng.randint(BRIDGE_LEN_RANGE[0], BRIDGE_LEN_RANGE[1])

    if tie:
        left_wins_tie = rng.random() < 0.5
    else:
        left_wins_tie = left_wins
    off_left, off_right = _horizontal_offsets(rng, h_left, h_right, tie, left_wins_tie)
    bridge_row = max(off_left, off_right)
    top_left = bridge_row - off_left
    top_right = bridge_row - off_right
    pair_height = max(top_left + h_left, top_right + h_right)
    pair_width = w_left + bridge_len + w_right

    if pair_height + 2 > grid.shape[0] or pair_width + 2 > grid.shape[1]:
        return False

    max_r0 = grid.shape[0] - pair_height - 1
    max_c0 = grid.shape[1] - pair_width - 1
    if max_r0 < 1 or max_c0 < 1:
        return False

    for _ in range(PAIR_TRIES):
        r0 = rng.randint(1, max_r0)
        c0 = rng.randint(1, max_c0)
        r1 = r0 + pair_height - 1
        c1 = c0 + pair_width - 1
        if not _bbox_free(occupied, r0, r1, c0, c1):
            continue

        left_top = r0 + top_left
        right_top = r0 + top_right
        left_col = c0
        right_col = c0 + w_left + bridge_len
        _draw_rect(grid, left_top, left_col, h_left, w_left, color_left)
        _draw_rect(grid, right_top, right_col, h_right, w_right, color_right)
        grid[r0 + bridge_row, c0 + w_left : c0 + w_left + bridge_len] = BRIDGE
        _mark_bbox(occupied, r0, r1, c0, c1)
        return True
    return False


def _place_vertical_pair(
    grid: np.ndarray,
    occupied: np.ndarray,
    rng: random.Random,
    tie: bool,
    color_top: int,
    color_bottom: int,
) -> bool:
    top_wins = True if tie else rng.random() < 0.5
    top_dims, bottom_dims = _choose_dims(rng, tie=tie, winner_first=top_wins)
    h_top, w_top = top_dims
    h_bottom, w_bottom = bottom_dims
    bridge_len = rng.randint(BRIDGE_LEN_RANGE[0], BRIDGE_LEN_RANGE[1])

    off_top = rng.randrange(w_top)
    off_bottom = rng.randrange(w_bottom)
    bridge_col = max(off_top, off_bottom)
    left_top = bridge_col - off_top
    left_bottom = bridge_col - off_bottom
    pair_height = h_top + bridge_len + h_bottom
    pair_width = max(left_top + w_top, left_bottom + w_bottom)

    if pair_height + 2 > grid.shape[0] or pair_width + 2 > grid.shape[1]:
        return False

    max_r0 = grid.shape[0] - pair_height - 1
    max_c0 = grid.shape[1] - pair_width - 1
    if max_r0 < 1 or max_c0 < 1:
        return False

    for _ in range(PAIR_TRIES):
        r0 = rng.randint(1, max_r0)
        c0 = rng.randint(1, max_c0)
        r1 = r0 + pair_height - 1
        c1 = c0 + pair_width - 1
        if not _bbox_free(occupied, r0, r1, c0, c1):
            continue

        top_col = c0 + left_top
        bottom_col = c0 + left_bottom
        top_row = r0
        bottom_row = r0 + h_top + bridge_len
        _draw_rect(grid, top_row, top_col, h_top, w_top, color_top)
        _draw_rect(grid, bottom_row, bottom_col, h_bottom, w_bottom, color_bottom)
        grid[r0 + h_top : r0 + h_top + bridge_len, c0 + bridge_col] = BRIDGE
        _mark_bbox(occupied, r0, r1, c0, c1)
        return True
    return False


def _place_free_rectangle(
    grid: np.ndarray,
    occupied: np.ndarray,
    rng: random.Random,
    color: int,
) -> bool:
    height, width = rng.choice(RECT_DIMS)
    if height + 2 > grid.shape[0] or width + 2 > grid.shape[1]:
        return False

    max_r0 = grid.shape[0] - height - 1
    max_c0 = grid.shape[1] - width - 1
    if max_r0 < 1 or max_c0 < 1:
        return False

    for _ in range(EXTRA_TRIES):
        r0 = rng.randint(1, max_r0)
        c0 = rng.randint(1, max_c0)
        r1 = r0 + height - 1
        c1 = c0 + width - 1
        if not _bbox_free(occupied, r0, r1, c0, c1):
            continue
        _draw_rect(grid, r0, c0, height, width, color)
        _mark_bbox(occupied, r0, r1, c0, c1)
        return True
    return False


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(GRID_H_RANGE[0], GRID_H_RANGE[1])
    w = rng.randint(GRID_W_RANGE[0], GRID_W_RANGE[1])
    grid = np.full((h, w), BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    max_pairs = 1
    if min(h, w) >= 20:
        max_pairs = 2
    if min(h, w) >= 24:
        max_pairs = 3
    pair_count = rng.randint(1, max_pairs)

    max_extra = min(2, len(COLOR_POOL) - 2 * pair_count)
    extra_count = rng.randint(0, max_extra)
    colors = rng.sample(COLOR_POOL, 2 * pair_count + extra_count)

    orientation_flags = _balanced_flags(rng, pair_count)
    tie_flags = _balanced_flags(rng, pair_count)

    color_idx = 0
    for use_horizontal, tie in zip(orientation_flags, tie_flags):
        color_a = colors[color_idx]
        color_b = colors[color_idx + 1]
        color_idx += 2
        if use_horizontal:
            ok = _place_horizontal_pair(grid, occupied, rng, tie, color_a, color_b)
        else:
            ok = _place_vertical_pair(grid, occupied, rng, tie, color_a, color_b)
        if not ok:
            return None

    for _ in range(extra_count):
        color = colors[color_idx]
        color_idx += 1
        if not _place_free_rectangle(grid, occupied, rng, color):
            return None

    try:
        out = main(grid)
        assert out.shape == grid.shape
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
