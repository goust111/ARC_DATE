from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8
MAROON = 9

PAYLOAD_COLORS = [BLUE, RED, GREEN, MAGENTA, AZURE, MAROON]

DIRS: dict[str, tuple[int, int]] = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

CHAIN_SPECS = [
    {
        "shape": (5, 5),
        "payload": (1, 1),
        "target": (3, 2),
        "stones": [(0, 1), (4, 0), (4, 4)],
    },
    {
        "shape": (5, 5),
        "payload": (1, 2),
        "target": (2, 1),
        "stones": [(0, 2), (3, 4), (3, 0)],
    },
]

MAX_GENERATE_ATTEMPTS = 300


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _payload_cells(top_left: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = top_left
    return [(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)]


def _adjacent_positions(top_left: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = top_left
    return {
        (r - 1, c),
        (r - 1, c + 1),
        (r + 2, c),
        (r + 2, c + 1),
        (r, c - 1),
        (r + 1, c - 1),
        (r, c + 2),
        (r + 1, c + 2),
    }


def _adjacent_stones(
    top_left: tuple[int, int], stones: set[tuple[int, int]]
) -> list[tuple[int, int]]:
    return sorted(_adjacent_positions(top_left) & stones)


def _find_payload(
    g: np.ndarray, room_cells: list[tuple[int, int]]
) -> tuple[tuple[int, int], int]:
    special = {BLACK, YELLOW, GREY, ORANGE}
    by_color: dict[int, list[tuple[int, int]]] = {}
    for r, c in room_cells:
        color = int(g[r, c])
        if color not in special:
            by_color.setdefault(color, []).append((r, c))
    assert len(by_color) == 1, "each room must contain exactly one payload color"
    color, cells = next(iter(by_color.items()))
    assert len(cells) == 4, "payload must have area 4"
    rows = sorted({r for r, _c in cells})
    cols = sorted({c for _r, c in cells})
    assert len(rows) == 2 and len(cols) == 2, "payload must be 2x2"
    top_left = (rows[0], cols[0])
    assert set(cells) == set(_payload_cells(top_left)), "payload must be a solid 2x2 block"
    return top_left, color


def _infer_stone_directions(
    g: np.ndarray,
    stones: set[tuple[int, int]],
    payload_top_left: tuple[int, int],
) -> dict[tuple[int, int], str]:
    payload_cells = set(_payload_cells(payload_top_left))
    h, w = g.shape
    directions: dict[tuple[int, int], str] = {}
    for stone in stones:
        sr, sc = stone
        hits: list[str] = []
        for name, (dr, dc) in DIRS.items():
            r, c = sr + dr, sc + dc
            while 0 <= r < h and 0 <= c < w:
                value = int(g[r, c])
                if value != BLACK:
                    if (r, c) in stones or (r, c) in payload_cells:
                        hits.append(name)
                    break
                r += dr
                c += dc
        assert len(hits) == 1, "each guide stone must have exactly one pointing direction"
        directions[stone] = hits[0]
    return directions


def _room_mask(room_cells: list[tuple[int, int]]) -> set[tuple[int, int]]:
    return set(room_cells)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transport each room's 2x2 payload along the guide-stone chain until its top-left reaches the yellow target.

    Input structure:
    - bg = black.
    - Grey walls partition the grid into closed rooms.
    - Each room contains orange singleton guide stones, one 2x2 payload block in a non-special color, and one yellow target pixel.
    - Every guide stone has exactly one orthogonal ray whose first non-background hit is another guide stone or the payload.

    Rule:
    - Solve each room independently.
    - Infer every guide stone's direction from that unique ray.
    - From the current payload position, take the unique orthogonally adjacent guide stone and move the payload one cell at a time in that direction until the moved payload first becomes adjacent to another guide stone, or until one more step would hit a wall.
    - Repeat until the payload's top-left reaches the yellow target; erase all guide stones and the original target, move the payload to its final position, and keep the walls unchanged.

    Key insight:
    - The orange stones do not store directions by color or shape: the direction comes from ray-casting, and the route comes from chaining those inferred directions through successive stopping contacts.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    room_components = _components(g != GREY)
    out = g.copy()

    for room_cells in room_components:
        payload_top_left, payload_color = _find_payload(g, room_cells)
        room_set = _room_mask(room_cells)
        stones = {(r, c) for r, c in room_cells if int(g[r, c]) == ORANGE}
        targets = [(r, c) for r, c in room_cells if int(g[r, c]) == YELLOW]
        assert len(targets) == 1, "each room must contain one target"
        target = targets[0]
        stone_dirs = _infer_stone_directions(g, stones, payload_top_left)

        current = payload_top_left
        for _ in range(len(stones) + 2):
            if current == target:
                break
            adjacent = _adjacent_stones(current, stones)
            assert len(adjacent) == 1, "payload must have exactly one active adjacent guide stone"
            dr, dc = DIRS[stone_dirs[adjacent[0]]]
            while True:
                nxt = (current[0] + dr, current[1] + dc)
                next_cells = _payload_cells(nxt)
                if any(cell not in room_set or cell in stones for cell in next_cells):
                    assert current == target, "wall stop must coincide with the target"
                    break
                current = nxt
                if current == target or _adjacent_stones(current, stones):
                    break
            if current == target:
                break
        else:
            raise AssertionError("payload path did not terminate")

        assert current == target, "payload must finish with its top-left on the target"
        for r, c in room_cells:
            if int(out[r, c]) == ORANGE or int(out[r, c]) == YELLOW or int(out[r, c]) == payload_color:
                out[r, c] = BLACK
        for r, c in _payload_cells(current):
            out[r, c] = payload_color

    return out


def _place_patch(room: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    mask = patch != BLACK
    room[r0 : r0 + h, c0 : c0 + w][mask] = patch[mask]


def _render_chain_patch(spec: dict[str, object], payload_color: int) -> np.ndarray:
    h, w = spec["shape"]
    patch = np.full((h, w), BLACK, dtype=int)
    pr, pc = spec["payload"]
    patch[pr : pr + 2, pc : pc + 2] = payload_color
    tr, tc = spec["target"]
    patch[tr, tc] = YELLOW
    for sr, sc in spec["stones"]:
        patch[sr, sc] = ORANGE
    return patch


def _random_chain_room(h: int, w: int, payload_color: int) -> np.ndarray:
    spec = random.choice(CHAIN_SPECS)
    patch = _render_chain_patch(spec, payload_color)
    if random.random() < 0.5:
        patch = np.fliplr(patch)
    patch = np.rot90(patch, random.randrange(4)).copy()
    ph, pw = patch.shape
    assert ph <= h and pw <= w
    room = np.full((h, w), BLACK, dtype=int)
    r0 = random.randint(0, h - ph)
    c0 = random.randint(0, w - pw)
    _place_patch(room, patch, r0, c0)
    return room


def _random_single_step_room(h: int, w: int, payload_color: int) -> np.ndarray:
    room = np.full((h, w), BLACK, dtype=int)
    direction = random.choice(list(DIRS))

    if direction in {"left", "right"}:
        pr = random.randint(0, h - 2)
        offset = random.randint(0, 1)
        if direction == "right":
            target_c = w - 2
            start_c = random.randint(1, target_c - 1)
            room[pr : pr + 2, start_c : start_c + 2] = payload_color
            room[pr + offset, start_c - 1] = ORANGE
            room[pr, target_c] = YELLOW
        else:
            target_c = 0
            start_c = random.randint(1, w - 3)
            room[pr : pr + 2, start_c : start_c + 2] = payload_color
            room[pr + offset, start_c + 2] = ORANGE
            room[pr, target_c] = YELLOW
    else:
        pc = random.randint(0, w - 2)
        offset = random.randint(0, 1)
        if direction == "down":
            target_r = h - 2
            start_r = random.randint(1, target_r - 1)
            room[start_r : start_r + 2, pc : pc + 2] = payload_color
            room[start_r - 1, pc + offset] = ORANGE
            room[target_r, pc] = YELLOW
        else:
            target_r = 0
            start_r = random.randint(1, h - 3)
            room[start_r : start_r + 2, pc : pc + 2] = payload_color
            room[start_r + 2, pc + offset] = ORANGE
            room[target_r, pc] = YELLOW

    return room


def _room_colors(count: int) -> list[int]:
    if count <= len(PAYLOAD_COLORS):
        return random.sample(PAYLOAD_COLORS, count)
    return [random.choice(PAYLOAD_COLORS) for _ in range(count)]


def _layout_choice() -> tuple[int, int]:
    return random.choice([(1, 1), (1, 2), (2, 1), (2, 2), (1, 3), (3, 1)])


def _build_grid() -> np.ndarray:
    n_rows, n_cols = _layout_choice()
    row_heights = [random.randint(5, 8) for _ in range(n_rows)]
    col_widths = [random.randint(5, 8) for _ in range(n_cols)]
    height = sum(row_heights) + n_rows + 1
    width = sum(col_widths) + n_cols + 1
    assert height <= 30 and width <= 30

    grid = np.full((height, width), GREY, dtype=int)
    room_count = n_rows * n_cols
    colors = _room_colors(room_count)

    room_index = 0
    r0 = 1
    for rh in row_heights:
        c0 = 1
        for cw in col_widths:
            payload_color = colors[room_index]
            if random.random() < 0.5:
                room = _random_single_step_room(rh, cw, payload_color)
            else:
                room = _random_chain_room(rh, cw, payload_color)
            grid[r0 : r0 + rh, c0 : c0 + cw] = room
            room_index += 1
            c0 += cw + 1
        r0 += rh + 1

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATE_ATTEMPTS):
        grid = _build_grid()
        try:
            _ = main(grid)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid guide-stone transport puzzle")
