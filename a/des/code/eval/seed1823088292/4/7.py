from __future__ import annotations

import random

import numpy as np


BG = 0
MAROON = 9
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8]
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
CLOCK_DIRECTIONS = {
    (0, 1): 0,  # up -> 0
    (1, 2): 1,  # right -> cw 90
    (2, 1): 2,  # down -> 180
    (1, 0): 3,  # left -> cw 270
}
SHAPE_BBOXES = [(2, 2), (2, 4), (4, 2), (3, 3)]


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _crop_mask(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask, (r0, c0, r1, c1)


def _rotate_coords(coords: np.ndarray, cw_turns: int) -> np.ndarray:
    mask, (r0, c0, _r1, _c1) = _crop_mask(coords)
    rotated = np.rot90(mask, (-cw_turns) % 4)
    h, w = mask.shape
    hr, wr = rotated.shape
    nr0 = r0 + (h - hr) // 2
    nc0 = c0 + (w - wr) // 2
    out = np.argwhere(rotated).astype(int)
    out[:, 0] += nr0
    out[:, 1] += nc0
    return out


def _clock_info(patch: np.ndarray) -> tuple[int, int] | None:
    if patch.shape != (3, 3):
        return None
    if int(patch[1, 1]) != BG:
        return None

    pointer_positions = [pos for pos in CLOCK_DIRECTIONS if int(patch[pos]) == MAROON]
    if len(pointer_positions) != 1:
        return None

    pointer = pointer_positions[0]
    ring_positions = [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 2),
        (2, 0),
        (2, 1),
        (2, 2),
    ]
    color_positions = [pos for pos in ring_positions if pos != pointer]
    colors = {int(patch[pos]) for pos in color_positions}
    if len(colors) != 1:
        return None
    color = next(iter(colors))
    if color in (BG, MAROON):
        return None
    return color, CLOCK_DIRECTIONS[pointer]


def _find_clocks(grid: np.ndarray) -> list[dict[str, int]]:
    h, w = grid.shape
    clocks: list[dict[str, int]] = []
    for r in range(h - 2):
        for c in range(w - 2):
            info = _clock_info(grid[r : r + 3, c : c + 3])
            if info is None:
                continue
            color, turns = info
            clocks.append({"r": int(r), "c": int(c), "color": int(color), "turns": int(turns)})
    return clocks


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each target shape by the angle encoded by the same-colored clock.

    Input structure:
    - bg = black.
    - Each clock is a 3x3 square border of color C with a black center and one
      maroon side-midpoint replacing the border color.
    - Each foreground color C appears exactly once as a clock border and once as
      a same-colored target shape elsewhere in the grid.

    Rule:
    - Match clocks to target shapes by color.
    - Read the maroon pointer direction:
      up = 0, right = cw 90, down = 180, left = cw 270.
    - Rotate the target shape around its bbox center and place the rotated shape
      back center-aligned to the original bbox.
    - Erase all clocks.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    clocks = _find_clocks(g)

    clock_by_color: dict[int, dict[str, int]] = {}
    clock_mask = np.zeros(g.shape, dtype=bool)
    for clock in clocks:
        color = int(clock["color"])
        if color in clock_by_color:
            continue
        clock_by_color[color] = clock
        r = int(clock["r"])
        c = int(clock["c"])
        clock_mask[r : r + 3, c : c + 3] = True

    out[clock_mask] = BG

    target_coords_by_color: dict[int, np.ndarray] = {}
    for color in clock_by_color:
        coords = np.argwhere((g == int(color)) & (~clock_mask))
        if coords.size == 0:
            continue
        target_coords_by_color[int(color)] = coords
        out[coords[:, 0], coords[:, 1]] = BG

    for color, coords in target_coords_by_color.items():
        rotated = _rotate_coords(coords, int(clock_by_color[color]["turns"]))
        out[rotated[:, 0], rotated[:, 1]] = int(color)

    return out


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def _neighbors4_in_shape(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out = []
    for dr, dc in NEIGHBORS4:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out.append((nr, nc))
    return out


def _is_clock_like_mask(mask: np.ndarray) -> bool:
    if mask.shape != (3, 3) or int(mask.sum()) != 7:
        return False
    ring = np.array(
        [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ],
        dtype=bool,
    )
    for pos in CLOCK_DIRECTIONS:
        candidate = ring.copy()
        candidate[pos] = False
        if np.array_equal(mask, candidate):
            return True
    return False


def _random_shape() -> np.ndarray | None:
    for _ in range(200):
        h, w = random.choice(SHAPE_BBOXES)
        area = random.randint(3, min(8, h * w))

        mask = np.zeros((h, w), dtype=bool)
        start = (random.randrange(h), random.randrange(w))
        mask[start] = True
        frontier = set(_neighbors4_in_shape(start[0], start[1], h, w))

        while int(mask.sum()) < area:
            if not frontier:
                break
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            mask[cell] = True
            for nb in _neighbors4_in_shape(cell[0], cell[1], h, w):
                if not mask[nb]:
                    frontier.add(nb)

        if int(mask.sum()) != area:
            continue
        if not np.all(mask.any(axis=0)) or not np.all(mask.any(axis=1)):
            continue
        if any(np.array_equal(mask, np.rot90(mask, k)) for k in (1, 2, 3)):
            continue
        if _is_clock_like_mask(mask):
            continue
        return np.rot90(mask, random.randrange(4))
    return None


def _find_slot(used: np.ndarray, rh: int, rw: int, pad: int = 1) -> tuple[int, int] | None:
    h, w = used.shape
    attempts = 300
    for _ in range(attempts):
        r0 = random.randint(0, h - rh)
        c0 = random.randint(0, w - rw)
        rr0 = max(0, r0 - pad)
        cc0 = max(0, c0 - pad)
        rr1 = min(h, r0 + rh + pad)
        cc1 = min(w, c0 + rw + pad)
        if not used[rr0:rr1, cc0:cc1].any():
            used[rr0:rr1, cc0:cc1] = True
            return r0, c0
    return None


def _clock_patch(color: int, turns: int) -> np.ndarray:
    patch = np.full((3, 3), color, dtype=int)
    patch[1, 1] = BG
    pointer = {
        0: (0, 1),
        1: (1, 2),
        2: (2, 1),
        3: (1, 0),
    }[int(turns)]
    patch[pointer] = MAROON
    return patch


def _balanced_turns(count: int) -> list[int]:
    turns: list[int] = []
    while len(turns) < count:
        batch = [0, 1, 2, 3]
        random.shuffle(batch)
        turns.extend(batch)
    return turns[:count]


def _paint_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    coords = np.argwhere(mask)
    grid[r0 + coords[:, 0], c0 + coords[:, 1]] = int(color)


def _generate_case() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(18, 28)
    w = random.randint(18, 28)
    pair_count = random.randint(2, 5)
    colors = random.sample(PALETTE, pair_count)
    turns = _balanced_turns(pair_count)

    shapes: list[dict[str, object]] = []
    for color, turn in zip(colors, turns):
        mask = _random_shape()
        if mask is None:
            return None
        sh, sw = mask.shape
        reserve = max(sh, sw)
        shapes.append(
            {
                "color": int(color),
                "turn": int(turn),
                "mask": mask,
                "h": int(sh),
                "w": int(sw),
                "reserve": int(reserve),
            }
        )

    grid = np.full((h, w), BG, dtype=int)
    expected = np.full((h, w), BG, dtype=int)
    used = np.zeros((h, w), dtype=bool)

    shape_order = sorted(range(pair_count), key=lambda i: int(shapes[i]["reserve"]), reverse=True)
    for idx in shape_order:
        shape = shapes[idx]
        reserve = int(shape["reserve"])
        slot = _find_slot(used, reserve, reserve, pad=1)
        if slot is None:
            return None
        sr, sc = slot
        r0 = sr + (reserve - int(shape["h"])) // 2
        c0 = sc + (reserve - int(shape["w"])) // 2
        shape["input_r0"] = int(r0)
        shape["input_c0"] = int(c0)
        _paint_mask(grid, shape["mask"], int(r0), int(c0), int(shape["color"]))

        coords = np.argwhere(np.asarray(shape["mask"], dtype=bool)).astype(int)
        coords[:, 0] += int(r0)
        coords[:, 1] += int(c0)
        rotated = _rotate_coords(coords, int(shape["turn"]))
        expected[rotated[:, 0], rotated[:, 1]] = int(shape["color"])

    for shape in shapes:
        slot = _find_slot(used, 3, 3, pad=1)
        if slot is None:
            return None
        cr, cc = slot
        shape["clock_r"] = int(cr)
        shape["clock_c"] = int(cc)
        grid[cr : cr + 3, cc : cc + 3] = _clock_patch(int(shape["color"]), int(shape["turn"]))

    found = _find_clocks(grid)
    if len(found) != pair_count:
        return None
    if {int(item["color"]) for item in found} != set(colors):
        return None

    if not np.array_equal(main(grid), expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(400):
        case = _generate_case()
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid clock-rotation input")
