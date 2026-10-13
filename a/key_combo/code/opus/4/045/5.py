from __future__ import annotations

import random

import numpy as np


BLACK = 0
RED = 2
GREY = 5
ORANGE = 7
TEAL = 8

DIRS_URDL = [(-1, 0), (0, 1), (1, 0), (0, -1)]
CYCLE = (GREY, RED, GREY, BLACK)
GEN_RETRIES = 400
ARM_PATTERNS = [
    (2, 2, 2, 2),
    (1, 2, 2, 2),
    (2, 1, 2, 2),
    (2, 2, 1, 2),
    (2, 2, 2, 1),
    (1, 2, 1, 2),
    (2, 1, 2, 1),
    (1, 2, 2, 1),
    (2, 1, 1, 2),
]
SIDE_CHOICES = (19, 21)


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def bbox(mask: np.ndarray):
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def right_dir(drc: tuple[int, int]) -> tuple[int, int]:
    return DIRS_URDL[(DIRS_URDL.index(drc) + 1) % 4]


def left_dir(drc: tuple[int, int]) -> tuple[int, int]:
    return DIRS_URDL[(DIRS_URDL.index(drc) - 1) % 4]


def frame_info(grid: np.ndarray) -> tuple[int, int, int, int, int]:
    frame_mask = grid == GREY
    box = bbox(frame_mask)
    assert box is not None, "expected one grey frame"
    r0, c0, r1, c1 = box
    side = r1 - r0 + 1
    assert side == c1 - c0 + 1 and side % 2 == 1 and side >= 7
    expected = np.zeros_like(frame_mask)
    expected[r0 : r1 + 1, c0] = True
    expected[r0 : r1 + 1, c1] = True
    expected[r0, c0 : c1 + 1] = True
    expected[r1, c0 : c1 + 1] = True
    assert np.array_equal(frame_mask, expected), "frame must be a complete 1-cell square"
    return r0, c0, r1, c1, side


def cross_info(grid: np.ndarray, box: tuple[int, int, int, int]):
    r0, c0, r1, c1 = box
    cr = (r0 + r1) // 2
    cc = (c0 + c1) // 2
    teal_mask = grid == TEAL
    coords = np.argwhere(teal_mask)
    assert coords.size > 0, "expected teal cross"
    assert teal_mask[cr, cc], "cross must pass through frame center"
    assert np.all((coords[:, 0] == cr) | (coords[:, 1] == cc)), "teal must form a cross"

    arm_lengths = []
    expected = {(cr, cc)}
    for dr, dc in DIRS_URDL:
        length = 0
        rr, cc2 = cr + dr, cc + dc
        while in_bounds(grid.shape, rr, cc2) and teal_mask[rr, cc2]:
            length += 1
            expected.add((rr, cc2))
            rr += dr
            cc2 += dc
        assert 1 <= length <= 2, "each arm length must be 1 or 2"
        arm_lengths.append(length)

    actual = {tuple(map(int, rc)) for rc in coords}
    assert actual == expected, "teal pixels must be exactly one centered cross"
    return (cr, cc), tuple(arm_lengths), teal_mask


def shell_depth(r: int, c: int, box: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = box
    return min(r - r0, r1 - r, c - c0, c1 - c)


def build_shells(grid: np.ndarray, box: tuple[int, int, int, int], teal_mask: np.ndarray) -> np.ndarray:
    out = grid.copy()
    r0, c0, r1, c1 = box
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if teal_mask[r, c]:
                continue
            out[r, c] = CYCLE[shell_depth(r, c, box) % 4]
    return out


def grow_paths(
    box: tuple[int, int, int, int],
    center: tuple[int, int],
    arm_lengths: tuple[int, int, int, int],
    teal_mask: np.ndarray,
):
    r0, c0, r1, c1 = box
    walkable = np.zeros_like(teal_mask, dtype=bool)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if not teal_mask[r, c] and shell_depth(r, c, box) % 4 == 3:
                walkable[r, c] = True

    occupied = teal_mask.copy()
    path_mask = np.zeros_like(teal_mask, dtype=bool)
    agents = []
    cr, cc = center
    for (dr, dc), arm_len in zip(DIRS_URDL, arm_lengths):
        endpoint = (cr + dr * arm_len, cc + dc * arm_len)
        agents.append(
            {
                "cur": endpoint,
                "dir": right_dir((dr, dc)),
                "done": False,
                "length": 0,
                "turns": 0,
            }
        )

    def can_enter(prev: tuple[int, int], nxt: tuple[int, int]) -> bool:
        nr, nc = nxt
        if not in_bounds(walkable.shape, nr, nc) or not walkable[nr, nc] or occupied[nr, nc]:
            return False
        for ar, ac in neighbors4(nr, nc):
            if in_bounds(walkable.shape, ar, ac) and occupied[ar, ac] and (ar, ac) != prev:
                return False
        return True

    def step_agent(agent: dict) -> bool:
        if agent["done"]:
            return False
        cur = agent["cur"]
        for nd in (right_dir(agent["dir"]), agent["dir"], left_dir(agent["dir"])):
            nxt = (cur[0] + nd[0], cur[1] + nd[1])
            if can_enter(cur, nxt):
                path_mask[nxt] = True
                occupied[nxt] = True
                agent["turns"] += int(nd != agent["dir"])
                agent["dir"] = nd
                agent["cur"] = nxt
                agent["length"] += 1
                return True
        agent["done"] = True
        return False

    guard = walkable.size + 10
    while guard > 0:
        guard -= 1
        changed = False
        for agent in agents:
            changed |= step_agent(agent)
        if not changed:
            break
    return path_mask, agents, walkable


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the square frame into cyclic distance shells, then grow orange spiral paths through the black shell corridors from the teal cross tips.

    Input structure:
    - The background is black.
    - Exactly one complete one-cell-thick grey square frame appears on the canvas, with odd side length so its center is a grid cell.
    - A teal centered cross sits at the frame center, with each arm extending one or two cells along the cardinal axes.
    - All non-frame, non-cross cells are black.

    Rule:
    - Use the grey frame bbox as the reference square and compute each cell's inward shell depth by its minimum distance to the four frame sides.
    - Recolor the whole square by the repeating shell cycle grey, red, grey, black, but never overwrite teal cross pixels.
    - Treat only the black shell cells as traversable; teal cross pixels stay teal and count as occupied blockers.
    - From each cross arm tip, launch a turtle whose initial heading is the outward arm direction rotated clockwise; on every step it tries right turn, then straight, then left, and it may enter only traversable cells that are not adjacent orthogonally to any occupied path cell except its predecessor.
    - Paint every reached traversable cell orange and leave all unreached black-shell cells black.

    Key insight:
    - The concentric shell coloring creates narrow one-cell corridors, and those corridors constrain the clockwise self-avoiding spiral growth so the frame geometry and cross-arm layout jointly determine the final orange path pattern.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(map(int, np.unique(g))) <= {BLACK, RED, GREY, ORANGE, TEAL}
    assert ORANGE not in g, "input should not already contain orange path pixels"

    box = frame_info(g)[:4]
    center, arm_lengths, teal_mask = cross_info(g, box)
    non_black = (g != BLACK) & (g != GREY) & (g != TEAL)
    assert not np.any(non_black), "input may only contain black, grey, teal"

    out = build_shells(g, box, teal_mask)
    path_mask, _agents, _walkable = grow_paths(box, center, arm_lengths, teal_mask)
    out[path_mask] = ORANGE
    return out


def make_input(
    side: int,
    pads: tuple[int, int, int, int],
    arm_lengths: tuple[int, int, int, int],
) -> np.ndarray:
    top, bottom, left, right = pads
    h = side + top + bottom
    w = side + left + right
    grid = np.zeros((h, w), dtype=int)
    r0, c0 = top, left
    r1, c1 = r0 + side - 1, c0 + side - 1
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY

    cr = (r0 + r1) // 2
    cc = (c0 + c1) // 2
    grid[cr, cc] = TEAL
    for (dr, dc), arm_len in zip(DIRS_URDL, arm_lengths):
        for step in range(1, arm_len + 1):
            grid[cr + dr * step, cc + dc * step] = TEAL
    return grid


def semantic_check(inp: np.ndarray, out: np.ndarray) -> bool:
    box = frame_info(inp)[:4]
    center, arm_lengths, teal_mask = cross_info(inp, box)
    path_mask, agents, walkable = grow_paths(box, center, arm_lengths, teal_mask)
    rebuilt = build_shells(inp, box, teal_mask)
    rebuilt[path_mask] = ORANGE
    if not np.array_equal(out, rebuilt):
        return False
    if np.count_nonzero(path_mask) < 8:
        return False
    active_agents = sum(agent["length"] > 0 for agent in agents)
    if active_agents < 2:
        return False
    if not any(agent["turns"] > 0 for agent in agents if agent["length"] > 0):
        return False
    if not np.all(out[path_mask] == ORANGE):
        return False
    if np.any(out[walkable & ~path_mask] != BLACK):
        return False
    return True


def sample_pads(side: int) -> tuple[int, int, int, int]:
    max_pad = min(3, 30 - side)
    while True:
        top = random.randint(0, max_pad)
        bottom = random.randint(0, max_pad)
        left = random.randint(0, max_pad)
        right = random.randint(0, max_pad)
        if side + top + bottom <= 30 and side + left + right <= 30:
            return top, bottom, left, right


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        side = random.choice(SIDE_CHOICES)
        arm_lengths = random.choice(ARM_PATTERNS)
        pads = sample_pads(side)
        inp = make_input(side, pads, arm_lengths)
        out = main(inp)
        if semantic_check(inp, out):
            return inp
    raise RuntimeError("failed to generate a valid concentric-shell spiral input")
