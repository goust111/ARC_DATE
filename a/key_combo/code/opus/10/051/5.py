from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
FILL = 2
STAIR = 5
ANCHOR = 8

CARDINALS = ((-1, 0), (0, 1), (1, 0), (0, -1))
CAP_TO_STEPS = {
    (-1, 0): ((-1, 0), (0, 1)),   # cap above: up then right
    (0, 1): ((0, 1), (1, 0)),     # cap right: right then down
    (1, 0): ((1, 0), (0, -1)),    # cap below: down then left
    (0, -1): ((0, -1), (-1, 0)),  # cap left: left then up
}

GEN_SIZE_MIN = 18
GEN_SIZE_MAX = 24
GEN_MIN_CHEBYSHEV_GAP = 5
GEN_MIN_MANHATTAN_GAP = 7
GEN_RETRIES_BY_COUNT = {4: 140, 3: 220, 2: 60}
GEN_COUNT_CYCLE = (4, 3, 4, 3)
GEN_CALL_INDEX = 0


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _touches_border(r: int, c: int, shape: tuple[int, int]) -> bool:
    h, w = shape
    return r == 0 or c == 0 or r == h - 1 or c == w - 1


def _parse_anchors(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    anchors: list[dict] = []
    h, w = g.shape
    for r in range(h):
        for c in range(w):
            if int(g[r, c]) != ANCHOR:
                continue
            hats: list[tuple[tuple[int, int], tuple[int, int]]] = []
            for dr, dc in CARDINALS:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) == STAIR:
                    hats.append(((dr, dc), (rr, cc)))
            assert len(hats) == 1, "each anchor must have exactly one adjacent direction cap"
            offset, hat = hats[0]
            anchors.append(
                {
                    "anchor": (r, c),
                    "hat": hat,
                    "cap_offset": offset,
                    "steps": CAP_TO_STEPS[offset],
                }
            )
    anchors.sort(key=lambda item: item["anchor"])
    assert 2 <= len(anchors) <= 4, "expected 2-4 anchors"
    return anchors


def _trace_arm(
    shape: tuple[int, int],
    start: tuple[int, int],
    steps: tuple[tuple[int, int], tuple[int, int]],
    occupied: np.ndarray,
    *,
    straight: bool = False,
    swap_order: bool = False,
) -> tuple[list[tuple[int, int]], str]:
    h, w = shape
    r, c = start
    out: list[tuple[int, int]] = []
    phase = 1 if swap_order else 0
    while True:
        dr, dc = steps[0 if straight else phase]
        segment_len = max(h, w) if straight else 2
        for _ in range(segment_len):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                return out, "boundary"
            if occupied[nr, nc]:
                return out, "collision"
            out.append((nr, nc))
            r, c = nr, nc
            if _touches_border(r, c, shape):
                return out, "boundary"
        if straight:
            continue
        phase ^= 1


def _fill_closed_regions(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    seen = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if int(out[r, c]) != BG or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            touches_border = False
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                if _touches_border(rr, cc, out.shape):
                    touches_border = True
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and int(out[nr, nc]) == BG and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            if not touches_border:
                for rr, cc in comp:
                    out[rr, cc] = FILL
    return out


def _simulate(
    grid: np.ndarray,
    *,
    erase_hats: bool = True,
    straight: bool = False,
    swap_order: bool = False,
    ignore_stair_collisions: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    anchors = _parse_anchors(g)
    anchor_mask = np.zeros(out.shape, dtype=bool)
    directions: list[tuple[int, int]] = []
    for item in anchors:
        ar, ac = item["anchor"]
        hr, hc = item["hat"]
        anchor_mask[ar, ac] = True
        directions.append(item["cap_offset"])
        if erase_hats:
            out[hr, hc] = BG

    occupied = anchor_mask.copy() if ignore_stair_collisions else (out != BG)
    arm_paths: list[list[tuple[int, int]]] = []
    stop_reasons: list[str] = []
    for item in anchors:
        coords, reason = _trace_arm(
            out.shape,
            item["anchor"],
            item["steps"],
            occupied,
            straight=straight,
            swap_order=swap_order,
        )
        arm_paths.append(coords)
        stop_reasons.append(reason)
        for rr, cc in coords:
            out[rr, cc] = STAIR
            if not ignore_stair_collisions:
                occupied[rr, cc] = True

    prefill = out.copy()
    filled = _fill_closed_regions(out)
    meta = {
        "anchors": anchors,
        "directions": directions,
        "arm_paths": arm_paths,
        "stop_reasons": stop_reasons,
    }
    return filled, prefill, meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow direction-coded staircase arms from teal anchors and flood-fill only the enclosed regions they create.

    Input structure:
    - The grid is a black background with 2-4 teal anchor pixels.
    - Each anchor has exactly one adjacent grey cap in a cardinal direction; that cap encodes the staircase growth orientation.
    - No other colored objects are required in the input.

    Rule:
    - Keep every teal anchor unchanged and remove the grey cap markers.
    - For each anchor, draw a grey staircase with alternating 2-cell segments, using the cap to choose both the first segment and the turn direction: above->up then right, right->right then down, below->down then left, left->left then up.
    - While drawing, stop an arm as soon as its next cell would leave the grid or hit an already-drawn non-black cell.
    - After all arms are drawn, fill every black 4-connected region that is not connected to the outside with red.

    Key insight:
    - The grey cap is only a direction code; solving requires converting it into the correct staircase orientation, tracing the interacting arms, and then identifying the emergent enclosed black regions.
    """
    out, _prefill, _meta = _simulate(grid)
    return out


def _cap_direction_name(offset: tuple[int, int]) -> str:
    return {
        (-1, 0): "U",
        (0, 1): "R",
        (1, 0): "D",
        (0, -1): "L",
    }[offset]


def _prefers_quadrant(
    h: int,
    w: int,
    direction_name: str,
) -> list[tuple[int, int]]:
    if direction_name == "U":
        rows = range(h // 2, h - 3)
        cols = range(3, max(4, w // 2 + 1))
    elif direction_name == "R":
        rows = range(3, max(4, h // 2 + 1))
        cols = range(3, max(4, w // 2 + 1))
    elif direction_name == "D":
        rows = range(3, max(4, h // 2 + 1))
        cols = range(max(3, w // 2), w - 3)
    else:
        rows = range(max(3, h // 2), h - 3)
        cols = range(max(3, w // 2), w - 3)
    return [(r, c) for r in rows for c in cols]


def _all_inner_cells(h: int, w: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(3, h - 3) for c in range(3, w - 3)]


def _far_enough(
    cell: tuple[int, int],
    anchors: list[tuple[int, int, str]],
) -> bool:
    r, c = cell
    for ar, ac, _name in anchors:
        if max(abs(r - ar), abs(c - ac)) < GEN_MIN_CHEBYSHEV_GAP:
            return False
        if abs(r - ar) + abs(c - ac) < GEN_MIN_MANHATTAN_GAP:
            return False
    return True


def _place_anchors(h: int, w: int, direction_names: list[str], *, broad_search: bool) -> list[tuple[int, int, str]] | None:
    placed: list[tuple[int, int, str]] = []
    occupied = np.zeros((h, w), dtype=bool)
    for direction_name in direction_names:
        pool = _all_inner_cells(h, w) if broad_search else _prefers_quadrant(h, w, direction_name)
        random.shuffle(pool)
        found = None
        for r, c in pool[:400]:
            if not _far_enough((r, c), placed):
                continue
            dr, dc = {
                "U": (-1, 0),
                "R": (0, 1),
                "D": (1, 0),
                "L": (0, -1),
            }[direction_name]
            hr, hc = r + dr, c + dc
            if not (0 <= hr < h and 0 <= hc < w):
                continue
            if occupied[r, c] or occupied[hr, hc]:
                continue
            found = (r, c, direction_name)
            occupied[r, c] = True
            occupied[hr, hc] = True
            break
        if found is None:
            return None
        placed.append(found)
    return placed


def _make_input_from_anchors(h: int, w: int, anchors: list[tuple[int, int, str]]) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    name_to_cap = {
        "U": (-1, 0),
        "R": (0, 1),
        "D": (1, 0),
        "L": (0, -1),
    }
    for r, c, direction_name in anchors:
        dr, dc = name_to_cap[direction_name]
        grid[r, c] = ANCHOR
        grid[r + dr, c + dc] = STAIR
    return grid


def _sample_direction_names(anchor_count: int) -> list[str]:
    if anchor_count >= 4:
        names = ["U", "R", "D", "L"]
        random.shuffle(names)
        return names[:anchor_count]
    if anchor_count == 3:
        names = ["U", "R", "D", "L"]
        random.shuffle(names)
        return names[:3]
    pair_bank = [
        ["U", "R"],
        ["R", "D"],
        ["D", "L"],
        ["L", "U"],
        ["U", "D"],
        ["R", "L"],
    ]
    return random.choice(pair_bank).copy()


def _wrong_straight(grid: np.ndarray) -> np.ndarray:
    out, _prefill, _meta = _simulate(grid, straight=True)
    return out


def _wrong_swapped_turn(grid: np.ndarray) -> np.ndarray:
    out, _prefill, _meta = _simulate(grid, swap_order=True)
    return out


def _wrong_ignore_collisions(grid: np.ndarray) -> np.ndarray:
    out, _prefill, _meta = _simulate(grid, ignore_stair_collisions=True)
    return out


def _wrong_keep_hats(grid: np.ndarray) -> np.ndarray:
    out, _prefill, _meta = _simulate(grid, erase_hats=False)
    return out


def _semantic_checks(grid: np.ndarray) -> bool:
    try:
        out, prefill, meta = _simulate(grid)
    except AssertionError:
        return False

    anchors = meta["anchors"]
    if not (2 <= len(anchors) <= 4):
        return False

    direction_names = {_cap_direction_name(item["cap_offset"]) for item in anchors}
    if len(direction_names) < 2:
        return False

    if "collision" not in meta["stop_reasons"]:
        return False
    if "boundary" not in meta["stop_reasons"]:
        return False

    border_touch = (
        np.any(prefill[0] != BG)
        or np.any(prefill[-1] != BG)
        or np.any(prefill[:, 0] != BG)
        or np.any(prefill[:, -1] != BG)
    )
    if not border_touch:
        return False

    red_count = int(np.count_nonzero(out == FILL))
    if red_count == 0 or red_count > grid.size // 3:
        return False

    if np.count_nonzero(prefill == STAIR) < 12:
        return False

    if np.array_equal(out, _wrong_straight(grid)):
        return False
    if np.array_equal(out, _wrong_swapped_turn(grid)):
        return False
    if np.array_equal(out, _wrong_ignore_collisions(grid)):
        return False
    if np.array_equal(out, _wrong_keep_hats(grid)):
        return False

    return True


def _gen_attempt(anchor_count: int) -> np.ndarray | None:
    h = random.randint(GEN_SIZE_MIN, GEN_SIZE_MAX)
    w = random.randint(GEN_SIZE_MIN, GEN_SIZE_MAX)
    direction_names = _sample_direction_names(anchor_count)
    anchors = _place_anchors(h, w, direction_names, broad_search=(anchor_count == 2))
    if anchors is None:
        return None
    grid = _make_input_from_anchors(h, w, anchors)
    if not _semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    global GEN_CALL_INDEX
    preferred = GEN_COUNT_CYCLE[GEN_CALL_INDEX % len(GEN_COUNT_CYCLE)]
    GEN_CALL_INDEX += 1

    counts = [preferred] + [count for count in (4, 3, 2) if count != preferred]
    for anchor_count in counts:
        for _ in range(GEN_RETRIES_BY_COUNT[anchor_count]):
            grid = _gen_attempt(anchor_count)
            if grid is not None:
                return grid
    raise RuntimeError("failed to generate a valid staircase-arm enclosure input")
