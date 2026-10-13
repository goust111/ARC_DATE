from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
CUP = 2
MARKER = 4
FILL = 8

DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
OPPOSITE = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
}
N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GEN_ATTEMPTS = 400
GEN_MIN_H = 10
GEN_MAX_H = 18
GEN_MIN_W = 14
GEN_MAX_W = 28
MIN_CUP_SIZE = 3
MAX_CUP_SIZE = 5
MIN_GAP = 2
EXTRA_ATTEMPTS = 120


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in N4:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    comps.sort(key=lambda coords: (int(coords[:, 0].min()), int(coords[:, 1].min())))
    return comps


def _cup_mask(h: int, w: int, open_dir: str) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    if open_dir == "up":
        mask[0, 1:-1] = False
    elif open_dir == "down":
        mask[-1, 1:-1] = False
    elif open_dir == "left":
        mask[1:-1, 0] = False
    else:
        assert open_dir == "right"
        mask[1:-1, -1] = False
    return mask


def _start_cells(cup: dict) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = cup["bbox"]
    open_dir = cup["open_dir"]
    if open_dir == "up":
        return [(r0 + 1, c) for c in range(c0 + 1, c1)]
    if open_dir == "down":
        return [(r1 - 1, c) for c in range(c0 + 1, c1)]
    if open_dir == "left":
        return [(r, c0 + 1) for r in range(r0 + 1, r1)]
    return [(r, c1 - 1) for r in range(r0 + 1, r1)]


def _opening_cells(cup: dict) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = cup["bbox"]
    open_dir = cup["open_dir"]
    if open_dir == "up":
        return [(r0, c) for c in range(c0 + 1, c1)]
    if open_dir == "down":
        return [(r1, c) for c in range(c0 + 1, c1)]
    if open_dir == "left":
        return [(r, c0) for r in range(r0 + 1, r1)]
    return [(r, c1) for r in range(r0 + 1, r1)]


def _interior_coords(cup: dict) -> np.ndarray:
    r0, c0, r1, c1 = cup["bbox"]
    cells = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
    return np.array(cells, dtype=int)


def _hit_side(cup: dict, r: int, c: int) -> str | None:
    r0, c0, r1, c1 = cup["bbox"]
    if r == r0 and c0 < c < c1:
        return "up"
    if r == r1 and c0 < c < c1:
        return "down"
    if c == c0 and r0 < r < r1:
        return "left"
    if c == c1 and r0 < r < r1:
        return "right"
    return None


def _parse_cups(grid: np.ndarray) -> list[dict]:
    red = grid == CUP
    cups: list[dict] = []
    for coords in _components(red):
        r0 = int(coords[:, 0].min())
        c0 = int(coords[:, 1].min())
        r1 = int(coords[:, 0].max())
        c1 = int(coords[:, 1].max())
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 3 or w < 3:
            raise AssertionError("cup bbox too small")
        local = np.zeros((h, w), dtype=bool)
        local[coords[:, 0] - r0, coords[:, 1] - c0] = True
        open_dirs = [name for name in DIRS if np.array_equal(local, _cup_mask(h, w, name))]
        if len(open_dirs) != 1:
            raise AssertionError("failed to identify cup orientation")
        cup = {
            "bbox": (r0, c0, r1, c1),
            "open_dir": open_dirs[0],
            "walls": coords,
        }
        cup["interior"] = _interior_coords(cup)
        cup["starts"] = _start_cells(cup)
        cup["opening"] = _opening_cells(cup)
        cups.append(cup)
    cups.sort(key=lambda cup: (cup["bbox"][0], cup["bbox"][1]))
    for idx, cup in enumerate(cups):
        cup["id"] = idx
    return cups


def _source_cup_id(grid: np.ndarray, cups: list[dict]) -> int:
    markers = np.argwhere(grid == MARKER)
    if len(markers) != 1:
        raise AssertionError("expected exactly one source marker")
    mr, mc = map(int, markers[0])
    owners = []
    for cup in cups:
        r0, c0, r1, c1 = cup["bbox"]
        if r0 < mr < r1 and c0 < mc < c1:
            owners.append(cup["id"])
    if len(owners) != 1:
        raise AssertionError("marker must lie inside exactly one cup")
    return owners[0]


def _cast_ray(out: np.ndarray, start: tuple[int, int], direction: str) -> tuple[int, int] | None:
    dr, dc = DIRS[direction]
    r, c = start
    while True:
        r += dr
        c += dc
        if not _in_bounds(out.shape, r, c):
            return None
        if out[r, c] != BG:
            return (r, c)
        out[r, c] = FILL


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the marked cup and recursively flood any other cups reached through correctly oriented openings.

    Input structure:
    - The grid has a black background with several red U-shaped cup outlines of varying sizes.
    - Exactly one cup contains a single yellow marker inside its enclosed interior.
    - Each cup is a one-pixel-thick rectangular outline missing one side segment, so its opening faces up, down, left, or right.

    Rule:
    - Find the cup containing the yellow marker and fill its interior with teal.
    - From every interior cell adjacent to that cup's opening, extend a teal ray straight outward until the ray leaves the grid or reaches a non-black cell.
    - If a ray reaches another cup through that target cup's opening direction, activate that cup too: fill its interior teal and repeat the same ray emission from its own opening. Red walls stay red, and cups never activate more than once.

    Key insight:
    - A cup's missing side defines a directed port: only rays aligned with the opening can chain into another cup, so reachability depends on spatial opening alignment rather than distance alone.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    cups = _parse_cups(g)
    source_id = _source_cup_id(g, cups)
    by_id = {cup["id"]: cup for cup in cups}

    wall_owner = {
        (int(r), int(c)): cup["id"]
        for cup in cups
        for r, c in cup["walls"]
    }

    out = g.copy()
    queue = deque([source_id])
    activated: set[int] = set()

    while queue:
        cup_id = queue.popleft()
        if cup_id in activated:
            continue
        cup = by_id[cup_id]
        activated.add(cup_id)

        interior = cup["interior"]
        out[interior[:, 0], interior[:, 1]] = FILL

        ray_dir = cup["open_dir"]
        for start in cup["starts"]:
            stop = _cast_ray(out, start, ray_dir)
            if stop is None:
                continue
            sr, sc = stop
            if out[sr, sc] != CUP:
                continue
            target_id = wall_owner.get((sr, sc))
            if target_id is None or target_id in activated:
                continue
            target = by_id[target_id]
            if _hit_side(target, sr, sc) != ray_dir:
                continue
            if target["open_dir"] != OPPOSITE[ray_dir]:
                continue
            queue.append(target_id)

    return out


def _boxes_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = MIN_GAP) -> bool:
    return not (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _cup_box(cup: dict) -> tuple[int, int, int, int]:
    return (cup["r"], cup["c"], cup["r"] + cup["h"] - 1, cup["c"] + cup["w"] - 1)


def _render_cup(grid: np.ndarray, cup: dict) -> None:
    mask = _cup_mask(cup["h"], cup["w"], cup["open_dir"])
    r0, c0 = cup["r"], cup["c"]
    grid[r0 : r0 + cup["h"], c0 : c0 + cup["w"]][mask] = CUP


def _corridor_cells(cup_a: dict, cup_b: dict, row: int) -> set[tuple[int, int]]:
    left = min(cup_a["c"], cup_b["c"])
    right = max(cup_a["c"] + cup_a["w"] - 1, cup_b["c"] + cup_b["w"] - 1)
    return {(row, c) for c in range(left, right + 1)}


def _try_place_extra(
    h: int,
    w: int,
    cups: list[dict],
    forbidden: set[tuple[int, int]],
) -> dict | None:
    boxes = [_cup_box(cup) for cup in cups]
    for _ in range(EXTRA_ATTEMPTS):
        cup_h = random.randint(MIN_CUP_SIZE, MAX_CUP_SIZE)
        cup_w = random.randint(MIN_CUP_SIZE, MAX_CUP_SIZE)
        r = random.randint(0, h - cup_h)
        c = random.randint(0, w - cup_w)
        open_dir = random.choice(list(DIRS))
        cup = {"r": r, "c": c, "h": cup_h, "w": cup_w, "open_dir": open_dir}
        box = _cup_box(cup)
        if any(_boxes_too_close(box, other) for other in boxes):
            continue
        blocked = False
        for rr in range(box[0], box[2] + 1):
            for cc in range(box[1], box[3] + 1):
                if (rr, cc) in forbidden:
                    blocked = True
                    break
            if blocked:
                break
        if blocked:
            continue
        return cup
    return None


def _build_candidate() -> np.ndarray | None:
    w_c = random.randint(3, 5)
    w_a = random.randint(3, 5)
    w_b = random.randint(3, 5)
    gap_ca = random.randint(2, 4)
    gap_ab = random.randint(3, 6)

    chain_c = {"r": 0, "c": 0, "h": 3, "w": w_c, "open_dir": "right"}
    chain_a = {"r": 2, "c": w_c + gap_ca, "h": 3, "w": w_a, "open_dir": "right"}
    chain_b = {
        "r": 0,
        "c": chain_a["c"] + w_a + gap_ab,
        "h": 5,
        "w": w_b,
        "open_dir": "left",
    }
    chain = [chain_c, chain_a, chain_b]

    motif_h = max(cup["r"] + cup["h"] for cup in chain)
    motif_w = max(cup["c"] + cup["w"] for cup in chain)

    grid_h = random.randint(max(GEN_MIN_H, motif_h + 2), GEN_MAX_H)
    grid_w = random.randint(max(GEN_MIN_W, motif_w + 2), GEN_MAX_W)
    off_r = random.randint(1, grid_h - motif_h - 1)
    off_c = random.randint(1, grid_w - motif_w - 1)

    cups = []
    for cup in chain:
        cups.append(
            {
                "r": cup["r"] + off_r,
                "c": cup["c"] + off_c,
                "h": cup["h"],
                "w": cup["w"],
                "open_dir": cup["open_dir"],
            }
        )

    source = cups[1]
    receiver = cups[2]
    sender = cups[0]

    path_row_ab = source["r"] + 1
    path_row_bc = sender["r"] + 1
    forbidden = _corridor_cells(source, receiver, path_row_ab)
    forbidden |= _corridor_cells(sender, receiver, path_row_bc)

    extra_count = random.randint(1, 3)
    for _ in range(extra_count):
        extra = _try_place_extra(grid_h, grid_w, cups, forbidden)
        if extra is None:
            return None
        cups.append(extra)

    grid = np.zeros((grid_h, grid_w), dtype=int)
    for cup in cups:
        _render_cup(grid, cup)

    marker_col = source["c"] + random.randint(1, source["w"] - 2)
    grid[source["r"] + 1, marker_col] = MARKER

    rot = random.randint(0, 3)
    if rot:
        grid = np.rot90(grid, rot).copy()
    return grid


def _activated_count(inp: np.ndarray, out: np.ndarray) -> int:
    cups = _parse_cups(inp)
    total = 0
    for cup in cups:
        interior = cup["interior"]
        if np.all(out[interior[:, 0], interior[:, 1]] == FILL):
            total += 1
    return total


def _valid_candidate(grid: np.ndarray) -> bool:
    try:
        cups = _parse_cups(grid)
        if not (4 <= len(cups) <= 6):
            return False
        out = main(grid)
    except Exception:
        return False

    activated = _activated_count(grid, out)
    if activated < 3 or activated >= len(cups):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_candidate()
        if grid is not None and _valid_candidate(grid):
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 2")
