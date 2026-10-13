from __future__ import annotations

from collections import Counter, deque, defaultdict
from dataclasses import dataclass
from importlib.machinery import SourceFileLoader
from pathlib import Path
import random

import numpy as np


_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
try:
    from solvers.arc_utils import Color, bbox_coords, components_mask, get_bg
except Exception:
    _arc_utils = SourceFileLoader("_arc_utils_chain_ignition", str(_ARC_UTILS_PATH)).load_module()
    Color = _arc_utils.Color
    bbox_coords = _arc_utils.bbox_coords
    components_mask = _arc_utils.components_mask
    get_bg = _arc_utils.get_bg


DIRS = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}
PALETTE = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.GREY, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]


@dataclass(frozen=True)
class Square:
    idx: int
    r0: int
    c0: int
    r1: int
    c1: int
    side: int
    color: int
    has_white_core: bool

    @property
    def center(self) -> tuple[int, int]:
        return (self.r0 + self.side // 2, self.c0 + self.side // 2)


def _component_to_square(grid: np.ndarray, idx: int, coords: np.ndarray, bg: int) -> Square:
    r0, c0, r1, c1 = bbox_coords(coords)
    side_h = r1 - r0 + 1
    side_w = c1 - c0 + 1
    assert side_h == side_w >= 3 and side_h % 2 == 1, "expected odd-sized solid squares"
    box_mask = np.zeros((side_h, side_w), dtype=bool)
    box_mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    assert bool(np.all(box_mask)), "component must fill its square bbox"

    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    colors = set(map(int, patch.flatten()))
    assert bg not in colors

    center_r = side_h // 2
    center_c = side_w // 2
    white_mask = patch == Color.WHITE
    has_white_core = bool(white_mask.any())

    if has_white_core:
        assert int(white_mask.sum()) == 1, "white marker must be unique"
        assert bool(white_mask[center_r, center_c]), "white marker must sit at the square center"
        colors.remove(Color.WHITE)
        assert len(colors) == 1, "igniter square must have one non-white body color"
    else:
        assert len(colors) == 1, "non-igniter squares must be monochrome"

    color = int(next(iter(colors)))
    assert color not in {bg, Color.WHITE}
    return Square(
        idx=idx,
        r0=int(r0),
        c0=int(c0),
        r1=int(r1),
        c1=int(c1),
        side=int(side_h),
        color=color,
        has_white_core=has_white_core,
    )


def _parse_squares(grid: np.ndarray) -> tuple[int, list[Square], int, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = int(get_bg(g))

    comps = components_mask(g != bg, connectivity=4)
    assert comps, "expected at least one square"

    occ = np.full(g.shape, -1, dtype=int)
    squares: list[Square] = []
    roots: list[int] = []

    for idx, coords in enumerate(comps):
        sq = _component_to_square(g, idx, coords, bg)
        occ[sq.r0 : sq.r1 + 1, sq.c0 : sq.c1 + 1] = sq.idx
        squares.append(sq)
        if sq.has_white_core:
            roots.append(sq.idx)

    assert len(roots) == 1, "expected exactly one white-core igniter"
    return bg, squares, roots[0], occ


def _trace_ray(square: Square, dr: int, dc: int, occ: np.ndarray) -> tuple[list[tuple[int, int]], int | None]:
    h, w = occ.shape
    cr, cc = square.center

    if dr == -1:
        r, c = square.r0 - 1, cc
    elif dr == 1:
        r, c = square.r1 + 1, cc
    elif dc == -1:
        r, c = cr, square.c0 - 1
    else:
        r, c = cr, square.c1 + 1

    cells: list[tuple[int, int]] = []
    while 0 <= r < h and 0 <= c < w:
        hit = int(occ[r, c])
        if hit >= 0:
            return cells, hit
        cells.append((int(r), int(c)))
        r += dr
        c += dc
    return cells, None


def _simulate(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg, squares, root_idx, occ = _parse_squares(g)

    out = g.copy()
    queue = deque([root_idx])
    fired: set[int] = {root_idx}
    depth = {root_idx: 0}
    incoming_first: dict[int, int] = {}
    rays: list[dict] = []
    ray_cells: Counter = Counter()

    while queue:
        sid = queue.popleft()
        square = squares[sid]

        for dname, (dr, dc) in DIRS.items():
            cells, hit = _trace_ray(square, dr, dc, occ)
            for cell in cells:
                out[cell] = square.color
                ray_cells[cell] += 1
            rays.append(
                {
                    "source": sid,
                    "dir": dname,
                    "cells": cells,
                    "hit": hit,
                }
            )
            if hit is not None and hit not in fired:
                fired.add(hit)
                depth[hit] = depth[sid] + 1
                incoming_first[hit] = sid
                queue.append(hit)

    return {
        "bg": bg,
        "squares": squares,
        "root": root_idx,
        "active": fired,
        "depth": depth,
        "incoming_first": incoming_first,
        "rays": rays,
        "ray_cells": ray_cells,
        "output": out,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Ignite the unique white-core square and draw the rays of every square reached by the chain.

    Input structure
    - A black background contains several separated solid odd-sized monochrome squares.
    - Exactly one square has a single white center cell; that marked square is the initial igniter.

    Rule
    - The igniter fires four cardinal rays along its own center row and center column using its non-white body color.
    - Each ray paints only background cells and stops at the grid edge or at the first square it meets.
    - Any struck square fires once by the same rule; squares that are never struck stay unchanged and never emit.

    Key insight
    - Emission is a contagious state rather than a property of every square, so the answer is the union of rays from the activation-reachable set starting at the white-core square.
    """
    return _simulate(np.asarray(grid, dtype=int))["output"]


def _bbox_from_centers(centers: list[tuple[int, int]], side: int) -> tuple[int, int, int, int]:
    half = side // 2
    rows = [r for r, _ in centers]
    cols = [c for _, c in centers]
    return min(rows) - half, min(cols) - half, max(rows) + half, max(cols) + half


def _build_chain_centers(side: int, orientation: str, count: int) -> list[tuple[int, int]]:
    centers = [(0, 0)]
    half = side // 2
    offsets = [delta for delta in range(-half, half + 1) if delta != 0]

    for _ in range(count - 1):
        gap = side + random.randint(2, 4)
        off = random.choice(offsets)
        pr, pc = centers[-1]

        if orientation == "R":
            child = (pr + off, pc + gap)
        else:
            child = (pr + gap, pc + off)
        centers.append(child)

    return centers


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _place_decoy_centers(
    active_centers: list[tuple[int, int]],
    side: int,
    height: int,
    width: int,
    count: int,
) -> list[tuple[int, int]]:
    half = side // 2
    placed: list[tuple[int, int]] = []
    active_rows = [r for r, _ in active_centers]
    active_cols = [c for _, c in active_centers]

    for _ in range(300):
        if len(placed) >= count:
            break
        r = random.randint(half, height - half - 1)
        c = random.randint(half, width - half - 1)

        if any(abs(r - ar) <= half for ar in active_rows):
            continue
        if any(abs(c - ac) <= half for ac in active_cols):
            continue

        new_box = (r - half, c - half, r + half, c + half)
        boxes = [
            (rr - half, cc - half, rr + half, cc + half)
            for rr, cc in active_centers + placed
        ]
        if any(_boxes_overlap(new_box, box) for box in boxes):
            continue
        placed.append((r, c))

    return placed


def _paint_square(grid: np.ndarray, center: tuple[int, int], side: int, color: int, white_core: bool) -> None:
    half = side // 2
    r, c = center
    grid[r - half : r + half + 1, c - half : c + half + 1] = color
    if white_core:
        grid[r, c] = Color.WHITE


def _children_from_incoming(active: set[int], incoming_first: dict[int, int]) -> dict[int, list[int]]:
    children: dict[int, list[int]] = {idx: [] for idx in active}
    for child, parent in incoming_first.items():
        children[parent].append(child)
    return children


def _validate_generated(grid: np.ndarray) -> bool:
    try:
        info = _simulate(grid)
    except AssertionError:
        return False

    bg = info["bg"]
    squares: list[Square] = info["squares"]
    active: set[int] = info["active"]
    depth: dict[int, int] = info["depth"]
    rays = info["rays"]
    out = info["output"]

    if len(active) < 3 or len(squares) - len(active) < 1:
        return False
    if max(depth.values()) < 2:
        return False
    if np.array_equal(out, grid):
        return False
    if np.count_nonzero(out != bg) > int(grid.size * 0.55):
        return False
    if any(count > 1 for count in info["ray_cells"].values()):
        return False

    earlier_hits: dict[int, int] = defaultdict(int)
    for ray in rays:
        source = ray["source"]
        hit = ray["hit"]
        if hit is None or hit not in active:
            continue
        if depth[source] < depth.get(hit, 10**9):
            earlier_hits[hit] += 1
        if hit != source and len(ray["cells"]) == 0:
            return False

    root = info["root"]
    for sid in active:
        if sid == root:
            continue
        if earlier_hits[sid] != 1:
            return False

    body_colors = [sq.color for sq in squares]
    if len(set(body_colors)) != len(body_colors):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    side = random.choices([3, 5], weights=[4, 1], k=1)[0]
    orientation = random.choice(["R", "D"])
    active_count = random.randint(3, 4 if side == 5 else 5)
    active_centers = _build_chain_centers(side, orientation, active_count)
    bbox = _bbox_from_centers(active_centers, side)
    r0, c0, r1, c1 = bbox
    span_h = r1 - r0 + 1
    span_w = c1 - c0 + 1
    if span_h > 24 or span_w > 24:
        return None

    if orientation == "R":
        top = random.randint(1, 4)
        left = random.randint(2, 6)
        max_extra_h = 30 - span_h - top
        max_extra_w = 30 - span_w - left
        if max_extra_h < 8 or max_extra_w < 1:
            return None
        height = span_h + top + random.randint(8, max_extra_h)
        width = span_w + left + random.randint(1, max_extra_w)
    else:
        top = random.randint(2, 6)
        left = random.randint(1, 4)
        max_extra_h = 30 - span_h - top
        max_extra_w = 30 - span_w - left
        if max_extra_h < 1 or max_extra_w < 8:
            return None
        height = span_h + top + random.randint(1, max_extra_h)
        width = span_w + left + random.randint(8, max_extra_w)

    dr = top - r0
    dc = left - c0
    active_centers = [(r + dr, c + dc) for r, c in active_centers]

    decoy_count = random.randint(1, 3)
    decoy_centers = _place_decoy_centers(active_centers, side, height, width, decoy_count)
    if len(decoy_centers) != decoy_count:
        return None

    all_centers = active_centers + decoy_centers
    half = side // 2
    boxes = []
    for r, c in all_centers:
        if not (half <= r < height - half and half <= c < width - half):
            return None
        box = (r - half, c - half, r + half, c + half)
        if any(_boxes_overlap(box, prev) for prev in boxes):
            return None
        boxes.append(box)

    colors = random.sample(PALETTE, len(all_centers))
    grid = np.zeros((height, width), dtype=int)

    for idx, center in enumerate(all_centers):
        _paint_square(grid, center, side, colors[idx], white_core=(idx == 0))

    if not _validate_generated(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid chain-ignition input")
