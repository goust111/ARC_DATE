from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2

GRID_SIDE_RANGE = (18, 24)
FRAME_SIDE_RANGE = (5, 7)
FRAME_COUNT = 2
GENERATE_ATTEMPTS = 500
INTERACTION_WEIGHTS = (("blue", 3), ("marker", 2))
ORIENTATION_CHOICES = ("SE", "SW")
CORNER_ORDER = ("NW", "NE", "SW", "SE")

CORNER_DATA = {
    "NW": {"inner": (1, 1), "outer": (-1, -1), "dir": (-1, -1)},
    "NE": {"inner": (1, -1), "outer": (-1, 1), "dir": (-1, 1)},
    "SW": {"inner": (-1, 1), "outer": (1, -1), "dir": (1, -1)},
    "SE": {"inner": (-1, -1), "outer": (1, 1), "dir": (1, 1)},
}


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
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
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _frame_border_cells(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells = {(r0, c) for c in range(c0, c1 + 1)}
    cells |= {(r1, c) for c in range(c0, c1 + 1)}
    cells |= {(r, c0) for r in range(r0, r1 + 1)}
    cells |= {(r, c1) for r in range(r0, r1 + 1)}
    return cells


def _frame_corners(bbox: tuple[int, int, int, int]) -> dict[str, tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return {
        "NW": (r0, c0),
        "NE": (r0, c1),
        "SW": (r1, c0),
        "SE": (r1, c1),
    }


def _corner_point(corner: tuple[int, int], offset: tuple[int, int]) -> tuple[int, int]:
    return corner[0] + offset[0], corner[1] + offset[1]


def _detect_frames(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    blue_components = _components(g == BLUE)
    frames: list[dict] = []
    for coords in blue_components:
        box = _bbox(coords)
        r0, c0, r1, c1 = box
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 5 or w < 5:
            continue
        cell_set = {tuple(map(int, rc)) for rc in coords}
        if cell_set != _frame_border_cells(box):
            continue
        corners = _frame_corners(box)
        inner_markers: dict[str, tuple[int, int]] = {}
        outer_markers: dict[str, tuple[int, int]] = {}
        for name in CORNER_ORDER:
            inner = _corner_point(corners[name], CORNER_DATA[name]["inner"])
            outer = _corner_point(corners[name], CORNER_DATA[name]["outer"])
            assert _in_bounds(g.shape, *inner), "inner marker must be in bounds"
            assert _in_bounds(g.shape, *outer), "outer marker destination must be in bounds"
            assert int(g[inner]) == RED, "expected a red inner-corner marker"
            inner_markers[name] = inner
            outer_markers[name] = outer
        frames.append(
            {
                "bbox": box,
                "corners": corners,
                "inner_markers": inner_markers,
                "outer_markers": outer_markers,
            }
        )

    frames.sort(key=lambda frame: frame["bbox"])
    assert len(frames) == FRAME_COUNT, f"expected {FRAME_COUNT} blue rectangular frames"

    expected_red = {
        marker
        for frame in frames
        for marker in frame["inner_markers"].values()
    }
    actual_red = {tuple(map(int, rc)) for rc in np.argwhere(g == RED)}
    assert actual_red == expected_red, "expected only the eight inner-corner red markers"
    return frames


def _paint_wings(
    out: np.ndarray,
    last: tuple[int, int],
    direction: tuple[int, int],
) -> int:
    r, c = last
    dr, dc = direction
    added = 0
    for wr, wc in ((r + dr, c), (r, c + dc)):
        if _in_bounds(out.shape, wr, wc) and int(out[wr, wc]) == BLACK:
            out[wr, wc] = RED
            added += 1
    return added


def _transform(grid: np.ndarray, *, with_meta: bool = False):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    frames = _detect_frames(g)

    out = g.copy()
    for frame in frames:
        for cell in frame["inner_markers"].values():
            out[cell] = BLACK
        for cell in frame["outer_markers"].values():
            out[cell] = RED

    rays: list[dict] = []
    for frame in frames:
        for name in CORNER_ORDER:
            rays.append(
                {
                    "start": frame["outer_markers"][name],
                    "dir": CORNER_DATA[name]["dir"],
                    "last": frame["outer_markers"][name],
                    "steps": 0,
                    "reason": None,
                    "wings": 0,
                }
            )

    active = set(range(len(rays)))
    reason_counts = {"boundary": 0, "blue": 0, "red": 0, "conflict": 0}
    wing_cells = 0

    while active:
        proposals: dict[tuple[int, int], list[int]] = {}
        stop_reasons: dict[int, str] = {}
        stopping: set[int] = set()
        movers: list[tuple[int, tuple[int, int]]] = []

        for idx in sorted(active):
            ray = rays[idx]
            r, c = ray["last"]
            dr, dc = ray["dir"]
            nr, nc = r + dr, c + dc
            if not _in_bounds(out.shape, nr, nc):
                stop_reasons[idx] = "boundary"
                stopping.add(idx)
                continue
            if int(out[nr, nc]) != BLACK:
                stop_reasons[idx] = "blue" if int(out[nr, nc]) == BLUE else "red"
                stopping.add(idx)
                continue
            proposals.setdefault((nr, nc), []).append(idx)

        for cell, ids in proposals.items():
            if len(ids) > 1:
                for idx in ids:
                    stop_reasons[idx] = "conflict"
                    stopping.add(idx)
            else:
                movers.append((ids[0], cell))

        for idx, cell in movers:
            out[cell] = RED
            rays[idx]["last"] = cell
            rays[idx]["steps"] += 1

        for idx in sorted(stopping):
            active.remove(idx)
            reason = stop_reasons[idx]
            rays[idx]["reason"] = reason
            reason_counts[reason] += 1

        for idx in sorted(stopping):
            added = _paint_wings(out, rays[idx]["last"], rays[idx]["dir"])
            rays[idx]["wings"] = added
            wing_cells += added

    if not with_meta:
        return out

    meta = {
        "reason_counts": reason_counts,
        "wing_cells": wing_cells,
        "max_steps": max(ray["steps"] for ray in rays),
        "ray_steps": [ray["steps"] for ray in rays],
        "total_new_red": int(np.count_nonzero(out == RED) - np.count_nonzero(g == RED)),
    }
    return out, meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each red inner-corner marker to the matching outer corner, then fire outward diagonal rays that sprout short side wakes when they stop.

    Input structure:
    - The background is black.
    - Exactly two blue single-pixel-thick hollow rectangular frames appear, each at least 5x5.
    - Each frame has one red marker at each of its four inner-corner cells, one step inside the blue border.

    Rule:
    - For each frame, erase the four red inner markers and recreate them at the four corresponding outer corners, one cell outside the frame.
    - Keep every blue frame cell unchanged.
    - From each new outer-corner marker, extend a red diagonal ray in the direction moving farther from that frame's center.
    - A ray advances only through black cells and stops before the grid edge or before any non-black cell, including another frame or a previously created red ray cell.
    - When a ray stops, add up to two one-cell red side wakes from its last red cell: one in the ray's row direction and one in its column direction, but only onto black cells.

    Key insight:
    - The moved corner markers are not just translated decorations; each outer corner becomes a directional launcher whose diagonal path and stopping wake are determined entirely by the geometry of its parent frame.
    """
    return _transform(grid)


def _draw_frame(grid: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    grid[r0, c0 : c1 + 1] = BLUE
    grid[r1, c0 : c1 + 1] = BLUE
    grid[r0 : r1 + 1, c0] = BLUE
    grid[r0 : r1 + 1, c1] = BLUE
    grid[r0 + 1, c0 + 1] = RED
    grid[r0 + 1, c1 - 1] = RED
    grid[r1 - 1, c0 + 1] = RED
    grid[r1 - 1, c1 - 1] = RED


def _expanded_overlap(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + 1 < br0 - 1
        or br1 + 1 < ar0 - 1
        or ac1 + 1 < bc0 - 1
        or bc1 + 1 < ac0 - 1
    )


def _bbox_gap(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> int:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    dr = max(0, br0 - ar1 - 1, ar0 - br1 - 1)
    dc = max(0, bc0 - ac1 - 1, ac0 - bc1 - 1)
    return max(dr, dc)


def _interaction_choice() -> str:
    labels = [label for label, _weight in INTERACTION_WEIGHTS]
    weights = [weight for _label, weight in INTERACTION_WEIGHTS]
    return random.choices(labels, weights=weights, k=1)[0]


def _build_layout(
    height: int,
    width: int,
    h1: int,
    w1: int,
    h2: int,
    w2: int,
    orientation: str,
    interaction: str,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]] | None:
    add = 0 if interaction == "blue" else 1
    max_d = min(
        7,
        height - h1 - h2 - add - 2,
        width - w1 - w2 - add - 2,
    )
    if max_d < 4:
        return None
    d = random.randint(4, max_d)

    if orientation == "SE":
        max_r0 = height - h1 - d - add - h2 - 1
        max_c0 = width - w1 - d - add - w2 - 1
        if max_r0 < 1 or max_c0 < 1:
            return None
        r0 = random.randint(1, max_r0)
        c0 = random.randint(1, max_c0)
        r2 = r0 + h1 + d + add
        c2 = c0 + w1 + d + add
    else:
        max_r0 = height - h1 - d - add - h2 - 1
        min_c0 = 1 + d + w2 + add
        max_c0 = width - w1 - 1
        if max_r0 < 1 or min_c0 > max_c0:
            return None
        r0 = random.randint(1, max_r0)
        c0 = random.randint(min_c0, max_c0)
        r2 = r0 + h1 + d + add
        c2 = c0 - d - w2 - add

    box1 = (r0, c0, r0 + h1 - 1, c0 + w1 - 1)
    box2 = (r2, c2, r2 + h2 - 1, c2 + w2 - 1)
    if _bbox_gap(box1, box2) < 4 or _expanded_overlap(box1, box2):
        return None
    return box1, box2


def _input_is_valid(grid: np.ndarray) -> bool:
    try:
        frames = _detect_frames(grid)
    except AssertionError:
        return False
    return len(frames) == FRAME_COUNT and int(np.count_nonzero(grid == RED)) == 8


def _accepts(grid: np.ndarray, interaction: str) -> bool:
    if not _input_is_valid(grid):
        return False
    out, meta = _transform(grid, with_meta=True)
    if np.array_equal(out, grid):
        return False
    if meta["max_steps"] < 2:
        return False
    if meta["wing_cells"] < 2:
        return False
    if meta["reason_counts"]["boundary"] < 1:
        return False
    if interaction == "blue":
        if meta["reason_counts"]["blue"] < 1:
            return False
    else:
        if meta["reason_counts"]["red"] + meta["reason_counts"]["conflict"] < 1:
            return False
    return meta["total_new_red"] >= 8


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_SIDE_RANGE)
    width = random.randint(*GRID_SIDE_RANGE)
    h1 = random.randint(*FRAME_SIDE_RANGE)
    w1 = random.randint(*FRAME_SIDE_RANGE)
    h2 = random.randint(*FRAME_SIDE_RANGE)
    w2 = random.randint(*FRAME_SIDE_RANGE)
    orientation = random.choice(ORIENTATION_CHOICES)
    interaction = _interaction_choice()

    layout = _build_layout(height, width, h1, w1, h2, w2, orientation, interaction)
    if layout is None:
        return None

    grid = np.zeros((height, width), dtype=int)
    for box in layout:
        r0, c0, r1, c1 = box
        _draw_frame(grid, r0, c0, r1 - r0 + 1, c1 - c0 + 1)

    if not _accepts(grid, interaction):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid frame-ray input")
