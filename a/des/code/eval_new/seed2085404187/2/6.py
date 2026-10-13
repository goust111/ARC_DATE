import random
from collections import deque

import numpy as np


BG = 0
MAGENTA = 6
COLOR_POOL = (1, 2, 3, 4, 7, 8, 9)
WORK_SIZE = 30
TOP_ROW_MAX = 11
BOTTOM_ROW_MIN = 15
GEN_RETRIES = 400
PAIR_COUNT_RANGE = (2, 4)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _rotate_offsets(offsets: list[tuple[int, int]], turns: int) -> list[tuple[int, int]]:
    out = [(int(dr), int(dc)) for dr, dc in offsets]
    for _ in range(turns % 4):
        out = [(-dc, dr) for dr, dc in out]
    return out


def _frame_mask(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _is_rect_frame(mask: np.ndarray) -> bool:
    h, w = mask.shape
    return h >= 3 and w >= 3 and np.array_equal(mask, _frame_mask((h, w)))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp each handled sprite into the matching colored frame after rotating it to the frame's anchor.

    Input structure:
    - The background is black.
    - In the upper part of the grid, each stamp is a 4-connected multicolor component made of one main color plus exactly one magenta handle pixel.
    - In the lower part of the grid, each target is a one-pixel-thick rectangular frame whose border color matches one stamp's main color, and its interior contains exactly one magenta anchor pixel.

    Rule:
    - Pair each stamp with the frame that has the same non-magenta color.
    - Remove the original stamp from the upper area.
    - Rotate the stamp by 0, 90, 180, or 270 degrees so that its magenta handle lands exactly on the frame's magenta anchor while every occupied pixel stays inside the frame interior.
    - The matching rotation is unique; paint the rotated stamp into the frame interior using the stamp's main color, including at the anchor cell, while preserving the colored frame border.

    Key insight:
    - Magenta has two different roles: on stamps it marks the rotation pivot, and inside frames it marks the placement anchor that determines the correct rotation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    frames: dict[int, dict] = {}
    frame_cells = np.zeros_like(g, dtype=bool)
    anchors_used = np.zeros_like(g, dtype=bool)

    for color in sorted(int(v) for v in np.unique(g) if int(v) not in {BG, MAGENTA}):
        for coords in _components(g == color):
            r0, c0, r1, c1 = _bbox(coords)
            local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            local[coords[:, 0] - r0, coords[:, 1] - c0] = True
            if not _is_rect_frame(local):
                continue
            inner = g[r0 + 1 : r1, c0 + 1 : c1]
            anchor_rel = np.argwhere(inner == MAGENTA)
            if anchor_rel.shape[0] != 1:
                continue
            assert color not in frames, "expected unique frame color"
            anchor = (int(anchor_rel[0, 0] + r0 + 1), int(anchor_rel[0, 1] + c0 + 1))
            frames[color] = {
                "bbox": (r0, c0, r1, c1),
                "anchor": anchor,
                "interior": (r0 + 1, c0 + 1, r1 - 1, c1 - 1),
            }
            frame_cells[coords[:, 0], coords[:, 1]] = True
            anchors_used[anchor] = True

    assert frames, "expected at least one target frame"

    temp = g.copy()
    temp[frame_cells] = BG
    temp[anchors_used] = BG

    stamps: dict[int, dict] = {}
    for coords in _components(temp != BG):
        vals = sorted(int(v) for v in np.unique(temp[coords[:, 0], coords[:, 1]]))
        assert MAGENTA in vals and len(vals) == 2, "stamp must contain one main color and one magenta handle"
        color = next(v for v in vals if v != MAGENTA)
        assert color not in stamps, "expected unique stamp color"
        handle_coords = coords[temp[coords[:, 0], coords[:, 1]] == MAGENTA]
        assert handle_coords.shape[0] == 1, "expected one magenta handle per stamp"
        hr, hc = int(handle_coords[0, 0]), int(handle_coords[0, 1])
        offsets = [(int(r - hr), int(c - hc)) for r, c in coords]
        stamps[color] = {"coords": coords, "offsets": offsets}

    assert set(stamps) == set(frames), "stamp colors and frame colors must match"

    out = g.copy()
    for stamp in stamps.values():
        out[stamp["coords"][:, 0], stamp["coords"][:, 1]] = BG

    for color in sorted(stamps):
        frame = frames[color]
        ir0, ic0, ir1, ic1 = frame["interior"]
        ar, ac = frame["anchor"]
        fitting: list[list[tuple[int, int]]] = []
        for turns in range(4):
            coords = []
            ok = True
            for dr, dc in _rotate_offsets(stamps[color]["offsets"], turns):
                rr = ar + dr
                cc = ac + dc
                if not (ir0 <= rr <= ir1 and ic0 <= cc <= ic1):
                    ok = False
                    break
                coords.append((rr, cc))
            if ok:
                fitting.append(coords)
        assert len(fitting) == 1, "expected a unique valid rotation"
        for rr, cc in fitting[0]:
            out[rr, cc] = color

    return out


def _random_shape_cells() -> set[tuple[int, int]] | None:
    h = random.randint(2, 4)
    w = random.randint(2, 4)
    if h * w < 5:
        return None
    max_area = min(7, h * w - 1)
    if max_area < 4:
        return None
    target = random.randint(4, max_area)
    start = (random.randrange(h), random.randrange(w))
    cells = {start}
    frontier = set()

    def add_frontier(r: int, c: int) -> None:
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                frontier.add((nr, nc))

    add_frontier(*start)
    while len(cells) < target and frontier:
        nxt = random.choice(tuple(frontier))
        frontier.remove(nxt)
        cells.add(nxt)
        add_frontier(*nxt)

    rows = {r for r, _ in cells}
    cols = {c for _, c in cells}
    if len(cells) != target or len(rows) < 2 or len(cols) < 2:
        return None
    r0, c0 = min(rows), min(cols)
    r1, c1 = max(rows), max(cols)
    if (r1 - r0 + 1) * (c1 - c0 + 1) == len(cells):
        return None
    return cells


def _search_target(
    offsets: list[tuple[int, int]], desired_turns: int
) -> tuple[int, int, int, int] | None:
    desired = _rotate_offsets(offsets, desired_turns)
    rows = [dr for dr, _ in desired]
    cols = [dc for _, dc in desired]
    min_h = max(rows) - min(rows) + 1
    min_w = max(cols) - min(cols) + 1

    h_choices = list(range(min_h, min(min_h + 2, 6) + 1))
    w_choices = list(range(min_w, min(min_w + 2, 6) + 1))
    random.shuffle(h_choices)
    random.shuffle(w_choices)

    for inner_h in h_choices:
        for inner_w in w_choices:
            row_range = list(range(-min(rows), inner_h - max(rows)))
            col_range = list(range(-min(cols), inner_w - max(cols)))
            random.shuffle(row_range)
            random.shuffle(col_range)
            for anchor_r in row_range:
                for anchor_c in col_range:
                    fits: list[int] = []
                    for turns in range(4):
                        ok = True
                        for dr, dc in _rotate_offsets(offsets, turns):
                            rr = anchor_r + dr
                            cc = anchor_c + dc
                            if not (0 <= rr < inner_h and 0 <= cc < inner_w):
                                ok = False
                                break
                        if ok:
                            fits.append(turns)
                    if fits == [desired_turns]:
                        return inner_h, inner_w, anchor_r, anchor_c
    return None


def _build_stamp_spec(desired_turns: int) -> dict | None:
    for _ in range(200):
        cells = _random_shape_cells()
        if cells is None:
            continue
        handles = list(cells)
        random.shuffle(handles)
        for hr, hc in handles:
            offsets = [(r - hr, c - hc) for r, c in cells]
            target = _search_target(offsets, desired_turns)
            if target is None:
                continue
            rows = [r for r, _ in cells]
            cols = [c for _, c in cells]
            r0, c0 = min(rows), min(cols)
            local_cells = sorted((r - r0, c - c0) for r, c in cells)
            return {
                "cells": local_cells,
                "handle": (hr - r0, hc - c0),
                "shape": (max(r for r, _ in local_cells) + 1, max(c for _, c in local_cells) + 1),
                "offsets": [(r - hr, c - hc) for r, c in cells],
                "target": {
                    "turns": desired_turns,
                    "inner_h": target[0],
                    "inner_w": target[1],
                    "anchor_r": target[2],
                    "anchor_c": target[3],
                },
            }
    return None


def _can_place(occ: np.ndarray, r: int, c: int, h: int, w: int) -> bool:
    rr0 = max(0, r - 1)
    cc0 = max(0, c - 1)
    rr1 = min(occ.shape[0], r + h + 1)
    cc1 = min(occ.shape[1], c + w + 1)
    return not occ[rr0:rr1, cc0:cc1].any()


def _occupy(occ: np.ndarray, r: int, c: int, h: int, w: int) -> None:
    occ[r : r + h, c : c + w] = True


def _place_group(
    occ: np.ndarray,
    sizes: list[tuple[int, int]],
    row_lo: int,
    row_hi: int,
) -> list[tuple[int, int]] | None:
    placements = [None] * len(sizes)
    order = sorted(range(len(sizes)), key=lambda i: sizes[i][0] * sizes[i][1], reverse=True)
    for idx in order:
        h, w = sizes[idx]
        choices = [
            (r, c)
            for r in range(row_lo, row_hi - h + 2)
            for c in range(0, WORK_SIZE - w + 1)
            if _can_place(occ, r, c, h, w)
        ]
        if not choices:
            return None
        r, c = random.choice(choices)
        placements[idx] = (r, c)
        _occupy(occ, r, c, h, w)
    return [p for p in placements if p is not None]


def _crop_random(grid: np.ndarray) -> np.ndarray:
    coords = np.argwhere(grid != BG)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    top = random.randint(0, min(2, int(r0)))
    left = random.randint(0, min(2, int(c0)))
    bottom = random.randint(0, min(2, int(grid.shape[0] - 1 - r1)))
    right = random.randint(0, min(2, int(grid.shape[1] - 1 - c1)))
    return grid[r0 - top : r1 + bottom + 1, c0 - left : c1 + right + 1]


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.randint(*PAIR_COUNT_RANGE)
    colors = random.sample(COLOR_POOL, pair_count)
    desired_turns = [random.randrange(4) for _ in range(pair_count)]

    specs = []
    for turns in desired_turns:
        spec = _build_stamp_spec(turns)
        if spec is None:
            return None
        specs.append(spec)

    occ = np.zeros((WORK_SIZE, WORK_SIZE), dtype=bool)
    stamp_sizes = [spec["shape"] for spec in specs]
    frame_sizes = [
        (spec["target"]["inner_h"] + 2, spec["target"]["inner_w"] + 2)
        for spec in specs
    ]

    stamp_pos = _place_group(occ, stamp_sizes, 0, TOP_ROW_MAX)
    if stamp_pos is None:
        return None
    frame_pos = _place_group(occ, frame_sizes, BOTTOM_ROW_MIN, WORK_SIZE - 1)
    if frame_pos is None:
        return None

    grid = np.zeros((WORK_SIZE, WORK_SIZE), dtype=int)

    for color, spec, (r0, c0) in zip(colors, specs, stamp_pos):
        for rr, cc in spec["cells"]:
            grid[r0 + rr, c0 + cc] = color
        hr, hc = spec["handle"]
        grid[r0 + hr, c0 + hc] = MAGENTA

    for color, spec, (r0, c0) in zip(colors, specs, frame_pos):
        outer_h, outer_w = spec["target"]["inner_h"] + 2, spec["target"]["inner_w"] + 2
        mask = _frame_mask((outer_h, outer_w))
        grid[r0 : r0 + outer_h, c0 : c0 + outer_w][mask] = color
        ar = spec["target"]["anchor_r"]
        ac = spec["target"]["anchor_c"]
        grid[r0 + 1 + ar, c0 + 1 + ac] = MAGENTA

    cropped = _crop_random(grid)
    if max(cropped.shape) > 30:
        return None

    try:
        _ = main(cropped)
    except AssertionError:
        return None
    return cropped


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
