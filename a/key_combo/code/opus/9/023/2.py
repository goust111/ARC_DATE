from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

import numpy as np


BG = 0
PATH = 2
ROD = 3
END = 5
HINGE = 7

FIXED = "fixed"
TAIL = "tail"

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


@dataclass(frozen=True)
class RodSpec:
    orientation: Literal["h", "v"]
    endpoint: tuple[int, int]
    hinge: tuple[int, int]
    input_colors: dict[tuple[int, int], int]
    output_colors: dict[tuple[int, int], int]
    fixed_cells: frozenset[tuple[int, int]]
    tail_input_cells: frozenset[tuple[int, int]]
    tail_output_cells: frozenset[tuple[int, int]]
    hit_kind: str | None = None
    tie_fold: bool = False


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _left(dr: int, dc: int) -> tuple[int, int]:
    return -dc, dr


def _right(dr: int, dc: int) -> tuple[int, int]:
    return dc, -dr


def _cw(dr: int, dc: int) -> tuple[int, int]:
    return dc, -dr


def _projection(r: int, c: int, dr: int, dc: int) -> int:
    return r * dr + c * dc


def _component_coords(mask: np.ndarray) -> list[list[tuple[int, int]]]:
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
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _fold_step(orientation: str, hinge: tuple[int, int], tail_dir: tuple[int, int], shape: tuple[int, int]) -> tuple[int, int]:
    r, c = hinge
    h, w = shape
    if orientation == "h":
        up = r
        down = h - 1 - r
        if up > down:
            return -1, 0
        if down > up:
            return 1, 0
        return _cw(*tail_dir)
    left = c
    right = w - 1 - c
    if left > right:
        return 0, -1
    if right > left:
        return 0, 1
    return _cw(*tail_dir)


def _parse_rods(grid: np.ndarray) -> list[dict]:
    rod_mask = np.isin(grid, [ROD, END, HINGE])
    comps = _component_coords(rod_mask)
    assert 3 <= len(comps) <= 5, "expected 3-5 rods"
    rods: list[dict] = []
    for idx, comp in enumerate(comps):
        vals = {int(grid[p]) for p in comp}
        assert vals <= {ROD, END, HINGE}
        endpoint_cells = [p for p in comp if int(grid[p]) == END]
        hinge_cells = [p for p in comp if int(grid[p]) == HINGE]
        assert len(endpoint_cells) == 1 and len(hinge_cells) == 1
        endpoint = endpoint_cells[0]
        hinge = hinge_cells[0]

        rows = {r for r, _ in comp}
        cols = {c for _, c in comp}
        assert len(rows) == 1 or len(cols) == 1, "input rods must be straight"
        orientation = "h" if len(rows) == 1 else "v"

        fixed_cells: list[tuple[int, int]] = []
        tail_input_cells: list[tuple[int, int]] = []
        tail_output_cells: list[tuple[int, int]] = []

        if orientation == "h":
            assert endpoint[0] == hinge[0]
            tail_dir = (0, 1) if endpoint[1] < hinge[1] else (0, -1)
            step = 1 if endpoint[1] < hinge[1] else -1
            for cc in range(endpoint[1], hinge[1] + step, step):
                fixed_cells.append((endpoint[0], cc))
            cc = hinge[1] + step
            while (endpoint[0], cc) in comp:
                tail_input_cells.append((endpoint[0], cc))
                cc += step
        else:
            assert endpoint[1] == hinge[1]
            tail_dir = (1, 0) if endpoint[0] < hinge[0] else (-1, 0)
            step = 1 if endpoint[0] < hinge[0] else -1
            for rr in range(endpoint[0], hinge[0] + step, step):
                fixed_cells.append((rr, endpoint[1]))
            rr = hinge[0] + step
            while (rr, endpoint[1]) in comp:
                tail_input_cells.append((rr, endpoint[1]))
                rr += step

        assert len(comp) == len(fixed_cells) + len(tail_input_cells)
        assert len(tail_input_cells) >= 1
        fold = _fold_step(orientation, hinge, tail_dir, grid.shape)
        for k in range(1, len(tail_input_cells) + 1):
            rr = hinge[0] + fold[0] * k
            cc = hinge[1] + fold[1] * k
            assert _in_bounds(grid.shape, rr, cc)
            tail_output_cells.append((rr, cc))

        rods.append(
            {
                "id": idx,
                "orientation": orientation,
                "endpoint": endpoint,
                "hinge": hinge,
                "fixed_cells": tuple(fixed_cells),
                "tail_input_cells": tuple(tail_input_cells),
                "tail_output_cells": tuple(tail_output_cells),
            }
        )
    return rods


def _fold_grid(grid: np.ndarray) -> tuple[np.ndarray, dict[tuple[int, int], tuple[int, str]], dict[tuple[int, str], tuple[tuple[int, int], ...]]]:
    g = np.array(grid, dtype=int)
    out = g.copy()
    rods = _parse_rods(g)
    fixed_owner: dict[tuple[int, int], tuple[int, str]] = {}
    segments: dict[tuple[int, str], tuple[tuple[int, int], ...]] = {}
    moved_from = set()

    for rod in rods:
        rid = int(rod["id"])
        fixed_cells = tuple(rod["fixed_cells"])
        tail_input = tuple(rod["tail_input_cells"])
        tail_output = tuple(rod["tail_output_cells"])

        segments[(rid, FIXED)] = fixed_cells
        segments[(rid, TAIL)] = tail_output

        for cell in fixed_cells:
            assert cell not in fixed_owner
            fixed_owner[cell] = (rid, FIXED)
        for cell in tail_input:
            moved_from.add(cell)

    for cell in moved_from:
        out[cell] = BG

    for rod in rods:
        for cell in rod["tail_output_cells"]:
            owner = fixed_owner.get(cell)
            assert owner is None, "folded tail overlaps a fixed segment"
            fixed_owner[cell] = (int(rod["id"]), TAIL)
            if int(out[cell]) not in (BG, ROD):
                raise AssertionError("folded tail collides with another marker")
            out[cell] = ROD

    return out, fixed_owner, segments


def _simulate_path(
    folded: np.ndarray,
    owner_map: dict[tuple[int, int], tuple[int, str]],
    segments: dict[tuple[int, str], tuple[tuple[int, int], ...]],
    kind_override: dict[tuple[int, int], str] | None = None,
) -> tuple[np.ndarray, list[str], list[int]]:
    g = np.array(folded, dtype=int)
    out = g.copy()
    starts = np.argwhere(g == PATH)
    assert starts.shape == (1, 2), "expected one red border marker"
    r, c = map(int, starts[0])
    h, w = g.shape
    if r == 0:
        dr, dc = 1, 0
    elif r == h - 1:
        dr, dc = -1, 0
    elif c == 0:
        dr, dc = 0, 1
    elif c == w - 1:
        dr, dc = 0, -1
    else:
        raise AssertionError("marker must lie on the border")

    hit_kinds: list[str] = []
    hit_rods: list[int] = []

    def paint(rr: int, cc: int) -> None:
        if int(out[rr, cc]) == BG:
            out[rr, cc] = PATH

    paint(r, c)
    while True:
        nr, nc = r + dr, c + dc
        if not _in_bounds(g.shape, nr, nc):
            break
        if (nr, nc) not in owner_map:
            r, c = nr, nc
            paint(r, c)
            continue

        rid, kind = owner_map[(nr, nc)]
        if kind_override is not None:
            kind = kind_override[(nr, nc)]
        hit_kinds.append(kind)
        hit_rods.append(rid)

        lateral = _left(dr, dc) if kind == FIXED else _right(dr, dc)
        segment_cells = segments[(rid, owner_map[(nr, nc)][1])]
        target_proj = max(_projection(rr, cc, dr, dc) for rr, cc in segment_cells) + 1

        r += lateral[0]
        c += lateral[1]
        assert _in_bounds(g.shape, r, c), "detour must stay in bounds"
        paint(r, c)
        while _projection(r, c, dr, dc) < target_proj:
            r += dr
            c += dc
            assert _in_bounds(g.shape, r, c), "detour forward step left the grid"
            paint(r, c)
        r -= lateral[0]
        c -= lateral[1]
        assert _in_bounds(g.shape, r, c), "detour must be able to return"
        paint(r, c)

    return out, hit_kinds, hit_rods


def _solve(grid: np.ndarray) -> np.ndarray:
    folded, owner_map, segments = _fold_grid(grid)
    out, _kinds, _rods = _simulate_path(folded, owner_map, segments)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold each marked rod into an L shape, then draw a border-entering path that detours left around fixed segments and right around folded segments.

    Input structure:
    - A black background contains 3-5 straight one-pixel rods made of green cells.
    - Each rod has one gray endpoint marker at an end and one orange hinge marker in its interior.
    - Exactly one red marker lies on a non-corner border cell and indicates the path entry point.

    Rule:
    - For every rod, keep the segment from the gray endpoint through the orange hinge fixed in place.
    - Rotate the tail on the far side of the hinge by 90 degrees toward the side with more edge-space; if both sides are equally large, use the clockwise turn.
    - After all rods are folded, start from the red border marker and move straight inward.
    - Whenever the next cell belongs to a fixed segment, make a one-cell left detour relative to the current travel direction; whenever it belongs to a folded tail, make the symmetric right detour.
    - During a detour, shift sideways by one cell, continue forward until one cell beyond that blocking segment's forward extent, then shift back; only black cells are painted red.

    Key insight:
    - The fold does two jobs at once: it changes where obstacles appear, and it labels each obstacle arm by its origin, so the solver must reconstruct which green cells stayed fixed and which green cells were moved.
    """
    return _solve(np.asarray(grid, dtype=int))


def _add_segment(cells: set[tuple[int, int]], a: tuple[int, int], b: tuple[int, int]) -> None:
    if a[0] == b[0]:
        step = 1 if b[1] >= a[1] else -1
        for cc in range(a[1], b[1] + step, step):
            cells.add((a[0], cc))
    else:
        step = 1 if b[0] >= a[0] else -1
        for rr in range(a[0], b[0] + step, step):
            cells.add((rr, a[1]))


def _expand(cells: set[tuple[int, int]], shape: tuple[int, int], radius: int = 1) -> set[tuple[int, int]]:
    h, w = shape
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((rr, cc))
    return out


def _rod_from_horizontal(
    shape: tuple[int, int],
    row: int,
    endpoint_col: int,
    hinge_col: int,
    tail_len: int,
    hit_kind: str | None = None,
) -> RodSpec | None:
    h, w = shape
    if endpoint_col == hinge_col:
        return None
    step = 1 if hinge_col > endpoint_col else -1
    tail_dir = (0, step)
    fold = _fold_step("h", (row, hinge_col), tail_dir, shape)
    avail = row if fold[0] < 0 else h - 1 - row
    if tail_len < 1 or tail_len > avail:
        return None

    input_colors: dict[tuple[int, int], int] = {}
    output_colors: dict[tuple[int, int], int] = {}
    fixed_cells: set[tuple[int, int]] = set()
    tail_input_cells: set[tuple[int, int]] = set()
    tail_output_cells: set[tuple[int, int]] = set()

    for cc in range(endpoint_col, hinge_col + step, step):
        cell = (row, cc)
        fixed_cells.add(cell)
        input_colors[cell] = ROD
        output_colors[cell] = ROD
    input_colors[(row, endpoint_col)] = END
    input_colors[(row, hinge_col)] = HINGE
    output_colors[(row, endpoint_col)] = END
    output_colors[(row, hinge_col)] = HINGE

    cc = hinge_col + step
    for _ in range(tail_len):
        if not _in_bounds(shape, row, cc):
            return None
        tail_input_cells.add((row, cc))
        input_colors[(row, cc)] = ROD
        cc += step

    for k in range(1, tail_len + 1):
        rr = row + fold[0] * k
        cc = hinge_col + fold[1] * k
        if not _in_bounds(shape, rr, cc):
            return None
        tail_output_cells.add((rr, cc))
        output_colors[(rr, cc)] = ROD

    tie_fold = row == (h - 1 - row)
    total_len = len(fixed_cells) + len(tail_input_cells)
    if not 5 <= total_len <= 9:
        return None
    return RodSpec(
        orientation="h",
        endpoint=(row, endpoint_col),
        hinge=(row, hinge_col),
        input_colors=input_colors,
        output_colors=output_colors,
        fixed_cells=frozenset(fixed_cells),
        tail_input_cells=frozenset(tail_input_cells),
        tail_output_cells=frozenset(tail_output_cells),
        hit_kind=hit_kind,
        tie_fold=tie_fold,
    )


def _rod_from_vertical(
    shape: tuple[int, int],
    col: int,
    endpoint_row: int,
    hinge_row: int,
    tail_len: int,
    hit_kind: str | None = None,
) -> RodSpec | None:
    h, w = shape
    if endpoint_row == hinge_row:
        return None
    step = 1 if hinge_row > endpoint_row else -1
    tail_dir = (step, 0)
    fold = _fold_step("v", (hinge_row, col), tail_dir, shape)
    avail = col if fold[1] < 0 else w - 1 - col
    if tail_len < 1 or tail_len > avail:
        return None

    input_colors: dict[tuple[int, int], int] = {}
    output_colors: dict[tuple[int, int], int] = {}
    fixed_cells: set[tuple[int, int]] = set()
    tail_input_cells: set[tuple[int, int]] = set()
    tail_output_cells: set[tuple[int, int]] = set()

    for rr in range(endpoint_row, hinge_row + step, step):
        cell = (rr, col)
        fixed_cells.add(cell)
        input_colors[cell] = ROD
        output_colors[cell] = ROD
    input_colors[(endpoint_row, col)] = END
    input_colors[(hinge_row, col)] = HINGE
    output_colors[(endpoint_row, col)] = END
    output_colors[(hinge_row, col)] = HINGE

    rr = hinge_row + step
    for _ in range(tail_len):
        if not _in_bounds(shape, rr, col):
            return None
        tail_input_cells.add((rr, col))
        input_colors[(rr, col)] = ROD
        rr += step

    for k in range(1, tail_len + 1):
        rr = hinge_row + fold[0] * k
        cc = col + fold[1] * k
        if not _in_bounds(shape, rr, cc):
            return None
        tail_output_cells.add((rr, cc))
        output_colors[(rr, cc)] = ROD

    tie_fold = col == (w - 1 - col)
    total_len = len(fixed_cells) + len(tail_input_cells)
    if not 5 <= total_len <= 9:
        return None
    return RodSpec(
        orientation="v",
        endpoint=(endpoint_row, col),
        hinge=(hinge_row, col),
        input_colors=input_colors,
        output_colors=output_colors,
        fixed_cells=frozenset(fixed_cells),
        tail_input_cells=frozenset(tail_input_cells),
        tail_output_cells=frozenset(tail_output_cells),
        hit_kind=hit_kind,
        tie_fold=tie_fold,
    )


def _input_cells(rod: RodSpec) -> set[tuple[int, int]]:
    return set(rod.input_colors)


def _output_cells(rod: RodSpec) -> set[tuple[int, int]]:
    return set(rod.output_colors)


def _placeable(
    rod: RodSpec,
    shape: tuple[int, int],
    occupied_in: set[tuple[int, int]],
    occupied_in_pad: set[tuple[int, int]],
    occupied_out: set[tuple[int, int]],
    occupied_out_pad: set[tuple[int, int]],
    reserved_input: set[tuple[int, int]] | None = None,
    reserved_output: set[tuple[int, int]] | None = None,
) -> bool:
    in_cells = _input_cells(rod)
    out_cells = _output_cells(rod)
    if reserved_input is not None and in_cells & reserved_input:
        return False
    if reserved_output is not None and out_cells & reserved_output:
        return False
    if in_cells & occupied_in:
        return False
    if out_cells & occupied_out:
        return False
    if in_cells & occupied_in_pad:
        return False
    if out_cells & occupied_out_pad:
        return False
    return True


def _try_fixed_hit(shape: tuple[int, int], lane: int, row: int, require_tie: bool) -> RodSpec | None:
    h, w = shape
    for _ in range(300):
        if require_tie:
            if h % 2 == 0 or row != h // 2:
                return None
        left_room = lane
        right_room = w - 1 - lane
        if left_room < 2 or right_room < 2:
            return None
        hinge_side = random.choice(["left", "right"])
        left_offset = random.randint(1, min(3, left_room))
        right_offset = random.randint(1, min(3, right_room))
        if hinge_side == "right":
            endpoint_col = lane - left_offset
            hinge_col = lane + right_offset
        else:
            endpoint_col = lane + right_offset
            hinge_col = lane - left_offset
        fixed_len = abs(endpoint_col - hinge_col) + 1
        max_tail = min(9 - fixed_len, h - 1)
        if max_tail < 1:
            continue
        tail_len = random.randint(1, min(4, max_tail))
        rod = _rod_from_horizontal(shape, row, endpoint_col, hinge_col, tail_len, FIXED)
        if rod is None:
            continue
        if require_tie and not rod.tie_fold:
            continue
        if (row, lane) not in rod.fixed_cells:
            continue
        if (row, lane) in {(rod.endpoint), (rod.hinge)}:
            continue
        return rod
    return None


def _try_tail_hit(shape: tuple[int, int], lane: int, row: int, require_tie: bool) -> RodSpec | None:
    h, w = shape
    for _ in range(400):
        if require_tie and w % 2 == 0:
            return None
        if random.random() < 0.5:
            col = random.randint(1, max(1, min(lane - 1, (w - 1) // 2)))
        else:
            col = random.randint(max(lane + 1, w // 2), w - 2)
        if require_tie:
            col = w // 2
            if col == lane:
                continue
        dist = abs(lane - col)
        if dist < 1:
            continue
        endpoint_above = random.random() < 0.5
        fixed_len = random.randint(2, 5)
        if endpoint_above:
            endpoint_row = row - (fixed_len - 1)
        else:
            endpoint_row = row + (fixed_len - 1)
        if not (0 <= endpoint_row < h):
            continue
        max_tail = 9 - fixed_len
        if max_tail < dist:
            continue
        tail_len = random.randint(dist + 1 if max_tail >= dist + 1 else dist, min(max_tail, dist + 2))
        rod = _rod_from_vertical(shape, col, endpoint_row, row, tail_len, TAIL)
        if rod is None:
            continue
        if require_tie and not rod.tie_fold:
            continue
        if (row, lane) not in rod.tail_output_cells:
            continue
        return rod
    return None


def _try_scenery(shape: tuple[int, int], lane: int, require_tie: bool) -> RodSpec | None:
    h, w = shape
    reserved_cols = {lane - 1, lane, lane + 1}
    for _ in range(500):
        orientation = random.choice(["h", "v"])
        if orientation == "h":
            if require_tie and h % 2 == 0:
                return None
            row = h // 2 if require_tie else random.randint(1, h - 2)
            endpoint_col = random.randint(0, w - 1)
            side = random.choice([-1, 1])
            hinge_dist = random.randint(2, 5)
            hinge_col = endpoint_col + side * hinge_dist
            if not (0 <= hinge_col < w):
                continue
            tail_len = random.randint(1, min(4, 9 - (abs(hinge_col - endpoint_col) + 1)))
            rod = _rod_from_horizontal(shape, row, endpoint_col, hinge_col, tail_len)
        else:
            if require_tie and w % 2 == 0:
                return None
            col = w // 2 if require_tie else random.choice([c for c in range(1, w - 1) if c not in reserved_cols])
            endpoint_row = random.randint(0, h - 1)
            side = random.choice([-1, 1])
            hinge_dist = random.randint(2, 5)
            hinge_row = endpoint_row + side * hinge_dist
            if not (0 <= hinge_row < h):
                continue
            tail_len = random.randint(1, min(4, 9 - (abs(hinge_row - endpoint_row) + 1)))
            rod = _rod_from_vertical(shape, col, endpoint_row, hinge_row, tail_len)
        if rod is None:
            continue
        if require_tie and not rod.tie_fold:
            continue
        if any(c in reserved_cols for _r, c in rod.output_colors):
            continue
        return rod
    return None


def _apply_rods(shape: tuple[int, int], rods: list[RodSpec], lane: int) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full(shape, BG, dtype=int)
    out = np.full(shape, BG, dtype=int)
    for rod in rods:
        for cell, color in rod.input_colors.items():
            assert inp[cell] == BG
            inp[cell] = color
        for cell, color in rod.output_colors.items():
            assert out[cell] == BG
            out[cell] = color
    inp[0, lane] = PATH
    out[0, lane] = PATH
    return inp, out


def _rotate_grid(grid: np.ndarray, k: int) -> np.ndarray:
    return np.rot90(grid, k=k)


def _build_variant(inp: np.ndarray, folded: np.ndarray) -> tuple[np.ndarray, dict[tuple[int, int], tuple[int, str]], dict[tuple[int, str], tuple[tuple[int, int], ...]]]:
    owner_map: dict[tuple[int, int], tuple[int, str]] = {}
    segments: dict[tuple[int, str], tuple[tuple[int, int], ...]] = {}
    rods = _parse_rods(inp)
    for rod in rods:
        rid = int(rod["id"])
        fixed_cells = tuple(rod["fixed_cells"])
        tail_output = tuple(rod["tail_output_cells"])
        segments[(rid, FIXED)] = fixed_cells
        segments[(rid, TAIL)] = tail_output
        for cell in fixed_cells:
            owner_map[cell] = (rid, FIXED)
        for cell in tail_output:
            owner_map[cell] = (rid, TAIL)
    return folded, owner_map, segments


def _valid_pair(inp: np.ndarray, out: np.ndarray) -> bool:
    try:
        folded, owner_map, segments = _fold_grid(inp)
        pred, hit_kinds, hit_rods = _simulate_path(folded, owner_map, segments)
        if not np.array_equal(pred, out):
            return False
        if len(set(hit_rods)) < 2:
            return False
        if FIXED not in hit_kinds or TAIL not in hit_kinds:
            return False

        kinds_left = {cell: FIXED for cell in owner_map}
        kinds_right = {cell: TAIL for cell in owner_map}
        all_left, _, _ = _simulate_path(folded, owner_map, segments, kinds_left)
        all_right, _, _ = _simulate_path(folded, owner_map, segments, kinds_right)
        if np.array_equal(all_left, out) or np.array_equal(all_right, out):
            return False

        straight = np.array(inp, dtype=int)
        straight_owner: dict[tuple[int, int], tuple[int, str]] = {}
        straight_segments: dict[tuple[int, str], tuple[tuple[int, int], ...]] = {}
        for rod in _parse_rods(inp):
            rid = int(rod["id"])
            fixed_cells = tuple(rod["fixed_cells"])
            tail_cells = tuple(rod["tail_input_cells"])
            straight_segments[(rid, FIXED)] = fixed_cells
            straight_segments[(rid, TAIL)] = tail_cells
            for cell in fixed_cells:
                straight_owner[cell] = (rid, FIXED)
            for cell in tail_cells:
                straight_owner[cell] = (rid, TAIL)
        no_fold, _, _ = _simulate_path(straight, straight_owner, straight_segments)
        if np.array_equal(no_fold, out):
            return False
    except Exception:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(18, 24)
    w = random.randint(18, 24)
    shape = (h, w)
    lane = random.randint(4, w - 5)
    rod_count = random.randint(3, 5)
    hit_count = random.randint(2, min(3, rod_count))
    scenery_count = rod_count - hit_count
    need_tie = random.random() < 0.35 and (h % 2 == 1 or w % 2 == 1)

    rows: list[int] = []
    if need_tie and h % 2 == 1:
        rows.append(h // 2)
    while len(rows) < hit_count:
        r = random.randint(3, h - 4)
        if all(abs(r - prev) >= 3 for prev in rows):
            rows.append(r)
    rows.sort()

    hit_types = [FIXED, TAIL]
    while len(hit_types) < hit_count:
        hit_types.append(random.choice([FIXED, TAIL]))
    random.shuffle(hit_types)

    rods: list[RodSpec] = []
    occupied_in: set[tuple[int, int]] = set()
    occupied_in_pad: set[tuple[int, int]] = set()
    occupied_out: set[tuple[int, int]] = set()
    occupied_out_pad: set[tuple[int, int]] = set()
    tie_used = False

    for row, hit_type in zip(rows, hit_types, strict=True):
        require_tie_here = need_tie and not tie_used and row == h // 2 and h % 2 == 1 and hit_type == FIXED
        builder = _try_fixed_hit if hit_type == FIXED else _try_tail_hit
        rod = builder(shape, lane, row, require_tie_here)
        if rod is None:
            return None
        if not _placeable(rod, shape, occupied_in, occupied_in_pad, occupied_out, occupied_out_pad):
            for _ in range(200):
                rod = builder(shape, lane, row, require_tie_here)
                if rod is not None and _placeable(rod, shape, occupied_in, occupied_in_pad, occupied_out, occupied_out_pad):
                    break
            else:
                return None
        rods.append(rod)
        occupied_in |= _input_cells(rod)
        occupied_out |= _output_cells(rod)
        occupied_in_pad |= _expand(_input_cells(rod), shape, radius=1)
        occupied_out_pad |= _expand(_output_cells(rod), shape, radius=1)
        tie_used |= rod.tie_fold

    for _ in range(scenery_count):
        require_tie_here = need_tie and not tie_used
        rod = _try_scenery(shape, lane, require_tie_here)
        if rod is None:
            if require_tie_here:
                return None
            continue
        for _ in range(200):
            if _placeable(rod, shape, occupied_in, occupied_in_pad, occupied_out, occupied_out_pad):
                break
            rod = _try_scenery(shape, lane, require_tie_here)
            if rod is None:
                break
        else:
            return None
        if rod is None or not _placeable(rod, shape, occupied_in, occupied_in_pad, occupied_out, occupied_out_pad):
            if require_tie_here:
                return None
            continue
        rods.append(rod)
        occupied_in |= _input_cells(rod)
        occupied_out |= _output_cells(rod)
        occupied_in_pad |= _expand(_input_cells(rod), shape, radius=1)
        occupied_out_pad |= _expand(_output_cells(rod), shape, radius=1)
        tie_used |= rod.tie_fold

    if len(rods) != rod_count:
        return None
    if need_tie and not tie_used:
        return None

    inp, _folded_only = _apply_rods(shape, rods, lane)
    k = random.randrange(4)
    inp = _rotate_grid(inp, k)
    expected = _solve(inp)
    if not _valid_pair(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(600):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fold-detour input")
