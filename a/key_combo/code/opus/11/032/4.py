from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
WIRE = 8
TOKEN_COLORS = (1, 2, 3, 4, 6, 7, 9)

SOLID_MASK = frozenset((r, c) for r in range(3) for c in range(3))
X_MASK = frozenset({(0, 0), (0, 2), (1, 1), (2, 0), (2, 2)})
DIAMOND_MASK = frozenset({(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)})
SHAPE_TO_MASK = {
    "solid": SOLID_MASK,
    "x": X_MASK,
    "diamond": DIAMOND_MASK,
}
SHAPE_CYCLE = ("solid", "x", "diamond")
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
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
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _ring_cells(r0: int, c0: int) -> list[tuple[int, int]]:
    cells = []
    for r in range(r0, r0 + 5):
        cells.append((r, c0))
        cells.append((r, c0 + 4))
    for c in range(c0 + 1, c0 + 4):
        cells.append((r0, c))
        cells.append((r0 + 4, c))
    return cells


def _classify_token(interior: np.ndarray) -> tuple[str | None, int | None]:
    colors = sorted({int(v) for v in interior.flatten()} - {BG})
    if not colors:
        return None, None
    assert len(colors) == 1, "token chamber must use a single non-background color"
    color = colors[0]
    mask = frozenset(
        (r, c) for r in range(3) for c in range(3) if int(interior[r, c]) == color
    )
    for state, expected in SHAPE_TO_MASK.items():
        if mask == expected:
            return state, color
    raise AssertionError("unrecognized 3x3 token shape")


def _rotate_shape(state: str, steps: int) -> str:
    idx = SHAPE_CYCLE.index(state)
    return SHAPE_CYCLE[(idx + steps) % 3]


def _paint_shape(grid: np.ndarray, r0: int, c0: int, state: str, color: int) -> None:
    grid[r0 + 1 : r0 + 4, c0 + 1 : c0 + 4] = BG
    for dr, dc in SHAPE_TO_MASK[state]:
        grid[r0 + 1 + dr, c0 + 1 + dc] = color


def _scan_chambers(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    chambers: list[dict] = []
    for r0 in range(h - 4):
        for c0 in range(w - 4):
            patch = g[r0 : r0 + 5, c0 : c0 + 5]
            if not (
                np.all(patch[0, :] == WIRE)
                and np.all(patch[4, :] == WIRE)
                and np.all(patch[:, 0] == WIRE)
                and np.all(patch[:, 4] == WIRE)
            ):
                continue
            if np.any(patch[1:4, 1:4] == WIRE):
                continue
            state, color = _classify_token(patch[1:4, 1:4])
            chambers.append(
                {
                    "bbox": (r0, c0, r0 + 4, c0 + 4),
                    "inner_bbox": (r0 + 1, c0 + 1, r0 + 3, c0 + 3),
                    "center": (r0 + 2, c0 + 2),
                    "state": state,
                    "color": color,
                    "border": _ring_cells(r0, c0),
                }
            )
    return chambers


def _analyze_instance(grid: np.ndarray) -> tuple[list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    chambers = _scan_chambers(g)
    assert chambers, "expected at least one chamber"

    border_mask = np.zeros_like(g, dtype=bool)
    border_to_chambers: dict[tuple[int, int], set[int]] = {}
    for idx, chamber in enumerate(chambers):
        for cell in chamber["border"]:
            border_mask[cell] = True
            border_to_chambers.setdefault(cell, set()).add(idx)

    wire_mask = (g == WIRE) & ~border_mask
    wires: list[dict] = []
    for comp in _components4(wire_mask):
        touched: set[int] = set()
        for r, c in comp:
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) in border_to_chambers:
                    touched.update(border_to_chambers[(nr, nc)])
        assert len(touched) == 2, "each wire must touch exactly two chambers"
        ids = sorted(touched)
        a = chambers[ids[0]]
        b = chambers[ids[1]]
        if a["state"] is None and b["state"] is not None:
            source_id, target_id = ids[1], ids[0]
        elif b["state"] is None and a["state"] is not None:
            source_id, target_id = ids[0], ids[1]
        else:
            raise AssertionError("each wire must connect one token chamber and one empty chamber")
        source = chambers[source_id]
        target = chambers[target_id]
        assert target["state"] is None and target["color"] is None
        target_state = _rotate_shape(source["state"], len(comp) % 3)
        wires.append(
            {
                "cells": comp,
                "length": len(comp),
                "source_id": source_id,
                "target_id": target_id,
                "source_state": source["state"],
                "target_state": target_state,
                "color": source["color"],
            }
        )
    return chambers, wires


def main(grid: np.ndarray) -> np.ndarray:
    """
    Send each 3x3 token through its teal wire, then fill the remote chamber with the length-modulated shape.

    Input structure:
    - Black background contains several 5x5 teal chamber frames with 3x3 interiors.
    - Some chamber interiors contain one colored 3x3 token shape: solid, X, or diamond.
    - Extra teal pixels outside chamber borders form 4-connected wires; each wire touches exactly one token chamber and one empty chamber.

    Rule:
    - Ignore teal pixels that belong to chamber borders; only the remaining teal pixels count as wire length.
    - For each wire, let L be its pixel count and advance the source token shape by L mod 3 along the cycle solid -> X -> diamond -> solid.
    - Draw that advanced shape in the empty chamber at the opposite end, using the source token's color.
    - Leave every source token and every teal structure unchanged.

    Key insight:
    - The wire is not just a connection cue: its physical teal pixel count encodes the shape transformation amount.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    chambers, wires = _analyze_instance(g)
    for wire in wires:
        target = chambers[wire["target_id"]]
        r0, c0, _, _ = target["bbox"]
        _paint_shape(out, r0, c0, wire["target_state"], wire["color"])
    return out


def _draw_chamber(grid: np.ndarray, r0: int, c0: int, state: str | None = None, color: int | None = None) -> None:
    grid[r0 : r0 + 5, c0 : c0 + 5] = BG
    grid[r0, c0 : c0 + 5] = WIRE
    grid[r0 + 4, c0 : c0 + 5] = WIRE
    grid[r0 : r0 + 5, c0] = WIRE
    grid[r0 : r0 + 5, c0 + 4] = WIRE
    if state is not None:
        assert color is not None
        _paint_shape(grid, r0, c0, state, color)


def _walk_segment(path: list[tuple[int, int]], nr: int, nc: int) -> None:
    r, c = path[-1]
    while r != nr:
        r += 1 if nr > r else -1
        path.append((r, c))
    while c != nc:
        c += 1 if nc > c else -1
        path.append((r, c))


def _build_wire_path(
    start: tuple[int, int],
    end: tuple[int, int],
    extra: int,
    detour_sign: int,
    detour_width: int,
    rng: random.Random,
) -> list[tuple[int, int]]:
    sr, sc = start
    er, ec = end
    assert sr == er and sc < ec
    path = [start]
    if extra == 0:
        _walk_segment(path, er, ec)
        return path

    height = extra // 2
    min_c = sc + 2
    max_c = ec - detour_width - 2
    bend_c = rng.randint(min_c, max_c)
    outer_r = sr + detour_sign * height
    _walk_segment(path, sr, bend_c)
    _walk_segment(path, outer_r, bend_c)
    _walk_segment(path, outer_r, bend_c + detour_width)
    _walk_segment(path, sr, bend_c + detour_width)
    _walk_segment(path, er, ec)
    return path


def _apply_symmetry(grid: np.ndarray, code: int) -> np.ndarray:
    if code == 0:
        return grid.copy()
    if code == 1:
        return np.fliplr(grid)
    if code == 2:
        return np.flipud(grid)
    if code == 3:
        return np.rot90(grid, 1)
    if code == 4:
        return np.rot90(grid, 2)
    if code == 5:
        return np.rot90(grid, 3)
    if code == 6:
        return np.transpose(grid)
    return np.flipud(np.transpose(grid))


def _build_attempt(rng: random.Random) -> np.ndarray:
    row_pitch = rng.choice([6, 7])
    top_margin = 1 if row_pitch == 7 else rng.randint(1, 2)
    bottom_margin = rng.randint(1, 2)
    left = rng.randint(1, 3)
    base_len = rng.randint(8, 12)
    right_margin = rng.randint(1, 3)
    right = left + 5 + base_len
    width = right + 5 + right_margin
    height = top_margin + 5 + 3 * row_pitch + bottom_margin
    assert height <= 30 and width <= 30

    grid = np.full((height, width), BG, dtype=int)
    row_tops = [top_margin + i * row_pitch for i in range(4)]

    colors = rng.sample(TOKEN_COLORS, 4)
    states = ["solid", "solid", "x", "diamond"]
    rng.shuffle(states)
    residues_by_state = {"solid": [0, 2], "x": [1], "diamond": [1]}
    assigned_residues: list[int] = []
    used = Counter()
    for state in states:
        options = residues_by_state[state]
        assigned_residues.append(options[used[state]])
        used[state] += 1

    extras: list[int] = []
    for residue in assigned_residues:
        for extra in (0, 2, 4):
            if (base_len + extra) % 3 == residue:
                extras.append(extra)
                break
        else:
            raise AssertionError("failed to realize requested residue")

    for r0, color, state in zip(row_tops, colors, states):
        _draw_chamber(grid, r0, left, state, color)
        _draw_chamber(grid, r0, right, None, None)

    occupied = set()
    for idx, (r0, extra) in enumerate(zip(row_tops, extras)):
        center_r = r0 + 2
        start = (center_r, left + 5)
        end = (center_r, right - 1)
        sign = -1 if idx % 2 == 0 else 1
        if extra:
            detour_row = center_r + sign * (extra // 2)
            if not (0 <= detour_row < height):
                sign *= -1
        path = _build_wire_path(start, end, extra, sign, rng.randint(1, 2), rng)
        for cell in path:
            if cell in occupied:
                raise AssertionError("wires overlapped")
            rr, cc = cell
            if grid[rr, cc] != BG:
                raise AssertionError("wire crossed a chamber")
            occupied.add(cell)
            grid[rr, cc] = WIRE

    grid = _apply_symmetry(grid, rng.randint(0, 7))
    _validate_generated(grid)
    return grid


def _validate_generated(grid: np.ndarray) -> None:
    chambers, wires = _analyze_instance(grid)
    assert len(chambers) == 8, "generator must create four source chambers and four target chambers"
    assert len(wires) == 4, "generator must create four independent wires"

    source_colors = [chambers[w["source_id"]]["color"] for w in wires]
    assert len(set(source_colors)) == 4, "source token colors must be unique"

    residues = [w["length"] % 3 for w in wires]
    assert set(residues) == {0, 1, 2}, "generator must cover all length mod 3 branches"

    states = [w["source_state"] for w in wires]
    target_states = [w["target_state"] for w in wires]
    assert states.count("solid") == 2 and "x" in states and "diamond" in states

    solid_targets = {
        w["target_state"]
        for w in wires
        if w["source_state"] == "solid"
    }
    assert len(solid_targets) == 2, "same source shape must yield different outputs under different wire lengths"

    same_residue_states = {
        residues[i]: {states[j] for j in range(len(wires)) if residues[j] == residues[i]}
        for i in range(len(wires))
    }
    assert any(len(v) >= 2 for v in same_residue_states.values()), "need repeated residue across different source shapes"

    out = main(grid)
    changed_targets = 0
    for wire in wires:
        target = chambers[wire["target_id"]]
        r0, c0, _, _ = target["bbox"]
        interior = out[r0 + 1 : r0 + 4, c0 + 1 : c0 + 4]
        state, color = _classify_token(interior)
        assert state == wire["target_state"]
        assert color == wire["color"]
        changed_targets += 1
    assert changed_targets == 4

    # Source chambers stay unchanged.
    for wire in wires:
        source = chambers[wire["source_id"]]
        r0, c0, _, _ = source["bbox"]
        before = grid[r0 + 1 : r0 + 4, c0 + 1 : c0 + 4]
        after = out[r0 + 1 : r0 + 4, c0 + 1 : c0 + 4]
        assert np.array_equal(before, after)


def generate_input() -> np.ndarray:
    for _ in range(300):
        seed = random.randrange(1 << 30)
        rng = random.Random(seed)
        try:
            return _build_attempt(rng)
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid instance")
