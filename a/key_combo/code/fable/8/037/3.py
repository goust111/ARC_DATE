from __future__ import annotations

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[int(np.argmax(counts))])


def _claim_disjoint(seen: set[tuple[int, int]], coords: list[tuple[int, int]]) -> None:
    for coord in coords:
        assert coord not in seen, "operator domains must be disjoint"
    seen.update(coords)


def _rotate_block_cw(block: np.ndarray) -> np.ndarray:
    tl = int(block[0, 0])
    tr = int(block[0, 1])
    br = int(block[1, 1])
    bl = int(block[1, 0])
    return np.array([[bl, tl], [br, tr]], dtype=int)


def _rotate_block_ccw(block: np.ndarray) -> np.ndarray:
    tl = int(block[0, 0])
    tr = int(block[0, 1])
    br = int(block[1, 1])
    bl = int(block[1, 0])
    return np.array([[tr, br], [tl, bl]], dtype=int)


def _apply_ops_to_cells(cells: np.ndarray, ops: list[dict], *, inverse: bool) -> np.ndarray:
    out = np.asarray(cells, dtype=int).copy()
    for op in ops:
        kind = op["kind"]
        r = int(op["r"])
        c = int(op["c"])
        if kind == "v":
            out[r, c], out[r, c + 1] = out[r, c + 1], out[r, c]
        elif kind == "h":
            out[r, c], out[r + 1, c] = out[r + 1, c], out[r, c]
        else:
            block = out[r : r + 2, c : c + 2]
            out[r : r + 2, c : c + 2] = (
                _rotate_block_ccw(block) if inverse else _rotate_block_cw(block)
            )
    return out


def _cells_for_op(kind: str, r: int, c: int) -> list[tuple[int, int]]:
    if kind == "v":
        return [(r, c), (r, c + 1)]
    if kind == "h":
        return [(r, c), (r + 1, c)]
    return [(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)]


def _marker_for_op(kind: str, r: int, c: int) -> tuple[int, int]:
    if kind == "v":
        return 2 * r, 2 * c + 1
    if kind == "h":
        return 2 * r + 1, 2 * c
    return 2 * r + 1, 2 * c + 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Apply separator-line operator markers before downsampling the hidden checkerboard of content cells.

    Input structure:
    - The grid has even height and even width.
    - Only even-row/even-column positions carry the content colors that survive into the output.
    - Odd rows and odd columns are separator lines filled with the background except for a few marker pixels.

    Rule:
    - A marker on an even row and odd column swaps the two neighboring content cells to its left and right.
    - A marker on an odd row and even column swaps the two neighboring content cells above and below it.
    - A marker on an odd row and odd column rotates its four diagonal neighboring content cells one step clockwise.
    - After all marked operations are executed on the full-resolution content grid, delete every odd row and odd column.

    Key insight:
    - The separator marks are operators, not output pixels: their row/column parity selects which local swap or 2x2 rotation must happen before compression.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height % 2 == 0 and width % 2 == 0, "grid shape must be even-by-even"

    bg = _get_bg(g)
    cells = g[::2, ::2].copy()
    cell_h, cell_w = map(int, cells.shape)
    seen: set[tuple[int, int]] = set()

    for r_raw, c_raw in np.argwhere(g != bg):
        r = int(r_raw)
        c = int(c_raw)
        if r % 2 == 0 and c % 2 == 0:
            continue

        if r % 2 == 0:
            rr = r // 2
            cc = c // 2
            assert 0 <= rr < cell_h and 0 <= cc < cell_w - 1
            _claim_disjoint(seen, [(rr, cc), (rr, cc + 1)])
            cells[rr, cc], cells[rr, cc + 1] = cells[rr, cc + 1], cells[rr, cc]
        elif c % 2 == 0:
            rr = r // 2
            cc = c // 2
            assert 0 <= rr < cell_h - 1 and 0 <= cc < cell_w
            _claim_disjoint(seen, [(rr, cc), (rr + 1, cc)])
            cells[rr, cc], cells[rr + 1, cc] = cells[rr + 1, cc], cells[rr, cc]
        else:
            rr = r // 2
            cc = c // 2
            assert 0 <= rr < cell_h - 1 and 0 <= cc < cell_w - 1
            _claim_disjoint(seen, [(rr, cc), (rr, cc + 1), (rr + 1, cc), (rr + 1, cc + 1)])
            cells[rr : rr + 2, cc : cc + 2] = _rotate_block_cw(cells[rr : rr + 2, cc : cc + 2])

    return cells


_GEN_RNG = np.random.default_rng()

_GEN_DIM_RANGE = (5, 9)
_GEN_TOTAL_OP_RANGE = (3, 6)
_GEN_ALL_COLORS = tuple(range(10))
_GEN_RETRY_LIMIT = 300
_GEN_TYPE_ORDER = ("v", "h", "x")


def _gen_choice(seq):
    return seq[int(_GEN_RNG.integers(len(seq)))]


def _gen_candidate_ops(cell_h: int, cell_w: int) -> dict[str, list[dict]]:
    candidates = {"v": [], "h": [], "x": []}
    for r in range(cell_h):
        for c in range(cell_w - 1):
            candidates["v"].append({"kind": "v", "r": r, "c": c})
    for r in range(cell_h - 1):
        for c in range(cell_w):
            candidates["h"].append({"kind": "h", "r": r, "c": c})
    for r in range(cell_h - 1):
        for c in range(cell_w - 1):
            candidates["x"].append({"kind": "x", "r": r, "c": c})
    return candidates


def _gen_pick_ops(cell_h: int, cell_w: int) -> list[dict] | None:
    area = cell_h * cell_w
    max_ops = min(_GEN_TOTAL_OP_RANGE[1], max(_GEN_TOTAL_OP_RANGE[0], (area - 1) // 2))
    total_ops = int(_GEN_RNG.integers(_GEN_TOTAL_OP_RANGE[0], max_ops + 1))

    op_kinds = list(_GEN_TYPE_ORDER)
    if total_ops > len(op_kinds):
        op_kinds.extend(
            str(k) for k in _GEN_RNG.choice(_GEN_TYPE_ORDER, size=total_ops - len(op_kinds), replace=True)
        )
    _GEN_RNG.shuffle(op_kinds)

    candidates = _gen_candidate_ops(cell_h, cell_w)
    occupied = np.zeros((cell_h, cell_w), dtype=bool)
    ops: list[dict] = []
    for kind in op_kinds:
        order = _GEN_RNG.permutation(len(candidates[kind]))
        chosen = None
        for idx in order:
            op = candidates[kind][int(idx)]
            cells = _cells_for_op(kind, int(op["r"]), int(op["c"]))
            if any(occupied[r, c] for r, c in cells):
                continue
            if len(cells) == area:
                continue
            chosen = {
                "kind": kind,
                "r": int(op["r"]),
                "c": int(op["c"]),
                "cells": cells,
                "marker": _marker_for_op(kind, int(op["r"]), int(op["c"])),
            }
            for r, c in cells:
                occupied[r, c] = True
            ops.append(chosen)
            break
        if chosen is None:
            return None

    assert len({op["marker"] for op in ops}) == len(ops)
    assert all(kind in {op["kind"] for op in ops} for kind in _GEN_TYPE_ORDER)
    return ops


def _gen_palette() -> tuple[int, int, list[int]]:
    bg, marker = _GEN_RNG.choice(_GEN_ALL_COLORS, size=2, replace=False)
    usable = [color for color in _GEN_ALL_COLORS if color not in {int(bg), int(marker)}]
    n_content = int(_GEN_RNG.integers(4, min(7, len(usable)) + 1))
    content = [int(color) for color in _GEN_RNG.choice(usable, size=n_content, replace=False)]
    return int(bg), int(marker), content


def _gen_output_cells(cell_h: int, cell_w: int, content_palette: list[int], ops: list[dict]) -> np.ndarray:
    out = _GEN_RNG.choice(content_palette, size=(cell_h, cell_w), replace=True).astype(int)

    for op in ops:
        cells = op["cells"]
        if op["kind"] in {"v", "h"}:
            (r0, c0), (r1, c1) = cells
            if int(out[r0, c0]) == int(out[r1, c1]):
                choices = [color for color in content_palette if color != int(out[r0, c0])]
                out[r1, c1] = _gen_choice(choices)
        else:
            vals = [int(out[r, c]) for r, c in cells]
            if len(set(vals)) < 3:
                replace_idx = int(_GEN_RNG.integers(len(cells)))
                rr, cc = cells[replace_idx]
                choices = [color for color in content_palette if color not in set(vals)]
                if not choices:
                    choices = [color for color in content_palette if color != vals[replace_idx]]
                out[rr, cc] = _gen_choice(choices)

    if len(set(map(int, out.ravel()))) < 3:
        flat = out.ravel()
        flat[0] = int(content_palette[0])
        flat[-1] = int(content_palette[1])
        flat[len(flat) // 2] = int(content_palette[2])
        out = flat.reshape((cell_h, cell_w))
    return out


def _gen_render_input(pre_cells: np.ndarray, bg: int, marker_color: int, ops: list[dict]) -> np.ndarray:
    cell_h, cell_w = map(int, pre_cells.shape)
    grid = np.full((2 * cell_h, 2 * cell_w), int(bg), dtype=int)
    grid[::2, ::2] = pre_cells
    for op in ops:
        mr, mc = op["marker"]
        grid[mr, mc] = int(marker_color)
    return grid


def _gen_validate(inp: np.ndarray, out: np.ndarray, pre_cells: np.ndarray, ops: list[dict], bg: int) -> bool:
    if not np.array_equal(main(inp), out):
        return False

    if np.array_equal(pre_cells, out):
        return False

    if np.any(inp[1::2, 1::2] != bg):
        marker_points = {op["marker"] for op in ops if op["kind"] == "x"}
        for r in range(1, inp.shape[0], 2):
            for c in range(1, inp.shape[1], 2):
                if (r, c) not in marker_points and int(inp[r, c]) != int(bg):
                    return False

    seen: set[tuple[int, int]] = set()
    total_changed = 0
    for op in ops:
        cells = op["cells"]
        if any(coord in seen for coord in cells):
            return False
        seen.update(cells)
        before = [int(pre_cells[r, c]) for r, c in cells]
        after = [int(out[r, c]) for r, c in cells]
        if before == after:
            return False
        total_changed += sum(a != b for a, b in zip(before, after))

    if total_changed < 6:
        return False

    if len({op["kind"] for op in ops}) != 3:
        return False

    if np.any(inp[1::2, ::2] != bg):
        allowed = {op["marker"] for op in ops if op["kind"] == "h"}
        for r in range(1, inp.shape[0], 2):
            for c in range(0, inp.shape[1], 2):
                if (r, c) not in allowed and int(inp[r, c]) != int(bg):
                    return False

    if np.any(inp[::2, 1::2] != bg):
        allowed = {op["marker"] for op in ops if op["kind"] == "v"}
        for r in range(0, inp.shape[0], 2):
            for c in range(1, inp.shape[1], 2):
                if (r, c) not in allowed and int(inp[r, c]) != int(bg):
                    return False

    return True


def _gen_attempt() -> np.ndarray | None:
    cell_h = int(_GEN_RNG.integers(*_GEN_DIM_RANGE))
    cell_w = int(_GEN_RNG.integers(*_GEN_DIM_RANGE))
    ops = _gen_pick_ops(cell_h, cell_w)
    if ops is None:
        return None

    bg, marker_color, content_palette = _gen_palette()
    out_cells = _gen_output_cells(cell_h, cell_w, content_palette, ops)
    pre_cells = _apply_ops_to_cells(out_cells, ops, inverse=True)
    inp = _gen_render_input(pre_cells, bg, marker_color, ops)

    if not _gen_validate(inp, out_cells, pre_cells, ops, bg):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid separator-operator input")
