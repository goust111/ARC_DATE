from __future__ import annotations

import math
import random

import numpy as np


BG = 0
SEP = 1
FRAME = 5
BOX = 5
INNER = 3
UNIT = 6
PAIR_W = 12
GAP = 1
KEY_POOL = (2, 3, 4, 6, 7, 8, 9)
ALL_OPS = ("flip_lr", "flip_ud", "rot_cw", "swap_colors")
MAX_GEN_RETRIES = 400


def _apply_op(patch: np.ndarray, op: str) -> np.ndarray:
    if op == "flip_lr":
        return np.fliplr(patch)
    if op == "flip_ud":
        return np.flipud(patch)
    if op == "rot_cw":
        return np.rot90(patch, -1)
    if op == "swap_colors":
        colors = [int(c) for c in np.unique(patch) if int(c) != BG]
        if len(colors) != 2:
            raise ValueError("swap_colors needs exactly two non-background colors")
        a, b = colors
        out = patch.copy()
        out[patch == a] = b
        out[patch == b] = a
        return out
    raise ValueError(f"unknown op: {op}")


def _infer_op(src: np.ndarray, dst: np.ndarray) -> str:
    matches = []
    for op in ALL_OPS:
        try:
            cand = _apply_op(src, op)
        except ValueError:
            continue
        if np.array_equal(cand, dst):
            matches.append(op)
    assert len(matches) == 1, "demo pair must define a unique operation"
    return matches[0]


def _is_box(g: np.ndarray, r0: int, c0: int) -> bool:
    if r0 + BOX > g.shape[0] or c0 + BOX > g.shape[1]:
        return False
    patch = g[r0 : r0 + BOX, c0 : c0 + BOX]
    if not np.all(patch[0, :] == FRAME):
        return False
    if not np.all(patch[-1, :] == FRAME):
        return False
    if not np.all(patch[:, 0] == FRAME):
        return False
    if not np.all(patch[:, -1] == FRAME):
        return False
    return not np.any(patch[1:-1, 1:-1] == FRAME)


def _find_boxes(g: np.ndarray) -> list[tuple[int, int, int, int]]:
    boxes = []
    for r0 in range(g.shape[0] - BOX + 1):
        for c0 in range(g.shape[1] - BOX + 1):
            if _is_box(g, r0, c0):
                boxes.append((r0, c0, r0 + BOX - 1, c0 + BOX - 1))
    assert boxes, "expected framed boxes"
    return boxes


def _connected(mask: np.ndarray) -> bool:
    cells = [tuple(x) for x in np.argwhere(mask)]
    if not cells:
        return False
    seen = {cells[0]}
    stack = [cells[0]]
    while stack:
        r, c = stack.pop()
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= rr < INNER and 0 <= cc < INNER and mask[rr, cc] and (rr, cc) not in seen:
                seen.add((rr, cc))
                stack.append((rr, cc))
    return len(seen) == len(cells)


def _random_mask() -> np.ndarray:
    target = random.randint(3, 6)
    for _ in range(100):
        mask = np.zeros((INNER, INNER), dtype=bool)
        start = (random.randrange(INNER), random.randrange(INNER))
        mask[start] = True
        while int(mask.sum()) < target:
            frontier = []
            for r, c in np.argwhere(mask):
                for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= rr < INNER and 0 <= cc < INNER and not mask[rr, cc]:
                        frontier.append((rr, cc))
            if not frontier:
                break
            rr, cc = random.choice(frontier)
            mask[rr, cc] = True
        coords = np.argwhere(mask)
        if int(mask.sum()) != target:
            continue
        if not _connected(mask):
            continue
        if len({int(r) for r, _c in coords}) < 2 or len({int(c) for _r, c in coords}) < 2:
            continue
        return mask
    raise RuntimeError("failed to build a non-degenerate mask")


def _colorize(mask: np.ndarray, colors: list[int]) -> np.ndarray:
    out = np.zeros((INNER, INNER), dtype=int)
    cells = [tuple(x) for x in np.argwhere(mask)]
    random.shuffle(cells)
    if len(colors) == 1:
        for r, c in cells:
            out[r, c] = colors[0]
        return out

    split = random.randint(1, len(cells) - 1)
    for r, c in cells[:split]:
        out[r, c] = colors[0]
    for r, c in cells[split:]:
        out[r, c] = colors[1]
    if np.sum(out == colors[0]) == 0 or np.sum(out == colors[1]) == 0:
        raise RuntimeError("both colors must appear")
    return out


def _make_pattern(op: str, palette: list[int], used: set[bytes]) -> np.ndarray:
    for _ in range(300):
        mask = _random_mask()
        n_colors = 2 if op == "swap_colors" else random.choice((1, 2))
        colors = random.sample(palette, n_colors)
        patch = _colorize(mask, colors)
        try:
            dst = _apply_op(patch, op)
        except ValueError:
            continue
        if np.array_equal(dst, patch):
            continue
        matches = []
        for cand_op in ALL_OPS:
            try:
                cand = _apply_op(patch, cand_op)
            except ValueError:
                continue
            if np.array_equal(cand, dst):
                matches.append(cand_op)
        if matches != [op]:
            continue
        sig = patch.tobytes() + b"|" + dst.tobytes()
        if sig in used:
            continue
        used.add(sig)
        return patch
    raise RuntimeError(f"failed to make a unique pattern for {op}")


def _frame(inner: np.ndarray) -> np.ndarray:
    box = np.full((BOX, BOX), FRAME, dtype=int)
    box[1:-1, 1:-1] = inner
    return box


def _legend_dims(num_ops: int) -> tuple[int, int, int]:
    cols = 2
    rows = int(math.ceil(num_ops / cols))
    h = rows * UNIT + (rows - 1) * GAP
    w = cols * PAIR_W + (cols - 1) * GAP
    return rows, h, w


def _main_dims(num_tasks: int, cols: int) -> tuple[int, int, int]:
    rows = int(math.ceil(num_tasks / cols))
    h = rows * UNIT + (rows - 1) * GAP
    w = cols * UNIT + (cols - 1) * GAP
    return rows, h, w


def _render_pair(canvas: np.ndarray, anchor_r: int, anchor_c: int, key: int, src: np.ndarray, dst: np.ndarray) -> None:
    canvas[anchor_r, anchor_c] = key
    canvas[anchor_r + 1 : anchor_r + 1 + BOX, anchor_c + 1 : anchor_c + 1 + BOX] = _frame(src)
    canvas[anchor_r + 1 : anchor_r + 1 + BOX, anchor_c + 7 : anchor_c + 7 + BOX] = _frame(dst)


def _render_unit(canvas: np.ndarray, anchor_r: int, anchor_c: int, key: int, src: np.ndarray) -> None:
    canvas[anchor_r, anchor_c] = key
    canvas[anchor_r + 1 : anchor_r + 1 + BOX, anchor_c + 1 : anchor_c + 1 + BOX] = _frame(src)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Learn each key color's transformation from top demo pairs, then apply it to the lower keyed boxes.

    Input structure:
    - A full-width separator row splits the grid into a top legend area and a lower task area.
    - Every picture sits inside a 5x5 frame whose 3x3 interior contains the actual pattern.
    - In the legend, each key pixel marks a left demo box and its right neighbor box; the right box is the transformed result.
    - In the lower area, each framed pattern has one key pixel at its upper-left outside corner.

    Rule:
    - For each top demo pair, compare the left and right interiors to infer the unique operation for that key color.
    - Supported operations are left-right mirror, up-down mirror, clockwise quarter-turn, and swapping the two non-background colors.
    - In the lower area, replace each interior by the result of applying its key color's learned operation.
    - Remove all lower key pixels and crop away the entire legend so the output keeps only the lower task area.

    Key insight:
    - The key colors do not have fixed meanings; the upper demo pairs teach the color-to-operation mapping that must then be reused below.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sep_rows = [r for r in range(g.shape[0]) if np.all(g[r, :] == SEP)]
    assert len(sep_rows) == 1, "expected one separator row"
    sep_r = sep_rows[0]

    boxes = _find_boxes(g)
    legend_boxes = [b for b in boxes if b[0] < sep_r]
    task_boxes = [b for b in boxes if b[0] > sep_r]
    assert legend_boxes and task_boxes, "expected both legend and task boxes"

    op_by_key: dict[int, str] = {}
    legend_box_set = {(r0, c0) for r0, c0, _r1, _c1 in legend_boxes}
    for r0, c0, _r1, _c1 in sorted(legend_boxes):
        if r0 == 0 or c0 == 0:
            continue
        key = int(g[r0 - 1, c0 - 1])
        if key in (BG, SEP, FRAME):
            continue
        partner_pos = (r0, c0 + 6)
        assert partner_pos in legend_box_set, "legend left box must have a right partner"
        src = g[r0 + 1 : r0 + 1 + INNER, c0 + 1 : c0 + 1 + INNER]
        dst = g[r0 + 1 : r0 + 1 + INNER, c0 + 7 : c0 + 7 + INNER]
        op_by_key[key] = _infer_op(src, dst)

    assert len(op_by_key) >= 3, "expected multiple learned mappings"

    min_r = min(r0 - 1 for r0, _c0, _r1, _c1 in task_boxes)
    min_c = min(c0 - 1 for _r0, c0, _r1, _c1 in task_boxes)
    max_r = max(r1 for _r0, _c0, r1, _c1 in task_boxes)
    max_c = max(c1 for _r0, _c0, _r1, c1 in task_boxes)
    out = g[min_r : max_r + 1, min_c : max_c + 1].copy()

    for r0, c0, _r1, _c1 in sorted(task_boxes):
        key = int(g[r0 - 1, c0 - 1])
        assert key in op_by_key, "every lower key must be defined by a demo"
        src = g[r0 + 1 : r0 + 1 + INNER, c0 + 1 : c0 + 1 + INNER]
        dst = _apply_op(src, op_by_key[key])
        rr = r0 - min_r
        cc = c0 - min_c
        out[rr - 1, cc - 1] = BG
        out[rr + 1 : rr + 1 + INNER, cc + 1 : cc + 1 + INNER] = dst

    return out


def _gen_attempt() -> np.ndarray | None:
    num_ops = random.choice((3, 4))
    ops = random.sample(list(ALL_OPS), num_ops)
    key_colors = random.sample(list(KEY_POOL), num_ops)
    op_by_key = dict(zip(key_colors, random.sample(ops, len(ops))))
    object_palette = [c for c in KEY_POOL if c not in key_colors]
    if len(object_palette) < 2:
        return None

    used_patterns: set[bytes] = set()
    demo_src: dict[int, np.ndarray] = {}
    demo_dst: dict[int, np.ndarray] = {}
    for key, op in op_by_key.items():
        src = _make_pattern(op, object_palette, used_patterns)
        demo_src[key] = src
        demo_dst[key] = _apply_op(src, op)

    num_tasks = random.randint(num_ops + 1, min(6, num_ops + 3))
    task_keys = key_colors[:]
    while len(task_keys) < num_tasks:
        task_keys.append(random.choice(key_colors))
    random.shuffle(task_keys)

    task_src = []
    task_used_outs: set[bytes] = set()
    for key in task_keys:
        op = op_by_key[key]
        for _ in range(200):
            src = _make_pattern(op, object_palette, used_patterns)
            dst = _apply_op(src, op)
            if dst.tobytes() in task_used_outs:
                continue
            task_used_outs.add(dst.tobytes())
            task_src.append(src)
            break
        else:
            return None

    legend_rows, legend_h, legend_w = _legend_dims(num_ops)
    legend_top = random.randint(0, 1)
    legend_left = random.randint(0, 2)

    main_cols_choices = [2, 3]
    random.shuffle(main_cols_choices)
    chosen_main = None
    for main_cols in main_cols_choices:
        main_rows, main_h, main_w = _main_dims(num_tasks, main_cols)
        main_top = legend_top + legend_h + 1 + random.randint(0, 1)
        bottom_pad = random.randint(0, 1)
        main_left = random.randint(0, 2)
        right_pad = random.randint(0, 2)
        width = max(legend_left + legend_w, main_left + main_w) + right_pad
        height = main_top + main_h + bottom_pad
        if height <= 30 and width <= 30:
            chosen_main = (main_cols, main_rows, main_h, main_w, main_top, main_left, height, width)
            break
    if chosen_main is None:
        return None

    main_cols, _main_rows, _main_h, _main_w, main_top, main_left, height, width = chosen_main
    canvas = np.full((height, width), BG, dtype=int)
    sep_r = legend_top + legend_h
    canvas[sep_r, :] = SEP

    sorted_keys = list(op_by_key)
    for idx, key in enumerate(sorted_keys):
        er = idx // 2
        ec = idx % 2
        anchor_r = legend_top + er * (UNIT + GAP)
        anchor_c = legend_left + ec * (PAIR_W + GAP)
        _render_pair(canvas, anchor_r, anchor_c, key, demo_src[key], demo_dst[key])

    task_anchors = []
    for idx, (key, src) in enumerate(zip(task_keys, task_src)):
        mr = idx // main_cols
        mc = idx % main_cols
        anchor_r = main_top + mr * (UNIT + GAP)
        anchor_c = main_left + mc * (UNIT + GAP)
        task_anchors.append((anchor_r, anchor_c, key, src))
        _render_unit(canvas, anchor_r, anchor_c, key, src)

    min_r = min(r for r, _c, _k, _src in task_anchors)
    min_c = min(c for _r, c, _k, _src in task_anchors)
    max_r = max(r + UNIT - 1 for r, _c, _k, _src in task_anchors)
    max_c = max(c + UNIT - 1 for _r, c, _k, _src in task_anchors)
    expected = canvas[min_r : max_r + 1, min_c : max_c + 1].copy()

    for r, c, key, src in task_anchors:
        op = op_by_key[key]
        rr = r - min_r
        cc = c - min_c
        expected[rr, cc] = BG
        expected[rr + 2 : rr + 2 + INNER, cc + 2 : cc + 2 + INNER] = _apply_op(src, op)

    try:
        actual = main(canvas)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    if len({_apply_op(src, op_by_key[key]).tobytes() for _r, _c, key, src in task_anchors}) < 3:
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid demo-legend ARC input")
