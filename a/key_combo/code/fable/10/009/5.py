import random
from itertools import permutations

import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each top block to the slot above its matching key, then fill the block's hole with the key color.

    Input structure:
    - The upper area contains several equal-width vertical color blocks laid side by side.
    - Each block has one unique main color and contains one background-colored hole shape fully inside the block.
    - A full background row separates the blocks from a lower area containing one colored key shape per destination slot.

    Rule:
    - Each key encodes three things at once: its shape identifies the unique source block whose hole has the same shape, its horizontal slot identifies that block's destination position, and its color identifies the hole-fill color.
    - Move every matched source block as a whole into the slot above its key, preserving all block pixels and the hole's relative position.
    - After the simultaneous permutation, fill each moved block's hole with its matching key color and clear the lower key area to background.

    Key insight:
    - Keys do not pair with blocks by left-to-right order; only exact hole-shape equality selects the source block, while key position independently selects the destination slot.
    """
    g = np.asarray(grid, dtype=int)
    info = _analyze_input(g)
    return _render_output(g, info)


BG = 0
COLOR_POOL = list(range(1, 10))
BLOCK_COUNT_CHOICES = (3, 4)
BLOCK_WIDTH_CHOICES = (6, 7)
TOP_HEIGHT_RANGE = (6, 8)
BOTTOM_HEIGHT_RANGE = (5, 8)
OUTER_MARGIN_CHOICES = (0, 1)
VERTICAL_MARGIN_CHOICES = (0, 1, 2)
MAX_ATTEMPTS = 400


def _shape(lines):
    coords = []
    for r, row in enumerate(lines):
        for c, ch in enumerate(row):
            if ch == "1":
                coords.append((r, c))
    return frozenset(coords)


SHAPE_LIBRARY = [
    _shape(["11", "10"]),
    _shape(["11", "01"]),
    _shape(["111", "010"]),
    _shape(["010", "111"]),
    _shape(["110", "011"]),
    _shape(["011", "110"]),
    _shape(["111", "100"]),
    _shape(["001", "111"]),
    _shape(["110", "111"]),
    _shape(["011", "111"]),
    _shape(["111", "101"]),
    _shape(["010", "111", "010"]),
    _shape(["110", "010", "011"]),
]


def _normalize(coords):
    cells = list(coords)
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return frozenset((int(r - r0), int(c - c0)) for r, c in cells)


def _shape_bbox(shape):
    rs = [r for r, _ in shape]
    cs = [c for _, c in shape]
    return max(rs) + 1, max(cs) + 1


def _is_connected(coords):
    cells = set(coords)
    if not cells:
        return False
    stack = [next(iter(cells))]
    seen = {stack[0]}
    while stack:
        r, c = stack.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(cells)


def _get_bg(grid):
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[np.argmax(counts)])


def _analyze_input(grid):
    assert grid.ndim == 2
    h, w = map(int, grid.shape)
    bg = _get_bg(grid)

    active_rows = np.any(grid != bg, axis=1)
    active_idx = np.flatnonzero(active_rows)
    assert active_idx.size > 0

    top_r0 = int(active_idx[0])
    top_r1 = top_r0
    while top_r1 + 1 < h and active_rows[top_r1 + 1]:
        top_r1 += 1
    sep_r = top_r1 + 1
    assert sep_r < h
    assert np.all(grid[sep_r] == bg)
    assert np.any(active_rows[sep_r + 1 :])

    top = grid[top_r0 : top_r1 + 1]
    top_h = int(top.shape[0])
    top_colors = [int(c) for c in np.unique(top) if int(c) != bg]
    assert len(top_colors) >= 2

    blocks = []
    for color in top_colors:
        coords = [tuple(map(int, rc)) for rc in np.argwhere(top == color)]
        assert _is_connected(coords)
        rs = [r for r, _ in coords]
        cs = [c for _, c in coords]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        assert r0 == 0 and r1 == top_h - 1
        patch = top[:, c0 : c1 + 1]
        assert np.all((patch == color) | (patch == bg))

        hole = [tuple(map(int, rc)) for rc in np.argwhere(patch == bg)]
        assert hole
        hole_rs = [r for r, _ in hole]
        hole_cs = [c for _, c in hole]
        assert min(hole_rs) > 0 and max(hole_rs) < top_h - 1
        assert min(hole_cs) > 0 and max(hole_cs) < (c1 - c0)
        hole_cells = frozenset(hole)
        hole_shape = _normalize(hole)
        blocks.append(
            {
                "color": color,
                "c0": int(c0),
                "c1": int(c1),
                "width": int(c1 - c0 + 1),
                "hole_cells": hole_cells,
                "hole_shape": hole_shape,
                "hole_area": len(hole_shape),
                "hole_bbox": _shape_bbox(hole_shape),
            }
        )

    blocks.sort(key=lambda b: b["c0"])
    widths = {b["width"] for b in blocks}
    assert len(widths) == 1
    for i in range(1, len(blocks)):
        assert blocks[i - 1]["c1"] + 1 == blocks[i]["c0"]
    assert len({b["hole_shape"] for b in blocks}) == len(blocks)

    bottom = grid[sep_r + 1 :]
    key_colors = [int(c) for c in np.unique(bottom) if int(c) != bg]
    assert len(key_colors) == len(blocks)

    slot_to_key = {}
    source_to_dest = {}
    keys = []
    for color in key_colors:
        coords = [tuple(map(int, rc)) for rc in np.argwhere(bottom == color)]
        assert _is_connected(coords)
        rs = [r for r, _ in coords]
        cs = [c for _, c in coords]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        shape = frozenset((int(r - r0), int(c - c0)) for r, c in coords)

        matching_slots = [i for i, b in enumerate(blocks) if b["c0"] <= c0 and c1 <= b["c1"]]
        assert len(matching_slots) == 1
        dest_slot = matching_slots[0]
        assert dest_slot not in slot_to_key
        slot_to_key[dest_slot] = color

        matching_sources = [i for i, b in enumerate(blocks) if b["hole_shape"] == shape]
        assert len(matching_sources) == 1
        source_idx = matching_sources[0]
        assert source_idx not in source_to_dest
        source_to_dest[source_idx] = dest_slot

        keys.append(
            {
                "color": color,
                "shape": shape,
                "dest_slot": dest_slot,
                "source_idx": source_idx,
            }
        )

    n = len(blocks)
    assert len(slot_to_key) == n
    assert len(source_to_dest) == n
    assert sorted(slot_to_key) == list(range(n))
    assert sorted(source_to_dest) == list(range(n))

    return {
        "bg": bg,
        "top_r0": top_r0,
        "top_r1": top_r1,
        "sep_r": sep_r,
        "blocks": blocks,
        "keys": keys,
        "source_to_dest": source_to_dest,
    }


def _render_output(grid, info):
    out = np.full_like(grid, info["bg"])
    top_r0 = info["top_r0"]
    top_r1 = info["top_r1"]
    blocks = info["blocks"]

    keys_by_source = {k["source_idx"]: k for k in info["keys"]}
    for source_idx, dest_slot in info["source_to_dest"].items():
        src = blocks[source_idx]
        dst = blocks[dest_slot]
        patch = grid[top_r0 : top_r1 + 1, src["c0"] : src["c1"] + 1]
        out[top_r0 : top_r1 + 1, dst["c0"] : dst["c1"] + 1] = patch
        fill_color = keys_by_source[source_idx]["color"]
        for dr, dc in src["hole_cells"]:
            out[top_r0 + dr, dst["c0"] + dc] = fill_color
    return out


def _sample_shapes(count):
    for _ in range(200):
        shapes = random.sample(SHAPE_LIBRARY, count)
        areas = [_shape_area(s) for s in shapes]
        bboxes = [_shape_bbox(s) for s in shapes]
        if len(set(shapes)) != count:
            continue
        if len(set(areas)) == count:
            continue
        if len(set(bboxes)) == count:
            continue
        return shapes
    raise RuntimeError("failed to sample informative shapes")


def _shape_area(shape):
    return len(shape)


def _sample_permutation(n):
    choices = [perm for perm in permutations(range(n)) if any(perm[i] != i for i in range(n))]
    return list(random.choice(choices))


def _same_slot_shortcut(grid, info):
    out = np.full_like(grid, info["bg"])
    top_r0 = info["top_r0"]
    top_r1 = info["top_r1"]
    blocks = info["blocks"]
    key_by_slot = {k["dest_slot"]: k for k in info["keys"]}
    for slot, block in enumerate(blocks):
        patch = grid[top_r0 : top_r1 + 1, block["c0"] : block["c1"] + 1]
        out[top_r0 : top_r1 + 1, block["c0"] : block["c1"] + 1] = patch
        fill_color = key_by_slot[slot]["color"]
        for dr, dc in block["hole_cells"]:
            out[top_r0 + dr, block["c0"] + dc] = fill_color
    return out


def _gen_attempt():
    n = random.choice(BLOCK_COUNT_CHOICES)
    block_w = random.choice(BLOCK_WIDTH_CHOICES)
    top_h = random.randint(*TOP_HEIGHT_RANGE)
    bottom_h = random.randint(*BOTTOM_HEIGHT_RANGE)
    left_margin = random.choice(OUTER_MARGIN_CHOICES)
    right_margin = random.choice(OUTER_MARGIN_CHOICES)
    top_margin = random.choice(VERTICAL_MARGIN_CHOICES)
    bottom_margin = random.choice(VERTICAL_MARGIN_CHOICES)

    width = left_margin + n * block_w + right_margin
    height = top_margin + top_h + 1 + bottom_h + bottom_margin
    if width > 30 or height > 30:
        return None

    colors = random.sample(COLOR_POOL, 2 * n)
    block_colors = colors[:n]
    key_colors = colors[n:]
    shapes = _sample_shapes(n)
    perm = _sample_permutation(n)

    grid = np.full((height, width), BG, dtype=int)
    top_r0 = top_margin
    top_r1 = top_r0 + top_h - 1
    bottom_r0 = top_r1 + 2

    chosen_shapes = []
    for i in range(n):
        c0 = left_margin + i * block_w
        c1 = c0 + block_w
        grid[top_r0 : top_r1 + 1, c0:c1] = block_colors[i]

        shape = shapes[i]
        sh_h, sh_w = _shape_bbox(shape)
        hole_r0 = random.randint(1, top_h - sh_h - 1)
        hole_c0 = random.randint(1, block_w - sh_w - 1)
        chosen_shapes.append((shape, hole_r0, hole_c0))
        for dr, dc in shape:
            grid[top_r0 + hole_r0 + dr, c0 + hole_c0 + dc] = BG

    for source_idx in range(n):
        dest_slot = perm[source_idx]
        slot_c0 = left_margin + dest_slot * block_w
        shape = shapes[source_idx]
        sh_h, sh_w = _shape_bbox(shape)
        key_r0 = random.randint(0, bottom_h - sh_h)
        key_c0 = random.randint(slot_c0, slot_c0 + block_w - sh_w)
        for dr, dc in shape:
            grid[bottom_r0 + key_r0 + dr, key_c0 + dc] = key_colors[source_idx]

    if _get_bg(grid) != BG:
        return None

    try:
        info = _analyze_input(grid)
    except AssertionError:
        return None

    if any(info["source_to_dest"][i] == i for i in range(n)) and all(info["source_to_dest"][i] == i for i in range(n)):
        return None

    true_out = _render_output(grid, info)
    if np.array_equal(true_out, _same_slot_shortcut(grid, info)):
        return None

    areas = [b["hole_area"] for b in info["blocks"]]
    bboxes = [b["hole_bbox"] for b in info["blocks"]]
    if len(set(areas)) == len(areas):
        return None
    if len(set(bboxes)) == len(bboxes):
        return None

    moved = sum(info["source_to_dest"][i] != i for i in range(n))
    if moved == 0:
        return None

    for src_idx, dst_slot in info["source_to_dest"].items():
        key = next(k for k in info["keys"] if k["source_idx"] == src_idx)
        if key["dest_slot"] != dst_slot:
            return None

    if not np.array_equal(main(grid), true_out):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid key-placement input")
