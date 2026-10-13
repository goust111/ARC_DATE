from __future__ import annotations

import random
from collections import Counter
from math import gcd

import numpy as np


BG = 0
GREEN = 3
YELLOW = 4
GEN_PALETTE = (1, 2, 5, 6, 7, 8, 9)
EDGE_OPTIONS = ("top", "bottom", "left", "right")


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _find_square_blocks(grid: np.ndarray, bg: int) -> list[dict[str, int]]:
    blocks: list[dict[str, int]] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        for pts in _components_of_color(grid, color):
            if len(pts) < 4:
                continue
            r0, c0, r1, c1 = _bbox(pts)
            height = r1 - r0 + 1
            width = c1 - c0 + 1
            if height != width or height < 2:
                continue
            if len(pts) != height * width:
                continue
            blocks.append({"r0": r0, "c0": c0, "size": height, "color": color})
    return sorted(blocks, key=lambda block: (block["r0"], block["c0"], block["color"]))


def _find_yellow_segment(grid: np.ndarray) -> dict[str, object]:
    h, w = grid.shape
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    pts = np.argwhere((grid == YELLOW) & border)
    assert len(pts) >= 2, "expected one yellow border segment"
    r0, c0, r1, c1 = _bbox(pts)
    if r0 == r1:
        cols = sorted(int(c) for c in pts[:, 1])
        assert cols == list(range(cols[0], cols[-1] + 1)), "yellow segment must be contiguous"
        edge = "top" if r0 == 0 else "bottom"
        return {
            "edge": edge,
            "axis": "h",
            "start": int(r0),
            "direction": 1 if edge == "top" else -1,
            "span0": int(c0),
            "span1": int(c1),
            "cells": {(int(r), int(c)) for r, c in pts},
        }
    assert c0 == c1, "yellow segment must be straight"
    rows = sorted(int(r) for r in pts[:, 0])
    assert rows == list(range(rows[0], rows[-1] + 1)), "yellow segment must be contiguous"
    edge = "left" if c0 == 0 else "right"
    return {
        "edge": edge,
        "axis": "v",
        "start": int(c0),
        "direction": 1 if edge == "left" else -1,
        "span0": int(r0),
        "span1": int(r1),
        "cells": {(int(r), int(c)) for r, c in pts},
    }


def _extract_legend(
    grid: np.ndarray,
    *,
    bg: int,
    blocks: list[dict[str, int]],
    segment_cells: set[tuple[int, int]],
) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    used = np.zeros_like(grid, dtype=bool)
    for block in blocks:
        r0 = block["r0"]
        c0 = block["c0"]
        size = block["size"]
        used[r0 : r0 + size, c0 : c0 + size] = True
    for r, c in segment_cells:
        used[r, c] = True

    legend_pts = np.argwhere((grid != bg) & (~used))
    assert len(legend_pts) >= 4, "legend must contain several colored pixels"
    r0, c0, r1, c1 = _bbox(legend_pts)
    legend = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
    local = [(int(r - r0), int(c - c0), int(grid[r, c])) for r, c in legend_pts]
    return legend, sorted(local)


def _dominant_block_size(blocks: list[dict[str, int]]) -> int:
    counts = Counter(block["size"] for block in blocks)
    return int(counts.most_common(1)[0][0])


def _infer_step(blocks: list[dict[str, int]]) -> int:
    diffs: list[int] = []
    rows = sorted({block["r0"] for block in blocks})
    cols = sorted({block["c0"] for block in blocks})
    for values in (rows, cols):
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                diff = int(values[j] - values[i])
                if diff > 0:
                    diffs.append(diff)
    assert diffs, "need at least two aligned anchors to infer block step"
    step = diffs[0]
    for diff in diffs[1:]:
        step = gcd(step, diff)
    assert step >= 2, "invalid block lattice step"
    return int(step)


def _candidate_bases(
    legend_cells: list[tuple[int, int, int]],
    blocks: list[dict[str, int]],
    step: int,
) -> list[tuple[int, int]]:
    candidates = set()
    for lr, lc, color in legend_cells:
        for block in blocks:
            if block["color"] == color:
                candidates.add((block["r0"] - lr * step, block["c0"] - lc * step))
    return sorted(candidates)


def _valid_base(
    *,
    base: tuple[int, int],
    legend: np.ndarray,
    legend_cells: list[tuple[int, int, int]],
    blocks: list[dict[str, int]],
    grid_shape: tuple[int, int],
    block_size: int,
    step: int,
) -> bool:
    h, w = grid_shape
    base_r, base_c = base
    anchors = {(block["r0"], block["c0"]): block["color"] for block in blocks}

    for block in blocks:
        dr = block["r0"] - base_r
        dc = block["c0"] - base_c
        if dr % step != 0 or dc % step != 0:
            return False
        lr = dr // step
        lc = dc // step
        if not (0 <= lr < legend.shape[0] and 0 <= lc < legend.shape[1]):
            return False
        if int(legend[lr, lc]) != block["color"]:
            return False

    for lr, lc, color in legend_cells:
        r0 = base_r + lr * step
        c0 = base_c + lc * step
        if not (0 <= r0 <= h - block_size and 0 <= c0 <= w - block_size):
            return False
        if (r0, c0) in anchors and anchors[(r0, c0)] != color:
            return False
    return True


def _infer_mapping(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    segment = _find_yellow_segment(g)
    raw_blocks = _find_square_blocks(g, bg)
    assert raw_blocks, "no square anchors found"
    block_size = _dominant_block_size(raw_blocks)
    blocks = [block for block in raw_blocks if block["size"] == block_size]
    legend, legend_cells = _extract_legend(g, bg=bg, blocks=blocks, segment_cells=segment["cells"])
    step = _infer_step(blocks)
    candidates = [
        base
        for base in _candidate_bases(legend_cells, blocks, step)
        if _valid_base(
            base=base,
            legend=legend,
            legend_cells=legend_cells,
            blocks=blocks,
            grid_shape=g.shape,
            block_size=block_size,
            step=step,
        )
    ]
    assert len(candidates) == 1, "legend-to-block mapping must be unique"
    return {
        "bg": bg,
        "segment": segment,
        "legend": legend,
        "legend_cells": legend_cells,
        "blocks": blocks,
        "block_size": block_size,
        "step": step,
        "base": candidates[0],
    }


def _complete_blocks(grid: np.ndarray, mapping: dict[str, object]) -> tuple[np.ndarray, list[dict[str, int]]]:
    out = np.array(grid, dtype=int, copy=True)
    base_r, base_c = mapping["base"]
    step = int(mapping["step"])
    block_size = int(mapping["block_size"])
    existing = {(block["r0"], block["c0"]) for block in mapping["blocks"]}
    new_blocks: list[dict[str, int]] = []
    for lr, lc, color in mapping["legend_cells"]:
        r0 = base_r + lr * step
        c0 = base_c + lc * step
        if (r0, c0) in existing:
            continue
        out[r0 : r0 + block_size, c0 : c0 + block_size] = color
        new_blocks.append({"r0": r0, "c0": c0, "size": block_size, "color": color})
    return out, new_blocks


def _stripe_positions(segment: dict[str, object], extent: int) -> list[tuple[int, int]]:
    colors = (YELLOW, YELLOW, GREEN)
    start = int(segment["start"])
    direction = int(segment["direction"])
    out: list[tuple[int, int]] = []
    idx = 0
    while True:
        pos = start + direction * (2 * idx)
        if not (0 <= pos < extent):
            break
        out.append((int(pos), int(colors[idx % 3])))
        idx += 1
    return out


def _apply_stripes(
    grid: np.ndarray,
    *,
    bg: int,
    segment: dict[str, object],
    new_blocks: list[dict[str, int]],
) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    h, w = out.shape
    span0 = int(segment["span0"])
    span1 = int(segment["span1"])

    if segment["axis"] == "h":
        for pos, color in _stripe_positions(segment, h):
            if color == GREEN:
                for block in new_blocks:
                    if block["r0"] == pos:
                        r0 = block["r0"]
                        c0 = block["c0"]
                        size = block["size"]
                        out[r0 : r0 + size, c0 : c0 + size] = GREEN
                row_mask = out[pos, :] != bg
                out[pos, row_mask] = GREEN
            for c in range(span0, span1 + 1):
                if out[pos, c] == bg:
                    out[pos, c] = color
    else:
        for pos, color in _stripe_positions(segment, w):
            if color == GREEN:
                for block in new_blocks:
                    if block["c0"] == pos:
                        r0 = block["r0"]
                        c0 = block["c0"]
                        size = block["size"]
                        out[r0 : r0 + size, c0 : c0 + size] = GREEN
                col_mask = out[:, pos] != bg
                out[col_mask, pos] = GREEN
            for r in range(span0, span1 + 1):
                if out[r, pos] == bg:
                    out[r, pos] = color
    return out


def _stripe_color_at(edge: str, coord: int, size: int) -> int | None:
    dist = coord if edge in ("top", "left") else size - 1 - coord
    if dist < 0 or dist % 2 != 0:
        return None
    return (YELLOW, YELLOW, GREEN)[(dist // 2) % 3]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a block lattice from a top-left thumbnail, then let propagated stripes modulate only the newly added blocks.

    Input structure:
    - The background is black.
    - A small thumbnail near the top-left encodes colored cells and black empties.
    - Elsewhere, some uniformly colored square blocks already occupy a regular lattice, and a short yellow segment sits on one border.

    Rule:
    - Infer the block size, lattice step, and lattice offset by matching the existing blocks to the thumbnail cells.
    - For every non-black thumbnail cell whose mapped lattice position is empty, add a square block of that thumbnail color.
    - Propagate one-cell stripes from the yellow border segment toward the opposite side with stride two and cycle yellow, yellow, green.
    - Yellow stripes draw only on background cells inside the segment span. Green stripes recolor every non-background pixel on their full row or column to green, and any newly added block whose lattice row or column lies on that green stripe becomes entirely green.

    Key insight:
    - The thumbnail decides where missing blocks belong, but the stripe system selectively overrides some of those new blocks, so the output comes from combining coordinate completion with stripe-driven recoloring.
    """
    mapping = _infer_mapping(np.asarray(grid, dtype=int))
    completed, new_blocks = _complete_blocks(grid, mapping)
    return _apply_stripes(
        completed,
        bg=int(mapping["bg"]),
        segment=mapping["segment"],
        new_blocks=new_blocks,
    )


def _make_mask(k: int, axis: str, axis_colors: list[int]) -> np.ndarray | None:
    mask = np.zeros((k, k), dtype=bool)
    if axis == "h":
        green_rows = [i for i, color in enumerate(axis_colors) if color == GREEN]
        yellow_rows = [i for i, color in enumerate(axis_colors) if color == YELLOW]
        if not green_rows or not yellow_rows:
            return None
        g_row = random.choice(green_rows)
        y_row = random.choice(yellow_rows)
        for c in random.sample(range(k), min(k, 3)):
            mask[g_row, c] = True
        for c in random.sample(range(k), min(k, 2)):
            mask[y_row, c] = True
        for r in range(k):
            if not mask[r].any():
                for c in random.sample(range(k), random.randint(1, min(3, k))):
                    mask[r, c] = True
    else:
        green_cols = [i for i, color in enumerate(axis_colors) if color == GREEN]
        yellow_cols = [i for i, color in enumerate(axis_colors) if color == YELLOW]
        if not green_cols or not yellow_cols:
            return None
        g_col = random.choice(green_cols)
        y_col = random.choice(yellow_cols)
        for r in random.sample(range(k), min(k, 3)):
            mask[r, g_col] = True
        for r in random.sample(range(k), min(k, 2)):
            mask[r, y_col] = True
        for c in range(k):
            if not mask[:, c].any():
                for r in random.sample(range(k), random.randint(1, min(3, k))):
                    mask[r, c] = True

    for r in range(k):
        for c in range(k):
            if random.random() < 0.35:
                mask[r, c] = True

    if mask.sum() < 6:
        return None
    rows = {r for r in range(k) if mask[r].any()}
    cols = {c for c in range(k) if mask[:, c].any()}
    if len(rows) < 2 or len(cols) < 2:
        return None
    return mask


def _assign_colors(mask: np.ndarray) -> np.ndarray:
    k = mask.shape[0]
    legend = np.zeros((k, k), dtype=int)
    cells = [(r, c) for r in range(k) for c in range(k) if mask[r, c]]
    palette = random.sample(GEN_PALETTE, random.choice((3, 4)))
    random.shuffle(cells)
    for color, (r, c) in zip(palette, cells):
        legend[r, c] = color
    for r, c in cells[len(palette) :]:
        legend[r, c] = random.choice(palette)
    return legend


def _choose_existing(mask: np.ndarray, axis: str, axis_colors: list[int]) -> set[tuple[int, int]] | None:
    filled = {(r, c) for r in range(mask.shape[0]) for c in range(mask.shape[1]) if mask[r, c]}
    if axis == "h":
        green_cells = [(r, c) for (r, c) in filled if axis_colors[r] == GREEN]
        yellow_cells = [(r, c) for (r, c) in filled if axis_colors[r] == YELLOW]
    else:
        green_cells = [(r, c) for (r, c) in filled if axis_colors[c] == GREEN]
        yellow_cells = [(r, c) for (r, c) in filled if axis_colors[c] == YELLOW]

    if len(green_cells) < 3 or len(yellow_cells) < 2 or len(filled) < 7:
        return None

    keep_green = random.choice(green_cells)
    keep_yellow = random.choice(yellow_cells)
    removable_green = [cell for cell in green_cells if cell != keep_green]
    removable_yellow = [cell for cell in yellow_cells if cell != keep_yellow]
    if len(removable_green) < 2 or len(removable_yellow) < 1:
        return None

    missing = set(random.sample(removable_green, 2))
    missing.add(random.choice(removable_yellow))
    extras = [cell for cell in filled if cell not in missing and cell not in {keep_green, keep_yellow}]
    random.shuffle(extras)
    target_missing = random.randint(3, min(len(filled) - 4, 5))
    for cell in extras:
        if len(missing) >= target_missing:
            break
        tentative = missing | {cell}
        existing = filled - tentative
        if len(existing) < 4:
            continue
        missing = tentative

    existing = filled - missing
    if len(existing) < 4 or len(missing) < 3:
        return None
    if len({r for r, _ in existing}) < 2 or len({c for _, c in existing}) < 2:
        return None
    return existing


def _axis_base_candidates(size: int, k: int, step: int, block_size: int, edge: str) -> list[tuple[int, list[int]]]:
    extent = (k - 1) * step + block_size
    candidates: list[tuple[int, list[int]]] = []
    for base in range(k + 3, size - extent):
        colors = [_stripe_color_at(edge, base + i * step, size) for i in range(k)]
        if GREEN in colors and YELLOW in colors:
            candidates.append((base, [int(color) if color is not None else -1 for color in colors]))
    return candidates


def _other_base_candidates(size: int, k: int, step: int, block_size: int) -> list[int]:
    extent = (k - 1) * step + block_size
    return list(range(k + 4, size - extent))


def _place_segment(grid: np.ndarray, edge: str, length: int, legend_size: int) -> None:
    h, w = grid.shape
    if edge in ("top", "bottom"):
        starts = [c for c in range(max(legend_size + 2, 1), w - length) if c >= 1]
        if not starts:
            starts = list(range(1, w - length))
        start = random.choice(starts)
        row = 0 if edge == "top" else h - 1
        grid[row, start : start + length] = YELLOW
    else:
        starts = [r for r in range(max(legend_size + 2, 1), h - length) if r >= 1]
        if not starts:
            starts = list(range(1, h - length))
        start = random.choice(starts)
        col = 0 if edge == "left" else w - 1
        grid[start : start + length, col] = YELLOW


def _render_input(
    *,
    h: int,
    w: int,
    legend: np.ndarray,
    edge: str,
    step: int,
    block_size: int,
    base_r: int,
    base_c: int,
    existing: set[tuple[int, int]],
) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    k = legend.shape[0]
    grid[1 : 1 + k, 1 : 1 + k] = legend
    for lr in range(k):
        for lc in range(k):
            color = int(legend[lr, lc])
            if color == BG or (lr, lc) not in existing:
                continue
            r0 = base_r + lr * step
            c0 = base_c + lc * step
            grid[r0 : r0 + block_size, c0 : c0 + block_size] = color
    _place_segment(grid, edge, random.randint(2, 3), k)
    return grid


def _semantic_check(inp: np.ndarray) -> bool:
    mapping = _infer_mapping(inp)
    completed, new_blocks = _complete_blocks(inp, mapping)
    out = _apply_stripes(completed, bg=int(mapping["bg"]), segment=mapping["segment"], new_blocks=new_blocks)
    if len(new_blocks) < 3 or np.array_equal(out, completed):
        return False

    segment = mapping["segment"]
    if segment["axis"] == "h":
        axis_len = inp.shape[0]
        stripe_lookup = {block["r0"]: _stripe_color_at(segment["edge"], block["r0"], axis_len) for block in new_blocks}
        existing_lookup = {_stripe_color_at(segment["edge"], block["r0"], axis_len) for block in mapping["blocks"]}
    else:
        axis_len = inp.shape[1]
        stripe_lookup = {block["c0"]: _stripe_color_at(segment["edge"], block["c0"], axis_len) for block in new_blocks}
        existing_lookup = {_stripe_color_at(segment["edge"], block["c0"], axis_len) for block in mapping["blocks"]}

    green_missing = sum(1 for color in stripe_lookup.values() if color == GREEN)
    yellow_missing = sum(1 for color in stripe_lookup.values() if color == YELLOW)
    if green_missing < 2 or yellow_missing < 1 or GREEN not in existing_lookup:
        return False

    yellow_bg_pixels = int(np.count_nonzero((out == YELLOW) & (completed == BG)))
    if yellow_bg_pixels < 2:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    k = random.randint(3, 5)
    block_size = random.randint(2, 4)
    gap = random.randint(1, 3)
    step = block_size + gap
    edge = random.choice(EDGE_OPTIONS)

    extent = (k - 1) * step + block_size
    min_size = max(18, extent + k + 6)
    if min_size > 30:
        return None
    h = random.randint(min_size, 30)
    w = random.randint(min_size, 30)

    if edge in ("top", "bottom"):
        axis_candidates = _axis_base_candidates(h, k, step, block_size, edge)
        other_candidates = _other_base_candidates(w, k, step, block_size)
        if not axis_candidates or not other_candidates:
            return None
        base_r, axis_colors = random.choice(axis_candidates)
        base_c = random.choice(other_candidates)
        axis = "h"
    else:
        axis_candidates = _axis_base_candidates(w, k, step, block_size, edge)
        other_candidates = _other_base_candidates(h, k, step, block_size)
        if not axis_candidates or not other_candidates:
            return None
        base_c, axis_colors = random.choice(axis_candidates)
        base_r = random.choice(other_candidates)
        axis = "v"

    mask = _make_mask(k, axis, axis_colors)
    if mask is None:
        return None
    existing = _choose_existing(mask, axis, axis_colors)
    if existing is None:
        return None
    legend = _assign_colors(mask)
    inp = _render_input(
        h=h,
        w=w,
        legend=legend,
        edge=edge,
        step=step,
        block_size=block_size,
        base_r=base_r,
        base_c=base_c,
        existing=existing,
    )

    try:
        if max(inp.shape) > 30 or max(main(inp).shape) > 30:
            return None
        if _get_bg(inp) != BG or not _semantic_check(inp):
            return None
    except AssertionError:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(800):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed after 800 attempts")
