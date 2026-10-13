from __future__ import annotations
import numpy as np
from arc_utils import get_bg, label_mask

def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate a repeating striped pattern from a yellow border segment toward the opposite edge, recoloring foreground objects on certain stripes.

    Input structure:
    - bg = most common color. Foreground blobs of another color scattered in the interior.
    - A short yellow (4) segment sits on one border edge.

    Rule:
    - From the yellow segment, march toward the opposite edge in steps of 2 cells, cycling colors Y, Y, G (4, 4, 3). The 1-cell gap between stripes is forced to bg.
    - Yellow stripes paint only within the segment's span.
    - Green stripes additionally recolor all non-bg pixels across the entire row/column to green.

    Key insight: Green stripes act as a "flood" that recolors existing objects on the full row/column, while yellow stripes are confined to the segment span.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    out = g.copy()

    Y = 4
    G = 3

    border = np.zeros((H, W), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    smask = (g == int(Y)) & border
    if not bool(smask.any()):
        return out

    lab, n = label_mask(smask, connectivity=4)
    if int(n) <= 0:
        return out

    best_id = max(range(1, int(n) + 1), key=lambda cid: int((lab == int(cid)).sum()))
    pts = np.argwhere(lab == int(best_id))

    rs = pts[:, 0]
    cs = pts[:, 1]
    r0, r1 = int(rs.min()), int(rs.max())
    c0, c1 = int(cs.min()), int(cs.max())

    if r0 == r1:
        orient = "h"
        start_pos = int(r0)
        cols = sorted(int(c) for c in cs)
        span = list(range(int(cols[0]), int(cols[-1]) + 1))
        dir1 = 1 if start_pos == 0 else -1
        limit = H
    elif c0 == c1:
        orient = "v"
        start_pos = int(c0)
        rows = sorted(int(r) for r in rs)
        span = list(range(int(rows[0]), int(rows[-1]) + 1))
        dir1 = 1 if start_pos == 0 else -1
        limit = W
    else:
        # Unexpected: not a straight edge segment.
        return out

    cycle = [int(Y), int(Y), int(G)]

    step = 0
    while True:
        pos = int(start_pos) + int(dir1) * (2 * int(step))
        if not (0 <= pos < int(limit)):
            break

        col = int(cycle[int(step) % 3])

        if orient == "h":
            if col == int(G):
                out[pos, :] = np.where(out[pos, :] != int(bg), int(G), out[pos, :])
            for c in span:
                out[pos, int(c)] = col
            # Force the 1-cell gap to background within the segment span.
            gap = int(start_pos) + int(dir1) * (2 * int(step) + 1)
            if 0 <= gap < H:
                for c in span:
                    out[int(gap), int(c)] = int(bg)
        else:
            if col == int(G):
                out[:, pos] = np.where(out[:, pos] != int(bg), int(G), out[:, pos])
            for r in span:
                out[int(r), pos] = col
            gap = int(start_pos) + int(dir1) * (2 * int(step) + 1)
            if 0 <= gap < W:
                for r in span:
                    out[int(r), int(gap)] = int(bg)

        step += 1

    return out


# --------------- generator ---------------
import random
from arc_utils import random_sprite, random_free_location


def _gen_attempt():
    rng = random.Random()

    H = rng.randint(25, 30)
    W = rng.randint(22, 30)

    # Pick bg and fg colors (not yellow=4 or green=3)
    available = [c for c in range(10) if c not in (3, 4)]
    bg, fg = rng.sample(available, 2)

    # Pick edge and yellow segment
    edge = rng.choice(['top', 'bottom', 'left', 'right'])
    seg_len = rng.randint(3, 8)

    if edge in ('top', 'bottom'):
        # horizontal segment: columns span, must not touch corners (col 0 or W-1)
        max_start = W - 1 - seg_len  # last col of span = start + seg_len - 1 <= W-2
        if max_start < 1:
            return None
        seg_start = rng.randint(1, max_start)
        span = list(range(seg_start, seg_start + seg_len))
        orient = 'h'
        edge_row = 0 if edge == 'top' else H - 1
    else:
        # vertical segment: rows span, must not touch corners (row 0 or H-1)
        max_start = H - 1 - seg_len
        if max_start < 1:
            return None
        seg_start = rng.randint(1, max_start)
        span = list(range(seg_start, seg_start + seg_len))
        orient = 'v'
        edge_col = 0 if edge == 'left' else W - 1

    # Start with bg grid
    grid = np.full((H, W), bg, dtype=int)

    # Place yellow segment on border
    if orient == 'h':
        for c in span:
            grid[edge_row, c] = 4
    else:
        for r in span:
            grid[r, edge_col] = 4

    # Determine which rows/cols are on the trajectory (both stripe and gap positions)
    # to avoid placing blobs on the edge row/col itself
    if orient == 'h':
        forbidden_rows = {edge_row}
    else:
        forbidden_cols = {edge_col}

    # Scatter 5-9 amorphous fg blobs in the interior
    n_blobs = rng.randint(5, 9)
    canvas = grid.copy()

    for _ in range(n_blobs):
        # Random blob size: aim for 5-67 pixels, bbox dims 2-15
        bh = rng.randint(2, 10)
        bw = rng.randint(2, 10)
        density = rng.uniform(0.4, 0.85)

        try:
            sprite = random_sprite(bh, bw, density=density, symmetry="not_symmetric",
                                   color_palette=[fg], connectivity=4, background=bg)
        except (AssertionError, RecursionError):
            continue

        # Trim bg border rows/cols from sprite
        mask = (sprite != bg)
        rows_any = np.any(mask, axis=1)
        cols_any = np.any(mask, axis=0)
        if not rows_any.any():
            continue
        r0, r1 = np.where(rows_any)[0][[0, -1]]
        c0, c1 = np.where(cols_any)[0][[0, -1]]
        sprite = sprite[r0:r1+1, c0:c1+1]

        sh, sw = sprite.shape
        pix = int(np.sum(sprite != bg))
        if pix < 5 or pix > 67:
            continue

        # Find placement: border=1 to keep away from grid edges, padding=1 for separation
        try:
            pr, pc = random_free_location(canvas, sprite, transparent=bg, border=1, padding=1)
        except ValueError:
            continue

        # Check forbidden row/col constraints
        sprite_mask = (sprite != bg)
        blob_rows = set(int(pr + dr) for dr in range(sh) if sprite_mask[dr].any())
        blob_cols = set(int(pc + dc) for dc in range(sw) if sprite_mask[:, dc].any())

        if orient == 'h' and blob_rows & forbidden_rows:
            continue
        if orient == 'v' and blob_cols & forbidden_cols:
            continue

        # Paste blob
        for dr in range(sh):
            for dc in range(sw):
                if sprite[dr, dc] != bg:
                    canvas[pr + dr, pc + dc] = sprite[dr, dc]

    # Count blobs placed
    fg_mask = (canvas != bg) & (canvas != 4)
    lab, n_placed = label_mask(fg_mask, connectivity=4)
    if n_placed < 5:
        return None

    # Now canvas is our INPUT grid. Build the expected output via solver.
    inp = canvas.copy()
    out = main(inp)

    # Verify bg is most common in input
    if get_bg(inp) != bg:
        return None

    # Round-trip check
    if not np.array_equal(main(inp), out):
        return None

    return inp#.tolist()


def generate_input():
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 300 attempts")
