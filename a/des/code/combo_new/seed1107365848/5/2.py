import math
import random
from collections import Counter

import numpy as np


BG = 0
GREY = 5
INK = 2
BORDER_POOL = (1, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 500


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
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
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _find_strip(grid: np.ndarray, bg: int) -> list[int]:
    h, w = grid.shape
    best: tuple[int, int, int] | None = None
    for r in range(1, h - 1):
        c = 0
        while c < w:
            top = int(grid[r - 1, c])
            bot = int(grid[r + 1, c])
            mid = int(grid[r, c])
            if top == bot != bg and mid != bg:
                c0 = c
                while c < w and int(grid[r - 1, c]) == top and int(grid[r + 1, c]) == top and int(grid[r, c]) != bg:
                    c += 1
                c1 = c - 1
                colors = [int(v) for v in grid[r, c0 : c1 + 1]]
                if len(colors) >= 2 and len(set(colors)) == len(colors) and top not in colors:
                    cand = (len(colors), r, c0)
                    if best is None or cand > best:
                        best = cand
            else:
                c += 1
    assert best is not None, "expected a 3-row color strip"
    _, r, c0 = best
    width = best[0]
    return [int(v) for v in grid[r, c0 : c0 + width]]


def _find_panel(grid: np.ndarray, color: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    comps = _components(grid == color)
    candidates: list[tuple[int, tuple[int, int, int, int]]] = []
    for comp in comps:
        if len(comp) <= 1:
            continue
        r0, c0, r1, c1 = _bbox(comp)
        side_r = r1 - r0 + 1
        side_c = c1 - c0 + 1
        if side_r != side_c or side_r < 3:
            continue
        expected = 4 * side_r - 4
        if len(comp) != expected:
            continue
        candidates.append((side_r, (r0, c0, r1, c1)))
    assert len(candidates) == 1, f"expected one panel for color {color}"
    _, (r0, c0, r1, c1) = candidates[0]
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    return patch, (r0, c0, r1, c1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Composite framed panel patterns onto one canvas using the color-strip stacking order.

    Input structure:
    - Background is black.
    - Near the top sits a 3-row color strip: one row of distinct colors, with matching grey boundary rows directly above and below it.
    - Below the strip are several equal-sized square panels separated by black cells; each panel has a 1-pixel border whose color appears exactly once in the strip.
    - Each panel interior contains a pattern drawn with a single non-background ink color on black.

    Rule:
    - Read the strip colors from left to right to get the bottom-to-top stacking order.
    - For each strip color, find the panel with the matching border color and take only its interior.
    - Start a black output grid the size of one panel interior, then process panels in strip order: every non-black interior cell paints the output with that panel's border color, overwriting earlier colors.

    Key insight:
    - The strip is not decoration; it is the ordering key that links each border color to a layer, and each panel's interior acts as a mask that recolors to the border color during compositing.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    order = _find_strip(g, bg)

    panel_by_color: dict[int, np.ndarray] = {}
    inner_shape: tuple[int, int] | None = None
    for color in order:
        patch, _ = _find_panel(g, color)
        inner = patch[1:-1, 1:-1]
        if inner_shape is None:
            inner_shape = inner.shape
        assert inner_shape == inner.shape, "all panels must share the same inner size"
        panel_by_color[color] = inner

    assert inner_shape is not None
    out = np.full(inner_shape, bg, dtype=int)
    for color in order:
        mask = panel_by_color[color] != bg
        out[mask] = int(color)
    return out


def _random_blob(inner: int, rng: random.Random) -> np.ndarray:
    mask = np.zeros((inner, inner), dtype=bool)
    target = rng.randint(1, max(1, (inner * inner * 2) // 3))
    start = (rng.randrange(inner), rng.randrange(inner))
    frontier = [start]
    mask[start] = True
    while frontier and int(mask.sum()) < target:
        r, c = frontier[rng.randrange(len(frontier))]
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        rng.shuffle(dirs)
        grew = False
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < inner and 0 <= nc < inner and not mask[nr, nc]:
                mask[nr, nc] = True
                frontier.append((nr, nc))
                grew = True
                break
        if not grew:
            frontier.remove((r, c))
            if not frontier and int(mask.sum()) < target:
                empties = [(rr, cc) for rr in range(inner) for cc in range(inner) if not mask[rr, cc]]
                if not empties:
                    break
                seed = rng.choice(empties)
                mask[seed] = True
                frontier.append(seed)
    return mask


def _random_line_mask(inner: int, rng: random.Random) -> np.ndarray:
    mask = np.zeros((inner, inner), dtype=bool)
    for _ in range(rng.randint(1, max(1, inner - 1))):
        if rng.random() < 0.5:
            mask[rng.randrange(inner), :] = True
        else:
            mask[:, rng.randrange(inner)] = True
    if rng.random() < 0.5:
        for d in range(inner):
            if rng.random() < 0.7:
                mask[d, d] = True
    else:
        for d in range(inner):
            if rng.random() < 0.7:
                mask[d, inner - 1 - d] = True
    return mask


def _random_sparse_mask(inner: int, rng: random.Random) -> np.ndarray:
    probs = [0.2, 0.3, 0.4, 0.5, 0.6]
    p = rng.choice(probs)
    mask = np.fromiter(
        (rng.random() < p for _ in range(inner * inner)),
        dtype=bool,
        count=inner * inner,
    ).reshape(inner, inner)
    return mask


def _random_mask(inner: int, rng: random.Random) -> np.ndarray:
    mode = rng.randrange(3)
    if mode == 0:
        mask = _random_blob(inner, rng)
    elif mode == 1:
        mask = _random_line_mask(inner, rng)
    else:
        mask = _random_sparse_mask(inner, rng)
    if not mask.any():
        mask[rng.randrange(inner), rng.randrange(inner)] = True
    if mask.all():
        mask[rng.randrange(inner), rng.randrange(inner)] = False
    return mask


def _compose_masks(order: list[int], masks: dict[int, np.ndarray], bg: int) -> tuple[np.ndarray, bool, Counter]:
    inner_shape = next(iter(masks.values())).shape
    out = np.full(inner_shape, bg, dtype=int)
    occupied = np.zeros(inner_shape, dtype=bool)
    overwrite = False
    visible = Counter()
    for color in order:
        mask = masks[color]
        overwrite |= bool((occupied & mask).any())
        out[mask] = int(color)
        occupied |= mask
    for color in order:
        visible[color] = int(((out == color) & masks[color]).sum())
    return out, overwrite, visible


def _layout_options(panel_count: int) -> list[tuple[int, int, int, int, int]]:
    options: list[tuple[int, int, int, int, int]] = []
    for side in range(4, 8):
        for rows in range(1, min(3, panel_count) + 1):
            cols = math.ceil(panel_count / rows)
            for hgap in range(1, 4):
                for vgap in range(1, 4):
                    panel_w = cols * side + (cols - 1) * hgap
                    panel_h = rows * side + (rows - 1) * vgap
                    if max(panel_w, panel_count) <= 30 and 4 + panel_h <= 30:
                        options.append((side, rows, cols, hgap, vgap))
    return options


def _build_input() -> np.ndarray:
    rng = random.Random()
    panel_count = rng.randint(2, 6)
    colors = rng.sample(BORDER_POOL, panel_count)
    order = colors[:]

    layout = rng.choice(_layout_options(panel_count))
    side, rows, cols, hgap, vgap = layout
    inner = side - 2
    panel_w = cols * side + (cols - 1) * hgap
    panel_h = rows * side + (rows - 1) * vgap

    min_h = 4 + panel_h
    min_w = max(panel_w, panel_count)
    height = rng.randint(min_h, 30)
    width = rng.randint(min_w, 30)

    strip_top = rng.randint(0, height - 4 - panel_h)
    panel_top = rng.randint(strip_top + 4, height - panel_h)
    panel_left = rng.randint(0, width - panel_w)
    strip_left = rng.randint(0, width - panel_count)

    grid = np.full((height, width), BG, dtype=int)
    grid[strip_top, strip_left : strip_left + panel_count] = GREY
    grid[strip_top + 1, strip_left : strip_left + panel_count] = np.array(order, dtype=int)
    grid[strip_top + 2, strip_left : strip_left + panel_count] = GREY

    placement_colors = order[:]
    rng.shuffle(placement_colors)
    if placement_colors == order:
        placement_colors = order[1:] + order[:1]

    masks: dict[int, np.ndarray] = {}
    for color in order:
        masks[color] = _random_mask(inner, rng)

    if rng.random() < 0.5:
        shared_cells = rng.sample(
            [(r, c) for r in range(inner) for c in range(inner)],
            k=rng.randint(1, max(1, inner)),
        )
        participants = rng.sample(order, k=rng.randint(2, panel_count))
        for color in participants:
            for r, c in shared_cells:
                masks[color][r, c] = True

    out, overwrite, visible = _compose_masks(order, masks, BG)
    if len(set(int(v) for v in out.ravel() if int(v) != BG)) < 2:
        raise ValueError("output needs at least two visible colors")
    if sum(1 for color in order if visible[color] > 0) < 2:
        raise ValueError("at least two panels must remain visible")
    if rng.random() < 0.5 and not overwrite:
        raise ValueError("expected an overlap-heavy sample")

    slots = []
    for idx in range(panel_count):
        rr = idx // cols
        cc = idx % cols
        r0 = panel_top + rr * (side + vgap)
        c0 = panel_left + cc * (side + hgap)
        slots.append((r0, c0))

    for color, (r0, c0) in zip(placement_colors, slots):
        patch = np.full((side, side), BG, dtype=int)
        patch[0, :] = color
        patch[-1, :] = color
        patch[:, 0] = color
        patch[:, -1] = color
        patch[1:-1, 1:-1][masks[color]] = INK
        grid[r0 : r0 + side, c0 : c0 + side] = patch

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            return _build_input()
        except ValueError:
            continue
    raise RuntimeError("failed to generate a valid panel-compositing input")
