import random

import numpy as np


GREY = 5
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _is_single_component(mask: np.ndarray) -> bool:
    if not np.any(mask):
        return False
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    start = tuple(np.argwhere(mask)[0])
    stack = [start]
    seen[start] = True
    size = 0
    while stack:
        r, c = stack.pop()
        size += 1
        for nr, nc in _neighbors4(r, c, h, w):
            if mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                stack.append((nr, nc))
    return size == int(mask.sum())


def _touches_border(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())


def _extract_line_colors(grid: np.ndarray, occluder: int, axis: int):
    n = grid.shape[axis]
    line_colors = []
    for idx in range(n):
        line = grid[idx, :] if axis == 0 else grid[:, idx]
        visible = line[line != occluder]
        if visible.size == 0:
            line_colors.append(None)
            continue
        colors = np.unique(visible)
        if colors.size != 1:
            return None
        line_colors.append(int(colors[0]))
    return line_colors


def _fit_cycle(line_colors: list[int | None]):
    n = len(line_colors)
    for period in range(2, min(4, n) + 1):
        seq = [None] * period
        ok = True
        for idx, color in enumerate(line_colors):
            if color is None:
                continue
            slot = idx % period
            if seq[slot] is None:
                seq[slot] = color
            elif seq[slot] != color:
                ok = False
                break
        if not ok or any(color is None for color in seq):
            continue
        if len(set(seq)) != period:
            continue
        return period, [int(color) for color in seq]
    return None


def _candidate_colors(grid: np.ndarray) -> list[int]:
    seen = []
    for value in grid.ravel():
        color = int(value)
        if color not in seen:
            seen.append(color)
    if GREY in seen:
        return [GREY] + [color for color in seen if color != GREY]
    return seen


def _infer_model(grid: np.ndarray):
    best = None
    for occluder in _candidate_colors(grid):
        mask = grid == occluder
        if not _is_single_component(mask):
            continue
        if _touches_border(mask):
            continue
        for orientation, axis in (("horizontal", 0), ("vertical", 1)):
            line_colors = _extract_line_colors(grid, occluder, axis)
            if line_colors is None:
                continue
            fit = _fit_cycle(line_colors)
            if fit is None:
                continue
            period, seq = fit
            score = sum(color is not None for color in line_colors)
            model = {
                "occluder": occluder,
                "mask": mask,
                "orientation": orientation,
                "period": period,
                "seq": seq,
                "score": score,
            }
            if best is None or model["score"] > best["score"]:
                best = model
    return best


def main(grid: np.ndarray) -> np.ndarray:
    """Replace the grey occluder with the periodic stripe colors hidden underneath.

    Input structure:
    - The grid is filled by either horizontal or vertical monochrome stripes.
    - The stripe colors repeat with a fixed period of 2 to 4, and an internal 4-connected grey blob overwrites part of them.
    - Outside the grey blob the original stripe fragments remain visible.

    Rule:
    - Find the occluding blob and determine whether rows or columns are the stripe-carrying lines by checking which visible lines stay monochrome.
    - Infer the smallest 2 to 4 color cycle that matches the visible line colors and align that cycle to absolute row or column indices.
    - Replace every grey cell by the stripe color dictated by its row index for horizontal stripes or its column index for vertical stripes.

    Key insight:
    - The blob's shape does not encode the answer; the visible stripe fragments reveal both the stripe orientation and the phase of the hidden periodic pattern.
    """

    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    model = _infer_model(g)
    assert model is not None, "could not infer a valid stripe model"

    out = g.copy()
    mask = model["mask"]
    seq = model["seq"]
    period = model["period"]

    if model["orientation"] == "horizontal":
        for r in range(g.shape[0]):
            out[r, mask[r, :]] = seq[r % period]
    else:
        for c in range(g.shape[1]):
            out[mask[:, c], c] = seq[c % period]
    return out


def _render_stripes(h: int, w: int, orientation: str, seq: list[int], phase: int) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    period = len(seq)
    if orientation == "horizontal":
        for r in range(h):
            grid[r, :] = seq[(r + phase) % period]
    else:
        for c in range(w):
            grid[:, c] = seq[(c + phase) % period]
    return grid


def _grow_blob(h: int, w: int, rng: random.Random) -> np.ndarray | None:
    max_box_h = min(10, h - 2)
    max_box_w = min(10, w - 2)
    for _ in range(80):
        box_h = rng.randint(4, max_box_h)
        box_w = rng.randint(4, max_box_w)
        top = rng.randint(1, h - box_h - 1)
        left = rng.randint(1, w - box_w - 1)

        area = box_h * box_w
        lo = max(7, area // 3)
        hi = min(area - 1, (2 * area) // 3)
        if lo > hi:
            continue
        target = rng.randint(lo, hi)

        seed = (rng.randrange(box_h), rng.randrange(box_w))
        cells = {seed}
        frontier = [seed]
        while frontier and len(cells) < target:
            idx = rng.randrange(len(frontier))
            r, c = frontier[idx]
            choices = []
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells:
                    choices.append((nr, nc))
            if not choices:
                frontier.pop(idx)
                continue
            nxt = rng.choice(choices)
            cells.add(nxt)
            frontier.append(nxt)

        if len(cells) != target:
            continue

        local = np.zeros((box_h, box_w), dtype=bool)
        for r, c in cells:
            local[r, c] = True
        rows = np.where(local.any(axis=1))[0]
        cols = np.where(local.any(axis=0))[0]
        if rows.size < 2 or cols.size < 2:
            continue
        tight = local[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
        if tight.all():
            continue

        mask = np.zeros((h, w), dtype=bool)
        mask[top : top + box_h, left : left + box_w] = local
        if _touches_border(mask):
            continue
        return mask
    return None


def _attempt_generate(rng: random.Random) -> np.ndarray | None:
    orientation = rng.choice(["horizontal", "vertical"])
    period = rng.randint(2, 4)

    min_h = max(8, period + 3)
    min_w = max(8, period + 3)
    h = rng.randint(min_h, 20)
    w = rng.randint(min_w, 20)

    palette = [color for color in range(10) if color != GREY]
    stripe_colors = rng.sample(palette, period)
    phase = rng.randrange(period)
    base = _render_stripes(h, w, orientation, stripe_colors, phase)

    blob = _grow_blob(h, w, rng)
    if blob is None:
        return None

    inp = base.copy()
    inp[blob] = GREY

    solved = main(inp)
    if not np.array_equal(solved, base):
        return None
    model = _infer_model(inp)
    if model is None or model["orientation"] != orientation or model["occluder"] != GREY:
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid stripe-occlusion puzzle")
