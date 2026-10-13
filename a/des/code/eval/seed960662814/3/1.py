import random
from collections import Counter

import numpy as np


ORIENTATIONS = ("H", "V", "D")
TEMPLATE_SIZE = 3
TEMPLATE_GAP = 1
GEN_RETRIES = 400
GRID_H_RANGE = (18, 26)
GRID_W_RANGE = (18, 26)
BLOCK_COUNT_OPTIONS = (3, 4)
BLOCK_PAD = 1
BLOCK_PLACE_TRIES = 80
H_BLOCK_H_RANGE = (4, 8)
H_BLOCK_W_RANGE = (5, 9)
V_BLOCK_H_RANGE = (5, 9)
V_BLOCK_W_RANGE = (4, 8)
D_BLOCK_SIDE_RANGE = (4, 7)


def _bg_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((int(cr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(pts)
    return out


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _line_orientation(points: set[tuple[int, int]], h: int, w: int) -> str | None:
    if any(all((r, c) in points for c in range(w)) for r in range(h)):
        return "H"
    if any(all((r, c) in points for r in range(h)) for c in range(w)):
        return "V"
    if h == w and all((i, i) in points for i in range(h)):
        return "D"
    if h == w and all((i, w - 1 - i) in points for i in range(h)):
        return "D"
    return None


def _classify_template(patch: np.ndarray) -> tuple[str, int] | None:
    colors = [int(c) for c in np.unique(patch)]
    if len(colors) != 2:
        return None
    h, w = patch.shape
    for line_color in colors:
        pts = {(int(r), int(c)) for r, c in np.argwhere(patch == int(line_color))}
        if len(pts) != 3:
            continue
        orientation = _line_orientation(pts, h, w)
        if orientation is None:
            continue
        fill_color = colors[0] if colors[1] == line_color else colors[1]
        return orientation, int(fill_color)
    return None


def _find_templates(grid: np.ndarray, bg: int) -> tuple[dict[str, int], np.ndarray]:
    h, w = grid.shape
    template_mask = np.zeros((h, w), dtype=bool)
    mapping: dict[str, int] = {}
    for comp in _components(grid != bg):
        if len(comp) != TEMPLATE_SIZE * TEMPLATE_SIZE:
            continue
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 + 1 != TEMPLATE_SIZE or c1 - c0 + 1 != TEMPLATE_SIZE:
            continue
        if not (r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1):
            continue
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        info = _classify_template(patch)
        if info is None:
            continue
        orientation, fill_color = info
        mapping[orientation] = int(fill_color)
        template_mask[r0 : r1 + 1, c0 : c1 + 1] = True
    assert set(ORIENTATIONS).issubset(mapping), "expected one template for each crack orientation"
    return mapping, template_mask


def _classify_crack(missing: np.ndarray, h: int, w: int) -> str | None:
    pts = {(int(r), int(c)) for r, c in missing}
    return _line_orientation(pts, h, w)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair cracked monochrome rectangles by matching crack orientation to an edge template.

    Input structure:
    - The most common color is background.
    - Several large monochrome rectangles have a one-cell-wide background crack passing
      through them horizontally, vertically, or diagonally.
    - A row or column of 3x3 edge templates uses a line to encode crack orientation;
      the non-line color is the fill color for that orientation.

    Rule:
    - Detect the templates and build an orientation -> fill-color mapping.
    - For each cracked rectangle, infer its crack orientation from the missing cells in
      the rectangle's bounding box.
    - Replace the crack's background cells with the mapped fill color.
    - Remove the template area by setting it back to background.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    mapping, template_mask = _find_templates(g, bg)

    out = g.copy()
    out[template_mask] = bg

    object_colors = [int(c) for c in np.unique(g[~template_mask]) if int(c) != bg]
    for color in object_colors:
        points = np.argwhere((g == int(color)) & (~template_mask))
        if points.size == 0:
            continue
        r0, c0 = points.min(axis=0)
        r1, c1 = points.max(axis=0)
        patch = out[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
        assert bool(np.isin(patch, [bg, int(color)]).all()), "object bbox must contain only its color and background"
        missing = np.argwhere(patch == bg)
        assert missing.size > 0, "object must contain a crack"
        orientation = _classify_crack(missing, patch.shape[0], patch.shape[1])
        assert orientation in mapping, "could not match crack orientation to a template"
        fill_color = mapping[orientation]
        repaired = patch.copy()
        repaired[repaired == bg] = fill_color
        out[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = repaired

    return out


def _template_patch(orientation: str, fill_color: int, line_color: int, diag_kind: str) -> np.ndarray:
    patch = np.full((TEMPLATE_SIZE, TEMPLATE_SIZE), int(fill_color), dtype=int)
    if orientation == "H":
        patch[1, :] = int(line_color)
    elif orientation == "V":
        patch[:, 1] = int(line_color)
    else:
        idx = np.arange(TEMPLATE_SIZE)
        if diag_kind == "main":
            patch[idx, idx] = int(line_color)
        else:
            patch[idx, TEMPLATE_SIZE - 1 - idx] = int(line_color)
    return patch


def _block_patch(orientation: str, color: int, bg: int, diag_kind: str) -> np.ndarray:
    if orientation == "H":
        h = random.randint(*H_BLOCK_H_RANGE)
        w = random.randint(*H_BLOCK_W_RANGE)
        patch = np.full((h, w), int(color), dtype=int)
        patch[random.randint(1, h - 2), :] = int(bg)
        return patch
    if orientation == "V":
        h = random.randint(*V_BLOCK_H_RANGE)
        w = random.randint(*V_BLOCK_W_RANGE)
        patch = np.full((h, w), int(color), dtype=int)
        patch[:, random.randint(1, w - 2)] = int(bg)
        return patch
    side = random.randint(*D_BLOCK_SIDE_RANGE)
    patch = np.full((side, side), int(color), dtype=int)
    idx = np.arange(side)
    if diag_kind == "main":
        patch[idx, idx] = int(bg)
    else:
        patch[idx, side - 1 - idx] = int(bg)
    return patch


def _reserve(mask: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> None:
    rr0 = max(0, int(r0) - pad)
    cc0 = max(0, int(c0) - pad)
    rr1 = min(mask.shape[0], int(r0) + int(h) + pad)
    cc1 = min(mask.shape[1], int(c0) + int(w) + pad)
    mask[rr0:rr1, cc0:cc1] = True


def _can_place(mask: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> bool:
    rr0 = max(0, int(r0) - pad)
    cc0 = max(0, int(c0) - pad)
    rr1 = min(mask.shape[0], int(r0) + int(h) + pad)
    cc1 = min(mask.shape[1], int(c0) + int(w) + pad)
    return not bool(mask[rr0:rr1, cc0:cc1].any())


def _template_slots(h: int, w: int, edge: str, order: list[str]) -> list[tuple[str, int, int]] | None:
    span = TEMPLATE_SIZE * len(order) + TEMPLATE_GAP * (len(order) - 1)
    slots: list[tuple[str, int, int]] = []
    if edge in ("top", "bottom"):
        if w < span:
            return None
        start = random.randint(0, w - span)
        row = 0 if edge == "top" else h - TEMPLATE_SIZE
        for i, orientation in enumerate(order):
            col = start + i * (TEMPLATE_SIZE + TEMPLATE_GAP)
            slots.append((orientation, int(row), int(col)))
        return slots
    if h < span:
        return None
    start = random.randint(0, h - span)
    col = 0 if edge == "left" else w - TEMPLATE_SIZE
    for i, orientation in enumerate(order):
        row = start + i * (TEMPLATE_SIZE + TEMPLATE_GAP)
        slots.append((orientation, int(row), int(col)))
    return slots


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    num_blocks = random.choice(BLOCK_COUNT_OPTIONS)
    needed_colors = 5 + num_blocks
    if needed_colors > 10:
        return None

    palette = random.sample(range(10), needed_colors)
    bg = int(palette[0])
    line_color = int(palette[1])
    fill_colors = dict(zip(ORIENTATIONS, map(int, palette[2:5])))
    block_colors = [int(c) for c in palette[5:]]

    grid = np.full((h, w), bg, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    diag_kind = random.choice(("main", "anti"))
    template_order = list(ORIENTATIONS)
    random.shuffle(template_order)
    edge = random.choice(("top", "bottom", "left", "right"))
    slots = _template_slots(h, w, edge, template_order)
    if slots is None:
        return None

    for orientation, r0, c0 in slots:
        patch = _template_patch(orientation, fill_colors[orientation], line_color, diag_kind)
        grid[r0 : r0 + TEMPLATE_SIZE, c0 : c0 + TEMPLATE_SIZE] = patch
        _reserve(reserved, r0, c0, TEMPLATE_SIZE, TEMPLATE_SIZE, pad=1)

    orientations = ["H", "V", "D"]
    if num_blocks == 4:
        orientations.append(random.choice(ORIENTATIONS))
    random.shuffle(orientations)

    blocks: list[tuple[str, np.ndarray]] = []
    for orientation, color in zip(orientations, block_colors):
        blocks.append((orientation, _block_patch(orientation, color, bg, diag_kind)))
    blocks.sort(key=lambda item: item[1].shape[0] * item[1].shape[1], reverse=True)

    for orientation, patch in blocks:
        ph, pw = patch.shape
        if h - ph < 2 or w - pw < 2:
            return None
        placed = False
        for _ in range(BLOCK_PLACE_TRIES):
            r0 = random.randint(1, h - ph - 1)
            c0 = random.randint(1, w - pw - 1)
            if not _can_place(reserved, r0, c0, ph, pw, pad=BLOCK_PAD):
                continue
            grid[r0 : r0 + ph, c0 : c0 + pw] = patch
            _reserve(reserved, r0, c0, ph, pw, pad=BLOCK_PAD)
            placed = True
            break
        if not placed:
            return None

    counts = Counter(map(int, grid.ravel()))
    if counts[bg] != max(counts.values()):
        return None

    try:
        main(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
