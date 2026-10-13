import random
from collections import Counter, deque

import numpy as np


BG = 0
YELLOW = 4
GREY = 5
PALETTE = [1, 2, 3, 6, 7, 8, 9]
ROT_K_BY_SIDE = {
    "top": 0,
    "right": 3,
    "bottom": 2,
    "left": 1,
}
OUTER_SIZE_CHOICES = {
    2: [9, 11, 13, 15],
    3: [9, 11, 13],
    4: [9, 11],
}
CANVAS_SIZE = 30
GEN_ATTEMPTS = 400


def _mask(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


PATTERN_TEMPLATES = [
    _mask(["100", "100", "110"]),
    _mask(["111", "010", "010"]),
    _mask(["110", "010", "011"]),
    _mask(["1100", "0100", "0111", "0001"]),
    _mask(["1110", "1000", "1100", "0100"]),
]


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[counts.argmax()])


def _components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    else:
        neighbors = [
            (dr, dc)
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if not (dr == 0 and dc == 0)
        ]

    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in neighbors:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _border_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _is_complete_ring(sub: np.ndarray, color: int) -> bool:
    h, w = sub.shape
    if h < 3 or w < 3:
        return False
    border = _border_mask(h, w)
    return bool(np.all(sub[border] == color) and np.all(sub[~border] != color))


def _legend_mapping(grid: np.ndarray, bg: int) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for coords in _components_mask(grid != bg, connectivity=4):
        if len(coords) != 2:
            continue
        r0, c0, r1, c1 = _bbox_coords(coords)
        if (r1 - r0 + 1, c1 - c0 + 1) != (2, 1):
            continue
        top = int(grid[r0, c0])
        bottom = int(grid[r1, c0])
        if top in (bg, GREY, YELLOW) or bottom in (bg, GREY, YELLOW) or top == bottom:
            continue
        assert top not in mapping, f"duplicate legend source color {top}"
        mapping[top] = bottom
    assert 2 <= len(mapping) <= 4, f"expected 2-4 legend entries, got {len(mapping)}"
    return mapping


def _gap_side(sub: np.ndarray, bg: int) -> str:
    h, w = sub.shape
    border = _border_mask(h, w)
    missing = np.argwhere(border & (sub != GREY))
    assert len(missing) == 1, f"expected one outer-frame gap, got {len(missing)}"
    mr, mc = map(int, missing[0])
    assert int(sub[mr, mc]) == bg, "outer-frame gap must expose the background"
    expected = {
        "top": (0, w // 2),
        "right": (h // 2, w - 1),
        "bottom": (h - 1, w // 2),
        "left": (h // 2, 0),
    }
    for side, pos in expected.items():
        if (mr, mc) == pos:
            return side
    raise AssertionError(f"gap is not at a side midpoint: {(mr, mc)} in {(h, w)}")


def _outer_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames = []
    for coords in _components_mask(grid == GREY, connectivity=4):
        r0, c0, r1, c1 = _bbox_coords(coords)
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        h, w = sub.shape
        if h < 5 or w < 5:
            continue
        border = _border_mask(h, w)
        if np.any(sub[1:-1, 1:-1] == GREY):
            continue
        if np.count_nonzero(sub == GREY) != int(border.sum()) - 1:
            continue
        side = _gap_side(sub, bg)
        frames.append({"bbox": (r0, c0, r1, c1), "gap_side": side})
    assert frames, "no valid outer frames found"
    frames.sort(key=lambda item: item["bbox"])
    return frames


def _inner_frame(grid: np.ndarray, outer_bbox: tuple[int, int, int, int], bg: int) -> tuple[int, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = outer_bbox
    sub = grid[r0 + 1 : r1, c0 + 1 : c1]
    candidates = []
    for color in sorted(set(map(int, np.unique(sub))) - {bg, GREY, YELLOW}):
        for coords in _components_mask(sub == color, connectivity=4):
            ir0, ic0, ir1, ic1 = _bbox_coords(coords)
            piece = sub[ir0 : ir1 + 1, ic0 : ic1 + 1]
            if _is_complete_ring(piece, color):
                candidates.append((color, (r0 + 1 + ir0, c0 + 1 + ic0, r0 + 1 + ir1, c0 + 1 + ic1)))
    assert len(candidates) == 1, f"expected one inner frame in {outer_bbox}, got {len(candidates)}"
    return candidates[0]


def _apply_rule(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    legend = _legend_mapping(g, bg)
    frames = _outer_frames(g, bg)

    out = g.copy()
    for frame in frames:
        inner_color, (ir0, ic0, ir1, ic1) = _inner_frame(g, frame["bbox"], bg)
        assert inner_color in legend, f"missing legend entry for inner-frame color {inner_color}"
        inner_region = g[ir0 + 1 : ir1, ic0 + 1 : ic1]
        assert inner_region.shape[0] == inner_region.shape[1], "inner frame interior must be square"
        colors_inside = set(map(int, np.unique(inner_region)))
        assert colors_inside <= {bg, YELLOW}, f"unexpected colors inside inner frame: {sorted(colors_inside)}"
        assert np.any(inner_region == YELLOW), "inner frame must contain a yellow pattern"

        rotated = np.rot90(inner_region == YELLOW, k=ROT_K_BY_SIDE[frame["gap_side"]])
        new_region = np.full(inner_region.shape, bg, dtype=int)
        new_region[rotated] = int(legend[inner_color])
        out[ir0 + 1 : ir1, ic0 + 1 : ic1] = new_region

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate and recolor each inner yellow pattern using the outer-frame gap and the color legend.

    Input structure:
    - The background is black.
    - The grid contains several nested structures: each one has a grey outer rectangular ring with a single midpoint gap on one side, plus a smaller colored inner ring centered inside it.
    - Inside each inner ring is a yellow asymmetric pattern on black, and elsewhere there is a separate legend made of isolated 1x2 vertical color pairs.

    Rule:
    - Read each legend pair as a color mapping from its top pixel to its bottom pixel.
    - For every nested structure, convert the outer-frame gap side into a rotation: top = 0 degrees, right = 90 degrees clockwise, bottom = 180 degrees, left = 270 degrees clockwise.
    - Rotate the yellow pattern within the inner frame by that amount, recolor the rotated pattern to the legend target color for the inner-frame color, and leave both frames and the rest of the grid unchanged.

    Key insight:
    - The outer gap encodes the spatial transform while the inner-frame color selects the attribute transform, so solving the task requires combining direction decoding with legend-based recoloring.
    """
    return _apply_rule(np.asarray(grid, dtype=int))


def _derangement(items: list[int], rng: random.Random) -> list[int]:
    if len(items) == 2:
        return [items[1], items[0]]
    arr = items[:]
    while True:
        rng.shuffle(arr)
        if all(a != b for a, b in zip(items, arr)):
            return arr[:]


def _eligible_templates(side: int) -> list[np.ndarray]:
    return [template for template in PATTERN_TEMPLATES if template.shape[0] <= side and template.shape[1] <= side]


def _offset_limit(slack: int, rng: random.Random) -> int:
    if slack <= 0:
        return 0
    if slack >= 2:
        return rng.randint(1, slack - 1)
    return rng.randint(0, slack)


def _group_patch(size: int, inner_color: int, gap_side: str, rng: random.Random) -> np.ndarray:
    patch = np.full((size, size), BG, dtype=int)
    patch[0, :] = GREY
    patch[-1, :] = GREY
    patch[:, 0] = GREY
    patch[:, -1] = GREY

    gap_positions = {
        "top": (0, size // 2),
        "right": (size // 2, size - 1),
        "bottom": (size - 1, size // 2),
        "left": (size // 2, 0),
    }
    gap_r, gap_c = gap_positions[gap_side]
    patch[gap_r, gap_c] = BG

    pad_options = [pad for pad in (2, 3) if size - 2 * pad >= 5]
    pad = rng.choice(pad_options)
    inner_size = size - 2 * pad
    r0 = c0 = pad
    r1 = c1 = pad + inner_size - 1
    patch[r0, c0 : c1 + 1] = inner_color
    patch[r1, c0 : c1 + 1] = inner_color
    patch[r0 : r1 + 1, c0] = inner_color
    patch[r0 : r1 + 1, c1] = inner_color

    inner_side = inner_size - 2
    template = rng.choice(_eligible_templates(inner_side))
    template = np.rot90(template, k=rng.randrange(4))
    ph, pw = template.shape
    off_r = _offset_limit(inner_side - ph, rng)
    off_c = _offset_limit(inner_side - pw, rng)
    region = patch[r0 + 1 : r1, c0 + 1 : c1]
    region[off_r : off_r + ph, off_c : off_c + pw][template] = YELLOW

    return patch


def _legend_patch(sources: list[int], targets: list[int], rng: random.Random) -> np.ndarray:
    order = list(range(len(sources)))
    rng.shuffle(order)
    patch = np.full((2, 2 * len(order) - 1), BG, dtype=int)
    for idx, src_idx in enumerate(order):
        col = 2 * idx
        patch[0, col] = sources[src_idx]
        patch[1, col] = targets[src_idx]
    return patch


def _place_patches(patches: list[np.ndarray], rng: random.Random) -> np.ndarray | None:
    canvas = np.full((CANVAS_SIZE, CANVAS_SIZE), BG, dtype=int)
    blocked = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=bool)
    order = sorted(range(len(patches)), key=lambda idx: patches[idx].shape[0] * patches[idx].shape[1], reverse=True)
    for idx in order:
        patch = patches[idx]
        ph, pw = patch.shape
        candidates = []
        for r in range(CANVAS_SIZE - ph + 1):
            for c in range(CANVAS_SIZE - pw + 1):
                rr0 = max(0, r - 1)
                cc0 = max(0, c - 1)
                rr1 = min(CANVAS_SIZE, r + ph + 1)
                cc1 = min(CANVAS_SIZE, c + pw + 1)
                if not blocked[rr0:rr1, cc0:cc1].any():
                    candidates.append((r, c))
        if not candidates:
            return None
        r, c = rng.choice(candidates)
        mask = patch != BG
        canvas[r : r + ph, c : c + pw][mask] = patch[mask]
        blocked[max(0, r - 1) : min(CANVAS_SIZE, r + ph + 1), max(0, c - 1) : min(CANVAS_SIZE, c + pw + 1)] = True
    return canvas


def _crop_canvas(canvas: np.ndarray, rng: random.Random) -> np.ndarray:
    coords = np.argwhere(canvas != BG)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    top = rng.randint(0, min(4, int(r0)))
    left = rng.randint(0, min(4, int(c0)))
    bottom = rng.randint(0, min(4, int(canvas.shape[0] - 1 - r1)))
    right = rng.randint(0, min(4, int(canvas.shape[1] - 1 - c1)))
    return canvas[r0 - top : r1 + 1 + bottom, c0 - left : c1 + 1 + right].copy()


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    group_count = rng.randint(2, 4)
    distinct_colors = rng.randint(2, min(4, group_count))
    sources = rng.sample(PALETTE, distinct_colors)
    targets = _derangement(sources, rng)

    group_colors = sources[:]
    while len(group_colors) < group_count:
        group_colors.append(rng.choice(sources))
    rng.shuffle(group_colors)

    gap_sides = rng.sample(list(ROT_K_BY_SIDE), group_count)
    size_choices = OUTER_SIZE_CHOICES[group_count]
    group_patches = [
        _group_patch(rng.choice(size_choices), group_colors[idx], gap_sides[idx], rng)
        for idx in range(group_count)
    ]
    legend = _legend_patch(sources, targets, rng)
    canvas = _place_patches(group_patches + [legend], rng)
    if canvas is None:
        return None

    grid = _crop_canvas(canvas, rng)
    try:
        out = main(grid)
    except AssertionError:
        return None
    if not np.any(out != grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(random)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_ATTEMPTS} attempts")
