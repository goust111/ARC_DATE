from __future__ import annotations

import numpy as np

from arc_utils import Color, colors_present, find_components, get_bg, is_contiguous, paste
import itertools
import math
import random


RED = Color.RED
GREEN = Color.GREEN


def _component_mask(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _has_flanked_boundary_gap(mask: np.ndarray) -> bool:
    h, w = mask.shape

    for c in range(w):
        if not mask[0, c] and mask[0, :c].any() and mask[0, c + 1 :].any():
            return True
        if h > 1 and not mask[h - 1, c] and mask[h - 1, :c].any() and mask[h - 1, c + 1 :].any():
            return True

    for r in range(h):
        if not mask[r, 0] and mask[:r, 0].any() and mask[r + 1 :, 0].any():
            return True
        if w > 1 and not mask[r, w - 1] and mask[:r, w - 1].any() and mask[r + 1 :, w - 1].any():
            return True

    return False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each foreground object horizontally to the edge indicated by whether its shape has an open notch.

    Input structure:
    - Yellow is the background and there is one blue foreground color.
    - The foreground is split into separated 4-connected objects.
    - Each object keeps its row span and shape in the output.

    Rule:
    - If an object's bbox border contains a background gap flanked by object pixels along that same border, slide it to the right edge and recolor it green.
    - Otherwise slide the object to the left edge and recolor it red.
    - Start from a blank background canvas; no moved objects overlap.

    Key insight:
    - The target side is encoded by the object's shape, not by its original position: open notches go right, while closed or solid shapes go left.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    fg_colors = colors_present(g, exclude={bg})
    assert len(fg_colors) == 1, "expected exactly one foreground color"
    fg = int(fg_colors[0])

    components = find_components(g, bg=bg, connectivity=4, monochromatic=True, colors=[fg])
    assert components, "expected at least one foreground object"

    out = np.full_like(g, bg)
    W = g.shape[1]
    for component in components:
        bbox = tuple(map(int, component["bbox"]))
        r0, _c0, r1, _c1 = bbox
        mask = _component_mask(component["coords"], bbox)
        h, w = mask.shape

        has_open_notch = _has_flanked_boundary_gap(mask)
        dst_c = W - w if has_open_notch else 0
        color = GREEN if has_open_notch else RED
        assert 0 <= dst_c and dst_c + w <= W, "moved object must fit within the grid"

        dst = out[r0 : r1 + 1, dst_c : dst_c + w]
        assert dst.shape == (h, w), "destination slice must match object bbox"
        assert np.all(dst[mask] == bg), "moved objects must not overlap"
        dst[mask] = color

    return out

# --------------- generator (o5x) ---------------

# Canvas and colors.
BG, BLUE, RED, GREEN = Color.YELLOW, Color.BLUE, Color.RED, Color.GREEN
GRID_H, GRID_W_MIN, GRID_W_MAX = 22, 19, 26

# Object counts, dimensions, and row separation.
OBJECT_COUNT_MIN, OBJECT_COUNT_MAX, NOTCH_OBJECT_COUNT = 3, 5, 2
OBJECT_H_MIN, OBJECT_H_MAX, OBJECT_W_MIN, OBJECT_W_MAX = 2, 6, 2, 9
MIN_ROW_GAP = 1

# Shape density and carving probabilities.
MIN_FILL_RATIO, CLOSED_LARGE_AREA_MIN = 0.64, 12
NOTCH_INTERIOR_CARVE_PROB, EXTRA_CARVE_PROB = 0.45, 0.20
CLOSED_CARVE_PROB_SMALL, CLOSED_CARVE_PROB_LARGE = 0.45, 0.88

# Placement and retry limits.
INTERIOR_COL_PAD, MAX_FOREGROUND_FRACTION = 1, 0.5
GENERATION_RETRIES = 300


def _has_flanked_boundary_gap(mask: np.ndarray) -> bool:
    h, w = mask.shape
    for c in range(w):
        if not mask[0, c] and mask[0, :c].any() and mask[0, c + 1 :].any():
            return True
        if h > 1 and not mask[h - 1, c] and mask[h - 1, :c].any() and mask[h - 1, c + 1 :].any():
            return True
    for r in range(h):
        if not mask[r, 0] and mask[:r, 0].any() and mask[r + 1 :, 0].any():
            return True
        if w > 1 and not mask[r, w - 1] and mask[:r, w - 1].any() and mask[r + 1 :, w - 1].any():
            return True
    return False


def _patch(mask: np.ndarray, color: int) -> np.ndarray:
    out = np.full(mask.shape, BG, dtype=int)
    out[mask] = int(color)
    return out


def _random_heights(n: int) -> list[int]:
    usable = GRID_H - MIN_ROW_GAP * (n - 1)
    choices = [
        hs
        for hs in itertools.product(range(OBJECT_H_MIN, OBJECT_H_MAX + 1), repeat=n)
        if sum(hs) <= usable
    ]
    return list(random.choice(choices))


def _row_starts(heights: list[int]) -> list[int]:
    free = GRID_H - sum(heights) - MIN_ROW_GAP * (len(heights) - 1)
    gaps = [0] * (len(heights) + 1)
    for _ in range(free):
        gaps[random.randrange(len(gaps))] += 1

    starts = []
    r = gaps[0]
    for i, h in enumerate(heights):
        starts.append(r)
        r += h
        if i + 1 < len(heights):
            r += MIN_ROW_GAP + gaps[i + 1]
    return starts


def _carve_interior_rect(mask: np.ndarray, max_remove: int, probability: float) -> int:
    if max_remove <= 0 or random.random() > probability:
        return 0

    h, w = mask.shape
    choices = []
    for r0 in range(1, h - 1):
        for c0 in range(1, w - 1):
            for r1 in range(r0, h - 1):
                for c1 in range(c0, w - 1):
                    area = (r1 - r0 + 1) * (c1 - c0 + 1)
                    if area <= max_remove and np.all(mask[r0 : r1 + 1, c0 : c1 + 1]):
                        choices.append((r0, c0, r1, c1, area))
    if not choices:
        return 0

    r0, c0, r1, c1, area = random.choice(choices)
    mask[r0 : r1 + 1, c0 : c1 + 1] = False
    return int(area)


def _make_notch_mask(h: int, w: int) -> np.ndarray | None:
    area = h * w
    max_remove = area - math.ceil(area * MIN_FILL_RATIO)
    edges = []
    if w >= OBJECT_W_MIN + 1:
        edges.extend(["top", "bottom"])
    if h >= OBJECT_H_MIN + 1:
        edges.extend(["left", "right"])
    if not edges or max_remove < 1:
        return None

    mask = np.ones((h, w), dtype=bool)
    edge = random.choice(edges)
    edge_len = w if edge in {"top", "bottom"} else h
    run = random.randint(1, min(edge_len - 2, max_remove))
    start = random.randint(1, edge_len - run - 1)

    if edge in {"top", "bottom"}:
        depth = random.randint(1, min(h - 1, max_remove // run))
        if edge == "top":
            mask[:depth, start : start + run] = False
        else:
            mask[h - depth :, start : start + run] = False
    elif edge == "left":
        depth = random.randint(1, min(w - 1, max_remove // run))
        mask[start : start + run, :depth] = False
    else:
        depth = random.randint(1, min(w - 1, max_remove // run))
        mask[start : start + run, w - depth :] = False

    removed = run * depth
    removed += _carve_interior_rect(mask, max_remove - removed, NOTCH_INTERIOR_CARVE_PROB)
    _carve_interior_rect(mask, max_remove - removed, EXTRA_CARVE_PROB)
    if not _has_flanked_boundary_gap(mask):
        return None
    if not is_contiguous(mask.astype(int), background=0, connectivity=4):
        return None
    return mask


def _make_closed_mask(h: int, w: int) -> np.ndarray | None:
    area = h * w
    mask = np.ones((h, w), dtype=bool)
    max_remove = area - math.ceil(area * MIN_FILL_RATIO)
    probability = CLOSED_CARVE_PROB_LARGE if area >= CLOSED_LARGE_AREA_MIN else CLOSED_CARVE_PROB_SMALL
    removed = _carve_interior_rect(mask, max_remove, probability)
    _carve_interior_rect(mask, max_remove - removed, EXTRA_CARVE_PROB)
    if _has_flanked_boundary_gap(mask):
        return None
    if not is_contiguous(mask.astype(int), background=0, connectivity=4):
        return None
    return mask


def _sample_layout() -> tuple[int, list[str], list[int], list[int]]:
    W = random.randint(GRID_W_MIN, GRID_W_MAX)
    n = random.randint(OBJECT_COUNT_MIN, OBJECT_COUNT_MAX)
    kinds = ["notch"] * NOTCH_OBJECT_COUNT + ["closed"] * (n - NOTCH_OBJECT_COUNT)
    random.shuffle(kinds)

    heights = _random_heights(n)
    starts = _row_starts(heights)
    return W, kinds, heights, starts


def _sample_objects(grid_w: int, kinds: list[str], starts: list[int], heights: list[int]):
    max_w = min(OBJECT_W_MAX, (grid_w - 1) // 2)

    objects = []
    for kind, r0, h in zip(kinds, starts, heights):
        if kind == "notch":
            widths = [w for w in range(OBJECT_W_MIN, max_w + 1) if w >= 3 or h >= 3]
            w = random.choice(widths)
            mask = _make_notch_mask(h, w)
            color, edge_c = GREEN, grid_w - w
        else:
            w = random.randint(OBJECT_W_MIN, max_w)
            mask = _make_closed_mask(h, w)
            color, edge_c = RED, 0
        if mask is None:
            return None
        objects.append((mask, r0, w, color, edge_c))
    return objects


def _render_pair(objects, grid_w: int) -> tuple[np.ndarray, np.ndarray] | None:
    inp = np.full((GRID_H, grid_w), BG, dtype=int)
    out = np.full((GRID_H, grid_w), BG, dtype=int)
    for mask, r0, w, color, edge_c in objects:
        paste(out, _patch(mask, color), r0, edge_c, transparent=BG)

    if sum(int(mask.sum()) for mask, *_ in objects) >= (GRID_H * grid_w) * MAX_FOREGROUND_FRACTION:
        return None

    for mask, r0, w, _color, _edge_c in objects:
        c0 = random.randint(INTERIOR_COL_PAD, grid_w - w - INTERIOR_COL_PAD)
        paste(inp, _patch(mask, BLUE), r0, c0, transparent=BG)
    return inp, out


def _oracle_accepts(inp: np.ndarray, out: np.ndarray) -> bool:
    try:
        solved = main(inp)
    except AssertionError:
        return False
    return bool(np.array_equal(solved, out))


def _gen_attempt() -> np.ndarray | None:
    W, kinds, heights, starts = _sample_layout()
    objects = _sample_objects(W, kinds, starts, heights)
    if objects is None:
        return None

    rendered = _render_pair(objects, W)
    if rendered is None:
        return None

    inp, out = rendered
    return inp if _oracle_accepts(inp, out) else None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 320afe60 input")
