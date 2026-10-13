from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
FRAME_COLOR = 1
ANCHOR_COLORS = (2, 3, 4, 5, 6, 7, 8, 9)
TILE_SIZES = (2, 3)
CANVAS_H_RANGE = (5, 10)
CANVAS_W_RANGE = (5, 11)
MARGIN_RANGE = (2, 5)
ANCHOR_COUNT_RANGE = (3, 6)
ANCHOR_SAMPLE_TRIES = 80
GEN_RETRIES = 500


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    return len(_components(mask)) == 1


def _enumerate_tile_masks(size: int) -> tuple[np.ndarray, ...]:
    shapes: list[np.ndarray] = []
    for bits in range(1, 1 << (size * size)):
        mask = np.zeros((size, size), dtype=bool)
        for idx in range(size * size):
            if bits & (1 << idx):
                mask[idx // size, idx % size] = True
        if np.all(mask):
            continue
        if not mask[0].any() or not mask[-1].any():
            continue
        if not mask[:, 0].any() or not mask[:, -1].any():
            continue
        if not _is_connected(mask):
            continue
        shapes.append(mask)
    return tuple(shapes)


TILE_LIBRARY = {size: _enumerate_tile_masks(size) for size in TILE_SIZES}


def _frame_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    frame_coords = np.argwhere(grid == FRAME_COLOR)
    assert frame_coords.size > 0, "missing frame"
    r0, c0 = frame_coords.min(axis=0)
    r1, c1 = frame_coords.max(axis=0)
    assert r1 - r0 >= 2 and c1 - c0 >= 2, "frame must enclose a canvas"

    expected = np.zeros_like(grid, dtype=bool)
    expected[r0, c0 : c1 + 1] = True
    expected[r1, c0 : c1 + 1] = True
    expected[r0 : r1 + 1, c0] = True
    expected[r0 : r1 + 1, c1] = True
    assert np.array_equal(grid == FRAME_COLOR, expected), "blue pixels must form one rectangular frame"
    return int(r0), int(c0), int(r1), int(c1)


def _extract_tile_and_anchors(
    grid: np.ndarray, frame_bbox: tuple[int, int, int, int]
) -> tuple[int, np.ndarray, list[tuple[int, int]]]:
    r0, c0, r1, c1 = frame_bbox
    canvas = grid[r0 + 1 : r1, c0 + 1 : c1]
    inner_colors = sorted(int(x) for x in np.unique(canvas) if int(x) not in (BG, FRAME_COLOR))
    assert len(inner_colors) == 1, "expected one anchor color inside the frame"
    anchor_color = inner_colors[0]

    anchors: list[tuple[int, int]] = []
    sample_comp: list[tuple[int, int]] | None = None
    for comp in _components(grid == anchor_color):
        inside = all(r0 < r < r1 and c0 < c < c1 for r, c in comp)
        if inside:
            anchors.extend((r - (r0 + 1), c - (c0 + 1)) for r, c in comp)
            continue
        assert sample_comp is None, "expected a single exterior tile sample"
        sample_comp = comp

    assert sample_comp is not None, "missing exterior tile sample"
    assert anchors, "missing anchors"

    sr0, sc0, sr1, sc1 = _bbox(sample_comp)
    tile_mask = np.zeros((sr1 - sr0 + 1, sc1 - sc0 + 1), dtype=bool)
    for r, c in sample_comp:
        tile_mask[r - sr0, c - sc0] = True

    th, tw = tile_mask.shape
    assert (th, tw) in ((2, 2), (3, 3)), "tile must be 2x2 or 3x3"
    assert _is_connected(tile_mask), "tile sample must be connected"
    return anchor_color, tile_mask, anchors


def _shifted_residues(tile_mask: np.ndarray, offset: tuple[int, int]) -> set[tuple[int, int]]:
    th, tw = tile_mask.shape
    dr, dc = offset
    residues: set[tuple[int, int]] = set()
    for tr, tc in np.argwhere(tile_mask):
        residues.add(((dr + int(tr)) % th, (dc + int(tc)) % tw))
    return residues


def _offset_candidates(
    canvas_shape: tuple[int, int], tile_mask: np.ndarray, anchors: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    del canvas_shape
    th, tw = tile_mask.shape
    candidates: list[tuple[int, int]] = []
    for dr in range(th):
        for dc in range(tw):
            residues = _shifted_residues(tile_mask, (dr, dc))
            if all((r % th, c % tw) in residues for r, c in anchors):
                candidates.append((dr, dc))
    return candidates


def _render_canvas(
    canvas_shape: tuple[int, int], tile_mask: np.ndarray, offset: tuple[int, int]
) -> np.ndarray:
    ch, cw = canvas_shape
    th, tw = tile_mask.shape
    residues = _shifted_residues(tile_mask, offset)
    canvas = np.zeros((ch, cw), dtype=bool)
    for r in range(ch):
        for c in range(cw):
            canvas[r, c] = (r % th, c % tw) in residues
    return canvas


def _apply_rule(grid: np.ndarray, tile_mask: np.ndarray, anchor_color: int, offset: tuple[int, int]) -> np.ndarray:
    out = np.array(grid, copy=True)
    r0, c0, r1, c1 = _frame_bbox(out)
    canvas_mask = _render_canvas((r1 - r0 - 1, c1 - c0 - 1), tile_mask, offset)
    for r in range(r1 - r0 - 1):
        for c in range(c1 - c0 - 1):
            out[r0 + 1 + r, c0 + 1 + c] = anchor_color if canvas_mask[r, c] else BG
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the periodic tiling phase from sparse anchors inside a blue frame,
    then use the exterior tile sample to fill the whole canvas.
    """
    g = np.asarray(grid, dtype=int)
    frame_bbox = _frame_bbox(g)
    anchor_color, tile_mask, anchors = _extract_tile_and_anchors(g, frame_bbox)

    r0, c0, r1, c1 = frame_bbox
    canvas_shape = (r1 - r0 - 1, c1 - c0 - 1)
    candidates = _offset_candidates(canvas_shape, tile_mask, anchors)
    assert len(candidates) == 1, "anchor phase must be uniquely determined"
    return _apply_rule(g, tile_mask, anchor_color, candidates[0])


def _choose_anchor_subset(
    rng: random.Random, full_points: list[tuple[int, int]], tile_mask: np.ndarray, canvas_shape: tuple[int, int]
) -> list[tuple[int, int]] | None:
    max_anchors = min(ANCHOR_COUNT_RANGE[1], len(full_points))
    if max_anchors < ANCHOR_COUNT_RANGE[0]:
        return None
    for _ in range(ANCHOR_SAMPLE_TRIES):
        count = rng.randint(ANCHOR_COUNT_RANGE[0], max_anchors)
        anchors = rng.sample(full_points, count)
        if len(_offset_candidates(canvas_shape, tile_mask, anchors)) == 1:
            return anchors
    return None


def _place_tile_sample(
    rng: random.Random,
    grid_shape: tuple[int, int],
    frame_bbox: tuple[int, int, int, int],
    tile_mask: np.ndarray,
) -> tuple[int, int]:
    grid_h, grid_w = grid_shape
    r0, c0, r1, c1 = frame_bbox
    th, tw = tile_mask.shape
    side = rng.choice(("top", "bottom", "left", "right"))

    if side == "top":
        sr0 = rng.randint(0, r0 - th - 1)
        sc0 = rng.randint(0, grid_w - tw)
    elif side == "bottom":
        sr0 = rng.randint(r1 + 2, grid_h - th)
        sc0 = rng.randint(0, grid_w - tw)
    elif side == "left":
        sr0 = rng.randint(0, grid_h - th)
        sc0 = rng.randint(0, c0 - tw - 1)
    else:
        sr0 = rng.randint(0, grid_h - th)
        sc0 = rng.randint(c1 + 2, grid_w - tw)
    return sr0, sc0


def _generate_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    size = rng.choice(TILE_SIZES)
    tile_mask = np.array(rng.choice(TILE_LIBRARY[size]), copy=True)

    canvas_h = rng.randint(max(CANVAS_H_RANGE[0], size + 1), CANVAS_H_RANGE[1])
    canvas_w = rng.randint(max(CANVAS_W_RANGE[0], size + 1), CANVAS_W_RANGE[1])
    offset = (rng.randrange(size), rng.randrange(size))
    full_mask = _render_canvas((canvas_h, canvas_w), tile_mask, offset)
    full_points = [tuple(map(int, rc)) for rc in np.argwhere(full_mask)]
    anchors = _choose_anchor_subset(rng, full_points, tile_mask, (canvas_h, canvas_w))
    if anchors is None:
        return None

    margins = {name: rng.randint(*MARGIN_RANGE) for name in ("top", "bottom", "left", "right")}
    margins["top"] = max(margins["top"], size + 1)
    margins["bottom"] = max(margins["bottom"], size + 1)
    margins["left"] = max(margins["left"], size + 1)
    margins["right"] = max(margins["right"], size + 1)

    grid_h = margins["top"] + canvas_h + 2 + margins["bottom"]
    grid_w = margins["left"] + canvas_w + 2 + margins["right"]
    frame_bbox = (
        margins["top"],
        margins["left"],
        margins["top"] + canvas_h + 1,
        margins["left"] + canvas_w + 1,
    )

    grid = np.zeros((grid_h, grid_w), dtype=int)
    r0, c0, r1, c1 = frame_bbox
    grid[r0, c0 : c1 + 1] = FRAME_COLOR
    grid[r1, c0 : c1 + 1] = FRAME_COLOR
    grid[r0 : r1 + 1, c0] = FRAME_COLOR
    grid[r0 : r1 + 1, c1] = FRAME_COLOR

    anchor_color = rng.choice(ANCHOR_COLORS)
    for ar, ac in anchors:
        grid[r0 + 1 + ar, c0 + 1 + ac] = anchor_color

    sr0, sc0 = _place_tile_sample(rng, (grid_h, grid_w), frame_bbox, tile_mask)
    for tr, tc in np.argwhere(tile_mask):
        grid[sr0 + int(tr), sc0 + int(tc)] = anchor_color

    out = _apply_rule(grid, tile_mask, anchor_color, offset)
    if not np.array_equal(main(grid), out):
        return None
    return grid, out


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        case = _generate_case(rng)
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid input")
