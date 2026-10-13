from __future__ import annotations

from collections import deque
from typing import Iterable

import numpy as np


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _largest_non_bg_component(grid: np.ndarray, bg: int) -> dict[str, object]:
    best: dict[str, object] | None = None
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for coords in _components(grid == color):
            area = int(coords.shape[0])
            if best is None or area > int(best["area"]):
                best = {"color": color, "area": area, "coords": coords, "bbox": _bbox(coords)}
    if best is None:
        raise ValueError("expected at least one non-background component")
    return best


def _runs(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    runs = []
    start = prev = indices[0]
    for value in indices[1:]:
        if value == prev + 1:
            prev = value
            continue
        runs.append((start, prev))
        start = prev = value
    runs.append((start, prev))
    return runs


def _slot_side(local_bbox: tuple[int, int, int, int], crop_shape: tuple[int, int]) -> str:
    r0, c0, r1, c1 = local_bbox
    h, w = crop_shape
    sides = []
    if r0 == 0:
        sides.append("top")
    if r1 == h - 1:
        sides.append("bottom")
    if c0 == 0:
        sides.append("left")
    if c1 == w - 1:
        sides.append("right")
    if len(sides) != 1:
        raise ValueError("each slot must touch exactly one template side")
    return sides[0]


def _track_color_and_anchor(
    grid: np.ndarray,
    template_bbox: tuple[int, int, int, int],
    slot_bbox: tuple[int, int, int, int],
    side: str,
    bg: int,
) -> tuple[int, tuple[int, int]]:
    tr0, tc0, tr1, tc1 = template_bbox
    sr0, sc0, sr1, sc1 = slot_bbox

    if side == "top":
        cells = [(tr0 - 1, c) for c in range(sc0, sc1 + 1) if tr0 > 0 and int(grid[tr0 - 1, c]) != bg]
    elif side == "bottom":
        cells = [(tr1 + 1, c) for c in range(sc0, sc1 + 1) if tr1 + 1 < grid.shape[0] and int(grid[tr1 + 1, c]) != bg]
    elif side == "left":
        cells = [(r, tc0 - 1) for r in range(sr0, sr1 + 1) if tc0 > 0 and int(grid[r, tc0 - 1]) != bg]
    else:
        cells = [(r, tc1 + 1) for r in range(sr0, sr1 + 1) if tc1 + 1 < grid.shape[1] and int(grid[r, tc1 + 1]) != bg]

    colors = {int(grid[r, c]) for r, c in cells}
    if len(cells) != 1 or len(colors) != 1:
        raise ValueError("slot must connect to a single one-pixel rail")
    return colors.pop(), cells[0]


def _verify_piece_and_rail(
    grid: np.ndarray,
    color: int,
    template_bbox: tuple[int, int, int, int],
    slot_bbox: tuple[int, int, int, int],
    side: str,
    anchor: tuple[int, int],
) -> None:
    tr0, tc0, tr1, tc1 = template_bbox
    sr0, sc0, sr1, sc1 = slot_bbox
    slot_h = sr1 - sr0 + 1
    slot_w = sc1 - sc0 + 1
    ar, ac = anchor

    if side == "top":
        full_rows = [r for r in range(0, tr0) if np.all(grid[r, sc0 : sc1 + 1] == color)]
        candidates = [(a, b) for a, b in _runs(full_rows) if b - a + 1 == slot_h]
        if len(candidates) != 1:
            raise ValueError("expected one matching piece above the template")
        pr0, pr1 = candidates[0]
        for r in range(pr1 + 1, tr0):
            row = grid[r, sc0 : sc1 + 1]
            if int(row[ac - sc0]) != color or int(np.count_nonzero(row == color)) != 1:
                raise ValueError("invalid vertical rail structure")
    elif side == "bottom":
        full_rows = [r for r in range(tr1 + 1, grid.shape[0]) if np.all(grid[r, sc0 : sc1 + 1] == color)]
        candidates = [(a, b) for a, b in _runs(full_rows) if b - a + 1 == slot_h]
        if len(candidates) != 1:
            raise ValueError("expected one matching piece below the template")
        pr0, _ = candidates[0]
        for r in range(tr1 + 1, pr0):
            row = grid[r, sc0 : sc1 + 1]
            if int(row[ac - sc0]) != color or int(np.count_nonzero(row == color)) != 1:
                raise ValueError("invalid vertical rail structure")
    elif side == "left":
        full_cols = [c for c in range(0, tc0) if np.all(grid[sr0 : sr1 + 1, c] == color)]
        candidates = [(a, b) for a, b in _runs(full_cols) if b - a + 1 == slot_w]
        if len(candidates) != 1:
            raise ValueError("expected one matching piece left of the template")
        _, pc1 = candidates[0]
        for c in range(pc1 + 1, tc0):
            col = grid[sr0 : sr1 + 1, c]
            if int(col[ar - sr0]) != color or int(np.count_nonzero(col == color)) != 1:
                raise ValueError("invalid horizontal rail structure")
    else:
        full_cols = [c for c in range(tc1 + 1, grid.shape[1]) if np.all(grid[sr0 : sr1 + 1, c] == color)]
        candidates = [(a, b) for a, b in _runs(full_cols) if b - a + 1 == slot_w]
        if len(candidates) != 1:
            raise ValueError("expected one matching piece right of the template")
        pc0, _ = candidates[0]
        for c in range(tc1 + 1, pc0):
            col = grid[sr0 : sr1 + 1, c]
            if int(col[ar - sr0]) != color or int(np.count_nonzero(col == color)) != 1:
                raise ValueError("invalid horizontal rail structure")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide colored fragments along one-pixel rails into the matching notches of a central template.
    """
    g = np.asarray(grid, dtype=int)
    bg = _mode_color(g)

    template = _largest_non_bg_component(g, bg)
    template_color = int(template["color"])
    template_coords = template["coords"]
    template_bbox = template["bbox"]
    tr0, tc0, tr1, tc1 = template_bbox

    out = np.full_like(g, bg)
    out[template_coords[:, 0], template_coords[:, 1]] = template_color

    crop = g[tr0 : tr1 + 1, tc0 : tc1 + 1]
    slot_mask = crop == bg
    for slot_coords in _components(slot_mask):
        local_bbox = _bbox(slot_coords)
        lr0, lc0, lr1, lc1 = local_bbox
        if slot_coords.shape[0] != (lr1 - lr0 + 1) * (lc1 - lc0 + 1):
            raise ValueError("slot must be rectangular")
        side = _slot_side(local_bbox, crop.shape)
        slot_bbox = (tr0 + lr0, tc0 + lc0, tr0 + lr1, tc0 + lc1)
        color, anchor = _track_color_and_anchor(g, template_bbox, slot_bbox, side, bg)
        _verify_piece_and_rail(g, color, template_bbox, slot_bbox, side, anchor)
        sr0, sc0, sr1, sc1 = slot_bbox
        out[sr0 : sr1 + 1, sc0 : sc1 + 1] = color

    return out


BACKGROUND = 0
TEMPLATE_COLOR = 5
PIECE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
SIDES = ("top", "bottom", "left", "right")
SHAPE_POOL = ((2, 2), (2, 3), (3, 2), (3, 3), (2, 4), (4, 2), (3, 4), (4, 3))
TEMPLATE_H_RANGE = (10, 17)
TEMPLATE_W_RANGE = (10, 17)
SIDE_COUNT_RANGE = (2, 5)
SLOT_EXTRA_RANGE = (0, 2)
CLEARANCE_IDLE_RANGE = (3, 7)
CLEARANCE_EXTRA_RANGE = (1, 5)
GEN_ATTEMPTS = 300


def _randint(rng: np.random.Generator, lo: int, hi: int) -> int:
    return int(rng.integers(lo, hi))


def _choose_sides(rng: np.random.Generator) -> list[str]:
    side_count = _randint(rng, SIDE_COUNT_RANGE[0], SIDE_COUNT_RANGE[1])
    chosen = list(rng.choice(SIDES, size=side_count, replace=False))
    total_slots = side_count + _randint(rng, SLOT_EXTRA_RANGE[0], SLOT_EXTRA_RANGE[1] + 1)
    while len(chosen) < total_slots:
        chosen.append(str(rng.choice(chosen)))
    rng.shuffle(chosen)
    return chosen


def _choose_shapes(rng: np.random.Generator, count: int) -> list[tuple[int, int]]:
    shapes = [tuple(map(int, SHAPE_POOL[_randint(rng, 0, len(SHAPE_POOL))])) for _ in range(count)]
    if count >= 2 and float(rng.random()) < 0.65:
        dup_shape = tuple(map(int, SHAPE_POOL[_randint(rng, 0, len(SHAPE_POOL))]))
        idxs = rng.choice(count, size=2, replace=False)
        shapes[int(idxs[0])] = dup_shape
        shapes[int(idxs[1])] = dup_shape
    return shapes


def _slot_clearance_need(slots: list[dict[str, int]], side: str) -> int:
    if side in ("top", "bottom"):
        vals = [slot["h"] for slot in slots if slot["side"] == side]
    else:
        vals = [slot["w"] for slot in slots if slot["side"] == side]
    return max(vals, default=0) + 1


def _place_slots(
    rng: np.random.Generator, th: int, tw: int, sides: list[str], shapes: list[tuple[int, int]]
) -> list[dict[str, int]] | None:
    occupied = np.zeros((th, tw), dtype=bool)
    slots: list[dict[str, int]] = []
    for side, (h, w) in zip(sides, shapes):
        if h >= th - 1 or w >= tw - 1:
            return None
        placed = False
        for _ in range(120):
            if side == "top":
                if tw - w <= 1:
                    return None
                r0, c0 = 0, _randint(rng, 1, tw - w)
            elif side == "bottom":
                if tw - w <= 1:
                    return None
                r0, c0 = th - h, _randint(rng, 1, tw - w)
            elif side == "left":
                if th - h <= 1:
                    return None
                r0, c0 = _randint(rng, 1, th - h), 0
            else:
                if th - h <= 1:
                    return None
                r0, c0 = _randint(rng, 1, th - h), tw - w
            r1, c1 = r0 + h - 1, c0 + w - 1
            pr0, pc0 = max(0, r0 - 1), max(0, c0 - 1)
            pr1, pc1 = min(th - 1, r1 + 1), min(tw - 1, c1 + 1)
            if occupied[pr0 : pr1 + 1, pc0 : pc1 + 1].any():
                continue
            occupied[r0 : r1 + 1, c0 : c1 + 1] = True
            if side in ("top", "bottom"):
                track_offset = _randint(rng, 0, w)
            else:
                track_offset = _randint(rng, 0, h)
            slots.append(
                {
                    "side": side,
                    "r0": r0,
                    "c0": c0,
                    "r1": r1,
                    "c1": c1,
                    "h": h,
                    "w": w,
                    "track_offset": track_offset,
                }
            )
            placed = True
            break
        if not placed:
            return None
    return slots


def _paint_rect(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = color


def _external_geometry(
    slot: dict[str, int], template_bbox: tuple[int, int, int, int], gap: int
) -> tuple[tuple[int, int, int, int], list[tuple[int, int]]]:
    tr0, tc0, tr1, tc1 = template_bbox
    sr0 = tr0 + slot["r0"]
    sc0 = tc0 + slot["c0"]
    sr1 = tr0 + slot["r1"]
    sc1 = tc0 + slot["c1"]
    side = slot["side"]
    h, w = slot["h"], slot["w"]

    if side == "top":
        piece_r1 = tr0 - gap - 1
        piece_r0 = piece_r1 - h + 1
        piece_bbox = (piece_r0, sc0, piece_r1, sc1)
        track_col = sc0 + slot["track_offset"]
        rail = [(r, track_col) for r in range(piece_r1 + 1, tr0)]
    elif side == "bottom":
        piece_r0 = tr1 + gap + 1
        piece_r1 = piece_r0 + h - 1
        piece_bbox = (piece_r0, sc0, piece_r1, sc1)
        track_col = sc0 + slot["track_offset"]
        rail = [(r, track_col) for r in range(tr1 + 1, piece_r0)]
    elif side == "left":
        piece_c1 = tc0 - gap - 1
        piece_c0 = piece_c1 - w + 1
        piece_bbox = (sr0, piece_c0, sr1, piece_c1)
        track_row = sr0 + slot["track_offset"]
        rail = [(track_row, c) for c in range(piece_c1 + 1, tc0)]
    else:
        piece_c0 = tc1 + gap + 1
        piece_c1 = piece_c0 + w - 1
        piece_bbox = (sr0, piece_c0, sr1, piece_c1)
        track_row = sr0 + slot["track_offset"]
        rail = [(track_row, c) for c in range(tc1 + 1, piece_c0)]

    return piece_bbox, rail


def _gen_attempt(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    th = _randint(rng, TEMPLATE_H_RANGE[0], TEMPLATE_H_RANGE[1])
    tw = _randint(rng, TEMPLATE_W_RANGE[0], TEMPLATE_W_RANGE[1])
    sides = _choose_sides(rng)
    shapes = _choose_shapes(rng, len(sides))
    slots = _place_slots(rng, th, tw, sides, shapes)
    if slots is None:
        return None

    clearances = {
        side: _slot_clearance_need(slots, side) + _randint(rng, CLEARANCE_EXTRA_RANGE[0], CLEARANCE_EXTRA_RANGE[1])
        if any(slot["side"] == side for slot in slots)
        else _randint(rng, CLEARANCE_IDLE_RANGE[0], CLEARANCE_IDLE_RANGE[1])
        for side in SIDES
    }

    h = clearances["top"] + th + clearances["bottom"]
    w = clearances["left"] + tw + clearances["right"]
    tr0, tc0 = clearances["top"], clearances["left"]
    tr1, tc1 = tr0 + th - 1, tc0 + tw - 1
    template_bbox = (tr0, tc0, tr1, tc1)

    grid = np.full((h, w), BACKGROUND, dtype=int)
    _paint_rect(grid, template_bbox, TEMPLATE_COLOR)

    out = np.full_like(grid, BACKGROUND)
    _paint_rect(out, template_bbox, TEMPLATE_COLOR)

    colors = list(rng.choice(PIECE_COLORS, size=len(slots), replace=False))
    for slot, color in zip(slots, colors):
        sr0 = tr0 + slot["r0"]
        sc0 = tc0 + slot["c0"]
        sr1 = tr0 + slot["r1"]
        sc1 = tc0 + slot["c1"]
        slot_bbox = (sr0, sc0, sr1, sc1)
        _paint_rect(grid, slot_bbox, BACKGROUND)
        _paint_rect(out, slot_bbox, int(color))

        if slot["side"] in ("top", "bottom"):
            max_gap = clearances[slot["side"]] - slot["h"]
        else:
            max_gap = clearances[slot["side"]] - slot["w"]
        if max_gap < 1:
            return None
        gap = _randint(rng, 1, max_gap + 1)
        piece_bbox, rail = _external_geometry(slot, template_bbox, gap)
        pr0, pc0, pr1, pc1 = piece_bbox
        if pr0 < 0 or pc0 < 0 or pr1 >= h or pc1 >= w:
            return None
        _paint_rect(grid, piece_bbox, int(color))
        for rr, cc in rail:
            grid[rr, cc] = int(color)

    return grid, out


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        sample = _gen_attempt(rng)
        if sample is None:
            continue
        inp, out = sample
        if np.array_equal(main(inp), out):
            return inp
    raise RuntimeError("failed to generate a valid puzzle instance")
