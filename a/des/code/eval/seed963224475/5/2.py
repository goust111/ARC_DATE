from __future__ import annotations

import random
from collections import Counter

import numpy as np


SIDES = ("top", "right", "bottom", "left")
CW_SIDE = {
    "top": "right",
    "right": "bottom",
    "bottom": "left",
    "left": "top",
}

BG = 0
FRAME_COLOR = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]

GRID_H_RANGE = (18, 28)
GRID_W_RANGE = (28, 37)
TEMPLATE_H_RANGE = (12, 19)
TEMPLATE_W_MIN = 10
TEMPLATE_W_MAX = 15
SLOT_COUNT_RANGE = (2, 4)
SLOT_DIM_RANGE = (2, 5)
GEN_ATTEMPTS = 400


def _most_common(values: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(values, dtype=int), return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                if rr > 0 and mask[rr - 1, cc] and not seen[rr - 1, cc]:
                    seen[rr - 1, cc] = True
                    stack.append((rr - 1, cc))
                if rr + 1 < h and mask[rr + 1, cc] and not seen[rr + 1, cc]:
                    seen[rr + 1, cc] = True
                    stack.append((rr + 1, cc))
                if cc > 0 and mask[rr, cc - 1] and not seen[rr, cc - 1]:
                    seen[rr, cc - 1] = True
                    stack.append((rr, cc - 1))
                if cc + 1 < w and mask[rr, cc + 1] and not seen[rr, cc + 1]:
                    seen[rr, cc + 1] = True
                    stack.append((rr, cc + 1))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _side_line(arr: np.ndarray, side: str) -> np.ndarray:
    if side == "top":
        return arr[0, :]
    if side == "bottom":
        return arr[-1, :]
    if side == "left":
        return arr[:, 0]
    if side == "right":
        return arr[:, -1]
    raise ValueError(side)


def _without_side(arr: np.ndarray, side: str) -> np.ndarray:
    mask = np.ones(arr.shape, dtype=bool)
    if side == "top":
        mask[0, :] = False
    elif side == "bottom":
        mask[-1, :] = False
    elif side == "left":
        mask[:, 0] = False
    elif side == "right":
        mask[:, -1] = False
    else:
        raise ValueError(side)
    return arr[mask]


def _paint_side(arr: np.ndarray, side: str, color: int) -> np.ndarray:
    out = arr.copy()
    if side == "top":
        out[0, :] = color
    elif side == "bottom":
        out[-1, :] = color
    elif side == "left":
        out[:, 0] = color
    elif side == "right":
        out[:, -1] = color
    else:
        raise ValueError(side)
    return out


def _analyze_fragment(arr: np.ndarray, bg: int) -> dict[str, object] | None:
    arr = np.asarray(arr, dtype=int)
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
        return None
    if np.any(arr == bg):
        return None

    hits: list[tuple[str, int, int]] = []
    for side in SIDES:
        line = _side_line(arr, side)
        marker_color = int(line[0])
        if not np.all(line == marker_color):
            continue
        other = _without_side(arr, side)
        if other.size == 0 or np.any(other == marker_color):
            continue
        other_colors = np.unique(other)
        if len(other_colors) != 1:
            continue
        base_color = int(other_colors[0])
        if base_color == marker_color:
            continue
        hits.append((side, marker_color, base_color))

    if len(hits) != 1:
        return None

    marker_side, marker_color, base_color = hits[0]
    colors = set(map(int, np.unique(arr)))
    if colors != {marker_color, base_color}:
        return None

    return {
        "arr": arr.copy(),
        "shape": tuple(int(v) for v in arr.shape),
        "marker_side": marker_side,
        "marker_color": marker_color,
        "base_color": base_color,
    }


def _template_component(grid: np.ndarray, bg: int) -> tuple[np.ndarray, tuple[int, int, int, int], list[np.ndarray], int]:
    comps = _components(grid != bg)
    assert comps, "expected non-background objects"
    best_idx = max(range(len(comps)), key=lambda idx: len(comps[idx]))
    box = _bbox(comps[best_idx])
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy(), box, comps, best_idx


def _find_slots(crop: np.ndarray, bg: int, frame_color: int) -> list[dict[str, object]]:
    h, w = crop.shape
    slots: list[dict[str, object]] = []
    for coords in _components(crop == bg):
        r0, c0, r1, c1 = _bbox(coords)
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if len(coords) != area:
            continue

        open_sides = []
        if r0 == 0:
            open_sides.append("top")
        if c1 == w - 1:
            open_sides.append("right")
        if r1 == h - 1:
            open_sides.append("bottom")
        if c0 == 0:
            open_sides.append("left")
        if len(open_sides) != 1:
            continue

        side_colors: dict[str, int] = {}
        if r0 > 0:
            line = crop[r0 - 1, c0 : c1 + 1]
            if np.all(line == line[0]):
                side_colors["top"] = int(line[0])
        if c1 + 1 < w:
            line = crop[r0 : r1 + 1, c1 + 1]
            if np.all(line == line[0]):
                side_colors["right"] = int(line[0])
        if r1 + 1 < h:
            line = crop[r1 + 1, c0 : c1 + 1]
            if np.all(line == line[0]):
                side_colors["bottom"] = int(line[0])
        if c0 > 0:
            line = crop[r0 : r1 + 1, c0 - 1]
            if np.all(line == line[0]):
                side_colors["left"] = int(line[0])

        marker_sides = [
            (side, color)
            for side, color in side_colors.items()
            if color not in (bg, frame_color)
        ]
        if len(marker_sides) != 1:
            continue

        open_side = open_sides[0]
        marker_side, marker_color = marker_sides[0]
        if marker_side != CW_SIDE[open_side]:
            continue

        slots.append(
            {
                "bbox": (r0, c0, r1, c1),
                "shape": (r1 - r0 + 1, c1 - c0 + 1),
                "open_side": open_side,
                "marker_side": marker_side,
                "marker_color": marker_color,
            }
        )
    return slots


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _most_common(g.ravel())

    crop, _box, comps, template_idx = _template_component(g, bg)
    frame_color = _most_common(crop[crop != bg])
    slots = _find_slots(crop, bg, frame_color)
    assert slots, "template must contain open rectangular slots"

    fragments: dict[int, dict[str, object]] = {}
    for idx, coords in enumerate(comps):
        if idx == template_idx:
            continue
        r0, c0, r1, c1 = _bbox(coords)
        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        if len(coords) != sub.size:
            continue
        info = _analyze_fragment(sub, bg)
        if info is None:
            continue
        marker_color = int(info["marker_color"])
        assert marker_color not in fragments, "fragment marker colors must be unique"
        fragments[marker_color] = info

    slot_colors = {int(slot["marker_color"]) for slot in slots}
    assert slot_colors == set(fragments), "slots and fragments must pair one-to-one by color"

    out = crop.copy()
    for slot in slots:
        marker_color = int(slot["marker_color"])
        piece = fragments[marker_color]
        placed = False
        for k in range(4):
            rotated = np.rot90(np.asarray(piece["arr"], dtype=int), k)
            info = _analyze_fragment(rotated, bg)
            if info is None:
                continue
            if tuple(rotated.shape) != tuple(slot["shape"]):
                continue
            if info["marker_side"] != slot["marker_side"]:
                continue
            r0, c0, r1, c1 = slot["bbox"]
            out[r0 : r1 + 1, c0 : c1 + 1] = rotated
            placed = True
            break
        assert placed, f"no rotation fits slot color {marker_color}"

    return out


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _sample_slot_bbox(
    side: str,
    th: int,
    tw: int,
    rng: random.Random,
) -> tuple[int, int, int, int] | None:
    min_dim, max_dim = SLOT_DIM_RANGE
    if side in ("top", "bottom"):
        max_w = min(max_dim, tw - 3)
        max_h = min(max_dim, th - 1)
        if max_w < min_dim or max_h < min_dim:
            return None
        sw = rng.randint(min_dim, max_w)
        sh = rng.randint(min_dim, max_h)
        c0 = rng.randint(1, tw - sw - 2)
        if side == "top":
            return (0, c0, sh - 1, c0 + sw - 1)
        return (th - sh, c0, th - 1, c0 + sw - 1)

    max_h = min(max_dim, th - 3)
    max_w = min(max_dim, tw - 1)
    if max_h < min_dim or max_w < min_dim:
        return None
    sh = rng.randint(min_dim, max_h)
    sw = rng.randint(min_dim, max_w)
    r0 = rng.randint(1, th - sh - 2)
    if side == "left":
        return (r0, 0, r0 + sh - 1, sw - 1)
    return (r0, tw - sw, r0 + sh - 1, tw - 1)


def _slot_marker_line(crop: np.ndarray, bbox: tuple[int, int, int, int], open_side: str, marker_color: int) -> None:
    r0, c0, r1, c1 = bbox
    marker_side = CW_SIDE[open_side]
    if marker_side == "top":
        crop[r0 - 1, c0 : c1 + 1] = marker_color
    elif marker_side == "right":
        crop[r0 : r1 + 1, c1 + 1] = marker_color
    elif marker_side == "bottom":
        crop[r1 + 1, c0 : c1 + 1] = marker_color
    elif marker_side == "left":
        crop[r0 : r1 + 1, c0 - 1] = marker_color
    else:
        raise ValueError(marker_side)


def _make_piece(shape: tuple[int, int], base_color: int, marker_color: int, marker_side: str) -> np.ndarray:
    arr = np.full(shape, base_color, dtype=int)
    return _paint_side(arr, marker_side, marker_color)


def _can_place(used: np.ndarray, r: int, c: int, h: int, w: int, pad: int = 1) -> bool:
    hh, ww = used.shape
    if r < 0 or c < 0 or r + h > hh or c + w > ww:
        return False
    rr0 = max(0, r - pad)
    cc0 = max(0, c - pad)
    rr1 = min(hh, r + h + pad)
    cc1 = min(ww, c + w + pad)
    return not used[rr0:rr1, cc0:cc1].any()


def _mark_used(used: np.ndarray, r: int, c: int, h: int, w: int, pad: int = 1) -> None:
    hh, ww = used.shape
    rr0 = max(0, r - pad)
    cc0 = max(0, c - pad)
    rr1 = min(hh, r + h + pad)
    cc1 = min(ww, c + w + pad)
    used[rr0:rr1, cc0:cc1] = True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    gh = rng.randint(*GRID_H_RANGE)
    gw = rng.randint(*GRID_W_RANGE)

    tw_hi = min(TEMPLATE_W_MAX, gw // 2 - 2, gw - 14)
    if tw_hi < TEMPLATE_W_MIN:
        return None
    tw = rng.randint(TEMPLATE_W_MIN, tw_hi)

    th_lo = max(TEMPLATE_H_RANGE[0], 10)
    th_hi = min(TEMPLATE_H_RANGE[1], gh - 2)
    if th_hi < th_lo:
        return None
    th = rng.randint(th_lo, th_hi)

    top = rng.randint(1, gh - th - 1)
    left_hi = max(1, gw // 2 - tw)
    left = rng.randint(1, left_hi)

    n_slots = rng.randint(*SLOT_COUNT_RANGE)
    sides = rng.sample(list(SIDES), n_slots)

    slot_boxes: list[tuple[int, int, int, int]] = []
    for side in sides:
        placed = False
        for _ in range(80):
            bbox = _sample_slot_bbox(side, th, tw, rng)
            if bbox is None:
                return None
            if all(_boxes_separated(bbox, other, pad=1) for other in slot_boxes):
                slot_boxes.append(bbox)
                placed = True
                break
        if not placed:
            return None

    colors = rng.sample(COLOR_POOL, 2 * n_slots)
    marker_colors = colors[:n_slots]
    base_colors = colors[n_slots:]

    template = np.full((th, tw), FRAME_COLOR, dtype=int)
    slots: list[dict[str, object]] = []
    solved_pieces: list[np.ndarray] = []
    rotations = [rng.randrange(4) for _ in range(n_slots)]
    if all(rot == 0 for rot in rotations):
        rotations[rng.randrange(n_slots)] = rng.choice([1, 2, 3])

    for side, bbox, marker_color, base_color, shown_rot in zip(
        sides,
        slot_boxes,
        marker_colors,
        base_colors,
        rotations,
    ):
        r0, c0, r1, c1 = bbox
        template[r0 : r1 + 1, c0 : c1 + 1] = BG
        _slot_marker_line(template, bbox, side, marker_color)
        shape = (r1 - r0 + 1, c1 - c0 + 1)
        marker_side = CW_SIDE[side]
        solved = _make_piece(shape, base_color, marker_color, marker_side)
        shown = np.rot90(solved, shown_rot)
        solved_pieces.append(solved)
        slots.append(
            {
                "bbox": bbox,
                "shape": shape,
                "open_side": side,
                "marker_side": marker_side,
                "marker_color": marker_color,
                "shown": shown,
            }
        )

    grid = np.full((gh, gw), BG, dtype=int)
    grid[top : top + th, left : left + tw] = template

    used = np.zeros((gh, gw), dtype=bool)
    _mark_used(used, top, left, th, tw, pad=1)

    zone_c0 = max(gw // 2 + 1, left + tw + 2)
    if zone_c0 >= gw - 2:
        return None

    order = list(range(n_slots))
    rng.shuffle(order)
    for idx in order:
        frag = np.asarray(slots[idx]["shown"], dtype=int)
        fh, fw = frag.shape
        candidates: list[tuple[int, int]] = []
        for r in range(1, gh - fh):
            for c in range(zone_c0, gw - fw):
                if _can_place(used, r, c, fh, fw, pad=1):
                    candidates.append((r, c))
        if not candidates:
            return None
        r, c = rng.choice(candidates)
        grid[r : r + fh, c : c + fw] = frag
        _mark_used(used, r, c, fh, fw, pad=1)

    expected = template.copy()
    for slot, solved in zip(slots, solved_pieces):
        r0, c0, r1, c1 = slot["bbox"]
        expected[r0 : r1 + 1, c0 : c1 + 1] = solved

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
