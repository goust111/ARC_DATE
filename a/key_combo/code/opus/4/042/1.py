from __future__ import annotations

from collections import deque
import random

import numpy as np


BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7
BLACK = 0


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _components(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                nbrs = neighbors4 if connectivity == 4 else neighbors8
                for nr, nc in nbrs(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(pts, dtype=int))
    return out


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _rotate180(coords: np.ndarray, pivot: tuple[int, int]) -> np.ndarray:
    pr, pc = map(int, pivot)
    out = np.empty_like(coords)
    out[:, 0] = 2 * pr - coords[:, 0]
    out[:, 1] = 2 * pc - coords[:, 1]
    return out


def _exterior_background(obstacle: np.ndarray) -> np.ndarray:
    h, w = obstacle.shape
    bg = ~obstacle
    exterior = np.zeros_like(bg, dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in (0, w - 1):
            if bool(bg[r, c]) and not bool(exterior[r, c]):
                exterior[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if bool(bg[r, c]) and not bool(exterior[r, c]):
                exterior[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and bool(bg[nr, nc]) and not bool(exterior[nr, nc]):
                exterior[nr, nc] = True
                q.append((nr, nc))
    return exterior


def _enclosed_components(obstacle: np.ndarray) -> list[np.ndarray]:
    enclosed = (~obstacle) & (~_exterior_background(obstacle))
    return _components(enclosed, connectivity=4)


def _fill_color(area: int) -> int:
    if area <= 2:
        return GREEN
    if area <= 5:
        return YELLOW
    return ORANGE


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep the blue shape, add its 180-degree copy around the grey pivot, then fill the newly enclosed background regions by area.

    Input structure:
    - A black background contains exactly one blue 8-connected foreground shape and one grey pivot pixel.
    - The blue shape is asymmetric, has a visible indentation, and does not itself enclose a hole.
    - Output-only colors red, green, and orange are absent from the input.

    Rule:
    - Rotate every blue cell 180 degrees around the grey pivot and place a red copy at the mapped location while keeping the original blue cells.
    - If a mapped red cell lands on an original blue cell, keep that cell blue; keep the grey pivot unchanged.
    - Treat blue, red, and grey as walls, flood from all boundary black cells using 4-connectivity, and regard the unreached black cells as enclosed regions.
    - Split enclosed black cells into 4-connected regions and recolor each region by area: 1-2 to green, 3-5 to yellow, and 6+ to orange.

    Key insight:
    - The holes do not exist in the input alone; they appear only after the original shape and its 180-degree copy coexist, so rotation must happen before flood-filling and area-based recoloring.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert int((g == GREY).sum()) == 1, "expected exactly one pivot"
    assert int((g == BLUE).sum()) >= 1, "expected blue foreground"

    pivot = tuple(map(int, np.argwhere(g == GREY)[0]))
    blue = np.argwhere(g == BLUE)
    red = _rotate180(blue, pivot)
    h, w = g.shape
    assert np.all((0 <= red[:, 0]) & (red[:, 0] < h) & (0 <= red[:, 1]) & (red[:, 1] < w)), "rotated copy left the grid"

    out = g.copy()
    for r, c in red:
        rr = int(r)
        cc = int(c)
        if out[rr, cc] in (BLUE, GREY):
            continue
        out[rr, cc] = RED

    obstacle = out != BLACK
    for comp in _enclosed_components(obstacle):
        color = _fill_color(int(comp.shape[0]))
        out[comp[:, 0], comp[:, 1]] = color
    return out


def _comb_mask(h: int, w: int, row_a: int, row_b: int, len_top: int, len_mid: int, len_low: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[:, 0] = True
    mask[0, :len_top] = True
    mask[row_a, :len_mid] = True
    mask[row_b, :len_low] = True
    return mask


def _spiral_mask(h: int, w: int, top_len: int, right_height: int, low_row: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :top_len] = True
    mask[:right_height, top_len - 1] = True
    mask[right_height - 1, 1:top_len] = True
    mask[2:right_height, 1] = True
    mask[low_row, 1 : w - 1] = True
    return mask


def _mask_without(mask: np.ndarray, *cells: tuple[int, int]) -> np.ndarray:
    out = mask.copy()
    for r, c in cells:
        out[int(r), int(c)] = False
    return out


def _tight_canvas(mask: np.ndarray, pivot: tuple[int, int]) -> tuple[np.ndarray, tuple[int, int]]:
    pr, pc = map(int, pivot)
    pts = np.argwhere(mask)
    mins = np.min(np.vstack([pts, np.array([[pr, pc]], dtype=int)]), axis=0)
    maxs = np.max(np.vstack([pts, np.array([[pr, pc]], dtype=int)]), axis=0)
    r0, c0 = map(int, mins)
    r1, c1 = map(int, maxs)
    canvas = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    canvas[pts[:, 0] - r0, pts[:, 1] - c0] = BLUE
    canvas[pr - r0, pc - c0] = GREY
    return canvas, (pr - r0, pc - c0)


def _orient(mask: np.ndarray, pivot: tuple[int, int], op: int) -> tuple[np.ndarray, tuple[int, int]]:
    canvas, _ = _tight_canvas(mask, pivot)
    arr = canvas
    if op >= 4:
        arr = np.fliplr(arr)
    arr = np.rot90(arr, op % 4)
    new_mask = arr == BLUE
    new_pivot = tuple(map(int, np.argwhere(arr == GREY)[0]))
    return new_mask, new_pivot


def _normalize_layout(mask: np.ndarray, pivot: tuple[int, int]) -> tuple[np.ndarray, tuple[int, int], np.ndarray]:
    blue = np.argwhere(mask)
    red = _rotate180(blue, pivot)
    all_pts = np.vstack([blue, red, np.array([pivot], dtype=int)])
    mins = all_pts.min(axis=0)
    shift = -mins
    blue2 = blue + shift
    red2 = red + shift
    pivot2 = (int(pivot[0] + shift[0]), int(pivot[1] + shift[1]))
    h = int(all_pts[:, 0].max() + shift[0] + 1)
    w = int(all_pts[:, 1].max() + shift[1] + 1)
    norm = np.zeros((h, w), dtype=bool)
    norm[blue2[:, 0], blue2[:, 1]] = True
    return norm, pivot2, red2


def _has_hole(mask: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(mask)
    crop = mask[r0 : r1 + 1, c0 : c1 + 1]
    exterior = _exterior_background(crop)
    holes = (~crop) & (~exterior)
    return bool(holes.any())


def _has_indentation(mask: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(mask)
    crop = mask[r0 : r1 + 1, c0 : c1 + 1]
    h, w = crop.shape
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if bool(crop[r, c]):
                continue
            touch = 0
            for nr, nc in neighbors4(r, c):
                touch += int(crop[nr, nc])
            if touch >= 2:
                return True
    return False


def _validate_shape(mask: np.ndarray) -> None:
    assert 10 <= int(mask.sum()) <= 18
    assert len(_components(mask, connectivity=8)) == 1
    assert not _has_hole(mask)
    assert _has_indentation(mask)
    assert not np.array_equal(mask, np.rot90(mask, 2))


def _analyze_case(inp: np.ndarray) -> dict:
    pivot = tuple(map(int, np.argwhere(inp == GREY)[0]))
    blue = np.argwhere(inp == BLUE)
    red = _rotate180(blue, pivot)
    blue_set = {(int(r), int(c)) for r, c in blue}
    obstacle_in = (inp == BLUE) | (inp == GREY)
    obstacle_out = obstacle_in.copy()
    obstacle_out[red[:, 0], red[:, 1]] = True
    comps = _enclosed_components(obstacle_out)
    areas = sorted(int(comp.shape[0]) for comp in comps)
    cats = {_fill_color(area) for area in areas}
    return {
        "overlap": int(sum((int(r), int(c)) in blue_set for r, c in red)),
        "input_holes": len(_enclosed_components(obstacle_in)),
        "areas": areas,
        "cats": cats,
    }


_TEMPLATES = [
    {
        "name": "comb_green_yellow_a",
        "mask": _comb_mask(7, 7, 4, 6, 7, 4, 2),
        "pivot": (3, 2),
        "weight": 3,
    },
    {
        "name": "comb_green_yellow_b",
        "mask": _comb_mask(8, 8, 4, 7, 3, 6, 4),
        "pivot": (3, 2),
        "weight": 2,
    },
    {
        "name": "comb_yellow_orange",
        "mask": _comb_mask(8, 5, 2, 6, 4, 5, 4),
        "pivot": (4, 2),
        "weight": 2,
    },
    {
        "name": "spiral_green_orange",
        "mask": _spiral_mask(7, 6, 6, 6, 5),
        "pivot": (1, 3),
        "weight": 3,
    },
]

for _template in _TEMPLATES:
    _validate_shape(_template["mask"])


def _render_input(template: dict, op: int, height: int, width: int, top: int, left: int) -> np.ndarray:
    mask, pivot = _orient(template["mask"], template["pivot"], op)
    norm_mask, norm_pivot, _red = _normalize_layout(mask, pivot)
    inp = np.zeros((height, width), dtype=int)
    blue = np.argwhere(norm_mask)
    inp[top + blue[:, 0], left + blue[:, 1]] = BLUE
    inp[top + norm_pivot[0], left + norm_pivot[1]] = GREY
    return inp


def _gen_attempt() -> np.ndarray | None:
    template = random.choices(_TEMPLATES, weights=[t["weight"] for t in _TEMPLATES], k=1)[0]
    op = random.randrange(8)
    mask, pivot = _orient(template["mask"], template["pivot"], op)
    norm_mask, norm_pivot, _red = _normalize_layout(mask, pivot)
    base_h, base_w = norm_mask.shape

    min_h = max(12, base_h + 2)
    min_w = max(12, base_w + 2)
    max_h = min(18, base_h + 7)
    max_w = min(18, base_w + 7)
    if min_h > max_h or min_w > max_w:
        return None

    height = random.randint(min_h, max_h)
    width = random.randint(min_w, max_w)
    top = random.randint(1, height - base_h - 1)
    left = random.randint(1, width - base_w - 1)
    inp = _render_input(template, op, height, width, top, left)

    blue_mask = inp == BLUE
    _validate_shape(blue_mask)
    info = _analyze_case(inp)
    if info["input_holes"] != 0:
        return None
    if info["overlap"] <= 0 or info["overlap"] >= int(blue_mask.sum()):
        return None
    if len(info["areas"]) < 2 or len(set(info["areas"])) < 2:
        return None
    if len(info["cats"]) < 2:
        return None

    out = main(inp)
    present = set(map(int, out.flatten()))
    if not ({BLUE, RED, GREY} <= present):
        return None
    if present - {BLACK, BLUE, RED, GREEN, YELLOW, GREY, ORANGE}:
        return None
    for area in info["areas"]:
        if _fill_color(area) not in present:
            return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid input")
