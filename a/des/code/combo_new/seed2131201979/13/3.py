from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
FRAME_COLOR = 5
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDE_ORDER = ["top", "bottom", "left", "right"]
MATCHINGS = [
    {"top": "TL", "right": "TR", "left": "BL", "bottom": "BR"},
    {"left": "TL", "top": "TR", "bottom": "BL", "right": "BR"},
]
OBJECT_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
_RNG = random.Random()


def _bg_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(grid), return_counts=True)
    return int(vals[counts.argmax()])


def _mono_components(grid: np.ndarray, ignore: set[int]) -> list[tuple[int, list[tuple[int, int]]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[tuple[int, list[tuple[int, int]]]] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color in ignore:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            pts = [(r, c)]
            while q:
                cr, cc = q.popleft()
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
                        pts.append((nr, nc))
            comps.append((color, pts))
    return comps


def _frame_candidate(
    grid: np.ndarray, color: int, pts: list[tuple[int, int]]
) -> tuple[int, int, int, int, int] | None:
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 5 or w < 5 or h % 2 == 0 or w % 2 == 0:
        return None
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    mask = sub == color
    rr = h // 2
    cc = w // 2
    expected = np.zeros((h, w), dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    expected[rr, :] = True
    expected[:, cc] = True
    if not np.array_equal(mask, expected):
        return None
    return r0, c0, r1, c1, color


def _find_frame(grid: np.ndarray) -> tuple[int, int, int, int, int]:
    bg = _bg_color(grid)
    candidates = []
    for color, pts in _mono_components(grid, {bg}):
        cand = _frame_candidate(grid, color, pts)
        if cand is not None:
            candidates.append(cand)
    assert len(candidates) == 1, "expected exactly one grey reference frame"
    return candidates[0]


def _component_bbox(pts: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_mask(pts: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _component_bbox(pts)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in pts:
        mask[r - r0, c - c0] = True
    return mask


def _place_object(
    out: np.ndarray,
    mask: np.ndarray,
    color: int,
    side: str,
    bbox: tuple[int, int, int, int],
    frame: tuple[int, int, int, int, int],
    bg: int,
) -> None:
    fr0, fc0, fr1, fc1, _ = frame
    r0, c0, r1, c1 = bbox
    h, w = mask.shape
    if side == "top":
        dst_r0 = fr0 + 1
        dst_c0 = c0
        patch = np.flipud(mask)
    elif side == "bottom":
        dst_r0 = fr1 - h
        dst_c0 = c0
        patch = np.flipud(mask)
    elif side == "left":
        dst_r0 = r0
        dst_c0 = fc0 + 1
        patch = np.fliplr(mask)
    else:
        dst_r0 = r0
        dst_c0 = fc1 - w
        patch = np.fliplr(mask)

    dst_r1 = dst_r0 + h - 1
    dst_c1 = dst_c0 + w - 1
    assert fr0 < dst_r0 <= dst_r1 < fr1
    assert fc0 < dst_c0 <= dst_c1 < fc1
    target = out[dst_r0 : dst_r1 + 1, dst_c0 : dst_c1 + 1]
    assert np.all(target[patch] == bg), "moved object overlaps the frame or another object"
    target[patch] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reflect each outer monochrome object into the matching quadrant of the central grey frame.

    Input structure:
    The background is black. One connected grey component forms a hollow rectangular frame
    whose center row and center column are also grey, splitting the interior into four equal
    quadrants. Outside the frame, each of the four cardinal directions may contain at most one
    monochrome 4-connected object, separated from the frame by at least one empty row or column.

    Rule:
    Clear every non-grey object from outside the frame. An object above or below the frame is
    flipped vertically and moved inside so it touches the corresponding inner top or bottom wall,
    while keeping its column span unchanged. An object left or right of the frame is flipped
    horizontally and moved inside so it touches the corresponding inner left or right wall,
    while keeping its row span unchanged.

    Key insight:
    The source side determines the reflection axis, and the unchanged orthogonal coordinates
    preserve which half of the frame the object enters.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _bg_color(g)
    frame = _find_frame(g)
    fr0, fc0, fr1, fc1, frame_color = frame

    out = np.full_like(g, bg)
    rr = fr1 - fr0 + 1
    cc = fc1 - fc0 + 1
    local = np.zeros((rr, cc), dtype=bool)
    local[0, :] = True
    local[-1, :] = True
    local[:, 0] = True
    local[:, -1] = True
    local[rr // 2, :] = True
    local[:, cc // 2] = True
    out[fr0 : fr1 + 1, fc0 : fc1 + 1][local] = frame_color

    for color, pts in _mono_components(g, {bg, frame_color}):
        r0, c0, r1, c1 = _component_bbox(pts)
        if r1 < fr0:
            side = "top"
        elif r0 > fr1:
            side = "bottom"
        elif c1 < fc0:
            side = "left"
        elif c0 > fc1:
            side = "right"
        else:
            raise AssertionError("object must lie completely outside the frame")
        _place_object(out, _crop_mask(pts), color, side, (r0, c0, r1, c1), frame, bg)
    return out


def _random_mask(max_h: int, max_w: int, side: str, rng: random.Random) -> np.ndarray | None:
    axis = "ud" if side in {"top", "bottom"} else "lr"
    for _ in range(200):
        h = rng.randint(2, max_h)
        w = rng.randint(2, max_w)
        target = rng.randint(max(3, (h + w) // 2), h * w)
        start = (rng.randrange(h), rng.randrange(w))
        cells = {start}
        while len(cells) < target:
            frontier = []
            for r, c in cells:
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(rng.choice(frontier))
        if len(cells) != target:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        coords = np.argwhere(mask)
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        mask = mask[r0 : r1 + 1, c0 : c1 + 1]
        if min(mask.shape) < 2:
            continue
        if axis == "ud" and np.array_equal(mask, np.flipud(mask)):
            continue
        if axis == "lr" and np.array_equal(mask, np.fliplr(mask)):
            continue
        return mask
    return None


def _inside_bbox(
    side: str, quad: str, mask: np.ndarray, qh: int, qw: int, rng: random.Random
) -> tuple[int, int, int, int]:
    h, w = mask.shape
    frame_h = 2 * qh + 3
    frame_w = 2 * qw + 3
    if side == "top":
        r0 = 1
        if quad == "TL":
            c0 = rng.randint(1, qw - w + 1)
        else:
            c0 = rng.randint(qw + 2, frame_w - 1 - w)
    elif side == "bottom":
        r0 = frame_h - 1 - h
        if quad == "BL":
            c0 = rng.randint(1, qw - w + 1)
        else:
            c0 = rng.randint(qw + 2, frame_w - 1 - w)
    elif side == "left":
        c0 = 1
        if quad == "TL":
            r0 = rng.randint(1, qh - h + 1)
        else:
            r0 = rng.randint(qh + 2, frame_h - 1 - h)
    else:
        c0 = frame_w - 1 - w
        if quad == "TR":
            r0 = rng.randint(1, qh - h + 1)
        else:
            r0 = rng.randint(qh + 2, frame_h - 1 - h)
    return r0, c0, r0 + h - 1, c0 + w - 1


def _draw_frame(grid: np.ndarray, r0: int, c0: int, qh: int, qw: int) -> None:
    h = 2 * qh + 3
    w = 2 * qw + 3
    sub = grid[r0 : r0 + h, c0 : c0 + w]
    sub[0, :] = FRAME_COLOR
    sub[-1, :] = FRAME_COLOR
    sub[:, 0] = FRAME_COLOR
    sub[:, -1] = FRAME_COLOR
    sub[h // 2, :] = FRAME_COLOR
    sub[:, w // 2] = FRAME_COLOR


def _paint_mask(grid: np.ndarray, r0: int, c0: int, mask: np.ndarray, color: int) -> None:
    sub = grid[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]]
    sub[mask] = int(color)


def _generate_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    qh = rng.randint(3, 5)
    qw = rng.randint(3, 5)
    frame_h = 2 * qh + 3
    frame_w = 2 * qw + 3

    matching = rng.choice(MATCHINGS)
    side_count = rng.choice([2, 3, 4])
    sides = rng.sample(SIDE_ORDER, side_count)
    colors = rng.sample(OBJECT_COLORS, side_count)

    records = []
    for side, color in zip(sides, colors):
        mask = _random_mask(qh, qw, side, rng)
        if mask is None:
            return None
        quad = matching[side]
        inside_bbox = _inside_bbox(side, quad, mask, qh, qw, rng)
        gap = rng.randint(2, 3)
        records.append(
            {
                "side": side,
                "color": color,
                "mask_out": mask,
                "mask_in": np.flipud(mask) if side in {"top", "bottom"} else np.fliplr(mask),
                "quad": quad,
                "inside_bbox": inside_bbox,
                "gap": gap,
            }
        )

    top_need = max((rec["gap"] + rec["mask_out"].shape[0] for rec in records if rec["side"] == "top"), default=0)
    bottom_need = max((rec["gap"] + rec["mask_out"].shape[0] for rec in records if rec["side"] == "bottom"), default=0)
    left_need = max((rec["gap"] + rec["mask_out"].shape[1] for rec in records if rec["side"] == "left"), default=0)
    right_need = max((rec["gap"] + rec["mask_out"].shape[1] for rec in records if rec["side"] == "right"), default=0)

    min_h = top_need + frame_h + bottom_need
    min_w = left_need + frame_w + right_need
    if min_h > 30 or min_w > 30:
        return None

    slack_h = min(30 - min_h, 4)
    slack_w = min(30 - min_w, 4)
    extra_top = rng.randint(0, slack_h)
    extra_bottom = rng.randint(0, slack_h - extra_top)
    extra_left = rng.randint(0, slack_w)
    extra_right = rng.randint(0, slack_w - extra_left)

    grid_h = min_h + extra_top + extra_bottom
    grid_w = min_w + extra_left + extra_right
    frame_r0 = top_need + extra_top
    frame_c0 = left_need + extra_left

    inp = np.full((grid_h, grid_w), BG, dtype=int)
    out = np.full((grid_h, grid_w), BG, dtype=int)
    _draw_frame(inp, frame_r0, frame_c0, qh, qw)
    _draw_frame(out, frame_r0, frame_c0, qh, qw)

    for rec in records:
        ir0, ic0, _, _ = rec["inside_bbox"]
        abs_ir0 = frame_r0 + ir0
        abs_ic0 = frame_c0 + ic0
        _paint_mask(out, abs_ir0, abs_ic0, rec["mask_in"], rec["color"])

        side = rec["side"]
        if side == "top":
            or0 = frame_r0 - rec["gap"] - rec["mask_out"].shape[0]
            oc0 = abs_ic0
        elif side == "bottom":
            or0 = frame_r0 + frame_h - 1 + rec["gap"]
            oc0 = abs_ic0
        elif side == "left":
            or0 = abs_ir0
            oc0 = frame_c0 - rec["gap"] - rec["mask_out"].shape[1]
        else:
            or0 = abs_ir0
            oc0 = frame_c0 + frame_w - 1 + rec["gap"]
        _paint_mask(inp, or0, oc0, rec["mask_out"], rec["color"])

    if np.array_equal(main(inp), out):
        return inp, out
    return None


def generate_input() -> np.ndarray:
    for _ in range(400):
        case = _generate_case(_RNG)
        if case is not None:
            return case[0]
    raise RuntimeError("generate_input failed after 400 attempts")
