from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
GEN_RETRIES = 400
OBJECT_RETRIES = 200
PLACEMENT_RETRIES = 200


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _find_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if seen[nr, nc] or int(grid[nr, nc]) == bg:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            colors = tuple(sorted({int(grid[rr, cc]) for rr, cc in coords}))
            comps.append(
                {
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "colors": colors,
                }
            )
    return comps


def _component_patch(grid: np.ndarray, comp: dict[str, object], bg: int) -> np.ndarray:
    r0, c0, r1, c1 = comp["bbox"]  # type: ignore[index]
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    coords = comp["coords"]  # type: ignore[index]
    for r, c in coords:
        patch[int(r) - r0, int(c) - c0] = int(grid[int(r), int(c)])
    return patch


def _trim_patch(patch: np.ndarray, bg: int) -> np.ndarray:
    pts = np.argwhere(patch != bg)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return patch[r0 : r1 + 1, c0 : c1 + 1].copy()


def _axis_candidate(
    patch: np.ndarray, bg: int, orient: str
) -> dict[str, object] | None:
    colors = tuple(int(x) for x in np.unique(patch[patch != bg]))
    if len(colors) != 2:
        return None

    h, w = patch.shape
    axis2 = w - 1 if orient == "v" else h - 1
    off_counts = {colors[0]: 0, colors[1]: 0}
    off_entries: list[tuple[int, int, int]] = []

    for r, c in np.argwhere(patch != bg):
        rr = int(r)
        cc = int(c)
        if orient == "v":
            mr, mc = rr, axis2 - cc
            if not (0 <= mc < w) or int(patch[mr, mc]) == bg:
                return None
            if 2 * cc != axis2 and int(patch[mr, mc]) == int(patch[rr, cc]):
                return None
            if 2 * cc < axis2:
                val = int(patch[rr, cc])
                off_counts[val] += 1
                off_entries.append((rr, cc, val))
        else:
            mr, mc = axis2 - rr, cc
            if not (0 <= mr < h) or int(patch[mr, mc]) == bg:
                return None
            if 2 * rr != axis2 and int(patch[mr, mc]) == int(patch[rr, cc]):
                return None
            if 2 * rr < axis2:
                val = int(patch[rr, cc])
                off_counts[val] += 1
                off_entries.append((rr, cc, val))

    if not off_entries:
        return None

    if off_counts[colors[0]] > off_counts[colors[1]]:
        static_color = colors[0]
    elif off_counts[colors[1]] > off_counts[colors[0]]:
        static_color = colors[1]
    else:
        static_color = min(off_entries)[2]
    shift_color = colors[1] if static_color == colors[0] else colors[0]

    return {
        "orient": orient,
        "axis2": int(axis2),
        "static_color": int(static_color),
        "shift_color": int(shift_color),
        "margin": int(abs(off_counts[colors[0]] - off_counts[colors[1]])),
        "off_count": int(len(off_entries)),
    }


def _component_rule_info(patch: np.ndarray, bg: int) -> dict[str, object]:
    candidates = []
    for orient in ("v", "h"):
        cand = _axis_candidate(patch, bg, orient)
        if cand is not None:
            candidates.append(cand)
    assert candidates, "component must have one anti-color symmetry axis"
    candidates.sort(
        key=lambda cand: (
            int(cand["margin"]),
            int(cand["off_count"]),
            1 if cand["orient"] == "v" else 0,
        ),
        reverse=True,
    )
    return candidates[0]


def _paint(out: np.ndarray, r: int, c: int, value: int, bg: int) -> None:
    h, w = out.shape
    assert 0 <= r < h and 0 <= c < w, "transformed pixels must stay in bounds"
    prev = int(out[r, c])
    assert prev in (bg, value), "transformed objects must not overlap inconsistently"
    out[r, c] = value


def _paint_transformed(
    out: np.ndarray,
    patch: np.ndarray,
    top: int,
    left: int,
    info: dict[str, object],
    bg: int,
) -> None:
    axis2 = int(info["axis2"])
    static_color = int(info["static_color"])
    shift_color = int(info["shift_color"])

    for r, c in np.argwhere(patch != bg):
        rr = int(r)
        cc = int(c)
        value = int(patch[rr, cc])
        gr = top + rr
        gc = left + cc
        if info["orient"] == "v":
            if 2 * cc < axis2:
                if value == static_color:
                    _paint(out, gr, gc, value, bg)
            elif 2 * cc > axis2:
                if value == shift_color:
                    _paint(out, gr, gc + 2, value, bg)
            else:
                _paint(out, gr, gc, value, bg)
                _paint(out, gr, gc + 2, value, bg)
        else:
            if 2 * rr < axis2:
                if value == static_color:
                    _paint(out, gr, gc, value, bg)
            elif 2 * rr > axis2:
                if value == shift_color:
                    _paint(out, gr + 2, gc, value, bg)
            else:
                _paint(out, gr, gc, value, bg)
                _paint(out, gr + 2, gc, value, bg)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split each two-color anti-symmetric object along its symmetry axis.

    The left/top half stays in place and keeps only its dominant off-axis color.
    The right/bottom half keeps the opposite color and shifts two cells away
    from the axis. Axis pixels, when present, belong to both halves.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    out = np.full_like(g, bg)

    for comp in _find_components(g, bg):
        patch = _component_patch(g, comp, bg)
        info = _component_rule_info(patch, bg)
        r0, c0, _, _ = comp["bbox"]  # type: ignore[index]
        _paint_transformed(out, patch, r0, c0, info, bg)
    return out


def _grow_connected(
    shape: tuple[int, int], target_area: int, required: list[tuple[int, int]]
) -> np.ndarray | None:
    h, w = shape
    if target_area < len(required) or target_area > h * w:
        return None

    occ = np.zeros((h, w), dtype=bool)
    frontier: set[tuple[int, int]] = set()
    for r, c in required:
        if not (0 <= r < h and 0 <= c < w):
            return None
        occ[r, c] = True

    for r, c in required:
        for nr, nc in _neighbors4(r, c, h, w):
            if not occ[nr, nc]:
                frontier.add((nr, nc))

    while int(occ.sum()) < target_area:
        if not frontier:
            return None
        r, c = random.choice(tuple(frontier))
        frontier.remove((r, c))
        occ[r, c] = True
        for nr, nc in _neighbors4(r, c, h, w):
            if not occ[nr, nc]:
                frontier.add((nr, nc))
    return occ


def _build_vertical_mask() -> np.ndarray | None:
    height = random.randint(4, 8)
    half_w = random.randint(2, 4)
    axis_present = random.randint(0, 1)
    reduced_w = half_w + axis_present
    bridge_row = random.randrange(height)

    required = [(bridge_row, half_w - 1)]
    if axis_present:
        required.append((bridge_row, half_w))

    total = height * reduced_w
    min_area = max(len(required) + 2, int(np.ceil(total * 0.35)))
    max_area = max(min_area, int(np.floor(total * 0.8)))

    for _ in range(OBJECT_RETRIES):
        target_area = random.randint(min_area, max_area)
        reduced = _grow_connected((height, reduced_w), target_area, required)
        if reduced is None:
            continue

        left_occ = reduced[:, :half_w]
        if int(left_occ.sum()) < 3 or int(left_occ.sum()) % 2 == 0:
            continue

        full = np.zeros((height, 2 * half_w + axis_present), dtype=bool)
        full[:, :half_w] = left_occ
        if axis_present:
            full[:, half_w] = reduced[:, half_w]
        full[:, half_w + axis_present :] = left_occ[:, ::-1]

        if int(full.sum()) >= 7:
            return full
    return None


def _color_vertical_mask(mask: np.ndarray, colors: tuple[int, int]) -> np.ndarray:
    a, b = colors
    h, w = mask.shape
    half_w = w // 2
    axis_present = w % 2
    phase = random.randint(0, 1)
    patch = np.full((h, w), BG, dtype=int)

    for r in range(h):
        for c in range(half_w):
            if not mask[r, c]:
                continue
            value = a if (r + c + phase) % 2 == 0 else b
            patch[r, c] = value
            patch[r, w - 1 - c] = b if value == a else a

    if axis_present:
        axis_col = half_w
        for r in range(h):
            if mask[r, axis_col]:
                patch[r, axis_col] = a if (r + axis_col + phase) % 2 == 0 else b
    return patch


def _make_object(orient: str, colors: tuple[int, int]) -> tuple[np.ndarray, dict[str, object]] | None:
    for _ in range(OBJECT_RETRIES):
        mask = _build_vertical_mask()
        if mask is None:
            continue
        patch = _color_vertical_mask(mask, colors)
        if orient == "h":
            patch = patch.T.copy()
        if random.random() < 0.5:
            patch = np.flipud(patch).copy()
        if random.random() < 0.5:
            patch = np.fliplr(patch).copy()
        patch = _trim_patch(patch, BG)

        comps = _find_components(patch, BG)
        if len(comps) != 1:
            continue

        info = _component_rule_info(patch, BG)
        if info["orient"] != orient:
            continue
        return patch, info
    return None


def _boxes_overlap(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _reserve_box(
    top: int, left: int, patch: np.ndarray, orient: str, h: int, w: int
) -> tuple[int, int, int, int]:
    ph, pw = patch.shape
    r1 = top + ph - 1 + (2 if orient == "h" else 0)
    c1 = left + pw - 1 + (2 if orient == "v" else 0)
    return (
        max(0, top - 1),
        max(0, left - 1),
        min(h - 1, r1 + 1),
        min(w - 1, c1 + 1),
    )


def _stamp_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    for r, c in np.argwhere(patch != BG):
        grid[top + int(r), left + int(c)] = int(patch[int(r), int(c)])


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(14, 24)
    w = random.randint(14, 24)
    grid = np.zeros((h, w), dtype=int)
    object_count = random.randint(1, 4)
    palette = random.sample(PALETTE, 2 * object_count)
    reserved: list[tuple[int, int, int, int]] = []

    for idx in range(object_count):
        orient = random.choice(("v", "h"))
        colors = (palette[2 * idx], palette[2 * idx + 1])
        made = _make_object(orient, colors)
        if made is None:
            return None
        patch, info = made
        ph, pw = patch.shape
        max_top = h - ph - (2 if orient == "h" else 0)
        max_left = w - pw - (2 if orient == "v" else 0)
        if max_top < 0 or max_left < 0:
            return None

        placed = False
        for _ in range(PLACEMENT_RETRIES):
            top = random.randint(0, max_top)
            left = random.randint(0, max_left)
            box = _reserve_box(top, left, patch, orient, h, w)
            if any(_boxes_overlap(box, other) for other in reserved):
                continue
            _stamp_patch(grid, patch, top, left)
            reserved.append(box)
            placed = True
            break
        if not placed:
            return None

    if int(np.count_nonzero(grid)) == 0:
        return None
    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid anti-symmetry input")
