from __future__ import annotations

import random
from typing import Iterable

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values = np.asarray(grid, dtype=int).ravel()
    counts = np.bincount(values, minlength=max(10, int(values.max()) + 1))
    return int(np.argmax(counts))


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r0 in range(h):
        for c0 in range(w):
            if visited[r0, c0] or int(g[r0, c0]) == bg:
                continue
            color = int(g[r0, c0])
            stack = [(r0, c0)]
            visited[r0, c0] = True
            coords: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                for nr, nc in _neighbors4(r, c):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if visited[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    stack.append((nr, nc))

            rows = [r for r, _ in coords]
            cols = [c for _, c in coords]
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                    "area": len(coords),
                }
            )
    return comps


def _component_patch(comp: dict, bg: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    for r, c in comp["coords"]:
        patch[int(r) - r0, int(c) - c0] = int(comp["color"])
    return patch, (r0, c0, r1, c1)


def _is_frame(comp: dict, grid: np.ndarray, bg: int) -> bool:
    color = int(comp["color"])
    r0, c0, r1, c1 = map(int, comp["bbox"])
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False

    patch = np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)
    if not np.all(patch[0, :] == color):
        return False
    if not np.all(patch[-1, :] == color):
        return False
    if not np.all(patch[:, 0] == color):
        return False
    if not np.all(patch[:, -1] == color):
        return False
    if not np.all(patch[1:-1, 1:-1] == bg):
        return False
    return int(comp["area"]) == int(2 * h + 2 * w - 4)


def _fragment_side(comp: dict, frame_bbox: tuple[int, int, int, int]) -> str:
    fr = np.mean([r for r, _ in comp["coords"]])
    fc = np.mean([c for _, c in comp["coords"]])
    r0, c0, r1, c1 = map(int, frame_bbox)
    cr = (r0 + r1) / 2.0
    cc = (c0 + c1) / 2.0
    dr = fr - cr
    dc = fc - cc
    if abs(dr) > abs(dc):
        return "top" if dr < 0 else "bottom"
    return "left" if dc < 0 else "right"


def _dock_origin(
    frame_bbox: tuple[int, int, int, int], frag_shape: tuple[int, int], side: str
) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, frame_bbox)
    sh, sw = map(int, frag_shape)
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    assert 1 <= sh <= inner_h and 1 <= sw <= inner_w

    if side == "top":
        return r0 + 1, c0 + 1 + (inner_w - sw) // 2
    if side == "bottom":
        return r1 - sh, c0 + 1 + (inner_w - sw) // 2
    if side == "left":
        return r0 + 1 + (inner_h - sh) // 2, c0 + 1
    if side == "right":
        return r0 + 1 + (inner_h - sh) // 2, c1 - sw
    raise AssertionError(f"unknown side: {side}")


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, transparent: int) -> None:
    h, w = patch.shape
    window = dst[r0 : r0 + h, c0 : c0 + w]
    mask = patch != int(transparent)
    window[mask] = patch[mask]


def main(grid: np.ndarray) -> np.ndarray:
    """Sort scattered fragments into same-color frames using their relative side.

    Input structure:
    - Background color `bg` is the most common color.
    - The grid contains several hollow rectangular frames, each drawn in a unique color, with 1-pixel borders and `bg` interiors.
    - Outside the frames are small same-color fragments, each a 1-3 pixel connected component matching exactly one frame color.

    Rule:
    - Detect the frame that matches each fragment by color.
    - Determine whether the fragment lies above, below, left, or right of that frame by comparing the fragment center with the frame center along the dominant offset axis.
    - Remove the original outside fragment and place the unchanged fragment shape inside the frame, flush against the corresponding inner edge and centered along the parallel axis.

    Key insight:
    - Color identifies the destination frame, while the fragment's original side relative to that frame determines which inner edge it docks to.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)

    comps = _find_components(g, bg)
    frames_by_color: dict[int, dict] = {}
    fragments: list[dict] = []
    for comp in comps:
        if _is_frame(comp, g, bg):
            frames_by_color[int(comp["color"])] = comp
        else:
            fragments.append(comp)

    out = g.copy()
    for frag in fragments:
        for r, c in frag["coords"]:
            out[int(r), int(c)] = int(bg)

    for frag in fragments:
        color = int(frag["color"])
        assert color in frames_by_color
        frame = frames_by_color[color]
        frag_patch, _ = _component_patch(frag, bg)
        side = _fragment_side(frag, frame["bbox"])
        dr, dc = _dock_origin(frame["bbox"], frag_patch.shape, side)
        _paste(out, frag_patch, dr, dc, transparent=bg)

    return out


# --------------- generator ---------------

BASE_CANVAS = 30
PACK_PADDING = 1
MAX_RETRIES = 400
PATCH_RETRIES = 120

FRAME_COUNT_RANGE = (2, 4)
INNER_HEIGHT_RANGE = (4, 6)
INNER_WIDTH_RANGE = (4, 6)
FRAGMENT_COUNT_RANGE = (2, 4)
SIDE_GAP_RANGE = (1, 3)
DIRECTIONS = ("top", "bottom", "left", "right")

FRAGMENT_SHAPES = [
    np.array([[1]], dtype=bool),
    np.array([[1, 1]], dtype=bool),
    np.array([[1], [1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1], [1], [1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[0, 1], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
]


def _dilate4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _draw_shape(canvas: np.ndarray, shape: np.ndarray, r0: int, c0: int, color: int) -> None:
    h, w = shape.shape
    view = canvas[r0 : r0 + h, c0 : c0 + w]
    view[shape] = int(color)


def _draw_frame(canvas: np.ndarray, r0: int, c0: int, h: int, w: int, color: int) -> None:
    canvas[r0, c0 : c0 + w] = int(color)
    canvas[r0 + h - 1, c0 : c0 + w] = int(color)
    canvas[r0 : r0 + h, c0] = int(color)
    canvas[r0 : r0 + h, c0 + w - 1] = int(color)


def _inner_dock_origin(side: str, shape: np.ndarray, inner_h: int, inner_w: int) -> tuple[int, int]:
    sh, sw = shape.shape
    if side == "top":
        return 0, (inner_w - sw) // 2
    if side == "bottom":
        return inner_h - sh, (inner_w - sw) // 2
    if side == "left":
        return (inner_h - sh) // 2, 0
    if side == "right":
        return (inner_h - sh) // 2, inner_w - sw
    raise AssertionError(f"unknown side: {side}")


def _choose_fragment_layout(inner_h: int, inner_w: int, sides: list[str]) -> dict[str, dict] | None:
    for _ in range(PATCH_RETRIES):
        order = sides[:]
        random.shuffle(order)
        occ = np.zeros((inner_h, inner_w), dtype=bool)
        chosen: dict[str, dict] = {}
        ok = True
        for side in order:
            shapes = FRAGMENT_SHAPES[:]
            random.shuffle(shapes)
            found = False
            for shape in shapes:
                sh, sw = shape.shape
                if sh > inner_h or sw > inner_w:
                    continue
                ir, ic = _inner_dock_origin(side, shape, inner_h, inner_w)
                cand = np.zeros((inner_h, inner_w), dtype=bool)
                cand[ir : ir + sh, ic : ic + sw] = shape
                if np.any(cand & occ):
                    continue
                if np.any(cand & _dilate4(occ)):
                    continue
                chosen[side] = {
                    "shape": shape,
                    "dock_inner": (ir, ic),
                    "gap": random.randint(*SIDE_GAP_RANGE),
                }
                occ |= cand
                found = True
                break
            if not found:
                ok = False
                break
        if ok:
            return chosen
    return None


def _frame_patch(color: int, bg: int) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(PATCH_RETRIES):
        inner_h = random.randint(*INNER_HEIGHT_RANGE)
        inner_w = random.randint(*INNER_WIDTH_RANGE)
        n_frag = random.randint(*FRAGMENT_COUNT_RANGE)
        sides = random.sample(DIRECTIONS, n_frag)
        layout = _choose_fragment_layout(inner_h, inner_w, sides)
        if layout is None:
            continue

        frame_h = inner_h + 2
        frame_w = inner_w + 2
        entries: list[tuple[str, np.ndarray, int, int, int, int]] = []
        min_r = 0
        min_c = 0
        max_r = frame_h - 1
        max_c = frame_w - 1

        for side in sides:
            shape = np.asarray(layout[side]["shape"], dtype=bool)
            sh, sw = shape.shape
            ir, ic = map(int, layout[side]["dock_inner"])
            out_r = 1 + ir
            out_c = 1 + ic
            gap = int(layout[side]["gap"])

            if side == "top":
                in_r = -gap - sh
                in_c = 1 + ic
            elif side == "bottom":
                in_r = frame_h + gap
                in_c = 1 + ic
            elif side == "left":
                in_r = 1 + ir
                in_c = -gap - sw
            elif side == "right":
                in_r = 1 + ir
                in_c = frame_w + gap
            else:
                raise AssertionError(side)

            entries.append((side, shape, in_r, in_c, out_r, out_c))
            min_r = min(min_r, in_r)
            min_c = min(min_c, in_c)
            max_r = max(max_r, in_r + sh - 1)
            max_c = max(max_c, in_c + sw - 1)

        patch_h = max_r - min_r + 1
        patch_w = max_c - min_c + 1
        shift_r = -min_r
        shift_c = -min_c
        frame_r = shift_r
        frame_c = shift_c

        inp = np.full((patch_h, patch_w), int(bg), dtype=int)
        out = np.full((patch_h, patch_w), int(bg), dtype=int)
        _draw_frame(inp, frame_r, frame_c, frame_h, frame_w, color)
        _draw_frame(out, frame_r, frame_c, frame_h, frame_w, color)

        for _side, shape, in_r, in_c, out_r, out_c in entries:
            _draw_shape(inp, shape, shift_r + in_r, shift_c + in_c, color)
            _draw_shape(out, shape, shift_r + out_r, shift_c + out_c, color)

        if np.array_equal(main(inp), out):
            return inp, out
    return None


def _rectangles_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _place_patches(patches: list[tuple[np.ndarray, np.ndarray]]) -> dict[int, tuple[int, int]] | None:
    order = sorted(range(len(patches)), key=lambda i: patches[i][0].shape[0] * patches[i][0].shape[1], reverse=True)
    placements: dict[int, tuple[int, int]] = {}
    expanded_rects: list[tuple[int, int, int, int]] = []

    for idx in order:
        patch_h, patch_w = patches[idx][0].shape
        candidates = [(r, c) for r in range(BASE_CANVAS - patch_h + 1) for c in range(BASE_CANVAS - patch_w + 1)]
        random.shuffle(candidates)
        placed = False
        for r0, c0 in candidates:
            rect = (
                r0 - PACK_PADDING,
                c0 - PACK_PADDING,
                r0 + patch_h + PACK_PADDING,
                c0 + patch_w + PACK_PADDING,
            )
            if any(_rectangles_overlap(rect, prev) for prev in expanded_rects):
                continue
            placements[idx] = (r0, c0)
            expanded_rects.append(rect)
            placed = True
            break
        if not placed:
            return None
    return placements


def _bbox_of_non_bg(grid: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    pts = np.argwhere(grid != int(bg))
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    bg = random.randint(0, 9)
    n_frames = random.randint(*FRAME_COUNT_RANGE)
    colors = random.sample([c for c in range(10) if c != bg], n_frames)

    patches: list[tuple[np.ndarray, np.ndarray]] = []
    for color in colors:
        patch = _frame_patch(color, bg)
        if patch is None:
            return None
        patches.append(patch)

    placements = _place_patches(patches)
    if placements is None:
        return None

    inp = np.full((BASE_CANVAS, BASE_CANVAS), int(bg), dtype=int)
    out = np.full((BASE_CANVAS, BASE_CANVAS), int(bg), dtype=int)
    for idx, (in_patch, out_patch) in enumerate(patches):
        r0, c0 = placements[idx]
        ph, pw = in_patch.shape
        inp[r0 : r0 + ph, c0 : c0 + pw] = in_patch
        out[r0 : r0 + ph, c0 : c0 + pw] = out_patch

    if _get_bg(inp) != bg:
        return None

    r0, c0, r1, c1 = _bbox_of_non_bg(inp, bg)
    top = random.randint(0, r0)
    left = random.randint(0, c0)
    bottom = random.randint(r1, BASE_CANVAS - 1)
    right = random.randint(c1, BASE_CANVAS - 1)
    cropped_inp = inp[top : bottom + 1, left : right + 1]
    cropped_out = out[top : bottom + 1, left : right + 1]

    if _get_bg(cropped_inp) != bg:
        return None
    if not np.array_equal(main(cropped_inp), cropped_out):
        return None
    return cropped_inp, cropped_out


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        case = _build_case()
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid input")
