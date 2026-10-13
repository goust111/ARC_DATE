from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
FRAME_COLOR = 5
FRAGMENT_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_ATTEMPTS = 400
SIDES = ("top", "bottom", "left", "right")


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _most_common(arr: np.ndarray) -> int:
    vals, counts = np.unique(arr, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _frame_bbox(component: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = _bbox(component)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None
    expected_area = 2 * (h + w) - 4
    if len(component) != expected_area:
        return None
    cells = set(component)
    for r in range(r0, r1 + 1):
        if (r, c0) not in cells or (r, c1) not in cells:
            return None
    for c in range(c0, c1 + 1):
        if (r0, c) not in cells or (r1, c) not in cells:
            return None
    return r0, c0, r1, c1


def _side_from_endpoint(endpoint: tuple[int, int], frame_bbox: tuple[int, int, int, int]) -> str | None:
    r, c = endpoint
    r0, c0, r1, c1 = frame_bbox
    if r == r0 - 1 and c0 <= c <= c1:
        return "top"
    if r == r1 + 1 and c0 <= c <= c1:
        return "bottom"
    if c == c0 - 1 and r0 <= r <= r1:
        return "left"
    if c == c1 + 1 and r0 <= r <= r1:
        return "right"
    return None


def _outward_step(side: str) -> tuple[int, int]:
    return {
        "top": (-1, 0),
        "bottom": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
    }[side]


def _has_off_axis_neighbor(cell: tuple[int, int], comp_set: set[tuple[int, int]], side: str) -> bool:
    r, c = cell
    if side in ("top", "bottom"):
        return (r, c - 1) in comp_set or (r, c + 1) in comp_set
    return (r - 1, c) in comp_set or (r + 1, c) in comp_set


def _extract_fragment_cells(
    component: list[tuple[int, int]],
    endpoint: tuple[int, int],
    side: str,
) -> set[tuple[int, int]]:
    comp_set = set(component)
    dr, dc = _outward_step(side)
    ray: list[tuple[int, int]] = []
    r, c = endpoint
    while (r, c) in comp_set:
        ray.append((r, c))
        r += dr
        c += dc
    assert len(ray) >= 2, "connector must have positive length and reach a fragment"

    split_idx = None
    for i, cell in enumerate(ray):
        if _has_off_axis_neighbor(cell, comp_set, side):
            split_idx = i
            break
    if split_idx is None:
        split_idx = len(ray) - 1

    fragment_cells = comp_set - set(ray[:split_idx])
    assert fragment_cells, "fragment must remain after removing the connector"
    comps = _components(_cells_to_mask(fragment_cells))
    assert len(comps) == 1, "fragment must stay 4-connected"
    return fragment_cells


def _cells_to_mask(cells: set[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(cells)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in cells:
        mask[r - r0, c - c0] = True
    return mask


def _place_fragment(
    out: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    fragment_cells: set[tuple[int, int]],
    color: int,
    side: str,
    touch: tuple[int, int],
) -> None:
    fr0, fc0, fr1, fc1 = frame_bbox
    sr0, sc0, sr1, sc1 = _bbox(fragment_cells)
    sh = sr1 - sr0 + 1
    sw = sc1 - sc0 + 1
    center_r = sh // 2
    center_c = sw // 2

    if side == "top":
        top = fr0 + 1
        left = touch[1] - center_c
    elif side == "bottom":
        top = fr1 - sh
        left = touch[1] - center_c
    elif side == "left":
        top = touch[0] - center_r
        left = fc0 + 1
    else:
        top = touch[0] - center_r
        left = fc1 - sw

    assert fr0 + 1 <= top and top + sh - 1 <= fr1 - 1
    assert fc0 + 1 <= left and left + sw - 1 <= fc1 - 1

    for r, c in fragment_cells:
        out[top + (r - sr0), left + (c - sc0)] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each colored fragment along its connector into the matched hollow frame and seat it against the entered wall.

    Input structure:
    - The background is black.
    - Grey one-pixel-thick hollow rectangular frames sit on the grid, with black interiors.
    - Outside the frames are several uniquely colored 4-connected fragments; each fragment is connected to exactly one frame by a same-colored one-pixel-wide straight horizontal or vertical connector.

    Rule:
    - For each colored fragment, use its connector to identify which frame it belongs to and which frame side it reaches.
    - Erase the fragment and its connector from the outside.
    - Keep the fragment's shape and color, place it inside the matched frame flush against the entered inner wall, and align the fragment's center row or center column with the connector's contact point on the frame edge.
    - Leave every frame unchanged.

    Key insight:
    - The connector carries two roles at once: it pairs a fragment with a specific frame and it encodes the side from which the fragment must be inserted, while the edge contact point fixes the final offset along that wall.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h <= 30 and w <= 30

    bg = _most_common(g)
    colors = [int(v) for v in np.unique(g) if int(v) != bg]

    frame_entries: list[dict] = []
    other_entries: list[dict] = []
    for color in colors:
        for component in _components(g == color):
            frame = _frame_bbox(component)
            entry = {"color": int(color), "cells": component}
            if frame is not None:
                entry["bbox"] = frame
                frame_entries.append(entry)
            else:
                other_entries.append(entry)

    assert frame_entries, "expected at least one frame"
    frame_sets = [set(entry["cells"]) for entry in frame_entries]
    out = g.copy()

    matched_frames: set[int] = set()
    for entry in other_entries:
        component = entry["cells"]
        color = int(entry["color"])
        touches: list[tuple[int, tuple[int, int], str]] = []
        for cell in component:
            r, c = cell
            for idx, cells in enumerate(frame_sets):
                for nr, nc in _neighbors4(r, c):
                    if (nr, nc) in cells:
                        side = _side_from_endpoint(cell, frame_entries[idx]["bbox"])
                        if side is not None:
                            touches.append((idx, cell, side))
        unique_touches = {(idx, cell, side) for idx, cell, side in touches}
        assert len(unique_touches) == 1, "each fragment-line component must touch exactly one frame at one endpoint"
        frame_idx, endpoint, side = next(iter(unique_touches))
        assert frame_idx not in matched_frames, "each frame must match exactly one fragment"
        matched_frames.add(frame_idx)

        fragment_cells = _extract_fragment_cells(component, endpoint, side)
        for r, c in component:
            out[r, c] = bg
        _place_fragment(out, frame_entries[frame_idx]["bbox"], fragment_cells, color, side, endpoint)

    assert len(matched_frames) == len(frame_entries), "every frame must have one connector-matched fragment"
    return out


def _normalize(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    r0, c0, _r1, _c1 = _bbox(cells)
    return {(r - r0, c - c0) for r, c in cells}


def _grow_shape(base_h: int, base_w: int, seed_cells: list[tuple[int, int]], target_area: int) -> set[tuple[int, int]]:
    cells = set(seed_cells)
    while len(cells) < target_area:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < base_h and 0 <= nc < base_w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            break
        cells.add(random.choice(frontier))
    return cells


def _attachment_candidates(cells: set[tuple[int, int]], side: str) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = _bbox(cells)
    candidates: list[tuple[int, int]] = []
    for r, c in cells:
        if side == "top" and r == r1 and ((r, c - 1) in cells or (r, c + 1) in cells):
            candidates.append((r, c))
        elif side == "bottom" and r == r0 and ((r, c - 1) in cells or (r, c + 1) in cells):
            candidates.append((r, c))
        elif side == "left" and c == c1 and ((r - 1, c) in cells or (r + 1, c) in cells):
            candidates.append((r, c))
        elif side == "right" and c == c0 and ((r - 1, c) in cells or (r + 1, c) in cells):
            candidates.append((r, c))
    return candidates


def _random_fragment(side: str) -> tuple[np.ndarray, tuple[int, int]]:
    sizes = (3, 5, 7)
    for _ in range(200):
        base_h = random.choice(sizes)
        base_w = random.choice(sizes)
        if side in ("top", "bottom"):
            seed_r = base_h - 1 if side == "top" else 0
            seed_c = random.randint(1, base_w - 2)
            neighbor = (seed_r, seed_c + random.choice((-1, 1)))
        else:
            seed_r = random.randint(1, base_h - 2)
            seed_c = base_w - 1 if side == "left" else 0
            neighbor = (seed_r + random.choice((-1, 1)), seed_c)
        target_area = random.randint(max(4, base_h * base_w // 3), base_h * base_w)
        cells = _grow_shape(base_h, base_w, [(seed_r, seed_c), neighbor], target_area)
        cells = _normalize(cells)
        r0, c0, r1, c1 = _bbox(cells)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h % 2 == 0 or w % 2 == 0:
            continue
        candidates = _attachment_candidates(cells, side)
        if not candidates:
            continue
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        return mask, random.choice(candidates)
    raise RuntimeError("failed to generate a valid fragment")


def _stamp_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    grid[top : top + h, left : left + w][mask] = color


def _build_pair_patch(color: int, side: str) -> tuple[np.ndarray, np.ndarray]:
    fragment_mask, attach = _random_fragment(side)
    frag_h, frag_w = fragment_mask.shape
    center_r = frag_h // 2
    center_c = frag_w // 2

    interior_h = frag_h + random.randint(0, 4)
    interior_w = frag_w + random.randint(0, 4)
    frame_h = interior_h + 2
    frame_w = interior_w + 2

    if side in ("top", "bottom"):
        target_left = random.randint(0, interior_w - frag_w)
        touch_col = 1 + target_left + center_c
        target_top = 1 if side == "top" else frame_h - 1 - frag_h
        touch_row = 0 if side == "top" else frame_h - 1
    else:
        target_top = random.randint(0, interior_h - frag_h)
        touch_row = 1 + target_top + center_r
        target_left = 1 if side == "left" else frame_w - 1 - frag_w
        touch_col = 0 if side == "left" else frame_w - 1

    line_len = random.randint(1, 5)
    attach_r, attach_c = attach
    frame_top = 0
    frame_left = 0
    if side == "top":
        frag_top = -line_len - 1 - attach_r
        frag_left = touch_col - attach_c
    elif side == "bottom":
        frag_top = frame_h + line_len - attach_r
        frag_left = touch_col - attach_c
    elif side == "left":
        frag_top = touch_row - attach_r
        frag_left = -line_len - 1 - attach_c
    else:
        frag_top = touch_row - attach_r
        frag_left = frame_w + line_len - attach_c

    min_r = min(frame_top, frag_top)
    min_c = min(frame_left, frag_left)
    max_r = max(frame_top + frame_h - 1, frag_top + frag_h - 1)
    max_c = max(frame_left + frame_w - 1, frag_left + frag_w - 1)
    patch_h = max_r - min_r + 1
    patch_w = max_c - min_c + 1

    input_patch = np.full((patch_h, patch_w), BG, dtype=int)
    output_patch = np.full((patch_h, patch_w), BG, dtype=int)
    fr0 = frame_top - min_r
    fc0 = frame_left - min_c
    ft = frag_top - min_r
    fl = frag_left - min_c

    input_patch[fr0, fc0 : fc0 + frame_w] = FRAME_COLOR
    input_patch[fr0 + frame_h - 1, fc0 : fc0 + frame_w] = FRAME_COLOR
    input_patch[fr0 : fr0 + frame_h, fc0] = FRAME_COLOR
    input_patch[fr0 : fr0 + frame_h, fc0 + frame_w - 1] = FRAME_COLOR
    output_patch[:] = input_patch

    _stamp_mask(input_patch, fragment_mask, ft, fl, color)
    if side == "top":
        for k in range(1, line_len + 1):
            input_patch[fr0 - k, fc0 + touch_col] = color
    elif side == "bottom":
        for k in range(1, line_len + 1):
            input_patch[fr0 + frame_h - 1 + k, fc0 + touch_col] = color
    elif side == "left":
        for k in range(1, line_len + 1):
            input_patch[fr0 + touch_row, fc0 - k] = color
    else:
        for k in range(1, line_len + 1):
            input_patch[fr0 + touch_row, fc0 + frame_w - 1 + k] = color

    if side in ("top", "bottom"):
        out_top = fr0 + target_top
        out_left = fc0 + 1 + target_left
    else:
        out_top = fr0 + 1 + target_top
        out_left = fc0 + target_left
    _stamp_mask(output_patch, fragment_mask, out_top, out_left, color)
    return input_patch, output_patch


def _place_patches(
    patches: list[tuple[np.ndarray, np.ndarray]],
    canvas_h: int,
    canvas_w: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    occupied = np.zeros((canvas_h, canvas_w), dtype=bool)
    inp = np.full((canvas_h, canvas_w), BG, dtype=int)
    out = np.full((canvas_h, canvas_w), BG, dtype=int)

    order = sorted(range(len(patches)), key=lambda i: patches[i][0].shape[0] * patches[i][0].shape[1], reverse=True)
    for idx in order:
        input_patch, output_patch = patches[idx]
        ph, pw = input_patch.shape
        if ph + 2 > canvas_h or pw + 2 > canvas_w:
            return None
        candidates: list[tuple[int, int]] = []
        for top in range(canvas_h - ph + 1):
            for left in range(canvas_w - pw + 1):
                r0 = max(0, top - 1)
                c0 = max(0, left - 1)
                r1 = min(canvas_h, top + ph + 1)
                c1 = min(canvas_w, left + pw + 1)
                if bool(occupied[r0:r1, c0:c1].any()):
                    continue
                candidates.append((top, left))
        if not candidates:
            return None
        top, left = random.choice(candidates)
        inp[top : top + ph, left : left + pw] = np.maximum(inp[top : top + ph, left : left + pw], input_patch)
        out[top : top + ph, left : left + pw] = np.maximum(out[top : top + ph, left : left + pw], output_patch)
        occupied[top : top + ph, left : left + pw] = True
    return inp, out


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.randint(1, 4)
    colors = random.sample(FRAGMENT_COLORS, pair_count)
    sides = [random.choice(SIDES) for _ in range(pair_count)]
    patches = [_build_pair_patch(color, side) for color, side in zip(colors, sides)]

    max_h = max(p[0].shape[0] for p in patches)
    max_w = max(p[0].shape[1] for p in patches)
    total_area = sum((p[0].shape[0] + 2) * (p[0].shape[1] + 2) for p in patches)
    min_side = max(12, int(np.ceil(np.sqrt(total_area))))
    min_canvas_h = max(max_h + 2, min_side)
    min_canvas_w = max(max_w + 2, min_side)
    if min_canvas_h > 30 or min_canvas_w > 30:
        return None
    canvas_h = random.randint(min_canvas_h, 30)
    canvas_w = random.randint(min_canvas_w, 30)

    placed = _place_patches(patches, canvas_h, canvas_w)
    if placed is None:
        return None
    inp, expected = placed
    if _most_common(inp) != BG:
        return None
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
