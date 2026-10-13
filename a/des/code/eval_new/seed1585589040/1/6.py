from __future__ import annotations

import random

import numpy as np


BG = 0
ARROW_COLOR = 9
COLOR_POOL = (1, 2, 3, 4, 5, 6, 7, 8)
CARDINALS = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_RETRIES = 500


def _get_bg(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(cnts))])


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]


def _components(mask: np.ndarray) -> list[set[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[set[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: set[tuple[int, int]] = set()
            while stack:
                rr, cc = stack.pop()
                comp.add((int(rr), int(cc)))
                for nr, nc in _neighbors4(int(rr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _bbox(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _is_rect_frame(cells: set[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(cells)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False
    expected: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        expected.add((r0, c))
        expected.add((r1, c))
    for r in range(r0, r1 + 1):
        expected.add((r, c0))
        expected.add((r, c1))
    return cells == expected


def _arrow_corner(cells: set[tuple[int, int]]) -> tuple[int, int] | None:
    if len(cells) != 3:
        return None
    for r, c in cells:
        nbrs = [(nr - r, nc - c) for nr, nc in _neighbors4(r, c) if (nr, nc) in cells]
        if len(nbrs) != 2:
            continue
        if nbrs[0][0] * nbrs[1][0] + nbrs[0][1] * nbrs[1][1] == 0:
            return (r, c)
    return None


def _contact_direction(
    frag: set[tuple[int, int]], arrow: dict[str, object]
) -> tuple[int, int] | None:
    cells: set[tuple[int, int]] = arrow["cells"]  # type: ignore[assignment]
    corner: tuple[int, int] = arrow["corner"]  # type: ignore[assignment]
    contacts: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for ar, ac in cells:
        for nr, nc in _neighbors4(ar, ac):
            if (nr, nc) in frag:
                contacts.append(((ar, ac), (nr, nc)))
    if not contacts:
        return None
    contacts.sort(key=lambda pair: (pair[0] != corner, pair[0], pair[1]))
    (ar, ac), (fr, fc) = contacts[0]
    dr, dc = ar - fr, ac - fc
    return (int(dr), int(dc))


def _translate_to_frame(
    frag: set[tuple[int, int]],
    frame_bbox: tuple[int, int, int, int],
    arrow_dir: tuple[int, int],
) -> set[tuple[int, int]]:
    fr0, fc0, fr1, fc1 = frame_bbox
    ir0, ic0, ir1, ic1 = fr0 + 1, fc0 + 1, fr1 - 1, fc1 - 1
    rows = [r for r, _c in frag]
    cols = [c for _r, c in frag]
    dr, dc = arrow_dir
    shift_r = 0
    shift_c = 0
    if (dr, dc) == (0, 1):
        shift_c = ic1 - max(cols)
    elif (dr, dc) == (0, -1):
        shift_c = ic0 - min(cols)
    elif (dr, dc) == (1, 0):
        shift_r = ir1 - max(rows)
    elif (dr, dc) == (-1, 0):
        shift_r = ir0 - min(rows)
    else:
        raise AssertionError(f"unexpected arrow direction {arrow_dir}")
    return {(r + shift_r, c + shift_c) for r, c in frag}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each colored fragment back into its matching hollow frame by reversing the nearby arrow direction.

    Input structure:
    - The background is black.
    - Several distinct colors each appear as one hollow rectangular frame and one smaller connected fragment of the same color.
    - Every fragment has an adjacent maroon 3-pixel L-shaped arrow marker.

    Rule:
    - Detect each maroon arrow and infer which fragment it belongs to from direct adjacency.
    - The arrow points away from the fragment, so the fragment must move in the opposite direction until it sits inside the same-colored frame and touches that frame's inner boundary.
    - Remove the fragment from its scattered location, erase the arrow, and redraw the fragment at its returned position inside the frame.

    Key insight:
    - Color matches fragment to frame, but the arrow gives the ejection direction rather than the movement direction, so the correct translation is the reverse of the visible pointer.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    out = g.copy()

    frames: dict[int, tuple[int, int, int, int]] = {}
    fragments: dict[int, set[tuple[int, int]]] = {}
    colors = [int(v) for v in np.unique(g) if int(v) not in {bg, ARROW_COLOR}]
    for color in colors:
        comps = _components(g == color)
        frame_comps = [comp for comp in comps if _is_rect_frame(comp)]
        frag_comps = [comp for comp in comps if not _is_rect_frame(comp)]
        assert len(frame_comps) == 1, f"expected one frame for color {color}"
        assert len(frag_comps) == 1, f"expected one fragment for color {color}"
        frames[color] = _bbox(frame_comps[0])
        fragments[color] = frag_comps[0]

    arrows = []
    for comp in _components(g == ARROW_COLOR):
        corner = _arrow_corner(comp)
        if corner is not None:
            arrows.append({"cells": comp, "corner": corner})

    used_arrows: set[int] = set()
    for color, frag in fragments.items():
        best_idx = None
        best_dir = None
        best_score = None
        for idx, arrow in enumerate(arrows):
            if idx in used_arrows:
                continue
            direction = _contact_direction(frag, arrow)
            if direction is None:
                continue
            corner = arrow["corner"]  # type: ignore[assignment]
            score = (corner not in frag, idx)
            if best_score is None or score < best_score:
                best_score = score
                best_idx = idx
                best_dir = direction
        assert best_idx is not None and best_dir is not None, f"missing arrow for color {color}"
        used_arrows.add(best_idx)

        for r, c in frag:
            out[r, c] = bg
        for r, c in arrows[best_idx]["cells"]:  # type: ignore[index]
            out[r, c] = bg

        translated = _translate_to_frame(frag, frames[color], best_dir)
        fr0, fc0, fr1, fc1 = frames[color]
        for r, c in translated:
            assert fr0 < r < fr1 and fc0 < c < fc1
            out[r, c] = color

    return out


def _local_is_frame(cells: set[tuple[int, int]]) -> bool:
    return _is_rect_frame(cells)


def _random_shape(height: int, width: int, side: str) -> tuple[set[tuple[int, int]], list[tuple[int, int]]]:
    edge_c = 0 if side == "left" else width - 1
    for _ in range(200):
        area_max = max(3, min(height * width, 8))
        area = random.randint(3, area_max)
        start = (random.randrange(height), edge_c)
        cells = {start}
        while len(cells) < area:
            br, bc = random.choice(tuple(cells))
            dr, dc = random.choice(CARDINALS)
            nr, nc = br + dr, bc + dc
            if 0 <= nr < height and 0 <= nc < width:
                cells.add((nr, nc))
        if _local_is_frame(cells):
            continue
        candidates: list[tuple[int, int]] = []
        for r, c in sorted(cells):
            if c != edge_c:
                continue
            for bend in (-1, 1):
                if (r + bend, c) in cells:
                    continue
                candidates.append((r, bend))
        if candidates:
            return cells, candidates
    raise RuntimeError("failed to create a valid fragment shape")


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color


def _place_cells(grid: np.ndarray, cells: set[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _shift_cells(cells: set[tuple[int, int]], dr: int, dc: int) -> set[tuple[int, int]]:
    return {(r + dr, c + dc) for r, c in cells}


def _band_offsets(frame_hs: list[int], gaps: list[int]) -> list[int]:
    tops = []
    cursor = gaps[0]
    for idx, fh in enumerate(frame_hs):
        tops.append(cursor)
        cursor += fh + gaps[idx + 1]
    return tops


def _gen_object(
    h: int, w: int, top: int, frame_h: int, color: int
) -> tuple[dict[str, object], tuple[np.ndarray, np.ndarray]] | None:
    frame_w = random.randint(6, min(10, w - 4))
    side = random.choice(("left", "right"))
    interior_h = frame_h - 2
    interior_w = frame_w - 2
    shape_h_max = max(2, min(4, interior_h - 1 if interior_h >= 4 else interior_h))
    shape_w_max = max(2, min(4, interior_w - 1 if interior_w >= 4 else interior_w))
    shape_h = random.randint(2, shape_h_max)
    shape_w = random.randint(2, shape_w_max)
    shape, anchors = _random_shape(shape_h, shape_w, side)

    row_offsets = list(range(interior_h - shape_h + 1))
    if interior_h - shape_h >= 2:
        row_offsets = list(range(1, interior_h - shape_h))
    if not row_offsets:
        row_offsets = [0]
    inner_row_offset = random.choice(row_offsets)

    min_dist = shape_w + 1
    max_dist = shape_w + 5
    valid = []
    for dist in range(min_dist, max_dist + 1):
        if side == "left":
            c0_min = dist + 1
            c0_max = w - frame_w - 1
        else:
            c0_min = 1
            c0_max = w - frame_w - dist - 1
        if c0_min <= c0_max:
            valid.append((dist, c0_min, c0_max))
    if not valid:
        return None
    dist, c0_min, c0_max = random.choice(valid)
    c0 = random.randint(c0_min, c0_max)

    frame_bbox = (top, c0, top + frame_h - 1, c0 + frame_w - 1)
    ir0, ic0, _ir1, ic1 = top + 1, c0 + 1, top + frame_h - 2, c0 + frame_w - 2
    final_r0 = ir0 + inner_row_offset
    final_c0 = ic0 if side == "left" else ic1 - shape_w + 1
    final_cells = _shift_cells(shape, final_r0, final_c0)

    input_shift_c = -dist if side == "left" else dist
    input_cells = _shift_cells(final_cells, 0, input_shift_c)
    input_r0 = min(r for r, _c in input_cells)
    input_r1 = max(r for r, _c in input_cells)
    input_c0 = min(c for _r, c in input_cells)
    input_c1 = max(c for _r, c in input_cells)
    if not (0 <= input_r0 <= input_r1 < h and 0 <= input_c0 <= input_c1 < w):
        return None

    anchor_options = []
    for local_r, bend in anchors:
        global_r = final_r0 + local_r
        if not (0 <= global_r + bend < h):
            continue
        anchor_options.append((local_r, bend))
    if not anchor_options:
        return None
    anchor_r, bend = random.choice(anchor_options)

    anchor_col = input_c0 if side == "left" else input_c1
    corner_c = anchor_col - 1 if side == "left" else anchor_col + 1
    end_c = corner_c - 1 if side == "left" else corner_c + 1
    corner_r = final_r0 + anchor_r
    arrow_cells = {
        (corner_r, corner_c),
        (corner_r, end_c),
        (corner_r + bend, corner_c),
    }
    if any(not (0 <= r < h and 0 <= c < w) for r, c in arrow_cells):
        return None

    occupied = final_cells | input_cells | arrow_cells
    if len(occupied) != len(final_cells) + len(input_cells) + len(arrow_cells):
        return None
    if any(frame_bbox[0] <= r <= frame_bbox[2] and frame_bbox[1] <= c <= frame_bbox[3] for r, c in input_cells):
        return None
    if any(frame_bbox[0] <= r <= frame_bbox[2] and frame_bbox[1] <= c <= frame_bbox[3] for r, c in arrow_cells):
        return None
    if not any(abs(ar - fr) + abs(ac - fc) == 1 for ar, ac in arrow_cells for fr, fc in input_cells):
        return None

    obj = {
        "frame_bbox": frame_bbox,
        "final_cells": final_cells,
        "input_cells": input_cells,
        "arrow_cells": arrow_cells,
        "color": color,
    }

    span = np.zeros((h, w), dtype=int)
    expected = np.zeros((h, w), dtype=int)
    _draw_frame(span, frame_bbox, color)
    _draw_frame(expected, frame_bbox, color)
    _place_cells(span, input_cells, color)
    _place_cells(span, arrow_cells, ARROW_COLOR)
    _place_cells(expected, final_cells, color)
    return obj, (span, expected)


def _gen_case() -> tuple[np.ndarray, np.ndarray] | None:
    n_objects = random.randint(2, 4)
    frame_hs = [random.randint(5, 8) for _ in range(n_objects)]
    gaps = [random.randint(1, 2) for _ in range(n_objects + 1)]
    h = sum(frame_hs) + sum(gaps)
    if h > 30:
        return None
    w = random.randint(20, 30)
    if w > 30:
        return None

    colors = random.sample(COLOR_POOL, n_objects)
    tops = _band_offsets(frame_hs, gaps)
    inp = np.full((h, w), BG, dtype=int)
    out = np.full((h, w), BG, dtype=int)

    for top, frame_h, color in zip(tops, frame_hs, colors, strict=False):
        built = _gen_object(h, w, top, frame_h, color)
        if built is None:
            return None
        _obj, (local_inp, local_out) = built
        if np.any((inp != BG) & (local_inp != BG)):
            return None
        inp = np.where(local_inp != BG, local_inp, inp)
        out = np.where(local_out != BG, local_out, out)

    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        sample = _gen_case()
        if sample is None:
            continue
        inp, out = sample
        try:
            if np.array_equal(main(inp), out):
                return inp
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid puzzle instance")
