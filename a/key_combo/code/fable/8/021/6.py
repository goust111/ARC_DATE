import random
from collections import Counter

import numpy as np


BACKGROUND = 0
ALL_FOREGROUND_COLORS = list(range(1, 10))
GENERATION_RETRIES = 500
PATH_ATTEMPTS = 200
LAYOUT_ATTEMPTS = 200
TEMP_PATH_SIZE = 9
PATH_STEPS = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1)]
_LAST_SPEC = None


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).flatten())).most_common(1)[0][0])


def neighbors(r: int, c: int, connectivity: int) -> list[tuple[int, int]]:
    if int(connectivity) == 4:
        return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
    return [
        (r - 1, c - 1),
        (r - 1, c),
        (r - 1, c + 1),
        (r, c - 1),
        (r, c + 1),
        (r + 1, c - 1),
        (r + 1, c),
        (r + 1, c + 1),
    ]


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = map(int, bbox)
    return np.asarray(grid)[r0 : r1 + 1, c0 : c1 + 1].copy()


def find_components(
    grid: np.ndarray,
    *,
    bg: int | None = None,
    connectivity: int = 4,
    monochromatic: bool = True,
) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    if bg is None:
        bg = get_bg(g)
    bg = int(bg)
    h, w = g.shape
    dirs = neighbors(0, 0, connectivity)
    out: list[dict] = []

    if monochromatic:
        for color in sorted(set(map(int, g.flatten())) - {bg}):
            seen = np.zeros((h, w), dtype=bool)
            for r in range(h):
                for c in range(w):
                    if seen[r, c] or int(g[r, c]) != color:
                        continue
                    stack = [(r, c)]
                    seen[r, c] = True
                    coords = []
                    while stack:
                        rr, cc = stack.pop()
                        coords.append((rr, cc))
                        for dr, dc in dirs:
                            nr, nc = rr + dr, cc + dc
                            if not (0 <= nr < h and 0 <= nc < w):
                                continue
                            if seen[nr, nc] or int(g[nr, nc]) != color:
                                continue
                            seen[nr, nc] = True
                            stack.append((nr, nc))
                    arr = np.array(coords, dtype=int)
                    out.append(
                        {
                            "color": int(color),
                            "coords": arr,
                            "bbox": bbox_coords(arr),
                            "area": int(arr.shape[0]),
                        }
                    )
        return out

    seen = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in dirs:
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(g[nr, nc]) == bg:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            out.append({"color": None, "coords": arr, "bbox": bbox_coords(arr), "area": int(arr.shape[0])})
    return out


def _is_hollow_rect_frame(grid: np.ndarray, comp: dict, bg: int) -> bool:
    color = int(comp["color"])
    r0, c0, r1, c1 = map(int, comp["bbox"])
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    patch = crop_bbox(grid, comp["bbox"])
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    if int(comp["area"]) != int(border.sum()):
        return False
    return bool(np.all(patch[border] == color) and np.all(patch[~border] == int(bg)))


def _analyze_input(grid: np.ndarray) -> tuple[int, tuple[int, int, int, int], int, list[tuple[int, int, int]]]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    frame_candidates = [
        comp for comp in find_components(g, bg=bg, connectivity=8, monochromatic=True) if _is_hollow_rect_frame(g, comp, bg)
    ]
    assert len(frame_candidates) == 1, "expected exactly one hollow rectangular frame"
    frame = frame_candidates[0]
    frame_color = int(frame["color"])
    frame_mask = np.zeros_like(g, dtype=bool)
    for r, c in frame["coords"]:
        frame_mask[int(r), int(c)] = True

    motif_coords = np.argwhere((g != int(bg)) & (~frame_mask))
    assert motif_coords.size > 0, "expected a non-frame motif"
    motif = [(int(r), int(c), int(g[int(r), int(c)])) for r, c in motif_coords]
    return int(bg), tuple(map(int, frame["bbox"])), int(frame_color), motif


def _develop_inside_window(
    base: np.ndarray,
    motif_cells: list[tuple[int, int, int]],
    frame_bbox: tuple[int, int, int, int],
    bg: int,
) -> np.ndarray:
    out = np.array(base, dtype=int, copy=True)
    r0, c0, r1, c1 = map(int, frame_bbox)
    ir0, ic0, ir1, ic1 = r0 + 1, c0 + 1, r1 - 1, c1 - 1
    if ir0 > ir1 or ic0 > ic1:
        return out

    for sr, sc, color in motif_cells:
        k0 = max(0, ir0 - int(sr), ic0 - int(sc))
        k1 = min(ir1 - int(sr), ic1 - int(sc))
        for k in range(int(k0), int(k1) + 1):
            rr = int(sr + k)
            cc = int(sc + k)
            prev = int(out[rr, cc])
            if prev not in (int(bg), int(color)):
                raise AssertionError("overlapping diagonal copies must agree in color")
            out[rr, cc] = int(color)
    return out


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Develop the diagonal-offset repetitions of a motif only inside a hollow rectangular window.

    Input structure:
    - Black background with one small multi-color motif near the upper-left part of the grid.
    - Elsewhere there is exactly one one-pixel-thick hollow rectangular frame whose interior is empty.
    - The motif and the frame are separated and do not overlap.

    Rule:
    - Treat every non-background motif pixel as being stamped again and again after shifts of (+1,+1), with background remaining transparent.
    - Keep the original motif and the rectangular frame unchanged.
    - Inside the frame interior only, draw every cell reached by at least one such diagonal copy, using the copied pixel's color; leave unreached interior cells as background.

    Key insight:
    - The visible grid never shows the repeated diagonal band directly; the motif fixes the phase, and the frame reveals only the clipped portion of that invisible infinite repetition.
    """
    g = np.asarray(input_grid, dtype=int)
    bg, frame_bbox, _frame_color, motif_cells = _analyze_input(g)
    return _develop_inside_window(g, motif_cells, frame_bbox, bg)


def _frame_border_cells(frame_bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = map(int, frame_bbox)
    cells = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        cells.append((r, c1))
    return cells


def _render_input_from_spec(spec: dict) -> np.ndarray:
    grid = np.full((spec["height"], spec["width"]), spec["bg"], dtype=int)
    for r, c, color in spec["motif_cells"]:
        grid[int(r), int(c)] = int(color)
    for r, c in _frame_border_cells(spec["frame_bbox"]):
        grid[int(r), int(c)] = int(spec["frame_color"])
    return grid


def _expected_output_from_spec(spec: dict) -> np.ndarray:
    inp = _render_input_from_spec(spec)
    return _develop_inside_window(inp, spec["motif_cells"], spec["frame_bbox"], spec["bg"])


def _sample_path(rng: random.Random) -> list[tuple[int, int]] | None:
    for _ in range(PATH_ATTEMPTS):
        target_len = rng.randint(4, 8)
        start = (TEMP_PATH_SIZE // 2, TEMP_PATH_SIZE // 2)
        path = [start]
        used_cells = {start}
        used_diagonals = {start[0] - start[1]}
        while len(path) < target_len:
            cur_r, cur_c = path[-1]
            candidates = []
            for dr, dc in PATH_STEPS:
                nr, nc = cur_r + dr, cur_c + dc
                if not (0 <= nr < TEMP_PATH_SIZE and 0 <= nc < TEMP_PATH_SIZE):
                    continue
                if (nr, nc) in used_cells:
                    continue
                if (nr - nc) in used_diagonals:
                    continue
                future_degree = 0
                for fdr, fdc in PATH_STEPS:
                    rr, cc = nr + fdr, nc + fdc
                    if not (0 <= rr < TEMP_PATH_SIZE and 0 <= cc < TEMP_PATH_SIZE):
                        continue
                    if (rr, cc) in used_cells or (rr - cc) in used_diagonals:
                        continue
                    future_degree += 1
                bbox_now = bbox_coords(np.array(path + [(nr, nc)], dtype=int))
                h_now = bbox_now[2] - bbox_now[0] + 1
                w_now = bbox_now[3] - bbox_now[1] + 1
                if h_now > 6 or w_now > 6:
                    continue
                candidates.append((future_degree, rng.random(), nr, nc))
            if not candidates:
                break
            candidates.sort(reverse=True)
            _, _, nr, nc = candidates[0]
            path.append((nr, nc))
            used_cells.add((nr, nc))
            used_diagonals.add(nr - nc)

        if len(path) != target_len:
            continue
        arr = np.array(path, dtype=int)
        r0, c0, r1, c1 = bbox_coords(arr)
        norm = [(int(r - r0), int(c - c0)) for r, c in path]
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if not (3 <= h <= 6 and 3 <= w <= 6):
            continue
        if len({r for r, _c in norm}) < 2 or len({c for _r, c in norm}) < 2:
            continue
        return norm
    return None


def _choose_colors(rng: random.Random, count: int) -> tuple[list[int], int] | None:
    if count < 4:
        return None
    palette_size = rng.randint(2, min(5, count))
    motif_palette = rng.sample(ALL_FOREGROUND_COLORS, palette_size)
    motif_colors = [rng.choice(motif_palette) for _ in range(count)]
    if len(set(motif_colors)) < 2:
        return None
    remaining = [c for c in ALL_FOREGROUND_COLORS if c not in set(motif_colors)]
    if not remaining:
        return None
    frame_color = rng.choice(remaining)
    return motif_colors, frame_color


def _sample_spec(rng: random.Random) -> dict | None:
    motif_shape = _sample_path(rng)
    if motif_shape is None:
        return None
    chosen = _choose_colors(rng, len(motif_shape))
    if chosen is None:
        return None
    motif_colors, frame_color = chosen

    mh = max(r for r, _c in motif_shape) + 1
    mw = max(c for _r, c in motif_shape) + 1
    motif_top = rng.randint(1, 4)
    motif_left = rng.randint(1, 4)
    motif_cells = [(motif_top + r, motif_left + c, motif_colors[i]) for i, (r, c) in enumerate(motif_shape)]
    motif_bottom = motif_top + mh - 1
    motif_right = motif_left + mw - 1

    for _ in range(LAYOUT_ATTEMPTS):
        frame_inner_h = rng.randint(5, 11)
        frame_inner_w = rng.randint(5, 11)
        frame_h = frame_inner_h + 2
        frame_w = frame_inner_w + 2

        min_fr0 = motif_bottom + 4
        min_fc0 = motif_right + 4
        max_fr0 = 30 - frame_h
        max_fc0 = 30 - frame_w
        if min_fr0 > max_fr0 or min_fc0 > max_fc0:
            continue
        frame_top = rng.randint(min_fr0, max_fr0)
        frame_left = rng.randint(min_fc0, max_fc0)
        height = rng.randint(frame_top + frame_h, 30)
        width = rng.randint(frame_left + frame_w, 30)
        frame_bbox = (frame_top, frame_left, frame_top + frame_h - 1, frame_left + frame_w - 1)

        spec = {
            "height": height,
            "width": width,
            "bg": BACKGROUND,
            "motif_cells": motif_cells,
            "frame_bbox": frame_bbox,
            "frame_color": frame_color,
        }
        inp = _render_input_from_spec(spec)
        out = _expected_output_from_spec(spec)
        ir0, ic0, ir1, ic1 = frame_top + 1, frame_left + 1, frame_top + frame_h - 2, frame_left + frame_w - 2
        interior = out[ir0 : ir1 + 1, ic0 : ic1 + 1]
        painted = interior[interior != BACKGROUND]
        if painted.size < 4:
            continue
        if painted.size >= interior.size:
            continue
        if len(set(map(int, painted.tolist()))) < 2:
            continue
        hit_diagonals = 0
        miss_diagonals = 0
        for r, c, _color in motif_cells:
            hit = False
            for k in range(0, max(height, width)):
                rr = r + k
                cc = c + k
                if rr > ir1 or cc > ic1:
                    break
                if ir0 <= rr <= ir1 and ic0 <= cc <= ic1:
                    hit = True
                    break
            if hit:
                hit_diagonals += 1
            else:
                miss_diagonals += 1
        if hit_diagonals < 2:
            continue
        if miss_diagonals < 1:
            continue
        return spec
    return None


def generate_input() -> np.ndarray:
    global _LAST_SPEC
    rng = random.Random()
    for _ in range(GENERATION_RETRIES):
        spec = _sample_spec(rng)
        if spec is None:
            continue
        inp = _render_input_from_spec(spec)
        out = _expected_output_from_spec(spec)
        solved = main(inp)
        if np.array_equal(out, solved):
            _LAST_SPEC = spec
            return inp
    raise RuntimeError("failed to generate a valid input for puzzle 6")
