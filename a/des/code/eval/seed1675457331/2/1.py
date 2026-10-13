from __future__ import annotations

import random
from collections import Counter, defaultdict

import numpy as np


BG = 8
FRAME_COLOR = 0
PALETTE = (1, 2, 3, 4, 5, 6, 7, 9)
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(cells: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0, c0, _r1, _c1 = _bbox(cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _is_connected(cells: list[tuple[int, int]] | set[tuple[int, int]]) -> bool:
    pts = set(cells)
    if not pts:
        return False
    stack = [next(iter(pts))]
    seen = {stack[0]}
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in pts and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(pts)


def _is_l_shape(cells: list[tuple[int, int]] | set[tuple[int, int]]) -> bool:
    pts = set(cells)
    if len(pts) not in (3, 4) or not _is_connected(pts):
        return False
    r0, c0, r1, c1 = _bbox(pts)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 2 or w < 2:
        return False
    norm = {(r - r0, c - c0) for r, c in pts}
    corners = ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1))
    for cr, cc in corners:
        row = {(cr, c) for c in range(w)}
        col = {(r, cc) for r in range(h)}
        if norm == row | col:
            return True
    return False


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _rect_border_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    pts = {(r0, c) for c in range(c0, c1 + 1)}
    pts |= {(r1, c) for c in range(c0, c1 + 1)}
    pts |= {(r, c0) for r in range(r0, r1 + 1)}
    pts |= {(r, c1) for r in range(r0, r1 + 1)}
    return pts


def _parse_seed(
    grid: np.ndarray, r0: int, c0: int, r1: int, c1: int, bg: int, frame_color: int
) -> dict:
    cells: list[tuple[int, int, int]] = []
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            v = int(grid[r, c])
            if v not in (bg, frame_color):
                cells.append((r, c, v))
    assert len(cells) == 2, "frame interior must contain exactly one seed pair"

    (r_a, c_a, col_a), (r_b, c_b, col_b) = cells
    assert col_a != col_b, "seed colors must differ"
    if r_a == r_b and abs(c_a - c_b) == 1:
        left, right = sorted(cells, key=lambda x: x[1])
        return {"orientation": "h", "cells": left, "other": right}
    if c_a == c_b and abs(r_a - r_b) == 1:
        top, bottom = sorted(cells, key=lambda x: x[0])
        return {"orientation": "v", "cells": top, "other": bottom}
    raise AssertionError("seed pair must be adjacent horizontally or vertically")


def _frame_candidates(grid: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for comp in _components(grid == color):
            if len(comp) < 12:
                continue
            r0, c0, r1, c1 = _bbox(comp)
            if r1 - r0 + 1 < 5 or c1 - c0 + 1 < 5:
                continue
            comp_set = set(comp)
            border = _rect_border_cells(r0, c0, r1, c1)
            if not comp_set <= border:
                continue
            gap = sorted(border - comp_set)
            if len(gap) not in (3, 4) or not _is_l_shape(gap):
                continue
            seed = _parse_seed(grid, r0, c0, r1, c1, bg, color)
            frames.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "border": border,
                    "gap": gap,
                    "gap_key": _normalize(gap),
                    "seed": seed,
                }
            )
    assert frames, "expected at least one incomplete frame"
    return frames


def _required_fragment_color(frame: dict) -> int:
    gap = frame["gap"]
    seed = frame["seed"]
    first = seed["cells"]
    second = seed["other"]
    d1 = min(abs(first[0] - r) + abs(first[1] - c) for r, c in gap)
    d2 = min(abs(second[0] - r) + abs(second[1] - c) for r, c in gap)
    assert d1 != d2, "gap side must distinguish one seed color"
    return int(first[2] if d1 < d2 else second[2])


def _paint_stripes(out: np.ndarray, frame: dict, bg: int) -> None:
    r0, c0, r1, c1 = frame["bbox"]
    orientation = frame["seed"]["orientation"]
    a = frame["seed"]["cells"]
    b = frame["seed"]["other"]

    if orientation == "h":
        sr = int(a[0])
        split_c = int(a[1])
        left_color = int(a[2])
        right_color = int(b[2])
        for r in range(r0 + 1, r1):
            swap = abs(r - sr) % 2
            for c in range(c0 + 1, c1):
                if int(out[r, c]) != bg:
                    continue
                on_left = c <= split_c
                if not swap:
                    out[r, c] = left_color if on_left else right_color
                else:
                    out[r, c] = right_color if on_left else left_color
    else:
        sc = int(a[1])
        split_r = int(a[0])
        top_color = int(a[2])
        bottom_color = int(b[2])
        for c in range(c0 + 1, c1):
            swap = abs(c - sc) % 2
            for r in range(r0 + 1, r1):
                if int(out[r, c]) != bg:
                    continue
                on_top = r <= split_r
                if not swap:
                    out[r, c] = top_color if on_top else bottom_color
                else:
                    out[r, c] = bottom_color if on_top else top_color


def _fragment_components(grid: np.ndarray, frames: list[dict], bg: int) -> list[dict]:
    inside_frame = np.zeros(grid.shape, dtype=bool)
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        inside_frame[r0 : r1 + 1, c0 : c1 + 1] = True

    fragments: list[dict] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for comp in _components(grid == color):
            if len(comp) not in (3, 4):
                continue
            if any(inside_frame[r, c] for r, c in comp):
                continue
            if not _is_l_shape(comp):
                continue
            fragments.append({"color": color, "cells": comp, "shape": _normalize(comp)})
    return fragments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use each frame's central seed pair to determine the stripe orientation and the fragment color,
    then translate the matching outside L-fragment into the frame gap.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    frames = _frame_candidates(g, bg)
    fragments = _fragment_components(g, frames, bg)

    by_key: dict[tuple[int, tuple[tuple[int, int], ...]], list[dict]] = defaultdict(list)
    for frag in fragments:
        by_key[(int(frag["color"]), frag["shape"])].append(frag)

    out = g.copy()
    for frame in frames:
        _paint_stripes(out, frame, bg)
        need_color = _required_fragment_color(frame)
        key = (need_color, frame["gap_key"])
        assert by_key[key], "missing matching fragment for frame gap"
        frag = by_key[key].pop()
        for r, c in frag["cells"]:
            out[r, c] = bg
        for r, c in frame["gap"]:
            out[r, c] = frame["color"]

    return out


def _gap_cells(
    r0: int, c0: int, r1: int, c1: int, corner: str, arm_row: int, arm_col: int
) -> list[tuple[int, int]]:
    if corner == "tl":
        pts = {(r0, c0 + k) for k in range(arm_col)} | {(r0 + k, c0) for k in range(arm_row)}
    elif corner == "tr":
        pts = {(r0, c1 - k) for k in range(arm_col)} | {(r0 + k, c1) for k in range(arm_row)}
    elif corner == "bl":
        pts = {(r1, c0 + k) for k in range(arm_col)} | {(r1 - k, c0) for k in range(arm_row)}
    else:
        pts = {(r1, c1 - k) for k in range(arm_col)} | {(r1 - k, c1) for k in range(arm_row)}
    pts_list = sorted(pts)
    assert _is_l_shape(pts_list)
    return pts_list


def _seed_layout(r0: int, c0: int, r1: int, c1: int, orientation: str) -> tuple[tuple[int, int], tuple[int, int]]:
    ih = r1 - r0 - 1
    iw = c1 - c0 - 1
    if orientation == "h":
        sr = r0 + 1 + ih // 2
        left = (sr, c0 + 1 + iw // 2 - 1)
        right = (sr, left[1] + 1)
        return left, right
    top = (r0 + 1 + ih // 2 - 1, c0 + 1 + iw // 2)
    bottom = (top[0] + 1, top[1])
    return top, bottom


def _seed_colors_for_gap(
    orientation: str,
    seed_cells: tuple[tuple[int, int], tuple[int, int]],
    gap: list[tuple[int, int]],
    colors: tuple[int, int],
) -> int:
    (r1, c1), (r2, c2) = seed_cells
    d1 = min(abs(r1 - gr) + abs(c1 - gc) for gr, gc in gap)
    d2 = min(abs(r2 - gr) + abs(c2 - gc) for gr, gc in gap)
    assert d1 != d2
    if orientation == "h":
        return colors[0] if d1 < d2 else colors[1]
    return colors[0] if d1 < d2 else colors[1]


def _paint_frame_input(grid: np.ndarray, frame: dict) -> None:
    r0, c0, r1, c1 = frame["bbox"]
    border = _rect_border_cells(r0, c0, r1, c1) - set(frame["gap"])
    for r, c in border:
        grid[r, c] = FRAME_COLOR
    (p1, p2) = frame["seed_cells"]
    grid[p1] = frame["seed_colors"][0]
    grid[p2] = frame["seed_colors"][1]


def _render_expected(inp: np.ndarray) -> np.ndarray:
    return main(inp)


def _empty_grid(h: int, w: int) -> np.ndarray:
    return np.full((h, w), BG, dtype=int)


def _rect_free(reserved: np.ndarray, r0: int, c0: int, h: int, w: int, margin: int) -> bool:
    rr0 = max(0, r0 - margin)
    cc0 = max(0, c0 - margin)
    rr1 = min(reserved.shape[0], r0 + h + margin)
    cc1 = min(reserved.shape[1], c0 + w + margin)
    return not reserved[rr0:rr1, cc0:cc1].any()


def _mark_rect(reserved: np.ndarray, r0: int, c0: int, h: int, w: int, margin: int) -> None:
    rr0 = max(0, r0 - margin)
    cc0 = max(0, c0 - margin)
    rr1 = min(reserved.shape[0], r0 + h + margin)
    cc1 = min(reserved.shape[1], c0 + w + margin)
    reserved[rr0:rr1, cc0:cc1] = True


def _sample_frame_specs(rng: random.Random, h: int, w: int) -> list[dict] | None:
    count = rng.choice((1, 2, 3))
    specs: list[dict] = []
    for _ in range(count):
        orientation = rng.choice(("h", "v"))
        if orientation == "h":
            ih = rng.choice((3, 5, 7))
            iw = rng.choice((4, 6, 8))
        else:
            ih = rng.choice((4, 6, 8))
            iw = rng.choice((3, 5, 7))
        fh = ih + 2
        fw = iw + 2
        if fh > h - 2 or fw > w - 2:
            return None

        corner = rng.choice(("tl", "tr", "bl", "br"))
        if rng.random() < 0.5:
            arm_row, arm_col = 2, 2
        elif rng.random() < 0.5:
            arm_row, arm_col = 3, 2
        else:
            arm_row, arm_col = 2, 3

        c1, c2 = rng.sample(PALETTE, 2)
        specs.append(
            {
                "orientation": orientation,
                "size": (fh, fw),
                "corner": corner,
                "arm_row": arm_row,
                "arm_col": arm_col,
                "seed_colors": (c1, c2),
            }
        )

    specs.sort(key=lambda spec: spec["size"][0] * spec["size"][1], reverse=True)

    reserved = np.zeros((h, w), dtype=bool)
    placed: list[dict] = []
    for spec in specs:
        fh, fw = spec["size"]
        choices = [
            (r0, c0)
            for r0 in range(1, h - fh)
            for c0 in range(1, w - fw)
            if _rect_free(reserved, r0, c0, fh, fw, margin=1)
        ]
        if not choices:
            return None
        r0, c0 = rng.choice(choices)
        _mark_rect(reserved, r0, c0, fh, fw, margin=1)
        r1 = r0 + fh - 1
        c1 = c0 + fw - 1
        gap = _gap_cells(r0, c0, r1, c1, spec["corner"], spec["arm_row"], spec["arm_col"])
        seed_cells = _seed_layout(r0, c0, r1, c1, spec["orientation"])
        need_color = _seed_colors_for_gap(spec["orientation"], seed_cells, gap, spec["seed_colors"])
        placed.append(
            {
                "bbox": (r0, c0, r1, c1),
                "orientation": spec["orientation"],
                "seed_cells": seed_cells,
                "seed_colors": spec["seed_colors"],
                "gap": gap,
                "need_color": need_color,
            }
        )

    return placed


def _place_fragments(
    rng: random.Random, grid: np.ndarray, frames: list[dict]
) -> list[dict] | None:
    h, w = grid.shape
    reserved = grid != BG
    fragments: list[dict] = []
    order = list(range(len(frames)))
    rng.shuffle(order)
    for idx in order:
        frame = frames[idx]
        shape = frame["gap"]
        norm = _normalize(shape)
        cells_rel = list(norm)
        sh = max(r for r, _ in cells_rel) + 1
        sw = max(c for _, c in cells_rel) + 1
        choices: list[tuple[int, int]] = []
        for r0 in range(1, h - sh):
            for c0 in range(1, w - sw):
                abs_cells = [(r0 + r, c0 + c) for r, c in cells_rel]
                rr0, cc0, rr1, cc1 = _bbox(abs_cells)
                if any(
                    frame_r0 <= rr <= frame_r1 and frame_c0 <= cc <= frame_c1
                    for rr, cc in abs_cells
                    for frame_r0, frame_c0, frame_r1, frame_c1 in [frame2["bbox"] for frame2 in frames]
                ):
                    continue
                if not _rect_free(reserved, rr0, cc0, rr1 - rr0 + 1, cc1 - cc0 + 1, margin=1):
                    continue
                choices.append((r0, c0))
        if not choices:
            return None
        r0, c0 = rng.choice(choices)
        abs_cells = [(r0 + r, c0 + c) for r, c in cells_rel]
        rr0, cc0, rr1, cc1 = _bbox(abs_cells)
        _mark_rect(reserved, rr0, cc0, rr1 - rr0 + 1, cc1 - cc0 + 1, margin=1)
        for r, c in abs_cells:
            grid[r, c] = frame["need_color"]
        fragments.append({"color": frame["need_color"], "cells": abs_cells})
    return fragments


def _build_case(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(18, 29)
    w = rng.randint(18, 29)
    frames = _sample_frame_specs(rng, h, w)
    if not frames:
        return None

    inp = _empty_grid(h, w)
    for frame in frames:
        _paint_frame_input(inp, frame)

    fragments = _place_fragments(rng, inp, frames)
    if fragments is None:
        return None

    try:
        out = _render_expected(inp)
    except Exception:
        return None

    if np.array_equal(inp, out):
        return None

    parsed_frames = _frame_candidates(inp, BG)
    if len(parsed_frames) != len(frames):
        return None
    parsed_frags = _fragment_components(inp, parsed_frames, BG)
    if len(parsed_frags) != len(fragments):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        seed = random.randrange(1 << 30)
        rng = random.Random(seed)
        grid = _build_case(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
