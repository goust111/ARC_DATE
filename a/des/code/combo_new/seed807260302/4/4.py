from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _components_of_color(g: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for sr, sc in np.argwhere(g == color):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        q: deque[tuple[int, int]] = deque([(sr, sc)])
        pts: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            pts.append((r, c))
            for dr, dc in NEIGHBORS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(np.array(pts, dtype=int))

    return comps


def _rect_frame_bbox(coords: np.ndarray) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None

    perimeter = 2 * h + 2 * w - 4
    if int(coords.shape[0]) != perimeter:
        return None

    rows = coords[:, 0]
    cols = coords[:, 1]
    on_border = (rows == r0) | (rows == r1) | (cols == c0) | (cols == c1)
    if not bool(np.all(on_border)):
        return None
    return r0, c0, r1, c1


def _find_frames(g: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    for raw_color in np.unique(g):
        color = int(raw_color)
        if color == bg:
            continue
        for coords in _components_of_color(g, color):
            box = _rect_frame_bbox(coords)
            if box is None:
                continue
            r0, c0, r1, c1 = box
            frames.append(
                {
                    "color": color,
                    "bbox": box,
                    "height": r1 - r0 + 1,
                    "width": c1 - c0 + 1,
                }
            )
    return frames


def _valid_pipe_values(values: list[int], bg: int) -> bool:
    if len(values) == 0 or len(values) % 2 != 0:
        return False
    if any(v == bg for v in values):
        return False
    return all(values[i] != values[i + 1] for i in range(0, len(values), 2))


def _horizontal_pipe(g: np.ndarray, rows: range, cols: range, bg: int) -> list[int] | None:
    rows_list = list(rows)
    cols_list = list(cols)
    if not rows_list or not cols_list or len(cols_list) % 2 != 0:
        return None

    candidates: list[list[int]] = []
    for r in rows_list:
        values = [int(g[r, c]) for c in cols_list]
        if _valid_pipe_values(values, bg):
            candidates.append(values)

    block = g[min(rows_list) : max(rows_list) + 1, min(cols_list) : max(cols_list) + 1]
    if len(candidates) != 1 or int(np.count_nonzero(block != bg)) != len(cols_list):
        return None
    return candidates[0]


def _vertical_pipe(g: np.ndarray, rows: range, cols: range, bg: int) -> list[int] | None:
    rows_list = list(rows)
    cols_list = list(cols)
    if not rows_list or not cols_list or len(rows_list) % 2 != 0:
        return None

    candidates: list[list[int]] = []
    for c in cols_list:
        values = [int(g[r, c]) for r in rows_list]
        if _valid_pipe_values(values, bg):
            candidates.append(values)

    block = g[min(rows_list) : max(rows_list) + 1, min(cols_list) : max(cols_list) + 1]
    if len(candidates) != 1 or int(np.count_nonzero(block != bg)) != len(rows_list):
        return None
    return candidates[0]


def _pipe_sequence_between(g: np.ndarray, a: dict, b: dict, bg: int) -> list[int] | None:
    ar0, ac0, ar1, ac1 = a["bbox"]
    br0, bc0, br1, bc1 = b["bbox"]

    if ac1 < bc0:
        rows = range(max(ar0, br0), min(ar1, br1) + 1)
        cols = range(ac1 + 1, bc0)
        return _horizontal_pipe(g, rows, cols, bg)
    if bc1 < ac0:
        rows = range(max(ar0, br0), min(ar1, br1) + 1)
        cols = range(bc1 + 1, ac0)
        seq = _horizontal_pipe(g, rows, cols, bg)
        return None if seq is None else list(reversed(seq))

    if ar1 < br0:
        rows = range(ar1 + 1, br0)
        cols = range(max(ac0, bc0), min(ac1, bc1) + 1)
        return _vertical_pipe(g, rows, cols, bg)
    if br1 < ar0:
        rows = range(br1 + 1, ar0)
        cols = range(max(ac0, bc0), min(ac1, bc1) + 1)
        seq = _vertical_pipe(g, rows, cols, bg)
        return None if seq is None else list(reversed(seq))

    return None


def _interior(frame: dict) -> tuple[slice, slice]:
    r0, c0, r1, c1 = frame["bbox"]
    return slice(r0 + 1, r1), slice(c0 + 1, c1)


def _find_structure(g: np.ndarray, bg: int) -> tuple[dict, dict, list[int]]:
    frames = _find_frames(g, bg)
    candidates: list[tuple[dict, dict, list[int]]] = []

    for i, a in enumerate(frames):
        for b in frames[i + 1 :]:
            if a["color"] == b["color"]:
                continue
            if a["height"] != b["height"] or a["width"] != b["width"]:
                continue
            seq = _pipe_sequence_between(g, a, b, bg)
            if seq is not None:
                candidates.append((a, b, seq))

    assert len(candidates) == 1, "expected one pair of same-size framed regions joined by one pipe"
    return candidates[0]


def _mapping_from_pipe(values: list[int]) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for i in range(0, len(values), 2):
        src = int(values[i])
        dst = int(values[i + 1])
        assert src != dst, "each pipe pair must contain two different colors"
        if src in mapping:
            assert mapping[src] == dst, "duplicate pipe keys must agree"
        mapping[src] = dst
    return mapping


def _apply_mapping(patch: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    out = patch.copy()
    for src, dst in mapping.items():
        out[patch == src] = dst
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy the richer framed interior into the poorer framed interior using pipe-coded color substitutions.

    Input structure:
    - The background is black.
    - Two same-size rectangular regions have one-pixel-thick borders in two different colors.
    - A single horizontal or vertical one-pixel-wide pipe connects the two frames; from either end it is read as adjacent color pairs.
    - One frame interior contains a richer multicolor pattern, while the other is empty or sparse.

    Rule:
    - Identify the source frame as the one whose interior has more non-background pixels; the other frame is the target.
    - Read the pipe from the source frame toward the target frame, grouping every two adjacent pipe cells as original color then replacement color.
    - Copy the entire source interior into the target interior, replacing each source pixel color according to those pipe pairs and leaving unmapped colors unchanged.
    - Keep the source frame, target border, pipe, and all non-target-interior cells unchanged.

    Key insight:
    The pipe is a directional legend for recoloring the copied pattern, and its direction is determined only after the richer source region is found.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = BG

    frame_a, frame_b, seq_ab = _find_structure(g, bg)
    ia = _interior(frame_a)
    ib = _interior(frame_b)
    count_a = int(np.count_nonzero(g[ia] != bg))
    count_b = int(np.count_nonzero(g[ib] != bg))
    assert count_a != count_b, "source and target interiors must have different richness"

    if count_a > count_b:
        source, target, pipe_values = frame_a, frame_b, seq_ab
    else:
        source, target, pipe_values = frame_b, frame_a, list(reversed(seq_ab))

    src_patch = g[_interior(source)]
    mapping = _mapping_from_pipe(pipe_values)
    recolored = _apply_mapping(src_patch, mapping)

    out = g.copy()
    out[_interior(target)] = recolored
    return out


# ---------------- generator ----------------

GEN_ATTEMPTS = 500
GEN_COLORS = list(range(1, 10))


def _paint_frame(grid: np.ndarray, r0: int, c0: int, h: int, w: int, color: int) -> None:
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color


def _place_patch(grid: np.ndarray, frame: dict, patch: np.ndarray) -> None:
    rs, cs = _interior(frame)
    grid[rs, cs] = patch


def _source_patch(ih: int, iw: int, keys: list[int], palette: list[int]) -> np.ndarray:
    patch = np.zeros((ih, iw), dtype=int)
    cells = [(r, c) for r in range(ih) for c in range(iw)]
    random.shuffle(cells)

    for color, (r, c) in zip(keys, cells):
        patch[r, c] = color

    area = ih * iw
    min_count = max(len(keys) + 2, area // 3)
    max_count = max(min_count, int(area * 0.72))
    target_count = random.randint(min_count, min(area, max_count))
    for r, c in cells[len(keys) : target_count]:
        patch[r, c] = random.choice(palette)

    # Add an occasional short bar to make some patterns less speckled.
    if ih >= 4 and iw >= 4 and random.random() < 0.55:
        color = random.choice(palette)
        if random.random() < 0.5:
            r = random.randrange(ih)
            c0 = random.randrange(0, iw - 2)
            patch[r, c0 : c0 + random.randint(2, min(4, iw - c0))] = color
        else:
            c = random.randrange(iw)
            r0 = random.randrange(0, ih - 2)
            patch[r0 : r0 + random.randint(2, min(4, ih - r0)), c] = color

    return patch


def _target_patch(ih: int, iw: int, palette: list[int], source_count: int) -> np.ndarray:
    patch = np.zeros((ih, iw), dtype=int)
    max_noise = min(2, ih * iw, source_count - 1)
    if max_noise <= 0 or random.random() < 0.55:
        return patch

    count = random.randint(1, max_noise)
    cells = random.sample([(r, c) for r in range(ih) for c in range(iw)], count)
    for r, c in cells:
        patch[r, c] = random.choice(palette)
    return patch


def _frame_dict(color: int, r0: int, c0: int, h: int, w: int) -> dict:
    return {
        "color": color,
        "bbox": (r0, c0, r0 + h - 1, c0 + w - 1),
        "height": h,
        "width": w,
    }


def _build_expected(grid: np.ndarray, source: dict, target: dict, mapping: dict[int, int]) -> np.ndarray:
    out = grid.copy()
    out[_interior(target)] = _apply_mapping(grid[_interior(source)], mapping)
    return out


def _gen_attempt() -> np.ndarray | None:
    frame_h = random.randint(5, 10)
    frame_w = random.randint(5, 10)
    ih = frame_h - 2
    iw = frame_w - 2

    frame_low_color, frame_high_color = random.sample(GEN_COLORS, 2)
    available = [c for c in GEN_COLORS if c not in (frame_low_color, frame_high_color)]
    pair_count = random.randint(1, min(3, len(available) // 2))
    pipe_source_to_target = random.sample(available, 2 * pair_count)
    keys = [pipe_source_to_target[i] for i in range(0, len(pipe_source_to_target), 2)]
    values = [pipe_source_to_target[i] for i in range(1, len(pipe_source_to_target), 2)]
    mapping = {src: dst for src, dst in zip(keys, values)}
    extra_colors = [c for c in available if c not in keys]
    pattern_palette = keys + random.sample(extra_colors, random.randint(1, min(3, len(extra_colors))))

    gap = len(pipe_source_to_target)
    orientation = random.choice(("horizontal", "vertical"))
    source_is_low = random.choice((False, True))
    pipe_low_to_high = pipe_source_to_target if source_is_low else list(reversed(pipe_source_to_target))

    if orientation == "horizontal":
        min_h = frame_h
        min_w = 2 * frame_w + gap
        if min_w > 30:
            return None
        height = random.randint(min_h, 30)
        width = random.randint(min_w, 30)
        r0 = random.randint(0, height - frame_h)
        c0 = random.randint(0, width - min_w)
        low = _frame_dict(frame_low_color, r0, c0, frame_h, frame_w)
        high = _frame_dict(frame_high_color, r0, c0 + frame_w + gap, frame_h, frame_w)
        pipe_r = random.randint(r0 + 1, r0 + frame_h - 2)
        pipe_cells = [(pipe_r, c) for c in range(c0 + frame_w, c0 + frame_w + gap)]
    else:
        min_h = 2 * frame_h + gap
        min_w = frame_w
        if min_h > 30:
            return None
        height = random.randint(min_h, 30)
        width = random.randint(min_w, 30)
        r0 = random.randint(0, height - min_h)
        c0 = random.randint(0, width - frame_w)
        low = _frame_dict(frame_low_color, r0, c0, frame_h, frame_w)
        high = _frame_dict(frame_high_color, r0 + frame_h + gap, c0, frame_h, frame_w)
        pipe_c = random.randint(c0 + 1, c0 + frame_w - 2)
        pipe_cells = [(r, pipe_c) for r in range(r0 + frame_h, r0 + frame_h + gap)]

    source = low if source_is_low else high
    target = high if source_is_low else low

    grid = np.zeros((height, width), dtype=int)
    _paint_frame(grid, *low["bbox"][:2], frame_h, frame_w, low["color"])
    _paint_frame(grid, *high["bbox"][:2], frame_h, frame_w, high["color"])
    for (r, c), color in zip(pipe_cells, pipe_low_to_high):
        grid[r, c] = color

    src_patch = _source_patch(ih, iw, keys, pattern_palette)
    src_count = int(np.count_nonzero(src_patch != BG))
    tgt_patch = _target_patch(ih, iw, pattern_palette, src_count)
    if src_count <= int(np.count_nonzero(tgt_patch != BG)):
        return None

    _place_patch(grid, source, src_patch)
    _place_patch(grid, target, tgt_patch)
    expected = _build_expected(grid, source, target, mapping)

    try:
        actual = main(grid)
    except Exception:
        return None
    if np.array_equal(actual, expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid pipe-recolor copy input")
