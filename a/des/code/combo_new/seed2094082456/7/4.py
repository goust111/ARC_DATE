from __future__ import annotations

import random

import numpy as np


def _mode_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0] or seen[r0, c0]:
                continue
            stack = [(int(r0), int(c0))]
            seen[r0, c0] = True
            cells: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                cells.append((int(r), int(c)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    rr = r + dr
                    cc = c + dc
                    if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        stack.append((int(rr), int(cc)))
            comps.append(np.array(cells, dtype=int))
    return comps


def _tight_crop(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _count_holes(mask: np.ndarray) -> int:
    h, w = mask.shape
    bg = ~mask
    seen = np.zeros((h, w), dtype=bool)
    holes = 0
    for r0 in range(h):
        for c0 in range(w):
            if not bg[r0, c0] or seen[r0, c0]:
                continue
            stack = [(int(r0), int(c0))]
            seen[r0, c0] = True
            touches_border = False
            while stack:
                r, c = stack.pop()
                if r == 0 or c == 0 or r == h - 1 or c == w - 1:
                    touches_border = True
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    rr = r + dr
                    cc = c + dc
                    if 0 <= rr < h and 0 <= cc < w and bg[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        stack.append((int(rr), int(cc)))
            if not touches_border:
                holes += 1
    return int(holes)


def _legend_colors(grid: np.ndarray, bg: int) -> list[int]:
    legend_rows = grid[-2:, :]
    active = (legend_rows[0] == legend_rows[1]) & (legend_rows[0] != int(bg))
    segments: list[tuple[int, int]] = []
    start = None
    for c in range(grid.shape[1]):
        if active[c] and start is None:
            start = c
        elif not active[c] and start is not None:
            segments.append((int(start), int(c)))
            start = None
    if start is not None:
        segments.append((int(start), int(grid.shape[1])))

    candidates: list[tuple[int, list[int]]] = []
    for c0, c1 in segments:
        seq = [int(v) for v in legend_rows[0, c0:c1].tolist()]
        if len(seq) % 2 == 0:
            continue
        colors = seq[0::2]
        separators = seq[1::2]
        if len(colors) == 0 or len(colors) != len(set(colors)):
            continue
        if separators and len(set(separators)) != 1:
            continue
        candidates.append((len(colors), colors))

    assert candidates, "expected a bottom legend strip with alternating color and separator columns"
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each monochrome closed object according to the number of holes it encloses.

    Input structure:
    - The background is black.
    - The bottom two rows contain a horizontal legend strip: 1-column color blocks separated by 1-column grey dividers.
    - Above the legend are several 4-connected objects, all drawn in the same source color; objects may contain zero or more enclosed black holes.

    Rule:
    - Read the legend colors from left to right, ignoring the grey separator columns.
    - Remove the bottom legend strip from the canvas.
    - For each foreground object in the remaining area, count its enclosed background holes inside the object's tight bounding box.
    - Recolor that object's pixels with the (hole_count + 1)-th legend color, leaving the holes themselves black.

    Key insight:
    The relevant feature is topological: objects are indexed by how many enclosed holes they contain, not by their size, shape, or position.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 3
    bg = _mode_color(g)
    palette = _legend_colors(g, bg)

    out = g[:-2, :].copy()
    comps = _components4(out != int(bg))
    for pts in comps:
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        mask = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
        mask[pts[:, 0] - int(r0), pts[:, 1] - int(c0)] = True
        holes = _count_holes(mask)
        assert holes < len(palette), "legend must provide a color for every observed hole count"
        out[pts[:, 0], pts[:, 1]] = int(palette[holes])
    return out


# --------------- generator ---------------

BG = 0
SEP = 5
GEN_RETRIES = 400
SHAPE_RETRIES = 240
PLACEMENT_RETRIES = 140
LEGEND_LEN_RANGE = (3, 5)
MAX_OBJECTS = 5


def _random_transform(mask: np.ndarray) -> np.ndarray:
    out = np.rot90(mask, random.randint(0, 3))
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = np.flipud(out)
    return np.array(out, dtype=bool)


def _carve_holes(mask: np.ndarray, hole_count: int) -> bool:
    if hole_count == 0:
        return True

    h, w = mask.shape
    forbidden = np.zeros((h, w), dtype=bool)
    hole_shapes = [(1, 1), (1, 2), (2, 1), (2, 2)]
    for _ in range(hole_count):
        placed = False
        for _ in range(120):
            hh, ww = random.choice(hole_shapes)
            if h - hh - 2 < 1 or w - ww - 2 < 1:
                continue
            r = random.randint(1, h - hh - 1)
            c = random.randint(1, w - ww - 1)
            r0 = max(0, r - 1)
            r1 = min(h, r + hh + 1)
            c0 = max(0, c - 1)
            c1 = min(w, c + ww + 1)
            if forbidden[r0:r1, c0:c1].any():
                continue
            mask[r : r + hh, c : c + ww] = False
            forbidden[r0:r1, c0:c1] = True
            placed = True
            break
        if not placed:
            return False
    return True


def _carve_notches(mask: np.ndarray, force_notch: bool) -> None:
    h, w = mask.shape
    notch_count = random.randint(1 if force_notch else 0, 3)
    for _ in range(notch_count):
        side = random.choice("TBLR")
        if side in "TB" and w < 4:
            continue
        if side in "LR" and h < 4:
            continue
        if side in "TB":
            depth = random.randint(1, 2 if h >= 7 else 1)
            span = random.randint(1, max(1, w - 2))
            c0 = random.randint(0, w - span)
            if side == "T":
                mask[:depth, c0 : c0 + span] = False
            else:
                mask[h - depth :, c0 : c0 + span] = False
        else:
            depth = random.randint(1, 2 if w >= 7 else 1)
            span = random.randint(1, max(1, h - 2))
            r0 = random.randint(0, h - span)
            if side == "L":
                mask[r0 : r0 + span, :depth] = False
            else:
                mask[r0 : r0 + span, w - depth :] = False


def _add_protrusions(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    canvas = np.zeros((h + 4, w + 4), dtype=bool)
    canvas[2 : 2 + h, 2 : 2 + w] = mask
    for _ in range(random.randint(0, 2)):
        coords = np.argwhere(canvas)
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        side = random.choice("TBLR")
        if side in "TB":
            span = random.randint(1, max(1, (int(c1 - c0) + 1) // 2))
            c = random.randint(int(c0), int(c1) - span + 1)
            thickness = random.randint(1, 2)
            if side == "T" and r0 >= thickness:
                canvas[int(r0) - thickness : int(r0), c : c + span] = True
            elif side == "B" and r1 + thickness < canvas.shape[0]:
                canvas[int(r1) + 1 : int(r1) + thickness + 1, c : c + span] = True
        else:
            span = random.randint(1, max(1, (int(r1 - r0) + 1) // 2))
            r = random.randint(int(r0), int(r1) - span + 1)
            thickness = random.randint(1, 2)
            if side == "L" and c0 >= thickness:
                canvas[r : r + span, int(c0) - thickness : int(c0)] = True
            elif side == "R" and c1 + thickness < canvas.shape[1]:
                canvas[r : r + span, int(c1) + 1 : int(c1) + thickness + 1] = True
    return _tight_crop(canvas)


def _generate_object(hole_count: int) -> np.ndarray:
    for _ in range(SHAPE_RETRIES):
        if hole_count == 0:
            h = random.randint(4, 8)
            w = random.randint(4, 9)
        else:
            cols = int(np.ceil(np.sqrt(hole_count)))
            rows = int(np.ceil(hole_count / cols))
            h_min = 2 * rows + 3
            w_min = 2 * cols + 3
            h = random.randint(h_min, min(12, h_min + 3))
            w = random.randint(w_min, min(14, w_min + 4))

        mask = np.ones((h, w), dtype=bool)
        if not _carve_holes(mask, hole_count):
            continue
        _carve_notches(mask, force_notch=(hole_count == 0))
        mask = _add_protrusions(mask)
        mask = _random_transform(mask)
        mask = _tight_crop(mask)

        if len(_components4(mask)) != 1:
            continue
        if _count_holes(mask) != hole_count:
            continue
        if mask.shape[0] > 12 or mask.shape[1] > 14:
            continue
        if int(mask.sum()) < max(8, hole_count * 4 + 6):
            continue
        return mask

    raise RuntimeError(f"failed to generate object with {hole_count} holes")


def _sample_hole_counts(legend_len: int) -> list[int]:
    max_holes = legend_len - 1
    distinct_target = random.randint(2, min(4, legend_len))
    chosen = {0, max_holes}
    middle = list(range(1, max_holes))
    random.shuffle(middle)
    while len(chosen) < distinct_target and middle:
        chosen.add(middle.pop())
    counts = list(chosen)
    total_objects = min(MAX_OBJECTS, len(counts) + random.randint(0, 2))
    while len(counts) < total_objects:
        counts.append(random.choice(counts))
    random.shuffle(counts)
    return [int(v) for v in counts]


def _dimension_choices(masks: list[np.ndarray], legend_span: int) -> list[tuple[int, int]]:
    padded_area = sum((mask.shape[0] + 2) * (mask.shape[1] + 2) for mask in masks)
    object_area = sum(int(mask.sum()) for mask in masks)
    max_h = max(mask.shape[0] for mask in masks)
    max_w = max(mask.shape[1] for mask in masks)
    min_h = max(9, int(max_h) + 1)
    min_w = max(legend_span, int(max_w) + 1, 12)
    choices: list[tuple[int, int]] = []
    for h in range(min_h, 29):
        for w in range(min_w, 31):
            if h * w >= padded_area + 16 and h * w >= 2 * object_area + 12:
                choices.append((int(h), int(w)))
    return choices


def _can_place(mask: np.ndarray, blocked: np.ndarray, top: int, left: int) -> bool:
    mh, mw = mask.shape
    return not np.any(blocked[top : top + mh, left : left + mw] & mask)


def _mark_blocked(mask: np.ndarray, blocked: np.ndarray, top: int, left: int) -> None:
    mh, mw = mask.shape
    for r, c in np.argwhere(mask):
        rr = top + int(r)
        cc = left + int(c)
        r0 = max(0, rr - 1)
        r1 = min(blocked.shape[0], rr + 2)
        c0 = max(0, cc - 1)
        c1 = min(blocked.shape[1], cc + 2)
        blocked[r0:r1, c0:c1] = True


def _place_objects(masks: list[np.ndarray], out_h: int, out_w: int) -> list[tuple[int, int]] | None:
    order = sorted(range(len(masks)), key=lambda i: masks[i].shape[0] * masks[i].shape[1], reverse=True)
    placements: list[tuple[int, int] | None] = [None] * len(masks)
    blocked = np.zeros((out_h, out_w), dtype=bool)

    for idx in order:
        mask = masks[idx]
        mh, mw = mask.shape
        candidates: list[tuple[int, int]] = []
        for top in range(out_h - mh + 1):
            for left in range(out_w - mw + 1):
                if _can_place(mask, blocked, top, left):
                    candidates.append((int(top), int(left)))
        if not candidates:
            return None
        top, left = random.choice(candidates)
        placements[idx] = (int(top), int(left))
        _mark_blocked(mask, blocked, top, left)

    assert all(placement is not None for placement in placements)
    return [
        (int(placement[0]), int(placement[1]))
        for placement in placements
        if placement is not None
    ]


def _render_case(
    masks: list[np.ndarray],
    placements: list[tuple[int, int]],
    counts: list[int],
    object_color: int,
    legend_colors: list[int],
    out_h: int,
    out_w: int,
) -> tuple[np.ndarray, np.ndarray]:
    expected = np.full((out_h, out_w), BG, dtype=int)
    inp = np.full((out_h + 2, out_w), BG, dtype=int)

    for mask, (top, left), count in zip(masks, placements, counts, strict=False):
        mh, mw = mask.shape
        inp_region = inp[top : top + mh, left : left + mw]
        out_region = expected[top : top + mh, left : left + mw]
        inp_region[mask] = int(object_color)
        out_region[mask] = int(legend_colors[count])

    seq: list[int] = []
    for i, color in enumerate(legend_colors):
        if i:
            seq.append(int(SEP))
        seq.append(int(color))
    legend = np.array(seq, dtype=int)
    left = random.randint(0, out_w - len(legend))
    inp[out_h : out_h + 2, left : left + len(legend)] = legend
    return inp, expected


def _generation_attempt() -> np.ndarray | None:
    legend_len = random.randint(*LEGEND_LEN_RANGE)
    counts = _sample_hole_counts(legend_len)

    palette_pool = [1, 2, 3, 4, 6, 7, 8, 9]
    object_color = random.choice(palette_pool)
    legend_pool = [c for c in palette_pool if c != object_color]
    legend_colors = random.sample(legend_pool, legend_len)

    try:
        masks = [_generate_object(count) for count in counts]
    except RuntimeError:
        return None
    legend_span = 2 * legend_len - 1
    choices = _dimension_choices(masks, legend_span)
    if not choices:
        return None

    for _ in range(PLACEMENT_RETRIES):
        out_h, out_w = random.choice(choices)
        placements = _place_objects(masks, out_h, out_w)
        if placements is None:
            continue
        inp, expected = _render_case(
            masks=masks,
            placements=placements,
            counts=counts,
            object_color=object_color,
            legend_colors=legend_colors,
            out_h=out_h,
            out_w=out_w,
        )
        if np.array_equal(main(inp), expected):
            return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generation_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC sample")
