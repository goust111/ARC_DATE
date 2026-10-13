from __future__ import annotations

import random

import numpy as np


BG = 0
MAX_ATTEMPTS = 400
FRAME_COLOR = 5
LABEL_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _scan_runs(row: np.ndarray, offset: int = 0) -> list[tuple[int, int, int]]:
    runs: list[tuple[int, int, int]] = []
    c = 0
    width = len(row)
    while c < width:
        color = int(row[c])
        if color == BG:
            c += 1
            continue
        start = c
        while c + 1 < width and int(row[c + 1]) == color:
            c += 1
        runs.append((start + offset, c + offset, color))
        c += 1
    return runs


def _is_border_color(g: np.ndarray, color: int) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(g == color)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None
    expected = 2 * ((r1 - r0 + 1) + (c1 - c0 + 1)) - 4
    if int(coords.shape[0]) != int(expected):
        return None
    if not np.all(g[r0, c0 : c1 + 1] == color):
        return None
    if not np.all(g[r1, c0 : c1 + 1] == color):
        return None
    if not np.all(g[r0 : r1 + 1, c0] == color):
        return None
    if not np.all(g[r0 : r1 + 1, c1] == color):
        return None
    mask = np.zeros_like(g, dtype=bool)
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    color_mask = g == color
    if not np.array_equal(mask, color_mask):
        return None
    return int(r0), int(c0), int(r1), int(c1)


def _find_frame(g: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int]]:
    candidates: list[tuple[int, tuple[int, int, int, int]]] = []
    for color in np.unique(g):
        color = int(color)
        if color == bg:
            continue
        bbox = _is_border_color(g, color)
        if bbox is None:
            continue
        r0, c0, r1, c1 = bbox
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        candidates.append((area, (color, bbox)))
    assert candidates, "expected one rectangular frame"
    candidates.sort(key=lambda item: item[0], reverse=True)
    top_area = candidates[0][0]
    best = [item[1] for item in candidates if item[0] == top_area]
    assert len(best) == 1, "frame must be unique"
    return best[0]


def _target_slots(
    g: np.ndarray, frame_bbox: tuple[int, int, int, int], frame_color: int
) -> list[dict[str, int]]:
    r0, c0, r1, c1 = frame_bbox
    assert r1 - r0 >= 3 and c1 - c0 >= 5
    row = r1 - 1
    interior = g[row, c0 + 1 : c1]
    runs = _scan_runs(interior, offset=c0 + 1)
    slots = []
    for start, end, color in runs:
        assert color != frame_color
        width = end - start + 1
        slots.append({"color": color, "c0": start, "c1": end, "width": width, "row": row})
    assert 3 <= len(slots) <= 5
    colors = [slot["color"] for slot in slots]
    assert len(colors) == len(set(colors))
    return slots


def _outside_frame(
    r: int, c0: int, c1: int, frame_bbox: tuple[int, int, int, int]
) -> bool:
    fr0, fc0, fr1, fc1 = frame_bbox
    return r < fr0 or r > fr1 or c1 < fc0 or c0 > fc1


def _extract_stamp_above_bar(
    g: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    color: int,
    width: int,
    bg: int,
    frame_color: int,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    h, w = g.shape
    matches: list[tuple[np.ndarray, tuple[int, int, int, int]]] = []
    for r in range(h):
        for c0, c1, run_color in _scan_runs(g[r, :]):
            if run_color != color or c1 - c0 + 1 != width:
                continue
            if not _outside_frame(r, c0, c1, frame_bbox):
                continue
            if r == 0 or r + 1 >= h:
                continue
            if not np.all(g[r + 1, c0 : c1 + 1] == bg):
                continue
            if np.any(g[r - 1, c0 : c1 + 1] == bg) or np.any(
                g[r - 1, c0 : c1 + 1] == frame_color
            ):
                continue

            top = r - 1
            while top - 1 >= 0 and np.all(g[top - 1, c0 : c1 + 1] != bg) and np.all(
                g[top - 1, c0 : c1 + 1] != frame_color
            ):
                top -= 1
            patch = g[top:r, c0 : c1 + 1]
            if patch.shape[0] < 2:
                continue
            if c0 > 0 and np.any(g[top : r + 1, c0 - 1] != bg):
                continue
            if c1 + 1 < w and np.any(g[top : r + 1, c1 + 1] != bg):
                continue
            if top > 0 and np.any(g[top - 1, c0 : c1 + 1] != bg):
                continue
            matches.append((patch.copy(), (top, c0, r, c1)))

    assert len(matches) == 1, f"expected one source stamp for color {color}"
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Reorder the labeled stamp patches into the frame slots named by matching color bars.

    Input structure:
    - The background is black.
    - Several multicolor rectangular stamps sit in the upper part of the grid, and each
      stamp has a one-cell-high monochrome bar directly beneath it with the same width.
    - A one-cell-thick rectangular frame sits lower in the grid; along the frame's bottom
      interior row are separated colored bars whose colors match the source bars.

    Rule:
    - Use the bar colors to pair every source stamp with exactly one bar inside the frame.
    - Erase each source stamp and its source bar back to the background color.
    - Paste the unchanged stamp directly above the matched frame bar so the stamp's bottom
      edge touches the bar and the stamp stays horizontally aligned to the bar width.

    Key insight:
    - The source bars are identity tags, while the bars inside the frame are ordered slots,
      so the output is determined by color matching plus the frame's left-to-right bar order.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    frame_color, frame_bbox = _find_frame(g, bg)
    slots = _target_slots(g, frame_bbox, frame_color)

    out = g.copy()
    for slot in slots:
        stamp, (sr0, sc0, bar_r, sc1) = _extract_stamp_above_bar(
            g,
            frame_bbox,
            slot["color"],
            slot["width"],
            bg,
            frame_color,
        )
        out[sr0 : bar_r + 1, sc0 : sc1 + 1] = bg
        dest_r0 = slot["row"] - stamp.shape[0]
        dest_r1 = slot["row"] - 1
        assert dest_r0 >= frame_bbox[0] + 1
        assert dest_r1 < frame_bbox[2]
        out[dest_r0 : dest_r1 + 1, slot["c0"] : slot["c1"] + 1] = stamp
    return out


def _random_stamp(h: int, w: int, label_colors: list[int]) -> np.ndarray:
    body_pool = [c for c in LABEL_COLORS if c not in label_colors]
    color_count = random.randint(2, min(4, len(body_pool)))
    colors = random.sample(body_pool, color_count)
    mode = random.choice(["noise", "hstripe", "vstripe", "checker", "blocks"])
    stamp = np.zeros((h, w), dtype=int)

    if mode == "noise":
        stamp[:, :] = np.random.choice(colors, size=(h, w))
    elif mode == "hstripe":
        chosen = [random.choice(colors) for _ in range(h)]
        for r in range(h):
            stamp[r, :] = chosen[r]
    elif mode == "vstripe":
        chosen = [random.choice(colors) for _ in range(w)]
        for c in range(w):
            stamp[:, c] = chosen[c]
    elif mode == "checker":
        shift = random.randint(0, len(colors) - 1)
        for r in range(h):
            for c in range(w):
                stamp[r, c] = colors[(r + c + shift) % len(colors)]
    else:
        split_r = random.randint(1, h - 1)
        split_c = random.randint(1, w - 1)
        fills = [random.choice(colors) for _ in range(4)]
        stamp[:split_r, :split_c] = fills[0]
        stamp[:split_r, split_c:] = fills[1]
        stamp[split_r:, :split_c] = fills[2]
        stamp[split_r:, split_c:] = fills[3]

    if len(np.unique(stamp)) == 1:
        r = random.randrange(h)
        c = random.randrange(w)
        alternatives = [color for color in colors if color != int(stamp[r, c])]
        stamp[r, c] = random.choice(alternatives)
    return stamp


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _can_place(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    if r0 < 0 or c0 < 0 or r1 >= occupied.shape[0] or c1 >= occupied.shape[1]:
        return False
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(occupied.shape[0] - 1, r1 + 1)
    cc1 = min(occupied.shape[1] - 1, c1 + 1)
    return not occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any()


def _try_place_sources(
    grid: np.ndarray,
    stamps: list[np.ndarray],
    label_colors: list[int],
    frame_bbox: tuple[int, int, int, int],
    gap_to_frame: int,
) -> list[tuple[int, int]] | None:
    frame_r0 = frame_bbox[0]
    occupied = np.zeros_like(grid, dtype=bool)
    fr0, fc0, fr1, fc1 = frame_bbox
    occupied[fr0 : fr1 + 1, fc0 : fc1 + 1] = True
    placements: list[tuple[int, int]] = []
    order = list(range(len(stamps)))
    random.shuffle(order)
    for idx in order:
        stamp = stamps[idx]
        sh, sw = stamp.shape
        object_h = sh + 1
        max_row = frame_r0 - gap_to_frame - object_h
        if max_row < 1:
            return None
        placed = False
        for _ in range(120):
            r0 = random.randint(1, max_row)
            c0 = random.randint(1, grid.shape[1] - sw - 2)
            if not _can_place(occupied, r0, c0, object_h, sw):
                continue
            grid[r0 : r0 + sh, c0 : c0 + sw] = stamp
            grid[r0 + sh, c0 : c0 + sw] = label_colors[idx]
            occupied[r0 : r0 + object_h, c0 : c0 + sw] = True
            placements.append((idx, c0))
            placed = True
            break
        if not placed:
            return None
    placements.sort(key=lambda item: item[1])
    return placements


def _target_order(colors: list[int]) -> list[int]:
    indices = list(range(len(colors)))
    while True:
        perm = random.sample(indices, len(indices))
        if perm != indices:
            return perm


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    n = random.randint(3, 5)
    label_colors = random.sample(LABEL_COLORS, n)
    dims = [(random.randint(3, 5), random.randint(3, 5)) for _ in range(n)]
    widths = [w for _h, w in dims]
    heights = [h for h, _w in dims]

    inner_left = random.randint(1, 3)
    inner_right = random.randint(1, 3)
    inner_w = inner_left + inner_right + sum(widths) + (n - 1)
    if inner_w > 28:
        return None
    inner_h = max(heights) + 1 + random.randint(0, 2)
    frame_h = inner_h + 2
    frame_w = inner_w + 2

    top_margin = random.randint(1, 3)
    gap_to_frame = random.randint(2, 4)
    bottom_margin = random.randint(1, 3)
    min_h = top_margin + max(heights) + 1 + gap_to_frame + frame_h + bottom_margin
    if min_h > 30:
        return None
    H = random.randint(min_h, min(30, min_h + 4))

    min_w = max(frame_w + 2, 18)
    if min_w > 30:
        return None
    W = random.randint(min_w, 30)
    frame_c0 = random.randint(1, W - frame_w - 1)
    frame_r0_low = top_margin + max(heights) + 1 + gap_to_frame
    frame_r0_high = H - bottom_margin - frame_h
    if frame_r0_low > frame_r0_high:
        return None
    frame_r0 = random.randint(frame_r0_low, frame_r0_high)
    frame_bbox = (frame_r0, frame_c0, frame_r0 + frame_h - 1, frame_c0 + frame_w - 1)

    base = np.full((H, W), BG, dtype=int)
    _draw_frame(base, frame_bbox, FRAME_COLOR)

    stamps = [_random_stamp(h, w, label_colors) for h, w in dims]
    perm = _target_order(label_colors)
    bar_row = frame_bbox[2] - 1
    cursor = frame_c0 + 1 + inner_left
    slot_starts: dict[int, int] = {}
    for idx in perm:
        sw = stamps[idx].shape[1]
        base[bar_row, cursor : cursor + sw] = label_colors[idx]
        slot_starts[idx] = cursor
        cursor += sw + 1

    expected = base.copy()
    for idx, stamp in enumerate(stamps):
        sh, sw = stamp.shape
        dest_c0 = slot_starts[idx]
        dest_r0 = bar_row - sh
        expected[dest_r0 : dest_r0 + sh, dest_c0 : dest_c0 + sw] = stamp

    inp = base.copy()
    source_order = _try_place_sources(inp, stamps, label_colors, frame_bbox, gap_to_frame)
    if source_order is None:
        return None
    source_colors = [label_colors[idx] for idx, _c0 in source_order]
    target_colors = [label_colors[idx] for idx in perm]
    if source_colors == target_colors:
        return None
    if len({int(pos[1]) for pos in source_order}) != len(source_order):
        return None
    if np.array_equal(main(inp), expected):
        return inp, expected
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        case = _build_case()
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate valid input")
