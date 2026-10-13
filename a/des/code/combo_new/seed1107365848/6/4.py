from __future__ import annotations

import random
from collections import deque

import numpy as np


DIRS: dict[str, tuple[int, int]] = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

ARROW_PATTERNS: dict[str, dict[str, object]] = {
    "up": {
        "shape": (3, 5),
        "cells": {
            (0, 2),
            (1, 1),
            (1, 2),
            (1, 3),
            (2, 0),
            (2, 1),
            (2, 2),
            (2, 3),
            (2, 4),
        },
        "apex": (0, 2),
    },
    "down": {
        "shape": (3, 5),
        "cells": {
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (1, 1),
            (1, 2),
            (1, 3),
            (2, 2),
        },
        "apex": (2, 2),
    },
    "left": {
        "shape": (5, 3),
        "cells": {
            (2, 0),
            (1, 1),
            (2, 1),
            (3, 1),
            (0, 2),
            (1, 2),
            (2, 2),
            (3, 2),
            (4, 2),
        },
        "apex": (2, 0),
    },
    "right": {
        "shape": (5, 3),
        "cells": {
            (0, 0),
            (1, 0),
            (2, 0),
            (3, 0),
            (4, 0),
            (1, 1),
            (2, 1),
            (3, 1),
            (2, 2),
        },
        "apex": (2, 2),
    },
}

FRAME_COLORS = tuple(range(1, 10))
BG = 0
GEN_ATTEMPTS = 400


def _mode_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[counts.argmax()])


def _component_from(mask: np.ndarray, start: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = mask.shape
    sr, sc = start
    if not mask[sr, sc]:
        return set()
    q = deque([(sr, sc)])
    seen = {(sr, sc)}
    while q:
        r, c = q.popleft()
        for dr, dc in DIRS.values():
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return seen


def _side_positions(
    bbox: tuple[int, int, int, int], side: str
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    if side == "up":
        return [(r0, c) for c in range(c0 + 1, c1)]
    if side == "down":
        return [(r1, c) for c in range(c0 + 1, c1)]
    if side == "left":
        return [(r, c0) for r in range(r0 + 1, r1)]
    return [(r, c1) for r in range(r0 + 1, r1)]


def _strict_gap_segment(
    comp: set[tuple[int, int]], positions: list[tuple[int, int]]
) -> list[tuple[int, int]] | None:
    if len(positions) < 3:
        return None
    missing = [idx for idx, pos in enumerate(positions) if pos not in comp]
    if not missing:
        return None
    if missing != list(range(missing[0], missing[-1] + 1)):
        return None
    if missing[0] == 0 or missing[-1] == len(positions) - 1:
        return None
    return positions[missing[0] : missing[-1] + 1]


def _frame_record(
    grid: np.ndarray,
    color: int,
    direction: str,
    arrow_cells: set[tuple[int, int]],
    frame_seed: tuple[int, int],
) -> dict[str, object] | None:
    color_mask = grid == int(color)
    frame_mask = color_mask.copy()
    for r, c in arrow_cells:
        frame_mask[r, c] = False

    comp = _component_from(frame_mask, frame_seed)
    if not comp:
        return None

    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    if r1 - r0 < 4 or c1 - c0 < 4:
        return None

    bbox = (r0, c0, r1, c1)
    corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    if not corners.issubset(comp):
        return None
    if any(r not in (r0, r1) and c not in (c0, c1) for r, c in comp):
        return None

    gaps: dict[str, list[tuple[int, int]]] = {}
    for side in DIRS:
        positions = _side_positions(bbox, side)
        gap = _strict_gap_segment(comp, positions)
        if side == direction:
            if gap is None:
                return None
            gaps[side] = gap
        else:
            if gap is not None:
                return None
            if any(pos not in comp for pos in positions):
                return None

    if direction == "right":
        if not all(c < c0 and r0 <= r <= r1 for r, c in arrow_cells):
            return None
        if frame_seed[1] != c0:
            return None
    elif direction == "left":
        if not all(c > c1 and r0 <= r <= r1 for r, c in arrow_cells):
            return None
        if frame_seed[1] != c1:
            return None
    elif direction == "down":
        if not all(r < r0 and c0 <= c <= c1 for r, c in arrow_cells):
            return None
        if frame_seed[0] != r0:
            return None
    else:
        if not all(r > r1 and c0 <= c <= c1 for r, c in arrow_cells):
            return None
        if frame_seed[0] != r1:
            return None

    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    content_mask = (interior != BG) & (interior != int(color))
    content_positions = np.argwhere(content_mask)
    content: list[tuple[int, int, int]] = []
    for rr, cc in content_positions:
        ar = int(r0 + 1 + rr)
        ac = int(c0 + 1 + cc)
        content.append((ar, ac, int(grid[ar, ac])))

    if not content:
        return None

    return {
        "color": int(color),
        "dir": direction,
        "bbox": bbox,
        "arrow": arrow_cells,
        "content": content,
        "gap": gaps[direction],
    }


def _detect_frames(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    frames: list[dict[str, object]] = []
    claimed_colors: set[int] = set()

    for color in sorted(int(v) for v in np.unique(grid) if int(v) != int(bg)):
        if color in claimed_colors:
            continue
        mask = grid == int(color)
        found = None
        for direction, spec in ARROW_PATTERNS.items():
            ph, pw = spec["shape"]  # type: ignore[index]
            cells = spec["cells"]  # type: ignore[index]
            apex_r, apex_c = spec["apex"]  # type: ignore[index]
            dr, dc = DIRS[direction]

            for r0 in range(h - ph + 1):
                if found is not None:
                    break
                for c0 in range(w - pw + 1):
                    ok = True
                    arrow_cells: set[tuple[int, int]] = set()
                    for rr in range(ph):
                        for cc in range(pw):
                            present = bool(mask[r0 + rr, c0 + cc])
                            expected = (rr, cc) in cells
                            if present != expected:
                                ok = False
                                break
                            if expected:
                                arrow_cells.add((r0 + rr, c0 + cc))
                        if not ok:
                            break
                    if not ok:
                        continue

                    apex = (int(r0 + apex_r), int(c0 + apex_c))
                    seed = (int(apex[0] + dr), int(apex[1] + dc))
                    if not (0 <= seed[0] < h and 0 <= seed[1] < w):
                        continue
                    if int(grid[seed]) != color:
                        continue

                    rec = _frame_record(grid, color, direction, arrow_cells, seed)
                    if rec is not None:
                        found = rec
                        break
        if found is not None:
            frames.append(found)
            claimed_colors.add(color)

    return frames


def _translate_points(
    coords: np.ndarray, base: np.ndarray, direction: str, bg: int
) -> np.ndarray:
    if coords.size == 0:
        return coords
    h, w = base.shape
    dr, dc = DIRS[direction]
    cur = coords.copy()
    while True:
        nxt = cur + np.array([dr, dc], dtype=int)
        if np.any(nxt[:, 0] < 0) or np.any(nxt[:, 0] >= h) or np.any(nxt[:, 1] < 0) or np.any(
            nxt[:, 1] >= w
        ):
            break
        if np.any(base[nxt[:, 0], nxt[:, 1]] != int(bg)):
            break
        cur = nxt
    return cur


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push each frame's interior pattern through its arrow-indicated gap and stack it outside the wall.

    Input structure:
    - The background is black.
    - Several colored one-pixel-thick rectangular frames appear; each frame has a single contiguous gap on one side.
    - Outside the side opposite the gap, a same-color solid triangular arrow is attached; the frame interior contains non-frame colored pixels.

    Rule:
    - Detect each outer triangular arrow and read its pointing direction; that direction is the gap side and the translation direction.
    - Remove all arrow pixels, clear the original interior content, then translate the whole interior pattern rigidly through the gap.
    - Keep moving until the next step would leave the grid or hit any remaining non-background pixel, and place the translated content there.
    - Preserve the frame walls and their gap exactly as they are.

    Key insight:
    - The triangle is not decoration: it identifies which side of the frame is open, and the interior pixels move as one rigid multi-color sprite rather than spilling out cell by cell.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = int(_mode_color(g))
    frames = _detect_frames(g, bg)

    base = g.copy()
    for frame in frames:
        for r, c in frame["arrow"]:
            base[r, c] = bg
        for r, c, _ in frame["content"]:
            base[r, c] = bg

    out = base.copy()
    for frame in frames:
        coords = np.array([(r, c) for r, c, _ in frame["content"]], dtype=int)
        colors = [int(v) for _, _, v in frame["content"]]
        moved = _translate_points(coords, base, str(frame["dir"]), bg)
        for (r, c), value in zip(moved.tolist(), colors, strict=False):
            out[int(r), int(c)] = int(value)

    return out


def _choose_gap_len(span: int) -> int:
    options = [1]
    if span >= 5:
        options.append(3)
    return random.choice(options)


def _grow_shape(height: int, width: int, n_cells: int) -> set[tuple[int, int]]:
    start = (random.randrange(height), random.randrange(width))
    cells = {start}
    while len(cells) < n_cells:
        frontier = []
        for r, c in cells:
            for dr, dc in DIRS.values():
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            break
        cells.add(random.choice(frontier))
    return cells


def _arrow_cells_for_frame(
    r0: int, c0: int, r1: int, c1: int, direction: str, lane: int
) -> set[tuple[int, int]]:
    spec = ARROW_PATTERNS[direction]
    shape_h, shape_w = spec["shape"]  # type: ignore[index]
    cells = spec["cells"]  # type: ignore[index]
    if direction == "right":
        top, left = lane - 2, c0 - 3
    elif direction == "left":
        top, left = lane - 2, c1 + 1
    elif direction == "down":
        top, left = r0 - 3, lane - 2
    else:
        top, left = r1 + 1, lane - 2

    return {
        (int(top + rr), int(left + cc))
        for rr in range(shape_h)
        for cc in range(shape_w)
        if (rr, cc) in cells
    }


def _gap_cells_for_frame(
    r0: int, c0: int, r1: int, c1: int, direction: str, lane: int, gap_len: int
) -> list[tuple[int, int]]:
    delta = gap_len // 2
    if direction == "right":
        return [(r, c1) for r in range(lane - delta, lane + delta + 1)]
    if direction == "left":
        return [(r, c0) for r in range(lane - delta, lane + delta + 1)]
    if direction == "down":
        return [(r1, c) for c in range(lane - delta, lane + delta + 1)]
    return [(r0, c) for c in range(lane - delta, lane + delta + 1)]


def _make_content(
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    direction: str,
    lane: int,
    gap_len: int,
    palette: list[int],
) -> dict[tuple[int, int], int]:
    interior_h = r1 - r0 - 1
    interior_w = c1 - c0 - 1
    delta = gap_len // 2

    if direction in ("right", "left"):
        band_top = lane - delta
        band_bottom = lane + delta
        box_h = random.randint(1, gap_len)
        top = random.randint(band_top, band_bottom - box_h + 1)
        box_w = random.randint(2, min(4, interior_w - 1))
        if direction == "right":
            left = random.randint(c0 + 1, c1 - box_w - 1)
        else:
            left = random.randint(c0 + 2, c1 - box_w)
    else:
        band_left = lane - delta
        band_right = lane + delta
        box_w = random.randint(1, gap_len)
        left = random.randint(band_left, band_right - box_w + 1)
        box_h = random.randint(2, min(4, interior_h - 1))
        if direction == "down":
            top = random.randint(r0 + 1, r1 - box_h - 1)
        else:
            top = random.randint(r0 + 2, r1 - box_h)

    area = box_h * box_w
    n_cells = random.randint(2, min(6, area))
    shape = _grow_shape(box_h, box_w, n_cells)
    n_colors = random.randint(1, min(3, len(palette)))
    colors = random.sample(palette, n_colors)

    content: dict[tuple[int, int], int] = {}
    for rr, cc in sorted(shape):
        content[(top + rr, left + cc)] = random.choice(colors)
    return content


def _frame_spec(
    r0: int,
    c0: int,
    height: int,
    width: int,
    direction: str,
    frame_color: int,
    content_palette: list[int],
    lane: int | None = None,
) -> dict[str, object]:
    r1 = r0 + height - 1
    c1 = c0 + width - 1

    if direction in ("right", "left"):
        lane_choices = list(range(r0 + 2, r1 - 1))
        side_span = height - 2
    else:
        lane_choices = list(range(c0 + 2, c1 - 1))
        side_span = width - 2
    if lane is None:
        lane = random.choice(lane_choices)
    gap_len = _choose_gap_len(side_span)

    frame_cells = {
        (r0, c) for c in range(c0, c1 + 1)
    } | {(r1, c) for c in range(c0, c1 + 1)} | {(r, c0) for r in range(r0, r1 + 1)} | {
        (r, c1) for r in range(r0, r1 + 1)
    }
    gap_cells = _gap_cells_for_frame(r0, c0, r1, c1, direction, lane, gap_len)
    for cell in gap_cells:
        frame_cells.discard(cell)

    arrow_cells = _arrow_cells_for_frame(r0, c0, r1, c1, direction, lane)
    content = _make_content(r0, c0, r1, c1, direction, lane, gap_len, content_palette)

    return {
        "bbox": (r0, c0, r1, c1),
        "dir": direction,
        "color": frame_color,
        "lane": lane,
        "gap_len": gap_len,
        "frame_cells": frame_cells,
        "arrow_cells": arrow_cells,
        "content": content,
        "gap_cells": gap_cells,
    }


def _mark_cells(mask: np.ndarray, cells: set[tuple[int, int]] | list[tuple[int, int]]) -> None:
    h, w = mask.shape
    for r, c in cells:
        if 0 <= r < h and 0 <= c < w:
            mask[r, c] = True


def _dilate(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    out = mask.copy()
    if radius <= 0:
        return out
    h, w = mask.shape
    pts = np.argwhere(mask)
    for r, c in pts:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr, nc = int(r + dr), int(c + dc)
                if 0 <= nr < h and 0 <= nc < w:
                    out[nr, nc] = True
    return out


def _reserve_actual(mask: np.ndarray, spec: dict[str, object]) -> None:
    _mark_cells(mask, spec["frame_cells"])
    _mark_cells(mask, spec["arrow_cells"])
    _mark_cells(mask, list(spec["content"].keys()))


def _boundary_corridor(
    spec: dict[str, object], shape: tuple[int, int]
) -> set[tuple[int, int]]:
    h, w = shape
    r0, c0, r1, c1 = spec["bbox"]
    lane = int(spec["lane"])
    gap_len = int(spec["gap_len"])
    delta = gap_len // 2
    cells: set[tuple[int, int]] = set()
    if spec["dir"] == "right":
        for r in range(lane - delta, lane + delta + 1):
            for c in range(c1 + 1, w):
                cells.add((r, c))
    elif spec["dir"] == "left":
        for r in range(lane - delta, lane + delta + 1):
            for c in range(0, c0):
                cells.add((r, c))
    elif spec["dir"] == "down":
        for c in range(lane - delta, lane + delta + 1):
            for r in range(r1 + 1, h):
                cells.add((r, c))
    else:
        for c in range(lane - delta, lane + delta + 1):
            for r in range(0, r0):
                cells.add((r, c))
    return cells


def _blocked_corridor(
    spec: dict[str, object], blocker_bbox: tuple[int, int, int, int]
) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = spec["bbox"]
    br0, bc0, br1, bc1 = blocker_bbox
    lane = int(spec["lane"])
    gap_len = int(spec["gap_len"])
    delta = gap_len // 2
    cells: set[tuple[int, int]] = set()
    if spec["dir"] == "right":
        for r in range(lane - delta, lane + delta + 1):
            for c in range(c1 + 1, bc0):
                cells.add((r, c))
    elif spec["dir"] == "left":
        for r in range(lane - delta, lane + delta + 1):
            for c in range(bc1 + 1, c0):
                cells.add((r, c))
    elif spec["dir"] == "down":
        for c in range(lane - delta, lane + delta + 1):
            for r in range(r1 + 1, br0):
                cells.add((r, c))
    else:
        for c in range(lane - delta, lane + delta + 1):
            for r in range(br1 + 1, r0):
                cells.add((r, c))
    return cells


def _placement_range_for_dir(
    shape: tuple[int, int], height: int, width: int, direction: str
) -> tuple[range, range]:
    h, w = shape
    min_r = 3 if direction == "down" else 0
    max_r = h - height - (3 if direction == "up" else 0)
    min_c = 3 if direction == "right" else 0
    max_c = w - width - (3 if direction == "left" else 0)
    return range(min_r, max_r + 1), range(min_c, max_c + 1)


def _envelope_mask_for_bbox(
    shape: tuple[int, int], bbox: tuple[int, int, int, int], direction: str
) -> np.ndarray:
    h, w = shape
    r0, c0, r1, c1 = bbox
    rr0, cc0, rr1, cc1 = r0, c0, r1, c1
    if direction == "right":
        cc0 -= 3
    elif direction == "left":
        cc1 += 3
    elif direction == "down":
        rr0 -= 3
    else:
        rr1 += 3
    rr0 = max(0, rr0)
    cc0 = max(0, cc0)
    rr1 = min(h - 1, rr1)
    cc1 = min(w - 1, cc1)
    mask = np.zeros(shape, dtype=bool)
    mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
    return _dilate(mask, 1)


def _sample_pair_bboxes(
    shape: tuple[int, int], pair_dir: str, b_dir: str, a_size: tuple[int, int], b_size: tuple[int, int]
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]] | None:
    h, w = shape
    ah, aw = a_size
    bh, bw = b_size
    b_rows, b_cols = _placement_range_for_dir(shape, bh, bw, b_dir)
    min_sep = 2

    for _ in range(120):
        br0 = random.choice(list(b_rows))
        bc0 = random.choice(list(b_cols))
        br1, bc1 = br0 + bh - 1, bc0 + bw - 1
        bb = (br0, bc0, br1, bc1)

        if pair_dir == "right":
            if bc0 - aw - min_sep < 3:
                continue
            ar_min = max(0, br0 - ah + 3)
            ar_max = min(h - ah, br1 - 2)
            if ar_min > ar_max:
                continue
            ac_min, ac_max = 3, bc0 - aw - min_sep
            ar0 = random.randint(ar_min, ar_max)
            ac0 = random.randint(ac_min, ac_max)
        elif pair_dir == "left":
            if bc1 + min_sep + aw > w - 4:
                continue
            ar_min = max(0, br0 - ah + 3)
            ar_max = min(h - ah, br1 - 2)
            if ar_min > ar_max:
                continue
            ac_min, ac_max = bc1 + min_sep, w - aw - 4
            ar0 = random.randint(ar_min, ar_max)
            ac0 = random.randint(ac_min, ac_max)
        elif pair_dir == "down":
            if br0 - ah - min_sep < 3:
                continue
            ac_min = max(0, bc0 - aw + 3)
            ac_max = min(w - aw, bc1 - 2)
            if ac_min > ac_max:
                continue
            ar_min, ar_max = 3, br0 - ah - min_sep
            ar0 = random.randint(ar_min, ar_max)
            ac0 = random.randint(ac_min, ac_max)
        else:
            if br1 + min_sep + ah > h - 4:
                continue
            ac_min = max(0, bc0 - aw + 3)
            ac_max = min(w - aw, bc1 - 2)
            if ac_min > ac_max:
                continue
            ar_min, ar_max = br1 + min_sep, h - ah - 4
            ar0 = random.randint(ar_min, ar_max)
            ac0 = random.randint(ac_min, ac_max)

        aa = (ar0, ac0, ar0 + ah - 1, ac0 + aw - 1)
        env_a = _envelope_mask_for_bbox(shape, aa, pair_dir)
        env_b = _envelope_mask_for_bbox(shape, bb, b_dir)
        if np.any(env_a & env_b):
            continue
        return aa, bb
    return None


def _pick_blocking_lane(
    a_bbox: tuple[int, int, int, int], b_bbox: tuple[int, int, int, int], pair_dir: str
) -> int | None:
    ar0, ac0, ar1, ac1 = a_bbox
    br0, bc0, br1, bc1 = b_bbox
    if pair_dir in ("right", "left"):
        lo = max(ar0 + 2, br0)
        hi = min(ar1 - 2, br1)
    else:
        lo = max(ac0 + 2, bc0)
        hi = min(ac1 - 2, bc1)
    if lo > hi:
        return None
    return random.randint(lo, hi)


def _spec_cells(spec: dict[str, object]) -> set[tuple[int, int]]:
    return set(spec["frame_cells"]) | set(spec["arrow_cells"]) | set(spec["content"].keys())


def _find_extra_boundary_spec(
    shape: tuple[int, int],
    direction: str,
    size: tuple[int, int],
    frame_color: int,
    palette: list[int],
    reserved: np.ndarray,
) -> dict[str, object] | None:
    h, w = shape
    fh, fw = size
    rows, cols = _placement_range_for_dir(shape, fh, fw, direction)
    row_list = list(rows)
    col_list = list(cols)
    random.shuffle(row_list)
    random.shuffle(col_list)

    for _ in range(140):
        r0 = random.choice(row_list)
        c0 = random.choice(col_list)
        bbox = (r0, c0, r0 + fh - 1, c0 + fw - 1)
        env = _envelope_mask_for_bbox(shape, bbox, direction)
        if np.any(env & reserved):
            continue

        if direction in ("right", "left"):
            lane_choices = list(range(r0 + 2, r0 + fh - 2))
        else:
            lane_choices = list(range(c0 + 2, c0 + fw - 2))
        random.shuffle(lane_choices)

        for lane in lane_choices:
            spec = _frame_spec(r0, c0, fh, fw, direction, frame_color, palette, lane=lane)
            cells = _spec_cells(spec)
            if any(reserved[r, c] for r, c in cells):
                continue
            corridor = _boundary_corridor(spec, shape)
            if any(reserved[r, c] for r, c in corridor):
                continue
            return spec
    return None


def _paint_specs(shape: tuple[int, int], specs: list[dict[str, object]]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for spec in specs:
        color = int(spec["color"])
        for r, c in spec["frame_cells"]:
            grid[r, c] = color
        for r, c in spec["arrow_cells"]:
            grid[r, c] = color
        for (r, c), value in spec["content"].items():
            grid[r, c] = int(value)
    return grid


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(20, 28)
    w = random.randint(20, 28)
    shape = (h, w)
    n_frames = random.choice([2, 3])

    frame_colors = random.sample(FRAME_COLORS, n_frames)
    content_palette = [c for c in FRAME_COLORS if c not in frame_colors]

    pair_dir = random.choice(list(DIRS))
    if pair_dir in ("right", "left"):
        b_dir = random.choice(["up", "down"])
    else:
        b_dir = random.choice(["left", "right"])

    a_size = (random.randint(7, 10), random.randint(7, 10))
    b_size = (random.randint(7, 10), random.randint(7, 10))
    pair_bboxes = _sample_pair_bboxes(shape, pair_dir, b_dir, a_size, b_size)
    if pair_bboxes is None:
        return None
    a_bbox, b_bbox = pair_bboxes
    lane = _pick_blocking_lane(a_bbox, b_bbox, pair_dir)
    if lane is None:
        return None

    a_r0, a_c0, a_r1, a_c1 = a_bbox
    b_r0, b_c0, b_r1, b_c1 = b_bbox

    spec_a = _frame_spec(
        a_r0,
        a_c0,
        a_r1 - a_r0 + 1,
        a_c1 - a_c0 + 1,
        pair_dir,
        frame_colors[0],
        content_palette,
        lane=lane,
    )
    spec_b = _frame_spec(
        b_r0,
        b_c0,
        b_r1 - b_r0 + 1,
        b_c1 - b_c0 + 1,
        b_dir,
        frame_colors[1],
        content_palette,
    )

    cells_a = _spec_cells(spec_a)
    cells_b = _spec_cells(spec_b)
    if cells_a & cells_b:
        return None

    reserved = np.zeros(shape, dtype=bool)
    _reserve_actual(reserved, spec_a)
    _reserve_actual(reserved, spec_b)
    _mark_cells(reserved, _blocked_corridor(spec_a, spec_b["bbox"]))
    _mark_cells(reserved, _boundary_corridor(spec_b, shape))
    reserved = _dilate(reserved, 0)

    specs = [spec_a, spec_b]

    if n_frames == 3:
        extra_dir = random.choice(list(DIRS))
        extra_size = (random.randint(7, 10), random.randint(7, 10))
        extra_spec = _find_extra_boundary_spec(
            shape,
            extra_dir,
            extra_size,
            frame_colors[2],
            content_palette,
            reserved,
        )
        if extra_spec is None:
            return None
        specs.append(extra_spec)

    grid = _paint_specs(shape, specs)

    frames = _detect_frames(grid, BG)
    if len(frames) != len(specs):
        return None
    out = main(grid)
    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
