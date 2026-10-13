from __future__ import annotations

from collections import defaultdict
import random

import numpy as np


SIDES = ("top", "bottom", "left", "right")
GEN_RETRIES = 400
PATCH_RETRIES = 120


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _sort_key(side: str, bbox: tuple[int, int, int, int], pos: tuple[int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    r, c = pos
    if side == "top":
        return (r0 - r, c)
    if side == "bottom":
        return (r - r1, c)
    if side == "left":
        return (c0 - c, r)
    return (c - c1, r)


def _target_cell(side: str, bbox: tuple[int, int, int, int], gap: tuple[int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    gr, gc = gap
    if side == "top":
        return (r1 - 1, gc)
    if side == "bottom":
        return (r0 + 1, gc)
    if side == "left":
        return (gr, c1 - 1)
    return (gr, c0 + 1)


def _is_singleton(g: np.ndarray, r: int, c: int) -> bool:
    color = int(g[r, c])
    h, w = g.shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) == color:
                return False
    return True


def _validate_frame(
    g: np.ndarray,
    bg: int,
    color: int,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
):
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None
    corners = ((r0, c0), (r0, c1), (r1, c0), (r1, c1))
    if any(int(g[r, c]) != color for r, c in corners):
        return None

    interior = g[r0 + 1 : r1, c0 + 1 : c1]
    if interior.size == 0 or np.any(interior != bg):
        return None

    side_cells = {
        "top": [(r0, c) for c in range(c0 + 1, c1)],
        "bottom": [(r1, c) for c in range(c0 + 1, c1)],
        "left": [(r, c0) for r in range(r0 + 1, r1)],
        "right": [(r, c1) for r in range(r0 + 1, r1)],
    }

    gap_side = None
    gaps: list[tuple[int, int]] = []
    for side, cells in side_cells.items():
        vals = [int(g[r, c]) for r, c in cells]
        if all(v == color for v in vals):
            continue
        if any(v not in (bg, color) for v in vals):
            return None
        bg_idx = [i for i, v in enumerate(vals) if v == bg]
        if not bg_idx:
            return None
        if any(b - a == 1 for a, b in zip(bg_idx, bg_idx[1:])):
            return None
        if gap_side is not None:
            return None
        gap_side = side
        gaps = [cells[i] for i in bg_idx]

    if gap_side is None or not gaps:
        return None

    return {
        "color": color,
        "bbox": (r0, c0, r1, c1),
        "side": gap_side,
        "gaps": gaps,
    }


def _detect_frames(g: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    colors = [int(v) for v in np.unique(g) if int(v) != bg]
    h, w = g.shape
    for color in colors:
        positions = [tuple(map(int, rc)) for rc in np.argwhere(g == color)]
        if len(positions) < 4:
            continue
        pos_set = set(positions)
        rows: dict[int, list[int]] = defaultdict(list)
        cols: dict[int, list[int]] = defaultdict(list)
        for r, c in positions:
            rows[r].append(c)
            cols[c].append(r)
        for cols_list in rows.values():
            cols_list.sort()
        for rows_list in cols.values():
            rows_list.sort()

        candidates: list[dict] = []
        seen = set()
        for r0, row_cols in rows.items():
            for i, c0 in enumerate(row_cols):
                lower_rows = [r1 for r1 in cols[c0] if r1 >= r0 + 2]
                if not lower_rows:
                    continue
                for c1 in row_cols[i + 1 :]:
                    if c1 < c0 + 2:
                        continue
                    for r1 in lower_rows:
                        if (r1, c1) not in pos_set:
                            continue
                        key = (r0, c0, r1, c1)
                        if key in seen:
                            continue
                        seen.add(key)
                        frame = _validate_frame(g, bg, color, r0, c0, r1, c1)
                        if frame is not None:
                            candidates.append(frame)

        used = np.zeros((h, w), dtype=bool)
        candidates.sort(key=lambda frame: (-_bbox_area(frame["bbox"]), frame["bbox"][0], frame["bbox"][1]))
        for frame in candidates:
            r0, c0, r1, c1 = frame["bbox"]
            if np.any(used[r0 : r1 + 1, c0 : c1 + 1]):
                continue
            used[r0 : r1 + 1, c0 : c1 + 1] = True
            frames.append(frame)

    frames.sort(key=lambda frame: (frame["bbox"][0], frame["bbox"][1], frame["color"]))
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pull same-colored outside single pixels through each frame-side gap and drop them against the opposite inner wall.

    Input structure:
    - The background is the dominant color.
    - Several one-pixel-thick hollow rectangular frames appear on the grid, each drawn in a single color and enclosing only background.
    - Every frame has one or more one-pixel gaps, and all of that frame's gaps lie on the same side.
    - Outside the frame, on the gap side, there are isolated single pixels of the same color as the frame.

    Rule:
    - Detect each rectangular frame and count its gaps; this count is N.
    - On the gap side, sort same-colored isolated outside pixels by perpendicular distance to that side, breaking ties by their position along the side, and select the nearest N.
    - Erase those N outside pixels from their original locations.
    - Sort the gaps in the same along-edge order, pair the selected pixels with the sorted gaps, and move them straight through the frame to the cell just before the opposite inner wall.

    Key insight:
    - The missing side simultaneously encodes the intake direction and the number of pixels to move, while the tie-breaking order along that side determines the pairing.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    out = g.copy()

    frames = _detect_frames(g, bg)
    singleton_by_color: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for r, c in np.argwhere(g != bg):
        rr, cc = map(int, (r, c))
        if _is_singleton(g, rr, cc):
            singleton_by_color[int(g[rr, cc])].append((rr, cc))

    available = {color: set(points) for color, points in singleton_by_color.items()}
    for frame in frames:
        color = frame["color"]
        bbox = frame["bbox"]
        side = frame["side"]
        gaps = sorted(frame["gaps"], key=lambda pos: (pos[1], pos[0]) if side in ("top", "bottom") else (pos[0], pos[1]))

        candidates = [
            pos
            for pos in available.get(color, set())
            if (
                (side == "top" and pos[0] < bbox[0] and bbox[1] < pos[1] < bbox[3])
                or (side == "bottom" and pos[0] > bbox[2] and bbox[1] < pos[1] < bbox[3])
                or (side == "left" and pos[1] < bbox[1] and bbox[0] < pos[0] < bbox[2])
                or (side == "right" and pos[1] > bbox[3] and bbox[0] < pos[0] < bbox[2])
            )
        ]
        candidates.sort(key=lambda pos: _sort_key(side, bbox, pos))
        chosen = candidates[: len(gaps)]
        if len(chosen) < len(gaps):
            continue

        for r, c in chosen:
            out[r, c] = bg
            available[color].remove((r, c))

        for gap in gaps:
            tr, tc = _target_cell(side, bbox, gap)
            out[tr, tc] = color

    return out


def _sample_nonadjacent_indices(length: int, count: int, rng: random.Random) -> list[int] | None:
    choices = list(range(length))
    for _ in range(80):
        rng.shuffle(choices)
        picked: list[int] = []
        for idx in choices:
            if all(abs(idx - prev) > 1 for prev in picked):
                picked.append(idx)
                if len(picked) == count:
                    return sorted(picked)
    return None


def _sparse_sample(
    candidates: list[tuple[int, int]],
    count: int,
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    for _ in range(120):
        order = candidates[:]
        rng.shuffle(order)
        picked: list[tuple[int, int]] = []
        for pos in order:
            if all(max(abs(pos[0] - pr), abs(pos[1] - pc)) > 1 for pr, pc in picked):
                picked.append(pos)
                if len(picked) == count:
                    return picked
    return None


def _strip_capacity(lanes: int, span: int) -> int:
    return ((lanes + 1) // 2) * ((span + 1) // 2)


def _frame_origin(side: str, outer_h: int, outer_w: int, max_dist: int) -> tuple[int, int, int, int]:
    if side == "top":
        return max_dist, 0, outer_h + max_dist, outer_w
    if side == "bottom":
        return 0, 0, outer_h + max_dist, outer_w
    if side == "left":
        return 0, max_dist, outer_h, outer_w + max_dist
    return 0, 0, outer_h, outer_w + max_dist


def _gap_positions(
    side: str,
    frame_top: int,
    frame_left: int,
    outer_h: int,
    outer_w: int,
    gap_indices: list[int],
) -> list[tuple[int, int]]:
    if side == "top":
        return [(frame_top, frame_left + 1 + idx) for idx in gap_indices]
    if side == "bottom":
        return [(frame_top + outer_h - 1, frame_left + 1 + idx) for idx in gap_indices]
    if side == "left":
        return [(frame_top + 1 + idx, frame_left) for idx in gap_indices]
    return [(frame_top + 1 + idx, frame_left + outer_w - 1) for idx in gap_indices]


def _free_candidates(
    side: str,
    frame_top: int,
    frame_left: int,
    outer_h: int,
    outer_w: int,
    max_dist: int,
) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []
    if side in ("top", "bottom"):
        for dist in range(2, max_dist + 1):
            row = frame_top - dist if side == "top" else frame_top + outer_h - 1 + dist
            for col in range(frame_left + 1, frame_left + outer_w - 1):
                points.append((row, col))
    else:
        for dist in range(2, max_dist + 1):
            col = frame_left - dist if side == "left" else frame_left + outer_w - 1 + dist
            for row in range(frame_top + 1, frame_top + outer_h - 1):
                points.append((row, col))
    return points


def _make_patch(color: int, bg: int, rng: random.Random):
    for _ in range(PATCH_RETRIES):
        side = rng.choice(SIDES)
        gap_count = rng.randint(1, 3)
        extra_count = rng.randint(0, 2)
        total_free = gap_count + extra_count
        max_dist = rng.randint(3, 5)

        if side in ("top", "bottom"):
            outer_w = rng.randint(max(5, 2 * gap_count + 1), 9)
            outer_h = rng.randint(4, 8)
            span = outer_w - 2
        else:
            outer_h = rng.randint(max(5, 2 * gap_count + 1), 9)
            outer_w = rng.randint(4, 8)
            span = outer_h - 2

        if _strip_capacity(max_dist - 1, span) < total_free:
            continue

        gap_indices = _sample_nonadjacent_indices(span, gap_count, rng)
        if gap_indices is None:
            continue

        frame_top, frame_left, patch_h, patch_w = _frame_origin(side, outer_h, outer_w, max_dist)
        patch = np.full((patch_h, patch_w), bg, dtype=int)
        bbox = (frame_top, frame_left, frame_top + outer_h - 1, frame_left + outer_w - 1)

        r0, c0, r1, c1 = bbox
        patch[r0, c0 : c1 + 1] = color
        patch[r1, c0 : c1 + 1] = color
        patch[r0 : r1 + 1, c0] = color
        patch[r0 : r1 + 1, c1] = color

        gaps = _gap_positions(side, frame_top, frame_left, outer_h, outer_w, gap_indices)
        for gr, gc in gaps:
            patch[gr, gc] = bg

        free_positions = _sparse_sample(
            _free_candidates(side, frame_top, frame_left, outer_h, outer_w, max_dist),
            total_free,
            rng,
        )
        if free_positions is None:
            continue
        for fr, fc in free_positions:
            patch[fr, fc] = color

        expected = patch.copy()
        free_positions.sort(key=lambda pos: _sort_key(side, bbox, pos))
        chosen = free_positions[:gap_count]
        for fr, fc in chosen:
            expected[fr, fc] = bg
        sorted_gaps = sorted(gaps, key=lambda pos: (pos[1], pos[0]) if side in ("top", "bottom") else (pos[0], pos[1]))
        for gap in sorted_gaps:
            tr, tc = _target_cell(side, bbox, gap)
            expected[tr, tc] = color

        if np.array_equal(main(patch), expected):
            return patch, expected
    return None


def _layout_options(n_items: int) -> list[tuple[int, int]]:
    if n_items == 1:
        return [(1, 1)]
    if n_items == 2:
        return [(1, 2), (2, 1), (2, 2)]
    return [(2, 2)]


def _paste_non_bg(dst: np.ndarray, src: np.ndarray, top: int, left: int, bg: int) -> None:
    mask = src != bg
    h, w = src.shape
    target = dst[top : top + h, left : left + w]
    target[mask] = src[mask]


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    bg = rng.randrange(10)
    palette = [c for c in range(10) if c != bg]
    rng.shuffle(palette)

    window_count = rng.randint(1, 3)
    if len(palette) < window_count:
        return None
    colors = palette[:window_count]

    patches = []
    for color in colors:
        made = _make_patch(color, bg, rng)
        if made is None:
            return None
        patches.append(made)

    heights = [patch.shape[0] for patch, _ in patches]
    widths = [patch.shape[1] for patch, _ in patches]
    base_h = max(heights)
    base_w = max(widths)

    layout = rng.choice(_layout_options(window_count))
    rows, cols = layout
    max_cell_h = (30 - (rows - 1)) // rows
    max_cell_w = (30 - (cols - 1)) // cols
    if base_h > max_cell_h or base_w > max_cell_w:
        return None

    cell_h = base_h + rng.randint(0, min(4, max_cell_h - base_h))
    cell_w = base_w + rng.randint(0, min(4, max_cell_w - base_w))
    grid_h = rows * cell_h + (rows - 1)
    grid_w = cols * cell_w + (cols - 1)

    extra_top = 1 if grid_h + 2 <= 30 else 0
    extra_left = 1 if grid_w + 2 <= 30 else 0
    grid_h += 2 * extra_top
    grid_w += 2 * extra_left

    grid = np.full((grid_h, grid_w), bg, dtype=int)
    expected = grid.copy()

    slots = [(r, c) for r in range(rows) for c in range(cols)]
    rng.shuffle(slots)
    for (patch, exp_patch), (slot_r, slot_c) in zip(patches, slots):
        ph, pw = patch.shape
        cell_top = extra_top + slot_r * (cell_h + 1)
        cell_left = extra_left + slot_c * (cell_w + 1)
        top = cell_top + rng.randint(0, cell_h - ph)
        left = cell_left + rng.randint(0, cell_w - pw)
        _paste_non_bg(grid, patch, top, left, bg)
        _paste_non_bg(expected, exp_patch, top, left, bg)

    if _bg_color(grid) != bg:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    if np.array_equal(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
