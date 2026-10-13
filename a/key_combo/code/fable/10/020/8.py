from __future__ import annotations

from collections import Counter, deque

import numpy as np


Coord = tuple[int, int]

ALL_COLORS = tuple(range(10))
MAX_GENERATION_ATTEMPTS = 600


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)


def _components(mask: np.ndarray) -> list[list[Coord]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[list[Coord]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            part: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                part.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(part)
    return parts


def _sorted_coords(coords: list[Coord] | np.ndarray) -> list[Coord]:
    return sorted((int(r), int(c)) for r, c in coords)


def _edge_run_candidates(g: np.ndarray, bg: int, marker_color: int) -> list[list[Coord]]:
    h, w = g.shape
    candidates: list[list[Coord]] = []

    def collect_line(coords: list[Coord]) -> None:
        run: list[Coord] = []
        for coord in coords + [(-1, -1)]:
            if coord != (-1, -1):
                r, c = coord
                val = int(g[r, c])
                usable = val != bg and val != marker_color
            else:
                usable = False

            if usable:
                run.append(coord)
                continue

            if len(run) >= 2:
                colors = [int(g[r, c]) for r, c in run]
                if len(set(colors)) == len(colors):
                    candidates.append(run.copy())
            run.clear()

    collect_line([(0, c) for c in range(w)])
    collect_line([(h - 1, c) for c in range(w)])
    collect_line([(r, 0) for r in range(h)])
    collect_line([(r, w - 1) for r in range(h)])
    return candidates


def _extract_scene(
    grid: np.ndarray,
) -> tuple[int, int, list[Coord], tuple[int, int], list[Coord], list[int], list[Coord], int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    values, counts = np.unique(g, return_counts=True)
    marker_candidates = [int(v) for v, n in zip(values, counts) if int(v) != bg and int(n) == 2]
    assert len(marker_candidates) == 1, "expected a unique two-pixel marker color"
    marker_color = marker_candidates[0]

    marker_coords = _sorted_coords(np.argwhere(g == marker_color))
    assert len(marker_coords) == 2
    vector = (
        marker_coords[1][0] - marker_coords[0][0],
        marker_coords[1][1] - marker_coords[0][1],
    )
    assert vector != (0, 0), "marker pixels must be distinct"
    assert vector[0] > 0 or vector[1] > 0, "reading-order vector must point down, right, or both"

    bar_candidates = _edge_run_candidates(g, bg, marker_color)
    assert len(bar_candidates) == 1, "expected a unique boundary color bar"
    bar_coords = bar_candidates[0]
    bar_colors = [int(g[r, c]) for r, c in bar_coords]

    masked = g.copy()
    for r, c in marker_coords + bar_coords:
        masked[r, c] = bg

    motif_mask = masked != bg
    motif_components = _components(motif_mask)
    assert len(motif_components) == 1, "expected one motif component after removing clues"
    motif_coords = _sorted_coords(motif_components[0])
    motif_colors = {int(g[r, c]) for r, c in motif_coords}
    assert len(motif_colors) == 1, "motif must be monochrome"
    motif_color = next(iter(motif_colors))

    assert motif_color != marker_color
    assert motif_color not in bar_colors
    assert marker_color not in bar_colors
    return bg, marker_color, marker_coords, vector, bar_coords, bar_colors, motif_coords, motif_color


def _stamp_copies(
    shape: tuple[int, int],
    motif_coords: list[Coord],
    motif_color: int,
    vector: tuple[int, int],
    bar_colors: list[int],
    bg: int,
    *,
    priority: bool,
    cycle: bool,
    stop_on_partial_exit: bool,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    h, w = shape
    out = np.full((h, w), bg, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    motif_size = len(motif_coords)
    stats: list[dict[str, object]] = []

    copy_idx = 0
    while True:
        shifted = [
            (r + copy_idx * vector[0], c + copy_idx * vector[1])
            for r, c in motif_coords
        ]
        in_bounds = [(r, c) for r, c in shifted if 0 <= r < h and 0 <= c < w]
        if not in_bounds:
            break
        if stop_on_partial_exit and len(in_bounds) < motif_size:
            break

        if copy_idx == 0:
            color = motif_color
        else:
            if cycle:
                color = int(bar_colors[(copy_idx - 1) % len(bar_colors)])
            else:
                color = int(bar_colors[0])

        overlap_count = sum(bool(occupied[r, c]) for r, c in in_bounds)
        new_count = 0
        if priority:
            for r, c in in_bounds:
                if occupied[r, c]:
                    continue
                out[r, c] = color
                occupied[r, c] = True
                new_count += 1
        else:
            for r, c in in_bounds:
                out[r, c] = color
                if not occupied[r, c]:
                    new_count += 1
                occupied[r, c] = True

        stats.append(
            {
                "copy_idx": copy_idx,
                "color": color,
                "inside_count": len(in_bounds),
                "overlap_count": overlap_count,
                "new_count": new_count,
                "all_inside": len(in_bounds) == motif_size,
                "coords": tuple(in_bounds),
            }
        )
        copy_idx += 1

    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repeat the motif along the marker-defined offset vector, recoloring each later copy by the boundary color bar until a full copy leaves the canvas.

    Input structure:
    - The grid has one monochrome connected motif, exactly two same-colored marker pixels, and one boundary-attached color bar made of distinct single cells in a straight line.
    - The marker color is different from the motif color and from every color used in the bar.
    - Aside from those clue objects, the rest of the grid is background.

    Rule:
    - Sort the two marker pixels in reading order and define the offset vector as second minus first.
    - Keep copy 0 of the motif at its original location and original color.
    - For copy i >= 1, translate the motif by i times that vector, recolor the whole translated copy with the i-th bar color cycling from the start of the bar, and stamp it without overwriting cells claimed by earlier copies.
    - Stop when the next translated copy is entirely outside the canvas, then erase the two marker pixels and the whole color bar to background.

    Key insight:
    - The motion is not drawn directly: the marker-pair difference creates the hidden translation vector, the boundary bar supplies the repeating recolor schedule, and the final visible extent comes from stopping only when a whole copy exits.
    """
    g = np.asarray(grid, dtype=int)
    bg, _marker_color, _marker_coords, vector, _bar_coords, bar_colors, motif_coords, motif_color = _extract_scene(g)
    out, _stats = _stamp_copies(
        g.shape,
        motif_coords,
        motif_color,
        vector,
        bar_colors,
        bg,
        priority=True,
        cycle=True,
        stop_on_partial_exit=False,
    )
    return out


def _normalize(coords: list[Coord]) -> list[Coord]:
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    return _sorted_coords([(r - min_r, c - min_c) for r, c in coords])


def _bbox(coords: list[Coord]) -> tuple[int, int]:
    return max(r for r, _ in coords) + 1, max(c for _, c in coords) + 1


def _gen_polyomino(rng: np.random.Generator) -> list[Coord] | None:
    for _ in range(100):
        box_h = int(rng.integers(2, 5))
        box_w = int(rng.integers(2, 6))
        area = box_h * box_w
        if area < 5:
            continue
        target = int(rng.integers(4, min(8, area)))
        start = (int(rng.integers(box_h)), int(rng.integers(box_w)))
        cells = {start}
        while len(cells) < target:
            frontier = set()
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(tuple(frontier)[int(rng.integers(len(frontier)))])
        if len(cells) != target:
            continue

        norm = _normalize(list(cells))
        h, w = _bbox(norm)
        if h < 2 or w < 2:
            continue
        if len(norm) == h * w:
            continue
        return norm
    return None


def _gen_vector(
    rng: np.random.Generator,
    motif_h: int,
    motif_w: int,
    need_overlap: bool,
) -> tuple[int, int] | None:
    for _ in range(80):
        orientation = str(rng.choice(["horizontal", "vertical", "diagonal"]))
        if orientation == "horizontal":
            if need_overlap:
                if motif_w < 2:
                    continue
                return 0, int(rng.integers(1, motif_w))
            return 0, int(rng.integers(motif_w, motif_w + 4))

        if orientation == "vertical":
            if need_overlap:
                if motif_h < 2:
                    continue
                return int(rng.integers(1, motif_h)), 0
            return int(rng.integers(motif_h, motif_h + 4)), 0

        if need_overlap:
            dr_hi = max(2, motif_h)
            dc_hi = max(2, motif_w)
            dr = int(rng.integers(1, dr_hi))
            dc = int(rng.integers(1, dc_hi))
            return dr, dc

        dr = int(rng.integers(1, motif_h + 3))
        dc = int(rng.integers(1, motif_w + 3))
        if dr >= motif_h or dc >= motif_w:
            return dr, dc
    return None


def _choose_counts(
    rng: np.random.Generator,
    need_wrap: bool,
) -> tuple[int, int]:
    if need_wrap:
        visible_copies = int(rng.integers(4, 8))
        bar_len = int(rng.integers(2, min(5, visible_copies - 1)))
        if bar_len >= visible_copies - 1:
            bar_len = visible_copies - 2
        return visible_copies, bar_len

    visible_copies = int(rng.integers(3, 7))
    bar_len = int(rng.integers(max(2, visible_copies - 1), 6))
    return visible_copies, bar_len


def _build_dimensions(
    rng: np.random.Generator,
    motif_h: int,
    motif_w: int,
    vector: tuple[int, int],
    visible_copies: int,
) -> tuple[int, int, int, int] | None:
    dr, dc = vector
    for _ in range(120):
        anchor_r = int(rng.integers(2, 6))
        anchor_c = int(rng.integers(2, 6))

        stop_axis_choices = []
        if dr > 0:
            stop_axis_choices.append("row")
        if dc > 0:
            stop_axis_choices.append("col")
        if not stop_axis_choices:
            return None
        stop_axis = str(rng.choice(stop_axis_choices))

        if stop_axis == "row":
            low = anchor_r + (visible_copies - 1) * dr + 1
            high = min(30, anchor_r + visible_copies * dr, anchor_r + (visible_copies - 1) * dr + motif_h - 1)
            if low > high:
                continue
            height = int(rng.integers(low, high + 1))
            width = anchor_c + (visible_copies - 1) * dc + motif_w + int(rng.integers(0, 5))
        else:
            low = anchor_c + (visible_copies - 1) * dc + 1
            high = min(30, anchor_c + visible_copies * dc, anchor_c + (visible_copies - 1) * dc + motif_w - 1)
            if low > high:
                continue
            width = int(rng.integers(low, high + 1))
            height = anchor_r + (visible_copies - 1) * dr + motif_h + int(rng.integers(0, 5))

        height = max(height, 9)
        width = max(width, 9)
        if height > 30 or width > 30:
            continue
        return height, width, anchor_r, anchor_c
    return None


def _mark_blocked(blocked: np.ndarray, coords: list[Coord] | tuple[Coord, ...]) -> None:
    for r, c in coords:
        blocked[r, c] = True


def _bar_slots(shape: tuple[int, int], length: int, blocked: np.ndarray) -> list[list[Coord]]:
    h, w = shape
    slots: list[list[Coord]] = []
    if w >= length + 2:
        for c0 in range(1, w - length):
            top = [(0, c) for c in range(c0, c0 + length)]
            bottom = [(h - 1, c) for c in range(c0, c0 + length)]
            if all(not blocked[r, c] for r, c in top):
                slots.append(top)
            if all(not blocked[r, c] for r, c in bottom):
                slots.append(bottom)
    if h >= length + 2:
        for r0 in range(1, h - length):
            left = [(r, 0) for r in range(r0, r0 + length)]
            right = [(r, w - 1) for r in range(r0, r0 + length)]
            if all(not blocked[r, c] for r, c in left):
                slots.append(left)
            if all(not blocked[r, c] for r, c in right):
                slots.append(right)
    return slots


def _marker_slots(shape: tuple[int, int], vector: tuple[int, int], blocked: np.ndarray) -> list[tuple[Coord, Coord]]:
    h, w = shape
    dr, dc = vector
    slots: list[tuple[Coord, Coord]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            r2 = r + dr
            c2 = c + dc
            if not (1 <= r2 < h - 1 and 1 <= c2 < w - 1):
                continue
            if blocked[r, c] or blocked[r2, c2]:
                continue
            slots.append(((r, c), (r2, c2)))
    return slots


def _semantic_checks(
    inp: np.ndarray,
    expected: np.ndarray,
    motif_coords: list[Coord],
    motif_color: int,
    vector: tuple[int, int],
    bar_colors: list[int],
    bg: int,
    stats: list[dict[str, object]],
    need_overlap: bool,
    need_wrap: bool,
) -> bool:
    if not np.array_equal(main(inp), expected):
        return False

    if len(stats) < 3:
        return False
    if not any(int(s["copy_idx"]) >= 1 for s in stats):
        return False
    if any(int(s["new_count"]) == 0 for s in stats[1:]):
        return False
    if not bool(stats[-1]["inside_count"] < len(motif_coords)):
        return False

    overlap_present = any(int(s["overlap_count"]) > 0 for s in stats[1:])
    if need_overlap != overlap_present:
        return False
    if need_overlap:
        overwrite_out, _ = _stamp_copies(
            inp.shape,
            motif_coords,
            motif_color,
            vector,
            bar_colors,
            bg,
            priority=False,
            cycle=True,
            stop_on_partial_exit=False,
        )
        if np.array_equal(overwrite_out, expected):
            return False

    if need_wrap:
        if len(stats) - 1 <= len(bar_colors):
            return False
        no_cycle_out, _ = _stamp_copies(
            inp.shape,
            motif_coords,
            motif_color,
            vector,
            bar_colors,
            bg,
            priority=True,
            cycle=False,
            stop_on_partial_exit=False,
        )
        if np.array_equal(no_cycle_out, expected):
            return False

    stop_partial_out, _ = _stamp_copies(
        inp.shape,
        motif_coords,
        motif_color,
        vector,
        bar_colors,
        bg,
        priority=True,
        cycle=True,
        stop_on_partial_exit=True,
    )
    if np.array_equal(stop_partial_out, expected):
        return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    motif_rel = _gen_polyomino(rng)
    if motif_rel is None:
        return None
    motif_h, motif_w = _bbox(motif_rel)

    need_overlap = bool(rng.integers(0, 2))
    need_wrap = bool(rng.integers(0, 2))
    vector = _gen_vector(rng, motif_h, motif_w, need_overlap)
    if vector is None:
        return None

    visible_copies, bar_len = _choose_counts(rng, need_wrap)
    dims = _build_dimensions(rng, motif_h, motif_w, vector, visible_copies)
    if dims is None:
        return None
    h, w, anchor_r, anchor_c = dims

    palette = list(rng.choice(ALL_COLORS, size=bar_len + 3, replace=False))
    bg = int(palette[0])
    motif_color = int(palette[1])
    marker_color = int(palette[2])
    bar_colors = [int(c) for c in palette[3:]]

    motif_coords = [(anchor_r + r, anchor_c + c) for r, c in motif_rel]
    expected, stats = _stamp_copies(
        (h, w),
        motif_coords,
        motif_color,
        vector,
        bar_colors,
        bg,
        priority=True,
        cycle=True,
        stop_on_partial_exit=False,
    )
    if len(stats) != visible_copies:
        return None

    blocked = np.zeros((h, w), dtype=bool)
    for s in stats:
        _mark_blocked(blocked, list(s["coords"]))

    bar_options = _bar_slots((h, w), bar_len, blocked)
    if not bar_options:
        return None
    bar_coords = bar_options[int(rng.integers(len(bar_options)))]
    _mark_blocked(blocked, bar_coords)

    marker_options = _marker_slots((h, w), vector, blocked)
    if not marker_options:
        return None
    marker_coords = marker_options[int(rng.integers(len(marker_options)))]

    inp = np.full((h, w), bg, dtype=int)
    for r, c in motif_coords:
        inp[r, c] = motif_color
    for (r, c), color in zip(bar_coords, bar_colors):
        inp[r, c] = color
    for r, c in marker_coords:
        inp[r, c] = marker_color

    if not _semantic_checks(
        inp,
        expected,
        motif_coords,
        motif_color,
        vector,
        bar_colors,
        bg,
        stats,
        need_overlap,
        need_wrap,
    ):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {MAX_GENERATION_ATTEMPTS} attempts")
