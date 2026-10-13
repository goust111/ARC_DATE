from __future__ import annotations

import random
from collections import Counter

import numpy as np


NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    order = np.argsort(-counts)
    bg = int(values[int(order[0])])
    assert counts[int(order[0])] > counts[int(order[1])] if len(order) > 1 else True
    return bg


def _draw_frame(grid: np.ndarray, top: int, left: int, height: int, width: int, color: int) -> None:
    bottom = top + height - 1
    right = left + width - 1
    grid[top, left : right + 1] = int(color)
    grid[bottom, left : right + 1] = int(color)
    grid[top : bottom + 1, left] = int(color)
    grid[top : bottom + 1, right] = int(color)


def _render_output(frames: list[dict[str, int]], bg: int) -> np.ndarray:
    ordered = sorted(frames, key=lambda frame: int(frame["marker_count"]))
    out_h = max(int(frame["height"]) for frame in ordered)
    out_w = sum(int(frame["width"]) for frame in ordered) + max(0, len(ordered) - 1)
    out = np.full((out_h, out_w), int(bg), dtype=int)

    col = 0
    for frame in ordered:
        height = int(frame["height"])
        width = int(frame["width"])
        top = (out_h - height) // 2
        _draw_frame(out, top, col, height, width, int(frame["color"]))
        col += width + 1

    return out


def _border_mask(shape: tuple[int, int], r0: int, c0: int, r1: int, c1: int) -> np.ndarray:
    border = np.zeros(shape, dtype=bool)
    border[r0, c0 : c1 + 1] = True
    border[r1, c0 : c1 + 1] = True
    border[r0 : r1 + 1, c0] = True
    border[r0 : r1 + 1, c1] = True
    return border


def _is_valid_frame_candidate(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> bool:
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False
    if not np.all(mask[r0, c0 : c1 + 1]):
        return False
    if not np.all(mask[r1, c0 : c1 + 1]):
        return False
    if not np.all(mask[r0 : r1 + 1, c0]):
        return False
    if not np.all(mask[r0 : r1 + 1, c1]):
        return False

    border = _border_mask(mask.shape, r0, c0, r1, c1)
    extras = np.argwhere(mask & (~border))
    height, width = mask.shape

    for rr, cc in extras:
        rr = int(rr)
        cc = int(cc)
        if rr < r0 - 1 or rr > r1 + 1 or cc < c0 - 1 or cc > c1 + 1:
            return False
        if not any(
            0 <= rr + dr < height and 0 <= cc + dc < width and border[rr + dr, cc + dc]
            for dr, dc in NEIGHBORS4
        ):
            return False

    return True


def _extract_frame(mask: np.ndarray, color: int) -> dict[str, int]:
    coords = np.argwhere(mask)
    assert len(coords) > 0, "frame color must appear"

    rows = sorted(set(int(r) for r in coords[:, 0]))
    cols = sorted(set(int(c) for c in coords[:, 1]))
    total_area = int(mask.sum())
    candidates: list[dict[str, int]] = []

    for i, r0 in enumerate(rows):
        for r1 in rows[i + 1 :]:
            if r1 - r0 < 2:
                continue
            for j, c0 in enumerate(cols):
                for c1 in cols[j + 1 :]:
                    if c1 - c0 < 2:
                        continue
                    if not _is_valid_frame_candidate(mask, r0, c0, r1, c1):
                        continue

                    frame_h = int(r1 - r0 + 1)
                    frame_w = int(c1 - c0 + 1)
                    border_area = int(2 * (frame_h + frame_w) - 4)
                    marker_count = int(total_area - border_area)
                    if marker_count <= 0:
                        continue

                    candidates.append(
                        {
                            "r0": int(r0),
                            "c0": int(c0),
                            "r1": int(r1),
                            "c1": int(c1),
                            "height": frame_h,
                            "width": frame_w,
                            "marker_count": marker_count,
                            "color": int(color),
                            "border_area": border_area,
                        }
                    )

    assert candidates, "failed to extract a clean frame"
    best_area = max(int(candidate["border_area"]) for candidate in candidates)
    best = [candidate for candidate in candidates if int(candidate["border_area"]) == best_area]
    assert len(best) == 1, "frame geometry must be unambiguous"
    return best[0]


def _extract_frames(grid: np.ndarray, bg: int) -> list[dict[str, int]]:
    g = np.asarray(grid, dtype=int)
    colors = [int(color) for color in np.unique(g) if int(color) != int(bg)]
    assert len(colors) >= 2, "expected multiple nested frames"

    frames = [_extract_frame(g == color, color) for color in colors]
    marker_counts = [int(frame["marker_count"]) for frame in frames]
    assert len(marker_counts) == len(set(marker_counts)), "marker counts must be distinct"

    centers = {(int(frame["r0"]) + int(frame["r1"]), int(frame["c0"]) + int(frame["c1"])) for frame in frames}
    assert len(centers) == 1, "all frames must be concentric"

    by_size = sorted(frames, key=lambda frame: int(frame["height"]) * int(frame["width"]), reverse=True)
    for outer, inner in zip(by_size, by_size[1:]):
        assert int(outer["r0"]) < int(inner["r0"]) < int(inner["r1"]) < int(outer["r1"])
        assert int(outer["c0"]) < int(inner["c0"]) < int(inner["c1"]) < int(outer["c1"])

    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split concentric marked frames apart and reorder them by marker count.

    Input structure:
    - The background is dominant.
    - Several one-pixel-thick concentric rectangular frames sit at the center.
    - Each frame uses a distinct color and has same-color extra pixels attached
      to its sides, locally making the border look double-thick.

    Rule:
    - Recover each clean rectangular frame and count its extra marker pixels.
    - Sort the frames from fewest markers to most markers.
    - Draw the clean frames left-to-right with one background column between them.
    - Do not keep the marker pixels in the output.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    frames = _extract_frames(g, bg)
    return _render_output(frames, bg)


# --------------- generator ---------------

COLOR_IDS = tuple(range(10))
FRAME_COUNT_RANGE = (2, 5)
GRID_MAX_SIZE = 30
OUTER_SIZE_MAX = 24
OUTER_MARGIN_RANGE = (1, 4)
MAX_MARKER_VALUE = 6
GENERATION_ATTEMPTS = 400
MARKER_ASSIGN_ATTEMPTS = 80


def _support_groups_for_bbox(
    bbox: tuple[int, int, int, int],
    shape: tuple[int, int],
    clean_pixels: set[tuple[int, int]],
) -> list[list[tuple[int, int]]]:
    r0, c0, r1, c1 = bbox
    height, width = shape
    groups: list[list[tuple[int, int]]] = []

    for c in range(c0 + 1, c1):
        top_options = []
        for rr in (r0 - 1, r0 + 1):
            cell = (int(rr), int(c))
            if 0 <= rr < height and cell not in clean_pixels:
                top_options.append(cell)
        if top_options:
            groups.append(top_options)

        bottom_options = []
        for rr in (r1 + 1, r1 - 1):
            cell = (int(rr), int(c))
            if 0 <= rr < height and cell not in clean_pixels:
                bottom_options.append(cell)
        if bottom_options:
            groups.append(bottom_options)

    for r in range(r0 + 1, r1):
        left_options = []
        for cc in (c0 - 1, c0 + 1):
            cell = (int(r), int(cc))
            if 0 <= cc < width and cell not in clean_pixels:
                left_options.append(cell)
        if left_options:
            groups.append(left_options)

        right_options = []
        for cc in (c1 + 1, c1 - 1):
            cell = (int(r), int(cc))
            if 0 <= cc < width and cell not in clean_pixels:
                right_options.append(cell)
        if right_options:
            groups.append(right_options)

    return groups


def _render_input(
    shape: tuple[int, int],
    bg: int,
    frame_colors: list[int],
    bboxes: list[tuple[int, int, int, int]],
    markers: list[list[tuple[int, int]]],
) -> np.ndarray:
    grid = np.full(shape, int(bg), dtype=int)
    for color, (r0, c0, r1, c1) in zip(frame_colors, bboxes):
        _draw_frame(grid, int(r0), int(c0), int(r1 - r0 + 1), int(c1 - c0 + 1), int(color))
    for color, marker_cells in zip(frame_colors, markers):
        for r, c in marker_cells:
            assert int(grid[r, c]) == int(bg), "markers must occupy empty cells"
            grid[r, c] = int(color)
    return grid


def _specs_from_bboxes(
    frame_colors: list[int],
    bboxes: list[tuple[int, int, int, int]],
    counts: list[int],
) -> list[dict[str, int]]:
    specs = []
    for color, (r0, c0, r1, c1), marker_count in zip(frame_colors, bboxes, counts):
        specs.append(
            {
                "color": int(color),
                "r0": int(r0),
                "c0": int(c0),
                "r1": int(r1),
                "c1": int(c1),
                "height": int(r1 - r0 + 1),
                "width": int(c1 - c0 + 1),
                "marker_count": int(marker_count),
            }
        )
    return specs


def _dominant_bg(grid: np.ndarray, bg: int) -> bool:
    counts = Counter(map(int, grid.ravel()))
    return all(color == int(bg) or count < counts[int(bg)] for color, count in counts.items())


def _gen_attempt() -> np.ndarray | None:
    n_frames = random.randint(*FRAME_COUNT_RANGE)
    outer_min = 5 + 4 * (n_frames - 1)

    outer_h = random.randint(outer_min, OUTER_SIZE_MAX)
    outer_w = random.randint(outer_min, OUTER_SIZE_MAX)

    max_margin_r = min(OUTER_MARGIN_RANGE[1], (GRID_MAX_SIZE - outer_h) // 2)
    max_margin_c = min(OUTER_MARGIN_RANGE[1], (GRID_MAX_SIZE - outer_w) // 2)
    if max_margin_r < OUTER_MARGIN_RANGE[0] or max_margin_c < OUTER_MARGIN_RANGE[0]:
        return None

    margin_r = random.randint(OUTER_MARGIN_RANGE[0], max_margin_r)
    margin_c = random.randint(OUTER_MARGIN_RANGE[0], max_margin_c)
    grid_h = outer_h + 2 * margin_r
    grid_w = outer_w + 2 * margin_c

    bg = random.choice(COLOR_IDS)
    frame_colors = random.sample([color for color in COLOR_IDS if color != bg], n_frames)

    bboxes: list[tuple[int, int, int, int]] = []
    clean_pixels: set[tuple[int, int]] = set()
    for layer in range(n_frames):
        r0 = margin_r + 2 * layer
        c0 = margin_c + 2 * layer
        r1 = grid_h - margin_r - 1 - 2 * layer
        c1 = grid_w - margin_c - 1 - 2 * layer
        bboxes.append((int(r0), int(c0), int(r1), int(c1)))

        for c in range(c0, c1 + 1):
            clean_pixels.add((int(r0), int(c)))
            clean_pixels.add((int(r1), int(c)))
        for r in range(r0 + 1, r1):
            clean_pixels.add((int(r), int(c0)))
            clean_pixels.add((int(r), int(c1)))

    support_groups = [_support_groups_for_bbox(bbox, (grid_h, grid_w), clean_pixels) for bbox in bboxes]
    min_groups = min(len(groups) for groups in support_groups)
    max_marker_value = min(MAX_MARKER_VALUE, min_groups - 1)
    if max_marker_value < n_frames:
        return None

    counts = random.sample(range(1, max_marker_value + 1), n_frames)

    for _ in range(MARKER_ASSIGN_ATTEMPTS):
        occupied = set(clean_pixels)
        markers: list[list[tuple[int, int]]] = [[] for _ in range(n_frames)]

        for idx in sorted(range(n_frames), key=lambda i: int(counts[i]), reverse=True):
            groups = support_groups[idx][:]
            random.shuffle(groups)
            chosen: list[tuple[int, int]] = []
            local_used: set[tuple[int, int]] = set()

            for options in groups:
                free = [cell for cell in options if cell not in occupied and cell not in local_used]
                if not free:
                    continue
                cell = random.choice(free)
                chosen.append(cell)
                local_used.add(cell)
                occupied.add(cell)
                if len(chosen) == int(counts[idx]):
                    break

            if len(chosen) != int(counts[idx]):
                break
            markers[idx] = chosen
        else:
            inp = _render_input((grid_h, grid_w), bg, frame_colors, bboxes, markers)
            specs = _specs_from_bboxes(frame_colors, bboxes, counts)
            expected = _render_output(specs, bg)
            if _dominant_bg(inp, bg) and np.array_equal(main(inp), expected):
                return inp

    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
