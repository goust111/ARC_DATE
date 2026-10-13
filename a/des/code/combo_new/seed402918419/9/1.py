from __future__ import annotations

import random

import numpy as np


DIR_VECS: dict[str, tuple[int, int]] = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

LEGEND_SLOT_ORIGINS = ((0, 0), (0, 4), (4, 0), (4, 4))
LEGEND_PATCH_SIZE = 7
GEN_ATTEMPTS = 400


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[counts.argmax()])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _is_rect_frame(comp: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(comp)
    if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
        return False
    expected: set[tuple[int, int]] = set()
    for r in range(r0, r1 + 1):
        expected.add((r, c0))
        expected.add((r, c1))
    for c in range(c0, c1 + 1):
        expected.add((r0, c))
        expected.add((r1, c))
    return set(comp) == expected


def _is_len2_segment(comp: list[tuple[int, int]]) -> bool:
    if len(comp) != 2:
        return False
    (r0, c0), (r1, c1) = comp
    return abs(r0 - r1) + abs(c0 - c1) == 1


def _find_frames(grid: np.ndarray, bg: int) -> tuple[int, list[tuple[int, int, int, int]]]:
    groups: dict[tuple[int, int, int], list[tuple[int, int, int, int]]] = {}
    for color in (int(v) for v in np.unique(grid) if int(v) != bg):
        for comp in _components(grid == color):
            if not _is_rect_frame(comp):
                continue
            box = _bbox(comp)
            h = box[2] - box[0] + 1
            w = box[3] - box[1] + 1
            groups.setdefault((color, h, w), []).append(box)
    assert groups, "expected at least one rectangular frame component"
    best_key, frames = max(groups.items(), key=lambda item: (len(item[1]), item[0][1] * item[0][2]))
    return int(best_key[0]), sorted(frames)


def _fragment_cells_in_frame(
    grid: np.ndarray,
    bg: int,
    frame_box: tuple[int, int, int, int],
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = frame_box
    cells = [
        (r, c)
        for r in range(r0 + 1, r1)
        for c in range(c0 + 1, c1)
        if int(grid[r, c]) != bg
    ]
    assert 2 <= len(cells) <= 3, "each frame should contain a short interior fragment"
    colors = {int(grid[r, c]) for r, c in cells}
    assert len(colors) == 1, "fragment must be monochrome"
    rows = {r for r, _ in cells}
    cols = {c for _, c in cells}
    assert len(rows) == 1 or len(cols) == 1, "fragment must be a straight line"
    return cells


def _infer_direction(
    fragment: list[tuple[int, int]],
    frame_box: tuple[int, int, int, int],
) -> str:
    r0, c0, r1, c1 = frame_box
    rows = {r for r, _ in fragment}
    cols = {c for _, c in fragment}
    center_r = (r0 + r1) / 2.0
    center_c = (c0 + c1) / 2.0
    if len(rows) == 1:
        mean_c = sum(c for _, c in fragment) / float(len(fragment))
        assert abs(mean_c - center_c) > 1e-6, "horizontal fragment may not be centered"
        return "left" if mean_c < center_c else "right"
    mean_r = sum(r for r, _ in fragment) / float(len(fragment))
    assert len(cols) == 1 and abs(mean_r - center_r) > 1e-6, "vertical fragment may not be centered"
    return "up" if mean_r < center_r else "down"


def _decode_legend_segment(
    grid: np.ndarray,
    comp: list[tuple[int, int]],
    grey_color: int,
    bg: int,
) -> tuple[str, int]:
    h, w = grid.shape
    rows = {r for r, _ in comp}
    cols = {c for _, c in comp}
    if len(rows) == 1:
        r = next(iter(rows))
        c0 = min(cols)
        c1 = max(cols)
        left_ok = c0 - 1 >= 0 and int(grid[r, c0 - 1]) not in (bg, grey_color)
        right_ok = c1 + 1 < w and int(grid[r, c1 + 1]) not in (bg, grey_color)
        assert left_ok ^ right_ok, "horizontal legend segment must have one colored endpoint"
        if left_ok:
            return "left", int(grid[r, c0 - 1])
        return "right", int(grid[r, c1 + 1])
    assert len(cols) == 1, "legend segment must be horizontal or vertical"
    c = next(iter(cols))
    r0 = min(rows)
    r1 = max(rows)
    up_ok = r0 - 1 >= 0 and int(grid[r0 - 1, c]) not in (bg, grey_color)
    down_ok = r1 + 1 < h and int(grid[r1 + 1, c]) not in (bg, grey_color)
    assert up_ok ^ down_ok, "vertical legend segment must have one colored endpoint"
    if up_ok:
        return "up", int(grid[r0 - 1, c])
    return "down", int(grid[r1 + 1, c])


def _find_legend_mapping(
    grid: np.ndarray,
    bg: int,
    frames: list[tuple[int, int, int, int]],
) -> dict[str, int]:
    blocked = np.zeros_like(grid, dtype=bool)
    for r0, c0, r1, c1 in frames:
        blocked[r0 : r1 + 1, c0 : c1 + 1] = True

    segments: list[tuple[int, list[tuple[int, int]]]] = []
    for color in (int(v) for v in np.unique(grid) if int(v) != bg):
        mask = (grid == color) & ~blocked
        for comp in _components(mask):
            if _is_len2_segment(comp):
                segments.append((color, comp))

    assert segments, "expected legend segments outside frames"
    color_counts: dict[int, int] = {}
    for color, _ in segments:
        color_counts[color] = color_counts.get(color, 0) + 1
    grey_color = max(color_counts, key=color_counts.get)

    mapping: dict[str, int] = {}
    for color, comp in segments:
        if color != grey_color:
            continue
        direction, mapped_color = _decode_legend_segment(grid, comp, grey_color, bg)
        mapping[direction] = mapped_color

    assert set(mapping) == set(DIR_VECS), "legend must define all four cardinal directions"
    return mapping


def _channel_start(frame_box: tuple[int, int, int, int], direction: str) -> tuple[int, int]:
    r0, c0, r1, c1 = frame_box
    mid_r = (r0 + r1) // 2
    mid_c = (c0 + c1) // 2
    if direction == "up":
        return r0 - 1, mid_c
    if direction == "down":
        return r1 + 1, mid_c
    if direction == "left":
        return mid_r, c0 - 1
    return mid_r, c1 + 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the corner legend to map directions to colors, then extend colored channels outward from each framed fragment.

    Input structure:
    - The background is a single dominant color.
    - Several same-sized one-pixel-thick rectangular frames share one color and do not overlap.
    - Each frame contains a short 2-3 pixel monochrome fragment inside the hollow area; the fragment is horizontal or vertical and offset toward one side.
    - A corner legend contains four grey 2-pixel direction segments, each with a single colored pixel attached at the pointed end.

    Rule:
    - Detect the legend and build a direction-to-color lookup from the grey segment orientation plus the adjacent colored endpoint.
    - For each frame, infer the fragment direction: horizontal fragments mean left or right, vertical fragments mean up or down, and the side-biased offset chooses which of the two.
    - Look up that direction's output color in the legend.
    - Starting from the outside midpoint of the matching frame side, paint a one-pixel-wide straight channel in that direction across background cells only, stopping before any non-background object or at the grid edge.
    - Leave the frames, fragments, and legend unchanged.

    Key insight:
    - The frame interior fragment encodes direction, while the corner legend independently encodes which visible color corresponds to that direction; both cues must be combined before drawing the channels.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _most_common_color(g)
    _frame_color, frames = _find_frames(g, bg)
    mapping = _find_legend_mapping(g, bg, frames)

    out = g.copy()
    for frame_box in frames:
        fragment = _fragment_cells_in_frame(g, bg, frame_box)
        direction = _infer_direction(fragment, frame_box)
        color = mapping[direction]
        dr, dc = DIR_VECS[direction]
        r, c = _channel_start(frame_box, direction)
        while 0 <= r < h and 0 <= c < w and int(g[r, c]) == bg:
            out[r, c] = color
            r += dr
            c += dc
    return out


def _expanded_bbox(
    box: tuple[int, int, int, int],
    pad: int,
    h: int,
    w: int,
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    return max(0, r0 - pad), max(0, c0 - pad), min(h - 1, r1 + pad), min(w - 1, c1 + pad)


def _mark_box(mask: np.ndarray, box: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = box
    mask[r0 : r1 + 1, c0 : c1 + 1] = True


def _channel_cells(
    frame_box: tuple[int, int, int, int],
    direction: str,
    h: int,
    w: int,
) -> list[tuple[int, int]]:
    dr, dc = DIR_VECS[direction]
    r, c = _channel_start(frame_box, direction)
    cells: list[tuple[int, int]] = []
    while 0 <= r < h and 0 <= c < w:
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _draw_frame(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color


def _valid_fragment_positions(frame_h: int, frame_w: int, direction: str) -> list[list[tuple[int, int]]]:
    ih = frame_h - 2
    iw = frame_w - 2
    center_r = (ih - 1) / 2.0
    center_c = (iw - 1) / 2.0
    positions: list[list[tuple[int, int]]] = []
    if direction in ("up", "down"):
        for length in (2, 3):
            for col in range(1, iw - 1):
                for start in range(1, ih - length):
                    coords = [(start + d, col) for d in range(length)]
                    mean_r = sum(r for r, _ in coords) / float(length)
                    if direction == "up" and mean_r < center_r:
                        positions.append(coords)
                    if direction == "down" and mean_r > center_r:
                        positions.append(coords)
    else:
        for length in (2, 3):
            for row in range(1, ih - 1):
                for start in range(1, iw - length):
                    coords = [(row, start + d) for d in range(length)]
                    mean_c = sum(c for _, c in coords) / float(length)
                    if direction == "left" and mean_c < center_c:
                        positions.append(coords)
                    if direction == "right" and mean_c > center_c:
                        positions.append(coords)
    return positions


def _legend_entry_cells(direction: str) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    if direction == "up":
        return [(1, 1), (2, 1)], (0, 1)
    if direction == "down":
        return [(0, 1), (1, 1)], (2, 1)
    if direction == "left":
        return [(1, 1), (1, 2)], (1, 0)
    return [(1, 0), (1, 1)], (1, 2)


def _place_legend(
    grid: np.ndarray,
    bg: int,
    grey_color: int,
    mapping: dict[str, int],
    corner: str,
) -> tuple[int, int, int, int]:
    patch = np.full((LEGEND_PATCH_SIZE, LEGEND_PATCH_SIZE), bg, dtype=int)
    slots = list(LEGEND_SLOT_ORIGINS)
    random.shuffle(slots)
    for direction, (sr, sc) in zip(mapping.keys(), slots, strict=False):
        grey_cells, color_cell = _legend_entry_cells(direction)
        for gr, gc in grey_cells:
            patch[sr + gr, sc + gc] = grey_color
        cr, cc = color_cell
        patch[sr + cr, sc + cc] = mapping[direction]

    h, w = grid.shape
    if corner == "tl":
        r0, c0 = 0, 0
    elif corner == "tr":
        r0, c0 = 0, w - LEGEND_PATCH_SIZE
    elif corner == "bl":
        r0, c0 = h - LEGEND_PATCH_SIZE, 0
    else:
        r0, c0 = h - LEGEND_PATCH_SIZE, w - LEGEND_PATCH_SIZE
    mask = patch != bg
    grid[r0 : r0 + LEGEND_PATCH_SIZE, c0 : c0 + LEGEND_PATCH_SIZE][mask] = patch[mask]
    return r0, c0, r0 + LEGEND_PATCH_SIZE - 1, c0 + LEGEND_PATCH_SIZE - 1


def _try_build_case() -> np.ndarray | None:
    h = random.randint(22, 30)
    w = random.randint(22, 30)
    colors = random.sample(range(10), 8)
    bg, frame_color, fragment_color, grey_color = colors[:4]
    mapping = {
        "up": colors[4],
        "down": colors[5],
        "left": colors[6],
        "right": colors[7],
    }

    grid = np.full((h, w), bg, dtype=int)
    corner = random.choice(("tl", "tr", "bl", "br"))
    legend_box = _place_legend(grid, bg, grey_color, mapping, corner)

    used = np.zeros((h, w), dtype=bool)
    _mark_box(used, _expanded_bbox(legend_box, 1, h, w))

    frame_count = random.randint(2, 4)
    directions = random.sample(list(DIR_VECS), frame_count)

    size_choices = [7] if frame_count == 4 else [7, 9]
    frame_h = random.choice(size_choices)
    frame_w = random.choice(size_choices)
    if frame_h >= h - 1 or frame_w >= w - 1:
        return None

    placements: list[tuple[tuple[int, int, int, int], str]] = []
    random.shuffle(directions)
    for direction in directions:
        frag_choices = _valid_fragment_positions(frame_h, frame_w, direction)
        if not frag_choices:
            return None
        placed = False
        for _ in range(300):
            r0 = random.randint(1, h - frame_h - 1)
            c0 = random.randint(1, w - frame_w - 1)
            box = (r0, c0, r0 + frame_h - 1, c0 + frame_w - 1)
            halo = _expanded_bbox(box, 1, h, w)
            if np.any(used[halo[0] : halo[2] + 1, halo[1] : halo[3] + 1]):
                continue
            path = _channel_cells(box, direction, h, w)
            if not path or any(used[r, c] for r, c in path):
                continue
            _mark_box(used, halo)
            for r, c in path:
                used[r, c] = True
            placements.append((box, direction))
            placed = True
            break
        if not placed:
            return None

    for box, direction in placements:
        _draw_frame(grid, box, frame_color)
        fragment = random.choice(_valid_fragment_positions(frame_h, frame_w, direction))
        r0, c0, _, _ = box
        for rr, cc in fragment:
            grid[r0 + 1 + rr, c0 + 1 + cc] = fragment_color

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _try_build_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid direction-legend frame puzzle")
