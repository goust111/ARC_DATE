from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]

BG = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
AREA_TO_LENGTH = [2, 3, 4, 5]
GEN_RETRIES = 400


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _row_segments(row: np.ndarray, row_index: int) -> list[dict]:
    segments: list[dict] = []
    start = 0
    color = int(row[0])
    width = int(row.shape[0])
    for c in range(1, width + 1):
        if c == width or int(row[c]) != color:
            segments.append(
                {
                    "row": int(row_index),
                    "left": int(start),
                    "right": int(c - 1),
                    "length": int(c - start),
                    "color": int(color),
                }
            )
            if c < width:
                start = c
                color = int(row[c])
    return segments


def _find_objects(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []

    for r in range(1, h - 1):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue

            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 1 <= nr < h - 1 and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            rows = [rr for rr, _cc in coords]
            cols = [cc for _rr, cc in coords]
            objects.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "area": int(len(coords)),
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each interior object to the unique border-bar segment whose color and length both match it.

    Input structure:
    - A black background surrounds all foreground content.
    - The top row and bottom row are full-width composite border bars made of contiguous colored segments.
    - Interior foreground pixels form small 4-connected monochrome objects whose colors also appear in the border bars.

    Rule:
    - Split the top and bottom border bars into maximal same-color segments and measure each segment's horizontal length.
    - For every interior object, compute its area. A border segment matches iff it has the same color and the same length as that area.
    - If exactly one matching segment exists, erase the object and redraw it as a 1-pixel-thick horizontal bar aligned to that segment: directly below a top-row match or directly above a bottom-row match.
    - If no unique matching segment exists, erase the object.

    Key insight:
    - Color alone is ambiguous and segment length alone is ambiguous; only their intersection identifies the correct destination.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h >= 4 and w >= 2

    bg = _background_color(g)
    segments = _row_segments(g[0], 0) + _row_segments(g[h - 1], h - 1)
    objects = _find_objects(g, bg)
    objects.sort(key=lambda obj: (int(obj["bbox"][1]), int(obj["bbox"][0])))

    out = np.full_like(g, bg)
    out[0, :] = g[0, :]
    out[h - 1, :] = g[h - 1, :]

    for obj in objects:
        matches = [
            seg
            for seg in segments
            if int(seg["color"]) == int(obj["color"]) and int(seg["length"]) == int(obj["area"])
        ]
        if len(matches) != 1:
            continue
        seg = matches[0]
        target_row = 1 if int(seg["row"]) == 0 else h - 2
        out[target_row, int(seg["left"]) : int(seg["right"]) + 1] = int(obj["color"])

    return out


def _normalize_cells(cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _c in cells)
    min_c = min(c for _r, c in cells)
    return sorted((r - min_r, c - min_c) for r, c in cells)


def _random_shape(area: int) -> np.ndarray:
    cells = {(0, 0)}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for dr, dc in DIR4:
                cell = (r + dr, c + dc)
                if cell not in cells:
                    frontier.append(cell)
        cells.add(random.choice(frontier))

    coords = np.array(_normalize_cells(cells), dtype=int)
    h = int(coords[:, 0].max()) + 1
    w = int(coords[:, 1].max()) + 1
    mask = np.zeros((h, w), dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True

    if random.random() < 0.5:
        mask = np.fliplr(mask)
    mask = np.rot90(mask, k=random.randrange(4))

    cropped = np.argwhere(mask)
    r0, c0 = cropped.min(axis=0)
    r1, c1 = cropped.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _layout_segments(pairs: list[tuple[int, int]], row_index: int) -> list[dict]:
    left = 0
    segments: list[dict] = []
    for color, length in pairs:
        segments.append(
            {
                "row": int(row_index),
                "left": int(left),
                "right": int(left + length - 1),
                "length": int(length),
                "color": int(color),
            }
        )
        left += length
    return segments


def _paint_bar_row(grid: np.ndarray, row_index: int, segments: list[dict]) -> None:
    for seg in segments:
        grid[int(row_index), int(seg["left"]) : int(seg["right"]) + 1] = int(seg["color"])


def _choose_blueprint(n_objects: int) -> tuple[list[int], list[int]]:
    if n_objects == 6:
        n_colors = 4
    else:
        n_colors = random.choice([3, 4])

    if n_colors == 3:
        lengths = list(random.choice([(2, 4, 5), (3, 4, 5)]))
    else:
        lengths = [2, 3, 4, 5]

    random.shuffle(lengths)
    colors = random.sample(PALETTE, n_colors)
    return colors, lengths


def _make_segments(colors: list[int], lengths: list[int]) -> tuple[list[dict], list[dict]]:
    top_pairs = [(colors[i], lengths[i]) for i in range(len(colors))]
    bottom_pairs = [(colors[i], lengths[(i + 1) % len(colors)]) for i in range(len(colors))]
    random.shuffle(top_pairs)
    random.shuffle(bottom_pairs)
    return _layout_segments(top_pairs, 0), _layout_segments(bottom_pairs, -1)


def _pick_targets(top_segments: list[dict], bottom_segments: list[dict], n_objects: int) -> list[dict]:
    k = len(top_segments)
    low = max(2, n_objects - k)
    high = min(k, n_objects - 2)
    n_top = random.randint(low, high)
    n_bottom = n_objects - n_top
    chosen = random.sample(top_segments, n_top) + random.sample(bottom_segments, n_bottom)
    random.shuffle(chosen)
    return chosen


def _cells_from_mask(mask: np.ndarray, r0: int, c0: int) -> set[tuple[int, int]]:
    cells = set()
    for r, c in np.argwhere(mask):
        cells.add((int(r0 + r), int(c0 + c)))
    return cells


def _halo(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr, dc in DIR8 + [(0, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                out.add((nr, nc))
    return out


def _place_objects(grid: np.ndarray, targets: list[dict]) -> bool:
    h, w = grid.shape
    occupied: set[tuple[int, int]] = set()
    blocked: set[tuple[int, int]] = set()

    objects = []
    for seg in targets:
        mask = _random_shape(int(seg["length"]))
        objects.append({"segment": seg, "mask": mask})

    objects.sort(
        key=lambda item: (
            -int(item["mask"].shape[0] * item["mask"].shape[1]),
            -int(item["segment"]["length"]),
            random.random(),
        )
    )

    for item in objects:
        mask = item["mask"]
        color = int(item["segment"]["color"])
        mh, mw = mask.shape

        candidates: list[tuple[int, int, set[tuple[int, int]]]] = []
        for r0 in range(2, h - mh - 1):
            for c0 in range(0, w - mw + 1):
                cells = _cells_from_mask(mask, r0, c0)
                if all(cell not in blocked for cell in cells):
                    candidates.append((r0, c0, cells))

        if not candidates:
            return False

        r0, c0, cells = random.choice(candidates)
        for r, c in cells:
            grid[r, c] = color
        occupied |= cells
        blocked |= _halo(cells, h, w)

    return True


def _validate_grid(grid: np.ndarray) -> bool:
    h, w = grid.shape
    if max(h, w) > 30:
        return False

    top_segments = _row_segments(grid[0], 0)
    bottom_segments = _row_segments(grid[h - 1], h - 1)
    all_segments = top_segments + bottom_segments
    pair_counts = Counter((int(seg["color"]), int(seg["length"])) for seg in all_segments)
    color_counts = Counter(int(seg["color"]) for seg in all_segments)
    length_counts = Counter(int(seg["length"]) for seg in all_segments)

    if set(int(seg["color"]) for seg in top_segments) != set(int(seg["color"]) for seg in bottom_segments):
        return False

    for seg in all_segments:
        if pair_counts[(int(seg["color"]), int(seg["length"]))] != 1:
            return False
        if color_counts[int(seg["color"])] < 2:
            return False
        if length_counts[int(seg["length"])] < 2:
            return False

    objects = _find_objects(grid, BG)
    if not 4 <= len(objects) <= 6:
        return False

    seen_pairs = set()
    top_hits = 0
    bottom_hits = 0
    for obj in objects:
        pair = (int(obj["color"]), int(obj["area"]))
        if pair in seen_pairs:
            return False
        seen_pairs.add(pair)

        matches = [
            seg
            for seg in all_segments
            if int(seg["color"]) == int(obj["color"]) and int(seg["length"]) == int(obj["area"])
        ]
        if len(matches) != 1:
            return False
        if int(matches[0]["row"]) == 0:
            top_hits += 1
        else:
            bottom_hits += 1

    if top_hits == 0 or bottom_hits == 0:
        return False

    try:
        out = main(grid)
    except AssertionError:
        return False

    if out.shape != grid.shape:
        return False
    if not np.array_equal(out[0], grid[0]) or not np.array_equal(out[h - 1], grid[h - 1]):
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    n_objects = random.randint(4, 6)
    colors, lengths = _choose_blueprint(n_objects)
    width = int(sum(lengths))
    min_height = max(12, 8 + n_objects)
    height = random.randint(min_height, min(min_height + 2, 16))

    top_segments, bottom_segments = _make_segments(colors, lengths)
    for seg in bottom_segments:
        seg["row"] = height - 1

    targets = _pick_targets(top_segments, bottom_segments, n_objects)

    grid = np.zeros((height, width), dtype=int)
    _paint_bar_row(grid, 0, top_segments)
    _paint_bar_row(grid, height - 1, bottom_segments)

    if not _place_objects(grid, targets):
        return None
    if not _validate_grid(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
