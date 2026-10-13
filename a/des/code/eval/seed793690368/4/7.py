from __future__ import annotations

import numpy as np


BG = 0
GREY = 5
RED = 2
ORANGE = 7
YELLOW = 4
GREEN = 3
BLUE = 1

RAINBOW_ORDER = [RED, ORANGE, YELLOW, GREEN, BLUE]
RAINBOW_RANK = {color: i for i, color in enumerate(RAINBOW_ORDER)}
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    return _bbox_from_coords(coords)


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _separator_row(grid: np.ndarray, bg: int) -> tuple[int, int]:
    for r in range(grid.shape[0]):
        row = grid[r]
        if np.all(row == row[0]) and int(row[0]) != bg:
            return int(r), int(row[0])
    raise AssertionError("expected a full-width separator line")


def _frame_crop(lower: np.ndarray, frame_color: int) -> tuple[tuple[int, int, int, int], np.ndarray]:
    bbox = _bbox_from_mask(lower == frame_color)
    if bbox is None:
        raise AssertionError("expected a grey frame below the separator")
    r0, c0, r1, c1 = bbox
    return bbox, lower[r0 : r1 + 1, c0 : c1 + 1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract monochrome stamp clusters from the upper region, then overlay them on the lower canvas
    in rainbow order: red -> orange -> yellow -> green -> blue.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _most_common_color(g)
    split_row, grey = _separator_row(g, bg)
    upper = g[:split_row]
    lower = g[split_row + 1 :]

    _frame_bbox, frame = _frame_crop(lower, grey)
    out = np.full(frame.shape, bg, dtype=int)
    out[0, :] = grey
    out[-1, :] = grey
    out[:, 0] = grey
    out[:, -1] = grey
    interior = out[1:-1, 1:-1]

    mask = (upper != bg) & (upper != grey)
    stamps: list[tuple[int, int, np.ndarray]] = []
    for coords in _connected_components(mask):
        colors = {int(upper[r, c]) for r, c in coords}
        assert len(colors) == 1
        color = colors.pop()
        if color not in RAINBOW_RANK:
            continue
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        patch = upper[r0 : r1 + 1, c0 : c1 + 1]
        stamps.append((RAINBOW_RANK[color], color, patch == color))

    assert stamps, "expected at least one stamp cluster"
    stamps.sort(key=lambda item: item[0])

    for _rank, color, patch_mask in stamps:
        ph, pw = patch_mask.shape
        rh = min(ph, interior.shape[0])
        rw = min(pw, interior.shape[1])
        view = interior[:rh, :rw]
        view[patch_mask[:rh, :rw]] = color

    return out


def _manhattan_path(
    rng: np.random.Generator, start: tuple[int, int], end: tuple[int, int]
) -> set[tuple[int, int]]:
    r, c = start
    tr, tc = end
    path = {(r, c)}
    while (r, c) != (tr, tc):
        options: list[tuple[int, int]] = []
        if r < tr:
            options.append((r + 1, c))
        elif r > tr:
            options.append((r - 1, c))
        if c < tc:
            options.append((r, c + 1))
        elif c > tc:
            options.append((r, c - 1))
        r, c = options[int(rng.integers(len(options)))]
        path.add((r, c))
    return path


def _random_stamp_mask(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    total = h * w
    for _ in range(200):
        center = (int(rng.integers(h)), int(rng.integers(w)))
        anchors = [
            (0, int(rng.integers(w))),
            (h - 1, int(rng.integers(w))),
            (int(rng.integers(h)), 0),
            (int(rng.integers(h)), w - 1),
        ]

        cells = {center}
        for anchor in anchors:
            cells |= _manhattan_path(rng, center, anchor)

        min_fill = len(cells)
        max_fill = max(min_fill, int(np.ceil(total * float(rng.uniform(0.55, 0.9)))))
        target = int(rng.integers(min_fill, max_fill + 1))
        while len(cells) < target:
            frontier: list[tuple[int, int]] = []
            seen = set()
            for r, c in cells:
                for dr, dc in NEIGHBORS4:
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells and (nr, nc) not in seen:
                        frontier.append((nr, nc))
                        seen.add((nr, nc))
            if not frontier:
                break
            cells.add(frontier[int(rng.integers(len(frontier)))])

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        bbox = _bbox_from_mask(mask)
        if bbox == (0, 0, h - 1, w - 1):
            return mask
    raise RuntimeError("failed to build a valid stamp mask")


def _render_output(masks: dict[int, np.ndarray], colors: list[int], h: int, w: int) -> np.ndarray:
    out = np.full((h + 2, w + 2), BG, dtype=int)
    out[0, :] = GREY
    out[-1, :] = GREY
    out[:, 0] = GREY
    out[:, -1] = GREY
    interior = out[1:-1, 1:-1]
    for color in sorted(colors, key=lambda value: RAINBOW_RANK[value]):
        interior[masks[color]] = color
    return out


def _composition_ok(masks: dict[int, np.ndarray], colors: list[int]) -> bool:
    ordered = sorted(colors, key=lambda value: RAINBOW_RANK[value])
    h, w = next(iter(masks.values())).shape
    overlap_count = np.zeros((h, w), dtype=int)
    owner = np.zeros((h, w), dtype=int)

    for color in ordered:
        mask = masks[color]
        overlap_count += mask.astype(int)
        owner[mask] = color

    if not np.any(overlap_count > 1):
        return False
    return all(np.any(owner == color) for color in ordered)


def _boxes_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    return not (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _place_stamps(
    rng: np.random.Generator, top_h: int, width: int, stamp_h: int, stamp_w: int, colors: list[int]
) -> dict[int, tuple[int, int]] | None:
    order = list(colors)
    rng.shuffle(order)
    boxes: list[tuple[int, int, int, int]] = []
    placements: dict[int, tuple[int, int]] = {}

    for color in order:
        for _ in range(200):
            r = int(rng.integers(0, top_h - stamp_h + 1))
            c = int(rng.integers(0, width - stamp_w + 1))
            box = (r, c, r + stamp_h - 1, c + stamp_w - 1)
            if any(_boxes_too_close(box, other) for other in boxes):
                continue
            placements[color] = (r, c)
            boxes.append(box)
            break
        else:
            return None
    return placements


def _build_attempt(rng: np.random.Generator) -> np.ndarray | None:
    stamp_h = int(rng.integers(3, 6))
    stamp_w = int(rng.integers(3, 6))
    color_count = int(rng.integers(2, len(RAINBOW_ORDER) + 1))
    color_idx = rng.choice(len(RAINBOW_ORDER), size=color_count, replace=False)
    colors = [RAINBOW_ORDER[int(i)] for i in color_idx]

    masks = {color: _random_stamp_mask(rng, stamp_h, stamp_w) for color in colors}
    if not _composition_ok(masks, colors):
        return None

    row_capacity = 2 if color_count >= 4 else 1
    top_h_min = max(stamp_h + 3, row_capacity * stamp_h + row_capacity + 1)
    top_h = int(rng.integers(top_h_min, top_h_min + 5))

    max_per_row = min(3, color_count)
    width_min = max(stamp_w + 4, max_per_row * stamp_w + (max_per_row + 1) * 2)
    width = int(rng.integers(width_min, width_min + 9))

    placements = _place_stamps(rng, top_h, width, stamp_h, stamp_w, colors)
    if placements is None:
        return None

    frame_h = stamp_h + 2
    frame_w = stamp_w + 2
    lower_gap = int(rng.integers(1, 4))
    lower_bottom = int(rng.integers(1, 4))
    total_h = top_h + 1 + lower_gap + frame_h + lower_bottom
    grid = np.full((total_h, width), BG, dtype=int)
    grid[top_h, :] = GREY

    for color, (r0, c0) in placements.items():
        mask = masks[color]
        region = grid[r0 : r0 + stamp_h, c0 : c0 + stamp_w]
        region[mask] = color

    frame_r = top_h + 1 + lower_gap
    frame_c = int(rng.integers(1, width - frame_w))
    grid[frame_r, frame_c : frame_c + frame_w] = GREY
    grid[frame_r + frame_h - 1, frame_c : frame_c + frame_w] = GREY
    grid[frame_r : frame_r + frame_h, frame_c] = GREY
    grid[frame_r : frame_r + frame_h, frame_c + frame_w - 1] = GREY

    expected = _render_output(masks, colors, stamp_h, stamp_w)
    if np.array_equal(main(grid), expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(400):
        grid = _build_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
