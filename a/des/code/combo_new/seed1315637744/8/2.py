import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
MAX_ATTEMPTS = 400


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _border_cells(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells = {(r0, c) for c in range(c0, c1 + 1)}
    cells.update((r1, c) for c in range(c0, c1 + 1))
    cells.update((r, c0) for r in range(r0, r1 + 1))
    cells.update((r, c1) for r in range(r0, r1 + 1))
    return cells


def _first_seen_colors(grid: np.ndarray, excluded: int) -> list[int]:
    seen = set()
    colors = []
    for value in grid.ravel():
        color = int(value)
        if color in {BG, excluded} or color in seen:
            continue
        seen.add(color)
        colors.append(color)
    return colors


def _rects_overlap(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    pad: int = 0,
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0 - pad
        or br1 + pad < ar0 - pad
        or ac1 + pad < bc0 - pad
        or bc1 + pad < ac0 - pad
    )


def _flip_rect(
    rect: tuple[int, int, int, int],
    height: int,
    width: int,
    flip_v: bool,
    flip_h: bool,
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = rect
    if flip_v:
        r0, r1 = height - 1 - r1, height - 1 - r0
    if flip_h:
        c0, c1 = width - 1 - c1, width - 1 - c0
    return (r0, c0, r1, c1)


def _sample_overlap_pair(
    rng: random.Random,
    height: int,
    width: int,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]] | None:
    if height < 6 or width < 6:
        return None

    for _ in range(120):
        ha = rng.randint(4, min(8, height - 1))
        wa = rng.randint(4, min(8, width - 1))
        dr = rng.randint(1, ha - 2)
        dc = rng.randint(1, wa - 2)
        extra_h = rng.randint(1, 3)
        extra_w = rng.randint(1, 3)
        if ha + extra_h > height or wa + extra_w > width:
            continue

        max_r0 = height - (ha + extra_h)
        max_c0 = width - (wa + extra_w)
        if max_r0 < 0 or max_c0 < 0:
            continue

        r0 = rng.randint(0, max_r0)
        c0 = rng.randint(0, max_c0)
        rect_a = (r0, c0, r0 + ha - 1, c0 + wa - 1)
        hb = ha - dr + extra_h
        wb = wa - dc + extra_w
        rect_b = (r0 + dr, c0 + dc, r0 + dr + hb - 1, c0 + dc + wb - 1)

        flip_v = rng.choice([False, True])
        flip_h = rng.choice([False, True])
        rect_a = _flip_rect(rect_a, height, width, flip_v, flip_h)
        rect_b = _flip_rect(rect_b, height, width, flip_v, flip_h)

        if _border_cells(rect_a) & _border_cells(rect_b):
            return rect_a, rect_b
    return None


def _sample_separated_rect(
    rng: random.Random,
    height: int,
    width: int,
    existing: list[tuple[int, int, int, int]],
    pad: int = 1,
) -> tuple[int, int, int, int] | None:
    for _ in range(150):
        rh = rng.randint(3, min(8, height))
        rw = rng.randint(3, min(8, width))
        r0 = rng.randint(0, height - rh)
        c0 = rng.randint(0, width - rw)
        rect = (r0, c0, r0 + rh - 1, c0 + rw - 1)
        if all(not _rects_overlap(rect, other, pad=pad) for other in existing):
            return rect
    return None


def _mandatory_corner_sets(
    rect: tuple[int, int, int, int],
    count: int,
) -> list[list[tuple[int, int]]]:
    r0, c0, r1, c1 = rect
    corner_triplets = [
        [(r0, c0), (r0, c1), (r1, c0)],
        [(r0, c0), (r0, c1), (r1, c1)],
        [(r0, c0), (r1, c0), (r1, c1)],
        [(r0, c1), (r1, c0), (r1, c1)],
    ]
    if count >= 4:
        corner_triplets.append([(r0, c0), (r0, c1), (r1, c0), (r1, c1)])
    return corner_triplets


def _scatter_points(
    rng: random.Random,
    rect: tuple[int, int, int, int],
    count: int,
    blocked: set[tuple[int, int]],
    avoid: set[tuple[int, int]] | None = None,
) -> set[tuple[int, int]] | None:
    avoid = set() if avoid is None else set(avoid)
    r0, c0, r1, c1 = rect
    all_cells = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
    viable = [cell for cell in all_cells if cell not in blocked and cell not in avoid]
    if len(viable) < count:
        return None

    patterns = _mandatory_corner_sets(rect, count)
    rng.shuffle(patterns)
    for mandatory in patterns:
        if any(cell in blocked or cell in avoid for cell in mandatory):
            continue
        remaining = [cell for cell in viable if cell not in mandatory]
        if len(remaining) < count - len(mandatory):
            continue
        chosen = set(mandatory)
        chosen.update(rng.sample(remaining, count - len(mandatory)))
        return chosen
    return None


def _has_any_border_overlap(rects: list[tuple[int, int, int, int]]) -> bool:
    for i in range(len(rects)):
        border_i = _border_cells(rects[i])
        for j in range(i + 1, len(rects)):
            if border_i & _border_cells(rects[j]):
                return True
    return False


def main(grid: np.ndarray) -> np.ndarray:
    """Remove the bottom exclusion strip and frame every other color with its bounding rectangle.

    Input structure:
    - Black is the background.
    - Several foreground colors appear as scattered single pixels or tiny sparse sets, with at least three pixels per color.
    - The bottom row is a full-width solid strip of one foreground color, and that same color also appears scattered above.

    Rule:
    - Read the bottom strip color as the exclusion color and drop that entire row from the output.
    - Leave the scattered exclusion-color pixels unchanged, but ignore them when drawing frames.
    - For every other foreground color, find the axis-aligned bounding box of all pixels of that color and draw a one-pixel-thick rectangular border in that color.
    - When multiple borders overlap, the color with fewer source pixels wins; original input pixels are never overwritten.

    Key insight:
    - The bottom strip is not an object to transform; it is a control signal telling you which color should be skipped before the remaining colors are summarized by their bounding boxes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2

    excluded = int(g[-1, 0])
    assert excluded != BG
    assert np.all(g[-1] == excluded)

    body = g[:-1].copy()
    overlay = np.full_like(body, BG)

    color_infos = []
    for color in _first_seen_colors(body, excluded):
        coords = np.argwhere(body == color)
        if coords.size == 0:
            continue
        color_infos.append(
            {
                "color": color,
                "count": int(coords.shape[0]),
                "bbox": _bbox(coords),
                "anchor": (int(coords[0, 0]), int(coords[0, 1])),
            }
        )

    color_infos.sort(key=lambda item: (-item["count"], item["anchor"][0], item["anchor"][1]))

    for info in color_infos:
        for r, c in _border_cells(info["bbox"]):
            overlay[r, c] = int(info["color"])

    out = body.copy()
    bg_mask = body == BG
    out[bg_mask] = overlay[bg_mask]
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_processed = rng.randint(2, 4)
    body_h = rng.randint(9 + 2 * n_processed, min(24, 14 + 3 * n_processed))
    width = rng.randint(10 + 2 * n_processed, min(26, 16 + 3 * n_processed))
    height = body_h + 1

    colors = rng.sample(PALETTE, n_processed + 1)
    excluded = colors[0]
    processed_colors = colors[1:]
    processed_counts = rng.sample(list(range(3, 8)), n_processed)
    excluded_count = rng.randint(3, 6)
    force_overlap = rng.choice([False, True])

    rects: list[tuple[int, int, int, int]] = []
    if force_overlap:
        pair = _sample_overlap_pair(rng, body_h, width)
        if pair is None:
            return None
        rects.extend(pair)
    else:
        first = _sample_separated_rect(rng, body_h, width, [])
        second = _sample_separated_rect(rng, body_h, width, [first] if first else [])
        if first is None or second is None:
            return None
        rects.extend([first, second])

    while len(rects) < n_processed:
        rect = _sample_separated_rect(rng, body_h, width, rects)
        if rect is None:
            return None
        rects.append(rect)

    exclude_rect = _sample_separated_rect(rng, body_h, width, rects)
    if exclude_rect is None:
        return None

    if _has_any_border_overlap(rects) != force_overlap:
        return None

    grid = np.zeros((height, width), dtype=int)
    used_cells: set[tuple[int, int]] = set()

    avoid_by_index: list[set[tuple[int, int]]] = [set() for _ in range(n_processed)]
    if force_overlap:
        intersections = _border_cells(rects[0]) & _border_cells(rects[1])
        avoid_by_index[0] = set(intersections)
        avoid_by_index[1] = set(intersections)

    for idx, (color, rect, count) in enumerate(zip(processed_colors, rects, processed_counts)):
        cells = _scatter_points(
            rng,
            rect,
            count,
            blocked=used_cells,
            avoid=avoid_by_index[idx],
        )
        if cells is None:
            return None
        for r, c in cells:
            grid[r, c] = color
        used_cells.update(cells)

    excluded_cells = _scatter_points(rng, exclude_rect, excluded_count, blocked=used_cells)
    if excluded_cells is None:
        return None
    for r, c in excluded_cells:
        grid[r, c] = excluded

    grid[-1, :] = excluded

    if not np.any(grid[:-1] == excluded):
        return None
    if max(grid.shape) > 30:
        return None

    out = main(grid)
    if out.shape != (height - 1, width):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
