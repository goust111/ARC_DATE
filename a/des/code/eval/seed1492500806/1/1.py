import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
CYAN = 8
MAROON = 9

PIECE_COLORS = [BLUE, RED, GREEN, YELLOW, MAGENTA, ORANGE, CYAN]
GENERATE_MAX_RETRIES = 400

# Quadrant order matches the required clockwise turns:
# top-left=0, top-right=1, bottom-right=2, bottom-left=3.
TL, TR, BR, BL = 0, 1, 2, 3


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(vals[np.argmax(counts)])


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    assert np.any(rows) and np.any(cols)
    return mask[rows][:, cols].copy()


def _rotate_cw(arr: np.ndarray, turns: int) -> np.ndarray:
    turns %= 4
    return np.rot90(np.asarray(arr), -turns).copy()


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(np.min(coords[:, 0])),
        int(np.min(coords[:, 1])),
        int(np.max(coords[:, 0])),
        int(np.max(coords[:, 1])),
    )


def _coords_to_mask(coords: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return (r0, c0, r1, c1), mask


def _mask_signature(mask: np.ndarray) -> tuple:
    mask = _trim_mask(mask)
    sigs = []
    for k in range(4):
        rot = _trim_mask(np.rot90(mask, k))
        sigs.append((int(rot.shape[0]), int(rot.shape[1]), tuple(int(x) for x in rot.astype(np.uint8).ravel())))
    return min(sigs)


def _rotation_count(mask: np.ndarray) -> int:
    mask = _trim_mask(mask)
    sigs = set()
    for k in range(4):
        rot = _trim_mask(np.rot90(mask, k))
        sigs.add((int(rot.shape[0]), int(rot.shape[1]), rot.astype(np.uint8).tobytes()))
    return len(sigs)


def _find_separator_row(grid: np.ndarray, bg: int) -> tuple[int, int]:
    h, w = grid.shape
    candidates = []
    for r in range(h):
        color = int(grid[r, 0])
        if color != bg and int(grid[r, w - 1]) == color:
            count = int(np.sum(grid[r] == color))
            candidates.append((count, r, color))
    assert candidates, "missing horizontal separator"
    count, row, color = max(candidates)
    assert count >= w // 3, "separator row too short"
    return int(row), int(color)


def _find_separator_col(grid: np.ndarray, bg: int) -> tuple[int, int]:
    h, w = grid.shape
    candidates = []
    for c in range(w):
        color = int(grid[0, c])
        if color != bg and int(grid[h - 1, c]) == color:
            count = int(np.sum(grid[:, c] == color))
            candidates.append((count, c, color))
    assert candidates, "missing vertical separator"
    count, col, color = max(candidates)
    assert count >= h // 3, "separator col too short"
    return int(col), int(color)


def _quadrant_from_bbox(bbox: tuple[int, int, int, int], sep_r: int, sep_c: int) -> int:
    r0, c0, r1, c1 = bbox
    cr = (r0 + r1) / 2.0
    cc = (c0 + c1) / 2.0
    if cr < sep_r and cc < sep_c:
        return TL
    if cr < sep_r and cc > sep_c:
        return TR
    if cr > sep_r and cc > sep_c:
        return BR
    return BL


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match outside pieces to frame holes by shape, then rotate by quadrant code while filling.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)

    sep_r, sep_color_r = _find_separator_row(g, bg)
    sep_c, sep_color_c = _find_separator_col(g, bg)
    assert sep_color_r == sep_color_c, "separator row/col must share a color"
    sep_color = sep_color_r

    comps = []
    for coords in _components((g != bg) & (g != sep_color)):
        bbox, mask = _coords_to_mask(coords)
        colors = {int(g[r, c]) for r, c in coords}
        comps.append(
            {
                "bbox": bbox,
                "mask": mask,
                "coords": coords,
                "colors": colors,
                "area": int(len(coords)),
            }
        )
    assert comps, "expected frame and pieces"

    frame = max(comps, key=lambda item: item["area"])
    pieces = [item for item in comps if item is not frame]
    assert pieces, "expected at least one outside piece"

    fr0, fc0, fr1, fc1 = frame["bbox"]
    crop = g[fr0 : fr1 + 1, fc0 : fc1 + 1].copy()

    hole_infos = []
    for coords in _components(crop == bg):
        if (
            np.any(coords[:, 0] == 0)
            or np.any(coords[:, 0] == crop.shape[0] - 1)
            or np.any(coords[:, 1] == 0)
            or np.any(coords[:, 1] == crop.shape[1] - 1)
        ):
            continue
        bbox, mask = _coords_to_mask(coords)
        hole_infos.append({"bbox": bbox, "mask": mask, "sig": _mask_signature(mask)})
    assert hole_infos, "frame must contain holes"
    assert len(hole_infos) == len(pieces), "hole count must match piece count"

    hole_by_sig = {}
    for hole in hole_infos:
        assert hole["sig"] not in hole_by_sig, "hole shapes must be unique up to rotation"
        hole_by_sig[hole["sig"]] = hole

    used = set()
    for piece in pieces:
        assert len(piece["colors"]) == 1, "each piece must be monochrome"
        sig = _mask_signature(piece["mask"])
        assert sig in hole_by_sig, "piece shape must match a hole"
        assert sig not in used, "each hole should be matched once"
        hole = hole_by_sig[sig]
        used.add(sig)

        turns = _quadrant_from_bbox(piece["bbox"], sep_r, sep_c)
        rotated = _rotate_cw(piece["mask"], turns)
        assert rotated.shape == hole["mask"].shape and np.array_equal(rotated, hole["mask"])

        color = next(iter(piece["colors"]))
        hr0, hc0, _, _ = hole["bbox"]
        patch = crop[hr0 : hr0 + rotated.shape[0], hc0 : hc0 + rotated.shape[1]]
        patch[rotated] = color

    assert len(used) == len(hole_infos)
    return crop


def _random_connected_mask() -> np.ndarray | None:
    box_h = random.randint(2, 4)
    box_w = random.randint(2, 4)
    max_area = min(box_h * box_w - 1, 6)
    if max_area < 3:
        return None

    target_area = random.randint(3, max_area)
    start = (random.randrange(box_h), random.randrange(box_w))
    cells = {start}
    while len(cells) < target_area:
        frontier = set()
        for r, c in cells:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if not frontier:
            return None
        cells.add(random.choice(tuple(frontier)))

    mask = np.zeros((box_h, box_w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    mask = _trim_mask(mask)
    if mask.shape[0] < 2 or mask.shape[1] < 2:
        return None
    if _rotation_count(mask) < 4:
        return None
    return mask


def _sample_unique_shapes(count: int) -> list[np.ndarray] | None:
    shapes = []
    seen = set()
    for _ in range(300):
        if len(shapes) == count:
            return shapes
        mask = _random_connected_mask()
        if mask is None:
            continue
        sig = _mask_signature(mask)
        if sig in seen:
            continue
        seen.add(sig)
        shapes.append(mask)
    return None


def _place_masks(area_h: int, area_w: int, masks: list[np.ndarray]) -> list[tuple[int, int]] | None:
    blocked = np.zeros((area_h, area_w), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(masks)
    order = sorted(range(len(masks)), key=lambda i: int(np.sum(masks[i])), reverse=True)

    for idx in order:
        mask = masks[idx]
        mh, mw = mask.shape
        if mh > area_h or mw > area_w:
            return None
        positions = [(r, c) for r in range(area_h - mh + 1) for c in range(area_w - mw + 1)]
        random.shuffle(positions)
        placed = False
        coords = np.argwhere(mask)
        for r0, c0 in positions:
            if np.any(blocked[r0 : r0 + mh, c0 : c0 + mw] & mask):
                continue
            placements[idx] = (r0, c0)
            for rr, cc in coords:
                r1 = max(0, r0 + rr - 1)
                r2 = min(area_h, r0 + rr + 2)
                c1 = max(0, c0 + cc - 1)
                c2 = min(area_w, c0 + cc + 2)
                blocked[r1:r2, c1:c2] = True
            placed = True
            break
        if not placed:
            return None

    return [tuple(x) for x in placements]  # type: ignore[arg-type]


def _quadrant_sequence(count: int) -> list[int]:
    first_round = [TL, TR, BR, BL]
    random.shuffle(first_round)
    out = first_round[:]
    while len(out) < count:
        extra = [TL, TR, BR, BL]
        random.shuffle(extra)
        out.extend(extra)
    return out[:count]


def _paste_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    h, w = mask.shape
    patch = grid[r0 : r0 + h, c0 : c0 + w]
    patch[mask] = color


def _gen_attempt() -> np.ndarray | None:
    piece_count = random.randint(4, 6)
    base_shapes = _sample_unique_shapes(piece_count)
    if base_shapes is None:
        return None

    frame_h = random.choice([11, 13, 15, 17])
    frame_w = random.choice([11, 13, 15, 17])
    inner_h = frame_h - 2
    inner_w = frame_w - 2

    hole_masks = [_rotate_cw(shape, random.randrange(4)) for shape in base_shapes]
    hole_positions = _place_masks(inner_h, inner_w, hole_masks)
    if hole_positions is None:
        return None

    quadrants = _quadrant_sequence(piece_count)
    quad_counts = [quadrants.count(q) for q in (TL, TR, BR, BL)]

    top_margin = random.randint(8, 10) + 2 * max(0, max(quad_counts[TL], quad_counts[TR]) - 1)
    bottom_margin = random.randint(8, 10) + 2 * max(0, max(quad_counts[BL], quad_counts[BR]) - 1)
    left_margin = random.randint(8, 10) + 2 * max(0, max(quad_counts[TL], quad_counts[BL]) - 1)
    right_margin = random.randint(8, 10) + 2 * max(0, max(quad_counts[TR], quad_counts[BR]) - 1)

    height = top_margin + frame_h + bottom_margin
    width = left_margin + frame_w + right_margin
    fr0, fc0 = top_margin, left_margin
    fr1, fc1 = fr0 + frame_h - 1, fc0 + frame_w - 1
    sep_r = fr0 + frame_h // 2
    sep_c = fc0 + frame_w // 2

    grid = np.full((height, width), BLACK, dtype=int)
    expected = np.full((frame_h, frame_w), MAROON, dtype=int)

    hole_infos = []
    for mask, (rr, cc) in zip(hole_masks, hole_positions):
        pr, pc = rr + 1, cc + 1
        _paste_mask(expected, mask, pr, pc, BLACK)
        hole_infos.append({"mask": mask, "offset": (pr, pc)})

    grid[fr0 : fr1 + 1, fc0 : fc1 + 1] = expected

    grid[sep_r, :fc0] = GREY
    grid[sep_r, fc1 + 1 :] = GREY
    grid[:fr0, sep_c] = GREY
    grid[fr1 + 1 :, sep_c] = GREY

    colors = random.sample(PIECE_COLORS, piece_count)
    piece_masks = [_rotate_cw(hole_masks[i], (-quadrants[i]) % 4) for i in range(piece_count)]

    regions = {
        TL: (1, 1, fr0 - 2, fc0 - 2),
        TR: (1, fc1 + 2, fr0 - 2, width - 2),
        BR: (fr1 + 2, fc1 + 2, height - 2, width - 2),
        BL: (fr1 + 2, 1, height - 2, fc0 - 2),
    }

    for quadrant in (TL, TR, BR, BL):
        idxs = [i for i, q in enumerate(quadrants) if q == quadrant]
        if not idxs:
            continue
        r0, c0, r1, c1 = regions[quadrant]
        area_h = r1 - r0 + 1
        area_w = c1 - c0 + 1
        if area_h <= 0 or area_w <= 0:
            return None
        local_masks = [piece_masks[i] for i in idxs]
        local_positions = _place_masks(area_h, area_w, local_masks)
        if local_positions is None:
            return None
        for idx, (rr, cc) in zip(idxs, local_positions):
            _paste_mask(grid, piece_masks[idx], r0 + rr, c0 + cc, colors[idx])

    for color, info in zip(colors, hole_infos):
        hr, hc = info["offset"]
        mask = info["mask"]
        patch = expected[hr : hr + mask.shape[0], hc : hc + mask.shape[1]]
        patch[mask] = color

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid instance")
