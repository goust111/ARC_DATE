from __future__ import annotations

import random
from math import isqrt

import numpy as np


def _components_of_mask(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components as coordinate arrays."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(cells, dtype=int))
    return comps


def _normalize_coords(coords: np.ndarray) -> np.ndarray:
    top_left = coords.min(axis=0)
    return coords - top_left


def _piece_dims(piece: np.ndarray) -> tuple[int, int]:
    return int(piece[:, 0].max()) + 1, int(piece[:, 1].max()) + 1


def _can_tile_rectangle(pieces: list[np.ndarray], height: int, width: int) -> bool:
    """Exact-cover search over translations only."""
    total_area = sum(int(piece.shape[0]) for piece in pieces)
    if total_area != height * width:
        return False

    max_h = max(_piece_dims(piece)[0] for piece in pieces)
    max_w = max(_piece_dims(piece)[1] for piece in pieces)
    if height < max_h or width < max_w:
        return False

    occupied = np.zeros((height, width), dtype=bool)
    used = [False] * len(pieces)
    order = sorted(
        range(len(pieces)),
        key=lambda i: (
            -int(pieces[i].shape[0]),
            -_piece_dims(pieces[i])[0],
            -_piece_dims(pieces[i])[1],
        ),
    )

    def search(used_count: int) -> bool:
        if used_count == len(pieces):
            return True

        r, c = map(int, np.argwhere(~occupied)[0])
        for idx in order:
            if used[idx]:
                continue
            piece = pieces[idx]
            tried: set[tuple[int, int]] = set()
            for pr, pc in piece:
                shift_r = int(r - pr)
                shift_c = int(c - pc)
                if (shift_r, shift_c) in tried:
                    continue
                tried.add((shift_r, shift_c))

                rows = piece[:, 0] + shift_r
                cols = piece[:, 1] + shift_c
                if rows.min() < 0 or cols.min() < 0 or rows.max() >= height or cols.max() >= width:
                    continue
                if bool(occupied[rows, cols].any()):
                    continue

                occupied[rows, cols] = True
                used[idx] = True
                if search(used_count + 1):
                    return True
                used[idx] = False
                occupied[rows, cols] = False
        return False

    return search(0)


def _find_rectangle_dims(pieces: list[np.ndarray]) -> tuple[int, int]:
    area = sum(int(piece.shape[0]) for piece in pieces)
    candidates: list[tuple[int, int]] = []
    for h in range(1, isqrt(area) + 1):
        if area % h != 0:
            continue
        w = area // h
        candidates.append((int(h), int(w)))
        if h != w:
            candidates.append((int(w), int(h)))

    valid = [(h, w) for h, w in candidates if _can_tile_rectangle(pieces, h, w)]
    assert len(valid) == 1, "same-color pieces must assemble into one unique rectangle"
    return valid[0]


def _extract_rectangles(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    items = []
    for color in np.unique(g):
        color = int(color)
        if color == 0:
            continue
        comps = _components_of_mask(g == color)
        assert 2 <= len(comps) <= 4, "each color must appear in 2 to 4 fragments"
        pieces = [_normalize_coords(comp) for comp in comps]
        height, width = _find_rectangle_dims(pieces)
        items.append(
            {
                "color": color,
                "height": int(height),
                "width": int(width),
                "area": int(height * width),
            }
        )

    assert 2 <= len(items) <= 4, "expected 2 to 4 non-background colors"
    areas = [item["area"] for item in items]
    assert len(areas) == len(set(areas)), "rectangle areas must be distinct"
    items.sort(key=lambda item: int(item["area"]), reverse=True)
    return items


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group same-color fragments into solid rectangles, then nest the rectangles
    from largest area to smallest with a 1-pixel ring between layers.
    """
    items = _extract_rectangles(grid)
    outer = items[0]
    out = np.zeros((int(outer["height"]), int(outer["width"])), dtype=int)

    for depth, item in enumerate(items):
        h = int(item["height"])
        w = int(item["width"])
        if depth:
            prev = items[depth - 1]
            assert int(prev["height"]) == h + 2 and int(prev["width"]) == w + 2, (
                "nested rectangles must differ by exactly 2 in both dimensions"
            )
        out[depth : depth + h, depth : depth + w] = int(item["color"])

    return out


# --------------- generator ---------------

_RNG = random.Random()

COLOR_POOL = (1, 2, 3, 4, 5, 6, 7, 8, 9)
COLOR_COUNT_RANGE = (2, 4)
PIECE_COUNT_RANGE = (2, 4)
INNER_SIDE_MIN = 2
INNER_SIDE_MAX = 6
MAX_TOTAL_AREA = 240
CANVAS_SIDE_RANGE = (22, 30)
PARTITION_RETRIES = 120
GENERATION_RETRIES = 400


def _randint(lo: int, hi: int) -> int:
    return _RNG.randint(int(lo), int(hi))


def _random_composition(total: int, parts: int, min_part: int) -> list[int]:
    slack = total - parts * min_part
    if slack < 0:
        raise ValueError("impossible composition")
    if parts == 1:
        return [total]
    cuts = sorted(_RNG.sample(range(slack + parts - 1), parts - 1))
    values = []
    prev = -1
    for cut in cuts + [slack + parts - 1]:
        values.append(cut - prev)
        prev = cut
    return [value + min_part - 1 for value in values]


def _owner_connected(owner: np.ndarray, value: int) -> bool:
    coords = np.argwhere(owner == value)
    if coords.size == 0:
        return False
    cell_set = {tuple(map(int, coord)) for coord in coords}
    stack = [next(iter(cell_set))]
    seen = {stack[0]}
    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (r + dr, c + dc)
            if nxt in cell_set and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(cell_set)


def _partition_rectangle(height: int, width: int, parts: int) -> list[np.ndarray] | None:
    total = height * width
    if total < 2 * parts:
        return None

    cells = [(r, c) for r in range(height) for c in range(width)]
    for _ in range(PARTITION_RETRIES):
        targets = _random_composition(total, parts, 2)
        seeds = _RNG.sample(cells, parts)
        owner = np.full((height, width), -1, dtype=int)
        frontiers = [set() for _ in range(parts)]
        sizes = [1] * parts

        for idx, (r, c) in enumerate(seeds):
            owner[r, c] = idx

        for idx, (r, c) in enumerate(seeds):
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < height and 0 <= nc < width and owner[nr, nc] == -1:
                    frontiers[idx].add((nr, nc))

        success = True
        while any(sizes[idx] < targets[idx] for idx in range(parts)):
            choices = [idx for idx in range(parts) if sizes[idx] < targets[idx] and frontiers[idx]]
            if not choices:
                success = False
                break

            weights = [targets[idx] - sizes[idx] for idx in choices]
            region = _RNG.choices(choices, weights=weights, k=1)[0]
            cell = _RNG.choice(tuple(frontiers[region]))
            rr, cc = cell
            owner[rr, cc] = region
            sizes[region] += 1

            for frontier in frontiers:
                frontier.discard(cell)
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < height and 0 <= nc < width and owner[nr, nc] == -1:
                    frontiers[region].add((nr, nc))

        if not success or (owner == -1).any():
            continue
        if not all(_owner_connected(owner, idx) for idx in range(parts)):
            continue

        pieces = []
        irregular = False
        for idx in range(parts):
            coords = np.argwhere(owner == idx)
            rel = _normalize_coords(coords)
            ph, pw = _piece_dims(rel)
            if int(rel.shape[0]) < ph * pw:
                irregular = True
            pieces.append(rel)
        if irregular:
            return pieces
    return None


def _piece_to_patch(piece: np.ndarray, color: int) -> np.ndarray:
    h, w = _piece_dims(piece)
    patch = np.zeros((h, w), dtype=int)
    patch[piece[:, 0], piece[:, 1]] = int(color)
    return patch


def _dilate8(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            src_r0 = max(0, -dr)
            src_r1 = min(h, h - dr)
            src_c0 = max(0, -dc)
            src_c1 = min(w, w - dc)
            dst_r0 = max(0, dr)
            dst_r1 = dst_r0 + (src_r1 - src_r0)
            dst_c0 = max(0, dc)
            dst_c1 = dst_c0 + (src_c1 - src_c0)
            out[dst_r0:dst_r1, dst_c0:dst_c1] |= mask[src_r0:src_r1, src_c0:src_c1]
    return out


def _place_patches(patches: list[np.ndarray], height: int, width: int) -> np.ndarray | None:
    grid = np.zeros((height, width), dtype=int)
    occupied = np.zeros((height, width), dtype=bool)
    blocked = np.zeros((height, width), dtype=bool)

    order = list(range(len(patches)))
    _RNG.shuffle(order)
    order.sort(
        key=lambda idx: (
            -int(np.count_nonzero(patches[idx])),
            -int(patches[idx].shape[0] * patches[idx].shape[1]),
        )
    )

    for idx in order:
        patch = patches[idx]
        mask = patch != 0
        ph, pw = patch.shape
        candidates = []
        for r0 in range(height - ph + 1):
            for c0 in range(width - pw + 1):
                if bool((blocked[r0 : r0 + ph, c0 : c0 + pw] & mask).any()):
                    continue
                candidates.append((r0, c0))
        if not candidates:
            return None

        r0, c0 = _RNG.choice(candidates)
        view = grid[r0 : r0 + ph, c0 : c0 + pw]
        view[mask] = patch[mask]
        occupied[r0 : r0 + ph, c0 : c0 + pw] |= mask
        blocked = _dilate8(occupied)

    return grid


def _build_nested_output(items: list[dict]) -> np.ndarray:
    out = np.zeros((int(items[0]["height"]), int(items[0]["width"])), dtype=int)
    for depth, item in enumerate(items):
        h = int(item["height"])
        w = int(item["width"])
        out[depth : depth + h, depth : depth + w] = int(item["color"])
    return out


def _sample_rectangles() -> list[dict] | None:
    color_count = _randint(*COLOR_COUNT_RANGE)
    colors = _RNG.sample(COLOR_POOL, color_count)
    piece_counts = [_randint(*PIECE_COUNT_RANGE) for _ in range(color_count)]

    max_inner_h = INNER_SIDE_MAX
    max_inner_w = INNER_SIDE_MAX
    inner_h = _randint(INNER_SIDE_MIN, max_inner_h)
    inner_w = _randint(INNER_SIDE_MIN, max_inner_w)

    items = []
    total_area = 0
    for idx in range(color_count):
        depth_from_outer = idx
        depth_from_inner = color_count - 1 - idx
        height = inner_h + 2 * depth_from_inner
        width = inner_w + 2 * depth_from_inner
        area = height * width
        total_area += area
        items.append(
            {
                "color": int(colors[idx]),
                "height": int(height),
                "width": int(width),
                "area": int(area),
                "piece_count": int(piece_counts[idx]),
            }
        )

    if total_area > MAX_TOTAL_AREA:
        return None

    items.sort(key=lambda item: int(item["area"]), reverse=True)
    for item in items:
        if int(item["area"]) < 2 * int(item["piece_count"]):
            return None
    return items


def _gen_attempt() -> np.ndarray | None:
    items = _sample_rectangles()
    if items is None:
        return None

    patches = []
    for item in items:
        pieces = _partition_rectangle(int(item["height"]), int(item["width"]), int(item["piece_count"]))
        if pieces is None:
            return None
        for piece in pieces:
            patches.append(_piece_to_patch(piece, int(item["color"])))

    min_side = max(
        max(patch.shape[0] for patch in patches),
        max(patch.shape[1] for patch in patches),
        int(items[0]["height"]) + 4,
        int(items[0]["width"]) + 4,
        CANVAS_SIDE_RANGE[0],
    )
    if min_side > CANVAS_SIDE_RANGE[1]:
        return None

    height = _randint(min_side, CANVAS_SIDE_RANGE[1])
    width = _randint(min_side, CANVAS_SIDE_RANGE[1])
    grid = _place_patches(patches, height, width)
    if grid is None:
        return None

    expected = _build_nested_output(items)
    try:
        if not np.array_equal(main(grid), expected):
            return None
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
