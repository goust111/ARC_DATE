from __future__ import annotations

import random

import numpy as np


BG = 0
FRAME = 1
RED = 2
GREEN = 3
DIVIDER = 5

GEN_RETRIES = 600


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
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
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _extract_single_sprite(panel: np.ndarray, color: int) -> np.ndarray:
    comps = _components(panel == color)
    assert len(comps) == 1, f"expected one {color} component"
    box = _bbox(comps[0])
    return _crop_bbox(panel, box)


def _find_divider_col(grid: np.ndarray) -> int:
    h, w = grid.shape
    cols = [c for c in range(w) if np.all(grid[:, c] == DIVIDER)]
    assert len(cols) == 1, "expected exactly one full-height grey divider"
    c = cols[0]
    assert 0 < c < w - 1 and h >= 1
    return c


def _find_frames(grid: np.ndarray, start_col: int) -> list[tuple[int, int, int, int]]:
    right = grid[:, start_col:]
    assert set(np.unique(right)).issubset({BG, FRAME}), "right panel should only contain blue frames on black"
    frames: list[tuple[int, int, int, int]] = []
    for coords in _components(right == FRAME):
        r0, c0, r1, c1 = _bbox(coords)
        assert r1 - r0 >= 2 and c1 - c0 >= 2, "frame must have non-empty interior"
        local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        local[coords[:, 0] - r0, coords[:, 1] - c0] = True
        expected = np.zeros_like(local)
        expected[0, :] = True
        expected[-1, :] = True
        expected[:, 0] = True
        expected[:, -1] = True
        assert np.array_equal(local, expected), "blue component must be a hollow rectangle"
        assert np.all(right[r0 + 1 : r1, c0 + 1 : c1] == BG), "frame interior must be black"
        frames.append((r0, start_col + c0, r1, start_col + c1))
    assert frames, "expected at least one frame"
    frames.sort()
    return frames


def _stamp_clipped(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    rr1 = min(dst.shape[0], r0 + h)
    cc1 = min(dst.shape[1], c0 + w)
    if rr1 <= r0 or cc1 <= c0:
        return
    clipped = patch[: rr1 - r0, : cc1 - c0]
    mask = clipped != BG
    dst_view = dst[r0:rr1, c0:cc1]
    dst_view[mask] = clipped[mask]


def _visible_stamp(patch: np.ndarray, inner_h: int, inner_w: int) -> np.ndarray:
    out = np.zeros((inner_h, inner_w), dtype=int)
    clipped = patch[:inner_h, :inner_w]
    out[: clipped.shape[0], : clipped.shape[1]] = clipped
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp the red template or the rotated green template into each hollow blue frame based on interior-area parity.

    Input structure:
    - A black background contains one full-height grey divider column.
    - Left of the divider there are exactly two isolated monochrome sprites: an upper red sprite and a lower green sprite.
    - Right of the divider there are several separated one-cell-thick blue hollow rectangles whose interiors are black.

    Rule:
    - For each blue frame, compute the area of its interior rectangle.
    - If the interior area is even, stamp the upper red sprite in its original orientation into the frame interior, aligned by top-left corner.
    - If the interior area is odd, rotate the lower green sprite 90 degrees clockwise, then stamp that rotated sprite into the frame interior, also aligned by top-left corner.
    - When the chosen sprite extends past the frame interior, clip the overflow and draw only the in-bounds sprite pixels.
    - Keep the divider, the original left-side sprites, the blue frame borders, and all untouched background pixels unchanged.

    Key insight:
    - The frame interior area parity does not directly choose a fill color; it chooses both which template to use and whether rotation is applied before the clipped stamp.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30

    divider_col = _find_divider_col(g)
    left = g[:, :divider_col]
    red_sprite = _extract_single_sprite(left, RED)
    green_sprite = _extract_single_sprite(left, GREEN)
    green_rot = np.rot90(green_sprite, -1)

    out = g.copy()
    frames = _find_frames(g, divider_col + 1)
    for r0, c0, r1, c1 in frames:
        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        sprite = red_sprite if (inner_h * inner_w) % 2 == 0 else green_rot
        _stamp_clipped(out, sprite, r0 + 1, c0 + 1)
    return out


def _shape_is_connected(mask: np.ndarray) -> bool:
    return len(_components(mask)) == 1


def _shape_is_rectangle(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    box = _bbox(coords)
    tight = mask[box[0] : box[2] + 1, box[1] : box[3] + 1]
    return bool(np.all(tight))


def _random_shape(min_cells: int = 4, max_cells: int = 8) -> np.ndarray:
    for _ in range(300):
        box_h = random.randint(3, 5)
        box_w = random.randint(3, 5)
        target = random.randint(min_cells, max_cells)
        start = (random.randrange(box_h), random.randrange(box_w))
        cells = {start}
        while len(cells) < target:
            frontier = []
            for r, c in list(cells):
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        mask = np.zeros((box_h, box_w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        coords = np.argwhere(mask)
        box = _bbox(coords)
        tight = mask[box[0] : box[2] + 1, box[1] : box[3] + 1]
        if not (min_cells <= int(tight.sum()) <= max_cells):
            continue
        if not _shape_is_connected(tight):
            continue
        if _shape_is_rectangle(tight):
            continue
        return tight
    raise RuntimeError("failed to sample connected non-rectangular shape")


def _shape_to_patch(mask: np.ndarray, color: int) -> np.ndarray:
    return np.where(mask, color, BG).astype(int)


def _all_dim_pairs() -> list[tuple[int, int]]:
    return [(h, w) for h in range(3, 8) for w in range(3, 8)]


def _parity_pairs(parity: int) -> list[tuple[int, int]]:
    return [(h, w) for h, w in _all_dim_pairs() if (h * w) % 2 == parity]


def _random_split(extra: int, slots: int) -> list[int]:
    if slots == 1:
        return [extra]
    cuts = sorted(random.randint(0, extra) for _ in range(slots - 1))
    prev = 0
    out = []
    for cut in cuts + [extra]:
        out.append(cut - prev)
        prev = cut
    return out


def _pick_distinct(cands: list[tuple[int, int]], used: set[tuple[int, int]] | None = None) -> tuple[int, int]:
    used = used or set()
    pool = [pair for pair in cands if pair not in used]
    if not pool:
        pool = cands
    return random.choice(pool)


def _build_frame_dimensions(red_shape: tuple[int, int], green_rot_shape: tuple[int, int]) -> list[tuple[int, int]]:
    rh, rw = red_shape
    gh, gw = green_rot_shape

    even_pairs = _parity_pairs(0)
    odd_pairs = _parity_pairs(1)

    even_clip = [(h, w) for h, w in even_pairs if h < rh or w < rw]
    even_full = [(h, w) for h, w in even_pairs if h >= rh and w >= rw]
    odd_full = [(h, w) for h, w in odd_pairs if h >= gh and w >= gw]
    odd_clip = [(h, w) for h, w in odd_pairs if h < gh or w < gw]

    assert even_clip and even_full and odd_full

    used: set[tuple[int, int]] = set()
    dims = [
        _pick_distinct(even_clip, used),
    ]
    used.add(dims[-1])
    dims.append(_pick_distinct(even_full, used))
    used.add(dims[-1])
    dims.append(_pick_distinct(odd_full, used))
    used.add(dims[-1])
    dims.append(_pick_distinct(odd_clip if odd_clip else odd_full, used))
    used.add(dims[-1])

    if random.random() < 0.6:
        any_pairs = even_pairs + odd_pairs
        dims.append(_pick_distinct(any_pairs, used))

    random.shuffle(dims)
    return dims


def _row_layout(k: int) -> list[int]:
    if k == 4:
        return random.choice([[2, 2], [1, 2, 1], [2, 1, 1], [1, 1, 2]])
    return random.choice([[2, 2, 1], [1, 2, 2], [2, 1, 2]])


def _pack_frames(frame_inner_dims: list[tuple[int, int]], left_w: int) -> tuple[int, int, list[tuple[int, int, int, int]]]:
    outer = [(h + 2, w + 2) for h, w in frame_inner_dims]
    order = list(range(len(outer)))
    random.shuffle(order)
    outer = [outer[i] for i in order]

    row_counts = _row_layout(len(outer))
    rows: list[list[tuple[int, int]]] = []
    idx = 0
    for count in row_counts:
        rows.append(outer[idx : idx + count])
        idx += count

    row_heights = [max(h for h, _ in row) for row in rows]
    row_widths = [sum(w for _, w in row) + (len(row) - 1) for row in rows]
    min_right_w = max(row_widths)
    max_right_w = 30 - left_w - 1
    if min_right_w > max_right_w:
        raise RuntimeError("frame layout too wide")
    right_w = random.randint(min_right_w, min(max_right_w, min_right_w + 4))

    min_h = max(sum(row_heights) + (len(rows) - 1), 10)
    if min_h > 30:
        raise RuntimeError("frame layout too tall")
    total_h = random.randint(min_h, min(30, min_h + 3))

    row_slack = _random_split(total_h - (sum(row_heights) + (len(rows) - 1)), len(rows) + 1)
    boxes: list[tuple[int, int, int, int]] = []
    r = row_slack[0]
    for row_i, row in enumerate(rows):
        width_used = sum(w for _, w in row) + (len(row) - 1)
        gap_slack = _random_split(right_w - width_used, len(row) + 1)
        c = gap_slack[0]
        row_h = row_heights[row_i]
        for j, (fh, fw) in enumerate(row):
            top_offset = 0 if row_h == fh else random.randint(0, row_h - fh)
            boxes.append((r + top_offset, c, r + top_offset + fh - 1, c + fw - 1))
            c += fw
            if j < len(row) - 1:
                c += 1 + gap_slack[j + 1]
        r += row_h
        if row_i < len(rows) - 1:
            r += 1 + row_slack[row_i + 1]
    return total_h, right_w, boxes


def _render_input(
    red_sprite: np.ndarray,
    green_sprite: np.ndarray,
    frame_inner_dims: list[tuple[int, int]],
) -> np.ndarray:
    left_w = max(red_sprite.shape[1], green_sprite.shape[1]) + random.randint(2, 4)
    h, right_w, frame_boxes = _pack_frames(frame_inner_dims, left_w)
    grid = np.zeros((h, left_w + 1 + right_w), dtype=int)
    grid[:, left_w] = DIVIDER

    red_r_max = max(1, h // 3 - red_sprite.shape[0])
    red_r0 = random.randint(0, red_r_max)
    green_r_min = max(red_r0 + red_sprite.shape[0] + 2, h // 2)
    green_r_max = h - green_sprite.shape[0]
    if green_r_min > green_r_max:
        raise RuntimeError("left templates do not fit vertically")
    green_r0 = random.randint(green_r_min, green_r_max)
    red_c0 = random.randint(0, left_w - red_sprite.shape[1] - 1)
    green_c0 = random.randint(0, left_w - green_sprite.shape[1] - 1)
    _stamp_clipped(grid, red_sprite, red_r0, red_c0)
    _stamp_clipped(grid, green_sprite, green_r0, green_c0)

    offset = left_w + 1
    for (r0, c0, r1, c1) in frame_boxes:
        grid[r0, offset + c0 : offset + c1 + 1] = FRAME
        grid[r1, offset + c0 : offset + c1 + 1] = FRAME
        grid[r0 : r1 + 1, offset + c0] = FRAME
        grid[r0 : r1 + 1, offset + c1] = FRAME
    return grid


def _semantic_check(inp: np.ndarray) -> None:
    out = main(inp)
    divider_col = _find_divider_col(inp)
    red_sprite = _extract_single_sprite(inp[:, :divider_col], RED)
    green_sprite = _extract_single_sprite(inp[:, :divider_col], GREEN)
    green_rot = np.rot90(green_sprite, -1)
    assert not np.array_equal(green_sprite, green_rot), "rotation branch must be semantically active"

    frames = _find_frames(inp, divider_col + 1)
    assert 4 <= len(frames) <= 5
    even_count = 0
    odd_count = 0
    clipped_count = 0
    unclipped_count = 0
    odd_rotation_matter = False
    interior_signatures = set()

    for r0, c0, r1, c1 in frames:
        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        area = inner_h * inner_w
        patch = red_sprite if area % 2 == 0 else green_rot
        visible = _visible_stamp(patch, inner_h, inner_w)
        actual = out[r0 + 1 : r1, c0 + 1 : c1]
        assert np.array_equal(actual, visible), "frame interior does not match stamped sprite"
        interior_signatures.add(actual.tobytes() + bytes((inner_h, inner_w)))

        if area % 2 == 0:
            even_count += 1
        else:
            odd_count += 1
            unrotated = _visible_stamp(green_sprite, inner_h, inner_w)
            if not np.array_equal(unrotated, visible):
                odd_rotation_matter = True

        if patch.shape[0] > inner_h or patch.shape[1] > inner_w:
            clipped_count += 1
        else:
            unclipped_count += 1

    assert even_count >= 1 and odd_count >= 1
    assert clipped_count >= 1 and unclipped_count >= 1
    assert odd_rotation_matter
    assert len(interior_signatures) >= 2


def _gen_attempt() -> np.ndarray | None:
    try:
        red_mask = _random_shape()
        green_mask = _random_shape()
        if np.array_equal(red_mask, green_mask):
            return None
        if max(red_mask.shape) < 4:
            return None
        green_rot_mask = np.rot90(green_mask, -1)
        if np.array_equal(green_mask, green_rot_mask):
            return None

        red_sprite = _shape_to_patch(red_mask, RED)
        green_sprite = _shape_to_patch(green_mask, GREEN)
        dims = _build_frame_dimensions(red_sprite.shape, np.rot90(green_sprite, -1).shape)
        inp = _render_input(red_sprite, green_sprite, dims)
        _semantic_check(inp)
        return inp
    except (AssertionError, RuntimeError, ValueError):
        return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate valid input")
