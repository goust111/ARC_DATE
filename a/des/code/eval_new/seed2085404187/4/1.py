from __future__ import annotations

import random

import numpy as np


BG = 0
MAX_SIDE = 30
GEN_ATTEMPTS = 400
PLACE_ATTEMPTS = 200


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components of a boolean mask as coordinate arrays."""
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _tight_crop(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _component_infos(grid: np.ndarray, color: int) -> list[dict[str, object]]:
    infos: list[dict[str, object]] = []
    for coords in _components(grid == int(color)):
        box = _bbox(coords)
        r0, c0, r1, c1 = box
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        holes = patch == BG
        assert bool(holes.any())
        hole_coords = np.argwhere(holes)
        assert np.all(hole_coords[:, 0] > 0) and np.all(hole_coords[:, 0] < patch.shape[0] - 1)
        assert np.all(hole_coords[:, 1] > 0) and np.all(hole_coords[:, 1] < patch.shape[1] - 1)
        infos.append(
            {
                "bbox": box,
                "rect_area": int((r1 - r0 + 1) * (c1 - c0 + 1)),
                "hole_mask": holes,
                "hole_crop": _tight_crop(holes),
                "hole_count": int(holes.sum()),
            }
        )
    return infos


def _pair_gap(
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    ar0, ac0, ar1, ac1 = box_a
    br0, bc0, br1, bc1 = box_b

    row0 = max(ar0, br0)
    row1 = min(ar1, br1)
    col0 = max(ac0, bc0)
    col1 = min(ac1, bc1)

    if ac1 < bc0 and row0 <= row1:
        return int(row0), int(ac1 + 1), int(row1), int(bc0 - 1)
    if bc1 < ac0 and row0 <= row1:
        return int(row0), int(bc1 + 1), int(row1), int(ac0 - 1)
    if ar1 < br0 and col0 <= col1:
        return int(ar1 + 1), int(col0), int(br0 - 1), int(col1)
    if br1 < ar0 and col0 <= col1:
        return int(br1 + 1), int(col0), int(ar0 - 1), int(col1)
    raise AssertionError("same-color rectangles must share a row band or a column band")


def _draw_centered(
    out: np.ndarray,
    color: int,
    pattern: np.ndarray,
    gap_box: tuple[int, int, int, int],
) -> None:
    r0, c0, r1, c1 = gap_box
    gh = int(r1 - r0 + 1)
    gw = int(c1 - c0 + 1)
    ph, pw = map(int, pattern.shape)
    assert ph <= gh and pw <= gw
    top = int(r0 + (gh - ph) // 2)
    left = int(c0 + (gw - pw) // 2)
    region = out[top : top + ph, left : left + pw]
    region[pattern] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw a centered scaled motif between each same-color rectangle pair using the smaller block's holes as the stamp.

    Input structure:
    - Black background with several pairs of same-color filled rectangles.
    - Each non-background color appears in exactly two disjoint rectangular blocks.
    - Every block contains one or more black hole pixels fully enclosed inside the rectangle; those holes form a small pattern.

    Rule:
    - For each color pair, extract the hole pattern from both rectangles and crop it to the tight hole bounding box.
    - Use the smaller rectangle as the stamp source, and use the larger rectangle's total hole-pixel count N as the scale factor.
    - Enlarge the stamp by replacing each of its pixels with an N x N solid square of the pair color.
    - Place that enlarged pattern in the background gap between the two rectangles, centered within the shared row-band or column-band gap, while keeping both original rectangles unchanged.

    Key insight:
    - The holes provide two separate signals: the smaller rectangle contributes the shape to draw, while the larger rectangle contributes only the count that determines the enlargement factor.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for color in sorted(int(v) for v in np.unique(g) if int(v) != BG):
        infos = _component_infos(g, int(color))
        assert len(infos) == 2
        infos.sort(
            key=lambda item: (
                int(item["rect_area"]),
                int(np.prod(np.array(item["hole_crop"]).shape)),
                int(item["hole_count"]),
            )
        )
        small, large = infos
        stamp = np.array(small["hole_crop"], dtype=bool)
        scale = int(large["hole_count"])
        assert scale > 0
        pattern = np.kron(stamp.astype(int), np.ones((scale, scale), dtype=int)).astype(bool)
        gap = _pair_gap(
            small["bbox"],  # type: ignore[arg-type]
            large["bbox"],  # type: ignore[arg-type]
        )
        _draw_centered(out, int(color), pattern, gap)

    return out


_BASE_SHAPES = [
    np.array([[1]], dtype=bool),
    np.array([[1, 1]], dtype=bool),
    np.array([[1], [1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1], [1], [1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
]

_SCALED_BASES = [
    np.array([[1]], dtype=bool),
    np.array([[1, 1]], dtype=bool),
    np.array([[1], [1]], dtype=bool),
]


def _dihedral(mask: np.ndarray) -> np.ndarray:
    out = np.rot90(mask, random.randrange(4))
    if random.random() < 0.5:
        out = np.fliplr(out)
    return np.array(out, dtype=bool, copy=True)


def _sample_holes(max_span: int) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(100):
        if random.random() < 0.5:
            stamp = _dihedral(random.choice(_BASE_SHAPES))
            large_hole = stamp.copy()
        else:
            stamp = _dihedral(random.choice(_SCALED_BASES))
            base_count = int(stamp.sum())
            if base_count == 1:
                factor = random.choice([2, 3])
            else:
                factor = 2
            large_hole = np.kron(stamp.astype(int), np.ones((factor, factor), dtype=int)).astype(bool)

        hole_count = int(large_hole.sum())
        out_h = int(stamp.shape[0] * hole_count)
        out_w = int(stamp.shape[1] * hole_count)
        if max(out_h, out_w) <= max_span:
            return stamp, large_hole
    raise RuntimeError("failed to sample compatible hole patterns")


def _rand_between(lo: int, hi: int) -> int:
    assert lo <= hi
    return int(random.randint(int(lo), int(hi)))


def _place_hole(rect: np.ndarray, hole: np.ndarray) -> None:
    h, w = map(int, rect.shape)
    hh, hw = map(int, hole.shape)
    top = _rand_between(1, h - hh - 1)
    left = _rand_between(1, w - hw - 1)
    rect[top : top + hh, left : left + hw][hole] = BG


def _build_pair(color: int, max_span: int) -> tuple[np.ndarray, np.ndarray]:
    stamp, large_hole = _sample_holes(max_span)
    hole_count = int(large_hole.sum())
    pattern = np.kron(stamp.astype(int), np.ones((hole_count, hole_count), dtype=int)).astype(bool)
    out_h, out_w = map(int, pattern.shape)
    small_h, small_w = map(int, stamp.shape)
    large_h, large_w = map(int, large_hole.shape)

    horizontal = bool(random.randrange(2))
    if horizontal:
        common_h = max(out_h + 2 * _rand_between(1, 3), small_h + 2, large_h + 2)
        small_rect_w = small_w + _rand_between(2, 5)
        large_rect_w = max(large_w + _rand_between(2, 6), small_rect_w + 1)
        gap_w = out_w + 2 * _rand_between(1, 3)

        small_rect = np.full((common_h, small_rect_w), int(color), dtype=int)
        large_rect = np.full((common_h, large_rect_w), int(color), dtype=int)
        _place_hole(small_rect, stamp)
        _place_hole(large_rect, large_hole)

        patch = np.zeros((common_h, small_rect_w + gap_w + large_rect_w), dtype=int)
        out = np.zeros_like(patch)
        if random.random() < 0.5:
            left_rect, right_rect = small_rect, large_rect
        else:
            left_rect, right_rect = large_rect, small_rect

        lw = int(left_rect.shape[1])
        rw = int(right_rect.shape[1])
        patch[:, :lw] = left_rect
        patch[:, lw + gap_w : lw + gap_w + rw] = right_rect
        out[:, :] = patch
        gap_box = (0, int(lw), int(common_h - 1), int(lw + gap_w - 1))
        _draw_centered(out, int(color), pattern, gap_box)

        small_area = int(np.prod(small_rect.shape))
        large_area = int(np.prod(large_rect.shape))
        assert large_area > small_area
        return patch, out

    common_w = max(out_w + 2 * _rand_between(1, 3), small_w + 2, large_w + 2)
    small_rect_h = small_h + _rand_between(2, 5)
    large_rect_h = max(large_h + _rand_between(2, 6), small_rect_h + 1)
    gap_h = out_h + 2 * _rand_between(1, 3)

    small_rect = np.full((small_rect_h, common_w), int(color), dtype=int)
    large_rect = np.full((large_rect_h, common_w), int(color), dtype=int)
    _place_hole(small_rect, stamp)
    _place_hole(large_rect, large_hole)

    patch = np.zeros((small_rect_h + gap_h + large_rect_h, common_w), dtype=int)
    out = np.zeros_like(patch)
    if random.random() < 0.5:
        top_rect, bottom_rect = small_rect, large_rect
    else:
        top_rect, bottom_rect = large_rect, small_rect

    th = int(top_rect.shape[0])
    bh = int(bottom_rect.shape[0])
    patch[:th, :] = top_rect
    patch[th + gap_h : th + gap_h + bh, :] = bottom_rect
    out[:, :] = patch
    gap_box = (int(th), 0, int(th + gap_h - 1), int(common_w - 1))
    _draw_centered(out, int(color), pattern, gap_box)
    assert int(np.prod(large_rect.shape)) > int(np.prod(small_rect.shape))
    return patch, out


def _reserve_ok(occupied: np.ndarray, top: int, left: int, h: int, w: int) -> bool:
    r0 = max(0, int(top) - 1)
    c0 = max(0, int(left) - 1)
    r1 = min(MAX_SIDE, int(top + h + 1))
    c1 = min(MAX_SIDE, int(left + w + 1))
    return not bool(occupied[r0:r1, c0:c1].any())


def _mark_reserved(occupied: np.ndarray, top: int, left: int, h: int, w: int) -> None:
    r0 = max(0, int(top) - 1)
    c0 = max(0, int(left) - 1)
    r1 = min(MAX_SIDE, int(top + h + 1))
    c1 = min(MAX_SIDE, int(left + w + 1))
    occupied[r0:r1, c0:c1] = True


def _content_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid != BG)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_with_pads(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    pads: tuple[int, int, int, int],
) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    top_pad, left_pad, bot_pad, right_pad = pads
    return grid[
        int(r0) - top_pad : int(r1) + bot_pad + 1,
        int(c0) - left_pad : int(c1) + right_pad + 1,
    ].copy()


def _generate_attempt() -> np.ndarray | None:
    n_pairs = random.choice([1, 2, 3])
    max_span = {1: 16, 2: 12, 3: 9}[n_pairs]
    colors = random.sample(range(1, 10), n_pairs)

    pair_patches: list[tuple[np.ndarray, np.ndarray]] = []
    for color in colors:
        pair = _build_pair(int(color), max_span)
        if max(pair[0].shape) > MAX_SIDE or max(pair[1].shape) > MAX_SIDE:
            return None
        pair_patches.append(pair)
    pair_patches.sort(key=lambda item: int(np.prod(item[0].shape)), reverse=True)

    inp = np.zeros((MAX_SIDE, MAX_SIDE), dtype=int)
    expected = np.zeros_like(inp)
    occupied = np.zeros_like(inp, dtype=bool)

    for patch, out_patch in pair_patches:
        ph, pw = map(int, patch.shape)
        placed = False
        for _ in range(PLACE_ATTEMPTS):
            top = random.randint(0, MAX_SIDE - ph)
            left = random.randint(0, MAX_SIDE - pw)
            if not _reserve_ok(occupied, top, left, ph, pw):
                continue
            inp[top : top + ph, left : left + pw] = patch
            expected[top : top + ph, left : left + pw] = out_patch
            _mark_reserved(occupied, top, left, ph, pw)
            placed = True
            break
        if not placed:
            return None

    bbox = _content_bbox(inp)
    r0, c0, r1, c1 = bbox
    pads = (
        random.randint(0, min(2, int(r0))),
        random.randint(0, min(2, int(c0))),
        random.randint(0, min(2, int(MAX_SIDE - 1 - r1))),
        random.randint(0, min(2, int(MAX_SIDE - 1 - c1))),
    )
    inp = _crop_with_pads(inp, bbox, pads)
    expected = _crop_with_pads(expected, bbox, pads)

    colors_present = [int(v) for v in np.unique(inp) if int(v) != BG]
    if len(colors_present) != n_pairs:
        return None
    for color in colors_present:
        if len(_component_infos(inp, color)) != 2:
            return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if inp.ndim != 2 or expected.ndim != 2:
        return None
    if max(inp.shape) > MAX_SIDE or max(expected.shape) > MAX_SIDE:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC input")
