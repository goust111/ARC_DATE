from __future__ import annotations

import numpy as np


BG = 0
MAX_SIDE = 30
GEN_RETRIES = 400
PAIR_COUNT_CHOICES = (1, 2, 3)
ORIENTATION_CHOICES = ("horizontal", "vertical")
RELATION_CHOICES = ("identical", "disjoint", "partial")


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
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _frame_components(grid: np.ndarray) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []
    for color in map(int, np.unique(grid)):
        if color == BG:
            continue
        for coords in _components(grid == color):
            rows = coords[:, 0]
            cols = coords[:, 1]
            r0, r1 = int(rows.min()), int(rows.max())
            c0, c1 = int(cols.min()), int(cols.max())
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if h < 3 or w < 3:
                continue
            if len(coords) != 2 * (h + w) - 4:
                continue

            ok = True
            for rr, cc in coords:
                if rr not in (r0, r1) and cc not in (c0, c1):
                    ok = False
                    break
            if not ok:
                continue
            if not (
                np.all(grid[r0, c0 : c1 + 1] == color)
                and np.all(grid[r1, c0 : c1 + 1] == color)
                and np.all(grid[r0 : r1 + 1, c0] == color)
                and np.all(grid[r0 : r1 + 1, c1] == color)
            ):
                continue

            frames.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "inner_shape": (h - 2, w - 2),
                }
            )
    return frames


def _ordered_pair(pair: list[dict[str, object]]) -> tuple[dict[str, object], dict[str, object]]:
    a, b = pair
    ar0, ac0, _, _ = map(int, a["bbox"])
    br0, bc0, _, _ = map(int, b["bbox"])
    if abs(ac0 - bc0) >= abs(ar0 - br0):
        return (a, b) if ac0 <= bc0 else (b, a)
    return (a, b) if ar0 <= br0 else (b, a)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-colored hollow frames, xor their interior patterns, and keep the result only in the later frame.

    Input structure:
    - The background is black.
    - Several one-pixel-thick rectangular hollow frames appear on the grid.
    - Each frame color appears exactly twice, and the two matching frames have equal interior size.
    - Each frame interior contains a pattern drawn with non-black pixels on a black background.

    Rule:
    - Group frames by border color.
    - For each same-colored pair, compare the two interiors cell by cell.
    - Keep a pixel only where exactly one interior has a non-black pixel at that relative position; erase positions where both do or both do not.
    - Write that xor pattern into the frame that is farther right or lower, and clear the left or upper frame interior to black while keeping all borders.

    Key insight:
    - Border color identifies which frames belong together, but the interior merge is not union: matching occupied cells cancel and only differences survive.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    by_color: dict[int, list[dict[str, object]]] = {}
    for frame in _frame_components(g):
        by_color.setdefault(int(frame["color"]), []).append(frame)

    for color, pair in by_color.items():
        if len(pair) != 2:
            continue
        first, second = _ordered_pair(pair)
        fr0, fc0, fr1, fc1 = map(int, first["bbox"])
        sr0, sc0, sr1, sc1 = map(int, second["bbox"])
        assert (fr1 - fr0, fc1 - fc0) == (sr1 - sr0, sc1 - sc0)

        left_or_upper = g[fr0 + 1 : fr1, fc0 + 1 : fc1]
        right_or_lower = g[sr0 + 1 : sr1, sc0 + 1 : sc1]

        only_first = (left_or_upper != BG) & (right_or_lower == BG)
        only_second = (left_or_upper == BG) & (right_or_lower != BG)

        out[fr0 + 1 : fr1, fc0 + 1 : fc1] = BG
        target = out[sr0 + 1 : sr1, sc0 + 1 : sc1]
        target[:] = BG
        target[only_first] = left_or_upper[only_first]
        target[only_second] = right_or_lower[only_second]

    return out


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    dst[r0 : r0 + h, c0 : c0 + w] = patch


def _mask_from_indices(shape: tuple[int, int], flat_idx: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask.flat[flat_idx] = True
    return mask


def _random_mask(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    cells = h * w
    max_count = max(2, min(cells - 1, (2 * cells) // 3))
    count = int(rng.integers(2, max_count + 1))
    flat_idx = rng.choice(cells, size=count, replace=False)
    return _mask_from_indices((h, w), flat_idx)


def _relation_masks(
    rng: np.random.Generator, h: int, w: int, mode: str
) -> tuple[np.ndarray, np.ndarray]:
    cells = h * w
    if mode == "identical":
        mask = _random_mask(rng, h, w)
        return mask, mask.copy()

    perm = rng.permutation(cells)
    if mode == "disjoint":
        total = int(rng.integers(4, min(cells, 8) + 1))
        count_a = int(rng.integers(2, total - 1))
        count_b = total - count_a
        return (
            _mask_from_indices((h, w), perm[:count_a]),
            _mask_from_indices((h, w), perm[count_a : count_a + count_b]),
        )

    for _ in range(50):
        shared = int(rng.integers(1, min(3, cells - 2) + 1))
        unique_a = int(rng.integers(1, min(3, cells - shared - 1) + 1))
        unique_b_max = min(3, cells - shared - unique_a)
        if unique_b_max < 1:
            continue
        unique_b = int(rng.integers(1, unique_b_max + 1))
        break
    else:
        raise RuntimeError("failed to sample partial-overlap masks")

    a = np.zeros((h, w), dtype=bool)
    b = np.zeros((h, w), dtype=bool)
    shared_idx = perm[:shared]
    a_only_idx = perm[shared : shared + unique_a]
    b_only_idx = perm[shared + unique_a : shared + unique_a + unique_b]
    a.flat[shared_idx] = True
    a.flat[a_only_idx] = True
    b.flat[shared_idx] = True
    b.flat[b_only_idx] = True
    return a, b


def _frame_patch(border_color: int, pattern_color: int, mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    patch = np.full((h + 2, w + 2), BG, dtype=int)
    patch[0, :] = border_color
    patch[-1, :] = border_color
    patch[:, 0] = border_color
    patch[:, -1] = border_color
    patch[1:-1, 1:-1][mask] = pattern_color
    return patch


def _pair_patches(
    rng: np.random.Generator, border_color: int, pattern_color: int
) -> tuple[np.ndarray, np.ndarray]:
    max_inner = 6
    min_inner = 3
    h = int(rng.integers(min_inner, max_inner + 1))
    w = int(rng.integers(min_inner, max_inner + 1))
    orientation = str(rng.choice(ORIENTATION_CHOICES))
    relation = str(rng.choice(RELATION_CHOICES))
    gap = int(rng.integers(1, 4))

    first_mask, second_mask = _relation_masks(rng, h, w, relation)
    xor_mask = np.logical_xor(first_mask, second_mask)

    first_frame = _frame_patch(border_color, pattern_color, first_mask)
    second_frame = _frame_patch(border_color, pattern_color, second_mask)
    xor_frame = _frame_patch(border_color, pattern_color, xor_mask)
    empty_frame = _frame_patch(border_color, pattern_color, np.zeros((h, w), dtype=bool))

    fh, fw = first_frame.shape
    if orientation == "horizontal":
        pair_h, pair_w = fh, 2 * fw + gap
        first_pos = (0, 0)
        second_pos = (0, fw + gap)
    else:
        pair_h, pair_w = 2 * fh + gap, fw
        first_pos = (0, 0)
        second_pos = (fh + gap, 0)

    inp = np.full((pair_h, pair_w), BG, dtype=int)
    out = np.full_like(inp, BG)
    _paste(inp, first_frame, *first_pos)
    _paste(inp, second_frame, *second_pos)
    _paste(out, empty_frame, *first_pos)
    _paste(out, xor_frame, *second_pos)
    return inp, out


def _can_place(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> bool:
    H, W = occupied.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(H, r0 + h + pad)
    cc1 = min(W, c0 + w + pad)
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _place_patch(
    rng: np.random.Generator, occupied: np.ndarray, patch: np.ndarray
) -> tuple[int, int] | None:
    H, W = occupied.shape
    h, w = patch.shape
    starts = [
        (r0, c0)
        for r0 in range(H - h + 1)
        for c0 in range(W - w + 1)
        if _can_place(occupied, r0, c0, h, w)
    ]
    if not starts:
        return None
    idx = int(rng.integers(0, len(starts)))
    return starts[idx]


def _crop_used(
    rng: np.random.Generator, inp: np.ndarray, out: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = np.where(inp != BG)
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())
    top_pad = int(rng.integers(0, min(3, r0) + 1))
    left_pad = int(rng.integers(0, min(3, c0) + 1))
    bottom_pad = int(rng.integers(0, min(3, inp.shape[0] - 1 - r1) + 1))
    right_pad = int(rng.integers(0, min(3, inp.shape[1] - 1 - c1) + 1))
    rr0 = r0 - top_pad
    cc0 = c0 - left_pad
    rr1 = r1 + bottom_pad + 1
    cc1 = c1 + right_pad + 1
    return inp[rr0:rr1, cc0:cc1], out[rr0:rr1, cc0:cc1]


def _generate_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    n_pairs = int(rng.choice(PAIR_COUNT_CHOICES))

    palette = rng.permutation(np.arange(1, 10))
    pattern_color = int(palette[0])
    frame_colors = [int(x) for x in palette[1 : 1 + n_pairs]]

    inp = np.full((MAX_SIDE, MAX_SIDE), BG, dtype=int)
    out = np.full_like(inp, BG)
    occupied = np.zeros_like(inp, dtype=bool)

    for border_color in frame_colors:
        pair_in, pair_out = _pair_patches(rng, border_color, pattern_color)
        placement = _place_patch(rng, occupied, pair_in)
        if placement is None:
            return None
        r0, c0 = placement
        h, w = pair_in.shape
        _paste(inp, pair_in, r0, c0)
        _paste(out, pair_out, r0, c0)
        occupied[r0 : r0 + h, c0 : c0 + w] = True

    inp, out = _crop_used(rng, inp, out)
    if inp.shape[0] > MAX_SIDE or inp.shape[1] > MAX_SIDE:
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid xor-frame ARC input")
