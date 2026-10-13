from __future__ import annotations

import random

import numpy as np


DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
BG = 0
COLOR_POOL = tuple(range(1, 10))
CANVAS_SIZE = 30
MAX_GENERATION_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 300
PAIR_COUNT_RANGE = (2, 4)
PAIR_HEIGHT_RANGE = (2, 6)
PAIR_WIDTH_RANGE = (4, 9)
PAIR_GAP_RANGE = (1, 4)
_RNG = random.Random()


def _components_from_mask(mask: np.ndarray) -> list[dict[str, object]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            components.append(
                {
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(arr.shape[0]),
                }
            )
    return components


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_connected(mask: np.ndarray) -> bool:
    return int(np.count_nonzero(mask)) > 0 and len(_components_from_mask(mask)) == 1


def _shift_coords(coords: np.ndarray, dr: int, dc: int) -> np.ndarray:
    shifted = np.array(coords, dtype=int, copy=True)
    shifted[:, 0] += int(dr)
    shifted[:, 1] += int(dc)
    return shifted


def _row_gap(left: dict[str, object], right: dict[str, object]) -> int:
    left_max: dict[int, int] = {}
    for r, c in np.asarray(left["coords"], dtype=int):
        left_max[int(r)] = max(left_max.get(int(r), -10**9), int(c))

    right_min: dict[int, int] = {}
    for r, c in np.asarray(right["coords"], dtype=int):
        right_min[int(r)] = min(right_min.get(int(r), 10**9), int(c))

    shared_rows = sorted(set(left_max) & set(right_min))
    assert shared_rows, "horizontal pair must share at least one row"
    gap = min(right_min[r] - left_max[r] - 1 for r in shared_rows)
    assert gap > 0, "horizontal fragments must be separated"
    return int(gap)


def _col_gap(top: dict[str, object], bottom: dict[str, object]) -> int:
    top_max: dict[int, int] = {}
    for r, c in np.asarray(top["coords"], dtype=int):
        top_max[int(c)] = max(top_max.get(int(c), -10**9), int(r))

    bottom_min: dict[int, int] = {}
    for r, c in np.asarray(bottom["coords"], dtype=int):
        bottom_min[int(c)] = min(bottom_min.get(int(c), 10**9), int(r))

    shared_cols = sorted(set(top_max) & set(bottom_min))
    assert shared_cols, "vertical pair must share at least one column"
    gap = min(bottom_min[c] - top_max[c] - 1 for c in shared_cols)
    assert gap > 0, "vertical fragments must be separated"
    return int(gap)


def _merged_coords(comp_a: dict[str, object], comp_b: dict[str, object]) -> np.ndarray:
    ar0, ac0, ar1, ac1 = comp_a["bbox"]
    br0, bc0, br1, bc1 = comp_b["bbox"]

    row_overlap = max(ar0, br0) <= min(ar1, br1)
    col_overlap = max(ac0, bc0) <= min(ac1, bc1)
    horizontal = row_overlap and (ac1 < bc0 or bc1 < ac0)
    vertical = col_overlap and (ar1 < br0 or br1 < ar0)
    assert horizontal ^ vertical, "each color pair must align on exactly one axis"

    if horizontal:
        left, right = sorted(
            (comp_a, comp_b),
            key=lambda comp: (int(comp["bbox"][1]), int(comp["bbox"][0])),
        )
        gap = _row_gap(left, right)
        merged = np.vstack(
            [
                np.asarray(left["coords"], dtype=int),
                _shift_coords(np.asarray(right["coords"], dtype=int), 0, -gap),
            ]
        )
    else:
        top, bottom = sorted(
            (comp_a, comp_b),
            key=lambda comp: (int(comp["bbox"][0]), int(comp["bbox"][1])),
        )
        gap = _col_gap(top, bottom)
        merged = np.vstack(
            [
                np.asarray(top["coords"], dtype=int),
                _shift_coords(np.asarray(bottom["coords"], dtype=int), -gap, 0),
            ]
        )

    uniq = {tuple(cell) for cell in merged.tolist()}
    assert len(uniq) == merged.shape[0], "translated fragments must not overlap"
    return np.array(sorted(uniq), dtype=int)


def _is_solid_rectangle(coords: np.ndarray) -> bool:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(coords.shape[0]) == int((r1 - r0 + 1) * (c1 - c0 + 1))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each same-colored fragment pair together, then keep only the pairs that become solid rectangles.

    Input structure:
    - The background is black.
    - Every non-background color appears in exactly two separate 4-connected monochrome fragments.
    - The two fragments of a color are aligned either left-right with overlapping rows, or top-bottom with overlapping columns.

    Rule:
    - For each color, keep the upper/left fragment fixed and slide the lower/right fragment inward along the shared axis until the pair first touches without overlap.
    - Consider the merged shape formed by the two translated fragments.
    - If that merged shape completely fills its bounding box, draw it in the output; otherwise erase that color entirely.

    Key insight:
    - The first step is to match fragments by color and infer the translation axis from their alignment, and the second step is to treat rectangle-completion as the real keep-or-delete criterion.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = BG if np.any(g == BG) else int(np.bincount(g.ravel()).argmax())

    out = np.full_like(g, bg)
    colors = [int(color) for color in np.unique(g) if int(color) != bg]
    for color in colors:
        comps = _components_from_mask(g == color)
        assert len(comps) == 2, "each foreground color must appear in exactly two fragments"
        merged = _merged_coords(comps[0], comps[1])
        if not _is_solid_rectangle(merged):
            continue
        for r, c in merged:
            assert out[int(r), int(c)] == bg, "kept merged shapes must remain disjoint"
            out[int(r), int(c)] = color
    return out


def _trim_pair_masks(
    union: np.ndarray, piece_a: np.ndarray, piece_b: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cols = np.where(union.any(axis=0))[0]
    union = union[:, cols[0] : cols[-1] + 1]
    piece_a = piece_a[:, cols[0] : cols[-1] + 1]
    piece_b = piece_b[:, cols[0] : cols[-1] + 1]
    return union, piece_a, piece_b


def _build_horizontal_pair(rectangular: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    for _ in range(200):
        h = _RNG.randint(*PAIR_HEIGHT_RANGE)
        w = _RNG.randint(*PAIR_WIDTH_RANGE)
        cut_hi = w - 1 if rectangular else w - 2
        cut = _RNG.randint(1, cut_hi)

        union = np.zeros((h, w), dtype=bool)
        if rectangular:
            union[:] = True
        else:
            start = _RNG.randint(0, cut - 1)
            end = _RNG.randint(cut, w - 1)
            for r in range(h):
                if r > 0:
                    start = max(0, min(cut - 1, start + _RNG.choice((-1, 0, 1))))
                    end = max(cut, min(w - 1, end + _RNG.choice((-1, 0, 1))))
                union[r, start : end + 1] = True
            if not union.any():
                continue

        piece_a = np.zeros_like(union)
        piece_b = np.zeros_like(union)
        for r in range(h):
            row = np.where(union[r])[0]
            if row.size == 0:
                break
            start = int(row[0])
            end = int(row[-1])
            if not (start <= cut - 1 and end >= cut):
                break
            piece_a[r, start:cut] = True
            piece_b[r, cut : end + 1] = True
        else:
            union, piece_a, piece_b = _trim_pair_masks(union, piece_a, piece_b)
            if rectangular or not np.all(union):
                if _is_connected(piece_a) and _is_connected(piece_b):
                    return union, piece_a, piece_b
    raise RuntimeError("failed to build a valid pair")


def _make_pair_pattern(rectangular: bool, horizontal: bool) -> np.ndarray:
    union, piece_a, piece_b = _build_horizontal_pair(rectangular)

    if _RNG.random() < 0.5:
        union = np.flipud(union)
        piece_a = np.flipud(piece_a)
        piece_b = np.flipud(piece_b)
    if _RNG.random() < 0.5:
        union = np.fliplr(union)
        piece_a = np.fliplr(piece_a)
        piece_b = np.fliplr(piece_b)
    if not horizontal:
        union = union.T
        piece_a = piece_a.T
        piece_b = piece_b.T

    piece_masks = [piece_a, piece_b]
    if horizontal:
        piece_masks.sort(key=lambda mask: (_mask_bbox(mask)[1], _mask_bbox(mask)[0]))
        gap = _RNG.randint(*PAIR_GAP_RANGE)
        h, w = union.shape
        pattern = np.zeros((h, w + gap), dtype=bool)
        pattern[:, :w] |= piece_masks[0]
        pattern[:, gap : gap + w] |= piece_masks[1]
    else:
        piece_masks.sort(key=lambda mask: (_mask_bbox(mask)[0], _mask_bbox(mask)[1]))
        gap = _RNG.randint(*PAIR_GAP_RANGE)
        h, w = union.shape
        pattern = np.zeros((h + gap, w), dtype=bool)
        pattern[:h, :] |= piece_masks[0]
        pattern[gap : gap + h, :] |= piece_masks[1]

    assert len(_components_from_mask(pattern)) == 2
    return pattern


def _balanced_flags(count: int) -> list[bool]:
    true_count = count // 2
    if count % 2 == 1 and _RNG.random() < 0.5:
        true_count += 1
    flags = [True] * true_count + [False] * (count - true_count)
    _RNG.shuffle(flags)
    return flags


def _place_patterns(
    patterns: list[np.ndarray], colors: list[int]
) -> np.ndarray | None:
    canvas = np.full((CANVAS_SIZE, CANVAS_SIZE), BG, dtype=int)
    blocked = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=bool)

    order = list(range(len(patterns)))
    _RNG.shuffle(order)
    for idx in order:
        pattern = patterns[idx]
        color = colors[idx]
        h, w = pattern.shape
        if h + 2 > CANVAS_SIZE or w + 2 > CANVAS_SIZE:
            return None

        placed = False
        max_r = CANVAS_SIZE - h - 1
        max_c = CANVAS_SIZE - w - 1
        for _ in range(PLACEMENT_ATTEMPTS):
            r = _RNG.randint(1, max_r)
            c = _RNG.randint(1, max_c)
            if blocked[r - 1 : r + h + 1, c - 1 : c + w + 1].any():
                continue
            canvas[r : r + h, c : c + w][pattern] = color
            blocked[r - 1 : r + h + 1, c - 1 : c + w + 1] = True
            placed = True
            break
        if not placed:
            return None

    coords = np.argwhere(canvas != BG)
    if coords.size == 0:
        return None

    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    top_pad = _RNG.randint(1, min(5, int(r0)))
    left_pad = _RNG.randint(1, min(5, int(c0)))
    bottom_pad = _RNG.randint(1, min(5, int(CANVAS_SIZE - 1 - r1)))
    right_pad = _RNG.randint(1, min(5, int(CANVAS_SIZE - 1 - c1)))
    return canvas[
        int(r0) - top_pad : int(r1) + bottom_pad + 1,
        int(c0) - left_pad : int(c1) + right_pad + 1,
    ]


def _gen_attempt() -> np.ndarray | None:
    pair_count = _RNG.randint(*PAIR_COUNT_RANGE)
    rectangular_flags = _balanced_flags(pair_count)
    horizontal_flags = _balanced_flags(pair_count)
    colors = _RNG.sample(list(COLOR_POOL), pair_count)

    patterns = [
        _make_pair_pattern(rectangular=rectangular, horizontal=horizontal)
        for rectangular, horizontal in zip(rectangular_flags, horizontal_flags)
    ]

    grid = _place_patterns(patterns, colors)
    if grid is None:
        return None

    for color in colors:
        if len(_components_from_mask(grid == color)) != 2:
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    kept_colors = {int(color) for color in np.unique(out) if int(color) != BG}
    expected_kept = {color for color, keep in zip(colors, rectangular_flags) if keep}
    if kept_colors != expected_kept:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
