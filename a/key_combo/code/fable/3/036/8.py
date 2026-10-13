from __future__ import annotations

import itertools
from collections import Counter

import numpy as np


RED = 2
GEN_ATTEMPTS = 800
GEN_BG_CANDIDATES = [0, 1, 3, 4, 5, 6, 7, 8, 9]
GEN_MOTIF_COLORS = [0, 1, 3, 4, 5, 6, 7, 8, 9]


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).flat)).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int | None = None) -> None:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    if transparent is None:
        view[:] = patch
        return
    mask = patch != int(transparent)
    view[mask] = patch[mask]


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if int(connectivity) == 8:
        neighbors += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    comps: list[np.ndarray] = []
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
                for dr, dc in neighbors:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _is_filled_square(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> bool:
    r0, c0, r1, c1 = box
    side_r = r1 - r0 + 1
    side_c = c1 - c0 + 1
    return side_r == side_c and np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == int(color))


def _scale_patch(patch: np.ndarray, factor: int) -> np.ndarray:
    k = int(factor)
    assert k >= 1
    return np.kron(patch, np.ones((k, k), dtype=patch.dtype))


def _parse_input(grid: np.ndarray) -> tuple[np.ndarray, int, int, list[int]]:
    comps = components_mask(grid != get_bg(grid), connectivity=4)
    assert comps, "expected foreground objects"
    bg = get_bg(grid)

    motif_patch: np.ndarray | None = None
    target_sizes: list[int] = []
    for coords in comps:
        box = bbox_coords(coords)
        patch = crop_bbox(grid, box)
        colors = {int(v) for v in patch.flatten() if int(v) != bg}
        assert colors, "component cannot be empty"
        if colors == {RED}:
            assert _is_filled_square(grid, box, RED), "target instructions must be solid red squares"
            side = box[2] - box[0] + 1
            target_sizes.append(int(side))
            continue

        assert RED in colors, "motif must contain the red ruler block"
        assert motif_patch is None, "expected exactly one mixed-color motif component"
        motif_patch = patch

    assert motif_patch is not None, "missing motif component"
    assert len(target_sizes) >= 2, "expected several target red squares"
    assert len(set(target_sizes)) == len(target_sizes), "target red square sizes must be distinct"

    red_comps = components_mask(motif_patch == RED, connectivity=4)
    assert len(red_comps) == 1, "motif must contain one red ruler block"
    red_box = bbox_coords(red_comps[0])
    assert _is_filled_square(motif_patch, red_box, RED), "ruler must be a solid square"
    m = red_box[2] - red_box[0] + 1
    assert m >= 1
    assert all(side % m == 0 for side in target_sizes), "target squares must encode integer scale factors"
    return motif_patch, bg, m, sorted(target_sizes)


def _build_staircase(motif_patch: np.ndarray, bg: int, ruler_size: int, target_sizes: list[int]) -> np.ndarray:
    copies: list[np.ndarray] = []
    for rank, side in enumerate(sorted(target_sizes), start=1):
        factor = int(side) // int(ruler_size)
        copy = _scale_patch(motif_patch, factor)
        if rank % 2 == 0:
            copy = np.fliplr(copy)
        copies.append(copy)

    out_h = sum(int(copy.shape[0]) for copy in copies)
    out_w = sum(int(copy.shape[1]) for copy in copies)
    out = np.full((out_h, out_w), int(bg), dtype=int)

    r0 = 0
    c0 = 0
    for copy in copies:
        paste(out, copy, r0, c0, transparent=bg)
        r0 += int(copy.shape[0])
        c0 += int(copy.shape[1])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Arrange scaled copies of the source motif into a diagonal staircase ordered by red-square size.

    Input structure:
    - A dominant background contains one connected multi-color motif and several separate pure-red solid squares.
    - Inside the motif, the red pixels form one solid m x m square ruler, and the motif is not horizontally symmetric.
    - Every other foreground object is a standalone red square whose side length is a distinct multiple of m.

    Rule:
    - Treat each standalone red square as one instruction to copy the motif.
    - Its side length divided by m gives the integer scaling factor for that copy.
    - Sort the copies from smallest scale to largest; after sorting, horizontally mirror the 2nd, 4th, 6th, ... copies.
    - Place the first copy at the output's top-left corner, then place each later copy with its top-left one row and one column beyond the previous copy's bounding box.

    Key insight:
    - The scattered red squares are scale commands, while queue rank decides whether a scaled copy keeps its facing direction or flips horizontally.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    motif_patch, bg, ruler_size, target_sizes = _parse_input(g)
    return _build_staircase(motif_patch, bg, ruler_size, target_sizes)


def _grow_connected_cells(
    rng: np.random.Generator,
    h: int,
    w: int,
    start: tuple[int, int],
    target_count: int,
) -> set[tuple[int, int]]:
    cells = {start}
    frontier = {start}
    while len(cells) < target_count and frontier:
        base = tuple(list(frontier)[int(rng.integers(len(frontier)))])
        options: list[tuple[int, int]] = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr = base[0] + dr
            nc = base[1] + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                options.append((nr, nc))
        if not options:
            frontier.remove(base)
            continue
        picked = options[int(rng.integers(len(options)))]
        cells.add(picked)
        frontier.add(picked)
    return cells


def _make_motif(rng: np.random.Generator) -> tuple[np.ndarray, int, int] | None:
    for _ in range(200):
        ruler_size = int(rng.integers(1, 4))
        logical_h = int(rng.integers(3, 6))
        logical_w = int(rng.integers(3, 6))
        anchor = (int(rng.integers(logical_h)), int(rng.integers(logical_w)))

        min_fill = max(5, int(np.ceil(0.45 * logical_h * logical_w)))
        max_fill = max(min_fill, int(np.floor(0.8 * logical_h * logical_w)))
        fill_count = int(rng.integers(min_fill, max_fill + 1))
        cells = _grow_connected_cells(rng, logical_h, logical_w, anchor, fill_count)
        if len(cells) != fill_count:
            continue

        coords = np.array(sorted(cells), dtype=int)
        r0, c0, r1, c1 = bbox_coords(coords)
        cells = {(r - r0, c - c0) for r, c in cells}
        anchor = (anchor[0] - r0, anchor[1] - c0)
        logical_h = r1 - r0 + 1
        logical_w = c1 - c0 + 1

        if logical_h < 3 or logical_w < 3:
            continue
        if anchor[0] in {0, logical_h - 1} or anchor[1] in {0, logical_w - 1}:
            continue

        bg = int(rng.choice(GEN_BG_CANDIDATES))
        palette = [color for color in GEN_MOTIF_COLORS if color not in {bg, RED}]
        n_colors = int(rng.integers(2, min(5, len(palette)) + 1))
        motif_colors = rng.choice(palette, size=n_colors, replace=False).astype(int).tolist()

        logical = np.full((logical_h, logical_w), bg, dtype=int)
        logical[anchor] = RED
        others = sorted(cells - {anchor})
        if len(others) < 4:
            continue
        for idx, (r, c) in enumerate(others):
            if idx < len(motif_colors):
                logical[r, c] = int(motif_colors[idx])
            else:
                logical[r, c] = int(motif_colors[int(rng.integers(len(motif_colors)))])

        if np.array_equal(logical, np.fliplr(logical)):
            continue

        patch = _scale_patch(logical, ruler_size)
        return patch, bg, ruler_size
    return None


def _choose_scales(
    rng: np.random.Generator,
    patch_h: int,
    patch_w: int,
) -> tuple[int, ...] | None:
    max_sum = min(30 // int(patch_h), 30 // int(patch_w))
    max_scale = min(6, max_sum)
    candidates: list[tuple[int, ...]] = []
    for count in range(2, 5):
        for combo in itertools.combinations(range(1, max_scale + 1), count):
            if sum(combo) <= max_sum:
                candidates.append(combo)
    if not candidates:
        return None
    weights = np.array([len(combo) for combo in candidates], dtype=float)
    weights /= weights.sum()
    return candidates[int(rng.choice(len(candidates), p=weights))]


def _boxes_too_close(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    gap: int = 1,
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    separated = (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )
    return not separated


def _place_objects(
    rng: np.random.Generator,
    motif_shape: tuple[int, int],
    target_sizes: list[int],
) -> tuple[int, int, dict[str, tuple[int, int]]] | None:
    items = [("motif", int(motif_shape[0]), int(motif_shape[1]))]
    items.extend((f"square_{idx}", int(side), int(side)) for idx, side in enumerate(target_sizes))
    items.sort(key=lambda item: item[1] * item[2], reverse=True)

    total_area = sum(h * w for _name, h, w in items)
    max_h = max(h for _name, h, _w in items)
    max_w = max(w for _name, _h, w in items)
    base = max(12, int(np.ceil(np.sqrt(total_area * 2.6))))
    min_h = max(base, max_h + 2)
    min_w = max(base, max_w + 2)
    if min_h > 30 or min_w > 30:
        return None

    for _ in range(300):
        height = int(rng.integers(min_h, 31))
        width = int(rng.integers(min_w, 31))
        placed: list[tuple[str, tuple[int, int, int, int]]] = []
        positions: dict[str, tuple[int, int]] = {}

        success = True
        for name, h, w in items:
            found = False
            for _ in range(600):
                r0 = int(rng.integers(0, height - h + 1))
                c0 = int(rng.integers(0, width - w + 1))
                box = (r0, c0, r0 + h - 1, c0 + w - 1)
                if any(_boxes_too_close(box, other_box, gap=1) for _other_name, other_box in placed):
                    continue
                placed.append((name, box))
                positions[name] = (r0, c0)
                found = True
                break
            if not found:
                success = False
                break
        if success:
            return height, width, positions
    return None


def _background_is_dominant(grid: np.ndarray, bg: int) -> bool:
    values, counts = np.unique(grid, return_counts=True)
    count_by_color = {int(v): int(c) for v, c in zip(values.tolist(), counts.tolist(), strict=False)}
    bg_count = count_by_color.get(int(bg), 0)
    return all(bg_count > count for color, count in count_by_color.items() if color != int(bg))


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    motif_data = _make_motif(rng)
    if motif_data is None:
        return None
    motif_patch, bg, ruler_size = motif_data
    scales = _choose_scales(rng, *motif_patch.shape)
    if scales is None:
        return None
    target_sizes = [int(ruler_size * scale) for scale in scales]

    placement = _place_objects(rng, motif_patch.shape, target_sizes)
    if placement is None:
        return None
    height, width, positions = placement

    grid = np.full((height, width), int(bg), dtype=int)
    motif_r0, motif_c0 = positions["motif"]
    paste(grid, motif_patch, motif_r0, motif_c0)
    for idx, side in enumerate(target_sizes):
        r0, c0 = positions[f"square_{idx}"]
        grid[r0 : r0 + side, c0 : c0 + side] = RED

    if not _background_is_dominant(grid, bg):
        return None

    expected = _build_staircase(motif_patch, bg, ruler_size, target_sizes)
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return None
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid staircase-queue input")
