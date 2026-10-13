import numpy as np


BLACK = 0
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7
MAROON = 9

LEGEND_COLORS = (GREEN, YELLOW, ORANGE, RED, MAROON)
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
SQUARE_SIZES = (3, 5, 7)
GEN_RETRIES = 400
PLACE_RETRIES = 300
SHAPE_RETRIES = 200

_RNG = np.random.default_rng()


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _has_neighbor(mask: np.ndarray, r: int, c: int) -> bool:
    h, w = mask.shape
    for dr, dc in DIRS4:
        rr = r + dr
        cc = c + dc
        if _in_bounds(h, w, rr, cc) and mask[rr, cc]:
            return True
    return False


def _find_components(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if _in_bounds(h, w, nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _mask_from_coords(coords: list[tuple[int, int]], box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    h, w = mask.shape
    start_r, start_c = map(int, coords[0])
    seen = {(start_r, start_c)}
    stack = [(start_r, start_c)]
    while stack:
        r, c = stack.pop()
        for dr, dc in DIRS4:
            nr = r + dr
            nc = c + dc
            if _in_bounds(h, w, nr, nc) and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _mirror_difference_pairs(mask: np.ndarray) -> int:
    h, w = mask.shape
    half = w // 2
    if half == 0:
        return 0
    left = mask[:, :half]
    right = np.fliplr(mask[:, w - half :])
    return int(np.count_nonzero(left != right))


def _legend_from_bottom_row(row: np.ndarray) -> list[int]:
    colors = []
    c = 0
    while c < row.shape[0]:
        color = int(row[c])
        if color == BLACK:
            c += 1
            continue
        colors.append(color)
        while c < row.shape[0] and int(row[c]) == color:
            c += 1
    assert len(colors) == 5, "expected a five-step legend on the bottom row"
    return colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each grey object by how many horizontal mirror-pairs in its square mask disagree.

    Input structure:
    - The background is Black.
    - The bottom row contains a five-step horizontal color legend.
    - Above the legend are several 4-connected Grey objects, and each object's bounding box is a square.

    Rule:
    - Read the bottom legend from left to right as the colors for difference counts 0, 1, 2, 3, and 4+.
    - For each Grey object, crop its square bounding box and compare every cell with its horizontal mirror partner inside that box.
    - Count how many mirrored left/right cell pairs disagree, cap the count at the final legend slot, and recolor the whole object with that legend color.
    - Keep the legend and the background unchanged.

    Key insight:
    - The deciding feature is left-right symmetry inside each object's own square box; size and position do not matter once the mirror-disagreement count is known.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2 and g.shape[1] >= 5

    legend = _legend_from_bottom_row(g[-1])
    out = g.copy()

    for coords in _find_components(g[:-1], GREY):
        box = _bbox(coords)
        r0, c0, r1, c1 = box
        assert (r1 - r0) == (c1 - c0), "foreground objects must have square bounding boxes"
        mask = _mask_from_coords(coords, box)
        diff_pairs = _mirror_difference_pairs(mask)
        color = int(legend[min(diff_pairs, 4)])
        for r, c in coords:
            out[r, c] = color

    return out


def _cross_mask(size: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    mid = size // 2
    mask[mid, :] = True
    mask[:, mid] = True
    return mask


def _pair_candidates(mask: np.ndarray) -> list[tuple[int, int]]:
    size = mask.shape[0]
    half = size // 2
    candidates = []
    for r in range(size):
        for c in range(half):
            mc = size - 1 - c
            if mask[r, c] or mask[r, mc]:
                continue
            if _has_neighbor(mask, r, c) and _has_neighbor(mask, r, mc):
                candidates.append((r, c))
    return candidates


def _single_candidates(mask: np.ndarray) -> list[tuple[int, int]]:
    size = mask.shape[0]
    candidates = []
    for r in range(size):
        for c in range(size):
            mc = size - 1 - c
            if c == mc or mask[r, c] or mask[r, mc]:
                continue
            if _has_neighbor(mask, r, c):
                candidates.append((r, c))
    return candidates


def _max_mismatch_pairs(size: int) -> int:
    half = size // 2
    return half * (size - 1)


def _build_object_mask(target_pairs: int) -> np.ndarray:
    size_options = [size for size in SQUARE_SIZES if _max_mismatch_pairs(size) >= target_pairs]
    for _ in range(SHAPE_RETRIES):
        size = int(_RNG.choice(size_options))
        mask = _cross_mask(size)

        pre_steps = int(_RNG.integers(0, size + 2))
        for _ in range(pre_steps):
            candidates = _pair_candidates(mask)
            if not candidates:
                break
            r, c = candidates[int(_RNG.integers(len(candidates)))]
            mc = size - 1 - c
            mask[r, c] = True
            mask[r, mc] = True

        ok = True
        for _ in range(target_pairs):
            candidates = _single_candidates(mask)
            if not candidates:
                ok = False
                break
            r, c = candidates[int(_RNG.integers(len(candidates)))]
            mask[r, c] = True
        if not ok:
            continue

        post_steps = int(_RNG.integers(0, size))
        for _ in range(post_steps):
            candidates = _pair_candidates(mask)
            if not candidates:
                break
            r, c = candidates[int(_RNG.integers(len(candidates)))]
            mc = size - 1 - c
            mask[r, c] = True
            mask[r, mc] = True

        if _is_connected(mask) and _mirror_difference_pairs(mask) == target_pairs:
            return mask

    raise RuntimeError(f"failed to build mask for mismatch count {target_pairs}")


def _sample_pair_targets() -> list[int]:
    targets = [0, 1, 2, 3, int(_RNG.integers(4, 7))]
    extra_count = int(_RNG.integers(0, 3))
    for _ in range(extra_count):
        slot = int(_RNG.integers(0, 5))
        if slot == 4:
            targets.append(int(_RNG.integers(4, 7)))
        else:
            targets.append(slot)
    return targets


def _place_masks(scene_h: int, scene_w: int, masks: list[np.ndarray]) -> list[tuple[int, int]] | None:
    blocked = np.zeros((scene_h, scene_w), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(masks)
    order = list(range(len(masks)))
    order.sort(key=lambda idx: (-masks[idx].shape[0], int(_RNG.integers(0, 1_000_000))))

    for idx in order:
        size = masks[idx].shape[0]
        placed = False
        for _ in range(PLACE_RETRIES):
            r0 = int(_RNG.integers(0, scene_h - size + 1))
            c0 = int(_RNG.integers(0, scene_w - size + 1))
            rr0 = max(0, r0 - 1)
            rr1 = min(scene_h, r0 + size + 1)
            cc0 = max(0, c0 - 1)
            cc1 = min(scene_w, c0 + size + 1)
            if blocked[rr0:rr1, cc0:cc1].any():
                continue
            blocked[rr0:rr1, cc0:cc1] = True
            placements[idx] = (r0, c0)
            placed = True
            break
        if not placed:
            return None

    return [(int(r0), int(c0)) for r0, c0 in placements]


def _draw_legend(grid: np.ndarray) -> None:
    row = grid[-1]
    widths = [int(_RNG.integers(1, 4)) for _ in LEGEND_COLORS]
    total = sum(widths)
    start = int(_RNG.integers(0, row.shape[0] - total + 1))
    c = start
    for color, width in zip(LEGEND_COLORS, widths):
        row[c : c + width] = color
        c += width


def _gen_attempt() -> np.ndarray | None:
    targets = _sample_pair_targets()
    masks = [_build_object_mask(target) for target in targets]

    size_sum = sum(mask.shape[0] + 1 for mask in masks)
    max_size = max(mask.shape[0] for mask in masks)
    scene_h_min = max(12, min(29, max_size + len(masks) + 1))
    scene_w_min = max(16, min(30, max(max_size + 8, size_sum // 2)))
    if scene_h_min > 29 or scene_w_min > 30:
        return None

    scene_h = int(_RNG.integers(scene_h_min, 30))
    scene_w = int(_RNG.integers(scene_w_min, 31))
    placements = _place_masks(scene_h, scene_w, masks)
    if placements is None:
        return None

    grid = np.full((scene_h + 1, scene_w), BLACK, dtype=int)
    _draw_legend(grid)

    for mask, (r0, c0) in zip(masks, placements):
        size = mask.shape[0]
        patch = grid[r0 : r0 + size, c0 : c0 + size]
        assert np.all(patch == BLACK)
        patch[mask] = GREY

    comps = _find_components(grid[:-1], GREY)
    if len(comps) != len(masks):
        return None

    bins_seen = set()
    for coords in comps:
        box = _bbox(coords)
        mask = _mask_from_coords(coords, box)
        bins_seen.add(min(_mirror_difference_pairs(mask), 4))
    if bins_seen != {0, 1, 2, 3, 4}:
        return None

    out = main(grid)
    assert out.shape == grid.shape
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid symmetry-gradient input")
