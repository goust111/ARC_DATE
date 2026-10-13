import random
from collections import defaultdict

import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GENERATION_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    excluded = set() if exclude is None else set(exclude)
    return sorted(int(v) for v in np.unique(grid) if int(v) not in excluded)


def find_components(
    grid: np.ndarray,
    bg: int,
    colors: list[int] | None = None,
    connectivity: int = 4,
) -> list[dict]:
    assert connectivity == 4
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    wanted = None if colors is None else set(colors)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color == bg or (wanted is not None and color not in wanted):
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            components.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(len(coords)),
                }
            )

    return components


def _normalized_mask(grid: np.ndarray, component: dict) -> np.ndarray:
    r0, c0, r1, c1 = component["bbox"]
    return grid[r0 : r1 + 1, c0 : c1 + 1] == int(component["color"])


def _merged_mask(mask_a: np.ndarray, mask_b: np.ndarray) -> np.ndarray:
    h = max(mask_a.shape[0], mask_b.shape[0])
    w = max(mask_a.shape[1], mask_b.shape[1])
    merged = np.zeros((h, w), dtype=bool)
    merged[: mask_a.shape[0], : mask_a.shape[1]] |= mask_a
    merged[: mask_b.shape[0], : mask_b.shape[1]] |= mask_b
    return merged


def _merged_top_left(component_a: dict, component_b: dict, merged: np.ndarray) -> tuple[int, int]:
    ar0, ac0, ar1, ac1 = component_a["bbox"]
    br0, bc0, br1, bc1 = component_b["bbox"]

    row_num = ar0 + ar1 + br0 + br1 - 2 * (merged.shape[0] - 1)
    col_num = ac0 + ac1 + bc0 + bc1 - 2 * (merged.shape[1] - 1)
    assert row_num % 4 == 0 and col_num % 4 == 0, "pair midpoint must align to the merged sprite center"
    return row_num // 4, col_num // 4


def main(grid: np.ndarray) -> np.ndarray:
    """Pair same-colored sprites, merge their normalized shapes, and redraw at the midpoint."""
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    components = find_components(g, bg=bg, connectivity=4)

    by_color: dict[int, list[dict]] = defaultdict(list)
    for component in components:
        by_color[int(component["color"])].append(component)

    out = np.full_like(g, bg)
    for color in colors_present(g, exclude={bg}):
        same_color = by_color[int(color)]
        assert len(same_color) == 2, "each non-background color must appear in exactly two sprites"
        comp_a, comp_b = same_color
        mask_a = _normalized_mask(g, comp_a)
        mask_b = _normalized_mask(g, comp_b)
        merged = _merged_mask(mask_a, mask_b)
        top, left = _merged_top_left(comp_a, comp_b, merged)
        h, w = merged.shape
        assert 0 <= top and 0 <= left and top + h <= g.shape[0] and left + w <= g.shape[1]
        target = out[top : top + h, left : left + w]
        assert np.all(target[merged] == bg), "merged sprites must not overlap"
        target[merged] = int(color)

    return out


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _random_connected_mask() -> np.ndarray:
    height = random.randint(1, 4)
    width = random.randint(1, 4)
    cells = [(r, c) for r in range(height) for c in range(width)]
    target_area = random.randint(1, min(height * width, 6))
    start = random.choice(cells)
    chosen = {start}
    frontier = set()

    def add_frontier(cell: tuple[int, int]) -> None:
        r, c = cell
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in chosen:
                frontier.add((nr, nc))

    add_frontier(start)
    while len(chosen) < target_area and frontier:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        chosen.add(cell)
        add_frontier(cell)

    mask = np.zeros((height, width), dtype=bool)
    for r, c in chosen:
        mask[r, c] = True
    return _crop_mask(mask)


def _random_pair_masks() -> tuple[np.ndarray, np.ndarray]:
    while True:
        mask_a = _random_connected_mask()
        same_shape = random.random() < 0.5
        if same_shape:
            mask_b = mask_a.copy()
        else:
            for _ in range(80):
                candidate = _random_connected_mask()
                if candidate.shape[0] % 2 != mask_a.shape[0] % 2:
                    continue
                if candidate.shape[1] % 2 != mask_a.shape[1] % 2:
                    continue
                if np.array_equal(candidate, mask_a):
                    continue
                merged = _merged_mask(mask_a, candidate)
                if merged.sum() == max(mask_a.sum(), candidate.sum()):
                    continue
                mask_b = candidate
                break
            else:
                continue

        if mask_b.shape[0] % 2 != mask_a.shape[0] % 2:
            continue
        if mask_b.shape[1] % 2 != mask_a.shape[1] % 2:
            continue
        return mask_a, mask_b


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _is_clear(occupied: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 1) -> bool:
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(occupied.shape[0] - 1, r1 + pad)
    cc1 = min(occupied.shape[1] - 1, c1 + pad)
    return not bool(occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _occupy(occupied: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    occupied[r0 : r1 + 1, c0 : c1 + 1] = True


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    rows, cols = np.where(mask)
    grid[top + rows, left + cols] = int(color)


def _pair_top_lefts(
    merged_top: int,
    merged_left: int,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    dr: int,
    dc: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    mh = max(mask_a.shape[0], mask_b.shape[0])
    mw = max(mask_a.shape[1], mask_b.shape[1])
    ar = merged_top + (mh - mask_a.shape[0]) // 2 - dr
    br = merged_top + (mh - mask_b.shape[0]) // 2 + dr
    ac = merged_left + (mw - mask_a.shape[1]) // 2 - dc
    bc = merged_left + (mw - mask_b.shape[1]) // 2 + dc
    return (ar, ac), (br, bc)


def _attempt_place_pair(
    grid: np.ndarray,
    input_occ: np.ndarray,
    output_occ: np.ndarray,
    color: int,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    orientation: str,
) -> bool:
    h, w = grid.shape
    merged = _merged_mask(mask_a, mask_b)
    mh, mw = merged.shape

    for _ in range(140):
        merged_top = random.randint(1, h - mh - 2)
        merged_left = random.randint(1, w - mw - 2)
        if orientation == "horizontal":
            dr = 0
            dc = random.randint(2, max(2, w // 4))
        elif orientation == "vertical":
            dr = random.randint(2, max(2, h // 4))
            dc = 0
        else:
            dr = random.randint(1, max(1, h // 5))
            dc = random.randint(1, max(1, w // 5))

        (ar, ac), (br, bc) = _pair_top_lefts(merged_top, merged_left, mask_a, mask_b, dr, dc)
        bbox_a = (ar, ac, ar + mask_a.shape[0] - 1, ac + mask_a.shape[1] - 1)
        bbox_b = (br, bc, br + mask_b.shape[0] - 1, bc + mask_b.shape[1] - 1)
        bbox_out = (merged_top, merged_left, merged_top + mh - 1, merged_left + mw - 1)

        if min(*bbox_a[:2], *bbox_b[:2]) < 1:
            continue
        if bbox_a[2] >= h - 1 or bbox_a[3] >= w - 1 or bbox_b[2] >= h - 1 or bbox_b[3] >= w - 1:
            continue
        if _rects_overlap(bbox_a, bbox_b):
            continue
        if not _is_clear(input_occ, bbox_a, pad=1) or not _is_clear(input_occ, bbox_b, pad=1):
            continue
        if not _is_clear(output_occ, bbox_out, pad=1):
            continue

        _paint_mask(grid, mask_a, ar, ac, color)
        _paint_mask(grid, mask_b, br, bc, color)
        _occupy(input_occ, bbox_a)
        _occupy(input_occ, bbox_b)
        _occupy(output_occ, bbox_out)
        return True

    return False


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(16, 24)
    width = random.randint(16, 24)
    bg = random.randint(0, 9)
    grid = np.full((height, width), bg, dtype=int)
    input_occ = np.zeros((height, width), dtype=bool)
    output_occ = np.zeros((height, width), dtype=bool)

    num_pairs = random.randint(2, 4)
    palette = [color for color in range(10) if color != bg]
    colors = random.sample(palette, num_pairs)
    orientations = ["horizontal", "vertical", "diagonal"]
    random.shuffle(orientations)

    for idx, color in enumerate(colors):
        mask_a, mask_b = _random_pair_masks()
        orientation = orientations[idx % len(orientations)] if idx < len(orientations) else random.choice(orientations)
        if not _attempt_place_pair(grid, input_occ, output_occ, color, mask_a, mask_b, orientation):
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    if len(find_components(out, bg=get_bg(out), connectivity=4)) != num_pairs:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
