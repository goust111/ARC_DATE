import numpy as np


GREY = 5
MAX_ATTEMPTS = 400
PALETTE = tuple(range(10))
NEIGHBORS8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS8:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) != bg:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append({"coords": coords, "bbox": _bbox_from_coords(coords)})
    return components


def _extract_box(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int).copy()


def _component_mask(box: np.ndarray, bg: int) -> np.ndarray:
    return np.asarray(box != bg, dtype=bool)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair two mirrored sprites by using the other sprite's visible colors to disambiguate grey pixels.

    Input structure:
    - The background is a single dominant color.
    - Exactly two non-background multicolor sprites appear, and their foreground masks are left-right mirrors of each other.
    - Grey pixels belong inside the sprites: some are true sprite color, while others are grey occlusion patches.

    Rule:
    - Find the two sprite components and align them by their own bounding boxes.
    - Compare each sprite cell with the horizontally mirrored cell in the other sprite.
    - If one side is grey and the opposite side is a non-grey color, the grey cell is a patch and must be recolored to that opposite color.
    - If both corresponding cells are grey, keep them grey because that pair is the sprite's native color. Preserve background and all non-grey visible pixels.

    Key insight:
    - Grey has two meanings, so a single sprite is insufficient; only the cross-sprite mirror correspondence reveals whether each grey cell is an occlusion or the real color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    components = _find_components(g, bg)
    assert len(components) == 2, "expected exactly two non-background sprites"

    comp_a, comp_b = components
    box_a = _extract_box(g, comp_a["bbox"])
    box_b = _extract_box(g, comp_b["bbox"])
    assert box_a.shape == box_b.shape, "mirrored sprites must share one bbox shape"

    mask_a = _component_mask(box_a, bg)
    mask_b = _component_mask(box_b, bg)
    assert np.array_equal(np.fliplr(mask_a), mask_b), "sprite masks must be horizontal mirrors"

    repaired_a = box_a.copy()
    repaired_b = box_b.copy()
    h, w = box_a.shape
    for r in range(h):
        for c in range(w):
            if not mask_a[r, c]:
                assert not mask_b[r, w - 1 - c]
                continue

            cc = w - 1 - c
            assert mask_b[r, cc]
            col_a = int(box_a[r, c])
            col_b = int(box_b[r, cc])

            if col_a == GREY and col_b != GREY:
                repaired_a[r, c] = col_b
            elif col_b == GREY and col_a != GREY:
                repaired_b[r, cc] = col_a
            elif col_a != GREY and col_b != GREY:
                assert col_a == col_b, "visible mirrored colors must agree"

    out = g.copy()
    ar0, ac0, ar1, ac1 = comp_a["bbox"]
    br0, bc0, br1, bc1 = comp_b["bbox"]
    out[ar0 : ar1 + 1, ac0 : ac1 + 1] = repaired_a
    out[br0 : br1 + 1, bc0 : bc1 + 1] = repaired_b
    return out


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    h, w = mask.shape
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in NEIGHBORS8:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _random_connected_mask(rng: np.random.Generator, h: int, w: int) -> np.ndarray | None:
    area_min = max(7, (h * w) // 3)
    area_max = min(h * w - 1, int(round(h * w * 0.72)))
    if area_min > area_max:
        return None

    for _ in range(120):
        target = int(rng.integers(area_min, area_max + 1))
        mask = np.zeros((h, w), dtype=bool)
        start = (int(rng.integers(0, h)), int(rng.integers(0, w)))
        mask[start] = True

        while int(mask.sum()) < target:
            frontier: set[tuple[int, int]] = set()
            filled = np.argwhere(mask)
            for r, c in filled:
                rr = int(r)
                cc = int(c)
                for dr, dc in NEIGHBORS8:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                        frontier.add((nr, nc))
            if not frontier:
                break
            nr, nc = list(frontier)[int(rng.integers(0, len(frontier)))]
            mask[nr, nc] = True

        if int(mask.sum()) != target:
            continue
        if not _is_connected(mask):
            continue
        if np.array_equal(mask, np.fliplr(mask)):
            continue
        rows = np.flatnonzero(mask.any(axis=1))
        cols = np.flatnonzero(mask.any(axis=0))
        if len(rows) == 0 or len(cols) == 0:
            continue
        if rows[0] != 0 or rows[-1] != h - 1 or cols[0] != 0 or cols[-1] != w - 1:
            continue
        return mask
    return None


def _colorize_sprite(
    rng: np.random.Generator, mask: np.ndarray, bg: int
) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]] | None:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    area = len(coords)
    if area < 7:
        return None

    non_grey_pool = [c for c in PALETTE if c not in {bg, GREY}]
    color_count = int(rng.integers(2, min(5, len(non_grey_pool)) + 1))
    chosen_colors = list(map(int, rng.choice(non_grey_pool, size=color_count, replace=False)))

    max_native = min(3, area - 2)
    native_grey_count = int(rng.integers(1, max_native + 1))
    native_indices = set(map(int, rng.choice(area, size=native_grey_count, replace=False)))

    sprite = np.full(mask.shape, bg, dtype=int)
    remaining = [idx for idx in range(area) if idx not in native_indices]
    rng.shuffle(remaining)
    needed = remaining[: len(chosen_colors)]
    assignments: dict[int, int] = {}
    for idx, color in zip(needed, chosen_colors):
        assignments[idx] = color
    for idx in remaining[len(chosen_colors) :]:
        assignments[idx] = int(chosen_colors[int(rng.integers(0, len(chosen_colors)))])

    native_coords: list[tuple[int, int]] = []
    patchable_coords: list[tuple[int, int]] = []
    for idx, (r, c) in enumerate(coords):
        if idx in native_indices:
            sprite[r, c] = GREY
            native_coords.append((r, c))
        else:
            sprite[r, c] = assignments[idx]
            patchable_coords.append((r, c))

    if len(patchable_coords) < 2:
        return None
    if np.array_equal(sprite, np.fliplr(sprite)):
        return None
    return sprite, native_coords, patchable_coords


def _place_pair(
    rng: np.random.Generator, left: np.ndarray, right: np.ndarray, bg: int
) -> tuple[np.ndarray, tuple[int, int, int, int], tuple[int, int, int, int]] | None:
    h, w = left.shape
    min_gap = int(rng.integers(3, 8))
    min_h = max(12, h + 4)
    max_h = min(24, h + 10)
    min_w = 2 * w + min_gap + 4
    max_w = min(30, 2 * w + min_gap + 12)
    if min_h > max_h or min_w > max_w:
        return None

    grid_h = int(rng.integers(min_h, max_h + 1))
    grid_w = int(rng.integers(min_w, max_w + 1))
    max_left_c = grid_w - (2 * w + min_gap) - 1
    if max_left_c < 1:
        return None

    left_r = int(rng.integers(1, grid_h - h))
    right_r = int(rng.integers(1, grid_h - h))
    left_c = int(rng.integers(1, max_left_c + 1))
    min_right_c = left_c + w + min_gap
    max_right_c = grid_w - w - 1
    if min_right_c > max_right_c:
        return None
    right_c = int(rng.integers(min_right_c, max_right_c + 1))

    grid = np.full((grid_h, grid_w), bg, dtype=int)
    grid[left_r : left_r + h, left_c : left_c + w] = left
    grid[right_r : right_r + h, right_c : right_c + w] = right
    left_bbox = (left_r, left_c, left_r + h - 1, left_c + w - 1)
    right_bbox = (right_r, right_c, right_r + h - 1, right_c + w - 1)
    return grid, left_bbox, right_bbox


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    bg = int(rng.choice([c for c in PALETTE if c != GREY]))
    sprite_h = int(rng.integers(4, 8))
    sprite_w = int(rng.integers(4, 8))

    mask = _random_connected_mask(rng, sprite_h, sprite_w)
    if mask is None:
        return None

    colored = _colorize_sprite(rng, mask, bg)
    if colored is None:
        return None
    left_full, native_grey_coords, patchable_coords = colored
    right_full = np.fliplr(left_full)

    if not native_grey_coords:
        return None
    if len(patchable_coords) < 2:
        return None

    patchable = patchable_coords[:]
    rng.shuffle(patchable)
    max_side_patches = max(1, min(len(patchable) // 2, max(2, len(patchable) // 4)))
    left_patch_count = int(rng.integers(1, max_side_patches + 1))
    right_patch_count = int(rng.integers(1, max_side_patches + 1))
    if left_patch_count + right_patch_count > len(patchable):
        return None

    left_patch_coords = patchable[:left_patch_count]
    right_patch_coords = patchable[left_patch_count : left_patch_count + right_patch_count]
    left_input = left_full.copy()
    right_input = right_full.copy()

    for r, c in left_patch_coords:
        left_input[r, c] = GREY
    for r, c in right_patch_coords:
        right_input[r, sprite_w - 1 - c] = GREY

    placed = _place_pair(rng, left_input, right_input, bg)
    if placed is None:
        return None
    inp, left_bbox, right_bbox = placed

    h, w = left_full.shape
    grid_h, grid_w = inp.shape

    output = np.full((grid_h, grid_w), bg, dtype=int)
    components = _find_components(inp, bg)
    if len(components) != 2:
        return None
    for comp in components:
        r0, c0, r1, c1 = comp["bbox"]
        if (r1 - r0 + 1, c1 - c0 + 1) != (h, w):
            return None
    repaired = main(inp)

    lr0, lc0, lr1, lc1 = left_bbox
    rr0, rc0, rr1, rc1 = right_bbox
    output[lr0 : lr1 + 1, lc0 : lc1 + 1] = left_full
    output[rr0 : rr1 + 1, rc0 : rc1 + 1] = right_full

    if not np.array_equal(repaired, output):
        return None
    if np.array_equal(inp, output):
        return None

    left_grey_pairs = 0
    right_grey_pairs = 0
    both_grey_pairs = 0
    for r in range(h):
        for c in range(w):
            if left_full[r, c] == bg:
                continue
            rc = w - 1 - c
            if left_input[r, c] == GREY and right_input[r, rc] != GREY:
                left_grey_pairs += 1
            elif right_input[r, rc] == GREY and left_input[r, c] != GREY:
                right_grey_pairs += 1
            elif left_input[r, c] == GREY and right_input[r, rc] == GREY:
                both_grey_pairs += 1
    if left_grey_pairs == 0 or right_grey_pairs == 0 or both_grey_pairs == 0:
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirrored-complement-repair input")
