import numpy as np
from collections import defaultdict, deque


BLACK = 0
COLOR_POOL = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int)
DIM_OPTIONS = ((4, 4), (4, 5), (4, 6), (5, 5), (5, 6), (6, 6))
PAIR_COUNT_CHOICES = (2, 3, 4)
GRID_SIDE_RANGE = (18, 31)
MASK_ATTEMPTS = 500
PLACEMENT_ATTEMPTS = 400
GENERATION_ATTEMPTS = 300
_RNG = np.random.default_rng()


def _neighbors(r: int, c: int, h: int, w: int, connectivity: int) -> list[tuple[int, int]]:
    if connectivity == 4:
        deltas = ((-1, 0), (1, 0), (0, -1), (0, 1))
    else:
        deltas = (
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        )
    out = []
    for dr, dc in deltas:
        rr = int(r + dr)
        cc = int(c + dc)
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _is_connected(mask: np.ndarray, value: bool = True, connectivity: int = 8) -> bool:
    coords = np.argwhere(mask == bool(value))
    if coords.size == 0:
        return False
    h, w = map(int, mask.shape)
    start = tuple(map(int, coords[0]))
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for rr, cc in _neighbors(int(r), int(c), h, w, connectivity):
            if bool(mask[rr, cc]) != bool(value):
                continue
            cell = (int(rr), int(cc))
            if cell in seen:
                continue
            seen.add(cell)
            q.append(cell)
    return len(seen) == int(coords.shape[0])


def _touches_all_sides(mask: np.ndarray, value: bool = True) -> bool:
    target = (mask == bool(value))
    return bool(target[0].any() and target[-1].any() and target[:, 0].any() and target[:, -1].any())


def _find_components(grid: np.ndarray, bg: int = BLACK, connectivity: int = 8) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == int(bg):
                continue
            color = int(g[r, c])
            q = deque([(int(r), int(c))])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for nr, nc in _neighbors(int(rr), int(cc), h, w, connectivity):
                    if seen[nr, nc] or int(g[nr, nc]) != int(color):
                        continue
                    seen[nr, nc] = True
                    q.append((int(nr), int(nc)))

            rows = [pt[0] for pt in coords]
            cols = [pt[1] for pt in coords]
            r0 = int(min(rows))
            r1 = int(max(rows))
            c0 = int(min(cols))
            c1 = int(max(cols))
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in coords:
                mask[int(rr - r0), int(cc - c0)] = True
            comps.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "mask": mask,
                    "area": int(mask.sum()),
                }
            )

    comps.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1], comp["bbox"][2], comp["bbox"][3]))
    return comps


def _mask_key(mask: np.ndarray) -> bytes:
    return np.asarray(mask, dtype=np.uint8).tobytes()


def _pair_components(components: list[dict]) -> list[dict]:
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for idx, comp in enumerate(components):
        groups[tuple(map(int, comp["mask"].shape))].append(int(idx))

    pairs = []
    for dims, indices in groups.items():
        by_key: dict[bytes, list[int]] = defaultdict(list)
        for idx in indices:
            by_key[_mask_key(components[idx]["mask"])].append(int(idx))

        used: set[int] = set()
        for idx in indices:
            if idx in used:
                continue
            comp = components[idx]
            need = _mask_key(~comp["mask"])
            candidates = [j for j in by_key.get(need, []) if j not in used and j != idx]
            if not candidates:
                raise ValueError("failed to find a complementary partner")
            j = int(candidates[0])
            used.add(int(idx))
            used.add(int(j))

            first = components[idx]
            second = components[j]
            h, w = map(int, dims)
            rect = np.full((h, w), int(second["color"]), dtype=int)
            rect[first["mask"]] = int(first["color"])
            pairs.append(
                {
                    "rect": rect,
                    "area": int(h * w),
                    "dims": (h, w),
                    "anchor": min((first["bbox"][0], first["bbox"][1]), (second["bbox"][0], second["bbox"][1])),
                }
            )

        if len(used) != len(indices):
            raise ValueError("some components were left unpaired")

    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair complementary fragments into solid rectangles, then reorder the rectangles by area.

    Input structure:
    - Black background with several scattered monochrome fragments.
    - Each fragment is a single connected component under 8-neighbor connectivity.
    - Two fragments belong together when their cropped bbox masks have the same size and one mask exactly fills the other's empty cells.

    Rule:
    - Extract every non-black component, crop it to its own minimal bounding box, and compare only the binary occupancy masks.
    - Pair equal-sized masks that are exact complements, then rebuild the full rectangle while keeping the two original fragment colors.
    - Sort the rebuilt rectangles by area from small to large and place them left-to-right with a one-column black gap on the smallest output canvas that fits them.

    Key insight:
    - Colors do not identify the partners; the true key is negative-space complement inside a shared bbox, and only after pairing does rectangle area determine the final order.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("grid must be 2D")

    # Equal-bbox complementary fragments are only realizable when diagonal contact can keep a
    # fragment whole, so components are intentionally read with 8-neighbor connectivity.
    components = _find_components(g, bg=BLACK, connectivity=8)
    if not components or len(components) % 2 != 0:
        raise ValueError("input must contain an even number of non-black components")

    pairs = _pair_components(components)
    pairs.sort(key=lambda item: (item["area"], item["dims"][0], item["dims"][1], item["anchor"]))

    out_h = int(max(pair["rect"].shape[0] for pair in pairs))
    out_w = int(sum(int(pair["rect"].shape[1]) for pair in pairs) + len(pairs) - 1)
    out = np.zeros((out_h, out_w), dtype=int)

    col = 0
    for pair in pairs:
        rect = pair["rect"]
        h, w = map(int, rect.shape)
        out[:h, col : col + w] = rect
        col += int(w + 1)

    return out


def _random_complementary_mask(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    total = int(h * w)
    min_area = max(4, min(h, w) - 1)
    for _ in range(MASK_ATTEMPTS):
        p = float(rng.uniform(0.35, 0.65))
        mask = rng.random((h, w)) < p
        area = int(mask.sum())
        if area == 0 or area == total or min(area, total - area) < min_area:
            continue
        if not _touches_all_sides(mask, True):
            continue
        if not _touches_all_sides(mask, False):
            continue
        if not _is_connected(mask, True, connectivity=8):
            continue
        if not _is_connected(mask, False, connectivity=8):
            continue
        return mask
    raise RuntimeError("failed to generate a valid complementary mask")


def _boxes_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = map(int, a)
    br0, bc0, br1, bc1 = map(int, b)
    separated = ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0
    return not separated


def _place_patches(patches: list[np.ndarray], rng: np.random.Generator) -> np.ndarray | None:
    max_h = max(int(patch.shape[0]) for patch in patches)
    max_w = max(int(patch.shape[1]) for patch in patches)
    min_h = max(18, max_h + 4)
    min_w = max(18, max_w + 10)
    h = int(rng.integers(min_h, GRID_SIDE_RANGE[1]))
    w = int(rng.integers(min_w, GRID_SIDE_RANGE[1]))

    grid = np.zeros((h, w), dtype=int)
    placed_boxes: list[tuple[int, int, int, int]] = []

    order = list(map(int, rng.permutation(len(patches))))
    for idx in order:
        patch = patches[idx]
        ph, pw = map(int, patch.shape)
        if ph + 2 > h or pw + 2 > w:
            return None
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            r0 = int(rng.integers(1, h - ph))
            c0 = int(rng.integers(1, w - pw))
            bbox = (r0, c0, r0 + ph - 1, c0 + pw - 1)
            if any(_boxes_too_close(bbox, other, gap=1) for other in placed_boxes):
                continue
            region = grid[r0 : r0 + ph, c0 : c0 + pw]
            if np.any((region != BLACK) & (patch != BLACK)):
                continue
            region[patch != BLACK] = patch[patch != BLACK]
            placed_boxes.append(bbox)
            placed = True
            break
        if not placed:
            return None

    return grid


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    pair_count = int(rng.choice(PAIR_COUNT_CHOICES))
    dim_indices = list(map(int, rng.choice(len(DIM_OPTIONS), size=pair_count, replace=False)))
    dims = [DIM_OPTIONS[idx] for idx in dim_indices]
    colors = list(map(int, rng.choice(COLOR_POOL, size=2 * pair_count, replace=False)))
    rng.shuffle(colors)

    patches = []
    for pair_idx, (h, w) in enumerate(dims):
        mask = _random_complementary_mask(int(h), int(w), rng)
        first_color = int(colors[2 * pair_idx])
        second_color = int(colors[2 * pair_idx + 1])
        first_mask = mask if bool(rng.integers(0, 2)) else ~mask
        second_mask = ~first_mask

        first = np.zeros((h, w), dtype=int)
        second = np.zeros((h, w), dtype=int)
        first[first_mask] = first_color
        second[second_mask] = second_color
        patches.extend([first, second])

    rng.shuffle(patches)
    grid = _place_patches(patches, rng)
    if grid is None:
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if grid.ndim != 2 or out.ndim != 2:
        return None
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return None
    if len(_find_components(grid, bg=BLACK, connectivity=8)) != 2 * pair_count:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(_RNG)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid complementary-rectangle input")
