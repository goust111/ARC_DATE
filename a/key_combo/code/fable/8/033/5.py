import numpy as np


BG = 0
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
NON_BG_COLORS = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=10)
    return int(counts.argmax())


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int).copy()


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    mask = np.asarray(grid == color, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _is_solid_2x2(grid: np.ndarray, coords: np.ndarray, color: int) -> bool:
    if len(coords) != 4:
        return False
    r0, c0, r1, c1 = bbox_coords(coords)
    if (r1 - r0 + 1, c1 - c0 + 1) != (2, 2):
        return False
    return bool(np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color))


def _detect_marker_pairs(grid: np.ndarray) -> dict[int, dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    markers: dict[int, dict[str, object]] = {}
    for color in sorted(int(v) for v in np.unique(g) if int(v) != bg):
        comps = _components_of_color(g, color)
        block_bboxes: list[tuple[int, int, int, int]] = []
        extra_sizes: list[int] = []
        for comp in comps:
            if _is_solid_2x2(g, comp, color):
                block_bboxes.append(bbox_coords(comp))
            else:
                extra_sizes.append(len(comp))
        if len(block_bboxes) != 2:
            continue
        if extra_sizes and extra_sizes != [1]:
            continue
        block_bboxes.sort(key=lambda b: (b[1], b[0]))
        left, right = block_bboxes
        if left[0] != right[0] or left[2] != right[2]:
            continue
        if left[3] + 1 >= right[1]:
            continue
        markers[color] = {"left": left, "right": right}
    return markers


def _analyze_instance(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    markers = _detect_marker_pairs(g)
    assert len(markers) >= 2, "expected at least two marker pairs"

    marker_colors = set(markers)
    stripes: dict[int, np.ndarray] = {}
    next_color: dict[int, int | None] = {}
    indegree = {color: 0 for color in markers}
    row_order: list[int] = []
    reserved = np.zeros_like(g, dtype=bool)
    widths = set()
    total_pointer_pixels = 0

    for color, info in markers.items():
        left = info["left"]
        right = info["right"]
        assert isinstance(left, tuple) and isinstance(right, tuple)
        stripe = crop_bbox(g, (left[0], left[3] + 1, left[2], right[1] - 1))
        assert stripe.shape[0] == 2 and stripe.shape[1] > 0
        widths.add(int(stripe.shape[1]))
        stripes[color] = stripe
        row_order.append(color)

        reserved[left[0] : left[2] + 1, left[1] : left[3] + 1] = True
        reserved[right[0] : right[2] + 1, right[1] : right[3] + 1] = True
        reserved[left[0] : left[2] + 1, left[3] + 1 : right[1]] = True

        ptr_hits: list[int] = []
        for marker_color in marker_colors:
            hit_count = int((stripe == marker_color).sum())
            ptr_hits.extend([marker_color] * hit_count)
        assert len(ptr_hits) <= 1, "each stripe may contain at most one marker-color pixel"
        if ptr_hits:
            target = ptr_hits[0]
            assert target != color, "pointer must target another marker pair"
            next_color[color] = target
            indegree[target] += 1
            total_pointer_pixels += 1
        else:
            next_color[color] = None

    assert len(widths) == 1, "all extracted stripes must share a common width"
    assert total_pointer_pixels == len(markers) - 1, "a chain of N stripes must contain N-1 pointers"
    assert all(v <= 1 for v in indegree.values()), "pointer graph must be a chain, not a fan-in"

    starts = [color for color, deg in indegree.items() if deg == 0]
    tails = [color for color, nxt in next_color.items() if nxt is None]
    assert len(starts) == 1, "expected a unique unpointed chain start"
    assert len(tails) == 1, "expected a unique chain tail"

    order: list[int] = []
    seen: set[int] = set()
    cur: int | None = starts[0]
    while cur is not None:
        assert cur not in seen, "pointer chain must be acyclic"
        seen.add(cur)
        order.append(cur)
        cur = next_color[cur]
    assert len(order) == len(markers), "pointer chain must visit every stripe exactly once"

    row_order.sort(key=lambda color: markers[color]["left"][0])  # type: ignore[index]
    return {
        "markers": markers,
        "marker_colors": marker_colors,
        "stripes": stripes,
        "next_color": next_color,
        "order": order,
        "row_order": row_order,
        "reserved": reserved,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the content between paired 2x2 markers and stack the strips by their hidden pointer chain.

    Input structure:
    - The grid contains background noise plus several marker colors, each forming exactly two solid 2x2 blocks on the same two rows.
    - For each marker color, the rectangle strictly between its left/right blocks is a 2-row content strip.
    - Noise never uses marker colors; each non-tail strip contains exactly one pixel whose color is another marker color, while the tail contains none.

    Rule:
    - Identify every row-aligned same-color pair of 2x2 marker blocks and crop the 2-row rectangle between them.
    - Interpret the unique marker-colored pixel inside a strip as a pointer to the strip owned by that marker color.
    - Start from the unique strip with indegree zero, follow the single pointer chain to the tail, and vertically stack the cropped strips in that chain order.
    - Keep all strip pixels unchanged, including pointer pixels, and discard the marker blocks plus all external noise.

    Key insight:
    - Spatial order is a distractor: the true stacking order is encoded only by the marker-color pointer hidden inside each cropped strip.
    """
    info = _analyze_instance(grid)
    stripes = info["stripes"]
    order = info["order"]
    return np.vstack([stripes[color] for color in order])


def _random_stripe(rng: np.random.Generator, width: int, colors: list[int]) -> np.ndarray:
    stripe = np.full((2, width), BG, dtype=int)
    cells = [(r, c) for r in range(2) for c in range(width)]
    rng.shuffle(cells)
    fill_count = int(rng.integers(max(3, width), min(2 * width, width + 4) + 1))
    chosen_colors = list(rng.choice(colors, size=int(rng.integers(2, min(4, len(colors)) + 1)), replace=False))
    for idx in range(fill_count):
        r, c = cells[idx]
        stripe[r, c] = int(rng.choice(chosen_colors))
    return stripe


def _pick_row_order(rng: np.random.Generator, n: int, chain_order: list[int]) -> list[int]:
    reverse_chain = list(reversed(chain_order))
    while True:
        row_order = list(map(int, rng.permutation(n)))
        if row_order != chain_order and row_order != reverse_chain:
            return row_order


def _stack_for_ids(stripes: dict[int, np.ndarray], order: list[int]) -> np.ndarray:
    return np.vstack([stripes[idx] for idx in order])


def _add_noise(
    rng: np.random.Generator,
    grid: np.ndarray,
    reserved: np.ndarray,
    noise_colors: list[int],
) -> None:
    free = np.argwhere(~reserved & (grid == BG))
    if free.size == 0:
        return
    rng.shuffle(free)
    count = min(len(free), int(rng.integers(5, 13)))
    used: set[tuple[int, int]] = set()
    for r, c in free:
        rr, cc = int(r), int(c)
        if len(used) >= count:
            break
        if any(max(abs(rr - ur), abs(cc - uc)) <= 1 for ur, uc in used):
            continue
        grid[rr, cc] = int(rng.choice(noise_colors))
        used.add((rr, cc))


def _build_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n = int(rng.integers(3, 6))
    width = int(rng.integers(4, 9))
    marker_colors = list(map(int, rng.choice(NON_BG_COLORS, size=n, replace=False)))
    filler_colors = [c for c in NON_BG_COLORS if c not in marker_colors]
    if len(filler_colors) < 3:
        return None

    chain_order = list(map(int, rng.permutation(n)))
    row_order = _pick_row_order(rng, n, chain_order)

    top_margin = int(rng.integers(1, 4))
    bottom_margin = int(rng.integers(1, 4))
    gaps = list(map(int, rng.integers(1, 4, size=n - 1)))
    height = top_margin + 2 * n + sum(gaps) + bottom_margin
    object_width = width + 4
    extra_width = int(rng.integers(4, 10))
    grid_width = object_width + extra_width
    if height > 30 or grid_width > 30:
        return None

    grid = np.full((height, grid_width), BG, dtype=int)
    stripes: dict[int, np.ndarray] = {}
    reserved = np.zeros_like(grid, dtype=bool)

    next_id = {chain_order[i]: chain_order[i + 1] for i in range(n - 1)}
    tail_id = chain_order[-1]
    stripe_keys: set[bytes] = set()

    row_starts: dict[int, int] = {}
    r = top_margin
    for pos, stripe_id in enumerate(row_order):
        row_starts[stripe_id] = r
        r += 2
        if pos < n - 1:
            r += gaps[pos]

    max_left = grid_width - object_width - 1
    if max_left < 1:
        return None

    for stripe_id in range(n):
        stripe = _random_stripe(rng, width, filler_colors)
        if stripe_id != tail_id:
            target_id = next_id[stripe_id]
            pr = int(rng.integers(0, 2))
            pc = int(rng.integers(0, width))
            stripe[pr, pc] = marker_colors[target_id]
        key = stripe.tobytes()
        if key in stripe_keys:
            return None
        stripe_keys.add(key)
        stripes[stripe_id] = stripe

    for stripe_id in row_order:
        row = row_starts[stripe_id]
        left_col = int(rng.integers(1, max_left + 1))
        color = marker_colors[stripe_id]
        grid[row : row + 2, left_col : left_col + 2] = color
        grid[row : row + 2, left_col + 2 : left_col + 2 + width] = stripes[stripe_id]
        grid[row : row + 2, left_col + 2 + width : left_col + 4 + width] = color
        reserved[row : row + 2, left_col : left_col + object_width] = True

    _add_noise(rng, grid, reserved, filler_colors)

    info = _analyze_instance(grid)
    inferred_order = info["order"]
    expected_color_order = [marker_colors[idx] for idx in chain_order]
    if inferred_order != expected_color_order:
        return None

    inferred_row_order = info["row_order"]
    expected_row_colors = [marker_colors[idx] for idx in row_order]
    if inferred_row_order != expected_row_colors:
        return None

    if int(np.count_nonzero(grid[~reserved])) < 5:
        return None

    chain_out = main(grid)
    row_out = np.vstack([info["stripes"][color] for color in inferred_row_order])
    rev_row_out = np.vstack([info["stripes"][color] for color in reversed(inferred_row_order)])
    if np.array_equal(chain_out, row_out) or np.array_equal(chain_out, rev_row_out):
        return None

    if chain_out.shape[0] > 30 or chain_out.shape[1] > 30:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid pointer-chain stacking input")
