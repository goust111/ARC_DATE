import random

import numpy as np


BG = 0
RED = 2
MARKER_COLORS = (3, 4, 7)
FRAME_COLOR_POOL = (1, 5, 6, 8, 9)
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": len(coords),
                }
            )
    return comps


def _is_hollow_frame(grid: np.ndarray, comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    color = int(comp["color"])
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    if int(comp["area"]) != 2 * h + 2 * w - 4:
        return False
    for r, c in comp["coords"]:
        if r not in (r0, r1) and c not in (c0, c1):
            return False
    if not np.all(grid[r0, c0 : c1 + 1] == color):
        return False
    if not np.all(grid[r1, c0 : c1 + 1] == color):
        return False
    if not np.all(grid[r0 : r1 + 1, c0] == color):
        return False
    if not np.all(grid[r0 : r1 + 1, c1] == color):
        return False
    return True


def _contains(outer: dict, inner: dict) -> bool:
    or0, oc0, or1, oc1 = outer["bbox"]
    ir0, ic0, ir1, ic1 = inner["bbox"]
    return or0 < ir0 and oc0 < ic0 and ir1 < or1 and ic1 < oc1


def _tight_crop(mask: np.ndarray) -> tuple[int, int, np.ndarray]:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    assert rows.size > 0 and cols.size > 0
    r0, r1 = int(rows[0]), int(rows[-1])
    c0, c1 = int(cols[0]), int(cols[-1])
    return r0, c0, mask[r0 : r1 + 1, c0 : c1 + 1]


def _transform_pattern(mask: np.ndarray, depth: int) -> np.ndarray:
    if depth % 2 == 0:
        return np.fliplr(mask)
    return np.rot90(mask, -1)


def _root_marker_color(grid: np.ndarray, frame: dict, bg: int) -> int:
    h, w = grid.shape
    r0, c0, _r1, _c1 = frame["bbox"]
    candidates = ((r0 - 1, c0 - 1), (r0 - 1, c0), (r0, c0 - 1))
    colors = []
    for rr, cc in candidates:
        if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) != bg:
            colors.append(int(grid[rr, cc]))
    assert len(colors) == 1
    return colors[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify nested hollow rectangular frames, transform each innermost red pattern
    according to its nesting-depth parity, and recolor it from the outermost marker.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _most_common_color(g)

    frames = [comp for comp in _components(g, bg) if _is_hollow_frame(g, comp)]
    assert frames

    parents = [-1] * len(frames)
    for i, frame in enumerate(frames):
        containers = [j for j, other in enumerate(frames) if i != j and _contains(other, frame)]
        if containers:
            parents[i] = min(
                containers,
                key=lambda j: (frames[j]["bbox"][2] - frames[j]["bbox"][0] + 1)
                * (frames[j]["bbox"][3] - frames[j]["bbox"][1] + 1),
            )

    children = [[] for _ in frames]
    for i, parent in enumerate(parents):
        if parent != -1:
            children[parent].append(i)

    depths = [0] * len(frames)
    for i in range(len(frames)):
        depth = 1
        cur = parents[i]
        while cur != -1:
            depth += 1
            cur = parents[cur]
        depths[i] = depth

    root_colors: dict[int, int] = {}
    for i, parent in enumerate(parents):
        if parent == -1:
            root_colors[i] = _root_marker_color(g, frames[i], bg)

    out = g.copy()
    for i, frame in enumerate(frames):
        if children[i]:
            continue

        root = i
        while parents[root] != -1:
            root = parents[root]
        target_color = root_colors[root]

        r0, c0, r1, c1 = frame["bbox"]
        interior_in = g[r0 + 1 : r1, c0 + 1 : c1]
        red_mask = interior_in == RED
        assert red_mask.any()

        local_r, local_c, patch = _tight_crop(red_mask)
        transformed = _transform_pattern(patch, depths[i])

        interior_out = out[r0 + 1 : r1, c0 + 1 : c1]
        interior_out[red_mask] = bg
        th, tw = transformed.shape
        assert local_r + th <= interior_out.shape[0]
        assert local_c + tw <= interior_out.shape[1]
        target = interior_out[local_r : local_r + th, local_c : local_c + tw]
        target[transformed] = target_color

    return out


# --------------- generator ---------------

WORK_SIZE = 30
PACK_PADDING = 1
GEN_ATTEMPTS = 400
DEPTH_OPTIONS = (
    (1, 2),
    (1, 4),
    (3, 2),
    (1, 2, 1),
    (1, 2, 2),
)
PATTERN_BOX_CHOICES = (2, 3, 4)


def _random_connected_pattern(size: int) -> np.ndarray | None:
    for _ in range(200):
        mask = np.zeros((size, size), dtype=bool)
        start = (random.randrange(size), random.randrange(size))
        mask[start] = True
        target = random.randint(max(3, size), size * size - 1)
        frontier = {start}

        while int(mask.sum()) < target and frontier:
            cell = random.choice(tuple(frontier))
            neighbors = []
            for dr, dc in DIR4:
                nr, nc = cell[0] + dr, cell[1] + dc
                if 0 <= nr < size and 0 <= nc < size and not mask[nr, nc]:
                    neighbors.append((nr, nc))
            if not neighbors:
                frontier.discard(cell)
                continue
            chosen = random.choice(neighbors)
            mask[chosen] = True
            frontier.add(chosen)
            frontier.add(cell)

        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        if rows.size == 0 or cols.size == 0:
            continue
        if int(rows[0]) != 0 or int(rows[-1]) != size - 1:
            continue
        if int(cols[0]) != 0 or int(cols[-1]) != size - 1:
            continue
        if np.array_equal(mask, np.fliplr(mask)):
            continue
        if np.array_equal(mask, np.rot90(mask, -1)):
            continue
        return mask
    return None


def _build_chain(depth: int, marker_color: int, frame_colors: list[int]) -> dict | None:
    inner_hi = 6 if depth <= 2 else 5
    leaf_inner_h = random.randint(4, inner_hi)
    leaf_inner_w = random.randint(4, inner_hi)
    leaf_h = leaf_inner_h + 2
    leaf_w = leaf_inner_w + 2

    boxes = [(0, 0, leaf_h, leaf_w)]
    cur_h, cur_w = leaf_h, leaf_w
    gap_hi = 2 if depth <= 2 else 1

    for _ in range(depth - 1):
        mt = random.randint(1, gap_hi)
        mb = random.randint(1, gap_hi)
        ml = random.randint(1, gap_hi)
        mr = random.randint(1, gap_hi)
        shift_r = 1 + mt
        shift_c = 1 + ml
        boxes = [(r + shift_r, c + shift_c, h, w) for r, c, h, w in boxes]
        cur_h = cur_h + mt + mb + 2
        cur_w = cur_w + ml + mr + 2
        boxes.insert(0, (0, 0, cur_h, cur_w))

    max_box = min(leaf_inner_h, leaf_inner_w, max(PATTERN_BOX_CHOICES))
    size_choices = [size for size in PATTERN_BOX_CHOICES if size <= max_box]
    if not size_choices:
        return None
    pattern_size = random.choice(size_choices)
    pattern_mask = _random_connected_pattern(pattern_size)
    if pattern_mask is None:
        return None

    leaf_r, leaf_c, _leaf_h, _leaf_w = boxes[-1]
    pat_r = random.randint(0, leaf_inner_h - pattern_size)
    pat_c = random.randint(0, leaf_inner_w - pattern_size)

    return {
        "depth": depth,
        "marker_color": marker_color,
        "boxes": boxes,
        "frame_colors": list(frame_colors),
        "pattern_mask": pattern_mask,
        "pattern_top": leaf_r + 1 + pat_r,
        "pattern_left": leaf_c + 1 + pat_c,
        "root_h": cur_h,
        "root_w": cur_w,
        "package_h": cur_h + 1,
        "package_w": cur_w + 1,
    }


def _rectangles_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], padding: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + padding < br0 or br1 + padding < ar0 or ac1 + padding < bc0 or bc1 + padding < ac0


def _pack_packages(chains: list[dict]) -> list[tuple[int, int]] | None:
    order = sorted(
        range(len(chains)),
        key=lambda i: chains[i]["package_h"] * chains[i]["package_w"],
        reverse=True,
    )
    placements: list[tuple[int, int] | None] = [None] * len(chains)
    placed_rects: list[tuple[int, int, int, int]] = []

    for idx in order:
        ph = chains[idx]["package_h"]
        pw = chains[idx]["package_w"]
        candidates = [(r, c) for r in range(WORK_SIZE - ph + 1) for c in range(WORK_SIZE - pw + 1)]
        random.shuffle(candidates)
        for r, c in candidates:
            rect = (r, c, r + ph - 1, c + pw - 1)
            if all(_rectangles_separated(rect, other, PACK_PADDING) for other in placed_rects):
                placements[idx] = (r, c)
                placed_rects.append(rect)
                break
        if placements[idx] is None:
            return None
    return [(int(r), int(c)) for r, c in placements]


def _draw_frame(grid: np.ndarray, top: int, left: int, height: int, width: int, color: int) -> None:
    bottom = top + height - 1
    right = left + width - 1
    grid[top, left : right + 1] = color
    grid[bottom, left : right + 1] = color
    grid[top : bottom + 1, left] = color
    grid[top : bottom + 1, right] = color


def _render_chain(inp: np.ndarray, expected: np.ndarray, chain: dict, slot_r: int, slot_c: int) -> None:
    inp[slot_r, slot_c] = chain["marker_color"]
    expected[slot_r, slot_c] = chain["marker_color"]

    base_r = slot_r + 1
    base_c = slot_c + 1
    for (rel_r, rel_c, height, width), color in zip(chain["boxes"], chain["frame_colors"]):
        _draw_frame(inp, base_r + rel_r, base_c + rel_c, height, width, color)
        _draw_frame(expected, base_r + rel_r, base_c + rel_c, height, width, color)

    pattern_mask = chain["pattern_mask"]
    pat_top = base_r + chain["pattern_top"]
    pat_left = base_c + chain["pattern_left"]
    pr, pc = np.where(pattern_mask)
    inp[pat_top + pr, pat_left + pc] = RED

    transformed = _transform_pattern(pattern_mask, chain["depth"])
    er, ec = np.where(transformed)
    expected[pat_top + er, pat_left + ec] = chain["marker_color"]


def _crop_pair(inp: np.ndarray, expected: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere(inp != BG)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    top = random.randint(0, min(int(r0), 3))
    left = random.randint(0, min(int(c0), 3))
    bottom = random.randint(0, min(int(inp.shape[0] - 1 - r1), 3))
    right = random.randint(0, min(int(inp.shape[1] - 1 - c1), 3))
    row_slice = slice(int(r0) - top, int(r1) + 1 + bottom)
    col_slice = slice(int(c0) - left, int(c1) + 1 + right)
    return inp[row_slice, col_slice], expected[row_slice, col_slice]


def _gen_attempt() -> np.ndarray | None:
    depths = list(random.choice(DEPTH_OPTIONS))
    random.shuffle(depths)

    marker_colors = random.sample(MARKER_COLORS, len(depths))
    frame_colors = random.sample(FRAME_COLOR_POOL, sum(depths))

    chains = []
    cursor = 0
    for depth, marker_color in zip(depths, marker_colors):
        colors = frame_colors[cursor : cursor + depth]
        cursor += depth
        chain = _build_chain(depth, marker_color, colors)
        if chain is None:
            return None
        chains.append(chain)

    placements = _pack_packages(chains)
    if placements is None:
        return None

    inp = np.full((WORK_SIZE, WORK_SIZE), BG, dtype=int)
    expected = np.full_like(inp, BG)
    for chain, (slot_r, slot_c) in zip(chains, placements):
        _render_chain(inp, expected, chain, slot_r, slot_c)

    inp, expected = _crop_pair(inp, expected)
    counts = np.bincount(inp.ravel(), minlength=10)
    if counts[BG] <= max(counts[c] for c in range(10) if c != BG):
        return None

    try:
        if not np.array_equal(main(inp), expected):
            return None
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_ATTEMPTS} attempts")
