import random

import numpy as np


BLACK = 0
MASTER_SIZE = 30
GEN_RETRIES = 400
PLACE_GAP = 1

SEED_TEMPLATES = (
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 1], [1, 0], [1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=bool),
    np.array([[1, 0, 1], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 1]], dtype=bool),
)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _crop_bool(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.where(mask)
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1].copy()


def _rot_cw(arr: np.ndarray, turns: int) -> np.ndarray:
    return np.rot90(arr, -turns % 4)


def _mono_components(grid: np.ndarray, bg: int = BLACK) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coords = np.asarray(cells, dtype=int)
            comps.append({"color": color, "coords": coords})
    return comps


def _frame_from_component(comp: dict) -> dict | None:
    coords = comp["coords"]
    color = int(comp["color"])
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None
    if int(coords.shape[0]) != 2 * h + 2 * w - 4:
        return None

    patch = np.zeros((h, w), dtype=bool)
    patch[coords[:, 0] - r0, coords[:, 1] - c0] = True
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    if not np.array_equal(patch, border):
        return None

    return {
        "color": color,
        "bbox": (r0, c0, r1, c1),
        "height": h,
        "width": w,
    }


def _contains(outer: dict, inner: dict) -> bool:
    or0, oc0, or1, oc1 = outer["bbox"]
    ir0, ic0, ir1, ic1 = inner["bbox"]
    return or0 < ir0 and oc0 < ic0 and or1 > ir1 and oc1 > ic1


def _interior_seed(grid: np.ndarray, frame: dict) -> dict | None:
    r0, c0, r1, c1 = frame["bbox"]
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    comps = _mono_components(interior, bg=BLACK)
    if not comps:
        return None
    if len(comps) != 1:
        raise AssertionError("leaf container should contain at most one seed object")
    comp = comps[0]
    coords = comp["coords"].copy()
    coords[:, 0] += r0 + 1
    coords[:, 1] += c0 + 1
    return {"color": int(comp["color"]), "coords": coords}


def _seed_patch(seed: dict) -> np.ndarray:
    coords = seed["coords"]
    color = int(seed["color"])
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    patch[coords[:, 0] - r0, coords[:, 1] - c0] = color
    return patch


def _center_top_left(frame: dict, patch_shape: tuple[int, int]) -> tuple[int, int]:
    inner_h = int(frame["height"] - 2)
    inner_w = int(frame["width"] - 2)
    ph, pw = patch_shape
    r0, c0, _r1, _c1 = frame["bbox"]
    return r0 + 1 + (inner_h - ph) // 2, c0 + 1 + (inner_w - pw) // 2


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each seed object by a number of quarter-turns determined by its container nesting depth, then center it back inside the innermost container.

    Input structure:
    - The background is black.
    - Several one-pixel-thick hollow rectangular containers appear on the grid; containers may be disjoint or strictly nested, but they never partially overlap.
    - Only innermost containers may contain a single monochrome 4-connected seed object, and that seed lies strictly inside the frame.
    - Containers without a seed either contain smaller child containers or have an all-black interior.

    Rule:
    - Detect every rectangular frame and compute each frame's nesting depth as the number of containing frames including itself.
    - For each leaf container that directly contains a seed object, crop the seed object's tight bounding box and rotate it clockwise by `depth * 90` degrees.
    - Erase the original seed pixels and redraw the rotated seed, keeping its color, centered inside that container's interior region.
    - Leave all container frame pixels and all empty containers unchanged.

    Key insight:
    - The visible rotation angle is not encoded by the seed itself; it is encoded by how many rectangular containers wrap around that seed's innermost frame.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    frames = []
    for comp in _mono_components(g, bg=BLACK):
        frame = _frame_from_component(comp)
        if frame is not None:
            frames.append(frame)

    for frame in frames:
        frame["depth"] = 1 + sum(_contains(other, frame) for other in frames if other is not frame)
        frame["has_child"] = any(_contains(frame, other) for other in frames if other is not frame)

    for frame in frames:
        if frame["has_child"]:
            continue
        seed = _interior_seed(g, frame)
        if seed is None:
            continue
        for r, c in seed["coords"]:
            out[int(r), int(c)] = BLACK

        patch = _seed_patch(seed)
        rot = _rot_cw(patch, int(frame["depth"]) % 4)
        top, left = _center_top_left(frame, rot.shape)
        target = out[top : top + rot.shape[0], left : left + rot.shape[1]]
        target[rot != BLACK] = rot[rot != BLACK]

    return out


def _draw_frame(arr: np.ndarray, color: int) -> None:
    arr[0, :] = color
    arr[-1, :] = color
    arr[:, 0] = color
    arr[:, -1] = color


def _choose_seed_mask(depth: int) -> np.ndarray:
    turns = depth % 4
    for _ in range(100):
        mask = _rot_cw(random.choice(SEED_TEMPLATES), random.randrange(4)).astype(bool)
        if turns and np.array_equal(_crop_bool(mask), _crop_bool(_rot_cw(mask, turns))):
            continue
        return mask
    return _rot_cw(SEED_TEMPLATES[0], random.randrange(4)).astype(bool)


def _candidate_interior_sizes(mask: np.ndarray, depth: int) -> list[tuple[int, int]]:
    turns = depth % 4
    rot_mask = _rot_cw(mask, turns)
    ph, pw = mask.shape
    rh, rw = rot_mask.shape
    candidates: list[tuple[int, int]] = []
    for ih in range(max(ph, rh) + 2, 9):
        for iw in range(max(pw, rw) + 2, 9):
            if (ih - rh) % 2 != 0 or (iw - rw) % 2 != 0:
                continue
            if ih - ph < 2 or iw - pw < 2:
                continue
            candidates.append((ih, iw))
    return candidates


def _leaf_positions(interior_h: int, interior_w: int, patch_h: int, patch_w: int) -> list[tuple[int, int]]:
    return [
        (r, c)
        for r in range(1, interior_h - patch_h)
        for c in range(1, interior_w - patch_w)
    ]


def _build_seed_chain(depth: int) -> tuple[np.ndarray, np.ndarray] | None:
    if depth < 1:
        return None

    mask = _choose_seed_mask(depth)
    candidates = _candidate_interior_sizes(mask, depth)
    if not candidates:
        return None
    interior_h, interior_w = random.choice(candidates)
    leaf_h = interior_h + 2
    leaf_w = interior_w + 2

    colors = random.sample(range(1, 10), depth + 1)
    frame_colors = colors[:depth]
    seed_color = colors[-1]

    patch_in = np.zeros((leaf_h, leaf_w), dtype=int)
    patch_out = np.zeros((leaf_h, leaf_w), dtype=int)
    _draw_frame(patch_in, frame_colors[-1])
    _draw_frame(patch_out, frame_colors[-1])

    ph, pw = mask.shape
    centered_in = ((interior_h - ph) // 2, (interior_w - pw) // 2)
    positions = _leaf_positions(interior_h, interior_w, ph, pw)
    if not positions:
        return None
    non_center = [pos for pos in positions if pos != centered_in]
    if non_center:
        positions = non_center
    seed_top, seed_left = random.choice(positions)
    patch_in[1 + seed_top : 1 + seed_top + ph, 1 + seed_left : 1 + seed_left + pw][mask] = seed_color

    rot_mask = _rot_cw(mask, depth % 4)
    rh, rw = rot_mask.shape
    out_top = (interior_h - rh) // 2
    out_left = (interior_w - rw) // 2
    patch_out[1 + out_top : 1 + out_top + rh, 1 + out_left : 1 + out_left + rw][rot_mask] = seed_color

    child_in = patch_in
    child_out = patch_out
    for level in range(depth - 2, -1, -1):
        gap_top = random.randint(1, 2)
        gap_bottom = random.randint(1, 2)
        gap_left = random.randint(1, 2)
        gap_right = random.randint(1, 2)
        parent_h = child_in.shape[0] + gap_top + gap_bottom + 2
        parent_w = child_in.shape[1] + gap_left + gap_right + 2
        parent_in = np.zeros((parent_h, parent_w), dtype=int)
        parent_out = np.zeros((parent_h, parent_w), dtype=int)
        _draw_frame(parent_in, frame_colors[level])
        _draw_frame(parent_out, frame_colors[level])
        rr = 1 + gap_top
        cc = 1 + gap_left
        parent_in[rr : rr + child_in.shape[0], cc : cc + child_in.shape[1]] = child_in
        parent_out[rr : rr + child_out.shape[0], cc : cc + child_out.shape[1]] = child_out
        child_in = parent_in
        child_out = parent_out

    return child_in, child_out


def _build_empty_frame() -> tuple[np.ndarray, np.ndarray]:
    h = random.randint(5, 10)
    w = random.randint(5, 10)
    color = random.randint(1, 9)
    patch = np.zeros((h, w), dtype=int)
    _draw_frame(patch, color)
    return patch, patch.copy()


def _paste(master: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    master[top : top + patch.shape[0], left : left + patch.shape[1]] = patch


def _place_patches(pairs: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray] | None:
    inp = np.zeros((MASTER_SIZE, MASTER_SIZE), dtype=int)
    out = np.zeros((MASTER_SIZE, MASTER_SIZE), dtype=int)
    blocked = np.zeros((MASTER_SIZE, MASTER_SIZE), dtype=bool)

    ordered = sorted(pairs, key=lambda pair: pair[0].shape[0] * pair[0].shape[1], reverse=True)
    for patch_in, patch_out in ordered:
        h, w = patch_in.shape
        positions = []
        for r in range(MASTER_SIZE - h + 1):
            for c in range(MASTER_SIZE - w + 1):
                rr0 = max(0, r - PLACE_GAP)
                cc0 = max(0, c - PLACE_GAP)
                rr1 = min(MASTER_SIZE, r + h + PLACE_GAP)
                cc1 = min(MASTER_SIZE, c + w + PLACE_GAP)
                if not blocked[rr0:rr1, cc0:cc1].any():
                    positions.append((r, c, rr0, cc0, rr1, cc1))
        if not positions:
            return None
        r, c, rr0, cc0, rr1, cc1 = random.choice(positions)
        _paste(inp, patch_in, r, c)
        _paste(out, patch_out, r, c)
        blocked[rr0:rr1, cc0:cc1] = True

    return inp, out


def _crop_pair(inp: np.ndarray, out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    occupied = (inp != BLACK) | (out != BLACK)
    rows, cols = np.where(occupied)
    r0 = int(rows.min())
    c0 = int(cols.min())
    r1 = int(rows.max())
    c1 = int(cols.max())

    extra_top = random.randint(0, min(3, r0))
    extra_left = random.randint(0, min(3, c0))
    extra_bottom = random.randint(0, min(3, MASTER_SIZE - 1 - r1))
    extra_right = random.randint(0, min(3, MASTER_SIZE - 1 - c1))

    rr0 = r0 - extra_top
    cc0 = c0 - extra_left
    rr1 = r1 + extra_bottom + 1
    cc1 = c1 + extra_right + 1
    return inp[rr0:rr1, cc0:cc1].copy(), out[rr0:rr1, cc0:cc1].copy()


def _gen_attempt() -> np.ndarray | None:
    primary_depth = random.choice((1, 2, 3, 4))
    seed_depths = [primary_depth]

    if primary_depth <= 2 and random.random() < 0.75:
        others = [d for d in (1, 2, 3, 4) if d != primary_depth]
        seed_depths.append(random.choice(others))
    elif primary_depth >= 3 and random.random() < 0.4:
        seed_depths.append(random.choice((1, 2)))

    if len(seed_depths) == 2 and random.random() < 0.35:
        more = [d for d in (1, 2, 3, 4) if d not in seed_depths]
        if more:
            seed_depths.append(random.choice(more))

    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for depth in seed_depths:
        pair = _build_seed_chain(depth)
        if pair is None:
            return None
        pairs.append(pair)

    empty_count = random.choice((0, 0, 1, 1, 2 if primary_depth <= 2 else 0))
    for _ in range(empty_count):
        pairs.append(_build_empty_frame())

    placed = _place_patches(pairs)
    if placed is None:
        return None
    full_in, full_out = placed
    inp, expected = _crop_pair(full_in, full_out)

    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return None
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-container rotation task")
