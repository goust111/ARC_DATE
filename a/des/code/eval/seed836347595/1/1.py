import numpy as np
from collections import deque


BG = 0
FRAME_COLOR = 5
ANCHOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
RNG = np.random.default_rng()


def neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def components_from_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def bbox_of(coords: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def crop_coords_to_patch(
    grid: np.ndarray, coords: list[tuple[int, int]] | set[tuple[int, int]], bg: int
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = bbox_of(coords)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    mask = np.zeros_like(patch, dtype=bool)
    for r, c in coords:
        patch[r - r0, c - c0] = int(grid[r, c])
        mask[r - r0, c - c0] = True
    return patch, mask, (r0, c0, r1, c1)


def mask_key(mask: np.ndarray) -> tuple[tuple[int, int], tuple[tuple[int, int], ...]]:
    pts = tuple((int(r), int(c)) for r, c in np.argwhere(mask))
    return tuple(int(x) for x in mask.shape), pts


def rotate_patch_and_mask(patch: np.ndarray, mask: np.ndarray, turns_cw: int) -> tuple[np.ndarray, np.ndarray]:
    k = turns_cw % 4
    if k == 0:
        return patch.copy(), mask.copy()
    return np.rot90(patch, -k), np.rot90(mask, -k)


def frame_perimeter_coords(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    out = [(r0, c) for c in range(c0, c1 + 1)]
    out += [(r, c1) for r in range(r0 + 1, r1 + 1)]
    out += [(r1, c) for c in range(c1 - 1, c0 - 1, -1)]
    out += [(r, c0) for r in range(r1 - 1, r0, -1)]
    return out


def find_frame(grid: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int]]:
    best = None
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        for comp in components_from_mask(grid == color):
            r0, c0, r1, c1 = bbox_of(comp)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if h < 6 or w < 6:
                continue
            sub = grid[r0 : r1 + 1, c0 : c1 + 1]
            rows = {r for r, _ in comp}
            cols = {c for _, c in comp}
            if not ({r0, r1} <= rows and {c0, c1} <= cols):
                continue
            if not np.any(sub == bg):
                continue
            score = (h * w, len(comp))
            if best is None or score > best[0]:
                best = (score, color, (r0, c0, r1, c1))
    assert best is not None, "expected a unique frame component"
    return int(best[1]), best[2]


def trace_simple_color_path(grid: np.ndarray, color: int, frame_bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    pts = {tuple(int(x) for x in rc) for rc in np.argwhere(grid == color)}
    assert pts, "path color must exist"
    degrees = {}
    for p in pts:
        degrees[p] = sum((n in pts) for n in neighbors4(*p))
    endpoints = [p for p, deg in degrees.items() if deg == 1]
    assert len(endpoints) == 2, "anchor path must be a simple chain"

    perimeter = set(frame_perimeter_coords(frame_bbox))
    frame_ep = next((p for p in endpoints if p in perimeter), None)
    assert frame_ep is not None, "one endpoint must be on the frame border"
    piece_ep = next(p for p in endpoints if p != frame_ep)

    path = [piece_ep]
    prev = None
    cur = piece_ep
    while cur != frame_ep:
        nxts = [n for n in neighbors4(*cur) if n in pts and n != prev]
        assert len(nxts) == 1, "path must not branch"
        nxt = nxts[0]
        path.append(nxt)
        prev, cur = cur, nxt
    return path


def count_turns(path: list[tuple[int, int]]) -> int:
    if len(path) < 3:
        return 0
    dirs = []
    for (r0, c0), (r1, c1) in zip(path, path[1:]):
        dirs.append((r1 - r0, c1 - c0))
    turns = 0
    for d0, d1 in zip(dirs, dirs[1:]):
        if d0 != d1:
            turns += 1
    return turns


def find_slots(grid: np.ndarray, frame_bbox: tuple[int, int, int, int], bg: int) -> list[dict]:
    r0, c0, r1, c1 = frame_bbox
    slot_mask = np.zeros_like(grid, dtype=bool)
    slot_mask[r0 + 1 : r1, c0 + 1 : c1] = grid[r0 + 1 : r1, c0 + 1 : c1] == bg
    slots = []
    for comp in components_from_mask(slot_mask):
        patch, mask, bbox = crop_coords_to_patch(grid, comp, bg)
        slots.append({"coords": comp, "bbox": bbox, "mask": mask, "patch": patch})
    assert slots, "expected at least one slot"
    return slots


def extract_piece_for_anchor(
    grid: np.ndarray,
    piece_anchor: tuple[int, int],
    frame_bbox: tuple[int, int, int, int],
    bg: int,
    frame_color: int,
    anchor_colors: set[int],
) -> tuple[np.ndarray, np.ndarray]:
    h, w = grid.shape
    fr0, fc0, fr1, fc1 = frame_bbox
    outside = np.ones_like(grid, dtype=bool)
    outside[fr0 : fr1 + 1, fc0 : fc1 + 1] = False

    body_mask = outside & (grid != bg) & (grid != frame_color)
    for color in anchor_colors:
        body_mask &= grid != int(color)

    seeds = []
    for nr, nc in neighbors4(*piece_anchor):
        if 0 <= nr < h and 0 <= nc < w and body_mask[nr, nc]:
            seeds.append((nr, nc))
    assert seeds, "piece anchor must touch a non-anchor body"

    q = deque(seeds)
    seen = set(seeds)
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and body_mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))

    piece_coords = set(seen)
    piece_coords.add(piece_anchor)
    patch, mask, _ = crop_coords_to_patch(grid, piece_coords, bg)
    return patch, mask


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    frame_color, frame_bbox = find_frame(g, bg)
    slots = find_slots(g, frame_bbox, bg)

    slot_map = {}
    for slot in slots:
        key = mask_key(slot["mask"])
        assert key not in slot_map, "slot shapes must be unique"
        slot_map[key] = slot

    perimeter = frame_perimeter_coords(frame_bbox)
    anchor_colors = {
        int(g[r, c])
        for r, c in perimeter
        if int(g[r, c]) not in (bg, frame_color)
    }
    assert anchor_colors, "expected colored frame anchors"

    out = np.full_like(g, bg)
    fr0, fc0, fr1, fc1 = frame_bbox
    frame_slice = g[fr0 : fr1 + 1, fc0 : fc1 + 1]
    out_slice = out[fr0 : fr1 + 1, fc0 : fc1 + 1]
    out_slice[frame_slice != bg] = frame_slice[frame_slice != bg]

    used_slots: set[tuple[tuple[int, int], tuple[tuple[int, int], ...]]] = set()
    for color in sorted(anchor_colors):
        path = trace_simple_color_path(g, color, frame_bbox)
        piece_anchor = path[0]
        turns = count_turns(path)
        piece_patch, piece_mask = extract_piece_for_anchor(
            g, piece_anchor, frame_bbox, bg, frame_color, anchor_colors
        )
        rot_patch, rot_mask = rotate_patch_and_mask(piece_patch, piece_mask, turns)
        key = mask_key(rot_mask)
        assert key in slot_map, "rotated piece must match exactly one slot"
        assert key not in used_slots, "two pieces cannot target the same slot"
        used_slots.add(key)
        slot = slot_map[key]
        sr0, sc0, sr1, sc1 = slot["bbox"]
        sh = sr1 - sr0 + 1
        sw = sc1 - sc0 + 1
        assert rot_patch.shape == (sh, sw)
        dest = out[sr0 : sr1 + 1, sc0 : sc1 + 1]
        dest[rot_mask] = rot_patch[rot_mask]

    return out


def randint(lo: int, hi: int) -> int:
    return int(RNG.integers(lo, hi + 1))


def choice(seq):
    return seq[int(RNG.integers(len(seq)))]


def sample(seq, k: int):
    idx = RNG.choice(len(seq), size=k, replace=False)
    return [seq[int(i)] for i in idx]


def edge_offset(size: int) -> int:
    if size <= 2:
        return randint(0, size - 1)
    if RNG.random() < 0.7:
        return randint(1, size - 2)
    return choice((0, size - 1))


def build_piece_patch(h: int, w: int, anchor_color: int, anchor_side: str, fill_colors: list[int]) -> np.ndarray:
    patch = np.zeros((h, w), dtype=int)
    n = len(fill_colors)
    pattern_kind = choice(("rows", "cols", "checker", "diag", "blocks"))

    for r in range(h):
        for c in range(w):
            if pattern_kind == "rows":
                val = fill_colors[r % n]
            elif pattern_kind == "cols":
                val = fill_colors[c % n]
            elif pattern_kind == "checker":
                val = fill_colors[(r + c) % n]
            elif pattern_kind == "diag":
                val = fill_colors[(2 * r + c) % n]
            else:
                rr = 0 if r < h // 2 else 1
                cc = 0 if c < w // 2 else 1
                val = fill_colors[(rr + 2 * cc) % n]
            patch[r, c] = int(val)

    if len(set(int(v) for v in patch.ravel())) == 1:
        patch[0, 0] = int(fill_colors[-1])

    if anchor_side == "left":
        ar, ac = edge_offset(h), 0
    elif anchor_side == "right":
        ar, ac = edge_offset(h), w - 1
    elif anchor_side == "top":
        ar, ac = 0, edge_offset(w)
    else:
        ar, ac = h - 1, edge_offset(w)
    patch[ar, ac] = int(anchor_color)
    return patch


def anchor_pos_in_patch(patch: np.ndarray, anchor_color: int) -> tuple[int, int]:
    pts = np.argwhere(patch == anchor_color)
    assert len(pts) == 1, "piece must contain exactly one anchor pixel"
    r, c = pts[0]
    return int(r), int(c)


def partition_frame(frame_h: int, frame_w: int, target: int = 4, min_side: int = 2):
    base = (1, frame_h - 2, 1, frame_w - 2)
    for _ in range(300):
        regions = [base]
        walls = []
        while len(regions) < target:
            options = []
            for i, (r0, r1, c0, c1) in enumerate(regions):
                rh = r1 - r0 + 1
                rw = c1 - c0 + 1
                if rh >= 2 * min_side + 1:
                    options.append((i, "h"))
                if rw >= 2 * min_side + 1:
                    options.append((i, "v"))
            if not options:
                break
            idx, orient = choice(options)
            r0, r1, c0, c1 = regions.pop(idx)
            if orient == "h":
                cut = randint(r0 + min_side, r1 - min_side)
                walls.append(("h", cut, c0, c1))
                regions.append((r0, cut - 1, c0, c1))
                regions.append((cut + 1, r1, c0, c1))
            else:
                cut = randint(c0 + min_side, c1 - min_side)
                walls.append(("v", cut, r0, r1))
                regions.append((r0, r1, c0, cut - 1))
                regions.append((r0, r1, cut + 1, c1))
        if len(regions) != target:
            continue
        dims = [(r1 - r0 + 1, c1 - c0 + 1) for r0, r1, c0, c1 in regions]
        if any(h == w for h, w in dims):
            continue
        norm = {tuple(sorted((h, w))) for h, w in dims}
        if len(norm) != target:
            continue
        return regions, walls
    return None


def segment_points(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    if c0 == c1:
        step = 1 if r1 >= r0 else -1
        return [(r, c0) for r in range(r0, r1 + step, step)]
    raise ValueError("segment must be axis-aligned")


def polyline_points(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out = [points[0]]
    for a, b in zip(points, points[1:]):
        seg = segment_points(a, b)
        out.extend(seg[1:])
    return out


def place_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> bool:
    h, w = grid.shape
    ph, pw = patch.shape
    if top < 0 or left < 0 or top + ph > h or left + pw > w:
        return False
    view = grid[top : top + ph, left : left + pw]
    if np.any(view != BG):
        return False
    view[:, :] = patch
    return True


def draw_frame(grid: np.ndarray, fr0: int, fc0: int, frame_h: int, frame_w: int, walls) -> None:
    fr1 = fr0 + frame_h - 1
    fc1 = fc0 + frame_w - 1
    grid[fr0, fc0 : fc1 + 1] = FRAME_COLOR
    grid[fr1, fc0 : fc1 + 1] = FRAME_COLOR
    grid[fr0 : fr1 + 1, fc0] = FRAME_COLOR
    grid[fr0 : fr1 + 1, fc1] = FRAME_COLOR
    for orient, cut, a0, a1 in walls:
        if orient == "h":
            grid[fr0 + cut, fc0 + a0 : fc0 + a1 + 1] = FRAME_COLOR
        else:
            grid[fr0 + a0 : fr0 + a1 + 1, fc0 + cut] = FRAME_COLOR


def paint_path(
    grid: np.ndarray,
    color: int,
    points: list[tuple[int, int]],
    frame_bbox: tuple[int, int, int, int],
    frame_anchor: tuple[int, int],
) -> bool:
    fr0, fc0, fr1, fc1 = frame_bbox
    path = polyline_points(points)
    for i, (r, c) in enumerate(path):
        if not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]):
            return False
        if i == 0 or i == len(path) - 1:
            if grid[r, c] != color:
                return False
        else:
            if not (grid[r, c] == BG):
                return False
            if fr0 <= r <= fr1 and fc0 <= c <= fc1:
                return False
    for r, c in path[1:-1]:
        grid[r, c] = color

    ar, ac = frame_anchor
    if ar == fr0 and len(path) >= 2 and path[-2] != (fr0 - 1, ac):
        return False
    if ar == fr1 and len(path) >= 2 and path[-2] != (fr1 + 1, ac):
        return False
    if ac == fc0 and len(path) >= 2 and path[-2] != (ar, fc0 - 1):
        return False
    if ac == fc1 and len(path) >= 2 and path[-2] != (ar, fc1 + 1):
        return False
    return True


def gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    frame_h = randint(12, 17)
    frame_w = randint(12, 17)
    partition = partition_frame(frame_h, frame_w, target=4, min_side=2)
    if partition is None:
        return None
    regions, walls = partition
    regions = sample(regions, len(regions))

    anchor_colors = sample(list(ANCHOR_POOL), 4)
    fill_palette = [c for c in ANCHOR_POOL if c not in anchor_colors]
    if len(fill_palette) < 3:
        return None

    side_order = ("left", "top", "right", "bottom")
    anchor_side_for_piece = {"left": "right", "top": "right", "right": "left", "bottom": "left"}

    specs = []
    for turn, side, region, anchor_color in zip(range(4), side_order, regions, anchor_colors):
        r0, r1, c0, c1 = region
        slot_h = r1 - r0 + 1
        slot_w = c1 - c0 + 1
        piece_h, piece_w = (slot_h, slot_w) if turn % 2 == 0 else (slot_w, slot_h)
        n_fill = randint(2, min(3, len(fill_palette)))
        fill_colors = sample(fill_palette, n_fill)
        patch = build_piece_patch(piece_h, piece_w, anchor_color, anchor_side_for_piece[side], fill_colors)
        rotated = np.rot90(patch, -turn)
        specs.append(
            {
                "turn": turn,
                "side": side,
                "region": region,
                "input_patch": patch,
                "output_patch": rotated,
                "anchor_color": anchor_color,
            }
        )

    left_spec = next(s for s in specs if s["side"] == "left")
    top_spec = next(s for s in specs if s["side"] == "top")
    right_spec = next(s for s in specs if s["side"] == "right")
    bottom_spec = next(s for s in specs if s["side"] == "bottom")

    left_margin = max(left_spec["input_patch"].shape[1] + 4, top_spec["input_patch"].shape[1] + 4, 7) + randint(0, 2)
    top_margin = max(top_spec["input_patch"].shape[0] + 4, 7) + randint(0, 2)
    right_margin = max(right_spec["input_patch"].shape[1] + 5, bottom_spec["input_patch"].shape[1] + 5, 8) + randint(0, 2)
    bottom_margin = max(bottom_spec["input_patch"].shape[0] + 6, 9) + randint(0, 2)

    H = top_margin + frame_h + bottom_margin
    W = left_margin + frame_w + right_margin
    fr0, fc0 = top_margin, left_margin
    fr1, fc1 = fr0 + frame_h - 1, fc0 + frame_w - 1
    frame_bbox = (fr0, fc0, fr1, fc1)

    grid = np.full((H, W), BG, dtype=int)
    draw_frame(grid, fr0, fc0, frame_h, frame_w, walls)

    left_anchor = (fr0 + randint(2, frame_h - 3), fc0)
    right_anchor = (fr0 + randint(2, frame_h - 3), fc1)
    top_anchor = (fr0, fc0 + randint(2, frame_w - 3))
    bottom_anchor = (fr1, fc0 + randint(2, frame_w - 3))

    if top_anchor[1] == bottom_anchor[1]:
        return None

    frame_anchors = {
        "left": left_anchor,
        "top": top_anchor,
        "right": right_anchor,
        "bottom": bottom_anchor,
    }
    for spec in specs:
        r, c = frame_anchors[spec["side"]]
        grid[r, c] = int(spec["anchor_color"])

    # Left piece: straight rightward path.
    patch = left_spec["input_patch"]
    h0, w0 = patch.shape
    gap = randint(2, left_margin - w0 - 1)
    anchor_r, anchor_c = anchor_pos_in_patch(patch, left_spec["anchor_color"])
    assert anchor_c == w0 - 1
    top = left_anchor[0] - anchor_r
    left = fc0 - gap - w0
    if not place_patch(grid, patch, top, left):
        return None
    piece_anchor = (top + anchor_r, left + anchor_c)
    if grid[piece_anchor] != left_spec["anchor_color"]:
        return None
    if not paint_path(grid, left_spec["anchor_color"], [piece_anchor, left_anchor], frame_bbox, left_anchor):
        return None

    # Top piece: right, then down.
    patch = top_spec["input_patch"]
    h1, w1 = patch.shape
    gap_v = randint(2, top_margin - h1 - 1)
    gap_h = randint(2, min(5, top_anchor[1] - (w1 - 1)))
    top = fr0 - gap_v - h1
    left = top_anchor[1] - gap_h - (w1 - 1)
    if top < 0 or left < 0:
        return None
    if not place_patch(grid, patch, top, left):
        return None
    anchor_r, anchor_c = anchor_pos_in_patch(patch, top_spec["anchor_color"])
    assert anchor_c == w1 - 1
    piece_anchor = (top + anchor_r, left + anchor_c)
    if piece_anchor[1] >= top_anchor[1] or piece_anchor[0] >= fr0:
        return None
    if not paint_path(
        grid,
        top_spec["anchor_color"],
        [piece_anchor, (piece_anchor[0], top_anchor[1]), top_anchor],
        frame_bbox,
        top_anchor,
    ):
        return None

    # Right piece: left, vertical, left.
    patch = right_spec["input_patch"]
    h2, w2 = patch.shape
    gap_h = randint(3, max(3, right_margin - w2))
    left = fc1 + gap_h
    if left + w2 > W:
        return None
    row_offset, anchor_c = anchor_pos_in_patch(patch, right_spec["anchor_color"])
    assert anchor_c == 0
    delta = choice((-4, -3, -2, 2, 3, 4))
    anchor_row = right_anchor[0] + delta
    top = anchor_row - row_offset
    if top < 0 or top + h2 > H:
        return None
    if not place_patch(grid, patch, top, left):
        return None
    piece_anchor = (anchor_row, left + anchor_c)
    if piece_anchor[1] <= fc1 + 2 or piece_anchor[0] == right_anchor[0]:
        return None
    bend_col = randint(fc1 + 1, piece_anchor[1] - 1)
    if not paint_path(
        grid,
        right_spec["anchor_color"],
        [piece_anchor, (piece_anchor[0], bend_col), (right_anchor[0], bend_col), right_anchor],
        frame_bbox,
        right_anchor,
    ):
        return None

    # Bottom piece: left, up, left, up.
    patch = bottom_spec["input_patch"]
    h3, w3 = patch.shape
    gap_v = randint(3, max(3, bottom_margin - h3))
    top = fr1 + gap_v
    if top + h3 > H:
        return None
    gap_h = randint(3, max(3, right_margin - w3))
    left = bottom_anchor[1] + gap_h
    if left + w3 > W:
        return None
    row_offset, anchor_c = anchor_pos_in_patch(patch, bottom_spec["anchor_color"])
    assert anchor_c == 0
    if not place_patch(grid, patch, top, left):
        return None
    piece_anchor = (top + row_offset, left + anchor_c)
    if piece_anchor[0] <= fr1 + 1 or piece_anchor[1] <= bottom_anchor[1] + 2:
        return None
    bend_col = randint(bottom_anchor[1] + 1, piece_anchor[1] - 1)
    bend_row = randint(fr1 + 1, piece_anchor[0] - 1)
    if not paint_path(
        grid,
        bottom_spec["anchor_color"],
        [piece_anchor, (piece_anchor[0], bend_col), (bend_row, bend_col), (bend_row, bottom_anchor[1]), bottom_anchor],
        frame_bbox,
        bottom_anchor,
    ):
        return None

    expected = np.full_like(grid, BG)
    frame_in = grid[fr0 : fr1 + 1, fc0 : fc1 + 1]
    frame_out = expected[fr0 : fr1 + 1, fc0 : fc1 + 1]
    frame_out[frame_in != BG] = frame_in[frame_in != BG]
    for spec in specs:
        lr0, lr1, lc0, lc1 = spec["region"]
        dest = expected[fr0 + lr0 : fr0 + lr1 + 1, fc0 + lc0 : fc0 + lc1 + 1]
        dest[:, :] = spec["output_patch"]

    krot = randint(0, 3)
    if krot:
        grid = np.rot90(grid, krot)
        expected = np.rot90(expected, krot)

    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = gen_attempt()
        if result is not None:
            return result[0]
    raise RuntimeError("failed to generate a valid sample")
