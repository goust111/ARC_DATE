import random
import numpy as np


BLACK = 0
COLORS = list(range(1, 10))
GEN_RETRIES = 400
GRID_MIN = 18
GRID_MAX = 30
MAX_TOTAL_FRAMES = 6
CHAIN_MARGIN = 1
CHAIN_GAP = 2
PLACE_TRIES = 250
LEAF_SIZE_RANGE = (5, 7)
NEST_GAP_RANGE = (1, 2)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _rect_boundary(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    pts: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        pts.add((r0, c))
        pts.add((r1, c))
    for r in range(r0 + 1, r1):
        pts.add((r, c0))
        pts.add((r, c1))
    return pts


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _rect_contains(
    outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]
) -> bool:
    or0, oc0, or1, oc1 = outer
    ir0, ic0, ir1, ic1 = inner
    return or0 < ir0 and oc0 < ic0 and ir1 < or1 and ic1 < oc1


def _find_frames(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    frames: list[dict] = []
    for color0 in np.unique(g):
        color = int(color0)
        if color == BLACK:
            continue
        for pts in _components(g == color):
            r0, c0, r1, c1 = _bbox(pts)
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue
            boundary = _rect_boundary(r0, c0, r1, c1)
            if set(pts) != boundary:
                continue
            frames.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "area": (r1 - r0 + 1) * (c1 - c0 + 1),
                }
            )
    assert frames, "expected at least one rectangular frame"
    return frames


def _build_tree(frames: list[dict]) -> tuple[list[int | None], list[list[int]], list[int]]:
    n = len(frames)
    parents: list[int | None] = [None] * n
    children: list[list[int]] = [[] for _ in range(n)]

    for i in range(n):
        bbox_i = frames[i]["bbox"]
        container = None
        container_area = None
        for j in range(n):
            if i == j:
                continue
            bbox_j = frames[j]["bbox"]
            if not _rect_contains(bbox_j, bbox_i):
                continue
            area_j = int(frames[j]["area"])
            if container is None or area_j < container_area:
                container = j
                container_area = area_j
        parents[i] = container
        if container is not None:
            children[container].append(i)

    depths = [0] * n

    def assign_depth(idx: int) -> int:
        if depths[idx]:
            return depths[idx]
        parent = parents[idx]
        depths[idx] = 1 if parent is None else assign_depth(parent) + 1
        return depths[idx]

    for i in range(n):
        assign_depth(i)
    return parents, children, depths


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect nested hollow rectangular fences and fill each fence interior by depth.

    Odd depths use the fence's own border color. Even depths use the unique seed
    color propagated from the innermost descendant fence. Seed pixels remain
    unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frames = _find_frames(g)
    parents, children, depths = _build_tree(frames)

    frame_mask = np.zeros_like(g, dtype=bool)
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        frame_mask[r0, c0 : c1 + 1] = True
        frame_mask[r1, c0 : c1 + 1] = True
        frame_mask[r0 : r1 + 1, c0] = True
        frame_mask[r0 : r1 + 1, c1] = True

    seed_mask = (g != BLACK) & (~frame_mask)
    seed_positions = [tuple(map(int, rc)) for rc in np.argwhere(seed_mask)]
    assert seed_positions, "expected at least one seed pixel"

    leaf_seed: dict[int, int] = {}
    for r, c in seed_positions:
        containing = [
            i
            for i, frame in enumerate(frames)
            if frame["bbox"][0] < r < frame["bbox"][2] and frame["bbox"][1] < c < frame["bbox"][3]
        ]
        assert containing, "every seed pixel must lie inside a frame"
        leaf = min(containing, key=lambda idx: int(frames[idx]["area"]))
        assert not children[leaf], "deepest containing frame must be a leaf"
        if leaf in leaf_seed:
            raise AssertionError("expected exactly one seed per leaf frame")
        leaf_seed[leaf] = int(g[r, c])

    for idx, child_ids in enumerate(children):
        if not child_ids:
            assert idx in leaf_seed, "every leaf frame must contain a seed pixel"

    propagated: dict[int, int] = {}

    def resolve_seed(idx: int) -> int:
        if idx in propagated:
            return propagated[idx]
        if not children[idx]:
            propagated[idx] = leaf_seed[idx]
            return propagated[idx]
        colors = {resolve_seed(child) for child in children[idx]}
        assert len(colors) == 1, "even-depth fill is ambiguous without a unique subtree seed color"
        propagated[idx] = next(iter(colors))
        return propagated[idx]

    out = g.copy()
    for idx, frame in enumerate(frames):
        r0, c0, r1, c1 = frame["bbox"]
        annulus = np.zeros_like(g, dtype=bool)
        annulus[r0 + 1 : r1, c0 + 1 : c1] = True
        for child in children[idx]:
            cr0, cc0, cr1, cc1 = frames[child]["bbox"]
            annulus[cr0 : cr1 + 1, cc0 : cc1 + 1] = False

        fill_color = int(frame["color"]) if depths[idx] % 2 == 1 else resolve_seed(idx)
        out[annulus] = fill_color

    out[seed_mask] = g[seed_mask]
    return out


def _shift_boxes(
    boxes: list[tuple[int, int, int, int]], dr: int, dc: int
) -> list[tuple[int, int, int, int]]:
    return [(r0 + dr, c0 + dc, r1 + dr, c1 + dc) for r0, c0, r1, c1 in boxes]


def _sample_depths() -> list[int]:
    for _ in range(100):
        chain_count = random.choices([1, 2, 3], weights=[2, 3, 2])[0]
        depths = [random.randint(1, 4) for _ in range(chain_count)]
        if sum(depths) > MAX_TOTAL_FRAMES:
            continue
        if all(depth == 1 for depth in depths) and random.random() < 0.75:
            continue
        return depths
    raise RuntimeError("failed to sample chain depths")


def _build_chain(
    depth: int, frame_colors: list[int], seed_color: int
) -> tuple[np.ndarray, np.ndarray]:
    leaf_h = random.randint(*LEAF_SIZE_RANGE)
    leaf_w = random.randint(*LEAF_SIZE_RANGE)
    boxes = [(0, 0, leaf_h - 1, leaf_w - 1)]  # inner -> outer
    seed = (random.randint(1, leaf_h - 2), random.randint(1, leaf_w - 2))
    cur_h, cur_w = leaf_h, leaf_w

    for _ in range(depth - 1):
        top_gap = random.randint(*NEST_GAP_RANGE)
        bottom_gap = random.randint(*NEST_GAP_RANGE)
        left_gap = random.randint(*NEST_GAP_RANGE)
        right_gap = random.randint(*NEST_GAP_RANGE)

        dr = top_gap + 1
        dc = left_gap + 1
        boxes = _shift_boxes(boxes, dr, dc)
        seed = (seed[0] + dr, seed[1] + dc)
        cur_h += top_gap + bottom_gap + 2
        cur_w += left_gap + right_gap + 2
        boxes.append((0, 0, cur_h - 1, cur_w - 1))

    boxes = list(reversed(boxes))  # outer -> inner

    inp = np.zeros((cur_h, cur_w), dtype=int)
    for bbox, color in zip(boxes, frame_colors, strict=True):
        _draw_frame(inp, bbox, color)
    inp[seed] = seed_color

    out = inp.copy()
    for idx, (bbox, color) in enumerate(zip(boxes, frame_colors, strict=True), start=1):
        r0, c0, r1, c1 = bbox
        annulus = np.zeros_like(out, dtype=bool)
        annulus[r0 + 1 : r1, c0 + 1 : c1] = True
        if idx < len(boxes):
            cr0, cc0, cr1, cc1 = boxes[idx]
            annulus[cr0 : cr1 + 1, cc0 : cc1 + 1] = False
        out[annulus] = color if idx % 2 == 1 else seed_color
    out[seed] = seed_color
    return inp, out


def _rects_clear(
    rect_a: tuple[int, int, int, int], rect_b: tuple[int, int, int, int], gap: int
) -> bool:
    ar0, ac0, ar1, ac1 = rect_a
    br0, bc0, br1, bc1 = rect_b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _place_patches(
    shapes: list[tuple[int, int]], grid_h: int, grid_w: int
) -> list[tuple[int, int]] | None:
    positions: list[tuple[int, int] | None] = [None] * len(shapes)
    placed_rects: list[tuple[int, int, int, int]] = []
    order = sorted(range(len(shapes)), key=lambda idx: shapes[idx][0] * shapes[idx][1], reverse=True)

    for idx in order:
        ph, pw = shapes[idx]
        if ph + 2 * CHAIN_MARGIN > grid_h or pw + 2 * CHAIN_MARGIN > grid_w:
            return None
        placed = False
        for _ in range(PLACE_TRIES):
            r0 = random.randint(CHAIN_MARGIN, grid_h - CHAIN_MARGIN - ph)
            c0 = random.randint(CHAIN_MARGIN, grid_w - CHAIN_MARGIN - pw)
            rect = (r0, c0, r0 + ph - 1, c0 + pw - 1)
            if all(_rects_clear(rect, other, CHAIN_GAP) for other in placed_rects):
                positions[idx] = (r0, c0)
                placed_rects.append(rect)
                placed = True
                break
        if not placed:
            return None

    return [(int(r), int(c)) for r, c in positions]


def _paste(dst: np.ndarray, src: np.ndarray, r0: int, c0: int) -> None:
    h, w = src.shape
    dst[r0 : r0 + h, c0 : c0 + w] = src


def _generate_case() -> tuple[np.ndarray, np.ndarray]:
    depths = _sample_depths()
    total_frames = sum(depths)
    border_colors = random.sample(COLORS, total_frames)
    seed_choices = [color for color in COLORS if color not in border_colors]
    assert seed_choices, "need at least one seed color not used by borders"

    patches_in: list[np.ndarray] = []
    patches_out: list[np.ndarray] = []
    color_offset = 0
    for depth in depths:
        frame_colors = border_colors[color_offset : color_offset + depth]
        color_offset += depth
        seed_color = random.choice(seed_choices)
        patch_in, patch_out = _build_chain(depth, frame_colors, seed_color)
        patches_in.append(patch_in)
        patches_out.append(patch_out)

    shapes = [patch.shape for patch in patches_in]
    min_area = sum((h + CHAIN_GAP) * (w + CHAIN_GAP) for h, w in shapes)
    for _ in range(80):
        grid_h = random.randint(GRID_MIN, GRID_MAX)
        grid_w = random.randint(GRID_MIN, GRID_MAX)
        usable_area = max(0, grid_h - 2 * CHAIN_MARGIN) * max(0, grid_w - 2 * CHAIN_MARGIN)
        if usable_area < min_area:
            continue
        positions = _place_patches(shapes, grid_h, grid_w)
        if positions is None:
            continue

        inp = np.zeros((grid_h, grid_w), dtype=int)
        out = np.zeros((grid_h, grid_w), dtype=int)
        for patch_in, patch_out, (r0, c0) in zip(patches_in, patches_out, positions, strict=True):
            _paste(inp, patch_in, r0, c0)
            _paste(out, patch_out, r0, c0)
        return inp, out
    raise RuntimeError("failed to place chain patches")


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            inp, expected = _generate_case()
        except RuntimeError:
            continue
        try:
            actual = main(inp)
        except Exception:
            continue
        if not np.array_equal(actual, expected):
            continue
        if np.array_equal(inp, expected):
            continue
        return inp
    raise RuntimeError("failed to generate a valid input")
