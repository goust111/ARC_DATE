import random
from collections import deque

import numpy as np


DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

ROTATE_COLOR = 2
HFLIP_COLOR = 1
VFLIP_COLOR = 3
SINGLE_MARKER_OPS = {
    ROTATE_COLOR: "rot90",
    HFLIP_COLOR: "flip_h",
    VFLIP_COLOR: "flip_v",
}

ALL_COLORS = tuple(range(10))
BG_CHOICES = (0, 4, 5, 6, 7, 8, 9)
SEED_DIMS = ((3, 3), (3, 5), (4, 4), (5, 3), (5, 5))
LAYER_RANGE = (2, 4)
GAP_CHOICES = (1, 2)
MARGIN_CHOICES = (1, 2, 3, 4)
MAX_SIDE = 40
GEN_RETRIES = 400


def _mode_color(arr: np.ndarray) -> int:
    values, counts = np.unique(arr, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _bbox_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return None
    rows = pts[:, 0]
    cols = pts[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(comp)
    return out


def _is_rectangular_ring(coords: list[tuple[int, int]]) -> bool:
    if len(coords) < 8:
        return False
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    if len(coords) != 2 * (h + w) - 4:
        return False
    pts = set(coords)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            on_border = r in (r0, r1) or c in (c0, c1)
            if on_border != ((r, c) in pts):
                return False
    return True


def _read_marker(grid: np.ndarray, bg: int, bbox: tuple[int, int, int, int]):
    h, w = grid.shape
    r0, _, _, c1 = bbox
    if c1 + 1 >= w:
        return None
    first = int(grid[r0, c1 + 1])
    if first == bg:
        return None
    second = int(grid[r0, c1 + 2]) if c1 + 2 < w else bg
    if second != bg:
        if first == second:
            return None
        return ("replace", first, second)
    kind = SINGLE_MARKER_OPS.get(first)
    if kind is None:
        return None
    return (kind,)


def _apply_op(pattern: np.ndarray, op) -> np.ndarray:
    kind = op[0]
    if kind == "rot90":
        return np.rot90(pattern, -1)
    if kind == "flip_h":
        return np.fliplr(pattern)
    if kind == "flip_v":
        return np.flipud(pattern)
    if kind == "replace":
        src, dst = int(op[1]), int(op[2])
        out = pattern.copy()
        out[out == src] = dst
        return out
    raise ValueError(f"unknown op: {op}")


def _marker_width(op) -> int:
    return 2 if op[0] == "replace" else 1


def _marker_colors(op) -> set[int]:
    if op[0] == "replace":
        return {int(op[1]), int(op[2])}
    if op[0] == "rot90":
        return {ROTATE_COLOR}
    if op[0] == "flip_h":
        return {HFLIP_COLOR}
    if op[0] == "flip_v":
        return {VFLIP_COLOR}
    raise ValueError(f"unknown op: {op}")


def _center_start(total: int, size: int) -> int:
    return (total - size) // 2


def main(grid: np.ndarray) -> np.ndarray:
    """
    Apply nested frame-corner operators from the innermost frame outward, then
    show only the transformed seed pattern on the background.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    bg = _mode_color(g)

    frames: list[dict] = []
    for color in np.unique(g):
        color = int(color)
        if color == bg:
            continue
        for comp in _components(g == color):
            if not _is_rectangular_ring(comp):
                continue
            bbox = _bbox(comp)
            op = _read_marker(g, bg, bbox)
            if op is None:
                continue
            frames.append({"bbox": bbox, "op": op})

    assert len(frames) >= 2, "expected 2 to 4 marked nested frames"
    frames.sort(key=lambda item: (item["bbox"][2] - item["bbox"][0] + 1) * (item["bbox"][3] - item["bbox"][1] + 1))

    for i in range(1, len(frames)):
        ir0, ic0, ir1, ic1 = frames[i - 1]["bbox"]
        or0, oc0, or1, oc1 = frames[i]["bbox"]
        assert or0 < ir0 and oc0 < ic0 and or1 > ir1 and oc1 > ic1, "frames must form one nested chain"

    r0, c0, r1, c1 = frames[0]["bbox"]
    interior = g[r0 + 1 : r1, c0 + 1 : c1]
    seed_box = _bbox_mask(interior != bg)
    assert seed_box is not None, "innermost frame must contain a seed pattern"
    sr0, sc0, sr1, sc1 = seed_box
    pattern = interior[sr0 : sr1 + 1, sc0 : sc1 + 1].copy()

    for frame in frames:
        pattern = _apply_op(pattern, frame["op"])

    out = np.full_like(g, bg)
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    ph, pw = map(int, pattern.shape)
    assert ph <= inner_h and pw <= inner_w, "transformed pattern must fit inside the innermost frame"
    pr = r0 + 1 + _center_start(inner_h, ph)
    pc = c0 + 1 + _center_start(inner_w, pw)
    out[pr : pr + ph, pc : pc + pw] = pattern
    return out


def _make_seed(rng: random.Random, bg: int) -> np.ndarray | None:
    for _ in range(200):
        h, w = rng.choice(SEED_DIMS)
        palette = rng.sample([c for c in ALL_COLORS if c != bg], rng.choice((2, 3)))
        fill_prob = rng.uniform(0.55, 0.8)
        seed = np.full((h, w), bg, dtype=int)
        for r in range(h):
            for c in range(w):
                if rng.random() < fill_prob:
                    seed[r, c] = rng.choice(palette)

        seed[0, rng.randrange(w)] = rng.choice(palette)
        seed[h - 1, rng.randrange(w)] = rng.choice(palette)
        seed[rng.randrange(h), 0] = rng.choice(palette)
        seed[rng.randrange(h), w - 1] = rng.choice(palette)

        if _bbox_mask(seed != bg) != (0, 0, h - 1, w - 1):
            continue
        density = float(np.count_nonzero(seed != bg)) / float(h * w)
        if not (0.4 <= density <= 0.9):
            continue
        if len({int(v) for v in np.unique(seed) if int(v) != bg}) < 2:
            continue
        if np.array_equal(seed, np.fliplr(seed)):
            continue
        if np.array_equal(seed, np.flipud(seed)):
            continue
        if np.array_equal(seed, np.rot90(seed, -1)):
            continue
        return seed
    return None


def _sample_ops(rng: random.Random, layer_count: int, seed: np.ndarray, bg: int):
    current = seed.copy()
    ops = []
    for _ in range(layer_count):
        chosen = None
        next_pattern = None
        for _ in range(40):
            kind = rng.choice(("rot90", "flip_h", "flip_v", "replace"))
            if kind == "replace":
                colors = [int(v) for v in np.unique(current) if int(v) != bg]
                if not colors:
                    continue
                src = rng.choice(colors)
                dst = rng.choice([c for c in ALL_COLORS if c != bg and c != src])
                candidate = ("replace", src, dst)
                updated = _apply_op(current, candidate)
            else:
                candidate = (kind,)
                updated = _apply_op(current, candidate)
                if np.array_equal(updated, current):
                    continue
            chosen = candidate
            next_pattern = updated
            break
        if chosen is None or next_pattern is None:
            return None
        ops.append(chosen)
        current = next_pattern
    return ops, current


def _sample_frame_colors(rng: random.Random, bg: int, ops, gaps):
    used: set[int] = set()
    colors: list[int] = []
    for i, op in enumerate(ops):
        forbidden = {bg}
        forbidden.update(_marker_colors(op))
        if i > 0 and gaps[i - 1] == 1 and ops[i - 1][0] == "replace":
            forbidden.add(int(ops[i - 1][2]))
        choices = [c for c in ALL_COLORS if c not in used and c not in forbidden]
        if not choices:
            return None
        color = rng.choice(choices)
        used.add(color)
        colors.append(color)
    return colors


def _paint_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _paint_marker(grid: np.ndarray, bbox: tuple[int, int, int, int], op) -> None:
    r0, _, _, c1 = bbox
    if op[0] == "replace":
        grid[r0, c1 + 1] = int(op[1])
        grid[r0, c1 + 2] = int(op[2])
    elif op[0] == "rot90":
        grid[r0, c1 + 1] = ROTATE_COLOR
    elif op[0] == "flip_h":
        grid[r0, c1 + 1] = HFLIP_COLOR
    elif op[0] == "flip_v":
        grid[r0, c1 + 1] = VFLIP_COLOR
    else:
        raise ValueError(f"unknown op: {op}")


def _expected_output(shape: tuple[int, int], bg: int, inner_bbox: tuple[int, int, int, int], pattern: np.ndarray) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    r0, c0, r1, c1 = inner_bbox
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    ph, pw = map(int, pattern.shape)
    pr = r0 + 1 + _center_start(inner_h, ph)
    pc = c0 + 1 + _center_start(inner_w, pw)
    out[pr : pr + ph, pc : pc + pw] = pattern
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    bg = rng.choice(BG_CHOICES)
    layer_count = rng.randint(LAYER_RANGE[0], LAYER_RANGE[1])

    seed = _make_seed(rng, bg)
    if seed is None:
        return None

    sampled = _sample_ops(rng, layer_count, seed, bg)
    if sampled is None:
        return None
    ops, final_pattern = sampled

    max_dim = max(seed.shape)
    inner_h = max_dim + 2 + 2 * rng.randint(0, 1)
    inner_w = max_dim + 2 + 2 * rng.randint(0, 1)

    rel_bboxes = [(0, 0, inner_h + 1, inner_w + 1)]
    gaps = []
    for _ in range(layer_count - 1):
        gap = rng.choice(GAP_CHOICES)
        gaps.append(gap)
        delta = gap + 2
        pr0, pc0, pr1, pc1 = rel_bboxes[-1]
        rel_bboxes.append((pr0 - delta, pc0 - delta, pr1 + delta, pc1 + delta))

    frame_colors = _sample_frame_colors(rng, bg, ops, gaps)
    if frame_colors is None:
        return None

    outer_bbox = rel_bboxes[-1]
    outer_h = outer_bbox[2] - outer_bbox[0] + 1
    outer_w = outer_bbox[3] - outer_bbox[1] + 1
    outer_marker_w = _marker_width(ops[-1])

    top_margin = rng.choice(MARGIN_CHOICES)
    left_margin = rng.choice(MARGIN_CHOICES)
    bottom_margin = rng.choice(MARGIN_CHOICES)
    right_margin = rng.choice(MARGIN_CHOICES)

    height = top_margin + outer_h + bottom_margin
    width = left_margin + outer_w + outer_marker_w + right_margin
    if height > MAX_SIDE or width > MAX_SIDE:
        return None

    shift_r = top_margin - outer_bbox[0]
    shift_c = left_margin - outer_bbox[1]
    bboxes = [
        (r0 + shift_r, c0 + shift_c, r1 + shift_r, c1 + shift_c)
        for r0, c0, r1, c1 in rel_bboxes
    ]

    grid = np.full((height, width), bg, dtype=int)
    for bbox, color in zip(bboxes, frame_colors):
        _paint_frame(grid, bbox, color)
    for bbox, op in zip(bboxes, ops):
        _paint_marker(grid, bbox, op)

    inner_bbox = bboxes[0]
    ir0, ic0, ir1, ic1 = inner_bbox
    inner_space_h = ir1 - ir0 - 1
    inner_space_w = ic1 - ic0 - 1
    sh, sw = map(int, seed.shape)
    sr = ir0 + 1 + _center_start(inner_space_h, sh)
    sc = ic0 + 1 + _center_start(inner_space_w, sw)
    grid[sr : sr + sh, sc : sc + sw] = seed

    if _mode_color(grid) != bg:
        return None

    expected = _expected_output(grid.shape, bg, inner_bbox, final_pattern)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-frame operator puzzle")
