from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

FRAME_COLORS = (BLUE, RED, GREEN, YELLOW, GREY)
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_RETRIES = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _border_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _frame_from_color(grid: np.ndarray, color: int) -> dict:
    mask = grid == int(color)
    box = _bbox(mask)
    assert box is not None
    r0, c0, r1, c1 = box
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    local_mask = patch == int(color)
    h, w = local_mask.shape
    assert h >= 3 and w >= 3
    assert np.array_equal(local_mask, _border_mask(h, w))
    return {
        "color": int(color),
        "bbox": (r0, c0, r1, c1),
        "h": h,
        "w": w,
        "area": h * w,
        "parent": None,
        "children": [],
        "depth": None,
    }


def _contains(outer: dict, inner: dict) -> bool:
    or0, oc0, or1, oc1 = outer["bbox"]
    ir0, ic0, ir1, ic1 = inner["bbox"]
    return or0 < ir0 and oc0 < ic0 and or1 > ir1 and oc1 > ic1


def _extract_frames(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert int(np.min(g)) >= 0 and int(np.max(g)) <= 9

    colors = sorted(int(v) for v in np.unique(g) if int(v) != BLACK)
    assert 3 <= len(colors) <= 5
    frames = [_frame_from_color(g, color) for color in colors]

    for i, frame in enumerate(frames):
        containers = [j for j, other in enumerate(frames) if i != j and _contains(other, frame)]
        if containers:
            parent = min(containers, key=lambda j: frames[j]["area"])
            frame["parent"] = int(parent)
            frames[parent]["children"].append(i)

    roots = [i for i, frame in enumerate(frames) if frame["parent"] is None]
    assert roots
    for root in roots:
        frames[root]["depth"] = 1
        stack = [root]
        while stack:
            cur = stack.pop()
            for child in frames[cur]["children"]:
                frames[child]["depth"] = int(frames[cur]["depth"]) + 1
                stack.append(child)

    assert all(frame["depth"] is not None for frame in frames)
    assert max(int(frame["depth"]) for frame in frames) >= 3
    return frames


def _depth_color(depth: int) -> int:
    return GREEN if depth % 2 == 0 else RED


def main(grid: np.ndarray) -> np.ndarray:
    """Flood-fill black regions by nested frame depth parity, then erase leaf-frame borders into their depth color.

    Input structure:
    - The grid has black background and 3-5 distinct one-pixel-thick hollow rectangular frames.
    - Every non-black color forms exactly one axis-aligned rectangular border, and frames are either disjoint or properly nested.
    - Nested frames are separated by at least one black cell gap, so each black area is a well-defined 4-connected region.

    Rule:
    - Detect all hollow rectangular frames and build their strict containment relation.
    - Assign frame depth 1 to outermost frames and increase depth by one for each nesting level.
    - For each 4-connected black region, find the deepest frame whose interior contains that region: depth 0 if none, otherwise that frame depth.
    - Recolor black regions green at even depth and red at odd depth, then replace every leaf frame border with the same parity color for that leaf depth while keeping non-leaf frame borders unchanged.

    Key insight:
    - The output is determined by recursive topology rather than local geometry: the same black color must be partitioned by frame-containment depth, and leaf erasure depends on identifying which frames have no nested descendants.
    """
    g = np.asarray(grid, dtype=int)
    frames = _extract_frames(g)
    out = g.copy()

    for comp in _components(g == BLACK):
        r, c = map(int, comp[0])
        region_depth = 0
        for frame in frames:
            r0, c0, r1, c1 = frame["bbox"]
            if r0 < r < r1 and c0 < c < c1:
                region_depth = max(region_depth, int(frame["depth"]))
        out[comp[:, 0], comp[:, 1]] = _depth_color(region_depth)

    for frame in frames:
        if frame["children"]:
            continue
        r0, c0, r1, c1 = frame["bbox"]
        fill = _depth_color(int(frame["depth"]))
        out[r0, c0 : c1 + 1] = fill
        out[r1, c0 : c1 + 1] = fill
        out[r0 : r1 + 1, c0] = fill
        out[r0 : r1 + 1, c1] = fill

    return out


def _normalize_boxes(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    min_r = min(r0 for r0, _c0, _r1, _c1 in boxes)
    min_c = min(c0 for _r0, c0, _r1, _c1 in boxes)
    return [(r0 - min_r, c0 - min_c, r1 - min_r, c1 - min_c) for r0, c0, r1, c1 in boxes]


def _chain_boxes(length: int) -> list[tuple[int, int, int, int]] | None:
    for _ in range(80):
        h = random.randint(3, 5)
        w = random.randint(3, 6)
        r0 = 0
        c0 = 0
        r1 = h - 1
        c1 = w - 1
        boxes = [(r0, c0, r1, c1)]
        for _layer in range(length - 1):
            top = random.randint(2, 3)
            bottom = random.randint(2, 3)
            left = random.randint(2, 3)
            right = random.randint(2, 3)
            r0 -= top
            c0 -= left
            r1 += bottom
            c1 += right
            boxes.append((r0, c0, r1, c1))
        boxes = _normalize_boxes(boxes)
        h_total = max(r1 for _r0, _c0, r1, _c1 in boxes) + 1
        w_total = max(c1 for _r0, _c0, _r1, c1 in boxes) + 1
        if h_total <= 26 and w_total <= 26:
            return boxes
    return None


def _branch_boxes() -> list[tuple[int, int, int, int]] | None:
    for _ in range(120):
        leaf_a_h = random.randint(4, 7)
        leaf_a_w = random.randint(4, 7)
        leaf_a = (0, 0, leaf_a_h - 1, leaf_a_w - 1)

        mid_chain = _chain_boxes(2)
        if mid_chain is None:
            continue
        leaf_b, mid_b = mid_chain[0], mid_chain[1]
        leaf_b_h = leaf_b[2] - leaf_b[0] + 1
        leaf_b_w = leaf_b[3] - leaf_b[1] + 1
        mid_b_h = mid_b[2] - mid_b[0] + 1
        mid_b_w = mid_b[3] - mid_b[1] + 1
        assert leaf_b_h >= 3 and leaf_b_w >= 3

        gap = random.randint(2, 4)
        orientation = random.choice(("h", "v"))
        if orientation == "h":
            offset_a = (random.randint(0, 2), 0)
            offset_b = (random.randint(0, 2), leaf_a_w + gap)
        else:
            offset_a = (0, random.randint(0, 2))
            offset_b = (leaf_a_h + gap, random.randint(0, 2))

        placed_leaf_a = (
            leaf_a[0] + offset_a[0],
            leaf_a[1] + offset_a[1],
            leaf_a[2] + offset_a[0],
            leaf_a[3] + offset_a[1],
        )
        placed_leaf_b = (
            leaf_b[0] + offset_b[0],
            leaf_b[1] + offset_b[1],
            leaf_b[2] + offset_b[0],
            leaf_b[3] + offset_b[1],
        )
        placed_mid_b = (
            mid_b[0] + offset_b[0],
            mid_b[1] + offset_b[1],
            mid_b[2] + offset_b[0],
            mid_b[3] + offset_b[1],
        )

        min_r = min(placed_leaf_a[0], placed_mid_b[0])
        min_c = min(placed_leaf_a[1], placed_mid_b[1])
        max_r = max(placed_leaf_a[2], placed_mid_b[2])
        max_c = max(placed_leaf_a[3], placed_mid_b[3])
        outer = (
            min_r - random.randint(2, 3),
            min_c - random.randint(2, 3),
            max_r + random.randint(2, 3),
            max_c + random.randint(2, 3),
        )
        boxes = _normalize_boxes([placed_leaf_a, placed_leaf_b, placed_mid_b, outer])
        h_total = max(r1 for _r0, _c0, r1, _c1 in boxes) + 1
        w_total = max(c1 for _r0, _c0, _r1, c1 in boxes) + 1
        if h_total <= 26 and w_total <= 26:
            return boxes
    return None


def _two_outer_boxes() -> list[tuple[int, int, int, int]] | None:
    for _ in range(120):
        left_chain = _chain_boxes(random.choice((3, 4)))
        right_chain = _chain_boxes(random.choice((1, 2)))
        if left_chain is None or right_chain is None:
            continue

        left_h = max(r1 for _r0, _c0, r1, _c1 in left_chain) + 1
        left_w = max(c1 for _r0, _c0, _r1, c1 in left_chain) + 1
        right_h = max(r1 for _r0, _c0, r1, _c1 in right_chain) + 1
        right_w = max(c1 for _r0, _c0, _r1, c1 in right_chain) + 1
        gap = random.randint(2, 4)

        if random.random() < 0.6:
            left_shift = (random.randint(0, 2), 0)
            right_shift = (random.randint(0, 2), left_w + gap)
        else:
            left_shift = (0, random.randint(0, 2))
            right_shift = (left_h + gap, random.randint(0, 2))

        boxes: list[tuple[int, int, int, int]] = []
        for r0, c0, r1, c1 in left_chain:
            boxes.append((r0 + left_shift[0], c0 + left_shift[1], r1 + left_shift[0], c1 + left_shift[1]))
        for r0, c0, r1, c1 in right_chain:
            boxes.append((r0 + right_shift[0], c0 + right_shift[1], r1 + right_shift[0], c1 + right_shift[1]))

        boxes = _normalize_boxes(boxes)
        h_total = max(r1 for _r0, _c0, r1, _c1 in boxes) + 1
        w_total = max(c1 for _r0, _c0, _r1, c1 in boxes) + 1
        if h_total <= 26 and w_total <= 26:
            return boxes
    return None


def _build_boxes() -> list[tuple[int, int, int, int]] | None:
    mode = random.choices(
        population=("chain", "branch", "two_outer"),
        weights=(3, 3, 2),
        k=1,
    )[0]
    if mode == "chain":
        return _chain_boxes(random.choice((3, 4, 5)))
    if mode == "branch":
        return _branch_boxes()
    return _two_outer_boxes()


def _paint_frames(shape: tuple[int, int], boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    colors = random.sample(FRAME_COLORS, len(boxes))
    random.shuffle(boxes)
    for color, (r0, c0, r1, c1) in zip(colors, boxes):
        grid[r0, c0 : c1 + 1] = color
        grid[r1, c0 : c1 + 1] = color
        grid[r0 : r1 + 1, c0] = color
        grid[r0 : r1 + 1, c1] = color
    return grid


def _semantic_ok(inp: np.ndarray, out: np.ndarray) -> bool:
    try:
        frames = _extract_frames(inp)
    except AssertionError:
        return False

    depths = sorted({int(frame["depth"]) for frame in frames})
    if depths[0] != 1 or len(depths) < 3:
        return False

    leaf_depths = sorted(int(frame["depth"]) for frame in frames if not frame["children"])
    if not leaf_depths:
        return False

    black_comps = _components(inp == BLACK)
    region_depths = set()
    for comp in black_comps:
        r, c = map(int, comp[0])
        depth = 0
        for frame in frames:
            r0, c0, r1, c1 = frame["bbox"]
            if r0 < r < r1 and c0 < c < c1:
                depth = max(depth, int(frame["depth"]))
        region_depths.add(depth)
    if region_depths != set(range(max(depths) + 1)):
        return False

    if not np.any(inp == BLACK):
        return False
    if not np.any(out != inp):
        return False
    if not np.any((inp != BLACK) & (out != inp)):
        return False
    if not ({RED, GREEN} <= set(map(int, np.unique(out)))):
        return False
    return True


def _attempt_generate() -> np.ndarray | None:
    boxes = _build_boxes()
    if boxes is None:
        return None
    if len(boxes) > len(FRAME_COLORS):
        return None

    local_h = max(r1 for _r0, _c0, r1, _c1 in boxes) + 1
    local_w = max(c1 for _r0, _c0, _r1, c1 in boxes) + 1
    if local_h > 29 or local_w > 29:
        return None

    h = random.randint(local_h, min(30, max(16, local_h + 4)))
    w = random.randint(local_w, min(30, max(16, local_w + 4)))
    shift_r = random.randint(0, h - local_h)
    shift_c = random.randint(0, w - local_w)
    shifted = [(r0 + shift_r, c0 + shift_c, r1 + shift_r, c1 + shift_c) for r0, c0, r1, c1 in boxes]
    inp = _paint_frames((h, w), shifted)

    try:
        out = main(inp)
    except AssertionError:
        return None
    if not _semantic_ok(inp, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-frame puzzle input")
