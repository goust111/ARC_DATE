from __future__ import annotations

import numpy as np


BLACK = 0
BLUE = 1
DEPTH_TO_COLOR = {0: 2, 1: 3, 2: 4, 3: 5}
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
MAX_GENERATION_ATTEMPTS = 500


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components of a boolean mask as coordinate arrays."""
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _frame_from_coords(grid: np.ndarray, coords: np.ndarray) -> dict:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    assert height >= 3 and width >= 3, "frames must be at least 3x3"

    patch = np.zeros((height, width), dtype=bool)
    rel = coords - np.array([r0, c0], dtype=int)
    patch[rel[:, 0], rel[:, 1]] = True
    ring = np.zeros((height, width), dtype=bool)
    ring[0, :] = True
    ring[-1, :] = True
    ring[:, 0] = True
    ring[:, -1] = True
    assert np.array_equal(patch, ring), "each blue object must be a complete 1-pixel rectangular frame"

    return {
        "bbox": (r0, c0, r1, c1),
        "height": height,
        "width": width,
        "area": height * width,
        "center_r2": r0 + r1,
        "center_c2": c0 + c1,
    }


def _find_frames(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    colors = sorted(set(map(int, g.flatten())))
    assert colors == [BLACK, BLUE] or colors == [BLACK], "expected only black background and blue frames"

    frames = [_frame_from_coords(g, comp) for comp in _components(g == BLUE)]
    assert frames, "expected at least one blue frame"
    return frames


def _contains(outer: dict, inner: dict) -> bool:
    r0, c0, r1, c1 = outer["bbox"]
    rr0, cc0, rr1, cc1 = inner["bbox"]
    return r0 < rr0 and c0 < cc0 and r1 > rr1 and c1 > cc1


def _depths(frames: list[dict]) -> list[int]:
    return [
        sum(_contains(other, frame) for other in frames if other is not frame)
        for frame in frames
    ]


def _axis_ranks(values: list[int]) -> dict[int, int]:
    ordered = sorted(set(map(int, values)))
    return {value: idx for idx, value in enumerate(ordered)}


def _compress(frames: list[dict], depths: list[int]) -> np.ndarray:
    row_rank = _axis_ranks([frame["center_r2"] for frame in frames])
    col_rank = _axis_ranks([frame["center_c2"] for frame in frames])
    out = np.zeros((len(row_rank), len(col_rank)), dtype=int)
    for frame, depth in zip(frames, depths):
        assert depth in DEPTH_TO_COLOR, "supported nesting depths are 0..3"
        rr = row_rank[int(frame["center_r2"])]
        cc = col_rank[int(frame["center_c2"])]
        assert out[rr, cc] == BLACK, "each compressed cell must correspond to exactly one frame"
        out[rr, cc] = DEPTH_TO_COLOR[depth]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress nested blue rectangular frames into a small grid whose colors encode nesting depth.

    Input structure:
    - The canvas is a black background with several blue 1-pixel-thick hollow rectangles.
    - Every blue connected component is a complete rectangular frame, and smaller frames may sit strictly inside larger ones.
    - Different frame groups can appear at different positions, so equal input color does not reveal the output color.

    Rule:
    - Detect every blue rectangular frame and recover its bounding box.
    - The depth of a frame is the number of larger frames that completely contain it.
    - Compress the frames by their bounding-box row and column positions: order the distinct vertical centers from top to bottom and the distinct horizontal centers from left to right.
    - Put one colored cell at the corresponding compressed row/column position using depth 0->red(2), 1->green(3), 2->yellow(4), 3->grey(5); empty cells stay black.

    Key insight:
    - All input frames share the same visible color, so the output depends on hidden containment depth plus spatial compression rather than on any frame's original color.
    """
    frames = _find_frames(np.asarray(grid, dtype=int))
    depths = _depths(frames)
    return _compress(frames, depths)


def _box_size(box: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    return r1 - r0 + 1, c1 - c0 + 1


def _place_inside_pair(
    rng: np.random.Generator,
    parent: tuple[int, int, int, int],
    top_right_size: tuple[int, int],
    bottom_left_size: tuple[int, int],
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    pr0, pc0, pr1, pc1 = parent
    ph, pw = _box_size(parent)
    trh, trw = top_right_size
    blh, blw = bottom_left_size

    min_h = trh + blh + 5
    min_w = trw + blw + 5
    assert ph >= min_h and pw >= min_w, "parent too small for opposite-corner children"

    slack_r = ph - min_h
    slack_c = pw - min_w
    top_jitter = int(rng.integers(0, slack_r + 1))
    left_jitter = int(rng.integers(0, slack_c + 1))
    bottom_jitter = slack_r - top_jitter
    right_jitter = slack_c - left_jitter

    tr0 = pr0 + 2 + top_jitter
    tc0 = pc1 - trw + 1 - 2 - right_jitter
    br0 = pr1 - blh + 1 - 2 - bottom_jitter
    bc0 = pc0 + 2 + left_jitter

    top_right = (tr0, tc0, tr0 + trh - 1, tc0 + trw - 1)
    bottom_left = (br0, bc0, br0 + blh - 1, bc0 + blw - 1)
    return top_right, bottom_left


def _place_child(
    rng: np.random.Generator,
    parent: tuple[int, int, int, int],
    size: tuple[int, int],
    anchor: str,
) -> tuple[int, int, int, int]:
    pr0, pc0, pr1, pc1 = parent
    ph, pw = _box_size(parent)
    ch, cw = size
    assert ph >= ch + 4 and pw >= cw + 4, "parent too small for nested child"

    if anchor not in {"tr", "bl", "tl", "br"}:
        raise ValueError(f"unknown anchor {anchor}")

    row_slack = ph - ch - 4
    col_slack = pw - cw - 4

    if anchor[0] == "t":
        r0 = pr0 + 2 + int(rng.integers(0, row_slack // 2 + 1))
    else:
        r0 = pr1 - ch + 1 - 2 - int(rng.integers(0, row_slack // 2 + 1))

    if anchor[1] == "l":
        c0 = pc0 + 2 + int(rng.integers(0, col_slack // 2 + 1))
    else:
        c0 = pc1 - cw + 1 - 2 - int(rng.integers(0, col_slack // 2 + 1))

    return (r0, c0, r0 + ch - 1, c0 + cw - 1)


def _rand_hw(
    rng: np.random.Generator,
    h_lo: int,
    h_hi: int,
    w_lo: int,
    w_hi: int,
) -> tuple[int, int]:
    return int(rng.integers(h_lo, h_hi + 1)), int(rng.integers(w_lo, w_hi + 1))


def _draw_boxes(shape: tuple[int, int], boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for r0, c0, r1, c1 in boxes:
        grid[r0, c0 : c1 + 1] = BLUE
        grid[r1, c0 : c1 + 1] = BLUE
        grid[r0 : r1 + 1, c0] = BLUE
        grid[r0 : r1 + 1, c1] = BLUE
    return grid


def _build_single_root_three(rng: np.random.Generator) -> list[tuple[int, int, int, int]]:
    child_tr = _rand_hw(rng, 3, 6, 4, 7)
    child_bl = _rand_hw(rng, 4, 7, 3, 6)
    root_h = child_tr[0] + child_bl[0] + int(rng.integers(5, 9))
    root_w = child_tr[1] + child_bl[1] + int(rng.integers(5, 9))
    canvas_h = root_h + int(rng.integers(2, 6))
    canvas_w = root_w + int(rng.integers(2, 6))
    r0 = int(rng.integers(1, canvas_h - root_h + 1))
    c0 = int(rng.integers(1, canvas_w - root_w + 1))
    root = (r0, c0, r0 + root_h - 1, c0 + root_w - 1)
    top_right, bottom_left = _place_inside_pair(rng, root, child_tr, child_bl)
    return [root, top_right, bottom_left]


def _build_single_root_four(rng: np.random.Generator) -> list[tuple[int, int, int, int]]:
    deep = _rand_hw(rng, 3, 5, 3, 5)
    branch = (deep[0] + int(rng.integers(4, 7)), deep[1] + int(rng.integers(4, 7)))
    peer = _rand_hw(rng, 4, 7, 4, 7)
    root_h = branch[0] + peer[0] + int(rng.integers(5, 9))
    root_w = branch[1] + peer[1] + int(rng.integers(5, 9))
    canvas_h = root_h + int(rng.integers(2, 6))
    canvas_w = root_w + int(rng.integers(2, 6))
    r0 = int(rng.integers(1, canvas_h - root_h + 1))
    c0 = int(rng.integers(1, canvas_w - root_w + 1))
    root = (r0, c0, r0 + root_h - 1, c0 + root_w - 1)
    peer_box, branch_box = _place_inside_pair(rng, root, peer, branch)
    deep_box = _place_child(rng, branch_box, deep, "tr")
    return [root, peer_box, branch_box, deep_box]


def _build_single_root_five(rng: np.random.Generator) -> list[tuple[int, int, int, int]]:
    deepest = _rand_hw(rng, 3, 4, 3, 4)
    depth2 = (deepest[0] + int(rng.integers(4, 7)), deepest[1] + int(rng.integers(4, 7)))
    depth1_main = (depth2[0] + int(rng.integers(4, 7)), depth2[1] + int(rng.integers(4, 7)))
    depth1_peer = _rand_hw(rng, 5, 7, 5, 7)
    root_h = depth1_main[0] + depth1_peer[0] + int(rng.integers(5, 8))
    root_w = depth1_main[1] + depth1_peer[1] + int(rng.integers(5, 8))
    canvas_h = root_h + int(rng.integers(2, 5))
    canvas_w = root_w + int(rng.integers(2, 5))
    r0 = int(rng.integers(1, canvas_h - root_h + 1))
    c0 = int(rng.integers(1, canvas_w - root_w + 1))
    root = (r0, c0, r0 + root_h - 1, c0 + root_w - 1)
    peer_box, main_box = _place_inside_pair(rng, root, depth1_peer, depth1_main)
    depth2_box = _place_child(rng, main_box, depth2, "tr")
    deepest_box = _place_child(rng, depth2_box, deepest, "bl")
    return [root, peer_box, main_box, depth2_box, deepest_box]


def _build_two_roots_four(rng: np.random.Generator) -> list[tuple[int, int, int, int]]:
    child1_size = _rand_hw(rng, 3, 5, 3, 5)
    child2_size = _rand_hw(rng, 3, 5, 3, 5)
    root1_size = (
        int(rng.integers(child1_size[0] + 4, 13)),
        int(rng.integers(child1_size[1] + 4, 13)),
    )
    root2_size = (
        int(rng.integers(child2_size[0] + 4, 13)),
        int(rng.integers(child2_size[1] + 4, 13)),
    )

    canvas_h = root1_size[0] + root2_size[0] + int(rng.integers(5, 9))
    canvas_w = root1_size[1] + root2_size[1] + int(rng.integers(5, 9))
    root1 = (1, 1, root1_size[0], root1_size[1])
    root2 = (
        canvas_h - root2_size[0] - 1,
        canvas_w - root2_size[1] - 1,
        canvas_h - 2,
        canvas_w - 2,
    )
    child1 = _place_child(rng, root1, child1_size, "tr")
    child2 = _place_child(rng, root2, child2_size, "bl")
    return [root1, child1, root2, child2]


def _build_two_roots_five(rng: np.random.Generator) -> list[tuple[int, int, int, int]]:
    peer = _rand_hw(rng, 3, 5, 3, 5)
    branch = _rand_hw(rng, 4, 6, 4, 6)
    root1_h = peer[0] + branch[0] + int(rng.integers(5, 8))
    root1_w = peer[1] + branch[1] + int(rng.integers(5, 8))
    root1_size = (root1_h, root1_w)

    child2_size = _rand_hw(rng, 3, 5, 3, 5)
    root2_size = (
        int(rng.integers(child2_size[0] + 4, 12)),
        int(rng.integers(child2_size[1] + 4, 12)),
    )

    canvas_h = max(root1_size[0], root2_size[0]) + int(rng.integers(8, 13))
    canvas_w = root1_size[1] + root2_size[1] + int(rng.integers(6, 10))
    root1 = (1, 1, root1_size[0], root1_size[1])
    root2 = (
        canvas_h - root2_size[0] - 1,
        canvas_w - root2_size[1] - 1,
        canvas_h - 2,
        canvas_w - 2,
    )
    child1_peer, child1_branch = _place_inside_pair(rng, root1, peer, branch)
    child2 = _place_child(rng, root2, child2_size, "bl")
    return [root1, child1_peer, child1_branch, root2, child2]


def _target_builders():
    return [
        (_build_single_root_three, 0.15),
        (_build_single_root_four, 0.25),
        (_build_single_root_five, 0.35),
        (_build_two_roots_four, 0.10),
        (_build_two_roots_five, 0.15),
    ]


def _choose_builder(rng: np.random.Generator):
    builders = _target_builders()
    weights = np.array([weight for _builder, weight in builders], dtype=float)
    weights /= weights.sum()
    idx = int(rng.choice(len(builders), p=weights))
    return builders[idx][0]


def _boxes_fit_limit(boxes: list[tuple[int, int, int, int]]) -> bool:
    return all(0 <= r0 < r1 < 30 and 0 <= c0 < c1 < 30 for r0, c0, r1, c1 in boxes)


def _boxes_set(boxes: list[tuple[int, int, int, int]]) -> set[tuple[int, int, int, int]]:
    return {tuple(map(int, box)) for box in boxes}


def _intended_depths(boxes: list[tuple[int, int, int, int]]) -> dict[tuple[int, int, int, int], int]:
    frames = [{"bbox": box} for box in boxes]
    depths = _depths(frames)
    return {box: depth for box, depth in zip(boxes, depths)}


def _semantic_check(grid: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if not _boxes_fit_limit(boxes):
        return False

    try:
        frames = _find_frames(grid)
        out = main(grid)
    except Exception:
        return False

    frame_boxes = [tuple(map(int, frame["bbox"])) for frame in frames]
    if _boxes_set(frame_boxes) != _boxes_set(boxes):
        return False

    intended_depths = _intended_depths(boxes)
    detected_depths = {
        tuple(map(int, frame["bbox"])): depth
        for frame, depth in zip(frames, _depths(frames))
    }
    if intended_depths != detected_depths:
        return False

    if np.count_nonzero(out) != len(boxes):
        return False
    if len(set(map(int, out[out != 0].flatten()))) < 2:
        return False
    if out.size <= len(boxes):
        return False

    box_areas = [(box[2] - box[0] + 1) * (box[3] - box[1] + 1) for box in boxes]
    if len(set(box_areas)) == 1:
        return False

    max_depth = max(intended_depths.values())
    if max_depth >= 3 and not np.any(out == DEPTH_TO_COLOR[3]):
        return False
    if max_depth >= 2 and not np.any(out == DEPTH_TO_COLOR[2]):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        rng = np.random.default_rng()
        builder = _choose_builder(rng)
        boxes = builder(rng)
        if not (3 <= len(boxes) <= 5):
            continue
        if not _boxes_fit_limit(boxes):
            continue
        height = max(box[2] for box in boxes) + 1
        width = max(box[3] for box in boxes) + 1
        if height > 30 or width > 30:
            continue
        grid = _draw_boxes((height, width), boxes)
        if _semantic_check(grid, boxes):
            return grid
    raise RuntimeError("failed to generate a valid nested-frame ARC input")
