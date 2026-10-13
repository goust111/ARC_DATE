import random
from collections import Counter
from itertools import permutations

import numpy as np


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _square_ring_mask(size: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _is_square_ring_component(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = bbox_coords(coords)
    size = r1 - r0 + 1
    if size != c1 - c0 + 1 or size < 5:
        return False
    rel = coords - np.array([r0, c0], dtype=int)
    mask = np.zeros((size, size), dtype=bool)
    mask[rel[:, 0], rel[:, 1]] = True
    return np.array_equal(mask, _square_ring_mask(size))


def _outside_contour(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    contour = np.zeros_like(mask, dtype=bool)
    for r, c in np.argwhere(mask):
        for nr, nc in neighbors4(int(r), int(c)):
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                contour[nr, nc] = True
    return contour


def _rarest_color(inner: np.ndarray) -> int:
    vals = [int(v) for v in inner.ravel() if int(v) != 0]
    counts = Counter(vals)
    least = min(counts.values())
    colors = [color for color, cnt in counts.items() if cnt == least]
    assert len(colors) == 1, "rarest interior color must be unique"
    return int(colors[0])


def _render_transformed_inner(inner: np.ndarray) -> np.ndarray:
    out = np.zeros_like(inner)
    contour = _outside_contour(inner != 0)
    out[contour] = _rarest_color(inner)
    return out


def _find_frames(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    frames = []
    candidates = []
    for color in sorted(set(int(v) for v in np.unique(g)) - {0}):
        for coords in _components(g == color):
            if _is_square_ring_component(coords):
                bbox = bbox_coords(coords)
                size = bbox[2] - bbox[0] + 1
                candidates.append((size, bbox, color))
    candidates.sort(key=lambda item: (-item[0], item[1][0], item[1][1], item[2]))

    for size, bbox, color in candidates:
        r0, c0, r1, c1 = bbox
        overlaps = False
        for frame in frames:
            fr0, fc0, fr1, fc1 = frame["bbox"]
            if not (r1 < fr0 or fr1 < r0 or c1 < fc0 or fc1 < c0):
                overlaps = True
                break
        if overlaps:
            continue
        patch = crop_bbox(g, bbox)
        frames.append(
            {
                "bbox": bbox,
                "size": int(size),
                "color": int(color),
                "patch": patch,
                "inner": patch[1:-1, 1:-1].copy(),
            }
        )
    return frames


def _infer_roles(frames: list[dict]) -> tuple[int, int, int]:
    matches = []
    for src_idx, dst_idx in permutations(range(len(frames)), 2):
        src_inner = frames[src_idx]["inner"]
        dst_inner = frames[dst_idx]["inner"]
        if src_inner.shape != dst_inner.shape or np.count_nonzero(src_inner) == 0:
            continue
        if np.array_equal(_render_transformed_inner(src_inner), dst_inner):
            matches.append((src_idx, dst_idx))
    assert len(matches) == 1, "expected exactly one training source->target pair"
    src_idx, dst_idx = matches[0]
    test_idx = [idx for idx in range(len(frames)) if idx not in matches[0]]
    assert len(test_idx) == 1
    return int(src_idx), int(dst_idx), int(test_idx[0])


def _build_output_frame(frame_color: int, inner: np.ndarray) -> np.ndarray:
    size = inner.shape[0] + 2
    out = np.zeros((size, size), dtype=int)
    out[0, :] = frame_color
    out[-1, :] = frame_color
    out[:, 0] = frame_color
    out[:, -1] = frame_color
    out[1:-1, 1:-1] = _render_transformed_inner(inner)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Learn the framed solid-to-outside-contour rule, then repaint the test frame with its own rarest brush color.

    Input structure:
    - The grid has black background and three disjoint hollow square frames in different colors.
    - Each frame contains a non-border interior pattern; two frames form a training source→target relation and the third is the test case.
    - The training source and the test show dense multicolor solid shapes, while the training target shows only the outside 4-neighbor contour.

    Rule:
    - Detect the three square frames and extract each interior.
    - Find the unique ordered pair where one interior becomes the other after erasing the solid shape and painting every background cell that is 4-adjacent to it.
    - The contour paint color is the unique rarest non-black color from the source interior.
    - Apply that same erase-and-trace rule to the remaining test interior, using the test interior's own rarest non-black color as the brush.
    - Return only the cropped test frame, preserving its border and replacing the interior with the painted outside contour.

    Key insight:
    - The training pair teaches the geometric operation, but the output color must be recomputed independently from color rarity inside the test frame.
    """
    g = np.asarray(grid, dtype=int)
    frames = _find_frames(g)
    assert len(frames) >= 3, "expected three square frames"
    frames = frames[:3]
    _src_idx, _dst_idx, test_idx = _infer_roles(frames)
    test_frame = frames[test_idx]
    return _build_output_frame(test_frame["color"], test_frame["inner"])


FRAME_SIZE = 9
INNER_SIZE = 7
CORE_SIZE = 5
GRID_MIN = 22
GRID_MAX = 30
MAX_ATTEMPTS = 400

CORE_LIBRARY = [
    (
        ".####",
        "#####",
        "#####",
        "#####",
        "#####",
    ),
    (
        "..###",
        "#####",
        "#####",
        "#####",
        "#####",
    ),
    (
        ".####",
        ".####",
        "#####",
        "#####",
        "#####",
    ),
    (
        "...##",
        "#####",
        "#####",
        "#####",
        "#####",
    ),
    (
        "..###",
        ".####",
        "#####",
        "#####",
        "#####",
    ),
    (
        "#.###",
        "#.###",
        "#####",
        "#####",
        "#####",
    ),
    (
        "...##",
        "##.##",
        "#####",
        "#####",
        "#####",
    ),
    (
        "....#",
        "#####",
        "#####",
        "#####",
        "#####",
    ),
]


def _core_from_lines(lines: tuple[str, ...]) -> np.ndarray:
    return np.array([[ch == "#" for ch in row] for row in lines], dtype=bool)


def _all_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants = []
    seen = set()
    for k in range(4):
        rotated = np.rot90(mask, k)
        for cand in (rotated, np.fliplr(rotated)):
            key = cand.tobytes()
            if key not in seen:
                seen.add(key)
                variants.append(cand.copy())
    return variants


def _make_shape_library() -> list[np.ndarray]:
    lib = []
    seen = set()
    for lines in CORE_LIBRARY:
        core = _core_from_lines(lines)
        for variant in _all_variants(core):
            mask = np.zeros((INNER_SIZE, INNER_SIZE), dtype=bool)
            mask[1 : 1 + CORE_SIZE, 1 : 1 + CORE_SIZE] = variant
            contour = _outside_contour(mask)
            if np.count_nonzero(mask) <= np.count_nonzero(contour):
                continue
            if _is_square_ring_component(np.argwhere(contour)):
                continue
            key = mask.tobytes()
            if key not in seen:
                seen.add(key)
                lib.append(mask)
    return lib


SHAPE_LIBRARY = _make_shape_library()


def _bbox_gap_ok(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _pick_frame_boxes(height: int, width: int) -> list[tuple[int, int, int, int]] | None:
    boxes: list[tuple[int, int, int, int]] = []
    for _ in range(500):
        r0 = random.randint(0, height - FRAME_SIZE)
        c0 = random.randint(0, width - FRAME_SIZE)
        box = (r0, c0, r0 + FRAME_SIZE - 1, c0 + FRAME_SIZE - 1)
        if all(_bbox_gap_ok(box, other, gap=1) for other in boxes):
            boxes.append(box)
            if len(boxes) == 3:
                random.shuffle(boxes)
                return boxes
    return None


def _bfs_order(mask: np.ndarray, start: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = mask.shape
    order = []
    seen = {start}
    queue = [start]
    while queue:
        rr, cc = queue.pop(0)
        order.append((rr, cc))
        nbrs = []
        for nr, nc in neighbors4(rr, cc):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                nbrs.append((nr, nc))
        nbrs.sort()
        for cell in nbrs:
            seen.add(cell)
            queue.append(cell)
    return order


def _farthest_cell(coords: list[tuple[int, int]], anchors: list[tuple[int, int]]) -> tuple[int, int]:
    def score(cell: tuple[int, int]) -> tuple[int, int, int]:
        dist = min(abs(cell[0] - ar) + abs(cell[1] - ac) for ar, ac in anchors)
        return (dist, -cell[0], -cell[1])

    return max(coords, key=score)


def _pick_color_counts(area: int, n_colors: int) -> list[int]:
    for _ in range(200):
        rare = random.randint(1, min(4, area - 2))
        if n_colors == 2:
            common = area - rare
            if common > rare:
                return [rare, common]
            continue
        mid_min = rare + 1
        mid_max = min(8, area - rare - 1)
        if mid_min > mid_max:
            continue
        mid = random.randint(mid_min, mid_max)
        common = area - rare - mid
        if common > mid and common != rare:
            return [rare, mid, common]
    if n_colors == 2:
        return [2, area - 2]
    return [2, 5, area - 7]


def _color_shape(mask: np.ndarray, palette: list[int]) -> np.ndarray:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    area = len(coords)
    counts = _pick_color_counts(area, len(palette))
    counts.sort()

    seeds = [random.choice(coords)]
    if len(palette) >= 2:
        seeds.append(_farthest_cell(coords, seeds))
    if len(palette) >= 3:
        seeds.append(_farthest_cell(coords, seeds))
    random.shuffle(seeds)

    inner = np.zeros(mask.shape, dtype=int)
    remaining = set(coords)
    for color, count, seed in zip(palette[:-1], counts[:-1], seeds[:-1]):
        order = [cell for cell in _bfs_order(mask, seed) if cell in remaining]
        if len(order) < count:
            order.extend(sorted(remaining - set(order)))
        chosen = order[:count]
        for cell in chosen:
            inner[cell] = int(color)
            remaining.remove(cell)
    for cell in remaining:
        inner[cell] = int(palette[-1])
    return inner


def _frame_patch(frame_color: int, inner: np.ndarray) -> np.ndarray:
    patch = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=int)
    patch[0, :] = frame_color
    patch[-1, :] = frame_color
    patch[:, 0] = frame_color
    patch[:, -1] = frame_color
    patch[1:-1, 1:-1] = inner
    return patch


def _partition_colors() -> tuple[list[int], list[int], list[int]]:
    colors = list(range(1, 10))
    random.shuffle(colors)
    frame_colors = colors[:3]
    remaining = colors[3:]
    train_count = random.choice([2, 3])
    test_count = 2 if train_count == 3 else random.choice([2, 3])
    train_palette = remaining[:train_count]
    test_palette = remaining[train_count : train_count + test_count]
    random.shuffle(train_palette)
    random.shuffle(test_palette)
    return frame_colors, train_palette, test_palette


def _render_grid(
    height: int,
    width: int,
    boxes: list[tuple[int, int, int, int]],
    roles: list[tuple[int, np.ndarray]],
) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    for bbox, (frame_color, inner) in zip(boxes, roles):
        r0, c0, _r1, _c1 = bbox
        patch = _frame_patch(frame_color, inner)
        grid[r0 : r0 + FRAME_SIZE, c0 : c0 + FRAME_SIZE] = patch
    return grid


def _semantic_checks(grid: np.ndarray) -> bool:
    frames = _find_frames(grid)
    if len(frames) != 3:
        return False
    try:
        src_idx, dst_idx, test_idx = _infer_roles(frames)
    except AssertionError:
        return False

    src_inner = frames[src_idx]["inner"]
    dst_inner = frames[dst_idx]["inner"]
    test_inner = frames[test_idx]["inner"]
    train_palette = set(int(v) for v in np.unique(src_inner) if int(v) != 0)
    test_palette = set(int(v) for v in np.unique(test_inner) if int(v) != 0)

    if train_palette & {frames[src_idx]["color"], frames[dst_idx]["color"], frames[test_idx]["color"]}:
        return False
    if test_palette & {frames[src_idx]["color"], frames[dst_idx]["color"], frames[test_idx]["color"]}:
        return False
    if train_palette == test_palette:
        return False
    if _rarest_color(src_inner) == _rarest_color(test_inner):
        return False
    if np.count_nonzero(src_inner) <= np.count_nonzero(dst_inner):
        return False

    expected = _build_output_frame(frames[test_idx]["color"], test_inner)
    if not np.array_equal(main(grid), expected):
        return False

    train_target_patch = frames[dst_idx]["patch"]
    if np.array_equal(expected, train_target_patch):
        return False

    wrong_brush = _build_output_frame(frames[test_idx]["color"], test_inner)
    wrong_brush[1:-1, 1:-1][_outside_contour(test_inner != 0)] = _rarest_color(src_inner)
    if np.array_equal(wrong_brush, expected):
        return False

    if np.array_equal(_outside_contour(src_inner != 0), _outside_contour(test_inner != 0)):
        return False
    return True


def _attempt_instance() -> np.ndarray | None:
    height = random.randint(GRID_MIN, GRID_MAX)
    width = random.randint(GRID_MIN, GRID_MAX)
    boxes = _pick_frame_boxes(height, width)
    if boxes is None:
        return None

    frame_colors, train_palette, test_palette = _partition_colors()
    src_mask = random.choice(SHAPE_LIBRARY)
    test_mask = random.choice(SHAPE_LIBRARY)

    for _ in range(40):
        if np.array_equal(src_mask, test_mask):
            test_mask = random.choice(SHAPE_LIBRARY)
        src_inner = _color_shape(src_mask, train_palette)
        dst_inner = _render_transformed_inner(src_inner)
        test_inner = _color_shape(test_mask, test_palette)
        if np.array_equal(_outside_contour(src_mask), _outside_contour(test_mask)):
            continue
        roles = [
            (frame_colors[0], src_inner),
            (frame_colors[1], dst_inner),
            (frame_colors[2], test_inner),
        ]
        random.shuffle(roles)
        grid = _render_grid(height, width, boxes, roles)
        if _semantic_checks(grid):
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _attempt_instance()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
