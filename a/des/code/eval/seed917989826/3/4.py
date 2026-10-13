import numpy as np

from collections import defaultdict, deque
from dataclasses import dataclass


BG = 0
GREY = 5
FRAME_COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_RETRIES = 400
RNG = np.random.default_rng()


@dataclass(frozen=True)
class Component:
    color: int
    coords: np.ndarray
    bbox: tuple[int, int, int, int]

    @property
    def size(self) -> int:
        return int(self.coords.shape[0])


@dataclass(frozen=True)
class Frame:
    color: int
    bbox: tuple[int, int, int, int]

    @property
    def area(self) -> int:
        r0, c0, r1, c1 = self.bbox
        return int((r1 - r0 + 1) * (c1 - c0 + 1))


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    points = np.argwhere(mask)
    if points.size == 0:
        return None
    return _bbox_from_coords(points)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return np.array(grid[r0 : r1 + 1, c0 : c1 + 1], copy=True)


def _interior_bbox(frame_bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = frame_bbox
    return int(r0 + 1), int(c0 + 1), int(r1 - 1), int(c1 - 1)


def _point_in_bbox(point: tuple[int, int], bbox: tuple[int, int, int, int]) -> bool:
    r, c = point
    r0, c0, r1, c1 = bbox
    return bool(r0 <= r <= r1 and c0 <= c <= c1)


def _contains_frame(parent: tuple[int, int, int, int], child: tuple[int, int, int, int]) -> bool:
    pr0, pc0, pr1, pc1 = parent
    cr0, cc0, cr1, cc1 = child
    return bool(pr0 + 1 < cr0 and pc0 + 1 < cc0 and cr1 < pr1 - 1 and cc1 < pc1 - 1)


def _find_components(grid: np.ndarray, bg: int) -> list[Component]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    components: list[Component] = []

    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(grid[r, c]) == int(bg):
                continue
            color = int(grid[r, c])
            q = deque([(int(r), int(c))])
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if visited[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    q.append((nr, nc))
            coord_arr = np.array(coords, dtype=int)
            components.append(Component(color=color, coords=coord_arr, bbox=_bbox_from_coords(coord_arr)))
    return components


def _is_rect_border(component: Component) -> bool:
    r0, c0, r1, c1 = component.bbox
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    if height < 3 or width < 3:
        return False
    expected = 2 * height + 2 * width - 4
    if component.size != expected:
        return False
    pts = {tuple(map(int, pt)) for pt in component.coords}
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            on_border = r in (r0, r1) or c in (c0, c1)
            if on_border != ((r, c) in pts):
                return False
    return True


def _longest_nested_chain(frames: list[Frame]) -> list[Frame]:
    if not frames:
        return []

    ordered = sorted(frames, key=lambda frame: (-frame.area, frame.bbox))
    contains: list[list[int]] = [[] for _ in ordered]
    for i, outer in enumerate(ordered):
        for j, inner in enumerate(ordered):
            if outer.area <= inner.area:
                continue
            if _contains_frame(outer.bbox, inner.bbox):
                contains[i].append(j)

    memo: dict[int, list[int]] = {}

    def best_from(idx: int) -> list[int]:
        if idx in memo:
            return memo[idx]
        best = [idx]
        for child_idx in contains[idx]:
            candidate = [idx] + best_from(child_idx)
            candidate_key = (
                len(candidate),
                ordered[candidate[0]].area,
                -ordered[candidate[0]].bbox[0],
                -ordered[candidate[0]].bbox[1],
            )
            best_key = (
                len(best),
                ordered[best[0]].area,
                -ordered[best[0]].bbox[0],
                -ordered[best[0]].bbox[1],
            )
            if candidate_key > best_key:
                best = candidate
        memo[idx] = best
        return best

    best_path = max((best_from(i) for i in range(len(ordered))), key=lambda path: (len(path), ordered[path[0]].area))
    return [ordered[idx] for idx in best_path]


def _read_marker_sequence(
    singletons: dict[int, list[tuple[int, int]]],
    frames: list[Frame],
) -> list[int]:
    frame_colors = {frame.color for frame in frames}
    by_col: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for color in frame_colors:
        for point in singletons.get(color, []):
            if any(_point_in_bbox(point, frame.bbox) for frame in frames):
                continue
            by_col[int(point[1])].append((int(point[0]), int(color)))

    candidates: list[tuple[int, int, list[int]]] = []
    target_len = len(frames)
    for col, items in by_col.items():
        items.sort()
        for start in range(len(items)):
            seq = [items[start]]
            for idx in range(start + 1, len(items)):
                if items[idx][0] - seq[-1][0] == 2:
                    seq.append(items[idx])
                elif items[idx][0] - seq[-1][0] > 2:
                    break
            colors = [color for _, color in seq]
            if len(seq) != target_len:
                continue
            if len(set(colors)) != target_len:
                continue
            if set(colors) != frame_colors:
                continue
            candidates.append((int(col), int(seq[0][0]), colors))

    if not candidates:
        raise ValueError("failed to locate the top-left frame-color sequence")
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2]


def _detect_pattern_bbox(
    grid: np.ndarray,
    bg: int,
    frames: list[Frame],
) -> tuple[int, int, int, int]:
    ir0, ic0, ir1, ic1 = _interior_bbox(frames[-1].bbox)
    inner = grid[ir0 : ir1 + 1, ic0 : ic1 + 1]
    frame_colors = np.array([frame.color for frame in frames], dtype=int)
    mask = (inner != int(bg)) & (~np.isin(inner, frame_colors))
    if not np.any(mask):
        mask = inner != int(bg)
    bbox = _bbox_from_mask(mask)
    if bbox is None:
        raise ValueError("innermost frame must contain a pattern")
    pr0, pc0, pr1, pc1 = bbox
    return int(ir0 + pr0), int(ic0 + pc0), int(ir0 + pr1), int(ic0 + pc1)


def _parse_instance(grid: np.ndarray) -> tuple[int, list[Frame], list[int], tuple[int, int, int, int]]:
    bg = _most_common_color(grid)
    components = _find_components(grid, bg)

    singletons: dict[int, list[tuple[int, int]]] = defaultdict(list)
    frame_candidates: list[Frame] = []
    for comp in components:
        if comp.size == 1:
            point = tuple(map(int, comp.coords[0]))
            singletons[comp.color].append(point)
        if _is_rect_border(comp):
            frame_candidates.append(Frame(color=comp.color, bbox=comp.bbox))

    frame_candidates = [frame for frame in frame_candidates if singletons.get(frame.color)]
    frames = _longest_nested_chain(frame_candidates)
    if len(frames) < 2:
        raise ValueError("expected a nested chain of at least two frames")

    sequence = _read_marker_sequence(singletons, frames)
    pattern_bbox = _detect_pattern_bbox(grid, bg, frames)
    return int(bg), frames, sequence, pattern_bbox


def _extract_ring(
    grid: np.ndarray,
    bg: int,
    frame_bbox: tuple[int, int, int, int],
    child_bbox: tuple[int, int, int, int],
) -> np.ndarray:
    ir0, ic0, ir1, ic1 = _interior_bbox(frame_bbox)
    sub = np.array(grid[ir0 : ir1 + 1, ic0 : ic1 + 1], copy=True)
    geom = np.ones(sub.shape, dtype=bool)

    cr0, cc0, cr1, cc1 = child_bbox
    rr0 = int(cr0 - ir0)
    cc0_rel = int(cc0 - ic0)
    rr1 = int(cr1 - ir0)
    cc1_rel = int(cc1 - ic0)
    sub[rr0 : rr1 + 1, cc0_rel : cc1_rel + 1] = int(bg)
    geom[rr0 : rr1 + 1, cc0_rel : cc1_rel + 1] = False

    crop = _bbox_from_mask(geom)
    if crop is None:
        raise ValueError("ring area must be non-empty")
    return _crop_bbox(sub, crop)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify the nested colored frame chain, reorder layers by the top-left color legend,
    and horizontally lay out each layer's ring content followed by the innermost pattern.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")

    bg, frames, sequence, pattern_bbox = _parse_instance(g)
    frame_by_color = {frame.color: frame for frame in frames}

    parts: list[np.ndarray] = []
    for color in sequence:
        frame = frame_by_color[int(color)]
        depth = frames.index(frame)
        child_bbox = frames[depth + 1].bbox if depth + 1 < len(frames) else pattern_bbox
        parts.append(_extract_ring(g, bg, frame.bbox, child_bbox))

    parts.append(_crop_bbox(g, pattern_bbox))

    out_h = max(part.shape[0] for part in parts)
    out_w = sum(part.shape[1] for part in parts) + (len(parts) - 1)
    out = np.full((out_h, out_w), int(bg), dtype=int)

    col = 0
    for idx, part in enumerate(parts):
        h, w = part.shape
        out[:h, col : col + w] = part
        col += w
        if idx != len(parts) - 1:
            out[:, col] = GREY
            col += 1

    return out


def _rand_int(low: int, high: int) -> int:
    return int(RNG.integers(low, high + 1))


def _choose_colors(pool: tuple[int, ...], count: int) -> list[int]:
    return [int(v) for v in RNG.choice(pool, size=count, replace=False)]


def _make_pattern(height: int, width: int, colors: list[int]) -> np.ndarray:
    while True:
        density = float(RNG.uniform(0.35, 0.7))
        mask = RNG.random((height, width)) < density
        for r in range(height):
            if not np.any(mask[r]):
                mask[r, _rand_int(0, width - 1)] = True
        for c in range(width):
            if not np.any(mask[:, c]):
                mask[_rand_int(0, height - 1), c] = True
        if not np.any(mask[0]):
            mask[0, _rand_int(0, width - 1)] = True
        if not np.any(mask[-1]):
            mask[-1, _rand_int(0, width - 1)] = True
        if not np.any(mask[:, 0]):
            mask[_rand_int(0, height - 1), 0] = True
        if not np.any(mask[:, -1]):
            mask[_rand_int(0, height - 1), width - 1] = True
        if int(mask.sum()) < 2:
            continue
        if int(mask.sum()) == height * width and height * width > 4:
            continue
        break

    pattern = np.full((height, width), BG, dtype=int)
    color_arr = np.array(colors, dtype=int)
    chosen = RNG.choice(color_arr, size=int(mask.sum()), replace=True)
    pattern[mask] = chosen
    return pattern


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)


def _ring_mask_from_boxes(
    frame_bbox: tuple[int, int, int, int],
    child_bbox: tuple[int, int, int, int],
) -> tuple[int, int, np.ndarray]:
    ir0, ic0, ir1, ic1 = _interior_bbox(frame_bbox)
    mask = np.ones((ir1 - ir0 + 1, ic1 - ic0 + 1), dtype=bool)
    cr0, cc0, cr1, cc1 = child_bbox
    mask[cr0 - ir0 : cr1 - ir0 + 1, cc0 - ic0 : cc1 - ic0 + 1] = False
    return int(ir0), int(ic0), mask


def _place_decorations(
    grid: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    child_bbox: tuple[int, int, int, int],
    colors: list[int],
    min_count: int,
    max_count: int,
) -> None:
    base_r, base_c, allowed = _ring_mask_from_boxes(frame_bbox, child_bbox)
    occupied = np.zeros_like(allowed, dtype=bool)
    shape_options = [(1, 2), (2, 1), (2, 2), (1, 3), (3, 1), (2, 3), (3, 2)]
    target = _rand_int(min_count, max_count)
    placed = 0

    for _ in range(60):
        if placed >= target:
            break
        sh, sw = shape_options[_rand_int(0, len(shape_options) - 1)]
        if sh > allowed.shape[0] or sw > allowed.shape[1]:
            continue
        rr = _rand_int(0, allowed.shape[0] - sh)
        cc = _rand_int(0, allowed.shape[1] - sw)
        region = allowed[rr : rr + sh, cc : cc + sw]
        if not np.all(region):
            continue
        if np.any(occupied[rr : rr + sh, cc : cc + sw]):
            continue
        color = colors[_rand_int(0, len(colors) - 1)]
        grid[base_r + rr : base_r + rr + sh, base_c + cc : base_c + cc + sw] = int(color)
        occupied[rr : rr + sh, cc : cc + sw] = True
        placed += 1


def _build_nested_layout(depth: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]], tuple[int, int], tuple[int, int]]:
    if depth == 4:
        pattern_h = _rand_int(2, 4)
        pattern_w = _rand_int(2, 4)
    else:
        pattern_h = _rand_int(2, 4)
        pattern_w = _rand_int(2, 5)

    pad_top = _rand_int(1, 2)
    pad_bottom = _rand_int(1, 2)
    pad_left = _rand_int(1, 2)
    pad_right = _rand_int(1, 2)

    sizes = [(0, 0)] * depth
    child_offsets = [(0, 0)] * (depth - 1)

    inner_h = pattern_h + pad_top + pad_bottom + 2
    inner_w = pattern_w + pad_left + pad_right + 2
    sizes[depth - 1] = (inner_h, inner_w)

    child_h, child_w = inner_h, inner_w
    for idx in range(depth - 2, -1, -1):
        gap_top = _rand_int(1, 2)
        gap_bottom = _rand_int(1, 2)
        gap_left = _rand_int(1, 2)
        gap_right = _rand_int(1, 2)
        parent_h = child_h + gap_top + gap_bottom + 2
        parent_w = child_w + gap_left + gap_right + 2
        sizes[idx] = (parent_h, parent_w)
        child_offsets[idx] = (gap_top, gap_left)
        child_h, child_w = parent_h, parent_w

    return sizes, child_offsets, (pad_top, pad_left), (pattern_h, pattern_w)


def _construct_grid() -> np.ndarray | None:
    depth = [2, 3, 4][_rand_int(0, 2)]
    frame_colors = _choose_colors(FRAME_COLOR_POOL, depth)
    free_colors = [color for color in FRAME_COLOR_POOL if color not in frame_colors]
    if len(free_colors) < 2:
        return None

    sizes, child_offsets, pattern_offset, pattern_shape = _build_nested_layout(depth)
    outer_h, outer_w = sizes[0]
    if outer_h > 24 or outer_w > 26:
        return None

    marker_col = _rand_int(0, 1)
    marker_row0 = _rand_int(0, 1)
    marker_last_row = marker_row0 + 2 * (depth - 1)

    top_margin = _rand_int(1, 5)
    left_margin = marker_col + _rand_int(6, 11)
    bottom_margin = _rand_int(1, 5)
    right_margin = _rand_int(1, 5)

    grid_h = max(top_margin + outer_h + bottom_margin, marker_last_row + _rand_int(2, 5))
    grid_w = left_margin + outer_w + right_margin
    grid = np.full((grid_h, grid_w), BG, dtype=int)

    frame_bboxes: list[tuple[int, int, int, int]] = [(-1, -1, -1, -1)] * depth
    outer_r0 = top_margin
    outer_c0 = left_margin
    frame_bboxes[0] = (outer_r0, outer_c0, outer_r0 + outer_h - 1, outer_c0 + outer_w - 1)

    for idx in range(depth - 1):
        parent = frame_bboxes[idx]
        child_h, child_w = sizes[idx + 1]
        gap_top, gap_left = child_offsets[idx]
        child_r0 = parent[0] + 1 + gap_top
        child_c0 = parent[1] + 1 + gap_left
        frame_bboxes[idx + 1] = (child_r0, child_c0, child_r0 + child_h - 1, child_c0 + child_w - 1)

    for color, bbox in zip(frame_colors, frame_bboxes):
        _draw_frame(grid, bbox, color)

    for idx in range(depth - 1):
        min_count = 1 if idx == 0 else 0
        max_count = 3 if idx == 0 else 2
        _place_decorations(grid, frame_bboxes[idx], frame_bboxes[idx + 1], free_colors, min_count, max_count)

    inner_frame = frame_bboxes[-1]
    inner_r0, inner_c0, _, _ = _interior_bbox(inner_frame)
    pad_top, pad_left = pattern_offset
    pattern_h, pattern_w = pattern_shape
    pattern_r0 = inner_r0 + pad_top
    pattern_c0 = inner_c0 + pad_left
    pattern_r1 = pattern_r0 + pattern_h - 1
    pattern_c1 = pattern_c0 + pattern_w - 1
    pattern_bbox = (pattern_r0, pattern_c0, pattern_r1, pattern_c1)

    pattern_color_count = _rand_int(1, min(3, len(free_colors)))
    pattern_colors = _choose_colors(tuple(free_colors), pattern_color_count)
    pattern = _make_pattern(pattern_h, pattern_w, pattern_colors)
    mask = pattern != BG
    grid[pattern_r0 : pattern_r1 + 1, pattern_c0 : pattern_c1 + 1][mask] = pattern[mask]

    legend = frame_colors[:]
    RNG.shuffle(legend)
    for idx, color in enumerate(legend):
        grid[marker_row0 + 2 * idx, marker_col] = int(color)

    try:
        bg, frames, sequence, parsed_pattern_bbox = _parse_instance(grid)
    except Exception:
        return None

    if bg != BG:
        return None
    if len(frames) != depth:
        return None
    if set(sequence) != set(frame_colors):
        return None
    if parsed_pattern_bbox != pattern_bbox:
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if out.size == 0 or out.shape[1] <= depth:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_RETRIES):
        grid = _construct_grid()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
