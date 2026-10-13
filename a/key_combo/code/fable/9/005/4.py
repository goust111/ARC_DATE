from __future__ import annotations

import itertools
import random

import numpy as np


BG = 0
FRAME = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]

GEN_ATTEMPTS = 600
GRID_MIN = 20
GRID_MAX = 30
FRAME_SIZE_MIN = 5
FRAME_SIZE_MAX = 11
LINE_PROB = 0.55
GREY_DECOY_PROB = 0.55


def _prefix_sum(mask: np.ndarray) -> np.ndarray:
    ps = np.zeros((mask.shape[0] + 1, mask.shape[1] + 1), dtype=int)
    ps[1:, 1:] = mask.astype(int).cumsum(axis=0).cumsum(axis=1)
    return ps


def _rect_sum(ps: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> int:
    if r0 > r1 or c0 > c1:
        return 0
    return int(ps[r1 + 1, c1 + 1] - ps[r0, c1 + 1] - ps[r1 + 1, c0] + ps[r0, c0])


def _max_depth(h: int, w: int) -> int:
    return (min(h, w) - 1) // 2


def _bbox_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _bbox_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    row_gap = max(b[0] - a[2] - 1, a[0] - b[2] - 1, 0)
    col_gap = max(b[1] - a[3] - 1, a[1] - b[3] - 1, 0)
    return max(row_gap, col_gap)


def _paint_frame(grid: np.ndarray, box: tuple[int, int, int, int], color: int = FRAME) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _paint_shell(grid: np.ndarray, box: tuple[int, int, int, int], depth: int, color: int) -> None:
    r0, c0, r1, c1 = box
    rr0, cc0 = r0 + depth, c0 + depth
    rr1, cc1 = r1 - depth, c1 - depth
    grid[rr0, cc0 : cc1 + 1] = color
    grid[rr1, cc0 : cc1 + 1] = color
    grid[rr0 : rr1 + 1, cc0] = color
    grid[rr0 : rr1 + 1, cc1] = color


def _frame_mask(shape: tuple[int, int], boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for box in boxes:
        r0, c0, r1, c1 = box
        mask[r0, c0 : c1 + 1] = True
        mask[r1, c0 : c1 + 1] = True
        mask[r0 : r1 + 1, c0] = True
        mask[r0 : r1 + 1, c1] = True
    return mask


def _dist_to_frame_border(coord: tuple[int, int], box: tuple[int, int, int, int]) -> int:
    r, c = coord
    r0, c0, r1, c1 = box
    if r0 <= r <= r1 and c0 <= c <= c1:
        return min(r - r0, r1 - r, c - c0, c1 - c)
    dr = 0 if r0 <= r <= r1 else min(abs(r - r0), abs(r - r1))
    dc = 0 if c0 <= c <= c1 else min(abs(c - c0), abs(c - c1))
    return max(dr, dc)


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    return [
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if not (dr == 0 and dc == 0)
    ]


def _all_singletons(mask: np.ndarray) -> bool:
    seen: set[tuple[int, int]] = set()
    rows, cols = mask.shape
    for r, c in np.argwhere(mask):
        start = (int(r), int(c))
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        size = 0
        while stack:
            cr, cc = stack.pop()
            size += 1
            if size > 1:
                return False
            for nr, nc in _neighbors8(cr, cc):
                if 0 <= nr < rows and 0 <= nc < cols and mask[nr, nc] and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    stack.append((nr, nc))
    return True


def _find_frames(g: np.ndarray) -> list[dict]:
    grey = g == FRAME
    fg = g != BG
    ps_grey = _prefix_sum(grey)
    ps_fg = _prefix_sum(fg)
    rows, cols = g.shape

    candidates: list[dict] = []
    for r0 in range(rows - 2):
        for r1 in range(r0 + 2, rows):
            h = r1 - r0 + 1
            for c0 in range(cols - 2):
                for c1 in range(c0 + 2, cols):
                    w = c1 - c0 + 1
                    perimeter_size = 2 * (h + w) - 4
                    total_grey = _rect_sum(ps_grey, r0, c0, r1, c1)
                    inner_grey = _rect_sum(ps_grey, r0 + 1, c0 + 1, r1 - 1, c1 - 1)
                    if total_grey - inner_grey != perimeter_size:
                        continue
                    if _rect_sum(ps_fg, r0 + 1, c0 + 1, r1 - 1, c1 - 1) != 0:
                        continue
                    candidates.append(
                        {
                            "bbox": (r0, c0, r1, c1),
                            "size": (h, w),
                            "perimeter": perimeter_size,
                            "max_depth": _max_depth(h, w),
                        }
                    )

    assert len(candidates) >= 2, "expected at least two frame candidates"

    best_pair: tuple[dict, dict] | None = None
    best_score: tuple[int, int] = (-1, -1)
    for i, first in enumerate(candidates):
        for second in candidates[i + 1 :]:
            if first["size"] == second["size"]:
                continue
            if _bbox_overlap(first["bbox"], second["bbox"]):
                continue
            mask = _frame_mask(g.shape, [first["bbox"], second["bbox"]])
            if not _all_singletons((g != BG) & ~mask):
                continue
            score = (
                int(first["perimeter"] + second["perimeter"]),
                int(np.prod(first["size"]) + np.prod(second["size"])),
            )
            if score > best_score:
                best_score = score
                best_pair = (first, second)

    assert best_pair is not None, "failed to isolate exactly two frames"
    frames = sorted(best_pair, key=lambda frame: frame["bbox"])
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """Assign each exterior marker to the nearer grey frame and copy its color into the equally deep interior shell.

    Input structure:
    - The canvas has a black background, exactly two complete one-cell-thick grey hollow rectangles, and scattered single-pixel markers outside the frames.
    - Each marker lies outside both frame bboxes; one decoy marker may also be grey and can touch a frame from the outside.
    - Every marker is closer in Chebyshev distance to one frame border than to the other, and markers assigned to the same frame use distinct distances.

    Rule:
    - Detect the two grey rectangular frames and keep their border pixels unchanged.
    - Erase every non-frame marker from the exterior.
    - For each marker, measure its Chebyshev distance d to the nearer frame border and paint the entire interior shell at depth d of that frame with the marker color.
    - Leave any shell depth that receives no marker black.

    Key insight:
    - The scattered pixels encode two quantities at once: which frame they belong to and which shell depth to light, with the outside distance reused verbatim as the inside layer index.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    frames = _find_frames(g)
    boxes = [frame["bbox"] for frame in frames]
    mask = _frame_mask(g.shape, boxes)

    markers = []
    seen_depths = [set(), set()]
    for r, c in np.argwhere((g != BG) & ~mask):
        coord = (int(r), int(c))
        distances = [_dist_to_frame_border(coord, box) for box in boxes]
        owner = int(np.argmin(distances))
        depth = int(distances[owner])
        other = int(distances[1 - owner])
        assert depth < other, "marker must have a unique nearer frame"
        assert 1 <= depth <= int(frames[owner]["max_depth"]), "marker depth exceeds frame interior"
        assert depth not in seen_depths[owner], "depths within a frame must be unique"
        seen_depths[owner].add(depth)
        markers.append({"coord": coord, "color": int(g[r, c]), "frame": owner, "depth": depth})

    out = np.full_like(g, BG)
    out[mask] = FRAME
    for marker in markers:
        _paint_shell(out, boxes[int(marker["frame"])], int(marker["depth"]), int(marker["color"]))
    return out


def _sample_frame_sizes() -> list[tuple[int, int]]:
    while True:
        sizes = [
            (
                random.randint(FRAME_SIZE_MIN, FRAME_SIZE_MAX),
                random.randint(FRAME_SIZE_MIN, FRAME_SIZE_MAX),
            )
            for _ in range(2)
        ]
        if sizes[0] != sizes[1]:
            return sizes


def _sample_boxes(shape: tuple[int, int], sizes: list[tuple[int, int]]) -> list[tuple[int, int, int, int]] | None:
    rows, cols = shape
    depth_limits = [_max_depth(h, w) for h, w in sizes]
    order = sorted(range(2), key=lambda idx: sizes[idx][0] * sizes[idx][1], reverse=True)
    placed: dict[int, tuple[int, int, int, int]] = {}

    for _ in range(200):
        placed.clear()
        ok = True
        for idx in order:
            h, w = sizes[idx]
            halo = depth_limits[idx]
            r_lo = halo
            r_hi = rows - halo - h
            c_lo = halo
            c_hi = cols - halo - w
            if r_lo > r_hi or c_lo > c_hi:
                ok = False
                break
            positions = [
                (r0, c0)
                for r0 in range(r_lo, r_hi + 1)
                for c0 in range(c_lo, c_hi + 1)
                if all(
                    _bbox_gap(
                        (r0 - halo, c0 - halo, r0 + h - 1 + halo, c0 + w - 1 + halo),
                        (
                            prev[0] - depth_limits[jdx],
                            prev[1] - depth_limits[jdx],
                            prev[2] + depth_limits[jdx],
                            prev[3] + depth_limits[jdx],
                        ),
                    )
                    >= 1
                    for jdx, prev in placed.items()
                )
            ]
            if not positions:
                ok = False
                break
            r0, c0 = random.choice(positions)
            placed[idx] = (r0, c0, r0 + h - 1, c0 + w - 1)
        if ok and len(placed) == 2:
            return [placed[0], placed[1]]
    return None


def _sample_depths(
    max_depth: int,
    *,
    min_count: int,
    require_depth_one: bool,
    prefer_empty_shell: bool,
) -> list[int]:
    all_depths = list(range(1, max_depth + 1))
    min_count = min(min_count, max_depth)
    high = max_depth
    if prefer_empty_shell and max_depth > min_count:
        high = max_depth - 1
    count = random.randint(min_count, high)
    if require_depth_one:
        rest = random.sample([d for d in all_depths if d != 1], count - 1)
        depths = [1, *rest]
    else:
        depths = random.sample(all_depths, count)
    return sorted(depths)


def _coord_adjacent_to_markers(coord: tuple[int, int], placed: list[tuple[int, int]]) -> bool:
    r, c = coord
    return any(max(abs(r - pr), abs(c - pc)) <= 1 for pr, pc in placed)


def _marker_valid(
    coord: tuple[int, int],
    frame_idx: int,
    depth: int,
    boxes: list[tuple[int, int, int, int]],
    placed: list[tuple[int, int]],
    shape: tuple[int, int],
) -> bool:
    r, c = coord
    rows, cols = shape
    if not (0 <= r < rows and 0 <= c < cols):
        return False
    if any(box[0] <= r <= box[2] and box[1] <= c <= box[3] for box in boxes):
        return False
    if _coord_adjacent_to_markers(coord, placed):
        return False
    own = _dist_to_frame_border(coord, boxes[frame_idx])
    other = _dist_to_frame_border(coord, boxes[1 - frame_idx])
    return own == depth and own < other


def _search_line_positions(
    frame_idx: int,
    depths: list[int],
    boxes: list[tuple[int, int, int, int]],
    placed: list[tuple[int, int]],
    shape: tuple[int, int],
) -> list[tuple[int, int]] | None:
    r0, c0, r1, c1 = boxes[frame_idx]
    options: list[list[tuple[int, int]]] = []

    for shared_r in range(r0, r1 + 1):
        for sides in itertools.product((-1, 1), repeat=len(depths)):
            coords = [
                (shared_r, c0 - depth if side < 0 else c1 + depth)
                for depth, side in zip(depths, sides)
            ]
            if len(set(coords)) != len(coords):
                continue
            if any(_coord_adjacent_to_markers(coord, placed) for coord in coords):
                continue
            if any(max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= 1 for a, b in itertools.combinations(coords, 2)):
                continue
            if all(_marker_valid(coord, frame_idx, depth, boxes, placed, shape) for coord, depth in zip(coords, depths)):
                options.append(coords)

    for shared_c in range(c0, c1 + 1):
        for sides in itertools.product((-1, 1), repeat=len(depths)):
            coords = [
                (r0 - depth if side < 0 else r1 + depth, shared_c)
                for depth, side in zip(depths, sides)
            ]
            if len(set(coords)) != len(coords):
                continue
            if any(_coord_adjacent_to_markers(coord, placed) for coord in coords):
                continue
            if any(max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= 1 for a, b in itertools.combinations(coords, 2)):
                continue
            if all(_marker_valid(coord, frame_idx, depth, boxes, placed, shape) for coord, depth in zip(coords, depths)):
                options.append(coords)

    if not options:
        return None
    return random.choice(options)


def _search_free_marker(
    frame_idx: int,
    depth: int,
    boxes: list[tuple[int, int, int, int]],
    placed: list[tuple[int, int]],
    shape: tuple[int, int],
) -> tuple[int, int] | None:
    rows, cols = shape
    candidates = [
        (r, c)
        for r in range(rows)
        for c in range(cols)
        if _marker_valid((r, c), frame_idx, depth, boxes, placed, shape)
    ]
    if not candidates:
        return None
    return random.choice(candidates)


def _expected_output(
    shape: tuple[int, int],
    boxes: list[tuple[int, int, int, int]],
    markers: list[dict],
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for box in boxes:
        _paint_frame(out, box, FRAME)
    for marker in markers:
        _paint_shell(out, boxes[int(marker["frame"])], int(marker["depth"]), int(marker["color"]))
    return out


def _gen_attempt() -> np.ndarray | None:
    shape = (
        random.randint(GRID_MIN, GRID_MAX),
        random.randint(GRID_MIN, GRID_MAX),
    )
    sizes = _sample_frame_sizes()
    boxes = _sample_boxes(shape, sizes)
    if boxes is None:
        return None

    max_depths = [_max_depth(h, w) for h, w in sizes]
    line_candidates = [idx for idx, md in enumerate(max_depths) if md >= 2]
    line_frame = random.choice(line_candidates) if line_candidates and random.random() < LINE_PROB else None
    grey_frame = random.randint(0, 1) if random.random() < GREY_DECOY_PROB else None

    marker_specs: list[list[dict]] = [[], []]
    for frame_idx, md in enumerate(max_depths):
        prefer_empty = md >= 3 and random.random() < 0.7
        require_depth_one = frame_idx == grey_frame
        min_count = 2 if frame_idx == line_frame else 1
        depths = _sample_depths(
            md,
            min_count=min_count,
            require_depth_one=require_depth_one,
            prefer_empty_shell=prefer_empty,
        )
        for depth in depths:
            color = FRAME if frame_idx == grey_frame and depth == 1 else random.choice(PALETTE)
            marker_specs[frame_idx].append({"frame": frame_idx, "depth": depth, "color": color})

    placed_coords: list[tuple[int, int]] = []
    markers: list[dict] = []

    frame_order = list(range(2))
    if line_frame is not None:
        frame_order.remove(line_frame)
        frame_order.insert(0, line_frame)

    for frame_idx in frame_order:
        specs = sorted(marker_specs[frame_idx], key=lambda item: int(item["depth"]), reverse=True)
        if frame_idx == line_frame:
            coords = _search_line_positions(
                frame_idx,
                [int(spec["depth"]) for spec in specs],
                boxes,
                placed_coords,
                shape,
            )
            if coords is None:
                return None
            for spec, coord in zip(specs, coords):
                markers.append({**spec, "coord": coord})
                placed_coords.append(coord)
        else:
            for spec in specs:
                coord = _search_free_marker(frame_idx, int(spec["depth"]), boxes, placed_coords, shape)
                if coord is None:
                    return None
                markers.append({**spec, "coord": coord})
                placed_coords.append(coord)

    inp = np.full(shape, BG, dtype=int)
    for box in boxes:
        _paint_frame(inp, box, FRAME)
    for marker in markers:
        r, c = marker["coord"]
        inp[r, c] = int(marker["color"])

    expected = _expected_output(shape, boxes, markers)
    if np.array_equal(expected, inp):
        return None
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for distance-to-shell task")
