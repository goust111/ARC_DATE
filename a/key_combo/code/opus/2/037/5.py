import math
import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GRAY = 5
MASK_COLORS = [3, 4, 6, 7, 8, 9]

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if not (dr == 0 and dc == 0)
]

SLOTS = ("TL", "TR", "BR", "BL")


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    dirs = DIR4 if connectivity == 4 else DIR8
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in dirs:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _extract_frames(grid: np.ndarray) -> tuple[list[dict], list[np.ndarray]]:
    red_components = _components_mask(grid == RED, connectivity=4)
    assert 3 <= len(red_components) <= 4

    frames = []
    for coords in red_components:
        r0, c0, r1, c1 = _bbox_coords(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert h >= 4 and w >= 4

        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros((h, w), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True

        assert int(coords.shape[0]) == int(border.sum())
        assert np.all(patch[border] == RED)
        assert np.all(patch[~border] != RED)

        interior = patch[1:-1, 1:-1]
        marker_coords = np.argwhere(interior == GRAY)
        assert marker_coords.shape[0] == 1
        marker_r, marker_c = map(int, marker_coords[0])
        assert marker_r == interior.shape[0] - 1

        mask_colors = set(map(int, np.unique(interior))) - {BG, GRAY}
        assert len(mask_colors) == 1
        mask_color = mask_colors.pop()
        assert np.any(interior == mask_color)

        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "center": ((r0 + r1) / 2.0, (c0 + c1) / 2.0),
                "mask_color": int(mask_color),
                "marker": (marker_r, marker_c),
            }
        )

    return frames, red_components


def _blue_cycle_edges(grid: np.ndarray, red_components: list[np.ndarray]) -> set[tuple[int, int]]:
    red_owner: dict[tuple[int, int], int] = {}
    for frame_id, coords in enumerate(red_components):
        for r, c in coords:
            red_owner[(int(r), int(c))] = frame_id

    edges: set[tuple[int, int]] = set()
    for coords in _components_mask(grid == BLUE, connectivity=4):
        touched = set()
        for r, c in coords:
            rr = int(r)
            cc = int(c)
            for dr, dc in DIR4:
                frame_id = red_owner.get((rr + dr, cc + dc))
                if frame_id is not None:
                    touched.add(frame_id)
        assert len(touched) == 2
        a, b = sorted(touched)
        edges.add((a, b))

    n = len(red_components)
    assert len(edges) == n
    degrees = [0] * n
    for a, b in edges:
        degrees[a] += 1
        degrees[b] += 1
    assert all(degree == 2 for degree in degrees)
    return edges


def _clockwise_order(frames: list[dict], edges: set[tuple[int, int]]) -> list[int]:
    mean_r = sum(frame["center"][0] for frame in frames) / len(frames)
    mean_c = sum(frame["center"][1] for frame in frames) / len(frames)
    angles = []
    for frame in frames:
        dr = frame["center"][0] - mean_r
        dc = frame["center"][1] - mean_c
        angles.append(math.atan2(dc, -dr) % (2.0 * math.pi))

    assert len({round(angle, 12) for angle in angles}) == len(angles)
    order = sorted(range(len(frames)), key=lambda idx: angles[idx])
    for a, b in zip(order, order[1:] + order[:1]):
        assert tuple(sorted((a, b))) in edges
    return order


def _propagate_column(
    out: np.ndarray,
    source_interior: np.ndarray,
    bbox: tuple[int, int, int, int],
    marker: tuple[int, int],
    color: int,
) -> None:
    r0, c0, _r1, _c1 = bbox
    marker_r, marker_c = marker
    abs_c = c0 + 1 + marker_c

    out[r0 + 1 + marker_r, abs_c] = int(color)
    for local_r in range(marker_r - 1, -1, -1):
        if int(source_interior[local_r, marker_c]) != BG:
            break
        out[r0 + 1 + local_r, abs_c] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each frame's marked interior column with the next clockwise frame's mask color.

    Input structure:
    - The grid has black background, 3-4 hollow red frames, and blue connectors forming one cycle.
    - Every frame contains one irregular single-color interior mask plus one gray marker pixel on the bottom interior row.
    - Each blue connected component touches exactly two frames and stays outside the frame interiors.

    Rule:
    - Recover the red frames and the blue cycle that links them.
    - Sort frame centers clockwise around their centroid, with angle measured from upward and increasing clockwise.
    - For each frame, take the unique mask color from the next clockwise frame.
    - Replace the gray marker with that color and keep filling upward in the same interior column through background cells only, stopping before the first mask pixel or the frame wall.
    - Keep the original mask pixels, red borders, blue connectors, and outside background unchanged.

    Key insight:
    - The marked column says where to paint, but the blue ring and clockwise order decide which other frame supplies the paint color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    frames, red_components = _extract_frames(g)
    edges = _blue_cycle_edges(g, red_components)
    order = _clockwise_order(frames, edges)

    out = g.copy()
    for idx, frame_id in enumerate(order):
        next_id = order[(idx + 1) % len(order)]
        frame = frames[frame_id]
        r0, c0, r1, c1 = frame["bbox"]
        source_interior = g[r0 + 1 : r1, c0 + 1 : c1]
        _propagate_column(
            out,
            source_interior,
            frame["bbox"],
            frame["marker"],
            frames[next_id]["mask_color"],
        )
    return out


def _line_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    if a[0] == b[0]:
        step = 1 if b[1] >= a[1] else -1
        return [(a[0], c) for c in range(a[1], b[1] + step, step)]
    assert a[1] == b[1]
    step = 1 if b[0] >= a[0] else -1
    return [(r, a[1]) for r in range(a[0], b[0] + step, step)]


def _polyline_cells(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for idx in range(len(points) - 1):
        segment = _line_cells(points[idx], points[idx + 1])
        if idx:
            segment = segment[1:]
        cells.extend(segment)
    return cells


def _paint_frame(grid: np.ndarray, top: int, left: int, size: int) -> None:
    bottom = top + size - 1
    right = left + size - 1
    grid[top, left : right + 1] = RED
    grid[bottom, left : right + 1] = RED
    grid[top : bottom + 1, left] = RED
    grid[top : bottom + 1, right] = RED


def _random_mask(size: int, color: int) -> tuple[np.ndarray, int]:
    inner = size - 2
    assert inner >= 4

    for _ in range(120):
        marker_col = random.randrange(inner)
        anchor_r = random.randint(0, inner - 4)
        target = random.randint(3, min(6, inner * inner - 1))

        blocked = {(inner - 1, c) for c in range(inner)}
        blocked |= {(r, marker_col) for r in range(anchor_r + 1, inner)}

        occupied = {(anchor_r, marker_col)}
        frontier = set()
        for dr, dc in DIR8:
            nr = anchor_r + dr
            nc = marker_col + dc
            if 0 <= nr < inner and 0 <= nc < inner and (nr, nc) not in blocked:
                frontier.add((nr, nc))

        while len(occupied) < target and frontier:
            cell = random.choice(tuple(frontier))
            frontier.discard(cell)
            occupied.add(cell)
            for dr, dc in DIR8:
                nr = cell[0] + dr
                nc = cell[1] + dc
                if 0 <= nr < inner and 0 <= nc < inner and (nr, nc) not in blocked and (nr, nc) not in occupied:
                    frontier.add((nr, nc))

        rows_in_marker_col = [r for r, c in occupied if c == marker_col]
        if not rows_in_marker_col or max(rows_in_marker_col) > inner - 4:
            continue

        mask = np.zeros((inner, inner), dtype=int)
        for r, c in occupied:
            mask[r, c] = int(color)

        used_rows = {r for r, _c in occupied}
        used_cols = {c for _r, c in occupied}
        if len(occupied) >= 3 and (len(used_rows) >= 2 or len(used_cols) >= 2):
            return mask, marker_col

    raise RuntimeError("failed to build a valid interior mask")


def _attachment_points(
    size: int,
    positions: dict[str, tuple[int, int]],
    ring: dict[str, int],
) -> dict[str, dict[str, tuple[int, int]]]:
    attachments: dict[str, dict[str, tuple[int, int]]] = {}

    top = random.sample(range(1, size - 1), 2)
    left = random.sample(range(1, size - 1), 2)
    right = random.sample(range(1, size - 1), 2)
    bottom = random.sample(range(1, size - 1), 2)

    tl_r, tl_c = positions["TL"]
    tr_r, tr_c = positions["TR"]
    br_r, br_c = positions["BR"]
    bl_r, bl_c = positions["BL"]

    attachments["TL"] = {
        "top": (ring["top_row"], tl_c + top[0]),
        "left": (tl_r + left[0], ring["left_col"]),
    }
    attachments["TR"] = {
        "top": (ring["top_row"], tr_c + top[1]),
        "right": (tr_r + right[0], ring["right_col"]),
    }
    attachments["BR"] = {
        "right": (br_r + right[1], ring["right_col"]),
        "bottom": (ring["bottom_row"], br_c + bottom[0]),
    }
    attachments["BL"] = {
        "bottom": (ring["bottom_row"], bl_c + bottom[1]),
        "left": (bl_r + left[1], ring["left_col"]),
    }
    return attachments


def _connector_paths(
    missing: str | None,
    attachments: dict[str, dict[str, tuple[int, int]]],
    ring: dict[str, int],
) -> list[list[tuple[int, int]]]:
    if missing is None:
        return [
            _polyline_cells([attachments["TL"]["top"], attachments["TR"]["top"]]),
            _polyline_cells([attachments["TR"]["right"], attachments["BR"]["right"]]),
            _polyline_cells([attachments["BR"]["bottom"], attachments["BL"]["bottom"]]),
            _polyline_cells([attachments["BL"]["left"], attachments["TL"]["left"]]),
        ]

    if missing == "TL":
        corner = (ring["top_row"], ring["left_col"])
        return [
            _polyline_cells([attachments["TR"]["right"], attachments["BR"]["right"]]),
            _polyline_cells([attachments["BR"]["bottom"], attachments["BL"]["bottom"]]),
            _polyline_cells([attachments["BL"]["left"], corner, attachments["TR"]["top"]]),
        ]
    if missing == "TR":
        corner = (ring["top_row"], ring["right_col"])
        return [
            _polyline_cells([attachments["BR"]["bottom"], attachments["BL"]["bottom"]]),
            _polyline_cells([attachments["BL"]["left"], attachments["TL"]["left"]]),
            _polyline_cells([attachments["TL"]["top"], corner, attachments["BR"]["right"]]),
        ]
    if missing == "BR":
        corner = (ring["bottom_row"], ring["right_col"])
        return [
            _polyline_cells([attachments["BL"]["left"], attachments["TL"]["left"]]),
            _polyline_cells([attachments["TL"]["top"], attachments["TR"]["top"]]),
            _polyline_cells([attachments["TR"]["right"], corner, attachments["BL"]["bottom"]]),
        ]

    assert missing == "BL"
    corner = (ring["bottom_row"], ring["left_col"])
    return [
        _polyline_cells([attachments["TL"]["top"], attachments["TR"]["top"]]),
        _polyline_cells([attachments["TR"]["right"], attachments["BR"]["right"]]),
        _polyline_cells([attachments["BR"]["bottom"], corner, attachments["TL"]["left"]]),
    ]


def _build_attempt() -> np.ndarray | None:
    frame_count = random.choice([3, 4])
    size = random.choice([6, 7, 8])
    margin = 2
    max_gap = 30 - 2 * size - 2 * margin
    if max_gap < 4:
        return None
    h_gap = random.randint(4, max_gap)
    v_gap = random.randint(4, max_gap)

    top = margin
    left = margin
    bottom = top + size + v_gap
    right = left + size + h_gap
    height = bottom + size + margin
    width = right + size + margin
    if max(height, width) > 30:
        return None

    positions = {
        "TL": (top, left),
        "TR": (top, right),
        "BR": (bottom, right),
        "BL": (bottom, left),
    }
    ring = {
        "top_row": top - 1,
        "right_col": right + size,
        "bottom_row": bottom + size,
        "left_col": left - 1,
    }

    missing = None if frame_count == 4 else random.choice(SLOTS)
    active_slots = [slot for slot in SLOTS if slot != missing]

    grid = np.zeros((height, width), dtype=int)
    colors = random.sample(MASK_COLORS, frame_count)
    random.shuffle(colors)

    for slot, color in zip(active_slots, colors, strict=False):
        fr, fc = positions[slot]
        _paint_frame(grid, fr, fc, size)
        mask, marker_col = _random_mask(size, color)
        interior = grid[fr + 1 : fr + size - 1, fc + 1 : fc + size - 1]
        interior[:, :] = BG
        interior[mask == color] = color
        interior[-1, marker_col] = GRAY

    attachments = _attachment_points(size, positions, ring)
    connector_paths = _connector_paths(missing, attachments, ring)

    used: set[tuple[int, int]] = set()
    for path in connector_paths:
        touched_red = set()
        for r, c in path:
            if not (0 <= r < height and 0 <= c < width):
                return None
            if grid[r, c] != BG or (r, c) in used:
                return None
            used.add((r, c))
            for dr, dc in DIR4:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < height and 0 <= nc < width and grid[nr, nc] == RED:
                    touched_red.add((nr, nc))
        if not touched_red:
            return None
        for r, c in path:
            grid[r, c] = BLUE

    try:
        solved = main(grid)
    except Exception:
        return None
    if np.array_equal(solved, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
