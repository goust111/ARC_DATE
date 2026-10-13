from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BLACK = 0
MAROON = 9
COLOR_POOL = (1, 2, 3, 4, 5, 6, 7, 8)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

# Generator tuning.
GRID_MIN = 16
GRID_MAX = 30
EDGE_BUFFER = 2
GROUP_PADDING = 2
GROUP_COUNT_CHOICES = (1, 2, 3)
SEGMENT_COUNT_CHOICES = (2, 3, 4)
ORIENTATIONS = ("h", "v")
STEP_CHOICES = (-4, -3, -2, -1, 1, 2, 3, 4)
ROW_GAP_MIN = 2
ROW_GAP_MAX = 4
LOCAL_GROUP_ATTEMPTS = 250
GLOBAL_ATTEMPTS = 400
GROUP_PLACEMENT_ATTEMPTS = 200
MARKER_PLACEMENT_ATTEMPTS = 100
MAX_LOCAL_SIDE = 12


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    banned = exclude or set()
    return sorted(int(color) for color in np.unique(grid) if int(color) not in banned)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(coords)
    return parts


def _line_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    if a[0] == b[0]:
        r = a[0]
        c0, c1 = sorted((a[1], b[1]))
        return [(r, c) for c in range(c0, c1 + 1)]
    if a[1] == b[1]:
        c = a[1]
        r0, r1 = sorted((a[0], b[0]))
        return [(r, c) for r in range(r0, r1 + 1)]
    raise AssertionError("line endpoints must be aligned")


def _clear_between(grid: np.ndarray, a: tuple[int, int], b: tuple[int, int]) -> bool:
    if a[0] == b[0]:
        r = a[0]
        c0, c1 = sorted((a[1], b[1]))
        return c1 - c0 >= 2 and np.all(grid[r, c0 + 1 : c1] == BLACK)
    if a[1] == b[1]:
        c = a[1]
        r0, r1 = sorted((a[0], b[0]))
        return r1 - r0 >= 2 and np.all(grid[r0 + 1 : r1, c] == BLACK)
    return False


def _parse_segment(grid: np.ndarray, color: int, coords: list[tuple[int, int]]) -> dict[str, object]:
    rows = sorted({r for r, _ in coords})
    cols = sorted({c for _, c in coords})
    if len(coords) < 2:
        raise AssertionError("each segment must have length at least 2")
    if len(rows) == 1:
        r = rows[0]
        c0, c1 = cols[0], cols[-1]
        if len(coords) != c1 - c0 + 1:
            raise AssertionError("horizontal segment must be contiguous")
        if not np.all(grid[r, c0 : c1 + 1] == color):
            raise AssertionError("horizontal segment pixels must all match the color")
        return {
            "orientation": "h",
            "coords": tuple(sorted(coords)),
            "bbox": (r, c0, r, c1),
            "endpoints": ((r, c0), (r, c1)),
        }
    if len(cols) == 1:
        c = cols[0]
        r0, r1 = rows[0], rows[-1]
        if len(coords) != r1 - r0 + 1:
            raise AssertionError("vertical segment must be contiguous")
        if not np.all(grid[r0 : r1 + 1, c] == color):
            raise AssertionError("vertical segment pixels must all match the color")
        return {
            "orientation": "v",
            "coords": tuple(sorted(coords)),
            "bbox": (r0, c, r1, c),
            "endpoints": ((r0, c), (r1, c)),
        }
    raise AssertionError("each component must be a straight segment")


def _extract_segments(grid: np.ndarray, color: int) -> list[dict[str, object]]:
    segments = [_parse_segment(grid, color, comp) for comp in _components(grid == color)]
    segments.sort(key=lambda seg: tuple(int(v) for v in seg["bbox"]))
    return segments


def _candidate_connections(grid: np.ndarray, segments: list[dict[str, object]]) -> list[dict[str, object]]:
    connections: list[dict[str, object]] = []
    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            seg_a = segments[i]
            seg_b = segments[j]
            if seg_a["orientation"] != seg_b["orientation"]:
                continue
            matches: list[tuple[int, int, tuple[int, int], tuple[int, int]]] = []
            endpoints_a = tuple(seg_a["endpoints"])
            endpoints_b = tuple(seg_b["endpoints"])
            for end_idx_a, end_a in enumerate(endpoints_a):
                for end_idx_b, end_b in enumerate(endpoints_b):
                    if seg_a["orientation"] == "h":
                        if end_a[1] != end_b[1]:
                            continue
                    else:
                        if end_a[0] != end_b[0]:
                            continue
                    if _clear_between(grid, end_a, end_b):
                        matches.append((end_idx_a, end_idx_b, end_a, end_b))
            if len(matches) > 1:
                raise AssertionError("a segment pair admits multiple bridge choices")
            if matches:
                end_idx_a, end_idx_b, end_a, end_b = matches[0]
                connections.append(
                    {
                        "a": i,
                        "b": j,
                        "a_end_idx": end_idx_a,
                        "b_end_idx": end_idx_b,
                        "a_end": end_a,
                        "b_end": end_b,
                    }
                )
    return connections


def _analyze_color_group(grid: np.ndarray, color: int) -> dict[str, object]:
    segments = _extract_segments(grid, color)
    if not 2 <= len(segments) <= 4:
        raise AssertionError("each color must contribute 2 to 4 segments")
    orientations = {str(seg["orientation"]) for seg in segments}
    if len(orientations) != 1:
        raise AssertionError("all segments in a color group must share an orientation")

    connections = _candidate_connections(grid, segments)
    if len(connections) != len(segments) - 1:
        raise AssertionError("a valid group must form a simple path over its segments")

    adjacency: dict[int, list[tuple[int, int]]] = defaultdict(list)
    endpoint_usage = [[0, 0] for _ in segments]
    for conn_idx, conn in enumerate(connections):
        a = int(conn["a"])
        b = int(conn["b"])
        adjacency[a].append((b, conn_idx))
        adjacency[b].append((a, conn_idx))
        endpoint_usage[a][int(conn["a_end_idx"])] += 1
        endpoint_usage[b][int(conn["b_end_idx"])] += 1

    degrees = [len(adjacency[idx]) for idx in range(len(segments))]
    if degrees.count(1) != 2 or any(degree not in (1, 2) for degree in degrees):
        raise AssertionError("the segment graph must be a single chain")

    for degree, usage in zip(degrees, endpoint_usage):
        if any(count > 1 for count in usage):
            raise AssertionError("an endpoint cannot connect to multiple bridges")
        if degree == 1 and sum(usage) != 1:
            raise AssertionError("an end segment must use exactly one endpoint")
        if degree == 2 and sorted(usage) != [1, 1]:
            raise AssertionError("an internal segment must use both endpoints once")

    start = min((idx for idx, degree in enumerate(degrees) if degree == 1), key=lambda idx: tuple(int(v) for v in segments[idx]["bbox"]))
    ordered: list[dict[str, object]] = []
    used_connections: set[int] = set()
    current = start
    while True:
        options = [(neighbor, conn_idx) for neighbor, conn_idx in adjacency[current] if conn_idx not in used_connections]
        if not options:
            break
        if len(options) != 1:
            raise AssertionError("the chain traversal must be unique")
        neighbor, conn_idx = options[0]
        conn = connections[conn_idx]
        if int(conn["a"]) == current:
            ordered.append({"from": current, "to": neighbor, "from_end": conn["a_end"], "to_end": conn["b_end"]})
        else:
            ordered.append({"from": current, "to": neighbor, "from_end": conn["b_end"], "to_end": conn["a_end"]})
        used_connections.add(conn_idx)
        current = neighbor

    if len(used_connections) != len(connections):
        raise AssertionError("the segment chain must be connected")

    return {"segments": segments, "connections": ordered}


def _analyze_grid(grid: np.ndarray) -> dict[int, dict[str, object]]:
    colors = _colors_present(grid, exclude={BLACK, MAROON})
    if not colors:
        raise AssertionError("expected at least one colored segment group")
    return {color: _analyze_color_group(grid, color) for color in colors}


def _edge_marker_coords(grid: np.ndarray) -> list[tuple[int, int]]:
    return [tuple(map(int, coord)) for coord in np.argwhere(grid == MAROON)]


def _validate_edge_markers(grid: np.ndarray) -> None:
    h, w = grid.shape
    markers = _edge_marker_coords(grid)
    if not markers:
        raise AssertionError("expected at least one maroon edge marker")
    for r, c in markers:
        if r not in (0, h - 1) and c not in (0, w - 1):
            raise AssertionError("maroon markers must lie on the edge")
        for dr, dc in DIR4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] != BLACK:
                raise AssertionError("maroon markers must stay isolated from segments")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-colored set of scattered line segments into a single orthogonal path and move the maroon marks onto the turns.

    Input structure:
    - The background is black.
    - Foreground colors other than maroon form isolated horizontal or vertical one-pixel-wide line segments, each at least two cells long.
    - Each non-maroon color appears as exactly one group of 2 to 4 parallel segments whose endpoints admit a unique bridge order.
    - Extra maroon single-pixel markers may sit on the grid edge and are not adjacent to any segment.

    Rule:
    - For each non-maroon color, identify its straight segments and recover the unique chain in which bridgeable endpoints align in one row or one column with only black cells between them.
    - Keep every original segment in place, then draw the missing bridge between each consecutive pair of aligned endpoints using that same color.
    - Recolor every bridge-to-segment junction cell to maroon, so the maroon marks land on the path turns instead of on the original edge distractors.
    - Erase all original maroon edge markers back to black.

    Key insight:
    - The task is solved by reasoning over endpoint topology rather than component order: group segments by color, infer the only endpoint-to-endpoint chain that yields a valid path, then transfer maroon from the edge noise to the actual turning points.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("expected a 2D grid")

    group_data = _analyze_grid(g)
    out = g.copy()
    out[out == MAROON] = BLACK
    for color, info in group_data.items():
        for conn in info["connections"]:
            for r, c in _line_cells(tuple(conn["from_end"]), tuple(conn["to_end"])):
                out[r, c] = color
            fr, fc = tuple(conn["from_end"])
            tr, tc = tuple(conn["to_end"])
            out[fr, fc] = MAROON
            out[tr, tc] = MAROON
    return out


def _build_local_group(rng: random.Random, orientation: str, count: int) -> np.ndarray | None:
    for _ in range(LOCAL_GROUP_ATTEMPTS):
        if orientation == "h":
            rows = [0]
            for _ in range(count - 1):
                rows.append(rows[-1] + rng.randint(ROW_GAP_MIN, ROW_GAP_MAX))
            bridges = [0]
            for _ in range(count - 2):
                bridges.append(bridges[-1] + rng.choice(STEP_CHOICES))
            free_start = bridges[0] + rng.choice(STEP_CHOICES)
            free_end = bridges[-1] + rng.choice(STEP_CHOICES)
            cols = bridges + [free_start, free_end]
            shift = -min(cols)
            bridges = [col + shift for col in bridges]
            free_start += shift
            free_end += shift
            width = max(bridges + [free_start, free_end]) + 1
            height = rows[-1] + 1
            if width > MAX_LOCAL_SIDE or height > MAX_LOCAL_SIDE:
                continue
            block = np.zeros((height, width), dtype=int)
            for idx, row in enumerate(rows):
                if idx == 0:
                    a, b = free_start, bridges[0]
                elif idx == count - 1:
                    a, b = bridges[-1], free_end
                else:
                    a, b = bridges[idx - 1], bridges[idx]
                c0, c1 = sorted((a, b))
                block[row, c0 : c1 + 1] = 1
        else:
            cols = [0]
            for _ in range(count - 1):
                cols.append(cols[-1] + rng.randint(ROW_GAP_MIN, ROW_GAP_MAX))
            bridges = [0]
            for _ in range(count - 2):
                bridges.append(bridges[-1] + rng.choice(STEP_CHOICES))
            free_start = bridges[0] + rng.choice(STEP_CHOICES)
            free_end = bridges[-1] + rng.choice(STEP_CHOICES)
            rows = bridges + [free_start, free_end]
            shift = -min(rows)
            bridges = [row + shift for row in bridges]
            free_start += shift
            free_end += shift
            height = max(bridges + [free_start, free_end]) + 1
            width = cols[-1] + 1
            if width > MAX_LOCAL_SIDE or height > MAX_LOCAL_SIDE:
                continue
            block = np.zeros((height, width), dtype=int)
            for idx, col in enumerate(cols):
                if idx == 0:
                    a, b = free_start, bridges[0]
                elif idx == count - 1:
                    a, b = bridges[-1], free_end
                else:
                    a, b = bridges[idx - 1], bridges[idx]
                r0, r1 = sorted((a, b))
                block[r0 : r1 + 1, col] = 1

        try:
            info = _analyze_grid(block)
        except AssertionError:
            continue
        if list(info.keys()) != [1]:
            continue
        if len(info[1]["segments"]) != count:
            continue
        return block
    return None


def _expanded_occupied(occupied: np.ndarray, top: int, left: int, height: int, width: int) -> bool:
    r0 = max(0, top - GROUP_PADDING)
    c0 = max(0, left - GROUP_PADDING)
    r1 = min(occupied.shape[0], top + height + GROUP_PADDING)
    c1 = min(occupied.shape[1], left + width + GROUP_PADDING)
    return bool(np.any(occupied[r0:r1, c0:c1]))


def _place_group(grid: np.ndarray, occupied: np.ndarray, block: np.ndarray, color: int, rng: random.Random) -> bool:
    h, w = grid.shape
    bh, bw = block.shape
    top_min = EDGE_BUFFER
    left_min = EDGE_BUFFER
    top_max = h - EDGE_BUFFER - bh
    left_max = w - EDGE_BUFFER - bw
    if top_max < top_min or left_max < left_min:
        return False
    for _ in range(GROUP_PLACEMENT_ATTEMPTS):
        top = rng.randint(top_min, top_max)
        left = rng.randint(left_min, left_max)
        if _expanded_occupied(occupied, top, left, bh, bw):
            continue
        dst = grid[top : top + bh, left : left + bw]
        if np.any(dst[block == 1] != BLACK):
            continue
        dst[block == 1] = color
        occupied[top : top + bh, left : left + bw][block == 1] = True
        return True
    return False


def _marker_candidates(grid: np.ndarray) -> list[tuple[int, int]]:
    h, w = grid.shape
    coords: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            if grid[r, c] != BLACK:
                continue
            ok = True
            for dr, dc in DIR4:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] != BLACK:
                    ok = False
                    break
            if ok:
                coords.append((r, c))
    return coords


def _stamp_markers(grid: np.ndarray, rng: random.Random) -> bool:
    for _ in range(MARKER_PLACEMENT_ATTEMPTS):
        candidates = _marker_candidates(grid)
        if not candidates:
            return False
        marker_count = rng.randint(1, min(4, len(candidates)))
        picked = rng.sample(candidates, marker_count)
        grid[picked[0][0], picked[0][1]] = MAROON
        for r, c in picked[1:]:
            grid[r, c] = MAROON
        return True
    return False


def _validate_generated(grid: np.ndarray) -> None:
    _validate_edge_markers(grid)
    info = _analyze_grid(grid)
    if not 1 <= len(info) <= 3:
        raise AssertionError("unexpected number of color groups")
    out = main(grid)
    expected_corners = sum(2 * (len(group["segments"]) - 1) for group in info.values())
    if int(np.count_nonzero(out == MAROON)) != expected_corners:
        raise AssertionError("corner recoloring count mismatch")
    if np.array_equal(out, grid):
        raise AssertionError("output should differ from input")


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    group_count = rng.choice(GROUP_COUNT_CHOICES)
    blocks: list[np.ndarray] = []
    for _ in range(group_count):
        count = rng.choice(SEGMENT_COUNT_CHOICES)
        orientation = rng.choice(ORIENTATIONS)
        block = _build_local_group(rng, orientation, count)
        if block is None:
            return None
        blocks.append(block)

    min_h = max(GRID_MIN, max(block.shape[0] for block in blocks) + 2 * EDGE_BUFFER)
    min_w = max(GRID_MIN, max(block.shape[1] for block in blocks) + 2 * EDGE_BUFFER)
    if min_h > GRID_MAX or min_w > GRID_MAX:
        return None

    h = rng.randint(min_h, GRID_MAX)
    w = rng.randint(min_w, GRID_MAX)
    grid = np.zeros((h, w), dtype=int)
    occupied = np.zeros_like(grid, dtype=bool)
    colors = rng.sample(COLOR_POOL, group_count)
    rng.shuffle(blocks)
    for color, block in zip(colors, blocks):
        if not _place_group(grid, occupied, block, color, rng):
            return None

    if not _stamp_markers(grid, rng):
        return None

    try:
        _validate_generated(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GLOBAL_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GLOBAL_ATTEMPTS} attempts")
