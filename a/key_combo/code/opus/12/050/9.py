from __future__ import annotations

import random
import itertools
from collections import defaultdict

import numpy as np


BG = 0
FILL = 4
MARKER = 5
COLOR_POOL = (1, 2, 3, 6, 7, 8, 9)
DIAGONALS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
ROW_DIRS = {0: ((-1, -1), (1, -1)), 1: ((-1, 1), (1, 1))}
COL_DIRS = {0: ((-1, -1), (-1, 1)), 1: ((1, -1), (1, 1))}
GEN_ATTEMPTS = 600


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _is_boundary(shape: tuple[int, int], r: int, c: int) -> bool:
    return r in (0, shape[0] - 1) or c in (0, shape[1] - 1)


def _ray_path(shape: tuple[int, int], source: tuple[int, int], direction: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = source
    dr, dc = direction
    path: list[tuple[int, int]] = []
    while _in_bounds(shape, r, c):
        path.append((r, c))
        r += dr
        c += dc
    return path


def _extract_objects(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    colors = [int(color) for color in np.unique(g) if color not in (BG, MARKER)]
    objects: list[dict[str, object]] = []
    for color in colors:
        coords = np.argwhere(g == color)
        assert len(coords) == 1, "each non-background color must be a single source pixel"
        source = tuple(map(int, coords[0]))

        marker_dirs = []
        for dr, dc in DIAGONALS:
            nr = source[0] + dr
            nc = source[1] + dc
            if _in_bounds(g.shape, nr, nc) and int(g[nr, nc]) == MARKER:
                marker_dirs.append((dr, dc))
        assert len(marker_dirs) == 1, "each source must have exactly one diagonal grey marker"

        direction = marker_dirs[0]
        path = _ray_path(g.shape, source, direction)
        objects.append(
            {
                "color": color,
                "source": source,
                "direction": direction,
                "marker": (source[0] + direction[0], source[1] + direction[1]),
                "path": path,
                "endpoint": path[-1],
            }
        )
    return objects


def _segments_from_endpoints(endpoints: list[tuple[int, int, int]]) -> list[tuple[str, int, int, int]]:
    rows: dict[int, list[tuple[int, int]]] = defaultdict(list)
    cols: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for r, c, color in endpoints:
        rows[r].append((c, color))
        cols[c].append((r, color))

    segments: list[tuple[str, int, int, int]] = []
    for r, items in rows.items():
        if len(items) < 2:
            continue
        items.sort()
        for (c0, color0), (c1, color1) in zip(items, items[1:]):
            if color0 != color1 and c1 - c0 > 1:
                segments.append(("row", r, c0, c1))
    for c, items in cols.items():
        if len(items) < 2:
            continue
        items.sort()
        for (r0, color0), (r1, color1) in zip(items, items[1:]):
            if color0 != color1 and r1 - r0 > 1:
                segments.append(("col", c, r0, r1))
    return segments


def _render_expected(shape: tuple[int, int], objects: list[dict[str, object]]) -> np.ndarray:
    base = np.full(shape, BG, dtype=int)
    for obj in objects:
        source_r, source_c = obj["source"]
        marker_r, marker_c = obj["marker"]
        color = int(obj["color"])
        base[source_r, source_c] = color
        base[marker_r, marker_c] = MARKER

    rays = base.copy()
    endpoints: list[tuple[int, int, int]] = []
    for obj in objects:
        color = int(obj["color"])
        path = obj["path"]
        for r, c in path:
            rays[r, c] = color
        end_r, end_c = path[-1]
        endpoints.append((end_r, end_c, color))

    rays[rays == MARKER] = BG
    out = rays.copy()
    for axis, fixed, start, stop in _segments_from_endpoints(endpoints):
        if axis == "row":
            for c in range(start + 1, stop):
                if rays[fixed, c] == BG:
                    out[fixed, c] = FILL
        else:
            for r in range(start + 1, stop):
                if rays[r, fixed] == BG:
                    out[r, fixed] = FILL
    return out


def _render_rays_only(shape: tuple[int, int], objects: list[dict[str, object]]) -> np.ndarray:
    rays = np.full(shape, BG, dtype=int)
    for obj in objects:
        color = int(obj["color"])
        for r, c in obj["path"]:
            rays[r, c] = color
    return rays


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each marked colored pixel along one diagonal to the border, then fill gaps between aligned ray endpoints.

    Input structure:
    - The background is black, with 3-4 single colored source pixels and grey direction markers.
    - Each source color appears exactly once, and its unique grey marker is on one diagonal-adjacent cell.
    - The marker's relative position encodes one of the four diagonal ray directions.

    Rule:
    - For every non-black, non-grey source pixel, read the diagonal direction from its grey neighbor.
    - Draw a same-color diagonal ray from the source to the grid edge, including the source and the terminal border pixel.
    - Remove all grey marker pixels.
    - Collect the border endpoints of all rays; within each row or column, sort aligned endpoints and fill the black cells between each adjacent endpoint pair with yellow, skipping any non-black cells already occupied by rays.

    Key insight:
    - The grey marker is only a direction code; the endpoint alignment that triggers yellow filling is not visible until the diagonal ray has first been extended to the boundary.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    objects = _extract_objects(g)
    out = g.copy()
    endpoints: list[tuple[int, int, int]] = []
    for obj in objects:
        color = int(obj["color"])
        path = obj["path"]
        for r, c in path:
            out[r, c] = color
        end_r, end_c = path[-1]
        endpoints.append((end_r, end_c, color))

    out[out == MARKER] = BG
    rays_only = out.copy()
    for axis, fixed, start, stop in _segments_from_endpoints(endpoints):
        if axis == "row":
            for c in range(start + 1, stop):
                if rays_only[fixed, c] == BG:
                    out[fixed, c] = FILL
        else:
            for r in range(start + 1, stop):
                if rays_only[r, fixed] == BG:
                    out[r, fixed] = FILL
    return out


def _endpoint_object(
    shape: tuple[int, int],
    endpoint: tuple[int, int],
    direction: tuple[int, int],
    color: int,
) -> dict[str, object] | None:
    er, ec = endpoint
    choices: list[dict[str, object]] = []
    for length in range(2, max(shape)):
        source = (er - length * direction[0], ec - length * direction[1])
        marker = (source[0] + direction[0], source[1] + direction[1])
        if not _in_bounds(shape, *source) or not _in_bounds(shape, *marker):
            continue
        if _is_boundary(shape, *source):
            continue
        path = _ray_path(shape, source, direction)
        if path[-1] != endpoint:
            continue
        choices.append(
            {
                "color": color,
                "source": source,
                "direction": direction,
                "marker": marker,
                "path": path,
                "endpoint": endpoint,
            }
        )
    if not choices:
        return None
    return random.choice(choices)


def _random_object(shape: tuple[int, int], color: int) -> dict[str, object] | None:
    for _ in range(120):
        source = (random.randrange(1, shape[0] - 1), random.randrange(1, shape[1] - 1))
        direction = random.choice(DIAGONALS)
        marker = (source[0] + direction[0], source[1] + direction[1])
        if not _in_bounds(shape, *marker):
            continue
        path = _ray_path(shape, source, direction)
        if len(path) < 3:
            continue
        return {
            "color": color,
            "source": source,
            "direction": direction,
            "marker": marker,
            "path": path,
            "endpoint": path[-1],
        }
    return None


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _can_add_object(objects: list[dict[str, object]], candidate: dict[str, object]) -> bool:
    candidate_cells = {tuple(candidate["source"]), tuple(candidate["marker"])}
    candidate_path = set(candidate["path"])
    for obj in objects:
        existing_cells = {tuple(obj["source"]), tuple(obj["marker"])}
        for a in candidate_cells:
            for b in existing_cells:
                if _chebyshev(a, b) <= 2:
                    return False
        if candidate_path & set(obj["path"]):
            return False
    return True


def _rows_with_counts(endpoints: list[tuple[int, int]]) -> dict[int, int]:
    counts: dict[int, int] = defaultdict(int)
    for r, _ in endpoints:
        counts[r] += 1
    return counts


def _cols_with_counts(endpoints: list[tuple[int, int]]) -> dict[int, int]:
    counts: dict[int, int] = defaultdict(int)
    for _, c in endpoints:
        counts[c] += 1
    return counts


def _has_blocked_segment(shape: tuple[int, int], objects: list[dict[str, object]]) -> bool:
    rays = _render_rays_only(shape, objects)
    endpoints = [(int(obj["endpoint"][0]), int(obj["endpoint"][1]), int(obj["color"])) for obj in objects]
    for axis, fixed, start, stop in _segments_from_endpoints(endpoints):
        if axis == "row":
            span = rays[fixed, start + 1 : stop]
        else:
            span = rays[start + 1 : stop, fixed]
        if span.size and np.any(span != BG) and np.any(span == BG):
            return True
    return False


def _features(shape: tuple[int, int], objects: list[dict[str, object]]) -> dict[str, bool]:
    endpoints = [tuple(map(int, obj["endpoint"])) for obj in objects]
    row_counts = _rows_with_counts(endpoints)
    col_counts = _cols_with_counts(endpoints)
    has_row = any(count >= 2 for count in row_counts.values())
    has_col = any(count >= 2 for count in col_counts.values())
    has_triple = any(count >= 3 for count in row_counts.values()) or any(count >= 3 for count in col_counts.values())
    return {
        "row": has_row,
        "col": has_col,
        "both": has_row and has_col,
        "triple": has_triple,
        "blocked": _has_blocked_segment(shape, objects),
    }


def _pick_distinct(values: list[int], count: int, min_gap: int) -> list[int] | None:
    candidates = []
    for combo in itertools.combinations(values, count):
        if all(b - a >= min_gap for a, b in zip(combo, combo[1:])):
            candidates.append(combo)
    if not candidates:
        return None
    return list(random.choice(candidates))


def _build_row_pair(shape: tuple[int, int], colors: list[int]) -> list[dict[str, object]] | None:
    row = random.randrange(2, shape[0] - 2)
    left_dir = random.choice(ROW_DIRS[0])
    right_dir = random.choice(ROW_DIRS[1])
    left = _endpoint_object(shape, (row, 0), left_dir, colors[0])
    right = _endpoint_object(shape, (row, shape[1] - 1), right_dir, colors[1])
    if left is None or right is None:
        return None
    if not _can_add_object([], left):
        return None
    if not _can_add_object([left], right):
        return None
    return [left, right]


def _build_col_pair(shape: tuple[int, int], colors: list[int]) -> list[dict[str, object]] | None:
    col = random.randrange(2, shape[1] - 2)
    top_dir = random.choice(COL_DIRS[0])
    bottom_dir = random.choice(COL_DIRS[1])
    top = _endpoint_object(shape, (0, col), top_dir, colors[0])
    bottom = _endpoint_object(shape, (shape[0] - 1, col), bottom_dir, colors[1])
    if top is None or bottom is None:
        return None
    if not _can_add_object([], top):
        return None
    if not _can_add_object([top], bottom):
        return None
    return [top, bottom]


def _build_top_triple(shape: tuple[int, int], colors: list[int]) -> list[dict[str, object]] | None:
    cols = _pick_distinct(list(range(1, shape[1] - 1)), 3, 2)
    if cols is None:
        return None
    objects: list[dict[str, object]] = []
    for color, col in zip(colors, cols):
        direction = random.choice(((-1, -1), (-1, 1)))
        obj = _endpoint_object(shape, (0, col), direction, color)
        if obj is None or not _can_add_object(objects, obj):
            return None
        objects.append(obj)
    return objects


def _build_blocker(shape: tuple[int, int], color: int, target_row: int, blocked_cols: set[int]) -> dict[str, object] | None:
    for _ in range(160):
        obj = _random_object(shape, color)
        if obj is None:
            continue
        rows = {r for r, _ in obj["path"]}
        if target_row not in rows:
            continue
        crossing_cols = {c for r, c in obj["path"] if r == target_row and c not in blocked_cols}
        if crossing_cols:
            return obj
    return None


def _build_objects_for_mode(shape: tuple[int, int], mode: str, colors: list[int]) -> list[dict[str, object]] | None:
    objects: list[dict[str, object]] = []
    used = 0
    if mode == "row":
        pair = _build_row_pair(shape, colors[:2])
        if pair is None:
            return None
        objects.extend(pair)
        used = 2
    elif mode == "col":
        pair = _build_col_pair(shape, colors[:2])
        if pair is None:
            return None
        objects.extend(pair)
        used = 2
    elif mode == "both":
        row_pair = _build_row_pair(shape, colors[:2])
        col_pair = _build_col_pair(shape, colors[2:4])
        if row_pair is None or col_pair is None:
            return None
        objects.extend(row_pair)
        for obj in col_pair:
            if not _can_add_object(objects, obj):
                return None
            objects.append(obj)
        used = 4
    elif mode == "triple":
        triple = _build_top_triple(shape, colors[:3])
        if triple is None:
            return None
        objects.extend(triple)
        used = 3
    else:
        pair = _build_row_pair(shape, colors[:2])
        if pair is None:
            return None
        objects.extend(pair)
        used = 2
        target_row = int(pair[0]["endpoint"][0])
        blocker = _build_blocker(shape, colors[2], target_row, {0, shape[1] - 1})
        if blocker is None or not _can_add_object(objects, blocker):
            return None
        objects.append(blocker)
        used = 3

    while used < len(colors):
        extra = _random_object(shape, colors[used])
        if extra is None or not _can_add_object(objects, extra):
            return None
        objects.append(extra)
        used += 1
    return objects


def _render_input(shape: tuple[int, int], objects: list[dict[str, object]]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for obj in objects:
        source_r, source_c = obj["source"]
        marker_r, marker_c = obj["marker"]
        grid[source_r, source_c] = int(obj["color"])
        grid[marker_r, marker_c] = MARKER
    return grid


def _validate_candidate(inp: np.ndarray, objects: list[dict[str, object]], mode: str) -> bool:
    parsed = _extract_objects(inp)
    parsed_signature = sorted(
        (int(obj["color"]), tuple(obj["source"]), tuple(obj["direction"]), tuple(obj["endpoint"]))
        for obj in parsed
    )
    built_signature = sorted(
        (int(obj["color"]), tuple(obj["source"]), tuple(obj["direction"]), tuple(obj["endpoint"]))
        for obj in objects
    )
    if parsed_signature != built_signature:
        return False

    expected = _render_expected(inp.shape, objects)
    if not np.array_equal(main(inp), expected):
        return False

    features = _features(inp.shape, objects)
    if mode == "row" and not features["row"]:
        return False
    if mode == "col" and not features["col"]:
        return False
    if mode == "both" and not features["both"]:
        return False
    if mode == "triple" and not features["triple"]:
        return False
    if mode == "blocked" and not features["blocked"]:
        return False
    return True


def generate_input() -> np.ndarray:
    mode = random.choices(
        population=["row", "col", "both", "triple", "blocked"],
        weights=[2, 2, 2, 2, 3],
        k=1,
    )[0]
    for _ in range(GEN_ATTEMPTS):
        height = random.randint(10, 16)
        width = random.randint(10, 16)
        if mode in {"both", "blocked"}:
            height = max(height, 12)
            width = max(width, 12)

        color_count = 4 if mode == "both" else random.choice((3, 4))
        colors = random.sample(COLOR_POOL, color_count)
        objects = _build_objects_for_mode((height, width), mode, colors)
        if objects is None:
            continue

        inp = _render_input((height, width), objects)
        if _validate_candidate(inp, objects, mode):
            return inp
    raise RuntimeError(f"failed to generate a valid input for mode {mode}")
