from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
MARKER_COLORS = (1, 2, 3, 4)
PINK = 6

TEMPLATE_REGION_SIDE = 4
TEMPLATE_SHAPES = (
    np.array([[PINK, PINK, PINK]], dtype=int),
    np.array([[PINK], [PINK], [PINK]], dtype=int),
    np.array(
        [
            [0, PINK, 0],
            [PINK, PINK, PINK],
            [0, PINK, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [PINK, PINK, PINK],
            [0, PINK, 0],
            [0, PINK, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, PINK, 0],
            [0, PINK, 0],
            [PINK, PINK, PINK],
        ],
        dtype=int,
    ),
    np.array(
        [
            [PINK, 0, 0],
            [PINK, PINK, PINK],
            [PINK, 0, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, 0, PINK],
            [PINK, PINK, PINK],
            [0, 0, PINK],
        ],
        dtype=int,
    ),
)

GEN_ATTEMPTS = 600
PLACEMENT_ATTEMPTS = 250
PAIRING_ATTEMPTS = 200
MIN_ROW_COL_GAP = 3
MIN_MARKER_CHEB = 5


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, bb: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = map(int, bb)
    return np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int).copy()


def paste(
    dst: np.ndarray,
    patch: np.ndarray,
    r0: int,
    c0: int,
    *,
    transparent: int | None = None,
    clip: bool = False,
) -> None:
    h, w = map(int, patch.shape)
    r0 = int(r0)
    c0 = int(c0)
    rr0, cc0 = 0, 0
    rr1, cc1 = h, w

    if clip:
        if r0 < 0:
            rr0 = -r0
            r0 = 0
        if c0 < 0:
            cc0 = -c0
            c0 = 0
        rr1 = min(rr1, dst.shape[0] - r0 + rr0)
        cc1 = min(cc1, dst.shape[1] - c0 + cc0)
        if rr0 >= rr1 or cc0 >= cc1:
            return
    elif r0 < 0 or c0 < 0 or r0 + h > dst.shape[0] or c0 + w > dst.shape[1]:
        raise ValueError("paste out of bounds")

    src = np.asarray(patch[rr0:rr1, cc0:cc1], dtype=int)
    view = dst[r0 : r0 + src.shape[0], c0 : c0 + src.shape[1]]
    if transparent is None:
        view[:, :] = src
        return
    mask = src != int(transparent)
    view[mask] = src[mask]


def _is_4_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if not in_bounds(mask.shape, nr, nc) or not bool(mask[nr, nc]):
                continue
            key = (int(nr), int(nc))
            if key not in seen:
                seen.add(key)
                q.append(key)
    return len(seen) == int(mask.sum())


def _nearest_horizontal_edge(r: int, h: int) -> int:
    top = int(r)
    bottom = int(h) - 1 - int(r)
    if top == bottom:
        raise ValueError("horizontal boundary tie is not allowed")
    return 0 if top < bottom else int(h) - 1


def _nearest_vertical_edge(c: int, w: int) -> int:
    left = int(c)
    right = int(w) - 1 - int(c)
    if left == right:
        raise ValueError("vertical boundary tie is not allowed")
    return 0 if left < right else int(w) - 1


def _extract_template(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    bb = bbox(grid == PINK)
    if bb is None:
        raise ValueError("missing template")
    return crop_bbox(grid, bb), bb


def _find_markers(grid: np.ndarray) -> list[dict[str, int]]:
    coords = np.argwhere((grid != BLACK) & (grid != PINK))
    markers: list[dict[str, int]] = []
    seen_rows: set[int] = set()
    seen_cols: set[int] = set()
    seen_colors: set[int] = set()
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        color = int(grid[rr, cc])
        if color not in MARKER_COLORS:
            raise ValueError("unexpected non-template foreground color")
        if rr in seen_rows or cc in seen_cols or color in seen_colors:
            raise ValueError("markers must have unique rows, cols, and colors")
        seen_rows.add(rr)
        seen_cols.add(cc)
        seen_colors.add(color)
        markers.append({"r": rr, "c": cc, "color": color})
    if len(markers) not in (3, 4):
        raise ValueError("expected 3 or 4 markers")
    markers.sort(key=lambda m: (m["r"], m["c"]))
    return markers


def _annotate_markers(markers: list[dict[str, int]], shape: tuple[int, int]) -> list[dict[str, int]]:
    h, w = map(int, shape)
    out: list[dict[str, int]] = []
    for idx, marker in enumerate(markers):
        r = int(marker["r"])
        c = int(marker["c"])
        v_edge = _nearest_horizontal_edge(r, h)
        h_edge = _nearest_vertical_edge(c, w)
        vr0, vr1 = sorted((r, v_edge))
        hc0, hc1 = sorted((c, h_edge))
        out.append(
            {
                "id": idx,
                "r": r,
                "c": c,
                "color": int(marker["color"]),
                "v_edge": int(v_edge),
                "h_edge": int(h_edge),
                "vr0": int(vr0),
                "vr1": int(vr1),
                "hc0": int(hc0),
                "hc1": int(hc1),
            }
        )
    return out


def _compute_intersections(markers: list[dict[str, int]]) -> list[tuple[int, int, int, int]]:
    points: list[tuple[int, int, int, int]] = []
    for row_marker in markers:
        for col_marker in markers:
            if row_marker["id"] == col_marker["id"]:
                continue
            if not (row_marker["hc0"] <= col_marker["c"] <= row_marker["hc1"]):
                continue
            if not (col_marker["vr0"] <= row_marker["r"] <= col_marker["vr1"]):
                continue
            points.append(
                (
                    int(row_marker["r"]),
                    int(col_marker["c"]),
                    int(row_marker["id"]),
                    int(col_marker["id"]),
                )
            )
    points.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    return points


def _naive_points(markers: list[dict[str, int]], *, include_self: bool) -> list[tuple[int, int, int, int]]:
    points = []
    for row_marker in markers:
        for col_marker in markers:
            if not include_self and row_marker["id"] == col_marker["id"]:
                continue
            points.append(
                (
                    int(row_marker["r"]),
                    int(col_marker["c"]),
                    int(row_marker["id"]),
                    int(col_marker["id"]),
                )
            )
    points.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    return points


def _draw_rays(base: np.ndarray, markers: list[dict[str, int]]) -> np.ndarray:
    out = np.asarray(base, dtype=int).copy()
    for marker in markers:
        color = int(marker["color"])
        out[marker["vr0"] : marker["vr1"] + 1, marker["c"]] = color
        out[marker["r"], marker["hc0"] : marker["hc1"] + 1] = color
    return out


def _stamp_points(
    base: np.ndarray,
    template: np.ndarray,
    points: list[tuple[int, int, int, int]],
    markers: list[dict[str, int]] | None,
    *,
    draw_rays: bool,
) -> np.ndarray:
    out = np.asarray(base, dtype=int).copy()
    if draw_rays and markers is not None:
        out = _draw_rays(out, markers)
    th, tw = map(int, template.shape)
    for r, c, _row_id, _col_id in sorted(points, key=lambda item: (item[0], item[1])):
        paste(out, template, int(r) - th // 2, int(c) - tw // 2, transparent=BLACK, clip=True)
    if markers is not None:
        for marker in markers:
            out[marker["r"], marker["c"]] = marker["color"]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend singleton-marker rays to the nearest borders, detect the genuine ray intersections, and stamp the magenta template there.

    Input structure:
    - A black background contains 3 or 4 isolated singleton markers in distinct colors chosen from blue, red, green, and yellow.
    - The lower-left 4x4 area contains one small 4-connected magenta template shape on black.
    - Marker rows and columns are unique, and each marker has a unique nearest horizontal edge and unique nearest vertical edge.

    Rule:
    - For each marker, draw its vertical ray from the marker to the nearest top/bottom border and its horizontal ray from the marker to the nearest left/right border, both inclusive and in the marker's own color.
    - A valid emergent node is a coordinate where one marker's horizontal segment and a different marker's vertical segment actually cross.
    - Crop the magenta template to its bounding box and stamp that cropped template at every valid node, center-aligned and clipped at the canvas boundary if necessary.
    - Draw all rays before any template copies, stamp nodes in row-major order, and finally restore the original marker pixels in their source colors.

    Key insight:
    - The rows and columns alone are not enough: a row/column pair becomes a node only when the two finite ray segments truly reach that coordinate, so nearest-border direction is essential.
    """
    g = np.asarray(grid, dtype=int)
    template, _template_bb = _extract_template(g)
    markers = _annotate_markers(_find_markers(g), g.shape)
    points = _compute_intersections(markers)
    return _stamp_points(g, template, points, markers, draw_rays=True)


def _sample_spaced(pool: list[int], count: int, gap: int) -> list[int] | None:
    if count == 0:
        return []
    if len(pool) < count:
        return None
    for _ in range(200):
        pick = sorted(random.sample(pool, count))
        if all(b - a >= int(gap) for a, b in zip(pick, pick[1:])):
            return pick
    return None


def _choose_template() -> np.ndarray:
    return np.array(random.choice(TEMPLATE_SHAPES), dtype=int, copy=True)


def _place_template(grid: np.ndarray, template: np.ndarray) -> None:
    h, w = map(int, template.shape)
    row0 = grid.shape[0] - TEMPLATE_REGION_SIDE + random.randint(0, TEMPLATE_REGION_SIDE - h)
    col0 = random.randint(0, TEMPLATE_REGION_SIDE - w)
    paste(grid, template, row0, col0, transparent=BLACK, clip=False)


def _pairwise_chebyshev_ok(markers: list[dict[str, int]]) -> bool:
    for i, a in enumerate(markers):
        for b in markers[i + 1 :]:
            if max(abs(a["r"] - b["r"]), abs(a["c"] - b["c"])) < MIN_MARKER_CHEB:
                return False
    return True


def _sample_marker_layout(h: int, w: int, count: int) -> list[dict[str, int]] | None:
    usable_rows = [r for r in range(1, h - TEMPLATE_REGION_SIDE) if 2 * r != h - 1]
    usable_cols = [c for c in range(TEMPLATE_REGION_SIDE, w - 1) if 2 * c != w - 1]
    top_rows = [r for r in usable_rows if r < h - 1 - r]
    bottom_rows = [r for r in usable_rows if r > h - 1 - r]
    left_cols = [c for c in usable_cols if c < w - 1 - c]
    right_cols = [c for c in usable_cols if c > w - 1 - c]

    if count == 3:
        families = (
            (top_rows, right_cols, False),   # U,R
            (top_rows, left_cols, True),     # U,L
            (bottom_rows, right_cols, True), # D,R
            (bottom_rows, left_cols, False), # D,L
        )
        for _ in range(PLACEMENT_ATTEMPTS):
            row_pool, col_pool, reverse_cols = random.choice(families)
            if len(row_pool) < 3 or len(col_pool) < 3:
                continue
            rows = sorted(random.sample(row_pool, 3))
            cols = sorted(random.sample(col_pool, 3), reverse=reverse_cols)
            markers = [{"r": int(r), "c": int(c)} for r, c in zip(rows, cols)]
            if _pairwise_chebyshev_ok(markers):
                colors = random.sample(list(MARKER_COLORS), 3)
                for marker, color in zip(markers, colors):
                    marker["color"] = int(color)
                return markers

    pair_families = (
        ("ur", "dl"),
        ("ul", "dr"),
    )
    for _ in range(PLACEMENT_ATTEMPTS):
        pair_a, pair_b = random.choice(pair_families)
        if len(top_rows) < 2 or len(bottom_rows) < 2 or len(left_cols) < 2 or len(right_cols) < 2:
            continue
        top_pair_rows = sorted(random.sample(top_rows, 2))
        bottom_pair_rows = sorted(random.sample(bottom_rows, 2))
        left_pair_cols = sorted(random.sample(left_cols, 2))
        right_pair_cols = sorted(random.sample(right_cols, 2))

        if pair_a == "ur":
            first_pair = [(top_pair_rows[0], right_pair_cols[0]), (top_pair_rows[1], right_pair_cols[1])]
            second_pair = [(bottom_pair_rows[0], left_pair_cols[0]), (bottom_pair_rows[1], left_pair_cols[1])]
        else:
            first_pair = [(top_pair_rows[0], left_pair_cols[1]), (top_pair_rows[1], left_pair_cols[0])]
            second_pair = [(bottom_pair_rows[0], right_pair_cols[1]), (bottom_pair_rows[1], right_pair_cols[0])]

        markers = [{"r": int(r), "c": int(c)} for r, c in first_pair + second_pair]
        if not _pairwise_chebyshev_ok(markers):
            continue
        colors = random.sample(list(MARKER_COLORS), 4)
        for marker, color in zip(markers, colors):
            marker["color"] = int(color)
        return markers
    return None


def _template_coords(template: np.ndarray) -> list[tuple[int, int]]:
    return [tuple(map(int, rc)) for rc in np.argwhere(template == PINK)]


def _template_is_valid(template: np.ndarray) -> bool:
    if template.ndim != 2 or template.size == 0:
        return False
    if template.shape[0] % 2 == 0 or template.shape[1] % 2 == 0:
        return False
    if not np.all((template == BLACK) | (template == PINK)):
        return False
    if not (3 <= int((template == PINK).sum()) <= 5):
        return False
    return _is_4_connected(template == PINK)


def _validate_generated_input(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False
    h, w = map(int, grid.shape)
    if h > 30 or w > 30:
        return False

    pink_coords = np.argwhere(grid == PINK)
    if pink_coords.size == 0:
        return False
    if np.any(pink_coords[:, 0] < h - TEMPLATE_REGION_SIDE) or np.any(pink_coords[:, 1] >= TEMPLATE_REGION_SIDE):
        return False

    try:
        template, template_bb = _extract_template(grid)
        markers = _annotate_markers(_find_markers(grid), grid.shape)
    except ValueError:
        return False

    if not _template_is_valid(template):
        return False
    tr0, tc0, tr1, tc1 = template_bb
    if tr0 < h - TEMPLATE_REGION_SIDE or tc0 < 0 or tr1 >= h or tc1 >= TEMPLATE_REGION_SIDE:
        return False

    template_mask = np.zeros_like(grid, dtype=bool)
    template_mask[grid == PINK] = True
    for marker in markers:
        if np.any(template_mask[marker["vr0"] : marker["vr1"] + 1, marker["c"]]):
            return False
        if np.any(template_mask[marker["r"], marker["hc0"] : marker["hc1"] + 1]):
            return False

    points = _compute_intersections(markers)
    naive_points = _naive_points(markers, include_self=False)
    self_points = _naive_points(markers, include_self=True)

    if not (2 <= len(points) <= 4):
        return False
    if len(points) >= len(naive_points):
        return False

    participants = {m["id"]: 0 for m in markers}
    for _r, _c, row_id, col_id in points:
        participants[row_id] += 1
        participants[col_id] += 1
    if any(hit == 0 for hit in participants.values()):
        return False

    actual_out = _stamp_points(grid, template, points, markers, draw_rays=True)
    rays_only_out = _draw_rays(grid, markers)
    stamps_only_out = _stamp_points(grid, template, points, markers, draw_rays=False)
    naive_out = _stamp_points(grid, template, naive_points, markers, draw_rays=True)
    self_out = _stamp_points(grid, template, self_points, markers, draw_rays=True)

    if np.array_equal(actual_out, rays_only_out):
        return False
    if np.array_equal(actual_out, stamps_only_out):
        return False
    if np.array_equal(actual_out, naive_out):
        return False
    if np.array_equal(actual_out, self_out):
        return False

    added_magenta = (actual_out == PINK) & (grid != PINK)
    if int(added_magenta.sum()) == 0:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        count = random.choice((3, 4))
        if count == 3:
            h = random.randint(20, 24)
            w = random.randint(20, 24)
        else:
            h = random.randint(20, 26)
            w = random.randint(20, 26)
        markers = _sample_marker_layout(h, w, count)
        if markers is None:
            continue

        grid = np.zeros((h, w), dtype=int)
        template = _choose_template()
        _place_template(grid, template)
        for marker in markers:
            grid[marker["r"], marker["c"]] = marker["color"]

        if _validate_generated_input(grid):
            return grid
    raise RuntimeError("failed to generate a valid ray-intersection stamping input")
