from __future__ import annotations

from collections import defaultdict
import random

import numpy as np


BLACK = 0
COLOR_POOL = tuple(range(1, 10))
CORNERS = ("tl", "tr", "bl", "br")
CORNER_TO_MARK = {
    "tl": (0, 0),
    "tr": (0, 1),
    "bl": (1, 0),
    "br": (1, 1),
}
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
NEIGHBORS8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)

GRID_MIN = {2: 18, 3: 22, 4: 24}
GRID_MAX = 30
ENVELOPE_COUNT_CHOICES = (2, 3, 4)
ENVELOPE_INTERIOR_MIN = 4
ENVELOPE_INTERIOR_MAX = 10
STAMP_BOX_MIN = 2
STAMP_BOX_MAX = 4
STAMP_AREA_MIN = 3
STAMP_AREA_MAX = 7
GEN_RETRIES = 300


def _prefix_sum(mask: np.ndarray) -> np.ndarray:
    return np.pad(mask.astype(int), ((1, 0), (1, 0)), constant_values=0).cumsum(0).cumsum(1)


def _rect_sum(ps: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> int:
    if r0 > r1 or c0 > c1:
        return 0
    return int(ps[r1 + 1, c1 + 1] - ps[r0, c1 + 1] - ps[r1 + 1, c0] + ps[r0, c0])


def _ring_coords(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    coords = [(r0, c) for c in range(c0, c1 + 1)]
    coords += [(r, c1) for r in range(r0 + 1, r1)]
    if r1 > r0:
        coords += [(r1, c) for c in range(c1, c0 - 1, -1)]
    if c1 > c0:
        coords += [(r, c0) for r in range(r1 - 1, r0, -1)]
    return coords


def _components(mask: np.ndarray, connectivity: int = 8) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    neighbors = NEIGHBORS8 if connectivity == 8 else NEIGHBORS4
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in neighbors:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _corner_name(r: int, c: int, r0: int, c0: int, r1: int, c1: int) -> str:
    if (r, c) == (r0, c0):
        return "tl"
    if (r, c) == (r0, c1):
        return "tr"
    if (r, c) == (r1, c0):
        return "bl"
    if (r, c) == (r1, c1):
        return "br"
    raise AssertionError("marker pixel must be on a rectangle corner")


def _inner_corner(env: dict) -> tuple[int, int]:
    r0, c0, r1, c1 = env["bbox"]
    corner = env["corner"]
    if corner == "tl":
        return (r0 + 1, c0 + 1)
    if corner == "tr":
        return (r0 + 1, c1 - 1)
    if corner == "bl":
        return (r1 - 1, c0 + 1)
    return (r1 - 1, c1 - 1)


def _find_envelopes(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    nonblack = g != BLACK
    ps = _prefix_sum(nonblack)
    h, w = g.shape

    rows_to_cols: dict[int, list[int]] = defaultdict(list)
    cols_to_rows: dict[int, list[int]] = defaultdict(list)
    for r, c in np.argwhere(nonblack):
        rows_to_cols[int(r)].append(int(c))
        cols_to_rows[int(c)].append(int(r))
    if not rows_to_cols:
        return []
    for cols in rows_to_cols.values():
        cols.sort()
    col_sets = {col: set(rows) for col, rows in cols_to_rows.items()}

    envelopes: list[dict] = []
    seen_bboxes: set[tuple[int, int, int, int]] = set()
    for r0, cols in rows_to_cols.items():
        for idx, c0 in enumerate(cols):
            lower_rows = [r1 for r1 in cols_to_rows[c0] if r1 >= r0 + 2]
            if not lower_rows:
                continue
            for c1 in cols[idx + 1 :]:
                if c1 < c0 + 2:
                    continue
                rows_at_c1 = col_sets[c1]
                for r1 in lower_rows:
                    if r1 not in rows_at_c1:
                        continue
                    bbox = (r0, c0, r1, c1)
                    if bbox in seen_bboxes:
                        continue
                    seen_bboxes.add(bbox)
                    total = _rect_sum(ps, r0, c0, r1, c1)
                    inner = _rect_sum(ps, r0 + 1, c0 + 1, r1 - 1, c1 - 1)
                    perimeter = 2 * ((r1 - r0 + 1) + (c1 - c0 + 1)) - 4
                    if inner != 0 or total - inner != perimeter:
                        continue
                    ring = _ring_coords(r0, c0, r1, c1)
                    counts: dict[int, int] = defaultdict(int)
                    for rr, cc in ring:
                        counts[int(g[rr, cc])] += 1
                    if BLACK in counts or len(counts) != 2:
                        continue
                    counts_by_freq = sorted(counts.items(), key=lambda item: item[1])
                    if counts_by_freq[0][1] != 1 or counts_by_freq[1][1] != perimeter - 1:
                        continue
                    mark_color = int(counts_by_freq[0][0])
                    border_color = int(counts_by_freq[1][0])
                    marker_coords = [
                        coord
                        for coord in ((r0, c0), (r0, c1), (r1, c0), (r1, c1))
                        if int(g[coord]) == mark_color
                    ]
                    if len(marker_coords) != 1:
                        continue
                    mr, mc = marker_coords[0]
                    env = {
                        "bbox": bbox,
                        "border_color": border_color,
                        "mark_color": mark_color,
                        "mark_coord": (mr, mc),
                        "corner": _corner_name(mr, mc, r0, c0, r1, c1),
                    }
                    env["inner_corner"] = _inner_corner(env)
                    envelopes.append(env)

    envelopes.sort(key=lambda env: env["bbox"])
    return envelopes


def _find_stamps(grid: np.ndarray, envelopes: list[dict]) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    marker_to_env = {env["mark_color"]: env for env in envelopes}
    assert len(marker_to_env) == len(envelopes), "marker colors must be unique"

    border_mask = np.zeros_like(g, dtype=bool)
    for env in envelopes:
        r0, c0, r1, c1 = env["bbox"]
        for rr, cc in _ring_coords(r0, c0, r1, c1):
            border_mask[rr, cc] = True

    stamp_mask = (g != BLACK) & ~border_mask
    components = _components(stamp_mask, connectivity=8)
    stamps: list[dict] = []
    for comp in components:
        assert len(comp) >= 2, "a stamp must have at least two pixels"
        marker_pixels = [
            (int(r), int(c), int(g[r, c]))
            for r, c in comp
            if int(g[r, c]) in marker_to_env
        ]
        assert len(marker_pixels) == 1, "each stamp must contain exactly one marker-colored pixel"
        sr, sc, special_color = marker_pixels[0]
        body_colors = {int(g[r, c]) for r, c in comp if (int(r), int(c)) != (sr, sc)}
        assert len(body_colors) == 1, "stamp body must use exactly one color"
        body_color = int(next(iter(body_colors)))
        rel_coords = [(int(r) - sr, int(c) - sc) for r, c in comp]
        stamps.append(
            {
                "special_color": special_color,
                "body_color": body_color,
                "special_coord": (sr, sc),
                "rel_coords": rel_coords,
            }
        )

    assert len(stamps) == len(envelopes), "expected one stamp per envelope"
    special_colors = {stamp["special_color"] for stamp in stamps}
    assert special_colors == set(marker_to_env), "stamp colors must match envelope markers exactly"
    return stamps


def _apply_stamps(grid: np.ndarray, envelopes: list[dict], stamps: list[dict]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    env_by_marker = {env["mark_color"]: env for env in envelopes}
    for stamp in stamps:
        env = env_by_marker[stamp["special_color"]]
        anchor_r, anchor_c = env["inner_corner"]
        r0, c0, r1, c1 = env["bbox"]
        for dr, dc in stamp["rel_coords"]:
            if dr == 0 and dc == 0:
                continue
            rr = anchor_r + dr
            cc = anchor_c + dc
            assert r0 < rr < r1 and c0 < cc < c1, "stamp must land inside its envelope"
            out[rr, cc] = stamp["body_color"]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp each outside sprite into the matching envelope corner chosen by the corner marker.

    Input structure:
    - The background is black.
    - Several large hollow rectangles ("envelopes") have single-pixel colored borders, black interiors, and exactly one corner pixel replaced by a unique marker color.
    - Several small connected sprites ("stamps") sit outside the envelopes. Each stamp has one special pixel whose color matches one envelope marker; every other pixel in that stamp shares one body color.

    Rule:
    - Detect every envelope, read its unique marker color, and note which corner contains that marker.
    - Detect every stamp and pair it with the envelope whose marker color matches the stamp's special pixel color.
    - Align the stamp's special pixel to the inner corner cell just inside the marked envelope corner, then copy the whole stamp into the envelope interior without rotating it.
    - Do not draw the special pixel itself, so the aligned anchor cell stays black. Leave the original stamps, the envelope borders, and the background unchanged.

    Key insight:
    - The hidden single-pixel color inside each stamp is the pairing key, while the marker corner on the envelope decides where that keyed stamp must be anchored.
    """
    g = np.asarray(grid, dtype=int)
    envelopes = _find_envelopes(g)
    assert envelopes, "expected at least one envelope"
    stamps = _find_stamps(g, envelopes)
    return _apply_stamps(g, envelopes, stamps)


def _grow_shape(
    rng: random.Random,
    box_h: int,
    box_w: int,
    start: tuple[int, int],
    target_area: int,
) -> set[tuple[int, int]]:
    cells = {start}
    frontier = [start]
    while len(cells) < target_area and frontier:
        base = rng.choice(frontier)
        candidates = []
        for dr, dc in NEIGHBORS4:
            nr, nc = base[0] + dr, base[1] + dc
            if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells:
                candidates.append((nr, nc))
        if not candidates:
            frontier.remove(base)
            continue
        pick = rng.choice(candidates)
        cells.add(pick)
        frontier.append(pick)
    return cells


def _make_stamp(rng: random.Random, corner: str, marker_color: int, body_color: int) -> dict | None:
    box_h = rng.randint(STAMP_BOX_MIN, STAMP_BOX_MAX)
    box_w = rng.randint(STAMP_BOX_MIN, STAMP_BOX_MAX)
    if corner == "tl":
        anchor = (0, 0)
    elif corner == "tr":
        anchor = (0, box_w - 1)
    elif corner == "bl":
        anchor = (box_h - 1, 0)
    else:
        anchor = (box_h - 1, box_w - 1)

    target_area = rng.randint(STAMP_AREA_MIN, min(STAMP_AREA_MAX, box_h * box_w))
    cells = _grow_shape(rng, box_h, box_w, anchor, target_area)
    if len(cells) < STAMP_AREA_MIN:
        return None

    rel_coords = [(r - anchor[0], c - anchor[1]) for r, c in cells]
    row_offsets = [dr for dr, _ in rel_coords]
    col_offsets = [dc for _, dc in rel_coords]
    needed_h = max(abs(min(row_offsets)), abs(max(row_offsets))) + 1
    needed_w = max(abs(min(col_offsets)), abs(max(col_offsets))) + 1
    return {
        "corner": corner,
        "marker_color": marker_color,
        "body_color": body_color,
        "rel_coords": rel_coords,
        "needed_h": needed_h,
        "needed_w": needed_w,
        "bbox_h": max(row_offsets) - min(row_offsets) + 1,
        "bbox_w": max(col_offsets) - min(col_offsets) + 1,
        "min_dr": min(row_offsets),
        "min_dc": min(col_offsets),
    }


def _can_place(occupied: np.ndarray, top: int, left: int, height: int, width: int, margin: int = 1) -> bool:
    h, w = occupied.shape
    r0 = top - margin
    c0 = left - margin
    r1 = top + height - 1 + margin
    c1 = left + width - 1 + margin
    if r0 < 0 or c0 < 0 or r1 >= h or c1 >= w:
        return False
    return not bool(occupied[r0 : r1 + 1, c0 : c1 + 1].any())


def _mark_occupied(occupied: np.ndarray, top: int, left: int, height: int, width: int, margin: int = 1) -> None:
    r0 = top - margin
    c0 = left - margin
    r1 = top + height - 1 + margin
    c1 = left + width - 1 + margin
    occupied[r0 : r1 + 1, c0 : c1 + 1] = True


def _place_bbox(
    rng: random.Random,
    occupied: np.ndarray,
    height: int,
    width: int,
    attempts: int = 200,
) -> tuple[int, int] | None:
    grid_h, grid_w = occupied.shape
    if height + 2 > grid_h or width + 2 > grid_w:
        return None
    for _ in range(attempts):
        top = rng.randint(1, grid_h - height - 1)
        left = rng.randint(1, grid_w - width - 1)
        if _can_place(occupied, top, left, height, width, margin=1):
            _mark_occupied(occupied, top, left, height, width, margin=1)
            return (top, left)
    return None


def _draw_envelope(grid: np.ndarray, env: dict) -> None:
    r0, c0, r1, c1 = env["bbox"]
    color = env["border_color"]
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    mark_r, mark_c = env["mark_coord"]
    grid[mark_r, mark_c] = env["mark_color"]


def _draw_stamp(grid: np.ndarray, stamp: dict, top: int, left: int) -> tuple[int, int]:
    coords = []
    for dr, dc in stamp["rel_coords"]:
        rr = top + (dr - stamp["min_dr"])
        cc = left + (dc - stamp["min_dc"])
        coords.append((rr, cc, dr, dc))
    special = None
    for rr, cc, dr, dc in coords:
        if dr == 0 and dc == 0:
            grid[rr, cc] = stamp["marker_color"]
            special = (rr, cc)
        else:
            grid[rr, cc] = stamp["body_color"]
    assert special is not None
    return special


def _expected_output(grid: np.ndarray, envelopes: list[dict], stamps: list[dict]) -> np.ndarray:
    return _apply_stamps(grid, envelopes, stamps)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_env = rng.choice(ENVELOPE_COUNT_CHOICES)
    grid_h = rng.randint(GRID_MIN[n_env], GRID_MAX)
    grid_w = rng.randint(GRID_MIN[n_env], GRID_MAX)
    grid = np.zeros((grid_h, grid_w), dtype=int)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)

    colors = list(COLOR_POOL)
    rng.shuffle(colors)
    border_colors = colors[:n_env]
    marker_colors = colors[n_env : 2 * n_env]
    if len(marker_colors) != n_env:
        return None

    spare_body_colors = [c for c in COLOR_POOL if c not in marker_colors and c not in border_colors]
    body_pool = spare_body_colors or [c for c in COLOR_POOL if c not in marker_colors]
    if not body_pool:
        return None

    corners = rng.sample(list(CORNERS), n_env)
    stamp_specs = []
    for idx in range(n_env):
        body_color = rng.choice(body_pool)
        stamp = _make_stamp(rng, corners[idx], marker_colors[idx], body_color)
        if stamp is None:
            return None
        ih_low = max(ENVELOPE_INTERIOR_MIN, stamp["needed_h"])
        iw_low = max(ENVELOPE_INTERIOR_MIN, stamp["needed_w"])
        ih_high = min(ENVELOPE_INTERIOR_MAX, ih_low + 4)
        iw_high = min(ENVELOPE_INTERIOR_MAX, iw_low + 4)
        if ih_low > ih_high or iw_low > iw_high:
            return None
        stamp["interior_h"] = rng.randint(ih_low, ih_high)
        stamp["interior_w"] = rng.randint(iw_low, iw_high)
        stamp_specs.append(stamp)

    envelopes: list[dict] = []
    place_order = sorted(range(n_env), key=lambda i: stamp_specs[i]["interior_h"] * stamp_specs[i]["interior_w"], reverse=True)
    for idx in place_order:
        stamp = stamp_specs[idx]
        env_h = stamp["interior_h"] + 2
        env_w = stamp["interior_w"] + 2
        pos = _place_bbox(rng, occupied, env_h, env_w)
        if pos is None:
            return None
        top, left = pos
        bbox = (top, left, top + env_h - 1, left + env_w - 1)
        mark_dr, mark_dc = CORNER_TO_MARK[stamp["corner"]]
        mark_coord = (bbox[0] if mark_dr == 0 else bbox[2], bbox[1] if mark_dc == 0 else bbox[3])
        env = {
            "bbox": bbox,
            "border_color": border_colors[idx],
            "mark_color": marker_colors[idx],
            "mark_coord": mark_coord,
            "corner": stamp["corner"],
        }
        env["inner_corner"] = _inner_corner(env)
        envelopes.append(env)
        _draw_envelope(grid, env)

    placed_stamps: list[dict] = []
    for idx in sorted(range(n_env), key=lambda i: stamp_specs[i]["bbox_h"] * stamp_specs[i]["bbox_w"], reverse=True):
        stamp = stamp_specs[idx]
        pos = _place_bbox(rng, occupied, stamp["bbox_h"], stamp["bbox_w"])
        if pos is None:
            return None
        top, left = pos
        special_coord = _draw_stamp(grid, stamp, top, left)
        placed_stamps.append(
            {
                "special_color": stamp["marker_color"],
                "body_color": stamp["body_color"],
                "special_coord": special_coord,
                "rel_coords": stamp["rel_coords"],
            }
        )

    envelopes_found = _find_envelopes(grid)
    if len(envelopes_found) != n_env:
        return None
    try:
        stamps_found = _find_stamps(grid, envelopes_found)
    except AssertionError:
        return None
    if len(stamps_found) != n_env:
        return None

    expected = _expected_output(grid, envelopes, placed_stamps)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    if not np.any(actual != grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid stamp-envelope input")
