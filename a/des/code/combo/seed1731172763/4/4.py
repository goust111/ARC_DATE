import random

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))
BRANCH_TYPES = (
    "top",
    "bottom",
    "left",
    "right",
    "h_over_v_top",
    "h_over_v_bottom",
    "top_over_bottom",
)

GRID_H_RANGE = (18, 31)
GRID_W_RANGE = (18, 31)
FRAME_COUNT_RANGE = (2, 4)
INTERIOR_H_RANGE = (3, 10)
INTERIOR_W_RANGE = (3, 11)
PLACEMENT_PAD = 2
MAX_GENERATION_ATTEMPTS = 400
GEOMETRY_ATTEMPTS = 200


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _is_rect_frame(cells: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(cells)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    expected = 2 * h + 2 * w - 4
    if len(cells) != expected:
        return False

    cellset = set(cells)
    for r in range(r0, r1 + 1):
        if (r, c0) not in cellset or (r, c1) not in cellset:
            return False
    for c in range(c0, c1 + 1):
        if (r0, c) not in cellset or (r1, c) not in cellset:
            return False
    for r, c in cells:
        if r0 < r < r1 and c0 < c < c1:
            return False
    return True


def _choose_edge_from_distances(d_top: int, d_bottom: int, d_left: int, d_right: int) -> str:
    # Horizontal edges win ties; among them top wins; otherwise use left before right
    # as a stable fallback for underspecified vertical-only ties.
    candidates = [
        ("top", d_top, 0, 0, 0),
        ("bottom", d_bottom, 0, 1, 0),
        ("left", d_left, 1, 1, 0),
        ("right", d_right, 1, 1, 1),
    ]
    return min(candidates, key=lambda item: item[1:])[0]


def _choose_edge_for_marker(
    bbox: tuple[int, int, int, int], marker: tuple[int, int]
) -> str:
    r0, c0, r1, c1 = bbox
    mr, mc = marker
    return _choose_edge_from_distances(
        mr - (r0 + 1),
        (r1 - 1) - mr,
        mc - (c0 + 1),
        (c1 - 1) - mc,
    )


def _paint_inner_line(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    edge: str,
    color: int,
) -> None:
    r0, c0, r1, c1 = bbox
    if edge == "top":
        grid[r0 + 1, c0 + 1 : c1] = color
    elif edge == "bottom":
        grid[r1 - 1, c0 + 1 : c1] = color
    elif edge == "left":
        grid[r0 + 1 : r1, c0 + 1] = color
    else:
        grid[r0 + 1 : r1, c1 - 1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    In each hollow rectangular frame, use the marker's distances to the four inner
    edges to choose the nearest edge, then draw a full same-colored inner line there.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    out = g.copy()

    frames: list[tuple[tuple[int, int, int, int], int]] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) != bg):
        for comp in _components(g == color):
            if _is_rect_frame(comp):
                frames.append((_bbox(comp), color))

    frames.sort(key=lambda item: item[0])

    for bbox, frame_color in frames:
        r0, c0, r1, c1 = bbox
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        marker_locs = np.argwhere((interior != bg) & (interior != frame_color))
        if len(marker_locs) != 1:
            continue

        mr_local, mc_local = map(int, marker_locs[0])
        mr = r0 + 1 + mr_local
        mc = c0 + 1 + mc_local
        marker_color = int(g[mr, mc])
        edge = _choose_edge_for_marker(bbox, (mr, mc))

        out[mr, mc] = bg
        _paint_inner_line(out, bbox, edge, marker_color)

    return out


def _branch_for_position(r: int, c: int, interior_h: int, interior_w: int) -> str | None:
    d_top = r
    d_bottom = interior_h - 1 - r
    d_left = c
    d_right = interior_w - 1 - c
    dists = {
        "top": d_top,
        "bottom": d_bottom,
        "left": d_left,
        "right": d_right,
    }
    min_dist = min(dists.values())
    mins = {edge for edge, dist in dists.items() if dist == min_dist}
    chosen = _choose_edge_from_distances(d_top, d_bottom, d_left, d_right)

    if len(mins) == 1:
        return chosen
    if chosen == "top" and mins == {"top", "bottom"}:
        return "top_over_bottom"
    if chosen == "top" and "bottom" not in mins:
        return "h_over_v_top"
    if chosen == "bottom" and "top" not in mins:
        return "h_over_v_bottom"
    return None


def _positions_for_branch(interior_h: int, interior_w: int, branch: str) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []
    for r in range(interior_h):
        for c in range(interior_w):
            if _branch_for_position(r, c, interior_h, interior_w) == branch:
                positions.append((r, c))
    return positions


def _sample_geometry(
    rng: random.Random, branch: str
) -> tuple[int, int, int, int] | None:
    for _ in range(GEOMETRY_ATTEMPTS):
        interior_h = rng.randint(*INTERIOR_H_RANGE)
        interior_w = rng.randint(*INTERIOR_W_RANGE)
        positions = _positions_for_branch(interior_h, interior_w, branch)
        if not positions:
            continue
        marker_r, marker_c = rng.choice(positions)
        return interior_h, interior_w, marker_r, marker_c
    return None


def _find_positions(
    occupied: np.ndarray, frame_h: int, frame_w: int
) -> list[tuple[int, int]]:
    h, w = occupied.shape
    positions: list[tuple[int, int]] = []
    for r0 in range(h - frame_h + 1):
        for c0 in range(w - frame_w + 1):
            rr0 = max(0, r0 - PLACEMENT_PAD)
            cc0 = max(0, c0 - PLACEMENT_PAD)
            rr1 = min(h, r0 + frame_h + PLACEMENT_PAD)
            cc1 = min(w, c0 + frame_w + PLACEMENT_PAD)
            if not occupied[rr0:rr1, cc0:cc1].any():
                positions.append((r0, c0))
    return positions


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _build_expected(
    inp: np.ndarray,
    specs: list[dict[str, object]],
) -> np.ndarray:
    out = inp.copy()
    for spec in specs:
        bbox = spec["bbox"]
        marker = spec["marker"]
        marker_color = spec["marker_color"]
        edge = spec["edge"]
        out[marker] = BG
        _paint_inner_line(out, bbox, edge, marker_color)
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    frame_count = rng.randint(*FRAME_COUNT_RANGE)

    frame_colors = rng.sample(COLORS, frame_count)
    marker_pool = [color for color in COLORS if color not in frame_colors]
    marker_colors = rng.sample(marker_pool, frame_count)
    branches = rng.sample(BRANCH_TYPES, frame_count)

    inp = np.full((h, w), BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    specs: list[dict[str, object]] = []

    order = list(range(frame_count))
    rng.shuffle(order)
    for idx in order:
        branch = branches[idx]
        geometry = _sample_geometry(rng, branch)
        if geometry is None:
            return None
        interior_h, interior_w, marker_r_local, marker_c_local = geometry
        frame_h = interior_h + 2
        frame_w = interior_w + 2

        positions = _find_positions(occupied, frame_h, frame_w)
        if not positions:
            return None
        r0, c0 = rng.choice(positions)
        r1 = r0 + frame_h - 1
        c1 = c0 + frame_w - 1
        bbox = (r0, c0, r1, c1)
        marker = (r0 + 1 + marker_r_local, c0 + 1 + marker_c_local)
        edge = _choose_edge_for_marker(bbox, marker)

        _draw_frame(inp, bbox, frame_colors[idx])
        inp[marker] = marker_colors[idx]

        rr0 = max(0, r0 - PLACEMENT_PAD)
        cc0 = max(0, c0 - PLACEMENT_PAD)
        rr1 = min(h, r1 + PLACEMENT_PAD + 1)
        cc1 = min(w, c1 + PLACEMENT_PAD + 1)
        occupied[rr0:rr1, cc0:cc1] = True
        specs.append(
            {
                "bbox": bbox,
                "frame_color": frame_colors[idx],
                "marker_color": marker_colors[idx],
                "marker": marker,
                "edge": edge,
                "branch": branch,
            }
        )

    if np.count_nonzero(inp == BG) <= np.count_nonzero(inp != BG):
        return None

    expected = _build_expected(inp, specs)
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GENERATION_ATTEMPTS} attempts")
