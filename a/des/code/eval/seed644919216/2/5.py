from __future__ import annotations

from collections import Counter

import numpy as np


MARKER_COLOR = 2
GRID_H_RANGE = (18, 24)
GRID_W_RANGE = (18, 24)
PAIR_COUNT_RANGE = (1, 3)
START_RAY_PROB = 0.75
MAX_GENERATE_ATTEMPTS = 400
SIDE_NAMES = ("up", "down", "left", "right")
SIDE_TO_DIR = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
PORTAL_COLOR_POOL = [1, 3, 4, 5, 6, 7, 8, 9]

SHAPE_POOL = (
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 0, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 0], [1, 1], [0, 1]], dtype=bool),
    np.array([[1, 1], [1, 1], [1, 0]], dtype=bool),
)


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _shape_key(coords: np.ndarray) -> tuple[tuple[int, int], tuple[tuple[int, int], ...]]:
    r0, c0, r1, c1 = _bbox(coords)
    rel = sorted((int(r - r0), int(c - c0)) for r, c in coords)
    return (int(r1 - r0 + 1), int(c1 - c0 + 1)), tuple(rel)


def _side_from_marker(
    bbox: tuple[int, int, int, int], marker: tuple[int, int]
) -> tuple[int, int] | None:
    r0, c0, r1, c1 = bbox
    mr, mc = marker
    if mr == r0 - 1 and c0 <= mc <= c1:
        return (-1, 0)
    if mr == r1 + 1 and c0 <= mc <= c1:
        return (1, 0)
    if mc == c0 - 1 and r0 <= mr <= r1:
        return (0, -1)
    if mc == c1 + 1 and r0 <= mr <= r1:
        return (0, 1)
    return None


def _find_portals(
    grid: np.ndarray, bg: int
) -> tuple[list[dict[str, object]], int, dict[int, int], dict[tuple[int, int], int], set[tuple[int, int]]]:
    unique_colors = [int(c) for c in np.unique(grid) if int(c) != int(bg)]
    singleton_by_cell: dict[tuple[int, int], int] = {}
    body_components: list[dict[str, object]] = []

    for color in unique_colors:
        for coords in _components(grid == int(color)):
            area = int(len(coords))
            box = _bbox(coords)
            if area == 1:
                singleton_by_cell[tuple(map(int, coords[0]))] = int(color)
                continue
            body_components.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": box,
                    "area": area,
                }
            )

    marker_votes: Counter[int] = Counter()
    raw_candidates: list[dict[str, object]] = []
    for comp in body_components:
        coords = np.asarray(comp["coords"], dtype=int)
        r0, c0, r1, c1 = comp["bbox"]
        area = int(comp["area"])
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        if area < 3 or h < 2 or w < 2 or h > 3 or w > 3:
            continue

        adj: dict[tuple[int, int], tuple[int, int]] = {}
        for r, c in coords:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                p = (int(r + dr), int(c + dc))
                if p not in singleton_by_cell:
                    continue
                side = _side_from_marker(comp["bbox"], p)
                if side is not None:
                    adj[p] = side

        if len(adj) != 1:
            continue
        marker_cell, direction = next(iter(adj.items()))
        marker_color = int(singleton_by_cell[marker_cell])
        marker_votes[marker_color] += 1
        raw_candidates.append(
            {
                "color": int(comp["color"]),
                "coords": coords,
                "bbox": comp["bbox"],
                "marker": marker_cell,
                "direction": direction,
                "shape_key": _shape_key(coords),
                "marker_color": marker_color,
            }
        )

    assert raw_candidates, "expected portal candidates"
    marker_color = max(marker_votes.items(), key=lambda item: (item[1], -item[0]))[0]
    portals = [cand for cand in raw_candidates if int(cand["marker_color"]) == int(marker_color)]
    assert portals, "expected at least one portal after marker-color inference"

    groups: dict[tuple[tuple[int, int], tuple[tuple[int, int], ...]], list[dict[str, object]]] = {}
    for portal in portals:
        groups.setdefault(portal["shape_key"], []).append(portal)

    paired_portals: list[dict[str, object]] = []
    pair_keys = [key for key, group in groups.items() if len(group) == 2]
    assert pair_keys, "expected at least one unambiguous portal pair"
    for key in sorted(pair_keys, key=str):
        group = groups[key]
        assert len({int(item["color"]) for item in group}) == 2, "portal pair colors must differ"
        paired_portals.extend(group)

    paired_portals.sort(key=lambda item: tuple(item["marker"]))
    for idx, portal in enumerate(paired_portals):
        portal["id"] = idx

    pair_of: dict[int, int] = {}
    marker_to_portal: dict[tuple[int, int], int] = {}
    by_shape: dict[tuple[tuple[int, int], tuple[tuple[int, int], ...]], list[int]] = {}
    body_cells: set[tuple[int, int]] = set()
    for portal in paired_portals:
        pid = int(portal["id"])
        marker_to_portal[tuple(portal["marker"])] = pid
        by_shape.setdefault(portal["shape_key"], []).append(pid)
        for cell in np.asarray(portal["coords"], dtype=int):
            body_cells.add(tuple(map(int, cell)))

    for ids in by_shape.values():
        assert len(ids) == 2, "every portal shape must pair exactly twice"
        a, b = ids
        pair_of[int(a)] = int(b)
        pair_of[int(b)] = int(a)

    return paired_portals, int(marker_color), pair_of, marker_to_portal, body_cells


def _find_start_ray(
    grid: np.ndarray,
    portal: dict[str, object],
    bg: int,
    marker_color: int,
    portal_body_cells: set[tuple[int, int]],
) -> dict[str, object] | None:
    dr, dc = portal["direction"]
    mr, mc = portal["marker"]
    nr, nc = int(mr + dr), int(mc + dc)
    if not _in_bounds(grid.shape, nr, nc):
        return None
    if (nr, nc) in portal_body_cells:
        return None

    ray_color = int(grid[nr, nc])
    if ray_color in (int(bg), int(marker_color)):
        return None

    cells: list[tuple[int, int]] = []
    rr, cc = nr, nc
    while _in_bounds(grid.shape, rr, cc) and int(grid[rr, cc]) == int(ray_color):
        if dr == 0:
            for odr in (-1, 1):
                nr2 = rr + odr
                if _in_bounds(grid.shape, nr2, cc) and int(grid[nr2, cc]) == int(ray_color):
                    return None
        else:
            for odc in (-1, 1):
                nc2 = cc + odc
                if _in_bounds(grid.shape, rr, nc2) and int(grid[rr, nc2]) == int(ray_color):
                    return None
        cells.append((int(rr), int(cc)))
        rr += int(dr)
        cc += int(dc)

    if not cells:
        return None
    return {
        "portal_id": int(portal["id"]),
        "color": int(ray_color),
        "direction": (int(dr), int(dc)),
    }


def _extend_ray(
    out: np.ndarray,
    start_portal_id: int,
    ray_color: int,
    direction: tuple[int, int],
    portals: list[dict[str, object]],
    pair_of: dict[int, int],
    marker_to_portal: dict[tuple[int, int], int],
    portal_body_cells: set[tuple[int, int]],
) -> None:
    dr, dc = direction
    current_portal = int(pair_of[start_portal_id])
    seen_states: set[tuple[int, int, int]] = set()

    while True:
        state = (int(current_portal), int(dr), int(dc))
        if state in seen_states:
            return
        seen_states.add(state)

        mr, mc = portals[current_portal]["marker"]
        rr, cc = int(mr + dr), int(mc + dc)
        while _in_bounds(out.shape, rr, cc):
            pos = (int(rr), int(cc))
            if pos in marker_to_portal:
                current_portal = int(pair_of[int(marker_to_portal[pos])])
                break
            if pos in portal_body_cells:
                return
            out[rr, cc] = int(ray_color)
            rr += int(dr)
            cc += int(dc)
        else:
            return


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find same-shape, different-color portal pairs, then extend each ray through
    paired portals while preserving its color and direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _background_color(g)
    portals, marker_color, pair_of, marker_to_portal, portal_body_cells = _find_portals(g, bg)

    out = g.copy()
    start_rays: list[dict[str, object]] = []
    for portal in portals:
        ray = _find_start_ray(g, portal, bg, marker_color, portal_body_cells)
        if ray is not None:
            start_rays.append(ray)

    assert start_rays, "expected at least one ray attached to a portal marker"
    start_rays.sort(key=lambda item: (int(item["portal_id"]), int(item["color"])))
    for ray in start_rays:
        _extend_ray(
            out,
            int(ray["portal_id"]),
            int(ray["color"]),
            tuple(ray["direction"]),
            portals,
            pair_of,
            marker_to_portal,
            portal_body_cells,
        )
    return out


def _anchors_for_side(mask: np.ndarray, side: str) -> list[int]:
    if side == "up":
        return [int(c) for c in np.flatnonzero(mask[0, :])]
    if side == "down":
        return [int(c) for c in np.flatnonzero(mask[-1, :])]
    if side == "left":
        return [int(r) for r in np.flatnonzero(mask[:, 0])]
    return [int(r) for r in np.flatnonzero(mask[:, -1])]


def _marker_to_body_top_left(
    marker: tuple[int, int], mask: np.ndarray, side: str, anchor: int
) -> tuple[int, int]:
    h, w = mask.shape
    mr, mc = marker
    if side == "up":
        return int(mr + 1), int(mc - anchor)
    if side == "down":
        return int(mr - h), int(mc - anchor)
    if side == "left":
        return int(mr - anchor), int(mc + 1)
    return int(mr - anchor), int(mc - w)


def _place_portal(
    grid: np.ndarray,
    reserved: np.ndarray,
    marker: tuple[int, int],
    mask: np.ndarray,
    color: int,
    side: str,
    anchor: int,
) -> bool:
    h, w = mask.shape
    r0, c0 = _marker_to_body_top_left(marker, mask, side, anchor)
    if r0 < 0 or c0 < 0 or r0 + h > grid.shape[0] or c0 + w > grid.shape[1]:
        return False

    body_cells: list[tuple[int, int]] = []
    for dr in range(h):
        for dc in range(w):
            if not mask[dr, dc]:
                continue
            body_cells.append((int(r0 + dr), int(c0 + dc)))

    all_cells = body_cells + [tuple(marker)]
    if any(reserved[r, c] for r, c in all_cells):
        return False

    min_r = max(0, min(r for r, _ in all_cells) - 1)
    max_r = min(grid.shape[0] - 1, max(r for r, _ in all_cells) + 1)
    min_c = max(0, min(c for _, c in all_cells) - 1)
    max_c = min(grid.shape[1] - 1, max(c for _, c in all_cells) + 1)
    if reserved[min_r : max_r + 1, min_c : max_c + 1].any():
        return False

    for r, c in body_cells:
        grid[r, c] = int(color)
    grid[marker] = MARKER_COLOR
    reserved[min_r : max_r + 1, min_c : max_c + 1] = True
    return True


def _draw_ray(
    grid: np.ndarray,
    marker: tuple[int, int],
    side: str,
    color: int,
    length: int,
) -> bool:
    dr, dc = SIDE_TO_DIR[side]
    cells: list[tuple[int, int]] = []
    rr, cc = int(marker[0] + dr), int(marker[1] + dc)
    for _ in range(length):
        if not _in_bounds(grid.shape, rr, cc) or int(grid[rr, cc]) != 0:
            return False
        cells.append((int(rr), int(cc)))
        rr += int(dr)
        cc += int(dc)
    for r, c in cells:
        grid[r, c] = int(color)
    return True


def _lane_candidates(size: int, need: int) -> list[int] | None:
    lanes = list(range(2, size - 2, 3))
    if len(lanes) < need:
        return None
    return lanes


def _coord_range(size: int, role: str, side: str) -> tuple[int, int] | None:
    mid = size // 2
    if side in ("right", "down"):
        low_role, high_role = "exit", "entry"
    else:
        low_role, high_role = "entry", "exit"

    if role == low_role:
        lo, hi = 4, mid - 2
    else:
        lo, hi = mid + 1, size - 5
    if lo > hi:
        return None
    return lo, hi


def _sample_coord(rng: np.random.Generator, size: int, role: str, side: str) -> int | None:
    bounds = _coord_range(size, role, side)
    if bounds is None:
        return None
    lo, hi = bounds
    return int(rng.integers(lo, hi + 1))


def _entry_ray_max_len(marker: tuple[int, int], side: str, shape: tuple[int, int]) -> int:
    if side == "up":
        return int(marker[0])
    if side == "down":
        return int(shape[0] - 1 - marker[0])
    if side == "left":
        return int(marker[1])
    return int(shape[1] - 1 - marker[1])


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(GRID_H_RANGE[0], GRID_H_RANGE[1] + 1))
    w = int(rng.integers(GRID_W_RANGE[0], GRID_W_RANGE[1] + 1))
    pair_count = int(rng.integers(PAIR_COUNT_RANGE[0], PAIR_COUNT_RANGE[1] + 1))
    side = str(rng.choice(SIDE_NAMES))

    want_chain = pair_count > 1 and bool(rng.integers(0, 2))
    chain_len = int(rng.integers(2, pair_count + 1)) if want_chain else 1

    pair_ids = list(range(pair_count))
    rng.shuffle(pair_ids)
    chain_pairs = pair_ids[:chain_len]
    isolated_pairs = pair_ids[chain_len:]

    shape_indices = rng.choice(len(SHAPE_POOL), size=pair_count, replace=False)
    colors = rng.choice(PORTAL_COLOR_POOL, size=pair_count * 2, replace=False)
    ray_palette = [color for color in PORTAL_COLOR_POOL if int(color) not in set(map(int, colors))]
    if not ray_palette:
        return None

    shapes: dict[int, np.ndarray] = {}
    anchors: dict[int, int] = {}
    portal_colors: dict[int, tuple[int, int]] = {}
    for pair_id, shape_idx in enumerate(shape_indices):
        mask = SHAPE_POOL[int(shape_idx)]
        choices = _anchors_for_side(mask, side)
        if not choices:
            return None
        shapes[pair_id] = mask
        anchors[pair_id] = int(rng.choice(choices))
        portal_colors[pair_id] = (int(colors[2 * pair_id]), int(colors[2 * pair_id + 1]))

    axis_size = h if side in ("up", "down") else w
    lane_size = w if side in ("up", "down") else h
    unique_lane_count = 2 * pair_count - chain_len + 1
    lanes = _lane_candidates(lane_size, unique_lane_count)
    if lanes is None:
        return None
    rng.shuffle(lanes)

    entry_lane: dict[int, int] = {}
    exit_lane: dict[int, int] = {}
    lane_iter = iter(lanes)
    for pair_id in range(pair_count):
        entry_lane[pair_id] = int(next(lane_iter))
    for i in range(chain_len - 1):
        exit_lane[chain_pairs[i]] = int(entry_lane[chain_pairs[i + 1]])
    exit_lane[chain_pairs[-1]] = int(next(lane_iter))
    for pair_id in isolated_pairs:
        exit_lane[pair_id] = int(next(lane_iter))

    markers: dict[tuple[int, str], tuple[int, int]] = {}
    for pair_id in range(pair_count):
        entry_coord = _sample_coord(rng, axis_size, "entry", side)
        exit_coord = _sample_coord(rng, axis_size, "exit", side)
        if entry_coord is None or exit_coord is None:
            return None
        if side in ("left", "right"):
            markers[(pair_id, "entry")] = (int(entry_lane[pair_id]), int(entry_coord))
            markers[(pair_id, "exit")] = (int(exit_lane[pair_id]), int(exit_coord))
        else:
            markers[(pair_id, "entry")] = (int(entry_coord), int(entry_lane[pair_id]))
            markers[(pair_id, "exit")] = (int(exit_coord), int(exit_lane[pair_id]))

    grid = np.zeros((h, w), dtype=int)
    reserved = np.zeros((h, w), dtype=bool)
    placement_order = [(pair_id, "entry") for pair_id in range(pair_count)] + [
        (pair_id, "exit") for pair_id in range(pair_count)
    ]
    for pair_id, role in placement_order:
        color = portal_colors[pair_id][0 if role == "entry" else 1]
        ok = _place_portal(
            grid,
            reserved,
            markers[(pair_id, role)],
            shapes[pair_id],
            int(color),
            side,
            anchors[pair_id],
        )
        if not ok:
            return None

    start_flags = {pair_id: rng.random() < START_RAY_PROB for pair_id in range(pair_count)}
    if not any(start_flags.values()):
        start_flags[chain_pairs[0]] = True

    for pair_id in range(pair_count):
        if not start_flags[pair_id]:
            continue
        marker = markers[(pair_id, "entry")]
        max_len = _entry_ray_max_len(marker, side, grid.shape)
        max_len = min(max_len, 4)
        if max_len < 1:
            return None
        ray_len = int(rng.integers(1, max_len + 1))
        ray_color = int(rng.choice(ray_palette))
        if not _draw_ray(grid, marker, side, ray_color, ray_len):
            return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATE_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid portal-ray puzzle sample")
