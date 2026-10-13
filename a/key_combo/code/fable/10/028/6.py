import random
from collections import Counter, defaultdict, deque

import numpy as np


PALETTE = list(range(10))
GEN_ATTEMPTS = 600


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    comps: list[dict] = []
    h, w = grid.shape
    for color in sorted(set(map(int, grid.ravel())) - {int(bg)}):
        mask = grid == color
        seen = np.zeros_like(mask, dtype=bool)
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
                    for nr, nc in neighbors4(rr, cc):
                        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            q.append((nr, nc))
                coords = np.array(pts, dtype=int)
                comps.append(
                    {
                        "color": int(color),
                        "coords": coords,
                        "bbox": _bbox(coords),
                        "area": int(len(pts)),
                    }
                )
    return comps


def _band_row_ranges(marker_rows: list[int], height: int) -> list[tuple[int, int]]:
    bands = []
    for i, row in enumerate(marker_rows):
        start = 0 if i == 0 else (marker_rows[i - 1] + row) // 2 + 1
        end = height - 1 if i == len(marker_rows) - 1 else (row + marker_rows[i + 1]) // 2
        bands.append((int(start), int(end)))
    return bands


def _assign_band(marker_rows: list[int], bbox: tuple[int, int, int, int]) -> int:
    center2 = int(bbox[0] + bbox[2])
    for i in range(len(marker_rows) - 1):
        boundary2 = int(marker_rows[i] + marker_rows[i + 1])
        assert center2 != boundary2, "object bbox center cannot lie exactly on a band midpoint"
    for i in range(len(marker_rows)):
        lower = -10**9 if i == 0 else int(marker_rows[i - 1] + marker_rows[i])
        upper = 10**9 if i == len(marker_rows) - 1 else int(marker_rows[i] + marker_rows[i + 1])
        if lower < center2 < upper:
            return i
    raise AssertionError("object center must belong to exactly one marker band")


def _object_side(coords: np.ndarray, marker_row: int) -> str:
    top = int(coords[:, 0].min())
    bottom = int(coords[:, 0].max())
    assert bottom < marker_row or top > marker_row, "objects may not cross their marker row"
    return "above" if bottom < marker_row else "below"


def _distance_to_marker(coords: np.ndarray, marker_row: int, side: str) -> int:
    if side == "above":
        return int(marker_row - int(coords[:, 0].max()) - 1)
    return int(int(coords[:, 0].min()) - marker_row - 1)


def _settle_coords(coords: np.ndarray, marker_row: int, occupied: np.ndarray, step: int) -> np.ndarray:
    cur = coords.copy()
    while True:
        nxt = cur.copy()
        nxt[:, 0] += int(step)
        if np.any(nxt[:, 0] == int(marker_row)):
            return cur
        if np.any(occupied[nxt[:, 0], nxt[:, 1]]):
            return cur
        cur = nxt


def _solve_with_info(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = get_bg(g)
    comps = _find_components(g, bg)
    markers = [comp for comp in comps if comp["area"] == 1]
    objects = [comp for comp in comps if comp["area"] >= 2]
    assert markers and objects, "expected singleton markers and larger foreground objects"

    marker_rows = [int(comp["coords"][0, 0]) for comp in markers]
    assert len(marker_rows) == len(set(marker_rows)), "marker rows must be unique"
    marker_colors = {int(comp["color"]) for comp in markers}
    object_colors = {int(comp["color"]) for comp in objects}
    assert marker_colors.isdisjoint(object_colors), "marker and object colors must be disjoint"

    markers_sorted = sorted(
        (
            {
                "row": int(comp["coords"][0, 0]),
                "col": int(comp["coords"][0, 1]),
                "color": int(comp["color"]),
            }
            for comp in markers
        ),
        key=lambda item: item["row"],
    )
    marker_rows = [item["row"] for item in markers_sorted]
    bands = _band_row_ranges(marker_rows, h)

    assigned: dict[int, list[dict]] = defaultdict(list)
    for comp in objects:
        band_idx = _assign_band(marker_rows, comp["bbox"])
        marker = markers_sorted[band_idx]
        side = _object_side(comp["coords"], marker["row"])
        dist = _distance_to_marker(comp["coords"], marker["row"], side)
        assert dist >= 0
        assigned[band_idx].append(
            {
                "color": int(comp["color"]),
                "coords": comp["coords"].copy(),
                "bbox": comp["bbox"],
                "band": int(band_idx),
                "marker_row": int(marker["row"]),
                "marker_color": int(marker["color"]),
                "marker_col": int(marker["col"]),
                "band_rows": bands[band_idx],
                "side": side,
                "input_gap": int(dist),
            }
        )

    out = np.full_like(g, bg)
    for marker in markers_sorted:
        out[marker["row"], marker["col"]] = marker["color"]

    global_occ = out != bg
    settled_info: list[dict] = []
    for band_idx in range(len(markers_sorted)):
        band_occ = np.zeros_like(g, dtype=bool)
        entries = assigned.get(band_idx, [])
        same_side_gaps: dict[str, list[int]] = defaultdict(list)
        for item in entries:
            same_side_gaps[item["side"]].append(item["input_gap"])
        for gaps in same_side_gaps.values():
            assert len(gaps) == len(set(gaps)), "same-side objects must have unique distances to the marker row"

        entries.sort(key=lambda item: (item["input_gap"], item["bbox"][0], item["bbox"][1], item["color"]))
        for item in entries:
            step = 1 if item["side"] == "above" else -1
            settled = _settle_coords(item["coords"], item["marker_row"], band_occ, step)
            final_gap = _distance_to_marker(settled, item["marker_row"], item["side"])
            assert final_gap >= 0
            assert not np.any(global_occ[settled[:, 0], settled[:, 1]]), "settled objects may not overlap across bands"
            out[settled[:, 0], settled[:, 1]] = item["marker_color"]
            global_occ[settled[:, 0], settled[:, 1]] = True
            band_occ[settled[:, 0], settled[:, 1]] = True
            item = dict(item)
            item["settled_coords"] = settled
            item["settled_bbox"] = _bbox(settled)
            item["final_gap"] = int(final_gap)
            item["blocked"] = bool(final_gap > 0)
            settled_info.append(item)

    return out, {"bg": int(bg), "markers": markers_sorted, "bands": bands, "objects": settled_info}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pull each foreground object vertically onto its assigned marker row and recolor it to that marker's color.

    Input structure:
    - The background is the most common color.
    - Foreground pixels are split into isolated one-pixel markers on distinct rows and larger 4-connected monochrome objects.
    - Marker colors and object colors are disjoint, and each larger object lies entirely above or below its eventual marker row.

    Rule:
    - Sort markers by row and split the canvas into horizontal gravity bands at the midpoints between consecutive marker rows.
    - Assign every larger object to the band containing the center of its bounding box.
    - Within each band, objects above the marker row slide downward and objects below it slide upward; process same-side objects from nearest to farthest from the marker row, stopping each one just before the marker row or just before a previously settled object in the same band.
    - Recolor every settled object to its assigned marker color while leaving the original marker pixels fixed in place.

    Key insight:
    - The markers define bidirectional attraction lines by row band, so the deciding clue is not an object's own color or nearest edge, but which marker row owns the object's bounding-box center.
    """
    return _solve_with_info(grid)[0]


def _mask_to_coords(mask: np.ndarray, top: int, left: int) -> np.ndarray:
    coords = np.argwhere(mask)
    coords[:, 0] += int(top)
    coords[:, 1] += int(left)
    return coords.astype(int)


def _random_connected_mask(max_h: int, max_w: int) -> np.ndarray:
    h = random.randint(1, max_h)
    w = random.randint(1, max_w)
    min_area = 2
    max_area = min(h * w, 6)
    if max_area < min_area:
        h = max(h, 2)
        w = max(w, 2 if h == 1 else 1)
        max_area = min(h * w, 6)
    area = random.randint(min_area, max_area)
    if random.random() < 0.4:
        if h * w < area:
            area = h * w
        mask = np.zeros((h, w), dtype=bool)
        if area == h * w:
            mask[:, :] = True
            return mask
    cells = {(random.randrange(h), random.randrange(w))}
    while len(cells) < area:
        rr, cc = random.choice(tuple(cells))
        options = []
        for nr, nc in neighbors4(rr, cc):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                options.append((nr, nc))
        if not options:
            continue
        cells.add(random.choice(options))
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    mask = np.zeros((max(rows) - min(rows) + 1, max(cols) - min(cols) + 1), dtype=bool)
    for r, c in cells:
        mask[r - min(rows), c - min(cols)] = True
    return mask


def _rect_mask(h: int, w: int) -> np.ndarray:
    mask = np.ones((h, w), dtype=bool)
    assert int(mask.sum()) >= 2
    return mask


def _reserve_ok(coords: np.ndarray, occupied: np.ndarray) -> bool:
    h, w = occupied.shape
    for r, c in coords:
        if not (0 <= int(r) < h and 0 <= int(c) < w):
            return False
        if occupied[int(r), int(c)]:
            return False
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and occupied[nr, nc] and (nr, nc) not in {tuple(x) for x in coords}:
                    return False
    return True


def _render_plan(shape: tuple[int, int], bg: int, markers: list[dict], objects: list[dict]) -> np.ndarray:
    grid = np.full(shape, int(bg), dtype=int)
    for marker in markers:
        grid[int(marker["row"]), int(marker["col"])] = int(marker["color"])
    for obj in objects:
        coords = obj["coords"]
        grid[coords[:, 0], coords[:, 1]] = int(obj["color"])
    return grid


def _planned_band_ok(markers: list[dict], bbox: tuple[int, int, int, int], band_idx: int) -> bool:
    rows = [int(marker["row"]) for marker in markers]
    try:
        return _assign_band(rows, bbox) == int(band_idx)
    except AssertionError:
        return False


def _simulate_plan(shape: tuple[int, int], bg: int, markers: list[dict], objects: list[dict]) -> tuple[np.ndarray, dict]:
    out = np.full(shape, int(bg), dtype=int)
    for marker in markers:
        out[int(marker["row"]), int(marker["col"])] = int(marker["color"])
    global_occ = out != bg
    settled: list[dict] = []
    grouped: dict[int, list[dict]] = defaultdict(list)
    for obj in objects:
        grouped[int(obj["band"])].append(dict(obj))
    for band in sorted(grouped):
        band_occ = np.zeros(shape, dtype=bool)
        entries = grouped[band]
        entries.sort(key=lambda item: (item["input_gap"], item["bbox"][0], item["bbox"][1], item["color"]))
        for item in entries:
            step = 1 if item["side"] == "above" else -1
            final_coords = _settle_coords(item["coords"], item["marker_row"], band_occ, step)
            assert not np.any(global_occ[final_coords[:, 0], final_coords[:, 1]])
            out[final_coords[:, 0], final_coords[:, 1]] = int(item["marker_color"])
            band_occ[final_coords[:, 0], final_coords[:, 1]] = True
            global_occ[final_coords[:, 0], final_coords[:, 1]] = True
            item["settled_coords"] = final_coords
            item["final_gap"] = _distance_to_marker(final_coords, item["marker_row"], item["side"])
            item["blocked"] = bool(item["final_gap"] > 0)
            settled.append(item)
    return out, {"objects": settled}


def _place_object(
    occupied: np.ndarray,
    band_rows: tuple[int, int],
    marker_row: int,
    side: str,
    mask: np.ndarray,
    gap: int,
    color: int,
    band_idx: int,
    marker_color: int,
    left_options: list[int] | None = None,
    role: str = "extra",
) -> dict | None:
    h, w = mask.shape
    start, end = band_rows
    if side == "above":
        top = int(marker_row - h - gap)
        if top < start:
            return None
    else:
        top = int(marker_row + gap + 1)
        if top + h - 1 > end:
            return None
    if left_options is None:
        left_options = list(range(1, occupied.shape[1] - w))
    random.shuffle(left_options)
    for left in left_options:
        coords = _mask_to_coords(mask, top, left)
        if _reserve_ok(coords, occupied):
            bbox = _bbox(coords)
            return {
                "color": int(color),
                "coords": coords,
                "bbox": bbox,
                "band": int(band_idx),
                "marker_row": int(marker_row),
                "marker_color": int(marker_color),
                "side": side,
                "input_gap": int(gap),
                "role": role,
            }
    return None


def _gen_band_layout(n_bands: int) -> tuple[list[int], list[tuple[int, int]]]:
    for _ in range(200):
        heights = [random.randint(6, 8) for _ in range(n_bands)]
        if sum(heights) <= 30:
            starts = []
            cur = 0
            for h in heights:
                starts.append((cur, cur + h - 1))
                cur += h
            return heights, starts
    raise RuntimeError("failed to choose band heights")


def _choose_markers(bands: list[tuple[int, int]], colors: list[int]) -> list[dict] | None:
    markers = []
    for (start, end), color in zip(bands, colors):
        height = end - start + 1
        offsets = list(range(2, height - 2))
        if not offsets:
            return None
        row = start + random.choice(offsets)
        markers.append({"row": int(row), "color": int(color)})
    return markers


def _gen_attempt() -> np.ndarray | None:
    n_bands = random.choice([3, 4])
    object_count = random.randint(4, 5)
    if n_bands + object_count > 9:
        return None
    heights, bands = _gen_band_layout(n_bands)
    width = random.randint(12, 20)
    bg = random.choice(PALETTE)
    colors = random.sample([c for c in PALETTE if c != bg], n_bands + object_count)
    marker_colors = colors[:n_bands]
    object_colors = colors[n_bands:]
    markers = _choose_markers(bands, marker_colors)
    if markers is None:
        return None

    collision_options = []
    for band_idx, ((start, end), marker) in enumerate(zip(bands, markers)):
        above_cap = marker["row"] - start
        below_cap = end - marker["row"]
        if above_cap >= 4:
            collision_options.append((band_idx, "above"))
        if below_cap >= 4:
            collision_options.append((band_idx, "below"))
    if not collision_options:
        return None
    collision_band, collision_side = random.choice(collision_options)
    opposite_side = "below" if collision_side == "above" else "above"

    shape = (sum(heights), width)
    occupied = np.zeros(shape, dtype=bool)
    for marker in markers:
        marker["col"] = random.randint(1, width - 2)
        occupied[marker["row"], marker["col"]] = True

    plans: list[dict] = []
    used_gaps: dict[tuple[int, str], set[int]] = defaultdict(set)

    # Collision pair: force a far object to stop on a previously settled nearer object.
    band_rows = bands[collision_band]
    marker = markers[collision_band]
    cap = marker["row"] - band_rows[0] if collision_side == "above" else band_rows[1] - marker["row"]
    pair_specs = []
    for h1 in (1, 2):
        for h2 in (1, 2):
            for d1 in (1, 2):
                d2_min = max(d1 + 2, h1 + d1 + 1)
                for d2 in range(d2_min, cap - h2 + 1):
                    pair_specs.append((h1, h2, d1, d2))
    if not pair_specs:
        return None
    h1, h2, d1, d2 = random.choice(pair_specs)
    w1 = random.randint(2, min(4, width - 3))
    w2 = random.randint(2, min(4, width - 3))
    mask1 = _rect_mask(h1, w1)
    mask2 = _rect_mask(h2, w2)

    pair_ok = False
    for _ in range(80):
        left1 = random.randint(1, width - w1 - 1)
        overlap_lo = max(1, left1 - w2 + 1)
        overlap_hi = min(width - w2 - 1, left1 + w1 - 1)
        if overlap_lo > overlap_hi:
            continue
        left2 = random.randint(overlap_lo, overlap_hi)
        near = _place_object(
            occupied,
            band_rows,
            marker["row"],
            collision_side,
            mask1,
            d1,
            object_colors[0],
            collision_band,
            marker["color"],
            left_options=[left1],
            role="collision_near",
        )
        if near is None:
            continue
        if not _planned_band_ok(markers, near["bbox"], collision_band):
            continue
        occupied[near["coords"][:, 0], near["coords"][:, 1]] = True
        far = _place_object(
            occupied,
            band_rows,
            marker["row"],
            collision_side,
            mask2,
            d2,
            object_colors[1],
            collision_band,
            marker["color"],
            left_options=[left2],
            role="collision_far",
        )
        if far is not None and _planned_band_ok(markers, far["bbox"], collision_band):
            occupied[far["coords"][:, 0], far["coords"][:, 1]] = True
            plans.extend([near, far])
            used_gaps[(collision_band, collision_side)].update({d1, d2})
            pair_ok = True
            break
        occupied[near["coords"][:, 0], near["coords"][:, 1]] = False
    if not pair_ok:
        return None

    # Guaranteed object on the opposite side to cover both movement directions.
    opposite_candidates = []
    for band_idx, (band_rows, marker) in enumerate(zip(bands, markers)):
        cap_side = marker["row"] - band_rows[0] if opposite_side == "above" else band_rows[1] - marker["row"]
        if cap_side >= 2:
            opposite_candidates.append(band_idx)
    if not opposite_candidates:
        return None
    band_idx = random.choice(opposite_candidates)
    band_rows = bands[band_idx]
    marker = markers[band_idx]
    mask = _random_connected_mask(3, 4)
    cap_side = marker["row"] - band_rows[0] if opposite_side == "above" else band_rows[1] - marker["row"]
    gaps = [g for g in range(1, cap_side - mask.shape[0] + 1) if g not in used_gaps[(band_idx, opposite_side)]]
    if not gaps:
        return None
    obj = _place_object(
        occupied,
        band_rows,
        marker["row"],
        opposite_side,
        mask,
        random.choice(gaps),
        object_colors[2],
        band_idx,
        marker["color"],
        role="opposite_side",
    )
    if obj is None:
        return None
    if not _planned_band_ok(markers, obj["bbox"], band_idx):
        return None
    occupied[obj["coords"][:, 0], obj["coords"][:, 1]] = True
    plans.append(obj)
    used_gaps[(band_idx, opposite_side)].add(obj["input_gap"])

    # Extra objects for variety and multi-band coverage.
    for color in object_colors[3:]:
        placed = None
        for _ in range(120):
            band_idx = random.randrange(n_bands)
            side = random.choice(["above", "below"])
            band_rows = bands[band_idx]
            marker = markers[band_idx]
            cap_side = marker["row"] - band_rows[0] if side == "above" else band_rows[1] - marker["row"]
            mask = _random_connected_mask(min(3, cap_side - 1 if cap_side >= 2 else 1), 4)
            if cap_side - mask.shape[0] < 1:
                continue
            gaps = [g for g in range(1, cap_side - mask.shape[0] + 1) if g not in used_gaps[(band_idx, side)]]
            if not gaps:
                continue
            placed = _place_object(
                occupied,
                band_rows,
                marker["row"],
                side,
                mask,
                random.choice(gaps),
                int(color),
                band_idx,
                marker["color"],
                role="extra",
            )
            if placed is not None and _planned_band_ok(markers, placed["bbox"], band_idx):
                occupied[placed["coords"][:, 0], placed["coords"][:, 1]] = True
                used_gaps[(band_idx, side)].add(placed["input_gap"])
                plans.append(placed)
                break
        if placed is None:
            return None

    inp = _render_plan(shape, bg, markers, plans)
    planned_out, planned_meta = _simulate_plan(shape, bg, markers, plans)
    solved_out, solved_meta = _solve_with_info(inp)
    if not np.array_equal(planned_out, solved_out):
        return None

    solved_by_color = {item["color"]: item for item in solved_meta["objects"]}
    if len(solved_by_color) != len(plans):
        return None
    roles = {item["color"]: item["role"] for item in plans}
    for item in plans:
        solved = solved_by_color[item["color"]]
        if solved["band"] != item["band"] or solved["side"] != item["side"]:
            return None
        if solved["marker_color"] != item["marker_color"]:
            return None

    if not any(item["side"] == "above" for item in solved_meta["objects"]):
        return None
    if not any(item["side"] == "below" for item in solved_meta["objects"]):
        return None
    if len({item["band"] for item in solved_meta["objects"]}) < 2:
        return None

    blocked = [item for item in solved_meta["objects"] if item["blocked"]]
    if len(blocked) < 1:
        return None
    if roles.get(blocked[0]["color"]) not in {"collision_far", "extra"}:
        return None
    if not any(roles.get(item["color"]) == "collision_far" and item["blocked"] for item in solved_meta["objects"]):
        return None
    if not any(roles.get(item["color"]) == "collision_near" and not item["blocked"] for item in solved_meta["objects"]):
        return None
    if np.array_equal(inp, solved_out):
        return None

    # Every object must move and the moved result must clear the source footprint.
    for item in solved_meta["objects"]:
        src = next(plan for plan in plans if plan["color"] == item["color"])
        if np.array_equal(src["coords"], item["settled_coords"]):
            return None
        if np.any(solved_out == item["color"]):
            return None

    # Validate markers remain unchanged in output.
    for marker in markers:
        if int(solved_out[marker["row"], marker["col"]]) != int(marker["color"]):
            return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gravity-band example")
