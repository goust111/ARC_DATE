import random
from collections import deque

import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]
PALETTE = tuple(range(10))
GEN_RNG = np.random.default_rng()


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    dirs = DIRS4 if int(connectivity) == 4 else DIRS8
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in dirs:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _parse_scene(grid: np.ndarray) -> tuple[int, list[dict], list[dict], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)

    markers: list[dict] = []
    for r in range(h):
        for c in range(w):
            if g[r, c] == bg:
                continue
            on_top = r == 0
            on_bottom = r == h - 1
            on_left = c == 0
            on_right = c == w - 1
            edge_count = sum((on_top, on_bottom, on_left, on_right))
            if edge_count != 1:
                continue
            if on_top:
                dr, dc = 1, 0
            elif on_bottom:
                dr, dc = -1, 0
            elif on_left:
                dr, dc = 0, 1
            else:
                dr, dc = 0, -1
            markers.append({"pos": (r, c), "dir": (dr, dc), "color": int(g[r, c])})

    inner_mask = np.zeros_like(g, dtype=bool)
    if h > 2 and w > 2:
        inner_mask[1:-1, 1:-1] = g[1:-1, 1:-1] != bg

    id_grid = np.full(g.shape, -1, dtype=int)
    objects: list[dict] = []
    for idx, coords in enumerate(_components_mask(inner_mask, connectivity=4)):
        colors = np.unique(g[coords[:, 0], coords[:, 1]])
        assert colors.size == 1, "each interior object must be monochromatic"
        obj = {
            "id": idx,
            "coords": coords,
            "color": int(colors[0]),
        }
        objects.append(obj)
        id_grid[coords[:, 0], coords[:, 1]] = idx

    return bg, markers, objects, id_grid


def _line_hits(id_grid: np.ndarray, marker: dict) -> list[tuple[int, int]]:
    h, w = id_grid.shape
    r, c = marker["pos"]
    dr, dc = marker["dir"]
    rr, cc = r + dr, c + dc
    dist = 1
    seen_ids: set[int] = set()
    hits: list[tuple[int, int]] = []
    while 0 <= rr < h and 0 <= cc < w:
        obj_id = int(id_grid[rr, cc])
        if obj_id >= 0 and obj_id not in seen_ids:
            seen_ids.add(obj_id)
            hits.append((obj_id, dist))
        rr += dr
        cc += dc
        dist += 1
    return hits


def _compute_claims(markers: list[dict], objects: list[dict], id_grid: np.ndarray) -> tuple[dict[int, list[dict]], dict[int, dict], list[list[tuple[int, int]]]]:
    claims = {obj["id"]: [] for obj in objects}
    line_sequences: list[list[tuple[int, int]]] = []
    for marker in markers:
        hits = _line_hits(id_grid, marker)
        line_sequences.append(hits)
        if not hits:
            continue
        obj_id, dist = hits[0]
        claims[obj_id].append({"marker": marker, "distance": int(dist)})

    winners: dict[int, dict] = {}
    for obj in objects:
        obj_claims = claims[obj["id"]]
        if not obj_claims:
            continue
        dists = [claim["distance"] for claim in obj_claims]
        best = min(dists)
        assert dists.count(best) == 1, "nearest marker must be unique on valid inputs"
        winner = min(obj_claims, key=lambda claim: claim["distance"])
        winners[obj["id"]] = winner
    return claims, winners, line_sequences


def main(grid: np.ndarray) -> np.ndarray:
    """Let boundary markers recolor the first visible object on their ray and erase unseen objects.

    Input structure:
    - The background is the dominant color.
    - The outer border contains several single-pixel markers, each on exactly one edge.
    - Inside the border are separated 4-connected monochrome objects whose colors do not encode ownership.

    Rule:
    - Each marker looks inward along its own row or column, perpendicular to the edge it sits on.
    - The first object touched by that ray is claimed; the entire object is recolored to the marker color, and the ray stops there.
    - If multiple markers claim the same object, the one with the smaller hit distance wins.
    - Objects claimed by no marker are removed, while the markers themselves stay unchanged.

    Key insight:
    - Visibility, not original color, decides each object's fate: front objects block rays, contested objects compare claim distances, and anything never seen disappears.
    """
    bg, markers, objects, id_grid = _parse_scene(grid)
    claims, winners, _line_sequences = _compute_claims(markers, objects, id_grid)

    out = np.full_like(np.asarray(grid, dtype=int), bg)
    for marker in markers:
        r, c = marker["pos"]
        out[r, c] = marker["color"]

    for obj in objects:
        winner = winners.get(obj["id"])
        if winner is None:
            continue
        color = int(winner["marker"]["color"])
        coords = obj["coords"]
        out[coords[:, 0], coords[:, 1]] = color

    return out


def _random_blob(rng: np.random.Generator, min_h: int = 2, max_h: int = 5, min_w: int = 2, max_w: int = 5, min_area: int = 3, max_area: int = 8) -> np.ndarray:
    h = int(rng.integers(min_h, max_h + 1))
    w = int(rng.integers(min_w, max_w + 1))
    area_cap = min(int(max_area), h * w)
    area = int(rng.integers(min(int(min_area), area_cap), area_cap + 1))

    mask = np.zeros((h, w), dtype=bool)
    seed = (int(rng.integers(0, h)), int(rng.integers(0, w)))
    mask[seed] = True
    while int(mask.sum()) < area:
        frontier: list[tuple[int, int]] = []
        seen = set()
        for r, c in np.argwhere(mask):
            for dr, dc in DIRS4:
                nr, nc = int(r + dr), int(c + dc)
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    frontier.append((nr, nc))
        if not frontier:
            break
        nr, nc = frontier[int(rng.integers(0, len(frontier)))]
        mask[nr, nc] = True
    return _crop_mask(mask)


def _coords_set(mask: np.ndarray, top: int, left: int) -> set[tuple[int, int]]:
    return {(top + int(r), left + int(c)) for r, c in np.argwhere(mask)}


def _can_place(coords: set[tuple[int, int]], occupied: set[tuple[int, int]], h: int, w: int) -> bool:
    for r, c in coords:
        if not (1 <= r < h - 1 and 1 <= c < w - 1):
            return False
        for dr, dc in DIRS8 + [(0, 0)]:
            if (r + dr, c + dc) in occupied:
                return False
    return True


def _place_in_region(
    rng: np.random.Generator,
    occupied: set[tuple[int, int]],
    h: int,
    w: int,
    row_range: tuple[int, int],
    col_range: tuple[int, int],
    extra_check=None,
    attempts: int = 80,
) -> dict | None:
    row_lo, row_hi = row_range
    col_lo, col_hi = col_range
    for _ in range(attempts):
        mask = _random_blob(rng)
        mh, mw = mask.shape
        max_top = min(int(row_hi), h - 1 - mh)
        max_left = min(int(col_hi), w - 1 - mw)
        min_top = max(1, int(row_lo))
        min_left = max(1, int(col_lo))
        if min_top > max_top or min_left > max_left:
            continue
        top = int(rng.integers(min_top, max_top + 1))
        left = int(rng.integers(min_left, max_left + 1))
        coords = _coords_set(mask, top, left)
        if not _can_place(coords, occupied, h, w):
            continue
        if extra_check is not None and not extra_check(coords):
            continue
        occupied.update(coords)
        return {"coords_set": coords, "mask": mask, "top": top, "left": left}
    return None


def _object_row_cols(obj: dict) -> dict[int, list[int]]:
    rows: dict[int, list[int]] = {}
    for r, c in obj["coords_set"]:
        rows.setdefault(int(r), []).append(int(c))
    for cols in rows.values():
        cols.sort()
    return rows


def _object_col_rows(obj: dict) -> dict[int, list[int]]:
    cols: dict[int, list[int]] = {}
    for r, c in obj["coords_set"]:
        cols.setdefault(int(c), []).append(int(r))
    for rows in cols.values():
        rows.sort()
    return cols


def _build_id_grid(shape: tuple[int, int], objects: list[dict]) -> np.ndarray:
    id_grid = np.full(shape, -1, dtype=int)
    for obj in objects:
        for r, c in obj["coords_set"]:
            id_grid[int(r), int(c)] = int(obj["id"])
    return id_grid


def _choose_marker_set(shape: tuple[int, int], objects: list[dict], contested_id: int, front_id: int, back_id: int, solo_id: int, rng: np.random.Generator) -> list[dict] | None:
    h, w = shape
    id_grid = _build_id_grid(shape, objects)
    contested = objects[contested_id]
    front = objects[front_id]
    back = objects[back_id]
    solo = objects[solo_id]

    contested_rows = _object_row_cols(contested)
    contested_cols = _object_col_rows(contested)
    left_options: list[tuple[int, int]] = []
    top_options: list[tuple[int, int]] = []
    for row, cols in contested_rows.items():
        marker = {"pos": (int(row), 0), "dir": (0, 1), "color": -1}
        hits = _line_hits(id_grid, marker)
        if hits and hits[0][0] == contested_id:
            left_options.append((int(row), int(hits[0][1])))
    for col, rows in contested_cols.items():
        marker = {"pos": (0, int(col)), "dir": (1, 0), "color": -1}
        hits = _line_hits(id_grid, marker)
        if hits and hits[0][0] == contested_id:
            top_options.append((int(col), int(hits[0][1])))

    contested_pairs = [
        (row, col, left_dist, top_dist)
        for row, left_dist in left_options
        for col, top_dist in top_options
        if left_dist != top_dist
    ]
    if not contested_pairs:
        return None
    rng.shuffle(contested_pairs)
    contested_row, contested_col, _left_dist, _top_dist = contested_pairs[0]

    common_rows = sorted(set(_object_row_cols(front)) & set(_object_row_cols(back)))
    occlusion_rows = []
    for row in common_rows:
        if int(row) == contested_row:
            continue
        marker = {"pos": (int(row), 0), "dir": (0, 1), "color": -1}
        hits = _line_hits(id_grid, marker)
        if len(hits) >= 2 and hits[0][0] == front_id and hits[1][0] == back_id:
            occlusion_rows.append(int(row))
    if not occlusion_rows:
        return None
    occlusion_row = int(occlusion_rows[int(rng.integers(0, len(occlusion_rows)))])

    solo_markers: list[dict] = []
    for col in _object_col_rows(solo):
        marker = {"pos": (h - 1, int(col)), "dir": (-1, 0), "color": -1}
        hits = _line_hits(id_grid, marker)
        if hits and hits[0][0] == solo_id:
            solo_markers.append(marker)
    for row in _object_row_cols(solo):
        marker = {"pos": (int(row), w - 1), "dir": (0, -1), "color": -1}
        hits = _line_hits(id_grid, marker)
        if hits and hits[0][0] == solo_id:
            solo_markers.append(marker)
    if not solo_markers:
        return None
    solo_marker = solo_markers[int(rng.integers(0, len(solo_markers)))]

    markers = [
        {"pos": (contested_row, 0), "dir": (0, 1), "color": -1},
        {"pos": (0, contested_col), "dir": (1, 0), "color": -1},
        {"pos": (occlusion_row, 0), "dir": (0, 1), "color": -1},
        solo_marker,
    ]
    positions = [marker["pos"] for marker in markers]
    if len(set(positions)) != len(positions):
        return None
    return markers


def _place_unseen_object(
    rng: np.random.Generator,
    occupied: set[tuple[int, int]],
    h: int,
    w: int,
    marker_rows: set[int],
    marker_cols: set[int],
) -> dict | None:
    def extra_check(coords: set[tuple[int, int]]) -> bool:
        rows = {r for r, _c in coords}
        cols = {c for _r, c in coords}
        return rows.isdisjoint(marker_rows) and cols.isdisjoint(marker_cols)

    return _place_in_region(
        rng,
        occupied,
        h,
        w,
        row_range=(max(2, h // 3), h - 4),
        col_range=(max(2, w // 3), w - 4),
        extra_check=extra_check,
        attempts=120,
    )


def _role_scene_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(16, 25))
    w = int(rng.integers(16, 25))
    bg = int(rng.integers(0, 10))
    occupied: set[tuple[int, int]] = set()
    objects: list[dict] = []

    contested = _place_in_region(
        rng,
        occupied,
        h,
        w,
        row_range=(2, max(3, h // 2)),
        col_range=(3, max(4, w // 2)),
    )
    if contested is None:
        return None
    contested["id"] = 0
    objects.append(contested)

    front = _place_in_region(
        rng,
        occupied,
        h,
        w,
        row_range=(max(3, h // 2 - 2), h - 6),
        col_range=(2, max(4, w // 3)),
    )
    if front is None:
        return None
    front["id"] = 1
    objects.append(front)

    front_rows = [r for r, _c in front["coords_set"]]
    overlap_lo = max(2, min(front_rows) - 1)
    overlap_hi = min(h - 4, max(front_rows))
    back = _place_in_region(
        rng,
        occupied,
        h,
        w,
        row_range=(overlap_lo, overlap_hi),
        col_range=(min(w - 4, front["left"] + front["mask"].shape[1] + 2), w - 4),
    )
    if back is None:
        return None
    back["id"] = 2
    if not (set(_object_row_cols(front)) & set(_object_row_cols(back))):
        return None
    objects.append(back)

    solo = _place_in_region(
        rng,
        occupied,
        h,
        w,
        row_range=(2, h - 5),
        col_range=(max(2, w // 2), w - 5),
    )
    if solo is None:
        return None
    solo["id"] = 3
    objects.append(solo)

    markers = _choose_marker_set((h, w), objects, 0, 1, 2, 3, rng)
    if markers is None:
        return None

    marker_rows = {marker["pos"][0] for marker in markers if marker["dir"][1] != 0}
    marker_cols = {marker["pos"][1] for marker in markers if marker["dir"][0] != 0}
    unseen = _place_unseen_object(rng, occupied, h, w, marker_rows, marker_cols)
    if unseen is None:
        return None
    unseen["id"] = 4
    objects.append(unseen)

    id_grid = _build_id_grid((h, w), objects)
    temp_markers = [{"pos": m["pos"], "dir": m["dir"], "color": 1} for m in markers]
    try:
        claims, _winners, _line_sequences = _compute_claims(temp_markers, objects, id_grid)
    except AssertionError:
        return None

    if len(claims[0]) < 2:
        return None
    if claims[2]:
        return None
    if claims[4]:
        return None

    marker_colors = list(rng.choice([c for c in PALETTE if c != bg], size=len(markers), replace=False))
    for marker, color in zip(markers, marker_colors):
        marker["color"] = int(color)

    try:
        _claims, winners, _line_sequences = _compute_claims(markers, objects, id_grid)
    except AssertionError:
        return None

    winner_colors = {
        obj_id: int(winners[obj_id]["marker"]["color"])
        for obj_id in winners
    }
    object_colors: dict[int, int] = {}
    for obj in objects:
        banned = {bg}
        if obj["id"] in winner_colors:
            banned.add(int(winner_colors[obj["id"]]))
        choices = [c for c in PALETTE if c not in banned]
        object_colors[obj["id"]] = int(rng.choice(choices))
        obj["color"] = object_colors[obj["id"]]

    grid = np.full((h, w), bg, dtype=int)
    for obj in objects:
        for r, c in obj["coords_set"]:
            grid[int(r), int(c)] = int(obj["color"])
    for marker in markers:
        r, c = marker["pos"]
        grid[int(r), int(c)] = int(marker["color"])

    rotations = int(rng.integers(0, 4))
    if rotations:
        grid = np.rot90(grid, k=rotations)
    if bool(rng.integers(0, 2)):
        grid = np.fliplr(grid)
    return grid.copy()


def _validate_generated_scene(grid: np.ndarray) -> bool:
    bg, markers, objects, id_grid = _parse_scene(grid)
    if len(markers) < 4 or len(objects) < 5:
        return False
    marker_colors = [marker["color"] for marker in markers]
    if len(set(marker_colors)) != len(marker_colors):
        return False

    try:
        claims, winners, line_sequences = _compute_claims(markers, objects, id_grid)
    except AssertionError:
        return False
    claimed_ids = {obj_id for obj_id, winner in winners.items() if winner is not None}
    unclaimed_ids = {obj["id"] for obj in objects if obj["id"] not in claimed_ids}
    if not claimed_ids or not unclaimed_ids:
        return False

    multi_claim_ok = False
    for obj_id, obj_claims in claims.items():
        if not obj_claims:
            continue
        dists = [claim["distance"] for claim in obj_claims]
        best = min(dists)
        if dists.count(best) != 1:
            return False
        if len(obj_claims) >= 2:
            multi_claim_ok = True
    if not multi_claim_ok:
        return False

    occluded_unclaimed = False
    for hits in line_sequences:
        if len(hits) < 2:
            continue
        for obj_id, _dist in hits[1:]:
            if obj_id in unclaimed_ids:
                occluded_unclaimed = True
                break
        if occluded_unclaimed:
            break
    if not occluded_unclaimed:
        return False

    marker_rows = {marker["pos"][0] for marker in markers if marker["dir"][1] != 0}
    marker_cols = {marker["pos"][1] for marker in markers if marker["dir"][0] != 0}
    isolated_unclaimed = False
    for obj in objects:
        if obj["id"] not in unclaimed_ids:
            continue
        rows = {int(r) for r, _c in obj["coords"]}
        cols = {int(c) for _r, c in obj["coords"]}
        if rows.isdisjoint(marker_rows) and cols.isdisjoint(marker_cols):
            isolated_unclaimed = True
            break
    if not isolated_unclaimed:
        return False

    try:
        out = main(grid)
    except AssertionError:
        return False
    if out.shape != grid.shape:
        return False
    if np.array_equal(out, grid):
        return False
    if any(np.any(out[obj["coords"][:, 0], obj["coords"][:, 1]] == bg) for obj in objects if obj["id"] in claimed_ids):
        return False
    if not any(np.all(out[obj["coords"][:, 0], obj["coords"][:, 1]] == bg) for obj in objects if obj["id"] in unclaimed_ids):
        return False

    winner_palette = {int(winners[obj_id]["marker"]["color"]) for obj_id in claimed_ids}
    return len(winner_palette) >= 2


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _role_scene_attempt(GEN_RNG)
        if grid is None:
            continue
        if _validate_generated_scene(grid):
            return grid
    raise RuntimeError("failed to generate a valid ray-claims-first-hit puzzle")
