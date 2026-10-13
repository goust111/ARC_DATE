from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NEIGHBORS8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]
GEN_ATTEMPTS = 400
SHAPE_ATTEMPTS = 250
LOCK_ATTEMPTS = 120
PLACE_ATTEMPTS = 250


def _find_components(mask: np.ndarray, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []
    neighbors = NEIGHBORS4 if connectivity == 4 else NEIGHBORS8
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((int(cr), int(cc)))
                for dr, dc in neighbors:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(coords)
    return components


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return np.zeros((0, 0), dtype=bool)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _normalize_coords(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    coords_list = list(coords)
    if not coords_list:
        return tuple()
    r0, c0, _, _ = _bbox(coords_list)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords_list))


def _mask_to_coords(mask: np.ndarray) -> list[tuple[int, int]]:
    return [(int(r), int(c)) for r, c in np.argwhere(mask)]


def _orientation_signatures_from_coords(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[tuple[int, int], ...], ...]:
    coords_list = list(coords)
    if not coords_list:
        return (tuple(),)
    r0, c0, r1, c1 = _bbox(coords_list)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in coords_list:
        mask[r - r0, c - c0] = True
    signatures: list[tuple[tuple[int, int], ...]] = []
    for k in range(4):
        rot = np.rot90(mask, k)
        signatures.append(_normalize_coords(_mask_to_coords(rot)))
    return tuple(signatures)


def _canonical_signature(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return min(_orientation_signatures_from_coords(coords))


def _holes_in_component_mask(component_mask: np.ndarray) -> list[list[tuple[int, int]]]:
    component_mask = np.asarray(component_mask, dtype=bool)
    h, w = component_mask.shape
    background = ~component_mask
    visited = np.zeros((h, w), dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if background[r, c] and not visited[r, c]:
                visited[r, c] = True
                queue.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if background[r, c] and not visited[r, c]:
                visited[r, c] = True
                queue.append((r, c))

    while queue:
        r, c = queue.popleft()
        for dr, dc in NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and background[nr, nc] and not visited[nr, nc]:
                visited[nr, nc] = True
                queue.append((nr, nc))

    hole_mask = background & ~visited
    return _find_components(hole_mask, connectivity=4)


def _holes_in_grid_component(grid: np.ndarray, coords: list[tuple[int, int]], bg: int = BG) -> list[list[tuple[int, int]]]:
    r0, c0, r1, c1 = _bbox(coords)
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    comp_mask = np.zeros(sub.shape, dtype=bool)
    for r, c in coords:
        comp_mask[r - r0, c - c0] = True
    bg_mask = sub == int(bg)

    h, w = sub.shape
    visited = np.zeros((h, w), dtype=bool)
    queue: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in (0, w - 1):
            if bg_mask[r, c] and not visited[r, c]:
                visited[r, c] = True
                queue.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if bg_mask[r, c] and not visited[r, c]:
                visited[r, c] = True
                queue.append((r, c))

    while queue:
        r, c = queue.popleft()
        for dr, dc in NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and bg_mask[nr, nc] and not visited[nr, nc]:
                visited[nr, nc] = True
                queue.append((nr, nc))

    holes_mask = bg_mask & ~visited
    holes_rel = _find_components(holes_mask, connectivity=4)
    holes_abs: list[list[tuple[int, int]]] = []
    for hole in holes_rel:
        holes_abs.append([(int(r0 + r), int(c0 + c)) for r, c in hole])
    return holes_abs


def _build_shell_from_cavity(mask: np.ndarray) -> np.ndarray | None:
    cavity = np.pad(np.asarray(mask, dtype=bool), 1)
    shell = np.zeros_like(cavity, dtype=bool)
    for r, c in np.argwhere(cavity):
        for dr, dc in NEIGHBORS4:
            nr, nc = int(r + dr), int(c + dc)
            if 0 <= nr < cavity.shape[0] and 0 <= nc < cavity.shape[1] and not cavity[nr, nc]:
                shell[nr, nc] = True

    if len(_find_components(shell, connectivity=8)) != 1:
        return None
    holes = _holes_in_component_mask(shell)
    if len(holes) != 1:
        return None
    if _canonical_signature(holes[0]) != _canonical_signature(_mask_to_coords(mask)):
        return None
    return shell


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each matching lock cavity with its key color, then erase the key itself.

    Input structure:
    - The background is black.
    - Some monochrome 4-connected objects are hollow closed outlines ("locks"), and each such object encloses one or more black cavities.
    - Other monochrome 4-connected objects are solid connected blocks ("keys"), with colors different from the lock-outline colors.

    Rule:
    - For every key, find the unique lock cavity whose shape is congruent to the key under a rotation of 0/90/180/270 degrees, without reflection.
    - Fill every black cell of that cavity with the key's color.
    - Remove the key object itself by turning all of its cells black.
    - Leave unmatched lock cavities and all lock outlines unchanged.

    Key insight:
    - Keys are not moved geometrically; only their shape is matched to a cavity up to rotation, and their color is transferred into that cavity while the original key disappears.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    locks: list[dict] = []
    keys: list[dict] = []
    for color in sorted(int(c) for c in np.unique(g) if int(c) != BG):
        for coords in _find_components(g == int(color), connectivity=8):
            holes = _holes_in_grid_component(g, coords, bg=BG)
            if holes:
                locks.append(
                    {
                        "color": int(color),
                        "coords": coords,
                        "holes": [
                            {
                                "coords": hole,
                                "signature": _canonical_signature(hole),
                            }
                            for hole in holes
                        ],
                    }
                )
            else:
                keys.append(
                    {
                        "color": int(color),
                        "coords": coords,
                        "signature": _canonical_signature(coords),
                    }
                )

    cavity_by_signature: dict[tuple[tuple[int, int], ...], dict] = {}
    for lock in locks:
        for hole in lock["holes"]:
            sig = hole["signature"]
            assert sig not in cavity_by_signature, "ambiguous duplicate cavity shapes"
            cavity_by_signature[sig] = hole

    out = g.copy()
    for key in keys:
        coords = key["coords"]
        for r, c in coords:
            out[r, c] = BG

        sig = key["signature"]
        assert sig in cavity_by_signature, "every key must match a lock cavity"
        hole = cavity_by_signature.pop(sig)
        for r, c in hole["coords"]:
            out[r, c] = int(key["color"])

    return out


def _random_connected_shape(rng: random.Random) -> dict | None:
    for _ in range(SHAPE_ATTEMPTS):
        h = rng.randint(2, 5)
        w = rng.randint(2, 5)
        max_area = min(h * w, 7)
        target = rng.randint(2, max_area)

        cells = {(rng.randrange(h), rng.randrange(w))}
        while len(cells) < target:
            frontier: list[tuple[int, int]] = []
            for r, c in cells:
                for dr, dc in NEIGHBORS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(rng.choice(frontier))

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _crop_mask(mask)
        if mask.size == 0 or mask.sum() < 2:
            continue
        if _holes_in_component_mask(mask):
            continue

        shell = _build_shell_from_cavity(mask)
        if shell is None:
            continue

        return {
            "mask": mask,
            "shell": shell,
            "signature": _canonical_signature(_mask_to_coords(mask)),
            "orientation_count": len(set(_orientation_signatures_from_coords(_mask_to_coords(mask)))),
        }
    return None


def _select_edge_anchor(mask: np.ndarray, side: str) -> tuple[int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0
    if side == "left":
        edge_value = int(coords[:, 1].min())
        edge = coords[coords[:, 1] == edge_value]
        target = float(coords[:, 0].mean())
        idx = int(np.argmin(np.abs(edge[:, 0] - target)))
    elif side == "right":
        edge_value = int(coords[:, 1].max())
        edge = coords[coords[:, 1] == edge_value]
        target = float(coords[:, 0].mean())
        idx = int(np.argmin(np.abs(edge[:, 0] - target)))
    elif side == "top":
        edge_value = int(coords[:, 0].min())
        edge = coords[coords[:, 0] == edge_value]
        target = float(coords[:, 1].mean())
        idx = int(np.argmin(np.abs(edge[:, 1] - target)))
    else:
        edge_value = int(coords[:, 0].max())
        edge = coords[coords[:, 0] == edge_value]
        target = float(coords[:, 1].mean())
        idx = int(np.argmin(np.abs(edge[:, 1] - target)))
    return int(edge[idx, 0]), int(edge[idx, 1])


def _bfs_path(obj_mask: np.ndarray, blocked_extra: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]] | None:
    h, w = obj_mask.shape
    blocked = obj_mask | blocked_extra
    allowed = ~blocked
    allowed[start] = True
    allowed[goal] = True

    queue: deque[tuple[int, int]] = deque([start])
    seen = np.zeros((h, w), dtype=bool)
    seen[start] = True
    prev: dict[tuple[int, int], tuple[int, int] | None] = {start: None}

    while queue:
        r, c = queue.popleft()
        if (r, c) == goal:
            break
        for dr, dc in NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and allowed[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                prev[(nr, nc)] = (r, c)
                queue.append((nr, nc))

    if not seen[goal]:
        return None

    path: list[tuple[int, int]] = []
    cur: tuple[int, int] | None = goal
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def _build_lock_patch(specs: list[dict], rng: random.Random) -> dict | None:
    desired_counter = Counter(spec["signature"] for spec in specs)

    for _ in range(LOCK_ATTEMPTS):
        oriented: list[dict] = []
        for spec in specs:
            k = rng.randrange(4)
            oriented.append(
                {
                    "shell": np.rot90(spec["shell"], k).copy(),
                    "signature": spec["signature"],
                }
            )

        axis = rng.choice(["horizontal", "vertical"]) if len(oriented) > 1 else "horizontal"
        gap = rng.randint(3, 5)
        margin = 2
        if axis == "horizontal":
            canvas_h = max(item["shell"].shape[0] for item in oriented) + margin * 2
            canvas_w = sum(item["shell"].shape[1] for item in oriented) + gap * (len(oriented) - 1) + margin * 2
        else:
            canvas_h = sum(item["shell"].shape[0] for item in oriented) + gap * (len(oriented) - 1) + margin * 2
            canvas_w = max(item["shell"].shape[1] for item in oriented) + margin * 2

        canvas = np.zeros((canvas_h, canvas_w), dtype=bool)
        cavity_block = np.zeros_like(canvas, dtype=bool)
        placements: list[dict] = []

        cursor = margin
        for item in oriented:
            shell = item["shell"]
            sh, sw = shell.shape
            if axis == "horizontal":
                r0 = (canvas_h - sh) // 2
                c0 = cursor
                cursor += sw + gap
            else:
                r0 = cursor
                c0 = (canvas_w - sw) // 2
                cursor += sh + gap

            canvas[r0 : r0 + sh, c0 : c0 + sw] |= shell
            holes = _holes_in_component_mask(shell)
            if len(holes) != 1:
                placements = []
                break
            for rr, cc in holes[0]:
                cavity_block[r0 + rr, c0 + cc] = True
            placements.append({"r0": r0, "c0": c0, "shell": shell, "signature": item["signature"]})

        if not placements:
            continue

        ok = True
        for left, right in zip(placements, placements[1:]):
            if axis == "horizontal":
                start_local = _select_edge_anchor(left["shell"], "right")
                goal_local = _select_edge_anchor(right["shell"], "left")
            else:
                start_local = _select_edge_anchor(left["shell"], "bottom")
                goal_local = _select_edge_anchor(right["shell"], "top")
            start = (left["r0"] + start_local[0], left["c0"] + start_local[1])
            goal = (right["r0"] + goal_local[0], right["c0"] + goal_local[1])
            path = _bfs_path(canvas, cavity_block, start, goal)
            if path is None:
                ok = False
                break
            for r, c in path:
                canvas[r, c] = True
        if not ok:
            continue

        canvas = _crop_mask(canvas)
        if len(_find_components(canvas, connectivity=8)) != 1:
            continue

        holes = _holes_in_component_mask(canvas)
        hole_counter = Counter(_canonical_signature(hole) for hole in holes)
        if hole_counter != desired_counter:
            continue

        holes_by_sig = { _canonical_signature(hole): hole for hole in holes }
        return {"patch": canvas, "holes_by_signature": holes_by_sig}
    return None


def _random_partition(total: int, parts: int, max_part: int, rng: random.Random) -> list[int] | None:
    for _ in range(80):
        if parts == 1:
            sizes = [total]
        else:
            cuts = sorted(rng.sample(range(1, total), parts - 1))
            nums = [0] + cuts + [total]
            sizes = [nums[i + 1] - nums[i] for i in range(parts)]
        if 1 <= min(sizes) and max(sizes) <= max_part:
            return sizes
    return None


def _place_objects(h: int, w: int, objects: list[dict], rng: random.Random) -> list[tuple[int, int]] | None:
    occupancy = np.zeros((h, w), dtype=bool)
    positions: list[tuple[int, int] | None] = [None] * len(objects)
    order = sorted(range(len(objects)), key=lambda i: -int(objects[i]["mask"].sum()))

    for idx in order:
        mask = objects[idx]["mask"]
        oh, ow = mask.shape
        if oh + 2 > h or ow + 2 > w:
            return None

        placed = False
        for _ in range(PLACE_ATTEMPTS):
            r0 = rng.randint(1, h - oh - 1)
            c0 = rng.randint(1, w - ow - 1)
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(h, r0 + oh + 1)
            cc1 = min(w, c0 + ow + 1)
            if occupancy[rr0:rr1, cc0:cc1].any():
                continue
            occupancy[r0 : r0 + oh, c0 : c0 + ow] |= mask
            positions[idx] = (r0, c0)
            placed = True
            break
        if not placed:
            return None

    return [(int(r), int(c)) for r, c in positions if r is not None and c is not None]


def _gen_attempt(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    total_cavities = rng.randint(2, 5)
    feasible_parts = [parts for parts in range(1, min(3, total_cavities) + 1) if total_cavities <= parts * 3]
    part_weights = [1 if parts == 1 else 3 if parts == 2 else 2 for parts in feasible_parts]
    num_locks = rng.choices(feasible_parts, weights=part_weights, k=1)[0]
    group_sizes = _random_partition(total_cavities, num_locks, 3, rng)
    if group_sizes is None:
        return None

    want_partial = total_cavities > 1 and rng.random() < 0.5

    specs: list[dict] = []
    seen_signatures: set[tuple[tuple[int, int], ...]] = set()
    while len(specs) < total_cavities:
        spec = _random_connected_shape(rng)
        if spec is None or spec["signature"] in seen_signatures:
            continue
        specs.append(spec)
        seen_signatures.add(spec["signature"])

    cavity_groups: list[list[dict]] = []
    start = 0
    for size in group_sizes:
        cavity_groups.append(specs[start : start + size])
        start += size

    global_index = 0
    cavities_meta: list[dict] = []
    for lock_index, group in enumerate(cavity_groups):
        for spec in group:
            cavities_meta.append({"lock_index": lock_index, "spec": spec, "global_index": global_index})
            global_index += 1

    if want_partial:
        matched_ids: set[int]
        if num_locks > 1 and rng.random() < 0.65:
            unmatched_lock = rng.randrange(num_locks)
            matched_ids = {
                item["global_index"]
                for item in cavities_meta
                if item["lock_index"] != unmatched_lock
            }
            if not matched_ids:
                matched_ids = {rng.randrange(total_cavities)}
            if len(matched_ids) > 1 and rng.random() < 0.4:
                drop_count = rng.randint(1, len(matched_ids) - 1)
                for drop in rng.sample(sorted(matched_ids), drop_count):
                    if len(matched_ids) == 1:
                        break
                    matched_ids.remove(drop)
        else:
            matched_count = rng.randint(1, total_cavities - 1)
            matched_ids = set(rng.sample(range(total_cavities), matched_count))
    else:
        matched_ids = set(range(total_cavities))

    if not any(specs[idx]["orientation_count"] > 1 for idx in matched_ids):
        return None

    lock_patches: list[dict] = []
    for group in cavity_groups:
        patch = _build_lock_patch(group, rng)
        if patch is None:
            return None
        lock_patches.append(patch)

    matched_specs = [specs[idx] for idx in sorted(matched_ids)]
    total_colors = num_locks + len(matched_specs)
    if total_colors > 9:
        return None
    palette = rng.sample(list(range(1, 10)), total_colors)
    lock_colors = palette[:num_locks]
    key_colors = palette[num_locks:]

    objects: list[dict] = []
    for lock_index, patch in enumerate(lock_patches):
        objects.append(
            {
                "kind": "lock",
                "mask": patch["patch"],
                "color": lock_colors[lock_index],
                "holes_by_signature": patch["holes_by_signature"],
            }
        )
    for spec, color in zip(matched_specs, key_colors):
        rot = rng.randrange(4)
        key_mask = np.rot90(spec["mask"], rot).copy()
        objects.append(
            {
                "kind": "key",
                "mask": key_mask,
                "color": int(color),
                "signature": spec["signature"],
            }
        )

    max_obj_h = max(obj["mask"].shape[0] for obj in objects)
    max_obj_w = max(obj["mask"].shape[1] for obj in objects)
    min_h = max(12, max_obj_h + 2, int(np.ceil(np.sqrt(sum(int(obj["mask"].sum()) for obj in objects) * 3))))
    min_w = max(12, max_obj_w + 2, int(np.ceil(np.sqrt(sum(int(obj["mask"].sum()) for obj in objects) * 3))))
    if min_h > 30 or min_w > 30:
        return None

    h = rng.randint(min_h, 30)
    w = rng.randint(min_w, 30)
    positions = _place_objects(h, w, objects, rng)
    if positions is None:
        return None

    grid = np.zeros((h, w), dtype=int)
    signature_to_hole: dict[tuple[tuple[int, int], ...], list[tuple[int, int]]] = {}
    key_records: list[dict] = []
    for obj, (r0, c0) in zip(objects, positions):
        mask = obj["mask"]
        oh, ow = mask.shape
        grid[r0 : r0 + oh, c0 : c0 + ow][mask] = int(obj["color"])
        if obj["kind"] == "lock":
            for sig, hole in obj["holes_by_signature"].items():
                signature_to_hole[sig] = [(int(r0 + r), int(c0 + c)) for r, c in hole]
        else:
            key_coords = [(int(r0 + r), int(c0 + c)) for r, c in _mask_to_coords(mask)]
            key_records.append({"signature": obj["signature"], "color": int(obj["color"]), "coords": key_coords})

    expected = grid.copy()
    for key in key_records:
        for r, c in key["coords"]:
            expected[r, c] = BG
        for r, c in signature_to_hole[key["signature"]]:
            expected[r, c] = int(key["color"])

    if np.array_equal(grid, expected):
        return None

    return grid, expected


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        result = _gen_attempt(rng)
        if result is None:
            continue
        grid, expected = result
        try:
            actual = main(grid)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
