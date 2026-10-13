from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
ANCHOR_COLOR = 5
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
OPPOSITE = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
}
MOTIF_BLOCKER_MASKS = [
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 1, 0], [0, 1, 1, 1]], dtype=bool),
    np.array([[1, 0, 1, 1], [1, 1, 1, 0]], dtype=bool),
]
MOTIF_BLOCKED_MASKS = [
    np.array([[1, 0], [1, 1], [0, 1]], dtype=bool),
    np.array([[1, 0], [1, 1], [1, 1], [0, 1]], dtype=bool),
    np.array([[0, 1], [1, 1], [1, 0], [1, 1]], dtype=bool),
]


def _neighbors(r: int, c: int, connectivity: int) -> list[tuple[int, int]]:
    if connectivity == 4:
        return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            out.append((r + dr, c + dc))
    return out


def _find_components(grid: np.ndarray, *, color: int | None = None, exclude: set[int] | None = None, connectivity: int = 8) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    seen = np.zeros((H, W), dtype=bool)
    out: list[dict] = []
    exclude = set() if exclude is None else set(exclude)

    for r in range(H):
        for c in range(W):
            if seen[r, c]:
                continue
            v = int(g[r, c])
            if color is not None:
                if v != int(color):
                    continue
            elif v in exclude:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors(rr, cc, connectivity):
                    if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc] and int(g[nr, nc]) == v:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            rows = [p[0] for p in coords]
            cols = [p[1] for p in coords]
            out.append(
                {
                    "color": v,
                    "coords": coords,
                    "coord_set": set(coords),
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                    "area": len(coords),
                }
            )
    return out


def _bbox_dims(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return r1 - r0 + 1, c1 - c0 + 1


def _center2_from_bbox(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return r0 + r1, c0 + c1


def _is_solid_anchor(comp: dict, grid: np.ndarray) -> bool:
    if int(comp["color"]) != ANCHOR_COLOR or int(comp["area"]) != 4:
        return False
    r0, c0, r1, c1 = comp["bbox"]
    if (r1 - r0 + 1, c1 - c0 + 1) != (2, 2):
        return False
    return bool((grid[r0 : r1 + 1, c0 : c1 + 1] == ANCHOR_COLOR).all())


def _shift_coords(coords: list[tuple[int, int]], dr: int, dc: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for r, c in coords]


def _frontier_cells(coord_set: set[tuple[int, int]], direction: str) -> list[tuple[int, int]]:
    dr, dc = DIRS[direction]
    return [(r, c) for r, c in coord_set if (r + dr, c + dc) not in coord_set]


def _anchor_edge_pixels(anchor_bbox: tuple[int, int, int, int], direction: str) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = anchor_bbox
    if direction == "right":
        return {(r, c0) for r in range(r0, r1 + 1)}
    if direction == "left":
        return {(r, c1) for r in range(r0, r1 + 1)}
    if direction == "down":
        return {(r0, c) for c in range(c0, c1 + 1)}
    return {(r1, c) for c in range(c0, c1 + 1)}


def _reflect_coords(coords: list[tuple[int, int]], anchor_bbox: tuple[int, int, int, int], direction: str) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = anchor_bbox
    mirrored = []
    for r, c in coords:
        if direction == "right":
            mirrored.append((r, 2 * c0 - 1 - c))
        elif direction == "left":
            mirrored.append((r, 2 * c1 + 1 - c))
        elif direction == "down":
            mirrored.append((2 * r0 - 1 - r, c))
        else:
            mirrored.append((2 * r1 + 1 - r, c))
    return mirrored


def _render(coords_by_color: list[tuple[list[tuple[int, int]], int]], shape: tuple[int, int]) -> np.ndarray:
    H, W = shape
    out = np.full((H, W), BG, dtype=int)
    for coords, color in coords_by_color:
        for r, c in coords:
            assert 0 <= r < H and 0 <= c < W
            assert int(out[r, c]) == BG
            out[r, c] = int(color)
    return out


def _component_distance(sprite_coords: list[tuple[int, int]], anchor_coords: list[tuple[int, int]]) -> int:
    best = None
    for sr, sc in sprite_coords:
        for ar, ac in anchor_coords:
            dist = abs(sr - ar) + abs(sc - ac)
            if best is None or dist < best:
                best = dist
    assert best is not None
    return int(best)


def _simulate(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    assert g.ndim == 2

    anchors = [comp for comp in _find_components(g, color=ANCHOR_COLOR, connectivity=8) if _is_solid_anchor(comp, g)]
    sprites = _find_components(g, exclude={BG, ANCHOR_COLOR}, connectivity=8)
    assert 2 <= len(sprites) <= 4
    assert len(anchors) == len(sprites)
    assert len({int(comp["color"]) for comp in sprites}) == len(sprites)

    sprite_by_idx = {i: sprite for i, sprite in enumerate(sprites)}
    anchor_by_idx = {i: anchor for i, anchor in enumerate(anchors)}

    sprite_choices: dict[int, tuple[int, int]] = {}
    pair_records = []

    for si, sprite in sprite_by_idx.items():
        ranked = []
        for ai, anchor in anchor_by_idx.items():
            dist2 = _component_distance(sprite["coords"], anchor["coords"])
            ranked.append((dist2, ai))
        ranked.sort()
        assert len(ranked) >= 2
        assert ranked[0][0] < ranked[1][0]
        sprite_choices[si] = ranked[0]

    used_anchors = set()
    for si, (dist2, ai) in sprite_choices.items():
        assert ai not in used_anchors
        used_anchors.add(ai)

        sprite = sprite_by_idx[si]
        anchor = anchor_by_idx[ai]
        s_row2, s_col2 = _center2_from_bbox(sprite["bbox"])
        a_row2, a_col2 = _center2_from_bbox(anchor["bbox"])
        same_row = s_row2 == a_row2
        same_col = s_col2 == a_col2
        assert same_row ^ same_col
        if same_row:
            direction = "right" if a_col2 > s_col2 else "left"
        else:
            direction = "down" if a_row2 > s_row2 else "up"
        pair_records.append(
            {
                "sprite_index": si,
                "anchor_index": ai,
                "sprite": sprite,
                "anchor": anchor,
                "distance2": int(dist2),
                "direction": direction,
            }
        )

    pair_records.sort(key=lambda item: item["distance2"])
    assert len({item["distance2"] for item in pair_records}) == len(pair_records)

    occupied_kind: dict[tuple[int, int], str] = {}
    painted: list[tuple[list[tuple[int, int]], int]] = []
    details = {"pairs": []}

    for pair in pair_records:
        direction = pair["direction"]
        dr, dc = DIRS[direction]
        anchor_edge = _anchor_edge_pixels(pair["anchor"]["bbox"], direction)
        cur_coords = list(pair["sprite"]["coords"])
        stop_event = None
        blocker_kind = None

        for _ in range(max(H, W) + 5):
            cur_set = set(cur_coords)
            front = _frontier_cells(cur_set, direction)
            forward_cells = {(r + dr, c + dc) for r, c in front}
            block_hits = forward_cells & set(occupied_kind)
            anchor_hits = forward_cells & anchor_edge
            assert not (block_hits and anchor_hits)
            if block_hits:
                stop_event = "blocked"
                kinds = {occupied_kind[cell] for cell in block_hits}
                assert len(kinds) == 1
                blocker_kind = next(iter(kinds))
                break
            if anchor_hits:
                stop_event = "anchor"
                break
            next_coords = _shift_coords(cur_coords, dr, dc)
            assert all(0 <= r < H and 0 <= c < W for r, c in next_coords)
            cur_coords = next_coords

        assert stop_event is not None
        cur_set = set(cur_coords)
        assert not (cur_set & set(occupied_kind))
        for cell in cur_coords:
            occupied_kind[cell] = "sprite"
        painted.append((cur_coords, int(pair["sprite"]["color"])))

        mirror_coords = None
        if stop_event == "anchor":
            mirror_coords = _reflect_coords(cur_coords, pair["anchor"]["bbox"], direction)
            mirror_set = set(mirror_coords)
            assert len(mirror_set) == len(mirror_coords)
            assert not (mirror_set & cur_set)
            assert not (mirror_set & set(occupied_kind))
            assert all(0 <= r < H and 0 <= c < W for r, c in mirror_coords)
            for cell in mirror_coords:
                occupied_kind[cell] = "mirror"
            painted.append((mirror_coords, ANCHOR_COLOR))

        details["pairs"].append(
            {
                "sprite_color": int(pair["sprite"]["color"]),
                "anchor_bbox": pair["anchor"]["bbox"],
                "distance2": pair["distance2"],
                "direction": direction,
                "event": stop_event,
                "blocker_kind": blocker_kind,
                "stop_coords": cur_coords,
                "mirror_coords": mirror_coords,
            }
        )

    return _render(painted, (H, W)), details


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair each colored sprite with its nearest gray 2x2 anchor, slide it toward that anchor, and mirror only the sprites that actually reach anchor contact.

    Input structure:
    - The background is black.
    - There are 2 to 4 differently colored monochrome 8-connected sprites, each with 4 to 8 pixels and not a solid 2x2.
    - There are equally many gray solid 2x2 anchor blocks.

    Rule:
    - Find all sprites and all gray 2x2 anchors, then pair every sprite with its unique nearest anchor by minimum Manhattan distance between sprite pixels and anchor pixels.
    - For each pair, use the unique shared row or shared column of those centers to determine the slide direction, and process pairs from smaller center distance to larger center distance.
    - Slide the current sprite one cell at a time along that direction until either its front edge would touch its paired anchor's facing edge, or its front edge would touch a previously stopped sprite or gray mirror copy.
    - If the sprite reaches its paired anchor, keep the moved sprite and add its reflection across the anchor-facing edge in gray; if it is blocked by an already stopped object, keep only the moved sprite.
    - Erase all original sprite positions and erase all anchors from the final grid.

    Key insight:
    - Nearest pairing fixes each sprite's motion axis, but the distance order is equally important because earlier stopped sprites become later collision obstacles, which decides whether a mirror appears at all.
    """
    out, _details = _simulate(np.asarray(grid, dtype=int))
    return out


def _mask_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return False
    seen = {tuple(map(int, pts[0]))}
    q = deque([tuple(map(int, pts[0]))])
    h, w = mask.shape
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors(r, c, 8):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _touches_all_edges(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _random_mask(direction: str) -> np.ndarray:
    if direction in {"left", "right"}:
        dims = [(2, 3), (2, 4), (4, 2), (4, 3), (4, 4)]
        mirror = np.fliplr
    else:
        dims = [(3, 2), (4, 2), (2, 4), (3, 4), (4, 4)]
        mirror = np.flipud

    for _ in range(300):
        h, w = random.choice(dims)
        area = random.randint(4, min(8, h * w))
        if h == 2 and w == 2 and area == 4:
            continue
        mask = np.zeros((h, w), dtype=bool)
        start = (random.randrange(h), random.randrange(w))
        mask[start] = True
        while int(mask.sum()) < area:
            frontier = set()
            for r, c in np.argwhere(mask):
                for nr, nc in _neighbors(int(r), int(c), 8):
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                        frontier.add((nr, nc))
            if not frontier:
                break
            pick = random.choice(list(frontier))
            mask[pick] = True
        if int(mask.sum()) != area:
            continue
        if not _mask_connected(mask):
            continue
        if not _touches_all_edges(mask):
            continue
        if h == 2 and w == 2 and mask.all():
            continue
        if np.array_equal(mask, mirror(mask)):
            continue
        return mask
    raise RuntimeError("failed to sample sprite mask")


def _mask_coords(mask: np.ndarray, top_left: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = top_left
    return [(r0 + int(r), c0 + int(c)) for r, c in np.argwhere(mask)]


def _in_bounds_box(H: int, W: int, r0: int, c0: int, h: int, w: int) -> bool:
    return 0 <= r0 and 0 <= c0 and r0 + h <= H and c0 + w <= W


def _shift_tl(top_left: tuple[int, int], direction: str, steps: int) -> tuple[int, int]:
    dr, dc = DIRS[direction]
    return top_left[0] + dr * steps, top_left[1] + dc * steps


def _anchor_tl_for_stop(stop_tl: tuple[int, int], shape: tuple[int, int], direction: str, lead_steps: int = 0) -> tuple[int, int]:
    r0, c0 = _shift_tl(stop_tl, direction, lead_steps)
    h, w = shape
    if direction == "right":
        return r0 + (h - 2) // 2, c0 + w
    if direction == "left":
        return r0 + (h - 2) // 2, c0 - 2
    if direction == "down":
        return r0 + h, c0 + (w - 2) // 2
    return r0 - 2, c0 + (w - 2) // 2


def _mirror_tl_for_stop(stop_tl: tuple[int, int], shape: tuple[int, int], direction: str) -> tuple[int, int]:
    h, w = shape
    ar, ac = _anchor_tl_for_stop(stop_tl, shape, direction, lead_steps=0)
    if direction == "right":
        return stop_tl[0], ac
    if direction == "left":
        return stop_tl[0], ac - w + 2
    if direction == "down":
        return ar, stop_tl[1]
    return ar - h + 2, stop_tl[1]


def _placeable(occupied: np.ndarray, coords: list[tuple[int, int]]) -> bool:
    H, W = occupied.shape
    for r, c in coords:
        if not (0 <= r < H and 0 <= c < W):
            return False
        r0 = max(0, r - 1)
        r1 = min(H, r + 2)
        c0 = max(0, c - 1)
        c1 = min(W, c + 2)
        if occupied[r0:r1, c0:c1].any():
            return False
    return True


def _mark_occupied(occupied: np.ndarray, coords: list[tuple[int, int]]) -> None:
    for r, c in coords:
        occupied[r, c] = True


def _sprite_bbox_from_tl(mask: np.ndarray, tl: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = mask.shape
    r0, c0 = tl
    return r0, c0, r0 + h - 1, c0 + w - 1


def _overlap_positions(size_a: int, size_b: int, start_a: int, end_a: int) -> list[int]:
    low = start_a - size_b + 1
    high = end_a
    return list(range(low, high + 1))


def _adjacent_stop_choices(obstacle_bbox: tuple[int, int, int, int], blocked_shape: tuple[int, int], direction: str) -> list[tuple[int, int]]:
    or0, oc0, or1, oc1 = obstacle_bbox
    h, w = blocked_shape
    out = []
    if direction == "right":
        c0 = oc0 - w
        for r0 in _overlap_positions(or1 - or0 + 1, h, or0, or1):
            out.append((r0, c0))
    elif direction == "left":
        c0 = oc1 + 1
        for r0 in _overlap_positions(or1 - or0 + 1, h, or0, or1):
            out.append((r0, c0))
    elif direction == "down":
        r0 = or0 - h
        for c0 in _overlap_positions(oc1 - oc0 + 1, w, oc0, oc1):
            out.append((r0, c0))
    else:
        r0 = or1 + 1
        for c0 in _overlap_positions(oc1 - oc0 + 1, w, oc0, oc1):
            out.append((r0, c0))
    random.shuffle(out)
    return out


def _render_input(H: int, W: int, pair_specs: list[dict]) -> np.ndarray:
    grid = np.full((H, W), BG, dtype=int)
    for spec in pair_specs:
        for r, c in _mask_coords(spec["mask"], spec["start_tl"]):
            assert int(grid[r, c]) == BG
            grid[r, c] = int(spec["color"])
        ar, ac = spec["anchor_tl"]
        patch = [(ar, ac), (ar + 1, ac), (ar, ac + 1), (ar + 1, ac + 1)]
        for r, c in patch:
            assert int(grid[r, c]) == BG
            grid[r, c] = ANCHOR_COLOR
    return grid


def _free_pair_candidates(H: int, W: int, direction: str, occupied: np.ndarray, color: int) -> list[dict]:
    mask = _random_mask(direction)
    h, w = mask.shape
    specs = []
    for _ in range(120):
        stop_tl = (random.randint(0, H - h), random.randint(0, W - w))
        start_steps = random.randint(1, 6)
        start_tl = _shift_tl(stop_tl, OPPOSITE[direction], start_steps)
        anchor_tl = _anchor_tl_for_stop(stop_tl, mask.shape, direction, lead_steps=0)
        mirror_tl = _mirror_tl_for_stop(stop_tl, mask.shape, direction)
        if not _in_bounds_box(H, W, start_tl[0], start_tl[1], h, w):
            continue
        if not _in_bounds_box(H, W, anchor_tl[0], anchor_tl[1], 2, 2):
            continue
        if not _in_bounds_box(H, W, mirror_tl[0], mirror_tl[1], h, w):
            continue
        sprite_coords = _mask_coords(mask, start_tl)
        anchor_coords = [(anchor_tl[0], anchor_tl[1]), (anchor_tl[0] + 1, anchor_tl[1]), (anchor_tl[0], anchor_tl[1] + 1), (anchor_tl[0] + 1, anchor_tl[1] + 1)]
        if not (_placeable(occupied, sprite_coords) and _placeable(occupied, anchor_coords)):
            continue
        specs.append(
            {
                "mask": mask,
                "color": color,
                "direction": direction,
                "stop_tl": stop_tl,
                "start_tl": start_tl,
                "anchor_tl": anchor_tl,
                "lead_steps": 0,
                "kind": "free",
            }
        )
    return specs


def _generate_attempt() -> np.ndarray | None:
    H = random.randint(18, 25)
    W = random.randint(18, 25)
    pair_count = random.randint(2, 4)
    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], pair_count)

    occupied = np.zeros((H, W), dtype=bool)
    pair_specs: list[dict] = []

    directions = [random.choice(list(DIRS)) for _ in range(pair_count)]
    for color, direction in zip(colors, directions):
        cands = _free_pair_candidates(H, W, direction, occupied, color)
        if not cands:
            return None
        spec = random.choice(cands)
        pair_specs.append(spec)
        _mark_occupied(occupied, _mask_coords(spec["mask"], spec["start_tl"]))
        _mark_occupied(
            occupied,
            [
                (spec["anchor_tl"][0], spec["anchor_tl"][1]),
                (spec["anchor_tl"][0] + 1, spec["anchor_tl"][1]),
                (spec["anchor_tl"][0], spec["anchor_tl"][1] + 1),
                (spec["anchor_tl"][0] + 1, spec["anchor_tl"][1] + 1),
            ],
        )

    inp = _render_input(H, W, pair_specs)
    try:
        out, details = _simulate(inp)
    except AssertionError:
        return None

    if out.shape != inp.shape:
        return None
    if sum(item["event"] == "anchor" for item in details["pairs"]) != len(details["pairs"]):
        return None
    if len({item["distance2"] for item in details["pairs"]}) != len(details["pairs"]):
        return None

    stop_signatures = set()
    for item in details["pairs"]:
        sig = tuple(sorted(item["stop_coords"]))
        if sig in stop_signatures:
            return None
        stop_signatures.add(sig)

    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input after many attempts")
