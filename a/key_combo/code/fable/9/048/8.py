from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
PALETTE = tuple(range(1, 10))
BG = 0
GEN_ATTEMPTS = 500
PLACEMENT_GAP = 1


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return np.zeros((0, 0), dtype=bool)
    return mask[int(rows[0]) : int(rows[-1]) + 1, int(cols[0]) : int(cols[-1]) + 1].copy()


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    tight = _crop_mask(mask.astype(bool))
    return tight.shape, tuple(int(v) for v in tight.ravel())


def _canonical_rotation_key(mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    return min(_mask_key(np.rot90(mask, k)) for k in range(4))


def _rotationally_asymmetric(mask: np.ndarray) -> bool:
    tight = _crop_mask(mask.astype(bool))
    return all(not np.array_equal(tight, np.rot90(tight, k)) for k in (1, 2, 3))


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _find_monochrome_components(grid: np.ndarray, bg: int) -> list[dict]:
    comps: list[dict] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        mask = grid == color
        for coords in _connected_components(mask):
            r0, c0, r1, c1 = _bbox_from_coords(coords)
            local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for r, c in coords:
                local[r - r0, c - c0] = True
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "mask": local,
                    "area": int(local.sum()),
                }
            )
    return comps


def _enclosed_holes(component: dict) -> list[dict]:
    mask = component["mask"]
    r0, c0, _, _ = component["bbox"]
    holes: list[dict] = []
    for hole_coords in _connected_components(~mask):
        touches_edge = any(
            rr == 0 or cc == 0 or rr == mask.shape[0] - 1 or cc == mask.shape[1] - 1
            for rr, cc in hole_coords
        )
        if touches_edge:
            continue
        hr0, hc0, hr1, hc1 = _bbox_from_coords(hole_coords)
        local = np.zeros((hr1 - hr0 + 1, hc1 - hc0 + 1), dtype=bool)
        global_coords: list[tuple[int, int]] = []
        for rr, cc in hole_coords:
            local[rr - hr0, cc - hc0] = True
            global_coords.append((r0 + rr, c0 + cc))
        holes.append(
            {
                "coords": global_coords,
                "bbox": (r0 + hr0, c0 + hc0, r0 + hr1, c0 + hc1),
                "mask": local,
                "area": int(local.sum()),
            }
        )
    return holes


def _classify_scene(grid: np.ndarray) -> tuple[list[dict], list[dict], int]:
    bg = _get_bg(grid)
    hosts: list[dict] = []
    sprites: list[dict] = []
    for comp in _find_monochrome_components(grid, bg):
        holes = _enclosed_holes(comp)
        if len(holes) == 1:
            hosts.append({**comp, "hole": holes[0]})
        elif len(holes) == 0:
            sprites.append(comp)
        else:
            raise ValueError("each component must be a free sprite or a one-hole host")
    return hosts, sprites, bg


def _matchings(hosts: list[dict], sprites: list[dict]) -> tuple[list[tuple[int, int, int]], list[int]]:
    all_matches: list[tuple[int, int, int]] = []
    needed_rotations: list[int] = []
    for si, sprite in enumerate(sprites):
        matches: list[tuple[int, int, int]] = []
        for hi, host in enumerate(hosts):
            hole_mask = host["hole"]["mask"]
            for k in range(4):
                if np.array_equal(np.rot90(sprite["mask"], k), hole_mask):
                    matches.append((si, hi, k))
        if len(matches) != 1:
            raise ValueError("each sprite must match exactly one host hole under exactly one rotation")
        all_matches.append(matches[0])
        needed_rotations.append(matches[0][2])

    host_hits = Counter(hi for _, hi, _ in all_matches)
    if len(host_hits) != len(hosts) or any(v != 1 for v in host_hits.values()):
        raise ValueError("sprite-hole relation must be bijective")
    return all_matches, needed_rotations


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each free sprite to fit the matching enclosed hole and erase the original sprite.

    Input structure:
    - The background contains several large monochrome host objects.
    - Each host encloses exactly one background hole, and the hole shapes are pairwise different up to rotation.
    - Elsewhere there are equally many monochrome free sprites; each sprite matches one hole only after a unique quarter-turn.

    Rule:
    - Detect the enclosed hole inside every host object.
    - For each free sprite, try the four quarter-turn rotations and find the unique hole whose negative-space mask it exactly fills.
    - Fill that hole with the sprite's own color, and clear the sprite's original cells to background.

    Key insight:
    - The template to match is not a drawn foreground object but the host's internal background hole, so the solver must compare sprites against negative space under rotation.
    """
    g = np.asarray(grid, dtype=int)
    hosts, sprites, bg = _classify_scene(g)
    if not hosts or len(hosts) != len(sprites):
        raise ValueError("expected equally many one-hole hosts and free sprites")

    matches, _ = _matchings(hosts, sprites)

    out = g.copy()
    for sprite in sprites:
        rows, cols = zip(*sprite["coords"])
        out[rows, cols] = bg
    for si, hi, _k in matches:
        fill_color = sprites[si]["color"]
        rows, cols = zip(*hosts[hi]["hole"]["coords"])
        out[rows, cols] = fill_color
    return out


def _b(mask_rows: list[str]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in mask_rows], dtype=bool)


SHAPE_LIBRARY = [
    _b(["11", "10"]),
    _b(["111", "100"]),
    _b(["111", "010"]),
    _b(["110", "011", "001"]),
    _b(["111", "100", "100"]),
    _b(["110", "100", "110"]),
    _b(["1111", "1000"]),
    _b(["111", "101", "100"]),
    _b(["1110", "0011"]),
    _b(["111", "001", "011"]),
    _b(["1111", "0100", "0110"]),
    _b(["111", "010", "011", "001"]),
]


for _shape in SHAPE_LIBRARY:
    if len(_connected_components(_shape)) != 1:
        raise AssertionError("library shape must be 4-connected")
    if not _rotationally_asymmetric(_shape):
        raise AssertionError("library shape must have a unique matching rotation")


SHAPE_META = [
    {
        "mask": shape.copy(),
        "area": int(shape.sum()),
        "bbox_key": tuple(sorted(shape.shape)),
        "canon": _canonical_rotation_key(shape),
    }
    for shape in SHAPE_LIBRARY
]


def _choose_hole_shapes(count: int) -> tuple[list[np.ndarray], dict]:
    indices = list(range(len(SHAPE_META)))
    for _ in range(400):
        chosen = random.sample(indices, count)
        metas = [SHAPE_META[i] for i in chosen]
        if len({meta["canon"] for meta in metas}) != count:
            continue
        same_area = any(metas[i]["area"] == metas[j]["area"] for i in range(count) for j in range(i + 1, count))
        same_bbox = any(metas[i]["bbox_key"] == metas[j]["bbox_key"] for i in range(count) for j in range(i + 1, count))
        if same_area or same_bbox:
            return [metas[i]["mask"].copy() for i in range(count)], {"same_area": same_area, "same_bbox": same_bbox}
    raise RuntimeError("failed to choose sufficiently confusable hole shapes")


def _random_colors(count: int, avoid: list[int] | None = None) -> list[int]:
    avoid = avoid or [BG]
    choices = [c for c in PALETTE if c not in set(avoid)]
    return [random.choice(choices) for _ in range(count)]


def _assign_colors(host_count: int) -> tuple[list[int], list[int]]:
    host_colors = _random_colors(host_count)
    sprite_colors = []
    for host_color in host_colors:
        sprite_colors.append(random.choice([c for c in PALETTE if c != host_color]))

    if host_count >= 3 and random.random() < 0.65:
        a, b = random.sample(range(host_count), 2)
        choices = [c for c in PALETTE if c != host_colors[a] and c != host_colors[b]]
        shared = random.choice(choices)
        sprite_colors[a] = shared
        sprite_colors[b] = shared
    if host_count >= 3 and random.random() < 0.45:
        a, b = random.sample(range(host_count), 2)
        shared = random.choice(PALETTE)
        host_colors[a] = shared
        host_colors[b] = shared
        for idx in (a, b):
            if sprite_colors[idx] == host_colors[idx]:
                sprite_colors[idx] = random.choice([c for c in PALETTE if c != host_colors[idx]])
    return host_colors, sprite_colors


def _build_host_mask(hole_mask: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    hh, hw = hole_mask.shape
    top = random.randint(1, 3)
    bottom = random.randint(1, 3)
    left = random.randint(1, 3)
    right = random.randint(1, 3)
    outer = np.ones((hh + top + bottom, hw + left + right), dtype=bool)
    hole_coords: list[tuple[int, int]] = []
    for r in range(hh):
        for c in range(hw):
            if hole_mask[r, c]:
                outer[top + r, left + c] = False
                hole_coords.append((top + r, left + c))

    corners = [
        ("tl", top, left),
        ("tr", top, right),
        ("bl", bottom, left),
        ("br", bottom, right),
    ]
    random.shuffle(corners)
    for corner, max_h, max_w in corners[: random.randint(0, 2)]:
        if max_h <= 0 or max_w <= 0:
            continue
        notch_h = random.randint(1, max_h)
        notch_w = random.randint(1, max_w)
        if corner == "tl":
            outer[:notch_h, :notch_w] = False
        elif corner == "tr":
            outer[:notch_h, -notch_w:] = False
        elif corner == "bl":
            outer[-notch_h:, :notch_w] = False
        else:
            outer[-notch_h:, -notch_w:] = False
    return outer, hole_coords


def _paint_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    h, w = patch.shape
    for r in range(h):
        for c in range(w):
            if patch[r, c] != BG:
                grid[top + r, left + c] = int(patch[r, c])
                coords.append((top + r, left + c))
    return coords


def _can_place(mask: np.ndarray, top: int, left: int, grid_shape: tuple[int, int], occupied: list[tuple[int, int]]) -> bool:
    h, w = mask.shape
    H, W = grid_shape
    if top < 1 or left < 1 or top + h >= H or left + w >= W:
        return False
    new_coords = [(top + r, left + c) for r in range(h) for c in range(w) if mask[r, c]]
    for nr, nc in new_coords:
        for orr, occ in occupied:
            if max(abs(nr - orr), abs(nc - occ)) <= PLACEMENT_GAP:
                return False
    return True


def _place_component(grid: np.ndarray, patch: np.ndarray, occupied: list[tuple[int, int]]) -> tuple[int, int, list[tuple[int, int]]] | None:
    h, w = patch.shape
    mask = patch != BG
    candidates = [(r, c) for r in range(1, grid.shape[0] - h) for c in range(1, grid.shape[1] - w)]
    random.shuffle(candidates)
    for top, left in candidates:
        if not _can_place(mask, top, left, grid.shape, occupied):
            continue
        coords = _paint_patch(grid, patch, top, left)
        occupied.extend(coords)
        return top, left, coords
    return None


def _rotation_requirements(count: int) -> list[int]:
    needed = [random.randint(0, 3) for _ in range(count)]
    if all(k == 0 for k in needed):
        needed[random.randrange(count)] = random.randint(1, 3)
    if count >= 3 and len(set(needed)) == 1:
        needed[random.randrange(count)] = (needed[0] + random.randint(1, 3)) % 4
    return needed


def _build_scene() -> tuple[np.ndarray, np.ndarray, dict] | None:
    height = random.randint(20, 30)
    width = random.randint(20, 30)
    host_count = random.randint(2, 4)

    hole_shapes, confusability = _choose_hole_shapes(host_count)
    host_colors, sprite_colors = _assign_colors(host_count)
    needed_rotations = _rotation_requirements(host_count)

    grid = np.full((height, width), BG, dtype=int)
    occupied: list[tuple[int, int]] = []
    blueprint_hosts: list[dict] = []
    blueprint_sprites: list[dict] = []

    host_order = list(range(host_count))
    random.shuffle(host_order)
    for idx in host_order:
        host_mask, local_hole_coords = _build_host_mask(hole_shapes[idx])
        patch = np.full(host_mask.shape, BG, dtype=int)
        patch[host_mask] = host_colors[idx]
        placed = _place_component(grid, patch, occupied)
        if placed is None:
            return None
        top, left, _coords = placed
        blueprint_hosts.append(
            {
                "idx": idx,
                "top": top,
                "left": left,
                "hole_coords": [(top + r, left + c) for r, c in local_hole_coords],
                "hole_mask": hole_shapes[idx].copy(),
                "host_color": host_colors[idx],
            }
        )

    sprite_order = list(range(host_count))
    random.shuffle(sprite_order)
    for idx in sprite_order:
        sprite_mask = np.rot90(hole_shapes[idx], (-needed_rotations[idx]) % 4)
        patch = np.full(sprite_mask.shape, BG, dtype=int)
        patch[sprite_mask] = sprite_colors[idx]
        placed = _place_component(grid, patch, occupied)
        if placed is None:
            return None
        top, left, coords = placed
        blueprint_sprites.append(
            {
                "idx": idx,
                "top": top,
                "left": left,
                "coords": coords,
                "mask": sprite_mask.copy(),
                "color": sprite_colors[idx],
                "needed_rotation": needed_rotations[idx],
            }
        )

    expected = grid.copy()
    for sprite in blueprint_sprites:
        rows, cols = zip(*sprite["coords"])
        expected[rows, cols] = BG
    hole_by_idx = {host["idx"]: host for host in blueprint_hosts}
    for sprite in blueprint_sprites:
        host = hole_by_idx[sprite["idx"]]
        rows, cols = zip(*host["hole_coords"])
        expected[rows, cols] = sprite["color"]

    meta = {
        "host_count": host_count,
        "confusability": confusability,
        "needed_rotations": needed_rotations,
        "hole_shapes": [shape.copy() for shape in hole_shapes],
    }
    return grid, expected, meta


def _scene_semantics_ok(inp: np.ndarray, expected: np.ndarray, meta: dict) -> bool:
    try:
        actual = main(inp)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    try:
        hosts, sprites, _bg = _classify_scene(inp)
        matches, needed = _matchings(hosts, sprites)
    except Exception:
        return False

    if len(hosts) != meta["host_count"] or len(sprites) != meta["host_count"]:
        return False
    if len(matches) != meta["host_count"]:
        return False
    if not any(k != 0 for k in needed):
        return False

    hole_masks = [host["hole"]["mask"] for host in hosts]
    same_area = any(int(hole_masks[i].sum()) == int(hole_masks[j].sum()) for i in range(len(hole_masks)) for j in range(i + 1, len(hole_masks)))
    same_bbox = any(tuple(sorted(hole_masks[i].shape)) == tuple(sorted(hole_masks[j].shape)) for i in range(len(hole_masks)) for j in range(i + 1, len(hole_masks)))
    if not (same_area or same_bbox):
        return False

    if np.array_equal(inp, expected):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        built = _build_scene()
        if built is None:
            continue
        inp, expected, meta = built
        if _scene_semantics_ok(inp, expected, meta):
            return inp
    raise RuntimeError("failed to generate a valid negative-space-key sample")
