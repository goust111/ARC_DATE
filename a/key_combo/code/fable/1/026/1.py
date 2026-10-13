from __future__ import annotations

from collections import deque

import numpy as np


BG = 0
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
PALETTE = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int)
MAX_GEN_ATTEMPTS = 300


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    return bbox_coords(coords)


def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def label_mask(mask: np.ndarray) -> tuple[np.ndarray, int]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    comp_id = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            comp_id += 1
            labels[r, c] = comp_id
            q: deque[tuple[int, int]] = deque([(r, c)])
            while q:
                rr, cc = q.popleft()
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = comp_id
                        q.append((nr, nc))
    return labels, comp_id


def dedup_preserve(seq: list[int]) -> list[int]:
    out: list[int] = []
    for value in seq:
        if value not in out:
            out.append(int(value))
    return out


def ring_distance(shape: tuple[int, int]) -> np.ndarray:
    h, w = map(int, shape)
    rr, cc = np.indices((h, w))
    return np.minimum.reduce([rr, cc, h - 1 - rr, w - 1 - cc])


def extract_ring_layers(patch: np.ndarray) -> list[int]:
    dist = ring_distance(patch.shape)
    n_layers = int(dist.max()) + 1
    layers: list[int] = []
    for layer in range(n_layers):
        colors = np.unique(patch[dist == layer])
        assert colors.size == 1, "each ring layer must be monochrome"
        layers.append(int(colors[0]))
    return layers


def count_enclosed_holes(obj_mask: np.ndarray) -> int:
    bg_mask = ~np.asarray(obj_mask, dtype=bool)
    labels, n_comp = label_mask(bg_mask)
    hole_count = 0
    h, w = obj_mask.shape
    for comp_id in range(1, n_comp + 1):
        coords = np.argwhere(labels == comp_id)
        touches_edge = np.any(
            (coords[:, 0] == 0) | (coords[:, 0] == h - 1) | (coords[:, 1] == 0) | (coords[:, 1] == w - 1)
        )
        if not touches_edge:
            hole_count += 1
    return int(hole_count)


def rotate_ring_colors(patch: np.ndarray, order: list[int], steps: int) -> np.ndarray:
    steps = int(steps) % len(order)
    mapping = {color: order[(idx - steps) % len(order)] for idx, color in enumerate(order)}
    out = np.array(patch, copy=True)
    src = np.array(patch, copy=True)
    for old_color, new_color in mapping.items():
        out[src == int(old_color)] = int(new_color)
    return out


def parse_objects(grid: np.ndarray) -> tuple[list[dict], dict[int, dict]]:
    labels, n_comp = label_mask(grid != BG)
    systems: list[dict] = []
    keys_by_color: dict[int, dict] = {}

    for comp_id in range(1, n_comp + 1):
        comp_mask = labels == comp_id
        box = bbox(comp_mask)
        assert box is not None
        r0, c0, r1, c1 = box
        local_grid = crop_bbox(grid, box)
        local_mask = comp_mask[r0 : r1 + 1, c0 : c1 + 1]
        h, w = local_grid.shape

        if int(local_mask.sum()) == int(h * w) and np.all(local_grid != BG):
            layers = extract_ring_layers(local_grid)
            order = dedup_preserve(layers)
            systems.append(
                {
                    "bbox": box,
                    "patch": local_grid,
                    "outer_color": int(layers[0]),
                    "layers": layers,
                    "order": order,
                }
            )
            continue

        colors = np.unique(local_grid[local_mask])
        assert colors.size == 1, "each key must be monochrome"
        key_color = int(colors[0])
        hole_count = count_enclosed_holes(local_mask)
        assert hole_count >= 1, "keys must carry at least one enclosed hole"
        assert key_color not in keys_by_color, "key colors must be unique"
        keys_by_color[key_color] = {
            "bbox": box,
            "mask": local_mask,
            "hole_count": hole_count,
        }

    assert systems, "expected at least one ring system"
    assert keys_by_color, "expected at least one key"
    return systems, keys_by_color


def main(grid: np.ndarray) -> np.ndarray:
    """Use each key's enclosed-hole count to rotate the paired concentric ring system's color cycle.

    Input structure:
    - Black background surrounds several disconnected solid rectangular blocks made of one-cell-thick concentric monochrome rings.
    - Separate square key sprites use a single non-black color and contain one or more enclosed black hole components.
    - Each key color matches exactly one ring system's outermost ring color, even if that color also appears deeper inside other systems.

    Rule:
    - Split non-black cells into connected objects; solid filled rectangles are ring systems and holed monochrome squares are keys.
    - For each ring system, read ring colors from outside to inside and keep only first appearances to form a cyclic color order.
    - Pair a key to the unique system whose outer ring shares the key color, rotate that system's color order outward by the key's hole count, recolor the system accordingly, and erase every key to background.

    Key insight:
    - The key is only a parameter carrier: its color addresses the lock by outermost ring color, and its enclosed-hole count sets the cycle shift on the system's deduplicated nesting order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    systems, keys_by_color = parse_objects(g)
    outer_colors = [int(system["outer_color"]) for system in systems]
    assert len(set(outer_colors)) == len(outer_colors), "outer ring colors must be unique"
    assert set(outer_colors) == set(keys_by_color), "key colors must match outer ring colors exactly"

    out = np.array(g, copy=True)
    for key in keys_by_color.values():
        r0, c0, r1, c1 = key["bbox"]
        local_mask = key["mask"]
        out[r0 : r1 + 1, c0 : c1 + 1][local_mask] = BG

    for system in systems:
        r0, c0, r1, c1 = system["bbox"]
        shift = int(keys_by_color[int(system["outer_color"])]["hole_count"])
        recolored = rotate_ring_colors(system["patch"], list(system["order"]), shift)
        out[r0 : r1 + 1, c0 : c1 + 1] = recolored

    return out


def render_ring_patch(shape: tuple[int, int], layers: list[int]) -> np.ndarray:
    patch = np.full(shape, BG, dtype=int)
    dist = ring_distance(shape)
    assert int(dist.max()) + 1 == len(layers)
    for layer_idx, color in enumerate(layers):
        patch[dist == layer_idx] = int(color)
    return patch


def build_cycle(
    rng: np.random.Generator,
    outer_color: int,
    all_outer_colors: list[int],
    cycle_len: int,
    force_borrow: bool,
) -> list[int]:
    assert cycle_len >= 2
    inner: list[int] = []
    if force_borrow:
        borrow_pool = [color for color in all_outer_colors if color != outer_color]
        assert borrow_pool
        inner.append(int(rng.choice(np.array(borrow_pool, dtype=int))))

    while len(inner) < cycle_len - 1:
        if rng.random() < 0.6:
            pool = [color for color in all_outer_colors if color != outer_color and color not in inner]
        else:
            pool = []
        if not pool:
            pool = [int(color) for color in PALETTE if int(color) != outer_color and int(color) not in inner]
        inner.append(int(rng.choice(np.array(pool, dtype=int))))

    return [int(outer_color)] + inner


def build_system_spec(
    rng: np.random.Generator,
    outer_color: int,
    all_outer_colors: list[int],
    force_borrow: bool,
    force_repeat: bool,
    force_wrap: bool,
) -> dict:
    cycle_len = int(rng.integers(2, 5))
    cycle = build_cycle(rng, int(outer_color), all_outer_colors, cycle_len, force_borrow)

    extra_rings = 0
    if force_repeat or rng.random() < 0.75:
        extra_rings = 1 + int(rng.integers(0, 2))
    ring_count = min(5, cycle_len + extra_rings)
    ring_count = max(ring_count, cycle_len + (1 if force_repeat else 0))

    base_dim = 2 * ring_count - int(rng.integers(0, 2))
    long_dim = int(rng.integers(base_dim, min(12, base_dim + 3) + 1))
    if rng.random() < 0.5:
        shape = (base_dim, long_dim)
    else:
        shape = (long_dim, base_dim)

    layers = list(cycle)
    while len(layers) < ring_count:
        layers.append(int(rng.choice(np.array(cycle, dtype=int))))

    order = dedup_preserve(layers)
    if force_wrap:
        hole_count = min(5, len(order) + int(rng.integers(1, 3)))
    else:
        hole_count = int(rng.integers(1, min(5, len(order) + 2) + 1))

    input_patch = render_ring_patch(shape, layers)
    output_patch = rotate_ring_colors(input_patch, order, hole_count)

    return {
        "kind": "system",
        "outer_color": int(outer_color),
        "shape": shape,
        "layers": layers,
        "order": order,
        "hole_count": int(hole_count),
        "input_patch": input_patch,
        "output_patch": output_patch,
        "uses_borrowed_outer_color": any(color in all_outer_colors and color != outer_color for color in order[1:]),
        "has_repeat": len(set(layers)) < len(layers),
        "wraps_cycle": int(hole_count) > len(order),
    }


def build_key_patch(
    rng: np.random.Generator,
    color: int,
    hole_count: int,
    force_large_hole: bool,
) -> dict | None:
    min_side = max(5, 2 * int(np.ceil(np.sqrt(hole_count))) + 3)
    max_side = min(9, min_side + 2)
    side = int(rng.integers(min_side, max_side + 1))
    patch = np.full((side, side), int(color), dtype=int)
    blocked = np.zeros((side, side), dtype=bool)
    hole_specs: list[tuple[int, int, int, int]] = []

    for hole_idx in range(hole_count):
        shapes = [(1, 1)] * 4 + [(1, 2), (2, 1), (2, 2)]
        if force_large_hole and hole_idx == 0:
            shapes = [(1, 2), (2, 1), (2, 2)]

        candidates: list[tuple[int, int, int, int]] = []
        for hh, ww in shapes:
            for r0 in range(1, side - hh):
                for c0 in range(1, side - ww):
                    r1 = r0 + hh - 1
                    c1 = c0 + ww - 1
                    rr0 = max(0, r0 - 1)
                    cc0 = max(0, c0 - 1)
                    rr1 = min(side, r1 + 2)
                    cc1 = min(side, c1 + 2)
                    if blocked[rr0:rr1, cc0:cc1].any():
                        continue
                    candidates.append((r0, c0, hh, ww))

        if not candidates:
            return None

        r0, c0, hh, ww = candidates[int(rng.integers(len(candidates)))]
        patch[r0 : r0 + hh, c0 : c0 + ww] = BG
        hole_specs.append((r0, c0, hh, ww))
        rr0 = max(0, r0 - 1)
        cc0 = max(0, c0 - 1)
        rr1 = min(side, r0 + hh + 1)
        cc1 = min(side, c0 + ww + 1)
        blocked[rr0:rr1, cc0:cc1] = True

    assert count_enclosed_holes(patch == int(color)) == int(hole_count)
    return {
        "kind": "key",
        "color": int(color),
        "shape": patch.shape,
        "patch": patch,
        "hole_specs": hole_specs,
        "has_large_hole": any(hh * ww > 1 for _r0, _c0, hh, ww in hole_specs),
    }


def fits(occupied: np.ndarray, r0: int, c0: int, shape: tuple[int, int], pad: int = 1) -> bool:
    h, w = map(int, shape)
    h0, w0 = occupied.shape
    if r0 < 0 or c0 < 0 or r0 + h > h0 or c0 + w > w0:
        return False
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h0, r0 + h + pad)
    cc1 = min(w0, c0 + w + pad)
    return not occupied[rr0:rr1, cc0:cc1].any()


def place_objects(
    rng: np.random.Generator,
    canvas_shape: tuple[int, int],
    systems: list[dict],
    keys: list[dict],
) -> dict | None:
    h, w = map(int, canvas_shape)
    occupied = np.zeros((h, w), dtype=bool)
    placements: dict[tuple[str, int], tuple[int, int]] = {}
    items: list[tuple[str, int, tuple[int, int], int]] = []

    for idx, system in enumerate(systems):
        sh, sw = system["input_patch"].shape
        items.append(("system", idx, (sh, sw), int(sh * sw)))
    for idx, key in enumerate(keys):
        kh, kw = key["patch"].shape
        items.append(("key", idx, (kh, kw), int(kh * kw)))

    items.sort(key=lambda item: item[3], reverse=True)

    for kind, idx, shape, _area in items:
        placed = False
        hh, ww = shape
        for _ in range(300):
            r0 = int(rng.integers(0, h - hh + 1))
            c0 = int(rng.integers(0, w - ww + 1))
            if not fits(occupied, r0, c0, shape, pad=1):
                continue
            occupied[r0 : r0 + hh, c0 : c0 + ww] = True
            placements[(kind, idx)] = (r0, c0)
            placed = True
            break
        if not placed:
            return None

    return placements


def build_case(rng: np.random.Generator) -> dict | None:
    n_systems = int(rng.integers(2, 4))
    outer_colors = [int(value) for value in rng.choice(PALETTE, size=n_systems, replace=False)]
    force_borrow_idx = int(rng.integers(n_systems))
    force_repeat_idx = int(rng.integers(n_systems))
    force_wrap_idx = int(rng.integers(n_systems))
    force_large_hole_idx = int(rng.integers(n_systems))

    systems = [
        build_system_spec(
            rng,
            outer_color=outer_colors[idx],
            all_outer_colors=outer_colors,
            force_borrow=idx == force_borrow_idx,
            force_repeat=idx == force_repeat_idx,
            force_wrap=idx == force_wrap_idx,
        )
        for idx in range(n_systems)
    ]

    keys: list[dict] = []
    for idx, system in enumerate(systems):
        key = build_key_patch(
            rng,
            color=int(system["outer_color"]),
            hole_count=int(system["hole_count"]),
            force_large_hole=idx == force_large_hole_idx,
        )
        if key is None:
            return None
        keys.append(key)

    total_area = sum(int(system["input_patch"].size) for system in systems) + sum(int(key["patch"].size) for key in keys)
    min_side = max(
        max(max(system["input_patch"].shape) for system in systems),
        max(max(key["patch"].shape) for key in keys),
        int(np.ceil(np.sqrt(total_area * 1.8))),
    )
    canvas_h = int(np.clip(min_side + int(rng.integers(-1, 3)), 20, 30))
    canvas_w = int(np.clip(min_side + int(rng.integers(-2, 4)), 20, 30))
    if canvas_h * canvas_w < int(total_area * 1.35):
        return None

    placements = place_objects(rng, (canvas_h, canvas_w), systems, keys)
    if placements is None:
        return None

    inp = np.full((canvas_h, canvas_w), BG, dtype=int)
    expected = np.full((canvas_h, canvas_w), BG, dtype=int)

    for idx, system in enumerate(systems):
        r0, c0 = placements[("system", idx)]
        patch_in = system["input_patch"]
        patch_out = system["output_patch"]
        sh, sw = patch_in.shape
        inp[r0 : r0 + sh, c0 : c0 + sw] = patch_in
        expected[r0 : r0 + sh, c0 : c0 + sw] = patch_out
        system["pos"] = (r0, c0)

    for idx, key in enumerate(keys):
        r0, c0 = placements[("key", idx)]
        patch = key["patch"]
        kh, kw = patch.shape
        inp[r0 : r0 + kh, c0 : c0 + kw] = patch
        key["pos"] = (r0, c0)

    case = {
        "input": inp,
        "expected": expected,
        "systems": systems,
        "keys": keys,
        "has_borrow": any(system["uses_borrowed_outer_color"] for system in systems),
        "has_repeat": any(system["has_repeat"] for system in systems),
        "has_wrap": any(system["wraps_cycle"] for system in systems),
        "has_large_hole": any(key["has_large_hole"] for key in keys),
    }

    if not (case["has_borrow"] and case["has_repeat"] and case["has_wrap"] and case["has_large_hole"]):
        return None
    if not np.array_equal(main(inp), expected):
        return None

    systems_found, keys_found = parse_objects(inp)
    if len(systems_found) != len(systems) or len(keys_found) != len(keys):
        return None

    return case


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GEN_ATTEMPTS):
        case = build_case(rng)
        if case is not None:
            return case["input"]
    raise RuntimeError(f"failed to generate a valid input after {MAX_GEN_ATTEMPTS} attempts")
