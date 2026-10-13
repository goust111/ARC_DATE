from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
MAX_ATTEMPTS = 400
VECTOR_CHOICES = [
    (-4, 0), (-3, 0), (-2, 3), (-2, 4), (-2, -3), (-2, -4),
    (0, -4), (0, -3), (0, 3), (0, 4),
    (2, -4), (2, -3), (2, 3), (2, 4),
    (3, -2), (3, 0), (3, 2),
    (4, 0),
]


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def component_dicts(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    seen = np.zeros(g.shape, dtype=bool)
    comps: list[dict] = []

    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            if seen[r, c] or int(g[r, c]) == bg:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1] and not seen[nr, nc] and int(g[nr, nc]) != bg:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            c0 = int(arr[:, 1].min())
            r1 = int(arr[:, 0].max())
            c1 = int(arr[:, 1].max())
            sprite = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
            sprite[arr[:, 0] - r0, arr[:, 1] - c0] = g[arr[:, 0], arr[:, 1]]
            comps.append(
                {
                    "coords": arr,
                    "colors": g[arr[:, 0], arr[:, 1]].astype(int),
                    "bbox": (r0, c0, r1, c1),
                    "sprite": sprite,
                }
            )
    return comps


def sprite_key(sprite: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    s = np.asarray(sprite, dtype=int)
    return s.shape, tuple(int(x) for x in s.ravel())


def shifted_coords(coords: np.ndarray, delta: tuple[int, int]) -> np.ndarray:
    dr, dc = delta
    return coords + np.array([[int(dr), int(dc)]], dtype=int)


def in_bounds_all(shape: tuple[int, int], coords: np.ndarray) -> bool:
    return (
        np.all(coords[:, 0] >= 0)
        and np.all(coords[:, 0] < shape[0])
        and np.all(coords[:, 1] >= 0)
        and np.all(coords[:, 1] < shape[1])
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the duplicated sprite pair and use their displacement vector to extend every sprite as a bidirectional translation chain.

    Input structure:
    - The background is black and the foreground consists of separated 4-connected multicolor sprites.
    - Exactly one tight colored sprite crop, including any internal background holes, appears twice with the same orientation.
    - Every other sprite crop appears exactly once.

    Rule:
    - Identify the duplicated sprite pair and read the non-zero translation vector between the two copies.
    - For every original sprite, repeatedly try to place full translated copies at offsets +v, +2v, ... and -v, -2v, ...
      from that sprite's original location.
    - In one direction, stop before the first copy that would leave the grid or overlap any already existing foreground pixel.
      Keep all original pixels and draw every successful copy.

    Key insight:
    - The only repeated sprite is the hidden key: its offset defines the lattice step that all other sprites must follow.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    bg = get_bg(g)

    components = component_dicts(g, bg)
    assert len(components) >= 3

    by_key: dict[tuple[tuple[int, int], tuple[int, ...]], list[dict]] = {}
    for comp in components:
        by_key.setdefault(sprite_key(comp["sprite"]), []).append(comp)

    twins = [group for group in by_key.values() if len(group) == 2]
    assert len(twins) == 1, "expected exactly one duplicated sprite crop"
    twin_pair = sorted(twins[0], key=lambda comp: (comp["bbox"][0], comp["bbox"][1]))
    v = (
        int(twin_pair[1]["bbox"][0] - twin_pair[0]["bbox"][0]),
        int(twin_pair[1]["bbox"][1] - twin_pair[0]["bbox"][1]),
    )
    assert v != (0, 0)

    out = g.copy()
    occupied = out != bg
    ordered = sorted(components, key=lambda comp: (comp["bbox"][0], comp["bbox"][1], comp["bbox"][2], comp["bbox"][3]))
    for comp in ordered:
        for sign in (1, -1):
            step = 1
            while True:
                delta = (sign * step * v[0], sign * step * v[1])
                target = shifted_coords(comp["coords"], delta)
                if not in_bounds_all(out.shape, target):
                    break
                rr = target[:, 0]
                cc = target[:, 1]
                if np.any(occupied[rr, cc]):
                    break
                out[rr, cc] = comp["colors"]
                occupied[rr, cc] = True
                step += 1

    return out


def random_connected_mask(h: int, w: int, area: int) -> np.ndarray:
    for _ in range(80):
        start = (random.randrange(h), random.randrange(w))
        cells = {start}
        while len(cells) < area:
            r, c = random.choice(tuple(cells))
            nbrs = []
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    nbrs.append((nr, nc))
            if not nbrs:
                break
            cells.add(random.choice(nbrs))
        if len(cells) != area:
            continue
        rs = {r for r, _ in cells}
        cs = {c for _, c in cells}
        if len(rs) < 2 or len(cs) < 2:
            continue
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        return mask
    raise RuntimeError("failed to generate connected mask")


def random_sprite(v: tuple[int, int], forbidden: set[tuple[tuple[int, int], tuple[int, ...]]]) -> np.ndarray:
    palette = list(range(1, 10))
    size_options = []
    for h in range(2, 6):
        for w in range(2, 6):
            lo = max(4, h + w - 1)
            hi = min(h * w - 1, 9)
            if lo <= hi:
                size_options.append((h, w, lo, hi))
    for _ in range(160):
        h, w, lo, hi = random.choice(size_options)
        area = random.randint(lo, hi)
        mask = random_connected_mask(h, w, area)
        n_colors = random.randint(2, min(4, area))
        colors = random.sample(palette, n_colors)
        coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
        random.shuffle(coords)
        sprite = np.zeros((h, w), dtype=int)
        for (r, c), color in zip(coords, colors):
            sprite[r, c] = color
        for r, c in coords[n_colors:]:
            sprite[r, c] = random.choice(colors)
        key = sprite_key(sprite)
        if key in forbidden:
            continue
        if not shift_is_separated(np.argwhere(sprite != BG), v):
            continue
        return sprite
    raise RuntimeError("failed to generate unique sprite")


def shift_is_separated(coords: np.ndarray, v: tuple[int, int]) -> bool:
    shifted = shifted_coords(coords, v)
    src = {tuple(map(int, rc)) for rc in coords}
    dst = {tuple(map(int, rc)) for rc in shifted}
    if src & dst:
        return False
    for r, c in dst:
        if (r - 1, c) in src or (r + 1, c) in src or (r, c - 1) in src or (r, c + 1) in src:
            return False
    return True


def slot_bounds(shape: tuple[int, int], rel_coords: np.ndarray, anchor: tuple[int, int], v: tuple[int, int]) -> tuple[int, int]:
    valid: list[int] = []
    for k in range(-30, 31):
        coords = rel_coords + np.array([[anchor[0] + k * v[0], anchor[1] + k * v[1]]], dtype=int)
        if in_bounds_all(shape, coords):
            valid.append(k)
    assert valid
    return min(valid), max(valid)


def lane_cells(
    sprite: np.ndarray,
    anchor: tuple[int, int],
    kmin: int,
    kmax: int,
    v: tuple[int, int],
) -> list[tuple[int, int, int]]:
    rel = np.argwhere(sprite != BG)
    colors = sprite[rel[:, 0], rel[:, 1]].astype(int)
    cells: list[tuple[int, int, int]] = []
    for k in range(kmin, kmax + 1):
        rr = rel[:, 0] + anchor[0] + k * v[0]
        cc = rel[:, 1] + anchor[1] + k * v[1]
        for r, c, color in zip(rr, cc, colors):
            cells.append((int(r), int(c), int(color)))
    return cells


def can_add_cells(cells: list[tuple[int, int, int]], occupied: np.ndarray) -> bool:
    for r, c, _color in cells:
        if occupied[r, c]:
            return False
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < occupied.shape[0] and 0 <= nc < occupied.shape[1] and occupied[nr, nc]:
                return False
    return True


def mode_ok(mode: str, kmin: int, kmax: int) -> bool:
    if mode == "twin":
        return kmin <= -1 and kmax >= 2 and (kmax - kmin + 1) >= 4
    if mode == "center":
        return kmin <= -1 and kmax >= 1 and (kmax - kmin + 1) >= 3
    if mode == "positive":
        return kmin == 0 and kmax >= 2
    if mode == "negative":
        return kmin <= -2 and kmax == 0
    raise AssertionError("unknown lane mode")


def find_lane_placement(
    shape: tuple[int, int],
    sprite: np.ndarray,
    v: tuple[int, int],
    mode: str,
    occupied: np.ndarray,
) -> tuple[tuple[int, int], int, int] | None:
    rel = np.argwhere(sprite != BG)
    h, w = sprite.shape
    candidates: list[tuple[tuple[int, int], int, int]] = []
    for r0 in range(shape[0] - h + 1):
        for c0 in range(shape[1] - w + 1):
            anchor = (r0, c0)
            kmin, kmax = slot_bounds(shape, rel, anchor, v)
            if not mode_ok(mode, kmin, kmax):
                continue
            cells = lane_cells(sprite, anchor, kmin, kmax, v)
            if can_add_cells(cells, occupied):
                candidates.append((anchor, kmin, kmax))
    if not candidates:
        return None
    return random.choice(candidates)


def build_grid(shape: tuple[int, int], lanes: list[dict], *, full_output: bool) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for lane in lanes:
        sprite = lane["sprite"]
        rel = np.argwhere(sprite != BG)
        colors = sprite[rel[:, 0], rel[:, 1]].astype(int)
        slots = range(lane["kmin"], lane["kmax"] + 1) if full_output else lane["seed_slots"]
        for k in slots:
            rr = rel[:, 0] + lane["anchor"][0] + k * lane["v"][0]
            cc = rel[:, 1] + lane["anchor"][1] + k * lane["v"][1]
            grid[rr, cc] = colors
    return grid


def validate_instance(inp: np.ndarray, expected: np.ndarray, v: tuple[int, int]) -> None:
    comps = component_dicts(inp, BG)
    keys = Counter(sprite_key(comp["sprite"]) for comp in comps)
    dup_keys = [key for key, count in keys.items() if count == 2]
    assert len(dup_keys) == 1

    twin_comps = [comp for comp in comps if sprite_key(comp["sprite"]) == dup_keys[0]]
    twin_comps = sorted(twin_comps, key=lambda comp: (comp["bbox"][0], comp["bbox"][1]))
    detected = (
        int(twin_comps[1]["bbox"][0] - twin_comps[0]["bbox"][0]),
        int(twin_comps[1]["bbox"][1] - twin_comps[0]["bbox"][1]),
    )
    assert detected == v or detected == (-v[0], -v[1])

    out = main(inp)
    assert np.array_equal(out, expected)
    assert np.count_nonzero(out) > np.count_nonzero(inp)

    # Require both termination modes to exist in every generated example:
    # the twin pair creates collision stops, and at least one lane must hit a boundary.
    collision_probe = shifted_coords(twin_comps[0]["coords"], detected)
    assert np.any(inp[collision_probe[:, 0], collision_probe[:, 1]] != BG)

    hit_boundary = False
    rel = np.argwhere(expected != BG)
    if rel.size:
        hit_boundary = (
            np.any(rel[:, 0] == 0)
            or np.any(rel[:, 0] == expected.shape[0] - 1)
            or np.any(rel[:, 1] == 0)
            or np.any(rel[:, 1] == expected.shape[1] - 1)
        )
    assert hit_boundary


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        h = random.randint(15, 22)
        w = random.randint(15, 22)
        v = random.choice(VECTOR_CHOICES)

        lane_count = random.randint(3, 4)
        unique_modes = ["center"]
        while len(unique_modes) < lane_count - 1:
            unique_modes.append(random.choice(["center", "positive", "negative"]))
        random.shuffle(unique_modes)
        modes = ["twin", *unique_modes]

        forbidden: set[tuple[tuple[int, int], tuple[int, ...]]] = set()
        sprites: list[np.ndarray] = []
        try:
            for _mode in modes:
                sprite = random_sprite(v, forbidden)
                forbidden.add(sprite_key(sprite))
                sprites.append(sprite)
        except RuntimeError:
            continue

        order = sorted(
            range(lane_count),
            key=lambda idx: (
                0 if modes[idx] == "twin" else 1,
                0 if modes[idx] in {"positive", "negative"} else 1,
                -(sprites[idx].shape[0] * sprites[idx].shape[1]),
            ),
        )

        occupied = np.zeros((h, w), dtype=bool)
        lanes: list[dict | None] = [None] * lane_count
        ok = True
        for idx in order:
            placement = find_lane_placement((h, w), sprites[idx], v, modes[idx], occupied)
            if placement is None:
                ok = False
                break
            anchor, kmin, kmax = placement
            cells = lane_cells(sprites[idx], anchor, kmin, kmax, v)
            for r, c, _color in cells:
                occupied[r, c] = True
            lanes[idx] = {
                "sprite": sprites[idx],
                "anchor": anchor,
                "kmin": kmin,
                "kmax": kmax,
                "v": v,
                "seed_slots": [0, 1] if modes[idx] == "twin" else [0],
                "mode": modes[idx],
            }

        if not ok:
            continue

        lane_list = [lane for lane in lanes if lane is not None]
        if len(lane_list) != lane_count:
            continue
        if not any(lane["mode"] in {"positive", "negative"} for lane in lane_list[1:]):
            continue

        inp = build_grid((h, w), lane_list, full_output=False)
        expected = build_grid((h, w), lane_list, full_output=True)
        try:
            validate_instance(inp, expected, v)
        except AssertionError:
            continue
        return inp

    raise RuntimeError("failed to generate a valid input")
