from __future__ import annotations

import itertools
import random
import sys

import numpy as np

UTILS_DIR = "/home/lixinyu/snap/new/solvers"
if UTILS_DIR not in sys.path:
    sys.path.insert(0, UTILS_DIR)

from arc_utils import bbox_coords, components_mask, get_bg, neighbors4


DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

SPRITE_COLORS = (1, 2, 4)
OBSTACLE = 5
BG = 0

# Canonical settled scaffold.
A_REL = ((0, 1), (1, 1), (2, 0), (2, 1), (2, 2))
B_REL_SMALL = ((0, 0), (1, 0), (2, 0), (3, 0), (3, 1))
B_REL_LARGE = ((0, 0), (1, 0), (2, 0), (2, 1), (3, 0), (3, 1))
C_REL_LARGE = ((0, 0), (1, 0), (1, 1), (1, 2), (1, 3), (2, 3))
C_REL_SMALL = ((0, 0), (1, 0), (1, 1), (1, 2), (0, 2))

FINAL_TOPS = {
    "A": (3, 4),
    "B": (2, 10),
    "C": (4, 12),
}

OBSTACLES_CANONICAL = {
    (6, 3), (7, 3), (6, 6), (7, 6), (8, 4), (8, 5),
    (6, 10), (7, 10), (6, 14), (7, 14), (8, 11), (8, 12), (8, 13),
}

GEN_ATTEMPTS = 400


def _bbox(coords: set[tuple[int, int]] | list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    arr = np.array(sorted(coords), dtype=int)
    return tuple(map(int, bbox_coords(arr)))


def _is_connected(coords: set[tuple[int, int]]) -> bool:
    start = next(iter(coords))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for nr, nc in neighbors4(r, c):
            if (nr, nc) in coords and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(coords)


def _candidate_direction(marker: tuple[int, int], body: set[tuple[int, int]]) -> str | None:
    r0, c0, r1, c1 = _bbox(body)
    mr, mc = marker
    if mr == r0 - 1 and c0 <= mc <= c1:
        return "up"
    if mr == r1 + 1 and c0 <= mc <= c1:
        return "down"
    if mc == c0 - 1 and r0 <= mr <= r1:
        return "left"
    if mc == c1 + 1 and r0 <= mr <= r1:
        return "right"
    return None


def _component_candidates(coords: np.ndarray, color: int) -> list[dict]:
    full = {tuple(map(int, rc)) for rc in coords}
    out: list[dict] = []
    for marker in sorted(full):
        body = full - {marker}
        if not (5 <= len(body) <= 8):
            continue
        deg = sum(((marker[0] + dr, marker[1] + dc) in body) for dr, dc in DIRS.values())
        if deg != 1:
            continue
        if not _is_connected(body):
            continue
        direction = _candidate_direction(marker, body)
        if direction is None:
            continue
        out.append(
            {
                "color": int(color),
                "marker": marker,
                "direction": direction,
                "body": frozenset(body),
                "area": len(body),
                "bbox": _bbox(body),
            }
        )
    return out


def _slide_body(
    body: frozenset[tuple[int, int]],
    direction: str,
    blockers: set[tuple[int, int]],
    shape: tuple[int, int],
) -> tuple[set[tuple[int, int]], int]:
    H, W = map(int, shape)
    dr, dc = DIRS[direction]
    cur = set(body)
    steps = 0
    while True:
        nxt = {(r + dr, c + dc) for r, c in cur}
        if any(r < 0 or r >= H or c < 0 or c >= W for r, c in nxt):
            return cur, steps
        if nxt & blockers:
            return cur, steps
        cur = nxt
        steps += 1


def _render_scene(
    shape: tuple[int, int],
    obstacles: set[tuple[int, int]],
    sprites: list[dict],
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for r, c in obstacles:
        out[r, c] = OBSTACLE
    for sprite in sprites:
        for r, c in sprite["coords"]:
            out[r, c] = int(sprite["color"])
    return out


def _enclosed_zero_components(grid: np.ndarray) -> list[np.ndarray]:
    H, W = map(int, grid.shape)
    enclosed: list[np.ndarray] = []
    for comp in components_mask(grid == BG, connectivity=4):
        if comp.size == 0:
            continue
        rows = comp[:, 0]
        cols = comp[:, 1]
        if np.any(rows == 0) or np.any(rows == H - 1) or np.any(cols == 0) or np.any(cols == W - 1):
            continue
        enclosed.append(comp)
    return enclosed


def _fill_cavities(base: np.ndarray, sprites: list[dict]) -> np.ndarray:
    out = base.copy()
    sprite_lookup: dict[tuple[int, int], int] = {}
    for idx, sprite in enumerate(sprites):
        for coord in sprite["coords"]:
            sprite_lookup[coord] = idx

    for comp in _enclosed_zero_components(base):
        touching: set[int] = set()
        for r, c in comp:
            rr = int(r)
            cc = int(c)
            for nr, nc in neighbors4(rr, cc):
                idx = sprite_lookup.get((int(nr), int(nc)))
                if idx is not None:
                    touching.add(idx)

        if not touching:
            continue

        winner_idx = min(
            touching,
            key=lambda idx: (
                int(sprites[idx]["area"]),
                int(sprites[idx]["bbox"][0]),
                int(sprites[idx]["bbox"][1]),
            ),
        )
        out[comp[:, 0], comp[:, 1]] = int(sprites[winner_idx]["color"])
    return out


def _infer_setup(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert int(get_bg(g)) == BG

    obstacles = {tuple(map(int, rc)) for rc in np.argwhere(g == OBSTACLE)}
    colors = [int(c) for c in np.unique(g) if int(c) not in (BG, OBSTACLE)]
    assert sorted(colors) == list(SPRITE_COLORS), "expected exactly blue/red/yellow sprite colors"

    candidate_lists: list[list[dict]] = []
    for color in SPRITE_COLORS:
        comps = components_mask(g == int(color), connectivity=4)
        assert len(comps) == 1, "each sprite color must form one connected sprite+marker component"
        candidates = _component_candidates(comps[0], int(color))
        assert candidates, "failed to identify any plausible marker/body split"
        candidate_lists.append(candidates)

    valid: list[dict] = []
    for choice_tuple in itertools.product(*candidate_lists):
        items = [dict(item) for item in choice_tuple]
        if len({item["direction"] for item in items}) == 1:
            continue
        items.sort(key=lambda item: (int(item["bbox"][0]), int(item["bbox"][1])))

        blockers = set(obstacles)
        stopped: list[dict] = []
        ok = True
        for item in items:
            final_coords, steps = _slide_body(item["body"], item["direction"], blockers, g.shape)
            if steps < 2:
                ok = False
                break
            blockers |= final_coords
            stopped.append(
                {
                    "color": int(item["color"]),
                    "coords": final_coords,
                    "area": int(item["area"]),
                    "bbox": _bbox(final_coords),
                    "orig_bbox": item["bbox"],
                    "direction": item["direction"],
                    "steps": int(steps),
                }
            )
        if not ok:
            continue

        base = _render_scene(g.shape, obstacles, stopped)
        cavities = _enclosed_zero_components(base)
        if len(cavities) < 2:
            continue

        valid.append({"obstacles": obstacles, "sprites": stopped, "base": base})

    assert len(valid) == 1, f"expected a unique valid sprite interpretation, found {len(valid)}"
    return valid[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each marked sprite until collision, then fill the enclosed cavities created by the stopped sprites.

    Input structure:
    - A black background contains three irregular 4-connected sprites in blue, red, and yellow.
    - Each sprite is fused to exactly one same-color single-pixel marker that sits just outside one cardinal side.
    - Grey obstacle pixels are scattered away from the initial sprites and remain fixed.

    Rule:
    - For each color component, treat the detachable singleton on one bbox side as the direction marker and the remaining pixels as the sprite body.
    - Process sprite bodies in ascending order of their original body-bbox top-left corner, sliding each body in its marked direction until the next step would hit a grey obstacle, the grid boundary, or an already stopped sprite.
    - Erase all markers and original sprite locations, keep only the stopped sprites, then find every black 4-connected region that cannot reach the grid edge.
    - Fill each enclosed region with the color of the unique touching sprite, or with the smallest touching sprite (breaking size ties by the upper-left final bbox) when multiple sprites touch its boundary; regions touching no sprite stay black.

    Key insight:
    - The markers only tell you how the sprites move; the actual fill targets are new cavities that appear only after the ordered collision stops are resolved.
    """
    setup = _infer_setup(np.asarray(grid, dtype=int))
    return _fill_cavities(setup["base"], setup["sprites"])


def _place_rel(top: tuple[int, int], rel: tuple[tuple[int, int], ...]) -> list[tuple[int, int]]:
    tr, tc = top
    return [(int(tr + r), int(tc + c)) for r, c in rel]


def _mode_shapes(mode: int) -> dict[str, tuple[tuple[int, int], ...]]:
    if mode == 0:
        return {"A": A_REL, "B": B_REL_SMALL, "C": C_REL_LARGE}
    if mode == 1:
        return {"A": A_REL, "B": B_REL_SMALL, "C": C_REL_SMALL}
    return {"A": A_REL, "B": B_REL_LARGE, "C": C_REL_SMALL}


def _apply_transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.rot90(arr, int(k))
    return np.fliplr(out) if flip else out


def _transform_direction(direction: str, k: int, flip: bool) -> str:
    patch = np.zeros((3, 3), dtype=int)
    patch[1, 1] = 1
    dr, dc = DIRS[direction]
    patch[1 + dr, 1 + dc] = 2
    transformed = _apply_transform(patch, k, flip)
    cr, cc = map(int, np.argwhere(transformed == 1)[0])
    tr, tc = map(int, np.argwhere(transformed == 2)[0])
    delta = (tr - cr, tc - cc)
    for name, vec in DIRS.items():
        if vec == delta:
            return name
    raise AssertionError("invalid transformed direction")


def _transform_component_shape(
    body_rel: tuple[tuple[int, int], ...],
    marker_rel: tuple[int, int],
    k: int,
    flip: bool,
) -> tuple[tuple[tuple[int, int], ...], tuple[int, int]]:
    all_coords = list(body_rel) + [marker_rel]
    rows = [r for r, _c in all_coords]
    cols = [c for _r, c in all_coords]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)

    patch = np.zeros((max_r - min_r + 1, max_c - min_c + 1), dtype=int)
    for r, c in body_rel:
        patch[r - min_r, c - min_c] = 1
    patch[marker_rel[0] - min_r, marker_rel[1] - min_c] = 2

    transformed = _apply_transform(patch, k, flip)
    body = {tuple(map(int, rc)) for rc in np.argwhere(transformed == 1)}
    marker = tuple(map(int, np.argwhere(transformed == 2)[0]))
    r0, c0, _r1, _c1 = _bbox(body)
    body_norm = tuple(sorted((r - r0, c - c0) for r, c in body))
    marker_norm = (int(marker[0] - r0), int(marker[1] - c0))
    return body_norm, marker_norm


def _build_transformed_final(
    shapes: dict[str, tuple[tuple[int, int], ...]],
    k: int,
    flip: bool,
) -> tuple[set[tuple[int, int]], dict[str, set[tuple[int, int]]]]:
    labels = {"A": 11, "B": 12, "C": 13}
    role_coords = {name: _place_rel(FINAL_TOPS[name], rel) for name, rel in shapes.items()}
    all_coords = set(OBSTACLES_CANONICAL)
    for coords in role_coords.values():
        all_coords.update(coords)
    rows = [r for r, _c in all_coords]
    cols = [c for _r, c in all_coords]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    patch = np.zeros((max_r - min_r + 1, max_c - min_c + 1), dtype=int)
    for r, c in OBSTACLES_CANONICAL:
        patch[r - min_r, c - min_c] = OBSTACLE
    for role, coords in role_coords.items():
        for r, c in coords:
            patch[r - min_r, c - min_c] = labels[role]

    transformed = _apply_transform(patch, k, flip)
    obstacles = {tuple(map(int, rc)) for rc in np.argwhere(transformed == OBSTACLE)}
    final_bodies = {
        role: {tuple(map(int, rc)) for rc in np.argwhere(transformed == label)}
        for role, label in labels.items()
    }
    return obstacles, final_bodies


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    mode = random.choice([0, 1, 2])
    k = random.randint(0, 3)
    flip = random.choice([False, True])
    colors = dict(zip(("A", "B", "C"), random.sample(list(SPRITE_COLORS), 3)))
    dists = {"A": random.randint(2, 4), "B": random.randint(2, 4), "C": random.randint(2, 4)}

    shapes = _mode_shapes(mode)
    marker_rels = {"A": (3, 1), "B": (4, 0), "C": (0, -1)}
    directions = {"A": "down", "B": "down", "C": "left"}

    obstacles_local, final_bodies_local = _build_transformed_final(shapes, k, flip)
    transformed_shapes = {
        role: _transform_component_shape(shapes[role], marker_rels[role], k, flip)
        for role in ("A", "B", "C")
    }
    transformed_dirs = {
        role: _transform_direction(directions[role], k, flip)
        for role in ("A", "B", "C")
    }

    input_bodies_local: dict[str, set[tuple[int, int]]] = {}
    markers_local: dict[str, tuple[int, int]] = {}
    for role in ("A", "B", "C"):
        dr, dc = DIRS[transformed_dirs[role]]
        d = dists[role]
        input_body = {(r - dr * d, c - dc * d) for r, c in final_bodies_local[role]}
        input_bodies_local[role] = input_body

        body_rel_t, marker_rel_t = transformed_shapes[role]
        if tuple(sorted((r, c) for r, c in body_rel_t)) != tuple(sorted((r - _bbox(input_body)[0], c - _bbox(input_body)[1]) for r, c in input_body)):
            return None
        r0, c0, _r1, _c1 = _bbox(input_body)
        markers_local[role] = (int(r0 + marker_rel_t[0]), int(c0 + marker_rel_t[1]))

    order = sorted(("A", "B", "C"), key=lambda role: (_bbox(input_bodies_local[role])[0], _bbox(input_bodies_local[role])[1]))
    if order.index("B") > order.index("C"):
        return None

    all_input_coords = set(obstacles_local)
    for role in ("A", "B", "C"):
        all_input_coords.update(input_bodies_local[role])
        all_input_coords.add(markers_local[role])

    occupied_input = {}
    for role in ("A", "B", "C"):
        for coord in input_bodies_local[role]:
            if coord in occupied_input or coord in obstacles_local:
                return None
            occupied_input[coord] = role
        if markers_local[role] in occupied_input or markers_local[role] in obstacles_local:
            return None
        occupied_input[markers_local[role]] = role

    for role in ("A", "B", "C"):
        for r, c in input_bodies_local[role]:
            for nr, nc in neighbors4(r, c):
                if (nr, nc) in obstacles_local:
                    return None

    rows = [r for r, _c in all_input_coords]
    cols = [c for _r, c in all_input_coords]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)

    pad_top = random.randint(1, 4)
    pad_bottom = random.randint(1, 4)
    pad_left = random.randint(1, 4)
    pad_right = random.randint(1, 4)
    H = (max_r - min_r + 1) + pad_top + pad_bottom
    W = (max_c - min_c + 1) + pad_left + pad_right
    if H > 30 or W > 30:
        return None

    shift_r = pad_top - min_r
    shift_c = pad_left - min_c

    def shift_set(coords: set[tuple[int, int]]) -> set[tuple[int, int]]:
        return {(int(r + shift_r), int(c + shift_c)) for r, c in coords}

    def shift_coord(coord: tuple[int, int]) -> tuple[int, int]:
        return int(coord[0] + shift_r), int(coord[1] + shift_c)

    obstacles = shift_set(obstacles_local)
    final_sprites = []
    for role in ("A", "B", "C"):
        coords = shift_set(final_bodies_local[role])
        final_sprites.append(
            {
                "role": role,
                "color": int(colors[role]),
                "coords": coords,
                "area": len(coords),
                "bbox": _bbox(coords),
                "direction": transformed_dirs[role],
            }
        )

    inp = np.zeros((H, W), dtype=int)
    for r, c in obstacles:
        inp[r, c] = OBSTACLE
    for role in ("A", "B", "C"):
        for r, c in shift_set(input_bodies_local[role]):
            inp[r, c] = int(colors[role])
        mr, mc = shift_coord(markers_local[role])
        inp[mr, mc] = int(colors[role])

    final_base = _render_scene((H, W), obstacles, final_sprites)
    out = _fill_cavities(final_base, final_sprites)

    if len(_enclosed_zero_components(inp)) != 0:
        return None
    final_cavities = _enclosed_zero_components(final_base)
    if len(final_cavities) != 2:
        return None
    if len(set(transformed_dirs.values())) == 1:
        return None

    touch_sets: list[set[str]] = []
    lookup = {}
    role_to_sprite = {sprite["role"]: sprite for sprite in final_sprites}
    for idx, sprite in enumerate(final_sprites):
        for coord in sprite["coords"]:
            lookup[coord] = idx
    for cavity in final_cavities:
        touching: set[str] = set()
        for r, c in cavity:
            for nr, nc in neighbors4(int(r), int(c)):
                idx = lookup.get((int(nr), int(nc)))
                if idx is not None:
                    touching.add(final_sprites[idx]["role"])
        touch_sets.append(touching)
    if {"A"} not in touch_sets:
        return None
    if {"B", "C"} not in touch_sets:
        return None

    if mode == 0 and not (role_to_sprite["B"]["area"] < role_to_sprite["C"]["area"]):
        return None
    if mode == 1 and not (role_to_sprite["B"]["area"] == role_to_sprite["C"]["area"]):
        return None
    if mode == 2 and not (role_to_sprite["C"]["area"] < role_to_sprite["B"]["area"]):
        return None

    for role in ("A", "B"):
        dr, dc = DIRS[transformed_dirs[role]]
        next_cells = {(r + dr, c + dc) for r, c in role_to_sprite[role]["coords"]}
        if not (next_cells & obstacles):
            return None

    dr, dc = DIRS[transformed_dirs["C"]]
    next_cells = {(r + dr, c + dc) for r, c in role_to_sprite["C"]["coords"]}
    if not (next_cells & role_to_sprite["B"]["coords"]):
        return None

    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        case = _build_case()
        if case is None:
            continue
        inp, expected = case
        produced = main(inp)
        if not np.array_equal(produced, expected):
            continue
        if len(_enclosed_zero_components(inp)) != 0:
            continue
        return inp
    raise RuntimeError("failed to generate a valid sliding-cavity ARC input")
