from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
PATH = 5
SPRITE_PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GEN_ATTEMPTS = 500


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _normalize(coords: set[tuple[int, int]] | list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    pts = list(coords)
    r0 = min(r for r, _ in pts)
    c0 = min(c for _, c in pts)
    return tuple(sorted((r - r0, c - c0) for r, c in pts))


def _bbox(coords: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    pts = list(coords)
    return (
        min(r for r, _ in pts),
        min(c for _, c in pts),
        max(r for r, _ in pts),
        max(c for _, c in pts),
    )


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _find_elbow(path_coords: list[tuple[int, int]]) -> tuple[int, int]:
    cells = set(path_coords)
    elbows = []
    for r, c in path_coords:
        dirs = []
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in cells:
                dirs.append((nr - r, nc - c))
        has_h = any(dc != 0 for _dr, dc in dirs)
        has_v = any(dr != 0 for dr, _dc in dirs)
        if has_h and has_v:
            elbows.append((r, c))
    assert len(elbows) == 1, "expected one unique elbow"
    er, ec = elbows[0]
    assert all(r == er or c == ec for r, c in path_coords), "L path must lie on elbow row/col"
    return er, ec


def _path_kind(path_coords: list[tuple[int, int]]) -> tuple[str, tuple[int, int], tuple[int, int]]:
    rows = {r for r, _ in path_coords}
    cols = {c for _, c in path_coords}
    if len(rows) == 1:
        row = next(iter(rows))
        c0 = min(c for _, c in path_coords)
        c1 = max(c for _, c in path_coords)
        return "horizontal", (row, 0), (2 * row, c0 + c1)
    if len(cols) == 1:
        col = next(iter(cols))
        r0 = min(r for r, _ in path_coords)
        r1 = max(r for r, _ in path_coords)
        return "vertical", (0, col), (r0 + r1, 2 * col)
    elbow = _find_elbow(path_coords)
    return "l", elbow, (2 * elbow[0], 2 * elbow[1])


def _transform_coords(
    coords: list[tuple[int, int]],
    kind: str,
    ref: tuple[int, int],
) -> list[tuple[int, int]]:
    if kind == "horizontal":
        axis_r = ref[0]
        return [(2 * axis_r - r, c) for r, c in coords]
    if kind == "vertical":
        axis_c = ref[1]
        return [(r, 2 * axis_c - c) for r, c in coords]
    center_r, center_c = ref
    return [(2 * center_r - r, 2 * center_c - c) for r, c in coords]


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    sprite_comps = _connected_components((g != BG) & (g != PATH))
    sprites = []
    coord_to_sprite: dict[tuple[int, int], int] = {}
    for comp in sprite_comps:
        colors = {int(g[r, c]) for r, c in comp}
        assert len(colors) == 1, "sprites must be monochrome"
        idx = len(sprites)
        coords = sorted(comp)
        color = next(iter(colors))
        sprites.append(
            {
                "coords": coords,
                "coord_set": set(coords),
                "color": color,
                "norm": _normalize(coords),
            }
        )
        for rc in coords:
            coord_to_sprite[rc] = idx

    path_comps = _connected_components(g == PATH)
    assert path_comps, "expected at least one path"
    paths = []
    sprite_degree = Counter()
    for comp in path_comps:
        coord_set = set(comp)
        deg = {rc: sum((nr, nc) in coord_set for nr, nc in _neighbors4(*rc)) for rc in comp}
        endpoints = [rc for rc, d in deg.items() if d == 1]
        assert len(endpoints) == 2, "path must have exactly two endpoints"
        sprite_ids = []
        for r, c in endpoints:
            touching = {
                coord_to_sprite[(nr, nc)]
                for nr, nc in _neighbors4(r, c)
                if (nr, nc) in coord_to_sprite
            }
            assert len(touching) == 1, "each path endpoint must touch exactly one sprite"
            sid = next(iter(touching))
            sprite_ids.append(sid)
            sprite_degree[sid] += 1
        assert sprite_ids[0] != sprite_ids[1], "a path must connect two distinct sprites"
        kind, ref, order_key = _path_kind(comp)
        paths.append(
            {
                "coords": sorted(comp),
                "coord_set": coord_set,
                "endpoints": endpoints,
                "sprite_ids": tuple(sprite_ids),
                "kind": kind,
                "ref": ref,
                "order_key": order_key,
                "bbox": _bbox(comp),
            }
        )
    assert all(sprite_degree[i] == 1 for i in range(len(sprites))), "each sprite must connect to exactly one path"
    return {"grid": g, "sprites": sprites, "paths": paths}


def _render_from_scene(
    scene: dict,
    *,
    swap_colors: bool = True,
    erase_paths: bool = True,
    force_kind: str | None = None,
    reverse_order: bool = False,
) -> np.ndarray:
    g = scene["grid"]
    out = g.copy()
    paths = sorted(scene["paths"], key=lambda item: item["order_key"], reverse=reverse_order)

    for path in paths:
        sid_a, sid_b = path["sprite_ids"]
        sprite_a = scene["sprites"][sid_a]
        sprite_b = scene["sprites"][sid_b]

        if force_kind is None:
            kind = path["kind"]
            ref = path["ref"]
        else:
            r0, c0, r1, c1 = path["bbox"]
            if force_kind == "horizontal":
                kind = "horizontal"
                ref = ((r0 + r1) // 2, 0)
            elif force_kind == "vertical":
                kind = "vertical"
                ref = (0, (c0 + c1) // 2)
            else:
                kind = "l"
                ref = ((r0 + r1) // 2, (c0 + c1) // 2)

        copy_a = _transform_coords(sprite_a["coords"], kind, ref)
        copy_b = _transform_coords(sprite_b["coords"], kind, ref)
        color_a = sprite_b["color"] if swap_colors else sprite_a["color"]
        color_b = sprite_a["color"] if swap_colors else sprite_b["color"]

        for (r, c), color in list(zip(copy_a, [color_a] * len(copy_a))) + list(zip(copy_b, [color_b] * len(copy_b))):
            if _in_bounds(out.shape, r, c):
                out[r, c] = color

    if erase_paths:
        for path in scene["paths"]:
            for r, c in path["coords"]:
                out[r, c] = BG
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reflect each path-linked sprite pair according to the path geometry, swap pair colors on the copies, then erase the path.

    Input structure:
    - Black background with several small monochrome 4-connected sprites and thin gray 4-connected paths.
    - Each gray path connects exactly two sprites by 4-neighbor contact at its two endpoints.
    - Every path is either a horizontal line, a vertical line, or a one-elbow L path.

    Rule:
    - Detect every gray path and the two sprites touched by its endpoints.
    - Horizontal paths use their row as a mirror axis; vertical paths use their column as a mirror axis; L paths use their elbow as a 180-degree rotation center.
    - Reflect both sprites with that geometry, recoloring each reflected copy with its partner sprite's color.
    - Draw reflected copies in path order from top-left to bottom-right using elbow position for L paths and straight-path midpoint for lines, with later copies overwriting earlier cells.
    - Erase the original gray path cells to black while keeping the original sprites.

    Key insight:
    - The path is both the pairing cue and the geometric code: endpoint connectivity picks the two sprites, and the path's physical shape selects the exact reflection rule.
    """
    scene = _parse_scene(np.asarray(grid, dtype=int))
    return _render_from_scene(scene)


_BASE_SHAPES = [
    {(0, 0), (1, 0), (1, 1)},
    {(0, 0), (0, 1), (1, 0), (2, 0)},
    {(0, 0), (0, 1), (1, 1), (1, 2)},
    {(0, 1), (1, 0), (1, 1), (1, 2)},
    {(0, 0), (1, 0), (1, 1), (2, 1), (2, 2)},
    {(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)},
    {(0, 0), (0, 1), (1, 1), (2, 1), (2, 2)},
    {(0, 1), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1)},
    {(0, 0), (1, 0), (1, 1), (1, 2), (2, 2), (2, 1)},
]


def _shape_transform(
    coords: set[tuple[int, int]],
    rot: int,
    flip: bool,
) -> tuple[tuple[int, int], ...]:
    pts = [(r, c) for r, c in coords]
    h = max(r for r, _ in pts) + 1
    w = max(c for _, c in pts) + 1
    cur = pts
    cur_h, cur_w = h, w
    for _ in range(rot % 4):
        cur = [(c, cur_h - 1 - r) for r, c in cur]
        cur_h, cur_w = cur_w, cur_h
    if flip:
        cur = [(r, cur_w - 1 - c) for r, c in cur]
    return _normalize(set(cur))


def _random_shape(
    seen_norms: set[tuple[tuple[int, int], ...]],
    kind: str,
    ref_hint: int,
) -> tuple[tuple[int, int], ...]:
    choices = []
    for base in _BASE_SHAPES:
        for rot in range(4):
            for flip in (False, True):
                norm = _shape_transform(base, rot, flip)
                if norm in seen_norms:
                    continue
                coords = list(norm)
                if len(coords) < 3 or len(coords) > 6:
                    continue
                if kind == "horizontal":
                    mirrored = _normalize({(2 * ref_hint - r, c) for r, c in coords})
                elif kind == "vertical":
                    mirrored = _normalize({(r, 2 * ref_hint - c) for r, c in coords})
                else:
                    mirrored = _normalize({(2 * ref_hint - r, 2 * ref_hint - c) for r, c in coords})
                if mirrored == norm:
                    continue
                choices.append(norm)
    assert choices, "expected at least one shape choice"
    return random.choice(choices)


def _boundary_candidates(shape: tuple[tuple[int, int], ...], side: str) -> list[tuple[int, int]]:
    coords = list(shape)
    if side == "top":
        edge = min(r for r, _ in coords)
        return [(r, c) for r, c in coords if r == edge]
    if side == "bottom":
        edge = max(r for r, _ in coords)
        return [(r, c) for r, c in coords if r == edge]
    if side == "left":
        edge = min(c for _, c in coords)
        return [(r, c) for r, c in coords if c == edge]
    edge = max(c for _, c in coords)
    return [(r, c) for r, c in coords if c == edge]


def _place_shape(
    shape: tuple[tuple[int, int], ...],
    target: tuple[int, int],
    side: str,
) -> list[tuple[int, int]]:
    port = random.choice(_boundary_candidates(shape, side))
    tr, tc = target
    pr, pc = port
    return [(tr - pr + r, tc - pc + c) for r, c in shape]


def _pad8(coords: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    out = set()
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if _in_bounds(shape, rr, cc):
                    out.add((rr, cc))
    return out


def _pair_ok(
    cells: set[tuple[int, int]],
    forbidden: set[tuple[int, int]],
    shape: tuple[int, int],
) -> bool:
    return bool(cells) and cells.isdisjoint(forbidden) and all(_in_bounds(shape, r, c) for r, c in cells)


def _make_horizontal_pair(
    shape_a: tuple[tuple[int, int], ...],
    shape_b: tuple[tuple[int, int], ...],
    color_a: int,
    color_b: int,
    h: int,
    w: int,
    forbidden: set[tuple[int, int]],
):
    for _ in range(120):
        axis_r = random.randint(4, max(4, h // 3 + 2))
        c1 = random.randint(2, max(2, w // 3))
        c2 = random.randint(max(c1 + 5, (2 * w) // 3), w - 3)
        upper_left = random.choice([True, False])
        target_a = (axis_r - 1, c1) if upper_left else (axis_r + 1, c1)
        target_b = (axis_r + 1, c2) if upper_left else (axis_r - 1, c2)
        side_a = "bottom" if upper_left else "top"
        side_b = "top" if upper_left else "bottom"
        coords_a = _place_shape(shape_a, target_a, side_a)
        coords_b = _place_shape(shape_b, target_b, side_b)
        path = {(axis_r, c) for c in range(c1, c2 + 1)}
        cells = set(coords_a) | set(coords_b) | path
        if not _pair_ok(cells, forbidden, (h, w)):
            continue
        return {
            "kind": "horizontal",
            "coords_a": coords_a,
            "coords_b": coords_b,
            "color_a": color_a,
            "color_b": color_b,
            "path": sorted(path),
        }
    return None


def _make_vertical_pair(
    shape_a: tuple[tuple[int, int], ...],
    shape_b: tuple[tuple[int, int], ...],
    color_a: int,
    color_b: int,
    h: int,
    w: int,
    forbidden: set[tuple[int, int]],
):
    for _ in range(120):
        axis_c = random.randint(4, max(4, w // 3 + 1))
        r1 = random.randint(max(h // 2 - 1, 8), max(h // 2 + 1, h - 9))
        r2 = random.randint(r1 + 4, h - 3)
        left_top = random.choice([True, False])
        target_a = (r1, axis_c - 1) if left_top else (r1, axis_c + 1)
        target_b = (r2, axis_c + 1) if left_top else (r2, axis_c - 1)
        side_a = "right" if left_top else "left"
        side_b = "left" if left_top else "right"
        coords_a = _place_shape(shape_a, target_a, side_a)
        coords_b = _place_shape(shape_b, target_b, side_b)
        path = {(r, axis_c) for r in range(r1, r2 + 1)}
        cells = set(coords_a) | set(coords_b) | path
        if not _pair_ok(cells, forbidden, (h, w)):
            continue
        return {
            "kind": "vertical",
            "coords_a": coords_a,
            "coords_b": coords_b,
            "color_a": color_a,
            "color_b": color_b,
            "path": sorted(path),
        }
    return None


def _make_l_pair(
    shape_a: tuple[tuple[int, int], ...],
    shape_b: tuple[tuple[int, int], ...],
    color_a: int,
    color_b: int,
    h: int,
    w: int,
    forbidden: set[tuple[int, int]],
):
    for _ in range(180):
        hdir = random.choice([-1, 1])
        vdir = random.choice([-1, 1])
        hlen = random.randint(4, 7)
        vlen = random.randint(4, 7)
        er = random.randint(6, h - 7)
        ec = random.randint(6, w - 7)
        term_a = (er, ec + hdir * (hlen - 1))
        term_b = (er + vdir * (vlen - 1), ec)
        target_a = (term_a[0], term_a[1] + hdir)
        target_b = (term_b[0] + vdir, term_b[1])
        side_a = "left" if hdir == 1 else "right"
        side_b = "top" if vdir == 1 else "bottom"
        coords_a = _place_shape(shape_a, target_a, side_a)
        coords_b = _place_shape(shape_b, target_b, side_b)
        path = {(er, c) for c in range(min(ec, term_a[1]), max(ec, term_a[1]) + 1)}
        path.update((r, ec) for r in range(min(er, term_b[0]), max(er, term_b[0]) + 1))
        cells = set(coords_a) | set(coords_b) | path
        if not _pair_ok(cells, forbidden, (h, w)):
            continue
        return {
            "kind": "l",
            "coords_a": coords_a,
            "coords_b": coords_b,
            "color_a": color_a,
            "color_b": color_b,
            "path": sorted(path),
        }
    return None


def _assemble_grid(pairs: list[dict], h: int, w: int) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for pair in pairs:
        for r, c in pair["path"]:
            grid[r, c] = PATH
        for r, c in pair["coords_a"]:
            grid[r, c] = pair["color_a"]
        for r, c in pair["coords_b"]:
            grid[r, c] = pair["color_b"]
    return grid


def _copy_overlap_score(scene: dict) -> int:
    total = 0
    occupied = set()
    for path in scene["paths"]:
        sid_a, sid_b = path["sprite_ids"]
        copy_a = set(_transform_coords(scene["sprites"][sid_a]["coords"], path["kind"], path["ref"]))
        copy_b = set(_transform_coords(scene["sprites"][sid_b]["coords"], path["kind"], path["ref"]))
        total += len((copy_a | copy_b) & occupied)
        occupied |= copy_a | copy_b
    return total


def _validate_scene(grid: np.ndarray) -> bool:
    try:
        scene = _parse_scene(grid)
        out = main(grid)
    except AssertionError:
        return False

    kinds = Counter(path["kind"] for path in scene["paths"])
    if set(kinds) != {"horizontal", "vertical", "l"}:
        return False

    if _copy_overlap_score(scene) > 2:
        return False

    for path in scene["paths"]:
        sid_a, sid_b = path["sprite_ids"]
        true_a = set(_transform_coords(scene["sprites"][sid_a]["coords"], path["kind"], path["ref"]))
        true_b = set(_transform_coords(scene["sprites"][sid_b]["coords"], path["kind"], path["ref"]))
        if true_a == set(scene["sprites"][sid_a]["coords"]):
            return False
        if true_b == set(scene["sprites"][sid_b]["coords"]):
            return False
        for rr, cc in true_a | true_b:
            if not _in_bounds(grid.shape, rr, cc):
                return False

    if np.array_equal(out, _render_from_scene(scene, swap_colors=False)):
        return False
    if np.array_equal(out, _render_from_scene(scene, erase_paths=False)):
        return False
    if np.array_equal(out, _render_from_scene(scene, force_kind="horizontal")):
        return False
    if np.array_equal(out, _render_from_scene(scene, force_kind="vertical")):
        return False
    if np.array_equal(out, _render_from_scene(scene, force_kind="l")):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        h = random.randint(20, 26)
        w = random.randint(20, 26)
        colors = random.sample(SPRITE_PALETTE, 6)
        seen_shapes: set[tuple[tuple[int, int], ...]] = set()
        shapes = []
        for kind, hint in (("horizontal", 1), ("horizontal", 1), ("vertical", 1), ("vertical", 1), ("l", 1), ("l", 1)):
            shape = _random_shape(seen_shapes, kind, hint)
            seen_shapes.add(shape)
            shapes.append(shape)

        forbidden: set[tuple[int, int]] = set()
        pairs = []

        builders = [
            _make_horizontal_pair(shapes[0], shapes[1], colors[0], colors[1], h, w, forbidden),
            None,
        ]
        if builders[0] is None:
            continue
        pairs.append(builders[0])
        forbidden |= _pad8(set(builders[0]["path"]) | set(builders[0]["coords_a"]) | set(builders[0]["coords_b"]), (h, w))

        vertical = _make_vertical_pair(shapes[2], shapes[3], colors[2], colors[3], h, w, forbidden)
        if vertical is None:
            continue
        pairs.append(vertical)
        forbidden |= _pad8(set(vertical["path"]) | set(vertical["coords_a"]) | set(vertical["coords_b"]), (h, w))

        l_pair = _make_l_pair(shapes[4], shapes[5], colors[4], colors[5], h, w, forbidden)
        if l_pair is None:
            continue
        pairs.append(l_pair)

        grid = _assemble_grid(pairs, h, w)
        if _validate_scene(grid):
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
