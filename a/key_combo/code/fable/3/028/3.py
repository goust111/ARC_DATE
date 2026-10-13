from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))
Coord = tuple[int, int]
Shape = tuple[Coord, ...]


def _bg_color(g: np.ndarray) -> int:
    return int(Counter(map(int, g.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        q: deque[Coord] = deque([(sr, sc)])
        seen[sr, sc] = True
        cells: list[Coord] = []
        while q:
            r, c = q.popleft()
            cells.append((r, c))
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(np.array(cells, dtype=int))
    return comps


def _find_components(g: np.ndarray, bg: int) -> list[dict]:
    out: list[dict] = []
    for color in sorted(set(map(int, g.ravel())) - {int(bg)}):
        for coords in _components_mask(g == color):
            out.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": _bbox_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return out


def _normalize(coords) -> Shape:
    arr = np.array(list(coords), dtype=int)
    arr = arr - arr.min(axis=0)
    return tuple(sorted((int(r), int(c)) for r, c in arr.tolist()))


def _shape_size(shape: Shape) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _dihedral_variants(shape: Shape) -> list[Shape]:
    pts = np.array(shape, dtype=int)
    seen: set[Shape] = set()
    variants: list[Shape] = []
    cur = pts.copy()
    for _ in range(4):
        for reflected in (False, True):
            cand = cur.copy()
            if reflected:
                cand[:, 1] = -cand[:, 1]
            norm = _normalize(cand)
            if norm not in seen:
                seen.add(norm)
                variants.append(norm)
        cur = np.column_stack((cur[:, 1], -cur[:, 0]))
    return variants


def _canonical(shape: Shape) -> Shape:
    return min(_dihedral_variants(shape))


def _shape_mask(shape: Shape, color: int, bg: int = BG) -> np.ndarray:
    h, w = _shape_size(shape)
    patch = np.full((h, w), int(bg), dtype=int)
    for r, c in shape:
        patch[r, c] = int(color)
    return patch


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int | None = None) -> None:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    if transparent is None:
        view[:, :] = patch
    else:
        m = patch != int(transparent)
        view[m] = patch[m]


def _random_free_location(
    occupied: np.ndarray,
    patch_mask: np.ndarray,
    *,
    border: int = 0,
    padding: int = 0,
) -> Coord:
    h, w = patch_mask.shape
    H, W = occupied.shape
    candidates: list[Coord] = []
    for r0 in range(border, H - h - border + 1):
        for c0 in range(border, W - w - border + 1):
            rr0 = max(0, r0 - padding)
            cc0 = max(0, c0 - padding)
            rr1 = min(H, r0 + h + padding)
            cc1 = min(W, c0 + w + padding)
            local = occupied[rr0:rr1, cc0:cc1]
            mask = np.zeros_like(local, dtype=bool)
            mr0 = r0 - rr0
            mc0 = c0 - cc0
            mask[mr0 : mr0 + h, mc0 : mc0 + w] = patch_mask
            if not bool((local & mask).any()):
                candidates.append((r0, c0))
    if not candidates:
        raise ValueError("no free location")
    return random.choice(candidates)


def _component_holes(g: np.ndarray, comp: dict, bg: int) -> list[dict]:
    r0, c0, r1, c1 = comp["bbox"]
    crop = g[r0 : r1 + 1, c0 : c1 + 1]
    local_bg = crop == int(bg)
    h, w = local_bg.shape
    outside = np.zeros((h, w), dtype=bool)
    q: deque[Coord] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if bool(local_bg[r, c]) and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if bool(local_bg[r, c]) and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and bool(local_bg[nr, nc]) and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))

    holes = []
    for local_coords in _components_mask(local_bg & ~outside):
        coords = local_coords + np.array([r0, c0])
        holes.append(
            {
                "coords": coords,
                "cells": frozenset((int(r), int(c)) for r, c in coords.tolist()),
                "bbox": _bbox_coords(coords),
                "area": int(coords.shape[0]),
            }
        )
    holes.sort(key=lambda item: item["bbox"])
    return holes


def _scene_parts(g: np.ndarray) -> tuple[int, list[dict], list[dict]]:
    bg = _bg_color(g)
    components = _find_components(g, bg)
    assert components, "expected foreground components"

    holed: list[tuple[int, list[dict]]] = []
    for index, comp in enumerate(components):
        holes = _component_holes(g, comp, bg)
        if holes:
            holed.append((index, holes))
    assert len(holed) == 1, "expected exactly one holed object"

    container_index, holes = holed[0]
    assert len(holes) >= 1, "expected enclosed background holes"

    sprites: list[dict] = []
    for index, comp in enumerate(components):
        if index == container_index:
            continue
        coords = comp["coords"]
        r0, c0, _r1, _c1 = comp["bbox"]
        shape = _normalize((int(r - r0), int(c - c0)) for r, c in coords.tolist())
        sprites.append(
            {
                "color": int(comp["color"]),
                "coords": coords,
                "shape": shape,
                "variants": _dihedral_variants(shape),
                "area": int(comp["area"]),
            }
        )
    assert len(sprites) >= 2 * len(holes), "expected enough sprites to fill the holes"
    return bg, holes, sprites


def _placements_for_shape(shape: Shape, hole: dict) -> list[frozenset[Coord]]:
    hole_cells = hole["cells"]
    if len(shape) > int(hole["area"]):
        return []
    max_r = max(r for r, _ in shape)
    max_c = max(c for _, c in shape)
    r0, c0, r1, c1 = hole["bbox"]
    out: list[frozenset[Coord]] = []
    for ar in range(r0 - max_r, r1 + 1):
        for ac in range(c0 - max_c, c1 + 1):
            cells = frozenset((ar + dr, ac + dc) for dr, dc in shape)
            if cells <= hole_cells:
                out.append(cells)
    return out


def _hole_pair_options(sprites: list[dict], hole: dict) -> list[tuple[tuple[int, frozenset[Coord]], tuple[int, frozenset[Coord]]]]:
    placements: list[list[frozenset[Coord]]] = []
    for sprite in sprites:
        seen: set[frozenset[Coord]] = set()
        cur: list[frozenset[Coord]] = []
        for variant in sprite["variants"]:
            for cells in _placements_for_shape(variant, hole):
                if cells not in seen:
                    seen.add(cells)
                    cur.append(cells)
        placements.append(cur)

    hole_cells = hole["cells"]
    hole_area = int(hole["area"])
    options = []
    for i in range(len(sprites)):
        if not placements[i]:
            continue
        for j in range(i + 1, len(sprites)):
            if int(sprites[i]["area"]) + int(sprites[j]["area"]) != hole_area:
                continue
            if not placements[j]:
                continue
            for cells_i in placements[i]:
                rest = hole_cells - cells_i
                if len(rest) != int(sprites[j]["area"]):
                    continue
                for cells_j in placements[j]:
                    if cells_j == rest:
                        options.append(((i, cells_i), (j, cells_j)))
    return options


def _render_solution(
    g: np.ndarray,
    bg: int,
    sprites: list[dict],
    assignments: list[tuple[int, tuple[tuple[int, frozenset[Coord]], tuple[int, frozenset[Coord]]]]],
) -> np.ndarray:
    out = g.copy()
    used = sorted({sprite_index for _hole_index, option in assignments for sprite_index, _cells in option})
    for sprite_index in used:
        coords = sprites[sprite_index]["coords"]
        out[coords[:, 0], coords[:, 1]] = int(bg)
    for _hole_index, option in assignments:
        for sprite_index, cells in option:
            color = int(sprites[sprite_index]["color"])
            for r, c in cells:
                out[r, c] = color
    return out


def _solve_output(g: np.ndarray, bg: int, holes: list[dict], sprites: list[dict]) -> np.ndarray:
    options_by_hole = [_hole_pair_options(sprites, hole) for hole in holes]
    assert all(options for options in options_by_hole), "every hole must have at least one two-sprite fill"
    order = sorted(range(len(holes)), key=lambda i: len(options_by_hole[i]))

    solutions: list[np.ndarray] = []
    solution_keys: set[bytes] = set()

    def search(pos: int, used: frozenset[int], assignments: list) -> None:
        if len(solutions) > 1:
            return
        if pos == len(order):
            out = _render_solution(g, bg, sprites, assignments)
            key = out.tobytes() + str(out.shape).encode()
            if key not in solution_keys:
                solution_keys.add(key)
                solutions.append(out)
            return

        hole_index = order[pos]
        for option in options_by_hole[hole_index]:
            ids = frozenset(sprite_index for sprite_index, _cells in option)
            if ids & used:
                continue
            search(pos + 1, used | ids, [*assignments, (hole_index, option)])

    search(0, frozenset(), [])
    assert len(solutions) == 1, "expected a globally unique visible two-piece tiling"
    return solutions[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the unique two-sprite tiling pair into each enclosed hole, allowing rotations and reflections.

    Input structure:
    - The dominant color is the background, and one large monochrome object contains enclosed background holes.
    - Outside that object are monochrome sprites; no sprite alone is congruent to a complete hole.
    - Each hole is exactly fillable by two outside sprites after independent rotations or reflections.
    - Extra outside sprites may be present and do not participate in any hole fill.

    Rule:
    - Preserve the large holed object.
    - For every hole, find the globally unique assignment of two distinct sprites whose transformed masks exactly partition that hole.
    - Erase the original locations of only the participating sprites.
    - Paint each transformed sprite into its matched hole with its original color, leaving distractor sprites unchanged.

    Key insight:
    - The target is not a single-shape match: each cavity is a two-piece jigsaw, and the moved sprites are determined by the unique global tiling rather than by color, size, or distance.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg, holes, sprites = _scene_parts(g)
    return _solve_output(g, bg, holes, sprites)


# ---------------- generator ----------------

GEN_RETRIES = 600
GRID_RANGE = (25, 30)
CONTAINER_RANGE = (14, 18)
HOLE_AREA_CHOICES = (8, 9, 10, 11)
HOLE_COUNT_WEIGHTS = (3, 2)
MAX_SHAPE_SIDE = 5


def _connected(shape: Shape) -> bool:
    cells = set(shape)
    if not cells:
        return False
    q = deque([next(iter(cells))])
    seen = {q[0]}
    while q:
        r, c = q.popleft()
        for nb in _neighbors4(r, c):
            if nb in cells and nb not in seen:
                seen.add(nb)
                q.append(nb)
    return len(seen) == len(cells)


def _random_connected_shape(area: int, max_side: int = MAX_SHAPE_SIDE, *, require_2d: bool = True) -> Shape:
    area = int(area)
    for _ in range(200):
        cells = {(max_side // 2, max_side // 2)}
        while len(cells) < area:
            frontier = set()
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < max_side and 0 <= nc < max_side and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(tuple(frontier)))
        if len(cells) == area:
            shape = _normalize(cells)
            h, w = _shape_size(shape)
            if (not require_2d or (h >= 2 and w >= 2)) and _connected(shape):
                return shape
    raise ValueError("failed to build connected shape")


def _grow_subset(cells: set[Coord], target: int) -> set[Coord] | None:
    start = random.choice(tuple(cells))
    out = {start}
    while len(out) < target:
        frontier = set()
        for r, c in out:
            for nb in _neighbors4(r, c):
                if nb in cells and nb not in out:
                    frontier.add(nb)
        if not frontier:
            return None
        out.add(random.choice(tuple(frontier)))
    return out


def _random_two_piece_hole(area: int) -> tuple[Shape, Shape, Shape] | None:
    for _ in range(300):
        hole = _random_connected_shape(area)
        hole_cells = set(hole)
        a_area = random.randint(2, int(area) - 2)
        piece_a = _grow_subset(hole_cells, a_area)
        if piece_a is None:
            continue
        piece_b = hole_cells - piece_a
        if len(piece_b) < 2:
            continue
        a_norm = _normalize(piece_a)
        b_norm = _normalize(piece_b)
        if not (_connected(a_norm) and _connected(b_norm)):
            continue
        if len(_dihedral_variants(a_norm)) == 1 and len(_dihedral_variants(b_norm)) == 1:
            continue
        return hole, tuple(sorted(piece_a)), tuple(sorted(piece_b))
    return None


def _place_hole_mask(occ: np.ndarray, shape: Shape) -> Coord | None:
    h, w = _shape_size(shape)
    H, W = occ.shape
    candidates: list[Coord] = []
    for r0 in range(1, H - h):
        for c0 in range(1, W - w):
            cells = [(r0 + r, c0 + c) for r, c in shape]
            ok = True
            for r, c in cells:
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        rr, cc = r + dr, c + dc
                        if 0 <= rr < H and 0 <= cc < W and bool(occ[rr, cc]):
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if ok:
                candidates.append((r0, c0))
    if not candidates:
        return None
    return random.choice(candidates)


def _build_holes(container_h: int, container_w: int, n_holes: int):
    hole_occ = np.zeros((container_h, container_w), dtype=bool)
    specs = []
    seen_holes: set[Shape] = set()
    first_area = random.choice(HOLE_AREA_CHOICES)
    areas = [first_area, first_area]
    while len(areas) < n_holes:
        areas.append(random.choice(HOLE_AREA_CHOICES))
    random.shuffle(areas)

    for area in areas:
        placed = False
        for _ in range(250):
            made = _random_two_piece_hole(area)
            if made is None:
                continue
            hole, piece_a, piece_b = made
            canon = _canonical(hole)
            if canon in seen_holes:
                continue
            loc = _place_hole_mask(hole_occ, hole)
            if loc is None:
                continue
            r0, c0 = loc
            cells = frozenset((r0 + r, c0 + c) for r, c in hole)
            for r, c in cells:
                hole_occ[r, c] = True
            seen_holes.add(canon)
            specs.append(
                {
                    "hole": hole,
                    "piece_a": piece_a,
                    "piece_b": piece_b,
                    "top_left": (r0, c0),
                    "cells": cells,
                }
            )
            placed = True
            break
        if not placed:
            return None
    return specs


def _cells_at(shape: Shape, r0: int, c0: int) -> frozenset[Coord]:
    return frozenset((r0 + r, c0 + c) for r, c in shape)


def _all_piece_specs(hole_specs: list[dict]) -> list[dict]:
    pieces = []
    for hole_index, spec in enumerate(hole_specs):
        hr, hc = spec["top_left"]
        for piece_key in ("piece_a", "piece_b"):
            target = tuple(sorted(spec[piece_key]))
            pieces.append(
                {
                    "hole_index": hole_index,
                    "target_shape": _normalize(target),
                    "target_cells_local": frozenset(target),
                    "target_cells": frozenset((hr + r, hc + c) for r, c in target),
                }
            )
    return pieces


def _assign_source_orientations(pieces: list[dict]) -> bool:
    transform_seen = False
    transformable = []
    for i, piece in enumerate(pieces):
        variants = _dihedral_variants(piece["target_shape"])
        if len(variants) > 1:
            transformable.append(i)
        source = random.choice(variants)
        piece["source_shape"] = source
        if source != piece["target_shape"]:
            transform_seen = True

    if not transform_seen and transformable:
        i = random.choice(transformable)
        choices = [v for v in _dihedral_variants(pieces[i]["target_shape"]) if v != pieces[i]["target_shape"]]
        pieces[i]["source_shape"] = random.choice(choices)
        transform_seen = True
    return transform_seen


def _make_distractor_shape(area: int, participant_shapes: list[Shape]) -> Shape | None:
    participant_canons = {_canonical(s) for s in participant_shapes if len(s) == int(area)}
    for _ in range(120):
        shape = _random_connected_shape(area, require_2d=False)
        if _canonical(shape) not in participant_canons:
            return shape
    return None


def _semantic_expected(
    inp: np.ndarray,
    bg: int,
    pieces: list[dict],
    distractor_coords: list[frozenset[Coord]],
) -> np.ndarray:
    out = inp.copy()
    for piece in pieces:
        for r, c in piece["source_cells"]:
            out[r, c] = bg
    for piece in pieces:
        for r, c in piece["target_cells_global"]:
            out[r, c] = int(piece["color"])
    for cells in distractor_coords:
        assert all(int(out[r, c]) != bg for r, c in cells)
    return out


def _attempt() -> np.ndarray | None:
    rows = random.randint(*GRID_RANGE)
    cols = random.randint(*GRID_RANGE)
    container_h = random.randint(*CONTAINER_RANGE)
    container_w = random.randint(*CONTAINER_RANGE)
    n_holes = random.choices((2, 3), weights=HOLE_COUNT_WEIGHTS, k=1)[0]

    hole_specs = _build_holes(container_h, container_w, n_holes)
    if hole_specs is None:
        return None
    pieces = _all_piece_specs(hole_specs)
    if not _assign_source_orientations(pieces):
        return None

    palette = list(COLORS)
    random.shuffle(palette)
    container_color = palette.pop()
    if len(palette) < len(pieces):
        return None
    for piece, color in zip(pieces, palette[: len(pieces)]):
        piece["color"] = int(color)
    spare_colors = palette[len(pieces) :]

    grid = np.full((rows, cols), BG, dtype=int)
    blocked = np.zeros((rows, cols), dtype=bool)
    cr = random.randint(2, rows - container_h - 2)
    cc = random.randint(2, cols - container_w - 2)
    grid[cr : cr + container_h, cc : cc + container_w] = container_color
    blocked[cr : cr + container_h, cc : cc + container_w] = True

    for spec in hole_specs:
        for lr, lc in spec["cells"]:
            grid[cr + lr, cc + lc] = BG
    for piece in pieces:
        piece["target_cells_global"] = frozenset((cr + r, cc + c) for r, c in piece["target_cells"])

    random.shuffle(pieces)
    participant_shapes = [piece["target_shape"] for piece in pieces]
    for piece in pieces:
        patch = _shape_mask(piece["source_shape"], int(piece["color"]))
        mask = patch != BG
        try:
            sr, sc = _random_free_location(blocked, mask, padding=1)
        except ValueError:
            return None
        _paste(grid, patch, sr, sc, transparent=BG)
        blocked[sr : sr + patch.shape[0], sc : sc + patch.shape[1]] |= mask
        piece["source_cells"] = _cells_at(piece["source_shape"], sr, sc)

    distractor_coords: list[frozenset[Coord]] = []
    distractor_count = random.randint(1, 2)
    participant_areas = [len(shape) for shape in participant_shapes]
    for i in range(distractor_count):
        area = random.choice(participant_areas) if i == 0 else random.randint(2, 6)
        shape = _make_distractor_shape(area, participant_shapes)
        if shape is None:
            return None
        color = random.choice(spare_colors or palette)
        patch = _shape_mask(shape, int(color))
        mask = patch != BG
        try:
            dr, dc = _random_free_location(blocked, mask, padding=1)
        except ValueError:
            return None
        _paste(grid, patch, dr, dc, transparent=BG)
        blocked[dr : dr + patch.shape[0], dc : dc + patch.shape[1]] |= mask
        distractor_coords.append(_cells_at(shape, dr, dc))

    expected = _semantic_expected(grid, BG, pieces, distractor_coords)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    bg, holes, sprites = _scene_parts(grid)
    if bg != BG or len(holes) != n_holes:
        return None
    if any(
        any(len(sprite["shape"]) == int(hole["area"]) and _canonical(sprite["shape"]) == _canonical(_normalize(hole["cells"])) for hole in holes)
        for sprite in sprites
    ):
        return None
    if not any(piece["source_shape"] != piece["target_shape"] for piece in pieces):
        return None
    if not np.any(actual != grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid two-piece hole tiling input")
