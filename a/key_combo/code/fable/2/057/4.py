from __future__ import annotations

from collections import Counter, deque
from functools import lru_cache
import random

import numpy as np


BLACK = 0
GREY = 5
SPRITE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]

BASE_SHAPES: tuple[tuple[tuple[int, int], ...], ...] = (
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 0)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 1), (1, 0), (1, 1), (2, 0)),
    ((0, 0), (1, 0), (1, 1), (1, 2), (2, 2)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 0)),
)

GEN_RETRIES = 400
SPRITE_PLACEMENT_TRIES = 600


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_components(grid: np.ndarray, *, bg: int, connectivity: int = 8) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    seen = np.zeros((H, W), dtype=bool)
    dirs = DIR8 if int(connectivity) == 8 else DIR4
    out: list[dict] = []

    for r in range(H):
        for c in range(W):
            color = int(g[r, c])
            if seen[r, c] or color == bg:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in dirs:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.array(coords, dtype=int)
            out.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": _bbox_coords(arr),
                    "area": int(arr.shape[0]),
                }
            )
    return out


def _rotate_coord_cw(r: int, c: int, anchor_top: tuple[int, int], turns: int) -> tuple[int, int]:
    ar, ac = map(int, anchor_top)
    rr, cc = int(r), int(c)
    for _ in range(int(turns) % 4):
        rr, cc = ar + cc - ac, ac + ar + 1 - rr
    return int(rr), int(cc)


def _min_anchor_distance(
    coords: list[tuple[int, int]] | set[tuple[int, int]] | np.ndarray,
    anchor_coords: list[tuple[int, int]],
    *,
    metric: str,
) -> int:
    best = 10**9
    for r, c in coords:
        rr, cc = int(r), int(c)
        for ar, ac in anchor_coords:
            if metric == "chebyshev":
                dist = max(abs(rr - ar), abs(cc - ac))
            elif metric == "manhattan":
                dist = abs(rr - ar) + abs(cc - ac)
            else:
                raise ValueError("unknown metric")
            if dist < best:
                best = dist
    return int(best)


def _shape_cell_distances(coords: set[tuple[int, int]], anchor_coords: list[tuple[int, int]]) -> set[int]:
    return {
        _min_anchor_distance([(int(r), int(c))], anchor_coords, metric="chebyshev")
        for r, c in coords
    }


def _parse_scene(grid: np.ndarray) -> tuple[np.ndarray, int, dict, list[tuple[int, int]], tuple[int, int], list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    comps = _find_components(g, bg=bg, connectivity=8)
    anchors = [
        comp
        for comp in comps
        if comp["color"] == GREY
        and comp["area"] == 4
        and comp["bbox"][2] - comp["bbox"][0] + 1 == 2
        and comp["bbox"][3] - comp["bbox"][1] + 1 == 2
    ]
    assert len(anchors) == 1, "expected exactly one grey 2x2 anchor"
    anchor = anchors[0]
    anchor_coords = [tuple(map(int, rc)) for rc in anchor["coords"]]
    sprites = [comp for comp in comps if comp is not anchor]
    assert sprites, "expected at least one sprite"
    sprites.sort(key=lambda comp: (comp["bbox"], comp["color"]))
    return g, bg, anchor, anchor_coords, anchor["bbox"][:2], sprites


def _apply_rule(grid: np.ndarray, turn_selector, *, strict: bool) -> np.ndarray | None:
    g, bg, _anchor, anchor_coords, anchor_top, sprites = _parse_scene(grid)
    H, W = g.shape
    out = g.copy()

    for comp in sprites:
        out[tuple(comp["coords"].T)] = bg

    written = {tuple(map(int, rc)) for rc in anchor_coords}
    for comp in sprites:
        coords = [tuple(map(int, rc)) for rc in comp["coords"]]
        d = _min_anchor_distance(coords, anchor_coords, metric="chebyshev")
        turns = int(turn_selector(coords, d, anchor_coords)) % 4
        for r, c in coords:
            rr, cc = _rotate_coord_cw(r, c, anchor_top, turns)
            if not (0 <= rr < H and 0 <= cc < W):
                if strict:
                    raise AssertionError("rotated sprite left the grid")
                return None
            if strict:
                assert (rr, cc) not in written, "rotated sprites must not overlap"
            out[rr, cc] = int(comp["color"])
            written.add((rr, cc))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each sprite clockwise around the grey 2x2 anchor by a number of quarter-turns set by its Chebyshev ring distance.

    Input structure:
    - A black background contains exactly one grey 2x2 anchor square.
    - The other foreground objects are disjoint monochrome sprites scattered around the anchor.
    - Each sprite can have any small connected shape, and different sprites may sit at different ring distances.

    Rule:
    - Keep the grey 2x2 anchor fixed.
    - For each sprite, compute the minimum Chebyshev distance from any sprite cell to any anchor cell.
    - Rotate the whole sprite clockwise around the geometric center of the 2x2 anchor by `distance * 90` degrees, clearing its old location.

    Key insight:
    - The anchor is the common rotation axis for every sprite, and the nearest anchor contact ring determines how many quarter-turns that particular sprite takes.
    """
    out = _apply_rule(
        np.asarray(grid, dtype=int),
        lambda _coords, d, _anchor_coords: d,
        strict=True,
    )
    assert out is not None
    return out


def _normalize_shape(coords: list[tuple[int, int]] | np.ndarray) -> tuple[tuple[int, int], ...]:
    arr = np.array(coords, dtype=int)
    arr[:, 0] -= arr[:, 0].min()
    arr[:, 1] -= arr[:, 1].min()
    items = sorted((int(r), int(c)) for r, c in arr.tolist())
    return tuple(items)


def _rotate_shape_cw(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return _normalize_shape([(c, -r) for r, c in shape])


@lru_cache(None)
def _shape_library() -> tuple[tuple[tuple[int, int], ...], ...]:
    shapes: set[tuple[tuple[int, int], ...]] = set()
    for base in BASE_SHAPES:
        base_shape = tuple((int(r), int(c)) for r, c in base)
        for flip in (False, True):
            shape = tuple((r, -c if flip else c) for r, c in base_shape)
            shape = _normalize_shape(list(shape))
            for _ in range(4):
                shapes.add(shape)
                shape = _rotate_shape_cw(shape)
    return tuple(sorted(shapes))


def _touches_any(coords: set[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    if not occupied:
        return False
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (r + dr, c + dc) in occupied:
                    return True
    return False


def _all_rotations_in_bounds(
    coords: set[tuple[int, int]],
    shape: tuple[int, int],
    anchor_top: tuple[int, int],
) -> bool:
    H, W = map(int, shape)
    for turns in range(4):
        for r, c in coords:
            rr, cc = _rotate_coord_cw(r, c, anchor_top, turns)
            if not (0 <= rr < H and 0 <= cc < W):
                return False
    return True


def _paint_coords(grid: np.ndarray, coords: set[tuple[int, int]], color: int) -> None:
    for r, c in coords:
        grid[int(r), int(c)] = int(color)


def _try_place_sprite(
    grid_shape: tuple[int, int],
    anchor_coords: list[tuple[int, int]],
    anchor_top: tuple[int, int],
    anchor_set: set[tuple[int, int]],
    input_occupied: set[tuple[int, int]],
    output_occupied: set[tuple[int, int]],
    *,
    color: int,
    target_d: int,
    diagonal_requirement: bool | None,
    varied_requirement: bool,
) -> dict | None:
    H, W = map(int, grid_shape)
    shapes = _shape_library()

    for _ in range(SPRITE_PLACEMENT_TRIES):
        shape = random.choice(shapes)
        h = 1 + max(r for r, _c in shape)
        w = 1 + max(c for _r, c in shape)
        r0 = random.randint(0, H - h)
        c0 = random.randint(0, W - w)
        coords = {(r0 + r, c0 + c) for r, c in shape}

        if coords & anchor_set:
            continue
        if _touches_any(coords, input_occupied):
            continue
        if not _all_rotations_in_bounds(coords, grid_shape, anchor_top):
            continue

        cheb = _min_anchor_distance(coords, anchor_coords, metric="chebyshev")
        if cheb != int(target_d):
            continue
        man = _min_anchor_distance(coords, anchor_coords, metric="manhattan")
        if diagonal_requirement is True and man == cheb:
            continue
        if diagonal_requirement is False and man != cheb:
            continue
        if varied_requirement and len(_shape_cell_distances(coords, anchor_coords)) < 2:
            continue

        rotated = {_rotate_coord_cw(r, c, anchor_top, cheb % 4) for r, c in coords}
        if cheb % 4 != 0 and rotated == coords:
            continue
        if rotated & anchor_set:
            continue
        if rotated & output_occupied:
            continue
        if _touches_any(rotated, output_occupied):
            continue

        return {
            "color": int(color),
            "coords": coords,
            "rotated": rotated,
            "distance": int(cheb),
            "diagonal": bool(man > cheb),
            "varied": len(_shape_cell_distances(coords, anchor_coords)) > 1,
        }
    return None


def _wrong_gap_turns(_coords, d: int, _anchor_coords) -> int:
    return max(int(d) - 1, 0)


def _wrong_manhattan_turns(coords, _d: int, anchor_coords: list[tuple[int, int]]) -> int:
    return _min_anchor_distance(coords, anchor_coords, metric="manhattan")


def _semantic_ok(grid: np.ndarray) -> bool:
    try:
        g, _bg, _anchor, anchor_coords, anchor_top, sprites = _parse_scene(grid)
        if len(sprites) < 4:
            return False

        distances = [_min_anchor_distance(comp["coords"], anchor_coords, metric="chebyshev") for comp in sprites]
        if not {1, 2, 3, 4}.issubset(set(distances)):
            return False

        if not any(_min_anchor_distance(comp["coords"], anchor_coords, metric="manhattan") > d for comp, d in zip(sprites, distances)):
            return False

        if not any(len(_shape_cell_distances({tuple(map(int, rc)) for rc in comp["coords"]}, anchor_coords)) > 1 for comp in sprites):
            return False

        out = main(g)
        if np.array_equal(out, g):
            return False

        saw_stationary = False
        saw_moved = False
        for comp, d in zip(sprites, distances):
            coords = {tuple(map(int, rc)) for rc in comp["coords"]}
            rotated = {_rotate_coord_cw(r, c, anchor_top, d % 4) for r, c in coords}
            if d % 4 == 0:
                if rotated != coords:
                    return False
                saw_stationary = True
            else:
                if rotated == coords:
                    return False
                saw_moved = True
        if not (saw_stationary and saw_moved):
            return False

        wrong_outputs = [
            _apply_rule(g, lambda _coords, _d, _anchor_coords: 1, strict=False),
            _apply_rule(g, _wrong_gap_turns, strict=False),
            _apply_rule(g, _wrong_manhattan_turns, strict=False),
        ]
        if any(w is not None and np.array_equal(w, out) for w in wrong_outputs):
            return False
    except AssertionError:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(18, 28)
    W = random.randint(18, 28)
    grid = np.full((H, W), BLACK, dtype=int)

    ar = random.randint(5, H - 7)
    ac = random.randint(5, W - 7)
    grid[ar : ar + 2, ac : ac + 2] = GREY

    anchor_coords = [(ar, ac), (ar, ac + 1), (ar + 1, ac), (ar + 1, ac + 1)]
    anchor_top = (ar, ac)
    anchor_set = set(anchor_coords)

    specs: list[dict] = [
        {"d": 1, "diag": False, "varied": False},
        {"d": 2, "diag": True, "varied": False},
        {"d": 3, "diag": None, "varied": True},
        {"d": 4, "diag": None, "varied": False},
    ]
    if random.random() < 0.55:
        specs.append(
            {
                "d": random.choice([1, 2, 3, 4]),
                "diag": random.choice([None, True, False]),
                "varied": random.choice([False, True]),
            }
        )

    colors = random.sample(SPRITE_COLORS, len(specs))
    input_occupied: set[tuple[int, int]] = set()
    output_occupied: set[tuple[int, int]] = set()
    placed: list[dict] = []

    for spec, color in zip(specs, colors):
        sprite = _try_place_sprite(
            grid.shape,
            anchor_coords,
            anchor_top,
            anchor_set,
            input_occupied,
            output_occupied,
            color=int(color),
            target_d=int(spec["d"]),
            diagonal_requirement=spec["diag"],
            varied_requirement=bool(spec["varied"]),
        )
        if sprite is None:
            return None
        placed.append(sprite)
        input_occupied |= sprite["coords"]
        output_occupied |= sprite["rotated"]

    for sprite in placed:
        _paint_coords(grid, sprite["coords"], sprite["color"])

    try:
        out = main(grid)
    except AssertionError:
        return None
    if out is None or not _semantic_ok(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid orbit-rotation input")
