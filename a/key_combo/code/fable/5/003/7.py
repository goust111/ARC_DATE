from __future__ import annotations

from collections import Counter, deque
import random
import sys

import numpy as np


_UTILS_DIR = "/home/lixinyu/snap/new/solvers"
if _UTILS_DIR not in sys.path:
    sys.path.append(_UTILS_DIR)

try:
    from arc_utils import get_bg as _arc_get_bg
except Exception:  # pragma: no cover - keeps SourceFileLoader robust.
    _arc_get_bg = None


Axis = tuple[str, int]
Coord = tuple[int, int]
CoordSet = frozenset[Coord]

ARC_COLORS = tuple(range(10))
GLOBAL_KINDS = ("H", "V", "D", "A")
GEN_ATTEMPTS = 900
ORBIT_ATTEMPTS = 260
COLOR_ATTEMPTS = 120


def _get_bg(grid: np.ndarray) -> int:
    if _arc_get_bg is not None:
        return int(_arc_get_bg(grid))
    return int(Counter(map(int, grid.reshape(-1))).most_common(1)[0][0])


def _neighbors(r: int, c: int, shape: tuple[int, int], connectivity: int) -> list[Coord]:
    H, W = map(int, shape)
    if int(connectivity) == 4:
        offsets = ((1, 0), (-1, 0), (0, 1), (0, -1))
    else:
        offsets = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc)
    out: list[Coord] = []
    for dr, dc in offsets:
        rr, cc = int(r) + dr, int(c) + dc
        if 0 <= rr < H and 0 <= cc < W:
            out.append((rr, cc))
    return out


def _components(mask: np.ndarray, connectivity: int = 8) -> list[list[Coord]]:
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[list[Coord]] = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        q: deque[Coord] = deque([(sr, sc)])
        seen[sr, sc] = True
        comp: list[Coord] = []
        while q:
            r, c = q.popleft()
            comp.append((r, c))
            for nr, nc in _neighbors(r, c, mask.shape, connectivity):
                if bool(mask[nr, nc]) and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _extract_objects(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    objects: list[dict[str, object]] = []
    for color in sorted(set(map(int, grid.reshape(-1))) - {int(bg)}):
        for comp in _components(grid == int(color), connectivity=8):
            coords = frozenset((int(r), int(c)) for r, c in comp)
            assert coords, "sprites may not be empty"
            objects.append({"color": int(color), "coords": coords})
    assert len(objects) >= 5, "expected several sprite components"
    return objects


def _reflect_coord(rc: Coord, axis: Axis) -> Coord:
    r, c = map(int, rc)
    kind, key = axis
    key = int(key)
    if kind == "H":
        return key - r, c
    if kind == "V":
        return r, key - c
    if kind == "D":
        return c + key, r - key
    if kind == "A":
        return key - c, key - r
    raise ValueError(f"unknown axis kind {kind!r}")


def _reflect_set(coords: CoordSet, axis: Axis) -> CoordSet:
    return frozenset(_reflect_coord(rc, axis) for rc in coords)


def _in_bounds(coords: CoordSet, shape: tuple[int, int]) -> bool:
    H, W = map(int, shape)
    return all(0 <= r < H and 0 <= c < W for r, c in coords)


def _all_axes(shape: tuple[int, int]) -> list[Axis]:
    H, W = map(int, shape)
    axes: list[Axis] = []
    axes.extend(("H", key) for key in range(0, 2 * H - 1))
    axes.extend(("V", key) for key in range(0, 2 * W - 1))
    axes.extend(("D", key) for key in range(-(W - 1), H))
    axes.extend(("A", key) for key in range(0, H + W - 1))
    return axes


def _side(coords: CoordSet, axis: Axis) -> int:
    kind, key = axis
    vals: list[int] = []
    for r, c in coords:
        if kind == "H":
            vals.append(2 * int(r) - int(key))
        elif kind == "V":
            vals.append(2 * int(c) - int(key))
        elif kind == "D":
            vals.append(int(r) - int(c) - int(key))
        elif kind == "A":
            vals.append(int(r) + int(c) - int(key))
        else:
            raise ValueError(f"unknown axis kind {kind!r}")
    if max(vals) < 0:
        return -1
    if min(vals) > 0:
        return 1
    return 0


def _geometric_candidate(objects: list[dict[str, object]], axis: Axis) -> dict[str, object] | None:
    by_coords = {obj["coords"]: idx for idx, obj in enumerate(objects)}
    if len(by_coords) != len(objects):
        return None

    paired: set[int] = set()
    pairs: list[tuple[int, int]] = []
    unpaired: list[int] = []

    for i, obj in enumerate(objects):
        if i in paired:
            continue
        mirrored = _reflect_set(obj["coords"], axis)  # type: ignore[arg-type]
        j = by_coords.get(mirrored)
        if j is None:
            unpaired.append(i)
            continue
        if j == i or j in paired:
            return None
        pairs.append((i, int(j)))
        paired.add(i)
        paired.add(int(j))

    if len(unpaired) != 1:
        return None
    if len(pairs) < 3:
        return None
    return {"axis": axis, "pairs": pairs, "unpaired": int(unpaired[0])}


def _single_cycle(mapping: dict[int, int]) -> bool:
    if len(mapping) < 3:
        return False
    colors = set(mapping)
    if set(mapping.values()) != colors:
        return False
    start = next(iter(colors))
    seen: list[int] = []
    cur = int(start)
    while cur not in seen:
        seen.append(cur)
        if cur not in mapping:
            return False
        cur = int(mapping[cur])
    return cur == start and len(seen) == len(colors)


def _color_mapping_from_pairs(
    objects: list[dict[str, object]], candidate: dict[str, object]
) -> dict[int, int] | None:
    axis = candidate["axis"]  # type: ignore[assignment]
    assert isinstance(axis, tuple)
    mapping: dict[int, int] = {}
    inverse: dict[int, int] = {}

    for i, j in candidate["pairs"]:  # type: ignore[assignment]
        left_side = _side(objects[int(i)]["coords"], axis)  # type: ignore[arg-type]
        right_side = _side(objects[int(j)]["coords"], axis)  # type: ignore[arg-type]
        if left_side == 0 or right_side == 0 or left_side == right_side:
            return None
        src = int(i) if left_side < 0 else int(j)
        dst = int(j) if left_side < 0 else int(i)
        src_color = int(objects[src]["color"])
        dst_color = int(objects[dst]["color"])
        if src_color == dst_color:
            return None
        if src_color in mapping and mapping[src_color] != dst_color:
            return None
        if dst_color in inverse and inverse[dst_color] != src_color:
            return None
        mapping[src_color] = dst_color
        inverse[dst_color] = src_color

    if not _single_cycle(mapping):
        return None
    return mapping


def _valid_chain(
    objects: list[dict[str, object]], candidate: dict[str, object]
) -> dict[int, int] | None:
    axis = candidate["axis"]  # type: ignore[assignment]
    assert isinstance(axis, tuple)
    unpaired = int(candidate["unpaired"])
    if _side(objects[unpaired]["coords"], axis) != -1:  # type: ignore[arg-type]
        return None

    mapping = _color_mapping_from_pairs(objects, candidate)
    if mapping is None:
        return None
    if int(objects[unpaired]["color"]) not in mapping:
        return None
    return mapping


def _solve_details(grid: np.ndarray) -> tuple[int, list[dict[str, object]], dict[str, object], dict[int, int], list[dict[str, object]]]:
    bg = _get_bg(grid)
    objects = _extract_objects(grid, bg)
    geometric: list[dict[str, object]] = []
    passing: list[tuple[dict[str, object], dict[int, int]]] = []

    for axis in _all_axes(grid.shape):
        candidate = _geometric_candidate(objects, axis)
        if candidate is None:
            continue
        geometric.append(candidate)
        mapping = _valid_chain(objects, candidate)
        if mapping is not None:
            passing.append((candidate, mapping))

    assert len(geometric) >= 2, "expected at least two geometrically plausible axes"
    assert len(passing) == 1, "exactly one axis must pass the color-chain test"
    candidate, mapping = passing[0]
    return bg, objects, candidate, mapping, geometric


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reject the false reflection axis by its color chain, then add the lone sprite's mirror partner.

    Input structure:
    - A dominant background surrounds several separated monochrome sprites.
    - At least two hidden horizontal, vertical, or diagonal reflection axes can
      pair all sprites except one if only geometry is considered.
    - For the true axis, reading each pair from the upper/left/NW side of that
      axis to the opposite side gives one consistent directed color cycle.

    Rule:
    - Enumerate all geometric candidate axes and their mirror pairings.
    - Keep only the axis whose paired sprite colors form one bijective directed
      color cycle; all geometric decoy axes violate that chain.
    - Mirror the single unpaired sprite across the true axis and color the copy
      with the next color in the inferred cycle. Leave all other cells unchanged.

    Key insight:
    - Geometry alone deliberately names more than one axis. The real axis is the
      one where shape pairing and side-to-side color succession agree, so the
      missing partner is determined only after rejecting the color-inconsistent
      fake axes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg, objects, candidate, mapping, _geometric = _solve_details(g)

    axis = candidate["axis"]  # type: ignore[assignment]
    assert isinstance(axis, tuple)
    unpaired = int(candidate["unpaired"])
    source = objects[unpaired]
    target = _reflect_set(source["coords"], axis)  # type: ignore[arg-type]
    assert _in_bounds(target, g.shape), "the mirrored sprite must stay on the canvas"
    assert all(int(g[r, c]) == int(bg) for r, c in target), "the missing mirror location must be empty"

    out = g.copy()
    new_color = int(mapping[int(source["color"])])
    for r, c in target:
        out[int(r), int(c)] = new_color
    return out


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


_MASKS = [
    np.array([[1]], dtype=bool),
    np.array([[1, 1]], dtype=bool),
    np.array([[1], [1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[0, 1], [1, 1]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 0], [1, 1], [0, 1]], dtype=bool),
]


def _global_axis(kind: str, side: int) -> Axis:
    if kind == "H":
        return "H", side - 1
    if kind == "V":
        return "V", side - 1
    if kind == "D":
        return "D", 0
    if kind == "A":
        return "A", side - 1
    raise ValueError(f"unknown axis kind {kind!r}")


def _max_orbit_size(side: int, axes: tuple[Axis, Axis]) -> int:
    best = 0
    for r in range(1, side - 1):
        for c in range(1, side - 1):
            orb = _orbit(frozenset({(r, c)}), axes)
            best = max(best, len(orb))
    return best


def _orbit(seed: CoordSet, axes: tuple[Axis, Axis]) -> list[CoordSet]:
    seen: set[CoordSet] = {seed}
    q: deque[CoordSet] = deque([seed])
    while q:
        cur = q.popleft()
        for axis in axes:
            nxt = _reflect_set(cur, axis)
            if nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return list(seen)


def _expanded(coords: CoordSet, side: int) -> set[Coord]:
    out: set[Coord] = set()
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = int(r) + dr, int(c) + dc
                if 0 <= rr < side and 0 <= cc < side:
                    out.add((rr, cc))
    return out


def _orbit_clear(orbit: list[CoordSet], occupied_halo: set[Coord], side: int) -> bool:
    local_halo: set[Coord] = set(occupied_halo)
    for coords in orbit:
        if not coords:
            return False
        if not all(1 <= r < side - 1 and 1 <= c < side - 1 for r, c in coords):
            return False
        if any(rc in local_halo for rc in coords):
            return False
        local_halo.update(_expanded(coords, side))
    return True


def _random_seed(side: int) -> CoordSet:
    mask = random.choice(_MASKS)
    h, w = map(int, mask.shape)
    r0 = random.randint(2, side - h - 2)
    c0 = random.randint(2, side - w - 2)
    return frozenset(
        (int(r0 + r), int(c0 + c))
        for r, c in np.argwhere(mask)
    )


def _place_orbit(
    side: int,
    axes: tuple[Axis, Axis],
    occupied_halo: set[Coord],
    expected_size: int,
    require_removable: bool,
) -> tuple[list[CoordSet], int | None] | None:
    true_axis, fake_axis = axes
    for _ in range(ORBIT_ATTEMPTS):
        seed = _random_seed(side)
        orbit = _orbit(seed, axes)
        if len(orbit) != expected_size:
            continue
        if not _orbit_clear(orbit, occupied_halo, side):
            continue
        removable: list[int] = []
        if require_removable:
            for idx, coords in enumerate(orbit):
                if _side(coords, true_axis) > 0 and _side(coords, fake_axis) > 0:
                    removable.append(idx)
            if not removable:
                continue
        removed_idx = random.choice(removable) if require_removable else None
        for coords in orbit:
            occupied_halo.update(_expanded(coords, side))
        return orbit, removed_idx
    return None


def _objects_from_coordsets(coordsets: list[CoordSet]) -> list[dict[str, object]]:
    return [{"color": -1, "coords": coords} for coords in coordsets]


def _assign_colors(
    coordsets: list[CoordSet],
    true_axis: Axis,
    fake_axis: Axis,
    bg: int,
) -> list[int] | None:
    base_objects = _objects_from_coordsets(coordsets)
    true_candidate = _geometric_candidate(base_objects, true_axis)
    fake_candidate = _geometric_candidate(base_objects, fake_axis)
    if true_candidate is None or fake_candidate is None:
        return None
    if _side(base_objects[int(true_candidate["unpaired"])]["coords"], true_axis) != -1:  # type: ignore[arg-type]
        return None
    if _side(base_objects[int(fake_candidate["unpaired"])]["coords"], fake_axis) != -1:  # type: ignore[arg-type]
        return None

    pair_count = len(true_candidate["pairs"])  # type: ignore[arg-type]
    palette = [c for c in ARC_COLORS if c != int(bg)]
    max_cycle = min(4, len(palette), pair_count)
    if max_cycle < 3:
        return None

    for _ in range(COLOR_ATTEMPTS):
        cycle_len = random.randint(3, max_cycle)
        cycle = random.sample(palette, cycle_len)
        edge_ids = list(range(cycle_len))
        while len(edge_ids) < pair_count:
            edge_ids.append(random.randrange(cycle_len))
        random.shuffle(edge_ids)

        colors = [-1] * len(coordsets)
        for (i, j), edge in zip(true_candidate["pairs"], edge_ids):  # type: ignore[assignment]
            si = _side(base_objects[int(i)]["coords"], true_axis)  # type: ignore[arg-type]
            src = int(i) if si < 0 else int(j)
            dst = int(j) if si < 0 else int(i)
            colors[src] = int(cycle[edge])
            colors[dst] = int(cycle[(edge + 1) % cycle_len])
        colors[int(true_candidate["unpaired"])] = int(random.choice(cycle))

        objects = [{"color": int(color), "coords": coords} for color, coords in zip(colors, coordsets)]
        if any(color < 0 for color in colors):
            continue
        if _valid_chain(objects, true_candidate) is None:
            continue
        if _color_mapping_from_pairs(objects, fake_candidate) is not None:
            continue
        return colors
    return None


def _render(side: int, bg: int, coordsets: list[CoordSet], colors: list[int]) -> np.ndarray:
    grid = np.full((side, side), int(bg), dtype=int)
    for coords, color in zip(coordsets, colors):
        for r, c in coords:
            grid[int(r), int(c)] = int(color)
    return grid


def _semantic_accepts(grid: np.ndarray, true_axis: Axis, fake_axis: Axis) -> bool:
    try:
        bg, objects, selected, mapping, geometric = _solve_details(grid)
        out = main(grid)
    except Exception:
        return False

    if selected["axis"] != true_axis:
        return False
    axes = {cand["axis"] for cand in geometric}
    if true_axis not in axes or fake_axis not in axes:
        return False
    if np.array_equal(out, grid):
        return False

    fake_candidates = [cand for cand in geometric if cand["axis"] == fake_axis]
    if len(fake_candidates) != 1:
        return False
    if _side(objects[int(fake_candidates[0]["unpaired"])]["coords"], fake_axis) != -1:  # type: ignore[arg-type]
        return False
    if _color_mapping_from_pairs(objects, fake_candidates[0]) is not None:
        return False

    unpaired = int(selected["unpaired"])
    target = _reflect_set(objects[unpaired]["coords"], selected["axis"])  # type: ignore[arg-type]
    added = [(r, c) for r, c in target if int(grid[r, c]) == int(bg)]
    if len(added) != len(target):
        return False
    expected_color = int(mapping[int(objects[unpaired]["color"])])
    return all(int(out[r, c]) == expected_color for r, c in target)


def _gen_attempt() -> np.ndarray | None:
    side = random.randint(17, 29)
    bg = random.choice(ARC_COLORS)
    true_kind, fake_kind = random.sample(GLOBAL_KINDS, 2)
    true_axis = _global_axis(true_kind, side)
    fake_axis = _global_axis(fake_kind, side)
    axes = (true_axis, fake_axis)
    expected_size = _max_orbit_size(side, axes)
    if expected_size < 4:
        return None

    full_count = 1 if expected_size >= 8 else random.randint(1, 2)
    occupied_halo: set[Coord] = set()
    coordsets: list[CoordSet] = []

    for _ in range(full_count):
        placed = _place_orbit(side, axes, occupied_halo, expected_size, require_removable=False)
        if placed is None:
            return None
        orbit, _removed = placed
        coordsets.extend(orbit)

    placed = _place_orbit(side, axes, occupied_halo, expected_size, require_removable=True)
    if placed is None:
        return None
    orbit, removed_idx = placed
    assert removed_idx is not None
    coordsets.extend(coords for idx, coords in enumerate(orbit) if idx != int(removed_idx))

    colors = _assign_colors(coordsets, true_axis, fake_axis, int(bg))
    if colors is None:
        return None
    grid = _render(side, int(bg), coordsets, colors)
    if not _semantic_accepts(grid, true_axis, fake_axis):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid fable/5/003/7 input")
