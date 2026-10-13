from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox_coords, components_mask


RED = Color.RED
BLACK = Color.BLACK
GREY = Color.GREY
YELLOW = Color.YELLOW
BLUE = Color.BLUE
GREEN = Color.GREEN
ORANGE = Color.ORANGE

SYMMETRY_TO_COLOR = {"H": BLUE, "V": GREEN, "D": ORANGE}


def _mask_from_coords(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = map(int, bbox)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _symmetry_kind(mask: np.ndarray) -> str:
    horizontal = bool(np.array_equal(mask, mask[::-1, :]))
    vertical = bool(np.array_equal(mask, mask[:, ::-1]))
    diagonal = bool(
        mask.shape[0] == mask.shape[1]
        and (
            np.array_equal(mask, mask.T)
            or np.array_equal(mask, np.flipud(np.fliplr(mask)).T)
        )
    )
    labels = [name for name, ok in (("H", horizontal), ("V", vertical), ("D", diagonal)) if ok]
    assert len(labels) == 1, "each non-singleton object must have exactly one symmetry kind"
    return labels[0]


def _analyze_red_objects(grid: np.ndarray) -> tuple[list[dict], list[np.ndarray]]:
    comps = components_mask(grid == RED, connectivity=4)
    objects: list[dict] = []
    singletons: list[np.ndarray] = []
    for coords in comps:
        if int(coords.shape[0]) == 1:
            singletons.append(coords)
            continue
        bbox = bbox_coords(coords)
        mask = _mask_from_coords(coords, bbox)
        kind = _symmetry_kind(mask)
        objects.append(
            {
                "coords": coords.astype(int),
                "bbox": bbox,
                "mask": mask,
                "kind": kind,
                "color": SYMMETRY_TO_COLOR[kind],
                "area": int(coords.shape[0]),
            }
        )
    return objects, singletons


def _dist_map_to_coords(shape: tuple[int, int], coords: np.ndarray) -> np.ndarray:
    h, w = map(int, shape)
    rows = np.arange(h)[:, None]
    cols = np.arange(w)[None, :]
    dist = np.full((h, w), h + w + 10, dtype=int)
    for r, c in coords:
        dist = np.minimum(dist, np.maximum(np.abs(rows - int(r)), np.abs(cols - int(c))))
    return dist


def _nearest_blocker_distance(coords: np.ndarray, blockers: np.ndarray) -> int:
    dist = np.abs(coords[:, None, :] - blockers[None, :, :])
    return int(np.maximum(dist[..., 0], dist[..., 1]).min())


def _solve(grid: np.ndarray) -> tuple[np.ndarray, list[dict], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    assert set(map(int, np.unique(g))).issubset({BLACK, RED, GREY})

    objects, singletons = _analyze_red_objects(g)
    blockers = np.argwhere(g == GREY)
    assert 2 <= len(blockers) <= 4, "expected 2-4 blockers"
    assert 3 <= len(objects) <= 5, "expected 3-5 non-singleton red objects"

    clean = g.copy()
    for coords in singletons:
        clean[coords[:, 0], coords[:, 1]] = BLACK

    out = clean.copy()
    claim_dist = np.full(g.shape, g.shape[0] + g.shape[1] + 10, dtype=int)
    claim_color = np.full(g.shape, -1, dtype=int)
    bg_mask = clean == BLACK

    for obj in objects:
        coords = obj["coords"]
        obj["D"] = _nearest_blocker_distance(coords, blockers)
        assert 2 <= obj["D"] <= 5
        out[coords[:, 0], coords[:, 1]] = int(obj["color"])

        dist_map = _dist_map_to_coords(g.shape, coords)
        obj["dist_map"] = dist_map
        paintable = bg_mask & (dist_map >= 1) & (dist_map < obj["D"])
        better = paintable & (dist_map < claim_dist)
        claim_dist[better] = dist_map[better]
        claim_color[better] = np.where(
            dist_map[better] % 2 == 1,
            int(obj["color"]),
            YELLOW,
        )

    fill_mask = claim_color >= 0
    out[fill_mask] = claim_color[fill_mask]
    return out, objects, np.asarray(singletons, dtype=object)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each red symmetric object and surround it with blocker-limited Chebyshev halo rings.

    Input structure:
    - A black background contains 3-5 red 4-connected foreground objects of size 3-6 and 2-4 grey blocker pixels.
    - Some red singleton pixels may appear as noise and must be erased before object analysis.
    - Every non-singleton red object has exactly one reflection-symmetry kind among horizontal, vertical, or diagonal.

    Rule:
    - Remove every red 4-connected component of area 1.
    - For each remaining red object, detect its unique symmetry kind and recolor the object itself:
      horizontal -> blue, vertical -> green, diagonal -> orange.
    - Let D be the minimum Chebyshev distance from any pixel of that object to any grey blocker pixel.
    - For each Chebyshev layer d = 1..D-1 from the object pixel set, color background cells on odd d with the
      object's symmetry color and on even d with yellow; never overwrite grey blockers or already-nonblack input cells.

    Key insight:
    - The halo color pattern comes from the object's symmetry type, while the halo radius comes from an independent
      geometric measurement to the nearest blocker, so both analyses are required to determine the output.
    """
    out, _objects, _singletons = _solve(grid)
    return out


_H_MASKS = [
    np.array([[1, 0], [1, 1], [1, 0]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 0], [1, 1, 0]], dtype=bool),
]
_V_MASKS = [mask.T.copy() for mask in _H_MASKS]
_D_MASKS = [
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[1, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[0, 1, 1], [1, 1, 1], [0, 1, 0]], dtype=bool),
]
_MASKS_BY_KIND = {"H": _H_MASKS, "V": _V_MASKS, "D": _D_MASKS}


for _kind, _masks in _MASKS_BY_KIND.items():
    for _mask in _masks:
        assert _symmetry_kind(_mask) == _kind
        assert 3 <= int(_mask.sum()) <= 6


def _placed_bbox(top: tuple[int, int], mask: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = map(int, top)
    h, w = mask.shape
    return r0, c0, r0 + h - 1, c0 + w - 1


def _expand_bbox(box: tuple[int, int, int, int], radius: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = map(int, box)
    d = int(radius)
    return r0 - d, c0 - d, r1 + d, c1 + d


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = map(int, a)
    br0, bc0, br1, bc1 = map(int, b)
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _render_coords(mask: np.ndarray, top: tuple[int, int]) -> np.ndarray:
    r0, c0 = map(int, top)
    coords = np.argwhere(mask)
    return coords + np.array([[r0, c0]])


def _cheb_between_coords(a: np.ndarray, b: np.ndarray) -> int:
    dist = np.abs(a[:, None, :] - b[None, :, :])
    return int(np.maximum(dist[..., 0], dist[..., 1]).min())


def _sample_spec() -> list[dict]:
    count = random.choice([3, 4])
    kinds = ["H", "V", "D"]
    if count == 4:
        kinds.append(random.choice(kinds))
    random.shuffle(kinds)

    shared_d = random.randint(2, 5)
    pair = random.sample(range(count), 2)
    while kinds[pair[0]] == kinds[pair[1]]:
        pair = random.sample(range(count), 2)

    specs = []
    for idx, kind in enumerate(kinds):
        specs.append(
            {
                "kind": kind,
                "mask": random.choice(_MASKS_BY_KIND[kind]).copy(),
                "D": shared_d if idx in pair else random.randint(2, 5),
            }
        )

    by_kind: dict[str, list[int]] = {}
    for idx, spec in enumerate(specs):
        by_kind.setdefault(spec["kind"], []).append(idx)
    for idxs in by_kind.values():
        if len(idxs) >= 2:
            base = specs[idxs[0]]["D"]
            if all(specs[i]["D"] == base for i in idxs[1:]):
                choices = [d for d in range(2, 6) if d != base]
                specs[idxs[-1]]["D"] = random.choice(choices)

    random.shuffle(specs)
    return specs


def _place_objects(height: int, width: int, specs: list[dict]) -> list[dict] | None:
    placed: list[dict] = []
    order = sorted(
        specs,
        key=lambda spec: (spec["mask"].shape[0] + 2 * (spec["D"] - 1)) * (spec["mask"].shape[1] + 2 * (spec["D"] - 1)),
        reverse=True,
    )

    for spec in order:
        mask = spec["mask"]
        halo = int(spec["D"]) - 1
        h, w = mask.shape
        positions = []
        for r0 in range(halo, height - h - halo + 1):
            for c0 in range(halo, width - w - halo + 1):
                bbox = _placed_bbox((r0, c0), mask)
                halo_box = _expand_bbox(bbox, halo)
                if any(_boxes_overlap(halo_box, other["halo_box"]) for other in placed):
                    continue
                positions.append((r0, c0))
        if not positions:
            return None
        top = random.choice(positions)
        coords = _render_coords(mask, top)
        bbox = _placed_bbox(top, mask)
        spec = dict(spec)
        spec["top"] = top
        spec["coords"] = coords
        spec["bbox"] = bbox
        spec["halo_box"] = _expand_bbox(bbox, halo)
        spec["dist_map"] = _dist_map_to_coords((height, width), coords)
        placed.append(spec)

    random.shuffle(placed)
    return placed


def _candidate_blockers(height: int, width: int, idx: int, placed: list[dict], chosen: set[tuple[int, int]]) -> list[tuple[int, int]]:
    obj = placed[idx]
    exact = obj["dist_map"] == int(obj["D"])
    candidates = []
    occupied = chosen.copy()
    for other in placed:
        occupied.update(map(tuple, other["coords"]))

    for r, c in np.argwhere(exact):
        cell = (int(r), int(c))
        if cell in occupied:
            continue
        ok = True
        for j, other in enumerate(placed):
            dist = int(other["dist_map"][r, c])
            if j == idx:
                if dist != int(obj["D"]):
                    ok = False
                    break
            elif dist < int(other["D"]):
                ok = False
                break
        if ok:
            candidates.append(cell)
    return candidates


def _place_blockers(height: int, width: int, placed: list[dict]) -> set[tuple[int, int]] | None:
    chosen: set[tuple[int, int]] = set()
    order = list(range(len(placed)))
    order.sort(key=lambda idx: placed[idx]["D"], reverse=True)
    for idx in order:
        candidates = _candidate_blockers(height, width, idx, placed, chosen)
        if not candidates:
            return None
        chosen.add(random.choice(candidates))
    return chosen


def _place_singletons(
    height: int,
    width: int,
    placed: list[dict],
    blockers: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    want_noise = random.random() < 0.8
    target = random.randint(1, 3) if want_noise else 0
    noises: list[tuple[int, int]] = []
    if target == 0:
        return noises

    occupied = set(blockers)
    for obj in placed:
        occupied.update(map(tuple, obj["coords"]))

    def valid(cell: tuple[int, int]) -> bool:
        if cell in occupied:
            return False
        if any(max(abs(cell[0] - r), abs(cell[1] - c)) <= 1 for r, c in blockers):
            return False
        for obj in placed:
            if int(obj["dist_map"][cell[0], cell[1]]) <= int(obj["D"]):
                return False
        if any(abs(cell[0] - r) + abs(cell[1] - c) <= 1 for r, c in noises):
            return False
        return True

    cells = [(r, c) for r in range(height) for c in range(width)]
    random.shuffle(cells)
    for cell in cells:
        if valid(cell):
            noises.append(cell)
            occupied.add(cell)
            if len(noises) == target:
                break
    return noises if len(noises) == target else []


def _oracle_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    objects, singletons = _analyze_red_objects(g)
    blockers = np.argwhere(g == GREY)
    clean = g.copy()
    for coords in singletons:
        clean[coords[:, 0], coords[:, 1]] = BLACK

    out = clean.copy()
    proposals: dict[tuple[int, int], tuple[int, int]] = {}
    for obj in objects:
        coords = obj["coords"]
        kind = obj["kind"]
        color = SYMMETRY_TO_COLOR[kind]
        D = _nearest_blocker_distance(coords, blockers)
        out[coords[:, 0], coords[:, 1]] = color
        for r in range(g.shape[0]):
            for c in range(g.shape[1]):
                if clean[r, c] != BLACK:
                    continue
                d = min(max(abs(r - rr), abs(c - cc)) for rr, cc in coords)
                if 1 <= d < D:
                    prev = proposals.get((r, c))
                    paint = color if d % 2 == 1 else YELLOW
                    if prev is None or d < prev[0]:
                        proposals[(r, c)] = (d, paint)
    for (r, c), (_d, color) in proposals.items():
        out[r, c] = color
    return out


def _validate_sample(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or max(g.shape) > 30:
        return False
    if not set(map(int, np.unique(g))).issubset({BLACK, RED, GREY}):
        return False

    objects, singletons = _analyze_red_objects(g)
    blockers = np.argwhere(g == GREY)
    if not (3 <= len(objects) <= 5 and 2 <= len(blockers) <= 4):
        return False
    if {obj["kind"] for obj in objects} != {"H", "V", "D"}:
        return False
    if any(obj["area"] < 3 or obj["area"] > 6 for obj in objects):
        return False

    Ds = []
    halo_maps = []
    for obj in objects:
        D = _nearest_blocker_distance(obj["coords"], blockers)
        if not (2 <= D <= 5):
            return False
        Ds.append(D)
        halo_map = _dist_map_to_coords(g.shape, obj["coords"]) < D
        halo_maps.append(halo_map)

    if len(set(Ds)) < 2:
        return False

    repeated_kind_diff_radius = False
    shared_radius_diff_kind = False
    for i, obj_i in enumerate(objects):
        for j, obj_j in enumerate(objects):
            if i >= j:
                continue
            if obj_i["kind"] == obj_j["kind"] and Ds[i] != Ds[j]:
                repeated_kind_diff_radius = True
            if obj_i["kind"] != obj_j["kind"] and Ds[i] == Ds[j]:
                shared_radius_diff_kind = True
    if not (repeated_kind_diff_radius or shared_radius_diff_kind):
        return False

    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            if np.any(halo_maps[i] & halo_maps[j]):
                return False

    for coords in singletons:
        r, c = map(int, coords[0])
        if any(max(abs(r - br), abs(c - bc)) <= 1 for br, bc in blockers):
            return False

    expected = _oracle_output(g)
    try:
        actual = main(g)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False
    if np.array_equal(actual, g):
        return False
    return True


def _attempt() -> np.ndarray | None:
    height = random.randint(18, 26)
    width = random.randint(18, 26)
    specs = _sample_spec()
    placed = _place_objects(height, width, specs)
    if placed is None:
        return None

    blockers = _place_blockers(height, width, placed)
    if blockers is None:
        return None

    noises = _place_singletons(height, width, placed, blockers)
    grid = np.zeros((height, width), dtype=int)
    for obj in placed:
        coords = obj["coords"]
        grid[coords[:, 0], coords[:, 1]] = RED
    for r, c in blockers:
        grid[r, c] = GREY
    for r, c in noises:
        grid[r, c] = RED

    if _validate_sample(grid):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
