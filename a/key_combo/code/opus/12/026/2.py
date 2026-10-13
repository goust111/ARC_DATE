from __future__ import annotations

import random
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np

try:
    from arc_utils import Color, bbox_coords, components_mask
except ModuleNotFoundError:
    _ARC_UTILS = SourceFileLoader(
        "_arc_utils_opus_12_026_2",
        str(Path("/home/lixinyu/snap/new/solvers/arc_utils.py")),
    ).load_module()
    Color = _ARC_UTILS.Color
    bbox_coords = _ARC_UTILS.bbox_coords
    components_mask = _ARC_UTILS.components_mask


BBox = tuple[int, int, int, int]
Coord = tuple[int, int]


GRID_SIDE_RANGE = (14, 18)
DOMINO_COUNT_RANGE = (2, 5)
SINGLE_COUNT_RANGE = (3, 8)
GEN_RETRIES = 400
PAIR_SEARCH_TRIES = 500


def _expanded_bbox(bbox: BBox, shape: tuple[int, int]) -> BBox:
    h, w = shape
    r0, c0, r1, c1 = bbox
    return max(0, r0 - 1), max(0, c0 - 1), min(h - 1, r1 + 1), min(w - 1, c1 + 1)


def _perimeter_coords(box: BBox) -> set[Coord]:
    r0, c0, r1, c1 = box
    coords: set[Coord] = set()
    for c in range(c0, c1 + 1):
        coords.add((r0, c))
        coords.add((r1, c))
    for r in range(r0, r1 + 1):
        coords.add((r, c0))
        coords.add((r, c1))
    return coords


def _rects_intersect(a: BBox, b: BBox) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _cells_are_4_separate(a: tuple[Coord, ...], b: tuple[Coord, ...]) -> bool:
    for r0, c0 in a:
        for r1, c1 in b:
            if abs(r0 - r1) + abs(c0 - c1) <= 1:
                return False
    return True


def _domino_orientation(coords: tuple[Coord, Coord]) -> str:
    return "H" if coords[0][0] == coords[1][0] else "V"


def _make_domino(coords: tuple[Coord, Coord], shape: tuple[int, int]) -> dict:
    pts = tuple(sorted((tuple(map(int, xy)) for xy in coords)))
    bbox = bbox_coords(np.array(pts, dtype=int))
    expanded = _expanded_bbox(bbox, shape)
    return {
        "pixels": pts,
        "bbox": bbox,
        "expanded": expanded,
        "outline": _perimeter_coords(expanded),
        "orientation": _domino_orientation(pts),
    }


def _ray_dirs(orientation: str) -> tuple[Coord, Coord]:
    return ((-1, 0), (1, 0)) if orientation == "H" else ((0, -1), (0, 1))


def _all_domino_candidates(shape: tuple[int, int]) -> list[dict]:
    h, w = shape
    out: list[dict] = []
    for r in range(h):
        for c in range(w - 1):
            out.append(_make_domino(((r, c), (r, c + 1)), shape))
    for r in range(h - 1):
        for c in range(w):
            out.append(_make_domino(((r, c), (r + 1, c)), shape))
    return out


def _domino_fits(candidate: dict, placed: list[dict]) -> bool:
    for other in placed:
        if not _cells_are_4_separate(candidate["pixels"], other["pixels"]):
            return False
        if _rects_intersect(candidate["expanded"], other["expanded"]):
            return False
    return True


def _touches_edge(domino: dict, shape: tuple[int, int]) -> bool:
    h, w = shape
    r0, c0, r1, c1 = domino["bbox"]
    return r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1


def _trace_ray(
    pixel: Coord,
    domino_index: int,
    direction: Coord,
    shape: tuple[int, int],
    red_cells: set[Coord],
    own_outline: set[Coord],
    outline_owner: dict[Coord, int],
) -> tuple[list[Coord], tuple[str, Coord, int | None] | None]:
    h, w = shape
    dr, dc = direction
    rr, cc = pixel[0] + dr, pixel[1] + dc
    path: list[Coord] = []
    while 0 <= rr < h and 0 <= cc < w:
        coord = (rr, cc)
        if coord in own_outline:
            rr += dr
            cc += dc
            continue
        owner = outline_owner.get(coord)
        if owner is not None:
            return path, ("outline", coord, owner)
        if coord in red_cells:
            return path, ("red", coord, None)
        path.append(coord)
        rr += dr
        cc += dc
    return path, None


def _predict_paths(dominoes: list[dict], singles: set[Coord], shape: tuple[int, int]):
    red_cells = set(singles)
    outline_owner: dict[Coord, int] = {}
    for idx, domino in enumerate(dominoes):
        red_cells.update(domino["pixels"])
        for coord in domino["outline"]:
            outline_owner[coord] = idx

    paths: list[set[Coord]] = []
    blockers: list[list[tuple[str, Coord, int | None]]] = []
    for idx, domino in enumerate(dominoes):
        path_cells: set[Coord] = set()
        hit_info: list[tuple[str, Coord, int | None]] = []
        for pixel in domino["pixels"]:
            for direction in _ray_dirs(domino["orientation"]):
                segment, blocker = _trace_ray(
                    pixel,
                    idx,
                    direction,
                    shape,
                    red_cells,
                    domino["outline"],
                    outline_owner,
                )
                path_cells.update(segment)
                if blocker is not None:
                    hit_info.append(blocker)
        paths.append(path_cells)
        blockers.append(hit_info)
    return paths, blockers


def _render_input(shape: tuple[int, int], dominoes: list[dict], singles: set[Coord]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for r, c in singles:
        grid[r, c] = Color.RED
    for domino in dominoes:
        for r, c in domino["pixels"]:
            grid[r, c] = Color.RED
    return grid


def _render_expected(shape: tuple[int, int], dominoes: list[dict], singles: set[Coord]) -> np.ndarray:
    out = _render_input(shape, dominoes, singles)
    for domino in dominoes:
        for r, c in sorted(domino["outline"]):
            if out[r, c] == Color.BLACK:
                out[r, c] = Color.GREEN
    paths, _ = _predict_paths(dominoes, singles, shape)
    for path in paths:
        for r, c in path:
            out[r, c] = Color.TEAL
    return out


def _blocked_by_other_outline(blockers: list[tuple[str, Coord, int | None]], self_index: int) -> bool:
    return any(kind == "outline" and owner is not None and owner != self_index for kind, _coord, owner in blockers)


def _metadata_is_valid(shape: tuple[int, int], dominoes: list[dict], singles: set[Coord]) -> bool:
    if not dominoes or any(len(domino["pixels"]) != 2 for domino in dominoes):
        return False
    for i, domino in enumerate(dominoes):
        for other in dominoes[i + 1 :]:
            if not _cells_are_4_separate(domino["pixels"], other["pixels"]):
                return False
            if _rects_intersect(domino["expanded"], other["expanded"]):
                return False

    paths, blockers = _predict_paths(dominoes, singles, shape)
    if not any(path for path in paths):
        return False

    seen: dict[Coord, int] = {}
    for idx, path in enumerate(paths):
        for coord in path:
            old = seen.get(coord)
            if old is not None and old != idx:
                return False
            seen[coord] = idx

    if not any(_blocked_by_other_outline(hit_info, idx) for idx, hit_info in enumerate(blockers)):
        return False

    inp = _render_input(shape, dominoes, singles)
    red_components = components_mask(inp == Color.RED, connectivity=4)
    if any(int(comp.shape[0]) not in (1, 2) for comp in red_components):
        return False

    expected = _render_expected(shape, dominoes, singles)
    try:
        actual = main(inp)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def _choose_blocked_pair(shape: tuple[int, int], candidates: list[dict]) -> list[dict] | None:
    source_orientation = random.choice(("H", "V"))
    sources = [candidate for candidate in candidates if candidate["orientation"] == source_orientation]
    if not sources:
        return None
    for _ in range(PAIR_SEARCH_TRIES):
        source = random.choice(sources)
        blocker = random.choice(candidates)
        if source["pixels"] == blocker["pixels"]:
            continue
        if not _domino_fits(blocker, [source]):
            continue
        paths, blockers = _predict_paths([source, blocker], set(), shape)
        if paths[0] and _blocked_by_other_outline(blockers[0], 0):
            return [source, blocker]
    return None


def _choose_extra_domino(
    shape: tuple[int, int],
    candidates: list[dict],
    placed: list[dict],
    *,
    orientation: str | None = None,
    edge_only: bool = False,
) -> dict | None:
    pool = []
    for candidate in candidates:
        if orientation is not None and candidate["orientation"] != orientation:
            continue
        if edge_only and not _touches_edge(candidate, shape):
            continue
        if _domino_fits(candidate, placed):
            pool.append(candidate)
    return random.choice(pool) if pool else None


def _single_candidates(shape: tuple[int, int], dominoes: list[dict], predicted_paths: list[set[Coord]]) -> list[Coord]:
    h, w = shape
    red_cells = {cell for domino in dominoes for cell in domino["pixels"]}
    forbidden = set().union(*(domino["outline"] for domino in dominoes))
    forbidden.update(red_cells)
    forbidden.update(*predicted_paths)
    for domino in dominoes:
        r0, c0, r1, c1 = domino["expanded"]
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                forbidden.add((r, c))

    out = []
    for r in range(h):
        for c in range(w):
            coord = (r, c)
            if coord in forbidden:
                continue
            if any(abs(r - rr) + abs(c - cc) <= 1 for rr, cc in red_cells):
                continue
            out.append(coord)
    return out


def _gen_attempt() -> np.ndarray | None:
    shape = (
        random.randint(GRID_SIDE_RANGE[0], GRID_SIDE_RANGE[1]),
        random.randint(GRID_SIDE_RANGE[0], GRID_SIDE_RANGE[1]),
    )
    target_dominoes = random.randint(DOMINO_COUNT_RANGE[0], DOMINO_COUNT_RANGE[1])
    target_singles = random.randint(SINGLE_COUNT_RANGE[0], SINGLE_COUNT_RANGE[1])
    candidates = _all_domino_candidates(shape)

    dominoes = _choose_blocked_pair(shape, candidates)
    if dominoes is None:
        return None

    orientations = {domino["orientation"] for domino in dominoes}
    if len(orientations) == 1:
        extra = _choose_extra_domino(
            shape,
            candidates,
            dominoes,
            orientation="V" if next(iter(orientations)) == "H" else "H",
        )
        if extra is None:
            return None
        dominoes.append(extra)

    if random.random() < 0.45 and not any(_touches_edge(domino, shape) for domino in dominoes):
        edge_domino = _choose_extra_domino(shape, candidates, dominoes, edge_only=True)
        if edge_domino is not None:
            dominoes.append(edge_domino)

    while len(dominoes) < target_dominoes:
        extra = _choose_extra_domino(shape, candidates, dominoes)
        if extra is None:
            break
        dominoes.append(extra)

    predicted_paths, _ = _predict_paths(dominoes, set(), shape)
    singles = set()
    options = _single_candidates(shape, dominoes, predicted_paths)
    random.shuffle(options)
    for coord in options:
        if len(singles) >= target_singles:
            break
        if any(abs(coord[0] - rr) + abs(coord[1] - cc) <= 1 for rr, cc in singles):
            continue
        singles.add(coord)
    if len(singles) < target_singles:
        return None

    if not _metadata_is_valid(shape, dominoes, singles):
        return None
    return _render_input(shape, dominoes, singles)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw green outline frames around red dominoes, then emit teal rays perpendicular to each domino until blocked.

    Input structure:
    - The grid is a black background containing only red foreground pixels.
    - Every 4-connected red object is either a single pixel or a 2-pixel horizontal/vertical domino.
    - Singles are inert; only dominoes participate in the added markings.

    Rule:
    - Preserve all original red pixels.
    - For each red domino, expand its bounding box by one cell, clip it to the grid, and paint the perimeter green wherever that perimeter cell is black.
    - After all green outlines are in place, shoot teal rays from each domino pixel in the two directions perpendicular to the domino axis.
    - A ray passes through the emitting domino's own outline, paints black cells teal, and stops before the first other non-black cell it encounters.

    Key insight:
    - The green outline created around one domino becomes a visible wall that can stop another domino's perpendicular teal rays.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(map(int, np.unique(g))).issubset({Color.BLACK, Color.RED})

    out = g.copy()
    dominoes: list[dict] = []
    for comp in components_mask(g == Color.RED, connectivity=4):
        area = int(comp.shape[0])
        assert area in (1, 2), "red objects must be singles or dominoes"
        if area == 1:
            continue
        bbox = bbox_coords(comp)
        assert (bbox[2] - bbox[0]) + (bbox[3] - bbox[1]) == 1, "2-pixel object must be a domino"
        coords = tuple(sorted((tuple(map(int, xy)) for xy in comp)))
        dominoes.append(_make_domino(coords, g.shape))

    assert dominoes, "at least one domino is required"

    for domino in dominoes:
        for r, c in sorted(domino["outline"]):
            if out[r, c] == Color.BLACK:
                out[r, c] = Color.GREEN

    h, w = g.shape
    for domino in dominoes:
        for r, c in domino["pixels"]:
            for dr, dc in _ray_dirs(domino["orientation"]):
                rr, cc = r + dr, c + dc
                while 0 <= rr < h and 0 <= cc < w:
                    coord = (rr, cc)
                    if coord in domino["outline"] and out[rr, cc] == Color.GREEN:
                        rr += dr
                        cc += dc
                        continue
                    if out[rr, cc] != Color.BLACK:
                        break
                    out[rr, cc] = Color.TEAL
                    rr += dr
                    cc += dc
    return out


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for opus/12/026/2")
