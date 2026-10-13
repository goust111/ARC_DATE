from __future__ import annotations

from collections import Counter, defaultdict, deque
from itertools import combinations
import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]

BG = 0
GRAY = 5
GLYPH_SIZE = 6
REFERENCE_SIZE = 4

_NEIGHBORS4: tuple[Coord, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
_NEIGHBORS8: tuple[Coord, ...] = tuple(
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if dr != 0 or dc != 0
)

# A visibly asymmetric gray reference shape.  Generated dot clusters are
# connected anchored subsets of its rotations.
_BASE_REF: frozenset[Coord] = frozenset(
    {
        (0, 0),
        (0, 2),
        (0, 3),
        (1, 1),
        (2, 0),
        (3, 1),
        (3, 2),
        (3, 3),
    }
)

_GLYPH_SPECS: dict[int, dict[str, tuple[Coord, ...]]] = {
    2: {
        "holes": ((1, 1), (1, 2), (3, 4)),
        "notches": ((0, 4), (0, 5), (1, 5)),
    },
    3: {
        "holes": ((1, 1), (2, 3), (2, 4), (4, 2)),
        "notches": ((5, 4), (5, 5)),
    },
    4: {
        "holes": ((1, 1), (1, 3), (3, 2), (4, 4)),
        "notches": ((0, 5),),
    },
    5: {
        "holes": ((1, 1), (1, 3), (2, 4), (3, 2), (4, 4)),
        "notches": (),
    },
}


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors(connectivity: int) -> tuple[Coord, ...]:
    assert connectivity in (4, 8)
    return _NEIGHBORS4 if int(connectivity) == 4 else _NEIGHBORS8


def _bbox(coords: set[Coord] | list[Coord] | np.ndarray) -> BBox:
    if isinstance(coords, np.ndarray):
        rows = coords[:, 0]
        cols = coords[:, 1]
        return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())
    pts = list(coords)
    rows = [int(r) for r, _ in pts]
    cols = [int(c) for _, c in pts]
    return min(rows), min(cols), max(rows), max(cols)


def _bbox_size(box: BBox) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, box)
    return int(r1 - r0 + 1), int(c1 - c0 + 1)


def _normalize(coords: set[Coord] | frozenset[Coord] | list[Coord]) -> frozenset[Coord]:
    pts = [(int(r), int(c)) for r, c in coords]
    r0 = min(r for r, _ in pts)
    c0 = min(c for _, c in pts)
    return frozenset((r - r0, c - c0) for r, c in pts)


def _component_masks(mask: np.ndarray, connectivity: int = 4) -> list[set[Coord]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[set[Coord]] = []
    nbrs = _neighbors(connectivity)
    for sr in range(h):
        for sc in range(w):
            if seen[sr, sc] or not bool(mask[sr, sc]):
                continue
            q: deque[Coord] = deque([(int(sr), int(sc))])
            seen[sr, sc] = True
            comp: set[Coord] = set()
            while q:
                r, c = q.popleft()
                comp.add((int(r), int(c)))
                for dr, dc in nbrs:
                    nr = int(r + dr)
                    nc = int(c + dc)
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and bool(mask[nr, nc])
                    ):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _components_by_color(g: np.ndarray, bg: int) -> list[dict]:
    comps: list[dict] = []
    for color in sorted(int(c) for c in np.unique(g) if int(c) != int(bg)):
        for comp in _component_masks(g == int(color), connectivity=8):
            box = _bbox(comp)
            comps.append(
                {
                    "color": int(color),
                    "coords": comp,
                    "bbox": box,
                    "area": int(len(comp)),
                }
            )
    return comps


def _crop(g: np.ndarray, box: BBox) -> np.ndarray:
    r0, c0, r1, c1 = map(int, box)
    return g[r0 : r1 + 1, c0 : c1 + 1].copy()


def _coords_to_mask(coords: set[Coord] | frozenset[Coord], shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    for r, c in coords:
        out[int(r), int(c)] = True
    return out


def _mask_to_coords(mask: np.ndarray) -> frozenset[Coord]:
    return frozenset((int(r), int(c)) for r, c in np.argwhere(mask))


def _rotate_bool(mask: np.ndarray, turns: int) -> np.ndarray:
    return np.rot90(mask, int(turns) % 4).copy()


def _rotate_coords_in_square(coords: set[Coord] | frozenset[Coord], size: int, turns: int) -> frozenset[Coord]:
    mask = _coords_to_mask(coords, (int(size), int(size)))
    return _mask_to_coords(_rotate_bool(mask, turns))


def _is_connected(coords: set[Coord] | frozenset[Coord], connectivity: int = 8) -> bool:
    pts = set(coords)
    if not pts:
        return False
    seen = {next(iter(pts))}
    stack = list(seen)
    nbrs = _neighbors(connectivity)
    while stack:
        r, c = stack.pop()
        for dr, dc in nbrs:
            nxt = (int(r + dr), int(c + dc))
            if nxt in pts and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(pts)


def _make_glyph(hole_count: int, color: int) -> np.ndarray:
    spec = _GLYPH_SPECS[int(hole_count)]
    out = np.full((GLYPH_SIZE, GLYPH_SIZE), int(color), dtype=int)
    for r, c in spec["holes"] + spec["notches"]:
        out[int(r), int(c)] = BG
    return out


def _count_holes(patch: np.ndarray, bg: int) -> int:
    zeros = patch == int(bg)
    count = 0
    h, w = patch.shape
    for comp in _component_masks(zeros, connectivity=4):
        if not any(r in (0, h - 1) or c in (0, w - 1) for r, c in comp):
            count += 1
    return int(count)


def _paint_patch(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int = BG) -> None:
    h, w = patch.shape
    assert 0 <= int(r0) and 0 <= int(c0)
    assert int(r0) + h <= dst.shape[0] and int(c0) + w <= dst.shape[1]
    view = dst[int(r0) : int(r0) + h, int(c0) : int(c0) + w]
    mask = patch != int(bg)
    view[mask] = patch[mask]


def _rotations_of_reference(ref_patch: np.ndarray, bg: int) -> list[frozenset[Coord]]:
    assert ref_patch.shape[0] == ref_patch.shape[1], "reference piece must have a square bbox"
    ref = ref_patch != int(bg)
    rots = [_mask_to_coords(_rotate_bool(ref, k)) for k in range(4)]
    assert len(set(rots)) == 4, "gray reference piece must have no rotational symmetry"
    return rots


def _infer_rotation(cluster_coords: set[Coord], ref_rots: list[frozenset[Coord]]) -> int:
    local = _normalize(cluster_coords)
    matches = [k for k, ref in enumerate(ref_rots) if local <= ref]
    assert len(matches) == 1, "dot cluster must identify exactly one rotation relative to the gray reference"
    return int(matches[0])


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    assert int(bg) == BG, "expected black background"

    comps = _components_by_color(g, bg)
    glyphs = []
    others = []
    for comp in comps:
        h, w = _bbox_size(comp["bbox"])
        if h == GLYPH_SIZE and w == GLYPH_SIZE and int(comp["area"]) >= 24:
            glyphs.append(comp)
        else:
            others.append(comp)

    assert len(glyphs) >= 4, "expected a glyph dictionary"
    legend_colors = {int(comp["color"]) for comp in glyphs}
    assert len(legend_colors) == 1, "glyphs must share one legend color"
    legend_color = next(iter(legend_colors))

    holes_to_glyph: dict[int, np.ndarray] = {}
    glyph_areas_by_hole: dict[int, int] = {}
    for comp in glyphs:
        patch = _crop(g, comp["bbox"])
        assert np.all((patch == bg) | (patch == legend_color))
        holes = _count_holes(patch, bg)
        assert holes not in holes_to_glyph, "glyph hole counts must be unique"
        holes_to_glyph[int(holes)] = patch
        glyph_areas_by_hole[int(holes)] = int(np.count_nonzero(patch != bg))

    area_to_holes: dict[int, set[int]] = defaultdict(set)
    for holes, area in glyph_areas_by_hole.items():
        area_to_holes[int(area)].add(int(holes))
    assert any(len(v) > 1 for v in area_to_holes.values()), "glyph area must not uniquely encode the hole count"

    ref_candidates = []
    for comp in others:
        h, w = _bbox_size(comp["bbox"])
        if int(comp["color"]) == GRAY and h == REFERENCE_SIZE and w == REFERENCE_SIZE:
            ref_candidates.append(comp)
    assert len(ref_candidates) == 1, "expected one gray reference piece"
    reference = ref_candidates[0]
    ref_patch = _crop(g, reference["bbox"])
    ref_rots = _rotations_of_reference(ref_patch, bg)

    cluster_comps = []
    for comp in others:
        if comp is reference:
            continue
        area = int(comp["area"])
        assert 2 <= area <= 5, "scene components must be 2..5 pixel dot clusters"
        assert int(comp["color"]) not in {legend_color, GRAY, bg}, "dot colors must not be legend or gray"
        rot = _infer_rotation(comp["coords"], ref_rots)
        assert area in holes_to_glyph, "each cluster count must have a matching glyph hole count"
        cluster_comps.append({**comp, "rotation": int(rot), "count": int(area)})

    assert len(cluster_comps) >= 4, "expected several dot clusters"

    color_to_counts: dict[int, set[int]] = defaultdict(set)
    count_to_colors: dict[int, set[int]] = defaultdict(set)
    rotations_seen: set[int] = set()
    for comp in cluster_comps:
        color_to_counts[int(comp["color"])].add(int(comp["count"]))
        count_to_colors[int(comp["count"])].add(int(comp["color"]))
        rotations_seen.add(int(comp["rotation"]))
    assert any(len(v) > 1 for v in color_to_counts.values()), "need same-color clusters with different counts"
    assert any(len(v) > 1 for v in count_to_colors.values()), "need different-color clusters with the same count"
    assert rotations_seen == {0, 1, 2, 3}, "all four rotation branches must appear"

    return {
        "grid": g,
        "bg": int(bg),
        "glyphs": glyphs,
        "reference": reference,
        "clusters": cluster_comps,
        "holes_to_glyph": holes_to_glyph,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp the glyph whose enclosed-hole count equals each dot-cluster size, rotated by that cluster's pose.

    Input structure:
    - The black background contains a compact legend of same-colored small glyphs, each with a unique number of enclosed background holes.
    - Beside the legend is one gray asymmetric reference piece; its visible orientation is the zero-rotation frame.
    - The scene contains several separated monochrome 8-connected dot clusters of size 2..5; their colors are decoys and may repeat across different sizes.
    - Each dot cluster is an anchored connected subset of one rotation of the gray reference piece, so its bbox top-left fixes its pose.

    Rule:
    - Count enclosed holes in every legend glyph and count pixels in every dot cluster.
    - For each cluster, select the glyph whose hole count equals the cluster pixel count.
    - Rotate that glyph by the cluster's unique rotation relative to the gray reference piece.
    - Clear the whole legend, the gray reference, and the original cluster pixels, then stamp the rotated glyph at the cluster bbox top-left in the cluster color.

    Key insight:
    - The answer is determined by three independent hidden keys: glyph choice comes from hole-count equality, stamp color comes from the cluster itself, and stamp rotation comes from the cluster pose relative to the gray reference.
    """
    parsed = _parse_scene(grid)
    g = parsed["grid"]
    bg = int(parsed["bg"])
    out = g.copy()

    for comp in parsed["glyphs"]:
        for r, c in comp["coords"]:
            out[int(r), int(c)] = bg
    for r, c in parsed["reference"]["coords"]:
        out[int(r), int(c)] = bg
    for comp in parsed["clusters"]:
        for r, c in comp["coords"]:
            out[int(r), int(c)] = bg

    for comp in parsed["clusters"]:
        r0, c0, _r1, _c1 = comp["bbox"]
        glyph = parsed["holes_to_glyph"][int(comp["count"])]
        rotated = np.rot90(glyph, int(comp["rotation"])).copy()
        stamp = np.full_like(rotated, bg)
        stamp[rotated != bg] = int(comp["color"])
        target = out[int(r0) : int(r0) + stamp.shape[0], int(c0) : int(c0) + stamp.shape[1]]
        assert target.shape == stamp.shape, "stamped glyph must fit within the grid"
        assert np.all(target[stamp != bg] == bg), "stamped glyphs must not overlap"
        _paint_patch(out, stamp, int(r0), int(c0), bg=bg)

    return out


# --------------- generator ---------------

GRID_H_RANGE = (24, 30)
GRID_W_RANGE = (28, 30)
GEN_ATTEMPTS = 300


def _good_cluster_subsets(ref_rots: list[frozenset[Coord]]) -> dict[tuple[int, int], list[frozenset[Coord]]]:
    choices: dict[tuple[int, int], list[frozenset[Coord]]] = {}
    for count in range(2, 6):
        for rot in range(4):
            good: list[frozenset[Coord]] = []
            for sub in combinations(sorted(ref_rots[rot]), count):
                pts = frozenset((int(r), int(c)) for r, c in sub)
                if min(r for r, _ in pts) != 0 or min(c for _, c in pts) != 0:
                    continue
                if not _is_connected(pts, connectivity=8):
                    continue
                matches = [k for k, ref in enumerate(ref_rots) if pts <= ref]
                if matches == [rot]:
                    good.append(pts)
            choices[(count, rot)] = good
    return choices


def _anchor_candidates(h: int, w: int, scene_c0: int) -> list[Coord]:
    row_start = random.randint(1, 2)
    col_start = scene_c0 + random.randint(0, 1)
    rows = list(range(row_start, h - GLYPH_SIZE + 1, GLYPH_SIZE + 1))
    cols = list(range(col_start, w - GLYPH_SIZE + 1, GLYPH_SIZE + 1))
    return [(int(r), int(c)) for r in rows for c in cols]


def _semantic_decoys(clusters: list[dict], glyphs: dict[int, np.ndarray]) -> bool:
    color_to_counts: dict[int, set[int]] = defaultdict(set)
    count_to_colors: dict[int, set[int]] = defaultdict(set)
    rotations = set()
    counts = set()
    for cluster in clusters:
        color_to_counts[int(cluster["color"])].add(int(cluster["count"]))
        count_to_colors[int(cluster["count"])].add(int(cluster["color"]))
        rotations.add(int(cluster["rotation"]))
        counts.add(int(cluster["count"]))
    if counts != {2, 3, 4, 5}:
        return False
    if rotations != {0, 1, 2, 3}:
        return False
    if not any(len(v) > 1 for v in color_to_counts.values()):
        return False
    if not any(len(v) > 1 for v in count_to_colors.values()):
        return False

    area_to_holes: dict[int, set[int]] = defaultdict(set)
    for holes, glyph in glyphs.items():
        area_to_holes[int(np.count_nonzero(glyph != BG))].add(int(holes))
    return any(len(v) > 1 for v in area_to_holes.values())


def _place_cluster(grid: np.ndarray, coords: frozenset[Coord], r0: int, c0: int, color: int) -> set[Coord]:
    placed: set[Coord] = set()
    for r, c in coords:
        rr = int(r0 + r)
        cc = int(c0 + c)
        assert int(grid[rr, cc]) == BG
        grid[rr, cc] = int(color)
        placed.add((rr, cc))
    return placed


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    scene_c0 = 15
    grid = np.full((h, w), BG, dtype=int)
    expected = np.full((h, w), BG, dtype=int)

    palette = [c for c in range(1, 10) if c != GRAY]
    legend_color = random.choice(palette)
    cluster_palette = [c for c in palette if c != legend_color]
    if len(cluster_palette) < 4:
        return None
    cluster_colors = random.sample(cluster_palette, 4)

    glyphs = {holes: _make_glyph(holes, legend_color) for holes in (2, 3, 4, 5)}
    glyph_origins = [(1, 1), (1, 8), (8, 1), (8, 8)]
    glyph_items = list(glyphs.items())
    random.shuffle(glyph_items)
    for (holes, glyph), (r0, c0) in zip(glyph_items, glyph_origins):
        _paint_patch(grid, glyph, r0, c0, bg=BG)
        assert _count_holes(glyph, BG) == int(holes)

    ref_turns = random.randrange(4)
    ref_coords = _rotate_coords_in_square(_BASE_REF, REFERENCE_SIZE, ref_turns)
    ref_mask = _coords_to_mask(ref_coords, (REFERENCE_SIZE, REFERENCE_SIZE))
    ref_patch = np.where(ref_mask, GRAY, BG).astype(int)
    ref_r = random.randint(15, h - REFERENCE_SIZE - 1)
    ref_c = random.randint(1, 10)
    if np.any(grid[ref_r : ref_r + REFERENCE_SIZE, ref_c : ref_c + REFERENCE_SIZE] != BG):
        return None
    _paint_patch(grid, ref_patch, ref_r, ref_c, bg=BG)

    ref_rots = _rotations_of_reference(ref_patch, BG)
    subset_choices = _good_cluster_subsets(ref_rots)
    if any(not subset_choices[(count, rot)] for count in range(2, 6) for rot in range(4)):
        return None

    duplicate_count = random.choice([2, 3, 4, 5])
    cluster_specs = [
        {"count": 2, "color": cluster_colors[0]},
        {"count": 3, "color": cluster_colors[0]},
        {"count": 4, "color": cluster_colors[1]},
        {"count": 5, "color": cluster_colors[2]},
        {"count": duplicate_count, "color": cluster_colors[3]},
    ]
    if random.random() < 0.45:
        cluster_specs.append(
            {
                "count": random.choice([2, 3, 4, 5]),
                "color": random.choice(cluster_colors),
            }
        )

    rotations = [0, 1, 2, 3]
    random.shuffle(rotations)
    for i, spec in enumerate(cluster_specs):
        spec["rotation"] = rotations[i] if i < 4 else random.randrange(4)

    candidates = _anchor_candidates(h, w, scene_c0)
    if len(candidates) < len(cluster_specs):
        return None
    random.shuffle(candidates)
    clusters: list[dict] = []

    occupied_stamp = np.zeros((h, w), dtype=bool)
    for spec, (r0, c0) in zip(cluster_specs, candidates):
        count = int(spec["count"])
        rot = int(spec["rotation"])
        color = int(spec["color"])
        if r0 + GLYPH_SIZE > h or c0 + GLYPH_SIZE > w:
            return None
        if np.any(occupied_stamp[r0 : r0 + GLYPH_SIZE, c0 : c0 + GLYPH_SIZE]):
            return None
        occupied_stamp[r0 : r0 + GLYPH_SIZE, c0 : c0 + GLYPH_SIZE] = True

        subset = random.choice(subset_choices[(count, rot)])
        placed = _place_cluster(grid, subset, r0, c0, color)

        glyph = np.rot90(glyphs[count], rot).copy()
        stamp = np.full_like(glyph, BG)
        stamp[glyph != BG] = color
        _paint_patch(expected, stamp, r0, c0, bg=BG)
        clusters.append(
            {
                "count": count,
                "rotation": rot,
                "color": color,
                "coords": placed,
                "bbox": _bbox(placed),
            }
        )

    if not _semantic_decoys(clusters, glyphs):
        return None

    try:
        actual = main(grid)
    except (AssertionError, ValueError):
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate valid fable/5/001/7 input")
