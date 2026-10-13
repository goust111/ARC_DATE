from __future__ import annotations

from collections import defaultdict
import importlib.util
import itertools
import math
from pathlib import Path
import random

import numpy as np


def _load_arc_utils():
    utils_path = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
    spec = importlib.util.spec_from_file_location("_arc_utils_residue_double_anchor", utils_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load arc_utils from {utils_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ARC = _load_arc_utils()
Color = _ARC.Color
crop_bbox = _ARC.crop_bbox
find_components = _ARC.find_components
get_bg = _ARC.get_bg
paste = _ARC.paste
random_free_location = _ARC.random_free_location


TRANSFORM_IDS = tuple(range(8))
BG = Color.BLACK
PALETTE = tuple(range(1, 10))
GRID_SIDE_RANGE = (18, 28)
PROTO_HEIGHT_RANGE = (3, 5)
PROTO_WIDTH_RANGE = (3, 5)
PROTO_AREA_RANGE = (5, 8)
PAIR_COUNT_RANGE = (2, 4)
GEN_RETRIES = 400


def _rigid_transform(patch: np.ndarray, transform_id: int) -> np.ndarray:
    x = np.asarray(patch, dtype=int)
    t = int(transform_id)
    if t >= 4:
        x = np.fliplr(x)
        t -= 4
    return np.rot90(x, k=t).copy()


def _fg_cells(patch: np.ndarray) -> list[tuple[int, int, int]]:
    return [(int(r), int(c), int(patch[r, c])) for r, c in np.argwhere(patch != BG)]


def _max_fg_distance_sq(patch: np.ndarray) -> int:
    coords = np.argwhere(patch != BG)
    best = 0
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            dr = int(coords[i, 0] - coords[j, 0])
            dc = int(coords[i, 1] - coords[j, 1])
            best = max(best, dr * dr + dc * dc)
    return best


def _connected_random_mask(height: int, width: int, area: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    start = (random.randrange(height), random.randrange(width))
    mask[start] = True
    frontier = set()
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = start[0] + dr, start[1] + dc
        if 0 <= rr < height and 0 <= cc < width:
            frontier.add((rr, cc))

    while int(mask.sum()) < area and frontier:
        rr, cc = random.choice(tuple(frontier))
        frontier.remove((rr, cc))
        mask[rr, cc] = True
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                frontier.add((nr, nc))
    return mask


def _make_prototype() -> tuple[np.ndarray, list[tuple[tuple[int, int], tuple[int, int]]]]:
    for _ in range(GEN_RETRIES):
        height = random.randint(*PROTO_HEIGHT_RANGE)
        width = random.randint(*PROTO_WIDTH_RANGE)
        max_area = min(PROTO_AREA_RANGE[1], height * width)
        area = random.randint(PROTO_AREA_RANGE[0], max_area)
        mask = _connected_random_mask(height, width, area)
        comp = np.argwhere(mask)
        r0, c0 = comp.min(axis=0)
        r1, c1 = comp.max(axis=0)
        mask = mask[r0 : r1 + 1, c0 : c1 + 1]
        if mask.shape[0] < 3 or mask.shape[1] < 3:
            continue
        coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
        eligible_pairs = []
        for (r1, c1), (r2, c2) in itertools.combinations(coords, 2):
            dr = r2 - r1
            dc = c2 - c1
            if max(abs(dr), abs(dc)) <= 1:
                continue
            if dr == 0 or dc == 0:
                continue
            if abs(dr) == abs(dc):
                continue
            eligible_pairs.append(((r1, c1), (r2, c2)))
        if len(eligible_pairs) < 3:
            continue
        colors = random.sample(PALETTE, len(coords))
        patch = np.zeros(mask.shape, dtype=int)
        for (r, c), color in zip(coords, colors):
            patch[r, c] = int(color)
        return patch, eligible_pairs
    raise RuntimeError("failed to generate a valid prototype")


def _residue_clusters(residue_cells: list[tuple[int, int, int]], max_dist_sq: int) -> list[list[tuple[int, int, int]]]:
    n = len(residue_cells)
    if n % 2 != 0:
        return []
    if n == 0:
        return []

    dists = [[0] * n for _ in range(n)]
    nearest: list[int | None] = [None] * n
    for i in range(n):
        best_dist = None
        best_indices: list[int] = []
        r1, c1, _ = residue_cells[i]
        for j in range(n):
            if i == j:
                continue
            r2, c2, _ = residue_cells[j]
            dist = (r1 - r2) ** 2 + (c1 - c2) ** 2
            dists[i][j] = dist
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_indices = [j]
            elif dist == best_dist:
                best_indices.append(j)
        if best_dist is None or best_dist > max_dist_sq or len(best_indices) != 1:
            return []
        nearest[i] = best_indices[0]

    used = set()
    clusters = []
    for i in range(n):
        if i in used:
            continue
        j = nearest[i]
        if j is None or j in used or nearest[j] != i or dists[i][j] > max_dist_sq:
            return []
        used.add(i)
        used.add(j)
        clusters.append([residue_cells[i], residue_cells[j]])

    if len(used) != n:
        return []
    return clusters


def _scene_parts(grid: np.ndarray):
    comps = find_components(grid, bg=BG, connectivity=4, monochromatic=False, return_format="dict")
    assert comps, "expected foreground"
    areas = [int(comp["area"]) for comp in comps]
    max_area = max(areas)
    prototype_candidates = [comp for comp in comps if int(comp["area"]) == max_area]
    assert len(prototype_candidates) == 1, "expected a unique full prototype"
    prototype = prototype_candidates[0]
    assert max_area >= 2, "prototype must have at least two cells"
    residue_cells = []
    for comp in comps:
        if comp is prototype:
            continue
        assert int(comp["area"]) == 1, "all non-prototype foreground must be singleton residue pixels"
        r, c = map(int, comp["coords"][0])
        residue_cells.append((r, c, int(grid[r, c])))
    prototype_patch = crop_bbox(grid, prototype["bbox"])
    return prototype_patch, prototype, residue_cells


def _pair_candidates(
    grid: np.ndarray,
    prototype_patch: np.ndarray,
    cluster: list[tuple[int, int, int]],
    all_foreground: set[tuple[int, int]],
) -> list[tuple[int, int, int]]:
    assert len(cluster) == 2
    candidates = set()
    cluster_points = {(int(r), int(c)) for r, c, _color in cluster}
    ordered_clusters = (cluster, [cluster[1], cluster[0]])
    h_grid, w_grid = grid.shape

    for transform_id in TRANSFORM_IDS:
        variant = _rigid_transform(prototype_patch, transform_id)
        by_color: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for r, c, color in _fg_cells(variant):
            by_color[color].append((r, c))
        for ordered in ordered_clusters:
            (r1, c1, color1), (r2, c2, color2) = ordered
            for q1 in by_color.get(int(color1), []):
                for q2 in by_color.get(int(color2), []):
                    if q1 == q2:
                        continue
                    dr = int(r1 - q1[0])
                    dc = int(c1 - q1[1])
                    if (q2[0] + dr, q2[1] + dc) != (int(r2), int(c2)):
                        continue
                    if dr < 0 or dc < 0 or dr + variant.shape[0] > h_grid or dc + variant.shape[1] > w_grid:
                        continue
                    placed_points = {
                        (dr + qr, dc + qc)
                        for qr, qc, _color in _fg_cells(variant)
                    }
                    if placed_points & all_foreground != cluster_points:
                        continue
                    candidates.add((int(transform_id), dr, dc))
    return sorted(candidates)


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Recover every erased prototype copy by using each nearby residue pair as a double anchor.

    Input structure:
    - The background is black.
    - One connected multicolor prototype appears in full.
    - Every other foreground mark is an isolated colored pixel, and these singleton residue pixels form well-separated nearby pairs.

    Rule:
    - Detect the unique complete prototype object.
    - Group the singleton residue pixels into pairs by proximity, using the prototype's maximal internal point distance as the pairing scale.
    - For each pair, search the prototype's 8 rigid poses and find the unique pose plus translation whose two matching prototype cells land exactly on the two residue pixels.
    - Stamp the whole transformed prototype at that pose; keep the original prototype unchanged.

    Key insight:
    - A single residue fixes only position, but two colored residues jointly lock both orientation and translation of the missing copy.
    """
    g = np.asarray(input_grid, dtype=int)
    assert g.ndim == 2
    bg = int(get_bg(g))
    assert bg == BG

    prototype_patch, _prototype, residue_cells = _scene_parts(g)
    out = g.copy()
    if not residue_cells:
        return out

    max_dist_sq = _max_fg_distance_sq(prototype_patch)
    clusters = _residue_clusters(residue_cells, max_dist_sq)
    assert all(len(cluster) == 2 for cluster in clusters), "each residue cluster must be exactly one pair"
    all_foreground = {tuple(map(int, rc)) for rc in np.argwhere(g != bg)}

    for cluster in clusters:
        candidates = _pair_candidates(g, prototype_patch, cluster, all_foreground)
        assert len(candidates) == 1, "each residue pair must determine a unique transformed copy"
        transform_id, r0, c0 = candidates[0]
        variant = _rigid_transform(prototype_patch, transform_id)
        paste(out, variant, r0, c0, transparent=bg)

    return out


def _render_input_output(
    shape: tuple[int, int],
    prototype_patch: np.ndarray,
    prototype_loc: tuple[int, int],
    copies: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.zeros(shape, dtype=int)
    out = np.zeros(shape, dtype=int)
    paste(inp, prototype_patch, prototype_loc[0], prototype_loc[1], transparent=BG)
    paste(out, prototype_patch, prototype_loc[0], prototype_loc[1], transparent=BG)

    for copy_info in copies:
        variant = copy_info["variant"]
        r0, c0 = copy_info["loc"]
        paste(out, variant, r0, c0, transparent=BG)
        for rr, cc in copy_info["residues"]:
            inp[rr, cc] = out[rr, cc]
    return inp, out


def _build_instance() -> tuple[np.ndarray, np.ndarray]:
    prototype_patch, eligible_pairs = _make_prototype()
    n_pairs = random.randint(*PAIR_COUNT_RANGE)
    side_h = random.randint(*GRID_SIDE_RANGE)
    side_w = random.randint(*GRID_SIDE_RANGE)
    shape = (side_h, side_w)
    occupancy = np.zeros(shape, dtype=int)

    prototype_loc = random_free_location(occupancy, prototype_patch, transparent=BG, border=1, padding=1)
    paste(occupancy, prototype_patch, prototype_loc[0], prototype_loc[1], transparent=BG)

    transform_ids = random.sample(TRANSFORM_IDS, n_pairs)
    copies = []
    for transform_id in transform_ids:
        variant = _rigid_transform(prototype_patch, transform_id)
        loc = random_free_location(occupancy, variant, transparent=BG, border=1, padding=1)
        paste(occupancy, variant, loc[0], loc[1], transparent=BG)

        anchor_a, anchor_b = random.choice(eligible_pairs)
        transformed_coords = np.argwhere(variant != BG)
        color_lookup = {(int(r), int(c)): int(variant[r, c]) for r, c in transformed_coords}
        base_variant = _rigid_transform(prototype_patch, transform_id)
        mapped_pair = []
        for anchor in (anchor_a, anchor_b):
            base_mask = np.zeros_like(prototype_patch)
            base_mask[anchor] = 1
            mapped = np.argwhere(_rigid_transform(base_mask, transform_id) == 1)
            assert mapped.shape == (1, 2)
            vr, vc = map(int, mapped[0])
            mapped_pair.append((loc[0] + vr, loc[1] + vc))
            assert color_lookup[(vr, vc)] == int(variant[vr, vc])

        copies.append(
            {
                "transform_id": int(transform_id),
                "variant": base_variant,
                "loc": (int(loc[0]), int(loc[1])),
                "residues": mapped_pair,
            }
        )

    inp, out = _render_input_output(shape, prototype_patch, prototype_loc, copies)
    return inp, out


def _semantic_checks(inp: np.ndarray, out: np.ndarray) -> bool:
    if inp.ndim != 2 or out.ndim != 2:
        return False
    if max(inp.shape) > 30 or max(out.shape) > 30:
        return False
    if np.array_equal(inp, out):
        return False

    try:
        solved = main(inp)
    except Exception:
        return False
    if not np.array_equal(solved, out):
        return False

    prototype_patch, _prototype, residue_cells = _scene_parts(inp)
    if len(residue_cells) < 4 or len(residue_cells) % 2 != 0:
        return False

    max_dist_sq = _max_fg_distance_sq(prototype_patch)
    clusters = _residue_clusters(residue_cells, max_dist_sq)
    if not clusters or any(len(cluster) != 2 for cluster in clusters):
        return False

    all_foreground = {tuple(map(int, rc)) for rc in np.argwhere(inp != BG)}
    candidate_signatures = set()
    for cluster in clusters:
        candidates = _pair_candidates(inp, prototype_patch, cluster, all_foreground)
        if len(candidates) != 1:
            return False
        candidate_signatures.add(tuple(candidates[0]))
        (r1, c1, _), (r2, c2, _) = cluster
        if max(abs(r1 - r2), abs(c1 - c2)) <= 1:
            return False
    return len(candidate_signatures) == len(clusters)


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            inp, out = _build_instance()
        except Exception:
            continue
        if _semantic_checks(inp, out):
            return inp
    raise RuntimeError("failed to generate a valid residue-double-anchor ARC instance")
