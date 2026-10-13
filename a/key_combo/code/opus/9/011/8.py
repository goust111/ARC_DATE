import random
import sys
from pathlib import Path

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import Color, bbox_coords, components_mask, crop_bbox, paste, random_free_location, scale_sprite


BG = Color.BLACK
MOTIF = Color.BLUE
ANCHOR = Color.RED
TARGET = Color.GREEN

MAX_GEN_ATTEMPTS = 400
TARGET_COUNT_CHOICES = (3, 4)
GRID_SIDE_RANGE = (22, 30)


def _bbox_chebyshev_distance(a, b):
    ar0, ac0, ar1, ac1 = map(int, a)
    br0, bc0, br1, bc1 = map(int, b)
    row_gap = max(0, br0 - ar1 - 1, ar0 - br1 - 1)
    col_gap = max(0, bc0 - ac1 - 1, ac0 - bc1 - 1)
    return max(int(row_gap), int(col_gap))


def _is_solid_square(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = bbox_coords(coords)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    return h == w and int(coords.shape[0]) == h * w


def _component_clusters(coords_list: list[np.ndarray], max_dist: int = 2) -> list[list[int]]:
    n = len(coords_list)
    boxes = [bbox_coords(coords) for coords in coords_list]
    seen = [False] * n
    clusters = []
    for start in range(n):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        cluster = []
        while stack:
            i = stack.pop()
            cluster.append(i)
            for j in range(n):
                if seen[j]:
                    continue
                if _bbox_chebyshev_distance(boxes[i], boxes[j]) <= max_dist:
                    seen[j] = True
                    stack.append(j)
        clusters.append(cluster)
    return clusters


def _extract_green_targets(grid: np.ndarray) -> list[dict]:
    comps = components_mask(grid == TARGET, connectivity=4)
    assert comps, "expected green components"
    boxes = [bbox_coords(coords) for coords in comps]
    clusters = _component_clusters(comps, max_dist=2)

    targets = []
    for cluster in clusters:
        areas = [int(comps[i].shape[0]) for i in cluster]
        order = sorted(cluster, key=lambda i: (int(comps[i].shape[0]), -i), reverse=True)
        best_idx = order[0]
        assert areas.count(max(areas)) == 1, "each green cluster must have a unique largest component"
        coords = comps[best_idx]
        box = boxes[best_idx]
        assert _is_solid_square(coords), "kept green component must be a solid square"
        targets.append(
            {
                "coords": coords,
                "bbox": box,
                "side": int(box[2] - box[0] + 1),
                "cluster_indices": cluster,
                "cluster_boxes": [boxes[i] for i in cluster],
                "cluster_area": sum(int(comps[i].shape[0]) for i in cluster),
            }
        )
    return sorted(targets, key=lambda item: (item["bbox"][0], item["bbox"][1]))


def _find_unique_square_component(grid: np.ndarray, color: int) -> tuple[np.ndarray, tuple[int, int, int, int], int]:
    comps = components_mask(grid == color, connectivity=4)
    assert len(comps) == 1, f"expected exactly one component of color {color}"
    coords = comps[0]
    box = bbox_coords(coords)
    assert _is_solid_square(coords), f"color {color} component must be a solid square"
    side = int(box[2] - box[0] + 1)
    return coords, box, side


def _motif_patch(grid: np.ndarray) -> np.ndarray:
    comps = components_mask(grid == MOTIF, connectivity=4)
    assert len(comps) == 1, "expected exactly one blue motif component"
    coords = comps[0]
    box = bbox_coords(coords)
    motif_only = np.zeros_like(grid)
    motif_only[grid == MOTIF] = MOTIF
    patch = crop_bbox(motif_only, box)
    assert patch.shape[0] >= 1 and patch.shape[1] >= 1
    assert np.any(patch == BG), "motif bbox should contain at least one background cell"
    return patch


def main(grid: np.ndarray) -> np.ndarray:
    """
    Filter small green distractors, decode scale from the red anchor square, and stamp the blue motif onto each target square.

    Input structure:
    - The background is black.
    - One blue connected motif sits near the upper-left area.
    - One red solid square is the anchor block.
    - Elsewhere, green pixels form several spatial clusters: in each cluster one largest solid square target and one or two smaller detached distractor components lie within Chebyshev distance 2 of that target.

    Rule:
    - Split green pixels into 4-connected components, then group those components into clusters using bbox Chebyshev distance <= 2.
    - In each cluster, keep only the unique largest component; it is the true target square and every smaller green component is removed.
    - Let the red anchor square side be a, and let each kept green square side be t. Compute the integer scale factor k = t / a.
    - Crop the blue motif to its bbox, scale that bbox by nearest-neighbor with factor k, align it to the target square's top-left corner, and overlay only blue cells onto the target square.

    Key insight:
    - The nearby green specks are not harmless decoration: they must be filtered out first, otherwise the target square size is misread and the anchor-derived scale factor becomes wrong.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    out[out == TARGET] = BG

    motif = _motif_patch(g)
    _red_coords, _red_box, anchor_side = _find_unique_square_component(g, ANCHOR)

    for target in _extract_green_targets(g):
        r0, c0, r1, c1 = target["bbox"]
        side = target["side"]
        assert side % anchor_side == 0, "target side must be divisible by anchor side"
        scale = side // anchor_side
        scaled = scale_sprite(motif, scale)
        assert scaled.shape[0] <= side and scaled.shape[1] <= side, "scaled motif must fit inside the target square"
        out[r0 : r1 + 1, c0 : c1 + 1] = TARGET
        paste(out, scaled, r0, c0, transparent=BG)

    return out


def _grow_connected_mask(h: int, w: int, target_cells: int) -> np.ndarray | None:
    for _ in range(80):
        mask = np.zeros((h, w), dtype=bool)
        seed = (random.randrange(h), random.randrange(w))
        mask[seed] = True
        frontier = set()
        for r, c in [seed]:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    frontier.add((nr, nc))
        while int(mask.sum()) < target_cells and frontier:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            if mask[cell]:
                continue
            mask[cell] = True
            r, c = cell
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.add((nr, nc))
        if int(mask.sum()) != target_cells:
            continue
        if not (mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        if target_cells == h * w:
            continue
        return mask
    return None


def _sample_motif(anchor_side: int) -> np.ndarray | None:
    if anchor_side == 2:
        motif = np.zeros((2, 2), dtype=int)
        missing = random.choice(((0, 0), (0, 1), (1, 0), (1, 1)))
        motif[:, :] = MOTIF
        motif[missing] = BG
        return motif

    dims = random.choice(((2, 3), (3, 2), (3, 3)))
    h, w = dims
    lo = 4
    hi = min(7, h * w - 1)
    if lo > hi:
        return None
    target_cells = random.randint(lo, hi)
    mask = _grow_connected_mask(h, w, target_cells)
    if mask is None:
        return None
    motif = np.zeros((h, w), dtype=int)
    motif[mask] = MOTIF
    return motif


def _build_source_patch(anchor_side: int) -> tuple[np.ndarray, np.ndarray] | None:
    motif = _sample_motif(anchor_side)
    if motif is None:
        return None
    mh, mw = motif.shape
    gap = random.randint(1, 2)
    if random.random() < 0.5:
        patch_h = mh + gap + anchor_side
        patch_w = max(mw, anchor_side)
        patch = np.zeros((patch_h, patch_w), dtype=int)
        motif_c = random.randint(0, patch_w - mw)
        anchor_c = random.randint(0, patch_w - anchor_side)
        paste(patch, motif, 0, motif_c, transparent=BG)
        patch[mh + gap : mh + gap + anchor_side, anchor_c : anchor_c + anchor_side] = ANCHOR
    else:
        patch_h = max(mh, anchor_side)
        patch_w = mw + gap + anchor_side
        patch = np.zeros((patch_h, patch_w), dtype=int)
        motif_r = random.randint(0, patch_h - mh)
        anchor_r = random.randint(0, patch_h - anchor_side)
        paste(patch, motif, motif_r, 0, transparent=BG)
        patch[anchor_r : anchor_r + anchor_side, mw + gap : mw + gap + anchor_side] = ANCHOR
    return patch, motif


def _distractor_candidates(side: int) -> list[tuple[tuple[int, int], ...]]:
    candidates: list[tuple[tuple[int, int], ...]] = [
        ((-1, -1),),
        ((-1, side),),
        ((side, -1),),
        ((side, side),),
    ]
    for c in range(side):
        candidates.append(((-2, c),))
        candidates.append(((side + 1, c),))
    for r in range(side):
        candidates.append(((r, -2),))
        candidates.append(((r, side + 1),))
    for c in range(max(1, side - 1)):
        candidates.append(((-2, c), (-2, c + 1)))
        candidates.append(((side + 1, c), (side + 1, c + 1)))
    for r in range(max(1, side - 1)):
        candidates.append(((r, -2), (r + 1, -2)))
        candidates.append(((r, side + 1), (r + 1, side + 1)))
    return candidates


def _build_target_cluster(side: int, force_expansion: bool) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    patch = np.zeros((side + 5, side + 5), dtype=int)
    sr = sc = 2
    patch[sr : sr + side, sc : sc + side] = TARGET
    distractor_total = random.randint(1, 2)
    chosen: list[tuple[tuple[int, int], ...]] = []
    candidates = _distractor_candidates(side)
    random.shuffle(candidates)

    forced = None
    if force_expansion:
        forced_options = [cand for cand in candidates if any(r == -1 or c == -1 or r == side or c == side for r, c in cand)]
        if forced_options:
            forced = random.choice(forced_options)
            chosen.append(forced)

    occupied = set()
    for rr in range(side):
        for cc in range(side):
            occupied.add((sr + rr, sc + cc))

    def valid(shape: tuple[tuple[int, int], ...]) -> bool:
        cells = [(sr + r, sc + c) for r, c in shape]
        for r, c in cells:
            if not (0 <= r < patch.shape[0] and 0 <= c < patch.shape[1]):
                return False
            if (r, c) in occupied:
                return False
        temp = np.zeros_like(patch, dtype=int)
        for r, c in cells:
            temp[r, c] = TARGET
        comps = components_mask(temp == TARGET, connectivity=4)
        return len(comps) == 1

    if forced is not None:
        for r, c in forced:
            occupied.add((sr + r, sc + c))

    for cand in candidates:
        if len(chosen) >= distractor_total:
            break
        if cand in chosen:
            continue
        if not valid(cand):
            continue
        cells = [(sr + r, sc + c) for r, c in cand]
        if any(any(abs(r - er) + abs(c - ec) == 1 for er, ec in occupied if (er, ec) not in {(sr + rr, sc + cc) for rr, cc in cand}) for r, c in cells):
            continue
        for r, c in cells:
            occupied.add((r, c))
        chosen.append(cand)

    if len(chosen) != distractor_total:
        return None

    for shape in chosen:
        for r, c in shape:
            patch[sr + r, sc + c] = TARGET

    comps = components_mask(patch == TARGET, connectivity=4)
    if len(comps) != 1 + distractor_total:
        return None
    cluster = _extract_green_targets(patch)
    if len(cluster) != 1 or cluster[0]["side"] != side:
        return None
    return patch, (sr, sc, sr + side - 1, sc + side - 1)


def _target_size_list(anchor_side: int, count: int) -> list[int] | None:
    choices = [size for size in range(3, 9) if size % anchor_side == 0]
    if len(choices) < 2:
        return None
    out = [random.choice(choices) for _ in range(count)]
    if len(set(out)) < 2:
        a, b = random.sample(choices, 2)
        out[0] = a
        out[1] = b
        random.shuffle(out)
    return out


def _expected_output(inp: np.ndarray, motif: np.ndarray, anchor_side: int, targets: list[tuple[int, int, int]]) -> np.ndarray:
    out = inp.copy()
    out[out == TARGET] = BG
    for r0, c0, side in targets:
        out[r0 : r0 + side, c0 : c0 + side] = TARGET
        scaled = scale_sprite(motif, side // anchor_side)
        paste(out, scaled, r0, c0, transparent=BG)
    return out


def _semantic_checks(grid: np.ndarray) -> bool:
    try:
        motif = _motif_patch(grid)
        _red_coords, _red_box, anchor_side = _find_unique_square_component(grid, ANCHOR)
        targets = _extract_green_targets(grid)
    except Exception:
        return False

    if motif.shape[0] > anchor_side or motif.shape[1] > anchor_side:
        return False
    if len(targets) < 3:
        return False
    if len({target["side"] for target in targets}) < 2:
        return False

    filtering_matters = False
    for target in targets:
        side = target["side"]
        if side % anchor_side != 0:
            return False
        cluster_box = target["cluster_boxes"][0]
        for other in target["cluster_boxes"][1:]:
            r0 = min(cluster_box[0], other[0])
            c0 = min(cluster_box[1], other[1])
            r1 = max(cluster_box[2], other[2])
            c1 = max(cluster_box[3], other[3])
            cluster_box = (r0, c0, r1, c1)
        union_side = max(cluster_box[2] - cluster_box[0] + 1, cluster_box[3] - cluster_box[1] + 1)
        if union_side != side or target["cluster_area"] != side * side:
            filtering_matters = True
    return filtering_matters


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(*GRID_SIDE_RANGE)
    cols = random.randint(*GRID_SIDE_RANGE)
    grid = np.zeros((rows, cols), dtype=int)

    anchor_side = random.choices([2, 3], weights=[2, 3])[0]
    source = _build_source_patch(anchor_side)
    if source is None:
        return None
    source_patch, motif = source
    source_r = random.randint(1, 2)
    source_c = random.randint(1, 2)
    if source_r + source_patch.shape[0] > rows or source_c + source_patch.shape[1] > cols:
        return None
    paste(grid, source_patch, source_r, source_c, transparent=BG)

    target_count = random.choice(TARGET_COUNT_CHOICES)
    sizes = _target_size_list(anchor_side, target_count)
    if sizes is None:
        return None

    targets: list[tuple[int, int, int]] = []
    occupied = grid.copy()
    forced_index = random.randrange(target_count)
    for idx, side in enumerate(sizes):
        cluster = _build_target_cluster(side, force_expansion=idx == forced_index)
        if cluster is None:
            return None
        patch, square_box = cluster
        try:
            place_r, place_c = random_free_location(
                occupied,
                patch,
                transparent=BG,
                border=1,
                padding=3,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(grid, patch, place_r, place_c, transparent=BG)
        paste(occupied, patch, place_r, place_c, transparent=BG)
        sr0, sc0, sr1, sc1 = square_box
        targets.append((place_r + sr0, place_c + sc0, side))

    expected = _expected_output(grid, motif, anchor_side, targets)
    if not _semantic_checks(grid):
        return None
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
