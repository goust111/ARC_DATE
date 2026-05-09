# Combo variant of 800d221b (index 19)
# combo_spec: D5/3+D6/1
# Name: 无边框种子+长度排序匹配
#
# Design analysis:
# - a) Concrete behavior trace:
#   - In the source task, a branch that physically touched a seed's P frame simply inherited that seed's majority
#     color by local adjacency. The P frame itself also became the same color.
#   - In this combo, there is no seed frame at all: a branch ends next to raw seed interior pixels. For example, a
#     short north branch may touch a 4x4 seed whose majority is blue, while a longer west branch touches a 2x3 seed
#     whose majority is red. The combo ignores those physical contacts. Instead, branches are sorted by visible path
#     length and seeds are sorted by visible rectangle area; the longest branch takes the largest seed's majority,
#     the second-longest branch takes the second-largest seed's majority, and so on.
#   - The crucial observable difference is therefore "who a branch touches" versus "which size-rank it has". The
#     generator deliberately makes those two relationships disagree, so the transformed branch colors cannot be
#     explained by local contact alone.
# - b) Complexity calibration:
#   - The scene keeps the original object family: one 3x3 P hub, four thin orthogonal branches, and four edge seeds.
#     This preserves the recognisable ARC vocabulary while changing only the intended reasoning dimensions.
#   - Seed areas use four distinct rectangle sizes and branch paths use four distinct length templates. Keeping both
#     rank sets small, repeated, and strictly ordered makes the pairing rule human-visible without introducing a large
#     combinatorial search space.
#   - The generator uses a derangement between physical seed slots and area ranks, so at least two branches would be
#     recolored differently under the original adjacency rule. Without that constraint the combo could visually
#     collapse back into the baseline task.
# - c) Hidden-assumption audit:
#   - Original assumption "every seed is surrounded by a P border" is invalid here. The solver removes all border
#     checks and instead recognises seeds directly as non-bg/non-P connected rectangles.
#   - Original assumption "branch-to-seed matching can be recovered from adjacency counts" is also invalid. The new
#     solver still records physical touches for validation, but output coloring ignores them and uses global sorting.
#   - Hub inference via the largest solid P square still holds because the hub geometry is unchanged.
#   - P-color inference via "largest network-like non-background component" still holds because the input still has
#     one dominant thin scaffold plus a compact hub, while seeds remain block-like.
#   - New combo assumptions: seed areas are all distinct, branch lengths are all distinct, branch count equals seed
#     count, and each branch touches exactly one seed component. The generator enforces each of these conditions.
# - d) Placement legality:
#   - Seeds are fixed to four edge slots that stay far from the central hub and from each other.
#   - Each branch follows a prevalidated orthogonal polyline in its own corridor, never overlaps another branch, never
#     enters a seed interior, and ends in a cell adjacent to exactly one seed rectangle.
#   - Rectangle sizes stay within the 27x27 canvas bounds for every rank assignment.
# - e) Visual observability:
#   - A human can compare branch lengths and seed rectangle areas directly from the pixels; no hidden numeric feature
#     such as color id, component index, or generation order is used in the rule.
#   - Because physical contact and sorted pairing intentionally disagree, the examples visibly teach that local
#     touching is not the governing rule anymore.
# - f) Visual cleanliness:
#   - The layout stays sparse and symmetric: one hub in the middle, one branch per cardinal direction, one seed on
#     each edge, and no extra legends or decorations. Objects do not overlap or stack multiple annotations.
# - g) Inferability of the color change:
#   - Only two seed colors are used globally, as in the source family. The color decision is supported by repeated
#     structure: the same seed majority visible inside a larger rectangle reappears on the correspondingly longer
#     branch even when that branch touches a different-colored seed.

from __future__ import annotations

from collections import Counter
import importlib.util
import itertools
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
UTIL_PATH = ROOT / "solvers" / "arc_utils.py"
SPEC = importlib.util.spec_from_file_location("arc_utils_combo_800d221b_19", UTIL_PATH)
arc_utils = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = arc_utils
assert SPEC.loader is not None
SPEC.loader.exec_module(arc_utils)

components_mask = arc_utils.components_mask
get_bg = arc_utils.get_bg
label_mask = arc_utils.label_mask

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GRID_SHAPE = (27, 27)
HUB_SIZE = 3
HUB_R0 = 12
HUB_C0 = 12
GEN_RETRIES = 200

SLOTS = ("N", "E", "S", "W")
BRANCH_RANK_BY_SLOT = {"N": 0, "S": 1, "E": 2, "W": 3}  # shortest -> longest
SIZE_BY_RANK = {
    0: (2, 3),  # area 6
    1: (2, 4),  # area 8
    2: (3, 4),  # area 12
    3: (4, 4),  # area 16
}
DERANGEMENTS = [
    perm
    for perm in itertools.permutations(range(4))
    if all(int(perm[i]) != i for i in range(4))
]


def _polyline(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for idx, ((r0, c0), (r1, c1)) in enumerate(zip(points, points[1:])):
        if r0 == r1:
            step = 1 if c1 >= c0 else -1
            seg = [(int(r0), int(c)) for c in range(int(c0), int(c1) + step, step)]
        elif c0 == c1:
            step = 1 if r1 >= r0 else -1
            seg = [(int(r), int(c0)) for r in range(int(r0), int(r1) + step, step)]
        else:
            raise AssertionError("branch template must be orthogonal")
        if idx:
            seg = seg[1:]
        cells.extend(seg)
    return cells


def _comp_deg_score(mask: np.ndarray) -> float:
    comps = components_mask(mask, connectivity=4)
    if not comps:
        return -1e9
    pts = max(comps, key=lambda coords: int(coords.shape[0]))
    H, W = mask.shape
    m = np.zeros((H, W), dtype=bool)
    m[pts[:, 0], pts[:, 1]] = True
    deg2 = 0
    deg4 = 0
    for r, c in pts:
        r = int(r)
        c = int(c)
        deg = 0
        if r > 0 and bool(m[r - 1, c]):
            deg += 1
        if r + 1 < H and bool(m[r + 1, c]):
            deg += 1
        if c > 0 and bool(m[r, c - 1]):
            deg += 1
        if c + 1 < W and bool(m[r, c + 1]):
            deg += 1
        if deg == 2:
            deg2 += 1
        elif deg == 4:
            deg4 += 1
    return float((deg2 - deg4) * int(pts.shape[0]) / max(1, int(pts.shape[0])))


def _infer_path_color(g: np.ndarray, bg: int) -> int:
    colors = sorted({int(x) for x in np.unique(g)} - {int(bg)})
    assert colors, "missing non-background colors"
    return int(max(colors, key=lambda color: _comp_deg_score(g == int(color))))


def _find_hub(g: np.ndarray, path_color: int) -> tuple[np.ndarray, int, int, int]:
    H, W = g.shape
    mP = (g == int(path_color)).astype(int)
    ps = mP.cumsum(axis=0).cumsum(axis=1)

    def rect_sum(r0: int, c0: int, r1: int, c1: int) -> int:
        total = int(ps[r1, c1])
        if r0 > 0:
            total -= int(ps[r0 - 1, c1])
        if c0 > 0:
            total -= int(ps[r1, c0 - 1])
        if r0 > 0 and c0 > 0:
            total += int(ps[r0 - 1, c0 - 1])
        return int(total)

    best: tuple[int, int, int] | None = None
    for k in range(2, min(H, W) + 1):
        found = None
        for r0 in range(H - k + 1):
            r1 = r0 + k - 1
            for c0 in range(W - k + 1):
                c1 = c0 + k - 1
                if rect_sum(r0, c0, r1, c1) == k * k:
                    found = (int(k), int(r0), int(c0))
        if found is not None:
            best = found

    assert best is not None, "hub square not found"
    k, r0, c0 = best
    hub = np.zeros(g.shape, dtype=bool)
    hub[r0 : r0 + k, c0 : c0 + k] = True
    return hub, int(k), int(r0), int(c0)


def _bbox_of_pts(pts: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(pts[:, 0].min()),
        int(pts[:, 1].min()),
        int(pts[:, 0].max()),
        int(pts[:, 1].max()),
    )


def _seed_infos(g: np.ndarray, bg: int, path_color: int) -> tuple[np.ndarray, list[dict]]:
    seed_mask = (g != int(bg)) & (g != int(path_color))
    seed_lab, seed_n = label_mask(seed_mask, connectivity=4)
    assert int(seed_n) > 0, "grid must contain seed regions"

    infos: list[dict] = []
    for sid in range(1, int(seed_n) + 1):
        pts = np.argwhere(seed_lab == int(sid))
        assert pts.size > 0
        r0, c0, r1, c1 = _bbox_of_pts(pts)
        rect = seed_mask[r0 : r1 + 1, c0 : c1 + 1]
        assert bool(np.all(rect)), "combo expects each seed to be a solid rectangle"
        vals = [int(g[int(r), int(c)]) for r, c in pts]
        counts = sorted(Counter(vals).items(), key=lambda kv: (-kv[1], kv[0]))
        assert len(counts) == 2, "each seed must contain exactly two colors"
        assert int(counts[0][1]) > int(counts[1][1]), "seed majority must be strict"
        infos.append(
            {
                "id": int(sid),
                "pts": pts,
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "area": int(pts.shape[0]),
                "majority": int(counts[0][0]),
            }
        )
    return seed_lab, infos


def _check_path_degrees(g: np.ndarray, path_color: int, hub: np.ndarray) -> None:
    pmask = g == int(path_color)
    H, W = g.shape
    for r in range(H):
        for c in range(W):
            if not bool(pmask[r, c]) or bool(hub[r, c]):
                continue
            deg = 0
            for dr, dc in DIRS4:
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < H and 0 <= cc < W and bool(pmask[rr, cc]):
                    deg += 1
            assert deg <= 2, "branch network must stay 1-pixel wide without T-junctions"


def _branch_infos(g: np.ndarray, path_color: int, hub: np.ndarray, seed_lab: np.ndarray) -> list[dict]:
    _check_path_degrees(g, path_color, hub)
    branch_mask = (g == int(path_color)) & (~hub)
    branch_lab, branch_n = label_mask(branch_mask, connectivity=4)
    H, W = g.shape
    infos: list[dict] = []
    for bid in range(1, int(branch_n) + 1):
        pts = np.argwhere(branch_lab == int(bid))
        touches: Counter[int] = Counter()
        for r, c in pts:
            r = int(r)
            c = int(c)
            for dr, dc in DIRS4:
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < H and 0 <= cc < W:
                    sid = int(seed_lab[rr, cc])
                    if sid:
                        touches[int(sid)] += 1
        assert touches, "every branch must touch a seed directly"
        assert len(touches) == 1, "each branch must touch exactly one seed"
        infos.append(
            {
                "id": int(bid),
                "pts": pts,
                "length": int(pts.shape[0]),
                "touch_seed_id": int(next(iter(touches))),
                "touch_count": int(next(iter(touches.values()))),
                "bbox": _bbox_of_pts(pts),
            }
        )
    return infos


def _scene_info(grid: np.ndarray) -> dict:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    path_color = _infer_path_color(g, bg)
    hub, hub_size, hub_r0, hub_c0 = _find_hub(g, path_color)
    seed_lab, seeds = _seed_infos(g, bg, path_color)
    branches = _branch_infos(g, path_color, hub, seed_lab)

    assert len(seeds) == len(branches), "combo expects one branch per seed"
    assert len({int(seed["area"]) for seed in seeds}) == len(seeds), "seed areas must be distinct"
    assert len({int(branch["length"]) for branch in branches}) == len(branches), "branch lengths must be distinct"

    return {
        "grid": g,
        "bg": int(bg),
        "path_color": int(path_color),
        "hub": hub,
        "hub_size": int(hub_size),
        "hub_r0": int(hub_r0),
        "hub_c0": int(hub_c0),
        "center": (int(hub_r0 + hub_size // 2), int(hub_c0 + hub_size // 2)),
        "seeds": seeds,
        "seeds_by_id": {int(seed["id"]): seed for seed in seeds},
        "branches": branches,
    }


def _vote_center_color(branches: list[dict], branch_colors: dict[int, int], hub: np.ndarray) -> int:
    H, W = hub.shape
    counts: Counter[int] = Counter()
    for branch in branches:
        color = int(branch_colors[int(branch["id"])])
        for r, c in branch["pts"]:
            r = int(r)
            c = int(c)
            for dr, dc in DIRS4:
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < H and 0 <= cc < W and bool(hub[rr, cc]):
                    counts[int(color)] += 1
    assert counts, "hub must be touched by recolored branches"
    return int(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0])


def _sorted_branch_target_colors(scene: dict) -> dict[int, int]:
    seed_order = sorted(scene["seeds"], key=lambda info: (-int(info["area"]), tuple(info["bbox"])))
    branch_order = sorted(scene["branches"], key=lambda info: (-int(info["length"]), tuple(info["bbox"])))
    return {
        int(branch["id"]): int(seed["majority"])
        for branch, seed in zip(branch_order, seed_order)
    }


def _adjacency_branch_target_colors(scene: dict) -> dict[int, int]:
    return {
        int(branch["id"]): int(scene["seeds_by_id"][int(branch["touch_seed_id"])]["majority"])
        for branch in scene["branches"]
    }


def _render_output(scene: dict, branch_colors: dict[int, int]) -> np.ndarray:
    out = scene["grid"].copy()
    for branch in scene["branches"]:
        color = int(branch_colors[int(branch["id"])])
        for r, c in branch["pts"]:
            out[int(r), int(c)] = int(color)
    out[scene["center"]] = _vote_center_color(scene["branches"], branch_colors, scene["hub"])
    return out


def _adjacency_baseline_output(grid: np.ndarray) -> np.ndarray:
    scene = _scene_info(grid)
    return _render_output(scene, _adjacency_branch_target_colors(scene))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor a hub-and-spoke path layout using size-rank matching instead of contact matching.

    Input structure:
    - The background is the most frequent color.
    - One color P forms a thin orthogonal network with a solid 3x3 hub near the middle.
    - Four seed rectangles sit on the outer edges. Each seed has no P border: its two interior colors are exposed
      directly to the background, and a single branch ends adjacent to the seed interior itself.

    Rule:
    - Infer P from the network-like component and locate the hub as the largest solid all-P square.
    - Treat every non-bg, non-P connected rectangle as one seed and compute its majority color plus its area.
    - Treat every non-hub P component as one branch and compute its path length.
    - Sort branches by length and seeds by area, pairing longest with largest, second-longest with second-largest,
      and so on; recolor each branch with the matched seed's majority color.
    - Recolor only the hub center pixel to the branch color that touches the hub most often.

    Key insight: Once the seed borders are removed, physical contact is no longer the intended matching signal. The
    stable rule is the cross-object ranking: branch length order determines which seed-area majority each branch copies.
    """
    scene = _scene_info(grid)
    return _render_output(scene, _sorted_branch_target_colors(scene))


def _slot_bbox(slot: str, size_rank: int) -> tuple[int, int, int, int]:
    h, w = SIZE_BY_RANK[int(size_rank)]
    H, W = GRID_SHAPE
    if slot == "N":
        return (0, 3, h - 1, 3 + w - 1)
    if slot == "E":
        return (3, W - w, 3 + h - 1, W - 1)
    if slot == "S":
        return (H - h, 19, H - 1, 19 + w - 1)
    if slot == "W":
        return (19, 0, 19 + h - 1, w - 1)
    raise AssertionError("unknown slot")


def _slot_endpoint(slot: str, bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    if slot == "N":
        return (1, 2)
    if slot == "E":
        return (2, 25)
    if slot == "S":
        return (25, 18)
    if slot == "W":
        return (18, 1)
    raise AssertionError("unknown slot")


def _branch_cells_for_slot(slot: str, endpoint: tuple[int, int]) -> list[tuple[int, int]]:
    if slot == "N":
        points = [(11, 13), (6, 13), (6, 2), (1, 2)]
    elif slot == "E":
        points = [(13, 15), (13, 21), (10, 21), (10, 18), (6, 18), (6, 22), (2, 22), (2, 25)]
    elif slot == "S":
        points = [(15, 13), (19, 13), (19, 20), (22, 20), (22, 16), (25, 16), (25, 18)]
    elif slot == "W":
        points = [(13, 11), (13, 7), (18, 7), (18, 10), (22, 10), (22, 4), (18, 4), (18, 1)]
    else:
        raise AssertionError("unknown slot")
    return _polyline(points)


def _seed_cells_from_bbox(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return [(int(r), int(c)) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def _fill_seed(g: np.ndarray, cells: list[tuple[int, int]], major: int, minor: int, rng: random.Random) -> None:
    n = len(cells)
    max_minor = max(1, min(n // 3, n // 2 - 1))
    n_minor = rng.randint(1, max_minor)
    order = cells[:]
    rng.shuffle(order)
    for r, c in order[:n_minor]:
        g[int(r), int(c)] = int(minor)
    for r, c in order[n_minor:]:
        g[int(r), int(c)] = int(major)


def _branch_rank_color(rank: int, color_a: int, color_b: int) -> int:
    # smallest, second, third, largest seed areas -> A, B, A, A
    return int({0: color_a, 1: color_b, 2: color_a, 3: color_a}[int(rank)])


def _validate_layout(seed_boxes: dict[str, tuple[int, int, int, int]], branch_cells: dict[str, list[tuple[int, int]]]) -> None:
    H, W = GRID_SHAPE
    hub_cells = {(r, c) for r in range(HUB_R0, HUB_R0 + HUB_SIZE) for c in range(HUB_C0, HUB_C0 + HUB_SIZE)}
    occupied: set[tuple[int, int]] = set(hub_cells)
    seed_sets = {slot: set(_seed_cells_from_bbox(bbox)) for slot, bbox in seed_boxes.items()}
    for slot, cells in seed_sets.items():
        assert occupied.isdisjoint(cells), f"seed {slot} overlaps existing object"
        occupied |= cells
    branch_seen: set[tuple[int, int]] = set()
    for slot, cells in branch_cells.items():
        assert cells, f"branch {slot} must be non-empty"
        cell_set = set(cells)
        assert len(cell_set) == len(cells), f"branch {slot} self-overlaps"
        assert branch_seen.isdisjoint(cell_set), f"branch {slot} overlaps another branch"
        assert hub_cells.isdisjoint(cell_set), f"branch {slot} enters hub"
        assert cell_set.isdisjoint(seed_sets[slot]), f"branch {slot} enters its seed"
        for other_slot, other_seed in seed_sets.items():
            if other_slot != slot:
                assert cell_set.isdisjoint(other_seed), f"branch {slot} enters seed {other_slot}"
        for r, c in cells:
            assert 0 <= int(r) < H and 0 <= int(c) < W, f"branch {slot} leaves grid"
        branch_seen |= cell_set


def _build_input(rng: random.Random) -> tuple[np.ndarray, dict]:
    H, W = GRID_SHAPE
    bg, path_color, color_a, color_b = rng.sample(list(range(10)), 4)
    seed_rank_by_slot = {slot: int(rank) for slot, rank in zip(SLOTS, rng.choice(DERANGEMENTS))}
    seed_boxes = {slot: _slot_bbox(slot, seed_rank_by_slot[slot]) for slot in SLOTS}
    endpoints = {slot: _slot_endpoint(slot, seed_boxes[slot]) for slot in SLOTS}
    branch_cells = {slot: _branch_cells_for_slot(slot, endpoints[slot]) for slot in SLOTS}
    _validate_layout(seed_boxes, branch_cells)

    g = np.full((H, W), int(bg), dtype=int)
    g[HUB_R0 : HUB_R0 + HUB_SIZE, HUB_C0 : HUB_C0 + HUB_SIZE] = int(path_color)

    physical_major_by_slot: dict[str, int] = {}
    combo_major_by_slot: dict[str, int] = {}
    for slot in SLOTS:
        rank = int(seed_rank_by_slot[slot])
        physical_major = _branch_rank_color(rank, color_a, color_b)
        combo_major = _branch_rank_color(BRANCH_RANK_BY_SLOT[slot], color_a, color_b)
        physical_major_by_slot[slot] = int(physical_major)
        combo_major_by_slot[slot] = int(combo_major)
        physical_minor = int(color_b if physical_major == color_a else color_a)
        _fill_seed(g, _seed_cells_from_bbox(seed_boxes[slot]), int(physical_major), physical_minor, rng)

    for slot in SLOTS:
        for r, c in branch_cells[slot]:
            g[int(r), int(c)] = int(path_color)

    mismatch_slots = sum(
        int(combo_major_by_slot[slot] != physical_major_by_slot[slot])
        for slot in SLOTS
    )
    meta = {
        "bg": int(bg),
        "path_color": int(path_color),
        "color_a": int(color_a),
        "color_b": int(color_b),
        "seed_rank_by_slot": seed_rank_by_slot,
        "physical_major_by_slot": physical_major_by_slot,
        "combo_major_by_slot": combo_major_by_slot,
        "mismatch_slots": int(mismatch_slots),
    }
    return g, meta


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        rng = random.Random(random.randrange(1 << 30))
        inp, meta = _build_input(rng)
        if int(meta["mismatch_slots"]) < 2:
            continue
        try:
            combo_out = main(inp)
            base_out = _adjacency_baseline_output(inp)
        except Exception:
            continue
        if np.array_equal(combo_out, base_out):
            continue
        return inp
    raise RuntimeError(f"generate_input failed after {GEN_RETRIES} attempts")
