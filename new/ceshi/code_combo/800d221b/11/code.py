# Combo variant of 800d221b (index 11)
# combo_spec: D1/0+D10/1
# Name: 少数色着色后删除短分支
#
# Design analysis:
# - a) Concrete behavior trace:
#   - In the source task, a seed interior with 11 teal cells and 4 red cells would make its attached spoke and
#     outer P frame become teal, and the spoke would always remain present.
#   - In this combo, the same seed first exports the minority red onto both its frame and its attached spoke-path.
#     Then, if that spoke-path is shorter than the learned length threshold, only the spoke-path is erased back to
#     background while the recolored frame stays visible.
#   - The important structural split is therefore: "seed border" is recolored but never deleted; "true branch path"
#     is recolored first and may be deleted second. This is the main difference from the original one-shot branch
#     component treatment.
# - b) Complexity calibration:
#   - The family still uses one hub, four edge seeds, one path color P, and 1-pixel orthogonal paths. That preserves
#     the original visual language instead of introducing new object types.
#   - Only two seed interior colors are used, and every sample contains both minority outcomes. This keeps the color
#     logic inferable from repeated local evidence rather than arbitrary palette remapping.
#   - The generator explicitly mixes short and long spokes in the same grid. Without both categories, the delete step
#     would collapse and the fusion would visually degenerate to the simpler minority-color variant.
# - c) Hidden-assumption audit:
#   - Original solver assumption "all non-hub P cells can be recolored as one component" no longer holds, because the
#     delete stage must spare seed borders while removing only short spoke paths.
#   - Inferring P as the most network-like non-background color still holds because inputs still contain one dominant
#     thin orthogonal scaffold plus a solid 3x3 hub.
#   - Hub detection by largest solid P square still holds because the hub shape is unchanged.
#   - Seed detection as connected non-bg/non-P components still holds because seed interiors remain untouched inputs.
#   - New assumption introduced by this combo: each non-hub, non-border P path component touches exactly one seed
#     border. The generator guarantees this by using disjoint cardinal corridors with one spoke per seed.
# - d) Placement legality:
#   - Seeds are fixed on four distinct grid edges with non-overlapping P border zones.
#   - Every branch template ends orthogonally adjacent to exactly one seed border, never overlaps the hub/interiors,
#     and branch corridors are disjoint, so no accidental merges or out-of-bounds placements occur.
# - e) Visual observability:
#   - Each output shows two repeated cues at once: the rarer seed color leaks onto the seed frame, and only the
#     visibly shorter spokes disappear. Because deleted spokes leave behind recolored frames, the two-stage process is
#     readable by eye from a few examples.
# - f) Visual cleanliness:
#   - The scene stays sparse: one thin spoke per seed, one central hub, and four rectangular seeds. No extra markers,
#     legends, or overlapping annotations are added.
# - g) Inferability of the color change:
#   - The rule depends only on visible minority/majority counts inside each seed, never on numeric color ids.
#   - The delete rule depends only on visible geometric length categories. The generator uses two clear buckets
#     (short/direct vs long/detoured) so samples support a human-learnable "short disappears" abstraction.

from __future__ import annotations

from collections import Counter
import importlib.util
from pathlib import Path
import random
import sys

import numpy as np

try:
    from arc_utils import components_mask, get_bg, label_mask
except ImportError:
    ROOT = Path(__file__).resolve().parents[4]
    SPEC = importlib.util.spec_from_file_location("arc_utils_combo_800d221b_11", ROOT / "solvers" / "arc_utils.py")
    _arc_utils = importlib.util.module_from_spec(SPEC)
    sys.modules[SPEC.name] = _arc_utils
    assert SPEC.loader is not None
    SPEC.loader.exec_module(_arc_utils)
    components_mask = _arc_utils.components_mask
    get_bg = _arc_utils.get_bg
    label_mask = _arc_utils.label_mask


GRID_SHAPE = (27, 27)
HUB_SIZE = 3
HUB_R0 = 12
HUB_C0 = 12
SHORT_BRANCH_THRESHOLD = 18
GEN_RETRIES = 200

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
CARDINALS = ("N", "E", "S", "W")

SEED_BBOXES: dict[str, tuple[int, int, int, int]] = {
    "N": (0, 3, 2, 7),
    "E": (3, 24, 7, 26),
    "S": (24, 19, 26, 23),
    "W": (19, 0, 23, 2),
}

BRANCH_POLYLINES: dict[str, dict[str, list[tuple[int, int]]]] = {
    "N": {
        "short": [(11, 13), (4, 13), (4, 5)],
        "long": [(11, 13), (8, 13), (8, 9), (5, 9), (5, 11), (4, 11), (4, 5)],
    },
    "E": {
        "short": [(13, 15), (13, 22), (5, 22)],
        "long": [(13, 15), (13, 18), (9, 18), (9, 16), (6, 16), (6, 22), (5, 22)],
    },
    "S": {
        "short": [(15, 13), (22, 13), (22, 21)],
        "long": [(15, 13), (18, 13), (18, 17), (21, 17), (21, 15), (22, 15), (22, 21)],
    },
    "W": {
        "short": [(13, 11), (13, 4), (21, 4)],
        "long": [(13, 11), (13, 8), (17, 8), (17, 10), (20, 10), (20, 4), (21, 4)],
    },
}

SHORT_DIR_OPTIONS = [
    ("N",),
    ("E",),
    ("S",),
    ("W",),
    ("N", "S"),
    ("E", "W"),
]


def _polyline(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for idx, ((r0, c0), (r1, c1)) in enumerate(zip(points, points[1:])):
        if r0 == r1:
            step = 1 if c1 >= c0 else -1
            seg = [(r0, c) for c in range(c0, c1 + step, step)]
        elif c0 == c1:
            step = 1 if r1 >= r0 else -1
            seg = [(r, c0) for r in range(r0, r1 + step, step)]
        else:
            raise AssertionError("branch template must be orthogonal")
        if idx:
            seg = seg[1:]
        cells.extend(seg)
    return cells


def _compute_border_cells(bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    H, W = shape
    out: set[tuple[int, int]] = set()
    if r0 > 0:
        out.update((r0 - 1, c) for c in range(c0, c1 + 1))
    if r1 < H - 1:
        out.update((r1 + 1, c) for c in range(c0, c1 + 1))
    if c0 > 0:
        out.update((r, c0 - 1) for r in range(r0, r1 + 1))
    if c1 < W - 1:
        out.update((r, c1 + 1) for r in range(r0, r1 + 1))
    if r0 > 0 and c0 > 0:
        out.add((r0 - 1, c0 - 1))
    if r0 > 0 and c1 < W - 1:
        out.add((r0 - 1, c1 + 1))
    if r1 < H - 1 and c0 > 0:
        out.add((r1 + 1, c0 - 1))
    if r1 < H - 1 and c1 < W - 1:
        out.add((r1 + 1, c1 + 1))
    return out


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
    return float((deg2 - deg4) / max(1, int(pts.shape[0])) * int(pts.shape[0]))


def _infer_path_color(g: np.ndarray, bg: int) -> int:
    colors = sorted({int(x) for x in np.unique(g)} - {int(bg)})
    assert colors, "missing non-background colors"
    return int(max(colors, key=lambda color: _comp_deg_score(g == int(color))))


def _find_hub(g: np.ndarray, path_color: int) -> tuple[np.ndarray, int, int, int]:
    H, W = g.shape
    mask = (g == int(path_color)).astype(int)
    ps = mask.cumsum(axis=0).cumsum(axis=1)

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
                    found = (k, r0, c0)
        if found is not None:
            best = found
    assert best is not None, "hub square not found"
    k, r0, c0 = best
    hub = np.zeros(g.shape, dtype=bool)
    hub[r0 : r0 + k, c0 : c0 + k] = True
    return hub, int(k), int(r0), int(c0)


def _seed_data(
    g: np.ndarray,
    bg: int,
    path_color: int,
) -> tuple[np.ndarray, dict[int, int], dict[int, int], dict[int, set[tuple[int, int]]], dict[tuple[int, int], int]]:
    H, W = g.shape
    seed_mask = (g != int(bg)) & (g != int(path_color))
    seed_lab, seed_n = label_mask(seed_mask, connectivity=4)
    assert int(seed_n) > 0, "grid must contain seeds"
    seed_major: dict[int, int] = {}
    seed_minor: dict[int, int] = {}
    seed_border_cells: dict[int, set[tuple[int, int]]] = {}
    border_owner: dict[tuple[int, int], int] = {}

    for sid in range(1, int(seed_n) + 1):
        pts = np.argwhere(seed_lab == int(sid))
        rn = int(pts[:, 0].min())
        rx = int(pts[:, 0].max())
        cn = int(pts[:, 1].min())
        cx = int(pts[:, 1].max())
        vals = [int(g[int(r), int(c)]) for r, c in pts]
        counts = Counter(vals)
        ranked_major = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        ranked_minor = sorted(counts.items(), key=lambda kv: (kv[1], kv[0]))
        assert len(ranked_major) >= 2, "seed interior must use at least two colors"
        assert int(ranked_major[0][1]) > int(ranked_minor[0][1]), "seed minority must be strict"
        seed_major[int(sid)] = int(ranked_major[0][0])
        seed_minor[int(sid)] = int(ranked_minor[0][0])

        borders: set[tuple[int, int]] = set()
        for rr in range(max(0, rn - 1), min(H - 1, rx + 1) + 1):
            for cc in range(max(0, cn - 1), min(W - 1, cx + 1) + 1):
                if rn <= rr <= rx and cn <= cc <= cx:
                    continue
                if int(g[rr, cc]) == int(path_color):
                    borders.add((int(rr), int(cc)))
        assert borders, "each seed must have an adjacent P border"
        seed_border_cells[int(sid)] = borders
        for cell in borders:
            prev = border_owner.get(cell)
            assert prev in {None, int(sid)}, "seed borders must not overlap"
            border_owner[cell] = int(sid)

    return seed_lab, seed_major, seed_minor, seed_border_cells, border_owner


def _branch_data(
    g: np.ndarray,
    path_color: int,
    hub: np.ndarray,
    border_owner: dict[tuple[int, int], int],
) -> tuple[np.ndarray, dict[int, int], dict[int, np.ndarray]]:
    H, W = g.shape
    border_mask = np.zeros((H, W), dtype=bool)
    for r, c in border_owner:
        border_mask[int(r), int(c)] = True

    branch_mask = (g == int(path_color)) & (~hub) & (~border_mask)
    branch_lab, branch_n = label_mask(branch_mask, connectivity=4)
    branch_seed: dict[int, int] = {}
    branch_pts: dict[int, np.ndarray] = {}

    for bid in range(1, int(branch_n) + 1):
        pts = np.argwhere(branch_lab == int(bid))
        branch_pts[int(bid)] = pts
        touches = Counter()
        for r, c in pts:
            r = int(r)
            c = int(c)
            for dr, dc in DIRS4:
                rr = int(r + dr)
                cc = int(c + dc)
                sid = border_owner.get((rr, cc))
                if sid is not None:
                    touches[int(sid)] += 1
        assert touches, "every branch path must touch a seed border"
        best_sid = int(touches.most_common(1)[0][0])
        assert len(touches) == 1, f"branch path must map to exactly one seed, got {touches}"
        branch_seed[int(bid)] = best_sid

    return branch_lab, branch_seed, branch_pts


def _solve_grid(grid: np.ndarray, *, use_minor: bool, delete_short: bool) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    path_color = _infer_path_color(g, bg)
    hub, k, r0, c0 = _find_hub(g, path_color)
    seed_lab, seed_major, seed_minor, seed_border_cells, border_owner = _seed_data(g, bg, path_color)
    branch_lab, branch_seed, branch_pts = _branch_data(g, path_color, hub, border_owner)

    out = g.copy()

    seed_pick = seed_minor if use_minor else seed_major
    for sid, cells in seed_border_cells.items():
        col = int(seed_pick[int(sid)])
        for r, c in cells:
            out[int(r), int(c)] = int(col)

    for bid, pts in branch_pts.items():
        sid = int(branch_seed[int(bid)])
        col = int(seed_pick[int(sid)])
        for r, c in pts:
            out[int(r), int(c)] = int(col)

    if delete_short:
        for bid, pts in branch_pts.items():
            if int(pts.shape[0]) < SHORT_BRANCH_THRESHOLD:
                for r, c in pts:
                    out[int(r), int(c)] = int(bg)

    touch_cols = Counter()
    H, W = g.shape
    for r in range(H):
        for c in range(W):
            if not bool(hub[r, c]):
                continue
            for dr, dc in DIRS4:
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < H and 0 <= cc < W and int(branch_lab[rr, cc]) != 0 and int(out[rr, cc]) != int(bg):
                    touch_cols[int(out[rr, cc])] += 1

    assert touch_cols, "at least one long branch must survive to color the hub center"
    out[int(r0 + k // 2), int(c0 + k // 2)] = int(touch_cols.most_common(1)[0][0])
    return out


def analyze_behavior(grid: np.ndarray) -> dict[str, int]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    path_color = _infer_path_color(g, bg)
    hub, _k, _r0, _c0 = _find_hub(g, path_color)
    _seed_lab, _seed_major, _seed_minor, _seed_border_cells, border_owner = _seed_data(g, bg, path_color)
    branch_lab, _branch_seed, branch_pts = _branch_data(g, path_color, hub, border_owner)

    combo = _solve_grid(g, use_minor=True, delete_short=True)
    minor_no_delete = _solve_grid(g, use_minor=True, delete_short=False)
    major_delete = _solve_grid(g, use_minor=False, delete_short=True)

    branch_lengths = [int(pts.shape[0]) for pts in branch_pts.values()]
    short_count = sum(length < SHORT_BRANCH_THRESHOLD for length in branch_lengths)
    long_count = sum(length >= SHORT_BRANCH_THRESHOLD for length in branch_lengths)
    deleted_pixels = int(np.count_nonzero((combo == int(bg)) & (minor_no_delete != int(bg))))
    minority_delta = int(np.count_nonzero(combo != major_delete))

    hub_adj_survivors = 0
    H, W = g.shape
    for r in range(H):
        for c in range(W):
            if not bool(hub[r, c]):
                continue
            for dr, dc in DIRS4:
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < H and 0 <= cc < W and int(branch_lab[rr, cc]) != 0 and int(combo[rr, cc]) != int(bg):
                    hub_adj_survivors += 1

    return {
        "branch_count": int(len(branch_pts)),
        "short_count": int(short_count),
        "long_count": int(long_count),
        "min_branch_length": int(min(branch_lengths)),
        "max_branch_length": int(max(branch_lengths)),
        "deleted_pixels": int(deleted_pixels),
        "minority_delta": int(minority_delta),
        "hub_adj_survivors": int(hub_adj_survivors),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each seed-linked spoke by the seed's rarer color, then erase only the short spokes.

    Input structure:
    - The background is the most common color.
    - One non-background color P forms a thin orthogonal network with a solid square hub near the center.
    - Four mixed-color seed rectangles sit on the grid edges and each seed has a P frame on its non-edge sides.

    Rule:
    - Infer P from the network-like component and locate the largest solid all-P square as the hub.
    - For each seed, read its interior colors only, identify the strict minority color, and recolor both the seed's
      P frame and its attached non-hub spoke-path to that minority color.
    - After recoloring, erase every spoke-path whose length is below the task threshold by painting only that path
      back to background; the recolored seed frame remains.
    - Recolor the hub center pixel to the most frequent surviving spoke color among the spoke pixels adjacent to the hub.

    Key insight:
    - The output is governed by two stacked cues: first the rarer seed color flows outward, then only the genuinely
      short path segments vanish. Treating seed frames separately from the removable spoke paths is essential.
    """
    return _solve_grid(grid, use_minor=True, delete_short=True)


def _sample_seed_fill(
    rng: random.Random,
    area: int,
    major: int,
    minor: int,
) -> list[int]:
    n_minor = rng.choice([4, 5])
    assert n_minor < area - n_minor, "minority count must stay strict"
    vals = [minor] * n_minor + [major] * (area - n_minor)
    rng.shuffle(vals)
    return vals


def _branch_cells_for_dirs(short_dirs: set[str]) -> dict[str, list[tuple[int, int]]]:
    cells: dict[str, list[tuple[int, int]]] = {}
    for direction in CARDINALS:
        mode = "short" if direction in short_dirs else "long"
        cells[direction] = _polyline(BRANCH_POLYLINES[direction][mode])
    return cells


def _validate_layout(branch_cells: dict[str, list[tuple[int, int]]]) -> None:
    H, W = GRID_SHAPE
    hub = {(r, c) for r in range(HUB_R0, HUB_R0 + HUB_SIZE) for c in range(HUB_C0, HUB_C0 + HUB_SIZE)}
    all_seed_cells: set[tuple[int, int]] = set()
    all_border_cells: set[tuple[int, int]] = set()
    for bbox in SEED_BBOXES.values():
        r0, c0, r1, c1 = bbox
        seed_cells = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
        border_cells = _compute_border_cells(bbox, GRID_SHAPE)
        assert all_seed_cells.isdisjoint(seed_cells)
        assert all_seed_cells.isdisjoint(border_cells)
        assert all_border_cells.isdisjoint(seed_cells)
        assert all_border_cells.isdisjoint(border_cells)
        assert hub.isdisjoint(seed_cells)
        assert hub.isdisjoint(border_cells)
        all_seed_cells |= seed_cells
        all_border_cells |= border_cells

    occupied: set[tuple[int, int]] = set()
    for direction, cells in branch_cells.items():
        cell_set = set(cells)
        assert len(cell_set) == len(cells), f"branch {direction} self-intersects"
        assert occupied.isdisjoint(cell_set), f"branches overlap at {direction}"
        assert hub.isdisjoint(cell_set), f"branch {direction} overlaps hub"
        assert all_seed_cells.isdisjoint(cell_set), f"branch {direction} overlaps seed interior"
        for r, c in cell_set:
            assert 0 <= r < H and 0 <= c < W, "branch out of bounds"
        occupied |= cell_set

    border_by_dir = {d: _compute_border_cells(SEED_BBOXES[d], GRID_SHAPE) for d in CARDINALS}
    for direction, cells in branch_cells.items():
        endpoint = cells[-1]
        touches = 0
        for dr, dc in DIRS4:
            if (endpoint[0] + dr, endpoint[1] + dc) in border_by_dir[direction]:
                touches += 1
        assert touches >= 1, f"branch {direction} must end next to its border"
        for other_dir in CARDINALS:
            if other_dir == direction:
                continue
            for dr, dc in DIRS4:
                assert (endpoint[0] + dr, endpoint[1] + dc) not in border_by_dir[other_dir], "branch endpoint ambiguous"


def _generate_one(rng: random.Random) -> np.ndarray:
    H, W = GRID_SHAPE
    bg, path_color, seed_a, seed_b = rng.sample(list(range(10)), 4)
    short_dirs = set(rng.choice(SHORT_DIR_OPTIONS))
    branch_cells = _branch_cells_for_dirs(short_dirs)
    _validate_layout(branch_cells)

    dominant_minor = rng.choice([seed_a, seed_b])
    other_minor = seed_b if dominant_minor == seed_a else seed_a
    desired_minor: dict[str, int] = {}

    long_dirs = [d for d in CARDINALS if d not in short_dirs]
    for direction in long_dirs:
        desired_minor[direction] = int(dominant_minor)

    short_list = list(short_dirs)
    rng.shuffle(short_list)
    if short_list:
        desired_minor[short_list[0]] = int(other_minor)
    for direction in short_list[1:]:
        desired_minor[direction] = int(rng.choice([dominant_minor, other_minor]))

    if len(set(desired_minor.values())) == 1:
        flip_dir = short_list[0] if short_list else CARDINALS[0]
        desired_minor[flip_dir] = int(other_minor if desired_minor[flip_dir] == dominant_minor else dominant_minor)

    g = np.full((H, W), int(bg), dtype=int)

    for direction, bbox in SEED_BBOXES.items():
        r0, c0, r1, c1 = bbox
        cells = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
        minor = int(desired_minor[direction])
        major = int(seed_b if minor == seed_a else seed_a)
        vals = _sample_seed_fill(rng, len(cells), major, minor)
        for (r, c), val in zip(cells, vals):
            g[int(r), int(c)] = int(val)
        for r, c in _compute_border_cells(bbox, GRID_SHAPE):
            g[int(r), int(c)] = int(path_color)

    for cells in branch_cells.values():
        for r, c in cells:
            g[int(r), int(c)] = int(path_color)

    g[HUB_R0 : HUB_R0 + HUB_SIZE, HUB_C0 : HUB_C0 + HUB_SIZE] = int(path_color)

    metrics = analyze_behavior(g)
    assert metrics["short_count"] >= 1, "combo requires at least one short branch"
    assert metrics["long_count"] >= 1, "combo requires at least one long branch"
    assert metrics["deleted_pixels"] > 0, "delete stage must change the output"
    assert metrics["minority_delta"] > 0, "minority recolor stage must differ from majority recolor"
    assert metrics["hub_adj_survivors"] > 0, "hub center needs surviving branches"
    return g


def generate_input() -> np.ndarray:
    for attempt in range(GEN_RETRIES):
        rng = random.Random(random.randrange(1 << 30) ^ attempt)
        try:
            grid = _generate_one(rng)
            main(grid)
            return grid
        except Exception:
            continue
    raise RuntimeError(f"generate_input failed after {GEN_RETRIES} attempts")
