# Combo variant of 800d221b (index 23)
# combo_spec: D9/0+D10/1
# Name: 着色删除后裁剪
#
# Design analysis:
# - a) Concrete behavior trace:
#   - In the source family, every seed exports its majority color onto the attached P frame and branch, and the output
#     keeps the original canvas size.
#   - In this combo, the same majority-color recolor happens first. Then every true spoke-path shorter than the task
#     threshold is erased back to background, while the recolored seed frame stays visible. Finally the whole result is
#     cropped to the tight bounding box of all remaining non-background content.
#   - Example: a short north spoke and a long east spoke may both first turn red from their own seed majorities; the
#     north corridor then disappears because it is short, the east corridor remains, and the output frame shrinks to
#     the minimal rectangle containing the hub, all seeds, and the surviving recolored structure.
# - b) Complexity calibration:
#   - The scene stays inside the original visual family: one central 3x3 hub, four edge-oriented seeds, one path color
#     P, and one orthogonal spoke per seed.
#   - The generator always mixes short and long spokes in the same sample, so the delete stage is observable and does
#     not collapse back to plain recoloring.
#   - All objects are inset from the canvas boundary by a fixed margin, which makes D9/0 consistently visible as a
#     size change instead of a no-op crop.
# - c) Hidden-assumption audit:
#   - The original assumptions for inferring P, detecting the solid hub square, and reading seed majorities still
#     hold because those structures are preserved.
#   - The original assumption "all non-hub P cells can be treated as one recolored component" is no longer valid. The
#     solver reconstructs seed-border cells from each seed bbox, removes them from the removable branch mask, and only
#     measures deletion length on the remaining corridor.
#   - The new combo assumes each removable corridor touches exactly one seed border. The fixed branch templates enforce
#     that and prevent overlaps or ambiguous contacts.
# - d) Visual observability and cleanliness:
#   - Majority color is readable directly from each mixed seed interior; deleted spokes are readable directly from
#     visibly shorter corridors; cropping is readable directly from the smaller output frame.
#   - Spatial roles stay separated: interior rectangle, one-pixel border, external corridor, central hub. No region is
#     overloaded with multiple competing meanings.

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
    SPEC = importlib.util.spec_from_file_location("arc_utils_combo_800d221b_23", ROOT / "arc_utils.py")
    arc_utils = importlib.util.module_from_spec(SPEC)
    sys.modules[SPEC.name] = arc_utils
    assert SPEC.loader is not None
    SPEC.loader.exec_module(arc_utils)
    components_mask = arc_utils.components_mask
    get_bg = arc_utils.get_bg
    label_mask = arc_utils.label_mask


GRID_SHAPE = (31, 31)
HUB_SIZE = 3
HUB_R0 = 14
HUB_C0 = 14
SHORT_BRANCH_THRESHOLD = 18
GEN_RETRIES = 200

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
CARDINALS = ("N", "E", "S", "W")

SEED_BBOXES: dict[str, tuple[int, int, int, int]] = {
    "N": (2, 5, 4, 9),
    "E": (5, 26, 9, 28),
    "S": (26, 21, 28, 25),
    "W": (21, 2, 25, 4),
}

BRANCH_POLYLINES: dict[str, dict[str, list[tuple[int, int]]]] = {
    "N": {
        "short": [(13, 15), (6, 15), (6, 7)],
        "long": [(13, 15), (10, 15), (10, 11), (7, 11), (7, 13), (6, 13), (6, 7)],
    },
    "E": {
        "short": [(15, 17), (15, 24), (7, 24)],
        "long": [(15, 17), (15, 20), (11, 20), (11, 18), (8, 18), (8, 24), (7, 24)],
    },
    "S": {
        "short": [(17, 15), (24, 15), (24, 23)],
        "long": [(17, 15), (20, 15), (20, 19), (23, 19), (23, 17), (24, 17), (24, 23)],
    },
    "W": {
        "short": [(15, 13), (15, 6), (23, 6)],
        "long": [(15, 13), (15, 10), (19, 10), (19, 12), (22, 12), (22, 6), (23, 6)],
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


def _compute_border_cells(bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    H, W = shape
    out: set[tuple[int, int]] = set()
    if r0 > 0:
        out.update((int(r0 - 1), int(c)) for c in range(c0, c1 + 1))
    if r1 < H - 1:
        out.update((int(r1 + 1), int(c)) for c in range(c0, c1 + 1))
    if c0 > 0:
        out.update((int(r), int(c0 - 1)) for r in range(r0, r1 + 1))
    if c1 < W - 1:
        out.update((int(r), int(c1 + 1)) for r in range(r0, r1 + 1))
    if r0 > 0 and c0 > 0:
        out.add((int(r0 - 1), int(c0 - 1)))
    if r0 > 0 and c1 < W - 1:
        out.add((int(r0 - 1), int(c1 + 1)))
    if r1 < H - 1 and c0 > 0:
        out.add((int(r1 + 1), int(c0 - 1)))
    if r1 < H - 1 and c1 < W - 1:
        out.add((int(r1 + 1), int(c1 + 1)))
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
    return float((deg2 - deg4) * int(pts.shape[0]) / max(1, int(pts.shape[0])))


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
                    found = (int(k), int(r0), int(c0))
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
) -> tuple[np.ndarray, dict[int, int], dict[int, set[tuple[int, int]]], dict[tuple[int, int], int]]:
    H, W = g.shape
    seed_mask = (g != int(bg)) & (g != int(path_color))
    seed_lab, seed_n = label_mask(seed_mask, connectivity=4)
    assert int(seed_n) > 0, "grid must contain seeds"

    seed_major: dict[int, int] = {}
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
        ranked = sorted(counts.items(), key=lambda kv: (-int(kv[1]), int(kv[0])))
        assert len(ranked) >= 2, "seed interior must use at least two colors"
        assert int(ranked[0][1]) > int(ranked[1][1]), "seed majority must be strict"
        seed_major[int(sid)] = int(ranked[0][0])

        borders = _compute_border_cells((rn, cn, rx, cx), (H, W))
        borders = {cell for cell in borders if int(g[cell[0], cell[1]]) == int(path_color)}
        assert borders, "each seed must have a path-color border"
        seed_border_cells[int(sid)] = borders
        for cell in borders:
            prev = border_owner.get(cell)
            assert prev in {None, int(sid)}, "seed borders must not overlap"
            border_owner[cell] = int(sid)

    return seed_lab, seed_major, seed_border_cells, border_owner


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
    assert int(branch_n) > 0, "grid must contain removable branches"

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
                sid = border_owner.get((int(r + dr), int(c + dc)))
                if sid is not None:
                    touches[int(sid)] += 1
        assert touches, "every branch path must touch a seed border"
        best_sid = int(touches.most_common(1)[0][0])
        assert len(touches) == 1, f"branch path must map to exactly one seed, got {touches}"
        branch_seed[int(bid)] = best_sid

    return branch_lab, branch_seed, branch_pts


def _crop_non_bg(grid: np.ndarray, bg: int) -> np.ndarray:
    pts = np.argwhere(grid != int(bg))
    assert pts.size > 0, "cropped output cannot be empty"
    r0 = int(pts[:, 0].min())
    r1 = int(pts[:, 0].max())
    c0 = int(pts[:, 1].min())
    c1 = int(pts[:, 1].max())
    return grid[r0 : r1 + 1, c0 : c1 + 1]


def _transform_grid(grid: np.ndarray, *, delete_short: bool, crop: bool) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    path_color = int(_infer_path_color(g, bg))
    hub, k, r0, c0 = _find_hub(g, path_color)
    _seed_lab, seed_major, seed_border_cells, border_owner = _seed_data(g, bg, path_color)
    branch_lab, branch_seed, branch_pts = _branch_data(g, path_color, hub, border_owner)

    out = g.copy()

    for sid, cells in seed_border_cells.items():
        col = int(seed_major[int(sid)])
        for rr, cc in cells:
            out[int(rr), int(cc)] = int(col)

    for bid, pts in branch_pts.items():
        sid = int(branch_seed[int(bid)])
        col = int(seed_major[int(sid)])
        for rr, cc in pts:
            out[int(rr), int(cc)] = int(col)

    if delete_short:
        for pts in branch_pts.values():
            if int(pts.shape[0]) < SHORT_BRANCH_THRESHOLD:
                for rr, cc in pts:
                    out[int(rr), int(cc)] = int(bg)

    touch_cols = Counter()
    H, W = g.shape
    for rr in range(H):
        for cc in range(W):
            if not bool(hub[rr, cc]):
                continue
            for dr, dc in DIRS4:
                nr = int(rr + dr)
                nc = int(cc + dc)
                if 0 <= nr < H and 0 <= nc < W and int(branch_lab[nr, nc]) != 0 and int(out[nr, nc]) != int(bg):
                    touch_cols[int(out[nr, nc])] += 1

    assert touch_cols, "at least one long branch must survive to color the hub center"
    out[int(r0 + k // 2), int(c0 + k // 2)] = int(touch_cols.most_common(1)[0][0])
    return _crop_non_bg(out, bg) if crop else out


def analyze_behavior(grid: np.ndarray) -> dict[str, int]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    path_color = int(_infer_path_color(g, bg))
    hub, _k, _r0, _c0 = _find_hub(g, path_color)
    _seed_lab, _seed_major, _seed_border_cells, border_owner = _seed_data(g, bg, path_color)
    _branch_lab, _branch_seed, branch_pts = _branch_data(g, path_color, hub, border_owner)

    combo_full = _transform_grid(g, delete_short=True, crop=False)
    recolor_full = _transform_grid(g, delete_short=False, crop=False)
    combo = _transform_grid(g, delete_short=True, crop=True)

    lengths = [int(pts.shape[0]) for pts in branch_pts.values()]
    deleted_pixels = int(np.count_nonzero((combo_full == int(bg)) & (recolor_full != int(bg))))

    return {
        "branch_count": int(len(branch_pts)),
        "short_count": int(sum(length < SHORT_BRANCH_THRESHOLD for length in lengths)),
        "long_count": int(sum(length >= SHORT_BRANCH_THRESHOLD for length in lengths)),
        "min_branch_length": int(min(lengths)),
        "max_branch_length": int(max(lengths)),
        "deleted_pixels": int(deleted_pixels),
        "cropped_h": int(combo.shape[0]),
        "cropped_w": int(combo.shape[1]),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each seed-linked branch by seed majority, delete short spokes, then crop tightly.

    Input structure:
    - The background is the most common color.
    - One non-background color P forms a thin orthogonal network with a solid square hub near the center.
    - Mixed-color seed rectangles have a one-pixel P border on all sides and one removable corridor connecting each
      border back to the hub.

    Rule:
    - Infer P and the hub as in the base task.
    - For each seed, read its strict interior majority color and recolor both its P border and its attached corridor
      to that majority.
    - After recoloring, erase every corridor shorter than the task threshold by painting only that corridor back to
      background.
    - Recolor the hub center pixel to the most frequent surviving corridor color adjacent to the hub.
    - Finally crop the output to the minimal bounding box of all non-background cells.

    Key insight:
    - The transformation is a visible three-step pipeline: majority-color flow, short-corridor deletion, then tight
      cropping.
    """
    return _transform_grid(grid, delete_short=True, crop=True)


def _sample_seed_fill(
    rng: random.Random,
    area: int,
    major: int,
    minor: int,
) -> list[int]:
    n_minor = rng.choice([4, 5])
    assert n_minor < area - n_minor, "minority count must stay strict"
    vals = [int(minor)] * int(n_minor) + [int(major)] * int(area - n_minor)
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
        seed_cells = {(int(r), int(c)) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
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
            if (int(endpoint[0] + dr), int(endpoint[1] + dc)) in border_by_dir[direction]:
                touches += 1
        assert touches >= 1, f"branch {direction} must end next to its border"
        for other_dir in CARDINALS:
            if other_dir == direction:
                continue
            for dr, dc in DIRS4:
                assert (int(endpoint[0] + dr), int(endpoint[1] + dc)) not in border_by_dir[other_dir], "branch endpoint ambiguous"


def _generate_one(rng: random.Random) -> np.ndarray:
    H, W = GRID_SHAPE
    bg, path_color, seed_a, seed_b = rng.sample(list(range(10)), 4)
    short_dirs = set(rng.choice(SHORT_DIR_OPTIONS))
    branch_cells = _branch_cells_for_dirs(short_dirs)
    _validate_layout(branch_cells)

    winner_major = int(rng.choice([seed_a, seed_b]))
    loser_major = int(seed_b) if int(winner_major) == int(seed_a) else int(seed_a)
    desired_major: dict[str, int] = {}

    long_dirs = [d for d in CARDINALS if d not in short_dirs]
    for direction in long_dirs:
        desired_major[direction] = int(winner_major)

    short_list = list(short_dirs)
    rng.shuffle(short_list)
    if short_list:
        desired_major[short_list[0]] = int(loser_major)
    for direction in short_list[1:]:
        desired_major[direction] = int(rng.choice([winner_major, loser_major]))

    g = np.full((H, W), int(bg), dtype=int)

    for direction, bbox in SEED_BBOXES.items():
        r0, c0, r1, c1 = bbox
        cells = [(int(r), int(c)) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
        major = int(desired_major[direction])
        minor = int(seed_b) if int(major) == int(seed_a) else int(seed_a)
        vals = _sample_seed_fill(rng, len(cells), major, minor)
        for (r, c), val in zip(cells, vals):
            g[int(r), int(c)] = int(val)
        for r, c in _compute_border_cells(bbox, GRID_SHAPE):
            g[int(r), int(c)] = int(path_color)

    g[HUB_R0 : HUB_R0 + HUB_SIZE, HUB_C0 : HUB_C0 + HUB_SIZE] = int(path_color)
    for cells in branch_cells.values():
        for r, c in cells:
            g[int(r), int(c)] = int(path_color)

    info = analyze_behavior(g)
    if info["branch_count"] != 4:
        raise AssertionError("expected four branches")
    if info["short_count"] < 1 or info["long_count"] < 2:
        raise AssertionError("generator must include both deleted and surviving branches")
    if info["deleted_pixels"] <= 0:
        raise AssertionError("combo must delete at least one short branch")
    if info["cropped_h"] >= H or info["cropped_w"] >= W:
        raise AssertionError("D9/0 crop must shrink the canvas")
    return g


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        rng = random.Random(random.randrange(1 << 30))
        try:
            return _generate_one(rng)
        except AssertionError:
            continue
    raise RuntimeError("failed to generate input for combo 800d221b D9/0+D10/1")
