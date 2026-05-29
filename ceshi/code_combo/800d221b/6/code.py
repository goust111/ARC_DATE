# Combo variant of 800d221b (index 6)
# combo_spec: D1/0+D6/2
# Name: 少数色+方向匹配
#
# Design analysis:
# - Concrete behavior trace:
#   A north-starting branch may physically terminate at the east seed border. In the original task, that whole
#   branch would inherit the touched east seed's majority color. In this combo, the branch instead reads the seed
#   that sits north of the hub and uses that seed's minority color. The touched border is therefore recolored by a
#   spatially different seed, which makes the D6 change visible, and the chosen color is the rarer seed color,
#   which makes the D1 change visible at the same time.
# - Complexity calibration:
#   The generator always makes four edge seeds, one in each cardinal direction, and four branches that leave the
#   hub through the four cardinal sides. Physical branch endpoints are cyclically shifted (clockwise or
#   counterclockwise), so direction matching never collapses back to contact matching. Minority colors use a 3:1
#   directional distribution to avoid hub-center ties while still leaving repeated evidence across samples.
# - Hidden-assumption audit:
#   The original adjacency-count solver assumed each branch's touched seed determines its color. That assumption is
#   intentionally broken here. The new solver instead relies on two guarantees from the generator:
#   exactly one seed lies in each of N/E/S/W around the hub, and each branch component touches the hub from exactly
#   one side. The generator keeps routes separated so components do not merge and no directional ambiguity is
#   introduced.
# - Placement legality and observability:
#   Seeds stay on distinct grid edges, the hub remains a unique 3x3 P-square, and routes use separated lanes in the
#   four quadrants so they neither overlap nor leave the grid. Because some branches visibly end at the "wrong"
#   seed while being recolored from the seed on their launch side, a human can infer the direction rule from only a
#   few examples.

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
    SPEC = importlib.util.spec_from_file_location("arc_utils_combo_800d221b_6", ROOT / "arc_utils.py")
    _arc_utils = importlib.util.module_from_spec(SPEC)
    sys.modules[SPEC.name] = _arc_utils
    assert SPEC.loader is not None
    SPEC.loader.exec_module(_arc_utils)
    components_mask = _arc_utils.components_mask
    get_bg = _arc_utils.get_bg
    label_mask = _arc_utils.label_mask


DIRS = ("N", "E", "S", "W")
DIR_VECS = {"N": (-1, 0), "E": (0, 1), "S": (1, 0), "W": (0, -1)}
CW_SHIFT = {"N": "E", "E": "S", "S": "W", "W": "N"}
CCW_SHIFT = {"N": "W", "E": "N", "S": "E", "W": "S"}

GRID_SIZE_MIN = 21
GRID_SIZE_MAX = 29
SEED_W_MIN = 4
SEED_W_MAX = 6
SEED_H_MIN = 3
SEED_H_MAX = 5
HUB_SIZE = 3
GEN_RETRIES = 200


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
        d = 0
        if r > 0 and bool(m[r - 1, c]):
            d += 1
        if r + 1 < H and bool(m[r + 1, c]):
            d += 1
        if c > 0 and bool(m[r, c - 1]):
            d += 1
        if c + 1 < W and bool(m[r, c + 1]):
            d += 1
        if d == 2:
            deg2 += 1
        elif d == 4:
            deg4 += 1
    size = int(pts.shape[0])
    return float((deg2 - deg4) / max(1, size) * size)


def _infer_path_color(g: np.ndarray, bg: int) -> int:
    colors = sorted({int(x) for x in np.unique(g)} - {int(bg)})
    assert colors, "missing non-background colors"
    return int(max(colors, key=lambda c: _comp_deg_score(g == int(c))))


def _largest_p_square(g: np.ndarray, P: int) -> tuple[int, int, int]:
    H, W = g.shape
    mask = (g == int(P)).astype(int)
    ps = mask.cumsum(axis=0).cumsum(axis=1)

    def rect_sum(r0: int, c0: int, r1: int, c1: int) -> int:
        s = int(ps[r1, c1])
        if r0 > 0:
            s -= int(ps[r0 - 1, c1])
        if c0 > 0:
            s -= int(ps[r1, c0 - 1])
        if r0 > 0 and c0 > 0:
            s += int(ps[r0 - 1, c0 - 1])
        return int(s)

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
    return best


def _minority_color(vals: list[int]) -> int:
    counts = Counter(int(v) for v in vals)
    assert len(counts) >= 2, "seed interior must contain at least two colors"
    items = sorted(counts.items(), key=lambda kv: (kv[1], kv[0]))
    assert items[0][1] < items[-1][1], "seed minority must be strict"
    return int(items[0][0])


def _seed_direction_from_bbox(bbox: tuple[int, int, int, int], hub_bbox: tuple[int, int, int, int]) -> str:
    sr0, sc0, sr1, sc1 = bbox
    hr0, hc0, hr1, hc1 = hub_bbox
    if sr1 < hr0:
        return "N"
    if sr0 > hr1:
        return "S"
    if sc1 < hc0:
        return "W"
    if sc0 > hc1:
        return "E"
    raise AssertionError("seed overlaps hub zone")


def _branch_launch_direction(pts: np.ndarray, hub_bbox: tuple[int, int, int, int]) -> str:
    hr0, hc0, hr1, hc1 = hub_bbox
    dirs = set()
    for r, c in pts:
        r = int(r)
        c = int(c)
        if hr0 <= r <= hr1 and hc0 <= c <= hc1:
            continue
        if c >= hc0 and c <= hc1 and r == hr0 - 1:
            dirs.add("N")
        elif c >= hc0 and c <= hc1 and r == hr1 + 1:
            dirs.add("S")
        elif r >= hr0 and r <= hr1 and c == hc0 - 1:
            dirs.add("W")
        elif r >= hr0 and r <= hr1 and c == hc1 + 1:
            dirs.add("E")
    assert len(dirs) == 1, f"expected one launch side, got {dirs}"
    return next(iter(dirs))


def _analyze_grid(grid: np.ndarray) -> dict[str, object]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    P = _infer_path_color(g, bg)
    k, r0, c0 = _largest_p_square(g, P)
    r1 = r0 + k - 1
    c1 = c0 + k - 1
    hub_bbox = (r0, c0, r1, c1)
    hub = np.zeros(g.shape, dtype=bool)
    hub[r0 : r1 + 1, c0 : c1 + 1] = True

    seed_mask = (g != bg) & (g != P)
    seed_lab, seed_n = label_mask(seed_mask, connectivity=4)
    assert int(seed_n) == 4, "generator must create four cardinal seeds"
    seeds_by_dir: dict[str, dict[str, object]] = {}
    for sid in range(1, int(seed_n) + 1):
        pts = np.argwhere(seed_lab == sid)
        bbox = (
            int(pts[:, 0].min()),
            int(pts[:, 1].min()),
            int(pts[:, 0].max()),
            int(pts[:, 1].max()),
        )
        sdir = _seed_direction_from_bbox(bbox, hub_bbox)
        vals = [int(g[int(r), int(c)]) for r, c in pts]
        seeds_by_dir[sdir] = {
            "id": int(sid),
            "bbox": bbox,
            "minority": _minority_color(vals),
        }
    assert set(seeds_by_dir) == set(DIRS), "need exactly one seed in each cardinal direction"

    branch_mask = (g == P) & (~hub)
    branch_lab, branch_n = label_mask(branch_mask, connectivity=4)
    assert int(branch_n) == 4, "generator must create four separate branch components"
    branches_by_dir: dict[str, dict[str, object]] = {}
    for bid in range(1, int(branch_n) + 1):
        pts = np.argwhere(branch_lab == bid)
        bdir = _branch_launch_direction(pts, hub_bbox)
        branches_by_dir[bdir] = {
            "id": int(bid),
            "pts": pts,
        }
    assert set(branches_by_dir) == set(DIRS), "need one branch launched from each side"
    return {
        "grid": g,
        "bg": bg,
        "P": P,
        "hub_bbox": hub_bbox,
        "hub_mask": hub,
        "seed_lab": seed_lab,
        "seed_dirs": seeds_by_dir,
        "branch_lab": branch_lab,
        "branch_dirs": branches_by_dir,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor a hub-and-branch network by directional seed lookup.

    Input structure:
    - The background is the most frequent color.
    - One color P forms a connected orthogonal network with a solid 3x3 hub near the middle.
    - Four mixed-color seed rectangles lie on the north, east, south, and west edges, each wrapped by P on the
      non-edge sides.

    Rule:
    - Infer P and locate the largest solid P square as the hub.
    - Split the remaining P cells into four branch components and identify each branch by the side from which it
      first leaves the hub.
    - Match that launch direction to the seed located in the same cardinal direction and recolor the whole branch
      component to that seed interior's minority color.
    - Recolor only the hub center pixel to the most frequent color among the recolored branch pixels that touch the
      hub.

    Key insight:
    Physical contact is a red herring here: a branch may end at one seed while borrowing its color from the seed on
    the same side as its launch direction, and the borrowed color is the seed's rarer interior color.
    """
    info = _analyze_grid(grid)
    g = info["grid"]
    hub = info["hub_mask"]
    seed_dirs = info["seed_dirs"]
    branch_lab = info["branch_lab"]
    branch_dirs = info["branch_dirs"]
    r0, c0, r1, c1 = info["hub_bbox"]

    out = g.copy()
    for bdir, branch in branch_dirs.items():
        col = int(seed_dirs[bdir]["minority"])
        bid = int(branch["id"])
        pts = np.argwhere(branch_lab == bid)
        for r, c in pts:
            out[int(r), int(c)] = col

    touch_cols = Counter()
    for bdir, branch in branch_dirs.items():
        bid = int(branch["id"])
        pts = np.argwhere(branch_lab == bid)
        for r, c in pts:
            r = int(r)
            c = int(c)
            if (r == r0 - 1 and c0 <= c <= c1) or (r == r1 + 1 and c0 <= c <= c1):
                touch_cols[int(out[r, c])] += 1
            elif (c == c0 - 1 and r0 <= r <= r1) or (c == c1 + 1 and r0 <= r <= r1):
                touch_cols[int(out[r, c])] += 1
    assert touch_cols, "hub must have adjacent recolored branches"
    out[r0 + 1, c0 + 1] = int(touch_cols.most_common(1)[0][0])
    return out


def _polyline(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for idx, ((r0, c0), (r1, c1)) in enumerate(zip(points, points[1:])):
        if r0 == r1:
            step = 1 if c1 >= c0 else -1
            seg = [(r0, c) for c in range(c0, c1 + step, step)]
        elif c0 == c1:
            step = 1 if r1 >= r0 else -1
            seg = [(r, c0) for r in range(r0, r1 + step, step)]
        else:
            raise AssertionError("non-orthogonal branch segment")
        if idx:
            seg = seg[1:]
        out.extend(seg)
    return out


def _fill_seed_interior(
    g: np.ndarray,
    bbox: tuple[int, int, int, int],
    major: int,
    minor: int,
) -> None:
    r0, c0, r1, c1 = bbox
    cells = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
    random.shuffle(cells)
    area = len(cells)
    n_minor = random.randint(max(2, area // 6), max(2, min(area // 3, area // 2 - 1)))
    for r, c in cells[:n_minor]:
        g[r, c] = minor
    for r, c in cells[n_minor:]:
        g[r, c] = major


def _place_border(g: np.ndarray, P: int, side: str, bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    H, W = g.shape
    r0, c0, r1, c1 = bbox
    cells: set[tuple[int, int]] = set()
    if side != "N":
        for c in range(c0, c1 + 1):
            cells.add((r0 - 1, c))
    if side != "S":
        for c in range(c0, c1 + 1):
            cells.add((r1 + 1, c))
    if side != "W":
        for r in range(r0, r1 + 1):
            cells.add((r, c0 - 1))
    if side != "E":
        for r in range(r0, r1 + 1):
            cells.add((r, c1 + 1))

    for rr, cc in ((r0 - 1, c0 - 1), (r0 - 1, c1 + 1), (r1 + 1, c0 - 1), (r1 + 1, c1 + 1)):
        if 0 <= rr < H and 0 <= cc < W:
            if side == "N" and rr < r0:
                continue
            if side == "S" and rr > r1:
                continue
            if side == "W" and cc < c0:
                continue
            if side == "E" and cc > c1:
                continue
            cells.add((rr, cc))

    for r, c in cells:
        g[r, c] = P
    return cells


def _seed_target(side: str, bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    if side == "N":
        return (r1 + 1, (c0 + c1) // 2)
    if side == "S":
        return (r0 - 1, (c0 + c1) // 2)
    if side == "W":
        return ((r0 + r1) // 2, c1 + 1)
    return ((r0 + r1) // 2, c0 - 1)


def _cardinal_seed_boxes(H: int, W: int, hr0: int, hc0: int) -> dict[str, tuple[int, int, int, int]]:
    cr = hr0 + 1
    cc = hc0 + 1

    top_h = random.randint(SEED_H_MIN, SEED_H_MAX)
    top_w = random.randint(SEED_W_MIN, SEED_W_MAX)
    bot_h = random.randint(SEED_H_MIN, SEED_H_MAX)
    bot_w = random.randint(SEED_W_MIN, SEED_W_MAX)
    left_h = random.randint(SEED_W_MIN, SEED_W_MAX)
    left_w = random.randint(SEED_H_MIN, SEED_H_MAX)
    right_h = random.randint(SEED_W_MIN, SEED_W_MAX)
    right_w = random.randint(SEED_H_MIN, SEED_H_MAX)

    top_c0 = min(max(2, cc - top_w // 2 + random.randint(-1, 1)), W - top_w - 3)
    bot_c0 = min(max(2, cc - bot_w // 2 + random.randint(-1, 1)), W - bot_w - 3)
    left_r0 = min(max(2, cr - left_h // 2 + random.randint(-1, 1)), H - left_h - 3)
    right_r0 = min(max(2, cr - right_h // 2 + random.randint(-1, 1)), H - right_h - 3)

    return {
        "N": (0, top_c0, top_h - 1, top_c0 + top_w - 1),
        "S": (H - bot_h, bot_c0, H - 1, bot_c0 + bot_w - 1),
        "W": (left_r0, 0, left_r0 + left_h - 1, left_w - 1),
        "E": (right_r0, W - right_w, right_r0 + right_h - 1, W - 1),
    }


def _route_cells(
    launch: str,
    physical: str,
    starts: dict[str, tuple[int, int]],
    targets: dict[str, tuple[int, int]],
    lanes: dict[str, int],
    center: tuple[int, int],
) -> set[tuple[int, int]]:
    cr, cc = center
    start = starts[launch]
    end = targets[physical]
    if launch == "N":
        row = lanes["N"]
        pts = [start, (row, cc), (row, end[1]), end]
    elif launch == "S":
        row = lanes["S"]
        pts = [start, (row, cc), (row, end[1]), end]
    elif launch == "E":
        col = lanes["E"]
        pts = [start, (cr, col), (end[0], col), end]
    else:
        col = lanes["W"]
        pts = [start, (cr, col), (end[0], col), end]
    return set(_polyline(pts))


def _choose_shift_map() -> dict[str, str]:
    return CW_SHIFT if random.random() < 0.5 else CCW_SHIFT


def _build_input() -> np.ndarray:
    H = random.randrange(GRID_SIZE_MIN, GRID_SIZE_MAX + 1, 2)
    W = random.randrange(GRID_SIZE_MIN, GRID_SIZE_MAX + 1, 2)
    bg, P, seed_a, seed_b = random.sample(list(range(10)), 4)

    g = np.full((H, W), bg, dtype=int)
    hr0 = H // 2 - 1
    hc0 = W // 2 - 1
    g[hr0 : hr0 + HUB_SIZE, hc0 : hc0 + HUB_SIZE] = P

    boxes = _cardinal_seed_boxes(H, W, hr0, hc0)

    dominant_minor = random.choice([seed_a, seed_b])
    odd_minor = seed_b if dominant_minor == seed_a else seed_a
    odd_dir = random.choice(DIRS)
    minor_by_dir = {d: (odd_minor if d == odd_dir else dominant_minor) for d in DIRS}
    major_by_dir = {d: (seed_b if minor_by_dir[d] == seed_a else seed_a) for d in DIRS}

    for side in DIRS:
        _fill_seed_interior(g, boxes[side], major_by_dir[side], minor_by_dir[side])
        _place_border(g, P, side, boxes[side])

    cr = hr0 + 1
    cc = hc0 + 1
    starts = {
        "N": (hr0 - 1, cc),
        "E": (cr, hc0 + HUB_SIZE),
        "S": (hr0 + HUB_SIZE, cc),
        "W": (cr, hc0 - 1),
    }
    targets = {side: _seed_target(side, boxes[side]) for side in DIRS}
    lanes = {
        "N": hr0 - random.randint(3, 4),
        "S": hr0 + HUB_SIZE - 1 + random.randint(3, 4),
        "W": hc0 - random.randint(3, 4),
        "E": hc0 + HUB_SIZE - 1 + random.randint(3, 4),
    }
    shift_map = _choose_shift_map()

    branch_cells: dict[str, set[tuple[int, int]]] = {}
    for launch in DIRS:
        physical = shift_map[launch]
        branch_cells[launch] = _route_cells(launch, physical, starts, targets, lanes, (cr, cc))

    all_cells: set[tuple[int, int]] = set()
    for launch in DIRS:
        cells = branch_cells[launch]
        if any(g[r, c] not in (bg, P) for r, c in cells):
            raise AssertionError("branch crossed seed interior")
        overlap = all_cells & cells
        overlap -= {targets[shift_map[launch]]}
        if overlap:
            raise AssertionError(f"branch overlap: {launch}")
        all_cells |= cells
        for r, c in cells:
            g[r, c] = P

    return g


def analyze_fusion_behavior(grid: np.ndarray) -> dict[str, object]:
    info = _analyze_grid(grid)
    g = info["grid"]
    seed_lab = info["seed_lab"]
    branch_lab = info["branch_lab"]
    seed_dirs = info["seed_dirs"]
    branch_dirs = info["branch_dirs"]

    touched_seed_dir: dict[str, str] = {}
    touched_seed_minor: dict[str, int] = {}
    for bdir, branch in branch_dirs.items():
        bid = int(branch["id"])
        pts = np.argwhere(branch_lab == bid)
        touched_ids = set()
        for r, c in pts:
            r = int(r)
            c = int(c)
            for dr, dc in DIR_VECS.values():
                rr = r + dr
                cc = c + dc
                if 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]:
                    sid = int(seed_lab[rr, cc])
                    if sid:
                        touched_ids.add(sid)
        assert len(touched_ids) == 1, f"branch should touch exactly one physical seed, got {touched_ids}"
        sid = next(iter(touched_ids))
        for sdir, seed in seed_dirs.items():
            if int(seed["id"]) == sid:
                touched_seed_dir[bdir] = sdir
                touched_seed_minor[bdir] = int(seed["minority"])
                break

    directional_minor = {bdir: int(seed_dirs[bdir]["minority"]) for bdir in DIRS}
    mismatch_count = sum(int(directional_minor[d] != touched_seed_minor[d]) for d in DIRS)
    return {
        "launch_to_physical_seed": touched_seed_dir,
        "launch_to_directional_minor": directional_minor,
        "launch_to_physical_minor": touched_seed_minor,
        "mismatch_count": mismatch_count,
    }


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            g = _build_input()
            info = analyze_fusion_behavior(g)
            if int(info["mismatch_count"]) < 2:
                continue
            main(g)
            return g
        except Exception:
            continue
    raise RuntimeError(f"generate_input failed after {GEN_RETRIES} attempts")
