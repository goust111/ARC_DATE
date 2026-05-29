# Combo variant of 800d221b (index 36)
# combo_spec: D9/0+D10/1+D17/0
# Name: 干扰路径+着色删除+裁剪输出
#
# Design analysis:
# - Concrete behavior trace:
#   In the source task, every non-hub P component is treated as a real branch, recolored once, and the full
#   canvas is returned. Here the same reading would fail on three steps. In a concrete example, a short north
#   branch that touches the hub and one seed must first be recolored to that seed's majority color, then erased
#   because its full component length is below the fixed threshold, while an isolated P squiggle elsewhere is
#   never recolored because it touches no seed or hub. After that, the answer is not the full canvas: it is the
#   tight crop of the remaining meaningful network, so a corner distractor may vanish simply because it lies
#   outside the cropped rectangle.
# - Complexity calibration:
#   The generator keeps one central 3x3 hub plus four framed seeds, so the visual family still matches the
#   original task. The fused difficulty comes from three observable effects that repeat in every sample:
#   one valid branch disappears after being recolored, at least two valid branches survive and vote on the hub
#   center, and at least one isolated P component is not part of that logic. Noise is limited to two short
#   segments and seed colors remain only A/B, which keeps the scene readable instead of introducing many new
#   degrees of freedom at once.
# - Hidden-assumption audit:
#   The original solver assumes every non-hub P component is meaningful; D17 breaks that, so this solver now
#   classifies components by whether they touch both a seed interior and the hub. The original center-color
#   logic also assumes recolored branches remain present; D10 breaks that, so the hub center is computed only
#   from surviving recolored branches after deletion. D9 breaks the "output shape equals input shape" assumption,
#   so the final return value is cropped to the bbox of the remaining meaningful structure rather than all
#   foreground cells. The generator guarantees the needed preconditions: one short valid branch, multiple long
#   valid branches, a unique seed-majority color per seed, and at least one noise component outside the crop.
# - Placement legality:
#   The canonical scene is a 23x23 canvas with the hub in the center, four interior seed rectangles with full
#   P borders, four non-overlapping orthogonal branch components, one interior distractor path, and one corner
#   distractor path. Rotation by 0/90/180/270 degrees preserves in-bounds placement, branch/noise separation,
#   and the fact that only the true branches touch both hub and seed.
# - Visual observability:
#   A human looking at 2-4 samples can see the pipeline: several branches recolor to the obvious majority color
#   of nearby seed interiors; the shortest true branch then disappears entirely; one disconnected P segment can
#   remain unchanged; and the output frame is visibly smaller than the input. These cues are all pixel-visible
#   and do not rely on color ids or hidden ordering.
# - Visual cleanliness:
#   The layout uses one sparse central network and small rectangular seeds, with distractors placed either far in
#   a corner or in a clear empty quadrant. There are no overlays, legends, or overlapping annotations, so each
#   object class remains separable by shape and connectivity.
# - Change inferability:
#   The branch recoloring still follows local seed evidence, the deletion rule follows a repeated size contrast
#   between one short true branch and longer survivors, and the crop is directly legible because margins and at
#   least one corner distractor disappear together. The generator rejects degenerate cases where those effects
#   would be absent.
# - Anti-degeneration metric:
#   For generated samples we track: how many valid branches are deleted, how many noise pixels remain inside the
#   crop, how many lie outside the crop and therefore get excluded, and how much area the crop removes. Accepted
#   samples must show all three fused dimensions actively changing the result.
# - New solver assumptions guaranteed by the generator:
#   each valid non-hub P component touches exactly one seed and the hub; each distractor component touches
#   neither; the deleted valid branch is the unique shortest true branch; the surviving hub-adjacent branch
#   colors have a strict winner; and the meaningful crop bbox is strictly smaller than the input canvas.

from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
for extra in (ROOT, ROOT / "solvers"):
    extra_s = str(extra)
    if extra_s not in sys.path:
        sys.path.insert(0, extra_s)

from arc_utils import components_mask, get_bg, label_mask


GRID_SIZE = 23
HUB_SIZE = 3
SHORT_BRANCH_THRESHOLD = 20
GEN_OUTER_RETRIES = 200

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

SEED_AREAS = {
    "N": (3, 10, 5, 12),   # 3x3
    "E": (9, 17, 13, 19),  # 5x3
    "S": (17, 9, 19, 13),  # 3x5
    "W": (9, 3, 13, 5),    # 5x3
}

BRANCH_POLYLINES = {
    "N": [(9, 11), (6, 11)],
    "E": [(11, 13), (11, 16)],
    "S": [(13, 11), (16, 11)],
    "W": [(11, 9), (11, 6)],
}

NOISE_POLYLINES = {
    "inside": [(5, 15), (5, 17), (4, 17)],
    "outside": [(0, 1), (0, 3), (1, 3)],
}

MAJORITY_PATTERNS = [
    {"N": 0, "E": 0, "S": 0, "W": 1},
    {"N": 1, "E": 0, "S": 1, "W": 1},
    {"N": 0, "E": 1, "S": 0, "W": 0},
    {"N": 1, "E": 1, "S": 1, "W": 0},
]


def _ordered_winner(counter: Counter[int]) -> int:
    items = list(counter.items())
    assert items, "expected non-empty counter"
    return int(sorted(items, key=lambda kv: (-int(kv[1]), int(kv[0])))[0][0])


def _comp_deg_score(mask: np.ndarray) -> float:
    h, w = mask.shape
    comps = components_mask(mask, connectivity=4)
    if not comps:
        return -1e9
    best_pts = max(comps, key=lambda coords: int(coords.shape[0]))
    best_sz = int(best_pts.shape[0])

    m = np.zeros(mask.shape, dtype=bool)
    m[best_pts[:, 0], best_pts[:, 1]] = True

    deg2 = 0
    deg4 = 0
    for r, c in best_pts:
        r = int(r)
        c = int(c)
        d = 0
        if r > 0 and bool(m[r - 1, c]):
            d += 1
        if r + 1 < h and bool(m[r + 1, c]):
            d += 1
        if c > 0 and bool(m[r, c - 1]):
            d += 1
        if c + 1 < w and bool(m[r, c + 1]):
            d += 1
        if d == 2:
            deg2 += 1
        elif d == 4:
            deg4 += 1
    frac2 = deg2 / max(1, best_sz)
    frac4 = deg4 / max(1, best_sz)
    return float((frac2 - frac4) * best_sz)


def _infer_path_color(g: np.ndarray, bg: int) -> int:
    colors = sorted({int(x) for x in np.unique(g)} - {int(bg)})
    assert colors, "grid must contain non-background colors"
    return int(max(colors, key=lambda c: _comp_deg_score(g == int(c))))


def _largest_p_square(g: np.ndarray, pcol: int) -> tuple[int, int, int]:
    h, w = g.shape
    m = g == int(pcol)
    ps = m.astype(int).cumsum(axis=0).cumsum(axis=1)

    def rect_sum(r0: int, c0: int, r1: int, c1: int) -> int:
        s = int(ps[int(r1), int(c1)])
        if r0 > 0:
            s -= int(ps[int(r0 - 1), int(c1)])
        if c0 > 0:
            s -= int(ps[int(r1), int(c0 - 1)])
        if r0 > 0 and c0 > 0:
            s += int(ps[int(r0 - 1), int(c0 - 1)])
        return int(s)

    best = None
    for k in range(2, int(min(h, w)) + 1):
        found = None
        for r0 in range(0, int(h - k + 1)):
            r1 = int(r0 + k - 1)
            for c0 in range(0, int(w - k + 1)):
                c1 = int(c0 + k - 1)
                if rect_sum(int(r0), int(c0), int(r1), int(c1)) == int(k * k):
                    found = (int(k), int(r0), int(c0))
        if found is not None:
            best = found
    assert best is not None, "hub square not found"
    return tuple(map(int, best))


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    pts = np.argwhere(mask)
    assert pts.size, "crop mask must be non-empty"
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_to_mask(arr: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox_from_mask(mask)
    return arr[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy(), (int(r0), int(c0), int(r1), int(c1))


def _polyline_cells(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for idx, ((r0, c0), (r1, c1)) in enumerate(zip(points, points[1:])):
        if r0 == r1:
            step = 1 if c1 >= c0 else -1
            seg = [(int(r0), int(c)) for c in range(int(c0), int(c1) + step, step)]
        elif c0 == c1:
            step = 1 if r1 >= r0 else -1
            seg = [(int(r), int(c0)) for r in range(int(r0), int(r1) + step, step)]
        else:
            raise ValueError("non-orthogonal polyline")
        if idx:
            seg = seg[1:]
        out.extend(seg)
    return out


def _paint_cells(g: np.ndarray, cells: list[tuple[int, int]] | set[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        g[int(r), int(c)] = int(color)


def _place_seed_border(g: np.ndarray, pcol: int, ir0: int, ic0: int, ir1: int, ic1: int) -> set[tuple[int, int]]:
    border_cells: set[tuple[int, int]] = set()
    for r in range(int(ir0 - 1), int(ir1 + 2)):
        for c in range(int(ic0 - 1), int(ic1 + 2)):
            if r in (int(ir0 - 1), int(ir1 + 1)) or c in (int(ic0 - 1), int(ic1 + 1)):
                border_cells.add((int(r), int(c)))
    _paint_cells(g, border_cells, pcol)
    return border_cells


def _fill_seed_interior(
    g: np.ndarray,
    bbox: tuple[int, int, int, int],
    majority: int,
    minority: int,
    rng: random.Random,
) -> None:
    ir0, ic0, ir1, ic1 = map(int, bbox)
    cells = [(r, c) for r in range(ir0, ir1 + 1) for c in range(ic0, ic1 + 1)]
    area = len(cells)
    assert area in (9, 15), "unexpected seed area"
    rng.shuffle(cells)
    if area == 9:
        n_minor = rng.choice([3, 4])
    else:
        n_minor = rng.choice([4, 5, 6])
    assert n_minor < area - n_minor, "majority must be strict"
    for r, c in cells[:n_minor]:
        g[int(r), int(c)] = int(minority)
    for r, c in cells[n_minor:]:
        g[int(r), int(c)] = int(majority)


def _analyze_scene(grid: np.ndarray) -> dict:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    h, w = g.shape
    pcol = int(_infer_path_color(g, bg))

    k, r0, c0 = _largest_p_square(g, pcol)
    r1 = int(r0 + k - 1)
    c1 = int(c0 + k - 1)
    hub = np.zeros((h, w), dtype=bool)
    hub[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = True

    seed_mask = (g != int(bg)) & (g != int(pcol))
    seed_lab, seed_n = label_mask(seed_mask, connectivity=4)
    assert int(seed_n) >= 3, "expected multiple seed interiors"

    seed_major: dict[int, int] = {}
    for sid in range(1, int(seed_n) + 1):
        pts = np.argwhere(seed_lab == int(sid))
        assert pts.size, "empty seed component"
        rn, cn = pts.min(axis=0)
        rx, cx = pts.max(axis=0)
        rn, cn, rx, cx = map(int, (rn, cn, rx, cx))
        assert int(pts.shape[0]) == int((rx - rn + 1) * (cx - cn + 1)), "seed interior must be rectangular"
        for c in range(cn, cx + 1):
            assert int(g[rn - 1, c]) == int(pcol), "seed missing top border"
            assert int(g[rx + 1, c]) == int(pcol), "seed missing bottom border"
        for r in range(rn, rx + 1):
            assert int(g[r, cn - 1]) == int(pcol), "seed missing left border"
            assert int(g[r, cx + 1]) == int(pcol), "seed missing right border"

        counts = Counter(int(g[int(r), int(c)]) for r, c in pts)
        assert len(counts) == 2, "each seed interior must use exactly two colors"
        ordered = sorted(counts.items(), key=lambda kv: (-int(kv[1]), int(kv[0])))
        assert int(ordered[0][1]) > int(ordered[1][1]), "seed majority must be unique"
        seed_major[int(sid)] = int(ordered[0][0])

    branch_mask = (g == int(pcol)) & (~hub)
    branch_lab, branch_n = label_mask(branch_mask, connectivity=4)

    valid_branches: dict[int, dict] = {}
    noise_ids: list[int] = []

    for bid in range(1, int(branch_n) + 1):
        pts = np.argwhere(branch_lab == int(bid))
        touches = Counter()
        hub_touch = 0
        for r, c in pts:
            r = int(r)
            c = int(c)
            for dr, dc in DIR4:
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < h and 0 <= cc < w:
                    sid = int(seed_lab[rr, cc])
                    if sid != 0:
                        touches[int(sid)] += 1
                    if bool(hub[rr, cc]):
                        hub_touch += 1
        if touches and hub_touch:
            valid_branches[int(bid)] = {
                "cells": [(int(r), int(c)) for r, c in pts],
                "seed_id": int(_ordered_winner(touches)),
                "length": int(pts.shape[0]),
            }
        else:
            noise_ids.append(int(bid))

    assert valid_branches, "expected at least one valid branch"
    return {
        "grid": g,
        "bg": int(bg),
        "P": int(pcol),
        "hub": hub,
        "hub_box": (int(k), int(r0), int(c0)),
        "seed_lab": seed_lab,
        "seed_major": seed_major,
        "branch_lab": branch_lab,
        "valid_branches": valid_branches,
        "noise_ids": noise_ids,
    }


def _solve_from_info(info: dict) -> dict:
    g = info["grid"]
    bg = int(info["bg"])
    hub = info["hub"]
    seed_lab = info["seed_lab"]
    branch_lab = info["branch_lab"]
    valid_branches = info["valid_branches"]
    seed_major = info["seed_major"]

    stage1 = g.copy()
    branch_color: dict[int, int] = {}
    for bid, branch in valid_branches.items():
        col = int(seed_major[int(branch["seed_id"])])
        branch_color[int(bid)] = int(col)
        for r, c in branch["cells"]:
            stage1[int(r), int(c)] = int(col)

    out = stage1.copy()
    deleted: set[int] = set()
    meaningful_mask = (seed_lab > 0) | hub
    for bid, branch in valid_branches.items():
        if int(branch["length"]) < int(SHORT_BRANCH_THRESHOLD):
            deleted.add(int(bid))
            for r, c in branch["cells"]:
                out[int(r), int(c)] = int(bg)
        else:
            for r, c in branch["cells"]:
                meaningful_mask[int(r), int(c)] = True

    touch_cols = Counter()
    h, w = g.shape
    for r in range(h):
        for c in range(w):
            if not bool(hub[int(r), int(c)]):
                continue
            for dr, dc in DIR4:
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < h and 0 <= cc < w:
                    bid = int(branch_lab[rr, cc])
                    if bid in valid_branches and bid not in deleted:
                        touch_cols[int(branch_color[int(bid)])] += 1

    assert touch_cols, "hub must touch at least one surviving branch"
    k, r0, c0 = info["hub_box"]
    cr = int(r0 + k // 2)
    cc = int(c0 + k // 2)
    out[int(cr), int(cc)] = int(_ordered_winner(touch_cols))
    meaningful_mask[int(cr), int(cc)] = True

    cropped, crop_box = _crop_to_mask(out, meaningful_mask)
    return {
        "stage1": stage1,
        "stage2": out,
        "cropped": cropped,
        "crop_box": crop_box,
        "meaningful_mask": meaningful_mask,
        "deleted": deleted,
        "branch_color": branch_color,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor a hub-and-branch network in stages, ignore isolated P-noise, then return only the tight answer crop.

    Input structure:
    - bg is the most frequent color.
    - One non-bg color P forms a solid square hub, several hub-to-seed branch components, and extra isolated
      P path segments that do not connect to the hub.
    - Seed interiors are rectangular two-color regions framed by P on all four sides.

    Rule:
    - Infer P and locate the largest solid P square as the hub.
    - Among the non-hub P components, treat only components touching both a seed interior and the hub as valid
      branches; recolor every valid branch to the majority color of the seed it touches most.
    - Delete every valid branch whose full component length is below the fixed threshold, recolor the hub center
      from the surviving hub-adjacent branch colors, and finally crop the result to the bbox of the remaining
      meaningful network rather than returning the full canvas.

    Key insight:
    - The task is a three-stage pipeline: disconnected P segments are distractors, short true branches are
      recolored before being erased, and the final answer frame is determined only after that deletion step.
    """
    info = _analyze_scene(grid)
    solved = _solve_from_info(info)
    return solved["cropped"]


def _build_case(rng: random.Random) -> np.ndarray:
    bg, pcol, a, b = rng.sample(list(range(10)), 4)
    g = np.full((GRID_SIZE, GRID_SIZE), int(bg), dtype=int)

    hub_r0 = hub_c0 = 10
    g[hub_r0 : hub_r0 + HUB_SIZE, hub_c0 : hub_c0 + HUB_SIZE] = int(pcol)

    for bbox in SEED_AREAS.values():
        _place_seed_border(g, int(pcol), *bbox)

    for poly in BRANCH_POLYLINES.values():
        _paint_cells(g, _polyline_cells(poly), int(pcol))

    for poly in NOISE_POLYLINES.values():
        _paint_cells(g, _polyline_cells(poly), int(pcol))

    palette = [int(a), int(b)]
    pattern = rng.choice(MAJORITY_PATTERNS)
    for name, bbox in SEED_AREAS.items():
        majority = int(palette[int(pattern[name])])
        minority = int(b if majority == int(a) else a)
        _fill_seed_interior(g, bbox, majority, minority, rng)

    rot_k = rng.randint(0, 3)
    if rot_k:
        g = np.rot90(g, k=rot_k)
    return g


def _fusion_metrics(grid: np.ndarray) -> dict[str, int]:
    info = _analyze_scene(grid)
    solved = _solve_from_info(info)
    crop_box = solved["crop_box"]
    noise_cells: list[tuple[int, int]] = []
    branch_lab = info["branch_lab"]
    for bid in info["noise_ids"]:
        pts = np.argwhere(branch_lab == int(bid))
        noise_cells.extend((int(r), int(c)) for r, c in pts)

    r0, c0, r1, c1 = map(int, crop_box)
    inside_crop_noise = 0
    outside_crop_noise = 0
    for r, c in noise_cells:
        if int(r0) <= int(r) <= int(r1) and int(c0) <= int(c) <= int(c1):
            inside_crop_noise += 1
        else:
            outside_crop_noise += 1

    return {
        "noise_components": int(len(info["noise_ids"])),
        "deleted_valid_branches": int(len(solved["deleted"])),
        "surviving_valid_branches": int(len(info["valid_branches"]) - len(solved["deleted"])),
        "deleted_pixels": int(np.count_nonzero(solved["stage1"] != solved["stage2"])),
        "inside_crop_noise_pixels": int(inside_crop_noise),
        "outside_crop_noise_pixels": int(outside_crop_noise),
        "crop_h": int(solved["cropped"].shape[0]),
        "crop_w": int(solved["cropped"].shape[1]),
        "input_h": int(grid.shape[0]),
        "input_w": int(grid.shape[1]),
        "crop_area_removed": int(grid.shape[0] * grid.shape[1] - solved["cropped"].shape[0] * solved["cropped"].shape[1]),
    }


def generate_input() -> np.ndarray:
    for _ in range(GEN_OUTER_RETRIES):
        rng = random.Random(random.randrange(1 << 30))
        g = _build_case(rng)
        try:
            out = main(g)
            metrics = _fusion_metrics(g)
        except Exception:
            continue
        if metrics["noise_components"] < 2:
            continue
        if metrics["deleted_valid_branches"] < 1:
            continue
        if metrics["surviving_valid_branches"] < 2:
            continue
        if metrics["deleted_pixels"] <= 0:
            continue
        if metrics["inside_crop_noise_pixels"] <= 0:
            continue
        if metrics["outside_crop_noise_pixels"] <= 0:
            continue
        if metrics["crop_h"] >= metrics["input_h"] or metrics["crop_w"] >= metrics["input_w"]:
            continue
        if metrics["crop_area_removed"] <= 0:
            continue
        if g.shape == out.shape and np.array_equal(g, out):
            continue
        return g
    raise RuntimeError(f"generate_input failed after {GEN_OUTER_RETRIES} attempts")
