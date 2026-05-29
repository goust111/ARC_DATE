# Combo variant of e3721c99 (index 35)
# combo_spec: D1/3+D9/0+D10/1
# Name: 宽高比匹配+相似图例+一对一

from __future__ import annotations

from pathlib import Path
import itertools
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.insert(0, str(SOLVER_DIR))

from arc_utils import bbox_coords, components_mask, get_bg, neighbors4, paste


# Design analysis for combo D1/3 + D9/0 + D10/1
# a) Concrete behavior trace:
#    - Original local logic:
#        legend A has 1 hole -> all grey sprites with 1 hole become color A
#        legend B has 2 holes -> all grey sprites with 2 holes become color B
#      The mapping is many-to-one and purely topological.
#    - New fused logic:
#        legend A and B can both have exactly one hole and nearly the same outline.
#        Suppose their bboxes are 5x4 and 5x6, while the two grey sprites are 8x7
#        and 9x6. The solver must compare width/height ratios:
#          5x4 -> 0.80, 5x6 -> 1.20, 8x7 -> 0.875, 9x6 -> 0.667
#        Then do a one-to-one assignment by closest ratio globally, so the taller
#        sprite takes the 0.80 legend color and the wider sprite takes the 1.20
#        legend color. Hole count is now deliberately useless.
#    - Visible difference:
#        multiple single-hole legends coexist, so the old "count holes and recolor"
#        rule cannot even build a unique legend table.
# b) Complexity calibration:
#    - Only 2-4 legend entries are used, all from the same simple one-hole ring
#      family, with widths in a narrow 4..7 band and common height 5 or 6.
#    - Each grid contains exactly the same number of grey sprites as legends, so the
#      assignment is small enough to reason about visually but still nontrivial.
#    - Sprite ratios are chosen to be close to their target legends without being
#      exact size copies, so the cue is bbox aspect ratio rather than raw size.
# c) Hidden assumptions broken from the original main():
#    - `hole_count -> color` uniqueness is invalid because every legend intentionally
#      has the same hole count.
#    - Independent per-sprite processing is invalid because the global one-to-one
#      constraint matters; a full assignment is solved, not isolated lookups.
#    - Distractors with "unlisted hole counts" disappear entirely. Under D9/0 the
#      puzzle becomes a perfect bijection between legend set and sprite set.
# d) Placement legality:
#    - A divider still carves off a panel on one side of the 30x30 grid.
#    - Legend rings stay fully inside that panel and never touch the outer border.
#    - Grey sprites are placed only in the workspace with a one-cell moat, so
#      component extraction and bbox measurement stay stable.
# e) Visual observability:
#    - All legends share the same one-hole structure, so the solver cannot explain
#      color changes by hole count. The only stable visible difference is width vs
#      height.
#    - Across samples, legends and sprites both repeat the same ring motif, making
#      the ratio cue readable from only a few examples.
# f) Visual cleanliness:
#    - The scene uses one simple legend family, a single divider, and only 2-4 grey
#      sprites. There are no overlapping marks, nested annotations, or mixed cues.
# g) Change inferability:
#    - Colors are not sorted, ranked, or used as hidden IDs. The recolor rule is
#      explained entirely by visibly matching grey bbox proportions to the colored
#      exemplars.
# h) Quantitative anti-degeneration target:
#    - Every sample must have duplicate legend hole counts (in practice, all are 1),
#      and the intended ratio assignment must be the unique global optimum.
# i) New solver assumptions and generator guarantees:
#    - Solver assumes the number of grey sprites equals the number of legend entries;
#      generator enforces strict equality.
#    - Solver assumes the ratio-matching optimum is unique or at least stably
#      tie-broken; generator rejects near-tied global assignments.
#    - Solver assumes legend components are the only non-border-touching colored
#      objects other than grey sprites; generator keeps divider segments on the
#      border and all legends in the separated panel.


BG = 0
DIV = 1
GREY = 5
GRID_SIZE = 30
LEGEND_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

LEGEND_COUNT_RANGE = (2, 4)
LEGEND_HEIGHT_OPTIONS = (5, 6)
LEGEND_WIDTH_POOL = (4, 5, 6, 7)

SPRITE_H_RANGE = (7, 10)
SPRITE_W_RANGE = (5, 12)
LOCAL_DIFF_MIN = 0.01
LOCAL_DIFF_MAX = 0.16
LOCAL_MARGIN_MIN = -0.02
LOCAL_MARGIN_MAX = 0.2
GLOBAL_MARGIN_MIN = 0.03
AMBIGUOUS_ENOUGH_MAX = 0.09
MAX_ELIMINATION_SPRITES = 1

GEN_RETRIES = 400


def _hole_count(mask: np.ndarray) -> int:
    """Count holes inside a cropped boolean mask using 4-connectivity."""
    h, w = mask.shape
    bg = ~mask
    vis = np.zeros((h, w), dtype=bool)
    holes = 0
    for r0 in range(h):
        for c0 in range(w):
            if not bg[r0, c0] or vis[r0, c0]:
                continue
            st = [(r0, c0)]
            vis[r0, c0] = True
            touches = False
            while st:
                r, c = st.pop()
                if r == 0 or c == 0 or r == h - 1 or c == w - 1:
                    touches = True
                for rr, cc in neighbors4(r, c):
                    if 0 <= rr < h and 0 <= cc < w and bg[rr, cc] and not vis[rr, cc]:
                        vis[rr, cc] = True
                        st.append((rr, cc))
            if not touches:
                holes += 1
    return holes


def _submask_from_points(pts: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = bbox_coords(pts)
    sub = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in pts:
        sub[r - r0, c - c0] = True
    return sub


def _ratio_from_points(pts: np.ndarray) -> float:
    r0, c0, r1, c1 = bbox_coords(pts)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    return float(w) / float(h)


def _solve_ratio_assignment(
    legend_ratios: list[float], sprite_ratios: list[float]
) -> tuple[tuple[int, ...], float, float | None]:
    """Return best sprite->legend permutation, best total cost, and runner-up total cost."""
    n = len(legend_ratios)
    assert n == len(sprite_ratios) and n > 0
    best_key = None
    best_perm = None
    best_total = None
    second_total = None
    for perm in itertools.permutations(range(n)):
        diffs = tuple(abs(sprite_ratios[i] - legend_ratios[perm[i]]) for i in range(n))
        total = float(sum(diffs))
        key = (round(total, 10), tuple(round(x, 10) for x in diffs), tuple(int(x) for x in perm))
        if best_key is None or key < best_key:
            if best_total is not None:
                second_total = best_total if second_total is None else min(second_total, best_total)
            best_key = key
            best_perm = tuple(int(x) for x in perm)
            best_total = total
        else:
            second_total = total if second_total is None else min(second_total, total)
    assert best_perm is not None and best_total is not None
    return best_perm, best_total, second_total


def _legend_entries(g: np.ndarray, bg: int, grey: int) -> list[dict[str, object]]:
    """
    Find one legend component per non-(bg,grey) color and record its bbox ratio.
    Divider segments are filtered out because they touch the outer border.
    """
    h, w = g.shape
    entries: list[dict[str, object]] = []
    colors = [c for c in sorted(set(map(int, np.unique(g)))) if c not in (bg, grey)]
    for color in colors:
        comps = components_mask(g == color, connectivity=4)
        candidates = []
        for pts in comps:
            r0, c0, r1, c1 = bbox_coords(pts)
            if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
                continue
            if (r1 - r0 + 1) < 4 or (c1 - c0 + 1) < 4:
                continue
            candidates.append(pts)
        if not candidates:
            continue
        assert len(candidates) == 1, f"color {color} must have exactly one legend component"
        pts = candidates[0]
        entries.append({
            "color": int(color),
            "points": pts,
            "ratio": _ratio_from_points(pts),
        })
    assert entries, "grid must contain legend components"
    return entries


def _grey_entries(g: np.ndarray, grey: int) -> list[dict[str, object]]:
    """Collect grey components together with their bbox ratios."""
    entries: list[dict[str, object]] = []
    for pts in components_mask(g == grey, connectivity=4):
        entries.append({
            "points": pts,
            "ratio": _ratio_from_points(pts),
        })
    return entries


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor grey sprites by one-to-one matching against the legends' bbox aspect ratios.

    Input structure:
    - The grid still has a divider-separated legend panel near one side and a workspace full of grey sprites.
    - Each colored legend is a small one-hole ring, and all legends are visually very similar except for slight
      width/height differences.
    - The workspace contains exactly one grey sprite per legend, again from the same one-hole ring family but at
      larger sizes.

    Rule:
    - Measure each legend component only by its bounding-box width/height ratio.
    - Measure each grey sprite the same way and solve a global one-to-one assignment minimizing ratio differences.
    - Recolor every grey sprite with the color of its assigned legend.

    Key insight:
    - Holes are decorative and repeated; the discriminative signal is the continuous bbox proportion, resolved under
      a strict bijection between legend set and sprite set.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    grey = GREY

    legends = _legend_entries(g, bg, grey)
    sprites = _grey_entries(g, grey)
    assert len(legends) == len(sprites), "one-to-one ratio matching requires equal legend and sprite counts"

    perm, _best_total, _second_total = _solve_ratio_assignment(
        [float(x["ratio"]) for x in legends],
        [float(x["ratio"]) for x in sprites],
    )

    out = g.copy()
    for sprite_idx, legend_idx in enumerate(perm):
        color = int(legends[legend_idx]["color"])
        for r, c in sprites[sprite_idx]["points"]:
            out[r, c] = color
    return out


# --------------- generator (one-hole ratio-family bijection) ---------------

def _ring_patch(h: int, w: int, color: int) -> np.ndarray:
    """Create a simple rectangular ring with exactly one interior hole."""
    patch = np.full((int(h), int(w)), int(color), dtype=int)
    patch[1:-1, 1:-1] = BG
    assert _hole_count(patch == color) == 1
    return patch


def _choose_legends() -> list[dict[str, object]]:
    """Choose 2-4 similar one-hole legends differing only by width/height ratio."""
    count = random.randint(*LEGEND_COUNT_RANGE)
    h = random.choice(LEGEND_HEIGHT_OPTIONS)
    widths = sorted(random.sample(list(LEGEND_WIDTH_POOL), count))
    colors = random.sample(LEGEND_COLORS, count)
    specs = [{
        "h": int(h),
        "w": int(w),
        "ratio": float(w) / float(h),
        "color": int(color),
    } for w, color in zip(widths, colors)]
    random.shuffle(specs)
    return specs


def _place_legend_panel(grid: np.ndarray, blocked: np.ndarray, legends: list[dict[str, object]]) -> None:
    """Place the legends in a divider-separated panel on one side of the grid."""
    side = random.choice(["top", "bottom", "left", "right"])
    if side in {"top", "bottom"}:
        total_w = sum(int(spec["w"]) for spec in legends) + len(legends) - 1
        panel_h = max(int(spec["h"]) for spec in legends) + 2
        c = random.randint(1, GRID_SIZE - 1 - total_w)
        if side == "top":
            div_r = panel_h
            grid[div_r, :] = DIV
            blocked[: div_r + 1, :] = 9
            r = 1
        else:
            div_r = GRID_SIZE - 1 - panel_h
            grid[div_r, :] = DIV
            blocked[div_r:, :] = 9
            r = div_r + 2
        for spec in legends:
            patch = _ring_patch(int(spec["h"]), int(spec["w"]), int(spec["color"]))
            paste(grid, patch, r, c)
            c += patch.shape[1] + 1
        return

    total_h = sum(int(spec["h"]) for spec in legends) + len(legends) - 1
    panel_w = max(int(spec["w"]) for spec in legends) + 2
    r = random.randint(1, GRID_SIZE - 1 - total_h)
    if side == "left":
        div_c = panel_w
        grid[:, div_c] = DIV
        blocked[:, : div_c + 1] = 9
        c = 1
    else:
        div_c = GRID_SIZE - 1 - panel_w
        grid[:, div_c] = DIV
        blocked[:, div_c:] = 9
        c = div_c + 2
    for spec in legends:
        patch = _ring_patch(int(spec["h"]), int(spec["w"]), int(spec["color"]))
        paste(grid, patch, r, c)
        r += patch.shape[0] + 1


def _sprite_candidates_for(idx: int, legend_ratios: list[float]) -> list[dict[str, object]]:
    """Enumerate larger ring sizes whose ratio is closest to the target legend."""
    lr = float(legend_ratios[idx])
    others = [legend_ratios[j] for j in range(len(legend_ratios)) if j != idx]
    out: list[dict[str, object]] = []
    for h in range(SPRITE_H_RANGE[0], SPRITE_H_RANGE[1] + 1):
        for w in range(SPRITE_W_RANGE[0], SPRITE_W_RANGE[1] + 1):
            sr = float(w) / float(h)
            diff = abs(sr - lr)
            if diff < LOCAL_DIFF_MIN or diff > LOCAL_DIFF_MAX:
                continue
            second = min(abs(sr - x) for x in others) if others else 1.0
            gap = second - diff
            if gap < LOCAL_MARGIN_MIN or gap > LOCAL_MARGIN_MAX:
                continue
            out.append({
                "h": int(h),
                "w": int(w),
                "ratio": sr,
                "diff": diff,
                "local_gap": gap,
            })
    random.shuffle(out)
    out.sort(key=lambda x: (round(float(x["diff"]), 6), -round(float(x["local_gap"]), 6)))
    return out


def _choose_sprite_specs(legends: list[dict[str, object]]) -> list[dict[str, object]] | None:
    """Choose one larger sprite size per legend and verify the global one-to-one optimum."""
    legend_ratios = [float(spec["ratio"]) for spec in legends]
    cand_lists = [_sprite_candidates_for(i, legend_ratios) for i in range(len(legends))]
    if any(not cands for cands in cand_lists):
        return None

    order = sorted(range(len(legends)), key=lambda i: len(cand_lists[i]))
    chosen: list[dict[str, object] | None] = [None] * len(legends)
    used_dims: set[tuple[int, int]] = set()

    def dfs(k: int) -> bool:
        if k == len(order):
            specs = [x for x in chosen if x is not None]
            sprite_ratios = [float(spec["ratio"]) for spec in specs]
            perm, best_total, second_total = _solve_ratio_assignment(legend_ratios, sprite_ratios)
            if perm != tuple(range(len(legends))):
                return False
            if second_total is None or second_total - best_total < GLOBAL_MARGIN_MIN:
                return False
            if sum(float(spec["local_gap"]) < 0.0 for spec in specs) > MAX_ELIMINATION_SPRITES:
                return False
            if min(float(spec["local_gap"]) for spec in specs) > AMBIGUOUS_ENOUGH_MAX:
                return False
            return True

        idx = order[k]
        for cand in cand_lists[idx]:
            dim = (int(cand["h"]), int(cand["w"]))
            if dim in used_dims:
                continue
            chosen[idx] = cand
            used_dims.add(dim)
            if dfs(k + 1):
                return True
            used_dims.remove(dim)
            chosen[idx] = None
        return False

    if not dfs(0):
        return None
    return [dict(chosen[i]) for i in range(len(legends)) if chosen[i] is not None]


def _can_place(static_blocked: np.ndarray, occupied: np.ndarray, patch: np.ndarray, r: int, c: int) -> bool:
    """Check bounds plus a one-cell moat from blocked cells and other sprites."""
    h, w = patch.shape
    rr0 = max(0, r - 1)
    cc0 = max(0, c - 1)
    rr1 = min(GRID_SIZE, r + h + 1)
    cc1 = min(GRID_SIZE, c + w + 1)
    if np.any(static_blocked[rr0:rr1, cc0:cc1] != BG):
        return False
    if np.any(occupied[rr0:rr1, cc0:cc1]):
        return False
    return True


def _random_location(static_blocked: np.ndarray, occupied: np.ndarray, patch: np.ndarray) -> tuple[int, int] | None:
    """Find a random valid placement in the workspace."""
    h, w = patch.shape
    candidates = [(r, c) for r in range(GRID_SIZE - h + 1) for c in range(GRID_SIZE - w + 1)]
    random.shuffle(candidates)
    for r, c in candidates:
        if _can_place(static_blocked, occupied, patch, r, c):
            return int(r), int(c)
    return None


def _place_sprites(
    grid: np.ndarray,
    expected: np.ndarray,
    blocked: np.ndarray,
    legends: list[dict[str, object]],
    sprites: list[dict[str, object]],
) -> bool:
    """Place exactly one grey sprite per legend and paint the expected output accordingly."""
    items = []
    for legend, sprite in zip(legends, sprites):
        patch = _ring_patch(int(sprite["h"]), int(sprite["w"]), GREY)
        items.append({
            "area": int(patch.size),
            "patch": patch,
            "color": int(legend["color"]),
        })
    random.shuffle(items)
    items.sort(key=lambda x: int(x["area"]), reverse=True)

    occupied = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    for item in items:
        patch = np.asarray(item["patch"], dtype=int)
        loc = _random_location(blocked, occupied, patch)
        if loc is None:
            return False
        r, c = loc
        paste(grid, patch, r, c, transparent=BG)
        out_patch = np.where(patch == GREY, int(item["color"]), BG)
        paste(expected, out_patch, r, c, transparent=BG)
        occupied[r : r + patch.shape[0], c : c + patch.shape[1]] |= (patch != BG)
    return True


def _gen_attempt() -> np.ndarray | None:
    """Build one full puzzle instance and verify it with the solver."""
    legends = _choose_legends()
    sprites = _choose_sprite_specs(legends)
    if sprites is None:
        return None

    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    blocked = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    _place_legend_panel(grid, blocked, legends)

    expected = grid.copy()
    if not _place_sprites(grid, expected, blocked, legends, sprites):
        return None

    if int(np.sum(grid == BG)) <= int(np.sum(grid != BG)):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            grid = _gen_attempt()
        except Exception:
            grid = None
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate e3721c99 combo input")
