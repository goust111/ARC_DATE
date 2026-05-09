# Combo variant of 800d221b (index 13)
# combo_spec: D1/0+D8/1
# Name: 路径宽度决定多数/少数色
#
# Design analysis:
# - Concrete behavior trace:
#   In the base task, every non-hub path component is recolored to the majority
#   color of the seed interior it touches. In this combo, the touched seed is
#   still identified the same way, but the recolor choice now depends on branch
#   thickness: a 2-pixel-wide branch uses the seed majority color, while a
#   1-pixel-wide branch uses the seed minority color. Example: if a seed
#   interior contains mostly red with a few green cells, then a thick branch to
#   that seed becomes red, but a thin branch to an equally red-heavy seed
#   becomes green instead. The hub-center rule stays the same and therefore
#   integrates both dimensions: the center takes the recolored branch color that
#   touches the hub most.
# - Complexity calibration:
#   The generator always includes both width classes in one grid and repeats the
#   two seed interior colors across multiple seeds. This keeps the width rule
#   observable in 2-4 examples and avoids a degenerate family where every branch
#   follows the same majority-only behavior.
# - Hidden-assumption audit:
#   The original solver assumed all branches were 1-pixel chains and used degree
#   constraints that break on 2-pixel corridors. The new solver instead groups
#   each seed border together with its attached path as one P-component, assigns
#   that component to the unique adjacent seed, and detects width by the
#   presence or absence of a 2x2 all-P block outside the hub.
# - Placement legality:
#   The generator uses one seed per side with straight inward corridors, so
#   seeds never overlap, paths never cross except through the hub, and each
#   branch component touches exactly one seed region.
# - Visual observability:
#   Width is a human-visible property, not a hidden numeric one. Every sample
#   contains both thick and thin branches, and both seed colors appear as both
#   majorities and minorities across the four seeds, giving repeated support for
#   the width-to-color-choice mapping.
# - Visual cleanliness:
#   The layout is kept orthogonal and side-separated: each seed occupies a clean
#   edge rectangle, each branch is a straight corridor, and only the hub is a
#   compact central block. This prevents ambiguous overlays.
# - Inferability of change:
#   Only two seed colors participate in recoloring, so the key deduction is not
#   an arbitrary color map but the structural rule "thick chooses majority,
#   thin chooses minority".

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_SIDES = ("top", "bottom", "left", "right")

HUB_SIZE = 3
GRID_SIZE_MIN = 16
GRID_SIZE_MAX = 26
SEED_HEIGHT_MIN = 2
SEED_HEIGHT_MAX = 4
SEED_WIDTH_MIN = 2
SEED_WIDTH_MAX = 4
MINORITY_FRAC_LO = 0.15
MINORITY_FRAC_HI = 0.40
GEN_RETRIES = 200


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(int(x) for x in grid.flatten()).most_common(1)[0][0])


def _label_mask(mask: np.ndarray) -> tuple[np.ndarray, int]:
    h, w = mask.shape
    lab = np.zeros((h, w), dtype=int)
    cur = 0
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or int(lab[r, c]) != 0:
                continue
            cur += 1
            q: deque[tuple[int, int]] = deque([(int(r), int(c))])
            lab[r, c] = cur
            while q:
                rr, cc = q.popleft()
                for dr, dc in _DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and int(lab[nr, nc]) == 0:
                        lab[nr, nc] = cur
                        q.append((nr, nc))
    return lab, cur


def _component_points(labels: np.ndarray, comp_id: int) -> np.ndarray:
    return np.argwhere(labels == int(comp_id))


def _largest_component_size(mask: np.ndarray) -> int:
    lab, n = _label_mask(mask)
    if int(n) == 0:
        return 0
    best = 0
    for cid in range(1, int(n) + 1):
        best = max(best, int(np.count_nonzero(lab == int(cid))))
    return int(best)


def _find_path_color(grid: np.ndarray, bg: int) -> int:
    colors = sorted({int(x) for x in np.unique(grid)} - {int(bg)})
    assert colors, "grid must contain non-background colors"
    return int(max(colors, key=lambda c: _largest_component_size(grid == int(c))))


def _find_hub(grid: np.ndarray, path_color: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    h, w = grid.shape
    mask = grid == int(path_color)
    ps = mask.astype(int).cumsum(axis=0).cumsum(axis=1)

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
    for k in range(min(h, w), HUB_SIZE - 1, -1):
        for r0 in range(0, h - k + 1):
            r1 = r0 + k - 1
            for c0 in range(0, w - k + 1):
                c1 = c0 + k - 1
                if rect_sum(r0, c0, r1, c1) == k * k:
                    best = (int(k), int(r0), int(c0))
                    break
            if best is not None:
                break
        if best is not None:
            break
    assert best is not None, "failed to locate solid path hub"

    k, r0, c0 = best
    hub = np.zeros((h, w), dtype=bool)
    hub[r0 : r0 + k, c0 : c0 + k] = True
    return hub, (int(r0), int(c0), int(r0 + k - 1), int(c0 + k - 1))


def _seed_stats(grid: np.ndarray, bg: int, path_color: int) -> tuple[np.ndarray, dict[int, dict[str, int]]]:
    seed_mask = (grid != int(bg)) & (grid != int(path_color))
    seed_lab, seed_n = _label_mask(seed_mask)
    stats: dict[int, dict[str, int]] = {}
    for sid in range(1, int(seed_n) + 1):
        pts = _component_points(seed_lab, sid)
        vals = [int(grid[int(r), int(c)]) for r, c in pts]
        counts = Counter(vals).most_common()
        assert len(counts) >= 2, "each seed interior must contain majority and minority colors"
        assert counts[0][1] != counts[-1][1], "seed majority/minority must not tie"
        stats[int(sid)] = {
            "majority": int(counts[0][0]),
            "minority": int(counts[-1][0]),
        }
    return seed_lab, stats


def _component_is_wide(comp_mask: np.ndarray) -> bool:
    return bool(
        np.any(
            comp_mask[:-1, :-1]
            & comp_mask[1:, :-1]
            & comp_mask[:-1, 1:]
            & comp_mask[1:, 1:]
        )
    )


def _solve_details(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.array(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)
    path_color = _find_path_color(g, bg)
    hub, hub_box = _find_hub(g, path_color)
    seed_lab, seed_info = _seed_stats(g, bg, path_color)

    branch_mask = (g == int(path_color)) & (~hub)
    branch_lab, branch_n = _label_mask(branch_mask)
    out = g.copy()
    comp_meta: list[dict[str, int]] = []

    for bid in range(1, int(branch_n) + 1):
        pts = _component_points(branch_lab, bid)
        comp_mask = branch_lab == int(bid)
        touches = Counter()
        for r, c in pts:
            rr = int(r)
            cc = int(c)
            for dr, dc in _DIRS4:
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w:
                    sid = int(seed_lab[nr, nc])
                    if sid != 0:
                        touches[sid] += 1
        assert touches, "every branch component must touch a seed region"
        seed_id = int(touches.most_common(1)[0][0])
        wide = _component_is_wide(comp_mask)
        color = int(seed_info[seed_id]["majority"] if wide else seed_info[seed_id]["minority"])
        out[comp_mask] = int(color)
        comp_meta.append(
            {
                "component_id": int(bid),
                "seed_id": int(seed_id),
                "width": int(2 if wide else 1),
                "recolor": int(color),
                "majority": int(seed_info[seed_id]["majority"]),
                "minority": int(seed_info[seed_id]["minority"]),
            }
        )

    touch_colors = Counter()
    for r in range(h):
        for c in range(w):
            if not bool(hub[r, c]):
                continue
            for dr, dc in _DIRS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and int(branch_lab[nr, nc]) != 0:
                    touch_colors[int(out[nr, nc])] += 1
    assert touch_colors, "hub must be adjacent to at least one recolored branch"

    hr0, hc0, hr1, hc1 = hub_box
    center_r = (hr0 + hr1) // 2
    center_c = (hc0 + hc1) // 2
    out[center_r, center_c] = int(touch_colors.most_common(1)[0][0])

    return out, {
        "bg": int(bg),
        "path_color": int(path_color),
        "hub_box": tuple(int(x) for x in hub_box),
        "components": comp_meta,
        "hub_touch_counts": {int(k): int(v) for k, v in touch_colors.items()},
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor a hub-and-branch path network where branch thickness changes the rule.

    Input structure:
    - The background is the most common color.
    - One non-background color P forms a connected orthogonal network with a solid
      3x3 hub near the middle and one edge seed on each side.
    - Every seed is a two-color rectangle touching a grid edge, wrapped on its
      inward-facing sides by P, and connected back to the hub by either a thin
      1-pixel path or a thick 2-pixel corridor.

    Rule:
    - Infer P, locate the solid hub, and split the remaining P cells into branch
      components, each consisting of one seed border plus its attached path.
    - For the seed region touched by a component, compute both the majority and
      minority interior colors.
    - Recolor a thick component (width 2) to that seed's majority color, but
      recolor a thin component (width 1) to that seed's minority color.
    - Leave the hub unchanged except for its center cell, which becomes the
      recolored branch color that is most adjacent to the hub.

    Key insight:
    Branch width is the control signal. The same seed geometry is used in both
    cases, so the solver must read visible path thickness before deciding whether
    to take the local majority or the local minority color.
    """
    out, _ = _solve_details(grid)
    return out


def _fill_seed_interior(
    grid: np.ndarray,
    ir0: int,
    ic0: int,
    ir1: int,
    ic1: int,
    majority: int,
    minority: int,
) -> None:
    cells = [(r, c) for r in range(ir0, ir1 + 1) for c in range(ic0, ic1 + 1)]
    n = len(cells)
    lo = max(1, int(round(n * MINORITY_FRAC_LO)))
    hi = max(lo, int(round(n * MINORITY_FRAC_HI)))
    hi = min(hi, (n - 1) // 2)
    lo = min(lo, hi)
    n_minor = random.randint(lo, hi)
    random.shuffle(cells)
    for r, c in cells[:n_minor]:
        grid[r, c] = int(minority)
    for r, c in cells[n_minor:]:
        grid[r, c] = int(majority)


def _draw_seed_and_branch(
    grid: np.ndarray,
    side: str,
    width: int,
    majority: int,
    minority: int,
    path_color: int,
    hub_box: tuple[int, int, int, int],
) -> None:
    h, w = grid.shape
    hr0, hc0, hr1, hc1 = hub_box
    assert width in (1, 2)

    if side == "top":
        lane_c = random.choice([hc0, hc0 + 1, hc1] if width == 1 else [hc0, hc0 + 1])
        lane_cols = list(range(lane_c, lane_c + width))
        ih = random.randint(SEED_HEIGHT_MIN, SEED_HEIGHT_MAX)
        extra_l = random.randint(1, 2)
        extra_r = random.randint(1, 2)
        ir0, ir1 = 0, ih - 1
        ic0 = max(0, lane_cols[0] - extra_l)
        ic1 = min(w - 1, lane_cols[-1] + extra_r)
        _fill_seed_interior(grid, ir0, ic0, ir1, ic1, majority, minority)
        br = ir1 + 1
        grid[br, ic0 : ic1 + 1] = int(path_color)
        if ic0 > 0:
            grid[ir0 : ir1 + 1, ic0 - 1] = int(path_color)
            grid[br, ic0 - 1] = int(path_color)
        if ic1 < w - 1:
            grid[ir0 : ir1 + 1, ic1 + 1] = int(path_color)
            grid[br, ic1 + 1] = int(path_color)
        for c in lane_cols:
            grid[br + 1 : hr0, c] = int(path_color)

    elif side == "bottom":
        lane_c = random.choice([hc0, hc0 + 1, hc1] if width == 1 else [hc0, hc0 + 1])
        lane_cols = list(range(lane_c, lane_c + width))
        ih = random.randint(SEED_HEIGHT_MIN, SEED_HEIGHT_MAX)
        extra_l = random.randint(1, 2)
        extra_r = random.randint(1, 2)
        ir0, ir1 = h - ih, h - 1
        ic0 = max(0, lane_cols[0] - extra_l)
        ic1 = min(w - 1, lane_cols[-1] + extra_r)
        _fill_seed_interior(grid, ir0, ic0, ir1, ic1, majority, minority)
        tr = ir0 - 1
        grid[tr, ic0 : ic1 + 1] = int(path_color)
        if ic0 > 0:
            grid[ir0 : ir1 + 1, ic0 - 1] = int(path_color)
            grid[tr, ic0 - 1] = int(path_color)
        if ic1 < w - 1:
            grid[ir0 : ir1 + 1, ic1 + 1] = int(path_color)
            grid[tr, ic1 + 1] = int(path_color)
        for c in lane_cols:
            grid[hr1 + 1 : tr, c] = int(path_color)

    elif side == "left":
        lane_r = random.choice([hr0, hr0 + 1, hr1] if width == 1 else [hr0, hr0 + 1])
        lane_rows = list(range(lane_r, lane_r + width))
        iw = random.randint(SEED_WIDTH_MIN, SEED_WIDTH_MAX)
        extra_u = random.randint(1, 2)
        extra_d = random.randint(1, 2)
        ic0, ic1 = 0, iw - 1
        ir0 = max(0, lane_rows[0] - extra_u)
        ir1 = min(h - 1, lane_rows[-1] + extra_d)
        _fill_seed_interior(grid, ir0, ic0, ir1, ic1, majority, minority)
        bc = ic1 + 1
        grid[ir0 : ir1 + 1, bc] = int(path_color)
        if ir0 > 0:
            grid[ir0 - 1, ic0 : ic1 + 1] = int(path_color)
            grid[ir0 - 1, bc] = int(path_color)
        if ir1 < h - 1:
            grid[ir1 + 1, ic0 : ic1 + 1] = int(path_color)
            grid[ir1 + 1, bc] = int(path_color)
        for r in lane_rows:
            grid[r, bc + 1 : hc0] = int(path_color)

    else:
        lane_r = random.choice([hr0, hr0 + 1, hr1] if width == 1 else [hr0, hr0 + 1])
        lane_rows = list(range(lane_r, lane_r + width))
        iw = random.randint(SEED_WIDTH_MIN, SEED_WIDTH_MAX)
        extra_u = random.randint(1, 2)
        extra_d = random.randint(1, 2)
        ic0, ic1 = w - iw, w - 1
        ir0 = max(0, lane_rows[0] - extra_u)
        ir1 = min(h - 1, lane_rows[-1] + extra_d)
        _fill_seed_interior(grid, ir0, ic0, ir1, ic1, majority, minority)
        lc = ic0 - 1
        grid[ir0 : ir1 + 1, lc] = int(path_color)
        if ir0 > 0:
            grid[ir0 - 1, ic0 : ic1 + 1] = int(path_color)
            grid[ir0 - 1, lc] = int(path_color)
        if ir1 < h - 1:
            grid[ir1 + 1, ic0 : ic1 + 1] = int(path_color)
            grid[ir1 + 1, lc] = int(path_color)
        for r in lane_rows:
            grid[r, hc1 + 1 : lc] = int(path_color)


def _pick_majorities(widths: dict[str, int], color_a: int, color_b: int) -> tuple[dict[str, int], dict[str, int]]:
    for _ in range(200):
        majority = {side: random.choice([color_a, color_b]) for side in _SIDES}
        if len(set(majority.values())) < 2:
            continue
        branch_colors: dict[str, int] = {}
        touch = Counter()
        for side in _SIDES:
            maj = int(majority[side])
            min_col = int(color_b if maj == color_a else color_a)
            branch_col = int(maj if int(widths[side]) == 2 else min_col)
            branch_colors[side] = branch_col
            touch[branch_col] += int(widths[side])
        ranked = touch.most_common()
        if len(ranked) == 1 or ranked[0][1] != ranked[1][1]:
            return majority, branch_colors
    raise RuntimeError("failed to pick non-tied seed majorities")


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    w = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    bg, path_color, color_a, color_b = random.sample(list(range(10)), 4)

    hr0 = random.randint(max(6, h // 2 - 1), min(h - 9, h // 2 + 1))
    hc0 = random.randint(max(6, w // 2 - 1), min(w - 9, w // 2 + 1))
    hub_box = (hr0, hc0, hr0 + HUB_SIZE - 1, hc0 + HUB_SIZE - 1)

    widths = {side: 1 for side in _SIDES}
    for side in random.sample(list(_SIDES), 2):
        widths[side] = 2
    majority, _branch_colors = _pick_majorities(widths, color_a, color_b)

    grid = np.full((h, w), int(bg), dtype=int)
    grid[hr0 : hr0 + HUB_SIZE, hc0 : hc0 + HUB_SIZE] = int(path_color)

    for side in _SIDES:
        maj = int(majority[side])
        minority = int(color_b if maj == color_a else color_a)
        _draw_seed_and_branch(
            grid=grid,
            side=side,
            width=int(widths[side]),
            majority=maj,
            minority=minority,
            path_color=int(path_color),
            hub_box=hub_box,
        )

    if _get_bg(grid) != int(bg):
        return None

    try:
        _out, info = _solve_details(grid)
    except Exception:
        return None

    comp_widths = sorted(int(meta["width"]) for meta in info["components"])
    if comp_widths != [1, 1, 2, 2]:
        return None
    if len({int(meta["recolor"]) for meta in info["components"]}) < 2:
        return None
    if not any(int(meta["recolor"]) == int(meta["majority"]) for meta in info["components"]):
        return None
    if not any(int(meta["recolor"]) == int(meta["minority"]) for meta in info["components"]):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_RETRIES} attempts")
