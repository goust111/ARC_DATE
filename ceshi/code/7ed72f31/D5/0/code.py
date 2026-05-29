# Variant of 7ed72f31
# Dimension: D5 — 镜像像素的覆盖规则 (transform)
# Change: 镜像像素直接覆盖写入，不检查目标位置是否已有非背景像素... → 仅覆盖背景像素

from __future__ import annotations

"""
Design analysis for the D5 overwrite-priority variant.

spec_variant:
  inherited_core:
    - Red markers still define reflection geometry exactly as in 7ed72f31.
    - Row/column red lines reflect monochrome payloads across the line.
    - Isolated red pixels point-reflect one adjacent 8-connected monochrome component.
    - Original payloads and markers remain in place.
  changed_dimension:
    - Old behavior: every reflected pixel overwrote whatever was already at the target.
    - New behavior: a reflected pixel is painted only when the current output target is still background.
    - Because the solver writes into `out` incrementally, this means both pre-existing input pixels and
      earlier reflected pixels have priority over later writes.

  a_concretization_and_trace:
    - Concrete scene: a horizontal red axis has green payload pixels below it. On the mirrored side there is
      already an unrelated blocker pixel of another color at one of the target coordinates.
    - Old execution:
      1. detect the row axis
      2. reflect each green payload pixel
      3. write every reflected green pixel unconditionally
      4. the blocker is erased
    - New execution:
      1. detect the same axis
      2. reflect the same payload pixels
      3. before each write, check whether `out[target] == bg`
      4. green writes land only on empty background; the blocker survives
    - Visible difference: the mirrored copy has a missing cell exactly where another non-background object
      already occupied the target.

  b_complexity_calibration:
    - If the generator kept the original non-overlap guarantees, the new rule would be invisible because old
      and new solvers would agree.
    - Therefore the generator deliberately creates a small number of safe collisions between mirrored targets
      and unrelated blocker pixels.
    - The blocker palette is kept simple: one global blocker color distinct from all payload colors.
    - Each grid has 3-5 marker groups and at least 2 blocked reflection attempts, so the D5 effect is repeated.

  c_implicit_assumption_audit_for_main:
    - Component detection via `label_mask(g != bg, connectivity=4)` still assumes each marker group forms a
      stable 4-connected component in the input. The generator preserves that.
    - Axis classification by red geometry (single point / one row / one column) still assumes blocker pixels do
      not attach 4-connectedly to the red marker component. The generator only places blockers on the mirrored
      side at reflection distance >= 2 from the axis / center.
    - Eligibility functions still depend on the old bbox heuristics. The generator keeps payload colors unique
      across groups, so blockers of a different color are never accidentally mirrored by another group.
    - The changed line is the write itself: previously `out[target] = v` was harmless because generation avoided
      collisions. In this variant that assumption is false, so unconditional overwrite would destroy blockers.

  d_placement_legality:
    - Each local patch is built so the original marker group fits completely within its own rectangle.
    - The mirrored copy also fits within that rectangle, even though it is absent from the input.
    - Patches are placed into the global grid only when their whole rectangle plus a one-cell margin is empty,
      preventing cross-group contact, overlap, and out-of-bounds placements.
    - Blockers are inserted only at reflected target cells that are not 4-connected to the originating marker
      component, so the solver still parses the intended structure.

  e_visual_observability:
    - Human-visible clue: most of a mirrored copy appears normally, but some cells do not replace pre-existing
      blocker pixels. Repeating this in multiple groups makes the "fill only empty space" rule inferable.
    - The generator does not hide the effect behind perfect symmetry or dense clutter; blockers are sparse and
      locally aligned with where a mirror would otherwise continue.

  f_visual_cleanliness:
    - Groups live in separated rectangles, so the scene reads as several independent reflection problems.
    - Only one extra blocker color is introduced, avoiding color noise.
    - Blockers are small and sparse, so they read as occupied cells rather than as competing marker systems.

  g_color_inference:
    - No color remapping rule is introduced.
    - Payload colors remain stable under reflection; blocker color is merely an occupied obstacle and is not tied
      to a transformation of its own.

  new_main_assumptions_to_satisfy:
    - The order-sensitive occupancy check now matters. The generator therefore ensures that blocked targets are
      already non-background in the input itself, so the intended skip does not rely on unpredictable ordering
      between separate reflection groups.
"""

import random
import sys
from pathlib import Path

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import get_bg, label_mask


MARK = 2
PALETTE = tuple(range(10))


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            out.append((r + dr, c + dc))
    return out


def _solve(grid: np.ndarray, *, bg_only: bool, collect_stats: bool = False):
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape

    lab, n = label_mask(g != bg, connectivity=4)
    comps = []
    for k in range(1, int(n) + 1):
        coords = [(int(r), int(c)) for r, c in np.argwhere(lab == int(k))]
        if not coords:
            continue
        vals = [int(g[r, c]) for r, c in coords]
        rs = [r for r, _c in coords]
        cs = [c for _r, c in coords]
        comps.append(
            {
                "coords": coords,
                "vals": vals,
                "bbox": (int(min(rs)), int(min(cs)), int(max(rs)), int(max(cs))),
            }
        )

    out = g.copy()
    stats = {
        "attempts": 0,
        "writes": 0,
        "blocked_existing": 0,
        "row_groups": 0,
        "col_groups": 0,
        "point_groups": 0,
        "isolated_markers": 0,
    }

    def inb(r: int, c: int) -> bool:
        return 0 <= int(r) < H and 0 <= int(c) < W

    def refl_point(p: tuple[int, int], r: int, c: int) -> tuple[int, int]:
        pr, pc = map(int, p)
        return int(2 * pr - r), int(2 * pc - c)

    def refl_row(r0: int, r: int, c: int) -> tuple[int, int]:
        return int(2 * int(r0) - int(r)), int(c)

    def refl_col(c0: int, r: int, c: int) -> tuple[int, int]:
        return int(r), int(2 * int(c0) - int(c))

    def paint_if_allowed(rr: int, cc: int, v: int) -> None:
        if not inb(rr, cc):
            return
        stats["attempts"] += 1
        if bg_only and int(out[rr, cc]) != bg:
            stats["blocked_existing"] += 1
            return
        out[rr, cc] = int(v)
        stats["writes"] += 1

    for comp in comps:
        coords = comp["coords"]
        vals = comp["vals"]
        if MARK not in set(map(int, vals)):
            continue

        reds = [(int(r), int(c)) for (r, c), v in zip(coords, vals) if int(v) == MARK]
        if not reds:
            continue

        rs = {int(r) for r, _c in reds}
        cs = {int(c) for _r, c in reds}
        if len(reds) == 1:
            axis = ("point", (int(reds[0][0]), int(reds[0][1])))
            stats["point_groups"] += 1
        elif len(rs) == 1:
            axis = ("row", int(next(iter(rs))))
            stats["row_groups"] += 1
        elif len(cs) == 1:
            axis = ("col", int(next(iter(cs))))
            stats["col_groups"] += 1
        else:
            continue

        r0, c0, r1, c1 = map(int, comp["bbox"])
        payload = {int(v) for v in vals if int(v) != MARK}
        if not payload:
            continue

        if axis[0] == "row":

            def eligible(r: int, c: int, v: int) -> bool:
                return int(v) in payload and int(c0) <= int(c) <= int(c1)

        elif axis[0] == "col":

            def eligible(r: int, c: int, v: int) -> bool:
                return int(v) in payload and int(r0) <= int(r) <= int(r1)

        else:
            rr0 = max(0, int(r0 - 1))
            rr1 = min(int(H - 1), int(r1 + 1))
            cc0 = max(0, int(c0 - 1))
            cc1 = min(int(W - 1), int(c1 + 1))

            def eligible(r: int, c: int, v: int) -> bool:
                return int(v) in payload and int(rr0) <= int(r) <= int(rr1) and int(cc0) <= int(c) <= int(cc1)

        for r in range(int(H)):
            for c in range(int(W)):
                v = int(g[int(r), int(c)])
                if v == bg or v == MARK:
                    continue
                if not eligible(int(r), int(c), int(v)):
                    continue

                if axis[0] == "row":
                    rr, cc = refl_row(int(axis[1]), int(r), int(c))
                elif axis[0] == "col":
                    rr, cc = refl_col(int(axis[1]), int(r), int(c))
                else:
                    rr, cc = refl_point(axis[1], int(r), int(c))

                paint_if_allowed(int(rr), int(cc), int(v))

    marker_pts = []
    for comp in comps:
        if len(comp["coords"]) == 1 and int(comp["vals"][0]) == MARK:
            marker_pts.append(comp["coords"][0])

    for pr, pc in marker_pts:
        pr = int(pr)
        pc = int(pc)
        stats["isolated_markers"] += 1
        neigh_colors: set[int] = set()
        neigh_pixels: list[tuple[int, int, int]] = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = int(pr + dr), int(pc + dc)
                if not inb(rr, cc):
                    continue
                v = int(g[rr, cc])
                if v == bg or v == MARK:
                    continue
                neigh_colors.add(int(v))
                neigh_pixels.append((int(rr), int(cc), int(v)))
        if not neigh_colors:
            continue

        for v in neigh_colors:
            labv, _nv = label_mask(g == int(v), connectivity=8)
            labs = {
                int(labv[int(rr), int(cc)])
                for rr, cc, vv in neigh_pixels
                if int(vv) == int(v) and int(labv[int(rr), int(cc)]) != 0
            }
            for lbl in labs:
                for r, c in np.argwhere(labv == int(lbl)):
                    rr, cc = refl_point((int(pr), int(pc)), int(r), int(c))
                    paint_if_allowed(int(rr), int(cc), int(v))

    if collect_stats:
        return out, stats
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Same reflection parsing as the original task, but mirrored pixels only paint background cells.
    """
    return _solve(np.array(grid, dtype=int), bg_only=True, collect_stats=False)


def _random_connected_cells(
    height: int,
    width: int,
    *,
    min_cells: int,
    max_cells: int,
    connectivity: int,
    require_first_row: bool = False,
    require_first_col: bool = False,
    require_second_row: bool = False,
    require_second_col: bool = False,
) -> set[tuple[int, int]]:
    for _ in range(200):
        target = random.randint(min_cells, max_cells)
        starts = [(r, c) for r in range(height) for c in range(width)]
        if require_first_row:
            starts = [(r, c) for r, c in starts if r == 0]
        if require_first_col:
            starts = [(r, c) for r, c in starts if c == 0]
        start = random.choice(starts)
        cells = {start}
        while len(cells) < target:
            frontier = []
            for r, c in cells:
                neigh = _neighbors4(r, c) if connectivity == 4 else _neighbors8(r, c)
                for nr, nc in neigh:
                    if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) < min_cells:
            continue
        if require_first_row and not any(r == 0 for r, _c in cells):
            continue
        if require_first_col and not any(c == 0 for _r, c in cells):
            continue
        if require_second_row and not any(r >= 1 for r, _c in cells):
            continue
        if require_second_col and not any(c >= 1 for _r, c in cells):
            continue
        return cells
    raise RuntimeError("failed to make a connected shape")


def _build_row_patch(payload_color: int, blocker_color: int, with_blockers: bool, bg: int) -> np.ndarray:
    line_len = random.randint(2, 5)
    depth = random.randint(3, 5)
    side = random.choice(["above", "below"])
    patch = np.full((2 * depth + 1, line_len), bg, dtype=int)
    axis_r = depth
    patch[axis_r, :] = MARK

    cells = _random_connected_cells(
        depth,
        line_len,
        min_cells=3,
        max_cells=min(8, depth * line_len),
        connectivity=4,
        require_first_row=True,
        require_second_row=with_blockers,
    )

    blocker_targets = []
    for pr, pc in cells:
        if side == "below":
            src_r = axis_r + 1 + pr
            dst_r = axis_r - 1 - pr
        else:
            src_r = axis_r - 1 - pr
            dst_r = axis_r + 1 + pr
        patch[src_r, pc] = payload_color
        if pr >= 1:
            blocker_targets.append((dst_r, pc))

    if with_blockers and blocker_targets:
        random.shuffle(blocker_targets)
        for rr, cc in blocker_targets[: random.randint(1, min(3, len(blocker_targets)))]:
            patch[rr, cc] = blocker_color

    return patch


def _build_col_patch(payload_color: int, blocker_color: int, with_blockers: bool, bg: int) -> np.ndarray:
    line_len = random.randint(2, 5)
    depth = random.randint(3, 5)
    side = random.choice(["left", "right"])
    patch = np.full((line_len, 2 * depth + 1), bg, dtype=int)
    axis_c = depth
    patch[:, axis_c] = MARK

    cells = _random_connected_cells(
        line_len,
        depth,
        min_cells=3,
        max_cells=min(8, line_len * depth),
        connectivity=4,
        require_first_col=True,
        require_second_col=with_blockers,
    )

    blocker_targets = []
    for pr, pc in cells:
        if side == "right":
            src_c = axis_c + 1 + pc
            dst_c = axis_c - 1 - pc
        else:
            src_c = axis_c - 1 - pc
            dst_c = axis_c + 1 + pc
        patch[pr, src_c] = payload_color
        if pc >= 1:
            blocker_targets.append((pr, dst_c))

    if with_blockers and blocker_targets:
        random.shuffle(blocker_targets)
        for rr, cc in blocker_targets[: random.randint(1, min(3, len(blocker_targets)))]:
            patch[rr, cc] = blocker_color

    return patch


def _build_isolated_patch(payload_color: int, blocker_color: int, with_blockers: bool, bg: int) -> np.ndarray:
    height = random.randint(3, 4)
    width = random.randint(3, 4)
    sr, sc = random.choice([(1, 1), (1, -1), (-1, 1), (-1, -1)])
    radius = max(height, width) + 1
    patch = np.full((2 * radius + 1, 2 * radius + 1), bg, dtype=int)
    center = radius
    patch[center, center] = MARK

    cells = _random_connected_cells(
        height,
        width,
        min_cells=2,
        max_cells=min(6, height * width),
        connectivity=8,
        require_first_row=True,
        require_first_col=True,
        require_second_row=with_blockers,
    )

    blocker_targets = []
    for pr, pc in cells:
        dr = sr * (pr + 1)
        dc = sc * (pc + 1)
        src_r, src_c = center + dr, center + dc
        dst_r, dst_c = center - dr, center - dc
        patch[src_r, src_c] = payload_color
        if pr >= 1 or pc >= 1:
            blocker_targets.append((dst_r, dst_c))

    if with_blockers and blocker_targets:
        random.shuffle(blocker_targets)
        for rr, cc in blocker_targets[: random.randint(1, min(2, len(blocker_targets)))]:
            patch[rr, cc] = blocker_color

    return patch


def _choose_group_types(n: int) -> list[str]:
    types = ["row", "col"]
    while len(types) < n:
        types.append(random.choices(["row", "col", "isolated"], weights=[4, 4, 2])[0])
    random.shuffle(types)
    return types


def _place_patch(grid: np.ndarray, patch: np.ndarray, bg: int) -> bool:
    H, W = grid.shape
    h, w = patch.shape
    for _ in range(400):
        top = random.randint(0, H - h)
        left = random.randint(0, W - w)
        r0 = max(0, top - 1)
        c0 = max(0, left - 1)
        r1 = min(H, top + h + 1)
        c1 = min(W, left + w + 1)
        if np.any(grid[r0:r1, c0:c1] != bg):
            continue
        sub = grid[top : top + h, left : left + w]
        mask = patch != bg
        sub[mask] = patch[mask]
        return True
    return False


def _make_instance() -> np.ndarray:
    H = random.randint(22, 28)
    W = random.randint(22, 28)
    bg = random.choice([c for c in PALETTE if c != MARK])
    grid = np.full((H, W), bg, dtype=int)

    usable = [c for c in PALETTE if c not in (bg, MARK)]
    blocker_color = random.choice(usable)
    payload_pool = [c for c in usable if c != blocker_color]

    n_groups = random.randint(3, 5)
    group_types = _choose_group_types(n_groups)
    payload_colors = random.sample(payload_pool, n_groups)

    blocker_flags = [False] * n_groups
    must_block = random.sample(range(n_groups), k=random.randint(2, max(2, n_groups - 1)))
    for idx in must_block:
        blocker_flags[idx] = True

    builders = {
        "row": _build_row_patch,
        "col": _build_col_patch,
        "isolated": _build_isolated_patch,
    }

    for gtype, color, with_blockers in zip(group_types, payload_colors, blocker_flags):
        placed = False
        for _ in range(80):
            patch = builders[gtype](color, blocker_color, with_blockers, bg)
            if _place_patch(grid, patch, bg):
                placed = True
                break
        if not placed:
            raise RuntimeError("failed to place patch")

    return grid


def generate_input() -> np.ndarray:
    """
    Construct a non-degenerate D5 input:
    - marker geometry follows the original task
    - some mirrored target cells are already occupied by blocker pixels
    - new solver keeps those blockers, old overwrite behavior would erase them
    """
    for _ in range(300):
        try:
            grid = _make_instance()
        except RuntimeError:
            continue
        out_new, stats = _solve(grid, bg_only=True, collect_stats=True)
        out_old = _solve(grid, bg_only=False, collect_stats=False)
        diff = int(np.sum(out_new != out_old))
        if stats["blocked_existing"] >= 2 and diff >= 2 and np.array_equal(main(grid), out_new):
            return grid
    raise RuntimeError("failed to generate a non-degenerate D5 variant input")
