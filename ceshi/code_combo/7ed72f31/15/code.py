# Combo variant of 7ed72f31 (index 15)
# combo_spec: D5/0+D9/0
# Name: 无损移动反射

from __future__ import annotations

"""
Design analysis
===============

combo_variant:
  summary:
    - Keep the original reflection geometries of task 7ed72f31.
    - Merge D9 move semantics with D5 non-destructive placement:
      payloads are erased from their source side, but mirrored pixels only land on cells
      that were background in the input. Existing blockers survive.

  a_concretization_and_trace:
    - Concrete row example:
      input patch:
        - red horizontal axis in the middle
        - green payload below the axis
        - one blue blocker already sitting at one mirrored target cell above the axis
      original task:
        1. find the red row
        2. reflect every green pixel upward
        3. write every reflected green pixel unconditionally
        4. keep the original green payload below
        => output shows both copies and the blue blocker is overwritten
      fused variant:
        1. find the same red row
        2. collect the green source pixels
        3. erase those source pixels from the output
        4. for each reflected target, only paint if that input cell was background
        5. skip the target that already contains the blue blocker
        => output looks like the sprite moved across the red axis, but the blocked cell is
           missing and the blue blocker remains visible
    - The same logic applies to column reflections and isolated red-point reflections:
      move, do not copy; preserve blockers, do not overwrite them.

  b_complexity_calibration:
    - If the generator preserved the original "no target collisions" guarantee, D5 would be
      invisible and the combo would collapse into pure D9.
    - If the generator allowed arbitrary same-color clutter, D9 deletion would become noisy
      because unrelated pixels might be erased.
    - Therefore each instance uses:
      - 4-6 reflection groups
      - unique payload colors per group
      - one shared blocker color distinct from every payload color
      - at least 2 blocked reflection attempts overall
      - at least one row, one column, and one isolated-point group when possible
    - This keeps the move+skip rule repeated enough to infer visually without introducing
      ambiguous matching.

  c_implicit_assumption_audit_for_main:
    - Parsing non-background 4-connected components still assumes each marker-bearing group is
      intact in the input. The generator preserves that by keeping blockers away from the red
      component.
    - Deletion breaks the original bbox-wide color scan assumption: a loose scan could erase
      unrelated same-colored pixels. The fused solver therefore only moves payload pixels that
      belong to the same marker-bearing component.
    - Non-destructive placement plus deletion introduces a new subtlety: erasing the source
      before painting would falsely open cells that were occupied in the input. To avoid that,
      occupancy is checked against the original input grid `g`, not against the partially erased
      output canvas.
    - Isolated red markers assume exactly one intended adjacent 8-connected payload component.
      The generator enforces a halo so blockers never enter the marker's 8-neighborhood.
    - Paint order should not matter. The generator guarantees that visible mirrored targets of
      different groups do not overlap, so the solver never depends on cross-group write order.

  d_placement_legality:
    - Each group is built as a local patch containing the union of:
      - red markers
      - source payload cells
      - mirrored target cells
    - Global placement uses a one-cell halo around that whole union, guaranteeing:
      - no out-of-bounds placement
      - no cross-group overlap in either input or output
      - no accidental touching that would merge components or pollute isolated markers

  e_visual_observability:
    - The move part is obvious because the source side clears out in the output.
    - The D5 part is obvious because some mirrored shapes are incomplete exactly where blocker
      pixels remain unchanged.
    - Repeating blocked reflections in multiple groups makes the "only move into empty space"
      rule inferable from a small number of examples.

  f_visual_cleanliness:
    - Groups live in separated rectangles with halos, so local reflection stories are easy to
      read.
    - Only one extra blocker color is added, which avoids color noise.
    - Blockers are sparse and placed at target cells, so they read as occupancy constraints
      rather than as a second transformation system.

  g_color_inference:
    - No color remapping is introduced.
    - Payload colors remain unchanged under reflection.
    - The blocker color is stable and semantically means "already occupied".

  h_generator_non_degeneracy_checks:
    - Generated inputs are accepted only if:
      - the fused solver matches the manually constructed expected output
      - at least 2 reflection writes are blocked by existing non-background pixels
      - the fused output differs from D9-only output (so the D5 rule matters)
      - the fused output differs from D5-only output (so the D9 rule matters)

  i_new_solver_assumptions_to_satisfy:
    - Every moved source pixel belongs to exactly one intended reflection group.
    - Any blocked target is occupied already in the input, not by a later write.
    - No valid mirrored target that should remain visible is shared by two groups.
"""

import random
import sys
from pathlib import Path

import numpy as np

SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import get_bg, label_mask


Coord = tuple[int, int]
MARK = 2
PALETTE = tuple(range(10))


def _solve(
    grid: np.ndarray,
    *,
    bg_only: bool,
    erase_sources: bool,
    collect_stats: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, int]]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    h, w = g.shape

    def inb(r: int, c: int) -> bool:
        return 0 <= int(r) < h and 0 <= int(c) < w

    def refl_point(center: Coord, r: int, c: int) -> Coord:
        cr, cc = center
        return int(2 * cr - r), int(2 * cc - c)

    def refl_row(r0: int, r: int, c: int) -> Coord:
        return int(2 * r0 - r), int(c)

    def refl_col(c0: int, r: int, c: int) -> Coord:
        return int(r), int(2 * c0 - c)

    out = g.copy()
    erase: set[Coord] = set()
    paint: list[tuple[int, int, int]] = []
    stats = {
        "mixed_groups": 0,
        "isolated_markers": 0,
        "row_groups": 0,
        "col_groups": 0,
        "point_groups": 0,
        "source_pixels": 0,
        "erase_count": 0,
        "paint_attempts": 0,
        "writes": 0,
        "blocked_existing": 0,
        "out_of_bounds": 0,
    }

    def record_move(src: Coord, dst: Coord, value: int) -> None:
        erase.add((int(src[0]), int(src[1])))
        stats["source_pixels"] += 1
        if not inb(int(dst[0]), int(dst[1])):
            stats["out_of_bounds"] += 1
            return
        stats["paint_attempts"] += 1
        if bg_only and int(g[int(dst[0]), int(dst[1])]) != bg:
            stats["blocked_existing"] += 1
            return
        paint.append((int(dst[0]), int(dst[1]), int(value)))
        stats["writes"] += 1

    lab, n = label_mask(g != bg, connectivity=4)
    comps: list[dict[str, object]] = []
    for k in range(1, int(n) + 1):
        coords = [(int(r), int(c)) for r, c in np.argwhere(lab == int(k))]
        if not coords:
            continue
        vals = [int(g[r, c]) for r, c in coords]
        comps.append({"coords": coords, "vals": vals})

    for comp in comps:
        coords = comp["coords"]
        vals = comp["vals"]
        reds = [(int(r), int(c)) for (r, c), v in zip(coords, vals) if int(v) == MARK]
        if not reds:
            continue

        payload = [(int(r), int(c), int(v)) for (r, c), v in zip(coords, vals) if int(v) != MARK]
        if not payload:
            continue

        rows = {int(r) for r, _c in reds}
        cols = {int(c) for _r, c in reds}
        if len(reds) == 1:
            axis = ("point", (int(reds[0][0]), int(reds[0][1])))
            stats["point_groups"] += 1
        elif len(rows) == 1:
            axis = ("row", int(next(iter(rows))))
            stats["row_groups"] += 1
        elif len(cols) == 1:
            axis = ("col", int(next(iter(cols))))
            stats["col_groups"] += 1
        else:
            continue

        stats["mixed_groups"] += 1
        for r, c, v in payload:
            if axis[0] == "row":
                dst = refl_row(int(axis[1]), int(r), int(c))
            elif axis[0] == "col":
                dst = refl_col(int(axis[1]), int(r), int(c))
            else:
                dst = refl_point(axis[1], int(r), int(c))
            record_move((int(r), int(c)), dst, int(v))

    isolated_markers: list[Coord] = []
    for comp in comps:
        coords = comp["coords"]
        vals = comp["vals"]
        if len(coords) == 1 and int(vals[0]) == MARK:
            isolated_markers.append((int(coords[0][0]), int(coords[0][1])))

    for pr, pc in isolated_markers:
        stats["isolated_markers"] += 1
        neighbor_pixels: list[tuple[int, int, int]] = []
        neighbor_colors: set[int] = set()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = int(pr + dr), int(pc + dc)
                if not inb(rr, cc):
                    continue
                value = int(g[rr, cc])
                if value == bg or value == MARK:
                    continue
                neighbor_pixels.append((int(rr), int(cc), int(value)))
                neighbor_colors.add(int(value))

        for color in neighbor_colors:
            labv, _ = label_mask(g == int(color), connectivity=8)
            labels = {
                int(labv[int(rr), int(cc)])
                for rr, cc, vv in neighbor_pixels
                if int(vv) == int(color) and int(labv[int(rr), int(cc)]) != 0
            }
            for lbl in labels:
                for r, c in np.argwhere(labv == int(lbl)):
                    src = (int(r), int(c))
                    dst = refl_point((int(pr), int(pc)), int(r), int(c))
                    record_move(src, dst, int(color))

    if erase_sources:
        for r, c in erase:
            if int(out[r, c]) != MARK:
                out[r, c] = bg
        stats["erase_count"] = len(erase)

    for r, c, v in paint:
        out[int(r), int(c)] = int(v)

    if collect_stats:
        return out, stats
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reflect payloads using red markers, erase the sources, and only place mirrored pixels onto
    cells that were background in the input.
    """
    return _solve(np.array(grid, dtype=int), bg_only=True, erase_sources=True, collect_stats=False)


def _grow_connected(
    height: int,
    width: int,
    size: int,
    *,
    connectivity: int,
    seed: Coord,
    rng: random.Random,
) -> set[Coord]:
    if not (0 <= seed[0] < height and 0 <= seed[1] < width):
        raise ValueError("seed outside box")
    if size < 1 or size > height * width:
        raise ValueError("invalid size")

    if connectivity == 4:
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        nbrs = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]

    cells = {seed}
    frontier: set[Coord] = set()

    def push_neighbors(r: int, c: int) -> None:
        for dr, dc in nbrs:
            rr, cc = int(r + dr), int(c + dc)
            if 0 <= rr < height and 0 <= cc < width and (rr, cc) not in cells:
                frontier.add((rr, cc))

    push_neighbors(*seed)
    while len(cells) < size:
        if not frontier:
            raise ValueError("growth got stuck")
        nxt = rng.choice(tuple(frontier))
        frontier.remove(nxt)
        cells.add(nxt)
        push_neighbors(*nxt)
    return cells


def _make_row_group(color: int, blocker_color: int, with_blockers: bool, rng: random.Random) -> dict[str, object]:
    for _ in range(120):
        red_len = rng.randint(2, 6)
        depth = rng.randint(3, 5)
        size = rng.randint(3, min(11, depth * red_len))
        side = rng.choice(["above", "below"])
        box = _grow_connected(depth, red_len, size, connectivity=4, seed=(0, rng.randrange(red_len)), rng=rng)
        blocker_choices = [(r, c) for r, c in box if r >= 1]
        if with_blockers and not blocker_choices:
            continue

        axis_r = depth
        reds = [(axis_r, c) for c in range(red_len)]
        src: list[tuple[int, int, int]] = []
        dst_all: list[tuple[int, int, int]] = []
        for pr, pc in box:
            if side == "below":
                src_r = axis_r + 1 + pr
                dst_r = axis_r - 1 - pr
            else:
                src_r = axis_r - 1 - pr
                dst_r = axis_r + 1 + pr
            src.append((int(src_r), int(pc), int(color)))
            dst_all.append((int(dst_r), int(pc), int(color)))

        blocked = set(rng.sample(blocker_choices, rng.randint(1, min(3, len(blocker_choices))))) if with_blockers else set()
        dst: list[tuple[int, int, int]] = []
        blockers: list[tuple[int, int, int]] = []
        for pr, pc in box:
            if side == "below":
                dst_r = axis_r - 1 - pr
            else:
                dst_r = axis_r + 1 + pr
            if (pr, pc) in blocked:
                blockers.append((int(dst_r), int(pc), int(blocker_color)))
            else:
                dst.append((int(dst_r), int(pc), int(color)))

        return {
            "type": "row",
            "height": 2 * depth + 1,
            "width": red_len,
            "reds": reds,
            "src": src,
            "dst": dst,
            "dst_all": dst_all,
            "blockers": blockers,
        }
    raise ValueError("failed to build row group")


def _make_col_group(color: int, blocker_color: int, with_blockers: bool, rng: random.Random) -> dict[str, object]:
    for _ in range(120):
        red_len = rng.randint(2, 6)
        depth = rng.randint(3, 5)
        size = rng.randint(3, min(11, red_len * depth))
        side = rng.choice(["left", "right"])
        box = _grow_connected(red_len, depth, size, connectivity=4, seed=(rng.randrange(red_len), 0), rng=rng)
        blocker_choices = [(r, c) for r, c in box if c >= 1]
        if with_blockers and not blocker_choices:
            continue

        axis_c = depth
        reds = [(r, axis_c) for r in range(red_len)]
        src: list[tuple[int, int, int]] = []
        dst_all: list[tuple[int, int, int]] = []
        for pr, pc in box:
            if side == "right":
                src_c = axis_c + 1 + pc
                dst_c = axis_c - 1 - pc
            else:
                src_c = axis_c - 1 - pc
                dst_c = axis_c + 1 + pc
            src.append((int(pr), int(src_c), int(color)))
            dst_all.append((int(pr), int(dst_c), int(color)))

        blocked = set(rng.sample(blocker_choices, rng.randint(1, min(3, len(blocker_choices))))) if with_blockers else set()
        dst: list[tuple[int, int, int]] = []
        blockers: list[tuple[int, int, int]] = []
        for pr, pc in box:
            if side == "right":
                dst_c = axis_c - 1 - pc
            else:
                dst_c = axis_c + 1 + pc
            if (pr, pc) in blocked:
                blockers.append((int(pr), int(dst_c), int(blocker_color)))
            else:
                dst.append((int(pr), int(dst_c), int(color)))

        return {
            "type": "col",
            "height": red_len,
            "width": 2 * depth + 1,
            "reds": reds,
            "src": src,
            "dst": dst,
            "dst_all": dst_all,
            "blockers": blockers,
        }
    raise ValueError("failed to build col group")


def _make_isolated_group(color: int, blocker_color: int, with_blockers: bool, rng: random.Random) -> dict[str, object]:
    for _ in range(120):
        quad_h = rng.randint(2, 4)
        quad_w = rng.randint(2, 4)
        min_size = 2 if with_blockers else 1
        size = rng.randint(min_size, min(8, quad_h * quad_w))
        box = _grow_connected(quad_h, quad_w, size, connectivity=8, seed=(0, 0), rng=rng)
        blocker_choices = [(r, c) for r, c in box if r >= 1 or c >= 1]
        if with_blockers and not blocker_choices:
            continue

        sr = rng.choice([-1, 1])
        sc = rng.choice([-1, 1])
        center = (quad_h, quad_w)
        reds = [center]
        src: list[tuple[int, int, int]] = []
        dst_all: list[tuple[int, int, int]] = []
        for pr, pc in box:
            dr, dc = int(pr + 1), int(pc + 1)
            src_r = int(center[0] + sr * dr)
            src_c = int(center[1] + sc * dc)
            dst_r = int(center[0] - sr * dr)
            dst_c = int(center[1] - sc * dc)
            src.append((int(src_r), int(src_c), int(color)))
            dst_all.append((int(dst_r), int(dst_c), int(color)))

        blocked = set(rng.sample(blocker_choices, rng.randint(1, min(2, len(blocker_choices))))) if with_blockers else set()
        dst: list[tuple[int, int, int]] = []
        blockers: list[tuple[int, int, int]] = []
        for pr, pc in box:
            dr, dc = int(pr + 1), int(pc + 1)
            dst_r = int(center[0] - sr * dr)
            dst_c = int(center[1] - sc * dc)
            if (pr, pc) in blocked:
                blockers.append((int(dst_r), int(dst_c), int(blocker_color)))
            else:
                dst.append((int(dst_r), int(dst_c), int(color)))

        return {
            "type": "isolated",
            "height": quad_h * 2 + 1,
            "width": quad_w * 2 + 1,
            "reds": reds,
            "src": src,
            "dst": dst,
            "dst_all": dst_all,
            "blockers": blockers,
        }
    raise ValueError("failed to build isolated group")


def _expand(coords: set[Coord], height: int, width: int, pad: int = 1) -> set[Coord]:
    out: set[Coord] = set()
    for r, c in coords:
        for dr in range(-pad, pad + 1):
            for dc in range(-pad, pad + 1):
                rr, cc = int(r + dr), int(c + dc)
                if 0 <= rr < height and 0 <= cc < width:
                    out.add((rr, cc))
    return out


def _place_group(
    inp: np.ndarray,
    expected: np.ndarray,
    occupied: np.ndarray,
    group: dict[str, object],
    rng: random.Random,
) -> bool:
    h, w = inp.shape
    gh = int(group["height"])
    gw = int(group["width"])
    reds = [(int(r), int(c)) for r, c in group["reds"]]
    src = [(int(r), int(c), int(v)) for r, c, v in group["src"]]
    dst = [(int(r), int(c), int(v)) for r, c, v in group["dst"]]
    dst_all = [(int(r), int(c), int(v)) for r, c, v in group["dst_all"]]
    blockers = [(int(r), int(c), int(v)) for r, c, v in group["blockers"]]

    footprint = {(r, c) for r, c in reds}
    footprint |= {(r, c) for r, c, _v in src}
    footprint |= {(r, c) for r, c, _v in dst_all}

    for _ in range(240):
        r0 = rng.randint(0, h - gh)
        c0 = rng.randint(0, w - gw)
        placed = {(r0 + r, c0 + c) for r, c in footprint}
        halo = _expand(placed, h, w, pad=1)
        if any(bool(occupied[r, c]) for r, c in halo):
            continue

        for r, c in halo:
            occupied[r, c] = True
        for r, c in reds:
            inp[r0 + r, c0 + c] = MARK
            expected[r0 + r, c0 + c] = MARK
        for r, c, v in src:
            inp[r0 + r, c0 + c] = int(v)
        for r, c, v in dst:
            expected[r0 + r, c0 + c] = int(v)
        for r, c, v in blockers:
            inp[r0 + r, c0 + c] = int(v)
            expected[r0 + r, c0 + c] = int(v)
        return True
    return False


def _choose_types(n_groups: int, rng: random.Random) -> list[str]:
    base = ["row", "col", "isolated"][: min(3, n_groups)]
    while len(base) < n_groups:
        base.append(rng.choices(["row", "col", "isolated"], weights=[4, 4, 2])[0])
    rng.shuffle(base)
    return base


def _build_instance(rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    h = rng.randint(20, 28)
    w = rng.randint(20, 28)
    bg = rng.choice([c for c in PALETTE if c != MARK])
    inp = np.full((h, w), bg, dtype=int)
    expected = np.full((h, w), bg, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    usable = [c for c in PALETTE if c not in (bg, MARK)]
    blocker_color = rng.choice(usable)
    payload_pool = [c for c in usable if c != blocker_color]

    n_groups = rng.randint(4, 6)
    types = _choose_types(n_groups, rng)
    payload_colors = rng.sample(payload_pool, n_groups)

    block_flags = [False] * n_groups
    blocked_group_count = rng.randint(2, min(4, n_groups))
    for idx in rng.sample(range(n_groups), blocked_group_count):
        block_flags[idx] = True

    builders = {
        "row": _make_row_group,
        "col": _make_col_group,
        "isolated": _make_isolated_group,
    }

    for typ, color, with_blockers in zip(types, payload_colors, block_flags):
        placed = False
        for _ in range(120):
            group = builders[typ](int(color), int(blocker_color), bool(with_blockers), rng)
            if _place_group(inp, expected, occupied, group, rng):
                placed = True
                break
        if not placed:
            raise ValueError(f"failed to place {typ} group")

    return inp, expected


def generate_input() -> np.ndarray:
    """
    Generate a non-degenerate combo instance where both dimensions matter:
    - D9: payloads are erased from the source side
    - D5: some mirrored targets are blocked and therefore remain occupied by blocker pixels
    """
    rng = random.Random()
    last_error: Exception | None = None
    for _ in range(300):
        try:
            inp, expected = _build_instance(rng)
            fused, fused_stats = _solve(inp, bg_only=True, erase_sources=True, collect_stats=True)
            d9_only = _solve(inp, bg_only=False, erase_sources=True, collect_stats=False)
            d5_only = _solve(inp, bg_only=True, erase_sources=False, collect_stats=False)

            diff_vs_d9 = int(np.sum(fused != d9_only))
            diff_vs_d5 = int(np.sum(fused != d5_only))
            if not np.array_equal(fused, expected):
                raise AssertionError("manual expected output mismatch")
            if fused_stats["blocked_existing"] < 2:
                raise AssertionError("not enough blocked reflections")
            if fused_stats["erase_count"] < 8:
                raise AssertionError("not enough moved source pixels")
            if diff_vs_d9 < 2:
                raise AssertionError("D5 effect is not observable")
            if diff_vs_d5 < 4:
                raise AssertionError("D9 effect is not observable")
            return inp
        except Exception as exc:  # pragma: no cover - retry logic
            last_error = exc
    raise RuntimeError(f"generate_input failed after retries: {last_error}")
