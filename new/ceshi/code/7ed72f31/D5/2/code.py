# Variant of 7ed72f31
# Dimension: D5 — 镜像像素的覆盖规则 (transform)
# Change: 镜像像素直接覆盖写入，不检查目标位置是否已有非背景像素... → 碰撞检测：重叠时删除两者
#
# Design analysis
# ---------------
# yaml:
#   source_task:
#     id: 409aa875
#     solver_status: solver code not found in the provided local workspace, so the
#       collision idea is implemented directly from the variant description rather
#       than copied from a local reference solver.
#   concretization_and_trace:
#     original_behavior:
#       - Every reflected payload pixel is written straight into the output.
#       - If a reflected pixel lands on an existing non-background pixel, the old
#         solver simply overwrites that cell with the reflected color.
#     variant_behavior:
#       - Reflection geometry is unchanged: red row -> row mirror, red column ->
#         column mirror, isolated red point -> point reflection of the adjacent
#         8-connected component, embedded single red point -> point reflection.
#       - Only the write rule changes: when at least one reflected pixel targets a
#         cell that is already non-background in the input, the destination becomes
#         background instead of being overwritten. Both the blocker pixel and the
#         incoming reflected pixel disappear.
#       - For malformed inputs with multiple reflected pixels targeting one empty
#         cell, the solver also cancels that stack to background. The generator
#         avoids such mirror-vs-mirror overlaps, so human-facing samples only need
#         the simpler "mirror hits existing object => both vanish" rule.
#     small_trace:
#       - Row example:
#         input: red row at r=5, source payload cell color 6 at (7, 9), unrelated
#           blocker cell color 8 already at mirror target (3, 9).
#         old solver: reflected target (3, 9) is written as 6, so blocker 8 is
#           overwritten and disappears only on one side.
#         variant solver: reflected target (3, 9) sees an existing non-background
#           pixel, so output(3, 9) becomes background 0/other-bg instead.
#         difference: the mirrored copy has a hole there, and the blocker also
#           disappears. Geometry stays the same; only collision resolution changes.
#   complexity_calibration:
#     - A single accidental collision would be too subtle. The generator enforces
#       at least 3 collision sites per sample and spreads them across at least two
#       row/column reflection groups.
#     - Marker types remain mostly the original ones (row / col plus optional
#       isolated point groups). No new transformation classes are introduced.
#   implicit_assumptions_review_of_original_main:
#     - `out = g.copy()` is still valid because original pixels remain unless a
#       reflected pixel collides with them.
#     - Every `out[rr, cc] = v` write in the original code assumed overwrite was
#       always correct. Under D5 this is false; those writes must be deferred and
#       resolved against the pre-existing non-background occupancy.
#     - The component detection, axis classification, bbox eligibility, and the
#       isolated-red component lookup all remain valid because D5 changes only the
#       write policy, not the reflection geometry.
#     - The original solver's bbox-based global scan can accidentally capture
#       unrelated same-color pixels. The generator therefore keeps payload colors
#       unique per reflection group and uses a separate blocker color that is never
#       reused as a payload color.
#   placement_legality:
#     - Reflection groups are still built output-footprint-first: each template
#       knows its input cells and its mirror-target cells.
#     - Groups are placed with a 1-cell Chebyshev moat using their full
#       no-collision output footprints, so groups do not merge, overlap, or leave
#       the grid.
#     - Blockers are inserted only on mirror-target cells that are 4-disconnected
#       from all input pixels; this prevents them from merging into a marker
#       component in the input while still guaranteeing a collision in the output.
#   visual_observability:
#     - In every sample, several blocker pixels appear inside regions where a human
#       expects the mirrored copy to grow. In the output those blocker pixels and
#       the corresponding mirrored pixels both vanish, producing visible holes.
#     - The generator rejects degenerate samples whose collision-aware output is
#       identical to the naive overlay output.
#   cleanliness:
#     - Collision blockers are small isolated pixels of one extra color rather than
#       many complex distractor sprites, so the grid remains readable.
#   new_solver_assumptions_and_generator_support:
#     - Assumption: payload colors are unique per reflection group, so bbox scans do
#       not capture other reflectable objects. Generator guarantees unique payload
#       colors.
#     - Assumption: blockers are not 4-connected to any non-background input cell.
#       Generator chooses only "safe" mirror-target cells for blockers.
#     - Assumption: mirror-vs-mirror overlaps are absent in generated data, keeping
#       collision semantics visually simple. Generator places groups by full output
#       footprints with a moat to guarantee this.

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import get_bg, label_mask


MARK = 2


def _inb(H: int, W: int, r: int, c: int) -> bool:
    return 0 <= int(r) < int(H) and 0 <= int(c) < int(W)


def _refl_point(center: tuple[int, int], r: int, c: int) -> tuple[int, int]:
    cr, cc = map(int, center)
    return int(2 * cr - r), int(2 * cc - c)


def _refl_row(r0: int, r: int, c: int) -> tuple[int, int]:
    return int(2 * int(r0) - int(r)), int(c)


def _refl_col(c0: int, r: int, c: int) -> tuple[int, int]:
    return int(r), int(2 * int(c0) - int(c))


def _build_components(g: np.ndarray, bg: int) -> list[dict[str, object]]:
    lab, n = label_mask(g != int(bg), connectivity=4)
    comps: list[dict[str, object]] = []
    for k in range(1, int(n) + 1):
        coords = [(int(r), int(c)) for r, c in np.argwhere(lab == int(k))]
        if not coords:
            continue
        vals = [int(g[r, c]) for r, c in coords]
        rs = [int(r) for r, _c in coords]
        cs = [int(c) for _r, c in coords]
        comps.append(
            {
                "coords": coords,
                "vals": vals,
                "bbox": (int(min(rs)), int(min(cs)), int(max(rs)), int(max(cs))),
            }
        )
    return comps


def _collect_reflection_hits(g: np.ndarray, bg: int) -> dict[tuple[int, int], list[int]]:
    H, W = g.shape
    comps = _build_components(g, int(bg))
    hits: dict[tuple[int, int], list[int]] = {}

    def emit(rr: int, cc: int, v: int) -> None:
        if _inb(H, W, int(rr), int(cc)):
            hits.setdefault((int(rr), int(cc)), []).append(int(v))

    # 1) Marker-containing components.
    for comp in comps:
        coords = comp["coords"]
        vals = comp["vals"]
        if int(MARK) not in set(map(int, vals)):
            continue

        reds = [(int(r), int(c)) for (r, c), v in zip(coords, vals) if int(v) == int(MARK)]
        if not reds:
            continue

        rs = {int(r) for r, _c in reds}
        cs = {int(c) for _r, c in reds}
        if len(reds) == 1:
            axis = ("point", (int(reds[0][0]), int(reds[0][1])))
        elif len(rs) == 1:
            axis = ("row", int(next(iter(rs))))
        elif len(cs) == 1:
            axis = ("col", int(next(iter(cs))))
        else:
            continue

        r0, c0, r1, c1 = map(int, comp["bbox"])
        payload = {int(v) for v in vals if int(v) != int(MARK)}
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
                if v == int(bg) or v == int(MARK):
                    continue
                if not eligible(int(r), int(c), int(v)):
                    continue

                if axis[0] == "row":
                    rr, cc = _refl_row(int(axis[1]), int(r), int(c))
                elif axis[0] == "col":
                    rr, cc = _refl_col(int(axis[1]), int(r), int(c))
                else:
                    rr, cc = _refl_point(axis[1], int(r), int(c))
                emit(int(rr), int(cc), int(v))

    # 2) Isolated red pixels: point-reflect each adjacent 8-connected component.
    for comp in comps:
        if len(comp["coords"]) != 1 or int(comp["vals"][0]) != int(MARK):
            continue
        pr, pc = map(int, comp["coords"][0])
        neigh_colors: set[int] = set()
        neigh_pixels: list[tuple[int, int, int]] = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = int(pr + dr), int(pc + dc)
                if not _inb(H, W, int(rr), int(cc)):
                    continue
                v = int(g[int(rr), int(cc)])
                if v == int(bg) or v == int(MARK):
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
                    rr, cc = _refl_point((int(pr), int(pc)), int(r), int(c))
                    emit(int(rr), int(cc), int(v))
    return hits


def _resolve_hits(g: np.ndarray, bg: int, hits: dict[tuple[int, int], list[int]], *, collision: bool) -> np.ndarray:
    out = np.array(g, dtype=int, copy=True)
    for (r, c), vals in hits.items():
        if collision:
            if int(out[int(r), int(c)]) != int(bg) or len(vals) >= 2:
                out[int(r), int(c)] = int(bg)
            else:
                out[int(r), int(c)] = int(vals[0])
        else:
            out[int(r), int(c)] = int(vals[-1])
    return out


def _overlay_main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    hits = _collect_reflection_hits(g, int(bg))
    return _resolve_hits(g, int(bg), hits, collision=False)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Original geometry, new D5 write rule:
    mirrored pixels are collected first, then any overlap with an existing
    non-background cell is resolved by deleting both to background.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    hits = _collect_reflection_hits(g, int(bg))
    return _resolve_hits(g, int(bg), hits, collision=True)


def _neighbors4(cell: tuple[int, int]) -> Iterable[tuple[int, int]]:
    r, c = map(int, cell)
    yield int(r - 1), int(c)
    yield int(r + 1), int(c)
    yield int(r), int(c - 1)
    yield int(r), int(c + 1)


def _neighbors8(cell: tuple[int, int]) -> Iterable[tuple[int, int]]:
    r, c = map(int, cell)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield int(r + dr), int(c + dc)


def _grow_connected(
    area: int,
    allowed: set[tuple[int, int]],
    seed: tuple[int, int],
    connectivity: int,
    rng: random.Random,
    tries: int = 240,
) -> set[tuple[int, int]] | None:
    neigh = _neighbors4 if int(connectivity) == 4 else _neighbors8
    allowed = {(int(r), int(c)) for r, c in allowed}
    seed = (int(seed[0]), int(seed[1]))
    if seed not in allowed:
        return None

    for _ in range(int(tries)):
        cells = {seed}
        while len(cells) < int(area):
            frontier = set()
            for cell in cells:
                for nxt in neigh(cell):
                    if nxt in allowed and nxt not in cells:
                        frontier.add(nxt)
            if not frontier:
                break
            cells.add(rng.choice(tuple(frontier)))
        if len(cells) == int(area):
            return cells
    return None


def _safe_blocker_candidates(
    mirror_cells: dict[tuple[int, int], int],
    input_cells: dict[tuple[int, int], int],
) -> list[tuple[int, int]]:
    input_pts = {(int(r), int(c)) for r, c in input_cells}
    safe: list[tuple[int, int]] = []
    for r, c in mirror_cells:
        if all((int(rr), int(cc)) not in input_pts for rr, cc in _neighbors4((int(r), int(c)))):
            safe.append((int(r), int(c)))
    return safe


def _normalize_group(
    input_cells: dict[tuple[int, int], int],
    mirror_cells: dict[tuple[int, int], int],
    meta: dict[str, object],
) -> dict[str, object]:
    full = dict(input_cells)
    full.update(mirror_cells)
    pts = list(full)
    min_r = min(int(r) for r, _c in pts)
    min_c = min(int(c) for _r, c in pts)

    in_norm = {(int(r - min_r), int(c - min_c)): int(v) for (r, c), v in input_cells.items()}
    mirror_norm = {(int(r - min_r), int(c - min_c)): int(v) for (r, c), v in mirror_cells.items()}
    full_norm = {(int(r - min_r), int(c - min_c)): int(v) for (r, c), v in full.items()}
    safe = _safe_blocker_candidates(mirror_norm, in_norm)
    h = 1 + max(int(r) for r, _c in full_norm)
    w = 1 + max(int(c) for _r, c in full_norm)

    meta = dict(meta)
    meta["height"] = int(h)
    meta["width"] = int(w)
    meta["mirror_pixels"] = int(len(mirror_norm))
    return {
        "in": in_norm,
        "mirror": mirror_norm,
        "out_full": full_norm,
        "safe_blockers": safe,
        "h": int(h),
        "w": int(w),
        "meta": meta,
    }


def _build_row_group(color: int, rng: random.Random) -> dict[str, object]:
    for _ in range(240):
        line_len = rng.randint(2, 6)
        max_depth = rng.randint(2, 5)
        area = rng.randint(3, min(11, int(line_len * max_depth)))
        allowed = {(int(r), int(c)) for r in range(max_depth) for c in range(line_len)}
        seed = (0, rng.randrange(line_len))
        payload = _grow_connected(area, allowed, seed, connectivity=4, rng=rng)
        if payload is None:
            continue
        if max(int(r) for r, _c in payload) < 1:
            continue

        side = rng.choice((-1, 1))
        input_cells: dict[tuple[int, int], int] = {}
        mirror_cells: dict[tuple[int, int], int] = {}
        for c in range(line_len):
            input_cells[(0, int(c))] = int(MARK)
        for r, c in payload:
            rr = int(side * (int(r) + 1))
            cc = int(c)
            input_cells[(rr, cc)] = int(color)
            mirror_cells[(-rr, cc)] = int(color)

        group = _normalize_group(input_cells, mirror_cells, {"type": "row", "color": int(color)})
        if len(group["safe_blockers"]) >= 2:
            return group
    raise RuntimeError("failed to build row group")


def _build_col_group(color: int, rng: random.Random) -> dict[str, object]:
    for _ in range(240):
        line_len = rng.randint(2, 6)
        max_depth = rng.randint(2, 5)
        area = rng.randint(3, min(11, int(line_len * max_depth)))
        allowed = {(int(r), int(c)) for r in range(line_len) for c in range(max_depth)}
        seed = (rng.randrange(line_len), 0)
        payload = _grow_connected(area, allowed, seed, connectivity=4, rng=rng)
        if payload is None:
            continue
        if max(int(c) for _r, c in payload) < 1:
            continue

        side = rng.choice((-1, 1))
        input_cells: dict[tuple[int, int], int] = {}
        mirror_cells: dict[tuple[int, int], int] = {}
        for r in range(line_len):
            input_cells[(int(r), 0)] = int(MARK)
        for r, c in payload:
            rr = int(r)
            cc = int(side * (int(c) + 1))
            input_cells[(rr, cc)] = int(color)
            mirror_cells[(rr, -cc)] = int(color)

        group = _normalize_group(input_cells, mirror_cells, {"type": "col", "color": int(color)})
        if len(group["safe_blockers"]) >= 2:
            return group
    raise RuntimeError("failed to build col group")


def _build_iso_group(color: int, rng: random.Random) -> dict[str, object]:
    for _ in range(240):
        max_h = rng.randint(2, 4)
        max_w = rng.randint(2, 4)
        area = rng.randint(1, min(8, int(max_h * max_w)))
        allowed = {(int(r), int(c)) for r in range(max_h) for c in range(max_w)}
        payload = _grow_connected(area, allowed, (0, 0), connectivity=8, rng=rng)
        if payload is None:
            continue

        sr, sc = rng.choice(((-1, -1), (-1, 1), (1, -1), (1, 1)))
        input_cells: dict[tuple[int, int], int] = {(0, 0): int(MARK)}
        mirror_cells: dict[tuple[int, int], int] = {}
        for r, c in payload:
            rr = int(sr * (int(r) + 1))
            cc = int(sc * (int(c) + 1))
            input_cells[(rr, cc)] = int(color)
            mirror_cells[(-rr, -cc)] = int(color)

        return _normalize_group(input_cells, mirror_cells, {"type": "iso", "color": int(color)})
    raise RuntimeError("failed to build isolated group")


def _build_group(kind: str, color: int, rng: random.Random) -> dict[str, object]:
    if kind == "row":
        return _build_row_group(int(color), rng)
    if kind == "col":
        return _build_col_group(int(color), rng)
    if kind == "iso":
        return _build_iso_group(int(color), rng)
    raise ValueError(f"unknown group kind: {kind}")


def _can_place(
    group: dict[str, object],
    top: int,
    left: int,
    H: int,
    W: int,
    forbidden: set[tuple[int, int]],
) -> bool:
    for r, c in group["out_full"]:
        rr = int(top + r)
        cc = int(left + c)
        if not _inb(H, W, int(rr), int(cc)):
            return False
        if (int(rr), int(cc)) in forbidden:
            return False
    return True


def _stamp_moat(forbidden: set[tuple[int, int]], cells: Iterable[tuple[int, int]]) -> None:
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                forbidden.add((int(r + dr), int(c + dc)))


def _is_blocker_safe(inp: np.ndarray, pos: tuple[int, int], chosen: set[tuple[int, int]], bg: int) -> bool:
    r, c = map(int, pos)
    if int(inp[int(r), int(c)]) != int(bg):
        return False
    H, W = inp.shape
    for rr, cc in _neighbors4((int(r), int(c))):
        if not _inb(H, W, int(rr), int(cc)):
            continue
        if (int(rr), int(cc)) in chosen:
            return False
        if int(inp[int(rr), int(cc)]) != int(bg):
            return False
    return True


def _place_blockers(
    inp: np.ndarray,
    placed: list[dict[str, object]],
    blocker_color: int,
    bg: int,
    rng: random.Random,
) -> list[tuple[int, int]]:
    carrier_groups = [p for p in placed if p["type"] in ("row", "col") and p["safe_abs"]]
    if len(carrier_groups) < 2:
        return []

    rng.shuffle(carrier_groups)
    chosen_groups = carrier_groups[: rng.randint(2, min(3, len(carrier_groups)))]
    chosen: set[tuple[int, int]] = set()

    # First guarantee coverage across at least two groups.
    for group in chosen_groups:
        cands = [tuple(map(int, pos)) for pos in group["safe_abs"] if _is_blocker_safe(inp, tuple(map(int, pos)), chosen, int(bg))]
        if not cands:
            return []
        pos = rng.choice(cands)
        chosen.add((int(pos[0]), int(pos[1])))

    target_total = rng.randint(max(3, len(chosen)), min(6, sum(len(p["safe_abs"]) for p in carrier_groups)))
    all_cands = []
    for group in carrier_groups:
        for pos in group["safe_abs"]:
            all_cands.append(tuple(map(int, pos)))
    rng.shuffle(all_cands)
    for pos in all_cands:
        if len(chosen) >= int(target_total):
            break
        if pos in chosen:
            continue
        if _is_blocker_safe(inp, pos, chosen, int(bg)):
            chosen.add((int(pos[0]), int(pos[1])))

    for r, c in chosen:
        inp[int(r), int(c)] = int(blocker_color)
    return sorted(chosen)


def generate_case(rng: random.Random | None = None) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    rng = rng or random.Random()

    for _ in range(500):
        H = rng.randint(18, 28)
        W = rng.randint(18, 28)
        bg = rng.choice([c for c in range(10) if c != int(MARK)])

        usable = [c for c in range(10) if c not in (int(bg), int(MARK))]
        rng.shuffle(usable)
        n_groups = rng.randint(3, min(5, len(usable) - 1))
        group_colors = usable[:n_groups]
        blocker_color = int(usable[n_groups])

        iso_n = rng.randint(0, min(1, n_groups - 2))
        base_kinds = ["row", "col"] + [rng.choice(["row", "col", "iso"]) for _ in range(n_groups - 2)]
        if iso_n == 0:
            base_kinds = [k for k in base_kinds if k != "iso"]
            while len(base_kinds) < n_groups:
                base_kinds.append(rng.choice(["row", "col"]))
        else:
            iso_positions = rng.sample(range(n_groups), iso_n)
            kinds = []
            for idx in range(n_groups):
                if idx in iso_positions:
                    kinds.append("iso")
                elif not kinds:
                    kinds.append("row")
                elif "col" not in kinds and idx == n_groups - 1:
                    kinds.append("col")
                else:
                    kinds.append(rng.choice(["row", "col"]))
            base_kinds = kinds
        if "row" not in base_kinds or "col" not in base_kinds:
            continue

        try:
            groups = [_build_group(kind, int(color), rng) for kind, color in zip(base_kinds, group_colors)]
        except RuntimeError:
            continue
        groups.sort(key=lambda g: len(g["out_full"]), reverse=True)

        inp = np.full((H, W), int(bg), dtype=int)
        forbidden: set[tuple[int, int]] = set()
        placed: list[dict[str, object]] = []
        ok = True

        for group in groups:
            gh = int(group["h"])
            gw = int(group["w"])
            placed_here = False
            for _attempt in range(700):
                top = rng.randint(0, H - gh)
                left = rng.randint(0, W - gw)
                if not _can_place(group, int(top), int(left), H, W, forbidden):
                    continue

                out_cells_abs = []
                for (r, c), v in group["in"].items():
                    inp[int(top + r), int(left + c)] = int(v)
                for r, c in group["out_full"]:
                    out_cells_abs.append((int(top + r), int(left + c)))

                _stamp_moat(forbidden, out_cells_abs)
                meta = dict(group["meta"])
                meta["top"] = int(top)
                meta["left"] = int(left)
                meta["safe_abs"] = [(int(top + r), int(left + c)) for r, c in group["safe_blockers"]]
                placed.append(meta)
                placed_here = True
                break

            if not placed_here:
                ok = False
                break

        if not ok:
            continue

        blocker_cells = _place_blockers(inp, placed, int(blocker_color), int(bg), rng)
        if len(blocker_cells) < 3:
            continue

        out = main(inp)
        overlay = _overlay_main(inp)
        if np.array_equal(out, overlay):
            continue

        collision_mask = (inp != int(bg)) & (out == int(bg)) & (overlay != int(bg))
        collision_pixels = int(np.sum(collision_mask))
        if collision_pixels < 3:
            continue

        if not np.array_equal(main(inp), out):
            continue

        meta = {
            "bg": int(bg),
            "grid_shape": (int(H), int(W)),
            "group_types": [str(p["type"]) for p in placed],
            "group_colors": [int(p["color"]) for p in placed],
            "blocker_color": int(blocker_color),
            "blocker_cells": blocker_cells,
            "collision_pixels": int(collision_pixels),
            "delta_vs_overlay": int(np.sum(out != overlay)),
        }
        return inp, out, meta

    raise RuntimeError("failed to generate a valid collision-reflection case")


def generate_input() -> np.ndarray:
    inp, _out, _meta = generate_case()
    return inp
