# Variant of 7ed72f31
# Dimension: D10 — 符号编码与图例机制 (symbol_instructions)
# Change: 无符号编码，红色固定为标记色，变换规则固定... → 标记像素的排列编码反射方向

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import get_bg, label_mask


# Design analysis for the D10 variant
# ----------------------------------
# spec_variant:
#   core_change:
#     original: red line geometry only said "this is a reflection axis"; the payload side itself
#       implicitly told the solver what would be copied.
#     variant: a red line still gives the axis location, but its pixel pattern now also encodes
#       direction. Solid line = positive direction, one internal gap = negative direction.
#       vertical: solid => copy left payload to the right; gapped => copy right payload to the left.
#       horizontal: solid => copy upper payload downward; gapped => copy lower payload upward.
#   concrete_pixel_behavior:
#     line_groups:
#       - each mixed red component contains payload on both sides of the axis.
#       - one side is a large asymmetric source blob; the other side is a small attached same-color stub.
#       - output copies only the source-side blob across the axis; the stub remains unchanged.
#       - therefore the code carried by "solid vs one-gap" is visible: the same kind of line can point
#         the copy to opposite sides.
#     isolated_red_pixels:
#       - unchanged from the original task: a lone red pixel point-reflects one adjacent 8-connected
#         monochrome component.
#   behavior_trace_example:
#     original_main:
#       - collect a marker-containing component, infer vertical axis from red column, then reflect
#         every eligible payload-colored pixel in the component/bbox.
#       - if payload appears on both sides, both sides get reflected.
#     variant_main:
#       - collect the same component, infer axis orientation, then inspect whether the red line is
#         solid or has exactly one internal gap.
#       - that pattern selects the source side only; payload on the other side is preserved but not copied.
#       - difference: the rule now depends on a symbol-like line pattern, not just on geometry.
#   complexity_calibration:
#     - only two code states are used: solid vs one-gap. This is enough to be learnable from 2-4 ARC
#       examples while still forcing non-trivial pattern recognition.
#     - every generated grid contains both states among line groups, so the mapping has repeated support.
#   implicit_assumption_audit:
#     old_code_dependencies:
#       - old row/col handling assumed that scanning all pixels of the same payload color within the
#         component bbox was harmless. That worked because original payload was effectively on one side.
#       - in this variant that would be wrong: the target-side stub shares the same color and must not
#         be copied. So line processing must be restricted to the marker component and the encoded source side.
#       - old code also assumed a red line was 4-connected as a single run. With a gap this can break,
#         so generation forces the source blob to bridge the two red segments through 4-connectivity.
#     new_solver_assumptions:
#       - a gapped line has exactly one internal missing cell, never an endpoint gap.
#       - the source blob stays within the axis span so source-side filtering by row/col span is valid.
#       - each marker-containing component has only one meaningful source side; generator guarantees this.
#   placement_legality:
#     - generation is patch-based. Each local patch already contains the full input/output relation.
#     - patches are placed using rectangle separation, so no object overlaps or cross-group accidental
#       adjacency can confuse 4-connected components or isolated-marker adjacency.
#     - output patches are validated by asserting main(input_grid) == output_grid before returning.
#   visual_observability:
#     - human-visible clue: some red lines are continuous, some have a single hole in the middle.
#     - human-visible effect: the big asymmetric blob is copied only according to that pattern, while
#       the tiny opposite-side stub is not mirrored back.
#   cleanliness:
#     - one payload color per group, limited group count, and sandboxed placement keep the grids legible.


MARK = 2
PALETTE = [0, 1, 3, 4, 5, 6, 7, 8, 9]


def _inb(H: int, W: int, r: int, c: int) -> bool:
    return 0 <= int(r) < int(H) and 0 <= int(c) < int(W)


def _refl_point(p: tuple[int, int], r: int, c: int) -> tuple[int, int]:
    pr, pc = map(int, p)
    return int(2 * pr - r), int(2 * pc - c)


def _refl_row(r0: int, r: int, c: int) -> tuple[int, int]:
    return int(2 * int(r0) - int(r)), int(c)


def _refl_col(c0: int, r: int, c: int) -> tuple[int, int]:
    return int(r), int(2 * int(c0) - int(c))


def _components(grid: np.ndarray, bg: int) -> list[dict]:
    lab, n = label_mask(grid != int(bg), connectivity=4)
    comps = []
    for k in range(1, int(n) + 1):
        coords = [(int(r), int(c)) for r, c in np.argwhere(lab == int(k))]
        if not coords:
            continue
        vals = [int(grid[r, c]) for r, c in coords]
        rs = [r for r, _c in coords]
        cs = [c for _r, c in coords]
        comps.append(
            {
                "coords": coords,
                "vals": vals,
                "bbox": (int(min(rs)), int(min(cs)), int(max(rs)), int(max(cs))),
            }
        )
    return comps


def _line_instruction(reds: list[tuple[int, int]]) -> tuple[str, int, str, tuple[int, int]] | None:
    if len(reds) < 2:
        return None

    rs = sorted({int(r) for r, _c in reds})
    cs = sorted({int(c) for _r, c in reds})

    if len(rs) == 1:
        row = int(rs[0])
        c0, c1 = int(min(cs)), int(max(cs))
        full = set(range(int(c0), int(c1) + 1))
        missing = sorted(full - {int(c) for _r, c in reds})
        if len(missing) == 0:
            return ("row", int(row), "down", (int(c0), int(c1)))
        if len(missing) == 1 and int(c0) < int(missing[0]) < int(c1):
            return ("row", int(row), "up", (int(c0), int(c1)))
        return None

    if len(cs) == 1:
        col = int(cs[0])
        r0, r1 = int(min(rs)), int(max(rs))
        full = set(range(int(r0), int(r1) + 1))
        missing = sorted(full - {int(r) for r, _c in reds})
        if len(missing) == 0:
            return ("col", int(col), "right", (int(r0), int(r1)))
        if len(missing) == 1 and int(r0) < int(missing[0]) < int(r1):
            return ("col", int(col), "left", (int(r0), int(r1)))
        return None

    return None


def _variant_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    comps = _components(g, bg)

    out = {
        "line_groups": 0,
        "solid_groups": 0,
        "gapped_groups": 0,
        "bilateral_groups": 0,
        "isolated_groups": 0,
    }
    for comp in comps:
        coords = comp["coords"]
        vals = comp["vals"]
        reds = [(int(r), int(c)) for (r, c), v in zip(coords, vals) if int(v) == int(MARK)]
        payload = [(int(r), int(c)) for (r, c), v in zip(coords, vals) if int(v) != int(MARK)]
        if len(coords) == 1 and vals[0] == MARK:
            out["isolated_groups"] += 1
            continue
        if len(reds) == 1 or not payload:
            continue
        info = _line_instruction(reds)
        if info is None:
            continue
        axis_kind, axis_pos, target_side, span = info
        out["line_groups"] += 1
        if target_side in {"right", "down"}:
            out["solid_groups"] += 1
        else:
            out["gapped_groups"] += 1
        if axis_kind == "col":
            left = any(c < axis_pos for r, c in payload if span[0] <= r <= span[1])
            right = any(c > axis_pos for r, c in payload if span[0] <= r <= span[1])
            if left and right:
                out["bilateral_groups"] += 1
        else:
            up = any(r < axis_pos for r, c in payload if span[0] <= c <= span[1])
            down = any(r > axis_pos for r, c in payload if span[0] <= c <= span[1])
            if up and down:
                out["bilateral_groups"] += 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape
    comps = _components(g, bg)
    out = g.copy()

    # 1) Marker-containing components: line markers use D10 code; single embedded red keeps point-reflection.
    for comp in comps:
        coords = comp["coords"]
        vals = comp["vals"]
        reds = [(int(r), int(c)) for (r, c), v in zip(coords, vals) if int(v) == int(MARK)]
        payload_coords = [(int(r), int(c), int(v)) for (r, c), v in zip(coords, vals) if int(v) != int(MARK)]
        if not reds or not payload_coords:
            continue

        if len(reds) == 1:
            pivot = (int(reds[0][0]), int(reds[0][1]))
            for r, c, v in payload_coords:
                rr, cc = _refl_point(pivot, int(r), int(c))
                if _inb(H, W, rr, cc):
                    out[int(rr), int(cc)] = int(v)
            continue

        info = _line_instruction(reds)
        if info is None:
            continue
        axis_kind, axis_pos, target_side, span = info

        src_pixels: list[tuple[int, int, int]] = []
        if axis_kind == "col":
            r0, r1 = span
            for r, c, v in payload_coords:
                if not (int(r0) <= int(r) <= int(r1)):
                    continue
                if target_side == "right" and int(c) < int(axis_pos):
                    src_pixels.append((int(r), int(c), int(v)))
                elif target_side == "left" and int(c) > int(axis_pos):
                    src_pixels.append((int(r), int(c), int(v)))
        else:
            c0, c1 = span
            for r, c, v in payload_coords:
                if not (int(c0) <= int(c) <= int(c1)):
                    continue
                if target_side == "down" and int(r) < int(axis_pos):
                    src_pixels.append((int(r), int(c), int(v)))
                elif target_side == "up" and int(r) > int(axis_pos):
                    src_pixels.append((int(r), int(c), int(v)))

        for r, c, v in src_pixels:
            if axis_kind == "col":
                rr, cc = _refl_col(int(axis_pos), int(r), int(c))
            else:
                rr, cc = _refl_row(int(axis_pos), int(r), int(c))
            if _inb(H, W, rr, cc):
                out[int(rr), int(cc)] = int(v)

    # 2) Isolated red pixels keep the original point-reflection rule.
    marker_pts: list[tuple[int, int]] = []
    for comp in comps:
        if len(comp["coords"]) == 1 and int(comp["vals"][0]) == int(MARK):
            marker_pts.append(comp["coords"][0])

    for pr, pc in marker_pts:
        pr = int(pr)
        pc = int(pc)
        neigh_colors: set[int] = set()
        neigh_pixels: list[tuple[int, int, int]] = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = int(pr + dr), int(pc + dc)
                if not _inb(H, W, rr, cc):
                    continue
                v = int(g[rr, cc])
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
                    if _inb(H, W, rr, cc):
                        out[int(rr), int(cc)] = int(v)

    return out


def _stamp(dst: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> None:
    h, w = patch.shape
    mask = patch != int(bg)
    sub = dst[int(top) : int(top) + int(h), int(left) : int(left) + int(w)]
    sub[mask] = patch[mask]


def _make_vertical_line_patch(source_left: bool, has_gap: bool, color: int, bg: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    L = int(rng.randint(4, 6))
    H = int(L + 2)
    W = 9
    axis = 4
    inp = np.full((H, W), int(bg), dtype=int)
    out = np.full((H, W), int(bg), dtype=int)
    line_rows = list(range(1, L + 1))

    source_low = bool(rng.randint(0, 1))
    if source_low:
        src_rows = line_rows[-3:]
        stub_rows = line_rows[: 1 + rng.randint(0, 1)]
    else:
        src_rows = line_rows[:3]
        stub_rows = line_rows[-(1 + rng.randint(0, 1)) :]

    gap_row = int(src_rows[1]) if has_gap else None
    for r in line_rows:
        if gap_row is not None and int(r) == int(gap_row):
            continue
        inp[int(r), int(axis)] = int(MARK)
        out[int(r), int(axis)] = int(MARK)

    s = -1 if source_left else 1
    t = -s
    template = int(rng.randint(0, 1))
    src_pixels: list[tuple[int, int]] = []
    for r in src_rows:
        src_pixels.append((int(r), int(axis + s * 1)))
    if template == 0:
        src_pixels.extend(
            [
                (int(src_rows[0]), int(axis + s * 2)),
                (int(src_rows[1]), int(axis + s * 2)),
                (int(src_rows[1]), int(axis + s * 3)),
            ]
        )
    else:
        src_pixels.extend(
            [
                (int(src_rows[1]), int(axis + s * 2)),
                (int(src_rows[2]), int(axis + s * 2)),
                (int(src_rows[1]), int(axis + s * 3)),
            ]
        )

    stub_pixels: list[tuple[int, int]] = []
    for r in stub_rows:
        stub_pixels.append((int(r), int(axis + t * 1)))
    if len(stub_rows) == 1 and rng.random() < 0.5:
        stub_pixels.append((int(stub_rows[0]), int(axis + t * 2)))

    src_pixels = sorted(set(src_pixels))
    stub_pixels = sorted(set(stub_pixels))
    for r, c in src_pixels + stub_pixels:
        inp[int(r), int(c)] = int(color)
        out[int(r), int(c)] = int(color)
    for r, c in src_pixels:
        rr, cc = _refl_col(int(axis), int(r), int(c))
        out[int(rr), int(cc)] = int(color)

    return inp, out


def _make_line_patch(orientation: str, code: str, color: int, bg: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    solid = code == "solid"
    if orientation == "col":
        return _make_vertical_line_patch(source_left=solid, has_gap=not solid, color=color, bg=bg, rng=rng)
    vinp, vout = _make_vertical_line_patch(source_left=solid, has_gap=not solid, color=color, bg=bg, rng=rng)
    return vinp.T.copy(), vout.T.copy()


def _make_isolated_patch(color: int, bg: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((7, 7), int(bg), dtype=int)
    out = np.full((7, 7), int(bg), dtype=int)
    pr, pc = 3, 3
    inp[pr, pc] = MARK
    out[pr, pc] = MARK

    templates = [
        [(-1, -1), (-2, -1), (-1, -2)],
        [(-1, -1), (-2, -1), (-2, -2), (-1, -2)],
        [(-1, -1), (-2, -1), (-3, -1), (-2, -2)],
        [(-1, -1), (-1, -2), (-2, -2), (-3, -2)],
    ]
    pts = templates[int(rng.randrange(len(templates)))]
    flip_r = -1 if rng.random() < 0.5 else 1
    flip_c = -1 if rng.random() < 0.5 else 1
    sprite = []
    for dr, dc in pts:
        r = int(pr + dr * flip_r)
        c = int(pc + dc * flip_c)
        sprite.append((int(r), int(c)))
    for r, c in sprite:
        inp[int(r), int(c)] = int(color)
        out[int(r), int(c)] = int(color)
    for r, c in sprite:
        rr, cc = _refl_point((pr, pc), int(r), int(c))
        out[int(rr), int(cc)] = int(color)
    return inp, out


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _attempt in range(300):
        H = int(rng.randint(20, 28))
        W = int(rng.randint(20, 28))
        bg = int(rng.choice([c for c in PALETTE if c != MARK]))
        inp = np.full((H, W), int(bg), dtype=int)
        expected = np.full((H, W), int(bg), dtype=int)
        rect_occ = np.zeros((H, W), dtype=bool)

        n_line = int(rng.randint(2, 4))
        n_iso = int(rng.randint(0, min(2, 6 - n_line)))
        codes = ["solid", "gap"]
        while len(codes) < n_line:
            codes.append("solid" if rng.random() < 0.5 else "gap")
        rng.shuffle(codes)
        orientations = [rng.choice(["row", "col"]) for _ in range(n_line)]

        colors = [c for c in PALETTE if c not in {bg, MARK}]
        rng.shuffle(colors)
        need = n_line + n_iso
        if len(colors) < need:
            continue

        patches: list[tuple[np.ndarray, np.ndarray]] = []
        for i in range(n_line):
            patches.append(_make_line_patch(orientations[i], codes[i], colors[i], bg, rng))
        for i in range(n_iso):
            patches.append(_make_isolated_patch(colors[n_line + i], bg, rng))
        rng.shuffle(patches)

        ok = True
        for pinp, pout in patches:
            ph, pw = map(int, pinp.shape)
            placed = False
            for _ in range(200):
                top = int(rng.randint(0, H - ph))
                left = int(rng.randint(0, W - pw))
                r0 = max(0, int(top - 1))
                r1 = min(H, int(top + ph + 1))
                c0 = max(0, int(left - 1))
                c1 = min(W, int(left + pw + 1))
                if np.any(rect_occ[r0:r1, c0:c1]):
                    continue
                _stamp(inp, pinp, top, left, bg)
                _stamp(expected, pout, top, left, bg)
                rect_occ[int(top) : int(top) + int(ph), int(left) : int(left) + int(pw)] = True
                placed = True
                break
            if not placed:
                ok = False
                break
        if not ok:
            continue

        produced = main(inp)
        if not np.array_equal(produced, expected):
            continue

        metrics = _variant_metrics(inp)
        if metrics["line_groups"] < 2:
            continue
        if metrics["solid_groups"] < 1 or metrics["gapped_groups"] < 1:
            continue
        if metrics["bilateral_groups"] < metrics["line_groups"]:
            continue
        return inp

    raise RuntimeError("failed to generate a valid D10 variant sample")
