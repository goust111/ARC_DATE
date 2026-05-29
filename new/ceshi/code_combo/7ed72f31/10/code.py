# Combo variant of 7ed72f31 (index 10)
# combo_spec: D3/2+D9/0
# Name: 变色传送
#
# Design analysis
# ===============
# yaml:
#   concretization_and_behavior_trace:
#     fusion_rule:
#       - Red geometry still determines the transform: red row -> row reflection, red column -> column
#         reflection, isolated red point -> point reflection of the adjacent 8-connected sprite.
#       - The reflected copy is recolored through a fixed task-level directed map.
#       - The source payload is then erased, while the red marker remains.
#     example_trace:
#       - Input: a blue (1) L-shape sits below a red horizontal line.
#       - Original 7ed72f31: the same blue L is copied above the line; the original blue L stays.
#       - D3-only variant: a teal (8) L is copied above the line; the original blue L stays.
#       - D9-only variant: a blue L appears above the line; the original blue L is erased.
#       - This combo: a teal (8) L appears above the line, the original blue (1) L is erased,
#         and the red axis stays. The sprite visually "teleports" across the axis and changes color.
#   complexity_calibration:
#     - Use a small fixed directed map: 1->8, 3->4, 5->6, 7->9.
#     - Each grid uses only 2 or 3 source colors, and each chosen source color is repeated across
#       multiple groups so the recolor rule is inferable rather than a one-off coincidence.
#     - Keep several marker geometries in play in the same grid (always row + col, usually isolated too),
#       so the task remains a reflection puzzle with a color/deletion twist, not a pure recoloring task.
#   implicit_assumption_review:
#     original_solver_risk_1:
#       code_pattern: scan the whole grid for same-color pixels and filter by bbox.
#       broken_by_fusion: repeated source colors are now necessary for mapping evidence, and deletion
#         would erase unrelated same-color sprites if the scan stayed global.
#       adaptation: only reflect payload pixels belonging to the marker-bearing component itself; for
#         isolated markers, label the actually touched 8-connected component only.
#     original_solver_risk_2:
#       code_pattern: write mirrored pixels directly into out=g.copy().
#       broken_by_fusion: move semantics require erasing the source payload; interleaving erase/paint
#         can create order-dependent bugs.
#       adaptation: collect source pixels in erase, mirrored recolored pixels in paint, then apply
#         erase first and paint second.
#     generator_risk:
#       - In the source task, output footprint matched the full local object. Here output omits the
#         source half, so placement must reserve the union of input and output footprints.
#       adaptation: normalize and place each group by the union of source, red marker, and reflected target.
#   placement_legality:
#     - Build each group as a local template containing source payload, red marker, and reflected target.
#     - Normalize the template by the union footprint, then place it with a 1-cell Chebyshev moat.
#     - This guarantees no overlap, no accidental merges, no out-of-bounds reflections, and no isolated
#       red marker touching extra components in its 8-neighborhood.
#   visual_observability:
#     - Payloads are small irregular connected shapes, not symmetric blocks, so the move across the axis
#       is visible.
#     - Every chosen source color reappears in multiple groups, making the source->target color map visible.
#     - Output shows a stable cue combination: source side cleared, destination side filled, and destination
#       color systematically differs from the source color.
#   cleanliness:
#     - No stray same-color distractor sprites are added; with deletion semantics they would create
#       low-signal clutter and could obscure which pixels were truly moved.
#   inferability_of_change:
#     - Only a few mapping pairs are active, and they are repeated.
#     - The color change is always tied to a visible geometric event: the reflected landing position.
#       A human can align input source shapes with output destination shapes and infer both motion and recolor.
#   new_solver_assumptions_and_generator_guarantees:
#     - Assumption: payload colors used as sources are keys in the directed map.
#       Guarantee: generator only samples source colors from the map domain.
#     - Assumption: mixed marker components form an unambiguous row / col / single-point marker shape.
#       Guarantee: generator constructs only those forms.
#     - Assumption: isolated red markers touch exactly one 8-connected source component diagonally.
#       Guarantee: generator uses a moat and diagonal-only placement for isolated groups.
#     - Assumption: reflected targets land on background rather than on another object.
#       Guarantee: placement reserves the full source+marker+target footprint before stamping a group.

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import get_bg, label_mask


MARK = 2
SWAP_PAIRS = ((1, 8), (3, 4), (5, 6), (7, 9))
COLOR_MAP = {int(src): int(dst) for src, dst in SWAP_PAIRS}
Coord = tuple[int, int]


def swap_color(v: int) -> int:
    return int(COLOR_MAP.get(int(v), int(v)))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reflect payloads across red markers, recolor the reflected copy, and delete the source payload.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    h, w = g.shape
    out = g.copy()

    def inb(r: int, c: int) -> bool:
        return 0 <= int(r) < h and 0 <= int(c) < w

    def refl_point(center: Coord, r: int, c: int) -> Coord:
        cr, cc = center
        return int(2 * cr - r), int(2 * cc - c)

    def refl_row(r0: int, r: int, c: int) -> Coord:
        return int(2 * int(r0) - int(r)), int(c)

    def refl_col(c0: int, r: int, c: int) -> Coord:
        return int(r), int(2 * int(c0) - int(c))

    lab, n = label_mask(g != int(bg), connectivity=4)
    comps: list[dict[str, object]] = []
    for k in range(1, int(n) + 1):
        coords = [(int(r), int(c)) for r, c in np.argwhere(lab == int(k))]
        if not coords:
            continue
        vals = [int(g[r, c]) for r, c in coords]
        comps.append({"coords": coords, "vals": vals})

    erase: set[Coord] = set()
    paint: list[tuple[int, int, int]] = []

    # 1) Mixed components containing red markers: move only that component's payload.
    for comp in comps:
        coords = comp["coords"]
        vals = comp["vals"]
        if int(MARK) not in vals:
            continue
        if len(coords) == 1 and int(vals[0]) == int(MARK):
            continue

        reds = [(int(r), int(c)) for (r, c), v in zip(coords, vals) if int(v) == int(MARK)]
        payload = [(int(r), int(c), int(v)) for (r, c), v in zip(coords, vals) if int(v) != int(MARK)]
        if not reds or not payload:
            continue

        rows = {int(r) for r, _c in reds}
        cols = {int(c) for _r, c in reds}
        if len(reds) == 1:
            axis = ("point", (int(reds[0][0]), int(reds[0][1])))
        elif len(rows) == 1:
            axis = ("row", int(next(iter(rows))))
        elif len(cols) == 1:
            axis = ("col", int(next(iter(cols))))
        else:
            continue

        for r, c, v in payload:
            if axis[0] == "row":
                rr, cc = refl_row(int(axis[1]), int(r), int(c))
            elif axis[0] == "col":
                rr, cc = refl_col(int(axis[1]), int(r), int(c))
            else:
                rr, cc = refl_point(axis[1], int(r), int(c))
            erase.add((int(r), int(c)))
            if inb(int(rr), int(cc)):
                paint.append((int(rr), int(cc), swap_color(int(v))))

    # 2) Isolated red pixels: move the adjacent 8-connected component and recolor the landing copy.
    for comp in comps:
        coords = comp["coords"]
        vals = comp["vals"]
        if len(coords) != 1 or int(vals[0]) != int(MARK):
            continue

        pr, pc = map(int, coords[0])
        neigh_pixels: list[tuple[int, int, int]] = []
        neigh_colors: set[int] = set()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = int(pr + dr), int(pc + dc)
                if not inb(rr, cc):
                    continue
                v = int(g[rr, cc])
                if v == int(bg) or v == int(MARK):
                    continue
                neigh_pixels.append((int(rr), int(cc), int(v)))
                neigh_colors.add(int(v))

        for color in neigh_colors:
            labv, _ = label_mask(g == int(color), connectivity=8)
            labels = {
                int(labv[int(rr), int(cc)])
                for rr, cc, vv in neigh_pixels
                if int(vv) == int(color) and int(labv[int(rr), int(cc)]) != 0
            }
            for lbl in labels:
                for r, c in np.argwhere(labv == int(lbl)):
                    erase.add((int(r), int(c)))
                    rr, cc = refl_point((int(pr), int(pc)), int(r), int(c))
                    if inb(int(rr), int(cc)):
                        paint.append((int(rr), int(cc), swap_color(int(color))))

    for r, c in erase:
        if int(out[r, c]) != int(MARK):
            out[int(r), int(c)] = int(bg)
    for r, c, v in paint:
        out[int(r), int(c)] = int(v)
    return out


def _neighbors4(cell: Coord) -> Iterable[Coord]:
    r, c = map(int, cell)
    yield int(r - 1), int(c)
    yield int(r + 1), int(c)
    yield int(r), int(c - 1)
    yield int(r), int(c + 1)


def _neighbors8(cell: Coord) -> Iterable[Coord]:
    r, c = map(int, cell)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield int(r + dr), int(c + dc)


def _grow_connected(
    area: int,
    allowed: set[Coord],
    seed: Coord,
    *,
    connectivity: int,
    rng: random.Random,
    tries: int = 200,
) -> set[Coord] | None:
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


def _normalize_group(
    in_cells: dict[Coord, int],
    out_cells: dict[Coord, int],
    meta: dict[str, object],
) -> dict[str, object]:
    footprint = set(in_cells) | set(out_cells)
    min_r = min(int(r) for r, _c in footprint)
    min_c = min(int(c) for _r, c in footprint)

    def shift(cells: dict[Coord, int]) -> dict[Coord, int]:
        return {(int(r - min_r), int(c - min_c)): int(v) for (r, c), v in cells.items()}

    in_norm = shift(in_cells)
    out_norm = shift(out_cells)
    footprint_norm = {(int(r - min_r), int(c - min_c)) for r, c in footprint}
    h = 1 + max(int(r) for r, _c in footprint_norm)
    w = 1 + max(int(c) for _r, c in footprint_norm)
    meta = dict(meta)
    meta["height"] = int(h)
    meta["width"] = int(w)
    return {
        "in": in_norm,
        "out": out_norm,
        "footprint": footprint_norm,
        "h": int(h),
        "w": int(w),
        "meta": meta,
    }


def _build_row_group(src: int, dst: int, rng: random.Random) -> dict[str, object]:
    for _ in range(200):
        line_len = rng.randint(2, 6)
        max_depth = rng.randint(2, 5)
        area = rng.randint(3, min(11, int(line_len * max_depth)))
        allowed = {(int(r), int(c)) for r in range(max_depth) for c in range(line_len)}
        seed = (0, rng.randrange(line_len))
        payload = _grow_connected(area, allowed, seed, connectivity=4, rng=rng)
        if payload is None:
            continue

        side = rng.choice((-1, 1))
        in_cells: dict[Coord, int] = {}
        out_cells: dict[Coord, int] = {}
        for c in range(line_len):
            in_cells[(0, int(c))] = int(MARK)
            out_cells[(0, int(c))] = int(MARK)

        for r, c in payload:
            rr = int(side * (int(r) + 1))
            cc = int(c)
            in_cells[(rr, cc)] = int(src)
            out_cells[(-rr, cc)] = int(dst)

        return _normalize_group(
            in_cells,
            out_cells,
            {"type": "row", "source": int(src), "target": int(dst), "moved_pixels": int(len(payload))},
        )
    raise RuntimeError("failed to build row group")


def _build_col_group(src: int, dst: int, rng: random.Random) -> dict[str, object]:
    for _ in range(200):
        line_len = rng.randint(2, 6)
        max_depth = rng.randint(2, 5)
        area = rng.randint(3, min(11, int(line_len * max_depth)))
        allowed = {(int(r), int(c)) for r in range(line_len) for c in range(max_depth)}
        seed = (rng.randrange(line_len), 0)
        payload = _grow_connected(area, allowed, seed, connectivity=4, rng=rng)
        if payload is None:
            continue

        side = rng.choice((-1, 1))
        in_cells: dict[Coord, int] = {}
        out_cells: dict[Coord, int] = {}
        for r in range(line_len):
            in_cells[(int(r), 0)] = int(MARK)
            out_cells[(int(r), 0)] = int(MARK)

        for r, c in payload:
            rr = int(r)
            cc = int(side * (int(c) + 1))
            in_cells[(rr, cc)] = int(src)
            out_cells[(rr, -cc)] = int(dst)

        return _normalize_group(
            in_cells,
            out_cells,
            {"type": "col", "source": int(src), "target": int(dst), "moved_pixels": int(len(payload))},
        )
    raise RuntimeError("failed to build col group")


def _build_iso_group(src: int, dst: int, rng: random.Random) -> dict[str, object]:
    for _ in range(200):
        max_h = rng.randint(2, 4)
        max_w = rng.randint(2, 4)
        area = rng.randint(1, min(8, int(max_h * max_w)))
        allowed = {(int(r), int(c)) for r in range(max_h) for c in range(max_w)}
        payload = _grow_connected(area, allowed, (0, 0), connectivity=8, rng=rng)
        if payload is None:
            continue

        sr, sc = rng.choice(((-1, -1), (-1, 1), (1, -1), (1, 1)))
        in_cells: dict[Coord, int] = {(0, 0): int(MARK)}
        out_cells: dict[Coord, int] = {(0, 0): int(MARK)}
        for r, c in payload:
            rr = int(sr * (int(r) + 1))
            cc = int(sc * (int(c) + 1))
            in_cells[(rr, cc)] = int(src)
            out_cells[(-rr, -cc)] = int(dst)

        return _normalize_group(
            in_cells,
            out_cells,
            {"type": "iso", "source": int(src), "target": int(dst), "moved_pixels": int(len(payload))},
        )
    raise RuntimeError("failed to build isolated group")


def _build_group(kind: str, src: int, dst: int, rng: random.Random) -> dict[str, object]:
    if kind == "row":
        return _build_row_group(int(src), int(dst), rng)
    if kind == "col":
        return _build_col_group(int(src), int(dst), rng)
    if kind == "iso":
        return _build_iso_group(int(src), int(dst), rng)
    raise ValueError(f"unknown group kind: {kind}")


def _can_place(group: dict[str, object], top: int, left: int, h: int, w: int, forbidden: set[Coord]) -> bool:
    for r, c in group["footprint"]:
        rr = int(top + r)
        cc = int(left + c)
        if not (0 <= rr < int(h) and 0 <= cc < int(w)):
            return False
        if (int(rr), int(cc)) in forbidden:
            return False
    return True


def _stamp_moat(forbidden: set[Coord], cells: Iterable[Coord]) -> None:
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                forbidden.add((int(r + dr), int(c + dc)))


def generate_case(rng: random.Random | None = None) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    rng = rng or random.Random()

    for _ in range(400):
        h = rng.randint(20, 28)
        w = rng.randint(20, 28)
        bg = rng.choice([c for c in range(10) if c != int(MARK)])

        usable_pairs = [(int(src), int(dst)) for src, dst in SWAP_PAIRS if bg not in (int(src), int(dst))]
        if len(usable_pairs) < 2:
            continue

        groups_n = rng.randint(4, 6)
        if int(groups_n) == 6 and len(usable_pairs) >= 3 and rng.random() < 0.5:
            n_sources = 3
        else:
            n_sources = 2
        chosen_pairs = rng.sample(usable_pairs, n_sources)
        source_to_target = {int(src): int(dst) for src, dst in chosen_pairs}
        source_colors = [int(src) for src, _dst in chosen_pairs]

        iso_n = 1 if int(groups_n) == 4 else rng.randint(1, 2)
        kinds = ["row", "col"] + ["iso"] * int(iso_n)
        while len(kinds) < int(groups_n):
            kinds.append(rng.choice(["row", "col"]))
        rng.shuffle(kinds)

        colors: list[int] = []
        for src in source_colors:
            colors.extend([int(src), int(src)])
        while len(colors) < int(groups_n):
            colors.append(rng.choice(source_colors))
        rng.shuffle(colors)
        colors = colors[: int(groups_n)]

        groups = []
        try:
            for kind, src in zip(kinds, colors):
                groups.append(_build_group(kind, int(src), int(source_to_target[int(src)]), rng))
        except RuntimeError:
            continue
        groups.sort(key=lambda g: len(g["footprint"]), reverse=True)

        inp = np.full((h, w), int(bg), dtype=int)
        out = np.full((h, w), int(bg), dtype=int)
        forbidden: set[Coord] = set()
        placements: list[dict[str, object]] = []
        ok = True

        for group in groups:
            placed = False
            gh = int(group["h"])
            gw = int(group["w"])
            for _attempt in range(600):
                top = rng.randint(0, h - gh)
                left = rng.randint(0, w - gw)
                if not _can_place(group, top, left, h, w, forbidden):
                    continue

                footprint_abs = []
                for r, c in group["footprint"]:
                    footprint_abs.append((int(top + r), int(left + c)))
                for (r, c), v in group["in"].items():
                    inp[int(top + r), int(left + c)] = int(v)
                for (r, c), v in group["out"].items():
                    out[int(top + r), int(left + c)] = int(v)

                _stamp_moat(forbidden, footprint_abs)
                meta = dict(group["meta"])
                meta["top"] = int(top)
                meta["left"] = int(left)
                placements.append(meta)
                placed = True
                break

            if not placed:
                ok = False
                break

        if not ok:
            continue

        solved = main(inp)
        if not np.array_equal(solved, out):
            continue

        moved_pixels = sum(int(p["moved_pixels"]) for p in placements)
        meta = {
            "bg": int(bg),
            "grid_shape": (int(h), int(w)),
            "placements": placements,
            "group_types": [str(p["type"]) for p in placements],
            "sources": [int(p["source"]) for p in placements],
            "targets": [int(p["target"]) for p in placements],
            "moved_pixels": int(moved_pixels),
            "recolored_pixels": int(moved_pixels),
            "deletion_pixels": int(moved_pixels),
        }
        return inp, out, meta

    raise RuntimeError("failed to generate a valid combo case")


def generate_input() -> np.ndarray:
    inp, _out, _meta = generate_case()
    return inp
