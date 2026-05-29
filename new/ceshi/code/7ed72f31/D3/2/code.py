# Variant of 7ed72f31
# Dimension: D3 — payload精灵的连通性和结构 (src)
# Change: payload为与红色标记4-连通的单色精灵，3-11像素，单一非红非背景颜色... → 反射时颜色交换
#
# Design analysis
# yaml:
#   source_task:
#     id: a25697e4
#     solver_status: solver code not found in the provided local workspace, so only the
#       stated "color_swap"-style mechanism is borrowed; no extra structural ideas are imported.
#   concretization:
#     core_change:
#       - Every reflected copy changes color via a fixed task-level mapping instead of keeping
#         the payload's original color.
#       - The geometric part of the task stays the same: red row -> row reflection, red column
#         -> column reflection, isolated red pixel -> point reflection of the adjacent 8-connected sprite.
#     color_map:
#       - Use a small, fixed set of directed swap pairs: 1->8, 3->4, 5->6, 7->9.
#       - The generator only uses the source side of the pair inside the input, so the reflected
#         copy is always visibly different and never ambiguous with the source payload.
#       - If the chosen background would collide with one pair, that pair is simply not used.
#     behavior_trace:
#       - Original row case:
#         input has a red horizontal line with a monochrome payload on one side; solver reflects
#         the payload and paints the mirror in the same color.
#       - Variant row case:
#         the same reflected coordinates are produced, but every mirrored pixel is painted with
#         the mapped color instead of the source color.
#       - Example:
#         source color 3 under a red row becomes color 4 above the row. The red axis and original
#         green payload remain unchanged; only the copied half changes color.
#   complexity_calibration:
#     - The color mapping uses only 2-3 source colors per grid, repeated across 4-6 groups.
#       This is enough evidence for a human to infer the mapping without exploding the color count.
#     - The generator preserves multiple marker types (row / col / isolated) so the task is still
#       recognizably the original reflection task rather than just a recoloring puzzle.
#   implicit_assumptions_review:
#     original_solver_lines_of_interest:
#       - The original marker-component branch used a payload-color set plus a bbox-based global scan.
#       - That was harmless when payload colors were effectively unique per group.
#       - In this variant, repeated source colors are intentionally necessary to make the swap table
#         visible. A global color scan would then mirror unrelated same-color groups.
#     adaptation:
#       - The new solver reflects only the payload pixels that belong to the marker-containing
#         connected component itself.
#       - The isolated-red branch can still identify the touching 8-connected component by label,
#         so repeated colors remain safe there as long as placement keeps components separate.
#   placement_legality:
#     - All groups are built output-first, then the mirrored half is removed to form the input.
#     - Different groups are placed with a 1-cell Chebyshev moat, preventing accidental merging,
#       accidental 8-neighbor contact to isolated markers, overlap, and out-of-bounds reflections.
#     - Isolated red markers only touch their source sprite diagonally, never orthogonally.
#   visual_observability:
#     - Monochrome payloads remain acceptable because the reflection geometry is still visible from
#       position, while the recolor is visible from the consistent source->target color change.
#     - Each grid repeats source colors across several groups, so the color-change rule is not a
#       one-off coincidence.
#   cleanliness:
#     - The generator does not add the original task's optional stray same-color sprites; they would
#       add visual clutter and are unnecessary for this D3 change.
#   new_solver_assumptions_and_generator_guarantees:
#     - Assumption: every reflected payload color is present in the fixed swap map.
#       Guarantee: generator only chooses source colors from usable swap pairs.
#     - Assumption: marker-containing components are unambiguous row / col / point markers.
#       Guarantee: generator builds only those exact shapes.
#     - Assumption: isolated red markers have exactly one adjacent 8-connected source component.
#       Guarantee: generator places isolated groups with a moat and diagonal-only contact.

from __future__ import annotations

import random
from typing import Iterable

import numpy as np

from arc_utils import get_bg, label_mask

MARK = 2
SWAP_PAIRS = ((1, 8), (3, 4), (5, 6), (7, 9))
COLOR_MAP = {a: b for a, b in SWAP_PAIRS}
COLOR_MAP.update({b: a for a, b in SWAP_PAIRS})


def swap_color(v: int) -> int:
    return int(COLOR_MAP.get(int(v), int(v)))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror colored sprites using red markers as axes / centers, but recolor the mirrored copy.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape
    out = g.copy()

    def inb(r: int, c: int) -> bool:
        return 0 <= int(r) < H and 0 <= int(c) < W

    def refl_point(center: tuple[int, int], r: int, c: int) -> tuple[int, int]:
        cr, cc = map(int, center)
        return int(2 * cr - r), int(2 * cc - c)

    def refl_row(r0: int, r: int, c: int) -> tuple[int, int]:
        return int(2 * int(r0) - int(r)), int(c)

    def refl_col(c0: int, r: int, c: int) -> tuple[int, int]:
        return int(r), int(2 * int(c0) - int(c))

    lab, n = label_mask(g != int(bg), connectivity=4)
    comps: list[dict[str, object]] = []
    for k in range(1, int(n) + 1):
        coords = [(int(r), int(c)) for r, c in np.argwhere(lab == int(k))]
        if not coords:
            continue
        vals = [int(g[r, c]) for r, c in coords]
        comps.append({"coords": coords, "vals": vals})

    # 1) Marker embedded in a mixed component: reflect only that component's payload pixels.
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

        for r, c, v in payload:
            if axis[0] == "row":
                rr, cc = refl_row(int(axis[1]), int(r), int(c))
            elif axis[0] == "col":
                rr, cc = refl_col(int(axis[1]), int(r), int(c))
            else:
                rr, cc = refl_point(axis[1], int(r), int(c))
            if inb(int(rr), int(cc)):
                out[int(rr), int(cc)] = swap_color(int(v))

    # 2) Isolated red pixel: reflect the adjacent 8-connected component, then recolor it.
    for comp in comps:
        if len(comp["coords"]) != 1 or int(comp["vals"][0]) != int(MARK):
            continue
        pr, pc = map(int, comp["coords"][0])
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

        for v in neigh_colors:
            labv, _ = label_mask(g == int(v), connectivity=8)
            labels = {
                int(labv[int(rr), int(cc)])
                for rr, cc, vv in neigh_pixels
                if int(vv) == int(v) and int(labv[int(rr), int(cc)]) != 0
            }
            for lbl in labels:
                for r, c in np.argwhere(labv == int(lbl)):
                    rr, cc = refl_point((int(pr), int(pc)), int(r), int(c))
                    if inb(int(rr), int(cc)):
                        out[int(rr), int(cc)] = swap_color(int(v))

    return out


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
    tries: int = 200,
) -> set[tuple[int, int]] | None:
    neigh = _neighbors4 if int(connectivity) == 4 else _neighbors8
    allowed = set((int(r), int(c)) for r, c in allowed)
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
    out_cells: dict[tuple[int, int], int],
    in_cells: dict[tuple[int, int], int],
    meta: dict[str, object],
) -> dict[str, object]:
    pts = list(out_cells)
    min_r = min(int(r) for r, _c in pts)
    min_c = min(int(c) for _r, c in pts)

    out_norm = {(int(r - min_r), int(c - min_c)): int(v) for (r, c), v in out_cells.items()}
    in_norm = {(int(r - min_r), int(c - min_c)): int(v) for (r, c), v in in_cells.items()}
    h = 1 + max(int(r) for r, _c in out_norm)
    w = 1 + max(int(c) for _r, c in out_norm)
    meta = dict(meta)
    meta["height"] = int(h)
    meta["width"] = int(w)
    return {"out": out_norm, "in": in_norm, "h": int(h), "w": int(w), "meta": meta}


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
        out_cells: dict[tuple[int, int], int] = {}
        in_cells: dict[tuple[int, int], int] = {}
        for c in range(line_len):
            out_cells[(0, int(c))] = int(MARK)
            in_cells[(0, int(c))] = int(MARK)

        for r, c in payload:
            rr = int(side * (int(r) + 1))
            cc = int(c)
            out_cells[(rr, cc)] = int(src)
            in_cells[(rr, cc)] = int(src)
            out_cells[(-rr, cc)] = int(dst)

        return _normalize_group(
            out_cells,
            in_cells,
            {"type": "row", "source": int(src), "target": int(dst), "mirror_pixels": int(len(payload))},
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
        out_cells: dict[tuple[int, int], int] = {}
        in_cells: dict[tuple[int, int], int] = {}
        for r in range(line_len):
            out_cells[(int(r), 0)] = int(MARK)
            in_cells[(int(r), 0)] = int(MARK)

        for r, c in payload:
            rr = int(r)
            cc = int(side * (int(c) + 1))
            out_cells[(rr, cc)] = int(src)
            in_cells[(rr, cc)] = int(src)
            out_cells[(rr, -cc)] = int(dst)

        return _normalize_group(
            out_cells,
            in_cells,
            {"type": "col", "source": int(src), "target": int(dst), "mirror_pixels": int(len(payload))},
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
        out_cells: dict[tuple[int, int], int] = {(0, 0): int(MARK)}
        in_cells: dict[tuple[int, int], int] = {(0, 0): int(MARK)}
        for r, c in payload:
            rr = int(sr * (int(r) + 1))
            cc = int(sc * (int(c) + 1))
            out_cells[(rr, cc)] = int(src)
            in_cells[(rr, cc)] = int(src)
            out_cells[(-rr, -cc)] = int(dst)

        return _normalize_group(
            out_cells,
            in_cells,
            {"type": "iso", "source": int(src), "target": int(dst), "mirror_pixels": int(len(payload))},
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


def _can_place(
    group: dict[str, object],
    top: int,
    left: int,
    H: int,
    W: int,
    forbidden: set[tuple[int, int]],
) -> bool:
    for r, c in group["out"]:
        rr = int(top + r)
        cc = int(left + c)
        if not (0 <= rr < int(H) and 0 <= cc < int(W)):
            return False
        if (int(rr), int(cc)) in forbidden:
            return False
    return True


def _stamp_moat(forbidden: set[tuple[int, int]], cells: Iterable[tuple[int, int]]) -> None:
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                forbidden.add((int(r + dr), int(c + dc)))


def generate_case(rng: random.Random | None = None) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    rng = rng or random.Random()

    for _ in range(400):
        H = rng.randint(20, 28)
        W = rng.randint(20, 28)
        bg = rng.choice([c for c in range(10) if c != int(MARK)])

        usable_pairs = [(int(src), int(dst)) for src, dst in SWAP_PAIRS if bg not in (int(src), int(dst))]
        if len(usable_pairs) < 2:
            continue

        n_sources = rng.choice([2, 3] if len(usable_pairs) >= 3 else [2])
        chosen_pairs = rng.sample(usable_pairs, n_sources)
        source_to_target = {int(src): int(dst) for src, dst in chosen_pairs}

        groups_n = rng.randint(max(4, int(n_sources) + 1), 6)
        iso_n = rng.randint(0, min(2, int(groups_n) - 2))
        kinds = ["row", "col"] + [rng.choice(["row", "col"]) for _ in range(groups_n - 2 - iso_n)] + ["iso"] * iso_n
        rng.shuffle(kinds)

        colors = [int(src) for src, _dst in chosen_pairs]
        while len(colors) < int(groups_n):
            colors.append(rng.choice([int(src) for src, _dst in chosen_pairs]))
        rng.shuffle(colors)

        groups = []
        try:
            for kind, src in zip(kinds, colors):
                groups.append(_build_group(kind, int(src), int(source_to_target[int(src)]), rng))
        except RuntimeError:
            continue
        groups.sort(key=lambda g: len(g["out"]), reverse=True)

        inp = np.full((H, W), int(bg), dtype=int)
        out = np.full((H, W), int(bg), dtype=int)
        forbidden: set[tuple[int, int]] = set()
        placements: list[dict[str, object]] = []
        ok = True

        for group in groups:
            placed = False
            gh = int(group["h"])
            gw = int(group["w"])
            for _attempt in range(600):
                top = rng.randint(0, H - gh)
                left = rng.randint(0, W - gw)
                if not _can_place(group, top, left, H, W, forbidden):
                    continue

                out_cells_abs = []
                for (r, c), v in group["out"].items():
                    rr = int(top + r)
                    cc = int(left + c)
                    out[rr, cc] = int(v)
                    out_cells_abs.append((rr, cc))
                for (r, c), v in group["in"].items():
                    inp[int(top + r), int(left + c)] = int(v)

                _stamp_moat(forbidden, out_cells_abs)
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

        recolored = sum(int(p["mirror_pixels"]) for p in placements)
        meta = {
            "bg": int(bg),
            "grid_shape": (int(H), int(W)),
            "placements": placements,
            "group_types": [str(p["type"]) for p in placements],
            "sources": [int(p["source"]) for p in placements],
            "targets": [int(p["target"]) for p in placements],
            "recolored_pixels": int(recolored),
        }
        return inp, out, meta

    raise RuntimeError("failed to generate a valid recolor-reflection case")


def generate_input() -> np.ndarray:
    inp, _out, _meta = generate_case()
    return inp
