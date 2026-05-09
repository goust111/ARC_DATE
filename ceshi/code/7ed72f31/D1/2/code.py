# Variant of 7ed72f31
# Dimension: D1 — 反射变换类型 (transform)
# Change: 根据红色标记几何形状选择镜像方式：红线→轴反射，单红像素→点反射... → 对角线反射

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import get_bg, label_mask


# Design analysis for the diagonal-reflection variant
# ==================================================
# a) Concrete behavior and trace
#    - Original line markers were horizontal/vertical and the solver chose row/col
#      reflection from "all red pixels share one row/col".
#    - In this variant, every generated line marker is a 45-degree diagonal:
#      either "\" (all red pixels satisfy r-c = const) or "/" (all red pixels
#      satisfy r+c = const). The payload is a monochrome 4-connected sprite
#      attached to one side of that diagonal. Output keeps the original and paints
#      its reflected copy on the opposite side.
#    - Small trace: if a "\" line is at (4,4),(5,5),(6,6) and a green payload
#      pixel is at (5,4), original code cannot use row/col reflection. Variant
#      code detects r-c=0 and maps (5,4)->(4,5), i.e. swap coordinates around the
#      diagonal. For "/" with invariant r+c=10, (4,5)->(5,4) because reflection is
#      (r,c)->(10-c,10-r).
# b) Complexity calibration
#    - Only two diagonal orientations are allowed. This keeps the concept small
#      enough for ARC while still forcing non-orthogonal reasoning.
#    - Payloads are 4-8 pixels, single-color, and intentionally asymmetric enough
#      to make the diagonal mirror visible. Marker lengths are 3-6 so the slanted
#      axis is obvious by eye.
# c) Implicit-assumption audit of the original main()
#    - `label_mask(..., connectivity=4)` relied on axis-aligned red lines being
#      4-connected. That breaks for diagonals, so mixed components must be labeled
#      with connectivity=8.
#    - `len(rs)==1` / `len(cs)==1` assumed every multi-pixel red marker was a row
#      or column. Variant adds checks for contiguous diagonals using constant
#      `r-c` or constant `r+c`.
#    - The original bbox-based "eligible" scan was harmless for orthogonal lines
#      but becomes too loose for slanted axes because a square bbox around a
#      diagonal captures unrelated same-color pixels more easily. Variant reflects
#      only the non-red pixels that belong to the same mixed component.
# d) Placement legality
#    - Generator constructs the full output group first: diagonal red line +
#      payload + mirrored payload. It only accepts groups whose mirrored copy stays
#      in-bounds, does not overlap the line/original, and remains on the opposite
#      side of the diagonal.
#    - Global placement uses a 1-cell moat around every output group so different
#      8-connected groups cannot merge accidentally.
# e) Visual observability
#    - Each grid contains 3-5 diagonal groups and both "\" and "/" appear when
#      there is room. Seeing both orientations across examples makes the rule
#      "reflect across the red diagonal" inferable instead of looking like an
#      arbitrary copy.
# f) Visual cleanliness
#    - No free noise sprites are generated. The original task allowed them, but
#      with diagonal bboxes they would make the scene needlessly messy and add no
#      information about D1.
# g) Color inference
#    - Colors do not encode logic. Each group uses one unique non-red color so the
#      only changing factor is geometry of the red diagonal.
# h) New solver assumptions and how the generator guarantees them
#    - Assumption: multi-pixel red markers are contiguous straight diagonals.
#      Generator builds them explicitly as consecutive lattice points.
#    - Assumption: the payload belonging to a diagonal marker is entirely in the
#      same 8-connected mixed component. Generator starts payload growth from a
#      cell orthogonally adjacent to the line.
#    - Assumption: separate groups are not 8-touching. Generator enforces a moat.

MARK = 2


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _classify_marker(reds: list[tuple[int, int]]) -> tuple[str, int | tuple[int, int]] | None:
    reds = sorted({(int(r), int(c)) for r, c in reds})
    if not reds:
        return None
    if len(reds) == 1:
        return ("point", (int(reds[0][0]), int(reds[0][1])))

    rs = {int(r) for r, _c in reds}
    cs = {int(c) for _r, c in reds}

    if len(rs) == 1:
        c0 = min(c for _r, c in reds)
        row = int(next(iter(rs)))
        expected = {(row, c0 + i) for i in range(len(reds))}
        if set(reds) == expected:
            return ("row", row)

    if len(cs) == 1:
        r0 = min(r for r, _c in reds)
        col = int(next(iter(cs)))
        expected = {(r0 + i, col) for i in range(len(reds))}
        if set(reds) == expected:
            return ("col", col)

    d0 = int(reds[0][0] - reds[0][1])
    if all(int(r - c) == d0 for r, c in reds):
        r0 = min(r for r, _c in reds)
        c0 = min(c for _r, c in reds)
        expected = {(r0 + i, c0 + i) for i in range(len(reds))}
        if set(reds) == expected:
            return ("diag_main", d0)

    s0 = int(reds[0][0] + reds[0][1])
    if all(int(r + c) == s0 for r, c in reds):
        r0 = min(r for r, _c in reds)
        c0 = max(c for _r, c in reds)
        expected = {(r0 + i, c0 - i) for i in range(len(reds))}
        if set(reds) == expected:
            return ("diag_anti", s0)

    return None


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape

    # Diagonal red lines are only 8-connected, so 4-connectivity would split the
    # marker into unrelated single red pixels and make the old logic unusable.
    lab, n = label_mask(g != int(bg), connectivity=8)
    comps: list[dict[str, object]] = []
    for k in range(1, int(n) + 1):
        coords = [(int(r), int(c)) for r, c in np.argwhere(lab == int(k))]
        if not coords:
            continue
        vals = [int(g[r, c]) for r, c in coords]
        comps.append({"coords": coords, "vals": vals})

    out = g.copy()

    def inb(r: int, c: int) -> bool:
        return 0 <= int(r) < H and 0 <= int(c) < W

    def refl_point(p: tuple[int, int], r: int, c: int) -> tuple[int, int]:
        pr, pc = map(int, p)
        return int(2 * pr - r), int(2 * pc - c)

    def refl_row(r0: int, r: int, c: int) -> tuple[int, int]:
        return int(2 * int(r0) - int(r)), int(c)

    def refl_col(c0: int, r: int, c: int) -> tuple[int, int]:
        return int(r), int(2 * int(c0) - int(c))

    def refl_diag_main(delta: int, r: int, c: int) -> tuple[int, int]:
        return int(c + delta), int(r - delta)

    def refl_diag_anti(total: int, r: int, c: int) -> tuple[int, int]:
        return int(total - c), int(total - r)

    for comp in comps:
        coords = comp["coords"]
        vals = comp["vals"]
        reds = [(int(r), int(c)) for (r, c), v in zip(coords, vals) if int(v) == int(MARK)]
        if not reds:
            continue

        axis = _classify_marker(reds)
        if axis is None:
            continue

        payload = [(int(r), int(c), int(v)) for (r, c), v in zip(coords, vals) if int(v) != int(MARK)]
        if not payload:
            continue

        for r, c, v in payload:
            if axis[0] == "row":
                rr, cc = refl_row(int(axis[1]), int(r), int(c))
            elif axis[0] == "col":
                rr, cc = refl_col(int(axis[1]), int(r), int(c))
            elif axis[0] == "diag_main":
                rr, cc = refl_diag_main(int(axis[1]), int(r), int(c))
            elif axis[0] == "diag_anti":
                rr, cc = refl_diag_anti(int(axis[1]), int(r), int(c))
            else:
                rr, cc = refl_point(axis[1], int(r), int(c))

            if inb(int(rr), int(cc)):
                out[int(rr), int(cc)] = int(v)

    # Keep the original isolated-red behavior unchanged for robustness, although
    # the generator for this variant does not emit this branch.
    marker_pts = []
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
                if not inb(rr, cc):
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
                    rr, cc = refl_point((int(pr), int(pc)), int(r), int(c))
                    if inb(rr, cc):
                        out[int(rr), int(cc)] = int(v)

    return out


def _build_diagonal_group(
    rng: np.random.Generator,
    *,
    bg: int,
    color: int,
    orientation: str,
) -> dict[str, object] | None:
    for _ in range(200):
        line_len = int(rng.integers(3, 7))
        payload_size = int(rng.integers(4, 9))
        pad = int(rng.integers(3, 5))
        side = int(rng.choice([-1, 1]))
        size = int(line_len + 2 * pad)

        if orientation == "main":
            line = [(pad + i, pad + i) for i in range(line_len)]
            line_value = 0

            def reflect(r: int, c: int) -> tuple[int, int]:
                return int(c), int(r)

            def side_ok(r: int, c: int) -> bool:
                return int(r - c) * side > 0

            def line_distance(r: int, c: int) -> int:
                return abs(int(r - c))

            def seeds_for(r: int, c: int) -> list[tuple[int, int]]:
                if side > 0:
                    return [(r + 1, c), (r, c - 1)]
                return [(r - 1, c), (r, c + 1)]

        else:
            line_value = int(2 * pad + line_len - 1)
            line = [(pad + i, pad + line_len - 1 - i) for i in range(line_len)]

            def reflect(r: int, c: int) -> tuple[int, int]:
                return int(line_value - c), int(line_value - r)

            def side_ok(r: int, c: int) -> bool:
                return int(r + c - line_value) * side > 0

            def line_distance(r: int, c: int) -> int:
                return abs(int(r + c - line_value))

            def seeds_for(r: int, c: int) -> list[tuple[int, int]]:
                if side > 0:
                    return [(r + 1, c), (r, c + 1)]
                return [(r - 1, c), (r, c - 1)]

        line_set = set(line)
        seeds = []
        for r, c in line:
            for rr, cc in seeds_for(int(r), int(c)):
                if 0 <= int(rr) < size and 0 <= int(cc) < size and side_ok(int(rr), int(cc)):
                    seeds.append((int(rr), int(cc)))
        seeds = sorted(set(seeds))
        if not seeds:
            continue

        payload = {seeds[int(rng.integers(0, len(seeds)))]}

        def allowed(r: int, c: int) -> bool:
            if not (0 <= int(r) < size and 0 <= int(c) < size):
                return False
            if (int(r), int(c)) in line_set or (int(r), int(c)) in payload:
                return False
            if not side_ok(int(r), int(c)):
                return False
            rr, cc = reflect(int(r), int(c))
            if not (0 <= int(rr) < size and 0 <= int(cc) < size):
                return False
            if (int(rr), int(cc)) in line_set:
                return False
            return True

        while len(payload) < payload_size:
            frontier = set()
            for r, c in payload:
                for rr, cc in _neighbors4(int(r), int(c)):
                    if allowed(int(rr), int(cc)):
                        frontier.add((int(rr), int(cc)))
            if not frontier:
                break
            options = sorted(frontier)
            payload.add(options[int(rng.integers(0, len(options)))])

        if len(payload) != payload_size:
            continue

        mirror = {reflect(int(r), int(c)) for r, c in payload}
        if payload & mirror or mirror & line_set:
            continue
        if max(line_distance(int(r), int(c)) for r, c in payload) < 2:
            continue

        rows = [int(r) for r, _c in payload]
        cols = [int(c) for _r, c in payload]
        if max(rows) == min(rows) or max(cols) == min(cols):
            continue

        patch_out = np.full((size, size), int(bg), dtype=int)
        for r, c in line:
            patch_out[int(r), int(c)] = int(MARK)
        for r, c in payload:
            patch_out[int(r), int(c)] = int(color)
        for r, c in mirror:
            patch_out[int(r), int(c)] = int(color)

        patch_in = patch_out.copy()
        for r, c in mirror:
            patch_in[int(r), int(c)] = int(bg)

        nz = np.argwhere(patch_out != int(bg))
        r0 = int(nz[:, 0].min())
        c0 = int(nz[:, 1].min())
        r1 = int(nz[:, 0].max())
        c1 = int(nz[:, 1].max())
        patch_out = patch_out[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
        patch_in = patch_in[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]

        return {
            "orientation": orientation,
            "line_len": line_len,
            "payload_size": payload_size,
            "input_patch": patch_in,
            "output_patch": patch_out,
        }

    return None


def _place_patch(
    rng: np.random.Generator,
    grid_in: np.ndarray,
    grid_out: np.ndarray,
    blocked: np.ndarray,
    patch_in: np.ndarray,
    patch_out: np.ndarray,
    bg: int,
) -> bool:
    H, W = grid_in.shape
    h, w = patch_out.shape
    out_cells = [(int(r), int(c), int(patch_out[r, c])) for r, c in np.argwhere(patch_out != int(bg))]
    in_cells = [(int(r), int(c), int(patch_in[r, c])) for r, c in np.argwhere(patch_in != int(bg))]

    for _ in range(300):
        top = int(rng.integers(0, H - h + 1))
        left = int(rng.integers(0, W - w + 1))
        ok = True
        for r, c, _v in out_cells:
            rr = int(top + r)
            cc = int(left + c)
            if blocked[int(rr), int(cc)]:
                ok = False
                break
        if not ok:
            continue

        for r, c, v in out_cells:
            grid_out[int(top + r), int(left + c)] = int(v)
        for r, c, v in in_cells:
            grid_in[int(top + r), int(left + c)] = int(v)

        for r, c, _v in out_cells:
            rr = int(top + r)
            cc = int(left + c)
            for r2 in range(max(0, int(rr) - 1), min(H, int(rr) + 2)):
                for c2 in range(max(0, int(cc) - 1), min(W, int(cc) + 2)):
                    blocked[int(r2), int(c2)] = True
        return True

    return False


def generate_input(
    rng: np.random.Generator | None = None,
    *,
    return_output: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng() if rng is None else rng

    palette = list(range(10))
    palette.remove(MARK)

    for _ in range(200):
        H = int(rng.integers(18, 29))
        W = int(rng.integers(18, 29))
        bg = int(palette[int(rng.integers(0, len(palette)))])

        colors = [c for c in range(10) if c not in (bg, MARK)]
        n_groups = int(rng.integers(3, min(6, len(colors)) + 1))
        chosen_colors = list(map(int, rng.choice(colors, size=n_groups, replace=False)))

        orientations = ["main", "anti"]
        while len(orientations) < n_groups:
            orientations.append(str(rng.choice(["main", "anti"])))
        rng.shuffle(orientations)

        grid_in = np.full((H, W), int(bg), dtype=int)
        grid_out = np.full((H, W), int(bg), dtype=int)
        blocked = np.zeros((H, W), dtype=bool)
        placed = 0

        for color, orientation in zip(chosen_colors, orientations):
            group = _build_diagonal_group(rng, bg=int(bg), color=int(color), orientation=str(orientation))
            if group is None:
                break
            ok = _place_patch(
                rng,
                grid_in,
                grid_out,
                blocked,
                group["input_patch"],
                group["output_patch"],
                int(bg),
            )
            if not ok:
                break
            placed += 1

        if placed != n_groups:
            continue

        solved = main(grid_in)
        if not np.array_equal(solved, grid_out):
            continue

        if return_output:
            return grid_in, grid_out
        return grid_in

    raise RuntimeError("failed to generate a valid diagonal-reflection sample")
