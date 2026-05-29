# Combo variant of 16b78196 (index 3)
# combo_spec: D1/1+D4/2+D8/0
# Name: 旋转拼图+颜色排序+扇形扩展

from __future__ import annotations

from itertools import combinations
from pathlib import Path
import random
import sys

import numpy as np

for _UTILS_DIR in (
    Path("/home/lixinyu/snap/new/solvers"),
    Path("/home/lixinyu/snap/new/arc2_solver_select"),
):
    if str(_UTILS_DIR) not in sys.path:
        sys.path.append(str(_UTILS_DIR))

from arc_utils import components_mask, find_components, get_bg


# Design analysis for the fused variant
# -------------------------------------
# a) Concrete behavior and trace
#    The preserved scaffold is still "one central full-span bar + 1-2 outward chains made
#    from scattered monochrome sprites". The fused changes alter three linked decisions:
#    1. D1/1 rotation puzzle:
#       A scattered sprite is no longer guaranteed to already face the correct way. The
#       solver must test unique dihedral variants and pick the transformed silhouette that
#       plugs the notch or fills the previous cavity.
#    2. D4/2 ordered assembly:
#       Chain order is no longer "any piece whose shape fits". Along each chain, colors
#       must become visibly brighter from bar -> outer tip. This is implemented via the
#       fixed ARC palette luminance order, not hidden numeric IDs, so the order is human-
#       observable from the colors themselves.
#    3. D8/0 fan expansion:
#       The strip is mutable. If the root spans width 4 or 5, the next link must span
#       width 5 or 6 by widening the chain one cell on a fixed side. Widths therefore form
#       a short ladder such as 4->5 or 4->5->6.
#    Small trace, horizontal bar, chain grows upward and fans right:
#    - Original 16b78196:
#      root width 4 plugs a notch; next piece keeps width 4 and fills the root bbox cavity.
#    - Fused variant:
#      input root might appear as a rotated/mirrored shape away from the bar. The solver
#      transforms it, places it as the darkest color at width 4, then only considers a
#      brighter piece whose transformed span is width 5 and whose cells exactly cover the
#      root cavity while widening the strip on the right. A third piece, if present, must
#      be brighter again and width 6. Rotation affects the footprint, the width step fixes
#      the allowed strip, and color order fixes which candidate can be next.
# b) Complexity calibration
#    To keep the rule visible but tractable:
#    - Chain lengths are restricted to 2-3 and at least one chain has length 3, so the
#      brightness and width progression is visible in most examples.
#    - Width ladders are only 4->5, 5->6, or 4->5->6.
#    - Generator uses one-sided widening templates with areas 8/10/12, which keeps the
#      search space small while making the fan direction easy to see.
#    - Root pieces are forced to use a non-identity transform in the input. Many non-root
#      pieces are also transformed, but the generator does not require every piece to be.
# c) Hidden-assumption audit against the original main()
#    - Original assumption: one fixed orientation per piece. Broken by D1/1. We replace the
#      single `shape` with deduplicated dihedral `variants` and store the chosen transform
#      in the chain head.
#    - Original assumption: the root strip [A,B] stays fixed forever. Broken by D8/0. We
#      now store mutable `A,B` and a per-chain `fan_dir`; each extension tries exactly one
#      wider strip.
#    - Original assumption: any remaining piece that fills the cavity is eligible. Broken
#      by D4/2. We now require strictly increasing visible brightness along each chain.
#    - New fused interaction: a transformed candidate may cover the cavity at multiple
#      offsets, but only one offset can simultaneously satisfy the widened strip and color
#      monotonicity. This coupling is deliberate and solver search is built around it.
# d) Placement legality
#    The generator is output-first. It first builds valid chains in output space, checking:
#    - every used notch is inside the bar and distinct from other holes;
#    - every piece stays in bounds, remains 4-connected, and does not overlap existing
#      occupied cells;
#    - chain widths widen by exactly 1 at each step on a fixed side.
#    Only after the output is valid are the exact piece masks scattered into the input.
#    Scattering preserves in-bounds placement and prevents overlap or touching the bar or
#    other pieces within an 8-neighborhood for visual cleanliness.
# e) Visual observability
#    Sorting by raw color ID would be a hidden rule, so the fusion instead uses the visible
#    ARC palette brightness order. The generator picks chain colors with clear luminance
#    gaps, and the direction is always "darker near the bar, brighter outward". Across 2-4
#    samples a human can see both the progressive widening and the dark->bright order.
# f) Visual cleanliness
#    The bar remains the dominant largest component. Pieces are sparse, non-touching, and
#    the widening template uses only three rows/columns per piece, so the grid stays legible
#    even after adding rotations. Decoy notches exist only on the bar boundary and do not
#    overlap the colored chains.
# g) Why the fusion does not collapse back to a single-factor variant
#    - Without transform search, the input roots are intentionally misoriented.
#    - Without brightness order, multiple width-compatible continuations could be explored.
#    - Without widening, the later cavity matches would reduce to the older fixed-strip
#      logic. The generator records all three signals and later tests verify they co-occur.
# h) New solver assumptions introduced here and generator guarantees
#    - Assumption: the correct chain colors are strictly increasing in visible brightness.
#      Guarantee: generator assigns every chain a strictly increasing brightness sequence.
#    - Assumption: every used root has a non-identity dihedral variant.
#      Guarantee: generator rejects samples where a root would be transform-symmetric.
#    - Assumption: non-root pieces cannot masquerade as used roots on the contiguous bar
#      notch. Guarantee: the widening template omits an interior back-row cell on non-root
#      links, while used notches are contiguous edge mouths, so only true roots fit.


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

ARC_RGB = {
    0: (0, 0, 0),
    1: (0, 116, 217),
    2: (255, 65, 54),
    3: (46, 204, 64),
    4: (255, 220, 0),
    5: (170, 170, 170),
    6: (240, 18, 190),
    7: (255, 133, 27),
    8: (127, 219, 255),
    9: (135, 12, 37),
}
BRIGHTNESS = {
    color: float(0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2])
    for color, rgb in ARC_RGB.items()
}
VISIBLE_COLOR_ORDER = [color for color in sorted(range(1, 10), key=lambda c: (BRIGHTNESS[c], c))]
_LAST_GEN_INFO: dict[str, object] = {}


def _shape_from_coords(coords: np.ndarray, bbox) -> dict[str, object]:
    r0, c0, r1, c1 = map(int, bbox)
    rel = sorted((int(r - r0), int(c - c0)) for r, c in coords)
    return {
        "rel": rel,
        "rel_set": set(rel),
        "h": int(r1 - r0 + 1),
        "w": int(c1 - c0 + 1),
    }


def _shape_from_cells(cells) -> dict[str, object]:
    coords = np.array(sorted((int(r), int(c)) for r, c in cells), dtype=int)
    rs = coords[:, 0]
    cs = coords[:, 1]
    bbox = (int(rs.min()), int(cs.min()), int(rs.max()), int(cs.max()))
    return _shape_from_coords(coords, bbox)


def _shape_variants(shape: dict[str, object]) -> tuple[list[dict[str, object]], tuple[tuple[int, int], ...]]:
    mask = np.zeros((int(shape["h"]), int(shape["w"])), dtype=bool)
    for r, c in shape["rel"]:
        mask[int(r), int(c)] = True

    base_key = tuple(shape["rel"])
    variants: list[dict[str, object]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for k in range(4):
        rot = np.rot90(mask, int(k))
        for flip in (False, True):
            arr = np.fliplr(rot) if flip else rot
            rel = tuple((int(r), int(c)) for r, c in np.argwhere(arr))
            if rel in seen:
                continue
            seen.add(rel)
            variants.append(
                {
                    "rel": list(rel),
                    "rel_set": set(rel),
                    "h": int(arr.shape[0]),
                    "w": int(arr.shape[1]),
                    "rot": int(k),
                    "flip": bool(flip),
                    "is_base": bool(rel == base_key),
                }
            )
    return variants, base_key


def _transform_name(rot: int, flip: bool) -> str:
    return ("f" if flip else "r") + str(int(rot) * 90)


def _brightness(color: int) -> float:
    return float(BRIGHTNESS[int(color)])


def _connected(cells) -> bool:
    if len(cells) <= 1:
        return True
    cells_set = set(cells)
    start = next(iter(cells_set))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in DIR4:
            nb = (int(r + dr), int(c + dc))
            if nb in cells_set and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return len(seen) == len(cells_set)


def _grow_connected(rng: random.Random, available, target: int, seeds, must_span_axis=None, span_range=None, retries: int = 3):
    for _ in range(retries):
        cells = set(seeds)
        if not cells.issubset(available):
            return None
        for __ in range(target * 20):
            if len(cells) >= target:
                break
            frontier = [
                nb
                for r, c in cells
                for dr, dc in DIR4
                if (nb := (int(r + dr), int(c + dc))) in available and nb not in cells
            ]
            if not frontier:
                break

            if must_span_axis and span_range:
                A, B = map(int, span_range)
                if must_span_axis == "cols":
                    cmin = min((c for _, c in cells), default=B)
                    cmax = max((c for _, c in cells), default=A)
                    if cmin > A:
                        opts = [nb for nb in frontier if nb[1] < cmin]
                        if opts:
                            cells.add(rng.choice(opts))
                            continue
                    if cmax < B:
                        opts = [nb for nb in frontier if nb[1] > cmax]
                        if opts:
                            cells.add(rng.choice(opts))
                            continue
                else:
                    rmin = min((r for r, _ in cells), default=B)
                    rmax = max((r for r, _ in cells), default=A)
                    if rmin > A:
                        opts = [nb for nb in frontier if nb[0] < rmin]
                        if opts:
                            cells.add(rng.choice(opts))
                            continue
                    if rmax < B:
                        opts = [nb for nb in frontier if nb[0] > rmax]
                        if opts:
                            cells.add(rng.choice(opts))
                            continue
            cells.add(rng.choice(frontier))

        if len(cells) < target:
            continue

        while len(cells) > target:
            removable = []
            for rc in cells:
                if rc in seeds:
                    continue
                if must_span_axis and span_range:
                    A, B = map(int, span_range)
                    if must_span_axis == "cols":
                        if rc[1] == A and sum(1 for _, c2 in cells if c2 == A) <= 1:
                            continue
                        if rc[1] == B and sum(1 for _, c2 in cells if c2 == B) <= 1:
                            continue
                    else:
                        if rc[0] == A and sum(1 for r2, _ in cells if r2 == A) <= 1:
                            continue
                        if rc[0] == B and sum(1 for r2, _ in cells if r2 == B) <= 1:
                            continue
                if _connected(cells - {rc}):
                    removable.append(rc)
            if not removable:
                break
            cells.remove(rng.choice(removable))

        if len(cells) != target:
            continue
        if must_span_axis and span_range:
            A, B = map(int, span_range)
            if must_span_axis == "cols":
                if min(c for _, c in cells) != A or max(c for _, c in cells) != B:
                    continue
            else:
                if min(r for r, _ in cells) != A or max(r for r, _ in cells) != B:
                    continue
        if _connected(cells):
            return cells
    return None


def _piece_bbox(coords):
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return int(min(rs)), int(min(cs)), int(max(rs)), int(max(cs))


def _pixels(shape: dict[str, object], top: int, left: int) -> set[tuple[int, int]]:
    return {(int(top + lr), int(left + lc)) for lr, lc in shape["rel"]}


def _placement_pts(shape: dict[str, object], top: int, left: int, H: int, W: int, occ, must_cover=None, strip=None):
    if top < 0 or left < 0 or top + int(shape["h"]) > H or left + int(shape["w"]) > W:
        return None
    if strip is not None:
        axis, A, B = strip
        if axis == "cols":
            if left != int(A) or int(shape["w"]) != int(B) - int(A) + 1:
                return None
        else:
            if top != int(A) or int(shape["h"]) != int(B) - int(A) + 1:
                return None
    pts = _pixels(shape, top, left)
    if pts & occ:
        return None
    if must_cover is not None and not (must_cover <= pts):
        return None
    return pts


def _aligned_positions(axis: str, A: int, cover, shape: dict[str, object]):
    if axis == "cols":
        left = int(A)
        tops = set()
        for rN, cN in cover:
            for lr, lc in shape["rel"]:
                if int(cN) == left + int(lc):
                    tops.add(int(rN - lr))
        for top in tops:
            yield int(top), int(left)
    else:
        top = int(A)
        lefts = set()
        for rN, cN in cover:
            for lr, lc in shape["rel"]:
                if int(rN) == top + int(lr):
                    lefts.add(int(cN - lc))
        for left in lefts:
            yield int(top), int(left)


def _touches(pts, other) -> bool:
    for r, c in pts:
        for dr, dc in DIR4:
            if (int(r + dr), int(c + dc)) in other:
                return True
    return False


def _hole_positions(hole, shape: dict[str, object]):
    pos = set()
    for rH, cH in hole:
        for lr, lc in shape["rel"]:
            pos.add((int(rH - lr), int(cH - lc)))
    return pos


def _span_len(shape: dict[str, object], axis: str) -> int:
    return int(shape["w"] if axis == "cols" else shape["h"])


def _span_range(axis: str, top: int, left: int, shape: dict[str, object]) -> tuple[int, int]:
    if axis == "cols":
        return int(left), int(left + int(shape["w"]) - 1)
    return int(top), int(top + int(shape["h"]) - 1)


def _interval_after_step(A: int, B: int, fan_dir: int) -> tuple[int, int]:
    return (int(A - 1), int(B)) if int(fan_dir) < 0 else (int(A), int(B + 1))


def _build_template_piece(orientation: str, side: str, interval: tuple[int, int], anchor_curr: int, prev_anchor: int, back_coord: int):
    A, B = map(int, interval)
    cells = set()

    if orientation == "horizontal":
        cols = list(range(A, B + 1))
        if side == "top":
            front_r, mid_r, back_r = int(back_coord - 2), int(back_coord - 1), int(back_coord)
        else:
            back_r, mid_r, front_r = int(back_coord), int(back_coord + 1), int(back_coord + 2)
        cells.add((int(front_r), int(anchor_curr)))
        for c in cols:
            cells.add((int(mid_r), int(c)))
        for c in cols:
            if int(c) != int(prev_anchor):
                cells.add((int(back_r), int(c)))
    else:
        rows = list(range(A, B + 1))
        if side == "left":
            front_c, mid_c, back_c = int(back_coord - 2), int(back_coord - 1), int(back_coord)
        else:
            back_c, mid_c, front_c = int(back_coord), int(back_coord + 1), int(back_coord + 2)
        cells.add((int(anchor_curr), int(front_c)))
        for r in rows:
            cells.add((int(r), int(mid_c)))
        for r in rows:
            if int(r) != int(prev_anchor):
                cells.add((int(r), int(back_c)))

    return cells


def _bar_notches(bar_mask: np.ndarray, bar_r0: int, bar_c0: int):
    empty = ~bar_mask
    h, w = map(int, empty.shape)
    notches = []
    for pts in components_mask(empty, connectivity=4):
        rr = pts[:, 0]
        cc = pts[:, 1]
        top = int((rr == 0).sum())
        bottom = int((rr == h - 1).sum())
        left = int((cc == 0).sum())
        right = int((cc == w - 1).sum())
        if top + bottom + left + right == 0:
            continue

        side, mouth_n = max(
            [(name, cnt) for name, cnt in [("top", top), ("bottom", bottom), ("left", left), ("right", right)] if cnt > 0],
            key=lambda x: x[1],
        )
        axis = "cols" if side in ("top", "bottom") else "rows"
        hole_gl = {(int(bar_r0 + r), int(bar_c0 + c)) for r, c in pts}
        notches.append({"axis": axis, "side": side, "hole": hole_gl, "mouth_n": int(mouth_n)})
    return notches


def _valid_color_combo(combo: tuple[int, ...]) -> bool:
    vals = [_brightness(int(c)) for c in combo]
    min_gap = 18.0
    min_span = 45.0 if len(combo) >= 3 else 30.0
    return all(vals[i + 1] - vals[i] >= min_gap for i in range(len(vals) - 1)) and vals[-1] - vals[0] >= min_span


def _pick_chain_color_sequences(rng: random.Random, chain_lens: list[int]) -> tuple[list[list[int]], int] | None:
    combos_by_len = {
        L: [combo for combo in combinations(VISIBLE_COLOR_ORDER, L) if _valid_color_combo(combo)]
        for L in sorted(set(chain_lens))
    }

    def rec(idx: int, available: set[int], acc: list[list[int]]):
        if idx == len(chain_lens):
            return [list(seq) for seq in acc]
        length = int(chain_lens[idx])
        candidates = [combo for combo in combos_by_len[length] if set(combo).issubset(available)]
        rng.shuffle(candidates)
        candidates.sort(key=lambda combo: -(_brightness(combo[-1]) - _brightness(combo[0])))
        for combo in candidates[:80]:
            acc.append(list(map(int, combo)))
            res = rec(idx + 1, available - set(combo), acc)
            if res is not None:
                return res
            acc.pop()
        return None

    piece_seqs = rec(0, set(VISIBLE_COLOR_ORDER), [])
    if piece_seqs is None:
        return None
    used = {int(c) for seq in piece_seqs for c in seq}
    bar_choices = [int(c) for c in range(1, 10) if int(c) not in used]
    if not bar_choices:
        return None
    return piece_seqs, int(rng.choice(bar_choices))


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    root_keep_per_piece = 8
    root_keep_total = 64
    extend_keep_per_piece = 4
    extend_keep_total = 24

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    comps.sort(key=lambda d: -d["area"])
    assert comps

    bar = comps[0]
    pieces_raw = comps[1:]

    out = np.full((H, W), int(bg), dtype=int)
    bar_r0, bar_c0, bar_r1, bar_c1 = map(int, bar["bbox"])
    bar_h, bar_w = int(bar_r1 - bar_r0 + 1), int(bar_c1 - bar_c0 + 1)
    bar_color = int(bar["color"])
    bar_occ = {(int(r), int(c)) for r, c in bar["coords"]}
    bar_mask = np.zeros((bar_h, bar_w), dtype=bool)
    for r, c in bar_occ:
        bar_mask[int(r - bar_r0), int(c - bar_c0)] = True
        out[int(r), int(c)] = int(bar_color)

    pieces = []
    for p in pieces_raw:
        shape = _shape_from_coords(p["coords"], p["bbox"])
        variants, base_key = _shape_variants(shape)
        r0, c0 = map(int, p["bbox"][:2])
        pieces.append(
            {
                "color": int(p["color"]),
                "brightness": _brightness(int(p["color"])),
                "orig": (int(r0), int(c0)),
                "base_key": base_key,
                "variants": variants,
            }
        )

    notches = _bar_notches(bar_mask, bar_r0, bar_c0)

    root_cands = []
    for ni, notch in enumerate(notches):
        axis = str(notch["axis"])
        for pi, piece in enumerate(pieces):
            local = []
            for variant in piece["variants"]:
                root_width = _span_len(variant, axis)
                if root_width not in (4, 5):
                    continue
                for top, left in _hole_positions(notch["hole"], variant):
                    pts = _placement_pts(variant, top, left, H, W, bar_occ, must_cover=notch["hole"])
                    if pts is None:
                        continue
                    A, B = _span_range(axis, int(top), int(left), variant)
                    mv = abs(int(top) - int(piece["orig"][0])) + abs(int(left) - int(piece["orig"][1]))
                    local.append(
                        (
                            (int(mv), int(root_width), piece["brightness"], -int(notch["mouth_n"])),
                            {
                                "ni": int(ni),
                                "pi": int(pi),
                                "top": int(top),
                                "left": int(left),
                                "axis": axis,
                                "A": int(A),
                                "B": int(B),
                                "shape": variant,
                            },
                        )
                    )
            local.sort(key=lambda x: x[0])
            root_cands.extend(local[:root_keep_per_piece])

    root_cands.sort(key=lambda x: x[0])
    cand_roots = [item[1] for item in root_cands[:root_keep_total]]

    best_out = out.copy()
    best_key = None

    def search(out_grid: np.ndarray, occ_set, chains, remaining) -> bool:
        nonlocal best_out, best_key
        placed_cnt = len(pieces) - len(remaining)
        key = (int(placed_cnt), -int(len(remaining)), -int(len(occ_set)))
        if best_key is None or key > best_key:
            best_key = key
            best_out = out_grid.copy()
        if not remaining:
            return True

        def candidate_strips(chain):
            axis = str(chain["axis"])
            A, B = int(chain["A"]), int(chain["B"])
            dirs = [-1, 1] if chain["fan_dir"] is None else [int(chain["fan_dir"])]
            ranges = []
            for fan_dir in dirs:
                nA, nB = _interval_after_step(A, B, int(fan_dir))
                if axis == "cols":
                    if 0 <= nA and nB < W:
                        ranges.append((int(fan_dir), int(nA), int(nB)))
                else:
                    if 0 <= nA and nB < H:
                        ranges.append((int(fan_dir), int(nA), int(nB)))
            return ranges

        def gen_extend_moves(chain, remaining_set):
            axis = str(chain["axis"])
            head = chain["head"]
            head_pts = _pixels(head["shape"], int(head["top"]), int(head["left"]))
            head_top, head_left = int(head["top"]), int(head["left"])
            head_h, head_w = int(head["shape"]["h"]), int(head["shape"]["w"])
            cavity = {
                (int(r), int(c))
                for r in range(head_top, head_top + head_h)
                for c in range(head_left, head_left + head_w)
                if int(out_grid[r, c]) == int(bg)
            }
            if not cavity:
                return []

            moves = []
            for fan_dir, nA, nB in candidate_strips(chain):
                expected_width = int(nB - nA + 1)
                for pi in remaining_set:
                    if pieces[pi]["brightness"] <= float(chain["last_brightness"]):
                        continue
                    piece_moves = []
                    for variant in pieces[pi]["variants"]:
                        if _span_len(variant, axis) != expected_width:
                            continue
                        for top, left in _aligned_positions(axis, nA, cavity, variant):
                            pts = _placement_pts(variant, top, left, H, W, occ_set, must_cover=cavity, strip=(axis, nA, nB))
                            if pts is None or not _touches(pts, head_pts):
                                continue
                            mv = abs(int(top) - int(pieces[pi]["orig"][0])) + abs(int(left) - int(pieces[pi]["orig"][1]))
                            piece_moves.append(
                                (
                                    (
                                        int(mv),
                                        pieces[pi]["brightness"] - float(chain["last_brightness"]),
                                    ),
                                    (
                                        int(pi),
                                        int(top),
                                        int(left),
                                        pts,
                                        variant,
                                        int(fan_dir),
                                        int(nA),
                                        int(nB),
                                    ),
                                )
                            )
                    piece_moves.sort(key=lambda x: x[0])
                    moves.extend(piece_moves[:extend_keep_per_piece])
            moves.sort(key=lambda x: x[0])
            return [m[1] for m in moves[:extend_keep_total]]

        chain_moves = []
        for idx, chain in enumerate(chains):
            moves = gen_extend_moves(chain, remaining)
            if moves:
                chain_moves.append((len(moves), int(idx), moves))
        if not chain_moves:
            return False

        chain_moves.sort(key=lambda x: x[0])
        for _, ch_idx, moves in chain_moves:
            ch = chains[ch_idx]
            for pi, top, left, pts, variant, fan_dir, nA, nB in moves:
                for r, c in pts:
                    out_grid[int(r), int(c)] = int(pieces[pi]["color"])
                old_state = (
                    int(ch["A"]),
                    int(ch["B"]),
                    ch["fan_dir"],
                    ch["head"],
                    float(ch["last_brightness"]),
                )
                ch["A"], ch["B"], ch["fan_dir"] = int(nA), int(nB), int(fan_dir)
                ch["head"] = {"top": int(top), "left": int(left), "shape": variant}
                ch["last_brightness"] = float(pieces[pi]["brightness"])
                occ2 = set(occ_set) | set(pts)
                rem2 = set(remaining)
                rem2.remove(int(pi))
                if search(out_grid, occ2, chains, rem2):
                    return True
                ch["A"], ch["B"], ch["fan_dir"], ch["head"], ch["last_brightness"] = old_state
                for r, c in pts:
                    out_grid[int(r), int(c)] = int(bg)
                for r, c in bar_occ:
                    out_grid[int(r), int(c)] = int(bar_color)
        return False

    for k in range(1, min(2, len(cand_roots)) + 1):
        for roots_sel in combinations(range(len(cand_roots)), k):
            used_p = set()
            used_n = set()
            occ0 = set(bar_occ)
            out0 = np.full((H, W), int(bg), dtype=int)
            for r, c in bar_occ:
                out0[int(r), int(c)] = int(bar_color)

            chains = []
            ok = True
            for idx in roots_sel:
                root = cand_roots[idx]
                if root["pi"] in used_p or root["ni"] in used_n:
                    ok = False
                    break
                pts = _pixels(root["shape"], int(root["top"]), int(root["left"]))
                if pts & occ0:
                    ok = False
                    break
                used_p.add(int(root["pi"]))
                used_n.add(int(root["ni"]))
                for r, c in pts:
                    out0[int(r), int(c)] = int(pieces[root["pi"]]["color"])
                occ0 |= pts
                chains.append(
                    {
                        "axis": root["axis"],
                        "A": int(root["A"]),
                        "B": int(root["B"]),
                        "fan_dir": None,
                        "last_brightness": float(pieces[root["pi"]]["brightness"]),
                        "head": {"top": int(root["top"]), "left": int(root["left"]), "shape": root["shape"]},
                    }
                )
            if not ok:
                continue
            remaining = set(range(len(pieces))) - used_p
            search(out0, occ0, chains, remaining)

    return best_out


def _scatter_piece(
    rng: random.Random,
    input_grid: np.ndarray,
    placed_occ: set[tuple[int, int]],
    variant: dict[str, object],
    color: int,
    orientation: str,
    bar_r0: int,
    bar_r1: int,
    bar_c0: int,
    bar_c1: int,
) -> bool:
    p_h, p_w = int(variant["h"]), int(variant["w"])
    H, W = map(int, input_grid.shape)
    for _ in range(220):
        gap = int(rng.randint(2, 7))
        if orientation == "horizontal":
            top = (int(bar_r0 - gap - p_h)) if rng.random() < 0.5 else int(bar_r1 + gap + 1)
            left = int(rng.randint(0, max(0, W - p_w)))
        else:
            left = (int(bar_c0 - gap - p_w)) if rng.random() < 0.5 else int(bar_c1 + gap + 1)
            top = int(rng.randint(0, max(0, H - p_h)))
        if top < 0 or left < 0 or top + p_h > H or left + p_w > W:
            continue
        pts = {(int(top + dr), int(left + dc)) for dr, dc in variant["rel"]}
        if pts & placed_occ:
            continue
        blocked = False
        for r, c in pts:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if (int(r + dr), int(c + dc)) in placed_occ:
                        blocked = True
                        break
                if blocked:
                    break
            if blocked:
                break
        if blocked:
            continue
        for r, c in pts:
            input_grid[int(r), int(c)] = int(color)
        placed_occ |= pts
        return True
    return False


def _build_generation_metrics(chain_defs: list[dict[str, object]], transformed_piece_count: int, transformed_root_count: int, transform_hist: dict[str, int]):
    return {
        "num_chains": int(len(chain_defs)),
        "chain_lengths": [int(len(ch["pieces"])) for ch in chain_defs],
        "width_sequences": [[int(w) for w in ch["widths"]] for ch in chain_defs],
        "brightness_sequences": [[round(_brightness(int(color)), 1) for color in ch["colors"]] for ch in chain_defs],
        "colors_by_chain": [[int(color) for color in ch["colors"]] for ch in chain_defs],
        "fan_dirs": [int(ch["fan_dir"]) for ch in chain_defs],
        "transformed_piece_count": int(transformed_piece_count),
        "transformed_root_count": int(transformed_root_count),
        "transform_hist": dict(transform_hist),
    }


def _gen_attempt(seed: int | None = None):
    rng = random.Random(seed)
    H, W = 30, 30
    bg = 0

    orientation = str(rng.choice(["horizontal", "vertical"]))
    thickness = int(rng.randint(5, 7))

    if orientation == "horizontal":
        center = int(rng.uniform(0.35, 0.53) * H)
        bar_r0 = max(0, int(center - thickness // 2))
        bar_r1 = int(bar_r0 + thickness - 1)
        if bar_r1 >= H:
            bar_r1 = H - 1
            bar_r0 = int(bar_r1 - thickness + 1)
        bar_c0, bar_c1 = 0, W - 1
        space_before, space_after = int(bar_r0), int(H - 1 - bar_r1)
    else:
        center = int(rng.uniform(0.35, 0.53) * W)
        bar_c0 = max(0, int(center - thickness // 2))
        bar_c1 = int(bar_c0 + thickness - 1)
        if bar_c1 >= W:
            bar_c1 = W - 1
            bar_c0 = int(bar_c1 - thickness + 1)
        bar_r0, bar_r1 = 0, H - 1
        space_before, space_after = int(bar_c0), int(W - 1 - bar_c1)

    num_chains = int(rng.choice([1, 2]))
    if num_chains == 1:
        chain_lens = [3]
    else:
        chain_lens = list(rng.choice([[3, 2], [2, 3], [3, 3]]))

    picked = _pick_chain_color_sequences(rng, chain_lens)
    if picked is None:
        return None
    chain_color_seqs, bar_color = picked

    bar_cells = {(int(r), int(c)) for r in range(bar_r0, bar_r1 + 1) for c in range(bar_c0, bar_c1 + 1)}
    occ_set = set(bar_cells)
    notch_holes: list[set[tuple[int, int]]] = []
    chain_defs: list[dict[str, object]] = []

    for chain_idx, chain_len in enumerate(chain_lens):
        base_width = 4 if int(chain_len) == 3 else int(rng.choice([4, 5]))
        fan_dir = int(rng.choice([-1, 1]))
        side_choices = ["top", "bottom"] if orientation == "horizontal" else ["left", "right"]
        rng.shuffle(side_choices)
        need_space = int(2 * chain_len)
        if orientation == "horizontal":
            side = next((s for s in side_choices if (space_before if s == "top" else space_after) >= need_space), None)
            if side is None:
                return None
            final_width = int(base_width + chain_len - 1)
            min_start = 1 + (chain_len - 1 if fan_dir < 0 else 0)
            max_start = (W - 1 - final_width) - (0 if fan_dir < 0 else 0)
            if max_start < min_start:
                return None
            start = int(rng.randint(min_start, max_start))
            A0, B0 = int(start), int(start + base_width - 1)
            anchor0 = int(B0 if fan_dir > 0 else A0)
            notch_row = int(bar_r0 if side == "top" else bar_r1)
            notch = {(int(notch_row), int(c)) for c in range(A0, B0 + 1) if int(c) != anchor0}
            back_coord = int(bar_r0 if side == "top" else bar_r1)
        else:
            side = next((s for s in side_choices if (space_before if s == "left" else space_after) >= need_space), None)
            if side is None:
                return None
            final_width = int(base_width + chain_len - 1)
            min_start = 1 + (chain_len - 1 if fan_dir < 0 else 0)
            max_start = (H - 1 - final_width) - (0 if fan_dir < 0 else 0)
            if max_start < min_start:
                return None
            start = int(rng.randint(min_start, max_start))
            A0, B0 = int(start), int(start + base_width - 1)
            anchor0 = int(B0 if fan_dir > 0 else A0)
            notch_col = int(bar_c0 if side == "left" else bar_c1)
            notch = {(int(r), int(notch_col)) for r in range(A0, B0 + 1) if int(r) != anchor0}
            back_coord = int(bar_c0 if side == "left" else bar_c1)

        all_holes = set().union(*notch_holes) if notch_holes else set()
        halo = {(int(r + dr), int(c + dc)) for r, c in all_holes for dr, dc in DIR4}
        if notch & (all_holes | halo):
            return None
        if not notch.issubset(bar_cells):
            return None
        bar_cells -= notch
        occ_set -= notch
        notch_holes.append(set(notch))

        interval = (int(A0), int(B0))
        prev_anchor = int(anchor0)
        piece_defs = []
        widths = []
        colors = []

        for piece_idx in range(chain_len):
            A, B = map(int, interval)
            width = int(B - A + 1)
            anchor_curr = int(B if fan_dir > 0 else A)
            piece_cells = _build_template_piece(orientation, side, interval, anchor_curr, prev_anchor, back_coord)
            if len(piece_cells) < 8 or len(piece_cells) > 13:
                return None
            if any(not (0 <= r < H and 0 <= c < W) for r, c in piece_cells):
                return None
            if piece_cells & occ_set:
                return None
            if not _connected(piece_cells):
                return None

            color = int(chain_color_seqs[chain_idx][piece_idx])
            piece_defs.append(
                {
                    "coords": set(piece_cells),
                    "color": int(color),
                    "is_root": bool(piece_idx == 0),
                    "chain_idx": int(chain_idx),
                }
            )
            widths.append(int(width))
            colors.append(int(color))
            occ_set |= piece_cells

            if orientation == "horizontal":
                front = int(min(r for r, _ in piece_cells) if side == "top" else max(r for r, _ in piece_cells))
            else:
                front = int(min(c for _, c in piece_cells) if side == "left" else max(c for _, c in piece_cells))
            back_coord = int(front)
            prev_anchor = int(anchor_curr)
            interval = _interval_after_step(A, B, fan_dir)

        chain_defs.append(
            {
                "chain_idx": int(chain_idx),
                "side": side,
                "fan_dir": int(fan_dir),
                "pieces": piece_defs,
                "widths": widths,
                "colors": colors,
            }
        )

    if not (3 <= sum(len(ch["pieces"]) for ch in chain_defs) <= 6):
        return None

    num_decoys = int(rng.randint(max(4, 6 - num_chains), 9 - num_chains))
    decoys = 0
    for _ in range(600):
        if decoys >= num_decoys:
            break
        depth = int(rng.randint(1, 2))
        area = int(rng.randint(1, min(6, depth * 5)))
        if orientation == "horizontal":
            ds = str(rng.choice(["top", "bottom"]))
            dr0 = int(bar_r0 if ds == "top" else bar_r1 - depth + 1)
            dr1 = int(bar_r0 + depth - 1 if ds == "top" else bar_r1)
            dc0 = int(rng.randint(bar_c0, bar_c1))
            dc1 = int(min(bar_c1, dc0 + rng.randint(0, 4)))
        else:
            ds = str(rng.choice(["left", "right"]))
            dc0 = int(bar_c0 if ds == "left" else bar_c1 - depth + 1)
            dc1 = int(bar_c0 + depth - 1 if ds == "left" else bar_c1)
            dr0 = int(rng.randint(bar_r0, bar_r1))
            dr1 = int(min(bar_r1, dr0 + rng.randint(0, 4)))

        d_avail = {(int(r), int(c)) for r in range(dr0, dr1 + 1) for c in range(dc0, dc1 + 1) if (int(r), int(c)) in bar_cells}
        all_holes = set().union(*notch_holes) if notch_holes else set()
        halo = {(int(r + dr), int(c + dc)) for r, c in all_holes for dr, dc in DIR4}
        d_avail -= all_holes | halo
        if len(d_avail) < 1:
            continue
        area = min(int(area), len(d_avail))
        dn = _grow_connected(rng, d_avail, int(area), {rng.choice(list(d_avail))})
        if dn is None or not _connected(bar_cells - dn):
            continue
        bar_cells -= dn
        notch_holes.append(set(dn))
        decoys += 1
    if decoys < num_decoys:
        return None

    bar_h, bar_w = int(bar_r1 - bar_r0 + 1), int(bar_c1 - bar_c0 + 1)
    bar_mask = np.zeros((bar_h, bar_w), dtype=bool)
    for r, c in bar_cells:
        bar_mask[int(r - bar_r0), int(c - bar_c0)] = True
    for pts in components_mask(~bar_mask, connectivity=4):
        rr = pts[:, 0]
        cc = pts[:, 1]
        if not ((rr == 0).any() or (rr == bar_h - 1).any() or (cc == 0).any() or (cc == bar_w - 1).any()):
            return None

    output = np.full((H, W), int(bg), dtype=int)
    for r, c in bar_cells:
        output[int(r), int(c)] = int(bar_color)
    all_pieces = []
    for chain in chain_defs:
        for piece in chain["pieces"]:
            all_pieces.append(piece)
            for r, c in piece["coords"]:
                output[int(r), int(c)] = int(piece["color"])

    piece_infos = []
    for piece in all_pieces:
        shape = _shape_from_cells(piece["coords"])
        variants, base_key = _shape_variants(shape)
        can_transform = any(tuple(v["rel"]) != base_key for v in variants)
        if piece["is_root"] and not can_transform:
            return None
        piece_infos.append(
            {
                **piece,
                "shape": shape,
                "base_key": base_key,
                "variants": variants,
                "can_transform": bool(can_transform),
            }
        )

    input_grid = np.full((H, W), int(bg), dtype=int)
    for r, c in bar_cells:
        input_grid[int(r), int(c)] = int(bar_color)
    placed_occ = set(bar_cells)

    order = list(range(len(piece_infos)))
    rng.shuffle(order)
    transformed_piece_count = 0
    transformed_root_count = 0
    transform_hist: dict[str, int] = {}

    for idx in order:
        piece = piece_infos[idx]
        non_base = [v for v in piece["variants"] if tuple(v["rel"]) != piece["base_key"]]
        base_only = [v for v in piece["variants"] if tuple(v["rel"]) == piece["base_key"]]
        if piece["is_root"]:
            candidate_vars = non_base[:]
            rng.shuffle(candidate_vars)
        else:
            if non_base and rng.random() < 0.75:
                candidate_vars = non_base[:] + base_only
            else:
                candidate_vars = piece["variants"][:]
            rng.shuffle(candidate_vars)

        placed = False
        for variant in candidate_vars:
            if _scatter_piece(
                rng=rng,
                input_grid=input_grid,
                placed_occ=placed_occ,
                variant=variant,
                color=int(piece["color"]),
                orientation=orientation,
                bar_r0=int(bar_r0),
                bar_r1=int(bar_r1),
                bar_c0=int(bar_c0),
                bar_c1=int(bar_c1),
            ):
                changed = tuple(variant["rel"]) != piece["base_key"]
                if changed:
                    transformed_piece_count += 1
                    if piece["is_root"]:
                        transformed_root_count += 1
                name = _transform_name(int(variant["rot"]), bool(variant["flip"]))
                transform_hist[name] = int(transform_hist.get(name, 0) + 1)
                placed = True
                break
        if not placed:
            return None

    if transformed_root_count != num_chains:
        return None

    info = _build_generation_metrics(chain_defs, transformed_piece_count, transformed_root_count, transform_hist)
    return input_grid, output, info


def generate_input() -> np.ndarray:
    global _LAST_GEN_INFO
    for attempt in range(300):
        seed = int(attempt * 10007 + np.random.randint(1, 1_000_000))
        result = _gen_attempt(seed)
        if result is None:
            continue
        inp, out, info = result
        if np.array_equal(main(inp), out):
            _LAST_GEN_INFO = dict(info)
            return inp
    raise RuntimeError("generate_input: failed after 300 attempts")
