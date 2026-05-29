# Combo variant of a25697e4 (index 3)
# combo_spec: D2/0+D6/2
# Name: 纯旋转奇偶变色
#
# Design analysis for this combo
# - Concrete behavior and trace:
#   The seed task has border frames with staircase cavities plus detached 2-color
#   tools. A tool is matched to a frame by cavity area, then some D4 transform is
#   used so one color region fits the cavity and the other stays outside. In the
#   seed task, color swap depends on whether the winning docking transform uses a
#   reflection. In this combo both affected stages change together:
#   1) D2/0 removes reflections entirely. Only 0/90/180/270-degree rotations may
#      dock the tool back into a frame.
#   2) D6/2 makes color swap depend on rotation parity instead of reflection:
#      odd quarter-turn counts (90/270) swap the two tool colors; even counts
#      (0/180) preserve them.
#   Small trace: suppose a cavity has area 6 and the detached tool has a 6-cell
#   red plug region plus a 7-cell blue handle. If the input tool must rotate 90
#   degrees clockwise to fit, the geometric plug still fills the cavity, but the
#   output colors become swapped: the cavity ends up blue and the outside handle
#   red. If another pair in the same grid docks with a 180-degree rotation, that
#   pair keeps its original plug/handle colors. Relative to the seed task, the
#   visible cue is no longer "flip happened" but "odd quarter-turn happened".
# - Complexity calibration:
#   If the generator allowed only even rotations, the combo would collapse to a
#   simpler rotation-only variant with no visible color change. If it allowed 90
#   degree rotationally symmetric tools, odd/even parity could become ambiguous.
#   The generator therefore forces at least one odd-rotation docking per sample,
#   uses two pairs often enough to show odd and even behavior side by side, and
#   rejects tools whose colored pattern is identical under an odd and an even
#   rotation. Handle and plug areas remain distinct, preserving unambiguous frame
#   pairing by cavity size.
# - Implicit-assumption audit of the seed solver:
#   The seed `gen_variants()` assumed "orientation identity" was fully captured by
#   `(mask bytes, colored bytes)` because only flip-vs-no-flip affected colors.
#   That is false here: two identical-looking rotations with different parity can
#   imply different output colors, so parity must be tracked explicitly. The seed
#   `_undock_tool()` also assumed only flipped input orientations need a pre-swap
#   when constructing the detached tool. Here the inverse construction must pre-
#   swap whenever the undocking rotation parity is odd, otherwise re-docking would
#   swap twice to the wrong colors. Finally, every branch that referenced `is_flip`
#   must now reason about `rot_k % 2`.
# - Placement legality:
#   Frame construction still follows the original output-first strategy, so all
#   frame pixels, cavities, and docked tools are placed while checking in-bounds
#   border anchoring and bbox separation. Detached tools are then placed with
#   `random_free_location(...)` on a canvas that already contains the frames and
#   docked output state, preventing overlap and border contact. The generator only
#   samples pure rotations, never reflections, and verifies `main(input)==output`
#   before accepting a sample.
# - Visual observability:
#   A human can still pair tools to frames by matching the staircase plug shape
#   and cavity area. The new visible regularity is that tools requiring a quarter-
#   turn swap their two colors, while tools requiring a half-turn or no turn keep
#   them. Because the generator always includes at least one odd case and often an
#   odd/even contrast within the same grid, the parity-color relation is exposed
#   directly in a few examples rather than hidden in rare cases.
# - Visual cleanliness:
#   The combo preserves the original scene grammar: sparse border frames, one or
#   two detached tools, and no extra markers. No region is overloaded with new
#   annotations; the only change is the final color assignment of the same two
#   regions after docking. This keeps the output readable while still making the
#   combo behavior visible.
# - Color-rule audit and anti-degeneration metric:
#   The only color mapping is a full swap of the tool's two existing colors, so a
#   human can infer it from the plug/handle partition rather than from raw color
#   ids. After implementation, the useful metric is:
#     metric A = number of placed tools whose chosen docking rotation is odd
#     metric B = whether combo output differs from a rotation-only/no-swap ablation
#   The generator guarantees metric A >= 1 for every sample, so metric B should be
#   true for every accepted sample as well; otherwise the combo would have fallen
#   back to a non-visible subset of the original behavior.
# - New solver assumptions and generator guarantees:
#   The solver now assumes each detached tool has exactly two colors and that its
#   intended docking parity is not ambiguous between an odd and an even rotation.
#   `_place_docked_pair()` rejects odd/even rotation collisions, `_pick_pair_specs`
#   keeps plug area distinct from handle area, and `_gen_attempt()` chooses target
#   undocking parities so every accepted sample exercises the combo-specific rule.

from __future__ import annotations

from collections import Counter
import itertools
import random
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import label

try:
    from arc_utils import STRUCT4, get_bg, paste, random_free_location, random_sprite
    from arc_utils import bbox as _bbox
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "arc2_solver_select"))
    from arc_utils import STRUCT4, get_bg, paste, random_free_location, random_sprite
    from arc_utils import bbox as _bbox


def _bbox_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    (ar0, ac0, ar1, ac1) = a
    (br0, bc0, br1, bc1) = b

    def _gap1(x0: int, x1: int, y0: int, y1: int) -> int:
        if y1 < x0:
            return int(x0 - y1)
        if x1 < y0:
            return int(y0 - x1)
        return 0

    return _gap1(ar0, ar1, br0, br1) + _gap1(ac0, ac1, bc0, bc1)


def _swap_colors(patch: np.ndarray, a: int, b: int) -> np.ndarray:
    out = patch.copy()
    ma = out == int(a)
    mb = out == int(b)
    out[ma] = int(b)
    out[mb] = int(a)
    return out


def _rotation_signature(arr: np.ndarray, bg: int, k: int) -> tuple[bytes, bytes]:
    q = np.rot90(arr, k)
    m = q != int(bg)
    return m.tobytes(), q[m].tobytes()


def _has_odd_even_rotation_collision(arr: np.ndarray, bg: int) -> bool:
    even = {_rotation_signature(arr, bg, 0), _rotation_signature(arr, bg, 2)}
    odd = {_rotation_signature(arr, bg, 1), _rotation_signature(arr, bg, 3)}
    return bool(even & odd)


def _rotation_variants_for_solver(
    patch: np.ndarray, mask: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray, int]]:
    uniq: list[tuple[np.ndarray, np.ndarray, int]] = []
    seen: set[tuple[bytes, bytes, int]] = set()
    for k in range(4):
        p = np.rot90(patch, k)
        m = np.rot90(mask, k)
        key = (m.tobytes(), p[m].tobytes(), int(k % 2))
        if key in seen:
            continue
        seen.add(key)
        uniq.append((p, m, int(k)))
    return uniq


def _extract_anchor_groups(g: np.ndarray, bg: int) -> tuple[np.ndarray, list[dict[str, object]]]:
    H, W = g.shape
    fg = g != int(bg)
    lab, n = label(fg, structure=STRUCT4)
    fixed_mask = np.zeros((H, W), dtype=bool)
    fixed_components: list[dict[str, object]] = []
    for cid in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(cid))
        pts_list = [(int(r), int(c)) for r, c in pts]
        if not any(r == 0 or r == H - 1 or c == 0 or c == W - 1 for r, c in pts_list):
            continue
        fixed_components.append(
            {
                "pts": pts_list,
                "bbox": (
                    int(pts[:, 0].min()),
                    int(pts[:, 1].min()),
                    int(pts[:, 0].max()),
                    int(pts[:, 1].max()),
                ),
            }
        )
        for r, c in pts_list:
            fixed_mask[r, c] = True

    assert fixed_components

    parent = list(range(len(fixed_components)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(fixed_components)):
        for j in range(i + 1, len(fixed_components)):
            if _bbox_gap(
                fixed_components[i]["bbox"], fixed_components[j]["bbox"]  # type: ignore[arg-type]
            ) <= 2:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(len(fixed_components)):
        groups.setdefault(find(i), []).append(i)

    anchor_groups: list[dict[str, object]] = []
    for idxs in groups.values():
        pts = [p for i in idxs for p in fixed_components[i]["pts"]]  # type: ignore[index]
        rs = [r for r, _ in pts]
        cs = [c for _, c in pts]
        bbox = (min(rs), min(cs), max(rs), max(cs))
        mask = np.zeros((H, W), dtype=bool)
        for r, c in pts:
            mask[r, c] = True

        sock = np.zeros((H, W), dtype=bool)
        for r in range(bbox[0], bbox[2] + 1):
            for c in range(bbox[1], bbox[3] + 1):
                if int(g[r, c]) != int(bg):
                    continue
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < H and 0 <= cc < W and mask[rr, cc]:
                        sock[r, c] = True
                        break

        cavity = np.zeros((H, W), dtype=bool)
        sub_bg = g[bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1] == int(bg)
        sub_lab, sub_n = label(sub_bg, structure=STRUCT4)
        best_id = None
        best_c = None
        for cid in range(1, int(sub_n) + 1):
            ptsc = np.argwhere(sub_lab == int(cid))
            area = int(len(ptsc))
            adj = False
            for rr, cc in ptsc:
                r = int(bbox[0] + rr)
                c = int(bbox[1] + cc)
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    r2, c2 = r + dr, c + dc
                    if bbox[0] <= r2 <= bbox[2] and bbox[1] <= c2 <= bbox[3] and mask[r2, c2]:
                        adj = True
                        break
                if adj:
                    break
            if not adj:
                continue
            cand = (area, -int(cid))
            if best_c is None or cand > best_c:
                best_c = cand
                best_id = int(cid)
        if best_id is not None:
            ptsc = np.argwhere(sub_lab == int(best_id))
            for rr, cc in ptsc:
                cavity[int(bbox[0] + rr), int(bbox[1] + cc)] = True

        dir_list = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        dir_counts = [0, 0, 0, 0]
        for r, c in pts:
            for di, (dr, dc) in enumerate(dir_list):
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == int(bg):
                    dir_counts[di] += 1
        open_dir = dir_list[int(np.argmax(dir_counts))]

        anchor_groups.append(
            {
                "mask": mask,
                "bbox": bbox,
                "socket_mask": sock,
                "cavity_mask": cavity,
                "open_dir": open_dir,
                "cavity_size": int(cavity.sum()),
                "size": int(mask.sum()),
            }
        )

    anchor_groups.sort(key=lambda d: -int(d["size"]))
    return fixed_mask, anchor_groups


def _extract_movable_components(
    g: np.ndarray, bg: int, fixed_mask: np.ndarray
) -> list[dict[str, object]]:
    fg = g != int(bg)
    mov_mask = fg & (~fixed_mask)
    mov_lab, mov_n = label(mov_mask, structure=STRUCT4)
    mov_comps: list[dict[str, object]] = []
    for cid in range(1, int(mov_n) + 1):
        pts = np.argwhere(mov_lab == int(cid))
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        m = mov_lab[r0 : r1 + 1, c0 : c1 + 1] == int(cid)
        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        patch[~m] = 0
        mov_comps.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "mask": m,
                "patch": patch,
                "size": int(len(pts)),
            }
        )
    mov_comps.sort(key=lambda d: -int(d["size"]))
    return mov_comps


def _solve_grid(
    grid: np.ndarray, *, swap_on_odd_rotation: bool, collect_meta: bool = False
) -> np.ndarray | tuple[np.ndarray, list[dict[str, int]]]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape
    fixed_mask, anchor_groups = _extract_anchor_groups(g, bg)
    mov_comps = _extract_movable_components(g, bg, fixed_mask)

    def choose_group(bbox: tuple[int, int, int, int]) -> int:
        best_i = 0
        best_d = None
        for i, grp in enumerate(anchor_groups):
            d = _bbox_gap(bbox, grp["bbox"])  # type: ignore[arg-type]
            if best_d is None or d < best_d:
                best_d = d
                best_i = i
        return int(best_i)

    comp_color_counts: list[dict[int, int]] = []
    for comp in mov_comps:
        vals = [int(x) for x in comp["patch"][comp["mask"]].flatten() if int(x) != 0]  # type: ignore[index]
        comp_color_counts.append(dict(Counter(vals)))

    def compatible(ci: int, gi: int) -> bool:
        cav_sz = int(anchor_groups[gi]["cavity_size"])
        return any(int(v) == cav_sz for v in comp_color_counts[ci].values())

    if len(mov_comps) == len(anchor_groups) and len(anchor_groups) > 1:
        best_perm = None
        best_score = None
        for perm in itertools.permutations(range(len(anchor_groups))):
            hits = 0
            dist_sum = 0
            ok = True
            for ci, gi in enumerate(perm):
                if not compatible(ci, gi):
                    ok = False
                    break
                hits += 1
                dist_sum += _bbox_gap(
                    mov_comps[ci]["bbox"], anchor_groups[gi]["bbox"]  # type: ignore[arg-type]
                )
            if not ok:
                continue
            score = (int(hits), -int(dist_sum))
            if best_score is None or score > best_score:
                best_score = score
                best_perm = perm
        assert best_perm is not None
        group_for_comp = [int(x) for x in best_perm]
    else:
        group_for_comp: list[int] = []
        for ci, comp in enumerate(mov_comps):
            best_g = None
            best_d = None
            for gi, grp in enumerate(anchor_groups):
                if not compatible(ci, gi):
                    continue
                d = _bbox_gap(comp["bbox"], grp["bbox"])  # type: ignore[arg-type]
                if best_d is None or d < best_d:
                    best_d = d
                    best_g = gi
            group_for_comp.append(
                int(best_g) if best_g is not None else int(choose_group(comp["bbox"]))  # type: ignore[arg-type]
            )

    out = np.full_like(g, int(bg))
    out[fixed_mask] = g[fixed_mask]
    placed_global = fixed_mask.copy()
    placed_groups = [grp["mask"].copy() for grp in anchor_groups]  # type: ignore[index]
    meta: list[dict[str, int]] = []

    for comp_index, (comp, gi) in enumerate(zip(mov_comps, group_for_comp)):
        grp = anchor_groups[gi]
        placed_grp = placed_groups[gi]
        sock = grp["socket_mask"]  # type: ignore[assignment]
        cav = grp["cavity_mask"]  # type: ignore[assignment]
        odr, odc = grp["open_dir"]  # type: ignore[misc]
        cav_sz = int(grp["cavity_size"])

        comp_vals = [int(x) for x in comp["patch"][comp["mask"]].flatten() if int(x) != 0]  # type: ignore[index]
        comp_counts = Counter(comp_vals)
        comp_colors = sorted(comp_counts.keys())
        assert len(comp_colors) == 2

        best_state = None
        best_score = None
        for p, m, rot_k in _rotation_variants_for_solver(comp["patch"], comp["mask"]):  # type: ignore[arg-type]
            h, w = m.shape
            coords = np.argwhere(m)
            cols_here = sorted({int(x) for x in p[m].flatten() if int(x) != 0})
            plug_colors = [int(col) for col in cols_here if int(((p == int(col)) & m).sum()) == cav_sz]
            assert plug_colors
            for r0 in range(0, H - h + 1):
                for c0 in range(0, W - w + 1):
                    ok = True
                    for rr, cc in coords:
                        r = int(r0 + rr)
                        c = int(c0 + cc)
                        if placed_global[r, c]:
                            ok = False
                            break
                    if not ok:
                        continue

                    adj = 0
                    socket_hits = 0
                    cavity_hits = 0
                    outside = 0
                    br0, bc0, br1, bc1 = grp["bbox"]  # type: ignore[misc]
                    for rr, cc in coords:
                        r = int(r0 + rr)
                        c = int(c0 + cc)
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            r2, c2 = r + dr, c + dc
                            if 0 <= r2 < H and 0 <= c2 < W and placed_grp[r2, c2]:
                                adj += 1
                        if sock[r, c]:
                            socket_hits += 1
                        if cav[r, c]:
                            cavity_hits += 1
                        if odc == -1 and c < bc0:
                            outside += 1
                        elif odc == 1 and c > bc1:
                            outside += 1
                        elif odr == -1 and r < br0:
                            outside += 1
                        elif odr == 1 and r > br1:
                            outside += 1
                    if adj == 0:
                        continue

                    plug_best = None
                    for plug_col in plug_colors:
                        plug_hits = 0
                        plug_outside = 0
                        nonplug_in_cav = 0
                        for rr, cc in coords:
                            r = int(r0 + rr)
                            c = int(c0 + cc)
                            in_cav = bool(cav[r, c])
                            col = int(p[rr, cc])
                            if col == int(plug_col):
                                if in_cav:
                                    plug_hits += 1
                                else:
                                    plug_outside += 1
                            elif in_cav:
                                nonplug_in_cav += 1
                        plug_full = int(
                            plug_hits == cav_sz and plug_outside == 0 and nonplug_in_cav == 0
                        )
                        plug_score = (
                            int(plug_full),
                            int(plug_hits),
                            -int(nonplug_in_cav),
                            -int(plug_outside),
                        )
                        if plug_best is None or plug_score > plug_best:
                            plug_best = plug_score
                    assert plug_best is not None

                    dist = _bbox_gap(
                        grp["bbox"],  # type: ignore[arg-type]
                        (int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1)),
                    )
                    score = tuple(plug_best) + (
                        int(cavity_hits),
                        int(adj),
                        int(outside),
                        int(socket_hits),
                        -int(dist),
                        -int(r0),
                        -int(c0),
                    )
                    if best_score is None or score > best_score:
                        best_score = score
                        best_state = (p, m, int(r0), int(c0), int(rot_k))

        assert best_state is not None
        p, m, r0, c0, rot_k = best_state
        if swap_on_odd_rotation and rot_k % 2 == 1:
            p = _swap_colors(p, comp_colors[0], comp_colors[1])

        for rr, cc in np.argwhere(m):
            r = int(r0 + rr)
            c = int(c0 + cc)
            out[r, c] = int(p[rr, cc])
            placed_global[r, c] = True
            placed_grp[r, c] = True

        meta.append(
            {
                "component_index": int(comp_index),
                "group_index": int(gi),
                "rotation_k": int(rot_k),
                "rotation_parity": int(rot_k % 2),
                "cavity_size": int(cav_sz),
            }
        )

    if collect_meta:
        return out, meta
    return out


def _solve_without_parity_swap(grid: np.ndarray) -> np.ndarray:
    return _solve_grid(grid, swap_on_odd_rotation=False)  # type: ignore[return-value]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reattach detached two-color tools to border frames using rotation-only docking.

    Input structure:
    - The background is the most frequent color.
    - Border-touching same-color pieces form frames whose interior bg cavities are staircase sockets.
    - Interior 2-color connected components are movable tools made of a cavity-sized plug and an outward handle.

    Rule:
    - Pair each tool with the frame whose cavity size matches one of the tool's two color-region sizes.
    - Dock the tool using only 0/90/180/270-degree rotations so that one region fills the cavity exactly and the other remains outside the frame.
    - After choosing the winning rotation, swap the tool's two colors iff the rotation count is odd (90 or 270 degrees); even rotations keep the colors unchanged.

    Key insight: area still identifies the matching frame, but the final plug/handle colors are determined by rotation parity rather than by any reflection.
    """
    return _solve_grid(grid, swap_on_odd_rotation=True)  # type: ignore[return-value]


# --------------- generator ---------------

# ── CFG: canvas ──
GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (25, 30)
NUM_PAIRS_CHOICES = [1, 2, 2]

# ── CFG: frame geometry ──
FRAME_W_RANGE = (4, 8)
NOTCH_N_RANGE = (1, 4)
TOP_H_RANGE = (1, 4)
BOT_H_RANGE = (2, 4)
PLUG_AREA_BOUNDS = (5, 8)

# ── CFG: tool sizing ──
HANDLE_AREA_RANGE = (4, 10)
TOOL_TOTAL_BOUNDS = (10, 15)
HANDLE_SPRITE_DIM = (2, 5)

# ── CFG: placement ──
UNION_BBOX_GAP_MIN = 2
FRAME_BBOX_GAP_MIN = 3
TOOL_BORDER = 1
TOOL_PADDING = 3
TOOL_PADDING_CONN = 4

# ── CFG: retry limits ──
PAIR_SPEC_RETRIES = 300
HANDLE_SPRITE_RETRIES = 30
GEN_ATTEMPT_RETRIES = 300


def _transform(arr: np.ndarray, k: int) -> np.ndarray:
    return np.rot90(arr, k)


def _orient_left_to_side(arr: np.ndarray, side: str) -> np.ndarray:
    if side == "left":
        return arr.copy()
    if side == "right":
        return np.fliplr(arr)
    if side == "top":
        return np.rot90(arr, 3)
    if side == "bottom":
        return np.rot90(arr, 1)
    raise ValueError(side)


def _notch_offsets(notch_n: int, fw: int, bot_h: int) -> list[tuple[int, int]]:
    if notch_n == 1:
        return [(0, fw - 1)]
    if notch_n == 2:
        return [(0, fw - 2), (0, fw - 1)]
    if notch_n == 3:
        if bot_h >= 2:
            return [(0, fw - 2), (0, fw - 1), (1, fw - 1)]
        return [(0, fw - 3), (0, fw - 2), (0, fw - 1)]
    raise ValueError(notch_n)


def _make_handle_sprite(handle_area: int, bg: int, handle_color: int) -> np.ndarray | None:
    lo, hi = HANDLE_SPRITE_DIM
    dims = [
        (h, w)
        for h in range(lo, hi)
        for w in range(lo, hi)
        if handle_area <= h * w <= handle_area * 2
    ]
    if not dims:
        return None
    for _ in range(HANDLE_SPRITE_RETRIES):
        h, w = random.choice(dims)
        sprite = random_sprite(
            h,
            w,
            density=handle_area / (h * w),
            color_palette=[handle_color],
            connectivity=4,
            background=bg,
        )
        if int(np.sum(sprite != bg)) == handle_area:
            return sprite
    return None


def _build_left_pair(
    *,
    bg: int,
    frame_color: int,
    plug_color: int,
    handle_color: int,
    fw: int,
    top_h: int,
    bot_h: int,
    notch_n: int,
    handle_area: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    fh = top_h + 1 + bot_h
    frame = np.full((fh, fw), bg, dtype=int)
    frame[:top_h, :] = frame_color
    frame[top_h + 1 :, :] = frame_color
    for dr, dc in _notch_offsets(notch_n, fw, bot_h):
        frame[top_h + 1 + dr, dc] = bg

    cavity = frame == bg
    attach_rows = {r for r in range(fh) if cavity[r, fw - 1]}

    handle_sprite = _make_handle_sprite(handle_area, bg, handle_color)
    if handle_sprite is None:
        return None

    hs_h, hs_w = handle_sprite.shape
    sprite_left_rows = {r for r in range(hs_h) if handle_sprite[r, 0] != bg}
    valid_offsets = [
        voff for voff in range(fh) if any((voff + sr) in attach_rows for sr in sprite_left_rows)
    ]
    if not valid_offsets:
        return None

    voff = random.choice(valid_offsets)
    total_h = max(fh, voff + hs_h)
    total_w = fw + hs_w

    frame_full = np.full((total_h, total_w), bg, dtype=int)
    frame_full[:fh, :fw] = frame

    tool = np.full((total_h, total_w), bg, dtype=int)
    tool[:fh, :fw][cavity] = plug_color
    handle_mask = handle_sprite != bg
    tool[voff : voff + hs_h, fw : fw + hs_w][handle_mask] = handle_color

    return frame_full, tool


def _build_param_options() -> list[tuple[int, int, int, int]]:
    fw_lo, fw_hi = FRAME_W_RANGE
    nn_lo, nn_hi = NOTCH_N_RANGE
    th_lo, th_hi = TOP_H_RANGE
    bh_lo, bh_hi = BOT_H_RANGE
    pa_lo, pa_hi = PLUG_AREA_BOUNDS
    return [
        (fw, top_h, bot_h, notch_n)
        for fw in range(fw_lo, fw_hi)
        for notch_n in range(nn_lo, nn_hi)
        if pa_lo <= fw + notch_n <= pa_hi
        for top_h in range(th_lo, th_hi)
        for bot_h in range(bh_lo, bh_hi)
    ]


PARAM_OPTIONS = _build_param_options()


def _pick_pair_specs(num_pairs: int) -> list[dict[str, int]]:
    ha_lo, ha_hi = HANDLE_AREA_RANGE
    tt_lo, tt_hi = TOOL_TOTAL_BOUNDS
    specs: list[dict[str, int]] = []
    for _ in range(PAIR_SPEC_RETRIES):
        specs.clear()
        used_plug_areas: set[int] = set()
        for _pair in range(num_pairs):
            candidates = [x for x in PARAM_OPTIONS if x[0] + x[3] not in used_plug_areas]
            if not candidates:
                specs = []
                break
            fw, top_h, bot_h, notch_n = random.choice(candidates)
            plug_area = fw + notch_n
            specs.append(
                {
                    "fw": int(fw),
                    "top_h": int(top_h),
                    "bot_h": int(bot_h),
                    "notch_n": int(notch_n),
                    "plug_area": int(plug_area),
                }
            )
            used_plug_areas.add(int(plug_area))
        if len(specs) != num_pairs:
            continue
        all_plug_areas = {int(s["plug_area"]) for s in specs}
        ok = True
        for spec in specs:
            pa = int(spec["plug_area"])
            handle_choices = [
                h
                for h in range(ha_lo, ha_hi)
                if tt_lo <= pa + h <= tt_hi and h != pa and h not in all_plug_areas
            ]
            if not handle_choices:
                ok = False
                break
            spec["handle_area"] = int(random.choice(handle_choices))
        if ok:
            return [dict(s) for s in specs]
    return []


def _candidate_origins(side: str, ph: int, pw: int, H: int, W: int) -> list[tuple[int, int]]:
    if side == "left":
        origins = [(r0, 0) for r0 in range(H - ph + 1)]
    elif side == "right":
        origins = [(r0, W - pw) for r0 in range(H - ph + 1)]
    elif side == "top":
        origins = [(0, c0) for c0 in range(W - pw + 1)]
    elif side == "bottom":
        origins = [(H - ph, c0) for c0 in range(W - pw + 1)]
    else:
        raise ValueError(side)
    random.shuffle(origins)
    return origins


def _rotation_choices_for_input(patch: np.ndarray, bg: int) -> list[tuple[np.ndarray, int]]:
    seen: set[tuple[bytes, bytes, int]] = set()
    out: list[tuple[np.ndarray, int]] = []
    for k in range(4):
        q = _transform(patch, k)
        m = q != int(bg)
        key = (m.tobytes(), q[m].tobytes(), int(k % 2))
        if key in seen:
            continue
        seen.add(key)
        out.append((q, int(k)))
    random.shuffle(out)
    return out


def _place_docked_pair(
    i: int,
    spec: dict[str, int],
    colors: list[int],
    side: str,
    bg: int,
    H: int,
    W: int,
    output: np.ndarray,
    input_grid: np.ndarray,
    union_bboxes: list[tuple[int, int, int, int]],
    frame_bboxes: list[tuple[int, int, int, int]],
) -> dict[str, object] | None:
    frame_color = int(colors[3 * i])
    plug_color = int(colors[3 * i + 1])
    handle_color = int(colors[3 * i + 2])

    result = _build_left_pair(
        bg=bg,
        frame_color=frame_color,
        plug_color=plug_color,
        handle_color=handle_color,
        fw=int(spec["fw"]),
        top_h=int(spec["top_h"]),
        bot_h=int(spec["bot_h"]),
        notch_n=int(spec["notch_n"]),
        handle_area=int(spec["handle_area"]),
    )
    if result is None:
        return None
    frame_left, tool_left = result

    scene_left = np.where(tool_left != bg, tool_left, frame_left)
    frame_scene = _orient_left_to_side(frame_left, side)
    tool = _orient_left_to_side(tool_left, side)
    scene = _orient_left_to_side(scene_left, side)
    if _has_odd_even_rotation_collision(tool, bg):
        return None

    ph, pw = scene.shape
    fr0, fc0, fr1, fc1 = _bbox(frame_scene != bg)

    origin = None
    for r0, c0 in _candidate_origins(side, ph, pw, H, W):
        union_bbox = (r0, c0, r0 + ph - 1, c0 + pw - 1)
        frame_bbox = (r0 + fr0, c0 + fc0, r0 + fr1, c0 + fc1)
        if side in ("left", "right") and (frame_bbox[0] == 0 or frame_bbox[2] == H - 1):
            continue
        if side in ("top", "bottom") and (frame_bbox[1] == 0 or frame_bbox[3] == W - 1):
            continue
        if any(_bbox_gap(union_bbox, prev) <= UNION_BBOX_GAP_MIN - 1 for prev in union_bboxes):
            continue
        if any(_bbox_gap(frame_bbox, prev) <= FRAME_BBOX_GAP_MIN - 1 for prev in frame_bboxes):
            continue
        if np.any((output[r0 : r0 + ph, c0 : c0 + pw] != bg) & (scene != bg)):
            continue
        origin = (int(r0), int(c0))
        break
    if origin is None:
        return None

    r0, c0 = origin
    paste(output, scene, r0, c0, transparent=bg)
    paste(input_grid, frame_scene, r0, c0, transparent=bg)

    union_bboxes.append((r0, c0, r0 + ph - 1, c0 + pw - 1))
    frame_bboxes.append((r0 + fr0, c0 + fc0, r0 + fr1, c0 + fc1))
    return {
        "tool": tool,
        "plug_color": int(plug_color),
        "handle_color": int(handle_color),
        "plug_area": int(spec["plug_area"]),
        "handle_area": int(spec["handle_area"]),
    }


def _undock_tool(
    info: dict[str, object],
    bg: int,
    input_grid: np.ndarray,
    placement_canvas: np.ndarray,
    required_parity: int,
) -> bool:
    tool = info["tool"]  # type: ignore[assignment]
    candidates = _rotation_choices_for_input(tool, bg)
    preferred = [x for x in candidates if int(x[1] % 2) == int(required_parity)]
    if not preferred:
        return False
    for patch, k in preferred:
        q = patch
        if int(k % 2) == 1:
            q = _swap_colors(q, int(info["plug_color"]), int(info["handle_color"]))
        try:
            r0, c0 = random_free_location(
                placement_canvas,
                q,
                transparent=bg,
                border=TOOL_BORDER,
                padding=TOOL_PADDING,
                padding_connectivity=TOOL_PADDING_CONN,
            )
        except ValueError:
            continue
        paste(input_grid, q, r0, c0, transparent=bg)
        paste(placement_canvas, q, r0, c0, transparent=bg)
        info["input_rotation_k"] = int(k)
        info["input_rotation_parity"] = int(k % 2)
        return True
    return False


def _required_parities(num_pairs: int) -> list[int]:
    if num_pairs == 1:
        return [1]
    order = [1, 0]
    random.shuffle(order)
    return order


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    bg = random.choice(list(range(10)))
    num_pairs = random.choice(NUM_PAIRS_CHOICES)

    pair_specs = _pick_pair_specs(num_pairs)
    if len(pair_specs) != num_pairs:
        return None

    colors = [c for c in range(1, 10) if c != bg]
    random.shuffle(colors)
    if len(colors) < 1 + (3 * num_pairs):
        return None

    sides = random.sample(["left", "right", "top", "bottom"], k=num_pairs)
    output = np.full((H, W), bg, dtype=int)
    input_grid = np.full((H, W), bg, dtype=int)
    union_bboxes: list[tuple[int, int, int, int]] = []
    frame_bboxes: list[tuple[int, int, int, int]] = []
    placed_tools: list[dict[str, object]] = []

    for i, spec in enumerate(pair_specs):
        info = _place_docked_pair(
            i,
            spec,
            colors,
            sides[i],
            bg,
            H,
            W,
            output,
            input_grid,
            union_bboxes,
            frame_bboxes,
        )
        if info is None:
            return None
        placed_tools.append(info)

    placement_canvas = output.copy()
    parities = _required_parities(num_pairs)
    for info, parity in zip(placed_tools, parities):
        if not _undock_tool(info, bg, input_grid, placement_canvas, parity):
            return None

    solved, meta = _solve_grid(input_grid, swap_on_odd_rotation=True, collect_meta=True)  # type: ignore[misc]
    if not np.array_equal(solved, output):
        return None
    if sum(int(m["rotation_parity"]) for m in meta) < 1:
        return None
    if np.array_equal(_solve_without_parity_swap(input_grid), output):
        return None
    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPT_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_ATTEMPT_RETRIES} attempts")
