# Combo variant of 247ef758 (index 11)
# combo_spec: D2/2+D8/0
# Name: 无分隔线+精灵分散分布

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import get_bg, label_mask


# Design analysis for the combo variant
# a) Concrete behavior and trace:
#    - Original task: sprites sit in a left strip, a full-height colored divider marks the split,
#      and the solver can safely read only columns left of the frame. Each sprite color that also
#      appears as mirrored row marks and mirrored column marks on the frame is erased from the
#      source area and stamped into the frame interior at every row_mark x col_mark target.
#    - Combo behavior: the divider disappears and sprites can be above, below, left of, or right
#      of the frame. The frame is no longer flush with the grid border; instead it sits inside a
#      larger background field with a one-cell moat. A color-4 sprite might now be above the frame:
#      old pipeline would miss it because `c < fc0` is false, while the combo solver first finds
#      the frame via its enclosed background cavity and then scans the entire exterior region.
#    - Visible difference: the input no longer contains any full-height separator column, and the
#      source blobs are not concentrated in a single strip. The output rule inside the frame stays
#      the same, so the fusion is "same stamping logic, different segmentation problem".
# b) Complexity calibration:
#    - Movers occupy distinct exterior side zones chosen from {top, bottom, left, right}; this
#      makes the distribution visually obvious without exploding the search space.
#    - There are still only 2-3 movers and 0-1 distractors, all with distinct colors, so the
#      color matching story remains simple and non-ambiguous.
#    - Marker counts stay at 1-2 rows x 1-2 cols per mover, which is enough repetition for humans
#      to infer the Cartesian-product stamping rule from a few examples.
# c) Hidden assumptions removed from the original solver:
#    - The divider scan assumed a near-solid non-bg column; invalid because D8/0 removes it.
#    - `left_cols = cols < fc0` assumed every sprite is left of the frame; invalid because D2/2
#      allows top/bottom/right sprites.
#    - Extracting sprites by color only works if frame-edge marker pixels of the same color are
#      excluded, so the combo solver masks out the entire frame rectangle before component labeling.
#    - `largest_enclosed_component_bbox(g == bg)` still works, but only because the generator keeps
#      the frame as the unique enclosed hole and ensures every outside gap touches the border.
# d) Placement legality:
#    - The generator uses disjoint side zones separated from the frame by a one-cell background
#      moat, so source sprites never overlap the frame or each other.
#    - Marker rows and columns are spaced using sprite dimensions so each centered stamp fits in
#      the frame interior before clipping.
# e) Visual observability:
#    - Every sample has no divider-like column and has movers occupying at least two different
#      exterior sides, so a human can see that "search only on the left" is no longer valid.
#    - Colors keep their original explicit role: a sprite color is meaningful only when mirrored on
#      both opposite frame edges.
# f) Visual cleanliness:
#    - One sprite per side zone avoids clutter. Top/bottom sprites are kept roughly aligned over
#      the frame span so the canvas reads as "objects around a central frame" rather than a random
#      scatter of blobs.
# g) Change inferability:
#    - No hidden color remapping is introduced. Distinct colors, repeated mirrored markers, and the
#      unchanged stamping rule give enough structure for few-shot inference.
# h) New solver assumptions and how the generator enforces them:
#    - Assumption: all non-bg cells outside the frame belong to sprite components. Enforced by
#      drawing nothing else outside the frame.
#    - Assumption: each sprite color appears as exactly one exterior component. Enforced by placing
#      at most one sprite per color.
#    - Assumption: the frame interior is the largest enclosed bg component. Enforced by keeping the
#      frame rectangular and leaving all exterior background connected to the border.


def largest_enclosed_component_bbox(
    mask: np.ndarray, *, connectivity: int = 4
) -> tuple[int, int, int, int] | None:
    """BBox of the largest connected component in `mask` that does not touch the border."""
    h, w = mask.shape
    best_area = -1
    best_bbox: tuple[int, int, int, int] | None = None

    lab, n = label_mask(mask.astype(bool), connectivity=int(connectivity))
    for k in range(1, int(n) + 1):
        coords = np.argwhere(lab == int(k))
        if coords.size == 0:
            continue
        rs = coords[:, 0]
        cs = coords[:, 1]
        if bool(
            (rs == 0).any()
            or (rs == int(h - 1)).any()
            or (cs == 0).any()
            or (cs == int(w - 1)).any()
        ):
            continue
        area = int(coords.shape[0])
        if area > best_area:
            best_area = area
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            best_bbox = (int(r0), int(c0), int(r1), int(c1))
    return best_bbox


def _extract_sprites_outside_frame(
    grid: np.ndarray,
    bg: int,
    frame_bbox: tuple[int, int, int, int],
) -> list[dict[str, object]]:
    """Return per-color 8-connected sprite components strictly outside the frame rectangle."""
    fr0, fc0, fr1, fc1 = map(int, frame_bbox)
    outside = np.ones(grid.shape, dtype=bool)
    outside[int(fr0) : int(fr1) + 1, int(fc0) : int(fc1) + 1] = False

    sprites: list[dict[str, object]] = []
    for color in sorted(set(map(int, grid.flatten())) - {int(bg)}):
        mask = (grid == int(color)) & outside
        if not bool(mask.any()):
            continue
        lab, n = label_mask(mask, connectivity=8)
        for sid in range(1, int(n) + 1):
            pts = np.argwhere(lab == int(sid))
            r0, c0 = pts.min(axis=0)
            r1, c1 = pts.max(axis=0)
            r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
            component = lab[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] == int(sid)
            sprites.append(
                {
                    "color": int(color),
                    "r0": int(r0),
                    "c0": int(c0),
                    "r1": int(r1),
                    "c1": int(c1),
                    "mask": component,
                }
            )
    return sprites


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the central frame, find sprites anywhere outside it, and stamp movers at the
    Cartesian-product targets defined by mirrored edge markers of the same color.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    inner_bbox = largest_enclosed_component_bbox(g == int(bg), connectivity=4)
    assert inner_bbox is not None
    ir0, ic0, ir1, ic1 = map(int, inner_bbox)
    fr0, fc0, fr1, fc1 = int(ir0 - 1), int(ic0 - 1), int(ir1 + 1), int(ic1 + 1)

    sprites = _extract_sprites_outside_frame(g, int(bg), (int(fr0), int(fc0), int(fr1), int(fc1)))
    move_colors = {int(s["color"]) for s in sprites}

    row_marks: dict[int, list[int]] = defaultdict(list)
    col_marks: dict[int, list[int]] = defaultdict(list)
    for r in range(int(ir0), int(ir1) + 1):
        cl = int(g[int(r), int(fc0)])
        cr = int(g[int(r), int(fc1)])
        if cl == cr and cl != int(bg) and cl in move_colors:
            row_marks[int(cl)].append(int(r))
    for c in range(int(ic0), int(ic1) + 1):
        ct = int(g[int(fr0), int(c)])
        cb = int(g[int(fr1), int(c)])
        if ct == cb and ct != int(bg) and ct in move_colors:
            col_marks[int(ct)].append(int(c))

    targets: dict[int, list[tuple[int, int]]] = {}
    for color in set(row_marks.keys()) & set(col_marks.keys()):
        targets[int(color)] = sorted(
            {(int(r), int(c)) for r in row_marks[int(color)] for c in col_marks[int(color)]}
        )

    out = g.copy()
    movers = [s for s in sprites if int(s["color"]) in targets]

    for s in movers:
        r0, c0 = int(s["r0"]), int(s["c0"])
        for rr, cc in np.argwhere(np.asarray(s["mask"], dtype=bool)):
            out[int(r0) + int(rr), int(c0) + int(cc)] = int(bg)

    # Lower sprites stamp first so higher sprites cover them, matching the source task.
    movers.sort(key=lambda s: (-((int(s["r0"]) + int(s["r1"])) / 2.0), int(s["c0"])))

    for s in movers:
        color = int(s["color"])
        cr = (int(s["r0"]) + int(s["r1"])) // 2
        cc = (int(s["c0"]) + int(s["c1"])) // 2
        pix = [
            (int(s["r0"]) + int(rr), int(s["c0"]) + int(cc2))
            for rr, cc2 in np.argwhere(np.asarray(s["mask"], dtype=bool))
        ]
        for tr, tc in targets[int(color)]:
            dr = int(tr - cr)
            dc = int(tc - cc)
            for r, c in pix:
                rr = int(r + dr)
                cc3 = int(c + dc)
                if int(ir0) <= rr <= int(ir1) and int(ic0) <= cc3 <= int(ic1):
                    out[int(rr), int(cc3)] = int(color)

    return out


def _pick_spaced(candidates: list[int], n: int, min_spacing: int) -> list[int] | None:
    """Pick up to n values with at least `min_spacing` between chosen values."""
    if not candidates:
        return None
    if n == 1:
        return [random.choice(candidates)]
    random.shuffle(candidates)
    picked: list[int] = []
    for value in candidates:
        if all(abs(int(value) - int(prev)) >= int(min_spacing) for prev in picked):
            picked.append(int(value))
            if len(picked) == int(n):
                return picked
    return picked if picked else None


def _make_sprite(color: int, max_h: int, max_w: int) -> np.ndarray | None:
    """Generate one cropped 8-connected monochrome sprite with 3-9 pixels and bbox <= 5x5."""
    max_h = min(5, int(max_h))
    max_w = min(5, int(max_w))
    if max_h < 1 or max_w < 1 or max_h * max_w < 3:
        return None

    dims = [(h, w) for h in range(1, max_h + 1) for w in range(1, max_w + 1) if h * w >= 3]
    if not dims:
        return None

    for _ in range(80):
        sh, sw = random.choice(dims)
        n_pixels = random.randint(3, min(9, sh * sw))
        pixels = {(sh // 2, sw // 2)}
        for _ in range(n_pixels * 40):
            if len(pixels) >= n_pixels:
                break
            pr, pc = random.choice(tuple(pixels))
            dr, dc = random.choice(
                [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
            )
            nr, nc = pr + dr, pc + dc
            if 0 <= nr < sh and 0 <= nc < sw:
                pixels.add((int(nr), int(nc)))
        if len(pixels) < 3:
            continue
        rows = [p[0] for p in pixels]
        cols = [p[1] for p in pixels]
        min_r, max_r = min(rows), max(rows)
        min_c, max_c = min(cols), max(cols)
        sprite = np.zeros((max_r - min_r + 1, max_c - min_c + 1), dtype=int)
        for pr, pc in pixels:
            sprite[int(pr - min_r), int(pc - min_c)] = int(color)
        return sprite
    return None


def _side_zone(
    side: str, h: int, w: int, fr0: int, fc0: int, fr1: int, fc1: int
) -> tuple[int, int, int, int] | None:
    if side == "top":
        zone = (0, int(fr0) - 2, 0, int(w) - 1)
    elif side == "bottom":
        zone = (int(fr1) + 2, int(h) - 1, 0, int(w) - 1)
    elif side == "left":
        zone = (int(fr0), int(fr1), 0, int(fc0) - 2)
    elif side == "right":
        zone = (int(fr0), int(fr1), int(fc1) + 2, int(w) - 1)
    else:
        raise ValueError(f"unknown side: {side}")

    zr0, zr1, zc0, zc1 = zone
    if zr0 > zr1 or zc0 > zc1:
        return None
    return zone


def _place_sprite_in_zone(
    grid: np.ndarray,
    sprite: np.ndarray,
    side: str,
    fr0: int,
    fc0: int,
    fr1: int,
    fc1: int,
) -> tuple[int, int] | None:
    zone = _side_zone(side, int(grid.shape[0]), int(grid.shape[1]), int(fr0), int(fc0), int(fr1), int(fc1))
    if zone is None:
        return None
    zr0, zr1, zc0, zc1 = zone
    sh, sw = sprite.shape
    top_min = int(zr0)
    top_max = int(zr1 - sh + 1)
    left_min = int(zc0)
    left_max = int(zc1 - sw + 1)
    if side in {"top", "bottom"}:
        left_min = max(int(left_min), int(fc0 - sw + 1))
        left_max = min(int(left_max), int(fc1))
    if top_min > top_max or left_min > left_max:
        return None
    top = random.randint(int(top_min), int(top_max))
    left = random.randint(int(left_min), int(left_max))
    for dr in range(int(sh)):
        for dc in range(int(sw)):
            if int(sprite[int(dr), int(dc)]) != 0:
                grid[int(top) + int(dr), int(left) + int(dc)] = int(sprite[int(dr), int(dc)])
    return int(top), int(left)


def _gen_attempt() -> np.ndarray | None:
    bg = 0

    top_pad = random.choice([2, 2, 3])
    bottom_pad = random.choice([2, 2, 3])
    left_pad = random.choice([2, 3, 4])
    right_pad = random.choice([2, 3, 4])
    interior_h = random.randint(6, 8)
    interior_w = random.randint(7, 10)

    frame_h = int(interior_h + 2)
    frame_w = int(interior_w + 2)
    h = int(top_pad + frame_h + bottom_pad)
    w = int(left_pad + frame_w + right_pad)
    if not (10 <= h <= 15 and 14 <= w <= 20):
        return None

    fr0, fc0 = int(top_pad), int(left_pad)
    fr1, fc1 = int(fr0 + frame_h - 1), int(fc0 + frame_w - 1)
    ir0, ic0 = int(fr0 + 1), int(fc0 + 1)
    ir1, ic1 = int(fr1 - 1), int(fc1 - 1)

    all_colors = list(range(1, 10))
    random.shuffle(all_colors)
    n_movers = random.randint(2, 3)
    n_distractors = random.choice([0, 0, 1])
    idx = 0
    frame_base = int(all_colors[idx])
    idx += 1
    mover_colors = [int(v) for v in all_colors[idx : idx + n_movers]]
    idx += n_movers
    distractor_colors = [int(v) for v in all_colors[idx : idx + n_distractors]]

    sides = ["top", "bottom", "left", "right"]
    mover_sides = random.sample(sides, k=n_movers)
    remaining_sides = [side for side in sides if side not in mover_sides]
    distractor_sides = random.sample(remaining_sides, k=n_distractors)

    mover_specs: list[dict[str, object]] = []
    for color, side in zip(mover_colors, mover_sides):
        zone = _side_zone(side, int(h), int(w), int(fr0), int(fc0), int(fr1), int(fc1))
        if zone is None:
            return None
        zr0, zr1, zc0, zc1 = zone
        sprite = _make_sprite(int(color), int(zr1 - zr0 + 1), int(zc1 - zc0 + 1))
        if sprite is None:
            return None
        mover_specs.append({"color": int(color), "side": side, "sprite": sprite})

    distractor_specs: list[dict[str, object]] = []
    for color, side in zip(distractor_colors, distractor_sides):
        zone = _side_zone(side, int(h), int(w), int(fr0), int(fc0), int(fr1), int(fc1))
        if zone is None:
            return None
        zr0, zr1, zc0, zc1 = zone
        sprite = _make_sprite(int(color), int(zr1 - zr0 + 1), int(zc1 - zc0 + 1))
        if sprite is None:
            return None
        distractor_specs.append({"color": int(color), "side": side, "sprite": sprite})

    mover_targets: dict[int, tuple[list[int], list[int]]] = {}
    occupied_rows: set[int] = set()
    occupied_cols: set[int] = set()
    for spec in mover_specs:
        sprite = np.asarray(spec["sprite"], dtype=int)
        sh, sw = sprite.shape
        min_row = int(ir0 + sh // 2)
        max_row = int(ir1 - (sh - 1 - sh // 2))
        min_col = int(ic0 + sw // 2)
        max_col = int(ic1 - (sw - 1 - sw // 2))
        if min_row > max_row or min_col > max_col:
            return None
        avail_rows = [r for r in range(min_row, max_row + 1) if r not in occupied_rows]
        avail_cols = [c for c in range(min_col, max_col + 1) if c not in occupied_cols]
        row_marks = _pick_spaced(avail_rows, random.randint(1, 2), sh)
        col_marks = _pick_spaced(avail_cols, random.randint(1, 2), sw)
        if not row_marks or not col_marks:
            return None
        mover_targets[int(spec["color"])] = (sorted(row_marks), sorted(col_marks))
        occupied_rows.update(row_marks)
        occupied_cols.update(col_marks)

    inp = np.full((h, w), bg, dtype=int)

    inp[int(fr0), int(fc0) : int(fc1) + 1] = int(frame_base)
    inp[int(fr1), int(fc0) : int(fc1) + 1] = int(frame_base)
    inp[int(fr0) : int(fr1) + 1, int(fc0)] = int(frame_base)
    inp[int(fr0) : int(fr1) + 1, int(fc1)] = int(frame_base)

    for color, (_rmarks, cmarks) in mover_targets.items():
        for col in cmarks:
            inp[int(fr0), int(col)] = int(color)
            inp[int(fr1), int(col)] = int(color)
    for color, (rmarks, _cmarks) in mover_targets.items():
        for row in rmarks:
            inp[int(row), int(fc0)] = int(color)
            inp[int(row), int(fc1)] = int(color)

    for spec in mover_specs + distractor_specs:
        top_left = _place_sprite_in_zone(
            inp,
            np.asarray(spec["sprite"], dtype=int),
            str(spec["side"]),
            int(fr0),
            int(fc0),
            int(fr1),
            int(fc1),
        )
        if top_left is None:
            return None

    vals, counts = np.unique(inp, return_counts=True)
    if int(vals[np.argmax(counts)]) != bg:
        return None

    # Fusion guards: no divider-like separator and movers truly distributed beyond a left strip.
    for c in range(int(w)):
        colv = inp[:, int(c)]
        non = colv[colv != int(bg)]
        if int(len(non)) >= int(h - 1) and len(set(map(int, non))) == 1:
            return None
    if not any(str(spec["side"]) != "left" for spec in mover_specs):
        return None
    if len({str(spec["side"]) for spec in mover_specs}) < 2:
        return None

    try:
        output = main(inp)
    except Exception:
        return None
    if np.array_equal(inp, output):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 400 attempts")
