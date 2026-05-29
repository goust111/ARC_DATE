# Variant of 247ef758
# Dimension: D2 — 源对象：精灵类型与属性 (src)
# Change: 左侧窄条中2-3个可移动精灵（8连通单色色块，3-9像素，外接矩形≤5x5）加0-1个干扰精灵，颜色各不相同... → 精灵分布在框架四周而非仅左侧

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import get_bg, label_mask


# Design analysis for the D2 variant
# - Source task: none, so the only baseline is the original 247ef758 solver/generator.
# - a) Concrete behavior:
#   Original inputs force every movable sprite into columns left of the frame, separated by a
#   full-height divider. The solver can therefore ignore the rest of the canvas and search only
#   `c < fc0`. In this variant the frame is moved away from the borders and every sprite sits in
#   one of four exterior side zones: above, below, left of, or right of the frame. The output
#   behavior inside the frame is unchanged: matching edge markers define target rows/cols, each
#   mover is erased from its source location, then stamped at the Cartesian-product targets, and
#   lower source sprites are stamped first so higher ones cover them.
#   Small trace:
#   old: frame at right edge, sprite color 4 found only in left strip -> erase left-strip 4 ->
#        stamp 4 at its row/col marker intersections.
#   new: frame centered with margins, sprite color 4 may sit above the frame -> erase that top
#        sprite from the whole exterior area -> stamp 4 at the same marker intersections.
#   The only rule change is where the source sprite is searched for.
# - b) Complexity calibration:
#   To make the change visible, movers are forced onto distinct sides. With 2-4 total sprites this
#   yields 2-4 occupied side zones and guarantees the input is not visually compatible with the old
#   "single left strip" heuristic. Colors remain unique to avoid source/target ambiguity.
# - c) Original hidden assumptions that break under the variant:
#   `left_cols = cols < fc0` assumes every sprite is left of the frame. Invalid now.
#   `divider_mask` assumes there is a uniform divider column. Removed.
#   Sprite extraction from `(g == color) & left_cols` would miss top/bottom/right sprites. Replaced
#   with extraction from the full exterior of the detected frame rectangle.
#   Erasing movers still works, but only if extraction returns the true exterior components.
# - d) Placement legality:
#   The generator keeps a one-cell background moat between frame and sprites. Each sprite gets its
#   own side zone, so sources never overlap each other or the frame. Target row/col markers are
#   still chosen so each stamped sprite fits in the frame interior before clipping.
# - e) Visual observability:
#   Humans can now see that source blobs are not confined to a left strip. Because movers occupy
#   multiple exterior sides, the solver must infer "search outside the frame" rather than "read the
#   left margin". The replicated stamping rule remains unchanged and therefore learnable.
# - f) Visual cleanliness:
#   One sprite per side zone prevents crowding. The one-cell moat avoids source blobs visually
#   merging with frame edges or marker colors.
# - g) Color inference:
#   Colors still serve only as explicit matches between source sprites and mirrored frame-edge
#   markers. No numeric color ordering or hidden color mapping is introduced.
# - h) New solver assumptions after the change:
#   There is exactly one exterior 8-connected component per sprite color, and the only non-bg cells
#   outside the frame are sprites. The generator enforces both conditions. The frame interior is the
#   largest enclosed background component; the generator keeps the frame as the only large enclosed
#   hole.


def largest_enclosed_component_bbox(
    mask: np.ndarray, *, connectivity: int = 4
) -> tuple[int, int, int, int] | None:
    """
    BBox of the largest connected component in `mask` that does not touch the grid border.
    """
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
        if bool((rs == 0).any() or (rs == int(h - 1)).any() or (cs == 0).any() or (cs == int(w - 1)).any()):
            continue
        area = int(coords.shape[0])
        if area > best_area:
            best_area = area
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            best_bbox = (int(r0), int(c0), int(r1), int(c1))
    return best_bbox


def _extract_sprites_outside_frame(
    grid: np.ndarray, bg: int, frame_bbox: tuple[int, int, int, int]
) -> list[dict[str, object]]:
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
    Find sprites anywhere outside the frame, then stamp them into the frame by color-matched markers.
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
        pts = {(int(r), int(c)) for r in row_marks[int(color)] for c in col_marks[int(color)]}
        targets[int(color)] = sorted(pts)

    out = g.copy()
    movers = [s for s in sprites if int(s["color"]) in targets]

    for s in movers:
        r0, c0 = int(s["r0"]), int(s["c0"])
        for rr, cc in np.argwhere(np.asarray(s["mask"], dtype=bool)):
            out[int(r0) + int(rr), int(c0) + int(cc)] = int(bg)

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
    """Pick up to n values from candidates with at least min_spacing between chosen values."""
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
    """Generate a cropped single-color 8-connected sprite with 3-9 pixels and bbox <= 5x5."""
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
    grid: np.ndarray, sprite: np.ndarray, side: str, fr0: int, fc0: int, fr1: int, fc1: int
) -> tuple[int, int] | None:
    zone = _side_zone(side, int(grid.shape[0]), int(grid.shape[1]), int(fr0), int(fc0), int(fr1), int(fc1))
    if zone is None:
        return None
    zr0, zr1, zc0, zc1 = zone
    sh, sw = sprite.shape
    top_max = int(zr1 - sh + 1)
    left_max = int(zc1 - sw + 1)
    top_min = int(zr0)
    left_min = int(zc0)
    if side in {"top", "bottom"}:
        # Keep top/bottom sprites visually tied to the frame instead of drifting into a faux left strip.
        left_min = max(int(left_min), int(fc0 - sw + 1))
        left_max = min(int(left_max), int(fc1))
    if top_min > top_max or left_min > left_max:
        return None
    top = random.randint(int(top_min), top_max)
    left = random.randint(int(left_min), left_max)
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
    frame_base = all_colors[idx]
    idx += 1
    mover_colors = all_colors[idx : idx + n_movers]
    idx += n_movers
    distractor_colors = all_colors[idx : idx + n_distractors]
    idx += n_distractors

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

    if not any(spec["side"] != "left" for spec in mover_specs):
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
