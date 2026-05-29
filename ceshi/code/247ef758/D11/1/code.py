# Variant of 247ef758
# Dimension: D11 — 笛卡尔积目标模式 (meta)
# Change: 每个精灵的目标位置是其行标记×列标记的完整笛卡尔积（如2行×2列=4个目标位置）... → 单一目标（行列标记的中位数交点）

"""
Design analysis for the D11 variant
===================================

source_task:
  none

variant_spec:
  unchanged_layout:
    - left sprite strip
    - one full-height divider column
    - one right-side rectangular frame whose interior is the largest enclosed bg area
    - frame edge markers still use mover colors and still appear symmetrically on opposite edges
  changed_rule:
    - original: every mover stamps at the full Cartesian product row_marks[color] x col_marks[color]
    - variant: every mover stamps exactly once at
      (median(row_marks[color]), median(col_marks[color]))
    - odd count median: the middle marked row/col
    - even count median: the midpoint of the two middle marks; the generator enforces same-parity pairs
      so this midpoint is an integer cell and visually centered between the two edge markers
  concrete_trace:
    - if a sprite has row marks [3, 7] and column marks [10, 14]:
      original output positions = (3,10), (3,14), (7,10), (7,14)
      variant output position = (5,12)
    - this keeps the same marker-reading step but changes only the target synthesis step
  complexity_calibration:
    - keep 1-2 row marks and 1-2 column marks per mover so the task stays close to the source task
    - force at least one mover to have 2 row marks and 2 column marks, otherwise the variant could
      collapse to the original 1x1 case and become visually indistinguishable
    - keep colors distinct and row/col marks non-reused across movers to avoid ambiguous matching
  implicit_assumption_audit:
    - collecting row_marks/col_marks by scanning opposite frame edges is still valid
    - the old `for r in rows for c in cols` Cartesian-product code is no longer valid under D11
    - stamping order remains bottom-to-top; this logic depends only on sprite positions, not on the
      number of targets, so it can stay unchanged
    - clipping to the interior remains valid, but the generator must ensure the median target also lies
      in the legal center range for each sprite
    - even-cardinality medians create a new assumption: the midpoint should be an integer grid cell.
      The generator guarantees this by choosing same-parity marker pairs; the solver uses floor division
      defensively if given an arbitrary grid
  placement_legality:
    - for each sprite, marker rows/cols are chosen only from center positions where that sprite can be
      centered without leaving the frame interior
    - reused row/col marks are forbidden across movers, as in the original task
    - median target rows/cols are also kept distinct across movers so single-target placements are less
      likely to collapse onto each other and the new rule remains legible
  visual_observability:
    - at least one mover always exhibits the 2x2-markers -> 1-target behavior
    - when a pair of markers is used, the stamped sprite appears halfway between the two matched edge
      marks instead of appearing at both marked rows/cols; this is directly visible from a few examples
  cleanliness:
    - no new decorative elements are added
    - overlap behavior is preserved but unnecessary target collisions are reduced by distinct median rows
      and columns across movers
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import random
import sys

import numpy as np

try:
    from arc_utils import get_bg, label_mask
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[5] / "solvers"))
    from arc_utils import get_bg, label_mask


def largest_enclosed_component_bbox(mask: np.ndarray, *, connectivity: int = 4) -> tuple[int, int, int, int] | None:
    """
    Local helper (only used by this solver): bbox of largest connected component in `mask`
    that does NOT touch the grid border.
    """
    H, W = mask.shape
    best_area = -1
    best_bbox: tuple[int, int, int, int] | None = None

    lab, n = label_mask(mask.astype(bool), connectivity=int(connectivity))
    for k in range(1, int(n) + 1):
        coords = np.argwhere(lab == int(k))
        if coords.size == 0:
            continue
        rs = coords[:, 0]
        cs = coords[:, 1]
        if bool((rs == 0).any() or (rs == int(H - 1)).any() or (cs == 0).any() or (cs == int(W - 1)).any()):
            continue
        area = int(coords.shape[0])
        if area > best_area:
            best_area = area
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            best_bbox = (int(r0), int(c0), int(r1), int(c1))

    return best_bbox


def _median_mark(marks: list[int]) -> int:
    """Coordinate-wise median; for even counts use the midpoint of the two middle marks."""
    ordered = sorted(map(int, marks))
    assert ordered
    n = len(ordered)
    if n % 2 == 1:
        return int(ordered[n // 2])
    return int((ordered[n // 2 - 1] + ordered[n // 2]) // 2)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move colored sprites from the left strip into the framed area.

    Variant rule:
    - For each mover color, collect all matched row markers and matched column markers exactly as before.
    - Instead of stamping at the full Cartesian product, stamp once at the median row / median column
      intersection. Bottom-to-top covering order is unchanged.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    # Largest bg component not touching boundary = frame interior.
    inner_bbox = largest_enclosed_component_bbox(g == int(bg), connectivity=4)
    assert inner_bbox is not None
    ir0, ic0, ir1, ic1 = map(int, inner_bbox)
    fr0, fc0, fr1, fc1 = int(ir0 - 1), int(ic0 - 1), int(ir1 + 1), int(ic1 + 1)

    # Divider columns on the left side.
    divider_mask = np.zeros((H, W), dtype=bool)
    for c in range(0, max(0, int(fc0))):
        colv = g[:, int(c)]
        non = colv[colv != int(bg)]
        if int(len(non)) >= int(H - 1) and len(set(map(int, non))) == 1:
            divider_mask[:, int(c)] = True

    left_cols = np.arange(W)[None, :] < int(fc0)

    # Per-color 8-connected components in the left area (excluding divider) are movable sprites.
    sprites = []
    for col in sorted(set(map(int, g.flatten())) - {int(bg)}):
        mask = (g == int(col)) & left_cols & (~divider_mask)
        if not bool(mask.any()):
            continue
        slab, sn = label_mask(mask, connectivity=8)
        for sid in range(1, int(sn) + 1):
            pts = np.argwhere(slab == int(sid))
            r0, c0 = pts.min(axis=0)
            r1, c1 = pts.max(axis=0)
            r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
            m = slab[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] == int(sid)
            sprites.append({"color": int(col), "r0": int(r0), "c0": int(c0), "r1": int(r1), "c1": int(c1), "mask": m})

    move_colors = {int(s["color"]) for s in sprites}

    # Marker-derived target rows/cols, restricted to colors that actually appear as sprites.
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

    targets: dict[int, tuple[int, int]] = {}
    for col in set(row_marks.keys()) & set(col_marks.keys()):
        targets[int(col)] = (_median_mark(row_marks[int(col)]), _median_mark(col_marks[int(col)]))

    out = g.copy()
    movers = [s for s in sprites if int(s["color"]) in targets]

    # Erase originals of moved sprites.
    for s in movers:
        r0, c0 = int(s["r0"]), int(s["c0"])
        for rr, cc in np.argwhere(s["mask"]):
            out[int(r0) + int(rr), int(c0) + int(cc)] = int(bg)

    # Lower sprites stamp first, upper sprites stamp later and can cover them.
    movers.sort(key=lambda s: (-((int(s["r0"]) + int(s["r1"])) / 2.0), int(s["c0"])))

    for s in movers:
        col = int(s["color"])
        cr = (int(s["r0"]) + int(s["r1"])) // 2
        cc = (int(s["c0"]) + int(s["c1"])) // 2
        pix = [(int(s["r0"]) + int(rr), int(s["c0"]) + int(cc2)) for rr, cc2 in np.argwhere(s["mask"])]

        tr, tc = targets[int(col)]
        dr = int(tr - cr)
        dc = int(tc - cc)
        for r, c in pix:
            rr = int(r + dr)
            cc3 = int(c + dc)
            if int(ir0) <= rr <= int(ir1) and int(ic0) <= cc3 <= int(ic1):
                out[rr, cc3] = int(col)

    return out


def _make_sprite(color: int, max_h: int, max_w: int) -> np.ndarray | None:
    """Generate a single-color 8-connected sprite cropped to its bounding box."""
    sh = min(random.choice([3, 3, 3, 4, 5]), max_h)
    sw = min(random.choice([3, 3, 3, 4, 5]), max_w)
    if sh < 1 or sw < 1:
        return None
    n_pixels = random.randint(3, min(9, sh * sw))
    pixels = {(sh // 2, sw // 2)}
    r, c = sh // 2, sw // 2
    for _ in range(n_pixels * 30):
        if len(pixels) >= n_pixels:
            break
        dr, dc = random.choice(
            [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        )
        nr, nc = r + dr, c + dc
        if 0 <= nr < sh and 0 <= nc < sw:
            pixels.add((nr, nc))
            r, c = nr, nc
    rows = [p[0] for p in pixels]
    cols = [p[1] for p in pixels]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    sprite = np.zeros((max_r - min_r + 1, max_c - min_c + 1), dtype=int)
    for pr, pc in pixels:
        sprite[pr - min_r, pc - min_c] = color
    return sprite


def _pick_marks(candidates: list[int], n: int, min_spacing: int, used_targets: set[int]) -> list[int] | None:
    """
    Pick 1 or 2 marker coordinates.

    For n == 2, require same parity so the median midpoint is an integer grid cell.
    The resulting median target must not be reused across movers.
    """
    ordered = sorted(set(map(int, candidates)))
    if n == 1:
        options = [[v] for v in ordered if int(v) not in used_targets]
    elif n == 2:
        options = []
        for i, a in enumerate(ordered):
            for b in ordered[i + 1 :]:
                if int(b - a) < int(min_spacing):
                    continue
                if (int(a) + int(b)) % 2 != 0:
                    continue
                median = (int(a) + int(b)) // 2
                if median in used_targets:
                    continue
                options.append([int(a), int(b)])
    else:
        raise ValueError("only 1 or 2 marks are supported in this variant generator")
    if not options:
        return None
    return random.choice(options)


def _gen_attempt() -> np.ndarray | None:
    bg = 0

    # Dimensions: keep the source task's global size regime.
    interior_h = random.randint(8, 13)
    interior_w = random.randint(8, 13)
    H = interior_h + 2
    if not 10 <= H <= 15:
        return None
    sprite_area_w = random.randint(3, 5)
    frame_w = interior_w + 2
    W = sprite_area_w + 1 + frame_w
    if not 14 <= W <= 20:
        return None

    div_col = sprite_area_w
    fc0, fc1 = sprite_area_w + 1, W - 1
    ir0, ic0 = 1, fc0 + 1
    ir1, ic1 = H - 2, fc1 - 1

    all_colors = list(range(1, 10))
    random.shuffle(all_colors)
    n_movers = random.randint(2, 3)
    n_distractors = random.choice([0, 0, 1])
    idx = 0
    divider_color = all_colors[idx]
    idx += 1
    frame_base = all_colors[idx]
    idx += 1
    mover_colors = all_colors[idx : idx + n_movers]
    idx += n_movers
    distractor_colors = all_colors[idx : idx + n_distractors]

    # Ensure one mover can visibly demonstrate the D11 change: 2 row marks + 2 column marks -> 1 target.
    focus_index = random.randrange(n_movers)
    sprites: list[np.ndarray] = []
    for i, mc in enumerate(mover_colors):
        max_h = 4 if i == focus_index else 5
        max_w = min(4, sprite_area_w) if i == focus_index else sprite_area_w
        s = _make_sprite(mc, max_h, max_w)
        if s is None:
            return None
        sprites.append(s)

    mover_targets: dict[int, tuple[list[int], list[int]]] = {}
    occupied_rows: set[int] = set()
    occupied_cols: set[int] = set()
    occupied_target_rows: set[int] = set()
    occupied_target_cols: set[int] = set()

    for i, (mc, sp) in enumerate(zip(mover_colors, sprites)):
        sh, sw = sp.shape
        min_row = ir0 + sh // 2
        max_row = ir1 - (sh - 1 - sh // 2)
        min_col = ic0 + sw // 2
        max_col = ic1 - (sw - 1 - sw // 2)
        if min_row > max_row or min_col > max_col:
            return None

        row_count = 2 if i == focus_index else random.choice([1, 2])
        col_count = 2 if i == focus_index else random.choice([1, 2])

        avail_rows = [r for r in range(min_row, max_row + 1) if r not in occupied_rows]
        row_marks = _pick_marks(avail_rows, row_count, sh, occupied_target_rows)
        if row_marks is None:
            return None

        avail_cols = [c for c in range(min_col, max_col + 1) if c not in occupied_cols]
        col_marks = _pick_marks(avail_cols, col_count, sw, occupied_target_cols)
        if col_marks is None:
            return None

        mover_targets[mc] = (sorted(row_marks), sorted(col_marks))
        occupied_rows.update(row_marks)
        occupied_cols.update(col_marks)
        occupied_target_rows.add(_median_mark(row_marks))
        occupied_target_cols.add(_median_mark(col_marks))

    inp = np.full((H, W), bg, dtype=int)
    inp[:, div_col] = divider_color

    top_edge = np.full(frame_w, frame_base, dtype=int)
    for mc, (_, cmarks) in mover_targets.items():
        for cc in cmarks:
            top_edge[int(cc - fc0)] = int(mc)
    inp[0, fc0 : fc0 + frame_w] = top_edge
    inp[H - 1, fc0 : fc0 + frame_w] = top_edge

    for r in range(1, H - 1):
        val = frame_base
        for mc, (rmarks, _) in mover_targets.items():
            if r in rmarks:
                val = mc
                break
        inp[r, fc0] = val
        inp[r, fc1] = val

    cur_row = 0
    all_items: list[tuple[int, np.ndarray]] = [(mc, sp) for mc, sp in zip(mover_colors, sprites)]
    for dc_color in distractor_colors:
        ds = _make_sprite(dc_color, min(5, H - cur_row), sprite_area_w)
        if ds is None:
            return None
        all_items.append((dc_color, ds))

    for _color, sp in all_items:
        sh, sw = sp.shape
        if sw > sprite_area_w or cur_row + sh > H:
            return None
        c_off = random.randint(0, sprite_area_w - sw)
        for dr in range(sh):
            for dc in range(sw):
                if sp[dr, dc] != bg:
                    inp[cur_row + dr, c_off + dc] = sp[dr, dc]
        cur_row += sh

    vals, counts = np.unique(inp, return_counts=True)
    if int(vals[np.argmax(counts)]) != bg:
        return None

    try:
        output = main(inp)
    except Exception:
        return None
    if np.array_equal(inp, output):
        return None

    # Guard against invisible degeneration: at least one mover must truly use 2x2 markers.
    if not any(len(rmarks) == 2 and len(cmarks) == 2 for rmarks, cmarks in mover_targets.values()):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 300 attempts")
