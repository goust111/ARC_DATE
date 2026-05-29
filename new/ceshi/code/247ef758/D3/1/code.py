# Variant of 247ef758
# Dimension: D3 — 目标位置：框架与标记结构 (dst)
# Change: 框架右侧为矩形框（单色边框+内部背景），边缘标记位于边框像素上，目标为行标记×列标记的笛卡尔积中心点，精灵印刻限制在框架内部
#   → 多个独立框架
#
# Design analysis
# ===============
# source_task_note:
#   - The repository does not contain a solver file for source task 8698868d.
#   - I searched the workspace and only found its ARC json, so the implementation
#     below uses that task's visible "multiple separated panels/containers" idea
#     as a layout reference, without borrowing any solver code.
#
# variant_spec:
#   grid:
#     - fixed height 15 to fit a clean 2x2 slot canvas of independent frames
#     - width 19-20: left sprite strip (3-4 cols) | divider (1 col) | right canvas (15 cols)
#   layout:
#     - left side still contains stacked single-color sprites
#     - one full-height divider column separates sprites from the frame canvas
#     - right canvas contains 2-3 independent 7x7 frames placed in distinct slots
#       among {(top-left), (top-right), (bottom-left), (bottom-right)}
#     - each frame has 5x5 bg interior, one unique base-color border, and markers
#       for exactly one mover color on opposite edges
#   movers:
#     - number of movers == number of frames == 2 or 3
#     - each mover color appears as row/col markers on exactly one frame
#     - each frame may have 1 or 2 row markers and independently 1 or 2 col markers
#     - hence different frames may generate 1, 2, or 4 stamp targets
#   distractors:
#     - optional 0-1 distractor sprites remain in the left strip and never appear on frames
#
# behavior_trace:
#   - Original task:
#       solver finds one largest enclosed bg rectangle, extracts one frame, then
#       uses matching opposite-edge colors inside that frame to place every mover.
#   - Variant:
#       solver must first enumerate *all* enclosed bg rectangles, expand each into
#       an independent frame, and only then collect row/col markers per frame.
#       A mover color is routed to the frame whose opposite-edge markers match it.
#   - Example:
#       if red sprite is on the left and only the top-right frame has red marks on
#       both left/right edges and top/bottom edges, red is erased from the strip and
#       stamped only inside that top-right frame at the Cartesian-product centers.
#       The old single-frame logic would process at most one frame and miss the rest.
#
# complexity_calibration:
#   - Multiple frames are visually obvious: several bordered boxes sit in different
#     slots with distinct base colors.
#   - Routing is still human-solvable because color remains the only assignment key:
#     sprite color matches marker color, but now the solver must also pick the box.
#   - Each frame owns one mover color to avoid ambiguous multi-color overlaps inside
#     a tiny 5x5 interior, while still allowing different marker-count patterns.
#
# implicit_assumption_audit:
#   - `largest_enclosed_component_bbox(...)` from the original is invalid now:
#     there is no single privileged frame.
#   - Restricting sprites to cols < fc0 of that single frame is invalid now:
#     frame positions vary, so the divider must be found directly.
#   - Clipping against one global interior bbox is invalid now:
#     each target must carry its own frame-local interior bounds.
#   - The generator therefore guarantees:
#       * exactly one full-height uniform divider column
#       * every frame interior is a solid bg rectangle fully enclosed by non-bg border
#       * each mover color appears on exactly one frame and has at least one row mark
#         and one col mark there
#       * frames are separated by bg gaps, so enclosed-bg detection is unambiguous
#
# placement_legality_and_visual_cleanliness:
#   - Frames occupy disjoint 7x7 slots separated by one bg row/col gap.
#   - Sprite sizes are capped to fit a 5x5 interior after centering.
#   - Row/col marker spacing is chosen so repeated stamps of the same sprite fit the
#     frame and do not leave the interior.
#   - The right canvas contains only frames and bg, so humans can clearly parse each box.

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np

from arc_utils import get_bg, label_mask


def _find_frames(grid: np.ndarray, bg: int) -> list[dict[str, int]]:
    """Detect all enclosed bg rectangles and return their frame/interior bounds."""
    g = np.array(grid, dtype=int)
    H, W = g.shape
    lab, n = label_mask(g == int(bg), connectivity=4)
    frames: list[dict[str, int]] = []

    for k in range(1, int(n) + 1):
        coords = np.argwhere(lab == int(k))
        if coords.size == 0:
            continue
        rs = coords[:, 0]
        cs = coords[:, 1]
        if bool((rs == 0).any() or (rs == int(H - 1)).any() or (cs == 0).any() or (cs == int(W - 1)).any()):
            continue
        ir0, ic0 = coords.min(axis=0)
        ir1, ic1 = coords.max(axis=0)
        ir0, ic0, ir1, ic1 = map(int, (ir0, ic0, ir1, ic1))
        fr0, fc0, fr1, fc1 = int(ir0 - 1), int(ic0 - 1), int(ir1 + 1), int(ic1 + 1)
        if fr0 < 0 or fc0 < 0 or fr1 >= H or fc1 >= W:
            continue
        if not bool(np.all(g[ir0 : ir1 + 1, ic0 : ic1 + 1] == int(bg))):
            continue

        ring = []
        ring.extend(map(int, g[fr0, fc0 : fc1 + 1]))
        ring.extend(map(int, g[fr1, fc0 : fc1 + 1]))
        ring.extend(map(int, g[ir0 : ir1 + 1, fc0]))
        ring.extend(map(int, g[ir0 : ir1 + 1, fc1]))
        if any(v == int(bg) for v in ring):
            continue

        corners = [int(g[fr0, fc0]), int(g[fr0, fc1]), int(g[fr1, fc0]), int(g[fr1, fc1])]
        base = int(Counter(ring).most_common(1)[0][0])
        if base == int(bg) or any(v != base for v in corners):
            continue

        frames.append(
            {
                "fr0": int(fr0),
                "fc0": int(fc0),
                "fr1": int(fr1),
                "fc1": int(fc1),
                "ir0": int(ir0),
                "ic0": int(ic0),
                "ir1": int(ir1),
                "ic1": int(ic1),
                "base": int(base),
            }
        )

    frames.sort(key=lambda d: (int(d["fr0"]), int(d["fc0"])))
    return frames


def _find_divider_mask(grid: np.ndarray, bg: int) -> tuple[np.ndarray, int]:
    """Find the full-height uniform divider column(s); return mask and left/right split."""
    g = np.array(grid, dtype=int)
    H, W = g.shape
    divider_mask = np.zeros((H, W), dtype=bool)
    divider_cols: list[int] = []
    for c in range(W):
        colv = g[:, int(c)]
        if bool((colv != int(bg)).all()) and len(set(map(int, colv))) == 1:
            divider_mask[:, int(c)] = True
            divider_cols.append(int(c))
    assert divider_cols
    return divider_mask, int(min(divider_cols))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move left-strip sprites into one of several independent marked frames.

    Rule:
    - Detect every enclosed rectangular frame on the right.
    - For each frame separately, read opposite-edge row/col markers.
    - A sprite color moves into the frame that contains matching row+col markers of that color.
    - Stamp the sprite centered at every row_mark × col_mark target inside that frame.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    frames = _find_frames(g, bg)
    assert len(frames) >= 2

    divider_mask, divider_col = _find_divider_mask(g, bg)
    left_cols = np.arange(W)[None, :] < int(divider_col)

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
            m = slab[r0 : r1 + 1, c0 : c1 + 1] == int(sid)
            sprites.append({"color": int(col), "r0": r0, "c0": c0, "r1": r1, "c1": c1, "mask": m})

    move_colors = {int(s["color"]) for s in sprites}
    targets: dict[int, list[tuple[int, int, int, int, int, int]]] = defaultdict(list)

    for frame in frames:
        ir0, ic0, ir1, ic1 = (int(frame[k]) for k in ("ir0", "ic0", "ir1", "ic1"))
        fr0, fc0, fr1, fc1 = (int(frame[k]) for k in ("fr0", "fc0", "fr1", "fc1"))
        row_marks: dict[int, list[int]] = defaultdict(list)
        col_marks: dict[int, list[int]] = defaultdict(list)

        for r in range(ir0, ir1 + 1):
            cl = int(g[int(r), int(fc0)])
            cr = int(g[int(r), int(fc1)])
            if cl == cr and cl != int(bg) and cl in move_colors:
                row_marks[int(cl)].append(int(r))
        for c in range(ic0, ic1 + 1):
            ct = int(g[int(fr0), int(c)])
            cb = int(g[int(fr1), int(c)])
            if ct == cb and ct != int(bg) and ct in move_colors:
                col_marks[int(ct)].append(int(c))

        for col in set(row_marks.keys()) & set(col_marks.keys()):
            for tr in row_marks[int(col)]:
                for tc in col_marks[int(col)]:
                    targets[int(col)].append((int(tr), int(tc), ir0, ic0, ir1, ic1))

    out = g.copy()
    movers = [s for s in sprites if int(s["color"]) in targets]

    for s in movers:
        r0, c0 = int(s["r0"]), int(s["c0"])
        for rr, cc in np.argwhere(s["mask"]):
            out[r0 + int(rr), c0 + int(cc)] = int(bg)

    movers.sort(key=lambda s: (-((int(s["r0"]) + int(s["r1"])) / 2.0), int(s["c0"])))

    for s in movers:
        col = int(s["color"])
        cr = (int(s["r0"]) + int(s["r1"])) // 2
        cc = (int(s["c0"]) + int(s["c1"])) // 2
        pix = [(int(s["r0"]) + int(rr), int(s["c0"]) + int(cc2)) for rr, cc2 in np.argwhere(s["mask"])]

        for tr, tc, ir0, ic0, ir1, ic1 in targets[int(col)]:
            dr = int(tr - cr)
            dc = int(tc - cc)
            for r, c in pix:
                rr = int(r + dr)
                cc3 = int(c + dc)
                if int(ir0) <= rr <= int(ir1) and int(ic0) <= cc3 <= int(ic1):
                    out[rr, cc3] = int(col)

    return out


def _pick_spaced(candidates: list[int], n: int, min_spacing: int) -> list[int] | None:
    """Pick exactly n values from candidates with at least min_spacing between each pair."""
    vals = list(candidates)
    random.shuffle(vals)
    picked: list[int] = []
    for v in vals:
        if all(abs(int(v) - int(p)) >= int(min_spacing) for p in picked):
            picked.append(int(v))
            if len(picked) == int(n):
                return sorted(picked)
    return None


def _make_sprite(color: int, max_h: int, max_w: int) -> np.ndarray | None:
    """Generate a cropped 8-connected single-color sprite that fits a 5x5 destination."""
    sh = min(random.choice([2, 3, 3, 4]), max_h)
    sw = min(random.choice([2, 3, 3, 4]), max_w)
    if sh < 2 or sw < 1:
        return None

    n_pixels = random.randint(3, min(7, sh * sw))
    pixels = {(sh // 2, sw // 2)}
    r, c = sh // 2, sw // 2
    for _ in range(n_pixels * 40):
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
        sprite[pr - min_r, pc - min_c] = int(color)
    return sprite


def _choose_marks(ir0: int, ic0: int, sprite: np.ndarray) -> tuple[list[int], list[int]] | None:
    """Choose 1-2 row marks and 1-2 col marks that keep the centered sprite inside a 5x5 interior."""
    sh, sw = map(int, sprite.shape)
    ir1, ic1 = int(ir0 + 4), int(ic0 + 4)
    min_row = int(ir0 + sh // 2)
    max_row = int(ir1 - (sh - 1 - sh // 2))
    min_col = int(ic0 + sw // 2)
    max_col = int(ic1 - (sw - 1 - sw // 2))
    if min_row > max_row or min_col > max_col:
        return None

    row_candidates = list(range(min_row, max_row + 1))
    col_candidates = list(range(min_col, max_col + 1))

    row_marks = None
    col_marks = None
    if random.random() < 0.6:
        row_marks = _pick_spaced(row_candidates, 2, max(2, sh))
    if row_marks is None:
        row_marks = [random.choice(row_candidates)]

    if random.random() < 0.6:
        col_marks = _pick_spaced(col_candidates, 2, max(2, sw))
    if col_marks is None:
        col_marks = [random.choice(col_candidates)]

    return sorted(map(int, row_marks)), sorted(map(int, col_marks))


def _frame_slots(frame_start_col: int) -> list[tuple[int, int]]:
    """Top-left coordinates of 7x7 frame slots in the right canvas."""
    return [
        (0, int(frame_start_col)),
        (0, int(frame_start_col + 8)),
        (8, int(frame_start_col)),
        (8, int(frame_start_col + 8)),
    ]


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    H = 15
    sprite_area_w = random.randint(3, 4)
    divider_col = int(sprite_area_w)
    frame_start_col = int(divider_col + 1)
    W = int(frame_start_col + 15)
    if not (14 <= W <= 20):
        return None

    n_frames = random.randint(2, 3)
    n_distractors = random.choice([0, 0, 1])

    all_colors = list(range(1, 10))
    random.shuffle(all_colors)
    idx = 0
    divider_color = int(all_colors[idx]); idx += 1
    frame_bases = list(map(int, all_colors[idx : idx + n_frames])); idx += n_frames
    mover_colors = list(map(int, all_colors[idx : idx + n_frames])); idx += n_frames
    distractor_colors = list(map(int, all_colors[idx : idx + n_distractors])); idx += n_distractors

    mover_sprites: list[np.ndarray] = []
    for mc in mover_colors:
        sp = _make_sprite(mc, 4, sprite_area_w)
        if sp is None:
            return None
        if sp.shape[0] > 5 or sp.shape[1] > 5:
            return None
        mover_sprites.append(sp)

    slots = _frame_slots(frame_start_col)
    chosen_slots = random.sample(slots, n_frames)

    frame_infos = []
    multi_target_frames = 0
    for (fr0, fc0), base, mc, sp in zip(chosen_slots, frame_bases, mover_colors, mover_sprites):
        marks = _choose_marks(fr0 + 1, fc0 + 1, sp)
        if marks is None:
            return None
        row_marks, col_marks = marks
        if len(row_marks) * len(col_marks) > 1:
            multi_target_frames += 1
        frame_infos.append(
            {
                "fr0": int(fr0),
                "fc0": int(fc0),
                "fr1": int(fr0 + 6),
                "fc1": int(fc0 + 6),
                "ir0": int(fr0 + 1),
                "ic0": int(fc0 + 1),
                "ir1": int(fr0 + 5),
                "ic1": int(fc0 + 5),
                "base": int(base),
                "mover": int(mc),
                "rows": list(map(int, row_marks)),
                "cols": list(map(int, col_marks)),
            }
        )

    if multi_target_frames == 0:
        return None

    inp = np.full((H, W), bg, dtype=int)
    inp[:, divider_col] = int(divider_color)

    for info in frame_infos:
        fr0, fc0, fr1, fc1 = (int(info[k]) for k in ("fr0", "fc0", "fr1", "fc1"))
        base = int(info["base"])
        mover = int(info["mover"])

        inp[fr0, fc0 : fc1 + 1] = base
        inp[fr1, fc0 : fc1 + 1] = base
        inp[fr0 : fr1 + 1, fc0] = base
        inp[fr0 : fr1 + 1, fc1] = base

        for r in info["rows"]:
            inp[int(r), fc0] = mover
            inp[int(r), fc1] = mover
        for c in info["cols"]:
            inp[fr0, int(c)] = mover
            inp[fr1, int(c)] = mover

    items: list[np.ndarray] = list(mover_sprites)
    for dc in distractor_colors:
        ds = _make_sprite(int(dc), 4, sprite_area_w)
        if ds is None:
            return None
        items.append(ds)
    random.shuffle(items)

    heights = [int(sp.shape[0]) for sp in items]
    needed = sum(heights)
    extra = H - needed
    if extra < 0:
        return None
    gaps = [0] * (len(items) + 1)
    for _ in range(extra):
        gaps[random.randrange(len(gaps))] += 1

    cur_row = gaps[0]
    for i, sp in enumerate(items):
        sh, sw = map(int, sp.shape)
        if cur_row + sh > H or sw > sprite_area_w:
            return None
        c_off = random.randint(0, sprite_area_w - sw)
        for dr in range(sh):
            for dc in range(sw):
                if int(sp[dr, dc]) != int(bg):
                    inp[cur_row + dr, c_off + dc] = int(sp[dr, dc])
        cur_row += sh + gaps[i + 1]

    vals, counts = np.unique(inp, return_counts=True)
    if int(vals[np.argmax(counts)]) != int(bg):
        return None

    try:
        output = main(inp)
    except Exception:
        return None
    if np.array_equal(inp, output):
        return None
    if len(_find_frames(inp, bg)) != n_frames:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 400 attempts")
