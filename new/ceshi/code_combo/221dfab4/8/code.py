# Combo variant of 221dfab4 (index 8)
# combo_spec: D2/1+D12/2
# Name: GCD间距+内嵌绿色编码周期

from __future__ import annotations

from pathlib import Path
import itertools
import math
import random
import sys

import numpy as np
from scipy import ndimage

HELPER_DIR = Path(__file__).resolve().parents[4] / "arc2_solver_select"
if str(HELPER_DIR) not in sys.path:
    sys.path.append(str(HELPER_DIR))

from arc_utils import get_bg, label_mask, random_sprite


# Design analysis for the combo variant
# a) Concrete behavior and trace:
#    Original 221dfab4 reads one solid yellow border segment, then paints stripes at rows/cols
#    `start, start+2, start+4, ...` with a fixed cycle Y,Y,G. Here both parameters are hidden.
#    Example with a top clue `4 4 3 4 4 3` and interior blob lead distances `3, 6, 12, 15`:
#    the solver reads the clue as cycle [Y,Y,G,Y,Y,G], measures the blob distances along the
#    downward axis, computes gcd(3,6,12,15)=3, then paints rows 0,3,6,9,... instead of 0,2,4,...
#    Rows 1-2, 4-5, 7-8, ... inside the segment span become background gaps. Whenever the decoded
#    cycle says green, the whole row/column's existing non-background pixels are recolored green.
#    The difference from the base task is therefore visible in two places at once: stripe spacing
#    no longer matches "every other row/col", and green stripes no longer follow the fixed 3-step
#    cadence because they are keyed to embedded green clue pixels.
# b) Complexity calibration:
#    The generator only uses stride values in {3,4,5}; this keeps the path visibly different from
#    the baseline stride-2 behavior without exploding the hypothesis space. The code cycle uses a
#    5-7 cell clue with 1-2 embedded greens, which is enough to create repeated evidence across the
#    grid while still being readable by eye as a short periodic code rather than a free-form string.
# c) Hidden assumptions audited against the original solver:
#    - `smask = (g == 4) & border` is no longer valid because clue cells may be green; border code
#      detection must look for the largest straight {3,4} border component and verify it is gap-free.
#    - `cycle = [4,4,3]` is no longer valid; the entire clue run is now semantic data.
#    - `pos = start + dir * (2 * step)` hard-codes stride 2; the new solver must decode stride from
#      interior components before marching.
#    - Clearing exactly one gap cell depended on stride 2; with variable stride we must clear all
#      intermediate positions between successive stripes inside the clue span.
#    - The old generator treated every non-yellow foreground as a blob. That would be ambiguous now
#      because green exists in the clue, so blob detection excludes both 3 and 4.
#    No new conflict is introduced if the generator guarantees: one straight border code only, at
#    least two interior foreground components, and no interior pixels of colors 3/4.
# d) Placement legality:
#    The generator places trimmed sprites one by one at exact lead distances chosen to encode the
#    stride, while scanning orthogonal offsets that stay inside a 1-cell border and avoid overlap
#    under a 1-cell 8-neighborhood padding mask. This keeps blobs disjoint, in-bounds, and away
#    from the clue itself.
# e) Visual observability:
#    A human can recover the rule from a few examples because the same two signals repeat. First,
#    blob leading edges line up at distances that share a non-trivial gcd, and the output stripes
#    honor that spacing rather than the original every-other-line spacing. Second, the clue itself
#    visibly contains green exceptions inside an otherwise yellow border run, and output green rows/
#    columns recur with that clue's period rather than the old Y,Y,G baseline.
# f) New solver assumptions and how the generator satisfies them:
#    - `_locate_border_code` assumes the {3,4} border component is a single straight contiguous run.
#      The generator writes exactly one such run and keeps corners empty.
#    - `_decode_stride` assumes at least two non-{3,4} foreground components exist and that their
#      lead distances have a meaningful gcd > 1. The generator places 5 disjoint blobs whose chosen
#      distances are multiples of one sampled stride and include distance = stride, forcing the gcd.
#    - `_apply_rule` assumes the decoded stride is positive and that the clue span lies on one edge.
#      Those are guaranteed by the construction and rechecked by round-tripping `main(inp)`.

Y = 4
G = 3


def _border_mask(shape: tuple[int, int]) -> np.ndarray:
    H, W = shape
    border = np.zeros((H, W), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return border


def _locate_border_code(g: np.ndarray) -> dict | None:
    H, W = g.shape
    smask = _border_mask((H, W)) & np.isin(g, [Y, G])
    if not bool(smask.any()):
        return None

    lab, n = label_mask(smask, connectivity=4)
    if int(n) <= 0:
        return None

    best_id = max(range(1, int(n) + 1), key=lambda cid: int((lab == int(cid)).sum()))
    pts = np.argwhere(lab == int(best_id))
    rs = pts[:, 0]
    cs = pts[:, 1]
    r0, r1 = int(rs.min()), int(rs.max())
    c0, c1 = int(cs.min()), int(cs.max())

    if r0 == r1:
        coords = sorted(int(c) for c in cs)
        if coords != list(range(int(coords[0]), int(coords[-1]) + 1)):
            return None
        cycle = [int(g[int(r0), int(c)]) for c in coords]
        if any(col not in (Y, G) for col in cycle) or Y not in cycle:
            return None
        return {
            "orient": "h",
            "start_pos": int(r0),
            "span": coords,
            "cycle": cycle,
            "dir1": 1 if int(r0) == 0 else -1,
            "limit": H,
        }

    if c0 == c1:
        coords = sorted(int(r) for r in rs)
        if coords != list(range(int(coords[0]), int(coords[-1]) + 1)):
            return None
        cycle = [int(g[int(r), int(c0)]) for r in coords]
        if any(col not in (Y, G) for col in cycle) or Y not in cycle:
            return None
        return {
            "orient": "v",
            "start_pos": int(c0),
            "span": coords,
            "cycle": cycle,
            "dir1": 1 if int(c0) == 0 else -1,
            "limit": W,
        }

    return None


def _stripe_positions(start_pos: int, dir1: int, limit: int, stride: int) -> list[int]:
    positions: list[int] = []
    step = 0
    while True:
        pos = int(start_pos) + int(dir1) * (int(stride) * int(step))
        if not (0 <= pos < int(limit)):
            break
        positions.append(int(pos))
        step += 1
    return positions


def _component_lead_distance(coords: np.ndarray, info: dict, shape: tuple[int, int]) -> int:
    H, W = shape
    if info["orient"] == "h":
        if int(info["dir1"]) > 0:
            return int(coords[:, 0].min())
        return int(H - 1 - coords[:, 0].max())
    if int(info["dir1"]) > 0:
        return int(coords[:, 1].min())
    return int(W - 1 - coords[:, 1].max())


def _decode_stride(g: np.ndarray, info: dict, bg: int) -> int | None:
    fg_mask = (g != int(bg)) & ~np.isin(g, [Y, G])
    lab, n = label_mask(fg_mask, connectivity=4)
    if int(n) < 2:
        return None

    distances: list[int] = []
    for cid in range(1, int(n) + 1):
        coords = np.argwhere(lab == int(cid))
        if coords.size == 0:
            continue
        dist = _component_lead_distance(coords, info, g.shape)
        if dist > 0:
            distances.append(int(dist))

    distances = sorted(set(distances))
    if len(distances) < 2:
        return None

    nums = list(distances)
    nums.extend(int(b - a) for a, b in zip(distances, distances[1:]) if int(b - a) > 0)
    stride = 0
    for val in nums:
        stride = math.gcd(int(stride), int(val))
    return int(stride) if int(stride) > 0 else None


def _apply_rule(grid: np.ndarray, info: dict, bg: int, stride: int, cycle: list[int]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    span = np.array(info["span"], dtype=int)

    for step, pos in enumerate(_stripe_positions(info["start_pos"], info["dir1"], info["limit"], stride)):
        col = int(cycle[int(step) % len(cycle)])

        if info["orient"] == "h":
            if col == G:
                out[int(pos), :] = np.where(out[int(pos), :] != int(bg), int(G), out[int(pos), :])
            out[int(pos), span] = int(col)
            for off in range(1, int(stride)):
                gap = int(pos) + int(info["dir1"]) * int(off)
                if 0 <= gap < out.shape[0]:
                    out[int(gap), span] = int(bg)
        else:
            if col == G:
                out[:, int(pos)] = np.where(out[:, int(pos)] != int(bg), int(G), out[:, int(pos)])
            out[span, int(pos)] = int(col)
            for off in range(1, int(stride)):
                gap = int(pos) + int(info["dir1"]) * int(off)
                if 0 <= gap < out.shape[1]:
                    out[span, int(gap)] = int(bg)

    return out


def _baseline_mismatch_count(cycle: list[int], n_steps: int) -> int:
    baseline = [Y, Y, G]
    return sum(int(cycle[i % len(cycle)] != baseline[i % len(baseline)]) for i in range(int(n_steps)))


def _visible_green_floods(inp: np.ndarray, info: dict, bg: int, stride: int) -> int:
    H, W = inp.shape
    span = set(int(x) for x in info["span"])
    count = 0
    for step, pos in enumerate(_stripe_positions(info["start_pos"], info["dir1"], info["limit"], stride)):
        if int(info["cycle"][step % len(info["cycle"])]) != G:
            continue
        if info["orient"] == "h":
            if any(int(inp[int(pos), c]) not in (int(bg), Y, G) for c in range(W) if c not in span):
                count += 1
        else:
            if any(int(inp[r, int(pos)]) not in (int(bg), Y, G) for r in range(H) if r not in span):
                count += 1
    return int(count)


def _variant_metrics(inp: np.ndarray) -> dict | None:
    g = np.array(inp, dtype=int)
    bg = int(get_bg(g))
    info = _locate_border_code(g)
    if info is None:
        return None
    stride = _decode_stride(g, info, bg)
    if stride is None:
        return None
    positions = _stripe_positions(info["start_pos"], info["dir1"], info["limit"], stride)
    green_steps = sum(int(info["cycle"][i % len(info["cycle"])] == G) for i in range(len(positions)))
    return {
        "stride": int(stride),
        "cycle_len": int(len(info["cycle"])),
        "green_steps": int(green_steps),
        "mismatch_vs_baseline": int(_baseline_mismatch_count(info["cycle"], len(positions))),
        "visible_green_floods": int(_visible_green_floods(g, info, bg, stride)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    info = _locate_border_code(g)
    if info is None:
        return g.copy()

    stride = _decode_stride(g, info, bg)
    if stride is None:
        return g.copy()

    return _apply_rule(g, info, bg, stride, info["cycle"])


def _trim_sprite(sprite: np.ndarray, bg: int) -> np.ndarray | None:
    mask = sprite != int(bg)
    rows_any = np.any(mask, axis=1)
    cols_any = np.any(mask, axis=0)
    if not rows_any.any():
        return None
    r0, r1 = np.where(rows_any)[0][[0, -1]]
    c0, c1 = np.where(cols_any)[0][[0, -1]]
    return sprite[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _build_code_cycle(seg_len: int, n_steps: int, rng: random.Random) -> list[int] | None:
    interior = list(range(1, int(seg_len) - 1))
    choices: list[list[int]] = []
    for n_green in (1, 2):
        if int(n_green) > len(interior):
            continue
        for combo in itertools.combinations(interior, n_green):
            if any((b - a) <= 1 for a, b in zip(combo, combo[1:])):
                continue
            cycle = [Y] * int(seg_len)
            for idx in combo:
                cycle[int(idx)] = G
            green_steps = sum(int(cycle[i % len(cycle)] == G) for i in range(int(n_steps)))
            mismatch = _baseline_mismatch_count(cycle, n_steps)
            if green_steps >= 2 and mismatch >= 3:
                choices.append(cycle)
    if not choices:
        return None
    return list(rng.choice(choices))


def _choose_stride(limit: int, rng: random.Random) -> int | None:
    choices = [s for s in (3, 4, 5) if len(_stripe_positions(0, 1, limit, s)) >= 6]
    if not choices:
        return None
    return int(rng.choice(choices))


def _pick_step_indices(cycle: list[int], n_steps: int, rng: random.Random) -> list[int] | None:
    if int(n_steps) < 5:
        return None

    green_ids = [i for i in range(1, int(n_steps)) if int(cycle[i % len(cycle)]) == G]
    yellow_ids = [i for i in range(1, int(n_steps)) if int(cycle[i % len(cycle)]) == Y]
    if len(green_ids) < 2 or len(yellow_ids) < 2:
        return None

    chosen: list[int] = [1]
    for idx in green_ids:
        if idx not in chosen:
            chosen.append(int(idx))
        if len([x for x in chosen if int(cycle[x % len(cycle)]) == G]) >= 2:
            break

    pool = yellow_ids + green_ids
    rng.shuffle(pool)
    for idx in pool:
        if idx not in chosen:
            chosen.append(int(idx))
        if len(chosen) >= 5:
            break

    chosen = sorted(set(chosen))
    if len(chosen) < 5:
        return None
    if math.gcd(*chosen) != 1:
        return None
    return chosen


def _orthogonal_hits_span(sprite: np.ndarray, base: int, orient: str, span: set[int], bg: int) -> bool:
    mask = sprite != int(bg)
    if orient == "h":
        cols = {int(base + dc) for dc in range(sprite.shape[1]) if bool(mask[:, dc].any())}
        return cols <= span
    rows = {int(base + dr) for dr in range(sprite.shape[0]) if bool(mask[dr].any())}
    return rows <= span


def _place_sprite_at_distance(
    canvas: np.ndarray,
    sprite: np.ndarray,
    bg: int,
    info: dict,
    distance: int,
    span: set[int],
    rng: random.Random,
) -> bool:
    H, W = canvas.shape
    h, w = sprite.shape
    mask = sprite != int(bg)
    blocked = ndimage.binary_dilation(canvas != int(bg), structure=np.ones((3, 3), dtype=bool), iterations=1)

    if info["orient"] == "h":
        if int(info["dir1"]) > 0:
            r0 = int(distance)
        else:
            r0 = int(H - int(distance) - h)
        if r0 < 1 or r0 + h > H - 1:
            return False
        cols = list(range(1, W - w))
        rng.shuffle(cols)
        for c0 in cols:
            if _orthogonal_hits_span(sprite, c0, "h", span, bg):
                continue
            if bool((blocked[r0 : r0 + h, c0 : c0 + w] & mask).any()):
                continue
            patch = canvas[r0 : r0 + h, c0 : c0 + w]
            patch[mask] = sprite[mask]
            return True
        return False

    if int(info["dir1"]) > 0:
        c0 = int(distance)
    else:
        c0 = int(W - int(distance) - w)
    if c0 < 1 or c0 + w > W - 1:
        return False
    rows = list(range(1, H - h))
    rng.shuffle(rows)
    for r0 in rows:
        if _orthogonal_hits_span(sprite, r0, "v", span, bg):
            continue
        if bool((blocked[r0 : r0 + h, c0 : c0 + w] & mask).any()):
            continue
        patch = canvas[r0 : r0 + h, c0 : c0 + w]
        patch[mask] = sprite[mask]
        return True
    return False


def _gen_attempt():
    rng = random.Random()

    H = rng.randint(26, 32)
    W = rng.randint(24, 32)

    available = [c for c in range(10) if c not in (G, Y)]
    bg, fg = rng.sample(available, 2)

    edge = rng.choice(["top", "bottom", "left", "right"])
    seg_len = rng.randint(5, 7)

    if edge in ("top", "bottom"):
        max_start = W - 1 - seg_len
        if max_start < 1:
            return None
        seg_start = rng.randint(1, max_start)
        span = list(range(seg_start, seg_start + seg_len))
        info = {
            "orient": "h",
            "start_pos": 0 if edge == "top" else H - 1,
            "span": span,
            "dir1": 1 if edge == "top" else -1,
            "limit": H,
        }
    else:
        max_start = H - 1 - seg_len
        if max_start < 1:
            return None
        seg_start = rng.randint(1, max_start)
        span = list(range(seg_start, seg_start + seg_len))
        info = {
            "orient": "v",
            "start_pos": 0 if edge == "left" else W - 1,
            "span": span,
            "dir1": 1 if edge == "left" else -1,
            "limit": W,
        }

    stride = _choose_stride(int(info["limit"]), rng)
    if stride is None:
        return None
    n_steps = len(_stripe_positions(info["start_pos"], info["dir1"], info["limit"], stride))
    cycle = _build_code_cycle(seg_len, n_steps, rng)
    if cycle is None:
        return None
    info["cycle"] = cycle

    step_ids = _pick_step_indices(cycle, n_steps, rng)
    if step_ids is None:
        return None

    grid = np.full((H, W), bg, dtype=int)
    if info["orient"] == "h":
        for offset, c in enumerate(span):
            grid[int(info["start_pos"]), int(c)] = int(cycle[offset])
    else:
        for offset, r in enumerate(span):
            grid[int(r), int(info["start_pos"])] = int(cycle[offset])

    canvas = grid.copy()
    span_set = set(int(x) for x in span)

    for step_idx in step_ids:
        placed = False
        for _ in range(20):
            bh = rng.randint(2, 6)
            bw = rng.randint(2, 6)
            density = rng.uniform(0.45, 0.85)
            try:
                sprite = random_sprite(
                    bh,
                    bw,
                    density=density,
                    symmetry="not_symmetric",
                    color_palette=[fg],
                    connectivity=4,
                    background=bg,
                )
            except (AssertionError, RecursionError):
                continue
            sprite = _trim_sprite(sprite, bg)
            if sprite is None:
                continue
            pix = int(np.sum(sprite != int(bg)))
            if pix < 5 or pix > 36:
                continue
            distance = int(step_idx) * int(stride)
            if _place_sprite_at_distance(canvas, sprite, bg, info, distance, span_set, rng):
                placed = True
                break
        if not placed:
            return None

    fg_mask = (canvas != int(bg)) & ~np.isin(canvas, [Y, G])
    _lab, n_placed = label_mask(fg_mask, connectivity=4)
    if int(n_placed) < 5:
        return None

    inp = canvas.copy()
    detected = _locate_border_code(inp)
    if detected is None:
        return None
    decoded_stride = _decode_stride(inp, detected, bg)
    if decoded_stride != int(stride):
        return None
    if detected["cycle"] != cycle:
        return None

    metrics = _variant_metrics(inp)
    if metrics is None:
        return None
    if int(metrics["stride"]) == 2:
        return None
    if int(metrics["mismatch_vs_baseline"]) < 3:
        return None
    if int(metrics["visible_green_floods"]) < 2:
        return None

    out = main(inp)
    if int(get_bg(inp)) != int(bg):
        return None
    if not np.array_equal(main(inp), out):
        return None

    return inp


def generate_input():
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 400 attempts")
