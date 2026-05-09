# Variant of 221dfab4
# Dimension: D13 — 前景色块的角色 (meta)
# Change: 前景色块是被动对象——仅被绿色条纹重着色，不影响条纹行为... → 色块作为障碍物阻断条纹

from __future__ import annotations

import random
import sys

import numpy as np

_ARC_UTILS_DIR = "/home/lixinyu/snap/new/arc2_solver_select"
if _ARC_UTILS_DIR not in sys.path:
    sys.path.append(_ARC_UTILS_DIR)

from arc_utils import get_bg, label_mask, random_free_location, random_sprite


# Design analysis for the D13 variant
# a) Concrete behavior:
#    The border yellow segment still defines a set of parallel 1-cell-wide tracks and the
#    color cycle is still Y, Y, G with a bg gap between consecutive stripes. The change is
#    that each track now advances cell-by-cell away from the border and stops forever at
#    its first foreground obstacle. Example for a top-edge segment spanning columns 5..8:
#    old logic on column 6: rows 0,2,4,6... are painted 4,4,3,4... even if row 5 already
#    contains a blob pixel, because blobs were passive and only green rows recolored them.
#    new logic on column 6: row 0=4, row 1=bg, row 2=4, row 3=bg, row 4=3, then if row 5
#    is foreground the propagation halts there and rows 6+ on that column stay unchanged.
#    Neighboring columns can continue deeper, so the stripe front becomes clipped by blob
#    silhouettes.
# b) Complexity calibration:
#    If every blocker hit were rare or all at the same depth, the variant would look too
#    close to the original. The generator therefore forces multiple blocked tracks at
#    different depths while keeping some open tracks, so the jagged truncation is visually
#    obvious and not ambiguous.
# c) Hidden assumptions in the original solver that break here:
#    - `out[pos, :] = ... green ...` assumed green stripes flood whole rows/cols; this
#      cannot remain because obstacles must stay intact and must not be recolored away.
#    - `for c in span: out[pos, c] = col` / `for r in span: ...` assumed every track is
#      paintable all the way through the span. In the variant this must become per-track
#      painting with an obstacle check before every cell write.
#    - gap clearing also used unconditional writes; now gaps are only forced to bg until
#      the first obstacle, after which the original obstacle and everything behind it must
#      remain untouched.
# d) Placement legality:
#    The generator keeps all blobs at least one cell away from each other and from the grid
#    border, never overlaps the yellow source segment, and explicitly places several
#    blockers inside the source span so that `main()` sees valid, non-overlapping obstacle
#    tracks.
# e) Visual observability:
#    Human-solvable examples need a visible difference from the seed task. Each generated
#    input therefore contains at least two blocked track groups at different depths and at
#    least two still-open tracks. Across 2-4 samples, the solver can see that blobs are
#    preserving their color and clipping later stripes behind them.


Y = 4
G = 3


def _find_source_segment(g: np.ndarray) -> tuple[str, int, list[int], int, int, np.ndarray] | None:
    H, W = g.shape
    border = np.zeros((H, W), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    smask = (g == Y) & border
    if not bool(smask.any()):
        return None

    lab, n = label_mask(smask, connectivity=4)
    if int(n) <= 0:
        return None

    best_id = max(range(1, int(n) + 1), key=lambda cid: int((lab == int(cid)).sum()))
    seg_mask = lab == int(best_id)
    pts = np.argwhere(seg_mask)

    rs = pts[:, 0]
    cs = pts[:, 1]
    r0, r1 = int(rs.min()), int(rs.max())
    c0, c1 = int(cs.min()), int(cs.max())

    if r0 == r1:
        start_pos = int(r0)
        span = list(range(int(c0), int(c1) + 1))
        dir1 = 1 if start_pos == 0 else -1
        return "h", start_pos, span, dir1, H, seg_mask
    if c0 == c1:
        start_pos = int(c0)
        span = list(range(int(r0), int(r1) + 1))
        dir1 = 1 if start_pos == 0 else -1
        return "v", start_pos, span, dir1, W, seg_mask
    return None


def _track_stats(g: np.ndarray) -> dict[str, object]:
    g = np.array(g, dtype=int)
    parsed = _find_source_segment(g)
    if parsed is None:
        return {
            "blocked_tracks": 0,
            "open_tracks": 0,
            "unique_stop_depths": 0,
            "stop_depths": [],
            "occluded_cells": 0,
        }

    orient, start_pos, span, dir1, limit, seg_mask = parsed
    bg = int(get_bg(g))
    obstacle = (g != bg) & (~seg_mask)

    stop_depths: list[int] = []
    open_tracks = 0
    occluded_cells = 0
    max_depth = int(limit) - 1

    for track in span:
        hit = None
        for dist in range(1, int(limit)):
            pos = int(start_pos) + int(dir1) * int(dist)
            if orient == "h":
                cell = obstacle[int(pos), int(track)]
            else:
                cell = obstacle[int(track), int(pos)]
            if bool(cell):
                hit = int(dist)
                break
        if hit is None:
            open_tracks += 1
        else:
            stop_depths.append(hit)
            occluded_cells += max(0, int(max_depth) - int(hit))

    return {
        "blocked_tracks": int(len(stop_depths)),
        "open_tracks": int(open_tracks),
        "unique_stop_depths": int(len(set(stop_depths))),
        "stop_depths": stop_depths,
        "occluded_cells": int(occluded_cells),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant rule: foreground blobs are blockers.

    The yellow border segment still seeds parallel tracks and the stripe colors still cycle
    Y, Y, G with a bg gap between stripes. The change is local to each track:
    once a non-background obstacle is encountered, that track stops and nothing behind the
    obstacle is painted. Obstacles keep their original color and are never recolored.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    parsed = _find_source_segment(g)
    if parsed is None:
        return out

    orient, start_pos, span, dir1, limit, seg_mask = parsed
    obstacle = (g != bg) & (~seg_mask)
    cycle = [Y, Y, G]

    for track in span:
        for dist in range(0, int(limit)):
            pos = int(start_pos) + int(dir1) * int(dist)
            if not (0 <= pos < int(limit)):
                break

            if orient == "h":
                r, c = int(pos), int(track)
            else:
                r, c = int(track), int(pos)

            if bool(obstacle[r, c]):
                break

            if dist % 2 == 1:
                out[r, c] = int(bg)
            else:
                out[r, c] = int(cycle[(dist // 2) % 3])

    return out


def _trim_sprite(sprite: np.ndarray, bg: int) -> np.ndarray | None:
    mask = sprite != int(bg)
    if not bool(mask.any()):
        return None
    rows_any = np.any(mask, axis=1)
    cols_any = np.any(mask, axis=0)
    r0, r1 = np.where(rows_any)[0][[0, -1]]
    c0, c1 = np.where(cols_any)[0][[0, -1]]
    return sprite[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _expanded_nonbg(canvas: np.ndarray, bg: int, pad: int) -> np.ndarray:
    occ = canvas != int(bg)
    if int(pad) <= 0:
        return occ

    H, W = canvas.shape
    out = occ.copy()
    rr, cc = np.nonzero(occ)
    for r, c in zip(rr, cc):
        r0 = max(0, int(r) - int(pad))
        r1 = min(H, int(r) + int(pad) + 1)
        c0 = max(0, int(c) - int(pad))
        c1 = min(W, int(c) + int(pad) + 1)
        out[r0:r1, c0:c1] = True
    return out


def _paste_patch(canvas: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> bool:
    H, W = canvas.shape
    h, w = patch.shape
    if top < 1 or left < 1 or top + h > H - 1 or left + w > W - 1:
        return False

    expanded = _expanded_nonbg(canvas, bg, pad=1)
    patch_mask = patch != int(bg)
    if bool((expanded[int(top) : int(top) + h, int(left) : int(left) + w] & patch_mask).any()):
        return False

    region = canvas[int(top) : int(top) + h, int(left) : int(left) + w]
    region[patch_mask] = patch[patch_mask]
    return True


def _make_blocker_patch(rng: random.Random, fg: int, bg: int) -> np.ndarray:
    h = rng.randint(2, 4)
    w = rng.randint(2, 4)
    patch = np.full((h, w), fg, dtype=int)

    # Knock out one corner sometimes so blockers are not all perfect rectangles.
    if h * w >= 6 and rng.random() < 0.65:
        corner = rng.choice([(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)])
        patch[corner] = int(bg)
    return patch


def _place_forced_blockers(
    canvas: np.ndarray,
    *,
    bg: int,
    fg: int,
    orient: str,
    start_pos: int,
    span: list[int],
    dir1: int,
    limit: int,
    rng: random.Random,
) -> bool:
    span_len = len(span)
    if span_len < 4:
        return False

    depth_min = 3
    depth_max = int(limit) - 6
    if depth_max <= depth_min + 2:
        return False

    widths: list[int] = []
    remaining = span_len
    for _ in range(2):
        max_w = min(4, max(2, remaining - 2))
        widths.append(rng.randint(2, max_w))
        remaining -= widths[-1]

    intervals: list[tuple[int, int]] = []
    cursor = 0
    for w in widths:
        max_start = span_len - w
        if cursor > max_start:
            return False
        start_idx = rng.randint(cursor, max_start)
        intervals.append((start_idx, start_idx + w - 1))
        cursor = start_idx + w + 1

    depth_choices = sorted(rng.sample(range(int(depth_min), int(depth_max) + 1), 2))
    if depth_choices[1] - depth_choices[0] < 3:
        depth_choices[1] = min(int(depth_max), depth_choices[0] + 3)
        if depth_choices[1] - depth_choices[0] < 3:
            return False

    for depth, (i0, i1) in zip(depth_choices, intervals):
        patch = _make_blocker_patch(rng, fg, bg)
        ph, pw = patch.shape
        first_track = span[int(i0)]
        last_track = span[int(i1)]

        if orient == "h":
            hit_row = int(start_pos) + int(dir1) * int(depth)
            top = hit_row if dir1 == 1 else hit_row - ph + 1
            left = int(first_track)
            if left + pw - 1 < int(last_track):
                left = int(last_track) - pw + 1
            if left > int(first_track):
                left = rng.randint(int(first_track), int(left))
        else:
            hit_col = int(start_pos) + int(dir1) * int(depth)
            left = hit_col if dir1 == 1 else hit_col - pw + 1
            top = int(first_track)
            if top + ph - 1 < int(last_track):
                top = int(last_track) - ph + 1
            if top > int(first_track):
                top = rng.randint(int(first_track), int(top))

        if not _paste_patch(canvas, patch, int(top), int(left), bg):
            return False

    return True


def _place_extra_blobs(
    canvas: np.ndarray,
    *,
    bg: int,
    fg: int,
    orient: str,
    span: list[int],
    rng: random.Random,
) -> None:
    extra_target = rng.randint(2, 4)
    placed = 0

    for _ in range(40):
        if placed >= extra_target:
            break

        try:
            sprite = random_sprite(
                rng.randint(2, 7),
                rng.randint(2, 7),
                density=rng.uniform(0.45, 0.8),
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

        pix = int(np.sum(sprite != bg))
        if pix < 4 or pix > 28:
            continue

        try:
            pr, pc = random_free_location(canvas, sprite, transparent=bg, border=1, padding=1)
        except ValueError:
            continue

        mask = sprite != bg
        sh, sw = sprite.shape
        blob_rows = {int(pr + dr) for dr in range(sh) if bool(mask[dr].any())}
        blob_cols = {int(pc + dc) for dc in range(sw) if bool(mask[:, dc].any())}

        if orient == "h" and blob_cols & set(span):
            continue
        if orient == "v" and blob_rows & set(span):
            continue

        canvas[int(pr) : int(pr) + sh, int(pc) : int(pc) + sw][mask] = fg
        placed += 1


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()

    H = rng.randint(24, 30)
    W = rng.randint(24, 30)

    available = [c for c in range(10) if c not in (G, Y)]
    bg, fg = rng.sample(available, 2)

    edge = rng.choice(["top", "bottom", "left", "right"])
    seg_len = rng.randint(6, 9)

    if edge in ("top", "bottom"):
        max_start = W - 1 - seg_len
        if max_start < 2:
            return None
        seg_start = rng.randint(1, max_start)
        span = list(range(seg_start, seg_start + seg_len))
        orient = "h"
        start_pos = 0 if edge == "top" else H - 1
        dir1 = 1 if edge == "top" else -1
        limit = H
    else:
        max_start = H - 1 - seg_len
        if max_start < 2:
            return None
        seg_start = rng.randint(1, max_start)
        span = list(range(seg_start, seg_start + seg_len))
        orient = "v"
        start_pos = 0 if edge == "left" else W - 1
        dir1 = 1 if edge == "left" else -1
        limit = W

    canvas = np.full((H, W), bg, dtype=int)

    if orient == "h":
        canvas[int(start_pos), span] = Y
    else:
        canvas[span, int(start_pos)] = Y

    if not _place_forced_blockers(
        canvas,
        bg=bg,
        fg=fg,
        orient=orient,
        start_pos=start_pos,
        span=span,
        dir1=dir1,
        limit=limit,
        rng=rng,
    ):
        return None

    _place_extra_blobs(canvas, bg=bg, fg=fg, orient=orient, span=span, rng=rng)

    fg_mask = (canvas != bg) & (canvas != Y)
    _, n_blobs = label_mask(fg_mask, connectivity=4)
    if int(n_blobs) < 4:
        return None

    stats = _track_stats(canvas)
    if int(stats["blocked_tracks"]) < 2:
        return None
    if int(stats["open_tracks"]) < 2:
        return None
    if int(stats["unique_stop_depths"]) < 2:
        return None
    if int(stats["occluded_cells"]) < 8:
        return None

    if get_bg(canvas) != bg:
        return None

    out = main(canvas)
    if out.shape != canvas.shape:
        return None
    if np.array_equal(out, canvas):
        return None

    return canvas


def generate_input():
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 300 attempts")
