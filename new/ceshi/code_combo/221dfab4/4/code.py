# Combo variant of 221dfab4 (index 4)
# combo_spec: D7/0+D13/0
# Name: 色块阻停+色块作障碍物

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/arc2_solver_select")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import get_bg, label_mask, random_sprite


# Design analysis for the combo D7/0 + D13/0 variant
# a) Concrete behavior and trace:
#    Keep the original source detection: the largest yellow border segment seeds a train
#    of parallel 1-cell stripes/gaps, with distances 0,1,2,3,... from the border and
#    colors on even distances cycling Y,Y,G. The fusion changes how foreground blobs
#    participate. Example with a top source spanning cols 4..8:
#      - Original task:
#          row 0 -> Y on cols 4..8
#          row 1 -> bg on cols 4..8
#          row 2 -> Y on cols 4..8
#          row 3 -> bg on cols 4..8
#          row 4 -> G on cols 4..8, plus recolor all existing non-bg on row 4 to green
#          continue all the way to the far edge, even through blobs
#      - This combo:
#          rows before the first obstacle line behave regularly inside the source span.
#          On the first line whose span intersects a foreground blob, the line is clipped:
#          obstacle cells stay as the blob color, unobstructed span cells still receive the
#          current stripe/gap color. Immediately after drawing that clipped frontier line,
#          propagation stops globally and no later rows/cols are touched.
#      - Difference:
#          D13 contributes the local "bite" taken out of the frontier line; D7 contributes
#          the global halt after that encounter, so open tracks do not continue past it.
# b) Complexity calibration:
#    If every blocker covered the full span, the output would look like a simple hard stop
#    and D13 would not be visible. If blockers were only beyond the stop line, the variant
#    would collapse back to the original. The generator therefore forces a frontier blocker
#    that intersects some but not all span tracks, and also leaves several clean tracks plus
#    several lines of free space beyond the frontier so the global truncation is observable.
# c) Implicit assumptions audited from the original solver:
#    - `out[pos, :] = ...` / `out[:, pos] = ...` assumed green stripes may freely recolor
#      any foreground cell. That no longer holds because blobs are obstacles and must keep
#      their color. Green is therefore treated like the other stripe colors: it paints only
#      reachable cells inside the seed span.
#    - Unconditional writes over the entire span assumed that every span cell is always
#      paintable on every frontier line. The combo breaks that assumption exactly on the
#      first hit line, so writes now check the original obstacle mask cell-by-cell.
#    - Advancing to the far edge assumed there is no stateful stop condition. The combo adds
#      a frontier-stop event: once any obstacle is encountered on the active span, later
#      distances are no longer processed.
#    - Obstacle detection must consult the original input `g`, not `out`, otherwise newly
#      painted stripes would falsely count as blockers.
# d) Placement legality:
#    The generator anchors the frontier blob so its nearest occupied cells lie exactly on the
#    intended stop line and only on a strict subset of the span. Extra blobs are placed with
#    one-cell separation, never overlap the source segment, and are forbidden from touching
#    the source span on or before the stop line, so the solver's first-hit assumption holds.
# e) Visual observability:
#    Across 2-4 examples, a human sees three cues repeatedly:
#      1. regular Y/Y/G stripes near the seed;
#      2. a final frontier stripe/gap whose span is partially eaten by a blob silhouette;
#      3. clean tracks that would have continued farther, but do not, proving the whole
#         process stopped after the encounter instead of merely painting around the blob.


Y = 4
G = 3
CYCLE = [Y, Y, G]


def _find_source_segment(g: np.ndarray) -> tuple[str, int, list[int], int, int, np.ndarray] | None:
    H, W = g.shape

    border = np.zeros((H, W), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    smask = (g == int(Y)) & border
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


def _line_coord(orient: str, pos: int, track: int) -> tuple[int, int]:
    if orient == "h":
        return int(pos), int(track)
    return int(track), int(pos)


def _simulate_combo(g: np.ndarray) -> tuple[np.ndarray, dict[str, int | None]]:
    g = np.array(g, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    parsed = _find_source_segment(g)
    if parsed is None:
        return out, {"stop_dist": None, "hit_tracks": 0, "open_tracks": 0}

    orient, start_pos, span, dir1, limit, seg_mask = parsed
    obstacle = (g != int(bg)) & (~seg_mask)

    for dist in range(int(limit)):
        pos = int(start_pos) + int(dir1) * int(dist)
        if not (0 <= pos < int(limit)):
            break

        color = int(bg) if dist % 2 == 1 else int(CYCLE[(dist // 2) % 3])
        hit_tracks = 0

        for track in span:
            r, c = _line_coord(orient, pos, int(track))
            if bool(obstacle[r, c]):
                hit_tracks += 1
                continue
            out[r, c] = int(color)

        if dist > 0 and hit_tracks > 0:
            return out, {
                "stop_dist": int(dist),
                "hit_tracks": int(hit_tracks),
                "open_tracks": int(len(span) - hit_tracks),
            }

    return out, {"stop_dist": None, "hit_tracks": 0, "open_tracks": int(len(span))}


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    out, _ = _simulate_combo(g)
    return out


def _original_main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    parsed = _find_source_segment(g)
    if parsed is None:
        return out

    orient, start_pos, span, dir1, limit, _seg_mask = parsed
    for step in range(int(limit)):
        pos = int(start_pos) + int(dir1) * (2 * int(step))
        if not (0 <= pos < int(limit)):
            break

        color = int(CYCLE[int(step) % 3])
        if orient == "h":
            if color == int(G):
                out[pos, :] = np.where(out[pos, :] != int(bg), int(G), out[pos, :])
            out[pos, span] = int(color)
            gap = int(start_pos) + int(dir1) * (2 * int(step) + 1)
            if 0 <= gap < int(limit):
                out[gap, span] = int(bg)
        else:
            if color == int(G):
                out[:, pos] = np.where(out[:, pos] != int(bg), int(G), out[:, pos])
            out[span, pos] = int(color)
            gap = int(start_pos) + int(dir1) * (2 * int(step) + 1)
            if 0 <= gap < int(limit):
                out[span, gap] = int(bg)
    return out


def _combo_metrics(g: np.ndarray) -> dict[str, int]:
    g = np.array(g, dtype=int)
    combo_out, meta = _simulate_combo(g)
    parsed = _find_source_segment(g)
    if parsed is None or meta["stop_dist"] is None:
        return {
            "stop_dist": -1,
            "hit_tracks": 0,
            "open_tracks": 0,
            "beyond_stop_suppressed": 0,
            "diff_vs_original": 0,
        }

    orient, start_pos, span, dir1, limit, _seg_mask = parsed
    orig_out = _original_main(g)
    stop_dist = int(meta["stop_dist"])

    suppressed = 0
    for dist in range(stop_dist + 1, int(limit)):
        pos = int(start_pos) + int(dir1) * int(dist)
        if not (0 <= pos < int(limit)):
            break
        for track in span:
            r, c = _line_coord(orient, pos, int(track))
            if orig_out[r, c] != g[r, c] and combo_out[r, c] == g[r, c]:
                suppressed += 1

    return {
        "stop_dist": int(stop_dist),
        "hit_tracks": int(meta["hit_tracks"]),
        "open_tracks": int(meta["open_tracks"]),
        "beyond_stop_suppressed": int(suppressed),
        "diff_vs_original": int(np.sum(combo_out != orig_out)),
    }


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

    canvas[int(top) : int(top) + h, int(left) : int(left) + w][patch_mask] = patch[patch_mask]
    return True


def _candidate_locations(H: int, W: int, h: int, w: int, border: int = 1) -> list[tuple[int, int]]:
    return [
        (int(r), int(c))
        for r in range(int(border), H - int(border) - int(h) + 1)
        for c in range(int(border), W - int(border) - int(w) + 1)
    ]


def _place_sprite(
    rng: random.Random,
    canvas: np.ndarray,
    sprite: np.ndarray,
    bg: int,
    allow_fn,
) -> bool:
    H, W = canvas.shape
    h, w = sprite.shape
    if h > H - 2 or w > W - 2:
        return False

    sprite_mask = sprite != int(bg)
    occ = _expanded_nonbg(canvas, bg, pad=1)
    candidates = _candidate_locations(H, W, h, w, border=1)
    rng.shuffle(candidates)

    for top, left in candidates:
        if bool((occ[int(top) : int(top) + h, int(left) : int(left) + w] & sprite_mask).any()):
            continue
        if not allow_fn(int(top), int(left), sprite_mask):
            continue
        canvas[int(top) : int(top) + h, int(left) : int(left) + w][sprite_mask] = sprite[sprite_mask]
        return True
    return False


def _make_frontier_patch(rng: random.Random, fg: int, bg: int, depth: int, cover: int) -> np.ndarray:
    if rng.random() < 0.55 and depth >= 3 and cover >= 3:
        patch = np.full((int(depth), int(cover)), int(fg), dtype=int)
        # Remove a far corner only, so the frontier-facing side still defines the first hit.
        corner = rng.choice([(depth - 1, 0), (depth - 1, cover - 1)])
        patch[int(corner[0]), int(corner[1])] = int(bg)
        return patch
    return np.full((int(depth), int(cover)), int(fg), dtype=int)


def _place_frontier_blocker(
    canvas: np.ndarray,
    *,
    bg: int,
    fg: int,
    orient: str,
    start_pos: int,
    span: list[int],
    dir1: int,
    stop_dist: int,
    rng: random.Random,
) -> bool:
    span_len = len(span)
    if span_len < 5:
        return False

    cover = rng.randint(2, min(4, span_len - 2))
    start_idx = rng.randint(0, span_len - cover)
    depth = rng.randint(2, 4)
    patch = _make_frontier_patch(rng, fg, bg, depth=depth, cover=cover)

    stop_pos = int(start_pos) + int(dir1) * int(stop_dist)
    first_track = int(span[int(start_idx)])

    if orient == "h":
        top = int(stop_pos) if dir1 == 1 else int(stop_pos) - patch.shape[0] + 1
        left = int(first_track)
    else:
        top = int(first_track)
        left = int(stop_pos) if dir1 == 1 else int(stop_pos) - patch.shape[1] + 1

    return _paste_patch(canvas, patch, int(top), int(left), int(bg))


def _random_blob_sprite(rng: random.Random, bg: int, fg: int) -> np.ndarray | None:
    for _ in range(40):
        try:
            sprite = random_sprite(
                rng.randint(2, 6),
                rng.randint(2, 6),
                density=rng.uniform(0.45, 0.82),
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
        if 4 <= pix <= 24:
            return sprite
    return None


def _sprite_hits_span_before_stop(
    sprite_mask: np.ndarray,
    *,
    top: int,
    left: int,
    orient: str,
    start_pos: int,
    span_set: set[int],
    dir1: int,
    stop_dist: int,
) -> bool:
    rr, cc = np.nonzero(sprite_mask)
    for dr, dc in zip(rr, cc):
        r = int(top + dr)
        c = int(left + dc)
        prop = int(r) if orient == "h" else int(c)
        track = int(c) if orient == "h" else int(r)
        dist = int(dir1) * (int(prop) - int(start_pos))
        if track in span_set and 1 <= dist <= int(stop_dist):
            return True
    return False


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()

    H = rng.randint(24, 30)
    W = rng.randint(24, 30)

    available = [c for c in range(10) if c not in (G, Y)]
    bg, fg = rng.sample(available, 2)

    edge = rng.choice(["top", "bottom", "left", "right"])
    seg_len = rng.randint(6, 9)

    canvas = np.full((H, W), int(bg), dtype=int)
    if edge in ("top", "bottom"):
        max_start = W - 1 - seg_len
        if max_start < 2:
            return None
        seg_start = rng.randint(1, max_start)
        span = list(range(seg_start, seg_start + seg_len))
        orient = "h"
        start_pos = 0 if edge == "top" else H - 1
        dir1 = 1 if edge == "top" else -1
        canvas[int(start_pos), span] = int(Y)
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
        canvas[span, int(start_pos)] = int(Y)
        limit = W

    if int(limit) < 12:
        return None

    stop_dist = rng.randint(4, int(limit) - 6)
    if not _place_frontier_blocker(
        canvas,
        bg=bg,
        fg=fg,
        orient=orient,
        start_pos=start_pos,
        span=span,
        dir1=dir1,
        stop_dist=stop_dist,
        rng=rng,
    ):
        return None

    span_set = set(int(x) for x in span)
    extra_target = rng.randint(2, 4)
    extras_placed = 0
    for _ in range(50):
        if extras_placed >= extra_target:
            break
        sprite = _random_blob_sprite(rng, bg, fg)
        if sprite is None:
            continue

        def allow_extra(top: int, left: int, sprite_mask: np.ndarray) -> bool:
            return not _sprite_hits_span_before_stop(
                sprite_mask,
                top=top,
                left=left,
                orient=orient,
                start_pos=start_pos,
                span_set=span_set,
                dir1=dir1,
                stop_dist=stop_dist,
            )

        if _place_sprite(rng, canvas, sprite, bg, allow_extra):
            extras_placed += 1

    fg_mask = (canvas != int(bg)) & (canvas != int(Y))
    _lab, n_blobs = label_mask(fg_mask, connectivity=4)
    if int(n_blobs) < 3:
        return None

    if int(get_bg(canvas)) != int(bg):
        return None

    metrics = _combo_metrics(canvas)
    if int(metrics["stop_dist"]) != int(stop_dist):
        return None
    if int(metrics["hit_tracks"]) < 1:
        return None
    if int(metrics["open_tracks"]) < 2:
        return None
    if int(metrics["beyond_stop_suppressed"]) < max(4, int(metrics["open_tracks"]) * 2):
        return None
    if int(metrics["diff_vs_original"]) < int(metrics["beyond_stop_suppressed"]) + 2:
        return None

    out = main(canvas)
    if out.shape != canvas.shape:
        return None
    if np.array_equal(out, canvas):
        return None

    return canvas


def generate_input():
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 400 attempts")
