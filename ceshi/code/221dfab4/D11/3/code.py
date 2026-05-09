# Variant of 221dfab4
# Dimension: D11 — 多源与推进顺序 (multi_step)
# Change: 单个源从一侧向对侧单次推进，无迭代或链式步骤
#   -> 两步流程：先射线分割再条纹涂色
#
# Design analysis for this variant
# - Concrete behavior:
#   The yellow border segment no longer directly drives one global pass.
#   First, its middle cell emits a 1-cell-thick yellow ray orthogonally to the
#   opposite edge, creating a visible splitter. Second, the original Y-Y-G
#   striped propagation is run twice, once on each side of that splitter, using
#   the left/right (or top/bottom) halves of the original segment as two child
#   sources.
# - Small trace:
#   Example: top-edge segment on columns 3..7. The original task would stripe
#   rows 0,2,4,... and any green row would recolor the whole row's non-bg cells.
#   Here the middle column 5 first becomes a full yellow ray. Then columns 3..4
#   stripe downward only in the left region and columns 6..7 stripe downward
#   only in the right region. On a green row, recoloring stops at the splitter
#   instead of crossing through it.
# - Why this is not too weak:
#   The output gains a new persistent yellow ray plus two clipped green floods,
#   so the "split first, then process each side" logic is visually legible from
#   a few examples instead of looking like a minor crop of the old rule.
# - Old implicit assumptions that break:
#   The original code assumed one source span and that green could recolor the
#   full row/column. That is no longer safe: green must be clipped to one side,
#   the gap fill must not erase the splitter, and empty child spans must be
#   skipped instead of treated as valid sources.
# - Generator guarantees:
#   It uses odd-length border segments so the middle splitter is unambiguous,
#   keeps blobs off the splitter line, places blobs on both sides, and requires
#   each side to have foreground on at least one green stripe coordinate so the
#   second stage is observable instead of degenerating into a one-sided case.

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_HELPER_DIR = Path("/home/lixinyu/snap/new/arc2_solver_select")
if str(_HELPER_DIR) not in sys.path:
    sys.path.append(str(_HELPER_DIR))

from arc_utils import get_bg, label_mask, random_sprite

Y = 4
G = 3


def _find_source(g: np.ndarray):
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
    pts = np.argwhere(lab == int(best_id))
    rs = pts[:, 0]
    cs = pts[:, 1]
    r0, r1 = int(rs.min()), int(rs.max())
    c0, c1 = int(cs.min()), int(cs.max())

    if r0 == r1:
        orient = "h"
        start_pos = int(r0)
        dir1 = 1 if int(start_pos) == 0 else -1
        limit = int(H)
        span = list(range(int(c0), int(c1) + 1))
        split = int(span[len(span) // 2])
        regions = []
        left_span = [int(c) for c in span if int(c) < int(split)]
        right_span = [int(c) for c in span if int(c) > int(split)]
        if left_span:
            regions.append(
                {
                    "name": "left",
                    "lo": 0,
                    "hi": int(split - 1),
                    "span": left_span,
                }
            )
        if right_span:
            regions.append(
                {
                    "name": "right",
                    "lo": int(split + 1),
                    "hi": int(W - 1),
                    "span": right_span,
                }
            )
    elif c0 == c1:
        orient = "v"
        start_pos = int(c0)
        dir1 = 1 if int(start_pos) == 0 else -1
        limit = int(W)
        span = list(range(int(r0), int(r1) + 1))
        split = int(span[len(span) // 2])
        regions = []
        top_span = [int(r) for r in span if int(r) < int(split)]
        bottom_span = [int(r) for r in span if int(r) > int(split)]
        if top_span:
            regions.append(
                {
                    "name": "top",
                    "lo": 0,
                    "hi": int(split - 1),
                    "span": top_span,
                }
            )
        if bottom_span:
            regions.append(
                {
                    "name": "bottom",
                    "lo": int(split + 1),
                    "hi": int(H - 1),
                    "span": bottom_span,
                }
            )
    else:
        return None

    return {
        "orient": orient,
        "start_pos": int(start_pos),
        "dir1": int(dir1),
        "limit": int(limit),
        "split": int(split),
        "regions": regions,
    }


def _green_positions(start_pos: int, dir1: int, limit: int) -> list[int]:
    positions: list[int] = []
    step = 0
    while True:
        pos = int(start_pos) + int(dir1) * (2 * int(step))
        if not (0 <= pos < int(limit)):
            break
        if int(step) % 3 == 2:
            positions.append(int(pos))
        step += 1
    return positions


def _apply_region_stripes(
    out: np.ndarray,
    bg: int,
    orient: str,
    start_pos: int,
    dir1: int,
    limit: int,
    region_lo: int,
    region_hi: int,
    region_span: list[int],
) -> None:
    if not region_span or int(region_lo) > int(region_hi):
        return

    cycle = [int(Y), int(Y), int(G)]
    step = 0
    while True:
        pos = int(start_pos) + int(dir1) * (2 * int(step))
        if not (0 <= pos < int(limit)):
            break

        col = int(cycle[int(step) % 3])
        if orient == "h":
            if col == int(G):
                seg = out[int(pos), int(region_lo) : int(region_hi) + 1]
                out[int(pos), int(region_lo) : int(region_hi) + 1] = np.where(
                    seg != int(bg), int(G), seg
                )
            for c in region_span:
                out[int(pos), int(c)] = int(col)
            gap = int(start_pos) + int(dir1) * (2 * int(step) + 1)
            if 0 <= gap < out.shape[0]:
                for c in region_span:
                    out[int(gap), int(c)] = int(bg)
        else:
            if col == int(G):
                seg = out[int(region_lo) : int(region_hi) + 1, int(pos)]
                out[int(region_lo) : int(region_hi) + 1, int(pos)] = np.where(
                    seg != int(bg), int(G), seg
                )
            for r in region_span:
                out[int(r), int(pos)] = int(col)
            gap = int(start_pos) + int(dir1) * (2 * int(step) + 1)
            if 0 <= gap < out.shape[1]:
                for r in region_span:
                    out[int(r), int(gap)] = int(bg)

        step += 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Two-stage variant:
    1) A yellow splitter ray is emitted from the middle of the border segment.
    2) The original Y-Y-G striped propagation is run independently on both sides.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    out = g.copy()

    src = _find_source(g)
    if src is None:
        return out

    if src["orient"] == "h":
        out[:, int(src["split"])] = int(Y)
    else:
        out[int(src["split"]), :] = int(Y)

    for region in src["regions"]:
        _apply_region_stripes(
            out,
            bg,
            str(src["orient"]),
            int(src["start_pos"]),
            int(src["dir1"]),
            int(src["limit"]),
            int(region["lo"]),
            int(region["hi"]),
            list(region["span"]),
        )

    return out


def _trim_sprite(sprite: np.ndarray, bg: int) -> np.ndarray | None:
    mask = sprite != int(bg)
    if not bool(mask.any()):
        return None
    rows = np.where(np.any(mask, axis=1))[0]
    cols = np.where(np.any(mask, axis=0))[0]
    return sprite[int(rows[0]) : int(rows[-1]) + 1, int(cols[0]) : int(cols[-1]) + 1]


def _can_place(canvas: np.ndarray, sprite: np.ndarray, pr: int, pc: int, bg: int) -> bool:
    H, W = canvas.shape
    sh, sw = sprite.shape
    if not (0 <= int(pr) and 0 <= int(pc) and int(pr) + int(sh) <= int(H) and int(pc) + int(sw) <= int(W)):
        return False

    mask = sprite != int(bg)
    for dr, dc in np.argwhere(mask):
        rr = int(pr + dr)
        cc = int(pc + dc)
        if int(canvas[int(rr), int(cc)]) != int(bg):
            return False
        r0 = max(0, int(rr - 1))
        r1 = min(int(H), int(rr + 2))
        c0 = max(0, int(cc - 1))
        c1 = min(int(W), int(cc + 2))
        if bool(np.any(canvas[int(r0) : int(r1), int(c0) : int(c1)] != int(bg))):
            return False
    return True


def _paste(canvas: np.ndarray, sprite: np.ndarray, pr: int, pc: int, bg: int) -> None:
    mask = sprite != int(bg)
    for dr, dc in np.argwhere(mask):
        canvas[int(pr + dr), int(pc + dc)] = int(sprite[int(dr), int(dc)])


def _covers_track(sprite: np.ndarray, bg: int, pr: int, pc: int, axis: str, track: int) -> bool:
    mask = sprite != int(bg)
    if axis == "row":
        rows = {int(pr + dr) for dr in range(sprite.shape[0]) if bool(mask[int(dr), :].any())}
        return int(track) in rows
    cols = {int(pc + dc) for dc in range(sprite.shape[1]) if bool(mask[:, int(dc)].any())}
    return int(track) in cols


def _place_blob(
    rng: random.Random,
    canvas: np.ndarray,
    bg: int,
    fg: int,
    row_lo: int,
    row_hi: int,
    col_lo: int,
    col_hi: int,
    axis: str,
    required_track: int | None = None,
) -> bool:
    if int(row_lo) > int(row_hi) or int(col_lo) > int(col_hi):
        return False

    for _ in range(90):
        bh = rng.randint(2, 9)
        bw = rng.randint(2, 9)
        density = rng.uniform(0.4, 0.85)

        try:
            sprite = random_sprite(
                bh,
                bw,
                density=density,
                symmetry="not_symmetric",
                color_palette=[int(fg)],
                connectivity=4,
                background=int(bg),
            )
        except (AssertionError, RecursionError):
            continue

        sprite = _trim_sprite(np.array(sprite, dtype=int), int(bg))
        if sprite is None:
            continue

        sh, sw = sprite.shape
        pix = int(np.sum(sprite != int(bg)))
        if pix < 5 or pix > 67:
            continue
        if int(sh) > int(row_hi - row_lo + 1) or int(sw) > int(col_hi - col_lo + 1):
            continue

        for _ in range(60):
            pr = rng.randint(int(row_lo), int(row_hi - sh + 1))
            pc = rng.randint(int(col_lo), int(col_hi - sw + 1))
            if required_track is not None and not _covers_track(
                sprite, int(bg), int(pr), int(pc), axis, int(required_track)
            ):
                continue
            if _can_place(canvas, sprite, int(pr), int(pc), int(bg)):
                _paste(canvas, sprite, int(pr), int(pc), int(bg))
                return True

    return False


def _variant_activity(grid: np.ndarray) -> int:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    src = _find_source(g)
    if src is None:
        return 0

    active = 0
    stripe_hits = _green_positions(int(src["start_pos"]), int(src["dir1"]), int(src["limit"]))
    if src["orient"] == "h":
        for region in src["regions"]:
            cols = slice(int(region["lo"]), int(region["hi"]) + 1)
            if stripe_hits and bool(np.any((g[stripe_hits, cols] != int(bg)) & (g[stripe_hits, cols] != int(Y)))):
                active += 1
    else:
        for region in src["regions"]:
            rows = slice(int(region["lo"]), int(region["hi"]) + 1)
            if stripe_hits and bool(np.any((g[rows, stripe_hits] != int(bg)) & (g[rows, stripe_hits] != int(Y)))):
                active += 1
    return int(active)


def _gen_attempt():
    rng = random.Random()

    H = rng.randint(25, 30)
    W = rng.randint(24, 30)

    available = [c for c in range(10) if c not in (int(G), int(Y))]
    bg, fg = rng.sample(available, 2)

    edge = rng.choice(["top", "bottom", "left", "right"])
    seg_len = rng.choice([5, 7, 9])

    grid = np.full((H, W), int(bg), dtype=int)

    if edge in ("top", "bottom"):
        orient = "h"
        edge_row = 0 if edge == "top" else int(H - 1)
        split = rng.randint(5, int(W - 6))
        seg_start = int(split - seg_len // 2)
        if seg_start < 1 or seg_start + seg_len - 1 > int(W - 2):
            return None
        span = list(range(int(seg_start), int(seg_start + seg_len)))
        green_tracks = _green_positions(int(edge_row), 1 if edge == "top" else -1, int(H))
        if len(green_tracks) < 2:
            return None
        left_box = (1, int(H - 2), 1, int(split - 1))
        right_box = (1, int(H - 2), int(split + 1), int(W - 2))
        if left_box[2] > left_box[3] or right_box[2] > right_box[3]:
            return None
        for c in span:
            grid[int(edge_row), int(c)] = int(Y)
        targeted = [
            (left_box, "row", rng.choice(green_tracks)),
            (right_box, "row", rng.choice(green_tracks)),
        ]
    else:
        orient = "v"
        edge_col = 0 if edge == "left" else int(W - 1)
        split = rng.randint(5, int(H - 6))
        seg_start = int(split - seg_len // 2)
        if seg_start < 1 or seg_start + seg_len - 1 > int(H - 2):
            return None
        span = list(range(int(seg_start), int(seg_start + seg_len)))
        green_tracks = _green_positions(int(edge_col), 1 if edge == "left" else -1, int(W))
        if len(green_tracks) < 2:
            return None
        top_box = (1, int(split - 1), 1, int(W - 2))
        bottom_box = (int(split + 1), int(H - 2), 1, int(W - 2))
        if top_box[0] > top_box[1] or bottom_box[0] > bottom_box[1]:
            return None
        for r in span:
            grid[int(r), int(edge_col)] = int(Y)
        targeted = [
            (top_box, "col", rng.choice(green_tracks)),
            (bottom_box, "col", rng.choice(green_tracks)),
        ]

    canvas = grid.copy()

    for box, axis, track in targeted:
        if not _place_blob(
            rng,
            canvas,
            int(bg),
            int(fg),
            int(box[0]),
            int(box[1]),
            int(box[2]),
            int(box[3]),
            axis,
            int(track),
        ):
            return None

    for box, axis, _track in targeted:
        extra = rng.randint(1, 3)
        for _ in range(extra):
            _place_blob(
                rng,
                canvas,
                int(bg),
                int(fg),
                int(box[0]),
                int(box[1]),
                int(box[2]),
                int(box[3]),
                axis,
                None,
            )

    fg_mask = (canvas != int(bg)) & (canvas != int(Y))
    _lab, n_placed = label_mask(fg_mask, connectivity=4)
    if int(n_placed) < 4:
        return None
    if int(get_bg(canvas)) != int(bg):
        return None
    if int(_variant_activity(canvas)) < 2:
        return None

    out = main(canvas)
    if not np.array_equal(main(canvas), out):
        return None

    return canvas


def generate_input():
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 400 attempts")
