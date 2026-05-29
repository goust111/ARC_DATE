# Combo variant of 221dfab4 (index 20)
# combo_spec: D6/0+D11/3
# Name: 单像素控制点+先分割后涂色

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_HELPER_DIR = Path("/home/lixinyu/snap/new/arc2_solver_select")
if str(_HELPER_DIR) not in sys.path:
    sys.path.append(str(_HELPER_DIR))

from arc_utils import get_bg, label_mask, random_sprite

"""
Design analysis for the combo variant.

a) Concrete behavior and trace
   Original 221dfab4:
   - a yellow border segment directly gives both the stripe span and the propagation source
   - one global Y,Y,G pass runs across the whole grid
   - every green stripe recolors the full row/column

   Combo behavior:
   - the source shrinks to one yellow border control pixel
   - looking inward along that control pixel's ray, the first non-bg component is a rectangular
     measuring bar; its orthogonal extent is the hidden total span
   - the control pixel's own row/column is promoted to a full yellow splitter ray
   - the inferred total span is cut by that splitter into two child spans
   - the original Y,Y,G propagation is then executed independently on both sides, with green floods
     clipped to the corresponding half-plane instead of crossing through the splitter

   Small trace:
   - Suppose the control pixel is at the top edge at column 8.
   - The first hit object on column 8 is an interior bar spanning columns 5..11.
   - The solver infers total span 5..11, draws a yellow splitter on column 8, then runs stripes for
     columns 5..7 only in the left region and columns 9..11 only in the right region.
   - Compared with the original task, stripe width is no longer explicit on the border, and green rows
     now stop at the splitter instead of recoloring through the whole row.

b) Complexity calibration
   The variant is not reduced to a trivial single-pixel stripe: the measuring bar preserves a wide span,
   and the splitter creates two coupled subproblems. To keep the rule learnable rather than noisy, the
   generator restricts the measuring bar to a simple rectangle, forces the control pixel to lie strictly
   inside that bar's span, and keeps both child spans non-empty. This gives only a few degrees of freedom
   while still making the visual difference from the original task obvious.

c) Implicit assumptions review
   Original assumptions that break:
   - "yellow border component extent == stripe span": false once the source is a singleton
   - "one source span drives one global pass": false once a splitter ray cuts the grid first
   - "green can recolor the full row/column": false once recoloring must stay on one side

   New solver assumptions and how the generator satisfies them:
   - there is one salient yellow singleton on the border
   - its inward ray reaches a unique first non-yellow component
   - that first component crosses the splitter line, so the control pixel really lies inside the
     inferred hidden span
   - blobs do not occupy the splitter line, so the measuring bar remains the first ray hit and the
     two post-split regions stay visually separable

   Line-by-line impact on the old logic:
   - source detection can no longer infer orientation from segment shape, so orientation comes from
     which border contains the control pixel
   - span extraction can no longer read the yellow object, so it scans to the first foreground hit
   - the stripe loop can no longer paint globally, so it is parameterized by per-region bounds
   - gap filling must only clear the child spans, otherwise it would erase the splitter

d) Placement legality
   The generator places the measuring bar first, then the border control pixel, then side-local blobs.
   Every extra blob is placed inside one side box, away from edges, without overlap and with one-cell
   padding from existing objects. The splitter line itself is left clear except where the measuring bar
   intentionally crosses it. This avoids out-of-bounds placement, accidental merges, and ambiguous
   first-hit components.

e) Visual observability
   A human sees three coupled cues across examples:
   - the border source is a single yellow dot, not a segment
   - an interior bar aligned with that dot predicts the eventual stripe width
   - the output contains a persistent yellow splitter plus clipped green floods on the two sides
   These three cues support the intended two-step inference from a few IO pairs.
"""

YELLOW = 4
GREEN = 3


def _pick_control_pixel(g: np.ndarray) -> tuple[int, int] | None:
    H, W = g.shape
    border = np.zeros((H, W), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    smask = (g == int(YELLOW)) & border
    if not bool(smask.any()):
        return None

    lab, n = label_mask(smask, connectivity=4)
    if int(n) <= 0:
        return None

    singleton_ids = [cid for cid in range(1, int(n) + 1) if int((lab == int(cid)).sum()) == 1]
    best_id = int(singleton_ids[0] if singleton_ids else 1)
    pts = np.argwhere(lab == int(best_id))
    if pts.shape[0] == 0:
        return None
    return tuple(int(v) for v in pts[0])


def _control_geometry(g: np.ndarray, ctrl: tuple[int, int]) -> dict[str, int]:
    H, W = g.shape
    r, c = ctrl

    if int(r) == 0:
        return {"orient": "h", "start_pos": 0, "dir1": 1, "limit": int(H), "split": int(c), "dr": 1, "dc": 0}
    if int(r) == int(H - 1):
        return {"orient": "h", "start_pos": int(H - 1), "dir1": -1, "limit": int(H), "split": int(c), "dr": -1, "dc": 0}
    if int(c) == 0:
        return {"orient": "v", "start_pos": 0, "dir1": 1, "limit": int(W), "split": int(r), "dr": 0, "dc": 1}
    if int(c) == int(W - 1):
        return {"orient": "v", "start_pos": int(W - 1), "dir1": -1, "limit": int(W), "split": int(r), "dr": 0, "dc": -1}
    raise ValueError("control pixel is not on the border")


def _build_regions(orient: str, split: int, span: list[int], H: int, W: int) -> list[dict[str, object]]:
    regions: list[dict[str, object]] = []
    if orient == "h":
        left_span = [int(c) for c in span if int(c) < int(split)]
        right_span = [int(c) for c in span if int(c) > int(split)]
        if left_span:
            regions.append({"name": "left", "lo": 0, "hi": int(split - 1), "span": left_span})
        if right_span:
            regions.append({"name": "right", "lo": int(split + 1), "hi": int(W - 1), "span": right_span})
    else:
        top_span = [int(r) for r in span if int(r) < int(split)]
        bottom_span = [int(r) for r in span if int(r) > int(split)]
        if top_span:
            regions.append({"name": "top", "lo": 0, "hi": int(split - 1), "span": top_span})
        if bottom_span:
            regions.append({"name": "bottom", "lo": int(split + 1), "hi": int(H - 1), "span": bottom_span})
    return regions


def _infer_source(g: np.ndarray, bg: int) -> dict[str, object] | None:
    H, W = g.shape
    ctrl = _pick_control_pixel(g)
    if ctrl is None:
        return None

    geom = _control_geometry(g, ctrl)
    rr = int(ctrl[0]) + int(geom["dr"])
    cc = int(ctrl[1]) + int(geom["dc"])

    fg_mask = (g != int(bg)) & (g != int(YELLOW))
    lab, _ = label_mask(fg_mask, connectivity=4)

    while 0 <= int(rr) < int(H) and 0 <= int(cc) < int(W):
        if bool(fg_mask[int(rr), int(cc)]):
            cid = int(lab[int(rr), int(cc)])
            if cid <= 0:
                return None
            pts = np.argwhere(lab == int(cid))
            if pts.shape[0] == 0:
                return None

            if str(geom["orient"]) == "h":
                lo = int(pts[:, 1].min())
                hi = int(pts[:, 1].max())
            else:
                lo = int(pts[:, 0].min())
                hi = int(pts[:, 0].max())

            if not (int(lo) <= int(geom["split"]) <= int(hi)):
                return None

            span = list(range(int(lo), int(hi) + 1))
            regions = _build_regions(str(geom["orient"]), int(geom["split"]), span, int(H), int(W))
            return {
                "ctrl": tuple(int(v) for v in ctrl),
                "orient": str(geom["orient"]),
                "start_pos": int(geom["start_pos"]),
                "dir1": int(geom["dir1"]),
                "limit": int(geom["limit"]),
                "split": int(geom["split"]),
                "span": span,
                "regions": regions,
                "hit": (int(rr), int(cc)),
            }

        rr += int(geom["dr"])
        cc += int(geom["dc"])

    return None


def _green_positions(start_pos: int, dir1: int, limit: int) -> list[int]:
    positions: list[int] = []
    step = 0
    while True:
        pos = int(start_pos) + int(dir1) * (2 * int(step))
        if not (0 <= int(pos) < int(limit)):
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

    cycle = [int(YELLOW), int(YELLOW), int(GREEN)]
    step = 0

    while True:
        pos = int(start_pos) + int(dir1) * (2 * int(step))
        if not (0 <= int(pos) < int(limit)):
            break

        col = int(cycle[int(step) % 3])
        gap = int(start_pos) + int(dir1) * (2 * int(step) + 1)

        if orient == "h":
            if col == int(GREEN):
                seg = out[int(pos), int(region_lo) : int(region_hi) + 1]
                out[int(pos), int(region_lo) : int(region_hi) + 1] = np.where(seg != int(bg), int(GREEN), seg)
            for span_c in region_span:
                out[int(pos), int(span_c)] = int(col)
            if 0 <= int(gap) < out.shape[0]:
                for span_c in region_span:
                    out[int(gap), int(span_c)] = int(bg)
        else:
            if col == int(GREEN):
                seg = out[int(region_lo) : int(region_hi) + 1, int(pos)]
                out[int(region_lo) : int(region_hi) + 1, int(pos)] = np.where(seg != int(bg), int(GREEN), seg)
            for span_r in region_span:
                out[int(span_r), int(pos)] = int(col)
            if 0 <= int(gap) < out.shape[1]:
                for span_r in region_span:
                    out[int(span_r), int(gap)] = int(bg)

        step += 1


def _unsplit_output(grid: np.ndarray, bg: int, src: dict[str, object]) -> np.ndarray:
    out = np.array(grid, dtype=int).copy()
    cycle = [int(YELLOW), int(YELLOW), int(GREEN)]
    step = 0

    while True:
        pos = int(src["start_pos"]) + int(src["dir1"]) * (2 * int(step))
        if not (0 <= int(pos) < int(src["limit"])):
            break

        col = int(cycle[int(step) % 3])
        gap = int(src["start_pos"]) + int(src["dir1"]) * (2 * int(step) + 1)

        if str(src["orient"]) == "h":
            if col == int(GREEN):
                out[int(pos), :] = np.where(out[int(pos), :] != int(bg), int(GREEN), out[int(pos), :])
            for span_c in src["span"]:
                out[int(pos), int(span_c)] = int(col)
            if 0 <= int(gap) < out.shape[0]:
                for span_c in src["span"]:
                    out[int(gap), int(span_c)] = int(bg)
        else:
            if col == int(GREEN):
                out[:, int(pos)] = np.where(out[:, int(pos)] != int(bg), int(GREEN), out[:, int(pos)])
            for span_r in src["span"]:
                out[int(span_r), int(pos)] = int(col)
            if 0 <= int(gap) < out.shape[1]:
                for span_r in src["span"]:
                    out[int(span_r), int(gap)] = int(bg)

        step += 1

    return out


def _active_regions(grid: np.ndarray, src: dict[str, object]) -> int:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    tracks = _green_positions(int(src["start_pos"]), int(src["dir1"]), int(src["limit"]))
    if not tracks:
        return 0

    active = 0
    if str(src["orient"]) == "h":
        for region in src["regions"]:
            cols = slice(int(region["lo"]), int(region["hi"]) + 1)
            if bool(np.any((g[tracks, cols] != int(bg)) & (g[tracks, cols] != int(YELLOW)))):
                active += 1
    else:
        for region in src["regions"]:
            rows = slice(int(region["lo"]), int(region["hi"]) + 1)
            if bool(np.any((g[rows, tracks] != int(bg)) & (g[rows, tracks] != int(YELLOW)))):
                active += 1
    return int(active)


def _fusion_divergence(grid: np.ndarray) -> int:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    src = _infer_source(g, int(bg))
    if src is None:
        return 0
    combo = main(g)
    unsplit = _unsplit_output(g, int(bg), src)
    return int(np.sum(combo != unsplit))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule:
    1) use the yellow border singleton and the first hit interior bar to infer a hidden total span
    2) draw a full yellow splitter on the control pixel's line
    3) run the original Y,Y,G striped propagation independently on both sides of the splitter
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    out = g.copy()

    src = _infer_source(g, int(bg))
    if src is None:
        return out

    if str(src["orient"]) == "h":
        out[:, int(src["split"])] = int(YELLOW)
    else:
        out[int(src["split"]), :] = int(YELLOW)

    for region in src["regions"]:
        _apply_region_stripes(
            out,
            int(bg),
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

    for _ in range(100):
        bh = rng.randint(2, 8)
        bw = rng.randint(2, 8)
        density = rng.uniform(0.42, 0.85)

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

        for _ in range(70):
            pr = rng.randint(int(row_lo), int(row_hi - sh + 1))
            pc = rng.randint(int(col_lo), int(col_hi - sw + 1))
            if required_track is not None and not _covers_track(sprite, int(bg), int(pr), int(pc), axis, int(required_track)):
                continue
            if _can_place(canvas, sprite, int(pr), int(pc), int(bg)):
                _paste(canvas, sprite, int(pr), int(pc), int(bg))
                return True

    return False


def _build_measuring_bar(
    rng: random.Random,
    grid: np.ndarray,
    fg: int,
    edge: str,
) -> dict[str, int]:
    H, W = grid.shape
    thickness = rng.randint(2, 3)

    if edge in ("top", "bottom"):
        split = rng.randint(5, int(W - 6))
        left_len = rng.randint(2, min(4, int(split - 2)))
        right_len = rng.randint(2, min(4, int(W - split - 3)))
        c0 = int(split - left_len)
        c1 = int(split + right_len)
        r0 = rng.randint(3, int(H - thickness - 4))
        r1 = int(r0 + thickness - 1)
        grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = int(fg)
        ctrl = (0 if edge == "top" else int(H - 1), int(split))
        grid[int(ctrl[0]), int(ctrl[1])] = int(YELLOW)
        return {
            "split": int(split),
            "span_start": int(c0),
            "span_end": int(c1),
            "ray_hit": int(r0),
            "ctrl_r": int(ctrl[0]),
            "ctrl_c": int(ctrl[1]),
        }

    split = rng.randint(5, int(H - 6))
    top_len = rng.randint(2, min(4, int(split - 2)))
    bottom_len = rng.randint(2, min(4, int(H - split - 3)))
    r0 = int(split - top_len)
    r1 = int(split + bottom_len)
    c0 = rng.randint(3, int(W - thickness - 4))
    c1 = int(c0 + thickness - 1)
    grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = int(fg)
    ctrl = (int(split), 0 if edge == "left" else int(W - 1))
    grid[int(ctrl[0]), int(ctrl[1])] = int(YELLOW)
    return {
        "split": int(split),
        "span_start": int(r0),
        "span_end": int(r1),
        "ray_hit": int(c0),
        "ctrl_r": int(ctrl[0]),
        "ctrl_c": int(ctrl[1]),
    }


def _gen_attempt():
    rng = random.Random()

    H = rng.randint(25, 30)
    W = rng.randint(25, 30)

    available = [c for c in range(10) if c not in (int(GREEN), int(YELLOW))]
    bg, fg = rng.sample(available, 2)
    edge = rng.choice(["top", "bottom", "left", "right"])

    canvas = np.full((H, W), int(bg), dtype=int)
    meta = _build_measuring_bar(rng, canvas, int(fg), edge)

    src = _infer_source(canvas, int(bg))
    if src is None or len(src["regions"]) != 2:
        return None

    green_tracks = [int(p) for p in _green_positions(int(src["start_pos"]), int(src["dir1"]), int(src["limit"])) if 1 <= int(p) <= int(src["limit"]) - 2]
    if len(green_tracks) < 2:
        return None

    if str(src["orient"]) == "h":
        split = int(src["split"])
        if int(split) < 4 or int(split) > int(W - 5):
            return None
        targets = [
            ((1, int(H - 2), 1, int(split - 1)), "row", rng.choice(green_tracks)),
            ((1, int(H - 2), int(split + 1), int(W - 2)), "row", rng.choice(green_tracks)),
        ]
    else:
        split = int(src["split"])
        if int(split) < 4 or int(split) > int(H - 5):
            return None
        targets = [
            ((1, int(split - 1), 1, int(W - 2)), "col", rng.choice(green_tracks)),
            ((int(split + 1), int(H - 2), 1, int(W - 2)), "col", rng.choice(green_tracks)),
        ]

    for box, axis, track in targets:
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

    for box, axis, _track in targets:
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

    fg_mask = (canvas != int(bg)) & (canvas != int(YELLOW))
    _, n_placed = label_mask(fg_mask, connectivity=4)
    if int(n_placed) < 5:
        return None

    src = _infer_source(canvas, int(bg))
    if src is None or len(src["regions"]) != 2:
        return None
    if any(len(region["span"]) < 2 for region in src["regions"]):
        return None

    inp = canvas.copy()
    out = main(inp)
    if int(get_bg(inp)) != int(bg):
        return None
    if np.array_equal(inp, out):
        return None
    if not np.array_equal(main(inp), out):
        return None
    if _active_regions(inp, src) < 2:
        return None

    divergence = _fusion_divergence(inp)
    if divergence < max(8, int(src["limit"]) // 2):
        return None

    border_y = np.zeros((H, W), dtype=bool)
    border_y[0, :] = True
    border_y[-1, :] = True
    border_y[:, 0] = True
    border_y[:, -1] = True
    if int(np.sum((inp == int(YELLOW)) & border_y)) != 1:
        return None

    if edge in ("top", "bottom"):
        if int(meta["ray_hit"]) <= 2:
            return None
    else:
        if int(meta["ray_hit"]) <= 2:
            return None

    return inp


def generate_input():
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 500 attempts")
