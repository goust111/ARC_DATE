# Variant of 221dfab4
# Dimension: D12 — 符号编码——线段属性的语义 (symbol_instructions)
# Change: 黄色线段仅编码位置和跨度信息，颜色固定为黄(4)和绿(3)... → 边缘上的图例定义颜色周期

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[5]
_ARC_UTILS_DIR = _ROOT / "arc2_solver_select"
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import get_bg, label_mask, random_free_location, random_sprite


# Design analysis for this D12 variant
# -----------------------------------
# Reference note:
# - The prompt cites e87109e9 as the source-task inspiration for symbol_instructions.
# - In this workspace the task data for e87109e9 is present, but no solver file was
#   available to read. The design below therefore matches the stated mechanism:
#   a small border legend must be decoded first and then used as an instruction.
#
# (a) Concrete behavior trace
# - Original input had exactly one border instruction object: a long yellow segment.
#   Its geometry told us where stripes start and how wide they are, while colors
#   were globally fixed to [4, 4, 3].
# - New input keeps that long yellow segment as the geometric source, but adds a
#   second border object: a 2-cell legend on a different edge. Reading the legend
#   along its own axis gives two colors [A, B].
# - The actual stripe cycle becomes [A, A, B].
#   Example:
#   - old: top-edge yellow segment spanning cols 8..12 -> rows 0,2,4,... use 4,4,3
#   - new: same segment, plus a right-edge vertical legend [6,2]
#   - output now paints rows 0,2,4,... with colors 6,6,2,6,6,2,...
#   - rows colored with B still perform the original "flood whole row/col of all
#     non-background pixels" behavior; rows colored with A still only paint inside
#     the segment span.
# - The behavioral delta is therefore exactly at the "choose stripe color" stage:
#   fixed constants are replaced by a decoded legend.
#
# (b) Complexity calibration
# - A 2-cell legend is small but still visibly different from the original task
#   because the produced stripes usually use colors unrelated to the yellow source.
# - To avoid ambiguity, generator constraints keep the source segment length >= 4,
#   the legend length = 2, and the legend on a perpendicular border. This makes the
#   long source and short legend visually distinct and easy to parse.
#
# (c) Old implicit assumptions that no longer hold
# - `smask = (g == 4) & border`: this only finds the source in the old task. In the
#   variant we must inspect border occupancy more generally because there is now a
#   second border instruction object.
# - `cycle = [4, 4, 3]`: hard-coded colors are invalid once the legend may vary.
# - `if col == 3`: the special flood color is no longer globally green; it must be
#   the legend's second color.
# - Any logic that silently assumes "all border non-bg cells belong to the source"
#   is also invalid. The solver must separate the long yellow segment from the short
#   legend before painting.
#
# (d) Generator legality / solver compatibility
# - Border non-bg occupancy is restricted to exactly two disconnected objects:
#   the source segment and the legend.
# - Interior blobs are still placed with `border=1`, so they never touch the border
#   and cannot be mistaken for instructions.
# - The legend is placed on rows/cols that are not flood-stripe positions, so the
#   output transformation does not immediately overwrite the instruction itself.
# - Legend colors, blob color, background, and the yellow source are chosen to be
#   mutually distinct enough for decoding and visual inspection.
#
# (e) Visual observability
# - The generator forces the first legend color A to be non-yellow, so the output's
#   stripe color cannot be explained by the source segment alone.
# - Because the source segment itself is repainted to A in the output, the examples
#   clearly show that yellow only marks the instruction location, while the legend
#   controls the stripe palette.


YELLOW = 4


def _border_mask(h: int, w: int) -> np.ndarray:
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return border


def _component_orientation(pts: np.ndarray) -> str | None:
    rs = pts[:, 0]
    cs = pts[:, 1]
    if int(rs.min()) == int(rs.max()):
        return "h"
    if int(cs.min()) == int(cs.max()):
        return "v"
    return None


def _decode_instructions(grid: np.ndarray, bg: int):
    """
    Decode the long yellow source segment plus the short border legend.

    Assumptions introduced by the new solver and guaranteed by the generator:
    - exactly one yellow straight border segment of length >= 3 is the source
    - exactly one other straight border component of length >= 2 is the legend
    - legend reading order is left->right for horizontal, top->bottom for vertical
    """
    h, w = grid.shape
    border = _border_mask(h, w)
    occ = (grid != int(bg)) & border
    lab, n = label_mask(occ, connectivity=4)
    if int(n) < 2:
        return None

    comps: list[dict] = []
    for cid in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(cid))
        orient = _component_orientation(pts)
        cols = {int(grid[int(r), int(c)]) for r, c in pts}
        comps.append(
            {
                "id": int(cid),
                "pts": pts,
                "orient": orient,
                "size": int(len(pts)),
                "colors": cols,
                "mono": len(cols) == 1,
            }
        )

    source_candidates = [
        comp
        for comp in comps
        if comp["orient"] is not None
        and comp["mono"]
        and comp["size"] >= 3
        and next(iter(comp["colors"])) == int(YELLOW)
    ]
    if not source_candidates:
        return None
    source = max(source_candidates, key=lambda comp: comp["size"])

    legend_candidates = [
        comp
        for comp in comps
        if comp["id"] != source["id"] and comp["orient"] is not None and comp["size"] >= 2
    ]
    if not legend_candidates:
        return None
    legend = min(legend_candidates, key=lambda comp: comp["size"])

    spts = source["pts"]
    rs = spts[:, 0]
    cs = spts[:, 1]
    r0, r1 = int(rs.min()), int(rs.max())
    c0, c1 = int(cs.min()), int(cs.max())

    if source["orient"] == "h":
        start_pos = int(r0)
        span = list(range(int(c0), int(c1) + 1))
        dir1 = 1 if start_pos == 0 else -1
        limit = h
    else:
        start_pos = int(c0)
        span = list(range(int(r0), int(r1) + 1))
        dir1 = 1 if start_pos == 0 else -1
        limit = w

    lpts = legend["pts"]
    if legend["orient"] == "h":
        ordered = lpts[np.argsort(lpts[:, 1])]
    else:
        ordered = lpts[np.argsort(lpts[:, 0])]
    legend_colors = [int(grid[int(r), int(c)]) for r, c in ordered]
    if len(legend_colors) < 2 or legend_colors[0] == legend_colors[1]:
        return None

    local_color = int(legend_colors[0])
    flood_color = int(legend_colors[1])
    return source["orient"], start_pos, span, dir1, limit, local_color, flood_color


def _stripe_positions(start_pos: int, dir1: int, limit: int):
    step = 0
    while True:
        pos = int(start_pos) + int(dir1) * (2 * int(step))
        if not (0 <= pos < int(limit)):
            break
        yield int(step), int(pos)
        step += 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant rule:
    - A long yellow border segment still gives the stripe direction and span.
    - A separate 2-cell border legend gives colors [A, B].
    - The stripe cycle is then A, A, B, where B keeps the original "recolor the
      whole row/column's non-background pixels" semantics.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    out = g.copy()

    decoded = _decode_instructions(g, bg)
    if decoded is None:
        return out

    orient, start_pos, span, dir1, limit, local_color, flood_color = decoded
    cycle = [int(local_color), int(local_color), int(flood_color)]

    for step, pos in _stripe_positions(start_pos, dir1, limit):
        col = int(cycle[int(step) % 3])

        if orient == "h":
            if col == int(flood_color):
                out[pos, :] = np.where(out[pos, :] != int(bg), int(flood_color), out[pos, :])
            out[pos, span] = int(col)
            gap = int(start_pos) + int(dir1) * (2 * int(step) + 1)
            if 0 <= gap < g.shape[0]:
                out[int(gap), span] = int(bg)
        else:
            if col == int(flood_color):
                out[:, pos] = np.where(out[:, pos] != int(bg), int(flood_color), out[:, pos])
            out[span, pos] = int(col)
            gap = int(start_pos) + int(dir1) * (2 * int(step) + 1)
            if 0 <= gap < g.shape[1]:
                out[span, int(gap)] = int(bg)

    return out


def _choose_legend_position(rng: random.Random, orient: str, start_pos: int, dir1: int, limit: int):
    flood_positions = {pos for step, pos in _stripe_positions(start_pos, dir1, limit) if step % 3 == 2}
    if orient == "h":
        candidates = [r for r in range(1, limit - 2) if r not in flood_positions and (r + 1) not in flood_positions]
    else:
        candidates = [c for c in range(1, limit - 2) if c not in flood_positions and (c + 1) not in flood_positions]
    if not candidates:
        return None
    return int(rng.choice(candidates))


def _gen_attempt():
    rng = random.Random()

    h = rng.randint(25, 30)
    w = rng.randint(22, 30)

    # Keep the source yellow, but force the legend's first color to be non-yellow
    # so the D12 change is visible in the output.
    usable = [c for c in range(10) if c != int(YELLOW)]
    bg, fg, local_color, flood_color = rng.sample(usable, 4)

    edge = rng.choice(["top", "bottom", "left", "right"])
    seg_len = rng.randint(4, 8)

    if edge in ("top", "bottom"):
        max_start = w - 1 - seg_len
        if max_start < 1:
            return None
        seg_start = rng.randint(1, max_start)
        span = list(range(seg_start, seg_start + seg_len))
        orient = "h"
        start_pos = 0 if edge == "top" else h - 1
        dir1 = 1 if start_pos == 0 else -1
        limit = h
    else:
        max_start = h - 1 - seg_len
        if max_start < 1:
            return None
        seg_start = rng.randint(1, max_start)
        span = list(range(seg_start, seg_start + seg_len))
        orient = "v"
        start_pos = 0 if edge == "left" else w - 1
        dir1 = 1 if start_pos == 0 else -1
        limit = w

    grid = np.full((h, w), int(bg), dtype=int)

    if orient == "h":
        grid[int(start_pos), span] = int(YELLOW)
        legend_start = _choose_legend_position(rng, orient, start_pos, dir1, h)
        if legend_start is None:
            return None
        legend_col = 0 if rng.random() < 0.5 else w - 1
        grid[int(legend_start), int(legend_col)] = int(local_color)
        grid[int(legend_start) + 1, int(legend_col)] = int(flood_color)
    else:
        grid[span, int(start_pos)] = int(YELLOW)
        legend_start = _choose_legend_position(rng, orient, start_pos, dir1, w)
        if legend_start is None:
            return None
        legend_row = 0 if rng.random() < 0.5 else h - 1
        grid[int(legend_row), int(legend_start)] = int(local_color)
        grid[int(legend_row), int(legend_start) + 1] = int(flood_color)

    canvas = grid.copy()

    target_blobs = rng.randint(5, 9)
    placed = 0
    attempts = 0
    while placed < target_blobs and attempts < target_blobs * 8:
        attempts += 1
        bh = rng.randint(2, 10)
        bw = rng.randint(2, 10)
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

        mask = sprite != int(bg)
        rows_any = np.any(mask, axis=1)
        cols_any = np.any(mask, axis=0)
        if not rows_any.any():
            continue

        r0, r1 = np.where(rows_any)[0][[0, -1]]
        c0, c1 = np.where(cols_any)[0][[0, -1]]
        sprite = sprite[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]

        pix = int(np.sum(sprite != int(bg)))
        if pix < 5 or pix > 67:
            continue

        try:
            pr, pc = random_free_location(canvas, sprite, transparent=int(bg), border=1, padding=1)
        except ValueError:
            continue

        sh, sw = sprite.shape
        sprite_mask = sprite != int(bg)
        for dr in range(sh):
            for dc in range(sw):
                if sprite_mask[int(dr), int(dc)]:
                    canvas[int(pr) + int(dr), int(pc) + int(dc)] = int(fg)
        placed += 1

    fg_mask = canvas == int(fg)
    _lab, n_placed = label_mask(fg_mask, connectivity=4)
    if int(n_placed) < 5:
        return None

    inp = canvas.copy()
    out = main(inp)
    decoded = _decode_instructions(inp, int(bg))
    if decoded is None:
        return None

    _, _, _, _, _, dec_local, dec_flood = decoded
    if int(dec_local) != int(local_color) or int(dec_flood) != int(flood_color):
        return None

    if int(get_bg(inp)) != int(bg):
        return None

    # Ensure the new symbol-instruction behavior is actually visible:
    # some former blob pixels must get recolored to the legend's flood color.
    changed_blob_pixels = int(np.sum((inp == int(fg)) & (out == int(flood_color))))
    if changed_blob_pixels < 4:
        return None

    # Ensure we did not degenerate back to the original fixed-color behavior.
    if int(local_color) == int(YELLOW) and int(flood_color) == 3:
        return None

    if np.array_equal(inp, out):
        return None

    return inp


def generate_input():
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 300 attempts")
