# Combo variant of 71e489b6 (index 25)
# combo_spec: D1/0+D2/2+D7/2
# Name: 十字光晕+泛化异常+仅覆写蓝色

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arc_utils import components_mask

# Design analysis:
# a) Behavioral trace:
#    Original task: first absorb blue spur pixels that sit in mostly black
#    territory, then search only black cells for two special cases: tiny detached
#    black components and leaf pixels on larger black components. Every such
#    target receives an orange 3x3 Chebyshev halo, and the halo can overwrite any
#    in-bounds color.
#    Fused task: the clean-up pass stays the same, but target detection and
#    painting both change. We now treat each large blue rectangle as the true
#    "foreground region" and mark every non-blue pixel inside that rectangle,
#    regardless of whether it is black, red, gray, etc. Each marked pixel emits
#    only a four-arm cross halo, and each halo arm paints orange only onto blue
#    cells. Example: if a red pixel sits in the middle of a blue block, the old
#    solver would ignore it entirely while the fused solver paints orange on the
#    four orthogonal neighbors. If a black intrusion touches the top edge of the
#    blue block, the fused solver paints only the left/right/down blue neighbors;
#    the black cell above it stays black because non-blue cells are never
#    overwritten by the halo.
# b) Complexity calibration:
#    A sample would collapse visually if it only contained black anomalies or if
#    every anomaly were deep inside blue with no clipped halo arms. The generator
#    therefore enforces three supports at once: at least one non-black anomaly so
#    D2/2 is visible, at least one strictly interior anomaly so cross-vs-square
#    geometry is visible, and at least one edge intrusion so "blue-only overwrite"
#    visibly clips a halo arm against black territory.
# c) Hidden assumptions audit:
#    Original STEP 1/2 depended on connected components of black cells; that
#    assumption breaks once anomalies can use arbitrary non-blue colors. The new
#    solver instead reconstructs large blue rectangles after spur absorption and
#    declares every non-blue cell inside those rectangles to be anomalous. The
#    original halo step also assumed any in-bounds cell could be painted orange;
#    D7/2 invalidates that, so painting now checks the cleaned base grid and only
#    recolors cells whose base color is blue. Centers no longer normalize to
#    black: they keep their original non-blue color, which is the natural
#    generalization of "leave the defect pixel itself unchanged."
# d) Placement legality:
#    Blue regions are generated as separated solid rectangles with at least a
#    one-cell black moat, so their bboxes are unambiguous for the solver. Small
#    internal anomaly components are placed strictly inside a rectangle and kept
#    apart by a Chebyshev buffer. Edge intrusions occupy cells on a rectangle
#    boundary and extend inward by at most two cells, so they remain part of the
#    blue region's bbox without disconnecting the region. Blue spur pixels are
#    added only at black cells that the absorption pass deterministically flips
#    back to black.
# e) Visual observability:
#    Across 2-4 examples, a human can infer the fused rule because the same
#    orange marker appears around black and non-black anomalies alike, diagonals
#    around interior anomalies stay blue, and edge-touching anomalies produce
#    visibly incomplete crosses where the adjacent black cell is not overwritten.
# f) Visual cleanliness:
#    Target groups are spaced apart so cross halos rarely merge into dense orange
#    carpets. The palette also stays compact: inputs use black/blue plus a few
#    anomaly colors, while outputs add only orange. That keeps the structure
#    readable even when multiple anomaly colors are present.
# g) Inferability:
#    The fused change relies only on visible structure: "inside a blue rectangle"
#    and "orthogonal neighbor is blue" are both easy to see. No hidden ordering
#    by numeric color id is required. Non-black anomaly colors appear as repeated
#    evidence that the detector is keyed to blue-vs-non-blue contrast rather than
#    to black specifically.

GRID_H_RANGE = (16, 19)
GRID_W_RANGE = (16, 19)
MIN_BLUE_REGION = 30
MIN_REGION_FILL = 0.78
PARTITION_STYLES = ["h_stripes", "v_panels", "blocks"]
INTERNAL_COUNT_RANGE = (2, 6)
EDGE_INTRUSION_RANGE = (1, 3)
SPUR_COUNT_RANGE = (2, 5)
TARGET_CLEARANCE = 2
DEFECT_PLACE_TRIES = 80
MIN_SPURS = 2
MIN_EDGE_TARGETS = 1
MAX_GEN_ATTEMPTS = 400

BLACK = 0
BLUE = 1
ORANGE = 7
ANOMALY_COLORS = [BLACK, 2, 3, 4, 5, 6, 8, 9]
NONBLACK_ANOMALY_COLORS = [2, 3, 4, 5, 6, 8, 9]
CROSS_OFFSETS = [(-1, 0), (0, -1), (0, 1), (1, 0)]
SQUARE_OFFSETS = [
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if not (dr == 0 and dc == 0)
]
ANOMALY_SHAPES = [
    [(0, 0)],
    [(0, 0), (1, 0)],
    [(0, 0), (0, 1)],
    [(0, 0), (1, 0), (2, 0)],
    [(0, 0), (0, 1), (0, 2)],
    [(0, 0), (1, 0), (1, 1)],
    [(0, 1), (1, 0), (1, 1)],
]


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _bbox_of_points(points: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _exclusion_zone(pixels: set[tuple[int, int]] | list[tuple[int, int]], dist: int) -> set[tuple[int, int]]:
    zone: set[tuple[int, int]] = set()
    for r, c in pixels:
        for dr in range(-dist, dist + 1):
            for dc in range(-dist, dist + 1):
                zone.add((r + dr, c + dc))
    return zone


def _absorb_blue_spurs(grid: np.ndarray) -> np.ndarray:
    """Mirror the original spur-absorption rule on blue pixels."""
    base = np.array(grid, dtype=int)
    H, W = base.shape
    for r in range(H):
        for c in range(W):
            if int(base[r, c]) != BLUE:
                continue
            black_n = 0
            oob = 0
            for rr, cc in _neighbors4(r, c):
                if not (0 <= rr < H and 0 <= cc < W):
                    oob += 1
                    continue
                if int(base[rr, cc]) == BLACK:
                    black_n += 1
            if oob >= 2:
                black_n += int(oob)
            if black_n >= 3:
                base[r, c] = BLACK
    return base


def _detect_blue_regions(base: np.ndarray) -> list[dict]:
    """Recover the intended large blue rectangles after spur absorption."""
    regions: list[dict] = []
    for coords in components_mask(base == BLUE, connectivity=4):
        cells = {(int(r), int(c)) for r, c in coords}
        if len(cells) < MIN_BLUE_REGION:
            continue
        r0, c0, r1, c1 = _bbox_of_points(cells)
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if len(cells) / area < MIN_REGION_FILL:
            continue
        regions.append({"bbox": (r0, c0, r1, c1), "cells": cells})
    regions.sort(key=lambda item: item["bbox"])
    return regions


def _collect_targets(base: np.ndarray) -> tuple[list[tuple[int, int]], list[dict]]:
    """Every non-blue pixel inside a blue rectangle is an anomaly target."""
    targets: set[tuple[int, int]] = set()
    regions = _detect_blue_regions(base)
    for region in regions:
        r0, c0, r1, c1 = region["bbox"]
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if int(base[r, c]) != BLUE:
                    targets.add((r, c))
    return sorted(targets), regions


def _render_output(
    base: np.ndarray,
    targets: list[tuple[int, int]],
    *,
    square: bool,
    blue_only: bool,
) -> np.ndarray:
    """Paint either cross or square halos; centers keep their original anomaly color."""
    H, W = base.shape
    out = np.array(base, dtype=int)
    offsets = SQUARE_OFFSETS if square else CROSS_OFFSETS
    for r, c in targets:
        for dr, dc in offsets:
            rr = r + dr
            cc = c + dc
            if not (0 <= rr < H and 0 <= cc < W):
                continue
            if blue_only and int(base[rr, cc]) != BLUE:
                continue
            out[rr, cc] = ORANGE
    for r, c in targets:
        out[r, c] = int(base[r, c])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mark every non-blue anomaly inside blue rectangles with an orange cross halo.

    Input structure:
    - One or more large blue rectangular regions sit in a black background.
    - Inside each blue rectangle there may be small black or colored anomaly
      pixels/components, plus occasional short non-blue intrusions entering from
      a rectangle edge.
    - Separate blue spur pixels may also appear in black areas; they are noise.

    Rule:
    - First absorb blue spur pixels that are mostly surrounded by black, exactly
      as in the source task.
    - Then detect every non-blue pixel lying inside the bbox of a large blue
      rectangle, regardless of that pixel's color.
    - Around each such pixel paint orange only on the four orthogonal neighbors,
      and only when that neighbor cell is blue. The anomaly pixel itself keeps
      its original non-blue color.

    Key insight: The detector has broadened from "special black defects" to "any
    blue-region outlier," while the marker has narrowed from an unrestricted
    3x3 halo to a blue-only cross clipped by local blue support.
    """
    g = np.array(grid, dtype=int)
    base = _absorb_blue_spurs(g)
    targets, _regions = _collect_targets(base)
    return _render_output(base, targets, square=False, blue_only=True)


def _make_h_stripes(H: int, W: int) -> list[tuple[int, int, int, int]]:
    n = random.randint(2, 3)
    for _ in range(40):
        heights = [random.randint(3, 5) for _ in range(n)]
        gaps = [random.randint(1, 2) for _ in range(n - 1)]
        total_h = sum(heights) + sum(gaps)
        if total_h > H - 2:
            continue
        start = random.randint(1, H - total_h - 1)
        rects = []
        row = start
        for idx, height in enumerate(heights):
            rects.append((row, 1, row + height - 1, W - 2))
            row += height
            if idx < len(gaps):
                row += gaps[idx]
        return rects
    raise RuntimeError("failed to build horizontal stripes")


def _make_v_panels(H: int, W: int) -> list[tuple[int, int, int, int]]:
    n = random.randint(2, 3)
    for _ in range(40):
        widths = [random.randint(3, 5) for _ in range(n)]
        gaps = [random.randint(1, 2) for _ in range(n - 1)]
        total_w = sum(widths) + sum(gaps)
        if total_w > W - 2:
            continue
        start = random.randint(1, W - total_w - 1)
        rects = []
        col = start
        for idx, width in enumerate(widths):
            rects.append((1, col, H - 2, col + width - 1))
            col += width
            if idx < len(gaps):
                col += gaps[idx]
        return rects
    raise RuntimeError("failed to build vertical panels")


def _make_blocks(H: int, W: int) -> list[tuple[int, int, int, int]]:
    mid_r = H // 2
    mid_c = W // 2
    rects = [
        (1, 1, mid_r - 1, mid_c - 1),
        (mid_r + 1, mid_c + 1, H - 2, W - 2),
    ]
    extra = random.choice(["bottom_left", "top_right", None])
    if extra == "bottom_left" and (H - mid_r - 2) * (mid_c - 2) >= MIN_BLUE_REGION:
        rects.append((mid_r + 1, 1, H - 2, mid_c - 2))
    elif extra == "top_right" and (mid_r - 2) * (W - mid_c - 2) >= MIN_BLUE_REGION:
        rects.append((1, mid_c + 1, mid_r - 2, W - 2))
    return rects


def _build_partition(H: int, W: int) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    style = random.choice(PARTITION_STYLES)
    if style == "h_stripes":
        rects = _make_h_stripes(H, W)
    elif style == "v_panels":
        rects = _make_v_panels(H, W)
    else:
        rects = _make_blocks(H, W)

    grid = np.zeros((H, W), dtype=int)
    for r0, c0, r1, c1 in rects:
        grid[r0 : r1 + 1, c0 : c1 + 1] = BLUE
    return grid, rects


def _shape_fits(rect: tuple[int, int, int, int], cells: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = rect
    for r, c in cells:
        if not (r0 + 1 <= r <= r1 - 1 and c0 + 1 <= c <= c1 - 1):
            return False
    return True


def _place_internal_anomalies(
    grid: np.ndarray,
    rects: list[tuple[int, int, int, int]],
    blocked: set[tuple[int, int]],
) -> tuple[set[tuple[int, int]], bool, bool]:
    placed_cells: set[tuple[int, int]] = set()
    has_nonblack = False
    has_black = False
    n_items = random.randint(*INTERNAL_COUNT_RANGE)

    for idx in range(n_items):
        force_nonblack = idx == 0 and not has_nonblack
        force_black = idx == 1 and not has_black
        placed = False
        for _ in range(DEFECT_PLACE_TRIES):
            rect = random.choice(rects)
            r0, c0, r1, c1 = rect
            shape = random.choice(ANOMALY_SHAPES)
            max_dr = max(dr for dr, _ in shape)
            max_dc = max(dc for _, dc in shape)
            if r1 - r0 < max_dr + 2 or c1 - c0 < max_dc + 2:
                continue
            anchor_r = random.randint(r0 + 1, r1 - 1 - max_dr)
            anchor_c = random.randint(c0 + 1, c1 - 1 - max_dc)
            cells = [(anchor_r + dr, anchor_c + dc) for dr, dc in shape]
            if not _shape_fits(rect, cells):
                continue
            if any((r, c) in blocked for r, c in cells):
                continue

            if force_nonblack:
                color = random.choice(NONBLACK_ANOMALY_COLORS)
            elif force_black:
                color = BLACK
            else:
                color = random.choice(ANOMALY_COLORS)
            for r, c in cells:
                grid[r, c] = color
            placed_cells.update(cells)
            blocked.update(_exclusion_zone(cells, TARGET_CLEARANCE))
            has_nonblack = has_nonblack or color != BLACK
            has_black = has_black or color == BLACK
            placed = True
            break
        if not placed:
            break

    return placed_cells, has_nonblack, has_black


def _stub_cells(rect: tuple[int, int, int, int], side: str, pos: int, length: int) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    if side == "top":
        return [(r0 + k, pos) for k in range(length)]
    if side == "bottom":
        return [(r1 - k, pos) for k in range(length)]
    if side == "left":
        return [(pos, c0 + k) for k in range(length)]
    return [(pos, c1 - k) for k in range(length)]


def _place_edge_intrusions(
    grid: np.ndarray,
    rects: list[tuple[int, int, int, int]],
    blocked: set[tuple[int, int]],
    has_nonblack: bool,
    has_black: bool,
) -> tuple[set[tuple[int, int]], bool, bool]:
    edge_cells: set[tuple[int, int]] = set()
    n_items = random.randint(*EDGE_INTRUSION_RANGE)

    for idx in range(n_items):
        force_nonblack = idx == 0 and not has_nonblack
        force_black = idx == 1 and not has_black
        placed = False
        for _ in range(DEFECT_PLACE_TRIES):
            rect = random.choice(rects)
            r0, c0, r1, c1 = rect
            side = random.choice(["top", "bottom", "left", "right"])
            length = random.randint(1, 2)
            if side in {"top", "bottom"}:
                if c1 - c0 < 4:
                    continue
                pos = random.randint(c0 + 2, c1 - 2)
            else:
                if r1 - r0 < 4:
                    continue
                pos = random.randint(r0 + 2, r1 - 2)
            cells = _stub_cells(rect, side, pos, length)
            if any((r, c) in blocked for r, c in cells):
                continue

            if force_nonblack:
                color = random.choice(NONBLACK_ANOMALY_COLORS)
            elif force_black:
                color = BLACK
            else:
                color = random.choice(ANOMALY_COLORS)
            for r, c in cells:
                grid[r, c] = color
            edge_cells.update(cells)
            blocked.update(_exclusion_zone(cells, TARGET_CLEARANCE))
            has_nonblack = has_nonblack or color != BLACK
            has_black = has_black or color == BLACK
            placed = True
            break
        if not placed:
            break

    return edge_cells, has_nonblack, has_black


def _find_spur_candidates(grid: np.ndarray, exclude: set[tuple[int, int]]) -> list[tuple[int, int]]:
    """Find black cells that the absorption pass would flip back from blue to black."""
    H, W = grid.shape
    black = grid == BLACK
    nb = np.zeros((H, W), dtype=int)
    nb[:-1, :] += black[1:, :]
    nb[1:, :] += black[:-1, :]
    nb[:, :-1] += black[:, 1:]
    nb[:, 1:] += black[:, :-1]

    oob = np.zeros((H, W), dtype=int)
    oob[0, :] += 1
    oob[-1, :] += 1
    oob[:, 0] += 1
    oob[:, -1] += 1

    effective = nb.copy()
    effective[oob >= 2] += oob[oob >= 2]

    rs, cs = np.where(black & (effective >= 3))
    candidates = [(int(r), int(c)) for r, c in zip(rs, cs) if (int(r), int(c)) not in exclude]
    random.shuffle(candidates)
    return candidates


def _behavior_metrics(grid: np.ndarray) -> dict[str, int]:
    """Quantify whether the sample visibly exercises each fused control."""
    base = _absorb_blue_spurs(grid)
    targets, regions = _collect_targets(base)
    fused = _render_output(base, targets, square=False, blue_only=True)
    square = _render_output(base, targets, square=True, blue_only=True)
    unrestricted = _render_output(base, targets, square=False, blue_only=False)

    edge_targets = 0
    interior_targets = 0
    for r, c in targets:
        for region in regions:
            r0, c0, r1, c1 = region["bbox"]
            if r0 <= r <= r1 and c0 <= c <= c1:
                if r in {r0, r1} or c in {c0, c1}:
                    edge_targets += 1
                else:
                    interior_targets += 1
                break

    return {
        "regions": len(regions),
        "targets": len(targets),
        "black_targets": sum(int(base[r, c]) == BLACK for r, c in targets),
        "nonblack_targets": sum(int(base[r, c]) not in {BLACK, BLUE} for r, c in targets),
        "edge_targets": edge_targets,
        "interior_targets": interior_targets,
        "delta_cross_vs_square": int(np.count_nonzero(fused != square)),
        "blocked_nonblue_cells": int(np.count_nonzero((unrestricted == ORANGE) & (fused != ORANGE))),
        "orange_cells": int(np.count_nonzero(fused == ORANGE)),
    }


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    base, rects = _build_partition(H, W)

    # Sanity: every intended blue rectangle should remain a large region after carving.
    for r0, c0, r1, c1 in rects:
        if (r1 - r0 + 1) * (c1 - c0 + 1) < MIN_BLUE_REGION:
            return None

    clean = np.array(base, dtype=int)
    blocked: set[tuple[int, int]] = set()
    internal_cells, has_nonblack, has_black = _place_internal_anomalies(clean, rects, blocked)
    if not internal_cells:
        return None
    edge_cells, has_nonblack, has_black = _place_edge_intrusions(
        clean, rects, blocked, has_nonblack, has_black
    )
    if len(edge_cells) < MIN_EDGE_TARGETS or not has_nonblack or not has_black:
        return None

    targets, regions = _collect_targets(clean)
    if not targets or len(regions) != len(rects):
        return None
    output = _render_output(clean, targets, square=False, blue_only=True)

    inp = np.array(clean, dtype=int)
    exclusion = blocked | _exclusion_zone(set(targets), 1)
    candidates = _find_spur_candidates(inp, exclusion)
    n_spurs = random.randint(*SPUR_COUNT_RANGE)
    if len(candidates) < max(MIN_SPURS, n_spurs):
        return None
    for r, c in candidates[:n_spurs]:
        inp[r, c] = BLUE

    metrics = _behavior_metrics(inp)
    if metrics["regions"] != len(rects):
        return None
    if metrics["black_targets"] <= 0:
        return None
    if metrics["nonblack_targets"] <= 0:
        return None
    if metrics["edge_targets"] <= 0:
        return None
    if metrics["interior_targets"] <= 0:
        return None
    if metrics["delta_cross_vs_square"] <= 0:
        return None
    if metrics["blocked_nonblue_cells"] <= 0:
        return None

    try:
        if np.array_equal(main(inp), output):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {MAX_GEN_ATTEMPTS} attempts")
