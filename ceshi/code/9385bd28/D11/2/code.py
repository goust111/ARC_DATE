# Variant of 9385bd28
# Dimension: D11 — 干扰项 (meta)
# Change: 0-1 个干扰 L 对（使用非图例颜色的 L 形角标记），解题者需忽略非图例色的标记... → 干扰使用图例中存在的颜色

# Design analysis
# - source_task: None, so the baseline is the original 9385bd28 solver/generator only.
# - Concrete variant: one legend color may own four visible L-corners instead of exactly two.
#   Two corners are the real opposite-diagonal pair that determines the fill diamond.
#   Two extra same-color corners are distractors placed on one side of a would-be rectangle
#   (top pair, bottom pair, left pair, or right pair), so they do not form a valid diagonal pair.
# - Behavior trace: original code unions all non-legend pixels of a legend color, takes one global bbox,
#   and infers the fill from that bbox's four outer corners. With same-color distractors, that bbox can
#   span both the true pair and the distractor, so the inferred diagonal is wrong or ambiguous. The new
#   solver must first split the color into connected components, classify 3-cell L components by corner
#   orientation, and then search for the unique pair whose anchors are exact opposite corners of one bbox.
# - Complexity calibration: the variant still changes only D11. Legend format, fill rule, and 2x2 block
#   behavior stay unchanged. The new freedom is tightly controlled: at most one same-color distractor pair
#   per grid, and only for a legend entry that already uses a true L-pair.
# - Implicit assumptions that break from the original main():
#   1. `marker_pts = all pixels of one color` no longer isolates one logical object.
#   2. A bbox over all same-color pixels is no longer the marker bbox.
#   3. Global corner-anchor scoring can be polluted by extra same-color corners.
#   4. A 2x2 block must be detected per connected structure, not over the whole color mask.
# - Placement legality: generator keeps all non-legend placements disjoint, respects the original forbidden
#   area around the legend, and only commits a same-color distractor if the combined mask still contains
#   exactly one valid opposite-corner L-pair for that marker color.
# - Visual observability: in input/output examples, the changed behavior is visible because a legend color
#   can appear in extra L-corners that survive unchanged in the output while only one diagonal pair drives
#   the diamond fill. This is inferable from a few examples without using color ids as hidden metadata.
# - Visual cleanliness: distractor corners are placed as a neat same-row or same-column pair rather than
#   being scattered arbitrarily, reducing clutter while still forcing spatial validation.
# - New solver assumptions after the change: each legend color has either exactly one actionable L-pair,
#   exactly one actionable 2x2 block, or no actionable structure at all; same-color extras must not create
#   a second valid opposite-corner pair. The generator enforces these assumptions by construction.

import random
from collections import Counter, deque

import numpy as np

from arc_utils import get_bg


def _find_legend(g: np.ndarray, bg: int) -> tuple[int, list[tuple[int, int, int]]]:
    """
    Legend is a vertical stack of pairs at bottom-left.
    Each legend row: (marker_color at col c, fill_color at col c+1).
    Returns (legend_col, entries) where entries are ordered bottom->top as (row, marker, fill).
    """
    h, w = g.shape
    start_r = None
    start_c = None
    for r in range(h - 1, -1, -1):
        nz = np.flatnonzero(g[r, :] != bg)
        if nz.size:
            start_r = r
            start_c = int(nz[0])
            break
    assert start_r is not None and start_c is not None and start_c < w - 1, (
        "legend must exist at bottom-left with at least 2 columns"
    )

    entries: list[tuple[int, int, int]] = []
    c = start_c
    for r in range(start_r, -1, -1):
        nz = np.flatnonzero(g[r, :] != bg)
        if nz.size == 0 or int(nz[0]) != c or int(g[r, c]) == bg:
            break
        marker = int(g[r, c])
        fill = int(g[r, c + 1]) if c + 1 < w else bg
        entries.append((r, marker, fill))
    return c, entries


def _diamond_mask(
    h: int,
    w: int,
    r0: int,
    r1: int,
    c0: int,
    c1: int,
    a: tuple[int, int],
    b: tuple[int, int],
) -> np.ndarray:
    """
    Mask for all cells inside [r0..r1]x[c0..c1] that lie on some shortest 4-neigh path between corners a and b:
      dist(a,p) + dist(p,b) == dist(a,b)
    """
    ar, ac = a
    br, bc = b
    dab = abs(ar - br) + abs(ac - bc)
    m = np.zeros((h, w), dtype=bool)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if abs(r - ar) + abs(c - ac) + abs(r - br) + abs(c - bc) == dab:
                m[r, c] = True
    return m


_L_OFFSETS = {
    "TL": [(0, 0), (0, 1), (1, 0)],
    "TR": [(0, 0), (0, -1), (1, 0)],
    "BL": [(0, 0), (0, 1), (-1, 0)],
    "BR": [(0, 0), (0, -1), (-1, 0)],
}

_L_PATTERNS = {
    "TL": {(0, 0), (0, 1), (1, 0)},
    "TR": {(0, 0), (0, 1), (1, 1)},
    "BL": {(0, 0), (1, 0), (1, 1)},
    "BR": {(0, 1), (1, 0), (1, 1)},
}


def _connected_components(mask: np.ndarray) -> list[tuple[tuple[int, int], ...]]:
    """Return 4-connected True components as sorted cell tuples."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[tuple[tuple[int, int], ...]] = []
    for r, c in map(tuple, np.argwhere(mask)):
        if seen[r, c]:
            continue
        q = deque([(int(r), int(c))])
        seen[r, c] = True
        cells: list[tuple[int, int]] = []
        while q:
            rr, cc = q.popleft()
            cells.append((rr, cc))
            for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(tuple(sorted(cells)))
    comps.sort(key=lambda comp: (comp[0][0], comp[0][1], len(comp)))
    return comps


def _single_l_cells(name: str, r: int, c: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for dr, dc in _L_OFFSETS[name]]


def _l_pair_cells(rmin: int, cmin: int, rmax: int, cmax: int, diag: str) -> list[tuple[int, int]]:
    """Return the 6 cells for an L-pair at opposite bbox corners. diag in {'TLBR','TRBL'}."""
    if diag == "TLBR":
        anchors = [("TL", rmin, cmin), ("BR", rmax, cmax)]
    else:
        anchors = [("TR", rmin, cmax), ("BL", rmax, cmin)]
    cells: list[tuple[int, int]] = []
    for name, r, c in anchors:
        cells.extend(_single_l_cells(name, r, c))
    return cells


def _l_pair_valid_bbox(h: int, w: int) -> bool:
    """True if (h, w) avoids an L-anchor score tie."""
    if h < 2 or w < 2:
        return False
    if h == 2 and w <= 3:
        return False
    if w == 2 and h <= 3:
        return False
    return True


def _classify_l_component(comp: tuple[tuple[int, int], ...]) -> dict | None:
    """Return orientation/anchor info if `comp` is an exact 3-cell L corner."""
    if len(comp) != 3:
        return None
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    rmin, rmax = min(rs), max(rs)
    cmin, cmax = min(cs), max(cs)
    if rmax - rmin != 1 or cmax - cmin != 1:
        return None
    norm = {(r - rmin, c - cmin) for r, c in comp}
    for name, pattern in _L_PATTERNS.items():
        if norm != pattern:
            continue
        if name == "TL":
            anchor = (rmin, cmin)
        elif name == "TR":
            anchor = (rmin, cmax)
        elif name == "BL":
            anchor = (rmax, cmin)
        else:
            anchor = (rmax, cmax)
        return {
            "kind": "L",
            "name": name,
            "anchor": anchor,
            "cells": set(comp),
        }
    return None


def _classify_block_component(comp: tuple[tuple[int, int], ...]) -> dict | None:
    if len(comp) != 4:
        return None
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    rmin, rmax = min(rs), max(rs)
    cmin, cmax = min(cs), max(cs)
    if rmax - rmin != 1 or cmax - cmin != 1:
        return None
    expected = {
        (rmin, cmin),
        (rmin, cmax),
        (rmax, cmin),
        (rmax, cmax),
    }
    if set(comp) != expected:
        return None
    return {
        "kind": "block",
        "cells": expected,
    }


def _pair_from_components(a: dict, b: dict) -> dict | None:
    names = {a["name"], b["name"]}
    if names == {"TL", "BR"}:
        tl = a if a["name"] == "TL" else b
        br = a if a["name"] == "BR" else b
        rmin, cmin = tl["anchor"]
        rmax, cmax = br["anchor"]
        diag = "TLBR"
    elif names == {"TR", "BL"}:
        tr = a if a["name"] == "TR" else b
        bl = a if a["name"] == "BL" else b
        rmin, cmin = bl["anchor"][0], bl["anchor"][1]
        rmax, cmax = tr["anchor"][0], tr["anchor"][1]
        diag = "TRBL"
    else:
        return None

    if rmin >= rmax or cmin >= cmax:
        return None
    if not _l_pair_valid_bbox(rmax - rmin + 1, cmax - cmin + 1):
        return None

    expected = set(_l_pair_cells(rmin, cmin, rmax, cmax, diag))
    actual = set(a["cells"]) | set(b["cells"])
    if actual != expected:
        return None

    return {
        "diag": diag,
        "bbox": (rmin, cmin, rmax, cmax),
        "cells": expected,
    }


def _analyze_marker_mask(mask: np.ndarray) -> dict:
    comps = _connected_components(mask)
    l_components: list[dict] = []
    blocks: list[dict] = []
    for comp in comps:
        block = _classify_block_component(comp)
        if block is not None:
            blocks.append(block)
        l_comp = _classify_l_component(comp)
        if l_comp is not None:
            l_components.append(l_comp)

    valid_pairs: list[dict] = []
    for i in range(len(l_components)):
        for j in range(i + 1, len(l_components)):
            pair = _pair_from_components(l_components[i], l_components[j])
            if pair is not None:
                valid_pairs.append(pair)
    return {
        "components": comps,
        "l_components": l_components,
        "blocks": blocks,
        "valid_pairs": valid_pairs,
    }


def _choose_marker_structure(mask: np.ndarray) -> dict | None:
    info = _analyze_marker_mask(mask)
    valid_pairs = info["valid_pairs"]
    blocks = info["blocks"]
    if len(valid_pairs) > 1:
        raise AssertionError("marker color has multiple valid opposite-corner pairs")
    if len(valid_pairs) == 1:
        if blocks:
            raise AssertionError("marker color mixes a valid pair and a valid block")
        pair = valid_pairs[0]
        return {
            "kind": "pair",
            "diag": pair["diag"],
            "bbox": pair["bbox"],
            "cells": pair["cells"],
        }
    if len(blocks) > 1:
        raise AssertionError("marker color has multiple valid 2x2 blocks")
    if len(blocks) == 1:
        return {
            "kind": "block",
            "cells": blocks[0]["cells"],
        }
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill diamonds indicated by the bottom-left legend, even when a legend color has extra same-color decoys.

    Input structure:
    - The lower-left corner contains a vertical 2-column legend; each row gives (marker_color, fill_color).
    - Outside the legend, a marker color can form either one true opposite-corner L-pair or one solid 2x2 block.
    - A true L-pair color may also have extra L-shaped corners in the same color, but those extras are not positioned
      as an opposite-corner pair of one rectangle and must be ignored.

    Rule:
    - Read legend rows from bottom to top.
    - For each marker color, split its non-legend pixels into connected components and search for the unique actionable
      structure: either one exact 2x2 block, or one pair of 3-cell L components whose orientations and anchors match
      opposite corners of a valid bbox.
    - Recolor a 2x2 block directly; for a true L-pair, fill the shortest-path diamond on the opposite diagonal of that bbox.
    - Preserve all legend cells, all later marker colors, and all marker-colored pixels of the current entry, including
      ignored same-color distractors.

    Key insight:
    - Color membership alone is no longer enough. A marker color is actionable only when two of its L-components fit the
      exact opposite-corner geometry of one bbox; same-color L corners in any other spatial arrangement are decoys.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg = get_bg(g)
    legend_c, entries = _find_legend(g, bg)
    assert entries, "legend must contain at least one (marker, fill) entry"

    legend_mask = np.zeros_like(g, dtype=bool)
    for r, _, _ in entries:
        if 0 <= r < h and 0 <= legend_c < w:
            legend_mask[r, legend_c] = True
        if 0 <= r < h and 0 <= legend_c + 1 < w:
            legend_mask[r, legend_c + 1] = True

    out = g.copy()
    marker_order = [m for _, m, _ in entries]

    for i, (_, marker, fill) in enumerate(entries):
        current_mask = (g == marker) & (~legend_mask)
        if not np.any(current_mask):
            continue

        structure = _choose_marker_structure(current_mask)
        if structure is None:
            continue

        future_markers = set(marker_order[i + 1 :])
        protect = legend_mask.copy()
        for fm in future_markers:
            protect |= (g == fm) & (~legend_mask)

        eff_fill = int(fill)
        if structure["kind"] == "block":
            block_mask = np.zeros_like(g, dtype=bool)
            for r, c in structure["cells"]:
                block_mask[r, c] = True
            out[block_mask & (~protect)] = eff_fill
            continue

        protect |= current_mask
        rmin, cmin, rmax, cmax = structure["bbox"]
        if structure["diag"] == "TLBR":
            a, b = (rmin, cmax), (rmax, cmin)
        else:
            a, b = (rmin, cmin), (rmax, cmax)
        diamond = _diamond_mask(h, w, rmin, rmax, cmin, cmax, a, b)
        out[diamond & (~protect)] = eff_fill

    return out


# --------------- generator (C1) ---------------

# ── CFG: canvas & grid ──────────────────────────────────────────────
GRID_H_RANGE = (10, 19)          # grid height bounds
GRID_W_RANGE = (10, 19)          # grid width bounds

# ── CFG: legend ─────────────────────────────────────────────────────
LEGEND_COL_RANGE = (0, 2)        # legend left column (marker at c, fill at c+1)
LEGEND_ENTRY_RANGE = (2, 5)      # number of legend entries

# ── CFG: marker types ──────────────────────────────────────────────
NOOP_PROB = 0.3                  # prob of one noop entry (marker with no pixels)
BLOCK_PROB = 0.25                # prob of one 2x2 block entry

# ── CFG: L-pair bbox shape ──────────────────────────────────────────
BBOX_H_MAX = 19                  # max bbox height
BBOX_W_MAX = 12                  # max bbox width
BBOX_SHAPE_WEIGHTS = [0.75, 0.15, 0.10]  # [both>=3, h=2/w>=4, h>=4/w=2]
TLBR_PROB = 0.8                  # prob of TL-BR diagonal (vs TR-BL)

# ── CFG: distractor ────────────────────────────────────────────────
DISTRACTOR_PROB = 0.7            # prob of adding one same-color invalid L-pair

# ── CFG: retry budgets ─────────────────────────────────────────────
BLOCK_RETRIES = 100              # placement attempts for a 2x2 block
LPAIR_RETRIES = 150              # placement attempts for a true L-pair
DISTRACTOR_RETRIES = 90          # placement attempts for same-color distractor
GEN_RETRIES = 300                # outer generate_input retry budget


def _pick_bbox_size(H: int, W: int) -> tuple[int, int] | None:
    """Pick a random valid L-pair bbox size that fits in (H, W)."""
    shape_choice = random.choices(
        ["both_ge3", "h2_w_ge4", "h_ge4_w2"],
        weights=BBOX_SHAPE_WEIGHTS,
    )[0]
    if shape_choice == "both_ge3":
        bh = random.randint(3, min(BBOX_H_MAX, H))
        bw = random.randint(3, min(BBOX_W_MAX, W))
    elif shape_choice == "h2_w_ge4":
        bh = 2
        w_hi = min(BBOX_W_MAX, W)
        if w_hi < 4:
            return None
        bw = random.randint(4, w_hi)
    else:
        h_hi = min(BBOX_H_MAX, H)
        if h_hi < 4:
            return None
        bh = random.randint(4, h_hi)
        bw = 2
    if bh > H or bw > W or not _l_pair_valid_bbox(bh, bw):
        return None
    return bh, bw


def _try_place_l_pair(H: int, W: int, color: int, retries: int, can_place, do_place) -> bool:
    """Try to place one true opposite-corner L-pair."""
    for _ in range(retries):
        size = _pick_bbox_size(H, W)
        if size is None:
            continue
        bh, bw = size
        rmin = random.randint(0, H - bh)
        cmin = random.randint(0, W - bw)
        diag = "TLBR" if random.random() < TLBR_PROB else "TRBL"
        cells = _l_pair_cells(rmin, cmin, rmin + bh - 1, cmin + bw - 1, diag)
        if can_place(cells):
            do_place(cells, color)
            return True
    return False


def _try_place_same_color_distractor(
    H: int,
    W: int,
    color: int,
    retries: int,
    current_mask: np.ndarray,
    can_place,
    do_place,
) -> bool:
    """Add two same-color L corners that do not form an opposite-corner pair."""
    base = _choose_marker_structure(current_mask)
    if base is None or base["kind"] != "pair":
        return False
    base_sig = (base["diag"], base["bbox"])

    layouts = ["top", "bottom", "left", "right"]
    for _ in range(retries):
        layout = random.choice(layouts)
        if layout in {"top", "bottom"}:
            w_hi = min(BBOX_W_MAX, W)
            if w_hi < 4:
                continue
            span = random.randint(4, w_hi)
            cmin = random.randint(0, W - span)
            cmax = cmin + span - 1
            if layout == "top":
                r = random.randint(0, H - 2)
                anchors = [("TL", r, cmin), ("TR", r, cmax)]
            else:
                r = random.randint(1, H - 1)
                anchors = [("BL", r, cmin), ("BR", r, cmax)]
        else:
            h_hi = min(BBOX_H_MAX, H)
            if h_hi < 4:
                continue
            span = random.randint(4, h_hi)
            rmin = random.randint(0, H - span)
            rmax = rmin + span - 1
            if layout == "left":
                c = random.randint(0, W - 2)
                anchors = [("TL", rmin, c), ("BL", rmax, c)]
            else:
                c = random.randint(1, W - 1)
                anchors = [("TR", rmin, c), ("BR", rmax, c)]

        cells: list[tuple[int, int]] = []
        for name, r, c in anchors:
            cells.extend(_single_l_cells(name, r, c))
        if len(set(cells)) != 6 or not can_place(cells):
            continue

        test_mask = current_mask.copy()
        for r, c in cells:
            test_mask[r, c] = True
        info = _analyze_marker_mask(test_mask)
        if len(info["valid_pairs"]) != 1:
            continue
        pair = info["valid_pairs"][0]
        if (pair["diag"], pair["bbox"]) != base_sig:
            continue

        do_place(cells, color)
        return True
    return False


def _build_legend(H, W, bg):
    """Create legend entries and the corresponding grid/mask arrays."""
    legend_c = random.randint(*LEGEND_COL_RANGE)
    legend_bot = random.choice([H - 1, H - 2])
    n_entries = random.randint(*LEGEND_ENTRY_RANGE)
    if legend_bot - n_entries + 1 < 0:
        return None

    pool = [c for c in range(10) if c != bg]
    if len(pool) < n_entries:
        return None
    marker_colors = random.sample(pool, n_entries)

    grid = np.full((H, W), bg, dtype=int)
    legend_rows: list[tuple[int, int, int]] = []
    legend_mask = np.zeros((H, W), dtype=bool)
    for i in range(n_entries):
        r = legend_bot - i
        m = marker_colors[i]
        fill = random.randint(0, 9)
        legend_rows.append((r, m, fill))
        grid[r, legend_c] = m
        legend_mask[r, legend_c] = True
        if legend_c + 1 < W:
            grid[r, legend_c + 1] = fill
            legend_mask[r, legend_c + 1] = True

    return grid, legend_rows, legend_mask, legend_c, legend_bot, marker_colors


def _build_forbidden(H, W, legend_c, legend_bot, legend_rows):
    """Build the forbidden-placement mask that protects legend scan integrity."""
    forbidden = np.zeros((H, W), dtype=bool)
    if legend_bot + 1 < H:
        forbidden[legend_bot + 1 :, :] = True
    forbidden[: legend_bot + 1, : legend_c + 1] = True
    for r, _, _ in legend_rows:
        forbidden[r, legend_c] = False
    return forbidden


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    bg = random.randint(0, 9)

    legend_result = _build_legend(H, W, bg)
    if legend_result is None:
        return None
    grid, legend_rows, legend_mask, legend_c, legend_bot, marker_colors = legend_result

    forbidden = _build_forbidden(H, W, legend_c, legend_bot, legend_rows)
    occupied = legend_mask.copy()

    n_entries = len(legend_rows)
    types = ["L"] * n_entries
    idx_pool = list(range(n_entries))
    random.shuffle(idx_pool)
    if random.random() < NOOP_PROB and len(idx_pool) > 1:
        types[idx_pool.pop()] = "noop"
    if random.random() < BLOCK_PROB and idx_pool:
        types[idx_pool.pop()] = "block"

    def can_place(cells):
        for r, c in cells:
            if not (0 <= r < H and 0 <= c < W):
                return False
            if occupied[r, c] or forbidden[r, c]:
                return False
        return True

    def do_place(cells, color):
        for r, c in cells:
            grid[r, c] = color
            occupied[r, c] = True

    for i, (_, m, _) in enumerate(legend_rows):
        t = types[i]
        if t == "noop":
            continue
        if t == "block":
            placed = False
            for _ in range(BLOCK_RETRIES):
                r0 = random.randint(0, H - 2)
                c0 = random.randint(0, W - 2)
                cells = [(r0, c0), (r0, c0 + 1), (r0 + 1, c0), (r0 + 1, c0 + 1)]
                if can_place(cells):
                    do_place(cells, m)
                    placed = True
                    break
            if not placed:
                return None
        else:
            if not _try_place_l_pair(H, W, m, LPAIR_RETRIES, can_place, do_place):
                return None

    want_distractor = random.random() < DISTRACTOR_PROB
    if want_distractor:
        candidates = [i for i, t in enumerate(types) if t == "L"]
        random.shuffle(candidates)
        placed_distractor = False
        for idx in candidates:
            marker = legend_rows[idx][1]
            current_mask = (grid == marker) & (~legend_mask)
            if _try_place_same_color_distractor(
                H,
                W,
                marker,
                DISTRACTOR_RETRIES,
                current_mask,
                can_place,
                do_place,
            ):
                placed_distractor = True
                break
        if candidates and not placed_distractor:
            return None

    counts = Counter(grid.flatten().tolist())
    if counts.most_common(1)[0][0] != bg:
        return None

    try:
        _ = main(grid)
    except Exception:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(
        "generate_input: failed to generate a valid input after %d attempts" % GEN_RETRIES
    )
