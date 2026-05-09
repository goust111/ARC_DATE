# Combo variant of a47bf94d (index 8)
# combo_spec: D5/0+D7/2
# Name: 叠加合并+堆叠放置

import random
import sys
from pathlib import Path

import numpy as np

try:
    from arc_utils import components_mask, get_bg, neighbors4
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[4] / "solvers"))
    from arc_utils import components_mask, get_bg, neighbors4

# Design analysis for the combo variant
# - concretization:
#   each destination is now a same-colored receiver hub instead of a single
#   remote empty slot. A hub has one nominal anchor diamond plus one extra
#   diamond stacked 4 cells to the right. Two same-colored solid senders route
#   to that same anchor from opposite wire endpoints. The first routed
#   complement completes the anchor, and the later complement completes the
#   next diamond in the hub row.
# - behavior trace:
#   original one channel: source solid -> local diamond; far empty spot gets X.
#   D5-only: source solid -> local diamond; far same-colored diamond becomes
#   solid by corner-overlay.
#   combo: two same-colored source solids target one hub. Input shows two
#   solids attached to the same wire line from opposite sides and a row of two
#   same-colored diamonds at the hub. Output turns the two sources into
#   diamonds and turns the hub row into two solids from left to right. The
#   difference from D5 is that several arrivals share the same nominal anchor,
#   so collision handling is required and visible.
# - complexity calibration:
#   stacking direction is fixed to the right with a 4-cell step. That removes
#   arbitrary per-sample direction choices while still preserving the core D7
#   idea: multiple arrivals compete for one destination and must spill into
#   adjacent placements. Using 2 hubs with 4 total senders keeps repeated
#   evidence in every grid without making the board cluttered.
# - hidden-assumption audit against the original solver:
#   1. original color uniqueness no longer holds; several tokens can share one
#      color because they belong to the same hub.
#   2. original far destination emptiness is false; receivers are present in the
#      input and multiple arrivals want the same anchor.
#   3. original one-sender-per-endpoint-family is false; several components can
#      map to the same anchor center.
#   4. the track mask must exclude receiver diamonds, or stacked hubs would be
#      misread as wire fragments.
#   5. destination writing cannot be a single overwrite; it must allocate the
#      next unused receiver slot in a deterministic left-to-right order.
# - placement legality:
#   source solids stay exactly two cells outward from degree-1 endpoints. Hub
#   diamonds are spaced by 4 columns, so their 3x3 footprints never overlap.
#   Hubs are separated in both row and column, and straight wire segments are
#   assigned to disjoint rows/columns, so components do not touch or cross.
# - visual observability:
#   humans can compare input/output and see two simultaneous regularities:
#   sources always lose corners, and same-colored receiver rows on the far side
#   fill from left to right into stacked full blocks. This is visibly different
#   from both the original empty-target behavior and the single-target D5-only
#   behavior.
# - cleanliness:
#   only same-colored receivers share a hub, which avoids mixed-color overlays.
#   The stack is a short two-block row rather than a dense local pile, so each
#   completed 3x3 block remains separable by eye.
# - generator guarantees for the new solver:
#   each hub has exactly two same-colored receiver diamonds, matching the two
#   arrivals aimed at that anchor; each wire component has one source endpoint
#   and one destination endpoint; all arrivals to one anchor share the hub
#   color; no two hubs overlap.

_N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIAMOND_OFF = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_SOLID_OFF = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 0),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]

STACK_STEP = 4
STACK_SLOT_OFFSETS = [(0, 0), (0, STACK_STEP), (0, 2 * STACK_STEP)]

GRID_H_RANGE = (21, 25)
GRID_W_RANGE = (24, 26)
TOKEN_COLORS_POOL = [1, 2, 3, 4, 6, 7]
GEN_MAX_RETRIES = 200

_OPP = {"N": "S", "S": "N", "E": "W", "W": "E"}
_DIR2FACE = {(1, 0): "N", (-1, 0): "S", (0, 1): "W", (0, -1): "E"}
_FACE2DIR = {"S": (1, 0), "N": (-1, 0), "E": (0, 1), "W": (0, -1)}


def _find_solid_3x3(g: np.ndarray, bg: int) -> dict[tuple[int, int], tuple[int, tuple[int, int]]]:
    """Return {center: (color, (r0,c0))} for maximal isolated 3x3 solids."""
    h, w = g.shape
    found: dict[tuple[int, int], tuple[int, tuple[int, int]]] = {}
    for r0 in range(h - 2):
        for c0 in range(w - 2):
            sub = g[r0 : r0 + 3, c0 : c0 + 3]
            col = int(sub[0, 0])
            if col == bg or np.any(sub != col):
                continue
            isolated = True
            for rr in range(r0, r0 + 3):
                for cc in range(c0, c0 + 3):
                    for nr, nc in neighbors4(rr, cc):
                        if 0 <= nr < h and 0 <= nc < w and not (r0 <= nr < r0 + 3 and c0 <= nc < c0 + 3):
                            if int(g[nr, nc]) == col:
                                isolated = False
                                break
                    if not isolated:
                        break
                if not isolated:
                    break
            if isolated:
                found[(r0 + 1, c0 + 1)] = (col, (r0, c0))
    return found


def _find_diamond_3x3(g: np.ndarray, bg: int) -> dict[tuple[int, int], tuple[int, tuple[int, int]]]:
    """Return {center: (color, (r0,c0))} for isolated 3x3 diamonds."""
    h, w = g.shape
    found: dict[tuple[int, int], tuple[int, tuple[int, int]]] = {}
    mask = np.zeros((3, 3), dtype=bool)
    for rr, cc in [(0, 1), (1, 0), (1, 2), (2, 1)]:
        mask[rr, cc] = True

    for r0 in range(h - 2):
        for c0 in range(w - 2):
            sub = g[r0 : r0 + 3, c0 : c0 + 3]
            vals = sub[mask]
            if vals.size == 0:
                continue
            col = int(vals[0])
            if col == bg or np.any(vals != col):
                continue
            if np.any(sub[~mask] != bg):
                continue
            isolated = True
            for dr, dc in _DIAMOND_OFF:
                rr, cc = r0 + 1 + dr, c0 + 1 + dc
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not (r0 <= nr < r0 + 3 and c0 <= nc < c0 + 3):
                        if int(g[nr, nc]) == col:
                            isolated = False
                            break
                if not isolated:
                    break
            if isolated:
                found[(r0 + 1, c0 + 1)] = (col, (r0, c0))
    return found


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=4)]


def _find_9rects(g: np.ndarray) -> list[tuple[set[tuple[int, int]], dict[str, list[tuple[int, int]]]]]:
    """Find color-9 rectangular blocks and their wire connections per face."""
    mask9 = g == 9
    if not mask9.any():
        return []
    h, w = g.shape
    rects = []
    for comp9 in components_mask(mask9, connectivity=4):
        cells = {(int(r), int(c)) for r, c in comp9}
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        faces: dict[str, list[tuple[int, int]]] = {}
        for r, c in cells:
            for nr, nc in neighbors4(r, c):
                if (nr, nc) not in cells and 0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) in (5, 8):
                    face = "N" if nr < r0 else "S" if nr > r1 else "W" if nc < c0 else "E"
                    faces.setdefault(face, []).append((nr, nc))
        rects.append((cells, faces))
    return rects


def _trace_wire(
    comp: set[tuple[int, int]],
    start: tuple[int, int],
    dir0: tuple[int, int],
    h: int,
    w: int,
    rects_9: list[tuple[set[tuple[int, int]], dict[str, list[tuple[int, int]]]]],
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Greedy wire following with optional color-9 rectangle pass-through."""
    cur, d, prev = start, dir0, None
    visited: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for _ in range(h * w * 4):
        if (cur, d) in visited:
            raise AssertionError("wire trace looped")
        visited.add((cur, d))

        jumped = False
        for cells9, faces9 in rects_9:
            if cur in cells9:
                entry_face = _DIR2FACE[d]
                opp = _OPP[entry_face]
                if opp in faces9:
                    exit_face = opp
                else:
                    others = [f for f in faces9 if f != entry_face]
                    if len(others) != 1:
                        raise AssertionError("ambiguous 9-rect exit")
                    exit_face = others[0]
                pts = faces9[exit_face]
                er, ec = cur
                if entry_face in ("N", "S"):
                    exit_cell = min(pts, key=lambda p: abs(p[1] - ec))
                else:
                    exit_cell = min(pts, key=lambda p: abs(p[0] - er))
                cur, d, prev = exit_cell, _FACE2DIR[exit_face], None
                jumped = True
                break
        if jumped:
            continue

        r, c = cur
        opts = [((nr, nc), nr - r, nc - c) for nr, nc in neighbors4(r, c) if (nr, nc) in comp]
        if not opts:
            return cur, d

        rev = (-d[0], -d[1])
        nonrev = [o for o in opts if (o[1], o[2]) != rev]
        opts2 = nonrev if nonrev else opts
        straight = [o for o in opts2 if (o[1], o[2]) == d]
        if straight:
            p, dr, dc = straight[0]
        else:
            p, dr, dc = opts2[0]

        if p == prev:
            return cur, d
        prev = cur
        cur = p
        d = (dr, dc)

    raise AssertionError("wire trace exceeded step limit")


def _stamp_token(grid: np.ndarray, center: tuple[int, int], color: int, offsets: list[tuple[int, int]]) -> None:
    cr, cc = center
    for dr, dc in offsets:
        grid[cr + dr, cc + dc] = color


def _slot_centers(anchor: tuple[int, int]) -> list[tuple[int, int]]:
    ar, ac = anchor
    return [(ar + dr, ac + dc) for dr, dc in STACK_SLOT_OFFSETS]


def _in_bounds_center(center: tuple[int, int], h: int, w: int) -> bool:
    r, c = center
    return 1 <= r < h - 1 and 1 <= c < w - 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route each solid sender's missing corner-set to a shared receiver hub, completing stacked solids.

    Input structure:
    - Background cells are empty.
    - Non-background cells form orthogonal wire components plus 3x3 tokens near degree-1 endpoints.
    - Source tokens are solid 3x3 blocks. Destination hubs already contain one or more same-colored diamond receivers.
    - Several different wire components can target the same hub anchor, and the hub's extra receivers are pre-stacked to the right.

    Rule:
    - Every solid sender loses its corners and becomes a diamond in place.
    - The missing corner-set travels along the attached wire to the far endpoint's nominal anchor.
    - At the destination hub, the first arrival completes the anchor diamond into a solid.
    - If more arrivals target that same anchor, they do not overwrite the finished block; instead they complete the next unused receiver diamond in the hub's rightward stack.

    Key insight:
    - The remote action is now both overlay and collision resolution: wires reveal which senders belong to the same anchor, then the output fills a same-colored receiver row from left to right with full 3x3 blocks.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = int(get_bg(g))

    solids = _find_solid_3x3(g, bg)
    diamonds = _find_diamond_3x3(g, bg)

    track_mask = g != bg
    for _, (_, (r0, c0)) in solids.items():
        track_mask[r0 : r0 + 3, c0 : c0 + 3] = False
    for (cr, cc), _ in diamonds.items():
        for dr, dc in _DIAMOND_OFF:
            track_mask[cr + dr, cc + dc] = False

    comps = _components(track_mask)
    rects_9 = _find_9rects(g)
    out = g.copy()

    for (cr, cc), (col, (r0, c0)) in solids.items():
        out[r0 : r0 + 3, c0 : c0 + 3] = bg
        _stamp_token(out, (cr, cc), col, _DIAMOND_OFF)

    arrivals: dict[tuple[tuple[int, int], int], list[tuple[int, int]]] = {}

    for pts in comps:
        comp = set(pts)
        if len(comp) < 2:
            continue

        deg = {p: 0 for p in comp}
        for r, c in comp:
            for nr, nc in neighbors4(r, c):
                if (nr, nc) in comp:
                    deg[(r, c)] += 1
        endpoints = [p for p in comp if deg[p] <= 1]
        if len(endpoints) < 2:
            continue

        for ep in endpoints:
            er, ec = ep
            for odr, odc in _N4:
                cen = (er + 2 * odr, ec + 2 * odc)
                if cen not in solids:
                    continue
                col, _ = solids[cen]
                end, dd = _trace_wire(comp, ep, (-odr, -odc), h, w, rects_9)
                anchor = (end[0] + 2 * dd[0], end[1] + 2 * dd[1])
                arrivals.setdefault((anchor, col), []).append(cen)

    for (anchor, col), src_centers in arrivals.items():
        used: set[tuple[int, int]] = set()
        for _ in sorted(src_centers):
            placed = False
            for slot in _slot_centers(anchor):
                if slot in used:
                    continue
                if slot in diamonds and diamonds[slot][0] == col:
                    sr0, sc0 = slot[0] - 1, slot[1] - 1
                    out[sr0 : sr0 + 3, sc0 : sc0 + 3] = col
                    used.add(slot)
                    placed = True
                    break
            if placed:
                continue
            for slot in _slot_centers(anchor):
                if slot in used or not _in_bounds_center(slot, h, w):
                    continue
                sr0, sc0 = slot[0] - 1, slot[1] - 1
                patch = out[sr0 : sr0 + 3, sc0 : sc0 + 3]
                if np.all((patch == bg) | (patch == col)):
                    out[sr0 : sr0 + 3, sc0 : sc0 + 3] = col
                    used.add(slot)
                    break

    return out


def _draw_vertical(grid: np.ndarray, col: int, r0: int, r1: int) -> None:
    lo, hi = sorted((r0, r1))
    grid[lo : hi + 1, col] = 8


def _draw_horizontal(grid: np.ndarray, row: int, c0: int, c1: int) -> None:
    lo, hi = sorted((c0, c1))
    grid[row, lo : hi + 1] = 8


def _embed_rect_on_vertical(grid: np.ndarray, col: int, r0: int, r1: int) -> None:
    lo, hi = sorted((r0, r1))
    safe_lo = lo + 3
    safe_hi = hi - 4
    if safe_lo > safe_hi or col + 1 >= grid.shape[1]:
        return
    rr = random.randint(safe_lo, safe_hi)
    patch = grid[rr : rr + 2, col : col + 2]
    if np.all((patch == 0) | (patch == 8)):
        grid[rr : rr + 2, col : col + 2] = 9


def _embed_rect_on_horizontal(grid: np.ndarray, row: int, c0: int, c1: int) -> None:
    lo, hi = sorted((c0, c1))
    safe_lo = lo + 3
    safe_hi = hi - 4
    if safe_lo > safe_hi or row + 1 >= grid.shape[0]:
        return
    cc = random.randint(safe_lo, safe_hi)
    patch = grid[row : row + 2, cc : cc + 2]
    if np.all((patch == 0) | (patch == 8)):
        grid[row : row + 2, cc : cc + 2] = 9


def _maybe_embed_rect(grid: np.ndarray, orient: str, fixed: int, a: int, b: int) -> None:
    if random.random() >= 0.45:
        return
    if orient == "V":
        _embed_rect_on_vertical(grid, fixed, a, b)
    else:
        _embed_rect_on_horizontal(grid, fixed, a, b)


def _hub_layout(h: int, w: int) -> list[tuple[int, int]]:
    hub1 = (random.choice([7, 8]), random.choice([6, 7]))
    # Keep the second hub far enough right that its vertical wires cannot cut
    # through the first hub's receiver stack.
    hub2 = (h - random.choice([8, 9]), w - random.choice([7, 8]))
    return [hub1, hub2]


def _source_spec(direction: str, hub: tuple[int, int], h: int, w: int) -> tuple[tuple[int, int], tuple[int, int], str, int, int]:
    hr, hc = hub
    if direction == "T":
        return (2, hc), (4, hc), "V", 4, hr - 2
    if direction == "B":
        return (h - 3, hc), (h - 5, hc), "V", hr + 2, h - 5
    if direction == "R":
        return (hr, w - 3), (hr, w - 5), "H", hc + 2, w - 5
    raise ValueError(direction)


def _build_input_output() -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    inp = np.zeros((h, w), dtype=int)
    out = np.zeros((h, w), dtype=int)

    hubs = _hub_layout(h, w)
    colors = random.sample(TOKEN_COLORS_POOL, len(hubs))
    counts = [2, 2]
    meta = {"stacked_hubs": 0, "extra_slots": 0, "sources": sum(counts)}

    for hub_idx, (hub, col, count) in enumerate(zip(hubs, colors, counts)):
        if count > 1:
            meta["stacked_hubs"] += 1
            meta["extra_slots"] += count - 1

        for slot in _slot_centers(hub)[:count]:
            _stamp_token(inp, slot, col, _DIAMOND_OFF)
            _stamp_token(out, slot, col, _SOLID_OFF)

        directions = ["T", "B"]
        for direction in directions:
            src_center, src_ep, orient, a, b = _source_spec(direction, hub, h, w)
            _stamp_token(inp, src_center, col, _SOLID_OFF)
            _stamp_token(out, src_center, col, _DIAMOND_OFF)

            if orient == "V":
                _draw_vertical(inp, src_ep[1], a, b)
                _draw_vertical(out, src_ep[1], a, b)
                _maybe_embed_rect(inp, "V", src_ep[1], a, b)
                out[:, :] = np.where(inp == 9, 9, out)
            else:
                _draw_horizontal(inp, src_ep[0], a, b)
                _draw_horizontal(out, src_ep[0], a, b)
                _maybe_embed_rect(inp, "H", src_ep[0], a, b)
                out[:, :] = np.where(inp == 9, 9, out)

        if hub_idx == 0:
            # Keep hub rows/cols visibly separated by leaving empty moat cells.
            continue

    return inp, out, meta


def _gen_attempt() -> tuple[np.ndarray, dict[str, int]] | None:
    inp, expected, meta = _build_input_output()
    try:
        actual = main(inp)
    except Exception:
        return None
    if np.array_equal(actual, expected):
        return inp, meta
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result[0]
    raise RuntimeError(f"Failed to generate valid input after {GEN_MAX_RETRIES} attempts")
