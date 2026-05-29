# Combo variant of 2c181942 (index 36)
# combo_spec: D1/0+D6/1+D10/0
# Name: 形状匹配+同色干扰+多hub

from __future__ import annotations

import random
from typing import Any

import numpy as np

from arc_utils import bbox_coords, get_bg, label_mask, paste, random_free_location


# ---------------------------------------------------------------------------
# Design analysis for the fused variant
# ---------------------------------------------------------------------------
#
# a) Concrete fused behavior and trace
#    - Visible objects:
#      1. Multiple hub-like objects are present. Each hub has:
#         * a 6x6 hollow square core with bg corners/interior;
#         * four colored core edges;
#         * a same-color 2-cell-thick dock around the core whose outer rim contains
#           side-specific notch patterns.
#      2. Several detached sprites all use exactly the same monochrome dock color.
#         Real sprites have a 2-row / 2-col contact mouth whose protrusion pattern is
#         complementary to one dock side. A same-color distractor sprite may also exist
#         but has no legal mouth in any rotation.
#    - Tiny execution trace:
#      original task:
#        * pick the center-most hub;
#        * use color to map one sprite to one side;
#        * keep the largest same-color component;
#        * rotate and attach.
#      fused task:
#        * enumerate all 6x6 hollow-square cores with legal docks;
#        * select the true hub as the unique candidate whose four core-edge colors are
#          all distinct, not the one nearest the center;
#        * gather every same-color component outside every hub bbox;
#        * for each side of the chosen hub, test every unused component in all 4
#          rotations against the side's visible notch pattern;
#        * attach only components with exactly one legal geometric match.
#      difference:
#        * D10: center proximity is now a decoy; a closer hub can be false.
#        * D1: sprites no longer match by hub-edge color; side identity comes from the
#          dock notch geometry.
#        * D6: a same-color distractor can be larger than a real sprite, so geometry
#          must filter components before selection.
#    - Example mental picture:
#      the true hub may have core edge colors (1,2,3,4) while a center decoy repeats
#      a color, e.g. (5,6,5,7). All docks and all sprites use color 9. The top dock's
#      outer rim exposes pattern 1000, so only the sprite whose mouth is "support row +
#      one-cell protrusion" can attach there; a bigger same-color blob is ignored.
#
# b) Complexity calibration
#    - The hub core keeps only four edge colors plus one shared dock/sprite color, so
#      the palette stays readable.
#    - Dock patterns use exactly four fixed 4-bit codes, one per side. This is enough
#      to make side matching observable without creating a combinatorial shape soup.
#    - Decoy hubs are structurally similar to the true hub and at least one is closer
#      to the grid center, so the D10 change is active in typical samples.
#    - Distractor sprites are same-color and often larger than true sprites, so "pick
#      the largest component" is visibly wrong.
#
# c) Implicit-assumption audit of the original solver
#    - Original `_find_hub()` assumed the nearest hollow square is the hub. Broken by
#      D10, so the new solver enumerates all hub candidates and scores `all_distinct`
#      before anything else.
#    - Original `(g == side_color) & (~hub_mask)` assumed side color identifies the
#      sprite pool. Broken by D1, because all sprites share one dock color. The new
#      solver instead reads the dock color from the hub candidate and inspects every
#      outside component of that one color.
#    - Original "largest monochrome component of that color is the sprite" is broken by
#      D6. The new solver tests every component independently and only accepts a
#      component if it has exactly one legal geometric match for that side.
#    - With multiple hubs, excluding only the chosen hub is unsafe because decoy hub
#      docks share the sprite color. The new solver excludes the full 10x10 bbox of
#      every candidate hub from sprite search.
#    - New solver assumption: exactly one hub candidate has four distinct core-edge
#      colors, and real sprites have exactly one valid (rotation, side) match while
#      distractors have zero. The generator enforces both.
#
# d) Placement legality
#    - Generation is output-first: construct the true hub plus attached real sprites,
#      then scatter rotated sprites into the input.
#    - Hubs reserve their full 10x10 dock bboxes so no sprite can land inside a hub's
#      hollow interior or dock holes.
#    - Scattered components use `random_free_location(..., padding=2)` to preserve an
#      8-connected gap from hubs and other sprites.
#    - Attached sprites may overlap only the bg notch cells of the chosen dock; any
#      overlap with colored hub pixels is rejected.
#
# e) Visual observability
#    - Across 2-4 examples, a human sees:
#      * several hub candidates, but only one has four different core-edge colors;
#      * all detached pieces share the dock color, so color cannot assign sides;
#      * each dock side has a different notch silhouette and only matching sprite mouths
#        reappear attached in the output.
#    - No hidden ordering by color id is used. Distinctness of four edge colors is a
#      human-visible predicate.
#
# f) Visual neatness
#    - The multicolored signal is confined to the 6x6 core. The same-color docking
#      mechanism sits outside it, so "choose hub" and "match sprite" cues are separated
#      spatially instead of overdrawn on the same pixels.
#    - Padding between hubs and sprites prevents merged components or cluttered overlap.
#
# g) Color-rule inferability
#    - Core edge colors do not drive attachment; they only identify the true hub via the
#      visible "all four different" property.
#    - All movable pieces reuse one dock color, which makes the shape rule salient and
#      avoids asking the solver to infer an arbitrary color mapping from few examples.


HUB_N = 6
EDGE_LEN = HUB_N - 2
DOCK_THICKNESS = 2
FULL_HUB_N = HUB_N + 2 * DOCK_THICKNESS
BG_COLOR = 8
SIDES = ("top", "bot", "left", "right")
PATTERNS = {
    "top": (1, 0, 0, 0),
    "bot": (1, 1, 0, 0),
    "left": (1, 0, 1, 0),
    "right": (1, 1, 1, 0),
}


def _crop_to_mask(arr: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.argwhere(mask)
    r0, c0, r1, c1 = bbox_coords(pts)
    return arr[r0 : r1 + 1, c0 : c1 + 1], mask[r0 : r1 + 1, c0 : c1 + 1]


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    dummy = np.where(mask, 1, 0).astype(int)
    _, cropped = _crop_to_mask(dummy, mask)
    return cropped


def _side_span(r: int, c: int, side: str) -> tuple[int, int]:
    if side in ("top", "bot"):
        return int(c + 1), int(c + HUB_N - 2)
    return int(r + 1), int(r + HUB_N - 2)


def _dock_inner_coord(r: int, c: int, side: str) -> int:
    if side == "top":
        return int(r - 1)
    if side == "bot":
        return int(r + HUB_N)
    if side == "left":
        return int(c - 1)
    return int(c + HUB_N)


def _dock_outer_coord(r: int, c: int, side: str) -> int:
    if side == "top":
        return int(r - 2)
    if side == "bot":
        return int(r + HUB_N + 1)
    if side == "left":
        return int(c - 2)
    return int(c + HUB_N + 1)


def _full_hub_bbox(r: int, c: int) -> tuple[int, int, int, int]:
    return int(r - 2), int(c - 2), int(r + HUB_N + 1), int(c + HUB_N + 1)


def _center_score_pos(r: int, c: int, h: int, w: int, H: int, W: int) -> float:
    cr = float(r + (h - 1) / 2)
    cc = float(c + (w - 1) / 2)
    return abs(float(cr - (H - 1) / 2)) + abs(float(cc - (W - 1) / 2))


def _contact_band(mask: np.ndarray, side: str, pattern: tuple[int, ...]) -> int | None:
    h, w = mask.shape
    L = len(pattern)
    hits: list[int] = []

    if side == "top":
        if h < 2 or w < L:
            return None
        for j in range(w - L + 1):
            if bool(mask[h - 2, :j].any()) or bool(mask[h - 2, j + L :].any()):
                continue
            if bool(mask[h - 1, :j].any()) or bool(mask[h - 1, j + L :].any()):
                continue
            if not bool(mask[h - 2, j : j + L].all()):
                continue
            if tuple(int(v) for v in mask[h - 1, j : j + L]) != pattern:
                continue
            hits.append(int(j))
    elif side == "bot":
        if h < 2 or w < L:
            return None
        for j in range(w - L + 1):
            if bool(mask[1, :j].any()) or bool(mask[1, j + L :].any()):
                continue
            if bool(mask[0, :j].any()) or bool(mask[0, j + L :].any()):
                continue
            if not bool(mask[1, j : j + L].all()):
                continue
            if tuple(int(v) for v in mask[0, j : j + L]) != pattern:
                continue
            hits.append(int(j))
    elif side == "left":
        if w < 2 or h < L:
            return None
        for j in range(h - L + 1):
            if bool(mask[:j, w - 2].any()) or bool(mask[j + L :, w - 2].any()):
                continue
            if bool(mask[:j, w - 1].any()) or bool(mask[j + L :, w - 1].any()):
                continue
            if not bool(mask[j : j + L, w - 2].all()):
                continue
            if tuple(int(v) for v in mask[j : j + L, w - 1]) != pattern:
                continue
            hits.append(int(j))
    else:
        if w < 2 or h < L:
            return None
        for j in range(h - L + 1):
            if bool(mask[:j, 1].any()) or bool(mask[j + L :, 1].any()):
                continue
            if bool(mask[:j, 0].any()) or bool(mask[j + L :, 0].any()):
                continue
            if not bool(mask[j : j + L, 1].all()):
                continue
            if tuple(int(v) for v in mask[j : j + L, 0]) != pattern:
                continue
            hits.append(int(j))

    if len(hits) != 1:
        return None
    return int(hits[0])


def _attachment_shift(mask: np.ndarray, side: str, hub_r: int, hub_c: int) -> tuple[int, int] | None:
    pos = _contact_band(mask, side, PATTERNS[side])
    if pos is None:
        return None

    span_start, _span_end = _side_span(hub_r, hub_c, side)
    outer = _dock_outer_coord(hub_r, hub_c, side)
    h, w = mask.shape

    if side == "top":
        return int(outer - (h - 1)), int(span_start - pos)
    if side == "bot":
        return int(outer), int(span_start - pos)
    if side == "left":
        return int(span_start - pos), int(outer - (w - 1))
    return int(span_start - pos), int(outer)


def _placeable(
    out: np.ndarray,
    arr: np.ndarray,
    mask: np.ndarray,
    dr: int,
    dc: int,
    bg: int,
) -> bool:
    H, W = out.shape
    for rr, cc in np.argwhere(mask):
        R = int(rr + dr)
        C = int(cc + dc)
        if not (0 <= R < H and 0 <= C < W):
            return False
        val = int(arr[rr, cc])
        if val == bg:
            continue
        if int(out[R, C]) != bg and int(out[R, C]) != val:
            return False
    return True


def _count_valid_matches(mask: np.ndarray) -> list[tuple[int, str]]:
    matches: list[tuple[int, str]] = []
    for krot in range(4):
        rot = _crop_mask(np.rot90(mask, int(krot)))
        for side, pattern in PATTERNS.items():
            if _contact_band(rot, side, pattern) is not None:
                matches.append((int(krot), side))
    return matches


def _match_component_to_side(
    comp_arr: np.ndarray,
    comp_mask: np.ndarray,
    side: str,
    hub_r: int,
    hub_c: int,
    out: np.ndarray,
    bg: int,
) -> tuple[np.ndarray, np.ndarray, int, int] | None:
    hits = []
    for krot in range(4):
        arr = np.rot90(comp_arr, int(krot))
        mask = np.rot90(comp_mask, int(krot))
        arr, mask = _crop_to_mask(arr, mask)
        shift = _attachment_shift(mask, side, hub_r, hub_c)
        if shift is None:
            continue
        dr, dc = shift
        if not _placeable(out, arr, mask, dr, dc, bg):
            continue
        hits.append((arr, mask, int(dr), int(dc)))
    if len(hits) != 1:
        return None
    return hits[0]


def _read_hub_candidate(g: np.ndarray, bg: int, r: int, c: int) -> dict[str, Any] | None:
    H, W = g.shape
    r0, c0, r1, c1 = _full_hub_bbox(r, c)
    if r0 < 0 or c0 < 0 or r1 >= H or c1 >= W:
        return None

    if (
        int(g[r, c]) != bg
        or int(g[r, c + HUB_N - 1]) != bg
        or int(g[r + HUB_N - 1, c]) != bg
        or int(g[r + HUB_N - 1, c + HUB_N - 1]) != bg
    ):
        return None

    inner = g[r + 1 : r + HUB_N - 1, c + 1 : c + HUB_N - 1]
    if not bool((inner == bg).all()):
        return None

    top = g[r, c + 1 : c + HUB_N - 1]
    bot = g[r + HUB_N - 1, c + 1 : c + HUB_N - 1]
    left = g[r + 1 : r + HUB_N - 1, c]
    right = g[r + 1 : r + HUB_N - 1, c + HUB_N - 1]
    if (
        bool((top == bg).any())
        or bool((bot == bg).any())
        or bool((left == bg).any())
        or bool((right == bg).any())
    ):
        return None
    if (
        len(set(map(int, top))) != 1
        or len(set(map(int, bot))) != 1
        or len(set(map(int, left))) != 1
        or len(set(map(int, right))) != 1
    ):
        return None

    dock_color: int | None = None
    patterns: dict[str, tuple[int, ...]] = {}
    for side in SIDES:
        start, end = _side_span(r, c, side)
        inner_coord = _dock_inner_coord(r, c, side)
        outer_coord = _dock_outer_coord(r, c, side)

        if side in ("top", "bot"):
            inner_band = g[inner_coord, start : end + 1]
            outer_band = g[outer_coord, start : end + 1]
        else:
            inner_band = g[start : end + 1, inner_coord]
            outer_band = g[start : end + 1, outer_coord]

        if bool((inner_band == bg).any()) or len(set(map(int, inner_band))) != 1:
            return None
        side_dock_color = int(inner_band[0])
        if dock_color is None:
            dock_color = side_dock_color
        elif side_dock_color != dock_color:
            return None

        pat = []
        for v in outer_band:
            vv = int(v)
            if vv == bg:
                pat.append(1)
            elif vv == dock_color:
                pat.append(0)
            else:
                return None
        patterns[side] = tuple(int(x) for x in pat)

    if dock_color is None or dock_color == bg:
        return None
    if patterns != PATTERNS:
        return None

    edge_colors = (int(top[0]), int(bot[0]), int(left[0]), int(right[0]))
    return {
        "r": int(r),
        "c": int(c),
        "edge_colors": edge_colors,
        "dock_color": int(dock_color),
        "all_distinct": len(set(edge_colors)) == 4,
        "center_score": float(_center_score_pos(r, c, HUB_N, HUB_N, H, W)),
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
    }


def _hub_candidates(g: np.ndarray, bg: int) -> list[dict[str, Any]]:
    H, W = g.shape
    out: list[dict[str, Any]] = []
    for r in range(2, H - HUB_N):
        for c in range(2, W - HUB_N):
            cand = _read_hub_candidate(g, bg, int(r), int(c))
            if cand is not None:
                out.append(cand)
    return out


def _all_hubs_mask(shape: tuple[int, int], candidates: list[dict[str, Any]]) -> np.ndarray:
    H, W = shape
    mask = np.zeros((H, W), dtype=bool)
    for cand in candidates:
        r0, c0, r1, c1 = cand["bbox"]
        mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _sprite_components(
    g: np.ndarray,
    bg: int,
    sprite_color: int,
    all_hubs_mask: np.ndarray,
) -> list[dict[str, Any]]:
    m = (g == int(sprite_color)) & (~all_hubs_mask)
    lab, nc = label_mask(m, connectivity=8)
    comps: list[dict[str, Any]] = []
    for cid in range(1, int(nc) + 1):
        full_mask = lab == int(cid)
        arr = np.where(full_mask, int(sprite_color), int(bg)).astype(int)
        arr, mask = _crop_to_mask(arr, full_mask)
        comps.append(
            {
                "id": int(cid),
                "arr": arr,
                "mask": mask,
                "area": int(mask.sum()),
            }
        )
    comps.sort(key=lambda item: -int(item["area"]))
    return comps


def _choose_hub_and_matches(
    g: np.ndarray,
    bg: int,
    candidates: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[tuple[str, np.ndarray, np.ndarray, int, int]]]:
    if not candidates:
        raise AssertionError("no hub candidates found")

    all_hubs_mask = _all_hubs_mask(g.shape, candidates)
    best_cand = None
    best_matches: list[tuple[str, np.ndarray, np.ndarray, int, int]] = []
    best_score = None

    for cand in candidates:
        comps = _sprite_components(g, bg, int(cand["dock_color"]), all_hubs_mask)
        r = int(cand["r"])
        c = int(cand["c"])
        r0, c0, r1, c1 = cand["bbox"]
        out = np.full_like(g, int(bg))
        out[r0 : r1 + 1, c0 : c1 + 1] = g[r0 : r1 + 1, c0 : c1 + 1]

        used: set[int] = set()
        matches: list[tuple[str, np.ndarray, np.ndarray, int, int]] = []
        total_area = 0

        for side in SIDES:
            found = None
            found_id = None
            for comp in comps:
                cid = int(comp["id"])
                if cid in used:
                    continue
                match = _match_component_to_side(
                    comp["arr"],
                    comp["mask"],
                    side,
                    r,
                    c,
                    out,
                    bg,
                )
                if match is None:
                    continue
                found = match
                found_id = cid
                break
            if found is None:
                continue
            arr, mask, dr, dc = found
            paste(out, arr, int(dr), int(dc), mask=mask, overwrite=True, clip=True)
            used.add(int(found_id))
            matches.append((side, arr, mask, int(dr), int(dc)))
            total_area += int(mask.sum())

        score = (
            int(cand["all_distinct"]),
            len(matches),
            int(total_area),
            -float(cand["center_score"]),
        )
        if best_score is None or score > best_score:
            best_score = score
            best_cand = cand
            best_matches = matches

    if best_cand is None:
        raise AssertionError("failed to choose hub")
    return best_cand, best_matches


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    candidates = _hub_candidates(g, bg)
    hub, matches = _choose_hub_and_matches(g, bg, candidates)

    r0, c0, r1, c1 = hub["bbox"]
    out = np.full_like(g, int(bg))
    out[r0 : r1 + 1, c0 : c1 + 1] = g[r0 : r1 + 1, c0 : c1 + 1]

    for _side, arr, mask, dr, dc in matches:
        paste(out, arr, int(dr), int(dc), mask=mask, overwrite=True, clip=True)

    return out


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def _add_body_noise(mask: np.ndarray, side: str, band_start: int, rng: random.Random) -> None:
    h, w = mask.shape

    if side == "top":
        if h >= 3:
            row = h - 3
            start = max(0, min(w - 1, band_start + rng.randint(-1, 1)))
            length = rng.randint(1, min(3, w - start))
            mask[row, start : start + length] = True
        for rr in range(0, max(0, h - 3)):
            for cc in range(w):
                if rng.random() < 0.3:
                    mask[rr, cc] = True
    elif side == "bot":
        if h >= 3:
            row = 2
            start = max(0, min(w - 1, band_start + rng.randint(-1, 1)))
            length = rng.randint(1, min(3, w - start))
            mask[row, start : start + length] = True
        for rr in range(3, h):
            for cc in range(w):
                if rng.random() < 0.3:
                    mask[rr, cc] = True
    elif side == "left":
        if w >= 3:
            col = w - 3
            start = max(0, min(h - 1, band_start + rng.randint(-1, 1)))
            length = rng.randint(1, min(3, h - start))
            mask[start : start + length, col] = True
        for rr in range(h):
            for cc in range(0, max(0, w - 3)):
                if rng.random() < 0.3:
                    mask[rr, cc] = True
    else:
        if w >= 3:
            col = 2
            start = max(0, min(h - 1, band_start + rng.randint(-1, 1)))
            length = rng.randint(1, min(3, h - start))
            mask[start : start + length, col] = True
        for rr in range(h):
            for cc in range(3, w):
                if rng.random() < 0.3:
                    mask[rr, cc] = True

    if side in ("top", "bot"):
        for cc in range(w):
            col_sum = int(mask[:, cc].sum())
            if col_sum and col_sum == h:
                cut_r = 0 if side == "top" else h - 1
                if not ((side == "top" and cut_r >= h - 2) or (side == "bot" and cut_r <= 1)):
                    mask[cut_r, cc] = False
    else:
        for rr in range(h):
            row_sum = int(mask[rr, :].sum())
            if row_sum and row_sum == w:
                cut_c = 0 if side == "left" else w - 1
                if not ((side == "left" and cut_c >= w - 2) or (side == "right" and cut_c <= 1)):
                    mask[rr, cut_c] = False


def _make_sprite_mask(pattern: tuple[int, ...], side: str, rng: random.Random) -> np.ndarray | None:
    L = len(pattern)
    for _ in range(500):
        h = rng.randint(4, 6)
        w = rng.randint(4, 6)
        if side in ("top", "bot") and w < L:
            w = L
        if side in ("left", "right") and h < L:
            h = L
        mask = np.zeros((h, w), dtype=bool)

        if side in ("top", "bot"):
            band_start = rng.randint(0, w - L)
        else:
            band_start = rng.randint(0, h - L)

        if side == "top":
            mask[h - 2, band_start : band_start + L] = True
            for i, v in enumerate(pattern):
                if v:
                    mask[h - 1, band_start + i] = True
        elif side == "bot":
            mask[1, band_start : band_start + L] = True
            for i, v in enumerate(pattern):
                if v:
                    mask[0, band_start + i] = True
        elif side == "left":
            mask[band_start : band_start + L, w - 2] = True
            for i, v in enumerate(pattern):
                if v:
                    mask[band_start + i, w - 1] = True
        else:
            mask[band_start : band_start + L, 1] = True
            for i, v in enumerate(pattern):
                if v:
                    mask[band_start + i, 0] = True

        _add_body_noise(mask, side, band_start, rng)

        if not bool(mask.any()):
            continue
        mask = _crop_mask(mask)
        if int(mask.sum()) < 6 or int(mask.sum()) > 14:
            continue

        _lab, nc = label_mask(mask, connectivity=8)
        if int(nc) != 1:
            continue

        if _count_valid_matches(mask) != [(0, side)]:
            continue
        return mask
    return None


def _grow_connected_mask(h: int, w: int, area: int, rng: random.Random) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    sr = rng.randrange(h)
    sc = rng.randrange(w)
    mask[sr, sc] = True

    while int(mask.sum()) < int(area):
        frontier: set[tuple[int, int]] = set()
        for rr, cc in np.argwhere(mask):
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and not bool(mask[nr, nc]):
                        frontier.add((nr, nc))
        if not frontier:
            break
        nr, nc = rng.choice(list(frontier))
        mask[int(nr), int(nc)] = True

    return _crop_mask(mask)


def _make_same_color_distractor_mask(min_area: int, rng: random.Random) -> np.ndarray | None:
    lo = max(int(min_area), 11)
    hi = 16
    if lo > hi:
        lo = hi

    for _ in range(400):
        h = rng.randint(4, 6)
        w = rng.randint(4, 6)
        area = rng.randint(lo, min(hi, h * w - 1))
        mask = _grow_connected_mask(h, w, area, rng)
        if mask.shape[0] < 3 or mask.shape[1] < 3:
            continue
        if int(mask.sum()) < int(lo):
            continue
        _lab, nc = label_mask(mask, connectivity=8)
        if int(nc) != 1:
            continue
        if _count_valid_matches(mask):
            continue
        return mask
    return None


def _draw_hub(
    canvas: np.ndarray,
    hub_r: int,
    hub_c: int,
    edge_colors: tuple[int, int, int, int],
    dock_color: int,
    bg: int,
) -> None:
    top_col, bot_col, left_col, right_col = map(int, edge_colors)

    for cc in range(1, HUB_N - 1):
        canvas[hub_r, hub_c + cc] = top_col
        canvas[hub_r + HUB_N - 1, hub_c + cc] = bot_col
    for rr in range(1, HUB_N - 1):
        canvas[hub_r + rr, hub_c] = left_col
        canvas[hub_r + rr, hub_c + HUB_N - 1] = right_col

    span_c0 = int(hub_c + 1)
    span_r0 = int(hub_r + 1)

    canvas[hub_r - 1, span_c0 : span_c0 + EDGE_LEN] = int(dock_color)
    canvas[hub_r + HUB_N, span_c0 : span_c0 + EDGE_LEN] = int(dock_color)
    canvas[span_r0 : span_r0 + EDGE_LEN, hub_c - 1] = int(dock_color)
    canvas[span_r0 : span_r0 + EDGE_LEN, hub_c + HUB_N] = int(dock_color)

    for i, v in enumerate(PATTERNS["top"]):
        if not v:
            canvas[hub_r - 2, span_c0 + i] = int(dock_color)
    for i, v in enumerate(PATTERNS["bot"]):
        if not v:
            canvas[hub_r + HUB_N + 1, span_c0 + i] = int(dock_color)
    for i, v in enumerate(PATTERNS["left"]):
        if not v:
            canvas[span_r0 + i, hub_c - 2] = int(dock_color)
    for i, v in enumerate(PATTERNS["right"]):
        if not v:
            canvas[span_r0 + i, hub_c + HUB_N + 1] = int(dock_color)

    # Keep all remaining bbox cells bg; this preserves visible notches and the hollow core.
    _ = bg


def _attach_sprite(
    out: np.ndarray,
    mask: np.ndarray,
    side: str,
    hub_r: int,
    hub_c: int,
    color: int,
    bg: int,
) -> tuple[np.ndarray, np.ndarray, int, int] | None:
    arr = np.where(mask, int(color), int(bg)).astype(int)
    shift = _attachment_shift(mask, side, hub_r, hub_c)
    if shift is None:
        return None
    dr, dc = shift
    if not _placeable(out, arr, mask, dr, dc, bg):
        return None
    paste(out, arr, int(dr), int(dc), mask=mask, overwrite=True, clip=False)
    return arr, mask, int(dr), int(dc)


def _list_bbox_positions(
    blocked: np.ndarray,
    h: int,
    w: int,
    *,
    padding: int = 0,
    r_min: int = 0,
    r_max: int | None = None,
    c_min: int = 0,
    c_max: int | None = None,
) -> list[tuple[int, int]]:
    H, W = blocked.shape
    if r_max is None:
        r_max = H - h
    if c_max is None:
        c_max = W - w
    out: list[tuple[int, int]] = []
    for r in range(int(r_min), int(r_max) + 1):
        for c in range(int(c_min), int(c_max) + 1):
            rr0 = max(0, int(r - padding))
            cc0 = max(0, int(c - padding))
            rr1 = min(H, int(r + h + padding))
            cc1 = min(W, int(c + w + padding))
            if not bool(blocked[rr0:rr1, cc0:cc1].any()):
                out.append((int(r), int(c)))
    return out


def _reserve_bbox(canvas: np.ndarray, r: int, c: int, h: int, w: int, mark: int) -> None:
    canvas[int(r) : int(r + h), int(c) : int(c + w)] = int(mark)


def _variant_metrics(inp: np.ndarray) -> dict[str, int]:
    g = np.array(inp, dtype=int)
    bg = int(get_bg(g))
    hubs = _hub_candidates(g, bg)
    if not hubs:
        return {}

    chosen, _matches = _choose_hub_and_matches(g, bg, hubs)
    all_hubs_mask = _all_hubs_mask(g.shape, hubs)
    comps = _sprite_components(g, bg, int(chosen["dock_color"]), all_hubs_mask)

    valid = 0
    invalid = 0
    for comp in comps:
        if _count_valid_matches(comp["mask"]):
            valid += 1
        else:
            invalid += 1

    closer_decoy = 0
    for cand in hubs:
        if cand is chosen:
            continue
        if float(cand["center_score"]) < float(chosen["center_score"]):
            closer_decoy = 1
            break

    return {
        "hub_count": int(len(hubs)),
        "distinct_hub_count": int(sum(1 for cand in hubs if cand["all_distinct"])),
        "valid_sprite_components": int(valid),
        "invalid_same_color_components": int(invalid),
        "closer_decoy": int(closer_decoy),
    }


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = BG_COLOR

    H = rng.randint(26, 32)
    W = rng.randint(26, 32)

    palette = [c for c in range(10) if c != bg]
    rng.shuffle(palette)
    if len(palette) < 7:
        return None

    dock_color = int(palette[0])
    true_edge_colors = tuple(int(x) for x in palette[1:5])

    sides = list(SIDES)
    K = rng.randint(2, 4)
    sprite_sides = rng.sample(sides, K)

    sprite_masks: dict[str, np.ndarray] = {}
    for side in sprite_sides:
        mask = _make_sprite_mask(PATTERNS[side], side, rng)
        if mask is None:
            return None
        sprite_masks[side] = mask

    blocked = np.zeros((H, W), dtype=bool)
    margin = 2
    all_positions = _list_bbox_positions(
        blocked,
        FULL_HUB_N,
        FULL_HUB_N,
        padding=1,
        r_min=margin,
        r_max=H - FULL_HUB_N - margin,
        c_min=margin,
        c_max=W - FULL_HUB_N - margin,
    )
    if len(all_positions) < 2:
        return None

    # Put the true hub away from center so a closer decoy can invalidate the old heuristic.
    sorted_far = sorted(
        all_positions,
        key=lambda rc: _center_score_pos(rc[0] + 2, rc[1] + 2, HUB_N, HUB_N, H, W),
        reverse=True,
    )
    true_box_r, true_box_c = rng.choice(sorted_far[: max(1, len(sorted_far) // 2)])
    true_hub_r = int(true_box_r + 2)
    true_hub_c = int(true_box_c + 2)
    blocked[true_box_r : true_box_r + FULL_HUB_N, true_box_c : true_box_c + FULL_HUB_N] = True

    decoy_positions = [
        rc
        for rc in _list_bbox_positions(blocked, FULL_HUB_N, FULL_HUB_N, padding=1)
        if _center_score_pos(rc[0] + 2, rc[1] + 2, HUB_N, HUB_N, H, W)
        < _center_score_pos(true_hub_r, true_hub_c, HUB_N, HUB_N, H, W)
    ]
    if not decoy_positions:
        return None
    decoy_positions.sort(key=lambda rc: _center_score_pos(rc[0] + 2, rc[1] + 2, HUB_N, HUB_N, H, W))
    main_decoy_box_r, main_decoy_box_c = decoy_positions[0]
    blocked[
        main_decoy_box_r : main_decoy_box_r + FULL_HUB_N,
        main_decoy_box_c : main_decoy_box_c + FULL_HUB_N,
    ] = True

    extra_decoys: list[tuple[int, int, tuple[int, int, int, int]]] = []
    remaining = [c for c in palette[5:] if c != dock_color]
    if len(remaining) < 2:
        return None

    def make_repeated_colors(pool: list[int]) -> tuple[int, int, int, int]:
        if len(pool) >= 3:
            a, b, c = map(int, pool[:3])
            return (a, b, a, c)
        a, b = map(int, pool[:2])
        return (a, b, a, b)

    main_decoy_colors = make_repeated_colors(remaining)

    if rng.random() < 0.6:
        extra_positions = _list_bbox_positions(blocked, FULL_HUB_N, FULL_HUB_N, padding=1)
        if extra_positions:
            extra_box_r, extra_box_c = rng.choice(extra_positions)
            blocked[
                extra_box_r : extra_box_r + FULL_HUB_N,
                extra_box_c : extra_box_c + FULL_HUB_N,
            ] = True
            pool2 = remaining[::-1]
            extra_decoys.append((int(extra_box_r), int(extra_box_c), make_repeated_colors(pool2)))

    out = np.full((H, W), int(bg), dtype=int)
    _draw_hub(out, true_hub_r, true_hub_c, true_edge_colors, dock_color, bg)

    attached: dict[str, tuple[np.ndarray, np.ndarray, int, int]] = {}
    for side in sprite_sides:
        placed = _attach_sprite(out, sprite_masks[side], side, true_hub_r, true_hub_c, dock_color, bg)
        if placed is None:
            return None
        attached[side] = placed

    inp = np.full((H, W), int(bg), dtype=int)
    _draw_hub(inp, true_hub_r, true_hub_c, true_edge_colors, dock_color, bg)
    _draw_hub(inp, int(main_decoy_box_r + 2), int(main_decoy_box_c + 2), main_decoy_colors, dock_color, bg)
    for box_r, box_c, edge_colors in extra_decoys:
        _draw_hub(inp, int(box_r + 2), int(box_c + 2), edge_colors, dock_color, bg)

    occupancy = np.full((H, W), int(bg), dtype=int)
    occ_mark = 0 if bg != 0 else 1
    _reserve_bbox(occupancy, true_box_r, true_box_c, FULL_HUB_N, FULL_HUB_N, occ_mark)
    _reserve_bbox(occupancy, main_decoy_box_r, main_decoy_box_c, FULL_HUB_N, FULL_HUB_N, occ_mark)
    for box_r, box_c, _edge_colors in extra_decoys:
        _reserve_bbox(occupancy, box_r, box_c, FULL_HUB_N, FULL_HUB_N, occ_mark)

    for side in sprite_sides:
        _arr, mask, _sr, _sc = attached[side]
        k = rng.choice([1, 2, 3])
        rot_mask = _crop_mask(np.rot90(mask, int(k)))
        patch = np.where(rot_mask, int(dock_color), int(bg)).astype(int)
        try:
            pr, pc = random_free_location(
                occupancy,
                patch,
                transparent=bg,
                padding=2,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(occupancy, patch, pr, pc, transparent=bg)
        paste(inp, patch, pr, pc, transparent=bg)

    if rng.random() < 0.75:
        min_area = max(int(mask.sum()) for mask in sprite_masks.values()) + 1
        dist_mask = _make_same_color_distractor_mask(min_area, rng)
        if dist_mask is None:
            return None
        dist_mask = _crop_mask(np.rot90(dist_mask, rng.choice([0, 1, 2, 3])))
        dist_patch = np.where(dist_mask, int(dock_color), int(bg)).astype(int)
        try:
            pr, pc = random_free_location(
                occupancy,
                dist_patch,
                transparent=bg,
                padding=2,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(occupancy, dist_patch, pr, pc, transparent=bg)
        paste(inp, dist_patch, pr, pc, transparent=bg)

    result = main(inp)
    if not np.array_equal(result, out):
        return None

    metrics = _variant_metrics(inp)
    if metrics.get("hub_count", 0) < 2:
        return None
    if metrics.get("distinct_hub_count", 0) != 1:
        return None
    if metrics.get("valid_sprite_components", 0) != K:
        return None
    if metrics.get("closer_decoy", 0) != 1:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 500 attempts")
