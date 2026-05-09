# Combo variant of 2c181942 (index 38)
# combo_spec: D1/0+D3/0+D7/2
# Name: 形状匹配+八变换+非对称
#
# Design analysis for the fused variant
# ------------------------------------
# a) Concrete fused behavior and trace
#    - Original task:
#      1. detect the hollow square hub
#      2. use edge color to decide which sprite belongs to which side
#      3. try 4 rotations, keep the one whose contact edge length fits and whose
#         whole silhouette is mirror-symmetric across the contact axis
#    - Fused variant:
#      1. all foreground objects use one shared non-bg color, so color pairing is gone
#      2. the central hub becomes a 6x6 hollow core plus a 2-cell-thick outer dock
#         on each side; the dock's outer rim encodes one of four visible notch
#         patterns of length 4
#      3. every scattered component outside the hub is treated as a sprite candidate
#      4. for each hub side, the solver tries every candidate under all 8 D4
#         transforms (4 rotations + 4 reflected rotations)
#      5. a transform is valid only if the sprite exposes exactly one support band
#         plus one protrusion band whose 4-bit silhouette matches that side's notch
#         pattern
#      6. the solver accepts only a unique component/transform match, then pastes
#         the sprite onto that side
#    - Tiny example:
#      if the top dock rim reads 1000, then the correct top sprite orientation must
#      have a full 4-cell support row just inside the hub and a bottom protrusion row
#      equal to 1000. In the input that sprite may be mirrored and rotated, so the
#      solver must search all 8 transforms to recover the unique fit.
# b) Complexity calibration
#    - Keeping the original 4x4 hub would leave only length-2 contacts, which is too
#      weak for visible shape coding. The hub is therefore enlarged to a fixed 6x6
#      core so every side gets a length-4 notch code.
#    - The four side codes are drawn from a small fixed set:
#        top=1000, bot=1100, left=1010, right=1110
#      They are all distinct and none is the reverse of another, so mirror/rotation
#      cannot silently turn one side into another.
#    - Sprite bodies are still small and monochrome, but they are forced to be fully
#      asymmetric across all 7 non-identity D4 transforms. This keeps the novelty
#      visible without exploding the search space.
# c) Implicit assumption audit against the original main()
#    - `m = (g == col) & (~hub_mask)` is invalid now because color no longer selects
#      a side. The new solver labels every non-hub component and tests them all.
#    - The old symmetry checks (`flipud` / `fliplr`) are invalid because sprites are
#      intentionally non-symmetric. Rotation disambiguation is now done by counting
#      valid D4-transform matches and requiring uniqueness.
#    - The old hub finder assumed four side colors may differ. Here the hub still has
#      pure side segments, but all hub edges share the same color; side identity comes
#      from dock geometry instead.
#    - Distractors can no longer be filtered by fresh colors, so the generator must
#      build same-color distractors with zero valid side matches under all 8
#      transforms.
# d) Placement legality
#    - Generation is still output-first: build hub+docks+attached sprites, then scatter.
#    - Hub placement reserves margin for the 2-cell dock plus the largest sprite.
#    - Attached sprites may overlap the hub only on same-color dock cells; any other
#      overlap or out-of-bounds placement rejects the sample.
#    - Scattered placement uses `random_free_location(..., padding=2, connectivity=8)`
#      so sprites stay visually separated from the hub and from each other.
# e) Visual observability
#    - The rule is visible in 2-4 examples because every sample shows one same-color
#      hub with four different notch shapes and several same-color sprites with matching
#      protrusion mouths.
#    - At least one scattered true sprite is always placed via a reflective transform,
#      so the eight-transform requirement is not latent.
# f) Visual cleanliness
#    - Only one marking system is used near the hub: dock bands and their outer notch
#      rim. There are no stacked annotations or hidden numeric cues.
#    - Objects stay separated by padding and the notch code is limited to one 4-cell
#      strip per side, keeping the grid readable.
# g) New solver assumptions and generator guarantees
#    - Assumption: the hub is the only component containing a 6x6 hollow square with
#      2-cell docks. Guarantee: the generator creates exactly one such central hub.
#    - Assumption: every true sprite has exactly one `(transform, side)` match across
#      all 8 transforms and 4 sides. Guarantee: `_make_sprite_mask()` enforces that.
#    - Assumption: same-color distractors have no valid matches. Guarantee:
#      `_make_distractor_mask()` rejects any mask with a legal match.
#    - Assumption: at most one unused component matches a given side. Guarantee:
#      the generator validates the final puzzle with the solver before accepting it.

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, get_bg, label_mask, paste, random_free_location


HUB_N = 6
EDGE_LEN = HUB_N - 2
DOCK_THICKNESS = 2
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
    return (
        arr[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1],
        mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1],
    )


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    dummy = np.zeros(mask.shape, dtype=int)
    _, cropped = _crop_to_mask(dummy, mask)
    return cropped


def _apply_transform(arr: np.ndarray, transform_id: int) -> np.ndarray:
    tid = int(transform_id)
    if tid < 4:
        return np.rot90(arr, tid)
    return np.rot90(np.fliplr(arr), tid - 4)


def _iter_d4_transforms(
    arr: np.ndarray,
    mask: np.ndarray,
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    out: list[tuple[int, np.ndarray, np.ndarray]] = []
    for tid in range(8):
        ta = _apply_transform(arr, tid)
        tm = _apply_transform(mask, tid)
        ta, tm = _crop_to_mask(ta, tm)
        out.append((int(tid), ta, tm))
    return out


def _is_fully_asymmetric(mask: np.ndarray) -> bool:
    base = _crop_mask(mask)
    for tid in range(1, 8):
        other = _crop_mask(_apply_transform(base, tid))
        if np.array_equal(base, other):
            return False
    return True


def _find_core_hub(g: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    H, W = g.shape
    best = None
    n = HUB_N
    for r in range(DOCK_THICKNESS, H - n - DOCK_THICKNESS):
        for c in range(DOCK_THICKNESS, W - n - DOCK_THICKNESS):
            if (
                int(g[r, c]) != bg
                or int(g[r, c + n - 1]) != bg
                or int(g[r + n - 1, c]) != bg
                or int(g[r + n - 1, c + n - 1]) != bg
            ):
                continue

            inner = g[r + 1 : r + n - 1, c + 1 : c + n - 1]
            if not bool((inner == bg).all()):
                continue

            top = g[r, c + 1 : c + n - 1]
            bot = g[r + n - 1, c + 1 : c + n - 1]
            left = g[r + 1 : r + n - 1, c]
            right = g[r + 1 : r + n - 1, c + n - 1]
            if (
                bool((top == bg).any())
                or bool((bot == bg).any())
                or bool((left == bg).any())
                or bool((right == bg).any())
            ):
                continue

            colors = set(map(int, np.concatenate([top, bot, left, right])))
            if len(colors) != 1:
                continue
            color = int(next(iter(colors)))
            if color == bg:
                continue

            cr = r + (n - 1) / 2
            cc = c + (n - 1) / 2
            score = abs(cr - (H - 1) / 2) + abs(cc - (W - 1) / 2)
            cand = (float(score), int(r), int(c), int(color))
            if best is None or cand[0] < best[0]:
                best = cand

    assert best is not None
    _, r, c, color = best
    return int(r), int(c), HUB_N, int(color)


def _side_span(r: int, c: int, side: str) -> tuple[int, int]:
    if side in ("top", "bot"):
        return int(c + 1), int(c + HUB_N - 2)
    return int(r + 1), int(r + HUB_N - 2)


def _dock_outer_coord(r: int, c: int, side: str) -> int:
    if side == "top":
        return int(r - 2)
    if side == "bot":
        return int(r + HUB_N + 1)
    if side == "left":
        return int(c - 2)
    return int(c + HUB_N + 1)


def _read_hub_patterns(g: np.ndarray, r: int, c: int, bg: int) -> dict[str, tuple[int, ...]]:
    patterns: dict[str, tuple[int, ...]] = {}
    for side in SIDES:
        start, end = _side_span(r, c, side)
        outer = _dock_outer_coord(r, c, side)
        vals: list[int] = []
        if side in ("top", "bot"):
            for cc in range(start, end + 1):
                vals.append(1 if int(g[int(outer), int(cc)]) == int(bg) else 0)
        else:
            for rr in range(start, end + 1):
                vals.append(1 if int(g[int(rr), int(outer)]) == int(bg) else 0)
        patterns[side] = tuple(vals)
    return patterns


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


def _attachment_shift(
    mask: np.ndarray,
    side: str,
    pattern: tuple[int, ...],
    hub_r: int,
    hub_c: int,
) -> tuple[int, int] | None:
    pos = _contact_band(mask, side, pattern)
    if pos is None:
        return None
    span_start, _ = _side_span(hub_r, hub_c, side)
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
        val = int(arr[int(rr), int(cc)])
        if val == int(bg):
            continue
        if int(out[int(R), int(C)]) != int(bg) and int(out[int(R), int(C)]) != val:
            return False
    return True


def _match_component_to_side(
    comp_arr: np.ndarray,
    comp_mask: np.ndarray,
    side: str,
    pattern: tuple[int, ...],
    hub_r: int,
    hub_c: int,
    out: np.ndarray,
    bg: int,
) -> tuple[int, np.ndarray, np.ndarray, int, int] | None:
    hits: list[tuple[int, np.ndarray, np.ndarray, int, int]] = []
    for tid, arr, mask in _iter_d4_transforms(comp_arr, comp_mask):
        shift = _attachment_shift(mask, side, pattern, hub_r, hub_c)
        if shift is None:
            continue
        dr, dc = shift
        if not _placeable(out, arr, mask, dr, dc, bg):
            continue
        hits.append((int(tid), arr, mask, int(dr), int(dc)))
    if len(hits) != 1:
        return None
    return hits[0]


def _count_valid_matches(mask: np.ndarray) -> list[tuple[int, str]]:
    dummy = np.where(mask, 1, 0).astype(int)
    hits: list[tuple[int, str]] = []
    for tid, arr, rot_mask in _iter_d4_transforms(dummy, mask):
        del arr
        for side, pattern in PATTERNS.items():
            if _contact_band(rot_mask, side, pattern) is not None:
                hits.append((int(tid), side))
    return hits


def _draw_hub(canvas: np.ndarray, hub_r: int, hub_c: int, color: int, bg: int) -> None:
    del bg
    n = HUB_N
    for cc in range(1, n - 1):
        canvas[int(hub_r), int(hub_c + cc)] = int(color)
        canvas[int(hub_r + n - 1), int(hub_c + cc)] = int(color)
    for rr in range(1, n - 1):
        canvas[int(hub_r + rr), int(hub_c)] = int(color)
        canvas[int(hub_r + rr), int(hub_c + n - 1)] = int(color)

    span_c0 = int(hub_c + 1)
    span_r0 = int(hub_r + 1)

    canvas[int(hub_r - 1), span_c0 : span_c0 + EDGE_LEN] = int(color)
    canvas[int(hub_r + HUB_N), span_c0 : span_c0 + EDGE_LEN] = int(color)
    canvas[span_r0 : span_r0 + EDGE_LEN, int(hub_c - 1)] = int(color)
    canvas[span_r0 : span_r0 + EDGE_LEN, int(hub_c + HUB_N)] = int(color)

    for i, v in enumerate(PATTERNS["top"]):
        if not v:
            canvas[int(hub_r - 2), int(span_c0 + i)] = int(color)
    for i, v in enumerate(PATTERNS["bot"]):
        if not v:
            canvas[int(hub_r + HUB_N + 1), int(span_c0 + i)] = int(color)
    for i, v in enumerate(PATTERNS["left"]):
        if not v:
            canvas[int(span_r0 + i), int(hub_c - 2)] = int(color)
    for i, v in enumerate(PATTERNS["right"]):
        if not v:
            canvas[int(span_r0 + i), int(hub_c + HUB_N + 1)] = int(color)


def _add_body_noise(mask: np.ndarray, side: str, band_start: int, rng: random.Random) -> None:
    h, w = mask.shape
    L = EDGE_LEN

    if side == "top":
        if h >= 3:
            col = max(0, min(w - 1, band_start + rng.randint(0, L - 1) + rng.choice([-1, 0, 1])))
            width = rng.randint(1, min(2, w))
            for rr in range(h - 3, -1, -1):
                c0 = max(0, min(w - width, col + rng.choice([-1, 0, 0, 1])))
                mask[int(rr), int(c0) : int(c0 + width)] = True
                if rng.random() < 0.45:
                    break
        for rr in range(0, max(0, h - 3)):
            for cc in range(w):
                if rng.random() < 0.22:
                    mask[int(rr), int(cc)] = True
    elif side == "bot":
        if h >= 3:
            col = max(0, min(w - 1, band_start + rng.randint(0, L - 1) + rng.choice([-1, 0, 1])))
            width = rng.randint(1, min(2, w))
            for rr in range(2, h):
                c0 = max(0, min(w - width, col + rng.choice([-1, 0, 0, 1])))
                mask[int(rr), int(c0) : int(c0 + width)] = True
                if rng.random() < 0.45:
                    break
        for rr in range(3, h):
            for cc in range(w):
                if rng.random() < 0.22:
                    mask[int(rr), int(cc)] = True
    elif side == "left":
        if w >= 3:
            row = max(0, min(h - 1, band_start + rng.randint(0, L - 1) + rng.choice([-1, 0, 1])))
            height = rng.randint(1, min(2, h))
            for cc in range(w - 3, -1, -1):
                r0 = max(0, min(h - height, row + rng.choice([-1, 0, 0, 1])))
                mask[int(r0) : int(r0 + height), int(cc)] = True
                if rng.random() < 0.45:
                    break
        for rr in range(h):
            for cc in range(0, max(0, w - 3)):
                if rng.random() < 0.22:
                    mask[int(rr), int(cc)] = True
    else:
        if w >= 3:
            row = max(0, min(h - 1, band_start + rng.randint(0, L - 1) + rng.choice([-1, 0, 1])))
            height = rng.randint(1, min(2, h))
            for cc in range(2, w):
                r0 = max(0, min(h - height, row + rng.choice([-1, 0, 0, 1])))
                mask[int(r0) : int(r0 + height), int(cc)] = True
                if rng.random() < 0.45:
                    break
        for rr in range(h):
            for cc in range(3, w):
                if rng.random() < 0.22:
                    mask[int(rr), int(cc)] = True


def _make_sprite_mask(pattern: tuple[int, ...], side: str, rng: random.Random) -> np.ndarray | None:
    L = len(pattern)
    for _ in range(1200):
        h = rng.randint(4, 7)
        w = rng.randint(4, 7)
        if side in ("top", "bot") and w < L:
            w = L
        if side in ("left", "right") and h < L:
            h = L

        mask = np.zeros((h, w), dtype=bool)
        band_start = rng.randint(0, (w - L) if side in ("top", "bot") else (h - L))

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

        area = int(mask.sum())
        if area < 6 or area > 16:
            continue
        lab, nc = label_mask(mask, connectivity=8)
        if int(nc) != 1:
            continue
        if not _is_fully_asymmetric(mask):
            continue
        if _count_valid_matches(mask) != [(0, side)]:
            continue
        return mask
    return None


def _make_distractor_mask(rng: random.Random) -> np.ndarray | None:
    for _ in range(1200):
        h = rng.randint(3, 6)
        w = rng.randint(3, 6)
        mask = np.zeros((h, w), dtype=bool)
        k = rng.randint(4, min(10, h * w))
        cells = rng.sample([(r, c) for r in range(h) for c in range(w)], k)
        for rr, cc in cells:
            mask[int(rr), int(cc)] = True
        mask = _crop_mask(mask)
        if int(mask.sum()) < 4:
            continue
        lab, nc = label_mask(mask, connectivity=8)
        if int(nc) != 1:
            continue
        if _count_valid_matches(mask):
            continue
        return mask
    return None


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
    shift = _attachment_shift(mask, side, PATTERNS[side], hub_r, hub_c)
    if shift is None:
        return None
    dr, dc = shift
    if not _placeable(out, arr, mask, dr, dc, bg):
        return None
    paste(out, arr, dr, dc, mask=mask, overwrite=True, clip=False)
    return arr, mask, int(dr), int(dc)


def _component_records(g: np.ndarray, bg: int) -> tuple[int, list[tuple[int, np.ndarray, np.ndarray]]]:
    hub_r, hub_c, _hub_n, _hub_color = _find_core_hub(g, bg)
    lab, nc = label_mask(g != bg, connectivity=8)
    hub_id = int(lab[int(hub_r), int(hub_c + 1)])
    comps: list[tuple[int, np.ndarray, np.ndarray]] = []
    for cid in range(1, int(nc) + 1):
        if int(cid) == int(hub_id):
            continue
        mask = lab == int(cid)
        arr = np.where(mask, g, bg)
        arr, mask = _crop_to_mask(arr, mask)
        comps.append((int(cid), arr, mask))
    return int(hub_id), comps


def _fusion_metrics(inp: np.ndarray) -> dict[str, int]:
    g = np.array(inp, dtype=int)
    bg = int(get_bg(g))
    non_bg_colors = len(set(map(int, g[g != bg].flatten()))) if bool((g != bg).any()) else 0
    hub_r, hub_c, _hub_n, _hub_color = _find_core_hub(g, bg)
    hub_patterns = _read_hub_patterns(g, hub_r, hub_c, bg)
    _, comps = _component_records(g, bg)

    unique_match_components = 0
    reflective_components = 0
    asymmetric_unique_components = 0
    for _cid, _arr, mask in comps:
        hits = _count_valid_matches(mask)
        if len(hits) == 1:
            unique_match_components += 1
            if int(hits[0][0]) >= 4:
                reflective_components += 1
            if _is_fully_asymmetric(mask):
                asymmetric_unique_components += 1

    return {
        "non_bg_colors": int(non_bg_colors),
        "unique_hub_patterns": int(len(set(hub_patterns.values()))),
        "unique_match_components": int(unique_match_components),
        "reflective_components": int(reflective_components),
        "asymmetric_unique_components": int(asymmetric_unique_components),
        "component_count": int(len(comps)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    hub_r, hub_c, hub_n, _hub_color = _find_core_hub(g, bg)
    assert int(hub_n) == int(HUB_N)

    lab, nc = label_mask(g != bg, connectivity=8)
    hub_id = int(lab[int(hub_r), int(hub_c + 1)])
    assert int(hub_id) > 0
    hub_mask = lab == int(hub_id)

    out = np.full((H, W), bg, dtype=int)
    out[hub_mask] = g[hub_mask]

    hub_patterns = _read_hub_patterns(g, hub_r, hub_c, bg)

    components: list[tuple[int, np.ndarray, np.ndarray, int]] = []
    for cid in range(1, int(nc) + 1):
        if int(cid) == int(hub_id):
            continue
        mask = lab == int(cid)
        arr = np.where(mask, g, bg)
        arr, mask = _crop_to_mask(arr, mask)
        components.append((int(cid), arr, mask, int(mask.sum())))
    components.sort(key=lambda item: -int(item[3]))

    used: set[int] = set()
    for side in SIDES:
        pattern = hub_patterns[side]
        candidates: list[tuple[int, int, np.ndarray, np.ndarray, int, int]] = []
        for cid, arr, mask, _area in components:
            if int(cid) in used:
                continue
            match = _match_component_to_side(arr, mask, side, pattern, hub_r, hub_c, out, bg)
            if match is None:
                continue
            tid, match_arr, match_mask, dr, dc = match
            candidates.append((int(cid), int(tid), match_arr, match_mask, int(dr), int(dc)))

        if len(candidates) != 1:
            continue

        cid, _tid, match_arr, match_mask, dr, dc = candidates[0]
        paste(out, match_arr, dr, dc, mask=match_mask, overwrite=True, clip=True)
        used.add(int(cid))

    return out


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = BG_COLOR
    color = int(rng.choice([c for c in range(10) if c != bg]))

    H = rng.randint(20, 28)
    W = rng.randint(22, 28)
    K = rng.randint(2, 4)
    sprite_sides = rng.sample(list(SIDES), K)

    sprite_masks: dict[str, np.ndarray] = {}
    for side in sprite_sides:
        mask = _make_sprite_mask(PATTERNS[side], side, rng)
        if mask is None:
            return None
        sprite_masks[side] = mask

    max_h = max(mask.shape[0] for mask in sprite_masks.values())
    max_w = max(mask.shape[1] for mask in sprite_masks.values())
    margin = DOCK_THICKNESS + max(max_h, max_w) + 2

    hub_r_min = margin
    hub_r_max = H - HUB_N - margin
    hub_c_min = margin
    hub_c_max = W - HUB_N - margin
    if hub_r_min > hub_r_max or hub_c_min > hub_c_max:
        return None

    hub_r = rng.randint(hub_r_min, hub_r_max)
    hub_c = rng.randint(hub_c_min, hub_c_max)

    out = np.full((H, W), bg, dtype=int)
    _draw_hub(out, hub_r, hub_c, color, bg)

    attached: dict[str, tuple[np.ndarray, np.ndarray, int, int]] = {}
    for side in sprite_sides:
        placed = _attach_sprite(out, sprite_masks[side], side, hub_r, hub_c, color, bg)
        if placed is None:
            return None
        attached[side] = placed

    inp = np.full((H, W), bg, dtype=int)
    _draw_hub(inp, hub_r, hub_c, color, bg)
    canvas = inp.copy()

    reflective_side = rng.choice(sprite_sides)
    for side in sprite_sides:
        arr, mask, _sr, _sc = attached[side]
        if side == reflective_side:
            tid = rng.choice([4, 5, 6, 7])
        else:
            tid = rng.choice([1, 2, 3, 4, 5, 6, 7])
        transformed_arr = _apply_transform(arr, tid)
        transformed_mask = _apply_transform(mask, tid)
        transformed_arr, transformed_mask = _crop_to_mask(transformed_arr, transformed_mask)
        patch = np.where(transformed_mask, transformed_arr, bg).astype(int)

        try:
            pr, pc = random_free_location(
                canvas,
                patch,
                transparent=bg,
                padding=2,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(canvas, patch, pr, pc, transparent=bg)
        paste(inp, patch, pr, pc, transparent=bg)

    if rng.random() < 0.35:
        dist_mask = _make_distractor_mask(rng)
        if dist_mask is not None:
            dist_arr = np.where(dist_mask, color, bg).astype(int)
            try:
                pr, pc = random_free_location(
                    canvas,
                    dist_arr,
                    transparent=bg,
                    padding=2,
                    padding_connectivity=8,
                )
                paste(canvas, dist_arr, pr, pc, transparent=bg)
                paste(inp, dist_arr, pr, pc, transparent=bg)
            except ValueError:
                pass

    result = main(inp)
    if not np.array_equal(result, out):
        return None

    metrics = _fusion_metrics(inp)
    if metrics["non_bg_colors"] != 1:
        return None
    if metrics["unique_hub_patterns"] != 4:
        return None
    if metrics["unique_match_components"] != K:
        return None
    if metrics["reflective_components"] < 1:
        return None
    if metrics["asymmetric_unique_components"] != K:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(600):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 600 attempts")
