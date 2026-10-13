from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(int(r), int(c))])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _analyze_l_shape(coords: list[tuple[int, int]]) -> dict[str, object] | None:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 2 or w < 2:
        return None

    mask = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True

    missing = ~mask
    if not missing.any():
        return None

    miss_coords = np.argwhere(missing)
    mr0, mc0 = miss_coords.min(axis=0)
    mr1, mc1 = miss_coords.max(axis=0)
    mr0 = int(mr0)
    mc0 = int(mc0)
    mr1 = int(mr1)
    mc1 = int(mc1)
    if not np.all(missing[mr0 : mr1 + 1, mc0 : mc1 + 1]):
        return None

    touches_top = mr0 == 0
    touches_bottom = mr1 == h - 1
    touches_left = mc0 == 0
    touches_right = mc1 == w - 1
    if touches_top == touches_bottom or touches_left == touches_right:
        return None

    corner = (
        "tl"
        if touches_top and touches_left
        else "tr"
        if touches_top and touches_right
        else "bl"
        if touches_bottom and touches_left
        else "br"
    )

    return {
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "missing_bbox": (int(r0 + mr0), int(c0 + mc0), int(r0 + mr1), int(c0 + mc1)),
        "missing_shape": (int(mr1 - mr0 + 1), int(mc1 - mc0 + 1)),
        "corner": corner,
    }


def _between_marker(
    pair: list[dict[str, object]], markers: list[dict[str, object]]
) -> dict[str, object]:
    a, b = pair
    ar0, ac0, ar1, ac1 = a["bbox"]
    br0, bc0, br1, bc1 = b["bbox"]
    candidates: dict[tuple[int, int], dict[str, object]] = {}

    row_lo = max(ar0, br0)
    row_hi = min(ar1, br1)
    if row_lo <= row_hi:
        if ac0 <= bc0:
            left_c1, right_c0 = ac1, bc0
        else:
            left_c1, right_c0 = bc1, ac0
        if left_c1 < right_c0 - 0:
            for marker in markers:
                r, c = marker["pos"]
                if row_lo <= r <= row_hi and left_c1 < c < right_c0:
                    candidates[(r, c)] = marker

    col_lo = max(ac0, bc0)
    col_hi = min(ac1, bc1)
    if col_lo <= col_hi:
        if ar0 <= br0:
            top_r1, bottom_r0 = ar1, br0
        else:
            top_r1, bottom_r0 = br1, ar0
        if top_r1 < bottom_r0 - 0:
            for marker in markers:
                r, c = marker["pos"]
                if top_r1 < r < bottom_r0 and col_lo <= c <= col_hi:
                    candidates[(r, c)] = marker

    if not candidates:
        ur0 = min(ar0, br0)
        uc0 = min(ac0, bc0)
        ur1 = max(ar1, br1)
        uc1 = max(ac1, bc1)
        for marker in markers:
            r, c = marker["pos"]
            inside_union = ur0 <= r <= ur1 and uc0 <= c <= uc1
            inside_a = ar0 <= r <= ar1 and ac0 <= c <= ac1
            inside_b = br0 <= r <= br1 and bc0 <= c <= bc1
            if inside_union and not inside_a and not inside_b:
                candidates[(r, c)] = marker

    assert len(candidates) == 1, "each L-shape pair must have a unique marker between the two bboxes"
    return next(iter(candidates.values()))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each paired L-shape into a rectangle using the color of the isolated marker between the pair.

    Input structure:
    - The background is black, and the non-background cells form monochrome L-shaped objects plus isolated one-pixel markers.
    - Each L-shape is a filled rectangle with one corner sub-rectangle removed, and each L-shape color appears in exactly two 4-connected components.
    - The two same-color L-shapes in a pair have the same missing-corner size but different missing corners.
    - Each pair has exactly one uniquely colored singleton marker located in the empty space between the pair's bounding boxes.

    Rule:
    - Group the L-shapes by color to recover the intended pairs.
    - For each pair, find the singleton marker positioned between the two same-color bounding boxes and read its color.
    - Fill each L-shape's missing corner rectangle with that marker color so the shape becomes a complete rectangle.
    - Remove the marker pixel itself by restoring it to the background.

    Key insight:
    - Color equality identifies which L-shapes belong together, and the otherwise unrelated singleton between them supplies the fill color for both missing corners.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    out = g.copy()

    l_by_color: dict[int, list[dict[str, object]]] = defaultdict(list)
    markers: list[dict[str, object]] = []

    for color in (int(v) for v in np.unique(g) if int(v) != bg):
        for comp in _components(g == color):
            if len(comp) == 1:
                markers.append({"color": int(color), "pos": comp[0]})
                continue
            info = _analyze_l_shape(comp)
            assert info is not None, "non-singleton foreground components must be L-shapes"
            info["color"] = int(color)
            l_by_color[int(color)].append(info)

    used_markers: set[tuple[int, int]] = set()
    for color, pair in l_by_color.items():
        assert len(pair) == 2, "each L color must appear exactly twice"
        assert pair[0]["missing_shape"] == pair[1]["missing_shape"], "paired L-shapes must share missing-corner dimensions"
        assert pair[0]["corner"] != pair[1]["corner"], "paired L-shapes must miss different corners"
        marker = _between_marker(pair, markers)
        mr, mc = marker["pos"]
        used_markers.add((mr, mc))
        fill = int(marker["color"])
        for item in pair:
            rr0, cc0, rr1, cc1 = item["missing_bbox"]
            out[rr0 : rr1 + 1, cc0 : cc1 + 1] = fill
        out[mr, mc] = bg

    assert len(used_markers) == len(markers), "all singleton markers should belong to exactly one pair"
    return out


_CORNERS = ("tl", "tr", "bl", "br")


def _l_mask(height: int, width: int, miss_h: int, miss_w: int, corner: str) -> np.ndarray:
    mask = np.ones((height, width), dtype=bool)
    if corner == "tl":
        mask[:miss_h, :miss_w] = False
    elif corner == "tr":
        mask[:miss_h, width - miss_w :] = False
    elif corner == "bl":
        mask[height - miss_h :, :miss_w] = False
    else:
        mask[height - miss_h :, width - miss_w :] = False
    return mask


def _missing_bbox(
    top: int, left: int, height: int, width: int, miss_h: int, miss_w: int, corner: str
) -> tuple[int, int, int, int]:
    if corner == "tl":
        return top, left, top + miss_h - 1, left + miss_w - 1
    if corner == "tr":
        return top, left + width - miss_w, top + miss_h - 1, left + width - 1
    if corner == "bl":
        return top + height - miss_h, left, top + height - 1, left + miss_w - 1
    return top + height - miss_h, left + width - miss_w, top + height - 1, left + width - 1


def _pick_corner_pair() -> tuple[str, str]:
    first = random.choice(_CORNERS)
    if random.random() < 0.5:
        opposites = {"tl": "br", "tr": "bl", "bl": "tr", "br": "tl"}
        second = opposites[first]
    else:
        adj = {
            "tl": random.choice(["tr", "bl"]),
            "tr": random.choice(["tl", "br"]),
            "bl": random.choice(["tl", "br"]),
            "br": random.choice(["tr", "bl"]),
        }
        second = adj[first]
    return first, second


def _reserve_free(used: np.ndarray, block_h: int, block_w: int) -> tuple[int, int] | None:
    h, w = used.shape
    if block_h + 2 > h or block_w + 2 > w:
        return None
    spots = [
        (r, c)
        for r in range(1, h - block_h)
        for c in range(1, w - block_w)
        if not used[r - 1 : r + block_h + 1, c - 1 : c + block_w + 1].any()
    ]
    if not spots:
        return None
    r0, c0 = random.choice(spots)
    used[r0 : r0 + block_h, c0 : c0 + block_w] = True
    return int(r0), int(c0)


def _draw_pair(
    inp: np.ndarray,
    out: np.ndarray,
    used: np.ndarray,
    l_color: int,
    fill_color: int,
) -> bool:
    outer_h = random.randint(3, 6)
    outer_w = random.randint(3, 6)
    miss_h = random.randint(1, outer_h - 1)
    miss_w = random.randint(1, outer_w - 1)
    gap = random.randint(3, 7)
    corner_a, corner_b = _pick_corner_pair()
    horizontal = random.random() < 0.5

    if horizontal:
        block_h = outer_h
        block_w = outer_w * 2 + gap
        origin = _reserve_free(used, block_h, block_w)
        if origin is None:
            return False
        top, left = origin
        a_top, a_left = top, left
        b_top, b_left = top, left + outer_w + gap
        marker_r = random.randint(top, top + outer_h - 1)
        marker_c = random.randint(left + outer_w, left + outer_w + gap - 1)
    else:
        block_h = outer_h * 2 + gap
        block_w = outer_w
        origin = _reserve_free(used, block_h, block_w)
        if origin is None:
            return False
        top, left = origin
        a_top, a_left = top, left
        b_top, b_left = top + outer_h + gap, left
        marker_r = random.randint(top + outer_h, top + outer_h + gap - 1)
        marker_c = random.randint(left, left + outer_w - 1)

    for obj_top, obj_left, corner in ((a_top, a_left, corner_a), (b_top, b_left, corner_b)):
        mask = _l_mask(outer_h, outer_w, miss_h, miss_w, corner)
        region_in = inp[obj_top : obj_top + outer_h, obj_left : obj_left + outer_w]
        region_out = out[obj_top : obj_top + outer_h, obj_left : obj_left + outer_w]
        region_in[mask] = l_color
        region_out[mask] = l_color
        mr0, mc0, mr1, mc1 = _missing_bbox(obj_top, obj_left, outer_h, outer_w, miss_h, miss_w, corner)
        region = out[mr0 : mr1 + 1, mc0 : mc1 + 1]
        region[:] = fill_color

    inp[marker_r, marker_c] = fill_color
    return True


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    height = random.randint(16, 30)
    width = random.randint(16, 30)
    inp = np.zeros((height, width), dtype=int)
    out = np.zeros((height, width), dtype=int)
    used = np.zeros((height, width), dtype=bool)

    max_pairs = min(3, 9 // 2)
    pair_count = random.choice([1, 2, 3][:max_pairs])
    palette = list(range(1, 10))
    random.shuffle(palette)
    l_colors = palette[:pair_count]
    fill_colors = palette[pair_count : 2 * pair_count]

    for l_color, fill_color in zip(l_colors, fill_colors, strict=False):
        if not _draw_pair(inp, out, used, int(l_color), int(fill_color)):
            return None

    if _get_bg(inp) != 0:
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(500):
        candidate = _gen_attempt()
        if candidate is None:
            continue
        inp, expected = candidate
        try:
            if np.array_equal(main(inp), expected):
                return inp
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid L-pair puzzle")
