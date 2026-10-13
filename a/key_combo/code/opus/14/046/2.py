from __future__ import annotations

import hashlib
import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
YELLOW = 4
ORANGE = 7

INPUT_BODY_COLORS = (3, 5, 6, 9)
GRID_SIDE_RANGE = (14, 18)
SPRITE_SIDE_RANGE = (4, 5)
OVERLAP_SIZE_OPTIONS = (
    (2, 3),
    (3, 2),
    (2, 4),
    (4, 2),
    (3, 3),
    (3, 4),
    (4, 3),
)
GEN_ATTEMPTS = 600
MASK_ATTEMPTS = 250
SHIFT_ATTEMPTS = 200

_SEEN_OUTPUT_SIGS: set[str] = set()


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _mask_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for rr, cc in _neighbors4(r, c):
            if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1] and mask[rr, cc] and (rr, cc) not in seen:
                seen.add((rr, cc))
                q.append((rr, cc))
    return len(seen) == int(mask.sum())


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        q = deque([(r, c)])
        seen[r, c] = True
        coords: list[tuple[int, int]] = []
        while q:
            rr, cc = q.popleft()
            coords.append((rr, cc))
            for nr, nc in _neighbors4(rr, cc):
                if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(np.array(coords, dtype=int))
    return comps


def _find_sprite_components(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    mask = g != BG
    anchors = _find_anchors(g)
    for r, c in anchors:
        mask[r, c] = False
    sprites: list[dict] = []
    for coords in _component_coords(mask):
        pts = [tuple(map(int, rc)) for rc in coords]
        vals = [int(g[r, c]) for r, c in pts]
        marker_pts = [(r, c) for (r, c), v in zip(pts, vals) if v == ORANGE]
        if len(marker_pts) != 1:
            continue
        bbox = _bbox_from_coords(pts)
        sprites.append(
            {
                "coords": pts,
                "bbox": bbox,
                "marker": marker_pts[0],
                "values": {pt: int(g[pt]) for pt in pts},
            }
        )
    assert len(sprites) == 2, "expected exactly two marked sprites"
    return sprites


def _find_anchors(grid: np.ndarray) -> list[tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    anchors: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if int(g[r, c]) != ORANGE:
                continue
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            ok = True
            for rr, cc in _neighbors4(r, c):
                if 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) != BG:
                    ok = False
                    break
            if ok:
                anchors.append((r, c))
    top = [(r, c) for r, c in anchors if r == 0 and 0 < c < w - 1]
    right = [(r, c) for r, c in anchors if c == w - 1 and 0 < r < h - 1]
    assert len(top) == 1 and len(right) == 1 and len(anchors) == 2
    return [top[0], right[0]]


def _translated_values(sprite: dict, dr: int, dc: int) -> tuple[dict[tuple[int, int], int], tuple[int, int, int, int]]:
    moved = {(r + dr, c + dc): color for (r, c), color in sprite["values"].items()}
    r0, c0, r1, c1 = sprite["bbox"]
    return moved, (r0 + dr, c0 + dc, r1 + dr, c1 + dc)


def _compare_sprites(sprite_a: dict, sprite_b: dict, top_anchor_col: int, right_anchor_row: int,
                     move_a: bool = True, move_b: bool = True) -> np.ndarray:
    a_mr, a_mc = sprite_a["marker"]
    b_mr, b_mc = sprite_b["marker"]
    a_dr, a_dc = 0, (top_anchor_col - a_mc) if move_a else 0
    b_dr, b_dc = (right_anchor_row - b_mr) if move_b else 0, 0

    a_vals, a_bbox = _translated_values(sprite_a, a_dr, a_dc)
    b_vals, b_bbox = _translated_values(sprite_b, b_dr, b_dc)

    r0 = max(a_bbox[0], b_bbox[0])
    c0 = max(a_bbox[1], b_bbox[1])
    r1 = min(a_bbox[2], b_bbox[2])
    c1 = min(a_bbox[3], b_bbox[3])
    assert r0 <= r1 and c0 <= c1, "translated sprite bboxes must overlap"

    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            av = int(a_vals.get((r, c), BG))
            bv = int(b_vals.get((r, c), BG))
            if av != BG and bv != BG:
                out[r - r0, c - c0] = av if av == bv else YELLOW
            elif av != BG:
                out[r - r0, c - c0] = RED
            elif bv != BG:
                out[r - r0, c - c0] = BLUE
            else:
                out[r - r0, c - c0] = BG
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide the two marked sprites onto anchor-determined axes, then crop their overlap and color the pixelwise comparison.

    Input structure:
    - The grid has black background, two interior 4-connected multicolor sprites, and exactly two isolated orange anchor pixels.
    - One anchor lies on the top edge and one lies on the right edge; each sprite contains exactly one orange marker pixel.
    - The sprites are initially separate, irregular, and use 2-3 visible colors including the marker.

    Rule:
    - Identify sprite A as the one whose orange marker has the smaller Manhattan distance to the top-edge anchor; sprite B is the other sprite.
    - Slide A horizontally until its marker column matches the top anchor column, and slide B vertically until its marker row matches the right anchor row.
    - Take the rectangular overlap of the two translated sprite bounding boxes and compare the translated sprites cell by cell inside that rectangle.
    - Copy matching shared colors, use yellow for mismatched shared colors, red for A-only cells, blue for B-only cells, black for double-background cells, and output only the cropped overlap rectangle.

    Key insight:
    - The anchors do not mark target cells directly; their border side determines which axis each sprite may move along, and the final answer depends on the overlap created only after both slides.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    top_anchor, right_anchor = _find_anchors(g)
    sprites = _find_sprite_components(g)
    dists = [
        abs(sprite["marker"][0] - top_anchor[0]) + abs(sprite["marker"][1] - top_anchor[1])
        for sprite in sprites
    ]
    assert dists[0] != dists[1], "top-anchor distance must identify a unique sprite A"
    a_idx = 0 if dists[0] < dists[1] else 1
    b_idx = 1 - a_idx
    return _compare_sprites(sprites[a_idx], sprites[b_idx], top_anchor[1], right_anchor[0], True, True)


def _random_mask(h: int, w: int) -> np.ndarray | None:
    min_area = max(8, (h * w + 1) // 2)
    max_area = h * w - 1
    for _ in range(MASK_ATTEMPTS):
        p = random.uniform(0.48, 0.78)
        mask = np.random.random((h, w)) < p
        if not (min_area <= int(mask.sum()) <= max_area):
            continue
        if not _mask_connected(mask):
            continue
        coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
        if _bbox_from_coords(coords) != (0, 0, h - 1, w - 1):
            continue
        if np.all(mask):
            continue
        row_counts = {int(mask[r].sum()) for r in range(h)}
        col_counts = {int(mask[:, c].sum()) for c in range(w)}
        if len(row_counts) == 1 and len(col_counts) == 1:
            continue
        return mask
    return None


def _occupied_global(local_patch: np.ndarray, top: int, left: int) -> set[tuple[int, int]]:
    pts = set()
    for r, c in np.argwhere(local_patch != BG):
        pts.add((top + int(r), left + int(c)))
    return pts


def _touching_or_overlapping(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> bool:
    for r, c in a:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if (rr, cc) in b:
                    return True
    return False


def _choose_marker(mask: np.ndarray, overlap_mask: np.ndarray) -> tuple[int, int] | None:
    choices = [tuple(map(int, rc)) for rc in np.argwhere(mask & ~overlap_mask)]
    if not choices:
        return None
    return random.choice(choices)


def _paint_sprite(mask: np.ndarray, overlap_mask: np.ndarray, marker: tuple[int, int], body_palette: tuple[int, ...]) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    body_choices = body_palette[: random.choice((1, 2))]
    for r, c in np.argwhere(mask):
        patch[int(r), int(c)] = random.choice(body_choices)
    patch[marker] = ORANGE
    colors = set(map(int, patch[mask]))
    assert 2 <= len(colors) <= 3
    return patch


def _render_overlap(a_patch: np.ndarray, b_patch: np.ndarray, a_top: int, a_left: int, b_top: int, b_left: int) -> np.ndarray:
    sprite_a = _sprite_from_patch(a_patch, a_top, a_left)
    sprite_b = _sprite_from_patch(b_patch, b_top, b_left)
    top_col = sprite_a["marker"][1]
    right_row = sprite_b["marker"][0]
    return _compare_sprites(sprite_a, sprite_b, top_col, right_row, True, True)


def _sprite_from_patch(patch: np.ndarray, top: int, left: int) -> dict:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(patch != BG)]
    global_coords = [(top + r, left + c) for r, c in coords]
    marker_candidates = [(top + r, left + c) for r, c in coords if int(patch[r, c]) == ORANGE]
    assert len(marker_candidates) == 1
    return {
        "coords": global_coords,
        "bbox": (top, left, top + patch.shape[0] - 1, left + patch.shape[1] - 1),
        "marker": marker_candidates[0],
        "values": {(top + r, left + c): int(patch[r, c]) for r, c in coords},
    }


def _make_patches(h_a: int, w_a: int, h_b: int, w_b: int, oh: int, ow: int) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(MASK_ATTEMPTS):
        a_mask = _random_mask(h_a, w_a)
        b_mask = _random_mask(h_b, w_b)
        if a_mask is None or b_mask is None:
            continue
        if a_mask.shape == b_mask.shape and np.array_equal(a_mask, b_mask):
            continue

        overlap_top_a = h_a - oh if random.random() < 0.5 else 0
        overlap_left_a = w_a - ow if random.random() < 0.5 else 0
        overlap_top_b = 0 if overlap_top_a == h_a - oh else h_b - oh
        overlap_left_b = 0 if overlap_left_a == w_a - ow else w_b - ow

        a_overlap = np.zeros_like(a_mask, dtype=bool)
        b_overlap = np.zeros_like(b_mask, dtype=bool)
        a_overlap[overlap_top_a : overlap_top_a + oh, overlap_left_a : overlap_left_a + ow] = True
        b_overlap[overlap_top_b : overlap_top_b + oh, overlap_left_b : overlap_left_b + ow] = True

        a_win = a_mask[a_overlap].reshape(oh, ow)
        b_win = b_mask[b_overlap].reshape(oh, ow)
        both = a_win & b_win
        a_only = a_win & ~b_win
        b_only = ~a_win & b_win
        neither = ~a_win & ~b_win
        if int(both.sum()) < 2 or not a_only.any() or not b_only.any() or not neither.any():
            continue

        marker_a = _choose_marker(a_mask, a_overlap)
        marker_b = _choose_marker(b_mask, b_overlap)
        if marker_a is None or marker_b is None:
            continue

        common = random.choice(INPUT_BODY_COLORS)
        remaining = [c for c in INPUT_BODY_COLORS if c != common]
        a_palette = (common, random.choice(remaining))
        b_palette = (common, random.choice(remaining))
        a_patch = _paint_sprite(a_mask, a_overlap, marker_a, a_palette)
        b_patch = _paint_sprite(b_mask, b_overlap, marker_b, b_palette)

        both_cells = [tuple(map(int, rc)) for rc in np.argwhere(both)]
        same_cell = random.choice(both_cells)
        diff_choices = [cell for cell in both_cells if cell != same_cell]
        diff_cell = random.choice(diff_choices)

        a_overlap_coords = [tuple(map(int, rc)) for rc in np.argwhere(a_overlap)]
        b_overlap_coords = [tuple(map(int, rc)) for rc in np.argwhere(b_overlap)]
        a_map = {(idx // ow, idx % ow): a_overlap_coords[idx] for idx in range(len(a_overlap_coords))}
        b_map = {(idx // ow, idx % ow): b_overlap_coords[idx] for idx in range(len(b_overlap_coords))}

        for rr, cc in both_cells:
            ar, ac = a_map[(rr, cc)]
            br, bc = b_map[(rr, cc)]
            if (rr, cc) == same_cell:
                a_patch[ar, ac] = common
                b_patch[br, bc] = common
            elif (rr, cc) == diff_cell:
                a_patch[ar, ac] = a_palette[1]
                b_patch[br, bc] = b_palette[1]
            elif random.random() < 0.5:
                a_patch[ar, ac] = common
                b_patch[br, bc] = common
            else:
                a_patch[ar, ac] = a_palette[1]
                b_patch[br, bc] = b_palette[1]

        if a_patch[marker_a] != ORANGE or b_patch[marker_b] != ORANGE:
            continue
        return a_patch, b_patch
    return None


def _place_final_bboxes(h: int, w: int, a_patch: np.ndarray, b_patch: np.ndarray,
                        oh: int, ow: int) -> tuple[int, int, int, int] | None:
    h_a, w_a = a_patch.shape
    h_b, w_b = b_patch.shape
    row_mode = random.choice((0, 1))
    col_mode = random.choice((0, 1))

    if row_mode == 0:
        min_t = h_a - oh
        max_t = h - h_b
    else:
        min_t = h_b - oh
        max_t = h - h_a
    if col_mode == 0:
        min_l = w_a - ow
        max_l = w - w_b
    else:
        min_l = w_b - ow
        max_l = w - w_a
    if min_t > max_t or min_l > max_l:
        return None

    overlap_top = random.randint(min_t, max_t)
    overlap_left = random.randint(min_l, max_l)
    a_top = overlap_top - (h_a - oh) if row_mode == 0 else overlap_top
    b_top = overlap_top if row_mode == 0 else overlap_top - (h_b - oh)
    a_left = overlap_left - (w_a - ow) if col_mode == 0 else overlap_left
    b_left = overlap_left if col_mode == 0 else overlap_left - (w_b - ow)
    if min(a_top, b_top, a_left, b_left) < 1:
        return None
    if a_top + h_a > h - 1 or b_top + h_b > h - 1 or a_left + w_a > w - 1 or b_left + w_b > w - 1:
        return None
    return a_top, a_left, b_top, b_left


def _find_input_positions(h: int, w: int, a_patch: np.ndarray, b_patch: np.ndarray,
                          a_final_top: int, a_final_left: int, b_final_top: int, b_final_left: int,
                          top_anchor: tuple[int, int], right_anchor: tuple[int, int]) -> tuple[int, int] | None:
    h_a, w_a = a_patch.shape
    h_b, w_b = b_patch.shape
    a_marker = tuple(map(int, np.argwhere(a_patch == ORANGE)[0]))
    b_marker = tuple(map(int, np.argwhere(b_patch == ORANGE)[0]))

    a_left_candidates = list(range(1, w - w_a))
    b_top_candidates = list(range(1, h - h_b))
    random.shuffle(a_left_candidates)
    random.shuffle(b_top_candidates)

    for a_left in a_left_candidates[:SHIFT_ATTEMPTS]:
        if a_left == a_final_left:
            continue
        a_occ = _occupied_global(a_patch, a_final_top, a_left)
        a_marker_global = (a_final_top + a_marker[0], a_left + a_marker[1])
        if a_marker_global[1] == top_anchor[1]:
            continue
        if (1, top_anchor[1]) in a_occ:
            continue

        for b_top in b_top_candidates[:SHIFT_ATTEMPTS]:
            if b_top == b_final_top:
                continue
            b_occ = _occupied_global(b_patch, b_top, b_final_left)
            b_marker_global = (b_top + b_marker[0], b_final_left + b_marker[1])
            if b_marker_global[0] == right_anchor[0]:
                continue
            if (right_anchor[0], w - 2) in b_occ:
                continue
            if _touching_or_overlapping(a_occ, b_occ):
                continue

            dist_a = a_marker_global[0] + abs(a_marker_global[1] - top_anchor[1])
            dist_b = b_marker_global[0] + abs(b_marker_global[1] - top_anchor[1])
            if dist_a >= dist_b:
                continue
            return a_left, b_top
    return None


def _semantic_checks(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        solved = main(inp)
    except Exception:
        return False
    if not np.array_equal(solved, expected):
        return False

    vals = set(map(int, expected.flat))
    if not ({BG, BLUE, RED, YELLOW} <= vals):
        return False
    if not any(v in INPUT_BODY_COLORS for v in vals):
        return False

    top_anchor, right_anchor = _find_anchors(inp)
    sprites = _find_sprite_components(inp)
    dists = [
        abs(sprite["marker"][0] - top_anchor[0]) + abs(sprite["marker"][1] - top_anchor[1])
        for sprite in sprites
    ]
    if dists[0] == dists[1]:
        return False
    a_idx = 0 if dists[0] < dists[1] else 1
    b_idx = 1 - a_idx
    sprite_a = sprites[a_idx]
    sprite_b = sprites[b_idx]

    def same_under_variant(sa: dict, sb: dict, top_col: int, right_row: int, move_a: bool, move_b: bool) -> bool:
        try:
            alt = _compare_sprites(sa, sb, top_col, right_row, move_a, move_b)
        except Exception:
            return False
        return bool(np.array_equal(alt, expected))

    if same_under_variant(sprite_a, sprite_b, top_anchor[1], right_anchor[0], False, True):
        return False
    if same_under_variant(sprite_a, sprite_b, top_anchor[1], right_anchor[0], True, False):
        return False
    if same_under_variant(sprite_b, sprite_a, top_anchor[1], right_anchor[0], True, True):
        return False

    for dc in (-1, 1):
        col = top_anchor[1] + dc
        if 0 < col < inp.shape[1] - 1 and same_under_variant(sprite_a, sprite_b, col, right_anchor[0], True, True):
            return False
    for dr in (-1, 1):
        row = right_anchor[0] + dr
        if 0 < row < inp.shape[0] - 1 and same_under_variant(sprite_a, sprite_b, top_anchor[1], row, True, True):
            return False
    return True


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(*GRID_SIDE_RANGE)
    w = random.randint(*GRID_SIDE_RANGE)
    oh, ow = random.choice(OVERLAP_SIZE_OPTIONS)
    h_a = random.randint(max(SPRITE_SIDE_RANGE[0], oh + 1), SPRITE_SIDE_RANGE[1])
    w_a = random.randint(max(SPRITE_SIDE_RANGE[0], ow + 1), SPRITE_SIDE_RANGE[1])
    h_b = random.randint(max(SPRITE_SIDE_RANGE[0], oh + 1), SPRITE_SIDE_RANGE[1])
    w_b = random.randint(max(SPRITE_SIDE_RANGE[0], ow + 1), SPRITE_SIDE_RANGE[1])

    patches = _make_patches(h_a, w_a, h_b, w_b, oh, ow)
    if patches is None:
        return None
    a_patch, b_patch = patches

    final_pos = _place_final_bboxes(h, w, a_patch, b_patch, oh, ow)
    if final_pos is None:
        return None
    a_top, a_left, b_top, b_left = final_pos

    a_marker_local = tuple(map(int, np.argwhere(a_patch == ORANGE)[0]))
    b_marker_local = tuple(map(int, np.argwhere(b_patch == ORANGE)[0]))
    top_anchor = (0, a_left + a_marker_local[1])
    right_anchor = (b_top + b_marker_local[0], w - 1)
    if not (0 < top_anchor[1] < w - 1 and 0 < right_anchor[0] < h - 1):
        return None

    shifted = _find_input_positions(h, w, a_patch, b_patch, a_top, a_left, b_top, b_left, top_anchor, right_anchor)
    if shifted is None:
        return None
    a_input_left, b_input_top = shifted

    inp = np.zeros((h, w), dtype=int)
    inp[top_anchor] = ORANGE
    inp[right_anchor] = ORANGE
    for r, c in np.argwhere(a_patch != BG):
        inp[a_top + int(r), a_input_left + int(c)] = int(a_patch[int(r), int(c)])
    for r, c in np.argwhere(b_patch != BG):
        inp[b_input_top + int(r), b_left + int(c)] = int(b_patch[int(r), int(c)])

    expected = _render_overlap(a_patch, b_patch, a_top, a_left, b_top, b_left)
    if not _semantic_checks(inp, expected):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        sample = _gen_attempt()
        if sample is None:
            continue
        inp, out = sample
        sig = hashlib.md5(out.tobytes() + str(out.shape).encode()).hexdigest()
        if sig in _SEEN_OUTPUT_SIGS:
            continue
        _SEEN_OUTPUT_SIGS.add(sig)
        return inp
    raise RuntimeError("failed to generate a valid input for puzzle 2")
