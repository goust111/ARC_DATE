from __future__ import annotations

from collections import deque
import math
import random

import numpy as np


BG = 0
ANCHOR = 8
BODY_COLORS = (1, 2, 3, 4, 5, 6, 7, 9)

FRAGMENT_COUNT_RANGE = (3, 5)
OUTPUT_SIDE_RANGE = (6, 12)
FRAGMENT_AREA_MIN = 6
PLACEMENT_GAP = 1
GEN_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 80
PLACEMENT_NODE_BUDGET = 6000


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _extract_fragments(grid: np.ndarray) -> list[dict]:
    comps = _components_mask(grid != BG)
    assert comps, "expected at least one fragment"

    fragments: list[dict] = []
    for coords in comps:
        r0, c0, r1, c1 = _bbox_coords(coords)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        h, w = patch.shape
        assert int(coords.shape[0]) == h * w, "each fragment must be a filled rectangle"
        assert np.all(patch != BG), "fragment interiors must be non-black"

        anchors = np.argwhere(patch == ANCHOR)
        assert anchors.shape == (1, 2), "each fragment must contain exactly one anchor pixel"
        ar, ac = map(int, anchors[0])

        fragments.append(
            {
                "bbox": (r0, c0, r1, c1),
                "patch": patch,
                "anchor": (ar, ac),
            }
        )

    return fragments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align every rectangular fragment by its cyan anchor and merge them into the hidden sprite.

    Input structure:
    - Black background contains several disjoint filled rectangular fragments.
    - Every fragment has exactly one cyan anchor pixel; every other fragment cell is a non-black sprite color.
    - All fragments are windows onto the same hidden sprite, but the complete sprite never appears directly.

    Rule:
    - Split the non-black cells into filled rectangles.
    - For each rectangle, treat its cyan pixel as the same sprite coordinate and translate the whole fragment so all anchors coincide.
    - Merge the aligned rectangles, requiring overlapping non-cyan cells to agree, then return the tight crop of the merged union while keeping the anchor cyan.

    Key insight:
    - The fragments do not need edge matching: the only reliable registration cue is that every rectangle shows the same sprite cell in cyan.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    fragments = _extract_fragments(g)

    aligned: dict[tuple[int, int], int] = {}
    for fragment in fragments:
        patch = fragment["patch"]
        ar, ac = fragment["anchor"]
        h, w = patch.shape
        for rr in range(h):
            for cc in range(w):
                value = int(patch[rr, cc])
                key = (rr - ar, cc - ac)
                if key in aligned:
                    assert aligned[key] == value, "aligned overlaps must agree"
                else:
                    aligned[key] = value

    assert aligned.get((0, 0)) == ANCHOR, "the shared anchor must be present"

    rows = [r for r, _c in aligned]
    cols = [c for _r, c in aligned]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)

    out = np.full((max_r - min_r + 1, max_c - min_c + 1), BG, dtype=int)
    for (rr, cc), value in aligned.items():
        out[rr - min_r, cc - min_c] = int(value)
    return out


def _rect_cells(rect: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _build_union(
    rects: list[tuple[int, int, int, int]], anchor_r: int, anchor_c: int, h: int, w: int
) -> tuple[np.ndarray, list[set[tuple[int, int]]], np.ndarray]:
    union_mask = np.zeros((h, w), dtype=bool)
    cover = np.zeros((h, w), dtype=int)
    rect_sets: list[set[tuple[int, int]]] = []
    for up, left, down, right in rects:
        r0 = anchor_r - up
        c0 = anchor_c - left
        r1 = anchor_r + down
        c1 = anchor_c + right
        rect = (r0, c0, r1, c1)
        cells = _rect_cells(rect)
        rect_sets.append(cells)
        for r, c in cells:
            union_mask[r, c] = True
            cover[r, c] += 1
    return union_mask, rect_sets, cover


def _valid_fragment_family(
    rects: list[tuple[int, int, int, int]], anchor_r: int, anchor_c: int, h: int, w: int
) -> bool:
    union_mask, rect_sets, cover = _build_union(rects, anchor_r, anchor_c, h, w)
    union_cells = set(map(tuple, np.argwhere(union_mask)))
    union_area = len(union_cells)

    if not (OUTPUT_SIDE_RANGE[0] <= h <= OUTPUT_SIDE_RANGE[1]):
        return False
    if not (OUTPUT_SIDE_RANGE[0] <= w <= OUTPUT_SIDE_RANGE[1]):
        return False
    if union_area < 18 or union_area >= h * w:
        return False

    anchor_offsets = set()
    has_interior_anchor = False
    has_boundary_anchor = False
    max_overlap = 1

    for i, (up, left, down, right) in enumerate(rects):
        frag_h = up + down + 1
        frag_w = left + right + 1
        if frag_h < 2 or frag_w < 2 or frag_h * frag_w < FRAGMENT_AREA_MIN:
            return False
        anchor_offsets.add((up, left))
        if up > 0 and down > 0 and left > 0 and right > 0:
            has_interior_anchor = True
        if up == 0 or down == 0 or left == 0 or right == 0:
            has_boundary_anchor = True

        others = set()
        for j, other in enumerate(rect_sets):
            if i != j:
                others |= other
        unique_cells = rect_sets[i] - others
        if len(unique_cells) < 2:
            return False
        if rect_sets[i] == union_cells:
            return False

    if len(anchor_offsets) < 2 or not has_interior_anchor or not has_boundary_anchor:
        return False

    for i in range(len(rect_sets)):
        for j in range(i + 1, len(rect_sets)):
            max_overlap = max(max_overlap, len(rect_sets[i] & rect_sets[j]))
    if max_overlap < 2:
        return False

    if not np.any(cover == 1) or not np.any(cover >= 2):
        return False

    return True


def _sample_fragment_family() -> tuple[list[tuple[int, int, int, int]], int, int, int, int] | None:
    for _ in range(300):
        count = random.randint(*FRAGMENT_COUNT_RANGE)
        rects: list[tuple[int, int, int, int]] = []
        while len(rects) < count:
            up = random.randint(0, 5)
            down = random.randint(0, 5)
            left = random.randint(0, 5)
            right = random.randint(0, 5)
            if up + down + 1 < 2 or left + right + 1 < 2:
                continue
            rect = (up, left, down, right)
            if rect not in rects:
                rects.append(rect)

        max_up = max(r[0] for r in rects)
        max_left = max(r[1] for r in rects)
        max_down = max(r[2] for r in rects)
        max_right = max(r[3] for r in rects)
        h = max_up + max_down + 1
        w = max_left + max_right + 1
        anchor_r = max_up
        anchor_c = max_left

        if _valid_fragment_family(rects, anchor_r, anchor_c, h, w):
            return rects, anchor_r, anchor_c, h, w
    return None


def _paint_union(union_mask: np.ndarray, anchor_r: int, anchor_c: int) -> np.ndarray | None:
    h, w = union_mask.shape
    body_coords = [(int(r), int(c)) for r, c in np.argwhere(union_mask) if (int(r), int(c)) != (anchor_r, anchor_c)]
    if len(body_coords) < 6:
        return None

    for _ in range(80):
        color_count = random.randint(2, 4)
        palette = random.sample(BODY_COLORS, color_count)
        mode = random.choice(("bands", "tile", "quadrant", "diag"))
        out = np.full((h, w), BG, dtype=int)

        if mode == "bands":
            row_step = random.randint(1, 3)
            col_step = random.randint(1, 3)
            offset = random.randrange(color_count)
            for r, c in body_coords:
                idx = ((r // row_step) + (c // col_step) + offset) % color_count
                out[r, c] = palette[idx]
        elif mode == "tile":
            tile_h = random.randint(2, 3)
            tile_w = random.randint(2, 4)
            tile = np.array(
                [[random.choice(palette) for _ in range(tile_w)] for _ in range(tile_h)],
                dtype=int,
            )
            rr_off = random.randrange(tile_h)
            cc_off = random.randrange(tile_w)
            for r, c in body_coords:
                out[r, c] = int(tile[(r + rr_off) % tile_h, (c + cc_off) % tile_w])
        elif mode == "quadrant":
            palette4 = [palette[i % color_count] for i in range(4)]
            parity = random.randint(0, 1)
            for r, c in body_coords:
                key = 2 * int(r >= anchor_r) + int(c >= anchor_c)
                if (abs(r - anchor_r) + abs(c - anchor_c) + parity) % 2:
                    key = (key + 1) % 4
                out[r, c] = int(palette4[key])
        else:
            a = random.randint(1, 3)
            b = random.randint(2, 5)
            offset = random.randrange(color_count)
            for r, c in body_coords:
                idx = (a * (r - anchor_r) + b * (c - anchor_c) + offset) % color_count
                out[r, c] = palette[idx]

        out[anchor_r, anchor_c] = ANCHOR
        used = sorted(set(map(int, out[(out != BG) & (out != ANCHOR)].flat)))
        if len(used) < 2:
            continue
        return out
    return None


def _fragment_patch(
    output: np.ndarray, anchor_r: int, anchor_c: int, rect: tuple[int, int, int, int]
) -> np.ndarray:
    up, left, down, right = rect
    patch = output[anchor_r - up : anchor_r + down + 1, anchor_c - left : anchor_c + right + 1].copy()
    assert np.all(patch != BG), "fragments must not include background cells"
    assert np.argwhere(patch == ANCHOR).shape == (1, 2)
    return patch


def _boxes_intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _expanded_box(box: tuple[int, int, int, int], h: int, w: int, gap: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    return (max(0, r0 - gap), max(0, c0 - gap), min(h - 1, r1 + gap), min(w - 1, c1 + gap))


def _place_patches(patches: list[np.ndarray], h: int, w: int) -> list[tuple[int, int]] | None:
    order = sorted(range(len(patches)), key=lambda i: patches[i].shape[0] * patches[i].shape[1], reverse=True)
    placements: list[tuple[int, int] | None] = [None] * len(patches)
    placed_boxes: list[tuple[int, int, int, int] | None] = [None] * len(patches)
    node_budget = PLACEMENT_NODE_BUDGET

    candidate_positions: dict[int, list[tuple[int, int]]] = {}
    for i, patch in enumerate(patches):
        ph, pw = patch.shape
        positions = [(r, c) for r in range(h - ph + 1) for c in range(w - pw + 1)]
        random.shuffle(positions)
        candidate_positions[i] = positions

    def rec(pos: int) -> bool:
        nonlocal node_budget
        if pos == len(order):
            return True
        if node_budget <= 0:
            return False
        node_budget -= 1

        idx = order[pos]
        ph, pw = patches[idx].shape
        for r, c in candidate_positions[idx]:
            box = (r, c, r + ph - 1, c + pw - 1)
            safe = _expanded_box(box, h, w, PLACEMENT_GAP)
            if any(
                other is not None and _boxes_intersect(safe, other)
                for other in placed_boxes
            ):
                continue
            placements[idx] = (r, c)
            placed_boxes[idx] = box
            if rec(pos + 1):
                return True
            placements[idx] = None
            placed_boxes[idx] = None
        return False

    if not rec(0):
        return None
    return [p for p in placements if p is not None]


def _sample_canvas_dims(patches: list[np.ndarray]) -> tuple[int, int]:
    total_area = sum(int(p.shape[0] * p.shape[1]) for p in patches)
    max_h = max(int(p.shape[0]) for p in patches)
    max_w = max(int(p.shape[1]) for p in patches)
    base = max(18, int(math.ceil(math.sqrt(total_area * 2.5))))
    min_h = max(base, max_h + 2)
    min_w = max(base, max_w + 2)
    return random.randint(min_h, 30), random.randint(min_w, 30)


def _render_input(patches: list[np.ndarray]) -> np.ndarray | None:
    for _ in range(PLACEMENT_ATTEMPTS):
        h, w = _sample_canvas_dims(patches)
        placements = _place_patches(patches, h, w)
        if placements is None:
            continue
        grid = np.full((h, w), BG, dtype=int)
        for patch, (r0, c0) in zip(patches, placements):
            ph, pw = patch.shape
            grid[r0 : r0 + ph, c0 : c0 + pw] = patch
        return grid
    return None


def _semantic_checks(
    inp: np.ndarray,
    expected: np.ndarray,
    rects: list[tuple[int, int, int, int]],
    anchor_r: int,
    anchor_c: int,
) -> bool:
    parsed = _extract_fragments(inp)
    if len(parsed) != len(rects):
        return False
    if not np.array_equal(main(inp), expected):
        return False

    union_mask = expected != BG
    if np.count_nonzero(expected == ANCHOR) != 1:
        return False
    if not union_mask[anchor_r, anchor_c]:
        return False
    if not np.any(expected == BG):
        return False

    _, rect_sets, cover = _build_union(rects, anchor_r, anchor_c, *expected.shape)
    union_cells = set(map(tuple, np.argwhere(union_mask)))
    if union_cells != set().union(*rect_sets):
        return False
    if not np.any(cover == 1) or not np.any(cover >= 2):
        return False

    for i, rect_set in enumerate(rect_sets):
        others = set()
        for j, other in enumerate(rect_sets):
            if i != j:
                others |= other
        if rect_set == union_cells:
            return False
        if len(rect_set - others) < 2:
            return False

    multicolor_fragments = 0
    for fragment in parsed:
        patch = fragment["patch"]
        colors = sorted(set(map(int, patch[(patch != BG) & (patch != ANCHOR)].flat)))
        if len(colors) >= 2:
            multicolor_fragments += 1
    if multicolor_fragments < 2:
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    sampled = _sample_fragment_family()
    if sampled is None:
        return None
    rects, anchor_r, anchor_c, h, w = sampled

    union_mask, _rect_sets, _cover = _build_union(rects, anchor_r, anchor_c, h, w)
    output = _paint_union(union_mask, anchor_r, anchor_c)
    if output is None:
        return None

    patches = [_fragment_patch(output, anchor_r, anchor_c, rect) for rect in rects]
    inp = _render_input(patches)
    if inp is None:
        return None

    if not _semantic_checks(inp, output, rects, anchor_r, anchor_c):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid anchor-fragment puzzle after {GEN_ATTEMPTS} attempts")
