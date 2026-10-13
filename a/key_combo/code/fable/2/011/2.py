from __future__ import annotations

from collections import defaultdict
import random

import numpy as np


BG = 0
PALETTE = list(range(1, 10))

OBJECT_COUNT_RANGE = (2, 4)
SPRITE_H_RANGE = (3, 6)
SPRITE_W_RANGE = (2, 5)
AXIS_OFFSET_RANGE = (2, 4)
PADDING = 1
GRID_MIN = 18
GRID_MAX = 30
GEN_RETRIES = 400
PLACE_RETRIES = 300
MASK_RETRIES = 120


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors(r: int, c: int, shape: tuple[int, int], connectivity: int) -> list[tuple[int, int]]:
    h, w = shape
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            if connectivity == 4 and abs(dr) + abs(dc) != 1:
                continue
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w:
                out.append((rr, cc))
    return out


def _components_from_mask(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors(rr, cc, mask.shape, connectivity):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _connected(mask: np.ndarray) -> bool:
    comps = _components_from_mask(mask, connectivity=4)
    return len(comps) == 1


def _touches_all_bbox_sides(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _random_sprite_mask(h: int, w: int) -> np.ndarray:
    total = h * w
    lo = max(3, (total + 1) // 2)
    hi = total
    for _ in range(MASK_RETRIES):
        target = random.randint(lo, hi)
        mask = np.ones((h, w), dtype=bool)
        while int(mask.sum()) > target:
            removable = []
            coords = np.argwhere(mask)
            for r_raw, c_raw in coords:
                r = int(r_raw)
                c = int(c_raw)
                if int(mask.sum()) <= 3:
                    break
                trial = mask.copy()
                trial[r, c] = False
                if _touches_all_bbox_sides(trial) and _connected(trial):
                    removable.append((r, c))
            if not removable:
                break
            rr, cc = random.choice(removable)
            mask[rr, cc] = False
        if _touches_all_bbox_sides(mask) and _connected(mask) and int(mask.sum()) >= 3:
            return mask
    raise RuntimeError("failed to sample a connected full-bbox sprite")


def _flip_bbox_h(bbox: tuple[int, int, int, int], width: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return r0, width - 1 - c1, r1, width - 1 - c0


def _flip_bbox_v(bbox: tuple[int, int, int, int], height: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return height - 1 - r1, c0, height - 1 - r0, c1


def _transpose_bbox(bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return c0, r0, c1, r1


def _build_vertical_patch(sprite_color: int, marker_color: int) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    h = random.randint(*SPRITE_H_RANGE)
    w = random.randint(*SPRITE_W_RANGE)
    offset = random.randint(*AXIS_OFFSET_RANGE)
    mask = _random_sprite_mask(h, w)

    axis_c = (w - 1) + offset
    patch_w = 2 * axis_c + 1
    inp = np.full((h, patch_w), BG, dtype=int)
    out = np.full((h, patch_w), BG, dtype=int)

    sprite_coords = np.argwhere(mask)
    inp[sprite_coords[:, 0], sprite_coords[:, 1]] = sprite_color
    out[sprite_coords[:, 0], sprite_coords[:, 1]] = sprite_color

    inp[0, axis_c] = marker_color
    inp[h - 1, axis_c] = marker_color
    out[0, axis_c] = marker_color
    out[h - 1, axis_c] = marker_color

    reflected = sprite_coords.copy()
    reflected[:, 1] = 2 * axis_c - reflected[:, 1]
    out[reflected[:, 0], reflected[:, 1]] = marker_color

    src_bbox = (0, 0, h - 1, w - 1)
    if random.random() < 0.5:
        inp = np.fliplr(inp).copy()
        out = np.fliplr(out).copy()
        src_bbox = _flip_bbox_h(src_bbox, patch_w)

    return inp, out, src_bbox


def _build_horizontal_patch(sprite_color: int, marker_color: int) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    inp, out, src_bbox = _build_vertical_patch(sprite_color, marker_color)
    inp = inp.T.copy()
    out = out.T.copy()
    src_bbox = _transpose_bbox(src_bbox)

    if random.random() < 0.5:
        inp = np.flipud(inp).copy()
        out = np.flipud(out).copy()
        src_bbox = _flip_bbox_v(src_bbox, inp.shape[0])

    return inp, out, src_bbox


def _choose_orientations(count: int) -> list[str]:
    vertical_count = random.randint(1, count - 1)
    orientations = ["V"] * vertical_count + ["H"] * (count - vertical_count)
    random.shuffle(orientations)
    return orientations


def _pick_canvas(patches: list[np.ndarray]) -> tuple[int, int] | None:
    max_h = max(p.shape[0] for p in patches) + 2
    max_w = max(p.shape[1] for p in patches) + 2
    need_area = sum((p.shape[0] + 2 * PADDING) * (p.shape[1] + 2 * PADDING) for p in patches)
    min_h = max(GRID_MIN, max_h)
    min_w = max(GRID_MIN, max_w)
    if min_h > GRID_MAX or min_w > GRID_MAX:
        return None

    choices = []
    for h in range(min_h, GRID_MAX + 1):
        for w in range(min_w, GRID_MAX + 1):
            if h * w >= int(need_area * 1.25):
                choices.append((h, w))
    return random.choice(choices) if choices else None


def _try_place_patch(
    inp: np.ndarray,
    expected: np.ndarray,
    reserved: np.ndarray,
    patch_in: np.ndarray,
    patch_out: np.ndarray,
    src_bbox: tuple[int, int, int, int],
    used_row_spans: set[tuple[int, int]],
    used_col_spans: set[tuple[int, int]],
) -> bool:
    h, w = inp.shape
    ph, pw = patch_in.shape
    min_r = 1 if h - ph >= 2 else 0
    max_r = h - ph - 1 if h - ph >= 2 else h - ph
    min_c = 1 if w - pw >= 2 else 0
    max_c = w - pw - 1 if w - pw >= 2 else w - pw
    if max_r < min_r or max_c < min_c:
        return False

    for _ in range(PLACE_RETRIES):
        r0 = random.randint(min_r, max_r)
        c0 = random.randint(min_c, max_c)

        rr0 = max(0, r0 - PADDING)
        cc0 = max(0, c0 - PADDING)
        rr1 = min(h, r0 + ph + PADDING)
        cc1 = min(w, c0 + pw + PADDING)
        if reserved[rr0:rr1, cc0:cc1].any():
            continue

        sr0, sc0, sr1, sc1 = src_bbox
        row_span = (r0 + sr0, r0 + sr1)
        col_span = (c0 + sc0, c0 + sc1)
        if row_span in used_row_spans or col_span in used_col_spans:
            continue

        inp[r0 : r0 + ph, c0 : c0 + pw] = np.maximum(inp[r0 : r0 + ph, c0 : c0 + pw], patch_in)
        expected[r0 : r0 + ph, c0 : c0 + pw] = np.maximum(expected[r0 : r0 + ph, c0 : c0 + pw], patch_out)
        reserved[rr0:rr1, cc0:cc1] = True
        used_row_spans.add(row_span)
        used_col_spans.add(col_span)
        return True

    return False


def _generate_case() -> tuple[np.ndarray, np.ndarray]:
    count = random.randint(*OBJECT_COUNT_RANGE)
    orientations = _choose_orientations(count)
    colors = random.sample(PALETTE, 2 * count)
    sprite_colors = colors[:count]
    marker_colors = colors[count:]

    patch_specs = []
    for orientation, sprite_color, marker_color in zip(orientations, sprite_colors, marker_colors):
        if orientation == "V":
            patch_in, patch_out, src_bbox = _build_vertical_patch(sprite_color, marker_color)
        else:
            patch_in, patch_out, src_bbox = _build_horizontal_patch(sprite_color, marker_color)
        patch_specs.append((patch_in, patch_out, src_bbox))

    canvas = _pick_canvas([spec[0] for spec in patch_specs])
    if canvas is None:
        raise RuntimeError("no valid canvas size")
    h, w = canvas

    inp = np.full((h, w), BG, dtype=int)
    expected = np.full((h, w), BG, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)
    used_row_spans: set[tuple[int, int]] = set()
    used_col_spans: set[tuple[int, int]] = set()

    random.shuffle(patch_specs)
    for patch_in, patch_out, src_bbox in patch_specs:
        if not _try_place_patch(inp, expected, reserved, patch_in, patch_out, src_bbox, used_row_spans, used_col_spans):
            raise RuntimeError("failed to place patch")

    if np.array_equal(inp, expected):
        raise RuntimeError("degenerate case: output identical to input")
    return inp, expected


def _solve_assignments(grid: np.ndarray) -> list[tuple[np.ndarray, str, int, int]]:
    bg = _most_common_color(grid)
    comps = _components_from_mask(grid != bg, connectivity=8)
    assert comps, "expected foreground components"

    markers_by_color: dict[int, list[tuple[int, int]]] = defaultdict(list)
    sprites: list[np.ndarray] = []
    sprite_boxes: list[tuple[int, int, int, int]] = []

    for coords in comps:
        if coords.shape[0] == 1:
            r, c = map(int, coords[0])
            markers_by_color[int(grid[r, c])].append((r, c))
        else:
            sprites.append(coords)
            sprite_boxes.append(_bbox_coords(coords))

    assert sprites, "expected at least one sprite"
    assert markers_by_color, "expected marker pairs"

    assignments: list[tuple[np.ndarray, str, int, int]] = []
    used_sprite_ids: set[int] = set()

    for color, pts in markers_by_color.items():
        assert len(pts) == 2, "each marker color must appear exactly twice"
        (r1, c1), (r2, c2) = pts
        if c1 == c2:
            axis_kind = "V"
            axis_pos = c1
            row_span = tuple(sorted((r1, r2)))
            candidates = [i for i, box in enumerate(sprite_boxes) if (box[0], box[2]) == row_span]
        else:
            assert r1 == r2, "marker pair must share a row or a column"
            axis_kind = "H"
            axis_pos = r1
            col_span = tuple(sorted((c1, c2)))
            candidates = [i for i, box in enumerate(sprite_boxes) if (box[1], box[3]) == col_span]

        assert len(candidates) == 1, "marker pair must match exactly one sprite bbox span"
        sprite_id = candidates[0]
        assert sprite_id not in used_sprite_ids, "each sprite should be matched once"
        used_sprite_ids.add(sprite_id)
        assignments.append((sprites[sprite_id], axis_kind, axis_pos, int(color)))

    assert len(assignments) == len(sprites), "expected one marker pair per sprite"
    return assignments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reflect each sprite across the invisible axis encoded by its marker pair and recolor the reflected copy to the marker color.

    Input structure:
    - The grid has a black background plus several multi-pixel sprites and several isolated marker pixels.
    - Each marker color appears exactly twice and does not appear inside any sprite.
    - A marker pair is either vertical (same column) on one sprite bbox's top/bottom rows, or horizontal (same row) on one sprite bbox's left/right columns.

    Rule:
    - Treat each same-column marker pair as a vertical mirror axis at that column, and each same-row pair as a horizontal mirror axis at that row.
    - Match every pair to the unique sprite whose bbox shares that pair's row span or column span.
    - Reflect the entire matched sprite across the inferred axis, paint the reflected cells with the marker color, and keep the original sprites and marker pixels unchanged.

    Key insight:
    - The axis itself is never drawn: the two marker pixels jointly encode the axis orientation, axis position, target sprite, and copy color through bbox-span alignment.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = _most_common_color(g)
    out = g.copy()
    pending = np.zeros(g.shape, dtype=bool)
    paint = np.zeros(g.shape, dtype=int)

    for coords, axis_kind, axis_pos, color in _solve_assignments(g):
        if axis_kind == "V":
            reflected = np.column_stack((coords[:, 0], 2 * axis_pos - coords[:, 1]))
        else:
            reflected = np.column_stack((2 * axis_pos - coords[:, 0], coords[:, 1]))

        assert np.all(reflected[:, 0] >= 0) and np.all(reflected[:, 0] < g.shape[0])
        assert np.all(reflected[:, 1] >= 0) and np.all(reflected[:, 1] < g.shape[1])

        for r_raw, c_raw in reflected:
            r = int(r_raw)
            c = int(c_raw)
            if int(g[r, c]) != bg:
                continue
            if pending[r, c]:
                assert int(paint[r, c]) == color, "reflected copies may not conflict"
            else:
                pending[r, c] = True
                paint[r, c] = int(color)

    out[pending] = paint[pending]
    return out


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            inp, expected = _generate_case()
        except RuntimeError:
            continue

        out = main(inp)
        if np.array_equal(out, expected):
            return inp

    raise RuntimeError("failed to generate a valid invisible-axis mirror case")
