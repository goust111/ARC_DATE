from __future__ import annotations

from collections import Counter, deque
from itertools import combinations
import random

import numpy as np


BLACK = 0
FRAME_COLORS = (1, 2, 3, 4, 7)
PATTERN_COLORS = (5, 6, 8, 9)
CANVAS_SIZE = 29
GEN_RETRY_LIMIT = 400


def get_bg(grid: np.ndarray) -> int:
    values = np.asarray(grid, dtype=int).ravel()
    return int(Counter(map(int, values)).most_common(1)[0][0])


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = map(int, bbox)
    return np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int).copy()


def paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int | None = None) -> None:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    if transparent is None:
        view[:] = patch
        return
    mask = patch != int(transparent)
    view[mask] = patch[mask]


def _mask_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    components: list[dict] = []
    for color in sorted(set(map(int, g.ravel())) - {int(bg)}):
        for coords in _mask_components(g == color):
            components.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": bbox_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return components


def _is_rectangular_perimeter(comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    expected = 2 * h + 2 * w - 4
    if int(comp["area"]) != expected:
        return False
    coords = comp["coords"]
    on_border = (
        (coords[:, 0] == r0)
        | (coords[:, 0] == r1)
        | (coords[:, 1] == c0)
        | (coords[:, 1] == c1)
    )
    return bool(np.all(on_border))


def _strictly_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = outer
    rr0, cc0, rr1, cc1 = inner
    return r0 < rr0 and c0 < cc0 and rr1 < r1 and cc1 < c1


def _coords_inside_open_bbox(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    return bool(
        np.all((coords[:, 0] > r0) & (coords[:, 0] < r1) & (coords[:, 1] > c0) & (coords[:, 1] < c1))
    )


def _center_origin(outer_h: int, outer_w: int, inner_h: int, inner_w: int) -> tuple[int, int]:
    dh = outer_h - inner_h
    dw = outer_w - inner_w
    assert dh >= 0 and dw >= 0 and dh % 2 == 0 and dw % 2 == 0
    return dh // 2, dw // 2


def _group_specs(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    comps = find_components(g, bg)
    assert comps, "expected foreground components"

    frames = [comp for comp in comps if _is_rectangular_perimeter(comp)]
    patterns = [comp for comp in comps if not _is_rectangular_perimeter(comp)]
    assert 2 <= len(patterns) <= 3, "expected 2-3 inner patterns"
    assert len(frames) >= 4, "expected nested rectangle frames"

    groups = []
    for pattern in patterns:
        containing = [
            frame
            for frame in frames
            if _strictly_contains(frame["bbox"], pattern["bbox"]) and _coords_inside_open_bbox(pattern["coords"], frame["bbox"])
        ]
        assert 1 <= len(containing) <= 3, "pattern must be enclosed by 1-3 frames"
        by_area = sorted(containing, key=lambda comp: (comp["bbox"][2] - comp["bbox"][0] + 1) * (comp["bbox"][3] - comp["bbox"][1] + 1))
        groups.append(
            {
                "pattern": pattern,
                "frames_inner_to_outer": by_area,
                "depth": len(by_area),
            }
        )

    roots = [tuple(group["frames_inner_to_outer"][-1]["bbox"]) for group in groups]
    assert len(set(roots)) == len(roots), "each pattern must belong to a distinct nested group"
    depths = sorted(group["depth"] for group in groups)
    assert any(depth > 1 for depth in depths)
    return groups


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate each innermost pattern by its nesting depth, then recolor the whole nest to the outermost frame color.

    Input structure:
    - Black cells are background.
    - The foreground is split into 2-3 disjoint groups of nested monochrome hollow rectangular frames.
    - Each group contains exactly one small 4-connected monochrome pattern strictly inside its innermost frame.

    Rule:
    - For each pattern, count how many rectangular frames strictly contain it; this count is the nesting depth.
    - Rotate the pattern clockwise by 90 degrees times that depth, keeping it centered inside the innermost frame.
    - Recolor every frame in the same group to the color of that group's outermost frame, while leaving the pattern color unchanged.

    Key insight:
    - The same containment hierarchy drives two separate actions at once: its depth sets the rotation angle, and its outermost frame supplies the final border color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    groups = _group_specs(g)
    out = g.copy()

    for group in groups:
        pattern = group["pattern"]
        frames = group["frames_inner_to_outer"]
        depth = int(group["depth"])
        inner = frames[0]
        outer = frames[-1]

        rr, cc = pattern["coords"][:, 0], pattern["coords"][:, 1]
        out[rr, cc] = bg

        sprite = crop_bbox(g, pattern["bbox"])
        rotated = np.rot90(sprite, -(depth % 4))

        ir0, ic0, ir1, ic1 = inner["bbox"]
        inner_h = ir1 - ir0 - 1
        inner_w = ic1 - ic0 - 1
        top_off, left_off = _center_origin(inner_h, inner_w, rotated.shape[0], rotated.shape[1])
        dest_r = ir0 + 1 + top_off
        dest_c = ic0 + 1 + left_off
        paste(out, rotated, dest_r, dest_c, transparent=bg)

        final_color = int(outer["color"])
        for frame in frames:
            fr, fc = frame["coords"][:, 0], frame["coords"][:, 1]
            out[fr, fc] = final_color

    return out


def _same_array(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and bool(np.array_equal(a, b))


def _mask_is_connected(mask: np.ndarray) -> bool:
    comps = _mask_components(np.asarray(mask, dtype=bool))
    return len(comps) == 1


def _bbox_fills_shape(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    r0, c0, r1, c1 = bbox_coords(coords)
    return (r0, c0, r1, c1) == (0, 0, mask.shape[0] - 1, mask.shape[1] - 1)


def _is_asymmetric(mask: np.ndarray) -> bool:
    variants = (
        np.fliplr(mask),
        np.flipud(mask),
        np.rot90(mask, 1),
        np.rot90(mask, 2),
        np.rot90(mask, 3),
    )
    return all(not _same_array(mask, variant) for variant in variants)


def _shape_library() -> dict[str, list[np.ndarray]]:
    all_shapes: list[np.ndarray] = []
    seen: set[tuple[bytes, tuple[int, int]]] = set()
    for h, w in ((3, 3), (3, 5), (5, 3)):
        cells = list(range(h * w))
        for area in range(3, 7):
            for picks in combinations(cells, area):
                mask = np.zeros((h, w), dtype=bool)
                for idx in picks:
                    mask[idx // w, idx % w] = True
                if not _bbox_fills_shape(mask):
                    continue
                if not _mask_is_connected(mask):
                    continue
                if not _is_asymmetric(mask):
                    continue
                key = (mask.tobytes(), mask.shape)
                if key in seen:
                    continue
                seen.add(key)
                all_shapes.append(mask.astype(int))

    by_depth_2 = [mask for mask in all_shapes if not _same_array(mask, np.rot90(mask, 2))]
    by_depth_3 = [mask for mask in all_shapes if not _same_array(mask, np.rot90(mask, 3))]
    nonsquare = [mask for mask in by_depth_3 if mask.shape[0] != mask.shape[1]]
    small_square = [mask for mask in all_shapes if mask.shape == (3, 3)]
    return {
        "all": all_shapes,
        "depth2": by_depth_2,
        "depth3": by_depth_3,
        "depth3_nonsquare": nonsquare,
        "small_square": small_square,
    }


SHAPES = _shape_library()


def _choose_pattern(depth: int, group_count: int) -> np.ndarray:
    if depth == 3 and group_count == 2 and random.random() < 0.55 and SHAPES["depth3_nonsquare"]:
        pool = SHAPES["depth3_nonsquare"]
    elif group_count == 3:
        pool = [mask for mask in SHAPES[f"depth{depth}"] if mask.shape == (3, 3)]
    else:
        pool = SHAPES[f"depth{depth}"]

    mask = random.choice(pool).copy()
    mask = np.rot90(mask, random.randrange(4))
    assert not _same_array(mask, np.rot90(mask, depth))
    return mask


def _choose_interior_dims(pattern: np.ndarray, depth: int, group_count: int) -> tuple[int, int]:
    max_dim = max(pattern.shape)
    options = []
    base_sizes = (5, 7) if group_count == 3 else (5, 7, 9)
    for h in base_sizes:
        for w in base_sizes:
            if h < max_dim + 2 or w < max_dim + 2:
                continue
            if (h - pattern.shape[0]) % 2 != 0 or (w - pattern.shape[1]) % 2 != 0:
                continue
            if (h - pattern.shape[1]) % 2 != 0 or (w - pattern.shape[0]) % 2 != 0:
                continue
            options.append((h, w))

    if depth == 3:
        options = [dims for dims in options if dims[0] <= 7 and dims[1] <= 7] or options
    if group_count == 3:
        compact = [dims for dims in options if dims[0] <= 5 and dims[1] <= 5]
        if depth == 3 and compact:
            options = compact
        else:
            options = sorted(options, key=lambda dims: (dims[0] * dims[1], dims[0] + dims[1]))[:3]
    assert options, "no interior dims fit the selected pattern"
    return random.choice(options)


def _expand_size(size: tuple[int, int], gap: int) -> tuple[int, int]:
    h, w = size
    pad = 2 * (gap + 1)
    return h + pad, w + pad


def _draw_frame(patch: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    patch[r0, c0 : c1 + 1] = int(color)
    patch[r1, c0 : c1 + 1] = int(color)
    patch[r0 : r1 + 1, c0] = int(color)
    patch[r0 : r1 + 1, c1] = int(color)


def _render_group(depth: int, group_count: int, outer_color: int) -> tuple[np.ndarray, np.ndarray, dict]:
    pattern_mask = _choose_pattern(depth, group_count)
    interior_h, interior_w = _choose_interior_dims(pattern_mask, depth, group_count)
    inner_frame_size = (interior_h + 2, interior_w + 2)

    gaps = [1] * (depth - 1)
    if group_count == 2 and depth == 2 and random.random() < 0.4:
        gaps[0] = 2
    elif depth == 3 and group_count == 2 and random.random() < 0.2:
        gaps[random.randrange(len(gaps))] = 2

    frame_sizes_inner_to_outer = [inner_frame_size]
    for gap in gaps:
        frame_sizes_inner_to_outer.append(_expand_size(frame_sizes_inner_to_outer[-1], gap))

    outer_h, outer_w = frame_sizes_inner_to_outer[-1]
    frame_sizes_outer_to_inner = list(reversed(frame_sizes_inner_to_outer))
    frame_colors_outer_to_inner = [outer_color]
    frame_colors_outer_to_inner.extend(
        random.sample([color for color in FRAME_COLORS if color != outer_color], depth - 1)
    )
    pattern_color = random.choice(PATTERN_COLORS)

    inp = np.zeros((outer_h, outer_w), dtype=int)
    out = np.zeros((outer_h, outer_w), dtype=int)
    frame_bboxes_outer_to_inner = []
    for size, color in zip(frame_sizes_outer_to_inner, frame_colors_outer_to_inner):
        top, left = _center_origin(outer_h, outer_w, size[0], size[1])
        bbox = (top, left, top + size[0] - 1, left + size[1] - 1)
        frame_bboxes_outer_to_inner.append(bbox)
        _draw_frame(inp, bbox, color)
        _draw_frame(out, bbox, outer_color)

    inner_bbox = frame_bboxes_outer_to_inner[-1]
    inner_r0, inner_c0, inner_r1, inner_c1 = inner_bbox
    origin_r = inner_r0 + 1
    origin_c = inner_c0 + 1
    top_off, left_off = _center_origin(interior_h, interior_w, pattern_mask.shape[0], pattern_mask.shape[1])
    pr = origin_r + top_off
    pc = origin_c + left_off
    pattern_patch = np.where(pattern_mask, pattern_color, BLACK)
    paste(inp, pattern_patch, pr, pc, transparent=BLACK)

    rotated_mask = np.rot90(pattern_mask, -depth)
    assert not _same_array(pattern_mask, rotated_mask)
    rotated_patch = np.where(rotated_mask, pattern_color, BLACK)
    rot_top, rot_left = _center_origin(interior_h, interior_w, rotated_patch.shape[0], rotated_patch.shape[1])
    paste(out, rotated_patch, origin_r + rot_top, origin_c + rot_left, transparent=BLACK)

    meta = {
        "depth": depth,
        "frame_colors_outer_to_inner": frame_colors_outer_to_inner,
        "outer_color": outer_color,
        "pattern_color": pattern_color,
    }
    return inp, out, meta


def _bbox_gap_ok(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    return bool(
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _place_patch(
    canvas: np.ndarray,
    patch: np.ndarray,
    placed: list[tuple[int, int, int, int]],
    *,
    gap: int,
) -> tuple[int, int] | None:
    h, w = patch.shape
    positions = [(r, c) for r in range(canvas.shape[0] - h + 1) for c in range(canvas.shape[1] - w + 1)]
    random.shuffle(positions)
    for r, c in positions:
        bbox = (r, c, r + h - 1, c + w - 1)
        if all(_bbox_gap_ok(bbox, other, gap) for other in placed):
            paste(canvas, patch, r, c, transparent=BLACK)
            placed.append(bbox)
            return r, c
    return None


def _depth_templates(group_count: int) -> list[tuple[int, ...]]:
    if group_count == 2:
        return [(2, 3), (3, 2)]
    return [(2, 3, 2), (3, 2, 3), (2, 3, 3), (3, 2, 2)]


def _build_case() -> tuple[np.ndarray, np.ndarray]:
    group_count = random.choice((2, 3))
    depths = list(random.choice(_depth_templates(group_count)))
    random.shuffle(depths)
    outer_colors = random.sample(FRAME_COLORS, group_count)

    input_patches = []
    output_patches = []
    metas = []
    for depth, outer_color in zip(depths, outer_colors):
        inp_patch, out_patch, meta = _render_group(depth, group_count, outer_color)
        input_patches.append(inp_patch)
        output_patches.append(out_patch)
        metas.append(meta)

    in_canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=int)
    out_canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=int)
    placed: list[tuple[int, int, int, int]] = []
    placements = []
    order = list(range(group_count))
    order.sort(key=lambda idx: input_patches[idx].shape[0] * input_patches[idx].shape[1], reverse=True)
    for idx in order:
        pos = _place_patch(in_canvas, input_patches[idx], placed, gap=2)
        if pos is None:
            raise ValueError("placement failed")
        paste(out_canvas, output_patches[idx], pos[0], pos[1], transparent=BLACK)
        placements.append((idx, pos))

    used = np.argwhere(in_canvas != BLACK)
    assert used.size > 0
    r0, c0, r1, c1 = bbox_coords(used)
    extra_top = random.randint(0, min(2, r0))
    extra_left = random.randint(0, min(2, c0))
    extra_bottom = random.randint(0, min(2, CANVAS_SIZE - 1 - r1))
    extra_right = random.randint(0, min(2, CANVAS_SIZE - 1 - c1))
    final_bbox = (r0 - extra_top, c0 - extra_left, r1 + extra_bottom, c1 + extra_right)
    inp = crop_bbox(in_canvas, final_bbox)
    out = crop_bbox(out_canvas, final_bbox)

    groups = _group_specs(inp)
    observed_depths = sorted(group["depth"] for group in groups)
    assert sorted(depths) == observed_depths
    assert set(observed_depths) == {2, 3}, "each sample must expose both depth branches"
    assert not np.array_equal(inp, out)
    for meta in metas:
        assert len(set(meta["frame_colors_outer_to_inner"])) == len(meta["frame_colors_outer_to_inner"])
        assert meta["pattern_color"] not in meta["frame_colors_outer_to_inner"]
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        try:
            inp, expected = _build_case()
        except (AssertionError, ValueError):
            continue
        try:
            actual = main(inp)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid nested-depth rotation puzzle")
