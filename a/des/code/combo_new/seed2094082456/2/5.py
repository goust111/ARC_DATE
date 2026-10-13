from __future__ import annotations

from collections import deque
from itertools import permutations
import random

import numpy as np


DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
ARC_COLORS = tuple(range(10))
PACK_CANVAS = 30
GROUP_GAP = 2
GENERATION_ATTEMPTS = 400
CROP_MARGIN_LIMIT = 4

LAYER_TUPLES_BY_FOCUS = {
    1: ((1, 1), (1, 1, 1), (1, 2), (1, 2, 1), (1, 3)),
    2: ((2, 1), (2, 1, 1), (2, 2), (2, 2, 1), (2, 3)),
    3: ((3, 1), (3, 1, 1), (3, 2), (3, 2, 1), (3, 2, 2)),
    4: ((4, 1), (4, 1, 1)),
}

INNER_OPTIONS = {
    1: ((5, 5), (5, 7), (7, 5), (7, 7)),
    2: ((5, 5), (5, 7), (7, 5), (7, 7)),
    3: ((5, 5), (5, 7), (7, 5)),
    4: ((5, 5),),
}

PATTERN_TEMPLATES = (
    ((0, 0), (1, 0), (1, 1)),
    ((-1, 0), (0, 0), (1, 0), (1, 1)),
    ((-1, -1), (-1, 0), (0, 0), (1, 0)),
    ((-1, 0), (0, 0), (0, 1), (1, 0)),
    ((-1, 0), (0, 0), (1, 0), (1, -1), (1, 1)),
)


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _rotate_offsets(offsets: tuple[tuple[int, int], ...], turns_cw: int) -> tuple[tuple[int, int], ...]:
    turns = int(turns_cw) % 4
    out = list(offsets)
    for _ in range(turns):
        out = [(dc, -dr) for dr, dc in out]
    return tuple(out)


def _bbox_of(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _component_mask(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _frame_bbox_if_hollow(coords: np.ndarray) -> tuple[int, int, int, int] | None:
    bbox = _bbox_of(coords)
    r0, c0, r1, c1 = bbox
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None
    mask = _component_mask(coords, bbox)
    expected = np.zeros_like(mask, dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    if np.array_equal(mask, expected):
        return bbox
    return None


def _find_monochrome_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(int(r), int(c))])
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            coords = np.asarray(cells, dtype=int)
            components.append({"color": color, "coords": coords, "bbox": _bbox_of(coords)})
    return components


def _rotate_cells_within_box(
    coords: np.ndarray,
    inner_box: tuple[int, int, int, int],
    turns_cw: int,
) -> np.ndarray:
    turns = int(turns_cw) % 4
    if turns == 0:
        return coords.copy()

    r0, c0, r1, c1 = inner_box
    center_r2 = r0 + r1
    center_c2 = c0 + c1
    rotated: list[tuple[int, int]] = []

    for r, c in coords:
        dr2 = 2 * int(r) - center_r2
        dc2 = 2 * int(c) - center_c2
        for _ in range(turns):
            dr2, dc2 = dc2, -dr2
        nr2 = center_r2 + dr2
        nc2 = center_c2 + dc2
        assert nr2 % 2 == 0 and nc2 % 2 == 0, "rotation must stay on grid cells"
        nr = nr2 // 2
        nc = nc2 // 2
        assert r0 <= nr <= r1 and c0 <= nc <= c1, "rotated pattern must remain inside the innermost cavity"
        rotated.append((int(nr), int(nc)))

    return np.asarray(rotated, dtype=int)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each innermost asymmetric sprite by an amount determined by its nested frame depth.

    Input structure:
    - The dominant color is the background.
    - The grid contains several disjoint groups of concentric hollow rectangular
      frames, with one background cell of spacing between consecutive layers.
    - Every group has 1 to 4 frame layers sharing the same center, and the
      innermost frame encloses exactly one small asymmetric monochrome sprite.

    Rule:
    - For each group, count its number of nested frame layers N.
    - Keep all rectangular frame pixels unchanged.
    - Rotate the enclosed sprite clockwise by N * 90 degrees around the center
      of the innermost frame's interior, then repaint it in place.

    Key insight:
    - The rectangular nesting depth is the control signal: each additional frame
      contributes one more clockwise quarter-turn for the innermost sprite.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _most_common_color(g)
    components = _find_monochrome_components(g, bg)
    frames: list[dict] = []
    patterns: list[dict] = []

    for component in components:
        frame_bbox = _frame_bbox_if_hollow(component["coords"])
        if frame_bbox is not None:
            frames.append({**component, "bbox": frame_bbox})
        else:
            patterns.append(component)

    assert frames, "expected at least one rectangular frame"
    assert patterns, "expected at least one inner sprite"

    groups_by_center: dict[tuple[int, int], list[dict]] = {}
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        groups_by_center.setdefault((r0 + r1, c0 + c1), []).append(frame)

    groups: list[dict] = []
    for same_center_frames in groups_by_center.values():
        same_center_frames.sort(
            key=lambda item: (
                item["bbox"][2] - item["bbox"][0],
                item["bbox"][3] - item["bbox"][1],
            ),
            reverse=True,
        )
        for outer, inner in zip(same_center_frames, same_center_frames[1:]):
            or0, oc0, or1, oc1 = outer["bbox"]
            ir0, ic0, ir1, ic1 = inner["bbox"]
            assert (ir0, ic0, ir1, ic1) == (or0 + 2, oc0 + 2, or1 - 2, oc1 - 2), (
                "consecutive nested frames must be separated by a one-cell background ring"
            )
        groups.append({"frames": same_center_frames, "inner_frame": same_center_frames[-1]})

    used_pattern_ids: set[int] = set()
    for group in groups:
        r0, c0, r1, c1 = group["inner_frame"]["bbox"]
        inner_box = (r0 + 1, c0 + 1, r1 - 1, c1 - 1)
        matches: list[int] = []
        for idx, pattern in enumerate(patterns):
            if idx in used_pattern_ids:
                continue
            pr0, pc0, pr1, pc1 = pattern["bbox"]
            if inner_box[0] <= pr0 and inner_box[1] <= pc0 and pr1 <= inner_box[2] and pc1 <= inner_box[3]:
                matches.append(idx)
        assert len(matches) == 1, "each nested frame group must contain exactly one inner sprite"
        match_idx = matches[0]
        used_pattern_ids.add(match_idx)
        group["pattern"] = patterns[match_idx]
        group["inner_box"] = inner_box

    assert len(used_pattern_ids) == len(patterns), "unexpected extra non-frame components detected"

    out = g.copy()
    for group in groups:
        pattern = group["pattern"]
        color = int(pattern["color"])
        coords = pattern["coords"]
        out[coords[:, 0], coords[:, 1]] = bg
        rotated = _rotate_cells_within_box(coords, group["inner_box"], len(group["frames"]))
        out[rotated[:, 0], rotated[:, 1]] = color
    return out


def _draw_frame(arr: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    arr[r0 : r1 + 1, c0] = color
    arr[r0 : r1 + 1, c1] = color
    arr[r0, c0 : c1 + 1] = color
    arr[r1, c0 : c1 + 1] = color


def _paste(dst: np.ndarray, src: np.ndarray, top: int, left: int, transparent: int | None = None) -> None:
    h, w = src.shape
    view = dst[top : top + h, left : left + w]
    if transparent is None:
        view[:] = src
    else:
        mask = src != int(transparent)
        view[mask] = src[mask]


def _choose_layers() -> tuple[int, ...]:
    focus = random.randint(1, 4)
    layers = list(random.choice(LAYER_TUPLES_BY_FOCUS[focus]))
    random.shuffle(layers)
    return tuple(int(x) for x in layers)


def _build_group(layers: int, frame_colors: list[int], bg: int, pattern_color: int) -> tuple[np.ndarray, np.ndarray]:
    inner_h, inner_w = random.choice(INNER_OPTIONS[int(layers)])
    outer_h = inner_h + 2 + 4 * (int(layers) - 1)
    outer_w = inner_w + 2 + 4 * (int(layers) - 1)

    inp = np.full((outer_h, outer_w), bg, dtype=int)
    out = np.full_like(inp, bg)
    for depth, color in enumerate(frame_colors):
        bbox = (2 * depth, 2 * depth, outer_h - 1 - 2 * depth, outer_w - 1 - 2 * depth)
        _draw_frame(inp, bbox, int(color))
        _draw_frame(out, bbox, int(color))

    template = random.choice(PATTERN_TEMPLATES)
    in_rot = random.randrange(4)
    out_rot = (in_rot + int(layers)) % 4
    center_r = outer_h // 2
    center_c = outer_w // 2

    for dr, dc in _rotate_offsets(template, in_rot):
        inp[center_r + dr, center_c + dc] = pattern_color
    for dr, dc in _rotate_offsets(template, out_rot):
        out[center_r + dr, center_c + dc] = pattern_color
    return inp, out


def _layout_two(
    rects: list[tuple[int, int]],
    gap: int,
    orientation: str,
) -> tuple[list[tuple[int, int]], tuple[int, int]] | None:
    (h0, w0), (h1, w1) = rects
    if orientation == "horizontal":
        container_h = max(h0, h1)
        container_w = w0 + gap + w1
        pos0 = (random.randint(0, container_h - h0), 0)
        pos1 = (random.randint(0, container_h - h1), w0 + gap)
    else:
        container_h = h0 + gap + h1
        container_w = max(w0, w1)
        pos0 = (0, random.randint(0, container_w - w0))
        pos1 = (h0 + gap, random.randint(0, container_w - w1))
    if container_h > PACK_CANVAS or container_w > PACK_CANVAS:
        return None
    return [pos0, pos1], (container_h, container_w)


def _layout_three(
    rects: list[tuple[int, int]],
    gap: int,
    kind: str,
) -> tuple[list[tuple[int, int]], tuple[int, int]] | None:
    (h0, w0), (h1, w1), (h2, w2) = rects

    if kind == "row":
        container_h = max(h0, h1, h2)
        container_w = w0 + gap + w1 + gap + w2
        positions = [
            (random.randint(0, container_h - h0), 0),
            (random.randint(0, container_h - h1), w0 + gap),
            (random.randint(0, container_h - h2), w0 + gap + w1 + gap),
        ]
    elif kind == "col":
        container_h = h0 + gap + h1 + gap + h2
        container_w = max(w0, w1, w2)
        positions = [
            (0, random.randint(0, container_w - w0)),
            (h0 + gap, random.randint(0, container_w - w1)),
            (h0 + gap + h1 + gap, random.randint(0, container_w - w2)),
        ]
    elif kind == "top2_bottom1":
        top_h = max(h0, h1)
        container_h = top_h + gap + h2
        container_w = max(w0 + gap + w1, w2)
        positions = [
            (random.randint(0, top_h - h0), 0),
            (random.randint(0, top_h - h1), w0 + gap),
            (top_h + gap, random.randint(0, container_w - w2)),
        ]
    elif kind == "top1_bottom2":
        bottom_h = max(h1, h2)
        container_h = h0 + gap + bottom_h
        container_w = max(w0, w1 + gap + w2)
        positions = [
            (0, random.randint(0, container_w - w0)),
            (h0 + gap + random.randint(0, bottom_h - h1), 0),
            (h0 + gap + random.randint(0, bottom_h - h2), w1 + gap),
        ]
    elif kind == "left2_right1":
        left_w = max(w0, w1)
        container_h = max(h0 + gap + h1, h2)
        container_w = left_w + gap + w2
        positions = [
            (0, random.randint(0, left_w - w0)),
            (h0 + gap, random.randint(0, left_w - w1)),
            (random.randint(0, container_h - h2), left_w + gap),
        ]
    else:
        right_w = max(w1, w2)
        container_h = max(h0, h1 + gap + h2)
        container_w = w0 + gap + right_w
        positions = [
            (random.randint(0, container_h - h0), 0),
            (0, w0 + gap + random.randint(0, right_w - w1)),
            (h1 + gap, w0 + gap + random.randint(0, right_w - w2)),
        ]

    if container_h > PACK_CANVAS or container_w > PACK_CANVAS:
        return None
    return positions, (container_h, container_w)


def _pack_rectangles(shapes: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
    candidates: list[list[tuple[int, int]]] = []
    n = len(shapes)
    assert n in {2, 3}

    for perm in permutations(range(n)):
        ordered = [shapes[idx] for idx in perm]
        if n == 2:
            layouts = (
                _layout_two(ordered, GROUP_GAP, "horizontal"),
                _layout_two(ordered, GROUP_GAP, "vertical"),
            )
        else:
            layouts = tuple(
                _layout_three(ordered, GROUP_GAP, kind)
                for kind in (
                    "row",
                    "col",
                    "top2_bottom1",
                    "top1_bottom2",
                    "left2_right1",
                    "left1_right2",
                )
            )

        for layout in layouts:
            if layout is None:
                continue
            local_positions, (container_h, container_w) = layout
            base_r = random.randint(0, PACK_CANVAS - container_h)
            base_c = random.randint(0, PACK_CANVAS - container_w)
            mapped = [(-1, -1)] * n
            for local_idx, original_idx in enumerate(perm):
                rr, cc = local_positions[local_idx]
                mapped[original_idx] = (base_r + rr, base_c + cc)
            candidates.append(mapped)

    if not candidates:
        return None
    return random.choice(candidates)


def _crop_case(inp: np.ndarray, out: np.ndarray, bg: int) -> tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere(inp != bg)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)

    top = random.randint(max(0, int(r0) - CROP_MARGIN_LIMIT), int(r0))
    left = random.randint(max(0, int(c0) - CROP_MARGIN_LIMIT), int(c0))
    bottom = random.randint(int(r1), min(inp.shape[0] - 1, int(r1) + CROP_MARGIN_LIMIT))
    right = random.randint(int(c1), min(inp.shape[1] - 1, int(c1) + CROP_MARGIN_LIMIT))

    cropped_inp = inp[top : bottom + 1, left : right + 1]
    cropped_out = out[top : bottom + 1, left : right + 1]
    if _most_common_color(cropped_inp) == bg:
        return cropped_inp, cropped_out
    return inp.copy(), out.copy()


def _gen_attempt() -> np.ndarray | None:
    layers_per_group = _choose_layers()
    total_layers = sum(layers_per_group)
    assert total_layers <= 8

    bg, pattern_color = random.sample(ARC_COLORS, 2)
    frame_palette = [color for color in ARC_COLORS if color not in {bg, pattern_color}]
    frame_colors = random.sample(frame_palette, total_layers)

    groups_inp: list[np.ndarray] = []
    groups_out: list[np.ndarray] = []
    shapes: list[tuple[int, int]] = []
    offset = 0
    for layers in layers_per_group:
        colors = frame_colors[offset : offset + layers]
        offset += layers
        group_inp, group_out = _build_group(layers, colors, bg, pattern_color)
        groups_inp.append(group_inp)
        groups_out.append(group_out)
        shapes.append(tuple(map(int, group_inp.shape)))

    positions = _pack_rectangles(shapes)
    if positions is None:
        return None

    inp_canvas = np.full((PACK_CANVAS, PACK_CANVAS), bg, dtype=int)
    out_canvas = np.full_like(inp_canvas, bg)
    for (top, left), group_inp, group_out in zip(positions, groups_inp, groups_out):
        _paste(inp_canvas, group_inp, int(top), int(left), transparent=bg)
        _paste(out_canvas, group_out, int(top), int(left), transparent=bg)

    inp, expected = _crop_case(inp_canvas, out_canvas, bg)
    if inp.ndim != 2 or expected.ndim != 2:
        return None
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return None

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid nested-frame rotation puzzle")


for _template in PATTERN_TEMPLATES:
    base = set(_template)
    assert len(base) == len(_template)
    for turns in (1, 2, 3):
        assert set(_rotate_offsets(_template, turns)) != base, "pattern templates must be rotationally asymmetric"
