from __future__ import annotations

import random

import numpy as np


BG = 0
SPRITE_COLOR = 5
STAMP_COLOR = 7
FRAME_COLORS = [1, 2, 3, 4, 6, 8, 9]

GEN_RETRIES = 400
PLACE_TRIES = 300
MARGIN = 1
PADDING = 1


BASE_SHAPES: dict[int, list[np.ndarray]] = {
    1: [np.array([[1]], dtype=bool)],
    2: [np.array([[1, 1]], dtype=bool)],
    3: [
        np.array([[1, 1, 1]], dtype=bool),
        np.array([[1, 0], [1, 1]], dtype=bool),
    ],
    4: [
        np.array([[1, 1], [1, 1]], dtype=bool),
        np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
        np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
        np.array([[0, 1, 1], [1, 1, 0]], dtype=bool),
    ],
    5: [
        np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
        np.array([[1, 1], [1, 1], [1, 0]], dtype=bool),
        np.array([[1, 0, 1], [1, 1, 1]], dtype=bool),
        np.array([[1, 1, 0], [0, 1, 1], [0, 1, 0]], dtype=bool),
    ],
}


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(pts: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    return min(rows), min(cols), max(rows), max(cols)


def _frame_boundary(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    boundary: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        boundary.add((r0, c))
        boundary.add((r1, c))
    for r in range(r0, r1 + 1):
        boundary.add((r, c0))
        boundary.add((r, c1))
    return boundary


def _frame_bbox_from_component(pts: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = _bbox(pts)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None
    if set(pts) != _frame_boundary(r0, c0, r1, c1):
        return None
    return r0, c0, r1, c1


def _solid_square_side(pts: list[tuple[int, int]]) -> int | None:
    r0, c0, r1, c1 = _bbox(pts)
    side_h = r1 - r0 + 1
    side_w = c1 - c0 + 1
    if side_h != side_w:
        return None
    if len(pts) != side_h * side_h:
        return None
    expected = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
    if set(pts) != expected:
        return None
    return side_h


def _find_frames(grid: np.ndarray) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == BG or color == SPRITE_COLOR:
            continue
        for pts in _components(grid == color):
            bbox = _frame_bbox_from_component(pts)
            if bbox is None:
                continue
            frames.append({"color": color, "bbox": bbox})
    frames.sort(key=lambda item: tuple(item["bbox"]))  # deterministic traversal
    return frames


def _find_stamps(grid: np.ndarray) -> dict[int, list[tuple[int, int]]]:
    stamps: dict[int, list[tuple[int, int]]] = {}
    for pts in _components(grid == STAMP_COLOR):
        side = _solid_square_side(pts)
        if side is None:
            continue
        stamps[side] = pts
    return stamps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each frame's grey-sprite area to an external square stamp size.

    For every hollow rectangular frame:
    - count the grey pixels inside the frame;
    - use that count A as the side length of the replacement square;
    - erase the grey sprite and place an A x A square centered inside the frame,
      using the frame's border color;
    - erase the external orange square stamp of side A.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    frames = _find_frames(g)
    stamps = _find_stamps(g)
    used_sizes: set[int] = set()

    for frame in frames:
        color = int(frame["color"])
        r0, c0, r1, c1 = frame["bbox"]
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        area = int(np.sum(inner == SPRITE_COLOR))
        assert 1 <= area <= 5
        ih, iw = map(int, inner.shape)
        assert area <= ih and area <= iw

        region = out[r0 + 1 : r1, c0 + 1 : c1]
        region[inner == SPRITE_COLOR] = BG

        top = r0 + 1 + (ih - area) // 2
        left = c0 + 1 + (iw - area) // 2
        out[top : top + area, left : left + area] = color
        used_sizes.add(area)

    for side in used_sizes:
        assert side in stamps
        for r, c in stamps[side]:
            out[r, c] = BG

    return out


def _shape_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[bytes] = set()
    for k in range(4):
        rot = np.rot90(mask, k)
        for candidate in (rot, np.fliplr(rot)):
            key = candidate.tobytes()
            if key in seen:
                continue
            seen.add(key)
            variants.append(candidate.copy())
    return variants


SHAPE_LIBRARY: dict[int, list[np.ndarray]] = {
    area: [variant for base in bases for variant in _shape_variants(base)]
    for area, bases in BASE_SHAPES.items()
}


def _place_rect(
    occupied: np.ndarray, obj_h: int, obj_w: int, margin: int = MARGIN, pad: int = PADDING
) -> tuple[int, int] | None:
    h, w = occupied.shape
    min_r = margin
    min_c = margin
    max_r = h - obj_h - margin
    max_c = w - obj_w - margin
    if max_r < min_r or max_c < min_c:
        return None
    for _ in range(PLACE_TRIES):
        r0 = random.randint(min_r, max_r)
        c0 = random.randint(min_c, max_c)
        rr0 = max(0, r0 - pad)
        cc0 = max(0, c0 - pad)
        rr1 = min(h, r0 + obj_h + pad)
        cc1 = min(w, c0 + obj_w + pad)
        if occupied[rr0:rr1, cc0:cc1].any():
            continue
        occupied[r0 : r0 + obj_h, c0 : c0 + obj_w] = True
        return r0, c0
    return None


def _draw_frame(canvas: np.ndarray, r0: int, c0: int, fh: int, fw: int, color: int) -> None:
    canvas[r0, c0 : c0 + fw] = color
    canvas[r0 + fh - 1, c0 : c0 + fw] = color
    canvas[r0 : r0 + fh, c0] = color
    canvas[r0 : r0 + fh, c0 + fw - 1] = color


def _paint_mask(canvas: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    canvas[top : top + h, left : left + w][mask] = color


def _candidate_grid_sizes(frame_specs: list[dict[str, object]], max_stamp_size: int) -> list[tuple[int, int]]:
    rect_area = sum(int(spec["frame_h"]) * int(spec["frame_w"]) for spec in frame_specs)
    rect_area += sum(size * size for size in range(1, max_stamp_size + 1))
    max_h = max([int(spec["frame_h"]) for spec in frame_specs] + [max_stamp_size])
    max_w = max([int(spec["frame_w"]) for spec in frame_specs] + [max_stamp_size])

    candidates: list[tuple[int, int]] = []
    for h in range(22, 31):
        for w in range(22, 31):
            if h < max_h + 2 * MARGIN or w < max_w + 2 * MARGIN:
                continue
            if h * w < int(np.ceil(rect_area * 1.65)):
                continue
            candidates.append((h, w))
    return candidates


def _build_frame_specs(used_sizes: list[int], colors: list[int]) -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    for area, color in zip(used_sizes, colors):
        shape = random.choice(SHAPE_LIBRARY[area]).copy()
        sh, sw = map(int, shape.shape)
        inner_h = area + random.choice([2, 2, 4])
        inner_w = area + random.choice([2, 2, 4])
        frame_h = inner_h + 2
        frame_w = inner_w + 2
        specs.append(
            {
                "area": area,
                "color": color,
                "shape": shape,
                "shape_h": sh,
                "shape_w": sw,
                "inner_h": inner_h,
                "inner_w": inner_w,
                "frame_h": frame_h,
                "frame_w": frame_w,
            }
        )
    return specs


def _gen_attempt() -> np.ndarray | None:
    max_stamp_size = random.randint(3, 5)
    frame_count = random.randint(2, min(4, max_stamp_size))
    used_sizes = sorted(random.sample(range(1, max_stamp_size + 1), frame_count))
    frame_colors = random.sample(FRAME_COLORS, frame_count)
    frame_specs = _build_frame_specs(used_sizes, frame_colors)

    candidates = _candidate_grid_sizes(frame_specs, max_stamp_size)
    if not candidates:
        return None
    h, w = random.choice(candidates)
    occupied = np.zeros((h, w), dtype=bool)

    frame_positions: list[tuple[dict[str, object], int, int]] = []
    for spec in sorted(frame_specs, key=lambda item: (int(item["frame_h"]) * int(item["frame_w"])), reverse=True):
        pos = _place_rect(occupied, int(spec["frame_h"]), int(spec["frame_w"]))
        if pos is None:
            return None
        frame_positions.append((spec, pos[0], pos[1]))

    stamp_positions: dict[int, tuple[int, int]] = {}
    for size in range(max_stamp_size, 0, -1):
        pos = _place_rect(occupied, size, size)
        if pos is None:
            return None
        stamp_positions[size] = pos

    inp = np.full((h, w), BG, dtype=int)
    expected = np.full((h, w), BG, dtype=int)

    for spec, r0, c0 in frame_positions:
        color = int(spec["color"])
        area = int(spec["area"])
        inner_h = int(spec["inner_h"])
        inner_w = int(spec["inner_w"])
        shape = spec["shape"]
        shape_h = int(spec["shape_h"])
        shape_w = int(spec["shape_w"])

        _draw_frame(inp, r0, c0, int(spec["frame_h"]), int(spec["frame_w"]), color)
        _draw_frame(expected, r0, c0, int(spec["frame_h"]), int(spec["frame_w"]), color)

        sprite_top = r0 + 1 + (inner_h - shape_h) // 2
        sprite_left = c0 + 1 + (inner_w - shape_w) // 2
        _paint_mask(inp, shape, sprite_top, sprite_left, SPRITE_COLOR)

        stamp_top = r0 + 1 + (inner_h - area) // 2
        stamp_left = c0 + 1 + (inner_w - area) // 2
        expected[stamp_top : stamp_top + area, stamp_left : stamp_left + area] = color

    for size, (r0, c0) in stamp_positions.items():
        inp[r0 : r0 + size, c0 : c0 + size] = STAMP_COLOR
        if size not in used_sizes:
            expected[r0 : r0 + size, c0 : c0 + size] = STAMP_COLOR

    produced = main(inp)
    if not np.array_equal(produced, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
