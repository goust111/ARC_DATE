from __future__ import annotations

from importlib.machinery import SourceFileLoader
from pathlib import Path
import random

import numpy as np


_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
try:
    _arc_utils = SourceFileLoader("_arc_utils_014_4", str(_ARC_UTILS_PATH)).load_module()
except Exception:  # pragma: no cover
    import arc_utils as _arc_utils  # type: ignore

get_bg = _arc_utils.get_bg
find_components = _arc_utils.find_components

_BG = 0
_COLOR_POOL = tuple(range(1, 10))

# Generator ranges and retry bounds.
_INNER_H_RANGE = (3, 7)
_INNER_W_RANGE = (3, 7)
_COLOR_PLACEMENT_MAX = 200
_TRIALS_PER_FRAME = 80
_FRAME_AND_MARKERS_TRIES = 80


def _frame_inner_layers(inner_h: int, inner_w: int) -> int:
    return min((int(inner_h) + 1) // 2, (int(inner_w) + 1) // 2)


def _is_hollow_rect_frame(g: np.ndarray, comp: dict) -> bool:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    color = int(comp["color"])

    if int(comp["area"]) <= 4:
        return False

    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False

    if int(comp["area"]) != 2 * (h + w) - 4:
        return False

    inner = g[r0 : r1 + 1, c0 : c1 + 1]
    if not np.all(inner[0, :] == color):
        return False
    if not np.all(inner[-1, :] == color):
        return False
    if not np.all(inner[:, 0] == color):
        return False
    if not np.all(inner[:, -1] == color):
        return False

    inner_h = inner[1:-1, 1:-1]
    if inner_h.size and np.any(inner_h == color):
        return False

    return True


def _detect_frames(g: np.ndarray, bg: int) -> list[dict]:
    components = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    frames: list[dict] = []
    for comp in components:
        if not _is_hollow_rect_frame(g, comp):
            continue

        r0, c0, r1, c1 = map(int, comp["bbox"])
        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        if inner_h <= 0 or inner_w <= 0:
            continue

        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "inner_h": inner_h,
                "inner_w": inner_w,
                "rings": _frame_inner_layers(inner_h, inner_w),
                "color": int(comp["color"]),
            }
        )

    return frames


def _axis_is_bg(
    g: np.ndarray,
    bg: int,
    r: int,
    c: int,
    dr: int,
    dc: int,
    steps: int,
) -> bool:
    for k in range(1, int(steps) + 1):
        rr = int(r + dr * k)
        cc = int(c + dc * k)
        if g[rr, cc] != int(bg):
            return False
    return True


def _marker_distance_for_frame(g: np.ndarray, bg: int, r: int, c: int, frame: dict) -> int | None:
    r0, c0, r1, c1 = frame["bbox"]

    if c0 <= c <= c1 and r < r0:
        d = r0 - r - 1
        if d > 0 and _axis_is_bg(g, bg, r, c, 1, 0, d):
            return int(d)

    if c0 <= c <= c1 and r > r1:
        d = r - r1 - 1
        if d > 0 and _axis_is_bg(g, bg, r, c, -1, 0, d):
            return int(d)

    if r0 <= r <= r1 and c < c0:
        d = c0 - c - 1
        if d > 0 and _axis_is_bg(g, bg, r, c, 0, 1, d):
            return int(d)

    if r0 <= r <= r1 and c > c1:
        d = c - c1 - 1
        if d > 0 and _axis_is_bg(g, bg, r, c, 0, -1, d):
            return int(d)

    return None


def _marker_frame_matches(g: np.ndarray, bg: int, r: int, c: int, frames: list[dict]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for idx, frame in enumerate(frames):
        d = _marker_distance_for_frame(g, bg, r, c, frame)
        if d is not None:
            out.append((idx, d))
    return out


def _ring_depth(ir: int, ic: int, h: int, w: int) -> int:
    return min(int(ir) + 1, int(ic) + 1, int(h) - int(ir), int(w) - int(ic))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill concentric interior rings of each empty rectangular frame according to the sightline distance of surrounding marker pixels.

    Input structure:
    - Background is the dominant color.
    - The grid contains exactly two one-pixel-thick hollow rectangular frames.
    - Marker pixels lie outside the frames, each in the same row or column as exactly one frame, with only background in-between.
    - Markers are one pixel each and for each frame they induce distances 1..N where N is the number of concentric ring layers inside that frame.

    Rule:
    - Detect both hollow frames and keep all frame pixels unchanged.
    - Assign each marker to its visible frame and distance d to the frame wall (background cells between marker and wall).
    - Paint the interior ring at depth d (outermost is d=1) with the marker color inside that frame.
    - Let the innermost remaining region after all outer rings be painted by its corresponding deepest ring distance.

    Key insight:
    - Ring depth is determined by geometric distance from the frame wall, not by marker order or color identity; row/column line-of-sight gives the frame and layer index.
    """

    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = int(get_bg(g))

    frames = _detect_frames(g, bg)
    assert len(frames) == 2, f"expected two hollow frames, found {len(frames)}"

    frame_mask = np.zeros_like(g, dtype=bool)
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        frame_mask[r0, c0 : c1 + 1] = True
        frame_mask[r1, c0 : c1 + 1] = True
        frame_mask[r0 : r1 + 1, c0] = True
        frame_mask[r0 : r1 + 1, c1] = True

    assignments: list[dict[int, int]] = [dict() for _ in frames]
    for r, c in np.argwhere(g != bg):
        rr, cc = int(r), int(c)
        if frame_mask[rr, cc]:
            continue

        color = int(g[rr, cc])
        matches = _marker_frame_matches(g, bg, rr, cc, frames)
        assert len(matches) == 1, "each marker must match exactly one frame by row/column sightline"
        frame_idx, depth = matches[0]
        assert 1 <= depth <= frames[frame_idx]["rings"], "marker depth must match frame inner ring count"

        assigned = assignments[frame_idx]
        assert depth not in assigned, "marker distances for one frame must be unique"
        assigned[depth] = color

    for frame_idx, frame in enumerate(frames):
        needed = list(range(1, frame["rings"] + 1))
        assigned = assignments[frame_idx]
        assert len(assigned) == len(needed), "each frame must supply every ring depth"
        assert sorted(assigned) == needed, "frame marker depths must be exactly 1..N"

    out = g.copy()
    for frame_idx, frame in enumerate(frames):
        colors_by_depth = assignments[frame_idx]
        r0, c0, r1, c1 = frame["bbox"]
        inner_h = frame["inner_h"]
        inner_w = frame["inner_w"]

        base_r = r0 + 1
        base_c = c0 + 1
        for ir in range(inner_h):
            for ic in range(inner_w):
                depth = _ring_depth(ir, ic, inner_h, inner_w)
                out[base_r + ir, base_c + ic] = colors_by_depth[depth]

    return out


def _rects_touch(rect_a: tuple[int, int, int, int], rect_b: tuple[int, int, int, int], margin: int) -> bool:
    a_r0, a_c0, a_r1, a_c1 = rect_a
    b_r0, b_c0, b_r1, b_c1 = rect_b
    if a_r1 + margin < b_r0:
        return False
    if b_r1 + margin < a_r0:
        return False
    if a_c1 + margin < b_c0:
        return False
    if b_c1 + margin < a_c0:
        return False
    return True


def _place_two_frames(
    H: int,
    W: int,
    outer_a: tuple[int, int],
    outer_b: tuple[int, int],
    margin: int,
) -> tuple[int, int, int, int, int, int, int, int] | None:
    oh_a, ow_a = map(int, outer_a)
    oh_b, ow_b = map(int, outer_b)

    if random.random() < 0.55:
        min_h = oh_a + oh_b + 2 * margin + 1
        if H < min_h:
            return None

        a_r0_min = margin
        a_r0_max = H - oh_a - oh_b - margin - 1
        if a_r0_min > a_r0_max:
            return None

        a_c0_min = margin
        a_c0_max = W - ow_a - margin
        b_c0_min = margin
        b_c0_max = W - ow_b - margin
        if a_c0_min > a_c0_max or b_c0_min > b_c0_max:
            return None

        for _ in range(_TRIALS_PER_FRAME):
            a_r0 = random.randint(a_r0_min, a_r0_max)
            b_r0_lo = a_r0 + oh_a + margin
            b_r0_hi = H - oh_b - margin - 1
            if b_r0_lo > b_r0_hi:
                continue
            b_r0 = random.randint(b_r0_lo, b_r0_hi)

            if b_r0 <= a_r0 + oh_a + margin - 1:
                continue

            a_c0 = random.randint(a_c0_min, a_c0_max)
            b_c0 = random.randint(b_c0_min, b_c0_max)

            a_r1 = a_r0 + oh_a - 1
            a_c1 = a_c0 + ow_a - 1
            b_r1 = b_r0 + oh_b - 1
            b_c1 = b_c0 + ow_b - 1

            a_rect = (a_r0, a_c0, a_r1, a_c1)
            b_rect = (b_r0, b_c0, b_r1, b_c1)
            if _rects_touch(a_rect, b_rect, margin=margin):
                continue
            return (a_r0, a_c0, a_r1, a_c1, b_r0, b_c0, b_r1, b_c1)

    min_w = ow_a + ow_b + 2 * margin + 1
    if W < min_w:
        return None

    a_c0_min = margin
    a_c0_max = W - ow_a - ow_b - margin - 1
    if a_c0_min > a_c0_max:
        return None

    a_r0_min = margin
    a_r0_max = H - oh_a - margin
    b_r0_min = margin
    b_r0_max = H - oh_b - margin
    if a_r0_min > a_r0_max or b_r0_min > b_r0_max:
        return None

    for _ in range(_TRIALS_PER_FRAME):
        a_c0 = random.randint(a_c0_min, a_c0_max)
        b_c0_lo = a_c0 + ow_a + margin
        b_c0_hi = W - ow_b - margin - 1
        if b_c0_lo > b_c0_hi:
            continue
        b_c0 = random.randint(b_c0_lo, b_c0_hi)

        if b_c0 <= a_c0 + ow_a + margin - 1:
            continue

        a_r0 = random.randint(a_r0_min, a_r0_max)
        b_r0 = random.randint(b_r0_min, b_r0_max)

        a_r1 = a_r0 + oh_a - 1
        a_c1 = a_c0 + ow_a - 1
        b_r1 = b_r0 + oh_b - 1
        b_c1 = b_c0 + ow_b - 1

        a_rect = (a_r0, a_c0, a_r1, a_c1)
        b_rect = (b_r0, b_c0, b_r1, b_c1)
        if _rects_touch(a_rect, b_rect, margin=margin):
            continue
        return (a_r0, a_c0, a_r1, a_c1, b_r0, b_c0, b_r1, b_c1)

    return None


def _draw_hollow_frame(canvas: np.ndarray, top_left: tuple[int, int], inner_h: int, inner_w: int, color: int) -> tuple[int, int, int, int]:
    r0, c0 = map(int, top_left)
    r1 = r0 + int(inner_h) + 1
    c1 = c0 + int(inner_w) + 1

    canvas[r0, c0 : c1 + 1] = color
    canvas[r1, c0 : c1 + 1] = color
    canvas[r0 : r1 + 1, c0] = color
    canvas[r0 : r1 + 1, c1] = color

    return (r0, c0, r1, c1)


def _axis_candidates(
    g: np.ndarray,
    bg: int,
    frame: dict,
    d: int,
    sides: tuple[str, ...],
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = frame["bbox"]
    h, w = g.shape
    out: list[tuple[int, int]] = []

    if "up" in sides:
        mr = r0 - int(d) - 1
        if 0 <= mr < h:
            for mc in range(c0, c1 + 1):
                if g[mr, mc] == bg and _axis_is_bg(g, bg, mr, mc, 1, 0, d):
                    out.append((mr, mc))

    if "down" in sides:
        mr = r1 + int(d) + 1
        if 0 <= mr < h:
            for mc in range(c0, c1 + 1):
                if g[mr, mc] == bg and _axis_is_bg(g, bg, mr, mc, -1, 0, d):
                    out.append((mr, mc))

    if "left" in sides:
        mc = c0 - int(d) - 1
        if 0 <= mc < w:
            for mr in range(r0, r1 + 1):
                if g[mr, mc] == bg and _axis_is_bg(g, bg, mr, mc, 0, 1, d):
                    out.append((mr, mc))

    if "right" in sides:
        mc = c1 + int(d) + 1
        if 0 <= mc < w:
            for mr in range(r0, r1 + 1):
                if g[mr, mc] == bg and _axis_is_bg(g, bg, mr, mc, 0, -1, d):
                    out.append((mr, mc))

    return out


def _place_frame_markers(
    g: np.ndarray,
    bg: int,
    frame: dict,
    frame_idx: int,
    frames: list[dict],
    marker_colors: list[int],
) -> bool:
    rings = frame["rings"]

    mode = random.random()
    if mode < 0.35:
        preferred: tuple[str, ...] = ("up", "down")
    elif mode < 0.70:
        preferred = ("left", "right")
    else:
        preferred = ("up", "down", "left", "right")

    for depth in range(1, rings + 1):
        candidates = _axis_candidates(g, bg, frame, depth, preferred)
        if not candidates:
            candidates = _axis_candidates(g, bg, frame, depth, ("up", "down", "left", "right"))
        if not candidates:
            return False

        valid_candidates: list[tuple[int, int]] = []
        for r, c in candidates:
            if _marker_frame_matches(g, bg, r, c, frames) == [(frame_idx, depth)]:
                valid_candidates.append((r, c))

        if not valid_candidates:
            return False

        r, c = random.choice(valid_candidates)
        g[r, c] = int(marker_colors[depth - 1])

    return True


def _build_candidate() -> np.ndarray | None:
    for _ in range(_FRAME_AND_MARKERS_TRIES):
        inner_h_a = random.randint(*_INNER_H_RANGE)
        inner_w_a = random.randint(*_INNER_W_RANGE)
        inner_h_b = random.randint(*_INNER_H_RANGE)
        inner_w_b = random.randint(*_INNER_W_RANGE)

        outer_h_a = inner_h_a + 2
        outer_w_a = inner_w_a + 2
        outer_h_b = inner_h_b + 2
        outer_w_b = inner_w_b + 2

        rings_a = _frame_inner_layers(inner_h_a, inner_w_a)
        rings_b = _frame_inner_layers(inner_h_b, inner_w_b)
        edge_margin = 1 + max(rings_a, rings_b)
        if edge_margin < 3:
            edge_margin = 3

        H = random.randint(max(outer_h_a, outer_h_b) + 2 * edge_margin + 3, 30)
        W = random.randint(max(outer_w_a, outer_w_b) + 2 * edge_margin + 3, 30)

        if H > 30 or W > 30:
            H = 30
            W = 30

        if H < outer_h_a + 2 * edge_margin or W < outer_w_a + 2 * edge_margin:
            continue

        place = _place_two_frames(H, W, (outer_h_a, outer_w_a), (outer_h_b, outer_w_b), margin=edge_margin)
        if place is None:
            continue

        a_r0, a_c0, a_r1, a_c1, b_r0, b_c0, b_r1, b_c1 = place

        if random.random() < 0.4:
            frame_color_a = random.choice(_COLOR_POOL)
            frame_color_b = frame_color_a
        else:
            frame_color_a, frame_color_b = random.sample(_COLOR_POOL, 2)

        marker_palette = [int(c) for c in _COLOR_POOL if c not in {frame_color_a, frame_color_b}]
        if len(marker_palette) < rings_a + rings_b:
            marker_palette = [int(c) for c in _COLOR_POOL if c != _BG]
            extra = rings_a + rings_b - len(marker_palette)
            marker_palette = marker_palette + [int(random.choice(marker_palette)) for _ in range(extra)]
            random.shuffle(marker_palette)
        else:
            random.shuffle(marker_palette)

        canvas = np.full((H, W), _BG, dtype=int)

        frame_a_bbox = _draw_hollow_frame(canvas, (a_r0, a_c0), inner_h_a, inner_w_a, frame_color_a)
        frame_b_bbox = _draw_hollow_frame(canvas, (b_r0, b_c0), inner_h_b, inner_w_b, frame_color_b)

        frame_a = {
            "bbox": frame_a_bbox,
            "rings": rings_a,
            "inner_h": inner_h_a,
            "inner_w": inner_w_a,
            "color": frame_color_a,
        }
        frame_b = {
            "bbox": frame_b_bbox,
            "rings": rings_b,
            "inner_h": inner_h_b,
            "inner_w": inner_w_b,
            "color": frame_color_b,
        }

        frames = [frame_a, frame_b]

        colors_a = marker_palette[:rings_a]
        colors_b = marker_palette[rings_a : rings_a + rings_b]

        if not _place_frame_markers(canvas, _BG, frame_a, 0, frames, colors_a):
            continue
        if not _place_frame_markers(canvas, _BG, frame_b, 1, frames, colors_b):
            continue

        try:
            out = main(canvas)
        except Exception:
            continue
        if np.array_equal(out, canvas):
            continue
        return canvas

    return None


def generate_input() -> np.ndarray:
    for _ in range(_COLOR_PLACEMENT_MAX):
        candidate = _build_candidate()
        if candidate is not None:
            return candidate

    raise RuntimeError("failed to generate valid input for fable/3/014/4")
