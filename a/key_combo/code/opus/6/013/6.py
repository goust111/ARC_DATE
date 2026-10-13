from __future__ import annotations

from collections import deque
from pathlib import Path
import random
import sys

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import colors_present, draw_line, find_components, get_bg


BG = 0
SPRITE_COLOR = 2
FRAME_COLORS = (1, 3, 4, 6, 7, 8, 9)
ANCHORS = ("top_left", "top_right", "bottom_left", "bottom_right")
OPENING_TO_ANCHOR = {
    "top_left": "bottom_right",
    "top_right": "bottom_left",
    "bottom_left": "top_right",
    "bottom_right": "top_left",
}
LOCAL_TO_OPENING = {
    frozenset({(0, 1), (1, 0), (1, 1)}): "top_left",
    frozenset({(0, 0), (1, 0), (1, 1)}): "top_right",
    frozenset({(0, 0), (0, 1), (1, 1)}): "bottom_left",
    frozenset({(0, 0), (0, 1), (1, 0)}): "bottom_right",
}
MARKER_MASKS = {
    opening: np.array(
        [[1 if (r, c) in cells else 0 for c in range(2)] for r in range(2)],
        dtype=bool,
    )
    for cells, opening in ((coords, opening) for coords, opening in LOCAL_TO_OPENING.items())
}

GEN_MAX_ATTEMPTS = 400
GRID_SIDE_RANGE = (18, 25)
FRAME_COUNT_OPTIONS = (2, 3)
INTERIOR_MIN = 4
INTERIOR_MAX = 8
FRAME_GAP = 4
OBJECT_GAP = 2
SPRITE_GAP = 2
SPRITE_AREA_RANGE = (4, 7)
SPRITE_BBOX_SIDE_RANGE = (2, 4)


def _rect_frame_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return None
    expected = np.zeros_like(mask, dtype=bool)
    expected[r0 : r1 + 1, c0] = True
    expected[r0 : r1 + 1, c1] = True
    expected[r0, c0 : c1 + 1] = True
    expected[r1, c0 : c1 + 1] = True
    if not np.array_equal(mask, expected):
        return None
    return int(r0), int(c0), int(r1), int(c1)


def _decode_marker(mask: np.ndarray) -> tuple[str, str] | None:
    coords = np.argwhere(mask)
    if coords.shape != (3, 2):
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    if (int(r1 - r0), int(c1 - c0)) != (1, 1):
        return None
    local = frozenset((int(r) - int(r0), int(c) - int(c0)) for r, c in coords)
    opening = LOCAL_TO_OPENING.get(local)
    if opening is None:
        return None
    return opening, OPENING_TO_ANCHOR[opening]


def _axis_map(index: int, src_len: int, dst_len: int, anchor_at_zero: bool) -> int:
    if src_len <= 1:
        return 0 if anchor_at_zero else dst_len - 1
    if anchor_at_zero:
        dist = int(index)
        mapped = int(round(dist * (dst_len - 1) / (src_len - 1)))
        return mapped
    dist = int(src_len - 1 - index)
    mapped = int(round(dist * (dst_len - 1) / (src_len - 1)))
    return int(dst_len - 1 - mapped)


def _fill_enclosed(stroke: np.ndarray) -> np.ndarray:
    h, w = stroke.shape
    open_space = ~stroke.astype(bool)
    padded = np.pad(open_space, 1, constant_values=True)
    seen = np.zeros_like(padded, dtype=bool)
    q: deque[tuple[int, int]] = deque()
    ph, pw = padded.shape

    for r in range(ph):
        for c in (0, pw - 1):
            if padded[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    for c in range(pw):
        for r in (0, ph - 1):
            if padded[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < ph and 0 <= nc < pw and padded[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))

    holes = open_space & ~seen[1 : h + 1, 1 : w + 1]
    return stroke.astype(bool) | holes


def _stretch_sprite(sprite: np.ndarray, target_shape: tuple[int, int], anchor: str) -> np.ndarray:
    src_h, src_w = sprite.shape
    dst_h, dst_w = target_shape
    assert src_h >= 1 and src_w >= 1 and dst_h >= 1 and dst_w >= 1

    row_anchor_zero = anchor.startswith("top")
    col_anchor_zero = anchor.endswith("left")
    coords = [tuple(map(int, rc)) for rc in np.argwhere(sprite)]
    mapped: dict[tuple[int, int], tuple[int, int]] = {}
    stroke = np.zeros((dst_h, dst_w), dtype=int)

    for sr, sc in coords:
        tr = _axis_map(sr, src_h, dst_h, row_anchor_zero)
        tc = _axis_map(sc, src_w, dst_w, col_anchor_zero)
        mapped[(sr, sc)] = (tr, tc)
        stroke[tr, tc] = 1

    coord_set = set(coords)
    for sr, sc in coords:
        for nr, nc in ((sr + 1, sc), (sr, sc + 1)):
            if (nr, nc) not in coord_set:
                continue
            r0, c0 = mapped[(sr, sc)]
            r1, c1 = mapped[(nr, nc)]
            draw_line(stroke, r0, c0, end_r=r1, end_c=c1, color=1)

    return _fill_enclosed(stroke)


def _extract_scene(
    g: np.ndarray,
) -> tuple[np.ndarray, list[dict], np.ndarray, np.ndarray]:
    bg = int(get_bg(g))
    assert bg == BG, "expected black background"
    out = np.asarray(g, dtype=int)
    colors = colors_present(out, exclude={bg})
    assert SPRITE_COLOR in colors, "expected one red source sprite"

    sprite_comps = find_components(out, bg=bg, colors=[SPRITE_COLOR], connectivity=4, return_format="dict")
    assert len(sprite_comps) == 1, "expected a single red 4-connected source sprite"
    sprite_coords = sprite_comps[0]["coords"]
    sr0, sc0, sr1, sc1 = sprite_comps[0]["bbox"]
    sprite = out[sr0 : sr1 + 1, sc0 : sc1 + 1] == SPRITE_COLOR

    frame_infos: list[dict] = []
    used_colors: set[int] = set()
    for color in colors:
        if int(color) == SPRITE_COLOR:
            continue
        comps = find_components(out, bg=bg, colors=[int(color)], connectivity=4, return_format="dict")
        assert len(comps) == 2, "each frame color must form exactly one frame and one marker"
        frame_comp = None
        marker_comp = None
        for comp in comps:
            mask = np.zeros_like(out, dtype=bool)
            mask[comp["coords"][:, 0], comp["coords"][:, 1]] = True
            frame_bbox = _rect_frame_bbox(mask)
            marker_info = _decode_marker(mask)
            if frame_bbox is not None:
                assert frame_comp is None, "frame color must have one rectangular border"
                frame_comp = (comp, frame_bbox)
            if marker_info is not None:
                assert marker_comp is None, "frame color must have one 3-pixel L marker"
                marker_comp = (comp, marker_info)
        assert frame_comp is not None and marker_comp is not None
        comp, bbox = frame_comp
        marker, (opening, anchor) = marker_comp
        fr0, fc0, fr1, fc1 = bbox
        assert fr1 - fr0 >= 2 and fc1 - fc0 >= 2
        interior_h = fr1 - fr0 - 1
        interior_w = fc1 - fc0 - 1
        assert interior_h >= 1 and interior_w >= 1
        assert int(color) not in used_colors
        used_colors.add(int(color))
        frame_infos.append(
            {
                "color": int(color),
                "bbox": bbox,
                "frame_coords": comp["coords"],
                "marker_coords": marker["coords"],
                "opening": opening,
                "anchor": anchor,
                "interior_shape": (interior_h, interior_w),
            }
        )

    assert 2 <= len(frame_infos) <= 3, "expected 2-3 frame/marker pairs"
    assert len({info["opening"] for info in frame_infos}) >= 2, "need at least two L orientations"
    return out, frame_infos, sprite, sprite_coords


def _render_solution(g: np.ndarray, frame_infos: list[dict], sprite: np.ndarray, sprite_coords: np.ndarray) -> np.ndarray:
    out = np.array(g, copy=True)
    out[sprite_coords[:, 0], sprite_coords[:, 1]] = BG
    for info in frame_infos:
        out[info["marker_coords"][:, 0], info["marker_coords"][:, 1]] = BG
        fr0, fc0, fr1, fc1 = info["bbox"]
        interior = _stretch_sprite(sprite, (fr1 - fr0 - 1, fc1 - fc0 - 1), info["anchor"])
        out[fr0 + 1 : fr1, fc0 + 1 : fc1][interior] = SPRITE_COLOR
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stretch the shared red sprite into each hollow frame, anchoring the mapping at the corner selected by that frame's L marker.

    Input structure:
    - The background is black.
    - There are 2-3 one-pixel-thick hollow rectangular frames, each with a unique non-red color.
    - Near each frame is a same-color 3-pixel L marker; elsewhere there is one red 4-connected source sprite shared by every frame.

    Rule:
    - Pair each L marker with the same-color frame.
    - Read the marker's opening corner and choose the opposite frame-interior corner as the stretch anchor.
    - Map the red sprite's tight bbox into each frame interior by aligning the chosen anchor corner and stretching the opposite bbox corner to the opposite interior corner.
    - Connect mapped positions of 4-neighbor sprite pixels by horizontal or vertical segments, fill enclosed background regions, keep the frame border, and erase the original markers and source sprite.

    Key insight:
    - The L marker does not say where to draw an L; it selects which interior corner stays fixed while the same source sprite is stretched into the frame.
    """
    g, frame_infos, sprite, sprite_coords = _extract_scene(np.asarray(grid, dtype=int))
    return _render_solution(g, frame_infos, sprite, sprite_coords)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(mask)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _is_connected(mask: np.ndarray) -> bool:
    comps = find_components(mask.astype(int), bg=0, colors=[1], connectivity=4, return_format="dict")
    return len(comps) == 1


def _random_sprite() -> np.ndarray | None:
    h = random.randint(*SPRITE_BBOX_SIDE_RANGE)
    w = random.randint(*SPRITE_BBOX_SIDE_RANGE)
    max_area = min(SPRITE_AREA_RANGE[1], h * w)
    if max_area < SPRITE_AREA_RANGE[0]:
        return None
    target = random.randint(SPRITE_AREA_RANGE[0], max_area)
    mask = np.zeros((h, w), dtype=bool)
    start = (random.randrange(h), random.randrange(w))
    mask[start] = True

    while int(mask.sum()) < target:
        frontier = set()
        for r, c in np.argwhere(mask):
            for nr, nc in ((int(r) - 1, int(c)), (int(r) + 1, int(c)), (int(r), int(c) - 1), (int(r), int(c) + 1)):
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.add((nr, nc))
        if not frontier:
            return None
        nr, nc = random.choice(tuple(frontier))
        mask[nr, nc] = True

    mask = _crop_mask(mask)
    hh, ww = mask.shape
    if hh < 2 or ww < 2:
        return None
    if np.all(mask):
        return None
    if np.array_equal(mask, np.fliplr(mask)):
        return None
    if np.array_equal(mask, np.flipud(mask)):
        return None
    if np.array_equal(mask, np.rot90(mask, 2)):
        return None
    if not _is_connected(mask):
        return None
    return mask


def _sample_orientations(k: int) -> list[str]:
    while True:
        openings = [random.choice(ANCHORS) for _ in range(k)]
        if len(set(openings)) >= 2:
            return openings


def _sample_frame_shapes(k: int, sprite_shape: tuple[int, int]) -> list[tuple[int, int]]:
    sh, sw = sprite_shape
    pairs: list[tuple[int, int]] = []
    for _ in range(k):
        ih = random.randint(max(INTERIOR_MIN, sh + 1), INTERIOR_MAX)
        iw = random.randint(max(INTERIOR_MIN, sw + 1), INTERIOR_MAX)
        pairs.append((ih, iw))
    if not any(h != w for h, w in pairs):
        idx = random.randrange(k)
        h, w = pairs[idx]
        if h == w:
            if w < INTERIOR_MAX:
                w += 1
            else:
                h -= 1
        pairs[idx] = (h, w)
    return pairs


def _boxes_separate(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _place_frames(shape: tuple[int, int], interiors: list[tuple[int, int]]) -> list[dict] | None:
    h, w = shape
    specs = []
    for ih, iw in sorted(interiors, key=lambda x: x[0] * x[1], reverse=True):
        oh = ih + 2
        ow = iw + 2
        candidates = []
        for r0 in range(h - oh + 1):
            for c0 in range(w - ow + 1):
                bbox = (r0, c0, r0 + oh - 1, c0 + ow - 1)
                if all(_boxes_separate(bbox, other["bbox"], FRAME_GAP) for other in specs):
                    candidates.append((r0, c0))
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        specs.append(
            {
                "bbox": (r0, c0, r0 + oh - 1, c0 + ow - 1),
                "interior_shape": (ih, iw),
            }
        )
    return specs


def _frame_pixels(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    pixels = []
    for r in range(r0, r1 + 1):
        pixels.append((r, c0))
        pixels.append((r, c1))
    for c in range(c0 + 1, c1):
        pixels.append((r0, c))
        pixels.append((r1, c))
    return pixels


def _min_manhattan(points_a: list[tuple[int, int]], points_b: list[tuple[int, int]]) -> int:
    return min(abs(ar - br) + abs(ac - bc) for ar, ac in points_a for br, bc in points_b)


def _enumerate_marker_positions(
    grid_shape: tuple[int, int],
    frame_bbox: tuple[int, int, int, int],
    opening: str,
    occupied: np.ndarray,
) -> list[tuple[int, int]]:
    h, w = grid_shape
    frame_cells = _frame_pixels(frame_bbox)
    patch = MARKER_MASKS[opening]
    coords = [tuple(map(int, rc)) for rc in np.argwhere(patch)]
    candidates: list[tuple[int, int]] = []

    for r0 in range(h - 1):
        for c0 in range(w - 1):
            pts = [(r0 + dr, c0 + dc) for dr, dc in coords]
            if any(occupied[r, c] for r, c in pts):
                continue
            if any(frame_bbox[0] <= r <= frame_bbox[2] and frame_bbox[1] <= c <= frame_bbox[3] for r, c in pts):
                continue
            if _min_manhattan(pts, frame_cells) not in (2, 3):
                continue
            too_close = False
            for r, c in pts:
                r_lo = max(0, r - 1)
                r_hi = min(h, r + 2)
                c_lo = max(0, c - 1)
                c_hi = min(w, c + 2)
                neighborhood = occupied[r_lo:r_hi, c_lo:c_hi].copy()
                for fr, fc in frame_cells:
                    if r_lo <= fr < r_hi and c_lo <= fc < c_hi:
                        neighborhood[fr - r_lo, fc - c_lo] = False
                if neighborhood.any():
                    too_close = True
                    break
            if too_close:
                continue
            candidates.append((r0, c0))
    return candidates


def _place_sprite(
    grid_shape: tuple[int, int],
    sprite_mask: np.ndarray,
    occupied: np.ndarray,
    frame_bboxes: list[tuple[int, int, int, int]],
) -> tuple[int, int] | None:
    h, w = grid_shape
    sh, sw = sprite_mask.shape
    sprite_pts = [tuple(map(int, rc)) for rc in np.argwhere(sprite_mask)]
    candidates = []
    for r0 in range(h - sh + 1):
        for c0 in range(w - sw + 1):
            pts = [(r0 + dr, c0 + dc) for dr, dc in sprite_pts]
            if any(occupied[r, c] for r, c in pts):
                continue
            bbox = (r0, c0, r0 + sh - 1, c0 + sw - 1)
            if any(not _boxes_separate(bbox, frame_bbox, 0) for frame_bbox in frame_bboxes):
                continue
            too_close = False
            for r, c in pts:
                r_lo = max(0, r - 1)
                r_hi = min(h, r + 2)
                c_lo = max(0, c - 1)
                c_hi = min(w, c + 2)
                if occupied[r_lo:r_hi, c_lo:c_hi].any():
                    too_close = True
                    break
            if too_close:
                continue
            candidates.append((r0, c0))
    return random.choice(candidates) if candidates else None


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color


def _paint_patch(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    coords = np.argwhere(mask)
    for dr, dc in coords:
        grid[r0 + int(dr), c0 + int(dc)] = int(color)


def _semantic_ok(sprite: np.ndarray, frames: list[dict], expected: np.ndarray) -> bool:
    if np.count_nonzero(expected == SPRITE_COLOR) <= int(sprite.sum()):
        return False

    interiors_by_frame = []
    for frame in frames:
        ih, iw = frame["interior_shape"]
        actual = _stretch_sprite(sprite, (ih, iw), frame["anchor"])
        if not actual.any():
            return False
        wrongs = [_stretch_sprite(sprite, (ih, iw), other) for other in ANCHORS if other != frame["anchor"]]
        if not any(not np.array_equal(actual, wrong) for wrong in wrongs):
            return False
        interiors_by_frame.append(actual.astype(np.uint8).tobytes() + bytes((ih, iw)))

    if len(set(interiors_by_frame)) < 2:
        return False
    return True


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    sprite = _random_sprite()
    if sprite is None:
        return None
    k = random.choice(FRAME_COUNT_OPTIONS)
    openings = _sample_orientations(k)
    colors = random.sample(FRAME_COLORS, k)

    grid_h = random.randint(*GRID_SIDE_RANGE)
    grid_w = random.randint(*GRID_SIDE_RANGE)
    frame_shapes = _sample_frame_shapes(k, sprite.shape)
    frames = _place_frames((grid_h, grid_w), frame_shapes)
    if frames is None:
        return None

    random.shuffle(frames)
    for frame, opening, color in zip(frames, openings, colors):
        frame["opening"] = opening
        frame["anchor"] = OPENING_TO_ANCHOR[opening]
        frame["color"] = color

    inp = np.full((grid_h, grid_w), BG, dtype=int)
    occupied = np.zeros_like(inp, dtype=bool)
    for frame in frames:
        _draw_frame(inp, frame["bbox"], frame["color"])
        for r, c in _frame_pixels(frame["bbox"]):
            occupied[r, c] = True

    for frame in frames:
        pos = _enumerate_marker_positions(inp.shape, frame["bbox"], frame["opening"], occupied)
        if not pos:
            return None
        mr, mc = random.choice(pos)
        frame["marker_top_left"] = (mr, mc)
        _paint_patch(inp, MARKER_MASKS[frame["opening"]], mr, mc, frame["color"])
        for dr, dc in np.argwhere(MARKER_MASKS[frame["opening"]]):
            occupied[mr + int(dr), mc + int(dc)] = True

    sprite_pos = _place_sprite(inp.shape, sprite, occupied, [frame["bbox"] for frame in frames])
    if sprite_pos is None:
        return None
    sr, sc = sprite_pos
    _paint_patch(inp, sprite, sr, sc, SPRITE_COLOR)

    expected = np.array(inp, copy=True)
    expected[sr : sr + sprite.shape[0], sc : sc + sprite.shape[1]][sprite] = BG
    for frame in frames:
        mr, mc = frame["marker_top_left"]
        expected[mr : mr + 2, mc : mc + 2][MARKER_MASKS[frame["opening"]]] = BG
        r0, c0, r1, c1 = frame["bbox"]
        interior = _stretch_sprite(sprite, frame["interior_shape"], frame["anchor"])
        expected[r0 + 1 : r1, c0 + 1 : c1][interior] = SPRITE_COLOR

    if not _semantic_ok(sprite, frames, expected):
        return None

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        built = _build_case()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid anchored-stretch ARC input")
