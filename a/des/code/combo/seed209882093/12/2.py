import random
from typing import Iterable

import numpy as np


BG = 0
MARKER = 9
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

GEN_RETRIES = 400
PLACEMENT_ATTEMPTS = 120
FRAME_COUNT_RANGE = (2, 4)
FRAME_COLORS = (1, 2, 3, 4, 5, 6, 7, 8)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    pts = list(points)
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    return min(rows), min(cols), max(rows), max(cols)


def _is_hollow_rectangle(points: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(points)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False
    border = {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if r in (r0, r1) or c in (c0, c1)
    }
    return set(points) == border


def _find_frames(grid: np.ndarray) -> dict[int, dict[str, object]]:
    frames: dict[int, dict[str, object]] = {}
    for color in sorted(int(v) for v in np.unique(grid) if int(v) not in (BG, MARKER)):
        frame_bboxes = []
        for comp in _components(grid == color):
            if _is_hollow_rectangle(comp):
                frame_bboxes.append(_bbox(comp))
        if len(frame_bboxes) != 1:
            continue
        r0, c0, r1, c1 = frame_bboxes[0]
        anchors = [
            (r, c)
            for r in range(r0 + 1, r1)
            for c in range(c0 + 1, c1)
            if int(grid[r, c]) == color
        ]
        if not (1 <= len(anchors) <= 2):
            continue
        frames[color] = {"bbox": frame_bboxes[0], "anchors": sorted(anchors)}
    return frames


def _find_fragments(grid: np.ndarray) -> dict[int, dict[str, object]]:
    fragments: dict[int, dict[str, object]] = {}
    for comp in _components(grid != BG):
        colors = {int(grid[r, c]) for r, c in comp}
        if MARKER not in colors:
            continue
        body_colors = colors - {MARKER}
        if len(body_colors) != 1:
            continue
        color = next(iter(body_colors))
        marker_cells = [(r, c) for r, c in comp if int(grid[r, c]) == MARKER]
        if len(marker_cells) != 1:
            continue
        fragments[color] = {"points": comp, "marker": marker_cells[0]}
    return fragments


def _translated(points: Iterable[tuple[int, int]], dr: int, dc: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for r, c in points]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each colored fragment into the matching hollow frame by aligning its maroon marker to a frame anchor.

    Frames are hollow 1-pixel rectangles. Each frame contains 1-2 same-color anchor pixels in its interior.
    Each fragment is a 4-connected mixed-color component made of one body color plus one maroon marker pixel.
    The fragment is translated, not rotated, so that its marker lands on the matching frame anchor that keeps
    the whole fragment inside the frame interior. The marker then recolors to the body color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    frames = _find_frames(g)
    fragments = _find_fragments(g)

    out = g.copy()

    for color, frame in frames.items():
        if color not in fragments:
            continue
        frag = fragments[color]
        marker_r, marker_c = frag["marker"]
        r0, c0, r1, c1 = frame["bbox"]
        placement = None
        for anchor_r, anchor_c in frame["anchors"]:
            shift_r = anchor_r - marker_r
            shift_c = anchor_c - marker_c
            moved = _translated(frag["points"], shift_r, shift_c)
            if all(r0 < r < r1 and c0 < c < c1 for r, c in moved):
                placement = moved
                break
        if placement is None:
            continue
        for r, c in frag["points"]:
            out[r, c] = BG
        for r, c in placement:
            out[r, c] = color

    return out


def _random_shape(rng: random.Random) -> tuple[list[tuple[int, int]], tuple[int, int], tuple[int, int]]:
    target_area = rng.randint(3, 6)
    for _ in range(200):
        pts = {(0, 0)}
        while len(pts) < target_area:
            frontier = []
            for r, c in pts:
                for dr, dc in NEIGHBORS4:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) in pts or abs(nr) > 3 or abs(nc) > 3:
                        continue
                    frontier.append((nr, nc))
            if not frontier:
                break
            pts.add(rng.choice(frontier))
        if len(pts) != target_area:
            continue
        r0, c0, r1, c1 = _bbox(pts)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h > 4 or w > 4:
            continue
        norm = sorted((r - r0, c - c0) for r, c in pts)
        marker = rng.choice(norm)
        return norm, marker, (h, w)
    raise RuntimeError("failed to sample a connected fragment")


def _expand_bbox(rect: tuple[int, int, int, int], pad: int, h: int, w: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = rect
    return max(0, r0 - pad), max(0, c0 - pad), min(h - 1, r1 + pad), min(w - 1, c1 + pad)


def _can_place(used: np.ndarray, rect: tuple[int, int, int, int], pad: int) -> bool:
    h, w = used.shape
    r0, c0, r1, c1 = _expand_bbox(rect, pad, h, w)
    return not np.any(used[r0 : r1 + 1, c0 : c1 + 1])


def _reserve(used: np.ndarray, rect: tuple[int, int, int, int], pad: int) -> None:
    h, w = used.shape
    r0, c0, r1, c1 = _expand_bbox(rect, pad, h, w)
    used[r0 : r1 + 1, c0 : c1 + 1] = True


def _place_rect(
    rng: random.Random,
    used: np.ndarray,
    height: int,
    width: int,
    pad: int,
) -> tuple[int, int] | None:
    h, w = used.shape
    if h - height - 1 < 1 or w - width - 1 < 1:
        return None
    for _ in range(PLACEMENT_ATTEMPTS):
        top = rng.randint(1, h - height - 1)
        left = rng.randint(1, w - width - 1)
        rect = (top, left, top + height - 1, left + width - 1)
        if _can_place(used, rect, pad):
            _reserve(used, rect, pad)
            return top, left
    return None


def _valid_anchor_cells(
    interior_h: int,
    interior_w: int,
    shape_h: int,
    shape_w: int,
    marker: tuple[int, int],
) -> set[tuple[int, int]]:
    mr, mc = marker
    cells = set()
    for top in range(interior_h - shape_h + 1):
        for left in range(interior_w - shape_w + 1):
            cells.add((top + mr, left + mc))
    return cells


def _build_pair_spec(rng: random.Random, color: int) -> dict[str, object]:
    shape_pts, marker, (shape_h, shape_w) = _random_shape(rng)

    interior_h = rng.randint(shape_h, min(shape_h + 4, 8))
    interior_w = rng.randint(shape_w, min(shape_w + 4, 8))
    outer_h = interior_h + 2
    outer_w = interior_w + 2

    top = rng.randint(0, interior_h - shape_h)
    left = rng.randint(0, interior_w - shape_w)
    anchor_local = (top + marker[0], left + marker[1])

    placed_cells_local = {(top + r, left + c) for r, c in shape_pts}
    valid_anchors = _valid_anchor_cells(interior_h, interior_w, shape_h, shape_w, marker)
    invalid_anchors = [
        (r, c)
        for r in range(interior_h)
        for c in range(interior_w)
        if (r, c) not in valid_anchors and (r, c) not in placed_cells_local
    ]

    anchors_local = [anchor_local]
    if invalid_anchors and rng.random() < 0.45:
        anchors_local.append(rng.choice(invalid_anchors))
    anchors_local = sorted(anchors_local)

    return {
        "color": color,
        "shape_pts": shape_pts,
        "marker_local": marker,
        "shape_h": shape_h,
        "shape_w": shape_w,
        "outer_h": outer_h,
        "outer_w": outer_w,
        "interior_h": interior_h,
        "interior_w": interior_w,
        "target_top": top,
        "target_left": left,
        "anchors_local": anchors_local,
    }


def _draw_frame(grid: np.ndarray, top: int, left: int, spec: dict[str, object]) -> None:
    color = int(spec["color"])
    outer_h = int(spec["outer_h"])
    outer_w = int(spec["outer_w"])
    bottom = top + outer_h - 1
    right = left + outer_w - 1
    grid[top, left : right + 1] = color
    grid[bottom, left : right + 1] = color
    grid[top : bottom + 1, left] = color
    grid[top : bottom + 1, right] = color
    for ar, ac in spec["anchors_local"]:
        grid[top + 1 + ar, left + 1 + ac] = color


def _draw_fragment(grid: np.ndarray, top: int, left: int, spec: dict[str, object]) -> list[tuple[int, int]]:
    color = int(spec["color"])
    marker = tuple(spec["marker_local"])
    cells: list[tuple[int, int]] = []
    for r, c in spec["shape_pts"]:
        rr = top + r
        cc = left + c
        grid[rr, cc] = MARKER if (r, c) == marker else color
        cells.append((rr, cc))
    return cells


def _expected_output(grid: np.ndarray, specs: list[dict[str, object]]) -> np.ndarray:
    out = grid.copy()
    for spec in specs:
        for r, c in spec["fragment_cells_abs"]:
            out[r, c] = BG
        color = int(spec["color"])
        frame_top = int(spec["frame_top"])
        frame_left = int(spec["frame_left"])
        target_top = int(spec["target_top"])
        target_left = int(spec["target_left"])
        for r, c in spec["shape_pts"]:
            rr = frame_top + 1 + target_top + r
            cc = frame_left + 1 + target_left + c
            out[rr, cc] = color
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    pair_count = rng.randint(*FRAME_COUNT_RANGE)
    min_side = 12 + 4 * pair_count
    height = rng.randint(min_side, min_side + 6)
    width = rng.randint(min_side, min_side + 8)
    if height < 8 or width < 8:
        return None

    colors = rng.sample(FRAME_COLORS, pair_count)
    specs = [_build_pair_spec(rng, color) for color in colors]

    used = np.zeros((height, width), dtype=bool)
    for spec in specs:
        frame_pos = _place_rect(rng, used, int(spec["outer_h"]), int(spec["outer_w"]), pad=1)
        if frame_pos is None:
            return None
        spec["frame_top"], spec["frame_left"] = frame_pos

    for spec in specs:
        frag_pos = _place_rect(rng, used, int(spec["shape_h"]), int(spec["shape_w"]), pad=1)
        if frag_pos is None:
            return None
        spec["fragment_top"], spec["fragment_left"] = frag_pos

    grid = np.zeros((height, width), dtype=int)
    for spec in specs:
        _draw_frame(grid, int(spec["frame_top"]), int(spec["frame_left"]), spec)
    for spec in specs:
        spec["fragment_cells_abs"] = _draw_fragment(
            grid,
            int(spec["fragment_top"]),
            int(spec["fragment_left"]),
            spec,
        )

    expected = _expected_output(grid, specs)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Randomly generate a valid puzzle instance for the anchor-alignment fragment task."""
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
