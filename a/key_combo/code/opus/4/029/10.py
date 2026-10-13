from __future__ import annotations

import random
import sys
from collections import deque
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, bbox_h, bbox_w, draw_line, find_components, get_bg


BG = Color.BLACK
BLUE = Color.BLUE
RED = Color.RED
GRAY = Color.GRAY
ORANGE = Color.ORANGE
TEAL = Color.TEAL

GRID_MIN = 16
GRID_MAX = 22
RECT_H_RANGE = (4, 6)
RECT_W_RANGE = (5, 7)
INTERIOR_H_RANGE = (4, 6)
INTERIOR_W_RANGE = (4, 6)
MAX_RETRIES = 600

_SPRITE_SHAPES = [
    np.array([[1, 1], [1, 1], [0, 1]], dtype=bool),
    np.array([[1, 1, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 0], [1, 1, 1], [0, 0, 1]], dtype=bool),
]
_ANCHORS = ("tl", "tr", "bl", "br")


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


def _corner_points(inner_bbox: tuple[int, int, int, int]) -> dict[str, tuple[int, int]]:
    r0, c0, r1, c1 = inner_bbox
    return {
        "tl": (int(r0), int(c0)),
        "tr": (int(r0), int(c1)),
        "bl": (int(r1), int(c0)),
        "br": (int(r1), int(c1)),
    }


def _nearest_corner(point: tuple[int, int], inner_bbox: tuple[int, int, int, int]) -> str:
    pr, pc = point
    corners = _corner_points(inner_bbox)
    l1 = {name: abs(pr - r) + abs(pc - c) for name, (r, c) in corners.items()}
    d2 = {name: (pr - r) ** 2 + (pc - c) ** 2 for name, (r, c) in corners.items()}
    best_l1 = [name for name, dist in l1.items() if dist == min(l1.values())]
    best_d2 = [name for name, dist in d2.items() if dist == min(d2.values())]
    assert len(best_l1) == 1 and len(best_d2) == 1 and best_l1[0] == best_d2[0]
    return best_l1[0]


def _scaled_index(i: int, src_len: int, dst_len: int) -> int:
    assert src_len >= 1 and dst_len >= 1
    if src_len == 1:
        return 0
    return int(round(int(i) * (dst_len - 1) / (src_len - 1)))


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
                q.append((int(r), int(c)))
    for c in range(pw):
        for r in (0, ph - 1):
            if padded[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((int(r), int(c)))

    while q:
        r, c = q.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < ph and 0 <= nc < pw and padded[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((int(nr), int(nc)))

    holes = open_space & ~seen[1 : h + 1, 1 : w + 1]
    return stroke.astype(bool) | holes


def _stretch_sprite(sprite: np.ndarray, target_shape: tuple[int, int], anchor: str, *, fill: bool) -> np.ndarray:
    src_h, src_w = sprite.shape
    dst_h, dst_w = target_shape
    assert sprite.any() and src_h >= 2 and src_w >= 2
    reverse_r = anchor[0] == "b"
    reverse_c = anchor[1] == "r"

    stroke = np.zeros((dst_h, dst_w), dtype=int)
    coords = np.argwhere(sprite)
    mapped: dict[tuple[int, int], tuple[int, int]] = {}
    for sr_raw, sc_raw in coords:
        sr = int(sr_raw)
        sc = int(sc_raw)
        tr = _scaled_index(sr, src_h, dst_h)
        tc = _scaled_index(sc, src_w, dst_w)
        if reverse_r:
            tr = dst_h - 1 - tr
        if reverse_c:
            tc = dst_w - 1 - tc
        mapped[(sr, sc)] = (int(tr), int(tc))
        stroke[tr, tc] = 1

    for sr_raw, sc_raw in coords:
        sr = int(sr_raw)
        sc = int(sc_raw)
        for nr, nc in ((sr + 1, sc), (sr, sc + 1)):
            if (nr, nc) not in mapped:
                continue
            r0, c0 = mapped[(sr, sc)]
            r1, c1 = mapped[(nr, nc)]
            draw_line(stroke, r0, c0, end_r=r1, end_c=c1, color=1)

    return _fill_enclosed(stroke.astype(bool)) if fill else stroke.astype(bool)


def _bbox_separated(b1: tuple[int, int, int, int], b2: tuple[int, int, int, int], gap: int = 1) -> bool:
    return (
        b1[2] + gap < b2[0]
        or b2[2] + gap < b1[0]
        or b1[3] + gap < b2[1]
        or b2[3] + gap < b1[1]
    )


def _paint_mask(grid: np.ndarray, top: int, left: int, mask: np.ndarray, color: int) -> None:
    h, w = mask.shape
    sub = grid[top : top + h, left : left + w]
    sub[mask] = int(color)


def _random_sprite_mask() -> np.ndarray:
    mask = random.choice(_SPRITE_SHAPES).copy()
    mask = np.rot90(mask, random.randint(0, 3))
    if random.random() < 0.5:
        mask = np.fliplr(mask)
    if random.random() < 0.5:
        mask = np.flipud(mask)
    return np.ascontiguousarray(mask, dtype=bool)


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert get_bg(g) == BG

    gray_components = find_components(g, bg=BG, colors=[GRAY], connectivity=4)
    rects = [comp for comp in gray_components if int(comp["area"]) == bbox_h(comp["bbox"]) * bbox_w(comp["bbox"])]
    assert len(rects) == 1, "expected one solid gray rectangle"
    rect = rects[0]
    rr0, rc0, rr1, rc1 = map(int, rect["bbox"])

    red_pts = np.argwhere(g == RED)
    teal_pts = np.argwhere(g == TEAL)
    assert red_pts.shape == (1, 2) and teal_pts.shape == (1, 2), "expected one red and one teal marker"
    red = tuple(map(int, red_pts[0]))
    teal = tuple(map(int, teal_pts[0]))
    assert red[0] == rr0 - 1 and teal[0] == rr0 - 1
    assert rc0 <= red[1] <= rc1 and rc0 <= teal[1] <= rc1 and red[1] != teal[1]

    frame_components = find_components(g, bg=BG, colors=[ORANGE], connectivity=4)
    frames = []
    for comp in frame_components:
        mask = np.zeros_like(g, dtype=bool)
        coords = comp["coords"]
        mask[coords[:, 0], coords[:, 1]] = True
        frame_bbox = _rect_frame_bbox(mask)
        if frame_bbox is not None:
            frames.append((comp, frame_bbox))
    assert len(frames) == 1, "expected one orange frame"
    frame_comp, frame_bbox = frames[0]
    fr0, fc0, fr1, fc1 = frame_bbox
    inner_bbox = (fr0 + 1, fc0 + 1, fr1 - 1, fc1 - 1)
    assert inner_bbox[0] <= inner_bbox[2] and inner_bbox[1] <= inner_bbox[3]
    assert np.all(g[inner_bbox[0] : inner_bbox[2] + 1, inner_bbox[1] : inner_bbox[3] + 1] == BG)

    sprite_components = find_components(g, bg=BG, colors=[BLUE], connectivity=4)
    assert len(sprite_components) == 1, "expected one blue sprite"
    sprite_comp = sprite_components[0]
    sr0, sc0, sr1, sc1 = map(int, sprite_comp["bbox"])
    sprite_patch = np.zeros((sr1 - sr0 + 1, sc1 - sc0 + 1), dtype=bool)
    rel = sprite_comp["coords"] - np.array([[sr0, sc0]])
    sprite_patch[rel[:, 0], rel[:, 1]] = True
    assert sprite_patch.any() and sprite_patch.shape[0] >= 2 and sprite_patch.shape[1] >= 2

    frame_region = np.zeros_like(g, dtype=bool)
    frame_region[fr0 : fr1 + 1, fc0 : fc1 + 1] = True
    assert not np.any((g == BLUE) & frame_region), "sprite must lie outside the frame"

    return {
        "grid": g,
        "rect_bbox": (rr0, rc0, rr1, rc1),
        "red": red,
        "teal": teal,
        "frame_bbox": frame_bbox,
        "inner_bbox": inner_bbox,
        "sprite_coords": sprite_comp["coords"],
        "sprite_patch": sprite_patch,
    }


def _render(scene: dict, exit_side: str, anchor_point: tuple[int, int]) -> np.ndarray:
    g = scene["grid"]
    rr0, rc0, rr1, rc1 = scene["rect_bbox"]
    inner_bbox = scene["inner_bbox"]
    ir0, ic0, ir1, ic1 = inner_bbox
    out = g.copy()

    anchor = _nearest_corner(anchor_point, inner_bbox)
    stretched = _stretch_sprite(
        scene["sprite_patch"],
        (ir1 - ir0 + 1, ic1 - ic0 + 1),
        anchor,
        fill=True,
    )

    out[scene["red"]] = BG
    out[scene["teal"]] = RED
    out[scene["sprite_coords"][:, 0], scene["sprite_coords"][:, 1]] = BG

    new_teal = (rr1 + 1, rc1 if exit_side == "right" else rc0)
    out[new_teal] = TEAL
    interior = out[ir0 : ir1 + 1, ic0 : ic1 + 1]
    interior[stretched] = BLUE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Use the red-cyan order to place a teal exit marker, then anchor a stretched blue sprite into the orange frame.

    Input structure:
    - The background is black.
    - One solid gray rectangle has exactly one red pixel and one teal pixel on the row immediately above it.
    - Elsewhere there is one orange one-cell-thick hollow rectangle with an empty black interior, plus one blue 4-connected sprite outside that frame.

    Rule:
    - Compare the marker columns: if red is left of teal, the exit side is the gray rectangle's right side; otherwise it is the left side.
    - Erase the original red pixel, recolor the original teal pixel to red, and place a new teal pixel one row below the gray rectangle at the exit-side corner column.
    - Choose the unique nearest inner corner of the orange frame to that new teal pixel.
    - Clear the original blue sprite, stretch its bbox into the frame interior with that corner acting as the anchor orientation, draw mapped 4-neighbor segments, and flood-fill any enclosed interior hole blue.
    - Leave the orange frame unchanged.

    Key insight:
    - The red/teal horizontal order is not the final output by itself; it first selects an exit corner below the gray rectangle, and that displaced teal marker then determines which anchored mirror of the stretched sprite appears inside the distant frame.
    """
    scene = _parse_scene(np.asarray(grid, dtype=int))
    exit_side = "right" if scene["red"][1] < scene["teal"][1] else "left"
    rr0, rc0, rr1, rc1 = scene["rect_bbox"]
    anchor_point = (rr1 + 1, rc1 if exit_side == "right" else rc0)
    return _render(scene, exit_side, anchor_point)


def _case_is_valid(grid: np.ndarray) -> bool:
    try:
        scene = _parse_scene(grid)
        rr0, rc0, rr1, rc1 = scene["rect_bbox"]
        exit_side = "right" if scene["red"][1] < scene["teal"][1] else "left"
        actual_point = (rr1 + 1, rc1 if exit_side == "right" else rc0)
        other_point = (rr1 + 1, rc0 if exit_side == "right" else rc1)
        actual_anchor = _nearest_corner(actual_point, scene["inner_bbox"])
        other_anchor = _nearest_corner(other_point, scene["inner_bbox"])
        original_anchor = _nearest_corner(scene["teal"], scene["inner_bbox"])
        if actual_anchor == other_anchor or actual_anchor == original_anchor:
            return False

        ir0, ic0, ir1, ic1 = scene["inner_bbox"]
        target_shape = (ir1 - ir0 + 1, ic1 - ic0 + 1)
        if target_shape == scene["sprite_patch"].shape:
            return False

        stroke = _stretch_sprite(scene["sprite_patch"], target_shape, actual_anchor, fill=False)
        filled = _stretch_sprite(scene["sprite_patch"], target_shape, actual_anchor, fill=True)
        if np.array_equal(stroke, filled):
            return False
        if np.array_equal(filled, _stretch_sprite(scene["sprite_patch"], target_shape, other_anchor, fill=True)):
            return False
        if np.array_equal(filled, _stretch_sprite(scene["sprite_patch"], target_shape, original_anchor, fill=True)):
            return False

        actual = main(grid)
        other = _render(scene, "left" if exit_side == "right" else "right", other_point)
        wrong_anchor = _render(scene, exit_side, scene["teal"])
        if np.array_equal(actual, other) or np.array_equal(actual, wrong_anchor):
            return False
        if np.array_equal(actual, grid):
            return False
        return True
    except Exception:
        return False


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    grid = np.full((h, w), BG, dtype=int)

    rect_h = random.randint(*RECT_H_RANGE)
    rect_w = random.randint(*RECT_W_RANGE)
    rect_r0 = random.randint(2, h - rect_h - 3)
    rect_c0 = random.randint(2, w - rect_w - 3)
    rect_r1 = rect_r0 + rect_h - 1
    rect_c1 = rect_c0 + rect_w - 1
    rect_bbox = (rect_r0, rect_c0, rect_r1, rect_c1)
    grid[rect_r0 : rect_r1 + 1, rect_c0 : rect_c1 + 1] = GRAY

    exit_side = random.choice(("left", "right"))
    cols = random.sample(range(rect_c0, rect_c1 + 1), 2)
    left_col, right_col = sorted(cols)
    if exit_side == "right":
        red_col, teal_col = left_col, right_col
    else:
        red_col, teal_col = right_col, left_col
    red = (rect_r0 - 1, red_col)
    teal = (rect_r0 - 1, teal_col)
    grid[red] = RED
    grid[teal] = TEAL

    actual_point = (rect_r1 + 1, rect_c1 if exit_side == "right" else rect_c0)
    other_point = (rect_r1 + 1, rect_c0 if exit_side == "right" else rect_c1)

    sprite_mask = _random_sprite_mask()
    src_h, src_w = sprite_mask.shape
    interior_h = random.randint(max(src_h, INTERIOR_H_RANGE[0]), INTERIOR_H_RANGE[1])
    interior_w = random.randint(max(src_w, INTERIOR_W_RANGE[0]), INTERIOR_W_RANGE[1])
    if interior_h == src_h and interior_w == src_w:
        if interior_h < INTERIOR_H_RANGE[1]:
            interior_h += 1
        elif interior_w < INTERIOR_W_RANGE[1]:
            interior_w += 1
        else:
            return None
    frame_h = interior_h + 2
    frame_w = interior_w + 2

    desired_anchor = random.choice(_ANCHORS)
    frame_candidates = []
    marker_cluster_bbox = (rect_r0 - 1, rect_c0, rect_r1, rect_c1)
    for fr0 in range(0, h - frame_h + 1):
        for fc0 in range(0, w - frame_w + 1):
            frame_bbox = (fr0, fc0, fr0 + frame_h - 1, fc0 + frame_w - 1)
            if not _bbox_separated(frame_bbox, marker_cluster_bbox, gap=2):
                continue
            inner_bbox = (fr0 + 1, fc0 + 1, fr0 + interior_h, fc0 + interior_w)
            try:
                actual_anchor = _nearest_corner(actual_point, inner_bbox)
                other_anchor = _nearest_corner(other_point, inner_bbox)
                original_anchor = _nearest_corner(teal, inner_bbox)
            except AssertionError:
                continue
            if actual_anchor != desired_anchor or actual_anchor == other_anchor or actual_anchor == original_anchor:
                continue
            frame_candidates.append((frame_bbox, inner_bbox))
    if not frame_candidates:
        return None
    frame_bbox, inner_bbox = random.choice(frame_candidates)
    fr0, fc0, fr1, fc1 = frame_bbox
    grid[fr0 : fr1 + 1, fc0] = ORANGE
    grid[fr0 : fr1 + 1, fc1] = ORANGE
    grid[fr0, fc0 : fc1 + 1] = ORANGE
    grid[fr1, fc0 : fc1 + 1] = ORANGE

    sprite_candidates = []
    for sr0 in range(0, h - src_h + 1):
        for sc0 in range(0, w - src_w + 1):
            sprite_bbox = (sr0, sc0, sr0 + src_h - 1, sc0 + src_w - 1)
            if not _bbox_separated(sprite_bbox, marker_cluster_bbox, gap=2):
                continue
            if not _bbox_separated(sprite_bbox, frame_bbox, gap=2):
                continue
            sprite_candidates.append((sr0, sc0))
    if not sprite_candidates:
        return None
    sr0, sc0 = random.choice(sprite_candidates)
    _paint_mask(grid, sr0, sc0, sprite_mask, BLUE)

    if not _case_is_valid(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid anchored-stretch ARC input")
