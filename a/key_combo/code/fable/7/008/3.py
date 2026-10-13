import random
import sys
from pathlib import Path

import numpy as np


_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import find_components, get_bg


BG = 0
PALETTE = tuple(range(1, 10))
SIDES = ("top", "right", "bottom", "left")
MAX_ATTEMPTS = 400


def _is_hollow_rect(comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    coords = {tuple(map(int, pt)) for pt in comp["coords"]}
    if len(coords) != 2 * (h + w) - 4:
        return False
    for r, c in coords:
        if r not in (r0, r1) and c not in (c0, c1):
            return False
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            on_border = r in (r0, r1) or c in (c0, c1)
            if on_border != ((r, c) in coords):
                return False
    return True


def _marker_and_frames(grid: np.ndarray) -> tuple[tuple[int, int, int], list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True)
    marker = None
    frames = []
    h, w = g.shape

    for comp in comps:
        if comp["area"] == 1:
            r, c = map(int, comp["coords"][0])
            if (r in (0, h - 1)) ^ (c in (0, w - 1)):
                if marker is not None:
                    raise AssertionError("expected exactly one edge marker")
                marker = (r, c, int(comp["color"]))
                continue
        if _is_hollow_rect(comp):
            frames.append(comp)
            continue
        raise AssertionError("unexpected foreground object")

    if marker is None or not frames:
        raise AssertionError("missing marker or frames")
    return marker, frames


def _crossed_frames(
    shape: tuple[int, int], marker: tuple[int, int, int], frames: list[dict]
) -> list[dict]:
    h, w = shape
    r, c, _color = marker
    if r == 0:
        crossed = [f for f in frames if f["bbox"][1] < c < f["bbox"][3]]
        return sorted(crossed, key=lambda f: f["bbox"][0])
    if r == h - 1:
        crossed = [f for f in frames if f["bbox"][1] < c < f["bbox"][3]]
        return sorted(crossed, key=lambda f: f["bbox"][2], reverse=True)
    if c == 0:
        crossed = [f for f in frames if f["bbox"][0] < r < f["bbox"][2]]
        return sorted(crossed, key=lambda f: f["bbox"][1])
    if c == w - 1:
        crossed = [f for f in frames if f["bbox"][0] < r < f["bbox"][2]]
        return sorted(crossed, key=lambda f: f["bbox"][3], reverse=True)
    raise AssertionError("marker must lie on exactly one edge")


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Fill each ray-crossed hollow rectangle interior with the previous station's color.

    Input structure:
    - A black background contains exactly one single-pixel colored marker on one grid edge.
    - The foreground also contains several one-cell-thick hollow rectangles with distinct border colors.
    - Some rectangles lie across the invisible orthogonal ray launched from the edge marker, while others are offset away from that ray.

    Rule:
    - Reconstruct the ray that starts at the edge marker and travels straight inward perpendicular to that edge.
    - Among hollow rectangles whose border is crossed by that ray, process them in geometric traversal order from the marker inward.
    - Fill the first crossed rectangle interior with the marker color, then fill each later crossed rectangle interior with the border color of the previously crossed rectangle; keep all borders, the marker, and all off-ray rectangles unchanged.

    Key insight:
    - The ray itself is never drawn, so the only way to solve the task is to infer which hollow rectangles the edge-to-interior line crosses and then relay colors along that ordered chain.
    """
    g = np.asarray(input_grid, dtype=int)
    marker, frames = _marker_and_frames(g)
    crossed = _crossed_frames(g.shape, marker, frames)

    out = g.copy()
    relay = int(marker[2])
    for frame in crossed:
        r0, c0, r1, c1 = frame["bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = relay
        relay = int(frame["color"])
    return out


def _draw_hollow_rect(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _fill_inside(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0 + 1 : r1, c0 + 1 : c1] = color


def _mark_forbidden(forbidden: np.ndarray, box: tuple[int, int, int, int]) -> None:
    h, w = forbidden.shape
    r0, c0, r1, c1 = box
    forbidden[max(0, r0 - 1) : min(h, r1 + 2), max(0, c0 - 1) : min(w, c1 + 2)] = True


def _clear_box(forbidden: np.ndarray, box: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = box
    return not bool(forbidden[r0 : r1 + 1, c0 : c1 + 1].any())


def _sample_hit_rows(hit_count: int) -> tuple[int, list[tuple[int, int]]]:
    for _ in range(200):
        heights = [random.randint(3, 6) for _ in range(hit_count)]
        gaps = [random.randint(1, 4)] + [random.randint(1, 3) for _ in range(hit_count - 1)] + [random.randint(1, 4)]
        height = 1 + gaps[0] + sum(heights) + sum(gaps[1:])
        if 12 <= height <= 28:
            rows = []
            r0 = 1 + gaps[0]
            for idx, rect_h in enumerate(heights):
                rows.append((r0, r0 + rect_h - 1))
                r0 += rect_h + gaps[idx + 1]
            return height, rows
    raise RuntimeError("failed to sample hit rows")


def _random_distractor_box(
    height: int,
    width: int,
    ray_c: int,
    forbidden: np.ndarray,
) -> tuple[int, int, int, int] | None:
    for _ in range(120):
        rect_h = random.randint(3, min(7, height - 2))
        rect_w = random.randint(3, min(10, width - 2))
        r0 = random.randint(1, height - rect_h - 1)
        r1 = r0 + rect_h - 1
        near_path = random.random() < 0.45
        candidate_cols = []
        if near_path:
            left_c0 = ray_c - rect_w
            if left_c0 >= 1:
                candidate_cols.append(left_c0)
            right_c0 = ray_c + 1
            if right_c0 + rect_w - 1 <= width - 2:
                candidate_cols.append(right_c0)
            if not candidate_cols:
                near_path = False
        if not near_path:
            c0 = random.randint(1, width - rect_w - 1)
            c1 = c0 + rect_w - 1
            if c0 < ray_c < c1:
                continue
            candidate_cols = [c0]
        random.shuffle(candidate_cols)
        for c0 in candidate_cols:
            c1 = c0 + rect_w - 1
            if c0 < ray_c < c1:
                continue
            box = (r0, c0, r1, c1)
            if _clear_box(forbidden, box):
                return box
    return None


def _build_canonical_case() -> tuple[np.ndarray, np.ndarray]:
    hit_count = random.choices([1, 2, 3, 4], weights=[1, 4, 4, 3], k=1)[0]
    distractor_count = random.choices([0, 1, 2, 3], weights=[1, 3, 4, 3], k=1)[0]
    color_count = 1 + hit_count + distractor_count
    marker_color, *frame_colors = random.sample(PALETTE, color_count)
    hit_colors = frame_colors[:hit_count]
    distractor_colors = frame_colors[hit_count:]

    for _ in range(200):
        height, row_spans = _sample_hit_rows(hit_count)
        width = random.randint(12, 26)
        ray_c = random.randint(3, width - 4)

        hit_boxes = []
        for r0, r1 in row_spans:
            options = []
            for rect_w in range(3, min(10, width - 2) + 1):
                c_min = max(1, ray_c - rect_w + 2)
                c_max = min(ray_c - 1, width - rect_w - 1)
                if c_min > c_max:
                    continue
                for c0 in range(c_min, c_max + 1):
                    options.append((r0, c0, r1, c0 + rect_w - 1))
            if not options:
                break
            hit_boxes.append(random.choice(options))
        if len(hit_boxes) != hit_count:
            continue

        inp = np.full((height, width), BG, dtype=int)
        out = inp.copy()
        inp[0, ray_c] = marker_color
        out[0, ray_c] = marker_color

        forbidden = np.zeros_like(inp, dtype=bool)
        _mark_forbidden(forbidden, (0, ray_c, 0, ray_c))
        for box, color in zip(hit_boxes, hit_colors):
            if not _clear_box(forbidden, box):
                break
            _draw_hollow_rect(inp, box, color)
            _draw_hollow_rect(out, box, color)
            _mark_forbidden(forbidden, box)
        else:
            distractor_boxes = []
            for _idx in range(distractor_count):
                box = _random_distractor_box(height, width, ray_c, forbidden)
                if box is None:
                    break
                distractor_boxes.append(box)
                _mark_forbidden(forbidden, box)
            if len(distractor_boxes) != distractor_count:
                continue

            for box, color in zip(distractor_boxes, distractor_colors):
                _draw_hollow_rect(inp, box, color)
                _draw_hollow_rect(out, box, color)

            relay = marker_color
            for box, color in zip(hit_boxes, hit_colors):
                _fill_inside(out, box, relay)
                relay = color
            return inp, out

    raise RuntimeError("failed to build canonical case")


def _rotate_case(inp: np.ndarray, out: np.ndarray, side: str) -> tuple[np.ndarray, np.ndarray]:
    if side == "top":
        return inp, out
    if side == "right":
        return np.rot90(inp, -1).copy(), np.rot90(out, -1).copy()
    if side == "bottom":
        return np.rot90(inp, 2).copy(), np.rot90(out, 2).copy()
    if side == "left":
        return np.rot90(inp, 1).copy(), np.rot90(out, 1).copy()
    raise ValueError("invalid side")


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        side = random.choice(SIDES)
        inp, expected = _build_canonical_case()
        rotated_inp, rotated_expected = _rotate_case(inp, expected, side)
        try:
            actual = main(rotated_inp)
        except Exception:
            continue
        if np.array_equal(actual, rotated_expected):
            return rotated_inp
    raise RuntimeError("failed to generate valid relay-ray input")
