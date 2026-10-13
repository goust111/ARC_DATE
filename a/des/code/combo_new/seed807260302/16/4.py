import random
from dataclasses import dataclass

import numpy as np


BG = 0
ARC_MAX = 30
RECT_COUNT_RANGE = (2, 5)
RECT_SIDE_MIN = 2
RECT_SIDE_MAX = 10
GENERATION_RETRIES = 400


@dataclass(frozen=True)
class Rect:
    top: int
    left: int
    height: int
    width: int
    color: int

    @property
    def bottom(self) -> int:
        return self.top + self.height - 1

    @property
    def right(self) -> int:
        return self.left + self.width - 1

    @property
    def area(self) -> int:
        return self.height * self.width


def _bbox_for_color(grid: np.ndarray, color: int) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(grid == color)
    if coords.size == 0:
        return None
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    return int(mins[0]), int(mins[1]), int(maxs[0]), int(maxs[1])


def _render(rects: list[Rect], shape: tuple[int, int]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for rect in rects:
        grid[rect.top : rect.top + rect.height, rect.left : rect.left + rect.width] = rect.color
    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """Restore the hidden solid rectangles and pack them left to right by area.

    Input structure:
    - Black background.
    - Each non-black color belongs to one solid axis-aligned rectangle.
    - Later rectangles may partially cover earlier ones, so some colors remain
      visible only as clipped remnants.
    - Every foreground color still has at least one visible pixel.

    Rule:
    - For each non-black color, find the min/max visible row and column.
    - Reconstruct that color as the full solid rectangle spanning that bounding box.
    - Sort the reconstructed rectangles by area from small to large; if areas tie,
      use shape and source position as deterministic structure-based tie-breakers.
    - Place the rectangles horizontally with one black column between neighbors
      and align their top edges to row 0.

    Key insight:
    - Occlusion changes which pixels survive, but the surviving pixels still reveal
      the rectangle's original height and width through their row and column spans.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    rect_specs: list[tuple[int, int, int, int, int, int]] = []
    for color in (int(v) for v in np.unique(g) if int(v) != BG):
        bbox = _bbox_for_color(g, color)
        if bbox is None:
            continue
        r0, c0, r1, c1 = bbox
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        rect_specs.append((height * width, height, width, r0, c0, color))

    assert rect_specs
    ordered = sorted(rect_specs, key=lambda item: (item[0], item[1], item[2], item[3], item[4]))

    out_h = max(item[1] for item in ordered)
    out_w = sum(item[2] for item in ordered) + len(ordered) - 1
    assert out_h <= ARC_MAX and out_w <= ARC_MAX

    out = np.full((out_h, out_w), BG, dtype=int)
    col = 0
    for _area, height, width, _r0, _c0, color in ordered:
        out[:height, col : col + width] = color
        col += width + 1
    return out


def _sample_dimensions(n_rects: int) -> list[tuple[int, int]] | None:
    for _ in range(200):
        remaining_width = ARC_MAX - (n_rects - 1)
        widths: list[int] = []
        for i in range(n_rects):
            rects_left = n_rects - i - 1
            max_width = min(RECT_SIDE_MAX, remaining_width - RECT_SIDE_MIN * rects_left)
            if max_width < RECT_SIDE_MIN:
                widths = []
                break
            widths.append(random.randint(RECT_SIDE_MIN, max_width))
            remaining_width -= widths[-1]
        if not widths:
            continue

        random.shuffle(widths)
        dims: list[tuple[int, int]] = []
        used_areas: set[int] = set()
        ok = True
        for width in widths:
            heights = [h for h in range(RECT_SIDE_MIN, RECT_SIDE_MAX + 1) if h * width not in used_areas]
            if not heights:
                ok = False
                break
            height = random.choice(heights)
            dims.append((height, width))
            used_areas.add(height * width)
        if ok:
            random.shuffle(dims)
            return dims
    return None


def _sample_canvas(dims: list[tuple[int, int]]) -> tuple[int, int]:
    max_h = max(h for h, _w in dims)
    max_w = max(w for _h, w in dims)
    total_h = sum(h for h, _w in dims)
    total_w = sum(w for _h, w in dims)

    min_h = max(max_h + 1, min(ARC_MAX, total_h // 3 + 4), 10)
    min_w = max(max_w + 1, min(ARC_MAX, total_w // 2 + 4), 10)
    max_h_choice = min(ARC_MAX, min_h + 8)
    max_w_choice = min(ARC_MAX, min_w + 8)
    return random.randint(min_h, max_h_choice), random.randint(min_w, max_w_choice)


def _sample_modes(n_rects: int) -> list[str]:
    modes = ["either"]
    choices = ("overlap", "separate", "either")
    for _ in range(n_rects - 1):
        modes.append(random.choice(choices))
    if "overlap" not in modes[1:]:
        modes[random.randint(1, n_rects - 1)] = "overlap"
    return modes


def _witness_cells(rect: Rect) -> set[tuple[int, int]]:
    cells = {
        (rect.top, rect.left + random.randrange(rect.width)),
        (rect.bottom, rect.left + random.randrange(rect.width)),
        (rect.top + random.randrange(rect.height), rect.left),
        (rect.top + random.randrange(rect.height), rect.right),
    }
    if rect.height >= 3 and rect.width >= 3 and random.random() < 0.6:
        cells.add(
            (
                rect.top + random.randrange(1, rect.height - 1),
                rect.left + random.randrange(1, rect.width - 1),
            )
        )
    return cells


def _candidate_positions(canvas: np.ndarray, protected: np.ndarray, h: int, w: int, mode: str) -> list[tuple[int, int, int]]:
    H, W = canvas.shape
    area = h * w
    primary: list[tuple[int, int, int]] = []
    fallback: list[tuple[int, int, int]] = []
    max_overlap = max(2, int(area * 0.6))

    for top in range(H - h + 1):
        for left in range(W - w + 1):
            if np.any(protected[top : top + h, left : left + w]):
                continue
            overlap = int(np.count_nonzero(canvas[top : top + h, left : left + w]))

            if mode == "overlap":
                if overlap == 0:
                    continue
                fallback.append((top, left, overlap))
                if 2 <= overlap <= max_overlap:
                    primary.append((top, left, overlap))
            elif mode == "separate":
                if overlap == 0:
                    primary.append((top, left, overlap))
            else:
                primary.append((top, left, overlap))

    if mode == "overlap":
        return primary or fallback
    return primary


def _pick_candidate(candidates: list[tuple[int, int, int]], mode: str, area: int) -> tuple[int, int]:
    if mode == "overlap":
        target = max(1, area // 3)
        ranked = sorted(candidates, key=lambda item: abs(item[2] - target))
        shortlist = ranked[: min(12, len(ranked))]
        top, left, _overlap = random.choice(shortlist)
        return top, left

    if mode == "either":
        overlapping = [item for item in candidates if item[2] > 0]
        separate = [item for item in candidates if item[2] == 0]
        if overlapping and separate:
            pool = overlapping if random.random() < 0.5 else separate
            top, left, _overlap = random.choice(pool)
            return top, left

    top, left, _overlap = random.choice(candidates)
    return top, left


def _place_rectangles(
    dims: list[tuple[int, int]],
    colors: list[int],
    modes: list[str],
    shape: tuple[int, int],
) -> list[Rect] | None:
    canvas = np.full(shape, BG, dtype=int)
    protected = np.zeros(shape, dtype=bool)
    rects: list[Rect] = []

    for (height, width), color, mode in zip(dims, colors, modes):
        candidates = _candidate_positions(canvas, protected, height, width, mode)
        if not candidates:
            return None
        top, left = _pick_candidate(candidates, mode, height * width)
        rect = Rect(top=top, left=left, height=height, width=width, color=color)
        rects.append(rect)
        for rr, cc in _witness_cells(rect):
            protected[rr, cc] = True
        canvas[top : top + height, left : left + width] = color

    return rects


def _intersection_area(a: Rect, b: Rect) -> int:
    overlap_h = min(a.bottom, b.bottom) - max(a.top, b.top) + 1
    overlap_w = min(a.right, b.right) - max(a.left, b.left) + 1
    if overlap_h <= 0 or overlap_w <= 0:
        return 0
    return overlap_h * overlap_w


def _valid_instance(rects: list[Rect], grid: np.ndarray) -> bool:
    if not np.any(grid == BG):
        return False

    has_overlap = any(_intersection_area(a, b) > 0 for i, a in enumerate(rects) for b in rects[i + 1 :])
    if not has_overlap:
        return False

    has_occlusion = False
    for rect in rects:
        bbox = _bbox_for_color(grid, rect.color)
        if bbox != (rect.top, rect.left, rect.bottom, rect.right):
            return False
        visible_count = int(np.count_nonzero(grid == rect.color))
        min_visible = max(3, int(np.ceil(rect.area * 0.3)))
        if visible_count < min_visible:
            return False
        if visible_count < rect.area:
            has_occlusion = True

    return has_occlusion


def _gen_attempt() -> np.ndarray | None:
    n_rects = random.randint(*RECT_COUNT_RANGE)
    dims = _sample_dimensions(n_rects)
    if dims is None:
        return None

    colors = random.sample(range(1, 10), n_rects)
    shape = _sample_canvas(dims)
    modes = _sample_modes(n_rects)
    rects = _place_rectangles(dims, colors, modes, shape)
    if rects is None:
        return None

    grid = _render(rects, shape)
    if not _valid_instance(rects, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GENERATION_RETRIES} attempts")
