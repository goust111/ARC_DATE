import random
from collections import deque

import numpy as np


BG = 0


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox_from_points(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _contains_bbox(
    outer: tuple[int, int, int, int],
    inner: tuple[int, int, int, int],
    *,
    strict: bool = True,
) -> bool:
    or0, oc0, or1, oc1 = outer
    ir0, ic0, ir1, ic1 = inner
    if strict:
        return or0 < ir0 and oc0 < ic0 and ir1 < or1 and ic1 < oc1
    return or0 <= ir0 and oc0 <= ic0 and ir1 <= or1 and ic1 <= oc1


def _component_local_mask(
    points: list[tuple[int, int]],
    bbox: tuple[int, int, int, int],
) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in points:
        mask[r - r0, c - c0] = True
    return mask


def _is_rectangular_frame(
    points: list[tuple[int, int]],
    bbox: tuple[int, int, int, int],
) -> bool:
    r0, c0, r1, c1 = bbox
    if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
        return False
    local = _component_local_mask(points, bbox)
    frame = np.zeros_like(local)
    frame[0, :] = True
    frame[-1, :] = True
    frame[:, 0] = True
    frame[:, -1] = True
    return np.array_equal(local, frame)


def _find_components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(grid[r, c]) == BG:
                continue

            color = int(grid[r, c])
            queue = deque([(r, c)])
            visited[r, c] = True
            points: list[tuple[int, int]] = []

            while queue:
                cr, cc = queue.popleft()
                points.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not visited[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        visited[nr, nc] = True
                        queue.append((nr, nc))

            bbox = _bbox_from_points(points)
            components.append(
                {
                    "color": color,
                    "points": points,
                    "bbox": bbox,
                    "is_frame": _is_rectangular_frame(points, bbox),
                }
            )

    return components


def _fill_bbox_interior(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    patch = grid[r0 + 1 : r1, c0 + 1 : c1]
    patch[patch == BG] = color


def _fill_annulus(
    grid: np.ndarray,
    inner_bbox: tuple[int, int, int, int],
    outer_bbox: tuple[int, int, int, int],
    color: int,
) -> None:
    mask = np.zeros(grid.shape, dtype=bool)
    or0, oc0, or1, oc1 = outer_bbox
    mask[or0 + 1 : or1, oc0 + 1 : oc1] = True

    ir0, ic0, ir1, ic1 = inner_bbox
    mask[ir0 : ir1 + 1, ic0 : ic1 + 1] = False
    grid[mask & (grid == BG)] = color


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate the innermost fill outward through nested rectangular frames, switching to each crossed frame's color.

    Input structure:
    - The background is black.
    - Several one-cell-thick axis-aligned rectangular frames are strictly nested and use distinct colors.
    - The innermost frame contains one non-frame filled region whose color differs from every frame color.
    - The gaps between consecutive frames are black.

    Rule:
    - Identify the non-frame seed region and the ordered chain of rectangular frames that strictly contain it.
    - Keep the seed region and every frame pixel unchanged.
    - Fill any black cell inside the innermost frame with the seed color.
    - For each consecutive inner/outer frame pair, fill the black annulus between them with the inner frame's color.
    - Leave the black background outside the outermost frame unchanged.

    Key insight:
    - Each band inherits its color from the frame just inside it, so the visible nesting order determines the color propagation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    components = _find_components(g)
    frames = [comp for comp in components if comp["is_frame"]]
    seeds = [comp for comp in components if not comp["is_frame"]]

    assert frames, "expected at least one rectangular frame"
    assert seeds, "expected a non-frame seed region"

    def seed_score(comp: dict) -> tuple[int, int]:
        depth = sum(_contains_bbox(frame["bbox"], comp["bbox"], strict=True) for frame in frames)
        return depth, -_bbox_area(comp["bbox"])

    seed = max(seeds, key=seed_score)
    chain = [
        frame
        for frame in frames
        if _contains_bbox(frame["bbox"], seed["bbox"], strict=True)
    ]
    assert chain, "seed region must be enclosed by at least one frame"
    chain.sort(key=lambda frame: _bbox_area(frame["bbox"]))

    for inner, outer in zip(chain, chain[1:]):
        assert _contains_bbox(outer["bbox"], inner["bbox"], strict=True), "frames must be properly nested"

    out = g.copy()
    _fill_bbox_interior(out, chain[0]["bbox"], int(seed["color"]))
    for inner, outer in zip(chain, chain[1:]):
        _fill_annulus(out, inner["bbox"], outer["bbox"], int(inner["color"]))
    return out


# --------------- generator ---------------

GRID_LIMIT = 30
LAYER_COUNTS = (2, 3, 4, 5)
GAP_RANGE = (1, 3)
OUTER_PAD_RANGE = (1, 4)
SEED_SIZE_MIN = 2
SEED_SIZE_MAX = 14
GEN_ATTEMPTS = 300


def _build_bboxes(
    outer_bbox: tuple[int, int, int, int],
    gaps: list[int],
) -> list[tuple[int, int, int, int]]:
    outer_to_inner = [outer_bbox]
    current = outer_bbox
    for gap in reversed(gaps):
        delta = gap + 1
        current = (
            current[0] + delta,
            current[1] + delta,
            current[2] - delta,
            current[3] - delta,
        )
        outer_to_inner.append(current)
    return list(reversed(outer_to_inner))


def _expected_output(
    inp: np.ndarray,
    frame_bboxes: list[tuple[int, int, int, int]],
    frame_colors: list[int],
) -> np.ndarray:
    out = inp.copy()
    for inner_bbox, outer_bbox, color in zip(frame_bboxes, frame_bboxes[1:], frame_colors):
        _fill_annulus(out, inner_bbox, outer_bbox, color)
    return out


def _gen_attempt() -> np.ndarray | None:
    layer_count = random.choice(LAYER_COUNTS)
    gaps = [random.randint(*GAP_RANGE) for _ in range(layer_count - 1)]
    pad_top = random.randint(*OUTER_PAD_RANGE)
    pad_bottom = random.randint(*OUTER_PAD_RANGE)
    pad_left = random.randint(*OUTER_PAD_RANGE)
    pad_right = random.randint(*OUTER_PAD_RANGE)

    shell = 2 + 2 * sum(gap + 1 for gap in gaps)
    max_seed_h = GRID_LIMIT - pad_top - pad_bottom - shell
    max_seed_w = GRID_LIMIT - pad_left - pad_right - shell
    if max_seed_h < SEED_SIZE_MIN or max_seed_w < SEED_SIZE_MIN:
        return None

    seed_h = random.randint(SEED_SIZE_MIN, min(SEED_SIZE_MAX, max_seed_h))
    seed_w = random.randint(SEED_SIZE_MIN, min(SEED_SIZE_MAX, max_seed_w))
    outer_h = seed_h + shell
    outer_w = seed_w + shell

    grid_h = pad_top + outer_h + pad_bottom
    grid_w = pad_left + outer_w + pad_right
    if grid_h > GRID_LIMIT or grid_w > GRID_LIMIT:
        return None

    colors = random.sample(range(1, 10), layer_count + 1)
    frame_colors = colors[:layer_count]
    seed_color = colors[-1]

    outer_bbox = (
        pad_top,
        pad_left,
        pad_top + outer_h - 1,
        pad_left + outer_w - 1,
    )
    frame_bboxes = _build_bboxes(outer_bbox, gaps)

    inp = np.zeros((grid_h, grid_w), dtype=int)
    for bbox, color in zip(frame_bboxes, frame_colors):
        _draw_frame(inp, bbox, color)

    ir0, ic0, ir1, ic1 = frame_bboxes[0]
    inp[ir0 + 1 : ir1, ic0 + 1 : ic1] = seed_color

    expected = _expected_output(inp, frame_bboxes, frame_colors)
    try:
        if np.array_equal(main(inp), expected):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
