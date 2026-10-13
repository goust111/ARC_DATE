import sys
from collections import deque
from pathlib import Path

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _ARC_UTILS_DIR.exists():
    sys.path.append(str(_ARC_UTILS_DIR))

try:
    from arc_utils import crop_bbox, find_components, get_bg
except Exception:
    def get_bg(grid: np.ndarray) -> int:
        values, counts = np.unique(grid, return_counts=True)
        return int(values[np.argmax(counts)])

    def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
        r0, c0, r1, c1 = map(int, box)
        return grid[r0 : r1 + 1, c0 : c1 + 1].copy()

    def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
        rows = [r for r, _ in coords]
        cols = [c for _, c in coords]
        return min(rows), min(cols), max(rows), max(cols)

    def find_components(
        grid: np.ndarray,
        *,
        bg: int | None = None,
        connectivity: int = 4,
        monochromatic: bool = True,
        colors: list[int] | None = None,
        min_area: int = 1,
        return_format: str = "dict",
    ) -> list[dict]:
        if return_format != "dict":
            raise ValueError("fallback find_components only supports return_format='dict'")
        if bg is None:
            bg = get_bg(grid)
        bg = int(bg)
        wanted = None if colors is None else {int(c) for c in colors if int(c) != bg}
        h, w = grid.shape
        seen = np.zeros((h, w), dtype=bool)
        if connectivity == 4:
            neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        else:
            neighbors = [
                (-1, -1), (-1, 0), (-1, 1),
                (0, -1),           (0, 1),
                (1, -1),  (1, 0),  (1, 1),
            ]
        out = []
        for r in range(h):
            for c in range(w):
                color = int(grid[r, c])
                if seen[r, c] or color == bg or (wanted is not None and color not in wanted):
                    continue
                queue = deque([(r, c)])
                seen[r, c] = True
                coords = []
                while queue:
                    cr, cc = queue.popleft()
                    coords.append((cr, cc))
                    for dr, dc in neighbors:
                        nr = cr + dr
                        nc = cc + dc
                        if not (0 <= nr < h and 0 <= nc < w) or seen[nr, nc]:
                            continue
                        if monochromatic:
                            ok = int(grid[nr, nc]) == color
                        else:
                            ncol = int(grid[nr, nc])
                            ok = ncol != bg and (wanted is None or ncol in wanted)
                        if ok:
                            seen[nr, nc] = True
                            queue.append((nr, nc))
                if len(coords) >= int(min_area):
                    out.append(
                        {
                            "color": color,
                            "coords": np.array(coords, dtype=int),
                            "bbox": _bbox_from_coords(coords),
                            "area": len(coords),
                        }
                    )
        return out


COMB_COLORS = (1, 2, 3, 4, 7)
HEIGHTS = (2, 3, 4, 5, 6)
MIN_COMBS = 3
MAX_COMBS = 5
MAX_OUTPUT_SIDE = 30
GEN_ATTEMPTS = 400


def _parse_comb(component: dict, grid: np.ndarray, bg: int) -> dict:
    color = int(component["color"])
    r0, c0, r1, c1 = map(int, component["bbox"])
    patch = crop_bbox(grid, (r0, c0, r1, c1))
    assert patch.shape[1] == 3, f"comb bbox must be 3 columns wide, got {patch.shape}"
    solid_rows = 0
    expected_is_solid = True
    for row in patch:
        mask = tuple(int(v == color) for v in row)
        if expected_is_solid:
            assert mask == (1, 1, 1), f"expected solid row, got {mask}"
            solid_rows += 1
        else:
            assert mask == (1, 0, 1), f"expected open-tooth row, got {mask}"
        expected_is_solid = not expected_is_solid
        assert all(v in (bg, color) for v in row), "comb bbox may only contain background and its own color"
    assert tuple(int(v == color) for v in patch[-1]) == (1, 1, 1), "comb must end with a solid row"
    assert patch.shape[0] == 2 * solid_rows - 1, "comb rows must alternate solid/open/solid"
    assert 2 <= solid_rows <= 6, f"comb height out of range: {solid_rows}"
    return {
        "color": color,
        "height": solid_rows,
        "bbox": (r0, c0, r1, c1),
        "top": r0,
        "left": c0,
    }


def _extract_combs(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = get_bg(g)
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, colors=list(COMB_COLORS))
    assert MIN_COMBS <= len(comps) <= MAX_COMBS, f"expected {MIN_COMBS}-{MAX_COMBS} combs"
    combs = [_parse_comb(comp, g, bg) for comp in comps]
    heights = [comb["height"] for comb in combs]
    colors = [comb["color"] for comb in combs]
    assert len(set(heights)) == len(heights), "comb heights must be unique"
    assert len(set(colors)) == len(colors), "comb colors must be unique"
    combs.sort(key=lambda item: item["left"])
    for left_a, left_b in zip([c["left"] for c in combs], [c["left"] for c in combs][1:]):
        assert left_b - left_a >= 5, "combs must be separated by at least two blank columns"
    return combs


def _build_output(combs: list[dict], *, use_total_rows: bool = False, reorder: str = "position") -> np.ndarray:
    items = [dict(comb) for comb in combs]
    if reorder == "height_asc":
        items.sort(key=lambda item: item["height"])
    elif reorder == "height_desc":
        items.sort(key=lambda item: (-item["height"], item["left"]))
    else:
        items.sort(key=lambda item: item["left"])

    if use_total_rows:
        starts = [2 * item["height"] - 1 for item in items]
    else:
        starts = [item["height"] for item in items]
    width = max(starts)
    assert len(items) * width <= MAX_OUTPUT_SIDE and width <= MAX_OUTPUT_SIDE
    out = np.zeros((len(items) * width, width), dtype=int)
    for block_idx, (item, start_len) in enumerate(zip(items, starts)):
        top = block_idx * width
        for r in range(width):
            fill = min(width, start_len + r)
            out[top + r, :fill] = item["color"]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count each comb's solid-row height, use it as the starting run length, and stack the resulting growing color runs in left-to-right order.

    Input structure:
    - The grid is a black background with 3 to 5 disconnected monochrome comb motifs.
    - Each comb has a 3-column bounding box and alternates solid rows (`xxx`) with open-tooth rows (`x.x`).
    - Every comb ends with a solid row, uses one distinct foreground color, and no two combs share the same solid-row count.

    Rule:
    - Detect every comb and count how many solid 3-cell rows it contains; that count is the comb's height value `c`.
    - Sort the combs by their leftmost column from left to right.
    - Let `W` be the largest height among all combs. For each sorted comb, create a `W x W` subgrid in its own color where row 1 fills `c` cells from the left, and each later row fills one more cell than the previous row, capped at `W`.
    - Stack those subgrids vertically without gaps to form the output.

    Key insight:
    - The comb height does not control the output block height; it controls the first row length of a uniformly tall stacked growth pattern, while the horizontal placement of the input combs determines the vertical order of those blocks.
    """
    combs = _extract_combs(np.asarray(grid, dtype=int))
    return _build_output(combs)


def _random_partition(rng: np.random.Generator, total: int, slots: int) -> list[int]:
    if slots == 1:
        return [total]
    cuts = np.sort(rng.integers(0, total + 1, size=slots - 1))
    prev = 0
    parts = []
    for cut in cuts:
        parts.append(int(cut - prev))
        prev = int(cut)
    parts.append(int(total - prev))
    return parts


def _draw_comb(grid: np.ndarray, top: int, left: int, height: int, color: int) -> None:
    for idx in range(2 * height - 1):
        r = top + idx
        if idx % 2 == 0:
            grid[r, left : left + 3] = color
        else:
            grid[r, left] = color
            grid[r, left + 2] = color


def _independent_expected(meta: list[dict]) -> np.ndarray:
    ordered = sorted(meta, key=lambda item: item["left"])
    width = max(item["height"] for item in ordered)
    out = np.zeros((len(ordered) * width, width), dtype=int)
    for idx, item in enumerate(ordered):
        top = idx * width
        for r in range(width):
            out[top + r, : min(width, item["height"] + r)] = item["color"]
    return out


def _validate_instance(grid: np.ndarray, meta: list[dict]) -> None:
    parsed = _extract_combs(grid)
    expected_pairs = [(item["left"], item["color"], item["height"]) for item in sorted(meta, key=lambda x: x["left"])]
    parsed_pairs = [(item["left"], item["color"], item["height"]) for item in parsed]
    assert parsed_pairs == expected_pairs, "parsed combs do not match generated metadata"

    correct = main(grid)
    expected = _independent_expected(meta)
    assert np.array_equal(correct, expected), "main() disagrees with independent rule reconstruction"

    height_order = [item["height"] for item in sorted(meta, key=lambda x: x["left"])]
    assert height_order != sorted(height_order), "left-to-right heights must not be ascending"
    assert height_order != sorted(height_order, reverse=True), "left-to-right heights must not be descending"

    wrong_asc = _build_output(parsed, reorder="height_asc")
    wrong_desc = _build_output(parsed, reorder="height_desc")
    assert not np.array_equal(correct, wrong_asc), "position order degenerates to ascending height order"
    assert not np.array_equal(correct, wrong_desc), "position order degenerates to descending height order"

    wrong_count = _build_output(parsed, use_total_rows=True)
    assert correct.shape != wrong_count.shape or not np.array_equal(correct, wrong_count), (
        "solid-row count must matter; total comb row count should not reproduce the answer"
    )

    assert max(grid.shape) <= 30, f"input exceeds ARC limits: {grid.shape}"
    assert max(correct.shape) <= 30, f"output exceeds ARC limits: {correct.shape}"


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        n_combs = int(rng.integers(MIN_COMBS, MAX_COMBS + 1))
        heights = list(rng.choice(HEIGHTS, size=n_combs, replace=False))
        rng.shuffle(heights)
        if heights == sorted(heights) or heights == sorted(heights, reverse=True):
            continue

        colors = list(rng.choice(COMB_COLORS, size=n_combs, replace=False))
        rng.shuffle(colors)

        min_width = 3 * n_combs + 2 * (n_combs - 1)
        grid_w = int(rng.integers(min_width, 31))
        extra_w = grid_w - min_width
        gap_extras = _random_partition(rng, extra_w, n_combs + 1)

        left = gap_extras[0]
        lefts = []
        for idx in range(n_combs):
            lefts.append(left)
            left += 3
            if idx < n_combs - 1:
                left += 2 + gap_extras[idx + 1]

        max_rows = max(2 * h - 1 for h in heights)
        min_height = max(max_rows + 1, 12)
        grid_h = int(rng.integers(min_height, 25))
        grid = np.zeros((grid_h, grid_w), dtype=int)

        tops = []
        for height in heights:
            tops.append(int(rng.integers(0, grid_h - (2 * height - 1) + 1)))

        meta = []
        for top, left, height, color in zip(tops, lefts, heights, colors):
            _draw_comb(grid, top, left, int(height), int(color))
            meta.append({"top": int(top), "left": int(left), "height": int(height), "color": int(color)})

        try:
            _validate_instance(grid, meta)
        except AssertionError:
            continue
        return grid

    raise RuntimeError(f"failed to generate a valid comb-count instance after {GEN_ATTEMPTS} attempts")
