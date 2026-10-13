from __future__ import annotations

import random

import numpy as np


BG = 0
MARKER = 5
RECT_DIMS = ((2, 3), (3, 2))
FILL_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_ATTEMPTS = 400

_GEN_CALLS = 0


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _scale_patch(patch: np.ndarray, factor: int) -> np.ndarray:
    return np.kron(patch, np.ones((factor, factor), dtype=int))


def _boxes_separate(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], padding: int = 0
) -> bool:
    return (
        a[2] + padding < b[0]
        or b[2] + padding < a[0]
        or a[3] + padding < b[1]
        or b[3] + padding < a[1]
    )


def _components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] == BG or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and grid[nr, nc] != BG:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _analyze_rectangles(grid: np.ndarray) -> list[dict[str, object]]:
    rects: list[dict[str, object]] = []
    for coords in _components(grid):
        box = _bbox(coords)
        patch = _crop(grid, box)
        h, w = patch.shape
        assert (h, w) in RECT_DIMS, "every object must be a 2x3 or 3x2 rectangle"
        assert len(coords) == h * w, "foreground object must fill its bounding box"
        assert np.all(patch != BG), "rectangle may not contain background holes"

        colors = {int(v) for v in np.unique(patch)}
        non_marker = colors - {MARKER}
        assert len(non_marker) == 1, "each rectangle must have one visible fill color"
        fill = next(iter(non_marker))
        assert fill != BG, "fill color must be foreground"

        rects.append(
            {
                "bbox": box,
                "patch": patch,
                "fill": fill,
                "markers": int(np.sum(patch == MARKER)),
                "shape": (h, w),
            }
        )

    assert 3 <= len(rects) <= 5, "expected 3 to 5 rectangles"
    fills = [int(rect["fill"]) for rect in rects]
    assert len(set(fills)) == len(fills), "rectangle fill colors must be distinct"
    rects.sort(key=lambda rect: tuple(int(v) for v in rect["bbox"]))
    return rects


def _recolored_scaled_template(
    template_patch: np.ndarray, template_fill: int, target_fill: int, factor: int
) -> np.ndarray:
    scaled = _scale_patch(template_patch, factor)
    scaled[scaled == template_fill] = target_fill
    return scaled


def _semantic_output(grid: np.ndarray) -> np.ndarray:
    rects = _analyze_rectangles(grid)
    marker_counts = [int(rect["markers"]) for rect in rects]
    minimum = min(marker_counts)
    template_ids = [i for i, count in enumerate(marker_counts) if count == minimum]
    assert len(template_ids) == 1, "template must be the unique least-marked rectangle"
    template_id = template_ids[0]
    template = rects[template_id]
    assert int(template["markers"]) in (0, 1), "template marker count must be 0 or 1"

    other_counts = [int(rect["markers"]) for i, rect in enumerate(rects) if i != template_id]
    assert all(2 <= count <= 3 for count in other_counts), "target marker counts must be 2 or 3"
    assert len(set(other_counts)) == len(other_counts), "target marker counts must be distinct"

    template_patch = np.asarray(template["patch"], dtype=int)
    template_fill = int(template["fill"])
    out = grid.copy()

    occupied = [tuple(int(v) for v in template["bbox"])]
    for i, rect in enumerate(rects):
        if i == template_id:
            continue
        factor = int(rect["markers"])
        r0, c0, _r1, _c1 = rect["bbox"]
        scaled = _recolored_scaled_template(template_patch, template_fill, int(rect["fill"]), factor)
        sh, sw = scaled.shape
        footprint = (r0, c0, r0 + sh - 1, c0 + sw - 1)
        assert footprint[2] < grid.shape[0] and footprint[3] < grid.shape[1], "scaled template left grid bounds"
        assert all(_boxes_separate(footprint, other, padding=0) for other in occupied), "scaled outputs overlap"
        occupied.append(footprint)

        region = grid[r0 : r0 + sh, c0 : c0 + sw]
        allowed = np.zeros((sh, sw), dtype=bool)
        rh, rw = rect["shape"]
        allowed[:rh, :rw] = True
        assert np.all(region[~allowed] == BG), "scaled output would cover another input object"
        out[r0 : r0 + sh, c0 : c0 + sw] = scaled

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Use the least-marked rectangle as a template and enlarge it at other rectangles by their marker counts.

    Input structure:
    - A black background contains several separated solid foreground rectangles.
    - Every rectangle is either 2x3 or 3x2, has its own fill color, and may contain grey marker pixels that replace fill cells.
    - Exactly one rectangle has the strictly fewest grey markers; the others have larger marker counts that act as scale factors.

    Rule:
    - Find the unique rectangle with the smallest grey-marker count and crop it as the template.
    - Keep that template rectangle unchanged in place.
    - For every other rectangle, scale the template by that rectangle's grey-marker count, recolor template-fill cells to the target rectangle's fill color, keep grey cells grey, and place the enlarged copy with its top-left corner anchored at the target rectangle's top-left corner.

    Key insight:
    - Grey markers have two roles at once: the sparsest rectangle selects the blueprint, while the marker counts inside the other rectangles encode how much that blueprint must be enlarged before recoloring.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D ARC grid"
    return _semantic_output(g)


def _make_rect_patch(height: int, width: int, fill: int, marker_count: int) -> np.ndarray:
    patch = np.full((height, width), fill, dtype=int)
    if marker_count:
        cells = random.sample([(r, c) for r in range(height) for c in range(width)], marker_count)
        for r, c in cells:
            patch[r, c] = MARKER
    return patch


def _branch_profile() -> tuple[int, tuple[int, int]]:
    global _GEN_CALLS
    index = _GEN_CALLS
    _GEN_CALLS += 1
    template_markers = index % 2
    template_dims = RECT_DIMS[(index // 2) % len(RECT_DIMS)]
    return template_markers, template_dims


def _random_anchor(
    grid_shape: tuple[int, int],
    size: tuple[int, int],
    placed: list[tuple[int, int, int, int]],
    *,
    border: int,
    padding: int,
) -> tuple[int, int] | None:
    h, w = grid_shape
    ph, pw = size
    candidates: list[tuple[int, int]] = []
    for r0 in range(border, h - ph - border + 1):
        for c0 in range(border, w - pw - border + 1):
            box = (r0, c0, r0 + ph - 1, c0 + pw - 1)
            if all(_boxes_separate(box, prev, padding=padding) for prev in placed):
                candidates.append((r0, c0))
    if not candidates:
        return None
    return random.choice(candidates)


def _render_expected(
    grid_shape: tuple[int, int],
    template_anchor: tuple[int, int],
    template_patch: np.ndarray,
    template_fill: int,
    targets: list[dict[str, object]],
) -> np.ndarray:
    out = np.zeros(grid_shape, dtype=int)
    tr, tc = template_anchor
    th, tw = template_patch.shape
    out[tr : tr + th, tc : tc + tw] = template_patch
    for target in targets:
        r0, c0 = target["anchor"]
        scaled = _recolored_scaled_template(
            template_patch,
            template_fill,
            int(target["fill"]),
            int(target["markers"]),
        )
        sh, sw = scaled.shape
        out[r0 : r0 + sh, c0 : c0 + sw] = scaled
    return out


def _validate_generated_input(grid: np.ndarray) -> None:
    rects = _analyze_rectangles(grid)
    counts = sorted(int(rect["markers"]) for rect in rects)
    assert counts[0] in (0, 1), "generator must expose both template branches"
    assert counts[1:] == [2, 3], "generator uses one 2x target and one 3x target to keep rules unambiguous"


def _generate_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    template_markers, (th, tw) = _branch_profile()
    colors = random.sample(FILL_COLORS, 3)
    template_fill = colors[0]
    target_fills = colors[1:]

    template_patch = _make_rect_patch(th, tw, template_fill, template_markers)
    target_specs = []
    for markers, fill in zip((2, 3), target_fills):
        rh, rw = random.choice(RECT_DIMS)
        target_specs.append(
            {
                "markers": markers,
                "fill": fill,
                "input_patch": _make_rect_patch(rh, rw, fill, markers),
                "input_shape": (rh, rw),
                "output_shape": (th * markers, tw * markers),
            }
        )
    random.shuffle(target_specs)

    max_h = max(th, *(int(spec["output_shape"][0]) for spec in target_specs))
    max_w = max(tw, *(int(spec["output_shape"][1]) for spec in target_specs))
    min_h = max(15, max_h + 4)
    min_w = max(15, max_w + 4)
    if min_h > 26 or min_w > 26:
        return None
    grid_shape = (
        random.randint(min_h, 26),
        random.randint(min_w, 26),
    )

    placed: list[tuple[int, int, int, int]] = []
    anchors: dict[str, tuple[int, int]] = {}
    placement_items = [
        ("target_3", next(spec for spec in target_specs if int(spec["markers"]) == 3)["output_shape"]),
        ("target_2", next(spec for spec in target_specs if int(spec["markers"]) == 2)["output_shape"]),
        ("template", (th, tw)),
    ]
    for name, size in placement_items:
        anchor = _random_anchor(grid_shape, size, placed, border=1, padding=1)
        if anchor is None:
            return None
        r0, c0 = anchor
        sh, sw = size
        placed.append((r0, c0, r0 + sh - 1, c0 + sw - 1))
        anchors[name] = anchor

    inp = np.zeros(grid_shape, dtype=int)
    tr, tc = anchors["template"]
    inp[tr : tr + th, tc : tc + tw] = template_patch

    targets: list[dict[str, object]] = []
    for spec in target_specs:
        key = f"target_{int(spec['markers'])}"
        r0, c0 = anchors[key]
        patch = np.asarray(spec["input_patch"], dtype=int)
        rh, rw = patch.shape
        inp[r0 : r0 + rh, c0 : c0 + rw] = patch
        targets.append(
            {
                "anchor": (r0, c0),
                "fill": int(spec["fill"]),
                "markers": int(spec["markers"]),
            }
        )

    expected = _render_expected(grid_shape, anchors["template"], template_patch, template_fill, targets)
    _validate_generated_input(inp)
    if not np.array_equal(main(inp), expected):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        generated = _generate_attempt()
        if generated is not None:
            inp, _expected = generated
            return inp
    raise RuntimeError("failed to generate a valid input for puzzle 8")
