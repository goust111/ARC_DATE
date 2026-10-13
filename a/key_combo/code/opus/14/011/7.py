from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BACKGROUND = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

CONTROL_COLORS = (BLUE, RED, YELLOW)
INNER_COLORS = (GREEN, PINK, TEAL, MAROON)
GEN_MAX_ATTEMPTS = 500
CANVAS_SIZE = 30


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _background_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components_of_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _line_runs(pattern: np.ndarray, axis: int) -> list[tuple[int, int, tuple[int, ...]]]:
    lines = pattern if axis == 0 else pattern.T
    runs: list[tuple[int, int, tuple[int, ...]]] = []
    start = 0
    for i in range(1, len(lines) + 1):
        if i == len(lines) or not np.array_equal(lines[i], lines[start]):
            runs.append((start, i, tuple(map(int, lines[start]))))
            start = i
    return runs


def _stretch_run_index(runs: list[tuple[int, int, tuple[int, ...]]]) -> int:
    color_counts = [len(set(line)) for _start, _end, line in runs]
    max_count = max(color_counts)
    candidates = [i for i, count in enumerate(color_counts) if count == max_count]
    if len(candidates) == 1:
        return candidates[0]

    lengths = [end - start for start, end, _line in runs]
    max_length = max(lengths[i] for i in candidates)
    candidates = [i for i in candidates if lengths[i] == max_length]
    assert len(candidates) == 1, "expected a unique stretchable run"
    return candidates[0]


def _axis_indices(pattern: np.ndarray, axis: int, target_length: int) -> list[int]:
    runs = _line_runs(pattern, axis)
    stretch_i = _stretch_run_index(runs)
    fixed_length = sum(end - start for i, (start, end, _line) in enumerate(runs) if i != stretch_i)
    stretch_length = int(target_length) - fixed_length
    assert stretch_length > 0, "target is too small for the fixed bands"

    indices: list[int] = []
    for i, (start, end, _line) in enumerate(runs):
        if i == stretch_i:
            indices.extend([start] * stretch_length)
        else:
            indices.extend(range(start, end))
    assert len(indices) == int(target_length)
    return indices


def _stretch_pattern(pattern: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    row_indices = _axis_indices(pattern, 0, shape[0])
    col_indices = _axis_indices(pattern, 1, shape[1])
    return pattern[np.ix_(row_indices, col_indices)]


def _hole_components(patch: np.ndarray) -> list[np.ndarray]:
    zero_comps = _components_of_mask(patch == BACKGROUND)
    h, w = patch.shape
    holes = []
    for comp in zero_comps:
        if np.any(comp[:, 0] == 0) or np.any(comp[:, 0] == h - 1) or np.any(comp[:, 1] == 0) or np.any(comp[:, 1] == w - 1):
            continue
        holes.append(comp)
    return holes


def _parse_input(grid: np.ndarray) -> tuple[dict[int, int], tuple[int, int, int, int], np.ndarray, list[tuple[int, int, int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    bg = _background_color(g)

    markers: dict[int, int] = {}
    structures: list[tuple[np.ndarray, tuple[int, int, int, int], np.ndarray]] = []
    for comp in _components_of_mask(g != bg):
        box = _bbox(comp)
        patch = _crop(g, box)
        if len(comp) == 1 and int(comp[0, 0]) == 0:
            col = int(comp[0, 1])
            color = int(g[0, col])
            assert color in CONTROL_COLORS, "top-row markers must use control colors"
            assert col not in markers, "control columns must be unique"
            markers[col] = color
        else:
            structures.append((comp, box, patch))

    assert 2 <= len(markers) <= 3, "expected 2-3 top-row control pixels"
    assert structures, "expected exemplar and targets"

    exemplar_candidates: list[tuple[int, int, int, int]] = []
    exemplar_patch: np.ndarray | None = None
    targets: list[tuple[int, int, int, int]] = []
    for _comp, box, patch in structures:
        corners = [int(patch[0, 0]), int(patch[0, -1]), int(patch[-1, 0]), int(patch[-1, -1])]
        assert all(v == ORANGE for v in corners), "all structural rectangles must use orange corners"
        if np.all(patch == ORANGE):
            targets.append(box)
            continue

        colors = sorted(set(map(int, np.unique(patch))) - {bg})
        assert len(colors) >= 2 and ORANGE in colors, "exemplar must be multicolor and include orange"
        exemplar_candidates.append(box)
        exemplar_patch = patch

    assert len(exemplar_candidates) == 1, "expected exactly one multicolor exemplar"
    assert exemplar_patch is not None
    assert 2 <= len(targets) <= 3, "expected 2-3 pure orange targets"
    assert len(markers) == len(targets), "marker count must equal target count"

    for box in targets:
        c0 = box[1]
        c1 = box[3]
        hits = [color for col, color in markers.items() if c0 <= col <= c1]
        assert len(hits) == 1, "each target must contain exactly one control column"

    return markers, exemplar_candidates[0], exemplar_patch, sorted(targets)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stretch the orange-framed exemplar into each solid orange target rectangle, then fill the stretched holes with the top-row control color assigned by column containment.

    Input structure:
    - Black background with 2-3 colored singleton control pixels on the top row.
    - One multicolor orange-framed exemplar rectangle contains enclosed background holes.
    - Two or three larger solid orange target rectangles are separated from the exemplar and from each other.

    Rule:
    - Identify the unique multicolor orange-framed exemplar and the solid orange target rectangles.
    - For each target, elastically stretch the exemplar by keeping all identical row/column runs fixed except the unique structurally richest interior run in each axis, which absorbs the size difference.
    - Detect the enclosed background regions in the stretched target patch, find the unique top-row control pixel whose column lies inside that target's column span, and fill all those holes with that control color.
    - Leave the exemplar and the top-row control pixels unchanged.

    Key insight:
    - The holes are created by the exemplar's internal structure, but their final fill color is injected from outside by the marker whose column falls inside the target rectangle.
    """
    g = np.asarray(grid, dtype=int)
    markers, _exemplar_box, exemplar, targets = _parse_input(g)

    out = g.copy()
    for box in targets:
        r0, c0, r1, c1 = box
        stretched = _stretch_pattern(exemplar, (r1 - r0 + 1, c1 - c0 + 1))
        control_color = [color for col, color in markers.items() if c0 <= col <= c1][0]
        patch = stretched.copy()
        for hole in _hole_components(stretched):
            patch[hole[:, 0], hole[:, 1]] = int(control_color)
        out[r0 : r1 + 1, c0 : c1 + 1] = patch
    return out


def _one_hole_exemplar() -> np.ndarray:
    a, b, c = random.sample(INNER_COLORS, 3)
    patch = np.array(
        [
            [ORANGE, ORANGE, ORANGE, ORANGE, ORANGE],
            [ORANGE, a, a, a, ORANGE],
            [ORANGE, a, BACKGROUND, b, ORANGE],
            [ORANGE, a, c, a, ORANGE],
            [ORANGE, ORANGE, ORANGE, ORANGE, ORANGE],
        ],
        dtype=int,
    )
    if random.random() < 0.5:
        patch = np.fliplr(patch)
    if random.random() < 0.5:
        patch = np.flipud(patch)
    return patch


def _two_hole_exemplar() -> np.ndarray:
    a, c = random.sample(INNER_COLORS, 2)
    patch = np.array(
        [
            [ORANGE, ORANGE, ORANGE, ORANGE, ORANGE, ORANGE],
            [ORANGE, a, a, a, a, ORANGE],
            [ORANGE, a, BACKGROUND, c, BACKGROUND, ORANGE],
            [ORANGE, a, c, a, a, ORANGE],
            [ORANGE, a, a, a, a, ORANGE],
            [ORANGE, ORANGE, ORANGE, ORANGE, ORANGE, ORANGE],
        ],
        dtype=int,
    )
    if random.random() < 0.5:
        patch = np.fliplr(patch)
    if random.random() < 0.5:
        patch = np.flipud(patch)
    return patch


def _random_exemplar() -> np.ndarray:
    return _one_hole_exemplar() if random.random() < 0.5 else _two_hole_exemplar()


def _expanded_box(box: tuple[int, int, int, int], pad: int = 1) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    return r0 - pad, c0 - pad, r1 + pad, c1 + pad


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _place_rectangles(sizes: list[tuple[int, int]]) -> list[tuple[int, int, int, int]] | None:
    order = sorted(range(len(sizes)), key=lambda i: sizes[i][0] * sizes[i][1], reverse=True)
    boxes: list[tuple[int, int, int, int] | None] = [None] * len(sizes)

    def backtrack(k: int) -> bool:
        if k == len(order):
            return True
        idx = order[k]
        h, w = sizes[idx]
        rows = list(range(2, CANVAS_SIZE - h))
        cols = list(range(0, CANVAS_SIZE - w + 1))
        random.shuffle(rows)
        random.shuffle(cols)
        for r0 in rows:
            for c0 in cols:
                box = (r0, c0, r0 + h - 1, c0 + w - 1)
                if box[2] >= CANVAS_SIZE or box[3] >= CANVAS_SIZE:
                    continue
                if any(
                    other is not None and _boxes_overlap(_expanded_box(box), _expanded_box(other))
                    for other in boxes
                ):
                    continue
                boxes[idx] = box
                if backtrack(k + 1):
                    return True
                boxes[idx] = None
        return False

    if backtrack(0):
        return [box for box in boxes if box is not None]
    return None


def _marker_columns(target_boxes: list[tuple[int, int, int, int]], exemplar_box: tuple[int, int, int, int]) -> list[int] | None:
    exemplar_cols = set(range(exemplar_box[1], exemplar_box[3] + 1))
    spans = [set(range(box[1], box[3] + 1)) for box in target_boxes]
    candidates: list[list[int]] = []
    for i, span in enumerate(spans):
        blocked = exemplar_cols.copy()
        for j, other in enumerate(spans):
            if i != j:
                blocked |= other
        allowed = sorted(span - blocked)
        if not allowed:
            return None
        candidates.append(allowed)
    return [random.choice(cols) for cols in candidates]


def _choose_target_sizes(ex_h: int, ex_w: int, count: int) -> list[tuple[int, int]] | None:
    h_hi = 11 if count == 3 else 13
    w_hi = 11 if count == 3 else 13
    options = [
        (h, w)
        for h in range(ex_h + 1, h_hi + 1)
        for w in range(ex_w + 1, w_hi + 1)
        if h + w <= (19 if count == 3 else 23)
    ]
    random.shuffle(options)
    chosen: list[tuple[int, int]] = []
    for h, w in options:
        if any((h, w) == prev for prev in chosen):
            continue
        chosen.append((h, w))
        if len(chosen) == count:
            break
    if len(chosen) != count:
        return None
    return chosen


def _render_input(
    exemplar: np.ndarray,
    exemplar_box: tuple[int, int, int, int],
    target_boxes: list[tuple[int, int, int, int]],
    marker_cols: list[int],
    marker_colors: list[int],
) -> np.ndarray:
    grid = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=int)
    er0, ec0, er1, ec1 = exemplar_box
    grid[er0 : er1 + 1, ec0 : ec1 + 1] = exemplar
    for box in target_boxes:
        r0, c0, r1, c1 = box
        grid[r0 : r1 + 1, c0 : c1 + 1] = ORANGE
    for col, color in zip(marker_cols, marker_colors, strict=True):
        grid[0, col] = int(color)
    return grid


def _render_expected_output(
    input_grid: np.ndarray,
    exemplar: np.ndarray,
    target_boxes: list[tuple[int, int, int, int]],
    marker_cols: list[int],
    marker_colors: list[int],
) -> np.ndarray:
    out = input_grid.copy()
    control_by_col = dict(zip(marker_cols, marker_colors, strict=True))
    for box in target_boxes:
        r0, c0, r1, c1 = box
        stretched = _stretch_pattern(exemplar, (r1 - r0 + 1, c1 - c0 + 1))
        holes = _hole_components(stretched)
        assert holes, "every target must produce at least one hole after stretching"
        hits = [control_by_col[col] for col in marker_cols if c0 <= col <= c1]
        assert len(hits) == 1
        patch = stretched.copy()
        for hole in holes:
            patch[hole[:, 0], hole[:, 1]] = int(hits[0])
        out[r0 : r1 + 1, c0 : c1 + 1] = patch
    return out


def _crop_canvas(inp: np.ndarray, out: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    coords = np.argwhere(inp != BACKGROUND)
    r0 = 0
    c0 = max(0, int(coords[:, 1].min()) - random.randint(0, 2))
    r1 = min(CANVAS_SIZE - 1, int(coords[:, 0].max()) + random.randint(0, 2))
    c1 = min(CANVAS_SIZE - 1, int(coords[:, 1].max()) + random.randint(0, 2))

    while r1 - r0 + 1 < 15 and r1 < CANVAS_SIZE - 1:
        r1 += 1
    while c1 - c0 + 1 < 18:
        if c0 > 0:
            c0 -= 1
        elif c1 < CANVAS_SIZE - 1:
            c1 += 1
        else:
            break

    return inp[r0 : r1 + 1, c0 : c1 + 1].copy(), out[r0 : r1 + 1, c0 : c1 + 1].copy(), c0


def _semantic_checks(
    inp: np.ndarray,
    out: np.ndarray,
    exemplar: np.ndarray,
    target_boxes: list[tuple[int, int, int, int]],
    marker_cols: list[int],
    marker_colors: list[int],
) -> bool:
    if not np.array_equal(main(inp), out):
        return False

    filled_colors = set(marker_colors)
    if len(filled_colors) != len(marker_colors):
        return False

    target_patches: list[bytes] = []
    for box, control_color in zip(target_boxes, marker_colors, strict=True):
        r0, c0, r1, c1 = box
        if not any(c0 <= col <= c1 for col in marker_cols):
            return False
        if exemplar.shape[0] >= (r1 - r0 + 1) or exemplar.shape[1] >= (c1 - c0 + 1):
            return False

        stretched = _stretch_pattern(exemplar, (r1 - r0 + 1, c1 - c0 + 1))
        holes = _hole_components(stretched)
        if not holes:
            return False
        hole_mask = np.zeros_like(stretched, dtype=bool)
        for hole in holes:
            hole_mask[hole[:, 0], hole[:, 1]] = True
        if np.count_nonzero(hole_mask) == 0:
            return False
        target_patch = out[r0 : r1 + 1, c0 : c1 + 1]
        if not np.all(target_patch[hole_mask] == int(control_color)):
            return False
        if not np.array_equal(target_patch[~hole_mask], stretched[~hole_mask]):
            return False
        target_patches.append(target_patch.tobytes() + str(target_patch.shape).encode())

    if len(set(target_patches)) != len(target_patches):
        return False
    return True


def _generate_case() -> tuple[np.ndarray, np.ndarray] | None:
    exemplar = _random_exemplar()
    target_count = 3 if random.random() < 0.45 else 2
    target_sizes = _choose_target_sizes(exemplar.shape[0], exemplar.shape[1], target_count)
    if target_sizes is None:
        return None

    sizes = [exemplar.shape, *target_sizes]
    boxes = _place_rectangles(sizes)
    if boxes is None:
        return None
    exemplar_box = boxes[0]
    target_boxes = boxes[1:]

    marker_cols = _marker_columns(target_boxes, exemplar_box)
    if marker_cols is None:
        return None
    marker_colors = random.sample(CONTROL_COLORS, target_count)

    full_inp = _render_input(exemplar, exemplar_box, target_boxes, marker_cols, marker_colors)
    full_out = _render_expected_output(full_inp, exemplar, target_boxes, marker_cols, marker_colors)
    if not _semantic_checks(full_inp, full_out, exemplar, target_boxes, marker_cols, marker_colors):
        return None

    inp, out, c_shift = _crop_canvas(full_inp, full_out)
    shifted_boxes = [(r0, c0 - c_shift, r1, c1 - c_shift) for r0, c0, r1, c1 in target_boxes]
    shifted_cols = [col - c_shift for col in marker_cols]

    if max(inp.shape) > 30 or max(out.shape) > 30:
        return None
    if not _semantic_checks(inp, out, exemplar, shifted_boxes, shifted_cols, marker_colors):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        case = _generate_case()
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid puzzle instance")
