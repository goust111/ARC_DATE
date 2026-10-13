import numpy as np


BG = 0
MARK = 9
RECT_COLORS = (1, 2, 3, 4, 5)
MAX_ATTEMPTS = 500


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def bbox_from_mask(mask: np.ndarray):
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def bbox_h(box) -> int:
    return int(box[2] - box[0] + 1)


def bbox_w(box) -> int:
    return int(box[3] - box[1] + 1)


def _recover_rectangles(grid: np.ndarray):
    bg = get_bg(grid)
    rectangles = []
    boxes = {}
    for color in sorted(int(v) for v in np.unique(grid) if int(v) not in {bg, MARK}):
        box = bbox_from_mask(grid == color)
        assert box is not None
        boxes[color] = box
        rectangles.append(
            {
                "color": color,
                "height": bbox_h(box),
                "width": bbox_w(box),
                "area": bbox_h(box) * bbox_w(box),
                "marks": [],
            }
        )

    for r, c in np.argwhere(grid == MARK):
        owners = []
        for rect in rectangles:
            box = boxes[rect["color"]]
            if box[0] <= int(r) <= box[2] and box[1] <= int(c) <= box[3]:
                owners.append(rect)
        assert len(owners) == 1
        owner = owners[0]
        box = boxes[owner["color"]]
        owner["marks"].append((int(r) - box[0], int(c) - box[1]))

    for rect in rectangles:
        rect["marks"].sort()
    return bg, rectangles


def _stack_rectangles(rectangles, bg: int):
    out_h = max(rect["height"] for rect in rectangles)
    out_w = max(rect["width"] for rect in rectangles)
    canvas = np.full((out_h, out_w), bg, dtype=int)
    order = sorted(rectangles, key=lambda rect: rect["area"], reverse=True)
    areas = [rect["area"] for rect in order]
    assert len(set(areas)) == len(areas)
    for rect in order:
        h = rect["height"]
        w = rect["width"]
        canvas[:h, :w] = rect["color"]
        for r, c in rect["marks"]:
            assert 0 <= r < h and 0 <= c < w
            canvas[r, c] = MARK
    return canvas


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stack the separated rectangles, count only the still-visible maroon markers, and repeat the stack horizontally by that count.

    Input structure:
    - A black background contains 4 to 5 separated filled rectangles of distinct non-black, non-maroon colors.
    - Some rectangles contain internal maroon marker pixels that replace the rectangle color at those cells.
    - Each rectangle's size is recovered from the bounding box of its own color, while maroon pixels belong to the rectangle whose box contains them.

    Rule:
    - Reconstruct one rectangle per foreground color, keeping each rectangle's internal maroon pixels at the same relative offsets.
    - Sort the rectangles by area from largest to smallest and draw them all with their upper-left corners aligned at the output origin, so later smaller rectangles cover earlier larger ones.
    - Count the maroon pixels that remain visible after stacking, then concatenate that stacked image horizontally exactly that many times.

    Key insight:
    - The marker count is not taken from the input directly; only the markers still visible after area-ordered stacking determine the repetition count.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    bg, rectangles = _recover_rectangles(g)
    stacked = _stack_rectangles(rectangles, bg)
    visible_marks = int(np.count_nonzero(stacked == MARK))
    assert visible_marks >= 1
    return np.tile(stacked, (1, visible_marks))


def _covered_by_later(rectangles, index: int, cell) -> bool:
    r, c = cell
    for later in rectangles[index + 1 :]:
        if r < later["height"] and c < later["width"]:
            return True
    return False


def _candidate_cells(rectangles):
    hidden = []
    visible = []
    for i, rect in enumerate(rectangles):
        rect_hidden = []
        rect_visible = []
        for r in range(1, rect["height"] - 1):
            for c in range(1, rect["width"] - 1):
                if _covered_by_later(rectangles, i, (r, c)):
                    rect_hidden.append((r, c))
                else:
                    rect_visible.append((r, c))
        hidden.append(rect_hidden)
        visible.append(rect_visible)
    return hidden, visible


def _sample_rectangles(rng: np.random.Generator):
    n_rects = int(rng.integers(4, 6))
    colors = list(rng.choice(RECT_COLORS, size=n_rects, replace=False))
    rectangles = []
    seen_areas = set()
    for _ in range(400):
        sizes = []
        seen_areas.clear()
        ok = True
        for _idx in range(n_rects):
            h = int(rng.integers(3, 9))
            w = int(rng.integers(3, 9))
            area = h * w
            if area in seen_areas:
                ok = False
                break
            seen_areas.add(area)
            sizes.append((h, w, area))
        if not ok:
            continue
        rectangles = [
            {"color": int(color), "height": h, "width": w, "area": area, "marks": []}
            for color, (h, w, area) in zip(colors, sizes)
        ]
        rectangles.sort(key=lambda rect: rect["area"], reverse=True)
        hidden, visible = _candidate_cells(rectangles)
        lower_visible = [
            i
            for i in range(len(rectangles) - 1)
            if hidden[i] and visible[i]
        ]
        if not lower_visible:
            continue
        max_visible = min(4, 30 // max(rect["width"] for rect in rectangles))
        if max_visible < 2:
            continue
        return rectangles, hidden, visible
    return None, None, None


def _pick_unused(rng: np.random.Generator, cells, used):
    choices = [cell for cell in cells if cell not in used]
    if not choices:
        return None
    return choices[int(rng.integers(len(choices)))]


def _assign_marks(rectangles, hidden, visible, rng: np.random.Generator):
    max_visible = min(4, 30 // max(rect["width"] for rect in rectangles))
    pivots = [i for i in range(len(rectangles) - 1) if hidden[i] and visible[i]]
    for _ in range(500):
        for rect in rectangles:
            rect["marks"] = []
        pivot = int(rng.choice(pivots))
        target_visible = int(rng.integers(2, max_visible + 1))
        target_marked_rects = int(rng.integers(2, 4))
        used = set()
        vis_count = 0
        vis_sources = set()

        cell = _pick_unused(rng, hidden[pivot], used)
        if cell is None:
            continue
        rectangles[pivot]["marks"].append(cell)
        used.add(cell)

        cell = _pick_unused(rng, visible[pivot], used)
        if cell is None:
            continue
        rectangles[pivot]["marks"].append(cell)
        used.add(cell)
        vis_count += 1
        vis_sources.add(pivot)

        ok = True
        while vis_count < target_visible:
            candidates = []
            for i, rect in enumerate(rectangles):
                if len(rect["marks"]) >= 2:
                    continue
                cell = _pick_unused(rng, visible[i], used)
                if cell is not None:
                    candidates.append((i, cell))
            if not candidates:
                ok = False
                break
            i, cell = candidates[int(rng.integers(len(candidates)))]
            rectangles[i]["marks"].append(cell)
            used.add(cell)
            vis_count += 1
            vis_sources.add(i)
        if not ok:
            continue

        while sum(1 for rect in rectangles if rect["marks"]) < target_marked_rects:
            candidates = []
            for i, rect in enumerate(rectangles):
                if len(rect["marks"]) >= 2:
                    continue
                cell = _pick_unused(rng, hidden[i], used)
                if cell is not None:
                    candidates.append((i, cell))
            if not candidates:
                ok = False
                break
            i, cell = candidates[int(rng.integers(len(candidates)))]
            rectangles[i]["marks"].append(cell)
            used.add(cell)
        if not ok:
            continue

        if rng.random() < 0.45:
            candidates = []
            for i, rect in enumerate(rectangles):
                if len(rect["marks"]) >= 2:
                    continue
                cell = _pick_unused(rng, hidden[i], used)
                if cell is not None:
                    candidates.append((i, cell))
            if candidates:
                i, cell = candidates[int(rng.integers(len(candidates)))]
                rectangles[i]["marks"].append(cell)
                used.add(cell)

        marked = [i for i, rect in enumerate(rectangles) if rect["marks"]]
        hidden_total = sum(
            1
            for i, rect in enumerate(rectangles)
            for cell in rect["marks"]
            if cell in hidden[i]
        )
        visible_total = sum(
            1
            for i, rect in enumerate(rectangles)
            for cell in rect["marks"]
            if cell in visible[i]
        )
        if len(marked) not in {2, 3}:
            continue
        if not all(1 <= len(rectangles[i]["marks"]) <= 2 for i in marked):
            continue
        if hidden_total < 1:
            continue
        if visible_total != target_visible:
            continue
        if len(vis_sources) < 2:
            continue
        if pivot not in vis_sources:
            continue
        return True
    return False


def _pack_rectangles(rectangles, rng: np.random.Generator):
    n = len(rectangles)
    order = list(rng.permutation(n))
    for _ in range(300):
        row_count = 2 if n == 4 or rng.random() < 0.65 else 3
        if row_count == 2:
            cut = int(rng.integers(1, n))
            groups = [order[:cut], order[cut:]]
        else:
            cut1 = int(rng.integers(1, n - 1))
            cut2 = int(rng.integers(cut1 + 1, n))
            groups = [order[:cut1], order[cut1:cut2], order[cut2:]]
        if any(len(group) == 0 for group in groups):
            continue

        left_margin = int(rng.integers(1, 4))
        right_margin = int(rng.integers(1, 4))
        top_margin = int(rng.integers(1, 4))
        bottom_margin = int(rng.integers(1, 4))
        row_gap_values = [int(rng.integers(2, 5)) for _ in range(len(groups) - 1)]
        x_gaps = {
            idx: [int(rng.integers(2, 5)) for _ in range(max(0, len(group) - 1))]
            for idx, group in enumerate(groups)
        }

        row_heights = [max(rectangles[i]["height"] for i in group) for group in groups]
        row_widths = []
        for idx, group in enumerate(groups):
            width_sum = sum(rectangles[i]["width"] for i in group)
            width_sum += sum(x_gaps[idx])
            row_widths.append(width_sum)

        min_h = top_margin + bottom_margin + sum(row_heights) + sum(row_gap_values)
        min_w = left_margin + right_margin + max(row_widths)
        if min_h > 30 or min_w > 30:
            continue

        extra_h = int(rng.integers(0, 31 - min_h))
        extra_w = int(rng.integers(0, 31 - min_w))
        canvas_h = min_h + extra_h
        canvas_w = min_w + extra_w
        offset_r = int(rng.integers(0, extra_h + 1))

        positions = {}
        current_r = top_margin + offset_r
        for row_idx, group in enumerate(groups):
            slack_x = canvas_w - left_margin - right_margin - row_widths[row_idx]
            start_c = left_margin + int(rng.integers(0, slack_x + 1))
            current_c = start_c
            row_h = row_heights[row_idx]
            for j, rect_idx in enumerate(group):
                rect = rectangles[rect_idx]
                max_jitter = row_h - rect["height"]
                jitter_r = int(rng.integers(0, max_jitter + 1))
                positions[rect_idx] = (current_r + jitter_r, current_c)
                current_c += rect["width"]
                if j < len(group) - 1:
                    current_c += x_gaps[row_idx][j]
            current_r += row_h
            if row_idx < len(groups) - 1:
                current_r += row_gap_values[row_idx]
        return canvas_h, canvas_w, positions
    return None


def _render_input(rectangles, canvas_h: int, canvas_w: int, positions):
    grid = np.full((canvas_h, canvas_w), BG, dtype=int)
    for idx, rect in enumerate(rectangles):
        r0, c0 = positions[idx]
        h = rect["height"]
        w = rect["width"]
        grid[r0 : r0 + h, c0 : c0 + w] = rect["color"]
        for dr, dc in rect["marks"]:
            grid[r0 + dr, c0 + dc] = MARK
    return grid


def _expected_output_from_world(rectangles):
    stacked = _stack_rectangles(rectangles, BG)
    visible_count = int(np.count_nonzero(stacked == MARK))
    return np.tile(stacked, (1, visible_count)), stacked, visible_count


def _validate_world(rectangles, grid, expected_output):
    assert 4 <= len(rectangles) <= 5
    areas = [rect["area"] for rect in rectangles]
    assert len(set(areas)) == len(areas)
    marked_rects = [rect for rect in rectangles if rect["marks"]]
    assert len(marked_rects) in {2, 3}
    assert all(1 <= len(rect["marks"]) <= 2 for rect in marked_rects)
    hidden, visible = _candidate_cells(sorted(rectangles, key=lambda rect: rect["area"], reverse=True))
    visible_sources = set()
    hidden_count = 0
    visible_count = 0
    lower_visible = False
    lower_pivot_ok = False
    for i, rect in enumerate(sorted(rectangles, key=lambda rect: rect["area"], reverse=True)):
        has_hidden = False
        has_visible = False
        for cell in rect["marks"]:
            if cell in hidden[i]:
                hidden_count += 1
                has_hidden = True
            if cell in visible[i]:
                visible_count += 1
                visible_sources.add(i)
                has_visible = True
                if i < len(rectangles) - 1:
                    lower_visible = True
        if i < len(rectangles) - 1 and has_hidden and has_visible:
            lower_pivot_ok = True

    assert hidden_count >= 1
    assert 2 <= visible_count <= 4
    assert len(visible_sources) >= 2
    assert lower_visible
    assert lower_pivot_ok
    assert max(grid.shape) <= 30
    assert max(expected_output.shape) <= 30
    solved = main(grid)
    assert np.array_equal(solved, expected_output)


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        rectangles, hidden, visible = _sample_rectangles(rng)
        if rectangles is None:
            continue
        if not _assign_marks(rectangles, hidden, visible, rng):
            continue
        packed = _pack_rectangles(rectangles, rng)
        if packed is None:
            continue
        canvas_h, canvas_w, positions = packed
        grid = _render_input(rectangles, canvas_h, canvas_w, positions)
        expected_output, _stacked, visible_count = _expected_output_from_world(rectangles)
        if visible_count < 2 or visible_count > 4:
            continue
        if expected_output.shape[1] > 30:
            continue
        _validate_world(rectangles, grid, expected_output)
        return grid
    raise RuntimeError("failed to generate a valid ARC input")
