import random
from collections import Counter, deque

import numpy as np


COLORS = tuple(range(10))

GEN_RETRIES = 400
GRID_H_RANGE = (18, 28)
GRID_W_RANGE = (18, 30)
PALETTE_ENTRIES_RANGE = (2, 4)
PALETTE_GAP_RANGE = (1, 2)
CONTAINER_COUNT_RANGE = (1, 4)
OUTER_H_RANGE = (7, 11)
OUTER_W_RANGE = (7, 11)
MAX_DISTRACTOR_COLORS = 2
PLACEMENT_RETRIES = 120


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _extract_palette_entries(row: np.ndarray, bg: int) -> list[tuple[int, int, int]]:
    entries: list[tuple[int, int, int]] = []
    width = int(row.shape[0])
    c = 0
    while c <= width - 3:
        if int(row[c]) != bg and int(row[c + 1]) == bg and int(row[c + 2]) != bg:
            left_ok = c == 0 or int(row[c - 1]) == bg
            right_ok = c + 3 == width or int(row[c + 3]) == bg
            if left_ok and right_ok:
                entries.append((c, int(row[c]), int(row[c + 2])))
                c += 3
                continue
        c += 1

    keys = [a for _, a, _ in entries]
    if len(keys) != len(set(keys)):
        return []
    return entries


def _find_palette(grid: np.ndarray, bg: int) -> tuple[dict[int, int], np.ndarray]:
    h, w = map(int, grid.shape)
    best_entries: list[tuple[int, int, int]] = []
    best_row = None

    for row in (0, h - 1):
        entries = _extract_palette_entries(grid[row], bg)
        if not entries:
            continue
        adj_row = 1 if row == 0 else h - 2
        isolation = 0
        for c, _a, _b in entries:
            if int(grid[adj_row, c]) == bg and int(grid[adj_row, c + 2]) == bg:
                isolation += 1
        score = (len(entries), isolation)
        best_score = (len(best_entries), -1) if best_row is None else (
            len(best_entries),
            sum(
                int(grid[1 if best_row == 0 else h - 2, c]) == bg
                and int(grid[1 if best_row == 0 else h - 2, c + 2]) == bg
                for c, _a, _b in best_entries
            ),
        )
        if best_row is None or score > best_score:
            best_entries = entries
            best_row = row

    palette_mask = np.zeros((h, w), dtype=bool)
    mapping: dict[int, int] = {}
    if best_row is None:
        return mapping, palette_mask

    for c, a, b in best_entries:
        mapping[a] = b
        palette_mask[best_row, c] = True
        palette_mask[best_row, c + 2] = True
    return mapping, palette_mask


def _frame_from_component(
    component: list[tuple[int, int]],
    height: int,
    width: int,
) -> tuple[list[tuple[int, int]], tuple[int, int, int, int]] | None:
    rows = [r for r, _c in component]
    cols = [c for _r, c in component]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)

    if r0 == 0 or c0 == 0 or r1 == height - 1 or c1 == width - 1:
        return None
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    expected = set()
    for c in range(c0, c1 + 1):
        expected.add((r0, c))
        expected.add((r1, c))
    for r in range(r0 + 1, r1):
        expected.add((r, c0))
        expected.add((r, c1))

    comp_set = set(component)
    if comp_set != expected:
        return None

    ordered = sorted(expected)
    return ordered, (r0, c0, r1, c1)


def _find_frame_components(
    grid: np.ndarray,
    bg: int,
    palette_mask: np.ndarray,
) -> list[tuple[list[tuple[int, int]], tuple[int, int, int, int]]]:
    h, w = map(int, grid.shape)
    visited = np.zeros((h, w), dtype=bool)
    frames = []

    for r in range(h):
        for c in range(w):
            if visited[r, c] or palette_mask[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            queue = deque([(r, c)])
            visited[r, c] = True
            component: list[tuple[int, int]] = []

            while queue:
                rr, cc = queue.popleft()
                component.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if visited[nr, nc] or palette_mask[nr, nc]:
                        continue
                    if int(grid[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    queue.append((nr, nc))

            frame = _frame_from_component(component, h, w)
            if frame is not None:
                frames.append(frame)

    return frames


def _container_actions(
    grid: np.ndarray,
    bg: int,
    mapping: dict[int, int],
    palette_mask: np.ndarray,
) -> list[tuple[list[tuple[int, int]], tuple[int, int, int, int], int, int]]:
    actions = []
    for border_cells, bbox in _find_frame_components(grid, bg, palette_mask):
        r0, c0, r1, c1 = bbox
        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        colors = [int(v) for v in interior.ravel() if int(v) != bg]
        if not colors:
            continue

        counts = Counter(colors)
        max_count = max(counts.values())
        main_colors = [color for color, count in counts.items() if count == max_count]
        if len(main_colors) != 1:
            continue

        main_color = int(main_colors[0])
        matching_keys = [color for color in counts if color in mapping]
        if len(matching_keys) != 1 or main_color != int(matching_keys[0]):
            continue

        actions.append((border_cells, bbox, main_color, int(mapping[main_color])))
    return actions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each rectangular container border by looking up the most frequent
    fragment color inside it in the border palette.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    mapping, palette_mask = _find_palette(g, bg)
    out = g.copy()

    for border_cells, _bbox, _main_color, fill_color in _container_actions(g, bg, mapping, palette_mask):
        for r, c in border_cells:
            out[r, c] = fill_color

    return out


def _random_fragment_counts(
    core_area: int,
    main_key: int,
    other_colors: list[int],
) -> dict[int, int] | None:
    for _ in range(40):
        main_count = random.randint(2, min(5, core_area))
        max_distractors = min(MAX_DISTRACTOR_COLORS, len(other_colors), core_area - main_count)
        n_distractors = random.randint(0, max_distractors)
        chosen = random.sample(other_colors, n_distractors)
        counts = {main_key: main_count}
        remaining = core_area - main_count
        ok = True
        for i, color in enumerate(chosen):
            future_needed = len(chosen) - i - 1
            cap = min(main_count - 1, remaining - future_needed)
            if cap < 1:
                ok = False
                break
            count = random.randint(1, cap)
            counts[color] = count
            remaining -= count
        if ok:
            return counts
    return None


def _draw_container(
    grid: np.ndarray,
    top: int,
    left: int,
    outer_h: int,
    outer_w: int,
    border_color: int,
    fragment_counts: dict[int, int],
) -> tuple[int, int, int, int]:
    bottom = top + outer_h - 1
    right = left + outer_w - 1

    grid[top, left : right + 1] = border_color
    grid[bottom, left : right + 1] = border_color
    grid[top : bottom + 1, left] = border_color
    grid[top : bottom + 1, right] = border_color

    core_positions = [
        (r, c)
        for r in range(top + 2, bottom - 1)
        for c in range(left + 2, right - 1)
    ]
    total = sum(fragment_counts.values())
    chosen_positions = random.sample(core_positions, total)
    idx = 0
    for color, count in fragment_counts.items():
        for _ in range(count):
            r, c = chosen_positions[idx]
            idx += 1
            grid[r, c] = color

    return top, left, bottom, right


def _place_palette(
    grid: np.ndarray,
    bg: int,
    mapping: dict[int, int],
    side: str,
) -> bool:
    h, w = map(int, grid.shape)
    assert side in ("top", "bottom")
    row = 0 if side == "top" else h - 1
    entries = list(mapping.items())
    gaps = [random.randint(*PALETTE_GAP_RANGE) for _ in range(max(0, len(entries) - 1))]
    needed = 3 * len(entries) + sum(gaps)
    if needed > w:
        return False

    start = random.randint(0, w - needed)
    c = start
    for i, (a, b) in enumerate(entries):
        grid[row, c] = a
        grid[row, c + 2] = b
        c += 3
        if i < len(gaps):
            c += gaps[i]
    return True


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(COLORS)
    n_entries = random.randint(*PALETTE_ENTRIES_RANGE)
    palette_keys = random.sample([c for c in COLORS if c != bg], n_entries)
    mapping = {
        key: random.choice([c for c in COLORS if c != bg and c != key])
        for key in palette_keys
    }

    min_width = max(GRID_W_RANGE[0], 3 * n_entries + (n_entries - 1) * PALETTE_GAP_RANGE[1])
    width = random.randint(min_width, GRID_W_RANGE[1])
    height = random.randint(*GRID_H_RANGE)
    grid = np.full((height, width), bg, dtype=int)

    palette_side = random.choice(("top", "bottom"))
    if not _place_palette(grid, bg, mapping, palette_side):
        return None

    blocked = np.zeros((height, width), dtype=bool)
    if palette_side == "top":
        blocked[:2, :] = True
    else:
        blocked[-2:, :] = True

    n_containers = random.randint(*CONTAINER_COUNT_RANGE)
    non_key_colors = [c for c in COLORS if c != bg and c not in palette_keys]
    expected: list[tuple[tuple[int, int, int, int], int, int]] = []

    for _ in range(n_containers):
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            outer_h = random.randint(*OUTER_H_RANGE)
            outer_w = random.randint(*OUTER_W_RANGE)
            if outer_h >= height - 2 or outer_w >= width - 2:
                continue

            top = random.randint(1, height - outer_h - 1)
            left = random.randint(1, width - outer_w - 1)
            pad_top = top - 1
            pad_left = left - 1
            pad_bottom = top + outer_h
            pad_right = left + outer_w
            if blocked[pad_top : pad_bottom + 1, pad_left : pad_right + 1].any():
                continue

            core_area = (outer_h - 4) * (outer_w - 4)
            if core_area < 2:
                continue

            main_key = random.choice(palette_keys)
            fragment_counts = _random_fragment_counts(core_area, main_key, non_key_colors)
            if fragment_counts is None:
                continue

            border_color_choices = [c for c in COLORS if c != bg and c != mapping[main_key]]
            border_color = random.choice(border_color_choices)
            bbox = _draw_container(grid, top, left, outer_h, outer_w, border_color, fragment_counts)
            blocked[pad_top : pad_bottom + 1, pad_left : pad_right + 1] = True
            expected.append((bbox, main_key, mapping[main_key]))
            placed = True
            break
        if not placed:
            return None

    if _most_common_color(grid) != bg:
        return None

    found_mapping, palette_mask = _find_palette(grid, bg)
    if found_mapping != mapping:
        return None

    frames = _find_frame_components(grid, bg, palette_mask)
    actual_bboxes = {bbox for _cells, bbox in frames}
    expected_bboxes = {bbox for bbox, _key, _fill in expected}
    if actual_bboxes != expected_bboxes:
        return None

    actions = _container_actions(grid, bg, found_mapping, palette_mask)
    actual = {(bbox, main_key, fill) for _cells, bbox, main_key, fill in actions}
    expected_set = set(expected)
    if actual != expected_set:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
