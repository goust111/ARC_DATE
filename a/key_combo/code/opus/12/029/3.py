from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
GREY = 5
FRAME_COLORS = (2, 3, 4, 6, 7, 8, 9)
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _rect_border_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _find_frames(grid: np.ndarray) -> list[dict]:
    frames: list[dict] = []
    for color in sorted(set(map(int, np.unique(grid))) - {BLACK, BLUE, GREY}):
        for coords in _components(grid == color):
            r0, c0, r1, c1 = _bbox(coords)
            height = r1 - r0 + 1
            width = c1 - c0 + 1
            assert height >= 4 and width >= 4, "frames must be at least 4x4"
            border = _rect_border_mask(height, width)
            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            assert int(coords.shape[0]) == int(border.sum()), "frame must be one-cell thick"
            assert np.all(patch[border] == color), "frame border must be monochrome"
            assert np.all(patch[~border] == BLACK), "frame interior must be black"
            frames.append({"color": color, "bbox": (r0, c0, r1, c1)})
    frames.sort(key=lambda frame: frame["bbox"])
    return frames


def _reachable_mask(grid: np.ndarray, seed: tuple[int, int]) -> np.ndarray:
    h, w = grid.shape
    reach = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque([seed])
    reach[seed] = True

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if reach[nr, nc] or int(grid[nr, nc]) != BLACK:
                continue
            reach[nr, nc] = True
            q.append((nr, nc))
    return reach


def _frame_touched(frame: dict, reach: np.ndarray) -> bool:
    r0, c0, r1, c1 = frame["bbox"]
    border_cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        border_cells.append((r0, c))
        border_cells.append((r1, c))
    for r in range(r0 + 1, r1):
        border_cells.append((r, c0))
        border_cells.append((r, c1))

    h, w = reach.shape
    for r, c in border_cells:
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and reach[nr, nc]:
                return True
    return False


def _render_output(shape: tuple[int, int], frames: list[dict], filled_colors: set[int]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for frame in frames:
        color = int(frame["color"])
        r0, c0, r1, c1 = frame["bbox"]
        out[r0, c0 : c1 + 1] = color
        out[r1, c0 : c1 + 1] = color
        out[r0 : r1 + 1, c0] = color
        out[r0 : r1 + 1, c1] = color
        if color in filled_colors:
            out[r0 + 1 : r1, c0 + 1 : c1] = BLUE
    return out


def _solve_details(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    seed_coords = np.argwhere(g == BLUE)
    assert len(seed_coords) == 1, "expected exactly one blue seed"
    assert np.count_nonzero(g == GREY) > 0, "expected grey walls"
    assert len(_components(g == GREY)) == 1, "grey walls must form one connected obstacle"

    seed = tuple(map(int, seed_coords[0]))
    frames = _find_frames(g)
    assert 3 <= len(frames) <= 5, "expected 3-5 hollow rectangle frames"
    reach = _reachable_mask(g, seed)
    filled = {int(frame["color"]) for frame in frames if _frame_touched(frame, reach)}
    out = _render_output(g.shape, frames, filled)
    return {"seed": seed, "frames": frames, "reach": reach, "filled": filled, "output": out}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Flood from the blue seed, keep only the rectangle frames, and fill exactly the flood-adjacent frame interiors blue.

    Input structure:
    - The grid has a black background, one blue seed pixel, one connected grey wall structure, and 3-5 one-cell-thick hollow rectangular frames.
    - Every frame uses a distinct non-black, non-blue, non-grey color and has a black interior.
    - Grey walls and colored frame borders act as impassable barriers that split the black background into separate chambers.

    Rule:
    - Starting from the blue seed, traverse only 4-connected black background cells; do not cross grey walls or any frame border.
    - For each hollow frame, test whether some border pixel has a 4-neighbor in the reached region, counting the seed cell itself as part of that region.
    - Fill the interior of every touched frame with blue and leave untouched frame interiors black.
    - Erase the seed, every reached background cell, and all grey wall pixels so that only the original frame borders remain in the output.

    Key insight:
    - The disappearing grey walls are the real selectors: they control which chamber the flood can occupy, and only frames adjacent to that reachable chamber get filled.
    """
    return _solve_details(grid)["output"]


def _sample_segments(total: int, count: int, min_size: int) -> list[int] | None:
    if total < count * min_size:
        return None
    lengths = [min_size] * count
    extra = total - count * min_size
    for _ in range(extra):
        lengths[random.randrange(count)] += 1
    random.shuffle(lengths)
    return lengths


def _cut_rows(lengths: list[int]) -> list[int]:
    cuts: list[int] = []
    cursor = 0
    for i, length in enumerate(lengths[:-1]):
        cursor += length
        cuts.append(cursor + i)
    return cuts


def _build_partition(
    height: int,
    width: int,
    left_cuts: list[int],
    right_cuts: list[int],
    split_c: int,
) -> tuple[np.ndarray, list[dict]]:
    grid = np.zeros((height, width), dtype=int)
    grid[:, split_c] = GREY
    for r in left_cuts:
        grid[r, : split_c + 1] = GREY
    for r in right_cuts:
        grid[r, split_c:] = GREY

    chambers: list[dict] = []
    start = 0
    for r in left_cuts + [height]:
        chambers.append({"side": "left", "box": (start, 0, r - 1, split_c - 1)})
        start = r + 1
    start = 0
    for r in right_cuts + [height]:
        chambers.append({"side": "right", "box": (start, split_c + 1, r - 1, width - 1)})
        start = r + 1

    chambers = [
        {
            **chamber,
            "height": chamber["box"][2] - chamber["box"][0] + 1,
            "width": chamber["box"][3] - chamber["box"][1] + 1,
        }
        for chamber in chambers
        if chamber["box"][0] <= chamber["box"][2] and chamber["box"][1] <= chamber["box"][3]
    ]
    return grid, chambers


def _boxes_touch_or_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (
        a[2] < b[0] - 1
        or b[2] < a[0] - 1
        or a[3] < b[1] - 1
        or b[3] < a[1] - 1
    )


def _place_single_frame(box: tuple[int, int, int, int]) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = box
    chamber_h = r1 - r0 + 1
    chamber_w = c1 - c0 + 1
    max_h = min(7, chamber_h - 2)
    max_w = min(7, chamber_w - 2)
    if max_h < 4 or max_w < 4:
        return None
    for _ in range(120):
        outer_h = random.randint(4, max_h)
        outer_w = random.randint(4, max_w)
        top = random.randint(r0 + 1, r1 - outer_h)
        left = random.randint(c0 + 1, c1 - outer_w)
        return (top, left, top + outer_h - 1, left + outer_w - 1)
    return None


def _place_two_frames(box: tuple[int, int, int, int]) -> list[tuple[int, int, int, int]] | None:
    r0, c0, r1, c1 = box
    chamber_h = r1 - r0 + 1
    chamber_w = c1 - c0 + 1
    options = []
    if chamber_w >= 11 and chamber_h >= 6:
        options.append("horizontal")
    if chamber_h >= 11 and chamber_w >= 6:
        options.append("vertical")
    random.shuffle(options)

    for orient in options:
        for _ in range(120):
            if orient == "horizontal":
                pair_ws = [
                    (w1, w2)
                    for w1 in range(4, min(6, chamber_w - 7) + 1)
                    for w2 in range(4, min(6, chamber_w - 3 - w1) + 1)
                    if w1 + w2 <= chamber_w - 3
                ]
                if not pair_ws:
                    continue
                outer_w1, outer_w2 = random.choice(pair_ws)
                outer_h1 = random.randint(4, min(6, chamber_h - 2))
                outer_h2 = random.randint(4, min(6, chamber_h - 2))
                usable_left = c0 + 1
                slack = chamber_w - 2 - (outer_w1 + outer_w2 + 1)
                lead = random.randint(0, slack)
                tail = random.randint(0, slack - lead)
                gap = 1 + (slack - lead - tail)
                left1 = usable_left + lead
                left2 = left1 + outer_w1 + gap
                top1 = random.randint(r0 + 1, r1 - outer_h1)
                top2 = random.randint(r0 + 1, r1 - outer_h2)
                boxes = [
                    (top1, left1, top1 + outer_h1 - 1, left1 + outer_w1 - 1),
                    (top2, left2, top2 + outer_h2 - 1, left2 + outer_w2 - 1),
                ]
            else:
                pair_hs = [
                    (h1, h2)
                    for h1 in range(4, min(6, chamber_h - 7) + 1)
                    for h2 in range(4, min(6, chamber_h - 3 - h1) + 1)
                    if h1 + h2 <= chamber_h - 3
                ]
                if not pair_hs:
                    continue
                outer_h1, outer_h2 = random.choice(pair_hs)
                outer_w1 = random.randint(4, min(6, chamber_w - 2))
                outer_w2 = random.randint(4, min(6, chamber_w - 2))
                usable_top = r0 + 1
                slack = chamber_h - 2 - (outer_h1 + outer_h2 + 1)
                head = random.randint(0, slack)
                foot = random.randint(0, slack - head)
                gap = 1 + (slack - head - foot)
                top1 = usable_top + head
                top2 = top1 + outer_h1 + gap
                left1 = random.randint(c0 + 1, c1 - outer_w1)
                left2 = random.randint(c0 + 1, c1 - outer_w2)
                boxes = [
                    (top1, left1, top1 + outer_h1 - 1, left1 + outer_w1 - 1),
                    (top2, left2, top2 + outer_h2 - 1, left2 + outer_w2 - 1),
                ]

            if not _boxes_touch_or_overlap(boxes[0], boxes[1]):
                return boxes
    return None


def _place_frames_in_chamber(
    box: tuple[int, int, int, int],
    count: int,
) -> list[tuple[int, int, int, int]] | None:
    if count == 1:
        single = _place_single_frame(box)
        return None if single is None else [single]
    if count == 2:
        return _place_two_frames(box)
    return None


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _inside_any_bbox(r: int, c: int, boxes: list[tuple[int, int, int, int]]) -> bool:
    return any(br0 <= r <= br1 and bc0 <= c <= bc1 for br0, bc0, br1, bc1 in boxes)


def _frame_border_cells(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        cells.append((r, c1))
    return cells


def _min_seed_distance(seed: tuple[int, int], bbox: tuple[int, int, int, int]) -> int:
    sr, sc = seed
    return min(abs(sr - r) + abs(sc - c) for r, c in _frame_border_cells(bbox))


def _transform_grid(grid: np.ndarray) -> np.ndarray:
    variants = [
        lambda x: x,
        lambda x: np.flipud(x).copy(),
        lambda x: np.fliplr(x).copy(),
        lambda x: np.flipud(np.fliplr(x)).copy(),
        lambda x: x.T.copy(),
        lambda x: np.flipud(x.T).copy(),
        lambda x: np.fliplr(x.T).copy(),
        lambda x: np.flipud(np.fliplr(x.T)).copy(),
    ]
    return random.choice(variants)(grid)


def _background_chamber_count(grid: np.ndarray, frames: list[dict]) -> int:
    blocked = grid == GREY
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        blocked[r0 : r1 + 1, c0 : c1 + 1] = True
    return len(_components(~blocked))


def _distance_confound(grid: np.ndarray, frames: list[dict], filled: set[int]) -> bool:
    seed = tuple(map(int, np.argwhere(grid == BLUE)[0]))
    selected = [_min_seed_distance(seed, frame["bbox"]) for frame in frames if frame["color"] in filled]
    unselected = [_min_seed_distance(seed, frame["bbox"]) for frame in frames if frame["color"] not in filled]
    return bool(selected and unselected and min(unselected) <= max(selected))


def _assign_chambers(
    chambers: list[dict],
    n_rects: int,
    n_selected: int,
) -> tuple[dict, list[dict]] | None:
    candidates = [ch for ch in chambers if _place_frames_in_chamber(ch["box"], n_selected) is not None]
    if not candidates:
        return None
    selected = random.choice(candidates)

    remaining = n_rects - n_selected
    others = [ch for ch in chambers if ch is not selected]
    if len(others) < remaining:
        return None

    same_side = [ch for ch in others if ch["side"] == selected["side"]]
    other_side = [ch for ch in others if ch["side"] != selected["side"]]
    chosen: list[dict] = []

    if remaining >= 2 and same_side and other_side:
        chosen.append(random.choice(same_side))
        opp = random.choice(other_side)
        if opp not in chosen:
            chosen.append(opp)
    elif remaining >= 1 and same_side and random.random() < 0.65:
        chosen.append(random.choice(same_side))

    pool = [ch for ch in others if ch not in chosen]
    random.shuffle(pool)
    for chamber in pool:
        if len(chosen) >= remaining:
            break
        chosen.append(chamber)

    if len(chosen) != remaining:
        return None
    return selected, chosen


def _build_candidate() -> tuple[np.ndarray, set[int]] | None:
    height = random.randint(18, 24)
    width = random.randint(18, 24)

    split_options = [c for c in range(6, width - 6) if c >= 6 and width - c - 1 >= 6]
    if not split_options:
        return None
    split_c = random.choice(split_options)

    pair_options = [(1, 1), (1, 2), (2, 1), (2, 2)]
    k_left, k_right = random.choice(pair_options)
    total_chambers = k_left + k_right + 2
    n_rects = random.randint(3, min(5, total_chambers + 1))
    allowed_selected = [s for s in (1, 2) if total_chambers - 1 >= n_rects - s]
    if not allowed_selected:
        return None
    n_selected = random.choice(allowed_selected)

    min_chamber_h = 6
    left_lengths = _sample_segments(height - k_left, k_left + 1, min_chamber_h)
    right_lengths = _sample_segments(height - k_right, k_right + 1, min_chamber_h)
    if left_lengths is None or right_lengths is None:
        return None

    grid, chambers = _build_partition(height, width, _cut_rows(left_lengths), _cut_rows(right_lengths), split_c)
    assignment = _assign_chambers(chambers, n_rects, n_selected)
    if assignment is None:
        return None
    selected_chamber, other_chambers = assignment

    frame_counts = {id(selected_chamber): n_selected}
    frame_counts.update({id(chamber): 1 for chamber in other_chambers})
    all_target_chambers = [selected_chamber] + other_chambers

    all_boxes: list[tuple[int, int, int, int]] = []
    chamber_boxes: dict[int, list[tuple[int, int, int, int]]] = {}
    for chamber in all_target_chambers:
        boxes = _place_frames_in_chamber(chamber["box"], frame_counts[id(chamber)])
        if boxes is None:
            return None
        chamber_boxes[id(chamber)] = boxes
        all_boxes.extend(boxes)

    colors = random.sample(FRAME_COLORS, n_rects)
    random.shuffle(all_boxes)
    selected_colors: set[int] = set()
    for color, bbox in zip(colors, all_boxes):
        _draw_frame(grid, bbox, color)
        if bbox in chamber_boxes[id(selected_chamber)]:
            selected_colors.add(color)

    seed_candidates = []
    chosen_boxes = chamber_boxes[id(selected_chamber)]
    r0, c0, r1, c1 = selected_chamber["box"]
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if grid[r, c] != BLACK or _inside_any_bbox(r, c, chosen_boxes):
                continue
            black_neighbors = sum(
                0 <= nr < height and 0 <= nc < width and grid[nr, nc] == BLACK
                for nr, nc in _neighbors4(r, c)
            )
            seed_candidates.append((black_neighbors, r, c))
    if not seed_candidates:
        return None
    rich = [cell for cell in seed_candidates if cell[0] >= 1]
    _, sr, sc = random.choice(rich if rich else seed_candidates)
    grid[sr, sc] = BLUE

    return _transform_grid(grid), selected_colors


def _validate_candidate(grid: np.ndarray, planned_selected: set[int]) -> bool:
    try:
        details = _solve_details(grid)
    except Exception:
        return False

    frames = details["frames"]
    filled = details["filled"]
    if filled != planned_selected:
        return False
    if not (1 <= len(filled) < len(frames)):
        return False
    if _background_chamber_count(grid, frames) < 2:
        return False
    if not _distance_confound(grid, frames, filled):
        return False

    expected = _render_output(grid.shape, frames, planned_selected)
    try:
        actual = main(grid)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        built = _build_candidate()
        if built is None:
            continue
        grid, selected = built
        if _validate_candidate(grid, selected):
            return grid
    raise RuntimeError("failed to generate a valid flood-reachability frame puzzle")
