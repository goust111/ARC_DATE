import numpy as np


class Color:
    BLACK = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    YELLOW = 4
    GREY = 5


COLOR_BY_COUNT = {
    0: Color.GREEN,
    1: Color.YELLOW,
    2: Color.RED,
    3: Color.BLUE,
}

FRAME_SIZE_CHOICES = (4, 5, 6)
GEN_RETRY_LIMIT = 500


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_mask4(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _segment_cells(start_r: int, start_c: int, length: int) -> list[tuple[int, int]]:
    return [(int(start_r - i), int(start_c + i)) for i in range(int(length))]


def _extract_segment(grid: np.ndarray) -> list[tuple[int, int]]:
    coords = np.argwhere(grid == Color.GREEN)
    assert 3 <= len(coords) <= 5, "expected one green anti-diagonal segment of length 3-5"
    diag_ids = {int(r + c) for r, c in coords}
    assert len(diag_ids) == 1, "green cells must lie on one anti-diagonal"
    ordered = sorted((int(r), int(c)) for r, c in coords)
    ordered.sort(key=lambda rc: rc[1])
    cols = [c for _, c in ordered]
    rows = [r for r, _ in ordered]
    assert len(set(cols)) == len(cols), "segment columns must be unique"
    assert cols == list(range(cols[0], cols[0] + len(cols))), "segment columns must be consecutive"
    assert rows == list(range(rows[0], rows[0] - len(rows), -1)), "segment rows must decrease by 1"
    return ordered


def _frame_border_set(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    border: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        border.add((r0, c))
        border.add((r1, c))
    for r in range(r0, r1 + 1):
        border.add((r, c0))
        border.add((r, c1))
    return border


def _extract_frames(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    frames: list[tuple[int, int, int, int]] = []
    for coords in _components_mask4(grid == Color.GREY):
        bbox = _bbox_coords(coords)
        r0, c0, r1, c1 = bbox
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert 4 <= h <= 6 and 4 <= w <= 6, "frame size must stay within 4x4 to 6x6"
        border = _frame_border_set(bbox)
        coord_set = {tuple(map(int, rc)) for rc in coords}
        assert coord_set == border, "grey component must be a single-pixel hollow rectangle"
        assert r0 > 0 and c0 > 0 and r1 < grid.shape[0] - 1 and c1 < grid.shape[1] - 1, (
            "frames must not touch the grid edge"
        )
        frames.append(bbox)
    assert 2 <= len(frames) <= 3, "expected 2-3 grey frames"
    frames.sort()
    return frames


def _count_crossed_frames(
    line: list[tuple[int, int]], frames: list[tuple[int, int, int, int]]
) -> tuple[int, list[tuple[int, int, int, int]]]:
    crossed: list[tuple[int, int, int, int]] = []
    for bbox in frames:
        r0, c0, r1, c1 = bbox
        if any(r0 < r < r1 and c0 < c < c1 for r, c in line):
            crossed.append(bbox)
    return len(crossed), crossed


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the green anti-diagonal segment to the next longer position, then recolor it by how many grey frames it passes through.

    Input structure:
    - The background is black.
    - Exactly one green segment lies on a single anti-diagonal, with consecutive columns and rows decreasing by 1.
    - Two or three grey 4-connected hollow rectangle frames sit away from the grid edge.

    Rule:
    - Erase the original green segment.
    - Let its length be L and its rightmost cell be the current segment end; draw a new segment of length L+1 that starts one step further right on the same anti-diagonal.
    - Count how many distinct grey frames contain at least one new segment cell strictly inside the frame interior.
    - Recolor the whole new segment by that count: 0 stays green, 1 becomes yellow, 2 becomes red, 3 becomes blue.
    - Leave the grey frames unchanged except where the new segment overwrites frame-border pixels.

    Key insight:
    - The final color is not determined by the original line alone: you must first compute the translated longer anti-diagonal segment, then count its frame crossings in the frame interiors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "grid must fit ARC size limits"
    assert set(map(int, np.unique(g))) <= {Color.BLACK, Color.GREEN, Color.GREY}, (
        "input may contain only black, green, and grey"
    )

    old_line = _extract_segment(g)
    frames = _extract_frames(g)
    old_end_r, old_end_c = old_line[-1]
    new_line = _segment_cells(old_end_r - 1, old_end_c + 1, len(old_line) + 1)
    assert all(0 <= r < g.shape[0] and 0 <= c < g.shape[1] for r, c in new_line), "new segment leaves the grid"

    crossed_count, _ = _count_crossed_frames(new_line, frames)
    assert crossed_count in COLOR_BY_COUNT, "crossed-frame count must be in 0..3"
    line_color = COLOR_BY_COUNT[crossed_count]

    out = g.copy()
    for r, c in old_line:
        out[r, c] = Color.BLACK
    for r, c in new_line:
        out[r, c] = line_color
    return out


def _bbox_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[2] + 1 < b[0] or b[2] + 1 < a[0] or a[3] + 1 < b[1] or b[3] + 1 < a[1]


def _cell_relation(
    cell: tuple[int, int], bbox: tuple[int, int, int, int]
) -> tuple[bool, bool]:
    r, c = cell
    r0, c0, r1, c1 = bbox
    in_box = r0 <= r <= r1 and c0 <= c <= c1
    in_interior = r0 < r < r1 and c0 < c < c1
    on_border = in_box and not in_interior
    return in_interior, on_border


def _enumerate_frame_candidates(
    shape: tuple[int, int],
    old_line: list[tuple[int, int]],
    new_line: list[tuple[int, int]],
) -> list[dict]:
    h, w = shape
    candidate_boxes: set[tuple[int, int, int, int]] = set()
    for r, c in new_line + old_line:
        for fh in FRAME_SIZE_CHOICES:
            for fw in FRAME_SIZE_CHOICES:
                for roff in range(fh):
                    for coff in range(fw):
                        r0 = r - roff
                        c0 = c - coff
                        bbox = (r0, c0, r0 + fh - 1, c0 + fw - 1)
                        if bbox[0] < 1 or bbox[1] < 1 or bbox[2] >= h - 1 or bbox[3] >= w - 1:
                            continue
                        candidate_boxes.add(bbox)

    candidates: list[dict] = []
    for bbox in sorted(candidate_boxes):
        new_inside: list[int] = []
        new_border: list[int] = []
        old_inside: list[int] = []
        old_border: list[int] = []
        for idx, cell in enumerate(new_line):
            inside, border = _cell_relation(cell, bbox)
            if inside:
                new_inside.append(idx)
            elif border:
                new_border.append(idx)
        for idx, cell in enumerate(old_line):
            inside, border = _cell_relation(cell, bbox)
            if inside:
                old_inside.append(idx)
            elif border:
                old_border.append(idx)
        if not (new_inside or new_border or old_inside or old_border):
            continue
        candidates.append(
            {
                "bbox": bbox,
                "new_inside": tuple(new_inside),
                "new_border": tuple(new_border),
                "old_inside": tuple(old_inside),
                "old_border": tuple(old_border),
            }
        )
    return candidates


def _sample_pool(candidates: list[dict], rng: np.random.Generator, limit: int = 160) -> list[dict]:
    if len(candidates) <= limit:
        return candidates
    order = rng.permutation(len(candidates))[:limit]
    return [candidates[int(i)] for i in order]


def _search_non_overlapping(
    candidates: list[dict],
    count: int,
    rng: np.random.Generator,
    *,
    require_multi_hit: bool = False,
    blocked: list[dict] | None = None,
) -> list[dict] | None:
    blocked = blocked or []
    shuffled = list(candidates)
    rng.shuffle(shuffled)

    def dfs(start: int, chosen: list[dict], have_multi: bool) -> list[dict] | None:
        if len(chosen) == count:
            if require_multi_hit and not have_multi:
                return None
            return list(chosen)
        remaining = count - len(chosen)
        if len(shuffled) - start < remaining:
            return None
        for i in range(start, len(shuffled)):
            cand = shuffled[i]
            if any(not _bbox_separated(cand["bbox"], other["bbox"]) for other in blocked):
                continue
            if any(not _bbox_separated(cand["bbox"], other["bbox"]) for other in chosen):
                continue
            chosen.append(cand)
            result = dfs(
                i + 1,
                chosen,
                have_multi or len(cand["new_inside"]) >= 2,
            )
            chosen.pop()
            if result is not None:
                return result
        return None

    return dfs(0, [], False)


def _build_grid(
    shape: tuple[int, int],
    old_line: list[tuple[int, int]],
    frames: list[dict],
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for frame in frames:
        for r, c in _frame_border_set(frame["bbox"]):
            grid[r, c] = Color.GREY
    for r, c in old_line:
        assert grid[r, c] != Color.GREY, "old line must not cover a frame border in the input"
        grid[r, c] = Color.GREEN
    return grid


def _semantic_checks(grid: np.ndarray) -> None:
    old_line = _extract_segment(grid)
    frames = _extract_frames(grid)
    old_end_r, old_end_c = old_line[-1]
    new_line = _segment_cells(old_end_r - 1, old_end_c + 1, len(old_line) + 1)

    frame_infos = []
    for bbox in frames:
        r0, c0, r1, c1 = bbox
        new_inside = [i for i, (r, c) in enumerate(new_line) if r0 < r < r1 and c0 < c < c1]
        new_border = [i for i, (r, c) in enumerate(new_line) if r0 <= r <= r1 and c0 <= c <= c1 and not (r0 < r < r1 and c0 < c < c1)]
        old_inside = [i for i, (r, c) in enumerate(old_line) if r0 < r < r1 and c0 < c < c1]
        frame_infos.append(
            {
                "new_inside": tuple(new_inside),
                "new_border": tuple(new_border),
                "old_inside": tuple(old_inside),
            }
        )

    k = sum(1 for info in frame_infos if info["new_inside"])
    assert 1 <= k <= 3, "generated inputs must cross 1-3 frame interiors after moving"

    interior_pixel_hits = sum(len(info["new_inside"]) for info in frame_infos)
    border_touch_frames = sum(1 for info in frame_infos if info["new_inside"] or info["new_border"])
    old_cross_count = sum(1 for info in frame_infos if info["old_inside"])
    assert interior_pixel_hits > k or border_touch_frames > k or old_cross_count != k, (
        "each sample must exclude at least one shortcut"
    )

    out = main(grid)
    expected_color = COLOR_BY_COUNT[k]
    moved_cells = np.argwhere(out == expected_color)
    assert len(moved_cells) >= len(old_line) + 1, "output must contain the moved segment"
    assert not np.array_equal(out, grid), "transform must change the grid"


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    k = int(rng.choice([1, 2, 3]))
    if k == 3:
        line_len = int(rng.choice([4, 5]))
    else:
        line_len = int(rng.choice([3, 4, 5]))

    h = int(rng.integers(17 if k >= 2 else 15, 23))
    w = int(rng.integers(17 if k >= 2 else 15, 23))
    if min(h, w) < 2 * line_len + 3:
        return None

    target_start_r = int(rng.integers(line_len, h - line_len))
    target_start_c = int(rng.integers(line_len, w - line_len))
    new_line = _segment_cells(target_start_r, target_start_c, line_len + 1)
    old_line = _segment_cells(target_start_r + line_len, target_start_c - line_len, line_len)

    if any(not (0 <= r < h and 0 <= c < w) for r, c in old_line + new_line):
        return None

    candidates = _enumerate_frame_candidates((h, w), old_line, new_line)
    selected_pool = [
        cand
        for cand in candidates
        if cand["new_inside"] and not cand["old_inside"] and not cand["old_border"]
    ]
    border_pool = [
        cand
        for cand in candidates
        if cand["new_border"] and not cand["new_inside"] and not cand["old_inside"] and not cand["old_border"]
    ]
    old_pool = [
        cand
        for cand in candidates
        if cand["old_inside"] and not cand["new_inside"] and not cand["old_border"]
    ]
    selected_pool = _sample_pool(selected_pool, rng)
    border_pool = _sample_pool(border_pool, rng)
    old_pool = _sample_pool(old_pool, rng)

    total_frames = 3 if k == 3 else (3 if rng.random() < 0.7 else 2)
    need_distractors = total_frames - k
    require_multi_hit = need_distractors == 0
    selected = _search_non_overlapping(
        selected_pool,
        k,
        rng,
        require_multi_hit=require_multi_hit,
    )
    if selected is None:
        return None

    distractors: list[dict] = []
    if need_distractors == 1:
        pools = [old_pool, border_pool] if rng.random() < 0.5 else [border_pool, old_pool]
        chosen = None
        for pool in pools:
            chosen = _search_non_overlapping(pool, 1, rng, blocked=selected)
            if chosen is not None:
                break
        if chosen is None:
            return None
        distractors = chosen
    elif need_distractors == 2:
        first_old = _search_non_overlapping(old_pool, 1, rng, blocked=selected)
        if first_old is not None:
            second_border = _search_non_overlapping(border_pool, 1, rng, blocked=selected + first_old)
            if second_border is not None:
                distractors = first_old + second_border
        if not distractors:
            combined = old_pool + border_pool
            distractors = _search_non_overlapping(combined, 2, rng, blocked=selected) or []
        if len(distractors) != 2:
            return None

    frames = selected + distractors
    grid = _build_grid((h, w), old_line, frames)
    _semantic_checks(grid)
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 9")
