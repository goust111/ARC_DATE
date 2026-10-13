from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(pts, dtype=int))
    return out


def _detect_shelves(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    shelves: list[dict] = []
    for r in range(h):
        row = grid[r]
        if int(row[0]) == bg or int(row[-1]) == bg:
            continue
        non_bg = row[row != bg]
        if non_bg.size == 0:
            continue
        colors = np.unique(non_bg)
        if len(colors) != 1:
            continue

        gap_cols = np.flatnonzero(row == bg)
        gap = None
        if gap_cols.size > 0:
            if int(gap_cols[0]) == 0 or int(gap_cols[-1]) == w - 1:
                continue
            if not np.all(np.diff(gap_cols) == 1):
                continue
            gap = (int(gap_cols[0]), int(gap_cols[-1]))

        shelves.append({"row": int(r), "color": int(colors[0]), "gap": gap})

    shelves.sort(key=lambda shelf: shelf["row"])
    assert len(shelves) >= 2, "expected at least two shelf lines"
    assert len({shelf["color"] for shelf in shelves}) == len(shelves), "shelf colors must be unique"
    assert all(shelf["gap"] is not None for shelf in shelves[:-1]), "all non-bottom shelves must have gaps"
    assert shelves[-1]["gap"] is None, "bottom shelf must be solid"
    return shelves


def _detect_blocks(grid: np.ndarray, bg: int, shelves: list[dict]) -> list[dict]:
    mask = grid != bg
    shelf_rows = [shelf["row"] for shelf in shelves]
    mask[shelf_rows, :] = False
    comps = _components(mask)
    assert comps, "expected at least one falling block"

    top_shelf_row = shelves[0]["row"]
    blocks: list[dict] = []
    for coords in comps:
        r0, c0, r1, c1 = _bbox(coords)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        assert patch.shape[0] * patch.shape[1] == coords.shape[0], "blocks must be solid rectangles"
        colors = np.unique(patch)
        assert len(colors) == 1 and int(colors[0]) != bg, "block must be monochromatic"
        assert r1 < top_shelf_row, "all blocks must start above the top shelf"
        blocks.append(
            {
                "bbox": (r0, c0, r1, c1),
                "color": int(colors[0]),
                "h": int(r1 - r0 + 1),
                "w": int(c1 - c0 + 1),
                "coords": coords,
            }
        )

    blocks.sort(key=lambda block: block["bbox"][1])
    intervals = [(block["bbox"][1], block["bbox"][3]) for block in blocks]
    assert all(intervals[i][1] < intervals[i + 1][0] for i in range(len(intervals) - 1)), "block spans must not overlap"
    return blocks


def _inside_gap(c0: int, c1: int, gap: tuple[int, int] | None) -> bool:
    return gap is not None and c0 >= gap[0] and c1 <= gap[1]


def _stop_index_for_span(c0: int, c1: int, shelves: list[dict]) -> int:
    for idx, shelf in enumerate(shelves):
        if not _inside_gap(c0, c1, shelf["gap"]):
            return idx
    raise AssertionError("the solid bottom shelf must stop every block")


def _width_only_stop_index(width: int, shelves: list[dict]) -> int:
    for idx, shelf in enumerate(shelves):
        gap = shelf["gap"]
        if gap is None:
            return idx
        if width > gap[1] - gap[0] + 1:
            return idx
    raise AssertionError("the solid bottom shelf must stop every block")


def _simulate(grid: np.ndarray, *, width_only: bool = False) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    shelves = _detect_shelves(g, bg)
    blocks = _detect_blocks(g, bg, shelves)

    out = g.copy()
    for block in blocks:
        rr, cc = block["coords"][:, 0], block["coords"][:, 1]
        out[rr, cc] = bg

    for block in blocks:
        r0, c0, r1, c1 = block["bbox"]
        stop_idx = _width_only_stop_index(block["w"], shelves) if width_only else _stop_index_for_span(c0, c1, shelves)
        shelf = shelves[stop_idx]
        dst_r0 = shelf["row"] - block["h"]
        dst_r1 = shelf["row"] - 1
        assert 0 <= dst_r0 <= dst_r1 < g.shape[0], "destination block bbox must stay in bounds"
        dst = out[dst_r0 : dst_r1 + 1, c0 : c1 + 1]
        assert np.all(dst == bg), "blocks may only settle into empty columns or passed shelf gaps"
        out[dst_r0 : dst_r1 + 1, c0 : c1 + 1] = shelf["color"]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop each top block straight down through shelf gaps, then recolor it by the shelf that catches it.

    Input structure:
    - The grid has a dominant background, several thin horizontal shelf lines, and a few solid rectangular blocks hanging above them.
    - Every non-bottom shelf is a single-color full-width line with one contiguous background gap; the bottom shelf is solid.
    - The blocks occupy disjoint column spans and each block is a monochromatic solid rectangle.

    Rule:
    - Keep every block's columns fixed and let it fall vertically.
    - When a block reaches a shelf, it passes through only if its entire column span lies inside that shelf's gap; otherwise it stops immediately above that shelf.
    - Repeat the same full-span gap test for lower shelves until some shelf catches the block; then erase the original block and repaint the landed block with that shelf's color.

    Key insight:
    - Passing depends on exact horizontal alignment of the whole block span with the gap, not just on the block being narrow enough, and the final color is determined by the landing shelf rather than by the input block color.
    """
    return _simulate(grid, width_only=False)


BG = 0
FG_COLORS = list(range(1, 10))
MAX_GEN_ATTEMPTS = 400


def _interval_len(interval: tuple[int, int]) -> int:
    return int(interval[1] - interval[0] + 1)


def _overlap(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int] | None:
    lo = max(a[0], b[0])
    hi = min(a[1], b[1])
    if lo > hi:
        return None
    return int(lo), int(hi)


def _sample_gap_chain(width: int, n_gapped: int) -> list[tuple[int, int]] | None:
    max_gap_width = min(width - 4, 11)
    if max_gap_width < 5:
        return None

    top_width = random.randint(max(6, n_gapped + 3), max_gap_width)
    top_start = random.randint(2, width - top_width - 2)
    gaps = [(top_start, top_start + top_width - 1)]
    intersection = gaps[0]

    for _ in range(1, n_gapped):
        candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for gap_width in range(4, max_gap_width + 1):
            for start in range(2, width - gap_width - 1):
                gap = (start, start + gap_width - 1)
                inter = _overlap(intersection, gap)
                if inter is None or _interval_len(inter) < 2:
                    continue
                if _interval_len(inter) == _interval_len(intersection):
                    continue
                candidates.append((gap, inter))
        if not candidates:
            return None
        gap, intersection = random.choice(candidates)
        gaps.append(gap)
    return gaps


def _sample_shelves() -> list[dict] | None:
    n_shelves = random.randint(3, 5)
    n_gapped = n_shelves - 1
    width = random.randint(18, 30)
    gaps = _sample_gap_chain(width, n_gapped)
    if gaps is None:
        return None

    top_row = random.randint(6, 9)
    rows = [top_row]
    for _ in range(1, n_shelves):
        rows.append(rows[-1] + random.randint(4, 6))
    height = rows[-1] + random.randint(2, 4)
    if height > 30:
        return None

    shelf_colors = random.sample(FG_COLORS, n_shelves)
    shelves = []
    for idx, row in enumerate(rows):
        gap = gaps[idx] if idx < n_gapped else None
        shelves.append({"row": int(row), "color": int(shelf_colors[idx]), "gap": gap})
    return shelves


def _candidate_spans(
    width: int,
    shelves: list[dict],
    target_idx: int,
    *,
    require_narrow_misaligned: bool = False,
) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    last = len(shelves) - 1
    for block_width in (3, 2, 4, 1):
        if block_width >= width - 1:
            continue
        for c0 in range(1, width - block_width):
            c1 = c0 + block_width - 1
            stop_idx = _stop_index_for_span(c0, c1, shelves)
            if stop_idx != target_idx:
                continue
            if require_narrow_misaligned:
                if target_idx == last:
                    continue
                gap = shelves[target_idx]["gap"]
                assert gap is not None
                if block_width > _interval_len(gap):
                    continue
                if _inside_gap(c0, c1, gap):
                    continue
            out.append((c0, c1))
    random.shuffle(out)
    return out


def _non_overlapping(span: tuple[int, int], chosen: list[tuple[int, int]]) -> bool:
    c0, c1 = span
    for old0, old1 in chosen:
        if not (c1 + 1 < old0 or old1 + 1 < c0):
            return False
    return True


def _choose_spans(shelves: list[dict]) -> list[tuple[int, int]] | None:
    last = len(shelves) - 1
    middle_target = random.randint(1, last - 1)
    roles = [
        {"target": 0, "narrow": True},
        {"target": middle_target, "narrow": False},
        {"target": last, "narrow": False},
    ]
    if random.random() < 0.65:
        extra_target = random.choice(list(range(last + 1)))
        roles.append({"target": extra_target, "narrow": extra_target < last and random.random() < 0.35})

    width = max((shelf["gap"][1] if shelf["gap"] is not None else 0) for shelf in shelves) + 3
    width = max(width, shelves[0]["gap"][1] + 3 if shelves[0]["gap"] is not None else 18)
    width = max(width, 18)
    width = min(width, 30)

    role_candidates: list[list[tuple[int, int]]] = []
    for role in roles:
        candidates = _candidate_spans(width, shelves, role["target"], require_narrow_misaligned=role["narrow"])
        if not candidates:
            return None
        role_candidates.append(candidates)

    order = sorted(range(len(roles)), key=lambda idx: len(role_candidates[idx]))
    chosen: list[tuple[int, int] | None] = [None] * len(roles)

    def backtrack(pos: int, used: list[tuple[int, int]]) -> bool:
        if pos == len(order):
            return True
        role_idx = order[pos]
        for span in role_candidates[role_idx]:
            if not _non_overlapping(span, used):
                continue
            chosen[role_idx] = span
            used.append(span)
            if backtrack(pos + 1, used):
                return True
            used.pop()
            chosen[role_idx] = None
        return False

    if not backtrack(0, []):
        return None
    return [span for span in chosen if span is not None]


def _build_grid(shelves: list[dict], spans: list[tuple[int, int]]) -> np.ndarray:
    bottom_margin_max = min(4, 30 - shelves[-1]["row"])
    if bottom_margin_max < 2:
        raise ValueError("not enough room below the bottom shelf")
    height = shelves[-1]["row"] + random.randint(2, bottom_margin_max)
    width = random.randint(max(18, max(c1 for _c0, c1 in spans) + 3), 30)
    for shelf in shelves:
        if shelf["gap"] is not None and shelf["gap"][1] >= width - 1:
            width = shelf["gap"][1] + 3
    if width > 30:
        raise ValueError("width overflow")

    grid = np.full((height, width), BG, dtype=int)
    for shelf in shelves:
        grid[shelf["row"], :] = shelf["color"]
        if shelf["gap"] is not None:
            g0, g1 = shelf["gap"]
            grid[shelf["row"], g0 : g1 + 1] = BG

    used_colors = {shelf["color"] for shelf in shelves}
    block_colors = [color for color in FG_COLORS if color not in used_colors]
    random.shuffle(block_colors)

    top_shelf_row = shelves[0]["row"]
    for idx, (c0, c1) in enumerate(spans):
        block_h = random.choice([1, 2, 2, 3, 3, 4])
        block_h = min(block_h, top_shelf_row - 2)
        if block_h < 1:
            raise ValueError("top shelf too close to top edge")
        max_r0 = top_shelf_row - block_h - 2
        if max_r0 < 0:
            raise ValueError("block does not fit above top shelf")
        r0 = random.randint(0, max_r0)
        color = block_colors[idx]
        grid[r0 : r0 + block_h, c0 : c1 + 1] = color
    return grid


def _semantic_checks(grid: np.ndarray) -> bool:
    try:
        shelves = _detect_shelves(grid, BG)
        blocks = _detect_blocks(grid, BG, shelves)
    except AssertionError:
        return False

    if len(blocks) < 3:
        return False

    true_stops = [_stop_index_for_span(block["bbox"][1], block["bbox"][3], shelves) for block in blocks]
    naive_stops = [_width_only_stop_index(block["w"], shelves) for block in blocks]
    last = len(shelves) - 1

    if len(set(true_stops)) < 3:
        return False
    if not any(stop == last for stop in true_stops):
        return False
    if not any(0 < stop < last for stop in true_stops):
        return False
    if not any(stop != naive for stop, naive in zip(true_stops, naive_stops, strict=False)):
        return False
    if not any(stop >= 2 for stop in true_stops):
        return False

    try:
        true_out = main(grid)
        naive_out = _simulate(grid, width_only=True)
    except AssertionError:
        return False

    if np.array_equal(true_out, grid):
        return False
    if np.array_equal(true_out, naive_out):
        return False
    if true_out.shape != grid.shape:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        shelves = _sample_shelves()
        if shelves is None:
            continue
        spans = _choose_spans(shelves)
        if spans is None:
            continue
        try:
            grid = _build_grid(shelves, spans)
        except ValueError:
            continue
        if _semantic_checks(grid):
            return grid
    raise RuntimeError("failed to generate a valid width-gated shelf-drop input")
