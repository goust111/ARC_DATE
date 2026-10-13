from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _frame_border_cells(bbox: BBox) -> set[Coord]:
    r0, c0, r1, c1 = bbox
    cells: set[Coord] = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0 + 1, r1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _is_rectangular_frame(coords: np.ndarray) -> bool:
    bbox = _bbox_coords(coords)
    r0, c0, r1, c1 = bbox
    if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
        return False
    want = _frame_border_cells(bbox)
    have = {(int(r), int(c)) for r, c in coords}
    return have == want


def _find_frames(grid: np.ndarray, bg: int) -> tuple[int, list[dict]]:
    candidates: list[dict] = []
    for color in sorted(set(map(int, np.unique(grid))) - {bg}):
        for coords in _components_mask(grid == color):
            if _is_rectangular_frame(coords):
                candidates.append(
                    {
                        "color": int(color),
                        "coords": coords,
                        "bbox": _bbox_coords(coords),
                    }
                )
    assert candidates, "expected rectangular frame containers"
    by_color = Counter(frame["color"] for frame in candidates)
    frame_color, frame_count = max(by_color.items(), key=lambda item: (item[1], item[0]))
    assert frame_count >= 2, "expected at least two containers sharing one frame color"
    frames = [frame for frame in candidates if frame["color"] == frame_color]
    return int(frame_color), frames


def _inside_bbox(bbox: BBox, r: int, c: int) -> bool:
    r0, c0, r1, c1 = bbox
    return r0 <= r <= r1 and c0 <= c <= c1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transfer every source container's liquid into the unique empty target container, ordered by pipe length and settled bottom-up.

    Input structure:
    - The grid has one background, several same-colored rectangular frame containers, and one-cell-wide connector pipes between them.
    - Exactly one frame is empty inside; every other frame contains scattered liquid pixels of one unique color.
    - Each source frame is connected to the empty target frame by its own pipe, and the pipe lengths are all distinct.

    Rule:
    - Count the liquid pixels inside each non-empty source frame, then erase those liquid pixels while keeping every frame and pipe unchanged.
    - Match each source to its pipe, sort the sources by increasing pipe length, and pour their pixel counts into the target frame in that order.
    - The target interior fills from the bottom row upward, and within each row from left to right; one color continues until its full count is exhausted before the next color begins.

    Key insight:
    - The pipes carry only quantity, not shape: source geometry is erased, pixel counts are conserved, and pipe length alone determines arrival order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    bg = _get_bg(g)

    frame_color, frames = _find_frames(g, bg)
    assert len(frames) >= 3, "expected one target and at least two sources"

    border_owner = np.full((h, w), -1, dtype=int)
    interior_owner = np.full((h, w), -1, dtype=int)
    for idx, frame in enumerate(frames):
        r0, c0, r1, c1 = frame["bbox"]
        for r, c in frame["coords"]:
            rr = int(r)
            cc = int(c)
            assert border_owner[rr, cc] == -1
            border_owner[rr, cc] = idx
        if r1 - r0 > 1 and c1 - c0 > 1:
            assert np.all(interior_owner[r0 + 1 : r1, c0 + 1 : c1] == -1)
            interior_owner[r0 + 1 : r1, c0 + 1 : c1] = idx

    target_idx: int | None = None
    sources: dict[int, dict] = {}
    for idx, frame in enumerate(frames):
        r0, c0, r1, c1 = frame["bbox"]
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        inner_colors = sorted(set(map(int, np.unique(interior))) - {bg})
        if not inner_colors:
            assert target_idx is None, "expected a unique empty target container"
            target_idx = idx
            continue
        assert len(inner_colors) == 1, "each source container must have one liquid color"
        liquid_color = int(inner_colors[0])
        liquid_count = int(np.count_nonzero(interior == liquid_color))
        assert liquid_count > 0
        sources[idx] = {
            "frame_idx": idx,
            "bbox": frame["bbox"],
            "color": liquid_color,
            "count": liquid_count,
        }

    assert target_idx is not None, "missing empty target container"
    assert len(sources) + 1 == len(frames), "every non-target frame must be a source"

    pipes: list[dict] = []
    other_colors = sorted(set(map(int, np.unique(g))) - {bg, frame_color})
    for color in other_colors:
        for coords in _components_mask(g == color):
            owners = {int(interior_owner[int(r), int(c)]) for r, c in coords if interior_owner[int(r), int(c)] >= 0}
            if owners:
                continue
            touched: set[int] = set()
            for r, c in coords:
                rr = int(r)
                cc = int(c)
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and border_owner[nr, nc] >= 0:
                        touched.add(int(border_owner[nr, nc]))
            if len(touched) == 2:
                pipes.append({"length": int(coords.shape[0]), "frames": touched})

    pipe_length_by_source: dict[int, int] = {}
    for pipe in pipes:
        if target_idx not in pipe["frames"]:
            continue
        source_candidates = list(pipe["frames"] - {target_idx})
        if len(source_candidates) != 1:
            continue
        source_idx = int(source_candidates[0])
        assert source_idx in sources, "pipes must connect target to a source container"
        assert source_idx not in pipe_length_by_source, "each source needs exactly one pipe"
        pipe_length_by_source[source_idx] = int(pipe["length"])

    assert len(pipe_length_by_source) == len(sources), "every source must connect to the target"
    assert len(set(pipe_length_by_source.values())) == len(sources), "pipe lengths must be unique"

    ordered_sources = sorted(
        (
            {
                "bbox": src["bbox"],
                "color": src["color"],
                "count": src["count"],
                "pipe_length": pipe_length_by_source[idx],
            }
            for idx, src in sources.items()
        ),
        key=lambda item: item["pipe_length"],
    )

    out = g.copy()
    for source in ordered_sources:
        r0, c0, r1, c1 = source["bbox"]
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        interior[interior != bg] = bg

    tr0, tc0, tr1, tc1 = frames[target_idx]["bbox"]
    out[tr0 + 1 : tr1, tc0 + 1 : tc1] = bg
    fill_cells = [(r, c) for r in range(tr1 - 1, tr0, -1) for c in range(tc0 + 1, tc1)]
    total = sum(source["count"] for source in ordered_sources)
    assert total <= len(fill_cells), "target container is too small"

    pos = 0
    for source in ordered_sources:
        for _ in range(source["count"]):
            r, c = fill_cells[pos]
            out[r, c] = source["color"]
            pos += 1
    return out


_SIDES = ("left", "right", "top", "bottom")
_PIPE_LENGTHS = (2, 3, 4, 5)
_COLOR_POOL = tuple(range(1, 10))
_RETRY_LIMIT = 400


def _draw_frame(grid: np.ndarray, bbox: BBox, color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _shift_bbox(bbox: BBox, dr: int, dc: int) -> BBox:
    r0, c0, r1, c1 = bbox
    return r0 + dr, c0 + dc, r1 + dr, c1 + dc


def _render_expected(inp: np.ndarray, target_bbox: BBox, sources: list[dict], bg: int) -> np.ndarray:
    out = inp.copy()
    for source in sources:
        r0, c0, r1, c1 = source["bbox"]
        window = out[r0 + 1 : r1, c0 + 1 : c1]
        window[window == source["color"]] = bg

    tr0, tc0, tr1, tc1 = target_bbox
    out[tr0 + 1 : tr1, tc0 + 1 : tc1] = bg
    fill_cells = [(r, c) for r in range(tr1 - 1, tr0, -1) for c in range(tc0 + 1, tc1)]
    pos = 0
    for source in sorted(sources, key=lambda item: item["pipe_length"]):
        for _ in range(source["count"]):
            r, c = fill_cells[pos]
            out[r, c] = source["color"]
            pos += 1
    return out


def _scatter_cells(cells: list[Coord], count: int) -> list[Coord]:
    picked = random.sample(cells, count)
    random.shuffle(picked)
    return picked


def _allocate_counts(total: int, capacities: list[int]) -> list[int]:
    counts = [1] * len(capacities)
    remaining = total - len(capacities)
    room = [cap - 1 for cap in capacities]
    while remaining > 0:
        choices = [idx for idx, cap in enumerate(room) if cap > 0]
        if not choices:
            raise AssertionError("insufficient source capacity")
        idx = random.choice(choices)
        counts[idx] += 1
        room[idx] -= 1
        remaining -= 1
    return counts


def _make_source_specs(target_h: int, target_w: int, sides: list[str], pipe_lengths: list[int]) -> list[dict]:
    specs: list[dict] = []
    for side, pipe_length in zip(sides, pipe_lengths):
        if side in ("left", "right"):
            outer_h = random.randint(4, target_h)
            outer_w = random.randint(4, 5)
            top = random.randint(0, target_h - outer_h)
            attach_row = random.randint(top + 1, top + outer_h - 2)
            if side == "left":
                bbox = (top, -pipe_length - outer_w, top + outer_h - 1, -pipe_length - 1)
                pipe = [(attach_row, c) for c in range(bbox[3] + 1, 0)]
            else:
                bbox = (top, target_w + pipe_length, top + outer_h - 1, target_w + pipe_length + outer_w - 1)
                pipe = [(attach_row, c) for c in range(target_w, bbox[1])]
        else:
            outer_h = random.randint(4, 5)
            outer_w = random.randint(4, target_w)
            left = random.randint(0, target_w - outer_w)
            attach_col = random.randint(left + 1, left + outer_w - 2)
            if side == "top":
                bbox = (-pipe_length - outer_h, left, -pipe_length - 1, left + outer_w - 1)
                pipe = [(r, attach_col) for r in range(bbox[2] + 1, 0)]
            else:
                bbox = (target_h + pipe_length, left, target_h + pipe_length + outer_h - 1, left + outer_w - 1)
                pipe = [(r, attach_col) for r in range(target_h, bbox[0])]
        specs.append(
            {
                "side": side,
                "bbox": bbox,
                "pipe": pipe,
                "pipe_length": int(pipe_length),
                "capacity": int((bbox[2] - bbox[0] - 1) * (bbox[3] - bbox[1] - 1)),
            }
        )
    return specs


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    n_sources = random.randint(2, 4)
    sides = random.sample(_SIDES, n_sources)
    pipe_lengths = random.sample(_PIPE_LENGTHS, n_sources)

    target_inner_h = random.randint(4, 6)
    target_inner_w = random.randint(5, 7)
    target_h = target_inner_h + 2
    target_w = target_inner_w + 2
    target_bbox = (0, 0, target_h - 1, target_w - 1)

    sources = _make_source_specs(target_h, target_w, sides, pipe_lengths)
    source_caps = [source["capacity"] for source in sources]
    target_cap = target_inner_h * target_inner_w
    max_total = min(sum(source_caps), target_cap - 1)
    if max_total < n_sources:
        return None

    totals = list(range(n_sources, max_total + 1))
    weights = [4 if total > target_inner_w else 1 for total in totals]
    fill_total = random.choices(totals, weights=weights, k=1)[0]
    counts = _allocate_counts(fill_total, source_caps)

    colors = random.sample(_COLOR_POOL, n_sources + 2)
    frame_color = colors[0]
    pipe_color = colors[1]
    liquid_colors = colors[2:]

    all_boxes = [target_bbox] + [source["bbox"] for source in sources]
    all_pipe_cells = [cell for source in sources for cell in source["pipe"]]
    min_r = min([box[0] for box in all_boxes] + [r for r, _ in all_pipe_cells])
    max_r = max([box[2] for box in all_boxes] + [r for r, _ in all_pipe_cells])
    min_c = min([box[1] for box in all_boxes] + [c for _, c in all_pipe_cells])
    max_c = max([box[3] for box in all_boxes] + [c for _, c in all_pipe_cells])

    pad_top = random.randint(1, 2)
    pad_bottom = random.randint(1, 2)
    pad_left = random.randint(1, 2)
    pad_right = random.randint(1, 2)
    dr = pad_top - min_r
    dc = pad_left - min_c
    height = max_r + dr + pad_bottom + 1
    width = max_c + dc + pad_right + 1
    if height > 30 or width > 30:
        return None

    grid = np.full((height, width), bg, dtype=int)
    shifted_target = _shift_bbox(target_bbox, dr, dc)
    _draw_frame(grid, shifted_target, frame_color)

    placed_sources: list[dict] = []
    for source, count, liquid_color in zip(sources, counts, liquid_colors):
        shifted_bbox = _shift_bbox(source["bbox"], dr, dc)
        _draw_frame(grid, shifted_bbox, frame_color)
        for r, c in source["pipe"]:
            grid[r + dr, c + dc] = pipe_color

        r0, c0, r1, c1 = shifted_bbox
        cells = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
        liquid_cells = _scatter_cells(cells, count)
        for r, c in liquid_cells:
            grid[r, c] = liquid_color

        placed_sources.append(
            {
                "bbox": shifted_bbox,
                "count": int(count),
                "color": int(liquid_color),
                "pipe_length": int(source["pipe_length"]),
            }
        )

    expected = _render_expected(grid, shifted_target, placed_sources, bg)

    if len(set(source["pipe_length"] for source in placed_sources)) != len(placed_sources):
        return None
    if sum(source["count"] for source in placed_sources) >= target_cap:
        return None
    if any(np.count_nonzero(grid[s["bbox"][0] + 1 : s["bbox"][2], s["bbox"][1] + 1 : s["bbox"][3]] == s["color"]) != s["count"] for s in placed_sources):
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid pipe-transfer ARC input")
