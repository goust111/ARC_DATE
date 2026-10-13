from __future__ import annotations

from collections import Counter
import itertools

import numpy as np


def _run_starts_and_counts(lines: np.ndarray) -> tuple[list[int], np.ndarray]:
    starts = [0]
    counts: list[int] = []
    count = 1
    for idx in range(1, lines.shape[0]):
        if np.array_equal(lines[idx], lines[idx - 1]):
            count += 1
        else:
            counts.append(count)
            starts.append(idx)
            count = 1
    counts.append(count)
    return starts, np.array(counts, dtype=int)


def _compress_grid(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    row_starts, row_runs = _run_starts_and_counts(grid)
    row_compressed = grid[row_starts, :]
    col_starts, col_runs = _run_starts_and_counts(row_compressed.T)
    compressed = row_compressed[:, col_starts]
    return compressed, row_runs, col_runs


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _find_wallpaper_colors(grid: np.ndarray) -> tuple[int, int]:
    counts = Counter(map(int, grid.flat))
    colors = [int(color) for color, _ in counts.most_common(2)]
    assert len(colors) == 2, "expected exactly two wallpaper colors to dominate the compressed grid"
    return colors[0], colors[1]


def _extract_rectangles(grid: np.ndarray, wallpaper_colors: tuple[int, int]) -> list[dict]:
    wallpaper = set(map(int, wallpaper_colors))
    comps = _components(~np.isin(grid, list(wallpaper)))
    items: list[dict] = []
    for coords in comps:
        colors = {int(grid[r, c]) for r, c in coords}
        assert len(colors) == 1, "foreground component must be monochrome"
        color = next(iter(colors))
        r0, c0, r1, c1 = _bbox(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        local = grid[r0 : r1 + 1, c0 : c1 + 1] == color
        if np.all(local):
            kind = "solid"
        else:
            border = np.zeros_like(local, dtype=bool)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True
            assert np.array_equal(local, border), "foreground component must be a solid rectangle or a hollow frame"
            kind = "frame"
        items.append(
            {
                "color": int(color),
                "kind": kind,
                "h": int(h),
                "w": int(w),
                "area": int(h * w),
                "r0": int(r0),
                "c0": int(c0),
                "r1": int(r1),
                "c1": int(c1),
            }
        )
    items.sort(key=lambda item: item["area"], reverse=True)
    areas = [item["area"] for item in items]
    assert len(items) >= 3 and len(areas) == len(set(areas)), "need 3-4 rectangles with unique compressed bbox areas"
    return items


def _usable_region(item: dict, top: int, left: int) -> tuple[int, int, int, int]:
    if item["kind"] == "frame":
        return int(top + 1), int(left + 1), int(item["h"] - 2), int(item["w"] - 2)
    return int(top), int(left), int(item["h"]), int(item["w"])


def _draw_item(canvas: np.ndarray, item: dict, top: int, left: int, clip_right: int) -> None:
    h, w, color = int(item["h"]), int(item["w"]), int(item["color"])
    for dr in range(h):
        rr = top + dr
        if rr < 0 or rr >= canvas.shape[0]:
            continue
        for dc in range(w):
            cc = left + dc
            if cc < 0 or cc >= canvas.shape[1] or cc > clip_right:
                continue
            if item["kind"] == "solid" or dr in (0, h - 1) or dc in (0, w - 1):
                canvas[rr, cc] = color


def _render_nested(items: list[dict]) -> np.ndarray:
    outer = items[0]
    canvas = np.zeros((int(outer["h"]), int(outer["w"])), dtype=int)
    top = 0
    left = 0
    _draw_item(canvas, outer, top, left, outer["w"] - 1)
    prev_top, prev_left = top, left

    for idx in range(1, len(items)):
        parent = items[idx - 1]
        item = items[idx]
        ur, uc, uh, uw = _usable_region(parent, prev_top, prev_left)
        assert item["h"] <= uh and item["w"] <= uw, "child must fit in the parent's unshifted usable region"
        top = int(ur + uh - item["h"])
        left = int(uc + item["offset"])
        clip_right = int(uc + uw - 1)
        _draw_item(canvas, item, top, left, clip_right)
        prev_top, prev_left = top, left

    return canvas


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress a wallpaper grid, read row-duplication offsets, and rebuild the rectangles as a shifted nesting.

    Input structure:
    - The grid is a 2-color wallpaper made of black/gray periodic cells, overlaid by 3-4 non-wallpaper rectangles.
    - Each foreground rectangle is either a solid block or a hollow rectangular frame with a unique color.
    - The whole scene has been expanded by repeating adjacent identical rows and columns.

    Rule:
    - Collapse every maximal run of identical neighboring rows and columns, keeping only the first row/column of each run.
    - In the compressed grid, ignore the two wallpaper colors and detect each monochrome rectangular component.
    - Sort those rectangles by compressed bbox area from largest to smallest, create a black canvas the size of the largest one,
      then place the chain largest-first: the largest at the top-left, each later rectangle bottom-aligned in the previous
      rectangle's usable lower-left corner, but shifted right by the sum of removed duplicate rows across that rectangle's
      original compressed row span; pixels beyond the parent's usable right edge are clipped.

    Key insight:
    - Compression is not just cleanup: the dropped row repetitions encode the horizontal offsets needed for the nesting layout.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    compressed, row_runs, _col_runs = _compress_grid(g)
    wallpaper = _find_wallpaper_colors(compressed)
    items = _extract_rectangles(compressed, wallpaper)
    for item in items:
        item["offset"] = int(np.sum(row_runs[item["r0"] : item["r1"] + 1] - 1))
    offsets = [item["offset"] for item in items]
    assert len(offsets) == len(set(offsets)) and max(offsets) <= 3, "row-duplication offsets must be unique and small"
    return _render_nested(items)


_RNG = np.random.default_rng()
_FG_COLORS = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)
_GEN_ATTEMPTS = 600


def _choice(rng: np.random.Generator, seq):
    return seq[int(rng.integers(len(seq)))]


def _usable_dims(kind: str, h: int, w: int) -> tuple[int, int]:
    if kind == "frame":
        return h - 2, w - 2
    return h, w


def _build_items(rng: np.random.Generator) -> list[dict] | None:
    count = int(rng.integers(3, 5))
    colors = list(map(int, rng.choice(_FG_COLORS, size=count, replace=False)))

    kinds = ["frame"]
    for idx in range(1, count):
        if idx == count - 1:
            kinds.append("solid")
        else:
            kinds.append(_choice(rng, ["solid", "frame"]))
    if "solid" not in kinds[1:]:
        kinds[-1] = "solid"

    outer_h = int(rng.integers(8, 12 + 1))
    outer_w = int(rng.integers(8, 12 + 1))
    items = [{"kind": "frame", "h": outer_h, "w": outer_w, "area": outer_h * outer_w}]
    used_areas = {outer_h * outer_w}

    def rec(parent: dict, kind_seq: list[str], acc: list[dict]) -> list[dict] | None:
        if not kind_seq:
            return acc
        ph, pw = _usable_dims(parent["kind"], int(parent["h"]), int(parent["w"]))
        kind = kind_seq[0]
        h_min = 3 if kind == "frame" else 2
        w_min = 3 if kind == "frame" else 2
        if ph < h_min or pw < w_min:
            return None

        candidates = [
            (h, w)
            for h in range(h_min, ph + 1)
            for w in range(w_min, pw + 1)
            if h * w < parent["area"] and h * w not in used_areas
        ]
        rng.shuffle(candidates)
        for h, w in candidates:
            item = {"kind": kind, "h": int(h), "w": int(w), "area": int(h * w)}
            uh, uw = _usable_dims(kind, int(h), int(w))
            if len(kind_seq) > 1:
                next_kind = kind_seq[1]
                need_h = 3 if next_kind == "frame" else 2
                need_w = 3 if next_kind == "frame" else 2
                if uh < need_h or uw < need_w:
                    continue
            used_areas.add(item["area"])
            out = rec(item, kind_seq[1:], acc + [item])
            if out is not None:
                return out
            used_areas.remove(item["area"])
        return None

    inner = rec(items[0], kinds[1:], [])
    if inner is None:
        return None
    full = items + inner
    for item, color in zip(full, colors):
        item["color"] = int(color)
    return full


def _place_items(rng: np.random.Generator, h: int, w: int, items: list[dict]) -> list[tuple[int, int]] | None:
    placements: list[tuple[int, int]] = []
    occupied = np.zeros((h, w), dtype=bool)
    for item in items:
        ih, iw = int(item["h"]), int(item["w"])
        candidates = [
            (r, c)
            for r in range(1, h - ih)
            for c in range(1, w - iw)
            if not occupied[r - 1 : r + ih + 1, c - 1 : c + iw + 1].any()
        ]
        if not candidates:
            return None
        rng.shuffle(candidates)
        r, c = candidates[0]
        placements.append((int(r), int(c)))
        occupied[r - 1 : r + ih + 1, c - 1 : c + iw + 1] = True
    return placements


def _row_top_candidates(height: int, grid_h: int, marks: tuple[int, int, int], offset: int) -> list[int]:
    return [
        top
        for top in range(1, grid_h - height)
        if sum(int(top <= mark < top + height) for mark in marks) == offset
    ]


def _separated(candidate: tuple[int, int, int, int], placed: list[tuple[int, int, int, int]]) -> bool:
    r0, c0, r1, c1 = candidate
    for pr0, pc0, pr1, pc1 in placed:
        if not (r1 + 1 < pr0 or pr1 + 1 < r0 or c1 + 1 < pc0 or pc1 + 1 < c0):
            return False
    return True


def _place_with_offsets(
    rng: np.random.Generator,
    grid_h: int,
    grid_w: int,
    items: list[dict],
    row_options: list[list[int]],
) -> list[tuple[int, int]] | None:
    placed: list[tuple[int, int, int, int]] = []
    answer: list[tuple[int, int]] = []

    def rec(idx: int) -> bool:
        if idx == len(items):
            return True
        item = items[idx]
        ih, iw = int(item["h"]), int(item["w"])
        positions = [
            (top, left)
            for top in row_options[idx]
            for left in range(1, grid_w - iw)
        ]
        rng.shuffle(positions)
        for top, left in positions:
            bbox = (int(top), int(left), int(top + ih - 1), int(left + iw - 1))
            if not _separated(bbox, placed):
                continue
            placed.append(bbox)
            answer.append((int(top), int(left)))
            if rec(idx + 1):
                return True
            placed.pop()
            answer.pop()
        return False

    return answer if rec(0) else None


def _wallpaper(h: int, w: int, phase: int) -> np.ndarray:
    rows = np.arange(h)[:, None]
    cols = np.arange(w)[None, :]
    return np.where(((rows + cols + phase) % 2) == 0, 0, 5).astype(int)


def _paint_input_rect(grid: np.ndarray, item: dict, top: int, left: int) -> None:
    h, w, color = int(item["h"]), int(item["w"]), int(item["color"])
    if item["kind"] == "solid":
        grid[top : top + h, left : left + w] = color
        return
    grid[top, left : left + w] = color
    grid[top + h - 1, left : left + w] = color
    grid[top : top + h, left] = color
    grid[top : top + h, left + w - 1] = color


def _adjacent_unique(grid: np.ndarray) -> bool:
    for r in range(grid.shape[0] - 1):
        if np.array_equal(grid[r], grid[r + 1]):
            return False
    for c in range(grid.shape[1] - 1):
        if np.array_equal(grid[:, c], grid[:, c + 1]):
            return False
    return True


def _offsets_from_rows(row_runs: np.ndarray, items: list[dict]) -> list[int]:
    return [
        int(np.sum(row_runs[int(item["r0"]) : int(item["r1"]) + 1] - 1))
        for item in items
    ]


def _clip_occurs(items: list[dict], offsets: list[int]) -> bool:
    placed = [(0, 0)]
    for idx in range(1, len(items)):
        parent = items[idx - 1]
        pr, pc = placed[idx - 1]
        ur, uc, _uh, uw = _usable_region(parent, pr, pc)
        if int(items[idx]["w"]) + int(offsets[idx]) > int(uw):
            return True
        top = int(ur + _uh - int(items[idx]["h"]))
        left = int(uc + int(offsets[idx]))
        placed.append((top, left))
    return False


def _stretch_bbox_area(item: dict, row_runs: np.ndarray, col_runs: np.ndarray) -> int:
    hh = int(np.sum(row_runs[int(item["r0"]) : int(item["r1"]) + 1]))
    ww = int(np.sum(col_runs[int(item["c0"]) : int(item["c1"]) + 1]))
    return int(hh * ww)


def _render_with_offsets(items: list[dict], offsets: list[int]) -> np.ndarray:
    copied = [{**item, "offset": int(offset)} for item, offset in zip(items, offsets)]
    return _render_nested(copied)


def _safe_render(items: list[dict], offsets: list[int]) -> np.ndarray | None:
    try:
        return _render_with_offsets(items, offsets)
    except AssertionError:
        return None


def _semantic_checks(
    base: np.ndarray,
    items: list[dict],
    row_runs: np.ndarray,
    col_runs: np.ndarray,
    stretched: np.ndarray,
) -> bool:
    compressed, got_row_runs, got_col_runs = _compress_grid(stretched)
    if not np.array_equal(base, compressed):
        return False
    if not np.array_equal(row_runs, got_row_runs) or not np.array_equal(col_runs, got_col_runs):
        return False

    wallpaper = _find_wallpaper_colors(base)
    extracted = _extract_rectangles(base, wallpaper)
    if [(it["kind"], it["h"], it["w"], it["color"]) for it in extracted] != [
        (it["kind"], it["h"], it["w"], it["color"]) for it in items
    ]:
        return False

    offsets = _offsets_from_rows(row_runs, items)
    if len(offsets) != len(set(offsets)) or max(offsets) > 3 or not any(offset > 0 for offset in offsets[1:]):
        return False

    expected = _render_with_offsets(items, offsets)
    if not np.array_equal(main(stretched), expected):
        return False

    if np.array_equal(_render_with_offsets(items, [0] * len(items)), expected):
        return False

    stretched_order = sorted(
        range(len(items)),
        key=lambda idx: _stretch_bbox_area(items[idx], row_runs, col_runs),
        reverse=True,
    )
    if stretched_order == list(range(len(items))):
        return False

    wrong_items = [items[idx] for idx in stretched_order]
    wrong_offsets = [offsets[idx] for idx in stretched_order]
    wrong_output = _safe_render(wrong_items, wrong_offsets)
    if wrong_output is not None and np.array_equal(wrong_output, expected):
        return False

    if max(stretched.shape) > 30 or max(expected.shape) > 30:
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    want_clip = bool(rng.integers(2))
    items = _build_items(rng)
    if items is None:
        return None

    min_h = max(12, int(max(item["h"] for item in items)) + 2)
    min_w = max(13, int(max(item["w"] for item in items)) + 2)
    h = int(rng.integers(min_h, 16 + 1))
    w = int(rng.integers(min_w, 18 + 1))
    if h < 6:
        return None

    mark_start = int(rng.integers(2, h - 3))
    marks = (mark_start, mark_start + 1, mark_start + 2)
    row_runs = np.ones(h, dtype=int)
    row_runs[list(marks)] = 2

    offset_perms = list(itertools.permutations(range(4), len(items)))
    rng.shuffle(offset_perms)
    for offsets in offset_perms:
        if not any(offset > 0 for offset in offsets[1:]):
            continue
        if _clip_occurs(items, list(offsets)) != want_clip:
            continue
        row_options = [
            _row_top_candidates(int(item["h"]), h, marks, int(offset))
            for item, offset in zip(items, offsets)
        ]
        if any(not options for options in row_options):
            continue

        placements = _place_with_offsets(rng, h, w, items, row_options)
        if placements is None:
            continue

        base = _wallpaper(h, w, int(rng.integers(2)))
        for item, (top, left) in zip(items, placements):
            item["r0"] = int(top)
            item["c0"] = int(left)
            item["r1"] = int(top + item["h"] - 1)
            item["c1"] = int(left + item["w"] - 1)
            _paint_input_rect(base, item, top, left)

        if not _adjacent_unique(base):
            continue

        counts = Counter(map(int, base.flat))
        if counts[0] <= max(counts[int(item["color"])] for item in items):
            continue
        if counts[5] <= max(counts[int(item["color"])] for item in items):
            continue

        boost_order = list(range(1, len(items)))
        rng.shuffle(boost_order)
        for boost_idx in boost_order:
            cols = list(range(int(items[boost_idx]["c0"]), int(items[boost_idx]["c1"]) + 1))
            subsets = []
            for size in range(len(cols), 0, -1):
                for combo in itertools.combinations(cols, size):
                    subsets.append(combo)
            rng.shuffle(subsets)
            for boosted_cols in subsets[: max(40, min(200, len(subsets)))]:
                col_runs = np.ones(w, dtype=int)
                col_runs[list(boosted_cols)] = 2
                stretched = np.repeat(np.repeat(base, row_runs, axis=0), col_runs, axis=1)
                if _semantic_checks(base, items, row_runs, col_runs, stretched):
                    return stretched

    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 10")
