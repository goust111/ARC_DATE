from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
SEP = 5
PALETTE = (1, 2, 3, 4)
TARGET_COUNTS = (4, 6, 8)
RECT_HEIGHT = 2
AREA_TO_WIDTH = {4: 2, 6: 3, 8: 4}
QUADRANT_HEIGHTS = (5, 6, 7)
QUADRANT_WIDTHS = (9, 10)
MAX_TRIES = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_cross(grid: np.ndarray) -> tuple[int, int]:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == SEP)]
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == SEP)]
    assert len(rows) == 1, "expected one solid separator row"
    assert len(cols) == 1, "expected one solid separator col"
    return rows[0], cols[0]


def _region_slices(cross_r: int, cross_c: int, h: int, w: int) -> list[tuple[slice, slice]]:
    return [
        (slice(0, cross_r), slice(0, cross_c)),
        (slice(0, cross_r), slice(cross_c + 1, w)),
        (slice(cross_r + 1, h), slice(0, cross_c)),
        (slice(cross_r + 1, h), slice(cross_c + 1, w)),
    ]


def _extract_rectangles(region: np.ndarray) -> list[dict[str, int]]:
    mask = region != BG
    h, w = region.shape
    seen = np.zeros_like(mask, dtype=bool)
    rects: list[dict[str, int]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            color = int(region[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and region[nr, nc] == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            rows = [rr for rr, _ in cells]
            cols = [cc for _, cc in cells]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            area = (r1 - r0 + 1) * (c1 - c0 + 1)
            assert len(cells) == area, "foreground objects must be solid rectangles"
            assert np.all(region[r0 : r1 + 1, c0 : c1 + 1] == color)
            rects.append({"r0": r0, "r1": r1, "c0": c0, "c1": c1, "color": color, "area": area})
    return rects


def _overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) <= min(a1, b1)


def _gap_candidates(region: np.ndarray, rects: list[dict[str, int]]) -> list[dict[str, int]]:
    out: list[dict[str, int]] = []
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            a = rects[i]
            b = rects[j]
            if a["c1"] < b["c0"]:
                left, right = a, b
            elif b["c1"] < a["c0"]:
                left, right = b, a
            else:
                continue
            row0 = max(left["r0"], right["r0"])
            row1 = min(left["r1"], right["r1"])
            if row0 > row1:
                continue
            gap0 = left["c1"] + 1
            gap1 = right["c0"] - 1
            if gap0 > gap1:
                continue
            blocked = False
            for k, blocker in enumerate(rects):
                if k in (i, j):
                    continue
                if _overlap(gap0, gap1, blocker["c0"], blocker["c1"]) and _overlap(
                    row0, row1, blocker["r0"], blocker["r1"]
                ):
                    blocked = True
                    break
            if blocked:
                continue
            assert np.all(region[row0 : row1 + 1, gap0 : gap1 + 1] == BG), "gap must be empty"
            out.append(
                {
                    "r0": row0,
                    "r1": row1,
                    "c0": gap0,
                    "c1": gap1,
                    "area": (row1 - row0 + 1) * (gap1 - gap0 + 1),
                }
            )
    return out


def _color_counts(region: np.ndarray) -> Counter[int]:
    vals = [int(v) for v in region.reshape(-1) if int(v) not in (BG, SEP)]
    return Counter(vals)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each quadrant's unique horizontal rectangle gap with the color needed to balance all quadrant multisets.

    Input structure:
    - A grey full row and a grey full column split the black background into four rectangular regions.
    - Each region contains 2 or 3 solid colored rectangles drawn with non-grey foreground colors.
    - In every region, exactly one pair of rectangles is horizontally separated, overlaps in rows, and has an unobstructed gap between their bounding boxes.

    Rule:
    - Find the separator cross and inspect each of the four regions independently.
    - In each region, detect the unique fillable horizontal gap as the shared-row-span rectangle between the two qualifying object boxes.
    - Count non-background, non-grey pixels by color in every region; for each color, the target count is the maximum count seen among the four regions.
    - Each region is short by exactly one color, and that deficit equals the local gap area; fill the entire gap with that missing color.

    Key insight:
    - Gap geometry is local, but the fill color is determined only by comparing color multisets across all four regions, so both constraints are necessary.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    cross_r, cross_c = _find_cross(g)
    regions = _region_slices(cross_r, cross_c, *g.shape)

    records = []
    for rs, cs in regions:
        region = g[rs, cs]
        rects = _extract_rectangles(region)
        assert 2 <= len(rects) <= 3, "each region must contain 2 or 3 rectangles"
        gaps = _gap_candidates(region, rects)
        assert len(gaps) == 1, "each region must have exactly one fillable gap"
        records.append({"slice": (rs, cs), "counts": _color_counts(region), "gap": gaps[0]})

    all_colors = sorted({color for rec in records for color in rec["counts"]})
    target = {color: max(rec["counts"].get(color, 0) for rec in records) for color in all_colors}
    out = g.copy()
    for rec in records:
        deficits = {color: target[color] - rec["counts"].get(color, 0) for color in all_colors}
        missing = [(color, need) for color, need in deficits.items() if need > 0]
        assert len(missing) == 1, "each region must need exactly one fill color"
        fill_color, need = missing[0]
        assert need == rec["gap"]["area"], "gap area must match the unique color deficit"
        rs, cs = rec["slice"]
        gap = rec["gap"]
        out[
            rs.start + gap["r0"] : rs.start + gap["r1"] + 1,
            cs.start + gap["c0"] : cs.start + gap["c1"] + 1,
        ] = fill_color
    return out


def _draw_rect(region: np.ndarray, top: int, left: int, color: int, area: int) -> None:
    width = AREA_TO_WIDTH[area]
    region[top : top + RECT_HEIGHT, left : left + width] = color


def _build_region(
    h: int,
    w: int,
    target_by_color: dict[int, int],
    missing_color: int,
    use_three_rects: bool,
) -> np.ndarray:
    region = np.zeros((h, w), dtype=int)
    existing = [color for color in target_by_color if color != missing_color]
    gap_width = target_by_color[missing_color] // RECT_HEIGHT
    pair_top = random.choice([0, h - RECT_HEIGHT])
    other_top = h - RECT_HEIGHT if pair_top == 0 else 0

    if use_three_rects:
        split_color = next(color for color in existing if target_by_color[color] == 8)
        other_color = next(color for color in existing if color != split_color)
        if random.random() < 0.5:
            pair_specs = [(split_color, 4), (split_color, 4)]
            third_spec = (other_color, target_by_color[other_color])
        else:
            if random.random() < 0.5:
                pair_specs = [(split_color, 4), (other_color, target_by_color[other_color])]
            else:
                pair_specs = [(other_color, target_by_color[other_color]), (split_color, 4)]
            third_spec = (split_color, 4)
    else:
        pair_specs = [(existing[0], target_by_color[existing[0]]), (existing[1], target_by_color[existing[1]])]
        random.shuffle(pair_specs)
        third_spec = None

    pair_width = AREA_TO_WIDTH[pair_specs[0][1]] + gap_width + AREA_TO_WIDTH[pair_specs[1][1]]
    left0 = random.randint(0, w - pair_width)
    left1 = left0
    left2 = left0 + AREA_TO_WIDTH[pair_specs[0][1]] + gap_width
    _draw_rect(region, pair_top, left1, pair_specs[0][0], pair_specs[0][1])
    _draw_rect(region, pair_top, left2, pair_specs[1][0], pair_specs[1][1])

    if third_spec is not None:
        third_area = third_spec[1]
        third_width = AREA_TO_WIDTH[third_area]
        choices = list(range(0, w - third_width + 1))
        random.shuffle(choices)
        third_left = choices[0]
        _draw_rect(region, other_top, third_left, third_spec[0], third_area)

    return region


def _build_input() -> np.ndarray:
    qh = random.choice(QUADRANT_HEIGHTS)
    qw = random.choice(QUADRANT_WIDTHS)
    colors = random.sample(PALETTE, 3)
    counts = list(TARGET_COUNTS)
    random.shuffle(counts)
    target_by_color = {color: count for color, count in zip(colors, counts)}
    repeated_missing = random.choice(colors)
    missing_colors = colors + [repeated_missing]
    random.shuffle(missing_colors)

    color_with_eight = next(color for color, count in target_by_color.items() if count == 8)
    candidate_three = [i for i, color in enumerate(missing_colors) if color != color_with_eight]
    use_three = [False] * 4
    if candidate_three:
        for idx in random.sample(candidate_three, random.randint(1, min(2, len(candidate_three)))):
            use_three[idx] = True

    regions = [
        _build_region(qh, qw, target_by_color, missing_colors[i], use_three[i]) for i in range(4)
    ]

    grid = np.zeros((2 * qh + 1, 2 * qw + 1), dtype=int)
    cross_r, cross_c = qh, qw
    grid[cross_r, :] = SEP
    grid[:, cross_c] = SEP
    slices = _region_slices(cross_r, cross_c, *grid.shape)
    for region, (rs, cs) in zip(regions, slices):
        grid[rs, cs] = region
    return grid


def _validate_instance(grid: np.ndarray) -> None:
    cross_r, cross_c = _find_cross(grid)
    assert cross_r == grid.shape[0] // 2 and cross_c == grid.shape[1] // 2
    regions = _region_slices(cross_r, cross_c, *grid.shape)
    records = []
    for rs, cs in regions:
        region = grid[rs, cs]
        rects = _extract_rectangles(region)
        assert 2 <= len(rects) <= 3
        gaps = _gap_candidates(region, rects)
        assert len(gaps) == 1
        records.append({"slice": (rs, cs), "gap": gaps[0], "counts": _color_counts(region)})

    all_colors = sorted({color for rec in records for color in rec["counts"]})
    assert len(all_colors) == 3
    target = {color: max(rec["counts"].get(color, 0) for rec in records) for color in all_colors}
    assert sorted(target.values()) == list(TARGET_COUNTS)

    expected = grid.copy()
    changed = set()
    for rec in records:
        deficits = {color: target[color] - rec["counts"].get(color, 0) for color in all_colors}
        missing = [(color, need) for color, need in deficits.items() if need > 0]
        assert len(missing) == 1
        fill_color, need = missing[0]
        gap = rec["gap"]
        assert need == gap["area"]
        rs, cs = rec["slice"]
        for r in range(gap["r0"], gap["r1"] + 1):
            for c in range(gap["c0"], gap["c1"] + 1):
                expected[rs.start + r, cs.start + c] = fill_color
                changed.add((rs.start + r, cs.start + c))

    out = main(grid)
    assert np.array_equal(out, expected)
    diff = {(int(r), int(c)) for r, c in np.argwhere(out != grid)}
    assert diff == changed

    out_regions = [out[rs, cs] for rs, cs in regions]
    out_counts = [_color_counts(region) for region in out_regions]
    for counts in out_counts:
        assert {color: counts.get(color, 0) for color in all_colors} == target


def generate_input() -> np.ndarray:
    for _ in range(MAX_TRIES):
        grid = _build_input()
        try:
            _validate_instance(grid)
            return grid
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid instance")
