import itertools
import random

import numpy as np


_CORRECTIONS = (-1, 0, 1)
_BG = 0
_GEN_RETRIES = 400


def _neighbors(r: int, c: int, connectivity: int) -> list[tuple[int, int]]:
    if connectivity == 4:
        return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
    return [
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if not (dr == 0 and dc == 0)
    ]


def _components(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors(rr, cc, connectivity):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _colors_present(grid: np.ndarray) -> list[int]:
    return sorted(int(x) for x in np.unique(grid) if int(x) != _BG)


def _interval_rows_from_mask(mask: np.ndarray) -> list[tuple[int, int]]:
    rows = []
    for r in range(mask.shape[0]):
        cols = np.flatnonzero(mask[r])
        rows.append((int(cols.min()), int(cols.max())))
    return rows


def _shear_mask(mask: np.ndarray, shear: int) -> np.ndarray:
    coords = np.argwhere(mask)
    bottom = mask.shape[0] - 1
    shifted = coords[:, 1] + int(shear) * (bottom - coords[:, 0])
    min_c = int(shifted.min())
    max_c = int(shifted.max())
    out = np.zeros((mask.shape[0], max_c - min_c + 1), dtype=bool)
    out[coords[:, 0], shifted - min_c] = True
    return out


def _candidate_from_coords(coords: np.ndarray, correction: int) -> dict:
    rows = coords[:, 0].astype(int)
    cols = coords[:, 1].astype(int)
    bottom = int(rows.max())
    y = bottom - rows
    x = cols + int(correction) * y
    bottom_x = x[y == 0]
    anchor_left = int(bottom_x.min())
    anchored_x = x - anchor_left
    points = frozenset((int(yy), int(xx)) for yy, xx in zip(y, anchored_x))
    xs = [pt[1] for pt in points]
    ys = [pt[0] for pt in points]
    return {
        "correction": int(correction),
        "points": points,
        "pixel_count": len(points),
        "width": max(xs) - min(xs) + 1,
        "height": max(ys) - min(ys) + 1,
        "bbox_area": (max(xs) - min(xs) + 1) * (max(ys) - min(ys) + 1),
        "bottom_span": int(bottom_x.max() - bottom_x.min()),
    }


def _local_candidate_score(candidate: dict) -> tuple[int, int]:
    return candidate["width"], candidate["bbox_area"]


def _best_combo(candidates_by_component: list[list[dict]]) -> tuple[tuple[dict, ...], int]:
    filtered = []
    for candidates in candidates_by_component:
        best_local = min(_local_candidate_score(candidate) for candidate in candidates)
        filtered.append(
            [candidate for candidate in candidates if _local_candidate_score(candidate) == best_local]
        )

    best_combo = None
    best_score = None
    ties = 0
    for combo in itertools.product(*filtered):
        if sum(candidate["correction"] == 0 for candidate in combo) != 1:
            continue
        union = set()
        for candidate in combo:
            union.update(candidate["points"])
        xs = [pt[1] for pt in union]
        ys = [pt[0] for pt in union]
        score = (
            -len(union),
            (max(xs) - min(xs) + 1) * (max(ys) - min(ys) + 1),
        )
        if best_score is None or score < best_score:
            best_score = score
            best_combo = combo
            ties = 1
        elif score == best_score:
            ties += 1
    assert best_combo is not None, "expected at least one valid normalization combo"
    return best_combo, ties


def _render_points(points: set[tuple[int, int]] | frozenset[tuple[int, int]], color: int) -> np.ndarray:
    xs = [pt[1] for pt in points]
    ys = [pt[0] for pt in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    out = np.zeros((max_y - min_y + 1, max_x - min_x + 1), dtype=int)
    for y, x in points:
        out[max_y - y, x - min_x] = int(color)
    return out


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    colors = _colors_present(g)
    assert len(colors) >= 3, "expected several differently colored copies"

    candidates_by_component = []
    upright_color = None
    for color in colors:
        mask = g == int(color)
        comps = _components(mask, connectivity=8)
        assert len(comps) == 1, "each color must form exactly one separated block"
        coords = comps[0]
        candidates = [_candidate_from_coords(coords, correction) for correction in _CORRECTIONS]
        candidates_by_component.append(candidates)

    combo, ties = _best_combo(candidates_by_component)
    assert ties == 1, "normalization must be uniquely determined"

    union = set()
    for color, candidate in zip(colors, combo):
        union.update(candidate["points"])
        if candidate["correction"] == 0:
            assert upright_color is None, "expected exactly one upright copy"
            upright_color = int(color)

    assert upright_color is not None
    return _render_points(union, upright_color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Undo the mixed left/right shears on matching fragments, union them, and emit the full upright template.

    Input structure:
    - A black background contains several separated foreground blocks in distinct colors.
    - Every block is a damaged copy of the same hidden template.
    - Exactly one copy is upright; every other copy is sheared left or right around its own bottom row.
    - Each copy still keeps the two endpoint pixels of its bottom row.

    Rule:
    - For each colored block, test the three possible normalizations: already upright, unshear-left, or unshear-right.
    - Keep only the normalization(s) that make each individual fragment as tight as possible in its upright frame.
    - Among those locally tight candidates, choose the unique combination with exactly one already-upright copy whose union restores the most template pixels.
    - Union the restored upright fragments in that shared template frame.
    - Output the union's minimal bounding box, colored with the upright copy's color.

    Key insight:
    - The visible pieces look unrelated only because their rows have been shifted relative to a shared bottom anchor; once the correct inverse shear is chosen for each copy, all fragments snap into one common sprite and their missing parts complement one another.
    """
    return _solve(grid)


def _generate_template() -> np.ndarray | None:
    height = random.randint(4, 6)
    width = random.randint(6, 8)
    rows_bottom_up = [(0, width - 1)]
    prev_l, prev_r = rows_bottom_up[0]
    for _ in range(height - 1):
        row_width = random.randint(5, width)
        starts = [
            start
            for start in range(width - row_width + 1)
            if start <= prev_r and start + row_width - 1 >= prev_l
        ]
        if not starts:
            return None
        smooth = [start for start in starts if abs(start - prev_l) <= 2]
        start = random.choice(smooth or starts)
        rows_bottom_up.append((start, start + row_width - 1))
        prev_l, prev_r = rows_bottom_up[-1]

    rows = list(reversed(rows_bottom_up))
    mask = np.zeros((height, width), dtype=bool)
    for r, (left, right) in enumerate(rows):
        mask[r, left : right + 1] = True

    if np.array_equal(mask, np.fliplr(mask)):
        return None
    if len({row for row in rows}) < 2:
        return None
    return mask


def _row_sections(row_width: int) -> tuple[int, int]:
    valid = [
        (p1, p2)
        for p1 in range(1, row_width - 2)
        for p2 in range(p1 + 2, row_width - 1)
    ]
    return random.choice(valid)


def _generate_fragments(template: np.ndarray) -> list[np.ndarray] | None:
    height, width = template.shape
    rows = _interval_rows_from_mask(template)
    fragments = [np.zeros((height, width), dtype=bool) for _ in range(3)]
    for fragment in fragments:
        fragment[-1] = template[-1]

    for r, (left, right) in enumerate(rows[:-1]):
        row_width = right - left + 1
        if row_width < 5:
            return None
        p1, p2 = _row_sections(row_width)
        spans = [
            (left, left + p1),
            (left + p1, left + p2),
            (left + p2, right),
        ]
        for fragment, (span_left, span_right) in zip(fragments, spans):
            fragment[r, span_left : span_right + 1] = True

    for fragment in fragments:
        if np.array_equal(fragment, template):
            return None
        if len(_components(fragment, connectivity=8)) != 1:
            return None
        if any(not fragment[r].any() for r in range(height)):
            return None

    union = fragments[0] | fragments[1] | fragments[2]
    if not np.array_equal(union, template):
        return None

    for idx, fragment in enumerate(fragments):
        others = np.zeros_like(template, dtype=bool)
        for j, other in enumerate(fragments):
            if j != idx:
                others |= other
        if not np.any(fragment & ~others):
            return None
    return fragments


def _patch_from_mask(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = int(color)
    return patch


def _render_input(patches: list[np.ndarray]) -> np.ndarray | None:
    max_h = max(patch.shape[0] for patch in patches)
    max_w = max(patch.shape[1] for patch in patches)
    min_h = max(12, max_h + 6)
    min_w = max(16, max_w + 8)
    for _ in range(80):
        height = random.randint(min_h, min(30, max(min_h, 20)))
        width = random.randint(min_w, min(30, max(min_w, 28)))
        grid = np.zeros((height, width), dtype=int)
        occupied = []
        ok = True
        for patch in sorted(patches, key=lambda arr: arr.shape[0] * arr.shape[1], reverse=True):
            ph, pw = patch.shape
            candidates = [(r, c) for r in range(1, height - ph) for c in range(1, width - pw)]
            random.shuffle(candidates)
            placed = False
            for r0, c0 in candidates:
                bbox = (r0 - 1, c0 - 1, r0 + ph, c0 + pw)
                if any(
                    not (
                        bbox[2] < old[0]
                        or old[2] < bbox[0]
                        or bbox[3] < old[1]
                        or old[3] < bbox[1]
                    )
                    for old in occupied
                ):
                    continue
                grid[r0 : r0 + ph, c0 : c0 + pw][patch != 0] = patch[patch != 0]
                occupied.append((r0, c0, r0 + ph - 1, c0 + pw - 1))
                placed = True
                break
            if not placed:
                ok = False
                break
        if ok:
            return grid
    return None


def _expected_output(template: np.ndarray, upright_color: int) -> np.ndarray:
    out = np.zeros(template.shape, dtype=int)
    out[template] = int(upright_color)
    return out


def _true_combo_matches(inp: np.ndarray, colors: list[int], true_corrections: list[int]) -> bool:
    candidates_by_component = []
    for color in colors:
        coords = _components(inp == int(color), connectivity=8)[0]
        candidates_by_component.append(
            [_candidate_from_coords(coords, correction) for correction in _CORRECTIONS]
        )
    combo, ties = _best_combo(candidates_by_component)
    if ties != 1:
        return False
    actual = [candidate["correction"] for candidate in combo]
    return actual == true_corrections


def _raw_union_differs(inp: np.ndarray, expected: np.ndarray) -> bool:
    colors = _colors_present(inp)
    union = set()
    for color in colors:
        coords = _components(inp == int(color), connectivity=8)[0]
        candidate = _candidate_from_coords(coords, 0)
        union.update(candidate["points"])
    raw = _render_points(union, 1)
    expected_mask = expected != 0
    raw_mask = raw != 0
    return not np.array_equal(raw_mask, expected_mask)


def _attempt() -> np.ndarray | None:
    template = _generate_template()
    if template is None:
        return None
    fragments = _generate_fragments(template)
    if fragments is None:
        return None

    shears = [0, -1, 1]
    random.shuffle(shears)
    corrections = [-shear for shear in shears]
    colors = random.sample(list(range(1, 10)), 3)
    patches = []
    upright_color = None
    for fragment, shear, color in zip(fragments, shears, colors):
        observed = _shear_mask(fragment, shear)
        if len(_components(observed, connectivity=8)) != 1:
            return None
        if shear == 0:
            upright_color = color
        patches.append(_patch_from_mask(observed, color))

    assert upright_color is not None
    inp = _render_input(patches)
    if inp is None:
        return None

    expected = _expected_output(template, upright_color)
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    if not _true_combo_matches(inp, colors, corrections):
        return None
    if not _raw_union_differs(inp, expected):
        return None
    if any(np.array_equal(_shear_mask(fragment, shear), fragment) for fragment, shear in zip(fragments, shears) if shear != 0):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid shear-normalization input")
