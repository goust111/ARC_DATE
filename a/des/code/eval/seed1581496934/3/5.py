from __future__ import annotations

from collections import deque

import numpy as np


BLACK = 0
YELLOW = 4
MAGENTA = 6
ORANGE = 7
PALETTE = (1, 2, 3, 5, 8, 9)

ARROW_MASKS = (
    np.array(
        [
            [0, 0, 1],
            [1, 1, 1],
            [0, 0, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1],
            [1, 0],
            [1, 1],
        ],
        dtype=bool,
    ),
)

GEN_RETRIES = 300


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        steps = ((1, 0), (-1, 0), (0, 1), (0, -1))
    else:
        steps = (
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        )

    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in steps:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _crop(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)


def _patch_key(patch: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    p = np.asarray(patch, dtype=int)
    return (tuple(map(int, p.shape)), tuple(map(int, p.flatten())))


def _vertical_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    return max(0, min(int(a[2]), int(b[2])) - max(int(a[0]), int(b[0])) + 1)


def _is_arrow_component(grid: np.ndarray, coords: np.ndarray) -> bool:
    vals = grid[coords[:, 0], coords[:, 1]]
    if not np.all(vals == YELLOW):
        return False
    b = _bbox(coords)
    patch = _crop(grid, b) == YELLOW
    for mask in ARROW_MASKS:
        if patch.shape == mask.shape and np.array_equal(patch, mask):
            return True
    return False


def _extract_samples(grid: np.ndarray, sep_row: int) -> dict[tuple[tuple[int, int], tuple[int, ...]], np.ndarray]:
    top = np.asarray(grid[:sep_row], dtype=int)
    comps = _components((top != BLACK) & (top != MAGENTA), connectivity=8)
    entries = [{"coords": comp, "bbox": _bbox(comp)} for comp in comps]

    arrows = [entry for entry in entries if _is_arrow_component(top, entry["coords"])]
    patterns = [entry for entry in entries if not _is_arrow_component(top, entry["coords"])]
    assert arrows, "expected at least one sample arrow"
    assert len(patterns) == 2 * len(arrows), "each arrow should pair exactly two patterns"

    mapping: dict[tuple[tuple[int, int], tuple[int, ...]], np.ndarray] = {}
    used_pattern_ids: set[int] = set()

    for arrow in sorted(arrows, key=lambda item: (item["bbox"][0], item["bbox"][1])):
        abox = arrow["bbox"]
        left_candidates = [
            (idx, item)
            for idx, item in enumerate(patterns)
            if idx not in used_pattern_ids
            and item["bbox"][3] < abox[1]
            and _vertical_overlap(item["bbox"], abox) > 0
        ]
        right_candidates = [
            (idx, item)
            for idx, item in enumerate(patterns)
            if idx not in used_pattern_ids
            and item["bbox"][1] > abox[3]
            and _vertical_overlap(item["bbox"], abox) > 0
        ]
        assert left_candidates and right_candidates, "failed to locate pattern pair around arrow"

        left_idx, left = min(
            left_candidates,
            key=lambda pair: (
                int(abox[1] - pair[1]["bbox"][3]),
                abs((pair[1]["bbox"][0] + pair[1]["bbox"][2]) - (abox[0] + abox[2])),
            ),
        )
        right_idx, right = min(
            right_candidates,
            key=lambda pair: (
                int(pair[1]["bbox"][1] - abox[3]),
                abs((pair[1]["bbox"][0] + pair[1]["bbox"][2]) - (abox[0] + abox[2])),
            ),
        )
        used_pattern_ids.add(int(left_idx))
        used_pattern_ids.add(int(right_idx))

        left_patch = _crop(top, left["bbox"])
        right_patch = _crop(top, right["bbox"])
        assert left_patch.shape == right_patch.shape, "sample input/output patches must share a shape"
        key = _patch_key(left_patch)
        assert key not in mapping, "duplicate sample source pattern"
        mapping[key] = right_patch.copy()

    assert len(mapping) == len(arrows)
    return mapping


def _separator_row(grid: np.ndarray) -> int:
    rows = np.flatnonzero(np.all(grid == MAGENTA, axis=1))
    assert len(rows) == 1, "expected one full-width magenta separator row"
    return int(rows[0])


def _target_bbox(grid: np.ndarray, sep_row: int) -> tuple[int, int, int, int]:
    orange = np.argwhere(grid[sep_row + 1 :] == ORANGE)
    assert orange.size > 0, "expected orange partition lines below separator"
    orange = orange.copy()
    orange[:, 0] += sep_row + 1
    return _bbox(orange)


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    sep_row = _separator_row(g)
    mapping = _extract_samples(g, sep_row)
    tr0, tc0, tr1, tc1 = _target_bbox(g, sep_row)
    target = np.asarray(g[tr0 : tr1 + 1, tc0 : tc1 + 1], dtype=int)

    out = g.copy()
    for region in _components(target != ORANGE, connectivity=4):
        vals = target[region[:, 0], region[:, 1]]
        feature_coords = region[vals != BLACK]
        assert feature_coords.size > 0, "each partition must contain one pattern"
        fr0, fc0, fr1, fc1 = _bbox(feature_coords)
        patch = np.asarray(target[fr0 : fr1 + 1, fc0 : fc1 + 1], dtype=int)
        key = _patch_key(patch)
        assert key in mapping, "region pattern did not match any sample source"
        out[tr0 + fr0 : tr0 + fr1 + 1, tc0 + fc0 : tc0 + fc1 + 1] = mapping[key]

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Learn local replacement pairs from the sample rows above the magenta divider,
    then apply the matching replacement inside every orange-partitioned region below.
    """
    return _solve(grid)


def _random_patch(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    n_colors = int(rng.integers(2, min(4, len(PALETTE)) + 1))
    colors = rng.choice(PALETTE, size=n_colors, replace=False)
    patch = rng.choice(colors, size=(h, w), replace=True).astype(int)
    for color in colors:
        if not np.any(patch == int(color)):
            rr = int(rng.integers(0, h))
            cc = int(rng.integers(0, w))
            patch[rr, cc] = int(color)
    if len(np.unique(patch)) < 2:
        patch[0, 0] = int(colors[0])
        patch[-1, -1] = int(colors[1])
    return patch


def _deranged_palette(colors: np.ndarray, rng: np.random.Generator) -> dict[int, int]:
    src = [int(v) for v in colors]
    if len(src) == 2:
        return {src[0]: src[1], src[1]: src[0]}
    for _ in range(60):
        dst = src.copy()
        rng.shuffle(dst)
        if all(a != b for a, b in zip(src, dst)):
            return {a: b for a, b in zip(src, dst)}
    dst = src[1:] + src[:1]
    return {a: b for a, b in zip(src, dst)}


def _transform_patch(left: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    geoms = (
        lambda p: p.copy(),
        lambda p: p[:, ::-1],
        lambda p: p[::-1, :],
        lambda p: p[::-1, ::-1],
    )
    colors = np.unique(left)
    mapping = _deranged_palette(colors, rng)
    for _ in range(40):
        geom = geoms[int(rng.integers(0, len(geoms)))]
        transformed = np.asarray(geom(left), dtype=int).copy()
        out = transformed.copy()
        for src, dst in mapping.items():
            out[transformed == int(src)] = int(dst)
        if not np.array_equal(out, left):
            return out
        mapping = _deranged_palette(colors, rng)
    raise RuntimeError("failed to create a distinct output patch")


def _render_arrow() -> np.ndarray:
    arrow = np.zeros(ARROW_MASKS[0].shape, dtype=int)
    arrow[ARROW_MASKS[0]] = YELLOW
    return arrow


def _build_sample_row(left: np.ndarray, right: np.ndarray, gap1: int, gap2: int) -> np.ndarray:
    arrow = _render_arrow()
    h = max(left.shape[0], arrow.shape[0], right.shape[0])
    w = left.shape[1] + gap1 + arrow.shape[1] + gap2 + right.shape[1]
    row = np.full((h, w), BLACK, dtype=int)

    lr = (h - left.shape[0]) // 2
    ar = (h - arrow.shape[0]) // 2
    rr = (h - right.shape[0]) // 2
    ac = left.shape[1] + gap1
    rc = ac + arrow.shape[1] + gap2

    row[lr : lr + left.shape[0], 0 : left.shape[1]] = left
    row[ar : ar + arrow.shape[0], ac : ac + arrow.shape[1]] = arrow
    row[rr : rr + right.shape[0], rc : rc + right.shape[1]] = right
    return row


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n_rules = int(rng.integers(2, 4))
    ph = int(rng.integers(3, 5))
    pw = int(rng.integers(3, 5))

    used_keys: set[tuple[tuple[int, int], tuple[int, ...]]] = set()
    rules: list[tuple[np.ndarray, np.ndarray]] = []
    for _ in range(n_rules):
        for _retry in range(100):
            left = _random_patch(rng, ph, pw)
            right = _transform_patch(left, rng)
            lkey = _patch_key(left)
            rkey = _patch_key(right)
            if lkey in used_keys or rkey in used_keys:
                continue
            used_keys.add(lkey)
            used_keys.add(rkey)
            rules.append((left, right))
            break
        else:
            return None

    row_order = list(range(n_rules))
    rng.shuffle(row_order)
    sample_rows = [
        _build_sample_row(rules[idx][0], rules[idx][1], int(rng.integers(2, 5)), int(rng.integers(2, 5)))
        for idx in row_order
    ]

    target_rows = 2
    min_cols = max(2, (n_rules + target_rows - 1) // target_rows)
    target_cols = int(rng.integers(min_cols, 5))

    inner_h = int(ph + rng.integers(2, 4))
    inner_w = int(pw + rng.integers(2, 4))
    box_h = target_rows * inner_h + target_rows + 1
    box_w = target_cols * inner_w + target_cols + 1
    box_in = np.full((box_h, box_w), ORANGE, dtype=int)
    box_out = np.full((box_h, box_w), ORANGE, dtype=int)

    cells = [(r, c) for r in range(target_rows) for c in range(target_cols)]
    rng.shuffle(cells)
    assignments = {cells[i]: i for i in range(n_rules)}
    for cell in cells[n_rules:]:
        assignments[cell] = int(rng.integers(0, n_rules))

    for r in range(target_rows):
        for c in range(target_cols):
            r0 = 1 + r * (inner_h + 1)
            c0 = 1 + c * (inner_w + 1)
            box_in[r0 : r0 + inner_h, c0 : c0 + inner_w] = BLACK
            box_out[r0 : r0 + inner_h, c0 : c0 + inner_w] = BLACK

            left, right = rules[int(assignments[(r, c)])]
            off_r = int(rng.integers(1, inner_h - ph))
            off_c = int(rng.integers(1, inner_w - pw))
            rr = r0 + off_r
            cc = c0 + off_c
            box_in[rr : rr + ph, cc : cc + pw] = left
            box_out[rr : rr + ph, cc : cc + pw] = right

    top_margin = int(rng.integers(1, 3))
    left_margin = int(rng.integers(1, 4))
    right_margin = int(rng.integers(1, 4))
    row_gap = int(rng.integers(1, 3))
    lower_gap = int(rng.integers(1, 3))
    bottom_margin = int(rng.integers(1, 3))
    row_starts = [int(rng.integers(1, 4)) for _ in sample_rows]
    target_start = int(rng.integers(1, 5))

    top_h = top_margin + sum(row.shape[0] for row in sample_rows) + row_gap * (len(sample_rows) - 1)
    full_h = top_h + 1 + lower_gap + box_h + bottom_margin
    full_w = max(
        max(start + row.shape[1] for start, row in zip(row_starts, sample_rows)),
        target_start + box_w,
    )
    full_w += left_margin + right_margin

    grid = np.full((full_h, full_w), BLACK, dtype=int)
    out = np.full((full_h, full_w), BLACK, dtype=int)

    cursor = top_margin
    for start, row in zip(row_starts, sample_rows):
        c0 = left_margin + start
        grid[cursor : cursor + row.shape[0], c0 : c0 + row.shape[1]] = row
        out[cursor : cursor + row.shape[0], c0 : c0 + row.shape[1]] = row
        cursor += row.shape[0] + row_gap
    sep_row = top_h
    grid[sep_row, :] = MAGENTA
    out[sep_row, :] = MAGENTA

    target_r0 = sep_row + 1 + lower_gap
    target_c0 = left_margin + target_start
    grid[target_r0 : target_r0 + box_h, target_c0 : target_c0 + box_w] = box_in
    out[target_r0 : target_r0 + box_h, target_c0 : target_c0 + box_w] = box_out

    try:
        solved = _solve(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    """
    Generate a black-background puzzle with sampled pattern-replacement rows
    above a magenta divider and orange-partitioned target regions below.
    """
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid partitioned replacement puzzle")
