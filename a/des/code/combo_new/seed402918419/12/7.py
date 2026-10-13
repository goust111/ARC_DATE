import numpy as np


BG = 0
STAMP_COLOR = 5
MOSAIC_POOL = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)
GEN_RETRIES = 300
RNG = np.random.default_rng()


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        seen[rr, cc] = True
        stack = [(rr, cc)]
        pts: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            pts.append((int(cr), int(cc)))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr = int(cr + dr)
                nc = int(cc + dc)
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(points: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(points[:, 0].min()),
        int(points[:, 1].min()),
        int(points[:, 0].max()),
        int(points[:, 1].max()),
    )


def _dominant_color(window: np.ndarray) -> tuple[int, bool]:
    counts: dict[int, int] = {}
    first: dict[int, tuple[int, int]] = {}
    for r in range(window.shape[0]):
        for c in range(window.shape[1]):
            color = int(window[r, c])
            counts[color] = counts.get(color, 0) + 1
            if color not in first:
                first[color] = (int(r), int(c))

    best_color = -1
    best_count = -1
    best_pos = (window.shape[0], window.shape[1])
    for color, count in counts.items():
        pos = first[color]
        if count > best_count or (count == best_count and pos < best_pos):
            best_color = int(color)
            best_count = int(count)
            best_pos = pos
    tied = sum(count == best_count for count in counts.values()) > 1
    return best_color, tied


def _find_mosaic_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    mask = (grid != BG) & (grid != STAMP_COLOR)
    pts = np.argwhere(mask)
    assert len(pts) > 0, "grid must contain a non-grey mosaic"
    r0 = int(pts[:, 0].min())
    c0 = int(pts[:, 1].min())
    r1 = int(pts[:, 0].max())
    c1 = int(pts[:, 1].max())
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert bool(((patch != BG) & (patch != STAMP_COLOR)).all())
    return r0, c0, r1, c1


def _stamp_components(grid: np.ndarray) -> list[np.ndarray]:
    stamps: list[np.ndarray] = []
    for pts in _components(grid == STAMP_COLOR):
        r0, c0, r1, c1 = _bbox(pts)
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        if h == w and h in (2, 3) and len(pts) == h * w:
            stamps.append(pts)
    stamps.sort(key=_bbox)
    assert stamps, "grid must contain at least one grey square stamp"
    return stamps


def _sampling_window(
    grid: np.ndarray,
    mosaic_bbox: tuple[int, int, int, int],
    stamp_bbox: tuple[int, int, int, int],
) -> np.ndarray:
    mr0, mc0, mr1, mc1 = mosaic_bbox
    sr0, sc0, sr1, sc1 = stamp_bbox
    rr0 = max(mr0, sr0)
    rr1 = min(mr1, sr1)
    if rr0 <= rr1:
        return grid[rr0 : rr1 + 1, mc0 : mc1 + 1]
    cc0 = max(mc0, sc0)
    cc1 = min(mc1, sc1)
    assert cc0 <= cc1, "stamp must align with the mosaic in rows or columns"
    return grid[mr0 : mr1 + 1, cc0 : cc1 + 1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each grey stamp by the dominant mosaic color sampled along its aligned rows or columns.

    Input structure:
    - The background is black.
    - One large rectangular mosaic sits near the center and is fully filled with non-black, non-grey colors.
    - Several solid grey 2x2 or 3x3 square stamps lie on the surrounding black area.
    - Every stamp overlaps the mosaic in either row range or column range, but not by direct contact.

    Rule:
    - Find the large colored mosaic rectangle and each grey square stamp.
    - For a stamp whose rows overlap the mosaic, sample the mosaic cells in those overlapping rows across the mosaic's full width.
    - Otherwise sample the mosaic cells in the overlapping columns across the mosaic's full height.
    - Replace the entire stamp with the most frequent sampled color, breaking ties by the earliest top-left occurrence inside the sampling window.

    Key insight:
    - The grey squares are spatial pointers: their position selects a row-slice or column-slice of the mosaic, and that slice is compressed to one representative color by majority counting.
    """
    g = np.asarray(grid, dtype=int)
    mosaic_bbox = _find_mosaic_bbox(g)
    out = g.copy()

    for stamp in _stamp_components(g):
        stamp_bbox = _bbox(stamp)
        window = _sampling_window(g, mosaic_bbox, stamp_bbox)
        fill, _tied = _dominant_color(window)
        out[stamp[:, 0], stamp[:, 1]] = int(fill)

    return out


def _split_length(total: int, parts: int, min_size: int) -> list[int]:
    assert parts * min_size <= total
    lengths = np.full(parts, min_size, dtype=int)
    for _ in range(total - parts * min_size):
        lengths[int(RNG.integers(parts))] += 1
    RNG.shuffle(lengths)
    return [int(x) for x in lengths]


def _random_partition(total: int, min_parts: int, max_parts: int, min_size: int) -> list[int]:
    hi = min(max_parts, total // min_size)
    lo = min(min_parts, hi)
    parts = int(RNG.integers(lo, hi + 1))
    return _split_length(total, parts, min_size)


def _color_layout(nr: int, nc: int, colors: np.ndarray) -> np.ndarray | None:
    min_colors = min(3, len(colors))
    for _ in range(200):
        layout = RNG.choice(colors, size=(nr, nc), replace=True)
        if len(set(map(int, layout.flat))) < min_colors:
            continue
        if any(len(set(map(int, layout[r, :]))) < 2 for r in range(nr)):
            continue
        if any(len(set(map(int, layout[:, c]))) < 2 for c in range(nc)):
            continue
        changes = 0
        for r in range(nr):
            for c in range(nc - 1):
                changes += int(layout[r, c] != layout[r, c + 1])
        for r in range(nr - 1):
            for c in range(nc):
                changes += int(layout[r, c] != layout[r + 1, c])
        if changes < nr + nc:
            continue
        return layout.astype(int)
    return None


def _make_mosaic(h: int, w: int) -> np.ndarray | None:
    row_sizes = _random_partition(h, 3, 5, 2)
    col_sizes = _random_partition(w, 3, 5, 2)
    palette_size = int(RNG.integers(3, 6))
    colors = RNG.choice(MOSAIC_POOL, size=palette_size, replace=False)
    layout = _color_layout(len(row_sizes), len(col_sizes), colors)
    if layout is None:
        return None

    mosaic = np.empty((h, w), dtype=int)
    rr = 0
    for r, rh in enumerate(row_sizes):
        cc = 0
        for c, cw in enumerate(col_sizes):
            mosaic[rr : rr + rh, cc : cc + cw] = int(layout[r, c])
            cc += cw
        rr += rh

    if len(set(map(int, mosaic[0, :]))) < 2 or len(set(map(int, mosaic[:, 0]))) < 2:
        return None
    return mosaic


def _rects_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _enumerate_candidates(
    shape: tuple[int, int], mosaic_bbox: tuple[int, int, int, int]
) -> list[dict[str, object]]:
    h, w = shape
    mr0, mc0, mr1, mc1 = mosaic_bbox
    out: list[dict[str, object]] = []

    for size in (2, 3):
        row_lo = max(0, mr0 - size + 1)
        row_hi = min(h - size, mr1)
        col_lo = max(0, mc0 - size + 1)
        col_hi = min(w - size, mc1)

        max_left_c = mc0 - size - 1
        if row_lo <= row_hi and max_left_c >= 0:
            for r in range(row_lo, row_hi + 1):
                for c in range(0, max_left_c + 1):
                    out.append(
                        {
                            "bbox": (int(r), int(c), int(r + size - 1), int(c + size - 1)),
                            "side": "left",
                            "branch": "row",
                            "size": int(size),
                        }
                    )

        min_right_c = mc1 + 2
        if row_lo <= row_hi and min_right_c <= w - size:
            for r in range(row_lo, row_hi + 1):
                for c in range(min_right_c, w - size + 1):
                    out.append(
                        {
                            "bbox": (int(r), int(c), int(r + size - 1), int(c + size - 1)),
                            "side": "right",
                            "branch": "row",
                            "size": int(size),
                        }
                    )

        max_top_r = mr0 - size - 1
        if col_lo <= col_hi and max_top_r >= 0:
            for r in range(0, max_top_r + 1):
                for c in range(col_lo, col_hi + 1):
                    out.append(
                        {
                            "bbox": (int(r), int(c), int(r + size - 1), int(c + size - 1)),
                            "side": "top",
                            "branch": "col",
                            "size": int(size),
                        }
                    )

        min_bottom_r = mr1 + 2
        if col_lo <= col_hi and min_bottom_r <= h - size:
            for r in range(min_bottom_r, h - size + 1):
                for c in range(col_lo, col_hi + 1):
                    out.append(
                        {
                            "bbox": (int(r), int(c), int(r + size - 1), int(c + size - 1)),
                            "side": "bottom",
                            "branch": "col",
                            "size": int(size),
                        }
                    )

    return out


def _candidate_with_stats(
    grid: np.ndarray, mosaic_bbox: tuple[int, int, int, int], cand: dict[str, object]
) -> dict[str, object]:
    bbox = cand["bbox"]
    assert isinstance(bbox, tuple)
    window = _sampling_window(grid, mosaic_bbox, bbox)
    dominant, tied = _dominant_color(window)
    enriched = dict(cand)
    enriched["dominant"] = int(dominant)
    enriched["tied"] = bool(tied)
    return enriched


def _pick_candidate(
    pool: list[dict[str, object]],
    selected: list[dict[str, object]],
    prefer=None,
) -> dict[str, object] | None:
    order = RNG.permutation(len(pool))
    fallback = None
    for idx in order:
        cand = pool[int(idx)]
        bbox = cand["bbox"]
        assert isinstance(bbox, tuple)
        if any(_rects_close(bbox, other["bbox"]) for other in selected):
            continue
        if prefer is None or prefer(cand):
            return cand
        if fallback is None:
            fallback = cand
    return fallback


def _select_stamps(candidates: list[dict[str, object]]) -> list[dict[str, object]] | None:
    row_cands = [cand for cand in candidates if cand["branch"] == "row"]
    col_cands = [cand for cand in candidates if cand["branch"] == "col"]
    if not row_cands or not col_cands:
        return None

    selected: list[dict[str, object]] = []
    first_pool, second_pool = (row_cands, col_cands) if RNG.random() < 0.5 else (col_cands, row_cands)

    first = _pick_candidate(first_pool, selected)
    if first is None:
        return None
    selected.append(first)

    second = _pick_candidate(
        second_pool,
        selected,
        prefer=lambda cand: cand["dominant"] != first["dominant"],
    )
    if second is None:
        return None
    selected.append(second)

    want_tie = RNG.random() < 0.45
    if want_tie:
        tie_cands = [cand for cand in candidates if cand["tied"]]
        extra = _pick_candidate(tie_cands, selected)
        if extra is not None:
            selected.append(extra)

    target = int(RNG.integers(2, 7))
    while len(selected) < target:
        used_sides = {str(cand["side"]) for cand in selected}
        used_colors = {int(cand["dominant"]) for cand in selected}
        extra = _pick_candidate(
            candidates,
            selected,
            prefer=lambda cand: str(cand["side"]) not in used_sides
            or int(cand["dominant"]) not in used_colors,
        )
        if extra is None:
            break
        if any(extra["bbox"] == old["bbox"] for old in selected):
            break
        selected.append(extra)

    if len(selected) < 2:
        return None
    if {str(cand["branch"]) for cand in selected} != {"row", "col"}:
        return None
    return selected


def _gen_attempt() -> np.ndarray | None:
    top = int(RNG.integers(4, 8))
    bottom = int(RNG.integers(4, 8))
    left = int(RNG.integers(4, 8))
    right = int(RNG.integers(4, 8))

    max_h = 30 - top - bottom
    max_w = 30 - left - right
    if max_h < 8 or max_w < 8:
        return None

    mosaic_h = int(RNG.integers(8, min(17, max_h) + 1))
    mosaic_w = int(RNG.integers(8, min(17, max_w) + 1))
    mosaic = _make_mosaic(mosaic_h, mosaic_w)
    if mosaic is None:
        return None

    h = top + mosaic_h + bottom
    w = left + mosaic_w + right
    grid = np.full((h, w), BG, dtype=int)
    mr0, mc0 = top, left
    mr1, mc1 = mr0 + mosaic_h - 1, mc0 + mosaic_w - 1
    mosaic_bbox = (mr0, mc0, mr1, mc1)
    grid[mr0 : mr1 + 1, mc0 : mc1 + 1] = mosaic

    candidates = [
        _candidate_with_stats(grid, mosaic_bbox, cand)
        for cand in _enumerate_candidates(grid.shape, mosaic_bbox)
    ]
    selected = _select_stamps(candidates)
    if selected is None:
        return None

    for cand in selected:
        r0, c0, r1, c1 = cand["bbox"]
        grid[r0 : r1 + 1, c0 : c1 + 1] = STAMP_COLOR

    if len(set(int(cand["dominant"]) for cand in selected)) < 2 and RNG.random() < 0.7:
        return None

    out = main(grid)
    for cand in selected:
        r0, c0, r1, c1 = cand["bbox"]
        fill = int(cand["dominant"])
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        if not bool((patch == fill).all()):
            return None

    if max(grid.shape) > 30 or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
