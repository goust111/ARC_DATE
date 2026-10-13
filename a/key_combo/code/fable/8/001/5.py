from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
PALETTE = tuple(range(1, 10))
GENERATION_ATTEMPTS = 400
GROUP_COUNT_BAG: list[int] = []


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.unique(grid)))
    if exclude:
        colors -= set(exclude)
    return sorted(colors)


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _paste(
    dst: np.ndarray,
    patch: np.ndarray,
    r0: int,
    c0: int,
    *,
    transparent: int | None = None,
) -> np.ndarray:
    r0 = int(r0)
    c0 = int(c0)
    h, w = patch.shape
    if r0 < 0 or c0 < 0 or r0 + h > dst.shape[0] or c0 + w > dst.shape[1]:
        raise ValueError("paste out of bounds")
    view = dst[r0 : r0 + h, c0 : c0 + w]
    if transparent is None:
        view[:] = patch
    else:
        mask = patch != int(transparent)
        view[mask] = patch[mask]
    return dst


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _count_components(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    count = 0
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        count += 1
        q = deque([(r, c)])
        seen[r, c] = True
        while q:
            rr, cc = q.popleft()
            for nr, nc in _neighbors4(rr, cc):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
    return count


def _is_isolated_foreground(grid: np.ndarray, coord: tuple[int, int]) -> bool:
    r, c = coord
    h, w = grid.shape
    for rr in range(max(0, r - 1), min(h, r + 2)):
        for cc in range(max(0, c - 1), min(w, c + 2)):
            if (rr, cc) != (r, c) and int(grid[rr, cc]) != BLACK:
                return False
    return True


def _find_group_for_color(grid: np.ndarray, color: int) -> dict | None:
    points = [tuple(map(int, rc)) for rc in np.argwhere(grid == color)]
    if len(points) < 6:
        return None
    point_set = set(points)
    rows = sorted({r for r, _ in points})
    cols = sorted({c for _, c in points})
    candidates = []

    for rt in rows:
        for rb in rows:
            if rb - rt < 2:
                continue
            for cl in cols:
                for cr in cols:
                    if cr - cl < 2:
                        continue
                    markers = ((rt, cl), (rt, cr), (rb, cl), (rb, cr))
                    if not all(p in point_set for p in markers):
                        continue

                    r0, c0 = rt + 1, cl + 1
                    r1, c1 = rb - 1, cr - 1
                    anchor_cells = [
                        p for p in points if r0 <= p[0] <= r1 and c0 <= p[1] <= c1
                    ]
                    if len(anchor_cells) != 1:
                        continue
                    anchor = anchor_cells[0]

                    used = set(markers)
                    used.add(anchor)
                    leftovers = [p for p in points if p not in used]
                    if len(leftovers) != 1:
                        continue
                    target = leftovers[0]
                    if not _is_isolated_foreground(grid, target):
                        continue

                    pattern = _crop_bbox(grid, (r0, c0, r1, c1))
                    if np.count_nonzero(pattern) < 2:
                        continue

                    candidates.append(
                        {
                            "color": int(color),
                            "window": (r0, c0, r1, c1),
                            "markers": tuple(markers),
                            "anchor": anchor,
                            "anchor_offset": (anchor[0] - r0, anchor[1] - c0),
                            "target": target,
                            "pattern": pattern,
                        }
                    )

    if not candidates:
        return None

    unique = []
    seen = set()
    for cand in candidates:
        key = (
            cand["window"],
            cand["anchor"],
            cand["target"],
            tuple(sorted(cand["markers"])),
        )
        if key not in seen:
            seen.add(key)
            unique.append(cand)
    assert len(unique) == 1, f"color {color} must determine one window/anchor/target triple"
    return unique[0]


def _find_groups(grid: np.ndarray) -> list[dict]:
    groups = []
    for color in _colors_present(grid, exclude={BLACK}):
        group = _find_group_for_color(grid, color)
        if group is not None:
            groups.append(group)
    assert groups, "expected at least one marker color"
    return groups


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each hidden-window pattern so its embedded same-color anchor lands on the far isolated target pixel.

    Input structure:
    - The grid is a black background containing one or more groups with distinct control colors.
    - In each group, four same-color pixels sit just outside the diagonal corners of one axis-aligned rectangle, defining an implicit window.
    - Inside that window is a multicolor pattern whose cells include exactly one pixel of the same control color, plus one isolated faraway pixel of that control color.

    Rule:
    - Infer each implicit window from its four outer corner markers and crop the full window interior pattern.
    - Translate that whole cropped pattern so its embedded same-color anchor pixel lands exactly on the isolated same-color target pixel.
    - Erase the source window contents, erase the four markers, and keep the translated pattern at its new location.

    Key insight:
    - One color plays three linked roles at once: outer corner markers reveal the hidden crop, the inner same-color pixel is the anchor, and the lone far pixel is the delivery destination.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert np.any(g != BLACK), "input must contain foreground"

    groups = _find_groups(g)
    out = g.copy()

    for group in groups:
        r0, c0, r1, c1 = group["window"]
        out[r0 : r1 + 1, c0 : c1 + 1] = BLACK
        for mr, mc in group["markers"]:
            out[mr, mc] = BLACK
        tr, tc = group["target"]
        out[tr, tc] = BLACK

    occupied: set[tuple[int, int]] = set()
    for group in groups:
        r0, c0, r1, c1 = group["window"]
        ar, ac = group["anchor"]
        tr, tc = group["target"]
        pattern = group["pattern"]
        h, w = pattern.shape
        dr = tr - ar
        dc = tc - ac
        nr0 = r0 + dr
        nc0 = c0 + dc
        assert 0 <= nr0 and 0 <= nc0 and nr0 + h <= g.shape[0] and nc0 + w <= g.shape[1]

        for pr, pc in np.argwhere(pattern != BLACK):
            rr = nr0 + int(pr)
            cc = nc0 + int(pc)
            assert out[rr, cc] == BLACK, "translated pattern must land on cleared background"
            assert (rr, cc) not in occupied, "translated patterns must not overlap"
            occupied.add((rr, cc))

        _paste(out, pattern, nr0, nc0, transparent=BLACK)

    return out


def _sample_group_count() -> int:
    global GROUP_COUNT_BAG
    if not GROUP_COUNT_BAG:
        GROUP_COUNT_BAG = [1, 2, 3, 2, 3, 1]
        random.shuffle(GROUP_COUNT_BAG)
    return GROUP_COUNT_BAG.pop()


def _bbox_empty(mask: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 0) -> bool:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(mask.shape[0], r0 + h + pad)
    cc1 = min(mask.shape[1], c0 + w + pad)
    return not bool(mask[rr0:rr1, cc0:cc1].any())


def _mark_bbox(mask: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 0) -> None:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(mask.shape[0], r0 + h + pad)
    cc1 = min(mask.shape[1], c0 + w + pad)
    mask[rr0:rr1, cc0:cc1] = True


def _make_pattern(h: int, w: int, anchor_color: int, fill_colors: list[int]) -> tuple[np.ndarray, tuple[int, int]]:
    cells = [(r, c) for r in range(h) for c in range(w) if (r, c) not in {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}]
    for _ in range(200):
        anchor = random.choice(cells)
        occupied = {anchor}
        component_count = random.randint(2, min(4, max(2, h * w - 1)))
        seeds = random.sample(
            [p for p in cells if p != anchor],
            k=min(component_count, len(cells) - 1),
        )
        for seed in seeds:
            occupied.add(seed)
            growth_steps = random.randint(0, 2)
            frontier = [seed]
            for _ in range(growth_steps):
                base = random.choice(frontier)
                options = []
                for nr, nc in _neighbors4(*base):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) != anchor:
                        options.append((nr, nc))
                if not options:
                    continue
                nxt = random.choice(options)
                occupied.add(nxt)
                frontier.append(nxt)

        if len(occupied) < 4:
            continue
        occ = np.zeros((h, w), dtype=bool)
        for r, c in occupied:
            occ[r, c] = True
        if _count_components(occ) < 2:
            continue
        coords = np.argwhere(occ)
        r0, c0, r1, c1 = _bbox_coords(coords)
        if r0 == 0 and r1 == h - 1:
            continue
        if c0 == 0 and c1 == w - 1:
            continue
        if max(abs(r - anchor[0]) + abs(c - anchor[1]) for r, c in occupied if (r, c) != anchor) < 2:
            continue

        pattern = np.zeros((h, w), dtype=int)
        pattern[anchor] = anchor_color
        other_cells = [p for p in occupied if p != anchor]
        random.shuffle(other_cells)
        distinct = random.randint(2, min(4, len(fill_colors)))
        chosen_colors = random.sample(fill_colors, k=distinct)
        for idx, (r, c) in enumerate(other_cells):
            pattern[r, c] = chosen_colors[idx % distinct]

        if len(set(int(v) for v in pattern[pattern != BLACK])) < 3:
            continue
        return pattern, anchor
    raise RuntimeError("failed to sample a non-degenerate pattern")


def _render_input(shape: tuple[int, int], specs: list[dict]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for spec in specs:
        r0, c0 = spec["source_top"]
        h, w = spec["pattern"].shape
        _paste(grid, spec["pattern"], r0, c0, transparent=BLACK)
        color = spec["color"]
        markers = ((r0 - 1, c0 - 1), (r0 - 1, c0 + w), (r0 + h, c0 - 1), (r0 + h, c0 + w))
        for mr, mc in markers:
            grid[mr, mc] = color
        tr, tc = spec["target"]
        grid[tr, tc] = color
    return grid


def _render_output(shape: tuple[int, int], specs: list[dict]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for spec in specs:
        ar, ac = spec["anchor"]
        tr, tc = spec["target"]
        nr0 = tr - ar
        nc0 = tc - ac
        _paste(out, spec["pattern"], nr0, nc0, transparent=BLACK)
    return out


def _placement_options(
    occupied: np.ndarray, h: int, w: int, require_markers: bool
) -> list[tuple[int, int]]:
    positions = []
    margin = 2 if require_markers else 1
    row_stop = occupied.shape[0] - h - margin + 1
    col_stop = occupied.shape[1] - w - margin + 1
    for r0 in range(margin, row_stop):
        for c0 in range(margin, col_stop):
            if require_markers:
                ok = _bbox_empty(occupied, r0 - 1, c0 - 1, h + 2, w + 2, pad=1)
            else:
                ok = _bbox_empty(occupied, r0, c0, h, w, pad=1)
            if ok:
                positions.append((r0, c0))
    return positions


def _validate_anchor_is_unique(pattern: np.ndarray, anchor: tuple[int, int], target: tuple[int, int], shape: tuple[int, int]) -> bool:
    correct = np.zeros(shape, dtype=int)
    dest_r0 = target[0] - anchor[0]
    dest_c0 = target[1] - anchor[1]
    _paste(correct, pattern, dest_r0, dest_c0, transparent=BLACK)

    for rr, cc in np.argwhere(pattern != BLACK):
        rr = int(rr)
        cc = int(cc)
        if (rr, cc) == anchor:
            continue
        alt_r0 = target[0] - rr
        alt_c0 = target[1] - cc
        if alt_r0 < 0 or alt_c0 < 0 or alt_r0 + pattern.shape[0] > shape[0] or alt_c0 + pattern.shape[1] > shape[1]:
            continue
        alt = np.zeros(shape, dtype=int)
        _paste(alt, pattern, alt_r0, alt_c0, transparent=BLACK)
        if np.array_equal(alt, correct):
            return False
    return True


def _validate_instance(inp: np.ndarray, expected: np.ndarray, specs: list[dict]) -> bool:
    try:
        solved = main(inp)
    except Exception:
        return False
    if not np.array_equal(solved, expected):
        return False

    detected = {group["color"]: group for group in _find_groups(inp)}
    if set(detected) != {spec["color"] for spec in specs}:
        return False

    for spec in specs:
        color = spec["color"]
        group = detected[color]
        h, w = spec["pattern"].shape
        src_r0, src_c0 = spec["source_top"]
        if group["window"] != (src_r0, src_c0, src_r0 + h - 1, src_c0 + w - 1):
            return False
        if group["anchor"] != (src_r0 + spec["anchor"][0], src_c0 + spec["anchor"][1]):
            return False
        if group["target"] != spec["target"]:
            return False
        if np.count_nonzero(inp == color) != 6:
            return False
        if not _is_isolated_foreground(inp, spec["target"]):
            return False
        nonzero_mask = spec["pattern"] != BLACK
        if _count_components(nonzero_mask) < 2:
            return False
        bbox = _bbox_coords(np.argwhere(nonzero_mask))
        if bbox[0] == 0 and bbox[2] == h - 1:
            return False
        if bbox[1] == 0 and bbox[3] == w - 1:
            return False
        if not _validate_anchor_is_unique(spec["pattern"], spec["anchor"], spec["target"], inp.shape):
            return False

        dst_r0 = spec["target"][0] - spec["anchor"][0]
        dst_c0 = spec["target"][1] - spec["anchor"][1]
        dest_view = inp[dst_r0 : dst_r0 + h, dst_c0 : dst_c0 + w]
        if np.count_nonzero(dest_view) != 1:
            return False
    return True


def _attempt_generate() -> np.ndarray | None:
    height = random.randint(18, 28)
    width = random.randint(18, 28)
    group_count = _sample_group_count()
    control_colors = random.sample(PALETTE, k=group_count)
    fill_colors = [color for color in PALETTE if color not in control_colors]

    occupied = np.zeros((height, width), dtype=bool)
    specs: list[dict] = []

    for color in control_colors:
        h = random.randint(4, 8)
        w = random.randint(4, 8)
        pattern, anchor = _make_pattern(h, w, color, fill_colors)

        source_options = _placement_options(occupied, h, w, require_markers=True)
        if not source_options:
            return None
        random.shuffle(source_options)

        placed = False
        for src_r0, src_c0 in source_options[:80]:
            src_center = (src_r0 + h // 2, src_c0 + w // 2)
            temp_occupied = occupied.copy()
            _mark_bbox(temp_occupied, src_r0 - 1, src_c0 - 1, h + 2, w + 2, pad=1)
            dest_options = _placement_options(temp_occupied, h, w, require_markers=False)
            if not dest_options:
                continue
            random.shuffle(dest_options)
            for dst_r0, dst_c0 in dest_options[:120]:
                if max(abs(dst_r0 - src_r0), abs(dst_c0 - src_c0)) < 4:
                    continue
                dst_center = (dst_r0 + h // 2, dst_c0 + w // 2)
                if abs(dst_center[0] - src_center[0]) + abs(dst_center[1] - src_center[1]) < 7:
                    continue
                target = (dst_r0 + anchor[0], dst_c0 + anchor[1])
                if not _bbox_empty(temp_occupied, dst_r0, dst_c0, h, w, pad=1):
                    continue

                spec = {
                    "color": color,
                    "pattern": pattern,
                    "anchor": anchor,
                    "source_top": (src_r0, src_c0),
                    "target": target,
                }
                specs.append(spec)
                _mark_bbox(occupied, src_r0 - 1, src_c0 - 1, h + 2, w + 2, pad=1)
                _mark_bbox(occupied, dst_r0, dst_c0, h, w, pad=1)
                placed = True
                break
            if placed:
                break
        if not placed:
            return None

    inp = _render_input((height, width), specs)
    expected = _render_output((height, width), specs)
    if not _validate_instance(inp, expected, specs):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hidden-window anchor-delivery input")
