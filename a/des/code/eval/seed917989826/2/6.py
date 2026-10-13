import random
from collections import deque

import numpy as np


BG = 0
FRAME = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 400
MISSING_CORNERS = ("tl", "tr", "bl", "br")
CARDINALS = [(-1, 0), (0, 1), (1, 0), (0, -1)]


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in CARDINALS:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _dist_to_edge(h: int, w: int, r: int, c: int) -> int:
    return min(r, c, h - 1 - r, w - 1 - c)


def _missing_to_pattern(missing: str) -> list[tuple[int, int]]:
    lookup = {
        "tl": [(0, 1), (1, 0), (1, 1)],
        "tr": [(0, 0), (1, 0), (1, 1)],
        "bl": [(0, 0), (0, 1), (1, 1)],
        "br": [(0, 0), (0, 1), (1, 0)],
    }
    return lookup[missing]


def _template_maps_from_missing(missing: str) -> tuple[dict[tuple[int, int], tuple[int, int]], dict[tuple[int, int], tuple[int, int]]]:
    occupied = _missing_to_pattern(missing)
    missing_rc = {
        "tl": (0, 0),
        "tr": (0, 1),
        "bl": (1, 0),
        "br": (1, 1),
    }[missing]
    elbow = (1 - missing_rc[0], 1 - missing_rc[1])

    elbow_dirs = [(r - elbow[0], c - elbow[1]) for r, c in occupied if (r, c) != elbow]
    missing_dirs = [(r - missing_rc[0], c - missing_rc[1]) for r, c in occupied if abs(r - missing_rc[0]) + abs(c - missing_rc[1]) == 1]

    elbow_map = {
        elbow_dirs[0]: elbow_dirs[1],
        elbow_dirs[1]: elbow_dirs[0],
    }
    missing_map = {
        missing_dirs[0]: missing_dirs[1],
        missing_dirs[1]: missing_dirs[0],
    }
    return elbow_map, missing_map


def _find_legend_frame(grid: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int]]:
    candidates: list[tuple[int, int, int, int, int, tuple[int, int, int, int]]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for pts in _components(grid == color):
            r0, c0, r1, c1 = _bbox(pts)
            hh = r1 - r0 + 1
            ww = c1 - c0 + 1
            if hh < 4 or ww < 4:
                continue
            if not np.all(grid[r0, c0 : c1 + 1] == color):
                continue
            if not np.all(grid[r1, c0 : c1 + 1] == color):
                continue
            if not np.all(grid[r0 : r1 + 1, c0] == color):
                continue
            if not np.all(grid[r0 : r1 + 1, c1] == color):
                continue
            area = len(pts)
            candidates.append((r0, -area, hh * ww, color, c0, (r0, c0, r1, c1)))
    if not candidates:
        raise ValueError("legend frame not found")
    candidates.sort()
    _, _, _, color, _, bbox = candidates[0]
    return int(color), bbox


def _parse_templates(
    grid: np.ndarray, bg: int, frame_color: int, bbox: tuple[int, int, int, int]
) -> dict[int, dict[str, object]]:
    r0, c0, r1, c1 = bbox
    full_cols = [c for c in range(c0, c1 + 1) if np.all(grid[r0 : r1 + 1, c] == frame_color)]
    boundaries = [c0] + [c for c in full_cols if c0 < c < c1] + [c1]
    templates: dict[int, dict[str, object]] = {}
    for left, right in zip(boundaries, boundaries[1:]):
        cs, ce = left + 1, right - 1
        if cs > ce:
            continue
        cell = grid[r0 + 1 : r1, cs : ce + 1]
        pts = [(int(rr), int(cc)) for rr, cc in np.argwhere((cell != bg) & (cell != frame_color))]
        if len(pts) != 3:
            continue
        colors = {int(cell[r, c]) for r, c in pts}
        if len(colors) != 1:
            continue
        color = colors.pop()
        br0, bc0, br1, bc1 = _bbox(pts)
        if br1 - br0 != 1 or bc1 - bc0 != 1:
            continue
        rel = {(r - br0, c - bc0) for r, c in pts}
        all2 = {(0, 0), (0, 1), (1, 0), (1, 1)}
        missing_rel = list(all2 - rel)
        if len(missing_rel) != 1:
            continue
        mr, mc = missing_rel[0]
        missing = {
            (0, 0): "tl",
            (0, 1): "tr",
            (1, 0): "bl",
            (1, 1): "br",
        }[(mr, mc)]
        elbow_map, missing_map = _template_maps_from_missing(missing)
        templates[color] = {
            "missing": missing,
            "elbow_map": elbow_map,
            "missing_map": missing_map,
        }
    if not templates:
        raise ValueError("no templates found")
    return templates


def _segment_data(points: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int], int, tuple[int, int], tuple[int, int]] | None:
    rows = sorted({r for r, _ in points})
    cols = sorted({c for _, c in points})
    if len(rows) == 1:
        r = rows[0]
        c0, c1 = min(cols), max(cols)
        if len(points) != c1 - c0 + 1:
            return None
        left = (r, c0)
        right = (r, c1)
        return left, right, len(points), (0, 1), (0, -1)
    if len(cols) == 1:
        c = cols[0]
        r0, r1 = min(rows), max(rows)
        if len(points) != r1 - r0 + 1:
            return None
        top = (r0, c)
        bottom = (r1, c)
        return top, bottom, len(points), (1, 0), (-1, 0)
    return None


def _choose_endpoint(
    h: int,
    w: int,
    end_a: tuple[int, int],
    end_b: tuple[int, int],
    inward_from_a: tuple[int, int],
    inward_from_b: tuple[int, int],
    template: dict[str, object],
) -> tuple[tuple[int, int], tuple[int, int]]:
    da = _dist_to_edge(h, w, end_a[0], end_a[1])
    db = _dist_to_edge(h, w, end_b[0], end_b[1])
    if da > db:
        return end_a, inward_from_a
    if db > da:
        return end_b, inward_from_b

    elbow_map = template["elbow_map"]
    missing_map = template["missing_map"]
    in_a_elbow = inward_from_a in elbow_map
    in_b_elbow = inward_from_b in elbow_map
    if in_a_elbow != in_b_elbow:
        return (end_a, inward_from_a) if in_a_elbow else (end_b, inward_from_b)
    in_a_missing = inward_from_a in missing_map
    in_b_missing = inward_from_b in missing_map
    if in_a_missing != in_b_missing:
        return (end_a, inward_from_a) if in_a_missing else (end_b, inward_from_b)
    return end_a, inward_from_a


def _turn_direction(template: dict[str, object], inward: tuple[int, int]) -> tuple[int, int]:
    elbow_map = template["elbow_map"]
    if inward in elbow_map:
        return elbow_map[inward]
    missing_map = template["missing_map"]
    if inward in missing_map:
        return missing_map[inward]
    raise ValueError("segment direction incompatible with template")


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _most_common_color(g)
    frame_color, legend_bbox = _find_legend_frame(g, bg)
    templates = _parse_templates(g, bg, frame_color, legend_bbox)

    out = g.copy()
    mask = (g != bg) & (g != frame_color)
    lr0, lc0, lr1, lc1 = legend_bbox
    for pts in _components(mask):
        if all(lr0 <= r <= lr1 and lc0 <= c <= lc1 for r, c in pts):
            continue
        seg = _segment_data(pts)
        if seg is None:
            continue
        end_a, end_b, seg_len, inward_from_a, inward_from_b = seg
        color = int(g[pts[0][0], pts[0][1]])
        template = templates.get(color)
        if template is None:
            continue
        endpoint, inward = _choose_endpoint(h, w, end_a, end_b, inward_from_a, inward_from_b, template)
        turn = _turn_direction(template, inward)
        for step in range(1, seg_len + 1):
            rr = endpoint[0] + turn[0] * step
            cc = endpoint[1] + turn[1] * step
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = color
    return out


def _reserve_box(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = 0) -> None:
    h, w = mask.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _reserve_points(mask: np.ndarray, points: list[tuple[int, int]], pad: int = 1) -> None:
    for r, c in points:
        _reserve_box(mask, r, c, r, c, pad=pad)


def _candidate_placements(
    h: int,
    w: int,
    reserved: np.ndarray,
    main_top: int,
    missing: str,
    rng: random.Random,
) -> list[dict[str, object]]:
    elbow_map, _ = _template_maps_from_missing(missing)
    candidates: list[dict[str, object]] = []
    max_len = max(2, min(6, max(h, w) // 2))
    pairs = list(elbow_map.items())
    rng.shuffle(pairs)
    for inward, turn in pairs:
        for seg_len in range(2, max_len + 1):
            for er in range(main_top, h):
                for ec in range(w):
                    segment = [(er + inward[0] * k, ec + inward[1] * k) for k in range(seg_len)]
                    extension = [(er + turn[0] * k, ec + turn[1] * k) for k in range(1, seg_len + 1)]
                    all_pts = segment + extension
                    if not all(0 <= r < h and 0 <= c < w for r, c in all_pts):
                        continue
                    if min(r for r, _ in all_pts) < main_top:
                        continue
                    if any(reserved[r, c] for r, c in all_pts):
                        continue
                    other = segment[-1]
                    if _dist_to_edge(h, w, er, ec) <= _dist_to_edge(h, w, other[0], other[1]):
                        continue
                    candidates.append(
                        {
                            "endpoint": (er, ec),
                            "inward": inward,
                            "turn": turn,
                            "length": seg_len,
                            "segment": segment,
                            "extension": extension,
                        }
                    )
    rng.shuffle(candidates)
    return candidates


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_templates = rng.randint(2, 4)
    cell_h = rng.randint(2, 4)
    cell_w = rng.randint(2, 4)
    legend_h = cell_h + 2
    legend_w = 2 + n_templates * cell_w + (n_templates - 1)

    h = rng.randint(max(legend_h + 7, 12), 22)
    w = rng.randint(max(legend_w + 1, 12), 24)
    if legend_w > w - 1:
        return None

    top_margin = rng.randint(0, 1)
    left_margin = rng.randint(0, w - legend_w)
    legend_r0 = top_margin
    legend_c0 = left_margin
    legend_r1 = legend_r0 + legend_h - 1
    legend_c1 = legend_c0 + legend_w - 1

    gap_below = rng.randint(1, 3)
    main_top = legend_r1 + 1 + gap_below
    if main_top >= h - 2:
        return None

    colors = rng.sample(COLOR_POOL, n_templates)
    missings = random.sample(list(MISSING_CORNERS), k=n_templates)

    grid = np.full((h, w), BG, dtype=int)
    expected = np.full((h, w), BG, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    _reserve_box(reserved, legend_r0, legend_c0, legend_r1, legend_c1, pad=1)

    grid[legend_r0, legend_c0 : legend_c1 + 1] = FRAME
    grid[legend_r1, legend_c0 : legend_c1 + 1] = FRAME
    grid[legend_r0 : legend_r1 + 1, legend_c0] = FRAME
    grid[legend_r0 : legend_r1 + 1, legend_c1] = FRAME
    expected[:] = grid

    cell_tops = legend_r0 + 1
    cur_c = legend_c0 + 1
    template_info: dict[int, dict[str, object]] = {}
    for idx, (color, missing) in enumerate(zip(colors, missings)):
        cell_left = cur_c
        cell_right = cell_left + cell_w - 1
        if idx < n_templates - 1:
            divider = cell_right + 1
            grid[legend_r0 : legend_r1 + 1, divider] = FRAME
            expected[legend_r0 : legend_r1 + 1, divider] = FRAME
        off_r = rng.randint(0, cell_h - 2)
        off_c = rng.randint(0, cell_w - 2)
        anchor_r = cell_tops + off_r
        anchor_c = cell_left + off_c
        for dr, dc in _missing_to_pattern(missing):
            grid[anchor_r + dr, anchor_c + dc] = color
            expected[anchor_r + dr, anchor_c + dc] = color
        elbow_map, missing_map = _template_maps_from_missing(missing)
        template_info[color] = {
            "missing": missing,
            "elbow_map": elbow_map,
            "missing_map": missing_map,
        }
        cur_c = cell_right + 2

    placements = []
    for color in colors:
        candidates = _candidate_placements(h, w, reserved, main_top, template_info[color]["missing"], rng)
        if not candidates:
            return None
        placement = candidates[0]
        placements.append((color, placement))
        for r, c in placement["segment"]:
            grid[r, c] = color
            expected[r, c] = color
        for r, c in placement["extension"]:
            expected[r, c] = color
        _reserve_points(reserved, placement["segment"] + placement["extension"], pad=1)

    if _most_common_color(grid) != BG:
        return None
    if np.array_equal(grid, expected):
        return None
    try:
        produced = main(grid)
    except Exception:
        return None
    if not np.array_equal(produced, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
