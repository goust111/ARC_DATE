from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


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
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _dominant_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _tight_mask(points: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(points)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in points:
        out[r - r0, c - c0] = True
    return out


def _normalize_signature(cells: list[tuple[int, int, int]]) -> tuple[tuple[int, int, int], ...]:
    rows = [r for r, _, _ in cells]
    cols = [c for _, c, _ in cells]
    r0 = min(rows)
    c0 = min(cols)
    return tuple(sorted((r - r0, c - c0, col) for r, c, col in cells))


def _row_runs(flags: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    i = 0
    while i < len(flags):
        if not flags[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(flags) and flags[j + 1]:
            j += 1
        runs.append((i, j))
        i = j + 1
    return runs


def _is_rectangle_frame(points: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(points)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    expected = 2 * (h + w) - 4
    if len(points) != expected:
        return False
    pset = set(points)
    for r in range(r0, r1 + 1):
        if (r, c0) not in pset or (r, c1) not in pset:
            return False
    for c in range(c0, c1 + 1):
        if (r0, c) not in pset or (r1, c) not in pset:
            return False
    return True


def _find_legend_frame(grid: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    h, w = grid.shape
    candidates: list[tuple[tuple[int, int, int], tuple[int, int, int, int]]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for pts in _components(grid == color):
            if not _is_rectangle_frame(pts):
                continue
            r0, c0, r1, c1 = _bbox(pts)
            if r1 - r0 + 1 < 8 or c1 - c0 + 1 < 8:
                continue
            interior = grid[r0 + 1 : r1, c0 + 1 : c1]
            non_bg = interior != bg
            if int(non_bg.sum()) < 10:
                continue
            run_count = len(_row_runs(np.any(non_bg, axis=1)))
            if run_count < 2:
                continue
            score = (r1 + c1, int(non_bg.sum()), len(pts))
            candidates.append((score, (r0, c0, r1, c1)))
    assert candidates, "expected a legend frame in the lower-right area"
    candidates.sort()
    return candidates[-1][1]


def _extract_legend_mapping(
    grid: np.ndarray, legend_bbox: tuple[int, int, int, int], bg: int
) -> dict[tuple[tuple[int, int, int], ...], np.ndarray]:
    r0, c0, r1, c1 = legend_bbox
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    runs = _row_runs(np.any(interior != bg, axis=1))
    mapping: dict[tuple[tuple[int, int, int], ...], np.ndarray] = {}
    for rs, re in runs:
        band = interior[rs : re + 1, :]
        comps = _components(band != bg)
        assert comps, "every legend band must contain at least one object"
        comps.sort(key=len, reverse=True)
        template_pts = comps[0]
        template_mask = _tight_mask(template_pts)
        template_set = set(template_pts)
        signature_cells = [
            (r, c, int(band[r, c]))
            for r in range(band.shape[0])
            for c in range(band.shape[1])
            if int(band[r, c]) != bg and (r, c) not in template_set
        ]
        assert signature_cells, "each legend entry must include a signature"
        key = _normalize_signature(signature_cells)
        assert key not in mapping, "legend signatures must be unique"
        mapping[key] = template_mask
    return mapping


def _boundary_of_filled(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    boundary = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            for nr, nc in _neighbors4(r, c):
                if not (0 <= nr < h and 0 <= nc < w) or not mask[nr, nc]:
                    boundary[r, c] = True
                    break
    return boundary


def _flood_from_seeds(mask: np.ndarray, seeds: set[tuple[int, int]]) -> set[tuple[int, int]]:
    h, w = mask.shape
    q = deque(seeds)
    seen = set(seeds)
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return seen


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer each closed region's template from its internal color signature, then replace the
    irregular outline with a scaled copy of that template.
    """
    g = np.asarray(grid, dtype=int)
    bg = _dominant_color(g)

    legend_bbox = _find_legend_frame(g, bg)
    legend_map = _extract_legend_mapping(g, legend_bbox, bg)

    legend_mask = np.zeros(g.shape, dtype=bool)
    lr0, lc0, lr1, lc1 = legend_bbox
    legend_mask[lr0 : lr1 + 1, lc0 : lc1 + 1] = True

    outside_legend = (~legend_mask) & (g != bg)
    colors = [int(x) for x in g[outside_legend]]
    border_color = Counter(colors).most_common(1)[0][0]

    out = g.copy()
    passable = (~legend_mask) & (g != border_color)
    for pts in _components(passable):
        if any(r == 0 or c == 0 or r == g.shape[0] - 1 or c == g.shape[1] - 1 for r, c in pts):
            continue
        signature = [(r, c, int(g[r, c])) for r, c in pts if int(g[r, c]) != bg]
        if not signature:
            continue
        key = _normalize_signature(signature)
        assert key in legend_map, "region signature must match a legend entry"
        template = legend_map[key]

        border_seed_pts: set[tuple[int, int]] = set()
        for r, c in pts:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1] and not legend_mask[nr, nc]:
                    if int(g[nr, nc]) == border_color:
                        border_seed_pts.add((nr, nc))
        assert border_seed_pts, "each enclosed region must have a border"

        border_pts = _flood_from_seeds((g == border_color) & (~legend_mask), border_seed_pts)

        clear_pts = set(pts) | border_pts
        for r, c in clear_pts:
            out[r, c] = bg

        br0, bc0, br1, bc1 = _bbox(list(border_pts))
        box_h = br1 - br0 + 1
        box_w = bc1 - bc0 + 1
        tpl_h, tpl_w = template.shape
        k = max(1, min(box_h // tpl_h, box_w // tpl_w))
        scaled = np.kron(template.astype(int), np.ones((k, k), dtype=int)).astype(bool)
        sh, sw = scaled.shape
        off_r = br0 + (box_h - sh) // 2
        off_c = bc0 + (box_w - sw) // 2
        out[off_r : off_r + sh, off_c : off_c + sw][scaled] = border_color

    return out


TEMPLATES = [
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),  # plus
    np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=bool),  # L
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),  # T
    np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1]], dtype=bool),  # U
    np.array([[0, 1, 1], [1, 1, 0], [1, 0, 0]], dtype=bool),  # zig
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool),  # hook
]

SIGNATURE_PATTERNS = {
    1: [[(0, 0)]],
    2: [[(0, 0), (0, 1)], [(0, 0), (1, 0)], [(0, 0), (1, 1)]],
    3: [[(0, 0), (0, 1), (1, 0)], [(0, 0), (1, 0), (1, 1)], [(0, 0), (0, 1), (0, 2)]],
}

GEN_RETRIES = 400


def _single_component(mask: np.ndarray) -> bool:
    comps = _components(mask)
    return len(comps) == 1


def _random_signature(
    rng: random.Random, colors: list[int], used: set[tuple[tuple[int, int, int], ...]]
) -> list[tuple[int, int, int]]:
    for _ in range(200):
        count = rng.choice([1, 2, 3])
        pattern = rng.choice(SIGNATURE_PATTERNS[count])
        chosen = rng.sample(colors, count)
        rng.shuffle(chosen)
        cells = [(r, c, col) for (r, c), col in zip(pattern, chosen, strict=False)]
        key = _normalize_signature(cells)
        if key not in used:
            used.add(key)
            return cells
    raise RuntimeError("failed to sample a unique signature")


def _render_legend(
    entries: list[dict[str, object]], bg: int, frame_color: int
) -> np.ndarray:
    n = len(entries)
    inner_h = 4 * n - 1
    inner_w = 9
    inner = np.full((inner_h, inner_w), bg, dtype=int)
    for idx, entry in enumerate(entries):
        rs = idx * 4
        template = entry["template"]
        template_color = int(entry["template_color"])
        signature = entry["signature"]
        inner[rs : rs + 3, 0:3][template] = template_color
        for dr, dc, col in signature:
            inner[rs + dr, 5 + dc] = col
    legend = np.full((inner_h + 2, inner_w + 2), bg, dtype=int)
    legend[0, :] = frame_color
    legend[-1, :] = frame_color
    legend[:, 0] = frame_color
    legend[:, -1] = frame_color
    legend[1:-1, 1:-1] = inner
    return legend


def _random_region_mask(h: int, w: int, rng: random.Random) -> np.ndarray | None:
    max_depth = max(1, min(h, w) // 3)
    for _ in range(120):
        mask = np.ones((h, w), dtype=bool)
        notch_count = rng.randint(1, 4)
        for _ in range(notch_count):
            edge = rng.choice(["top", "bottom", "left", "right"])
            if edge in ("top", "bottom"):
                if w < 5:
                    continue
                depth = rng.randint(1, max_depth)
                c0 = rng.randint(1, w - 3)
                c1 = rng.randint(c0 + 1, w - 2)
                if edge == "top":
                    mask[0:depth, c0 : c1 + 1] = False
                else:
                    mask[h - depth : h, c0 : c1 + 1] = False
            else:
                if h < 5:
                    continue
                depth = rng.randint(1, max_depth)
                r0 = rng.randint(1, h - 3)
                r1 = rng.randint(r0 + 1, h - 2)
                if edge == "left":
                    mask[r0 : r1 + 1, 0:depth] = False
                else:
                    mask[r0 : r1 + 1, w - depth : w] = False
        if _bbox(list(zip(*np.where(mask), strict=False))) != (0, 0, h - 1, w - 1):
            continue
        if not _single_component(mask):
            continue
        boundary = _boundary_of_filled(mask)
        interior = mask & ~boundary
        if interior.sum() < 6:
            continue
        if not _single_component(interior):
            continue
        return mask
    return None


def _place_signature(
    interior: np.ndarray, signature: list[tuple[int, int, int]], rng: random.Random
) -> list[tuple[int, int, int]] | None:
    sig_h = max(r for r, _, _ in signature) + 1
    sig_w = max(c for _, c, _ in signature) + 1
    candidates: list[tuple[int, int]] = []
    for r0 in range(interior.shape[0] - sig_h + 1):
        for c0 in range(interior.shape[1] - sig_w + 1):
            if all(interior[r0 + dr, c0 + dc] for dr, dc, _ in signature):
                candidates.append((r0, c0))
    if not candidates:
        return None
    off_r, off_c = rng.choice(candidates)
    return [(off_r + dr, off_c + dc, col) for dr, dc, col in signature]


def _can_place(occ: np.ndarray, r0: int, c0: int, h: int, w: int, gap: int) -> bool:
    rr0 = max(0, r0 - gap)
    cc0 = max(0, c0 - gap)
    rr1 = min(occ.shape[0], r0 + h + gap)
    cc1 = min(occ.shape[1], c0 + w + gap)
    return not occ[rr0:rr1, cc0:cc1].any()


def _mark_occ(occ: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    occ[r0 : r0 + h, c0 : c0 + w] = True


def _build_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    h = rng.randint(26, 34)
    w = rng.randint(26, 34)
    all_colors = list(range(10))
    bg, border_color, legend_frame = rng.sample(all_colors, 3)
    remaining = [c for c in all_colors if c not in {bg, border_color, legend_frame}]

    template_count = rng.randint(3, 4)
    template_indices = rng.sample(range(len(TEMPLATES)), template_count)
    template_colors = rng.sample(remaining, template_count)
    signature_palette = [c for c in remaining if c not in set(template_colors)]
    assert len(signature_palette) >= 3

    used_signatures: set[tuple[tuple[int, int, int], ...]] = set()
    entries: list[dict[str, object]] = []
    for tpl_idx, tpl_color in zip(template_indices, template_colors, strict=False):
        signature = _random_signature(rng, signature_palette, used_signatures)
        entries.append(
            {
                "template": TEMPLATES[tpl_idx].copy(),
                "template_color": tpl_color,
                "signature": signature,
            }
        )

    legend = _render_legend(entries, bg, legend_frame)
    if legend.shape[0] + 4 > h or legend.shape[1] + 4 > w:
        return None

    margin_bottom = rng.randint(0, 1)
    margin_right = rng.randint(0, 1)
    legend_r0 = h - legend.shape[0] - margin_bottom
    legend_c0 = w - legend.shape[1] - margin_right

    inp = np.full((h, w), bg, dtype=int)
    out = np.full((h, w), bg, dtype=int)
    inp[legend_r0 : legend_r0 + legend.shape[0], legend_c0 : legend_c0 + legend.shape[1]] = legend
    out[legend_r0 : legend_r0 + legend.shape[0], legend_c0 : legend_c0 + legend.shape[1]] = legend

    occ = np.zeros((h, w), dtype=bool)
    _mark_occ(occ, legend_r0, legend_c0, legend.shape[0], legend.shape[1])
    region_count = rng.randint(2, 4)

    if region_count == 4:
        size_choices = [2, 3]
    elif region_count == 3:
        size_choices = [2, 3, 4]
    else:
        size_choices = [3, 4, 5]
    for _ in range(region_count):
        entry = rng.choice(entries)
        k = rng.choice(size_choices)
        box_h = 3 * k
        box_w = 3 * k
        region_mask = _random_region_mask(box_h, box_w, rng)
        if region_mask is None:
            return None
        boundary = _boundary_of_filled(region_mask)
        interior = region_mask & ~boundary
        placed_signature = _place_signature(interior, entry["signature"], rng)
        if placed_signature is None:
            return None

        positions = [
            (r0, c0)
            for r0 in range(1, h - box_h)
            for c0 in range(1, w - box_w)
            if _can_place(occ, r0, c0, box_h, box_w, gap=1)
        ]
        if not positions:
            return None
        r0, c0 = rng.choice(positions)
        _mark_occ(occ, r0, c0, box_h, box_w)

        sub_in = inp[r0 : r0 + box_h, c0 : c0 + box_w]
        sub_in[boundary] = border_color
        for sr, sc, col in placed_signature:
            sub_in[sr, sc] = col

        scaled = np.kron(entry["template"].astype(int), np.ones((k, k), dtype=int)).astype(bool)
        out[r0 : r0 + box_h, c0 : c0 + box_w][scaled] = border_color

    if not np.array_equal(main(inp), out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        case = _build_case(random.Random())
        if case is not None:
            return case[0]
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
