from __future__ import annotations

import numpy as np


BG = 0
DIVIDER = 5
HINT = 9
BORDER_COLORS = (1, 2, 3, 4)
PATTERN_COLORS = (6, 7, 8)
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
BRANCHES = ("same", "swap", "square")
MAX_GEN_RETRIES = 400


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    arr = np.asarray(mask, dtype=bool)
    return tuple(map(int, arr.shape)), arr.tobytes()


def _components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _monochrome_components(grid: np.ndarray, skip_colors: set[int] | None = None) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    skip = set() if skip_colors is None else set(map(int, skip_colors))
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            col = int(g[r, c])
            if col in skip or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(int(r), int(c))]
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == col:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            rs = [p[0] for p in coords]
            cs = [p[1] for p in coords]
            comps.append(
                {
                    "color": col,
                    "coords": coords,
                    "bbox": (min(rs), min(cs), max(rs), max(cs)),
                    "area": len(coords),
                }
            )
    return comps


def _tight_mask_from_coords(coords: list[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    rs = [p[0] for p in coords]
    cs = [p[1] for p in coords]
    r0, c0, r1, c1 = min(rs), min(cs), max(rs), max(cs)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[int(r - r0), int(c - c0)] = True
    return mask, (int(r0), int(c0), int(r1), int(c1))


def _is_window(panel: np.ndarray, comp: dict, bg: int) -> bool:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    patch = panel[r0 : r1 + 1, c0 : c1 + 1]
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    col = int(comp["color"])
    comp_mask = np.zeros((h, w), dtype=bool)
    for r, c in comp["coords"]:
        comp_mask[int(r - r0), int(c - c0)] = True
    if not np.array_equal(comp_mask, border):
        return False
    if not np.all(patch[border] == col):
        return False
    interior = patch[1:-1, 1:-1]
    return bool(np.all((interior == bg) | (interior != col)))


def _extract_windows(panel: np.ndarray, bg: int) -> list[dict]:
    windows: list[dict] = []
    for comp in _monochrome_components(panel, skip_colors={bg}):
        if not _is_window(panel, comp, bg):
            continue
        r0, c0, r1, c1 = map(int, comp["bbox"])
        inner = panel[r0 + 1 : r1, c0 + 1 : c1]
        hints = [(int(r), int(c)) for r, c in np.argwhere(inner != bg)]
        windows.append(
            {
                "bbox": (r0, c0, r1, c1),
                "inner_shape": tuple(map(int, inner.shape)),
                "area": int(inner.shape[0] * inner.shape[1]),
                "border_color": int(comp["color"]),
                "hints": hints,
            }
        )
    windows.sort(key=lambda d: (d["bbox"][0], d["bbox"][1]))
    return windows


def _extract_patterns(panel: np.ndarray, bg: int) -> list[dict]:
    patterns: list[dict] = []
    for comp in _monochrome_components(panel, skip_colors={bg}):
        mask, bbox = _tight_mask_from_coords(comp["coords"])
        patterns.append(
            {
                "bbox": bbox,
                "mask": mask,
                "color": int(comp["color"]),
                "count": int(mask.sum()),
                "bbox_area": int(mask.shape[0] * mask.shape[1]),
            }
        )
    patterns.sort(key=lambda d: (d["bbox"][0], d["bbox"][1]))
    return patterns


def _candidate_orientations(mask: np.ndarray, target_shape: tuple[int, int]) -> list[np.ndarray]:
    seen: set[tuple[tuple[int, int], bytes]] = set()
    out: list[np.ndarray] = []
    for k in range(4):
        rot = np.rot90(mask, k)
        if tuple(map(int, rot.shape)) != tuple(map(int, target_shape)):
            continue
        key = _mask_key(rot)
        if key in seen:
            continue
        seen.add(key)
        out.append(rot)
    return out


def _pick_assignment(
    windows: list[dict],
    patterns: list[dict],
    require_area_match: bool,
) -> dict[int, tuple[int, np.ndarray]] | None:
    cand_lists: dict[int, list[tuple[tuple[int, int, int, int], int, np.ndarray]]] = {}
    for pi, pat in enumerate(patterns):
        cands: list[tuple[tuple[int, int, int, int], int, np.ndarray]] = []
        for wi, win in enumerate(windows):
            area_score = 0
            if int(win["area"]) == int(pat["count"]):
                area_score += 2
            if int(win["area"]) == int(pat["bbox_area"]):
                area_score += 1
            if require_area_match and area_score == 0:
                continue
            valid: list[np.ndarray] = []
            for oriented in _candidate_orientations(pat["mask"], win["inner_shape"]):
                if all(bool(oriented[r, c]) for r, c in win["hints"]):
                    valid.append(oriented)
            if not valid:
                continue
            best = min(valid, key=lambda arr: arr.tobytes())
            sort_key = (-area_score, len(valid), wi, int(best.sum()))
            cands.append((sort_key, wi, best))
        if not cands:
            return None
        cands.sort(key=lambda item: item[0])
        cand_lists[pi] = cands

    order = sorted(cand_lists, key=lambda pi: (len(cand_lists[pi]), -patterns[pi]["bbox_area"], -patterns[pi]["count"], pi))
    used: set[int] = set()
    chosen: dict[int, tuple[int, np.ndarray]] = {}

    def dfs(idx: int) -> bool:
        if idx == len(order):
            return True
        pi = order[idx]
        for _sort_key, wi, oriented in cand_lists[pi]:
            if wi in used:
                continue
            used.add(wi)
            chosen[pi] = (wi, oriented)
            if dfs(idx + 1):
                return True
            used.remove(wi)
            del chosen[pi]
        return False

    if dfs(0):
        return chosen
    return None


def _find_divider_row(grid: np.ndarray, bg: int) -> int:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    rows = []
    for r in range(h):
        if np.all(g[r, :] == g[r, 0]) and int(g[r, 0]) != bg:
            rows.append(int(r))
    assert rows, "expected a full-width divider row"
    rows.sort(key=lambda r: (abs(r - h // 2), r))
    return int(rows[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each upper window with the matching lower pattern after choosing the correct rotation.

    Input structure:
    - The background is black, and a full-width grey row splits the grid into upper and lower panels.
    - The upper panel contains 2-3 colored rectangular windows with one-cell-thick borders.
    - Each window interior is mostly background, but includes 1-2 maroon hint pixels.
    - The lower panel contains the same number of monochrome connected patterns, one per color.

    Rule:
    - Pair each lower pattern with the upper window whose interior area matches that pattern's footprint size.
    - Tight-crop the pattern; if its height/width are swapped relative to the window interior, rotate it.
    - When multiple rotations fit, keep the rotation whose colored cells cover the maroon hint pixels.
    - Paint the rotated pattern into the window interior, keep the border unchanged, and clear the lower panel.

    Key insight:
    - The match is determined by size, not by color or location, and the hint pixels resolve the rotation ambiguity inside each frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _most_common_color(g)
    divider_row = _find_divider_row(g, bg)

    top = g[:divider_row, :]
    bottom = g[divider_row + 1 :, :]

    windows = _extract_windows(top, bg)
    patterns = _extract_patterns(bottom, bg)
    assert windows and patterns and len(windows) == len(patterns)

    assignment = _pick_assignment(windows, patterns, require_area_match=True)
    if assignment is None:
        assignment = _pick_assignment(windows, patterns, require_area_match=False)
    assert assignment is not None, "expected a consistent pattern-window assignment"

    out = g.copy()
    out[divider_row + 1 :, :] = bg
    for pi, pat in enumerate(patterns):
        wi, oriented = assignment[pi]
        win = windows[wi]
        r0, c0, r1, c1 = map(int, win["bbox"])
        inner = out[r0 + 1 : r1, c0 + 1 : c1]
        inner[:] = bg
        inner[np.asarray(oriented, dtype=bool)] = int(pat["color"])
    return out


def _is_connected(mask: np.ndarray) -> bool:
    return len(_components_mask(mask)) == 1


def _touches_full_bbox(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def _symmetry_pairs(h: int, w: int) -> list[tuple[tuple[int, int], ...]]:
    seen: set[tuple[int, int]] = set()
    pairs: list[tuple[tuple[int, int], ...]] = []
    for r in range(h):
        for c in range(w):
            if (r, c) in seen:
                continue
            mate = (h - 1 - r, w - 1 - c)
            if mate == (r, c):
                pair = ((int(r), int(c)),)
            else:
                pair = tuple(sorted(((int(r), int(c)), (int(mate[0]), int(mate[1])))))
            for cell in pair:
                seen.add(cell)
            pairs.append(pair)
    return pairs


def _random_symmetric_mask(
    h: int,
    w: int,
    rng: np.random.Generator,
    require_not_90: bool,
) -> np.ndarray | None:
    for _ in range(80):
        mask = np.ones((h, w), dtype=bool)
        pairs = _symmetry_pairs(h, w)
        rng.shuffle(pairs)
        max_remove = max(1, min(len(pairs) - 1, (h * w) // 4))
        target_remove = int(rng.integers(1, max_remove + 1))
        removed = 0
        for pair in pairs:
            if removed >= target_remove and rng.random() < 0.7:
                break
            cand = mask.copy()
            for r, c in pair:
                cand[r, c] = False
            if not cand.any():
                continue
            if not _touches_full_bbox(cand):
                continue
            if not _is_connected(cand):
                continue
            mask = cand
            removed += 1
        if removed == 0:
            continue
        if require_not_90 and np.array_equal(mask, np.rot90(mask)):
            continue
        return mask
    return None


def _pick_hints(
    display_mask: np.ndarray,
    target_mask: np.ndarray,
    rng: np.random.Generator,
) -> list[tuple[int, int]] | None:
    target_shape = tuple(map(int, target_mask.shape))
    rotations = _candidate_orientations(display_mask, target_shape)
    others = [rot for rot in rotations if not np.array_equal(rot, target_mask)]
    exclusive = []
    for r, c in np.argwhere(target_mask):
        rr, cc = int(r), int(c)
        if all(not bool(other[rr, cc]) for other in others):
            exclusive.append((rr, cc))
    if not exclusive:
        return None
    rng.shuffle(exclusive)
    take = 2 if len(exclusive) >= 2 and rng.random() < 0.5 else 1
    return exclusive[:take]


def _window_dim_candidates(branch: str) -> list[tuple[int, int]]:
    if branch == "square":
        return [(3, 3), (4, 4), (5, 5)]
    dims = [
        (2, 3),
        (2, 4),
        (2, 5),
        (3, 4),
        (3, 5),
        (4, 5),
        (3, 2),
        (4, 2),
        (5, 2),
        (4, 3),
        (5, 3),
        (5, 4),
    ]
    return dims


def _choose_specs(rng: np.random.Generator, n: int) -> list[dict] | None:
    for _ in range(120):
        specs: list[dict] = []
        used_areas: set[int] = set()
        branches = [BRANCHES[int(rng.integers(0, len(BRANCHES)))] for _ in range(n)]
        for branch in branches:
            dims = _window_dim_candidates(branch)
            order = list(range(len(dims)))
            rng.shuffle(order)
            picked = None
            for idx in order:
                h, w = dims[idx]
                area = int(h * w)
                if area in used_areas:
                    continue
                base = _random_symmetric_mask(h, w, rng, require_not_90=(branch == "square"))
                if base is None:
                    continue
                if branch == "same":
                    display = base.copy()
                    target = base.copy()
                elif branch == "swap":
                    display = np.rot90(base)
                    target = base.copy()
                else:
                    display = base.copy() if rng.random() < 0.5 else np.rot90(base)
                    target = display.copy() if rng.random() < 0.5 else np.rot90(display)
                hints = _pick_hints(display, target, rng)
                if hints is None:
                    continue
                picked = {
                    "branch": branch,
                    "inner_shape": (int(h), int(w)),
                    "area": area,
                    "display_mask": display,
                    "target_mask": target,
                    "count": int(display.sum()),
                    "hints": hints,
                }
                break
            if picked is None:
                break
            used_areas.add(int(picked["area"]))
            specs.append(picked)
        if len(specs) != n:
            continue
        area_values = {int(spec["area"]) for spec in specs}
        if any(int(spec["count"]) in area_values for spec in specs):
            continue
        return specs
    return None


def _draw_rect_border(grid: np.ndarray, r0: int, c0: int, h: int, w: int, color: int) -> None:
    grid[r0, c0 : c0 + w] = color
    grid[r0 + h - 1, c0 : c0 + w] = color
    grid[r0 : r0 + h, c0] = color
    grid[r0 : r0 + h, c0 + w - 1] = color


def _can_place_bbox(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> bool:
    hh, ww = occupied.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(hh, r0 + h + pad)
    cc1 = min(ww, c0 + w + pad)
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _place_patterns(
    specs: list[dict],
    bottom_h: int,
    width: int,
    rng: np.random.Generator,
) -> dict[int, tuple[int, int]] | None:
    occupied = np.zeros((bottom_h, width), dtype=bool)
    order = list(range(len(specs)))
    order.sort(key=lambda idx: int(specs[idx]["display_mask"].shape[0] * specs[idx]["display_mask"].shape[1]), reverse=True)
    rng.shuffle(order)
    positions: dict[int, tuple[int, int]] = {}
    for idx in order:
        h, w = map(int, specs[idx]["display_mask"].shape)
        candidates = [(r, c) for r in range(bottom_h - h + 1) for c in range(width - w + 1)]
        rng.shuffle(candidates)
        placed = None
        for r0, c0 in candidates:
            if _can_place_bbox(occupied, int(r0), int(c0), h, w, pad=1):
                placed = (int(r0), int(c0))
                occupied[r0 : r0 + h, c0 : c0 + w] = True
                break
        if placed is None:
            return None
        positions[idx] = placed
    return positions


def _build_input_and_output(specs: list[dict], rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    n = len(specs)
    border_colors = list(BORDER_COLORS[:n])
    pattern_colors = list(PATTERN_COLORS[:n])
    rng.shuffle(border_colors)
    rng.shuffle(pattern_colors)
    rng.shuffle(specs)
    for idx, spec in enumerate(specs):
        spec["border_color"] = int(border_colors[idx])
        spec["pattern_color"] = int(pattern_colors[idx])

    outer_ws = [int(spec["inner_shape"][1] + 2) for spec in specs]
    outer_hs = [int(spec["inner_shape"][0] + 2) for spec in specs]
    left = int(rng.integers(1, 3))
    right = int(rng.integers(1, 3))
    gaps = [int(rng.integers(1, 4)) for _ in range(max(0, n - 1))]
    min_w = left + sum(outer_ws) + sum(gaps) + right
    width = max(min_w, int(rng.integers(18, 27)))
    if width > 30:
        return None

    top_h = max(outer_hs) + 3 + int(rng.integers(0, 3))
    bottom_h = max(int(spec["display_mask"].shape[0]) for spec in specs) + 4 + int(rng.integers(0, 4))
    height = top_h + 1 + bottom_h
    if height > 30:
        return None

    window_positions: list[tuple[int, int]] = []
    c = left
    for idx, spec in enumerate(specs):
        oh = int(spec["inner_shape"][0] + 2)
        ow = int(spec["inner_shape"][1] + 2)
        max_r0 = top_h - oh
        if max_r0 < 1:
            return None
        r0 = int(rng.integers(1, max_r0 + 1))
        window_positions.append((r0, c))
        c += ow
        if idx < n - 1:
            c += gaps[idx]

    pattern_positions = _place_patterns(specs, bottom_h, width, rng)
    if pattern_positions is None:
        return None

    divider_row = top_h
    inp = np.full((height, width), BG, dtype=int)
    out = np.full((height, width), BG, dtype=int)
    inp[divider_row, :] = DIVIDER
    out[divider_row, :] = DIVIDER

    for idx, spec in enumerate(specs):
        r0, c0 = window_positions[idx]
        ih, iw = map(int, spec["inner_shape"])
        oh, ow = ih + 2, iw + 2
        _draw_rect_border(inp, r0, c0, oh, ow, int(spec["border_color"]))
        _draw_rect_border(out, r0, c0, oh, ow, int(spec["border_color"]))
        for hr, hc in spec["hints"]:
            inp[r0 + 1 + hr, c0 + 1 + hc] = HINT
        target_view = out[r0 + 1 : r0 + 1 + ih, c0 + 1 : c0 + 1 + iw]
        target_view[np.asarray(spec["target_mask"], dtype=bool)] = int(spec["pattern_color"])

        pr0, pc0 = pattern_positions[idx]
        pr0 += divider_row + 1
        mask = np.asarray(spec["display_mask"], dtype=bool)
        patch = inp[pr0 : pr0 + mask.shape[0], pc0 : pc0 + mask.shape[1]]
        patch[mask] = int(spec["pattern_color"])

    return inp, out


def _generate_case(rng: np.random.Generator) -> np.ndarray | None:
    n = int(rng.integers(2, 4))
    specs = _choose_specs(rng, n)
    if specs is None:
        return None
    built = _build_input_and_output(specs, rng)
    if built is None:
        return None
    inp, expected = built
    got = main(inp)
    if not np.array_equal(got, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_RETRIES):
        rng = np.random.default_rng()
        grid = _generate_case(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
