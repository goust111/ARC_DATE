from collections import Counter, deque
import numpy as np


BG = 0
MAX_SIDE = 30
MAX_RETRIES = 300
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NEIGHBORS8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


def _components_mask(mask: np.ndarray, connectivity: int) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    dirs = NEIGHBORS4 if connectivity == 4 else NEIGHBORS8
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in dirs:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _orbit_full(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    return [(r, c), (r, w - 1 - c), (h - 1 - r, c), (h - 1 - r, w - 1 - c)]


def _distinct_orbits(h: int, w: int) -> list[list[tuple[int, int]]]:
    seen: set[tuple[tuple[int, int], ...]] = set()
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            orbit = tuple(sorted(set(_orbit_full(r, c, h, w))))
            if orbit in seen:
                continue
            seen.add(orbit)
            out.append(list(orbit))
    return out


def _restore_patch(patch: np.ndarray, pattern_color: int) -> np.ndarray:
    h, w = patch.shape
    restored = np.full((h, w), BG, dtype=int)
    valid = {BG, int(pattern_color)}
    seen: set[tuple[tuple[int, int], ...]] = set()
    for r in range(h):
        for c in range(w):
            orbit_full = _orbit_full(r, c, h, w)
            orbit_key = tuple(sorted(set(orbit_full)))
            if orbit_key in seen:
                continue
            seen.add(orbit_key)
            kept = [int(patch[rr, cc]) for rr, cc in orbit_full if int(patch[rr, cc]) in valid]
            if not kept:
                raise AssertionError("each symmetry orbit must keep at least one non-noise pixel")
            counts = Counter(kept)
            top_count = max(counts.values())
            winners = [color for color, cnt in counts.items() if cnt == top_count]
            if len(winners) != 1:
                raise AssertionError("symmetry vote must have a unique winner")
            color = int(winners[0])
            for rr, cc in orbit_key:
                restored[rr, cc] = color
    return restored


def _count_holes(mask: np.ndarray) -> int:
    holes = 0
    for comp in _components_mask(~mask, connectivity=4):
        if any(r == 0 or c == 0 or r == mask.shape[0] - 1 or c == mask.shape[1] - 1 for r, c in comp):
            continue
        holes += 1
    return holes


def _scale_patch(patch: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 0:
        raise AssertionError("scale factor must be positive")
    if factor == 1:
        return patch.copy()
    return np.repeat(np.repeat(patch, factor, axis=0), factor, axis=1)


def _paste_nonzero(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    mask = patch != BG
    block = dst[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]]
    block[mask] = patch[mask]


def _render_rows(row_groups: list[list[np.ndarray]]) -> np.ndarray:
    row_heights = [max(p.shape[0] for p in group) for group in row_groups]
    row_widths = [sum(p.shape[1] for p in group) + len(group) - 1 for group in row_groups]
    out_h = sum(row_heights) + len(row_groups) - 1
    out_w = max(row_widths)
    out = np.full((out_h, out_w), BG, dtype=int)
    cursor_r = 0
    for row_h, group in zip(row_heights, row_groups):
        cursor_c = 0
        for patch in group:
            _paste_nonzero(out, patch, cursor_r, cursor_c)
            cursor_c += patch.shape[1] + 1
        cursor_r += row_h + 1
    return out


def _group_objects_by_rows(objects: list[dict]) -> list[list[dict]]:
    ordered = sorted(objects, key=lambda obj: (obj["r0"], obj["c0"]))
    groups: list[dict] = []
    for obj in ordered:
        if not groups or obj["r0"] > groups[-1]["bottom"]:
            groups.append({"bottom": obj["r1"], "items": [obj]})
        else:
            groups[-1]["bottom"] = max(groups[-1]["bottom"], obj["r1"])
            groups[-1]["items"].append(obj)
    return [sorted(group["items"], key=lambda obj: obj["c0"]) for group in groups]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct fourfold-symmetric monochrome objects corrupted by per-object noise,
    count enclosed background holes in each restored object, then scale each object by
    its own hole count and re-layout the objects in reading order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    comps = _components_mask(g != BG, connectivity=8)
    assert comps, "expected at least one foreground object"

    objects = []
    for comp in comps:
        r0, c0, r1, c1 = _bbox(comp)
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        nonzero = [int(v) for v in patch.ravel() if int(v) != BG]
        assert nonzero, "object bbox must contain non-background pixels"
        pattern_color = Counter(nonzero).most_common(1)[0][0]
        restored = _restore_patch(patch, int(pattern_color))
        mask = restored == int(pattern_color)
        hole_count = _count_holes(mask)
        assert hole_count >= 1, "restored objects must contain at least one closed hole"
        scaled = _scale_patch(restored, int(hole_count))
        objects.append(
            {
                "r0": int(r0),
                "c0": int(c0),
                "r1": int(r1),
                "c1": int(c1),
                "scaled": scaled,
            }
        )

    grouped = _group_objects_by_rows(objects)
    return _render_rows([[obj["scaled"] for obj in group] for group in grouped])


def _center_rect(h: int, w: int) -> tuple[int, int, int, int]:
    ch = 1 if h % 2 else 2
    cw = 1 if w % 2 else 2
    r0 = (h - ch) // 2
    c0 = (w - cw) // 2
    return r0, r0 + ch, c0, c0 + cw


def _carve_center(mask: np.ndarray) -> tuple[int, int, int, int]:
    r0, r1, c0, c1 = _center_rect(*mask.shape)
    mask[r0:r1, c0:c1] = 0
    return r0, r1, c0, c1


def _carve_horizontal_pair(
    rng: np.random.Generator,
    mask: np.ndarray,
    center_rect: tuple[int, int, int, int] | None,
) -> bool:
    h, w = mask.shape
    hole_h = 1 if h % 2 else 2
    row0 = (h - hole_h) // 2
    hole_w_options = [1, 2] if w >= 8 else [1]
    rng.shuffle(hole_w_options)
    for hole_w in hole_w_options:
        if center_rect is None:
            max_left = (w - 2 * hole_w - 1) // 2
            lo = 1
            hi = max_left
        else:
            _, _, cc0, _ = center_rect
            lo = 1
            hi = cc0 - hole_w - 1
        if hi < lo:
            continue
        left0 = int(rng.integers(lo, hi + 1))
        right0 = w - left0 - hole_w
        if right0 + hole_w > w - 1:
            continue
        if center_rect is not None:
            _, _, cc0, cc1 = center_rect
            if left0 + hole_w > cc0 - 1 or right0 < cc1 + 1:
                continue
        mask[row0 : row0 + hole_h, left0 : left0 + hole_w] = 0
        mask[row0 : row0 + hole_h, right0 : right0 + hole_w] = 0
        return True
    return False


def _carve_vertical_pair(
    rng: np.random.Generator,
    mask: np.ndarray,
    center_rect: tuple[int, int, int, int] | None,
) -> bool:
    h, w = mask.shape
    hole_w = 1 if w % 2 else 2
    col0 = (w - hole_w) // 2
    hole_h_options = [1, 2] if h >= 8 else [1]
    rng.shuffle(hole_h_options)
    for hole_h in hole_h_options:
        if center_rect is None:
            max_top = (h - 2 * hole_h - 1) // 2
            lo = 1
            hi = max_top
        else:
            cr0, cr1, _, _ = center_rect
            lo = 1
            hi = cr0 - hole_h - 1
        if hi < lo:
            continue
        top0 = int(rng.integers(lo, hi + 1))
        bottom0 = h - top0 - hole_h
        if bottom0 + hole_h > h - 1:
            continue
        if center_rect is not None:
            cr0, cr1, _, _ = center_rect
            if top0 + hole_h > cr0 - 1 or bottom0 < cr1 + 1:
                continue
        mask[top0 : top0 + hole_h, col0 : col0 + hole_w] = 0
        mask[bottom0 : bottom0 + hole_h, col0 : col0 + hole_w] = 0
        return True
    return False


def _add_edge_notches(rng: np.random.Generator, mask: np.ndarray) -> None:
    h, w = mask.shape
    if w >= 5 and rng.random() < 0.5:
        notch_w = 1 if w % 2 else 2
        c0 = (w - notch_w) // 2
        mask[0, c0 : c0 + notch_w] = 0
        mask[h - 1, c0 : c0 + notch_w] = 0
    if h >= 5 and rng.random() < 0.5:
        notch_h = 1 if h % 2 else 2
        r0 = (h - notch_h) // 2
        mask[r0 : r0 + notch_h, 0] = 0
        mask[r0 : r0 + notch_h, w - 1] = 0


def _make_symmetric_mask(rng: np.random.Generator, hole_target: int, max_side: int) -> np.ndarray | None:
    min_side = 5 if hole_target < 3 else 7
    max_side = max(min_side, max_side)
    for _ in range(60):
        h = int(rng.integers(min_side, max_side + 1))
        w = int(rng.integers(min_side, max_side + 1))
        if hole_target == 3:
            if rng.random() < 0.5:
                w = max(w, 7)
            else:
                h = max(h, 7)
        mask = np.ones((h, w), dtype=bool)
        center_rect = None
        if hole_target in (1, 3):
            center_rect = _carve_center(mask)
        if hole_target in (2, 3):
            orient = str(rng.choice(["h", "v"]))
            ok = _carve_horizontal_pair(rng, mask, center_rect) if orient == "h" else _carve_vertical_pair(rng, mask, center_rect)
            if not ok:
                continue
        _add_edge_notches(rng, mask)
        if not mask[0, 0] or not mask[0, -1] or not mask[-1, 0] or not mask[-1, -1]:
            continue
        if len(_components_mask(mask, connectivity=4)) != 1:
            continue
        if _count_holes(mask) != hole_target:
            continue
        return mask.astype(int)
    return None


def _noise_candidates(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    candidates: list[list[tuple[int, int]]] = []
    for orbit in _distinct_orbits(h, w):
        if len(orbit) <= 1:
            continue
        base = int(mask[orbit[0]])
        if base == 0:
            attached = False
            for r, c in orbit:
                for dr, dc in NEIGHBORS8:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                        attached = True
                        break
                if attached:
                    break
            if not attached:
                continue
        candidates.append(orbit)
    return candidates


def _apply_noise(
    rng: np.random.Generator,
    mask: np.ndarray,
    pattern_color: int,
    noise_color: int,
) -> np.ndarray | None:
    patch = np.where(mask, int(pattern_color), BG).astype(int)
    candidates = _noise_candidates(mask)
    if not candidates:
        return None
    rng.shuffle(candidates)
    orbit_count = int(rng.integers(1, min(len(candidates), 4) + 1))
    chosen = candidates[:orbit_count]
    noise_pixels = 0
    for orbit in chosen:
        k = int(rng.integers(1, len(orbit)))
        idxs = rng.choice(len(orbit), size=k, replace=False)
        for idx in idxs:
            r, c = orbit[int(idx)]
            patch[r, c] = int(noise_color)
            noise_pixels += 1
    if noise_pixels == 0:
        return None
    nonzero = [int(v) for v in patch.ravel() if int(v) != BG]
    counts = Counter(nonzero)
    if counts[int(pattern_color)] <= counts[int(noise_color)]:
        return None
    return patch


def _layout_output_shapes(rows: list[list[np.ndarray]]) -> tuple[int, int]:
    row_heights = [max(p.shape[0] for p in row) for row in rows]
    row_widths = [sum(p.shape[1] for p in row) + len(row) - 1 for row in rows]
    return sum(row_heights) + len(rows) - 1, max(row_widths)


def _row_structures() -> list[tuple[int, ...]]:
    return [
        (2,),
        (1, 1),
        (3,),
        (2, 1),
        (1, 2),
        (1, 1, 1),
        (2, 2),
        (2, 1, 1),
        (1, 2, 1),
        (1, 1, 2),
    ]


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    row_structure = _row_structures()[int(rng.integers(0, len(_row_structures())))]
    object_count = sum(row_structure)
    colors = rng.choice(np.arange(1, 10), size=2 * object_count, replace=False)
    pattern_colors = [int(v) for v in colors[:object_count]]
    noise_colors = [int(v) for v in colors[object_count:]]

    objects: list[dict] = []
    idx = 0
    rows_expected: list[list[np.ndarray]] = []
    for row_len in row_structure:
        row_expected: list[np.ndarray] = []
        for _ in range(row_len):
            hole_target = int(rng.integers(1, 4))
            max_side = 9 - row_len
            mask = _make_symmetric_mask(rng, hole_target, max_side=max_side)
            if mask is None:
                return None
            noisy = _apply_noise(rng, mask, pattern_colors[idx], noise_colors[idx])
            if noisy is None:
                return None
            restored = np.where(mask, pattern_colors[idx], BG).astype(int)
            scaled = _scale_patch(restored, hole_target)
            objects.append({"input": noisy, "scaled": scaled})
            row_expected.append(scaled)
            idx += 1
        rows_expected.append(row_expected)

    out_h, out_w = _layout_output_shapes(rows_expected)
    if out_h > MAX_SIDE or out_w > MAX_SIDE:
        return None

    row_heights = [max(obj["input"].shape[0] for obj in objects[start : start + row_len]) for start, row_len in zip(np.cumsum((0,) + row_structure[:-1]), row_structure)]
    row_widths = []
    cursor = 0
    row_gaps = [int(rng.integers(2, 5)) for _ in range(max(0, len(row_structure) - 1))]
    x_gaps_per_row: list[list[int]] = []
    for row_len in row_structure:
        widths = [objects[cursor + i]["input"].shape[1] for i in range(row_len)]
        gaps = [int(rng.integers(2, 5)) for _ in range(max(0, row_len - 1))]
        x_gaps_per_row.append(gaps)
        row_widths.append(sum(widths) + sum(gaps))
        cursor += row_len

    top_margin = int(rng.integers(1, 4))
    bottom_margin = int(rng.integers(1, 4))
    left_margin = int(rng.integers(1, 4))
    right_margin = int(rng.integers(1, 4))
    grid_h = top_margin + sum(row_heights) + sum(row_gaps) + bottom_margin
    grid_w = left_margin + max(row_widths) + right_margin
    if grid_h > MAX_SIDE or grid_w > MAX_SIDE:
        return None

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    row_r = top_margin
    obj_idx = 0
    for row_idx, row_len in enumerate(row_structure):
        extra = grid_w - left_margin - right_margin - row_widths[row_idx]
        row_c = left_margin + int(rng.integers(0, extra + 1))
        for j in range(row_len):
            patch = objects[obj_idx]["input"]
            _paste_nonzero(grid, patch, row_r, row_c)
            row_c += patch.shape[1]
            if j + 1 < row_len:
                row_c += x_gaps_per_row[row_idx][j]
            obj_idx += 1
        row_r += row_heights[row_idx]
        if row_idx + 1 < len(row_structure):
            row_r += row_gaps[row_idx]

    if len(_components_mask(grid != BG, connectivity=8)) != object_count:
        return None

    expected = _render_rows(rows_expected)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC sample")
