from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 7
BLACK = 0
BLUE = 1
RED = 2
SEP = 5
GEN_ATTEMPTS = 500


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _find_separator_col(grid: np.ndarray) -> int:
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == SEP)]
    assert len(cols) == 1, f"expected one full-height separator column, got {cols}"
    return int(cols[0])


def _sorted_black_components_left(grid: np.ndarray, sep_col: int) -> list[np.ndarray]:
    comps = _components4(grid[:, :sep_col] == BLACK)
    comps.sort(key=lambda pts: (int(pts[:, 0].min()), int(pts[:, 1].min())))
    return comps


def _slide_black_components(grid: np.ndarray, sep_col: int) -> tuple[np.ndarray, list[dict]]:
    out = np.full_like(grid, BG)
    out[:, sep_col] = SEP
    infos: list[dict] = []
    for coords in _sorted_black_components_left(grid, sep_col):
        shift = (sep_col - 1) - int(coords[:, 1].max())
        shifted = coords.copy()
        shifted[:, 1] += shift
        out[shifted[:, 0], shifted[:, 1]] = BLACK
        infos.append(
            {
                "src_coords": coords,
                "coords": shifted,
                "bbox": _bbox(shifted),
                "shift": int(shift),
            }
        )
    return out, infos


def _valid_holes_for_component(
    grid: np.ndarray,
    coords: np.ndarray,
    sep_col: int,
) -> list[np.ndarray]:
    h, w = grid.shape
    r0, c0, r1, _c1 = _bbox(coords)
    comp = {(int(r), int(c)) for r, c in coords}
    visited: set[tuple[int, int]] = set()
    holes: list[np.ndarray] = []

    for sr in range(r0, r1 + 1):
        for sc in range(c0, sep_col):
            if grid[sr, sc] != BG or (sr, sc) in visited:
                continue
            q = deque([(sr, sc)])
            visited.add((sr, sc))
            region: list[tuple[int, int]] = []
            closed = True
            touches_sep = False
            while q:
                r, c = q.popleft()
                region.append((r, c))
                for nr, nc in _neighbors4(r, c):
                    if not _in_bounds((h, w), nr, nc):
                        closed = False
                        continue
                    if c0 <= nc < sep_col and r0 <= nr <= r1 and grid[nr, nc] == BG:
                        if (nr, nc) not in visited:
                            visited.add((nr, nc))
                            q.append((nr, nc))
                        continue
                    if nc == sep_col and grid[nr, nc] == SEP:
                        touches_sep = True
                        continue
                    if grid[nr, nc] == BLACK and (nr, nc) in comp:
                        continue
                    closed = False
            if closed and touches_sep:
                holes.append(np.array(region, dtype=int))
    holes.sort(key=lambda pts: (int(pts[:, 0].min()), int(pts[:, 1].min())))
    return holes


def _paint_blue_copies(out: np.ndarray, coords: np.ndarray, sep_col: int, copies: int) -> None:
    r0, c0, _r1, c1 = _bbox(coords)
    rel = coords - np.array([r0, c0], dtype=int)
    tile_w = int(c1 - c0 + 1)
    h, w = out.shape
    for k in range(copies):
        base_c = sep_col + 1 + k * tile_w
        for rr, cc in rel:
            ar = int(r0 + rr)
            ac = int(base_c + cc)
            if 0 <= ar < h and 0 <= ac < w:
                out[ar, ac] = BLUE


def _render_transformed(slid_grid: np.ndarray, infos: list[dict], sep_col: int) -> tuple[np.ndarray, list[dict]]:
    out = slid_grid.copy()
    summaries: list[dict] = []
    for info in infos:
        holes = _valid_holes_for_component(slid_grid, info["coords"], sep_col)
        ray_rows: list[int] = []
        for hole in holes:
            ray_rows.extend(sorted({int(r) for r in hole[:, 0]}))
        ray_rows = sorted(set(ray_rows))
        for r in ray_rows:
            out[r, sep_col + 1 :] = RED
        _paint_blue_copies(out, info["coords"], sep_col, len(holes))
        summaries.append(
            {
                "bbox": info["bbox"],
                "hole_count": len(holes),
                "ray_rows": ray_rows,
            }
        )
    return out, summaries


def _analyze_black_components(grid: np.ndarray, sep_col: int) -> list[dict]:
    comps = _sorted_black_components_left(grid, sep_col)
    result = []
    for coords in comps:
        holes = _valid_holes_for_component(grid, coords, sep_col)
        result.append(
            {
                "bbox": _bbox(coords),
                "hole_count": len(holes),
                "ray_rows": sorted({int(r) for hole in holes for r in hole[:, 0]}),
            }
        )
    return result


def main(grid: np.ndarray) -> np.ndarray:
    """Slide black shapes onto the separator, then use separator-touching holes to emit rays and count blue repeats.

    Input structure:
    - The grid has orange background, one full-height grey separator column, and 2-3 black 4-connected shapes on the left.
    - Each black shape contains one or more orange cavity segments that are open toward the separator side before sliding.
    - The right side of the separator is otherwise background.

    Rule:
    - Translate every black shape horizontally right until its rightmost black pixel is flush with the separator's left side.
    - In the slid state, find orange 4-connected regions enclosed by that shape together with the separator; keep only enclosed regions touching the separator.
    - For every row occupied by such a hole, paint a red ray across the separator's right side, then tile as many blue copies of the slid black shape as there are valid holes, starting immediately to the right of the separator.

    Key insight:
    - The same separator-touching enclosed holes serve two roles at once: their row spans decide where red rays appear, and their count decides how many blue shape copies are repeated.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    sep_col = _find_separator_col(g)
    slid, infos = _slide_black_components(g, sep_col)
    out, _summaries = _render_transformed(slid, infos, sep_col)
    return out


def _build_shape_patch(hole_count: int) -> tuple[np.ndarray, list[list[int]]]:
    assert 1 <= hole_count <= 3
    for _ in range(200):
        width = random.randint(3, 5)
        heights = []
        for _idx in range(hole_count):
            heights.append(random.randint(1, 3 if hole_count == 1 else 2))
        area = (hole_count + 1) * width
        hole_rows_black = 0
        row_fill_starts: list[int] = []
        for hh in heights:
            for _ in range(hh):
                extra_black = random.randint(0, max(0, width - 3))
                row_fill_starts.append(1 + extra_black)
                hole_rows_black += 1 + extra_black
        area += hole_rows_black
        if not (8 <= area <= 20):
            continue

        total_h = sum(heights) + hole_count + 1
        patch = np.full((total_h, width), BG, dtype=int)
        hole_segments: list[list[int]] = []
        r = 0
        patch[r, :] = BLACK
        r += 1
        fill_idx = 0
        for hh in heights:
            rows = []
            for _ in range(hh):
                patch[r, 0] = BLACK
                fill_start = row_fill_starts[fill_idx]
                if fill_start > 1:
                    patch[r, 1:fill_start] = BLACK
                rows.append(r)
                fill_idx += 1
                r += 1
            patch[r, :] = BLACK
            r += 1
            hole_segments.append(rows)
        assert r == total_h
        comps = _components4(patch == BLACK)
        if len(comps) != 1:
            continue
        return patch, hole_segments
    raise RuntimeError("failed to build shape patch")


def _choose_layout(shape_specs: list[dict]) -> tuple[int, list[int]]:
    heights = [int(spec["patch"].shape[0]) for spec in shape_specs]
    gaps = [random.randint(1, 3) for _ in range(len(shape_specs) - 1)]
    top_margin = random.randint(1, 2)
    bottom_margin = random.randint(1, 2)
    h = top_margin + sum(heights) + sum(gaps) + bottom_margin
    if h > 30:
        raise RuntimeError("layout too tall")
    rows = []
    cur = top_margin
    for idx, hh in enumerate(heights):
        rows.append(cur)
        cur += hh
        if idx < len(gaps):
            cur += gaps[idx]
    return h, rows


def _make_output_from_specs(
    h: int,
    w: int,
    sep_col: int,
    shape_specs: list[dict],
    top_rows: list[int],
) -> np.ndarray:
    out = np.full((h, w), BG, dtype=int)
    out[:, sep_col] = SEP
    for spec, top in zip(shape_specs, top_rows):
        patch = spec["patch"]
        left = sep_col - patch.shape[1]
        out[top : top + patch.shape[0], left : left + patch.shape[1]][patch == BLACK] = BLACK
    for spec, top in zip(shape_specs, top_rows):
        for rows in spec["hole_rows"]:
            for lr in rows:
                out[top + lr, sep_col + 1 :] = RED
        patch = spec["patch"]
        coords = np.argwhere(patch == BLACK)
        bbox_w = patch.shape[1]
        for copy_idx in range(spec["hole_count"]):
            base_c = sep_col + 1 + copy_idx * bbox_w
            for rr, cc in coords:
                ar = int(top + rr)
                ac = int(base_c + cc)
                if 0 <= ar < h and 0 <= ac < w:
                    out[ar, ac] = BLUE
    return out


def _semantic_signature(items: list[dict]) -> list[tuple[int, int, tuple[int, ...]]]:
    sig = []
    for item in items:
        r0, c0, r1, c1 = item["bbox"]
        sig.append((r0, r1, int(item["hole_count"]), tuple(int(x) for x in item["ray_rows"])))
    return sig


def _attempt_input() -> np.ndarray | None:
    n_shapes = random.randint(2, 3)
    shape_specs = []
    for _ in range(n_shapes):
        hole_count = random.randint(1, 3)
        patch, hole_rows = _build_shape_patch(hole_count)
        shape_specs.append(
            {
                "patch": patch,
                "hole_count": hole_count,
                "hole_rows": hole_rows,
            }
        )

    max_repeat_width = max(int(spec["patch"].shape[1]) * int(spec["hole_count"]) for spec in shape_specs)
    sep_min = 9
    sep_max = min(14, 29 - max_repeat_width)
    if sep_max < sep_min:
        return None
    sep_col = random.randint(sep_min, sep_max)
    extra_right = random.randint(0, min(3, 29 - sep_col - max_repeat_width))
    right_w = max_repeat_width + extra_right
    w = sep_col + 1 + right_w
    if w > 30:
        return None

    try:
        h, top_rows = _choose_layout(shape_specs)
    except RuntimeError:
        return None

    expected = _make_output_from_specs(h, w, sep_col, shape_specs, top_rows)

    inp = np.full((h, w), BG, dtype=int)
    inp[:, sep_col] = SEP
    planned_slid = []
    planned_input = []
    for spec, top in zip(shape_specs, top_rows):
        patch = spec["patch"]
        flush_left = sep_col - patch.shape[1]
        if flush_left < 1:
            return None
        shift_left = random.randint(1, flush_left)
        src_left = flush_left - shift_left
        inp[top : top + patch.shape[0], src_left : src_left + patch.shape[1]][patch == BLACK] = BLACK
        planned_slid.append(
            {
                "bbox": (top, flush_left, top + patch.shape[0] - 1, flush_left + patch.shape[1] - 1),
                "hole_count": spec["hole_count"],
                "ray_rows": sorted(top + lr for rows in spec["hole_rows"] for lr in rows),
            }
        )
        planned_input.append(
            {
                "bbox": (top, src_left, top + patch.shape[0] - 1, src_left + patch.shape[1] - 1),
                "hole_count": 0,
                "ray_rows": [],
            }
        )

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    input_analysis = _analyze_black_components(inp, sep_col)
    if _semantic_signature(input_analysis) != _semantic_signature(planned_input):
        return None

    slid, infos = _slide_black_components(inp, sep_col)
    slid_analysis = _analyze_black_components(slid, sep_col)
    planned_slid.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    if _semantic_signature(slid_analysis) != _semantic_signature(planned_slid):
        return None

    _, output_summaries = _render_transformed(slid, infos, sep_col)
    if _semantic_signature(output_summaries) != _semantic_signature(planned_slid):
        return None

    if not np.any(actual == RED) or not np.any(actual == BLUE):
        return None
    if np.array_equal(inp, actual):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_input()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
