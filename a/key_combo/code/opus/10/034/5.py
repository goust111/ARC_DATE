import random
from typing import Iterable

import numpy as np


BG = 0
LEGEND_BG = 5
ANGLES = (0, 90, 180, 270)
ANGLE_TO_MARKER = {
    0: (1, 0),
    90: (1, 1),
    180: (0, 0),
    270: (0, 1),
}
MARKER_TO_ANGLE = {v: k for k, v in ANGLE_TO_MARKER.items()}
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GEN_ATTEMPTS = 400


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
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
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _line_runs(pattern: np.ndarray, axis: int) -> list[tuple[int, int, np.ndarray]]:
    count = pattern.shape[axis]
    runs: list[tuple[int, int, np.ndarray]] = []
    start = 0

    def line_at(idx: int) -> np.ndarray:
        return pattern[idx, :] if axis == 0 else pattern[:, idx]

    for idx in range(1, count + 1):
        if idx == count or not np.array_equal(line_at(idx), line_at(start)):
            runs.append((start, idx - start, line_at(start).copy()))
            start = idx
    return runs


def _choose_stretch_run(pattern: np.ndarray, axis: int) -> int:
    runs = _line_runs(pattern, axis)
    if len(runs) == 1:
        return 0
    candidates = list(range(1, len(runs) - 1))
    if not candidates:
        raise ValueError("stretch axis needs an interior run")
    scores = [len(set(map(int, runs[i][2].tolist()))) for i in candidates]
    best_score = max(scores)
    best = [idx for idx, score in zip(candidates, scores) if score == best_score]
    if len(best) == 1:
        return best[0]
    widths = [runs[i][1] for i in best]
    best_width = max(widths)
    if widths.count(best_width) == 1:
        return best[widths.index(best_width)]
    raise ValueError("stretch run must be uniquely identifiable")


def _resize_axis(pattern: np.ndarray, target_len: int, axis: int) -> np.ndarray:
    runs = _line_runs(pattern, axis)
    stretch_idx = _choose_stretch_run(pattern, axis)
    fixed = sum(width for i, (_, width, _) in enumerate(runs) if i != stretch_idx)
    new_width = int(target_len) - fixed
    if new_width < 1:
        raise ValueError("target too small for elastic stretch")

    pieces = []
    for idx, (_, width, line) in enumerate(runs):
        repeat = new_width if idx == stretch_idx else width
        if axis == 0:
            pieces.append(np.repeat(line[np.newaxis, :], repeat, axis=0))
        else:
            pieces.append(np.repeat(line[:, np.newaxis], repeat, axis=1))
    return np.concatenate(pieces, axis=axis)


def _stretch_pattern(pattern: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    stretched = _resize_axis(pattern, target_shape[0], axis=0)
    stretched = _resize_axis(stretched, target_shape[1], axis=1)
    return stretched


def _rotate_clockwise(pattern: np.ndarray, angle: int) -> np.ndarray:
    return np.rot90(pattern, k=(-angle // 90) % 4)


def _parse_legend(legend: np.ndarray) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for color in sorted(set(map(int, legend.flatten())) - {BG, LEGEND_BG}):
        coords = np.argwhere(legend == color)
        if coords.size == 0:
            continue
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        if (r1 - r0 + 1, c1 - c0 + 1) != (2, 2):
            raise ValueError("legend cell must occupy a 2x2 bbox")
        missing = []
        for dr in range(2):
            for dc in range(2):
                if legend[r0 + dr, c0 + dc] != color:
                    missing.append((dr, dc))
        if len(missing) != 1 or missing[0] not in MARKER_TO_ANGLE:
            raise ValueError("legend marker must be a single missing corner pixel")
        mapping[color] = MARKER_TO_ANGLE[missing[0]]
    if not mapping:
        raise ValueError("legend mapping not found")
    return mapping


def _extract_regions(grid: np.ndarray) -> list[dict]:
    regions = []
    for coords in _components(grid != BG):
        bbox = _bbox_from_coords(coords)
        patch = _crop_bbox(grid, bbox)
        area = int(coords.shape[0])
        rect_area = patch.shape[0] * patch.shape[1]
        if area != rect_area:
            raise ValueError("all foreground components must be solid rectangles")
        regions.append({"bbox": bbox, "patch": patch, "area": area})
    return regions


def _find_exemplar(regions: list[dict]) -> tuple[dict, int]:
    candidates = []
    for region in regions:
        patch = region["patch"]
        colors = set(map(int, np.unique(patch)))
        if len(colors) < 3:
            continue
        outer = int(patch[0, 0])
        corners = {int(patch[0, 0]), int(patch[0, -1]), int(patch[-1, 0]), int(patch[-1, -1])}
        inner_colors = colors - {outer}
        if len(corners) == 1 and len(inner_colors) >= 2:
            candidates.append((region, outer))
    if len(candidates) != 1:
        raise ValueError("expected exactly one multicolor exemplar rectangle")
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate the exemplar by a legend-encoded angle, elastically stretch it, and fill each matching target rectangle.

    Input structure:
    - A black-background grid has a grey legend band at the top and a main field below.
    - Each legend entry is a 2x2 colored cell with one missing corner pixel; that missing-corner position encodes a rotation angle for that color.
    - The main field contains one multicolor exemplar rectangle and several solid-color target rectangles whose colors appear exactly once in the legend.

    Rule:
    - Remove the grey legend band from the output and decode a color -> angle map from the 2x2 legend cells.
    - Find the unique multicolor exemplar rectangle whose four corners share one outer color.
    - For each solid target rectangle, rotate the exemplar by that target color's decoded angle.
    - Elastically stretch the rotated exemplar to the target size by keeping boundary row/column runs fixed and resizing the uniquely selected interior run on each axis.
    - Replace each solid target rectangle with the stretched pattern while leaving the exemplar itself unchanged.

    Key insight:
    - The legend colors do not name replacement colors; they encode geometric rotation parameters, and rotating before stretching changes which internal row/column band expands.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("grid must be 2D")

    grey_rows = np.where(np.any(g == LEGEND_BG, axis=1))[0]
    if grey_rows.size == 0 or not np.array_equal(grey_rows, np.arange(grey_rows[-1] + 1)):
        raise ValueError("legend band must be the contiguous top grey rows")
    legend_h = int(grey_rows[-1] + 1)
    legend = g[:legend_h, :]
    mapping = _parse_legend(legend)

    base = g[legend_h:, :].copy()
    regions = _extract_regions(base)
    exemplar_region, _outer = _find_exemplar(regions)

    out = base.copy()
    exemplar_bbox = exemplar_region["bbox"]
    exemplar = exemplar_region["patch"]

    for region in regions:
        if region["bbox"] == exemplar_bbox:
            continue
        patch = region["patch"]
        unique = np.unique(patch)
        if len(unique) != 1:
            raise ValueError("non-exemplar rectangles must be monochrome")
        color = int(unique[0])
        if color not in mapping:
            raise ValueError("target color missing from legend")
        rotated = _rotate_clockwise(exemplar, mapping[color])
        filled = _stretch_pattern(rotated, patch.shape)
        r0, c0, r1, c1 = region["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = filled
    return out


def _repeat_coarse(coarse: np.ndarray, row_runs: list[int], col_runs: list[int]) -> np.ndarray:
    expanded = np.repeat(coarse, row_runs, axis=0)
    expanded = np.repeat(expanded, col_runs, axis=1)
    return expanded


def _build_random_exemplar(colors: list[int]) -> np.ndarray | None:
    outer = colors[0]
    inner = colors[1:]
    for _ in range(200):
        n_rows = random.randint(3, 5)
        n_cols = random.randint(4, 6)
        row_runs = [random.randint(1, 2) for _ in range(n_rows)]
        col_runs = [random.randint(1, 2) for _ in range(n_cols)]
        if random.random() < 0.35:
            row_runs[random.randint(1, n_rows - 2)] += 1
        if random.random() < 0.35:
            col_runs[random.randint(1, n_cols - 2)] += 1
        if sum(row_runs) == sum(col_runs):
            continue

        coarse = np.full((n_rows, n_cols), outer, dtype=int)
        for r in range(1, n_rows - 1):
            for c in range(1, n_cols - 1):
                coarse[r, c] = random.choice(inner)

        if len(set(map(int, coarse[1:-1, 1:-1].flatten()))) < 2:
            continue
        exemplar = _repeat_coarse(coarse, row_runs, col_runs)
        if max(exemplar.shape) > 8 or min(exemplar.shape) < 4:
            continue
        if len(set(map(int, np.unique(exemplar)))) < 3:
            continue
        try:
            for angle in ANGLES:
                rot = _rotate_clockwise(exemplar, angle)
                if len(_line_runs(rot, 0)) < 3 or len(_line_runs(rot, 1)) < 3:
                    raise ValueError
                _choose_stretch_run(rot, 0)
                _choose_stretch_run(rot, 1)
        except ValueError:
            continue
        return exemplar
    return None


def _can_stretch_to(pattern: np.ndarray, shape: tuple[int, int]) -> bool:
    try:
        _stretch_pattern(pattern, shape)
        return True
    except ValueError:
        return False


def _candidate_sizes(pattern: np.ndarray) -> list[tuple[int, int]]:
    h0, w0 = pattern.shape
    sizes = []
    for h in range(max(3, h0 - 1), min(12, h0 + 5)):
        for w in range(max(3, w0 - 1), min(12, w0 + 6)):
            if (h, w) == (h0, w0):
                continue
            if _can_stretch_to(pattern, (h, w)):
                sizes.append((h, w))
    return sizes


def _choose_target_specs(exemplar: np.ndarray, target_colors: list[int], angles: list[int]) -> list[dict] | None:
    rotated = {angle: _rotate_clockwise(exemplar, angle) for angle in set(angles)}
    pair = None
    for h in range(4, 12):
        for w in range(4, 12):
            a0, a1 = angles[0], angles[1]
            p0 = rotated[a0]
            p1 = rotated[a1]
            if not _can_stretch_to(p0, (h, w)) or not _can_stretch_to(p1, (h, w)):
                continue
            if (h, w) == p0.shape or (h, w) == p1.shape:
                continue
            if np.array_equal(_stretch_pattern(p0, (h, w)), _stretch_pattern(p1, (h, w))):
                continue
            pair = (h, w)
            break
        if pair is not None:
            break
    if pair is None:
        return None

    specs = [
        {"color": target_colors[0], "angle": angles[0], "shape": pair},
        {"color": target_colors[1], "angle": angles[1], "shape": pair},
    ]

    for color, angle in zip(target_colors[2:], angles[2:]):
        pattern = rotated[angle]
        choices = _candidate_sizes(pattern)
        if not choices:
            return None
        same_row = [shape for shape in choices if shape[0] == pattern.shape[0] and shape[1] != pattern.shape[1]]
        same_col = [shape for shape in choices if shape[0] != pattern.shape[0] and shape[1] == pattern.shape[1]]
        both = [shape for shape in choices if shape[0] != pattern.shape[0] and shape[1] != pattern.shape[1]]
        buckets = [bucket for bucket in (same_row, same_col, both) if bucket]
        shape = random.choice(random.choice(buckets))
        specs.append({"color": color, "angle": angle, "shape": shape})
    return specs


def _rect_ok(occ: np.ndarray, top: int, left: int, shape: tuple[int, int], gap: int = 1) -> bool:
    h, w = shape
    h0, w0 = occ.shape
    if top < 0 or left < 0 or top + h > h0 or left + w > w0:
        return False
    r0 = max(0, top - gap)
    c0 = max(0, left - gap)
    r1 = min(h0, top + h + gap)
    c1 = min(w0, left + w + gap)
    return not occ[r0:r1, c0:c1].any()


def _place_rectangles(main_shape: tuple[int, int], shapes: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
    occ = np.zeros(main_shape, dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(shapes)
    order = sorted(range(len(shapes)), key=lambda i: shapes[i][0] * shapes[i][1], reverse=True)
    for idx in order:
        h, w = shapes[idx]
        candidates = []
        for top in range(main_shape[0] - h + 1):
            for left in range(main_shape[1] - w + 1):
                if _rect_ok(occ, top, left, (h, w), gap=1):
                    candidates.append((top, left))
        if not candidates:
            return None
        top, left = random.choice(candidates)
        occ[top : top + h, left : left + w] = True
        placements[idx] = (top, left)
    return [p for p in placements if p is not None]


def _build_legend(width: int, target_specs: list[dict]) -> np.ndarray | None:
    legend = np.full((2, width), LEGEND_BG, dtype=int)
    starts = []
    for _ in range(100):
        starts = sorted(random.sample(range(width - 1), k=len(target_specs)))
        if all(starts[i + 1] - starts[i] >= 3 for i in range(len(starts) - 1)):
            break
    if not starts or any(starts[i + 1] - starts[i] < 3 for i in range(len(starts) - 1)):
        return None
    for start, spec in zip(starts, target_specs):
        if start + 2 > width:
            return None
        color = spec["color"]
        marker = ANGLE_TO_MARKER[spec["angle"]]
        legend[:, start : start + 2] = color
        legend[marker[0], start + marker[1]] = BG
    return legend


def _make_instance() -> tuple[np.ndarray, np.ndarray] | None:
    target_count = random.choices([2, 3, 4], weights=[1, 3, 3])[0]
    target_angles = random.sample(list(ANGLES), k=target_count)
    usable = list(COLOR_POOL)
    random.shuffle(usable)
    target_colors = usable[:target_count]
    exemplar_colors = usable[target_count : target_count + 4]
    if len(exemplar_colors) < 3:
        return None
    exemplar = _build_random_exemplar(exemplar_colors)
    if exemplar is None:
        return None

    target_specs = _choose_target_specs(exemplar, target_colors, target_angles)
    if target_specs is None:
        return None

    shapes = [exemplar.shape] + [spec["shape"] for spec in target_specs]
    max_h = max(h for h, _ in shapes)
    max_w = max(w for _, w in shapes)
    for _ in range(80):
        main_h = random.randint(max(12, max_h + 4), min(24, max_h + 12))
        main_w = random.randint(max(18, max_w + 8), 28)
        placements = _place_rectangles((main_h, main_w), shapes)
        if placements is None:
            continue
        legend = _build_legend(main_w, target_specs)
        if legend is None:
            continue

        main_grid = np.zeros((main_h, main_w), dtype=int)
        exp_top, exp_left = placements[0]
        eh, ew = exemplar.shape
        main_grid[exp_top : exp_top + eh, exp_left : exp_left + ew] = exemplar

        expected = main_grid.copy()
        for (top, left), spec in zip(placements[1:], target_specs):
            h, w = spec["shape"]
            color = spec["color"]
            angle = spec["angle"]
            main_grid[top : top + h, left : left + w] = color
            expected[top : top + h, left : left + w] = _stretch_pattern(
                _rotate_clockwise(exemplar, angle), (h, w)
            )

        grid = np.vstack([legend, main_grid])
        if not _semantic_check(grid, expected, target_specs):
            continue
        return grid, expected
    return None


def _semantic_check(grid: np.ndarray, expected: np.ndarray, target_specs: list[dict]) -> bool:
    try:
        produced = main(grid)
    except Exception:
        return False
    if not np.array_equal(produced, expected):
        return False

    legend = grid[:2]
    mapping = _parse_legend(legend)
    if set(mapping) != {spec["color"] for spec in target_specs}:
        return False
    if len(set(mapping.values())) != len(mapping):
        return False

    base = grid[2:]
    regions = _extract_regions(base)
    if len(regions) != len(target_specs) + 1:
        return False
    exemplar_region, _ = _find_exemplar(regions)
    if exemplar_region["patch"].shape[0] == exemplar_region["patch"].shape[1]:
        return False

    pair_shapes = {}
    for spec in target_specs:
        pair_shapes.setdefault(spec["shape"], []).append(spec)
    if not any(
        len(specs) >= 2
        and len({spec["angle"] for spec in specs}) >= 2
        and len(
            {
                _stretch_pattern(
                    _rotate_clockwise(exemplar_region["patch"], spec["angle"]),
                    spec["shape"],
                ).tobytes()
                for spec in specs
            }
        )
        >= 2
        for specs in pair_shapes.values()
    ):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        built = _make_instance()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid ARC instance")
