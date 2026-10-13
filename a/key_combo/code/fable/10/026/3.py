from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
DIR8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]
TEMPLATE_LIBRARY = [
    np.array([[1, 1, 0], [0, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 1, 0], [0, 1, 0, 0], [0, 1, 1, 1]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1], [0, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 1, 0, 0], [0, 1, 1, 1]], dtype=bool),
    np.array([[0, 1, 0, 0], [1, 1, 1, 0], [0, 1, 0, 1], [0, 1, 1, 1]], dtype=bool),
]
MAX_ATTEMPTS = 500


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_of_color(grid: np.ndarray, color: int, connectivity: int = 8) -> list[np.ndarray]:
    mask = grid == color
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    nbrs = DIR8 if connectivity == 8 else [(-1, 0), (1, 0), (0, -1), (0, 1)]
    h, w = grid.shape
    for sr, sc in np.argwhere(mask):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue
        q = deque([(sr, sc)])
        seen[sr, sc] = True
        cells: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            cells.append((r, c))
            for dr, dc in nbrs:
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    q.append((rr, cc))
        comps.append(np.array(cells, dtype=int))
    return comps


def _sign(x: float) -> int:
    return int((x > 0) - (x < 0))


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for dr, dc in DIR8]


def _base_and_legends(grid: np.ndarray) -> tuple[list[int], dict[int, int], dict[int, tuple[int, int]]]:
    counts = Counter(map(int, grid.ravel()))
    assert counts[GREY] == 1, "expected exactly one grey indicator"
    assert BLACK in counts and counts[BLACK] >= 2, "expected a black template"

    other = [color for color in counts if color not in (BLACK, GREY)]
    assert len(other) == 4, "expected two base colors and two legend colors"
    ordered = sorted(other, key=lambda color: (-counts[color], color))
    base_colors = ordered[:2]
    legend_colors = ordered[2:]
    assert counts[base_colors[1]] > counts[legend_colors[0]], "base/legend split must be unambiguous"

    base_to_legend: dict[int, int] = {}
    legend_pos: dict[int, tuple[int, int]] = {}
    for legend in legend_colors:
        pos = np.argwhere(grid == legend)
        assert pos.shape == (1, 2), "each legend color must appear once"
        r, c = map(int, pos[0])
        legend_pos[legend] = (r, c)
        neighbor_bases = [int(grid[rr, cc]) for rr, cc in _neighbors8(r, c) if _in_bounds(grid.shape, rr, cc) and int(grid[rr, cc]) in base_colors]
        assert neighbor_bases, "legend must touch its base region"
        local_counts = Counter(neighbor_bases)
        best = local_counts.most_common()
        assert len(best) == 1 or best[0][1] > best[1][1], "legend base must be unique"
        base_color = int(best[0][0])
        assert base_color not in base_to_legend, "each base color must have one legend color"
        base_to_legend[base_color] = legend

    assert set(base_to_legend) == set(base_colors)
    return base_colors, base_to_legend, legend_pos


def _region_labels(grid: np.ndarray, base_colors: list[int], legend_pos: dict[int, tuple[int, int]]) -> np.ndarray:
    label_of_base = {int(base_colors[0]): 0, int(base_colors[1]): 1}
    labels = np.full(grid.shape, -1, dtype=int)
    for base_color, label in label_of_base.items():
        labels[grid == base_color] = label

    for legend, (r, c) in legend_pos.items():
        neighbor_labels = [int(labels[rr, cc]) for rr, cc in _neighbors8(r, c) if _in_bounds(grid.shape, rr, cc) and labels[rr, cc] != -1]
        assert neighbor_labels, "legend must have known neighboring base cells"
        local_counts = Counter(neighbor_labels)
        best = local_counts.most_common()
        assert len(best) == 1 or best[0][1] > best[1][1], "legend label must be unique"
        labels[r, c] = int(best[0][0])

    h, w = grid.shape
    seen = np.zeros(grid.shape, dtype=bool)
    for sr in range(h):
        for sc in range(w):
            if labels[sr, sc] != -1 or seen[sr, sc]:
                continue
            q = deque([(sr, sc)])
            seen[sr, sc] = True
            cells: list[tuple[int, int]] = []
            boundary = set()
            while q:
                r, c = q.popleft()
                cells.append((r, c))
                for rr, cc in _neighbors4(r, c):
                    if not _in_bounds(grid.shape, rr, cc):
                        continue
                    if labels[rr, cc] == -1:
                        if not seen[rr, cc]:
                            seen[rr, cc] = True
                            q.append((rr, cc))
                    else:
                        boundary.add(int(labels[rr, cc]))
            assert len(boundary) == 1, "hidden hole must belong to a unique base region"
            fill_label = boundary.pop()
            for r, c in cells:
                labels[r, c] = fill_label
    return labels


def _parse_template(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int], tuple[int, int], tuple[int, int], tuple[int, int]]:
    comps = _components_of_color(grid, BLACK, connectivity=8)
    assert len(comps) == 1, "expected exactly one black template"
    coords = comps[0]
    r0, c0, r1, c1 = _bbox(coords)
    indicator = np.argwhere(grid == GREY)
    ir, ic = map(int, indicator[0])
    adjacent = [(int(r), int(c)) for r, c in coords if max(abs(int(r) - ir), abs(int(c) - ic)) == 1]
    assert len(adjacent) == 1, "indicator must identify exactly one probe cell"
    probe = adjacent[0]
    center_r = (r0 + r1) / 2.0
    center_c = (c0 + c1) / 2.0
    direction = (_sign(ir - center_r), _sign(ic - center_c))
    assert direction != (0, 0), "indicator must determine a direction"
    return coords, (r0, c0, r1, c1), (ir, ic), probe, direction


def _copy_records(
    shape: tuple[int, int],
    rel_coords: np.ndarray,
    top_left: tuple[int, int],
    probe_rel: tuple[int, int],
    step: tuple[int, int],
    labels: np.ndarray,
) -> list[dict]:
    h, w = shape
    records: list[dict] = []
    k = 0
    while True:
        shift_r = top_left[0] + k * step[0]
        shift_c = top_left[1] + k * step[1]
        probe_r = shift_r + probe_rel[0]
        probe_c = shift_c + probe_rel[1]
        if not _in_bounds(shape, probe_r, probe_c):
            break

        abs_coords = rel_coords + np.array([[shift_r, shift_c]])
        visible = []
        for r, c in abs_coords:
            rr = int(r)
            cc = int(c)
            if _in_bounds(shape, rr, cc):
                visible.append((rr, cc))
        if not visible:
            break

        records.append(
            {
                "probe": (probe_r, probe_c),
                "label": int(labels[probe_r, probe_c]),
                "visible": visible,
                "full": len(visible) == len(rel_coords),
            }
        )
        k += 1
        if k > max(shape) + 2:
            raise AssertionError("unexpectedly long copy chain")
    return records


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repeat the black template along the indicator direction and recolor each copy by the legend of the probe cell's base region.

    Input structure:
    - The canvas is filled by two large base-color regions with an irregular boundary.
    - Each base-color region contains one isolated legend pixel of a distinct non-base color.
    - One black template touches a single grey indicator pixel.

    Rule:
    - The template cell adjacent to the grey indicator is the probe cell.
    - Let the movement direction be the sign of the vector from the template bbox center to the indicator; translate the template by `(bbox_height + 1, bbox_width + 1)` in that direction, repeating from the original copy outward.
    - Recolor every copy uniformly to the legend color of the base-color region containing that copy's probe cell; clip pasted pixels at the grid edge and keep the grey indicator and both legend pixels unchanged.

    Key insight:
    - The indicator fixes two things at once: the travel direction and the template-relative probe cell whose landing region chooses the copy color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30

    base_colors, base_to_legend, legend_pos = _base_and_legends(g)
    labels = _region_labels(g, base_colors, legend_pos)
    coords, bbox, indicator, probe, direction = _parse_template(g)
    r0, c0, r1, c1 = bbox
    step = (direction[0] * (r1 - r0 + 2), direction[1] * (c1 - c0 + 2))
    rel_coords = coords - np.array([[r0, c0]])
    probe_rel = (probe[0] - r0, probe[1] - c0)
    records = _copy_records(g.shape, rel_coords, (r0, c0), probe_rel, step, labels)
    assert records, "expected at least the original copy"

    region_to_legend = {int(labels[rr, cc]): legend for legend, (rr, cc) in legend_pos.items()}
    protected = {indicator, *legend_pos.values()}
    out = g.copy()
    for record in records:
        color = int(region_to_legend[int(record["label"])])
        for cell in record["visible"]:
            if cell not in protected:
                out[cell] = color
    return out


def _transformed_template(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    if random.random() < 0.5:
        out = np.fliplr(out)
    out = np.rot90(out, random.randrange(4))
    coords = np.argwhere(out)
    r0, c0, r1, c1 = _bbox(coords)
    return out[r0 : r1 + 1, c0 : c1 + 1]


def _template_options(mask: np.ndarray) -> list[dict]:
    coords = np.argwhere(mask).astype(int)
    r0, c0, r1, c1 = _bbox(coords)
    center_r = (r0 + r1) / 2.0
    center_c = (c0 + c1) / 2.0
    shape_cells = {tuple(map(int, cell)) for cell in coords}
    candidates = set()
    for r, c in shape_cells:
        for dr, dc in DIR8:
            ir = r + dr
            ic = c + dc
            if (ir, ic) not in shape_cells:
                candidates.add((ir, ic))

    options = []
    for ir, ic in candidates:
        adjacent = [(r, c) for r, c in shape_cells if max(abs(r - ir), abs(c - ic)) == 1]
        if len(adjacent) != 1:
            continue
        direction = (_sign(ir - center_r), _sign(ic - center_c))
        if direction == (0, 0):
            continue
        probe = adjacent[0]
        options.append(
            {
                "coords": coords.copy(),
                "bbox": (r0, c0, r1, c1),
                "indicator_rel": (int(ir), int(ic)),
                "probe_rel": (int(probe[0]), int(probe[1])),
                "direction": direction,
            }
        )
    return options


def _boundary_partition(h: int, w: int) -> np.ndarray:
    vertical = random.random() < 0.5
    if vertical:
        labels = np.zeros((h, w), dtype=int)
        split = random.randint(w // 3, (2 * w) // 3)
        for r in range(h):
            if r > 0:
                split = max(4, min(w - 5, split + random.choice([-2, -1, 0, 1, 2])))
            labels[r, split + 1 :] = 1
    else:
        labels = np.zeros((h, w), dtype=int)
        split = random.randint(h // 3, (2 * h) // 3)
        for c in range(w):
            if c > 0:
                split = max(4, min(h - 5, split + random.choice([-2, -1, 0, 1, 2])))
            labels[split + 1 :, c] = 1
    return labels


def _interior_cells(labels: np.ndarray, region: int) -> list[tuple[int, int]]:
    h, w = labels.shape
    cells = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(labels[r, c]) != region:
                continue
            if all(int(labels[rr, cc]) == region for rr, cc in _neighbors8(r, c)):
                cells.append((r, c))
    return cells


def _placement_records(
    labels: np.ndarray,
    option: dict,
    top_left: tuple[int, int],
) -> list[dict] | None:
    coords = option["coords"]
    r0 = int(top_left[0] + option["bbox"][0])
    c0 = int(top_left[1] + option["bbox"][1])
    r1 = int(top_left[0] + option["bbox"][2])
    c1 = int(top_left[1] + option["bbox"][3])
    expand = (r0 - 1, c0 - 1, r1 + 1, c1 + 1)
    if min(expand) < 0 or expand[2] >= labels.shape[0] or expand[3] >= labels.shape[1]:
        return None
    patch = labels[expand[0] : expand[2] + 1, expand[1] : expand[3] + 1]
    if len(set(map(int, patch.ravel()))) != 1:
        return None

    indicator = (top_left[0] + option["indicator_rel"][0], top_left[1] + option["indicator_rel"][1])
    if not _in_bounds(labels.shape, indicator[0], indicator[1]):
        return None
    if int(labels[indicator]) != int(patch[0, 0]):
        return None

    h = option["bbox"][2] - option["bbox"][0] + 1
    w = option["bbox"][3] - option["bbox"][1] + 1
    step = (option["direction"][0] * (h + 1), option["direction"][1] * (w + 1))
    rel_coords = coords - np.array([[option["bbox"][0], option["bbox"][1]]])
    probe_rel = (option["probe_rel"][0] - option["bbox"][0], option["probe_rel"][1] - option["bbox"][1])
    return _copy_records(labels.shape, rel_coords, (r0, c0), probe_rel, step, labels)


def _choose_legend_positions(
    labels: np.ndarray,
    records: list[dict],
    occupied: set[tuple[int, int]],
    want_overlap: bool,
) -> dict[int, tuple[int, int]] | None:
    candidates = {region: [cell for cell in _interior_cells(labels, region) if cell not in occupied] for region in (0, 1)}
    if not candidates[0] or not candidates[1]:
        return None

    chosen: dict[int, tuple[int, int]] = {}
    if want_overlap:
        overlap_choices = []
        for idx, record in enumerate(records[1:], start=1):
            region = int(record["label"])
            probe = tuple(map(int, record["probe"]))
            for cell in record["visible"]:
                if cell == probe or cell in occupied:
                    continue
                if cell in candidates[region]:
                    overlap_choices.append((idx, region, cell))
        if overlap_choices:
            _idx, region, cell = random.choice(overlap_choices)
            chosen[region] = cell
        else:
            return None

    used = set(chosen.values())
    for region in (0, 1):
        if region in chosen:
            continue
        pool = [cell for cell in candidates[region] if cell not in used]
        if not pool:
            return None
        chosen[region] = random.choice(pool)
        used.add(chosen[region])
    return chosen


def _paint_expected(
    grid: np.ndarray,
    records: list[dict],
    region_to_legend: dict[int, int],
    protected: set[tuple[int, int]],
) -> np.ndarray:
    out = grid.copy()
    for record in records:
        color = int(region_to_legend[int(record["label"])])
        for cell in record["visible"]:
            if cell not in protected:
                out[cell] = color
    return out


def _build_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(16, 28)
    w = random.randint(16, 28)
    labels = _boundary_partition(h, w)
    ratio = float(np.mean(labels == 0))
    if not (0.35 <= ratio <= 0.65):
        return None
    if min(len(_interior_cells(labels, 0)), len(_interior_cells(labels, 1))) < 20:
        return None

    base0, base1, legend0, legend1 = random.sample(PALETTE, 4)
    mask = _transformed_template(random.choice(TEMPLATE_LIBRARY))
    options = _template_options(mask)
    if not options:
        return None

    want_clipped = random.random() < 0.5
    want_overlap = random.random() < 0.35
    placements = []
    for option in options:
        min_r = -int(option["bbox"][0])
        max_r = h - int(option["bbox"][2]) - 1
        min_c = -int(option["bbox"][1])
        max_c = w - int(option["bbox"][3]) - 1
        for top_r in range(min_r, max_r + 1):
            for top_c in range(min_c, max_c + 1):
                records = _placement_records(labels, option, (top_r, top_c))
                if records is None or len(records) < 3:
                    continue
                label_seq = [int(record["label"]) for record in records]
                if len(set(label_seq)) != 2:
                    continue
                clipped_count = sum(not bool(record["full"]) for record in records)
                if want_clipped and clipped_count == 0:
                    continue
                if not want_clipped and clipped_count != 0:
                    continue
                if len(records) > 6:
                    continue
                placements.append((option, (top_r, top_c), records))
    if not placements:
        return None

    option, top_left, records = random.choice(placements)
    coords = option["coords"] + np.array([[top_left[0], top_left[1]]])
    indicator = (top_left[0] + option["indicator_rel"][0], top_left[1] + option["indicator_rel"][1])
    occupied = {tuple(map(int, cell)) for cell in coords}
    occupied.add(indicator)
    legend_pos = _choose_legend_positions(labels, records, occupied, want_overlap)
    if legend_pos is None:
        return None

    grid = np.where(labels == 0, base0, base1).astype(int)
    grid[legend_pos[0]] = legend0
    grid[legend_pos[1]] = legend1
    for r, c in coords:
        grid[int(r), int(c)] = BLACK
    grid[indicator] = GREY

    protected = {indicator, legend_pos[0], legend_pos[1]}
    expected = _paint_expected(grid, records, {0: legend0, 1: legend1}, protected)

    overlap_count = sum(cell in {legend_pos[0], legend_pos[1]} for record in records[1:] for cell in record["visible"])
    if want_overlap and overlap_count == 0:
        return None
    if np.array_equal(expected, grid):
        return None
    if len(set(int(record["label"]) for record in records)) != 2:
        return None
    if len(set(int(x) for x in expected.ravel())) < 5:
        return None
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        built = _build_attempt()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid probe-coloring ARC input")
