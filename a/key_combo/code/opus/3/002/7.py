import importlib.util
from collections import deque
from pathlib import Path

import numpy as np


_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
_ARC_UTILS_SPEC = importlib.util.spec_from_file_location("arc_utils_for_task_7", _ARC_UTILS_PATH)
assert _ARC_UTILS_SPEC is not None and _ARC_UTILS_SPEC.loader is not None
_ARC_UTILS = importlib.util.module_from_spec(_ARC_UTILS_SPEC)
_ARC_UTILS_SPEC.loader.exec_module(_ARC_UTILS)

Color = _ARC_UTILS.Color
bbox_coords = _ARC_UTILS.bbox_coords
bbox_h = _ARC_UTILS.bbox_h
bbox_w = _ARC_UTILS.bbox_w
crop_bbox = _ARC_UTILS.crop_bbox
find_components = _ARC_UTILS.find_components
in_bounds = _ARC_UTILS.in_bounds
label_mask = _ARC_UTILS.label_mask
neighbors4 = _ARC_UTILS.neighbors4


RNG = np.random.default_rng()
ATTEMPT_LIMIT = 400

LEFT_BG = Color.GREY
WALL = Color.RED
SEPARATOR = Color.BLACK
TEMPLATE_COLOR = Color.BLUE
MIRROR_COLOR = Color.GREEN

HEIGHT_RANGE = (12, 19)
LEFT_WIDTH_RANGE = (10, 15)
RIGHT_WIDTH_RANGE = (4, 7)
WALL_COUNT_RANGE = (2, 5)

TEMPLATE_AREA_RANGE = (4, 6)
TEMPLATE_BOX_SHAPE = (3, 3)


def _round_half_up(value: float) -> int:
    return int(np.floor(float(value) + 0.5))


def _region_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    r0, c0, r1, c1 = bbox
    return (r0 + r1) / 2.0, (c0 + c1) / 2.0


def _find_separator_column(grid: np.ndarray) -> int:
    candidates: list[int] = []
    height, width = grid.shape
    for col in range(1, width - 1):
        if not np.all(grid[:, col] == SEPARATOR):
            continue
        left = grid[:, :col]
        right = grid[:, col + 1 :]
        if left.size == 0 or right.size == 0:
            continue

        left_colors = set(map(int, np.unique(left)))
        right_colors = set(map(int, np.unique(right)))
        if left_colors <= {LEFT_BG, WALL} and LEFT_BG in left_colors and right_colors <= {SEPARATOR, TEMPLATE_COLOR} and TEMPLATE_COLOR in right_colors:
            candidates.append(col)

    assert len(candidates) == 1, f"expected one separator column, found {candidates}"
    return int(candidates[0])


def _extract_template(right_panel: np.ndarray) -> np.ndarray:
    components = find_components(
        right_panel,
        bg=SEPARATOR,
        connectivity=4,
        monochromatic=True,
        colors=[TEMPLATE_COLOR],
        return_format="dict",
    )
    assert len(components) == 1, "expected exactly one blue template component"
    bbox = components[0]["bbox"]
    template = crop_bbox(right_panel == TEMPLATE_COLOR, bbox)
    assert template.shape == TEMPLATE_BOX_SHAPE, f"expected template bbox {TEMPLATE_BOX_SHAPE}, got {template.shape}"
    assert template.sum() in range(*TEMPLATE_AREA_RANGE), "template area must be 4 or 5"
    assert not np.array_equal(template, template[:, ::-1]), "template must be horizontally asymmetric"
    return template


def _build_region_graph(left_panel: np.ndarray) -> tuple[np.ndarray, dict[int, set[int]]]:
    region_labels, n_regions = label_mask(left_panel == LEFT_BG, connectivity=4)
    assert n_regions >= 2, "walls must split the grey background"
    graph: dict[int, set[int]] = {region_id: set() for region_id in range(1, n_regions + 1)}

    height, width = left_panel.shape
    for r, c in np.argwhere(left_panel == WALL):
        r = int(r)
        c = int(c)
        if 0 < r < height - 1:
            a = int(region_labels[r - 1, c])
            b = int(region_labels[r + 1, c])
            if a > 0 and b > 0 and a != b:
                graph[a].add(b)
                graph[b].add(a)
        if 0 < c < width - 1:
            a = int(region_labels[r, c - 1])
            b = int(region_labels[r, c + 1])
            if a > 0 and b > 0 and a != b:
                graph[a].add(b)
                graph[b].add(a)

    return region_labels, graph


def _bipartition(graph: dict[int, set[int]], start_region: int) -> dict[int, int]:
    parity = {int(start_region): 0}
    queue: deque[int] = deque([int(start_region)])
    while queue:
        region = int(queue.popleft())
        for neighbor in sorted(graph[region]):
            expected = 1 - parity[region]
            if neighbor in parity:
                assert parity[neighbor] == expected, "region graph must be bipartite"
            else:
                parity[neighbor] = expected
                queue.append(neighbor)

    assert set(parity) == set(graph), "all regions must belong to one wall-adjacency graph"
    return parity


def _clip_stamp_cells(
    region_labels: np.ndarray,
    region_id: int,
    region_bbox: tuple[int, int, int, int],
    template_mask: np.ndarray,
) -> set[tuple[int, int]]:
    height, width = template_mask.shape
    region_center_r, region_center_c = _region_center(region_bbox)
    template_center_r = (height - 1) / 2.0
    template_center_c = (width - 1) / 2.0
    top = _round_half_up(region_center_r - template_center_r)
    left = _round_half_up(region_center_c - template_center_c)

    kept: set[tuple[int, int]] = set()
    for tr, tc in np.argwhere(template_mask):
        rr = int(top + tr)
        cc = int(left + tc)
        if in_bounds(region_labels.shape, rr, cc) and int(region_labels[rr, cc]) == int(region_id):
            kept.add((rr, cc))
    return kept


def _solve(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    separator_col = _find_separator_column(g)
    left_panel = g[:, :separator_col]
    right_panel = g[:, separator_col + 1 :]

    assert int(left_panel[0, 0]) == LEFT_BG, "left-panel upper-left cell must be inside the A-side seed region"

    template = _extract_template(right_panel)
    region_labels, graph = _build_region_graph(left_panel)

    start_region = int(region_labels[0, 0])
    assert start_region > 0, "upper-left cell must be in a grey region"
    parity = _bipartition(graph, start_region)

    out = g.copy()
    region_bboxes: dict[int, tuple[int, int, int, int]] = {}
    for region_id in sorted(graph):
        coords = np.argwhere(region_labels == region_id)
        region_bbox = bbox_coords(coords)
        region_bboxes[region_id] = region_bbox

        if parity[region_id] == 0:
            stamp = template
            color = TEMPLATE_COLOR
        else:
            stamp = template[:, ::-1]
            color = MIRROR_COLOR

        for rr, cc in _clip_stamp_cells(region_labels, region_id, region_bbox, stamp):
            out[rr, cc] = color

    info = {
        "separator_col": separator_col,
        "left_panel": left_panel,
        "right_panel": right_panel,
        "template": template,
        "region_labels": region_labels,
        "graph": graph,
        "parity": parity,
        "region_bboxes": region_bboxes,
    }
    return out, info


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp a wall-partitioned map with the template's original or mirrored orientation according to the regions' bipartite side.

    Input structure:
    - The grid contains a left panel of grey background cut by one-cell-thick red walls, a full black separator column, and a right panel on black background holding one blue connected template.
    - The red walls split the grey left panel into multiple 4-connected background regions.
    - The right-panel template is horizontally asymmetric, so its mirror is visibly different.

    Rule:
    - Label the left-panel grey regions and connect two region labels whenever a red wall pixel has those two labels on opposite sides.
    - Mark the region containing the left panel's upper-left pixel as side A, then alternate A/B across every region-graph edge.
    - Extract the blue template from the right panel.
    - For every A-side region, stamp the original blue template at that region bbox center; for every B-side region, stamp the horizontally mirrored template in green at that region bbox center.
    - Keep the walls, separator, and right panel unchanged, and clip away any stamped pixels that would land outside the target region or on walls.

    Key insight:
    - The red wall topology defines a bipartite region graph, and the two graph sides control template orientation rather than just a flat recoloring.
    """
    return _solve(grid)[0]


def _generate_template_mask() -> np.ndarray:
    for _ in range(200):
        target_area = int(RNG.integers(*TEMPLATE_AREA_RANGE))
        cells = {(1, 1)}
        while len(cells) < target_area:
            base_r, base_c = tuple(cells)[int(RNG.integers(len(cells)))]
            options = [
                (nr, nc)
                for nr, nc in neighbors4(base_r, base_c)
                if 0 <= nr < TEMPLATE_BOX_SHAPE[0] and 0 <= nc < TEMPLATE_BOX_SHAPE[1] and (nr, nc) not in cells
            ]
            if not options:
                break
            cells.add(options[int(RNG.integers(len(options)))])

        if len(cells) != target_area:
            continue

        mask = np.zeros(TEMPLATE_BOX_SHAPE, dtype=bool)
        for r, c in cells:
            mask[int(r), int(c)] = True

        if bbox_h(bbox_coords(np.argwhere(mask))) != TEMPLATE_BOX_SHAPE[0]:
            continue
        if bbox_w(bbox_coords(np.argwhere(mask))) != TEMPLATE_BOX_SHAPE[1]:
            continue
        if np.array_equal(mask, mask[:, ::-1]):
            continue
        return mask

    raise RuntimeError("failed to generate an asymmetric connected template")


def _split_candidates(
    region: tuple[int, int, int, int],
    min_region_area: int,
) -> list[tuple[str, int, int, int]]:
    r0, c0, r1, c1 = region
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    candidates: list[tuple[str, int, int, int]] = []

    for wall_col in range(c0 + 1, c1):
        left_width = wall_col - c0
        right_width = c1 - wall_col
        if left_width * height >= min_region_area and right_width * height >= min_region_area:
            candidates.append(("v", wall_col, left_width, right_width))

    for wall_row in range(r0 + 1, r1):
        top_height = wall_row - r0
        bottom_height = r1 - wall_row
        if top_height * width >= min_region_area and bottom_height * width >= min_region_area:
            candidates.append(("h", wall_row, top_height, bottom_height))

    return candidates


def _apply_split(
    grid: np.ndarray,
    regions: list[tuple[int, int, int, int]],
    region_index: int,
    split: tuple[str, int, int, int],
) -> None:
    r0, c0, r1, c1 = regions.pop(region_index)
    orientation, wall_pos, _side_a, _side_b = split

    if orientation == "v":
        grid[r0 : r1 + 1, wall_pos] = WALL
        regions.append((r0, c0, r1, wall_pos - 1))
        regions.append((r0, wall_pos + 1, r1, c1))
    else:
        grid[wall_pos, c0 : c1 + 1] = WALL
        regions.append((r0, c0, wall_pos - 1, c1))
        regions.append((wall_pos + 1, c0, r1, c1))


def _choose_split(
    options: list[tuple[int, tuple[str, int, int, int]]],
    prefer_narrow: bool,
) -> tuple[int, tuple[str, int, int, int]]:
    if not prefer_narrow:
        return options[int(RNG.integers(len(options)))]

    widths = np.array([min(option[1][2], option[1][3]) for option in options], dtype=float)
    weights = 1.0 / widths
    weights /= weights.sum()
    choice = int(RNG.choice(len(options), p=weights))
    return options[choice]


def _generate_left_panel(height: int, width: int, min_region_area: int, want_clip: bool) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    grid = np.full((height, width), LEFT_BG, dtype=int)
    regions = [(0, 0, height - 1, width - 1)]
    n_walls = int(RNG.integers(*WALL_COUNT_RANGE))

    for _ in range(n_walls):
        split_options: list[tuple[int, tuple[str, int, int, int]]] = []
        for region_index, region in enumerate(regions):
            for split in _split_candidates(region, min_region_area):
                split_options.append((region_index, split))

        if not split_options:
            raise ValueError("no feasible wall split")
        region_index, split = _choose_split(split_options, prefer_narrow=want_clip)
        _apply_split(grid, regions, region_index, split)

    return grid, regions


def _place_template(right_panel: np.ndarray, template_mask: np.ndarray) -> None:
    ph, pw = template_mask.shape
    top = int(RNG.integers(0, right_panel.shape[0] - ph + 1))
    left = int(RNG.integers(0, right_panel.shape[1] - pw + 1))
    right_panel[top : top + ph, left : left + pw][template_mask] = TEMPLATE_COLOR


def _visible_shape_difference(info: dict) -> bool:
    template = info["template"]
    mirrored = template[:, ::-1]
    for region_id, parity in info["parity"].items():
        if parity != 1:
            continue
        region_bbox = info["region_bboxes"][region_id]
        mirrored_cells = _clip_stamp_cells(info["region_labels"], region_id, region_bbox, mirrored)
        original_cells = _clip_stamp_cells(info["region_labels"], region_id, region_bbox, template)
        if mirrored_cells != original_cells:
            return True
    return False


def _count_clipped_regions(info: dict) -> int:
    clipped = 0
    for region_id, parity in info["parity"].items():
        stamp = info["template"] if parity == 0 else info["template"][:, ::-1]
        visible = _clip_stamp_cells(info["region_labels"], region_id, info["region_bboxes"][region_id], stamp)
        if len(visible) < int(stamp.sum()):
            clipped += 1
    return clipped


def _validate_generated_example(grid: np.ndarray, require_clip: bool | None) -> bool:
    try:
        out, info = _solve(grid)
    except AssertionError:
        return False

    template_area = int(info["template"].sum())
    template_bbox_area = int(np.prod(info["template"].shape))
    if not (4 <= template_area <= 5):
        return False

    if len(info["parity"]) < 3:
        return False
    if set(info["parity"].values()) != {0, 1}:
        return False

    for region_id in info["parity"]:
        area = int(np.sum(info["region_labels"] == region_id))
        if area < 2 * template_bbox_area:
            return False

    if not _visible_shape_difference(info):
        return False

    clipped_regions = _count_clipped_regions(info)
    if require_clip is True and clipped_regions == 0:
        return False
    if require_clip is False and clipped_regions != 0:
        return False

    if np.array_equal(out, grid):
        return False
    if np.count_nonzero(out == TEMPLATE_COLOR) == 0 or np.count_nonzero(out == MIRROR_COLOR) == 0:
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    height = int(RNG.integers(*HEIGHT_RANGE))
    left_width = int(RNG.integers(*LEFT_WIDTH_RANGE))
    right_width = int(RNG.integers(*RIGHT_WIDTH_RANGE))
    if max(height, left_width + 1 + right_width) > 30:
        return None

    template_mask = _generate_template_mask()
    require_clip = RNG.choice([True, False, None], p=[0.45, 0.25, 0.30])
    min_region_area = 2 * int(np.prod(template_mask.shape))

    try:
        left_panel, _regions = _generate_left_panel(height, left_width, min_region_area, want_clip=(require_clip is True))
    except ValueError:
        return None

    right_panel = np.full((height, right_width), SEPARATOR, dtype=int)
    _place_template(right_panel, template_mask)

    grid = np.concatenate(
        [left_panel, np.full((height, 1), SEPARATOR, dtype=int), right_panel],
        axis=1,
    )

    if _validate_generated_example(grid, require_clip=require_clip):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(ATTEMPT_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid wall-bipartite stamp input")
