from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import itertools
from pathlib import Path

import numpy as np


_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
_ARC_SPEC = importlib.util.spec_from_file_location("arc_utils_shared_041_8", _ARC_UTILS_PATH)
assert _ARC_SPEC is not None and _ARC_SPEC.loader is not None
_ARC_UTILS = importlib.util.module_from_spec(_ARC_SPEC)
_ARC_SPEC.loader.exec_module(_ARC_UTILS)

get_bg = _ARC_UTILS.get_bg
find_components = _ARC_UTILS.find_components


BG = 0
LID_COLOR = 8
ALL_OPENINGS = ("up", "down", "left", "right")
GEN_ATTEMPTS = 300
PALETTE = tuple(range(1, 10))


BBox = tuple[int, int, int, int]
Coord = tuple[int, int]


@dataclass(frozen=True)
class DetectedContainer:
    bbox: BBox
    wall_color: int
    opening: str
    sealed: bool
    lid_color: int | None


@dataclass(frozen=True)
class SceneContainer:
    name: str
    bbox: BBox
    wall_color: int
    opening: str
    sealed: bool
    lid_color: int | None


@dataclass(frozen=True)
class ScenePixel:
    token: str
    container_name: str
    coord: Coord
    color: int


def _inner_bbox(bbox: BBox) -> BBox:
    r0, c0, r1, c1 = bbox
    assert r1 - r0 >= 2 and c1 - c0 >= 2
    return (r0 + 1, c0 + 1, r1 - 1, c1 - 1)


def _interior_cells(bbox: BBox) -> list[Coord]:
    r0, c0, r1, c1 = _inner_bbox(bbox)
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def _opening_cells(bbox: BBox, opening: str) -> list[Coord]:
    r0, c0, r1, c1 = bbox
    if opening == "up":
        return [(r0, c) for c in range(c0 + 1, c1)]
    if opening == "down":
        return [(r1, c) for c in range(c0 + 1, c1)]
    if opening == "left":
        return [(r, c0) for r in range(r0 + 1, r1)]
    if opening == "right":
        return [(r, c1) for r in range(r0 + 1, r1)]
    raise ValueError(f"unknown opening {opening}")


def _wall_cells(bbox: BBox, opening: str) -> list[Coord]:
    r0, c0, r1, c1 = bbox
    cells: set[Coord] = set()
    if opening == "up":
        for c in range(c0, c1 + 1):
            cells.add((r1, c))
        for r in range(r0, r1 + 1):
            cells.add((r, c0))
            cells.add((r, c1))
    elif opening == "down":
        for c in range(c0, c1 + 1):
            cells.add((r0, c))
        for r in range(r0, r1 + 1):
            cells.add((r, c0))
            cells.add((r, c1))
    elif opening == "left":
        for c in range(c0, c1 + 1):
            cells.add((r0, c))
            cells.add((r1, c))
        for r in range(r0, r1 + 1):
            cells.add((r, c1))
    elif opening == "right":
        for c in range(c0, c1 + 1):
            cells.add((r0, c))
            cells.add((r1, c))
        for r in range(r0, r1 + 1):
            cells.add((r, c0))
    else:
        raise ValueError(f"unknown opening {opening}")
    return sorted(cells)


def _classify_u_component(coords: np.ndarray, bbox: BBox) -> str | None:
    r0, c0, r1, c1 = bbox
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None
    coord_set = {tuple(map(int, coord)) for coord in coords}
    matches = [opening for opening in ALL_OPENINGS if coord_set == set(_wall_cells(bbox, opening))]
    if len(matches) != 1:
        return None
    return matches[0]


def _detect_containers(g: np.ndarray) -> tuple[list[DetectedContainer], dict[Coord, int]]:
    bg = int(get_bg(g))
    components = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")

    containers: list[DetectedContainer] = []
    wall_cells: set[Coord] = set()
    opening_cells: set[Coord] = set()
    for comp in components:
        bbox = tuple(map(int, comp["bbox"]))
        opening = _classify_u_component(comp["coords"], bbox)
        if opening is None:
            continue

        cell_list = _opening_cells(bbox, opening)
        values = {int(g[r, c]) for r, c in cell_list}
        if values == {bg}:
            sealed = False
            lid_color = None
        else:
            assert len(values) == 1 and bg not in values, "opening must be entirely empty or fully sealed"
            sealed = True
            lid_color = int(next(iter(values)))

        for cell in _wall_cells(bbox, opening):
            assert cell not in wall_cells, "container walls must not overlap"
            wall_cells.add(cell)
        for cell in cell_list:
            assert cell not in opening_cells, "openings must not overlap"
            opening_cells.add(cell)

        containers.append(
            DetectedContainer(
                bbox=bbox,
                wall_color=int(comp["color"]),
                opening=opening,
                sealed=sealed,
                lid_color=lid_color,
            )
        )

    assert containers, "expected at least one U-shaped container"
    containers.sort(key=lambda item: (item.bbox[0], item.bbox[1], item.bbox[2], item.bbox[3]))

    owner_by_content: dict[Coord, int] = {}
    lid_coords = {
        cell
        for container in containers
        if container.sealed
        for cell in _opening_cells(container.bbox, container.opening)
    }

    for r_raw, c_raw in np.argwhere(g != bg):
        cell = (int(r_raw), int(c_raw))
        if cell in wall_cells or cell in lid_coords:
            continue

        owners = [
            idx
            for idx, container in enumerate(containers)
            if _in_interior(container.bbox, cell)
        ]
        assert len(owners) == 1, "every non-wall, non-lid pixel must belong to exactly one container interior"
        owner_by_content[cell] = owners[0]

    return containers, owner_by_content


def _in_interior(bbox: BBox, cell: Coord) -> bool:
    r, c = cell
    r0, c0, r1, c1 = bbox
    return r0 < r < r1 and c0 < c < c1


def _settle_by_column(
    base: np.ndarray,
    blocker_mask: np.ndarray,
    landing_mask: np.ndarray,
    moving_items: list[tuple[str, int, int, int]],
) -> tuple[np.ndarray, dict[str, Coord]]:
    out = base.copy()
    traces: dict[str, Coord] = {}
    height, width = map(int, base.shape)
    by_col: dict[int, list[tuple[int, str, int]]] = {}
    for token, r, c, color in moving_items:
        by_col.setdefault(int(c), []).append((int(r), token, int(color)))

    for c in range(width):
        items = sorted(by_col.get(c, []), key=lambda item: item[0])
        if not items:
            continue

        blockers = list(map(int, np.flatnonzero(blocker_mask[:, c])))
        prev_blocker = -1
        ptr = 0
        for blocker in blockers + [height]:
            segment_items: list[tuple[int, str, int]] = []
            while ptr < len(items) and items[ptr][0] < blocker:
                segment_items.append(items[ptr])
                ptr += 1
            landing_rows = [r for r in range(prev_blocker + 1, blocker) if landing_mask[r, c]]
            assert len(segment_items) <= len(landing_rows), "not enough landing cells in column segment"
            target_rows = landing_rows[len(landing_rows) - len(segment_items) :]
            for target_row, (_row, token, color) in zip(target_rows, segment_items):
                out[target_row, c] = color
                traces[token] = (target_row, c)
            prev_blocker = blocker

        assert ptr == len(items)

    return out, traces


def _simulate_detected(g: np.ndarray) -> np.ndarray:
    containers, owner_by_content = _detect_containers(g)
    bg = int(get_bg(g))
    base = g.copy()
    moving_items: list[tuple[str, int, int, int]] = []

    for cell, owner in owner_by_content.items():
        r, c = cell
        container = containers[owner]
        if container.sealed:
            continue
        moving_items.append((f"{r},{c}", r, c, int(g[r, c])))
        base[r, c] = bg

    blocker_mask = base != bg
    landing_mask = base == bg
    out, _ = _settle_by_column(base, blocker_mask, landing_mask, moving_items)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Let the contents of every unsealed U-container fall straight down through the shared scene geometry.

    Input structure:
    - The dominant background contains several one-cell-thick monochrome U-shaped containers, with openings that may face up, down, or sideways.
    - Some openings are completely covered by a straight lid line of another color.
    - Interior cells may contain colored content pixels, including suspended pixels.

    Rule:
    - A container whose entire opening is sealed keeps all of its content fixed in place.
    - Every content pixel inside an unsealed container falls vertically in its absolute grid column.
    - Container walls, every lid, the grid bottom, and already-settled pixels block motion; within each column, falling pixels keep their original top-to-bottom order.
    - Pixels may pass through a downward opening, continue through open air, land on another container's lid, or enter a lower upward-opening container; walls and lids themselves never move.

    Key insight:
    - Opening direction does not define a special flow direction: all motion is the same global vertical gravity, and only the exact wall-and-lid geometry decides which pixels move and where they finally rest.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30
    return _simulate_detected(g)


def _render_scene(shape: tuple[int, int], containers: list[SceneContainer], pixels: list[ScenePixel]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for container in containers:
        for r, c in _wall_cells(container.bbox, container.opening):
            assert grid[r, c] == BG
            grid[r, c] = container.wall_color
        if container.sealed:
            assert container.lid_color is not None
            for r, c in _opening_cells(container.bbox, container.opening):
                assert grid[r, c] == BG
                grid[r, c] = container.lid_color
    for pixel in pixels:
        r, c = pixel.coord
        assert grid[r, c] == BG
        grid[r, c] = pixel.color
    return grid


def _simulate_scene(
    shape: tuple[int, int],
    containers: list[SceneContainer],
    pixels: list[ScenePixel],
    *,
    move_sealed: bool = False,
    movable_openings: set[str] | None = None,
    lid_blocks: bool = True,
    synthetic_opening_floors: bool = False,
) -> tuple[np.ndarray, dict[str, Coord]]:
    base = np.full(shape, BG, dtype=int)
    blocker_mask = np.zeros(shape, dtype=bool)
    landing_mask = np.ones(shape, dtype=bool)
    by_name = {container.name: container for container in containers}

    for container in containers:
        for r, c in _wall_cells(container.bbox, container.opening):
            base[r, c] = container.wall_color
            blocker_mask[r, c] = True
            landing_mask[r, c] = False
        if container.sealed:
            assert container.lid_color is not None
            for r, c in _opening_cells(container.bbox, container.opening):
                base[r, c] = container.lid_color
                landing_mask[r, c] = False
                if lid_blocks:
                    blocker_mask[r, c] = True
        elif synthetic_opening_floors:
            for r, c in _opening_cells(container.bbox, container.opening):
                blocker_mask[r, c] = True
                landing_mask[r, c] = False

    moving_items: list[tuple[str, int, int, int]] = []
    for pixel in pixels:
        container = by_name[pixel.container_name]
        can_move = not container.sealed or move_sealed
        if movable_openings is not None and container.opening not in movable_openings:
            can_move = False

        r, c = pixel.coord
        if can_move:
            moving_items.append((pixel.token, r, c, pixel.color))
        else:
            base[r, c] = pixel.color
            blocker_mask[r, c] = True
            landing_mask[r, c] = False

    out, traces = _settle_by_column(base, blocker_mask, landing_mask, moving_items)
    return out, traces


def _content_palette(rng: np.random.Generator, wall_colors: list[int]) -> list[int]:
    content_colors = [color for color in PALETTE if color not in {LID_COLOR, *wall_colors}]
    rng.shuffle(content_colors)
    return content_colors


def _make_container(
    name: str,
    bbox: BBox,
    wall_color: int,
    opening: str,
    sealed: bool,
) -> SceneContainer:
    return SceneContainer(
        name=name,
        bbox=bbox,
        wall_color=wall_color,
        opening=opening,
        sealed=sealed,
        lid_color=LID_COLOR if sealed else None,
    )


def _scene_attempt(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    wall_colors = [color for color in PALETTE if color != LID_COLOR]
    rng.shuffle(wall_colors)
    wall_colors = wall_colors[:4]
    content_colors = _content_palette(rng, wall_colors)

    recv_open_w = int(rng.integers(2, 5))
    sealed_open_w = int(rng.integers(2, 5))
    air_gap_w = int(rng.integers(1, 3))
    floor_air_w = int(rng.integers(1, 3))

    source_inner_h = int(rng.integers(3, 5))
    recv_inner_h = int(rng.integers(4, 6))
    sealed_inner_h = int(rng.integers(3, 5))
    left_inner_w = int(rng.integers(2, 4))
    left_inner_h = int(rng.integers(3, 6))

    source_inner_w = recv_open_w + sealed_open_w + air_gap_w + 3 + floor_air_w
    source_w = source_inner_w + 2
    source_h = source_inner_h + 2
    recv_w = recv_open_w + 2
    recv_h = recv_inner_h + 2
    sealed_w = sealed_open_w + 2
    sealed_h = sealed_inner_h + 2
    left_w = left_inner_w + 2
    left_h = left_inner_h + 2

    left_c0 = int(rng.integers(1, 4))
    side_gap = int(rng.integers(3, 7))
    source_c0 = left_c0 + left_w + side_gap

    source_r0 = int(rng.integers(1, 4))
    vertical_gap = int(rng.integers(3, 7))
    recv_r0 = source_r0 + source_h + vertical_gap
    sealed_r0 = recv_r0 + int(rng.integers(-1, 2))
    left_r0 = max(1, recv_r0 + int(rng.integers(-2, 3)))

    recv_c0 = source_c0
    sealed_c0 = recv_c0 + recv_w + air_gap_w

    containers = [
        _make_container("source", (source_r0, source_c0, source_r0 + source_h - 1, source_c0 + source_w - 1), wall_colors[0], "down", False),
        _make_container("receiver", (recv_r0, recv_c0, recv_r0 + recv_h - 1, recv_c0 + recv_w - 1), wall_colors[1], "up", False),
        _make_container("sealed", (sealed_r0, sealed_c0, sealed_r0 + sealed_h - 1, sealed_c0 + sealed_w - 1), wall_colors[2], "up", True),
        _make_container("lefty", (left_r0, left_c0, left_r0 + left_h - 1, left_c0 + left_w - 1), wall_colors[3], "left", False),
    ]

    max_r = max(container.bbox[2] for container in containers)
    max_c = max(container.bbox[3] for container in containers)
    height = max(22, max_r + 1 + int(rng.integers(2, 5)))
    width = max(24, max_c + 1 + int(rng.integers(2, 5)))
    if height > 30 or width > 30:
        return None

    for first in range(len(containers)):
        r0a, c0a, r1a, c1a = containers[first].bbox
        for second in range(first + 1, len(containers)):
            r0b, c0b, r1b, c1b = containers[second].bbox
            overlap = not (r1a < r0b or r1b < r0a or c1a < c0b or c1b < c0a)
            if overlap:
                return None

    color_cycle = itertools.cycle(content_colors)
    pixels: list[ScenePixel] = []
    occupied: set[Coord] = set()

    def add_pixel(token: str, container_name: str, coord: Coord) -> None:
        assert coord not in occupied
        occupied.add(coord)
        pixels.append(
            ScenePixel(
                token=token,
                container_name=container_name,
                coord=coord,
                color=int(next(color_cycle)),
            )
        )

    source_box = containers[0].bbox
    recv_box = containers[1].bbox
    sealed_box = containers[2].bbox
    left_box = containers[3].bbox

    source_rows = list(range(source_box[0] + 1, source_box[2]))
    recv_rows = list(range(recv_box[0] + 1, recv_box[2]))
    sealed_rows = list(range(sealed_box[0] + 1, sealed_box[2]))
    left_rows = list(range(left_box[0] + 1, left_box[2]))

    recv_cols = list(range(recv_box[1] + 1, recv_box[3]))
    sealed_cols = list(range(sealed_box[1] + 1, sealed_box[3]))
    floor_cols = list(range(sealed_box[3] + 1, source_box[3]))

    if not recv_cols or not sealed_cols or not floor_cols:
        return None

    source_stack_col = recv_cols[0]
    source_other_recv_col = recv_cols[-1]
    source_sealed_col = sealed_cols[len(sealed_cols) // 2]
    source_floor_col = floor_cols[-1]

    add_pixel("src_a", "source", (source_rows[0], source_stack_col))
    add_pixel("src_b", "source", (source_rows[1], source_stack_col))
    add_pixel("src_c", "source", (source_rows[0], source_other_recv_col))
    add_pixel("src_d", "source", (source_rows[0], source_sealed_col))
    add_pixel("src_e", "source", (source_rows[1], source_floor_col))

    add_pixel("recv_a", "receiver", (recv_rows[0], source_stack_col))
    add_pixel("recv_b", "receiver", (recv_rows[1], source_other_recv_col))

    add_pixel("seal_a", "sealed", (sealed_rows[0], sealed_cols[0]))
    add_pixel("seal_b", "sealed", (sealed_rows[1], sealed_cols[-1]))

    left_col = left_box[1] + 1
    if left_inner_w > 1 and rng.random() < 0.5:
        left_col = left_box[1] + 2
    add_pixel("left_a", "lefty", (left_rows[0], left_col))
    add_pixel("left_b", "lefty", (left_rows[1], left_col))
    if len(left_rows) >= 3 and left_inner_w >= 2:
        other_left_col = left_box[1] + left_inner_w
        if other_left_col == left_col:
            other_left_col = left_box[1] + 1
        add_pixel("left_c", "lefty", (left_rows[2], other_left_col))

    inp = _render_scene((height, width), containers, pixels)
    expected, traces = _simulate_scene((height, width), containers, pixels)

    if not np.array_equal(main(inp), expected):
        return None

    if np.array_equal(inp, expected):
        return None

    sealed_coords = {pixel.coord for pixel in pixels if pixel.container_name == "sealed"}
    if not sealed_coords:
        return None
    floating_sealed = [
        coord
        for coord in sealed_coords
        if coord[0] + 1 < height and inp[coord[0] + 1, coord[1]] == BG
    ]
    if not floating_sealed:
        return None
    if any(expected[coord] != inp[coord] for coord in sealed_coords):
        return None

    if not any(
        pixel.container_name == "lefty" and traces[pixel.token][0] > pixel.coord[0]
        for pixel in pixels
        if pixel.container_name == "lefty"
    ):
        return None

    receiver_final = [
        traces[pixel.token]
        for pixel in pixels
        if pixel.container_name == "source" and _in_interior(recv_box, traces[pixel.token])
    ]
    if not receiver_final:
        return None

    lid_landings = [
        traces[pixel.token]
        for pixel in pixels
        if pixel.container_name == "source"
        and traces[pixel.token][0] + 1 < height
        and (traces[pixel.token][0] + 1, traces[pixel.token][1]) in set(_opening_cells(sealed_box, "up"))
    ]
    if not lid_landings:
        return None

    if not any(
        pixel.container_name == "source" and traces[pixel.token][0] == height - 1
        for pixel in pixels
        if pixel.container_name == "source"
    ):
        return None

    moving_by_col: dict[int, list[tuple[int, int]]] = {}
    for pixel in pixels:
        if pixel.container_name == "sealed":
            continue
        moving_by_col.setdefault(pixel.coord[1], []).append((pixel.coord[0], traces[pixel.token][0]))
    crowded_columns = [pairs for pairs in moving_by_col.values() if len(pairs) >= 2]
    if not crowded_columns:
        return None
    for pairs in crowded_columns:
        src_rows = [row for row, _ in sorted(pairs)]
        dst_rows = [row for _, row in sorted(pairs)]
        if dst_rows != sorted(dst_rows):
            return None
        if src_rows != sorted(src_rows):
            return None

    alt_oriented, _ = _simulate_scene((height, width), containers, pixels, movable_openings={"down"})
    alt_move_sealed, _ = _simulate_scene((height, width), containers, pixels, move_sealed=True)
    alt_ignore_lids, _ = _simulate_scene((height, width), containers, pixels, lid_blocks=False)
    alt_block_openings, _ = _simulate_scene((height, width), containers, pixels, synthetic_opening_floors=True)

    if np.array_equal(expected, alt_oriented):
        return None
    if np.array_equal(expected, alt_move_sealed):
        return None
    if np.array_equal(expected, alt_ignore_lids):
        return None
    if np.array_equal(expected, alt_block_openings):
        return None

    detected_out = _simulate_detected(inp)
    if not np.array_equal(expected, detected_out):
        return None

    return inp, expected


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        sample = _scene_attempt(rng)
        if sample is None:
            continue
        inp, _ = sample
        assert inp.ndim == 2 and max(inp.shape) <= 30
        return inp
    raise RuntimeError("failed to generate a valid layered-hourglass input")
