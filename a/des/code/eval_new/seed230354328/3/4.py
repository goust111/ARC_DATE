from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from numpy.random import default_rng


BLACK = 0
GREY = 5
MARKER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
OPEN_SIDES = ("top", "bottom", "left", "right")

GEN_RETRIES = 400
PLACEMENT_RETRIES = 240
MASK_RETRIES = 120


@dataclass(frozen=True)
class Container:
    top: int
    left: int
    height: int
    width: int
    open_side: str
    marker_color: int
    marker_local: tuple[int, int]
    coords: np.ndarray


@dataclass(frozen=True)
class Stamp:
    color: int
    top: int
    left: int
    mask: np.ndarray
    coords: np.ndarray


def _neighbors(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _wall_mask(height: int, width: int, open_side: str) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    if open_side == "top":
        mask[:, 0] = True
        mask[:, -1] = True
        mask[-1, :] = True
    elif open_side == "bottom":
        mask[:, 0] = True
        mask[:, -1] = True
        mask[0, :] = True
    elif open_side == "left":
        mask[0, :] = True
        mask[-1, :] = True
        mask[:, -1] = True
    elif open_side == "right":
        mask[0, :] = True
        mask[-1, :] = True
        mask[:, 0] = True
    else:
        raise ValueError(open_side)
    return mask


def _inner_wall_mask(height: int, width: int, open_side: str) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    if open_side == "top":
        mask[1:-1, 0] = True
        mask[1:-1, -1] = True
        mask[-1, 1:-1] = True
    elif open_side == "bottom":
        mask[0, 1:-1] = True
        mask[1:-1, 0] = True
        mask[1:-1, -1] = True
    elif open_side == "left":
        mask[0, 1:-1] = True
        mask[-1, 1:-1] = True
        mask[1:-1, -1] = True
    elif open_side == "right":
        mask[0, 1:-1] = True
        mask[-1, 1:-1] = True
        mask[1:-1, 0] = True
    else:
        raise ValueError(open_side)
    return mask


def _component_mask(coords: np.ndarray, top: int, left: int, height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[coords[:, 0] - top, coords[:, 1] - left] = True
    return mask


def _find_components(
    grid: np.ndarray,
    *,
    monochromatic: bool,
    ignore: np.ndarray | None = None,
) -> list[np.ndarray]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    if ignore is None:
        ignore = np.zeros((h, w), dtype=bool)

    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if visited[r, c] or ignore[r, c] or int(grid[r, c]) == BLACK:
                continue

            color = int(grid[r, c])
            q = deque([(r, c)])
            visited[r, c] = True
            cells: list[tuple[int, int]] = []

            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors(rr, cc, h, w):
                    if visited[nr, nc] or ignore[nr, nc] or int(grid[nr, nc]) == BLACK:
                        continue
                    if monochromatic and int(grid[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    q.append((nr, nc))

            components.append(np.array(cells, dtype=int))
    return components


def _detect_container(grid: np.ndarray, coords: np.ndarray) -> Container | None:
    top, left, bottom, right = _bbox(coords)
    height = bottom - top + 1
    width = right - left + 1
    if height < 3 or width < 3:
        return None

    patch = grid[top : bottom + 1, left : right + 1]
    occupied = patch != BLACK
    if not np.any(patch == GREY):
        return None

    matches: list[Container] = []
    for open_side in OPEN_SIDES:
        wall = _wall_mask(height, width, open_side)
        if not np.array_equal(occupied, wall):
            continue

        marker_cells = np.argwhere((patch != BLACK) & (patch != GREY))
        if marker_cells.shape[0] != 1:
            continue
        marker_r, marker_c = map(int, marker_cells[0])
        if not _inner_wall_mask(height, width, open_side)[marker_r, marker_c]:
            continue
        if np.any(patch[wall & ~((np.arange(height)[:, None] == marker_r) & (np.arange(width)[None, :] == marker_c))] != GREY):
            continue

        matches.append(
            Container(
                top=top,
                left=left,
                height=height,
                width=width,
                open_side=open_side,
                marker_color=int(patch[marker_r, marker_c]),
                marker_local=(marker_r, marker_c),
                coords=coords,
            )
        )

    if len(matches) != 1:
        return None
    return matches[0]


def _stamp_from_coords(grid: np.ndarray, coords: np.ndarray) -> Stamp:
    color = int(grid[tuple(coords[0])])
    assert color not in (BLACK, GREY), "stamps must use non-black, non-grey colors"
    assert np.all(grid[coords[:, 0], coords[:, 1]] == color), "stamp must be monochrome"
    top, left, bottom, right = _bbox(coords)
    height = bottom - top + 1
    width = right - left + 1
    mask = _component_mask(coords, top, left, height, width)
    return Stamp(color=color, top=top, left=left, mask=mask, coords=coords)


def _target_top_left(container: Container, stamp_mask: np.ndarray) -> tuple[int, int]:
    stamp_h, stamp_w = stamp_mask.shape
    if container.open_side == "top":
        return container.top + container.height, container.left + (container.width - stamp_w) // 2
    if container.open_side == "bottom":
        return container.top - stamp_h, container.left + (container.width - stamp_w) // 2
    if container.open_side == "left":
        return container.top + (container.height - stamp_h) // 2, container.left + container.width
    if container.open_side == "right":
        return container.top + (container.height - stamp_h) // 2, container.left - stamp_w
    raise ValueError(container.open_side)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each marked grey container to its same-color stamp and press the stamp onto the exterior opposite the opening.

    Input structure:
    - The background is black.
    - Grey pixels form single-cell-thick three-sided containers with one open side; each such container has exactly one colored marker embedded in an inner wall cell.
    - Separate from the containers are compact connected monochrome stamp shapes whose colors match the container markers.

    Rule:
    - Identify every container, determine which side is open, and pair it with the stamp of the same color as its wall marker.
    - Remove the original stamp from its source position.
    - Repaint the matched stamp grey and place it immediately outside the side opposite the opening, centered along that closed side.

    Key insight:
    - The colored wall pixel is a label for matching, but the placement is controlled by geometry: the stamp moves to the exterior of the closed side opposite the opening.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert np.all((0 <= g) & (g <= 9)), "expected ARC color ids"

    nonblack_components = _find_components(g, monochromatic=False)
    containers: list[Container] = []
    container_mask = np.zeros_like(g, dtype=bool)
    for coords in nonblack_components:
        container = _detect_container(g, coords)
        if container is None:
            continue
        containers.append(container)
        container_mask[coords[:, 0], coords[:, 1]] = True

    assert containers, "expected at least one container"
    by_marker: dict[int, Container] = {}
    for container in containers:
        assert container.marker_color not in by_marker, "marker colors must identify containers uniquely"
        by_marker[container.marker_color] = container

    stamp_components = _find_components(g, monochromatic=True, ignore=container_mask)
    assert stamp_components, "expected at least one stamp"
    by_color: dict[int, Stamp] = {}
    for coords in stamp_components:
        stamp = _stamp_from_coords(g, coords)
        assert stamp.color not in by_color, "stamp colors must identify stamps uniquely"
        by_color[stamp.color] = stamp

    assert set(by_color) == set(by_marker), "container markers and stamp colors must match one-to-one"

    out = g.copy()
    for stamp in by_color.values():
        out[stamp.coords[:, 0], stamp.coords[:, 1]] = BLACK

    for color, container in by_marker.items():
        stamp = by_color[color]
        top, left = _target_top_left(container, stamp.mask)
        local = np.argwhere(stamp.mask)
        rows = top + local[:, 0]
        cols = left + local[:, 1]
        assert rows.min() >= 0 and cols.min() >= 0
        assert rows.max() < g.shape[0] and cols.max() < g.shape[1]
        assert np.all(out[rows, cols] == BLACK), "imprint area must be empty"
        out[rows, cols] = GREY

    return out


def _random_stamp_mask(rng: np.random.Generator) -> np.ndarray:
    for _ in range(MASK_RETRIES):
        height = int(rng.integers(2, 5))
        width = int(rng.integers(2, 5))
        max_area = min(height * width, 7)
        min_area = min(max_area, max(3, (height * width) // 2))
        target_area = int(rng.integers(min_area, max_area + 1))

        mask = np.zeros((height, width), dtype=bool)
        seed = (int(rng.integers(0, height)), int(rng.integers(0, width)))
        mask[seed] = True

        while int(mask.sum()) < target_area:
            frontier: list[tuple[int, int]] = []
            seen = set()
            for r, c in zip(*np.where(mask)):
                for nr, nc in _neighbors(int(r), int(c), height, width):
                    if mask[nr, nc] or (nr, nc) in seen:
                        continue
                    seen.add((nr, nc))
                    frontier.append((nr, nc))
            if not frontier:
                break
            nr, nc = frontier[int(rng.integers(0, len(frontier)))]
            mask[nr, nc] = True

        coords = np.argwhere(mask)
        top, left, bottom, right = _bbox(coords)
        cropped = mask[top : bottom + 1, left : right + 1]
        if cropped.shape[0] < 2 or cropped.shape[1] < 2:
            continue
        if int(cropped.sum()) < 3:
            continue
        return cropped
    raise RuntimeError("failed to build a stamp mask")


def _inner_wall_positions(height: int, width: int, open_side: str) -> list[tuple[int, int]]:
    return [tuple(map(int, cell)) for cell in np.argwhere(_inner_wall_mask(height, width, open_side))]


def _container_dims(
    rng: np.random.Generator,
    stamp_mask: np.ndarray,
    open_side: str,
) -> tuple[int, int]:
    stamp_h, stamp_w = stamp_mask.shape
    if open_side in ("top", "bottom"):
        width_options = [w for w in range(max(4, stamp_w), 9) if (w - stamp_w) % 2 == 0]
        return int(rng.integers(4, 8)), int(rng.choice(width_options))
    height_options = [h for h in range(max(4, stamp_h), 9) if (h - stamp_h) % 2 == 0]
    return int(rng.choice(height_options)), int(rng.integers(4, 8))


def _bounds_for_container(
    grid_h: int,
    grid_w: int,
    container_h: int,
    container_w: int,
    stamp_h: int,
    stamp_w: int,
    open_side: str,
) -> tuple[int, int, int, int]:
    row_lo, row_hi = 1, grid_h - container_h - 2
    col_lo, col_hi = 1, grid_w - container_w - 2

    if open_side == "top":
        row_hi = grid_h - container_h - stamp_h - 2
    elif open_side == "bottom":
        row_lo = stamp_h + 1
    elif open_side == "left":
        col_hi = grid_w - container_w - stamp_w - 2
    elif open_side == "right":
        col_lo = stamp_w + 1
    else:
        raise ValueError(open_side)

    return row_lo, row_hi, col_lo, col_hi


def _expanded_rect(
    top: int,
    left: int,
    bottom: int,
    right: int,
    gap: int,
    grid_h: int,
    grid_w: int,
) -> tuple[int, int, int, int]:
    return (
        max(0, top - gap),
        max(0, left - gap),
        min(grid_h - 1, bottom + gap),
        min(grid_w - 1, right + gap),
    )


def _rect_is_free(occupied: np.ndarray, rect: tuple[int, int, int, int]) -> bool:
    top, left, bottom, right = rect
    return not bool(occupied[top : bottom + 1, left : right + 1].any())


def _mark_rect(occupied: np.ndarray, rect: tuple[int, int, int, int]) -> None:
    top, left, bottom, right = rect
    occupied[top : bottom + 1, left : right + 1] = True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    pair_count = int(rng.integers(1, 4))
    min_side = 16 + 2 * pair_count
    grid_h = int(rng.integers(min_side, 31))
    grid_w = int(rng.integers(min_side, 31))

    colors = [int(c) for c in rng.choice(np.array(MARKER_COLORS), size=pair_count, replace=False)]
    directions = list(OPEN_SIDES)
    rng.shuffle(directions)
    directions = directions[:pair_count]

    specs = []
    for color, open_side in zip(colors, directions):
        stamp_mask = _random_stamp_mask(rng)
        container_h, container_w = _container_dims(rng, stamp_mask, open_side)
        marker_choices = _inner_wall_positions(container_h, container_w, open_side)
        marker_local = marker_choices[int(rng.integers(0, len(marker_choices)))]
        specs.append(
            {
                "color": color,
                "open_side": open_side,
                "stamp_mask": stamp_mask,
                "container_h": container_h,
                "container_w": container_w,
                "marker_local": marker_local,
            }
        )

    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    for spec in specs:
        stamp_h, stamp_w = spec["stamp_mask"].shape
        row_lo, row_hi, col_lo, col_hi = _bounds_for_container(
            grid_h,
            grid_w,
            spec["container_h"],
            spec["container_w"],
            stamp_h,
            stamp_w,
            spec["open_side"],
        )
        if row_lo > row_hi or col_lo > col_hi:
            return None

        placed = False
        for _ in range(PLACEMENT_RETRIES):
            top = int(rng.integers(row_lo, row_hi + 1))
            left = int(rng.integers(col_lo, col_hi + 1))
            container = Container(
                top=top,
                left=left,
                height=spec["container_h"],
                width=spec["container_w"],
                open_side=spec["open_side"],
                marker_color=spec["color"],
                marker_local=spec["marker_local"],
                coords=np.empty((0, 2), dtype=int),
            )
            imprint_top, imprint_left = _target_top_left(container, spec["stamp_mask"])
            imprint_bottom = imprint_top + stamp_h - 1
            imprint_right = imprint_left + stamp_w - 1
            union_rect = (
                min(top, imprint_top),
                min(left, imprint_left),
                max(top + spec["container_h"] - 1, imprint_bottom),
                max(left + spec["container_w"] - 1, imprint_right),
            )
            reserve = _expanded_rect(*union_rect, gap=1, grid_h=grid_h, grid_w=grid_w)
            if not _rect_is_free(occupied, reserve):
                continue
            _mark_rect(occupied, reserve)
            spec["container_top"] = top
            spec["container_left"] = left
            placed = True
            break
        if not placed:
            return None

    for spec in specs:
        stamp_h, stamp_w = spec["stamp_mask"].shape
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            top = int(rng.integers(1, grid_h - stamp_h))
            left = int(rng.integers(1, grid_w - stamp_w))
            reserve = _expanded_rect(top, left, top + stamp_h - 1, left + stamp_w - 1, gap=1, grid_h=grid_h, grid_w=grid_w)
            if not _rect_is_free(occupied, reserve):
                continue
            _mark_rect(occupied, reserve)
            spec["stamp_top"] = top
            spec["stamp_left"] = left
            placed = True
            break
        if not placed:
            return None

    grid = np.zeros((grid_h, grid_w), dtype=int)
    for spec in specs:
        top = spec["container_top"]
        left = spec["container_left"]
        height = spec["container_h"]
        width = spec["container_w"]
        wall = _wall_mask(height, width, spec["open_side"])
        patch = grid[top : top + height, left : left + width]
        patch[wall] = GREY
        mr, mc = spec["marker_local"]
        patch[mr, mc] = spec["color"]

        stamp_top = spec["stamp_top"]
        stamp_left = spec["stamp_left"]
        stamp_patch = grid[stamp_top : stamp_top + spec["stamp_mask"].shape[0], stamp_left : stamp_left + spec["stamp_mask"].shape[1]]
        stamp_patch[spec["stamp_mask"]] = spec["color"]

    try:
        out = main(grid)
    except AssertionError:
        return None

    assert out.shape == grid.shape
    assert max(grid.shape) <= 30 and max(out.shape) <= 30
    return grid


def generate_input() -> np.ndarray:
    rng = default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
