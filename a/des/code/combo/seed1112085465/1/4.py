from __future__ import annotations

import random

import numpy as np


AZURE = 8
RED = 2
GREY = 5
MAROON = 9

ALL_TRANSFORMS = (
    "id",
    "rot90",
    "rot180",
    "rot270",
    "flip_lr",
    "flip_ud",
    "transpose",
    "anti_transpose",
)

INVERSE_TRANSFORM = {
    "id": "id",
    "rot90": "rot270",
    "rot180": "rot180",
    "rot270": "rot90",
    "flip_lr": "flip_lr",
    "flip_ud": "flip_ud",
    "transpose": "transpose",
    "anti_transpose": "anti_transpose",
}

GEN_RETRIES = 300
SHAPE_RETRIES = 200
PANEL_SIZE_RANGE = (5, 9)
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _valid_transforms(shape: tuple[int, int]) -> tuple[str, ...]:
    height, width = shape
    if height == width:
        return ALL_TRANSFORMS
    return ("id", "rot180", "flip_lr", "flip_ud")


def _transform_coord(
    coord: tuple[int, int], shape: tuple[int, int], name: str
) -> tuple[int, int]:
    row, col = map(int, coord)
    height, width = map(int, shape)
    if name == "id":
        return row, col
    if name == "rot180":
        return height - 1 - row, width - 1 - col
    if name == "flip_lr":
        return row, width - 1 - col
    if name == "flip_ud":
        return height - 1 - row, col

    if height != width:
        raise ValueError(f"transform {name!r} requires a square panel")

    side = height
    if name == "rot90":
        return side - 1 - col, row
    if name == "rot270":
        return col, side - 1 - row
    if name == "transpose":
        return col, row
    if name == "anti_transpose":
        return side - 1 - col, side - 1 - row
    raise ValueError(f"unknown transform {name!r}")


def _apply_transform(arr: np.ndarray, name: str) -> np.ndarray:
    out = np.empty_like(arr)
    height, width = map(int, arr.shape)
    for row in range(height):
        for col in range(width):
            new_row, new_col = _transform_coord((row, col), arr.shape, name)
            out[new_row, new_col] = arr[row, col]
    return out


def _mask_signature(mask: np.ndarray) -> bytes:
    return mask.astype(np.uint8).tobytes()


def _canonical_mask(mask: np.ndarray) -> np.ndarray:
    best = None
    best_sig = None
    for name in _valid_transforms(mask.shape):
        candidate = _apply_transform(mask, name)
        signature = _mask_signature(candidate)
        if best is None or signature < best_sig:
            best = candidate
            best_sig = signature
    assert best is not None
    return best


def _split_panels(grid: np.ndarray) -> list[np.ndarray]:
    sep_rows = np.where(np.all(grid == MAROON, axis=1))[0]
    sep_cols = np.where(np.all(grid == MAROON, axis=0))[0]
    assert len(sep_rows) == 1 and len(sep_cols) == 1

    sep_row = int(sep_rows[0])
    sep_col = int(sep_cols[0])
    height, width = map(int, grid.shape)

    assert sep_row * 2 + 1 == height
    assert sep_col * 2 + 1 == width

    panels = [
        grid[:sep_row, :sep_col],
        grid[:sep_row, sep_col + 1 :],
        grid[sep_row + 1 :, :sep_col],
        grid[sep_row + 1 :, sep_col + 1 :],
    ]
    panel_shape = panels[0].shape
    assert all(panel.shape == panel_shape for panel in panels)
    return panels


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the shared panel pattern up to a deterministic canonical orientation,
    then undo the marked panel's dihedral transform to map the red pixel back.
    """
    g = np.asarray(grid, dtype=int)
    panels = _split_panels(g)

    masks: list[np.ndarray] = []
    marked_panels: list[tuple[int, tuple[int, int]]] = []
    for idx, panel in enumerate(panels):
        red_pixels = np.argwhere(panel == RED)
        assert len(red_pixels) <= 1
        if len(red_pixels) == 1:
            marked_panels.append((idx, tuple(map(int, red_pixels[0]))))

        mask = (panel == GREY) | (panel == RED)
        assert bool(mask.any())
        masks.append(mask)

    assert len(marked_panels) == 1
    master_mask = _canonical_mask(masks[0])
    assert all(np.array_equal(_canonical_mask(mask), master_mask) for mask in masks[1:])

    valid_names = _valid_transforms(master_mask.shape)
    for mask in masks:
        assert any(np.array_equal(_apply_transform(master_mask, name), mask) for name in valid_names)

    marked_idx, red_pos = marked_panels[0]
    matched_transforms = [
        name
        for name in valid_names
        if np.array_equal(_apply_transform(master_mask, name), masks[marked_idx])
    ]
    assert matched_transforms

    mapped_positions = {
        _transform_coord(red_pos, master_mask.shape, INVERSE_TRANSFORM[name])
        for name in matched_transforms
    }
    assert len(mapped_positions) == 1
    mark_row, mark_col = mapped_positions.pop()
    assert bool(master_mask[mark_row, mark_col])

    out = np.full(master_mask.shape, AZURE, dtype=int)
    out[master_mask] = GREY
    out[mark_row, mark_col] = RED
    return out


def _random_connected_mask(side: int) -> np.ndarray | None:
    min_cells = max(5, side - 1)
    max_cells = min((side * side) // 2, side + 6)

    for _ in range(SHAPE_RETRIES):
        target_cells = random.randint(min_cells, max_cells)
        start = (random.randrange(side), random.randrange(side))
        cells = {start}
        cell_list = [start]
        failed_steps = 0

        while len(cells) < target_cells and failed_steps < target_cells * 20:
            base_row, base_col = random.choice(cell_list)
            d_row, d_col = random.choice(DIR4)
            new_row = base_row + d_row
            new_col = base_col + d_col
            if not (0 <= new_row < side and 0 <= new_col < side):
                failed_steps += 1
                continue
            if (new_row, new_col) in cells:
                failed_steps += 1
                continue
            cells.add((new_row, new_col))
            cell_list.append((new_row, new_col))

        if len(cells) != target_cells:
            continue

        mask = np.zeros((side, side), dtype=bool)
        for row, col in cells:
            mask[row, col] = True

        points = np.argwhere(mask)
        row_span = int(points[:, 0].max() - points[:, 0].min())
        col_span = int(points[:, 1].max() - points[:, 1].min())
        if row_span < 2 or col_span < 2:
            continue

        orbit = {
            _mask_signature(_apply_transform(mask, name))
            for name in ALL_TRANSFORMS
        }
        if len(orbit) != 8:
            continue

        return _canonical_mask(mask)

    return None


def _render_panel(mask: np.ndarray) -> np.ndarray:
    panel = np.full(mask.shape, AZURE, dtype=int)
    panel[mask] = GREY
    return panel


def _build_example() -> tuple[np.ndarray, np.ndarray]:
    master_mask = None
    side = PANEL_SIZE_RANGE[0]
    for _ in range(SHAPE_RETRIES):
        side = random.randint(*PANEL_SIZE_RANGE)
        master_mask = _random_connected_mask(side)
        if master_mask is not None:
            break
    if master_mask is None:
        raise RuntimeError("failed to sample an asymmetric master pattern")

    transform_names = random.sample(list(ALL_TRANSFORMS), 4)
    marked_panel_idx = random.randrange(4)
    mark_candidates = [tuple(map(int, coord)) for coord in np.argwhere(master_mask)]
    master_mark = random.choice(mark_candidates)

    panels = []
    for idx, name in enumerate(transform_names):
        panel_mask = _apply_transform(master_mask, name)
        panel = _render_panel(panel_mask)
        if idx == marked_panel_idx:
            mark_row, mark_col = _transform_coord(master_mark, panel.shape, name)
            panel[mark_row, mark_col] = RED
        panels.append(panel)

    grid = np.full((2 * side + 1, 2 * side + 1), AZURE, dtype=int)
    grid[side, :] = MAROON
    grid[:, side] = MAROON

    grid[:side, :side] = panels[0]
    grid[:side, side + 1 :] = panels[1]
    grid[side + 1 :, :side] = panels[2]
    grid[side + 1 :, side + 1 :] = panels[3]

    expected = _render_panel(master_mask)
    expected[master_mark] = RED
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid, expected = _build_example()
        try:
            solved = main(grid)
        except Exception:
            continue
        if np.array_equal(solved, expected):
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
