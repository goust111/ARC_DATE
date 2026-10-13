from __future__ import annotations

import random

import numpy as np


GREY = 5
ARC_COLORS = tuple(range(10))
GEN_ATTEMPTS = 300
CORE_SHAPES = (
    np.array([[1]], dtype=bool),
    np.array([[1, 1]], dtype=bool),
    np.array([[1], [1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[0, 1], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[1], [1], [1]], dtype=bool),
    np.array([[1, 1], [1, 0], [1, 0]], dtype=bool),
    np.array([[1, 1], [0, 1], [0, 1]], dtype=bool),
    np.array([[1, 0], [1, 1], [0, 1]], dtype=bool),
    np.array([[0, 1], [1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [1, 1], [1, 1]], dtype=bool),
)
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _components_for_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    mask = grid == int(color)
    seen = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape
    comps: list[np.ndarray] = []

    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for dr, dc in NEIGHBORS4:
                    nr = int(cr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _component_mask(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([r0, c0], dtype=int)
    mask[rel[:, 0], rel[:, 1]] = True
    return mask


def _is_rect_frame(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    if height < 3 or width < 3:
        return False

    mask = _component_mask(coords, (r0, c0, r1, c1))
    expected = np.zeros((height, width), dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    return bool(np.array_equal(mask, expected))


def _contains_with_gap(
    outer_bbox: tuple[int, int, int, int],
    inner_bbox: tuple[int, int, int, int],
    gap: int = 2,
) -> bool:
    or0, oc0, or1, oc1 = outer_bbox
    ir0, ic0, ir1, ic1 = inner_bbox
    return (
        int(or0 + gap) <= int(ir0)
        and int(oc0 + gap) <= int(ic0)
        and int(or1 - gap) >= int(ir1)
        and int(oc1 - gap) >= int(ic1)
    )


def _isolated_patch(
    coords: np.ndarray,
    color: int,
    bbox: tuple[int, int, int, int],
    bg: int,
) -> np.ndarray:
    patch = np.full((bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1), int(bg), dtype=int)
    rel = coords - np.array([bbox[0], bbox[1]], dtype=int)
    patch[rel[:, 0], rel[:, 1]] = int(color)
    return patch


def _center_paste(out: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> None:
    height, width = patch.shape
    tile = out[:, left : left + width]
    mask = patch != int(bg)
    tile[:, :] = tile
    tile[0:height, 0:width][mask] = patch[mask]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split a concentric nested frame stack into separate depth-ordered panels.

    Input structure:
    - The grid has a dominant background color.
    - Near the center sits one nested stack of axis-aligned one-pixel-thick rectangular frames, with a background gap between adjacent frame layers.
    - Inside the innermost frame there is one small solid core object whose color differs from every frame color.

    Rule:
    - Identify the core object and every rectangular frame layer, then order them by nesting depth from inner to outer.
    - Create an output grid with the same height as the input and width equal to the input width multiplied by the total number of layers (core included).
    - Split the output into equal-width panels from left to right, mark the panel boundaries with one-pixel grey vertical separators, and place the core in the leftmost panel followed by the frame layers from inner to outer.
    - Each isolated layer keeps its original shape and color and is centered within its panel on the original background color.

    Key insight:
    - The nested structure must be decomposed by containment depth: the output is not a copy of the whole stack, but a left-to-right peeling of core, then inner frame, then outer frames.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    colors = [int(c) for c in np.unique(g) if int(c) != int(bg)]
    assert colors, "expected at least one non-background layer"

    components: list[dict[str, object]] = []
    for color in colors:
        for coords in _components_for_color(g, int(color)):
            b = _bbox(coords)
            components.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": b,
                    "is_frame": _is_rect_frame(coords),
                }
            )

    frames = [comp for comp in components if bool(comp["is_frame"])]
    cores = [comp for comp in components if not bool(comp["is_frame"])]
    assert len(cores) == 1, "expected exactly one core object"
    assert frames, "expected at least one frame layer"

    frames.sort(
        key=lambda comp: (
            int(comp["bbox"][2] - comp["bbox"][0] + 1),
            int(comp["bbox"][3] - comp["bbox"][1] + 1),
            int(comp["bbox"][0]),
            int(comp["bbox"][1]),
        )
    )

    layers = [cores[0]] + frames
    for inner, outer in zip(layers, layers[1:]):
        assert _contains_with_gap(outer["bbox"], inner["bbox"]), "layers must nest with a background moat"

    height, width = g.shape
    panel_count = len(layers)
    out = np.full((height, width * panel_count), int(bg), dtype=int)

    for idx, layer in enumerate(layers):
        bbox = layer["bbox"]
        patch = _isolated_patch(layer["coords"], int(layer["color"]), bbox, int(bg))
        ph, pw = patch.shape
        panel_left = idx * width
        row = (height - ph) // 2
        col = panel_left + (width - pw) // 2
        mask = patch != int(bg)
        out[row : row + ph, col : col + pw][mask] = patch[mask]

    for idx in range(panel_count - 1):
        out[:, (idx + 1) * width - 1] = GREY

    return out


def _random_shape() -> np.ndarray:
    shape = np.array(random.choice(CORE_SHAPES), dtype=bool)
    if random.random() < 0.5:
        shape = np.flip(shape, axis=0)
    if random.random() < 0.5:
        shape = np.flip(shape, axis=1)
    return shape.copy()


def _centered_split(extra: int) -> tuple[int, int]:
    base = extra // 2
    choices = {base, extra - base}
    if base > 0:
        choices.add(base - 1)
    if base + 1 <= extra:
        choices.add(base + 1)
    top = random.choice(sorted(c for c in choices if 0 <= c <= extra))
    return int(top), int(extra - top)


def _draw_frame(grid: np.ndarray, top: int, left: int, height: int, width: int, color: int) -> None:
    grid[top, left : left + width] = int(color)
    grid[top + height - 1, left : left + width] = int(color)
    grid[top : top + height, left] = int(color)
    grid[top : top + height, left + width - 1] = int(color)


def _generate_attempt() -> np.ndarray | None:
    usable_colors = [c for c in ARC_COLORS if c != GREY]
    bg, core_color, inner_color, outer_color = random.sample(usable_colors, 4)
    core = _random_shape()
    ch, cw = map(int, core.shape)

    inner_h, inner_w = ch + 4, cw + 4
    outer_h, outer_w = ch + 8, cw + 8
    if outer_w > 10:
        return None

    grid_w = outer_w + random.randint(0, min(1, 10 - outer_w))
    extra_h_max = min(10, 30 - outer_h)
    if extra_h_max < 1:
        return None
    grid_h = outer_h + random.randint(1, extra_h_max)

    extra_h = grid_h - outer_h
    extra_w = grid_w - outer_w
    top_margin, _bottom_margin = _centered_split(extra_h)
    left_margin, _right_margin = _centered_split(extra_w)

    grid = np.full((grid_h, grid_w), int(bg), dtype=int)

    outer_top = top_margin
    outer_left = left_margin
    inner_top = outer_top + 2
    inner_left = outer_left + 2
    core_top = inner_top + 2
    core_left = inner_left + 2

    _draw_frame(grid, outer_top, outer_left, outer_h, outer_w, int(outer_color))
    _draw_frame(grid, inner_top, inner_left, inner_h, inner_w, int(inner_color))
    grid[core_top : core_top + ch, core_left : core_left + cw][core] = int(core_color)

    if int(np.sum(grid == int(bg))) <= max(
        int(np.sum(grid == int(core_color))),
        int(np.sum(grid == int(inner_color))),
        int(np.sum(grid == int(outer_color))),
    ):
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if out.shape[0] > 30 or out.shape[1] > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-frame input")
