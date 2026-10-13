import random

import numpy as np


BG = 0
COLOR_POOL = tuple(range(1, 10))
GEN_ATTEMPTS = 300
GEN_MAX_OUTER_HEIGHT = 20
GEN_CANVAS_SLACK = 8


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    (r0, c0), (r1, c1) = coords.min(axis=0), coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _perimeter_values(arr: np.ndarray) -> np.ndarray:
    h, w = map(int, arr.shape)
    if h == 1:
        return arr[0, :].reshape(-1)
    if w == 1:
        return arr[:, 0].reshape(-1)
    parts = [arr[0, :], arr[-1, :], arr[1:-1, 0], arr[1:-1, -1]]
    return np.concatenate([part.reshape(-1) for part in parts])


def _draw_border(canvas: np.ndarray, r0: int, c0: int, h: int, w: int, color: int) -> None:
    canvas[r0, c0 : c0 + w] = int(color)
    canvas[r0 + h - 1, c0 : c0 + w] = int(color)
    canvas[r0 : r0 + h, c0] = int(color)
    canvas[r0 : r0 + h, c0 + w - 1] = int(color)


def _make_border_sprite(h: int, w: int, color: int) -> np.ndarray:
    assert h >= 3 and w >= 3, "border sprites must be at least 3x3"
    sprite = np.zeros((h, w), dtype=int)
    _draw_border(sprite, 0, 0, h, w, color)
    return sprite


def _extract_elements(grid: np.ndarray) -> list[np.ndarray]:
    current = np.asarray(grid, dtype=int)
    elements: list[np.ndarray] = []

    while True:
        box = _bbox(current != BG)
        if box is None:
            break

        r0, c0, r1, c1 = box
        cropped = current[r0 : r1 + 1, c0 : c1 + 1]
        edge_colors = set(map(int, _perimeter_values(cropped)))
        assert edge_colors and BG not in edge_colors and len(edge_colors) == 1, "expected one monochrome outer layer"
        color = edge_colors.pop()

        if bool(np.all(cropped == color)):
            elements.append(np.full(cropped.shape, int(color), dtype=int))
            break

        h, w = map(int, cropped.shape)
        assert h >= 3 and w >= 3, "frame layers must be at least 3x3"
        assert np.all(cropped[0, :] == color) and np.all(cropped[-1, :] == color), "top and bottom edges must be solid"
        assert np.all(cropped[:, 0] == color) and np.all(cropped[:, -1] == color), "left and right edges must be solid"
        elements.append(_make_border_sprite(h, w, color))
        current = cropped[1:-1, 1:-1]

    assert elements, "expected at least one nested element"
    return elements


def _layout_elements(elements: list[np.ndarray]) -> np.ndarray:
    out_h = int(elements[0].shape[0])
    out_w = sum(int(element.shape[1]) for element in elements) + len(elements) - 1
    out = np.zeros((out_h, out_w), dtype=int)

    col = 0
    for element in elements:
        h, w = map(int, element.shape)
        assert out_h >= h and (out_h - h) % 2 == 0, "nested rectangles must share a common center"
        row = (out_h - h) // 2
        out[row : row + h, col : col + w] = element
        col += w + 1

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand concentric nested rectangles into a left-to-right sequence of standalone rectangles.

    Input structure:
    - The background is black.
    - One nested structure is present: several concentric one-pixel rectangular frames,
      with exactly one black-cell gap between successive frame layers.
    - The frame colors differ by layer, and the innermost layer may instead be a solid rectangle.

    Rule:
    - Read the nested elements from the outside inward, preserving each element's color and size.
    - Build an output whose height equals the outermost frame height and whose width is the
      sum of the extracted element widths plus one black separator column between neighbors.
    - Place the extracted elements from left to right in outer-to-inner order, vertically centered.
    - If the innermost element is solid rather than hollow, place that solid rectangle as the final item.

    Key insight:
    - The concentric nesting is a compressed ordered list: depth determines left-to-right position,
      while each layer's geometry stays unchanged after the structure is unfolded.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    elements = _extract_elements(g)
    return _layout_elements(elements)


def _spec_to_sprite(kind: str, h: int, w: int, color: int) -> np.ndarray:
    if kind == "solid":
        return np.full((h, w), int(color), dtype=int)
    assert kind == "border"
    return _make_border_sprite(h, w, color)


def _build_specs() -> list[tuple[str, int, int, int]]:
    has_solid = random.random() < 0.5
    kinds = ["border", "border", "solid"] if has_solid else ["border"] * random.choice([2, 3])
    item_count = len(kinds)

    min_last = 1 if kinds[-1] == "solid" else 3
    min_outer_h = 4 * (item_count - 1) + min_last
    min_outer_w = 4 * (item_count - 1) + min_last

    width_options = []
    for outer_w in range(min_outer_w, 31):
        widths = [outer_w - 4 * depth for depth in range(item_count)]
        if sum(widths) + item_count - 1 <= 30:
            width_options.append(outer_w)
    assert width_options, "generator requires at least one feasible outer width"

    outer_h = random.randint(min_outer_h, GEN_MAX_OUTER_HEIGHT)
    outer_w = random.choice(width_options)
    colors = random.sample(COLOR_POOL, item_count)

    return [
        (kind, outer_h - 4 * depth, outer_w - 4 * depth, color)
        for depth, (kind, color) in enumerate(zip(kinds, colors))
    ]


def _render_nested_input(specs: list[tuple[str, int, int, int]]) -> np.ndarray:
    outer_h = specs[0][1]
    outer_w = specs[0][2]
    structure = np.zeros((outer_h, outer_w), dtype=int)

    for kind, h, w, color in specs:
        r0 = (outer_h - h) // 2
        c0 = (outer_w - w) // 2
        if kind == "solid":
            structure[r0 : r0 + h, c0 : c0 + w] = int(color)
        else:
            _draw_border(structure, r0, c0, h, w, color)

    canvas_h = random.randint(outer_h, min(30, outer_h + GEN_CANVAS_SLACK))
    canvas_w = random.randint(outer_w, min(30, outer_w + GEN_CANVAS_SLACK))
    row = random.randint(0, canvas_h - outer_h)
    col = random.randint(0, canvas_w - outer_w)

    grid = np.zeros((canvas_h, canvas_w), dtype=int)
    grid[row : row + outer_h, col : col + outer_w] = structure
    return grid


def _expected_from_specs(specs: list[tuple[str, int, int, int]]) -> np.ndarray:
    sprites = [_spec_to_sprite(kind, h, w, color) for kind, h, w, color in specs]
    return _layout_elements(sprites)


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        specs = _build_specs()
        inp = _render_nested_input(specs)
        expected = _expected_from_specs(specs)
        try:
            actual = main(inp)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
