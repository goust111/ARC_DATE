import random
from collections import deque

import numpy as np


BG = 0
TEMPLATE_COLOR = 5
MARKER_COLORS = (1, 2, 3, 4)
GRID_SIZES = (13, 15, 17, 19)
GEN_ATTEMPTS = 400

L_TO_QUADRANT = {
    frozenset(((0, 0), (0, 1), (1, 0))): "top_left",
    frozenset(((0, 0), (0, 1), (1, 1))): "top_right",
    frozenset(((0, 0), (1, 0), (1, 1))): "bottom_left",
    frozenset(((0, 1), (1, 0), (1, 1))): "bottom_right",
}

QUADRANT_TO_TRANSFORM = {
    "top_left": lambda x: x.copy(),
    "top_right": np.fliplr,
    "bottom_left": np.flipud,
    "bottom_right": lambda x: np.flipud(np.fliplr(x)),
}

QUADRANT_ORDER = ("top_left", "top_right", "bottom_left", "bottom_right")


def _bbox_from_coords(coords):
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(coords)
    return components


def _extract_template(grid):
    coords = list(map(tuple, np.argwhere(grid == TEMPLATE_COLOR)))
    assert coords, "expected a central grey template"
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    template = grid[r0 : r1 + 1, c0 : c1 + 1] == TEMPLATE_COLOR
    return template, (r0, c0, r1, c1)


def _marker_quadrant(coords):
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    assert (r1 - r0 + 1, c1 - c0 + 1) == (2, 2), "marker must occupy a 2x2 box"
    local = frozenset((r - r0, c - c0) for r, c in coords)
    assert local in L_TO_QUADRANT, "marker must be a 3-pixel L"
    return L_TO_QUADRANT[local], (r0, c0, r1, c1)


def _detect_markers(grid):
    markers = []
    for color in sorted(set(map(int, grid.flatten())) - {BG, TEMPLATE_COLOR}):
        comps = _find_components(grid == color)
        assert len(comps) == 1, "each marker color must identify exactly one L marker"
        coords = comps[0]
        assert len(coords) == 3, "each marker must contain exactly 3 cells"
        quadrant, bbox = _marker_quadrant(coords)
        markers.append({"color": color, "quadrant": quadrant, "bbox": bbox})
    assert 2 <= len(markers) <= 4, "expected 2 to 4 corner markers"
    return markers


def _stamp_into_corner(out, mask, quadrant, color):
    h, w = mask.shape
    gh, gw = out.shape
    if quadrant == "top_left":
        rs = slice(0, h)
        cs = slice(0, w)
    elif quadrant == "top_right":
        rs = slice(0, h)
        cs = slice(gw - w, gw)
    elif quadrant == "bottom_left":
        rs = slice(gh - h, gh)
        cs = slice(0, w)
    else:
        rs = slice(gh - h, gh)
        cs = slice(gw - w, gw)
    patch = out[rs, cs]
    patch[mask] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the central grey motif, flip it by corner-marker orientation, and stamp recolored copies into the canvas corners.

    Input structure:
    - The grid is a black odd square with one central grey motif and 2-4 colored 3-pixel L markers.
    - Each marker is monochromatic, 4-connected, and fits in a 2x2 box near one outer corner.
    - Grey pixels form the template content; marker colors are chosen from the non-grey corner palette.

    Rule:
    - Crop the tight bounding box of all grey pixels to obtain the template mask.
    - Decode each L marker by its missing cell: missing BR/TR/BL/TL maps to the target quadrant TL/BL/TR/BR respectively.
    - Flip the template according to that target quadrant: TL none, TR horizontal, BL vertical, BR both.
    - Clear the whole grid to black, then stamp each flipped template into the corresponding outer corner and recolor template pixels to the marker color.

    Key insight:
    - The L marker orientation does double duty: it selects both which corner receives the stamp and which flip of the central motif is required there.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    template, _ = _extract_template(g)
    markers = _detect_markers(g)

    out = np.zeros_like(g)
    for marker in markers:
        transformed = QUADRANT_TO_TRANSFORM[marker["quadrant"]](template)
        _stamp_into_corner(out, transformed, marker["quadrant"], marker["color"])
    return out


def _all_transforms(mask):
    return {
        mask.tobytes(),
        np.fliplr(mask).tobytes(),
        np.flipud(mask).tobytes(),
        np.flipud(np.fliplr(mask)).tobytes(),
    }


def _build_template():
    for _ in range(200):
        h = random.randint(3, 5)
        w = random.randint(3, 5)
        area = h * w
        count = random.randint(3, min(8, area - 1))
        flat = np.zeros(area, dtype=bool)
        flat[random.sample(range(area), count)] = True
        mask = flat.reshape(h, w)
        if not mask[0].any() or not mask[-1].any() or not mask[:, 0].any() or not mask[:, -1].any():
            continue
        if len(_all_transforms(mask)) != 4:
            continue
        return mask
    raise RuntimeError("failed to build a non-symmetric template")


def _choose_quadrants():
    count = random.choice((2, 3, 4))
    for _ in range(100):
        chosen = sorted(random.sample(QUADRANT_ORDER, count), key=QUADRANT_ORDER.index)
        has_top = any(q.startswith("top_") for q in chosen)
        has_bottom = any(q.startswith("bottom_") for q in chosen)
        has_left = any(q.endswith("_left") for q in chosen)
        has_right = any(q.endswith("_right") for q in chosen)
        if has_top and has_bottom and has_left and has_right:
            return chosen
    raise RuntimeError("failed to choose quadrants")


def _marker_anchor(size, quadrant, color, template):
    template_score = int(np.sum(template * np.arange(1, template.size + 1).reshape(template.shape)))
    offset = (color + template_score + len(template) + template.shape[1]) % 3
    if quadrant == "top_left":
        return offset, (2 * offset + color) % 3
    if quadrant == "top_right":
        return offset, size - 2 - ((offset + color) % 3)
    if quadrant == "bottom_left":
        return size - 2 - offset, (offset + 2 * color) % 3
    return size - 2 - offset, size - 2 - ((2 * offset + color) % 3)


def _place_marker(grid, quadrant, color, template):
    r0, c0 = _marker_anchor(grid.shape[0], quadrant, color, template)
    for dr, dc in next(local for local, q in L_TO_QUADRANT.items() if q == quadrant):
        grid[r0 + dr, c0 + dc] = color
    return (r0, c0, r0 + 1, c0 + 1)


def _expected_output_from_spec(grid):
    template, _ = _extract_template(grid)
    out = np.zeros_like(grid)
    for marker in _detect_markers(grid):
        transformed = QUADRANT_TO_TRANSFORM[marker["quadrant"]](template)
        _stamp_into_corner(out, transformed, marker["quadrant"], marker["color"])
    return out


def _validate_generated(grid):
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2
    h, w = grid.shape
    assert h == w and h in GRID_SIZES and h <= 30
    assert np.all(np.isin(grid, (0, 1, 2, 3, 4, 5)))

    template, box = _extract_template(grid)
    assert 3 <= int(template.sum()) <= 8
    assert len(_all_transforms(template)) == 4
    r0, c0, r1, c1 = box
    mid = h // 2
    assert r0 >= mid - 2 and c0 >= mid - 2
    assert r1 <= mid + 2 and c1 <= mid + 2
    assert template.shape[0] <= h // 2 and template.shape[1] <= w // 2

    markers = _detect_markers(grid)
    colors = [m["color"] for m in markers]
    assert len(colors) == len(set(colors))
    assert set(colors).issubset(MARKER_COLORS)
    assert any(m["quadrant"].startswith("top_") for m in markers)
    assert any(m["quadrant"].startswith("bottom_") for m in markers)
    assert any(m["quadrant"].endswith("_left") for m in markers)
    assert any(m["quadrant"].endswith("_right") for m in markers)

    for marker in markers:
        mr0, mc0, mr1, mc1 = marker["bbox"]
        if marker["quadrant"] == "top_left":
            assert mr0 <= 2 and mc0 <= 2
        elif marker["quadrant"] == "top_right":
            assert mr0 <= 2 and mc1 >= w - 3
        elif marker["quadrant"] == "bottom_left":
            assert mr1 >= h - 3 and mc0 <= 2
        else:
            assert mr1 >= h - 3 and mc1 >= w - 3

    expected = _expected_output_from_spec(grid)
    actual = main(grid)
    assert np.array_equal(actual, expected)

    for marker in markers:
        color = marker["color"]
        quadrant = marker["quadrant"]
        transformed = QUADRANT_TO_TRANSFORM[quadrant](template)
        th, tw = transformed.shape
        if quadrant == "top_left":
            corner = actual[:th, :tw]
        elif quadrant == "top_right":
            corner = actual[:th, -tw:]
        elif quadrant == "bottom_left":
            corner = actual[-th:, :tw]
        else:
            corner = actual[-th:, -tw:]
        assert np.array_equal(corner == color, transformed)

    for quadrant in QUADRANT_ORDER:
        if quadrant in {m["quadrant"] for m in markers}:
            continue
        if quadrant == "top_left":
            window = actual[: template.shape[0], : template.shape[1]]
        elif quadrant == "top_right":
            window = actual[: template.shape[0], -template.shape[1] :]
        elif quadrant == "bottom_left":
            window = actual[-template.shape[0] :, : template.shape[1]]
        else:
            window = actual[-template.shape[0] :, -template.shape[1] :]
        assert not np.any(window)


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        size = random.choice(GRID_SIZES)
        grid = np.zeros((size, size), dtype=int)

        template = _build_template()
        th, tw = template.shape
        mid = size // 2
        r0 = mid - th // 2
        c0 = mid - tw // 2
        grid[r0 : r0 + th, c0 : c0 + tw][template] = TEMPLATE_COLOR

        quadrants = _choose_quadrants()
        colors = random.sample(MARKER_COLORS, len(quadrants))
        for quadrant, color in zip(quadrants, colors):
            bbox = _place_marker(grid, quadrant, color, template)
            br0, bc0, br1, bc1 = bbox
            template_box = (r0, c0, r0 + th - 1, c0 + tw - 1)
            if not (br1 < template_box[0] or template_box[2] < br0 or bc1 < template_box[1] or template_box[3] < bc0):
                break
        else:
            try:
                _validate_generated(grid)
            except AssertionError:
                continue
            return grid

    raise RuntimeError("failed to generate a valid puzzle input")
