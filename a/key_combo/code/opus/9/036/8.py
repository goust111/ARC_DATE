import numpy as np


BLACK = 0
BLUE = 1
SPRITE_COLORS = (2, 3, 4)

TYPE_VERTICAL = "vertical"
TYPE_HORIZONTAL = "horizontal"
TYPE_BOTH = "both"
ALL_TYPES = (TYPE_VERTICAL, TYPE_HORIZONTAL, TYPE_BOTH)

_RNG = np.random.default_rng()
_GEN_ATTEMPTS = 400


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _border_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _coords_to_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([r0, c0], dtype=int)
    mask[rel[:, 0], rel[:, 1]] = True
    return mask


def _classify_sprite_coords(coords: np.ndarray) -> str:
    mask = _coords_to_mask(coords)
    vertical = bool(np.array_equal(mask, mask[:, ::-1]))
    horizontal = bool(np.array_equal(mask, mask[::-1, :]))
    if vertical and horizontal:
        return TYPE_BOTH
    if vertical:
        return TYPE_VERTICAL
    if horizontal:
        return TYPE_HORIZONTAL
    raise AssertionError("sprite must have at least one symmetry axis")


def _fill_pattern(inner_h: int, inner_w: int, color: int, sprite_type: str) -> np.ndarray:
    patch = np.zeros((inner_h, inner_w), dtype=int)
    if sprite_type == TYPE_VERTICAL:
        patch[:, 0::2] = int(color)
    elif sprite_type == TYPE_HORIZONTAL:
        patch[0::2, :] = int(color)
    elif sprite_type == TYPE_BOTH:
        rr, cc = np.indices((inner_h, inner_w))
        patch[(rr + cc) % 2 == 0] = int(color)
    else:
        raise AssertionError("unknown sprite type")
    return patch


def _extract_frames(grid: np.ndarray) -> list[tuple[tuple[int, int, int, int], np.ndarray]]:
    blue_comps = _components4(grid == BLUE)
    assert 2 <= len(blue_comps) <= 3, "expected 2 or 3 blue frames"
    frames: list[tuple[tuple[int, int, int, int], np.ndarray]] = []
    for coords in blue_comps:
        bbox = _bbox_from_coords(coords)
        r0, c0, r1, c1 = bbox
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert h >= 6 and w >= 6, "frame interior must be at least 4x4"
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        assert np.array_equal(patch == BLUE, _border_mask(h, w)), "blue component must be a hollow rectangle"
        frames.append((bbox, coords))
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode each centered sprite's symmetry type and replace the frame interior with the matching fill pattern.

    Input structure:
    - The grid has a black background and contains 2 or 3 non-touching blue hollow rectangular frames.
    - Each frame encloses a single centered monochrome sprite in red, green, or yellow; the sprite is 4-connected and does not touch the frame.
    - Every sprite has exactly one of three symmetry classes: vertical only, horizontal only, or both vertical and horizontal.

    Rule:
    - Detect each blue rectangular frame and read the unique non-blue connected sprite inside it.
    - If the sprite is only vertically symmetric, fill the whole interior with that sprite's color in even-indexed interior columns.
    - If the sprite is only horizontally symmetric, fill the whole interior with that sprite's color in even-indexed interior rows; if it has both symmetries, fill interior cells of even (row + col) parity.

    Key insight:
    - The sprite itself is only an instruction token: its symmetry class selects the fill pattern, and its color selects the paint color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    frames = _extract_frames(g)
    for bbox, _coords in frames:
        r0, c0, r1, c1 = bbox
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        assert inner.shape[0] >= 4 and inner.shape[1] >= 4
        assert not np.any(inner == BLUE), "frame interior must not contain blue"
        sprite_mask = inner != BLACK
        comps = _components4(sprite_mask)
        assert len(comps) == 1, "each frame interior must contain exactly one sprite"
        sprite = comps[0]
        sprite_colors = {int(inner[r, c]) for r, c in sprite}
        assert len(sprite_colors) == 1 and next(iter(sprite_colors)) in SPRITE_COLORS
        color = next(iter(sprite_colors))
        sprite_type = _classify_sprite_coords(sprite)
        out[r0 + 1 : r1, c0 + 1 : c1] = _fill_pattern(inner.shape[0], inner.shape[1], color, sprite_type)
    return out


_VERTICAL_TEMPLATES = (
    np.array([[0, 1], [1, 0], [1, 1], [1, 2]], dtype=int),
    np.array([[0, 1], [1, 1], [2, 0], [2, 1], [2, 2]], dtype=int),
    np.array([[0, 0], [0, 2], [1, 0], [1, 1], [1, 2], [2, 1]], dtype=int),
    np.array([[0, 0], [0, 1], [0, 2], [1, 1], [2, 1]], dtype=int),
)
_HORIZONTAL_TEMPLATES = tuple(np.stack((tpl[:, 1], tpl[:, 0]), axis=1) for tpl in _VERTICAL_TEMPLATES)
_BOTH_TEMPLATES = (
    np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=int),
    np.array([[0, 1], [1, 0], [1, 1], [1, 2], [2, 1]], dtype=int),
    np.array([[0, 0], [0, 1], [0, 2]], dtype=int),
    np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]], dtype=int),
)
_TEMPLATE_LIBRARY = {
    TYPE_VERTICAL: _VERTICAL_TEMPLATES,
    TYPE_HORIZONTAL: _HORIZONTAL_TEMPLATES,
    TYPE_BOTH: _BOTH_TEMPLATES,
}


def _choose_type_multiset(num_frames: int) -> list[str]:
    if num_frames == 2:
        picked = _RNG.choice(ALL_TYPES, size=2, replace=False)
        return [str(x) for x in picked]

    if float(_RNG.random()) < 0.6:
        picked = list(ALL_TYPES)
        _RNG.shuffle(picked)
        return picked

    base = list(_RNG.choice(ALL_TYPES, size=2, replace=False))
    extra = str(_RNG.choice(base))
    picked = [str(base[0]), str(base[1]), extra]
    _RNG.shuffle(picked)
    return picked


def _choose_sprite_template(sprite_type: str) -> np.ndarray:
    templates = _TEMPLATE_LIBRARY[sprite_type]
    return np.array(templates[int(_RNG.integers(len(templates)))], dtype=int)


def _frame_spec(sprite_type: str, color: int) -> dict:
    sprite = _choose_sprite_template(sprite_type)
    sprite_h = int(sprite[:, 0].max()) + 1
    sprite_w = int(sprite[:, 1].max()) + 1
    extra_h = int(_RNG.integers(0, 3)) * 2
    extra_w = int(_RNG.integers(0, 3)) * 2
    inner_h = sprite_h + 2 + extra_h
    inner_w = sprite_w + 2 + extra_w
    return {
        "type": sprite_type,
        "color": int(color),
        "sprite": sprite,
        "inner_h": int(inner_h),
        "inner_w": int(inner_w),
        "outer_h": int(inner_h + 2),
        "outer_w": int(inner_w + 2),
    }


def _build_expected(grid: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = grid.copy()
    for spec in specs:
        r0, c0 = spec["top"], spec["left"]
        inner_h = spec["inner_h"]
        inner_w = spec["inner_w"]
        out[r0 + 1 : r0 + 1 + inner_h, c0 + 1 : c0 + 1 + inner_w] = _fill_pattern(
            inner_h,
            inner_w,
            spec["color"],
            spec["type"],
        )
    return out


def _place_specs(specs: list[dict], height: int, width: int) -> bool:
    occupied = np.zeros((height, width), dtype=bool)
    order = list(range(len(specs)))
    _RNG.shuffle(order)
    for idx in order:
        spec = specs[idx]
        oh = spec["outer_h"]
        ow = spec["outer_w"]
        if oh + 2 > height or ow + 2 > width:
            return False
        placed = False
        for _ in range(80):
            top = int(_RNG.integers(1, height - oh))
            left = int(_RNG.integers(1, width - ow))
            r0 = max(0, top - 1)
            c0 = max(0, left - 1)
            r1 = min(height, top + oh + 1)
            c1 = min(width, left + ow + 1)
            if occupied[r0:r1, c0:c1].any():
                continue
            occupied[r0:r1, c0:c1] = True
            spec["top"] = top
            spec["left"] = left
            placed = True
            break
        if not placed:
            return False
    return True


def _draw_input(height: int, width: int, specs: list[dict]) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    for spec in specs:
        top = spec["top"]
        left = spec["left"]
        oh = spec["outer_h"]
        ow = spec["outer_w"]
        grid[top : top + oh, left : left + ow][_border_mask(oh, ow)] = BLUE

        sprite = spec["sprite"]
        sprite_h = int(sprite[:, 0].max()) + 1
        sprite_w = int(sprite[:, 1].max()) + 1
        sprite_top = top + 1 + (spec["inner_h"] - sprite_h) // 2
        sprite_left = left + 1 + (spec["inner_w"] - sprite_w) // 2
        abs_coords = sprite + np.array([sprite_top, sprite_left], dtype=int)
        grid[abs_coords[:, 0], abs_coords[:, 1]] = spec["color"]
    return grid


def _validate_instance(grid: np.ndarray, specs: list[dict]) -> None:
    assert grid.ndim == 2 and max(grid.shape) <= 30
    assert len({spec["color"] for spec in specs}) == len(specs)
    assert len({spec["type"] for spec in specs}) >= 2

    for spec in specs:
        sprite_type = _classify_sprite_coords(spec["sprite"])
        assert sprite_type == spec["type"]
        sprite_h = int(spec["sprite"][:, 0].max()) + 1
        sprite_w = int(spec["sprite"][:, 1].max()) + 1
        assert spec["inner_h"] >= sprite_h + 2
        assert spec["inner_w"] >= sprite_w + 2

    expected = _build_expected(grid, specs)
    actual = main(grid)
    assert np.array_equal(actual, expected), "solver output must match construction metadata"
    assert not np.array_equal(grid, actual), "input must differ from output"

    blue_mask = grid == BLUE
    comps = _components4(blue_mask)
    assert len(comps) == len(specs)
    for comp in comps:
        bbox = _bbox_from_coords(comp)
        r0, c0, r1, c1 = bbox
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        assert np.array_equal(patch == BLUE, _border_mask(patch.shape[0], patch.shape[1]))


def _gen_attempt() -> np.ndarray | None:
    num_frames = int(_RNG.choice((2, 3), p=(0.35, 0.65)))
    types = _choose_type_multiset(num_frames)
    colors = list(_RNG.permutation(np.array(SPRITE_COLORS, dtype=int))[:num_frames])
    specs = [_frame_spec(types[i], colors[i]) for i in range(num_frames)]

    height = int(_RNG.integers(14, 25))
    width = int(_RNG.integers(14, 25))
    if not _place_specs(specs, height, width):
        return None

    grid = _draw_input(height, width, specs)
    try:
        _validate_instance(grid, specs)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
