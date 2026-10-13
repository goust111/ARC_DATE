import numpy as np


BG_COLOR = 0
FRAME_COLOR = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 400
MAX_SIDE = 72
_RNG = np.random.default_rng()

# The four L orientations are encoded by which corner of a 2x2 box is missing.
# We treat the missing-corner cycle TL, TR, BR, BL as directions up, right, down, left.
_BASE_L = np.array([[0, 1], [1, 1]], dtype=bool)
L_MASKS = tuple(np.rot90(_BASE_L, -k) for k in range(4))


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _is_rect_frame(grid: np.ndarray, color: int, box: tuple[int, int, int, int], area: int) -> bool:
    r0, c0, r1, c1 = box
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 4 or w < 4:
        return False
    if area != 2 * h + 2 * w - 4:
        return False
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    if not np.all(patch[0, :] == color):
        return False
    if not np.all(patch[-1, :] == color):
        return False
    if not np.all(patch[:, 0] == color):
        return False
    if not np.all(patch[:, -1] == color):
        return False
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return not np.any((patch == color) & ~border)


def _shape_index(mask: np.ndarray) -> int:
    for i, ref in enumerate(L_MASKS):
        if np.array_equal(mask, ref):
            return i
    raise AssertionError("mask is not a 3-cell L")


def _rotation_steps(src: np.ndarray, dst: np.ndarray) -> int:
    for k in range(4):
        if np.array_equal(np.rot90(src, -k), dst):
            return k
    raise AssertionError("fragment is not a rotation of template")


def _extract_payload(
    grid: np.ndarray, box: tuple[int, int, int, int], bg: int, frame_color: int
) -> dict[str, object]:
    r0, c0, r1, c1 = box
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    colors = [int(v) for v in np.unique(inner) if int(v) not in (bg, frame_color)]
    assert len(colors) == 1, "each frame must contain exactly one colored fragment"
    color = colors[0]
    pts = np.argwhere(inner == color)
    assert pts.shape[0] == 3, "fragment must occupy exactly three cells"
    pr0, pc0, pr1, pc1 = _bbox(pts)
    assert pr1 - pr0 == 1 and pc1 - pc0 == 1, "fragment must fit in a 2x2 box"
    mask = np.zeros((2, 2), dtype=bool)
    mask[pts[:, 0] - pr0, pts[:, 1] - pc0] = True
    _shape_index(mask)
    blank_count = int(np.sum(inner == bg))
    assert blank_count == int(inner.size - 3), "region interior must be otherwise empty"
    return {"color": color, "mask": mask, "blank_count": blank_count}


def _paint_extension(
    out: np.ndarray,
    box: tuple[int, int, int, int],
    direction: int,
    length: int,
    color: int,
) -> None:
    r0, c0, r1, c1 = box
    if direction == 0:
        assert r0 - length >= 0
        out[r0 - length : r0, c0 : c1 + 1] = color
    elif direction == 1:
        assert c1 + length < out.shape[1]
        out[r0 : r1 + 1, c1 + 1 : c1 + 1 + length] = color
    elif direction == 2:
        assert r1 + length < out.shape[0]
        out[r1 + 1 : r1 + 1 + length, c0 : c1 + 1] = color
    else:
        assert c0 - length >= 0
        out[r0 : r1 + 1, c0 - length : c0] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each colored fragment to its same-color template, recover its orientation,
    and extend the corresponding side of the enclosing grey rectangle outward.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)

    raw_frames: list[tuple[int, tuple[int, int, int, int]]] = []
    for color in np.unique(g):
        color = int(color)
        if color == bg:
            continue
        for comp in _components(g == color):
            box = _bbox(comp)
            if _is_rect_frame(g, color, box, int(comp.shape[0])):
                raw_frames.append((color, box))

    assert raw_frames, "no rectangular frames found"
    frame_colors = [color for color, _box in raw_frames]
    color_vals, color_counts = np.unique(frame_colors, return_counts=True)
    frame_color = int(color_vals[np.argmax(color_counts)])
    frames = [box for color, box in raw_frames if color == frame_color]

    templates: dict[int, np.ndarray] = {}
    targets: list[tuple[tuple[int, int, int, int], dict[str, object]]] = []
    for box in frames:
        r0, c0, r1, c1 = box
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        payload = _extract_payload(g, box, bg, frame_color)
        if h == 4 and w == 4:
            color = int(payload["color"])
            assert color not in templates, "template colors must be unique"
            templates[color] = np.asarray(payload["mask"], dtype=bool)
        else:
            targets.append((box, payload))

    assert templates, "no template dictionary found"
    assert targets, "no target regions found"

    out = g.copy()
    for box, payload in targets:
        color = int(payload["color"])
        assert color in templates, "target color missing from template dictionary"
        template_mask = templates[color]
        template_dir = _shape_index(template_mask)
        fragment_mask = np.asarray(payload["mask"], dtype=bool)
        rotation = _rotation_steps(template_mask, fragment_mask)
        actual_dir = (template_dir + rotation) % 4
        blank_count = int(payload["blank_count"])
        _paint_extension(out, box, actual_dir, blank_count, color)

    return out


def _draw_frame(grid: np.ndarray, r0: int, c0: int, h: int, w: int, color: int) -> None:
    grid[r0, c0 : c0 + w] = color
    grid[r0 + h - 1, c0 : c0 + w] = color
    grid[r0 : r0 + h, c0] = color
    grid[r0 : r0 + h, c0 + w - 1] = color


def _draw_mask(grid: np.ndarray, r0: int, c0: int, mask: np.ndarray, color: int) -> None:
    h, w = mask.shape
    patch = grid[r0 : r0 + h, c0 : c0 + w]
    patch[mask] = color


def _sample_direction_list(count: int) -> list[int]:
    dirs = list(_RNG.permutation(4)[: min(count, 4)])
    while len(dirs) < count:
        dirs.append(int(_RNG.integers(4)))
    _RNG.shuffle(dirs)
    return dirs


def _sample_size() -> int:
    return int(_RNG.choice([3, 4, 5, 6], p=[0.35, 0.35, 0.20, 0.10]))


def _build_region_specs(template_dirs: dict[int, int], template_colors: list[int]) -> list[dict[str, int]]:
    region_count = int(_RNG.integers(2, 6))
    directions = _sample_direction_list(region_count)
    colors = [int(_RNG.choice(template_colors)) for _ in range(region_count)]
    if region_count >= 2 and len(set(colors)) == 1 and len(template_colors) >= 2:
        swap = [c for c in template_colors if c != colors[0]]
        colors[1] = int(_RNG.choice(swap))

    specs: list[dict[str, int]] = []
    for color, actual_dir in zip(colors, directions):
        inner_h = _sample_size()
        inner_w = _sample_size()
        outer_h = inner_h + 2
        outer_w = inner_w + 2
        blank_count = inner_h * inner_w - 3
        template_dir = template_dirs[color]
        rotation = (actual_dir - template_dir) % 4
        frag_mask = np.rot90(L_MASKS[template_dir], -rotation)
        offset_r = int(_RNG.integers(0, inner_h - 1))
        offset_c = int(_RNG.integers(0, inner_w - 1))

        if actual_dir in (0, 2):
            footprint_h = outer_h + blank_count
            footprint_w = outer_w
        else:
            footprint_h = outer_h
            footprint_w = outer_w + blank_count

        specs.append(
            {
                "color": color,
                "actual_dir": actual_dir,
                "inner_h": inner_h,
                "inner_w": inner_w,
                "outer_h": outer_h,
                "outer_w": outer_w,
                "blank_count": blank_count,
                "offset_r": offset_r,
                "offset_c": offset_c,
                "footprint_h": footprint_h,
                "footprint_w": footprint_w,
                "frag_code": sum((1 << i) for i, v in enumerate(frag_mask.ravel()) if v),
            }
        )
    return specs


def _decode_frag_mask(code: int) -> np.ndarray:
    mask = np.zeros((2, 2), dtype=bool)
    for i in range(4):
        if code & (1 << i):
            mask.ravel()[i] = True
    return mask


def _pack_regions(dict_width: int, specs: list[dict[str, int]]) -> list[tuple[int, int]] | None:
    max_fw = max(int(spec["footprint_w"]) for spec in specs)
    total_fw = sum(int(spec["footprint_w"]) for spec in specs)
    row_limit = max(dict_width, max_fw, min(56, total_fw // 2 + int(_RNG.integers(8, 16))))
    start_r = 4 + int(_RNG.integers(2, 5))

    order = list(_RNG.permutation(len(specs)))
    positions: list[tuple[int, int] | None] = [None] * len(specs)
    cursor_r = start_r
    cursor_c = 0
    row_h = 0
    for idx in order:
        spec = specs[idx]
        fw = int(spec["footprint_w"])
        fh = int(spec["footprint_h"])
        if cursor_c > 0 and cursor_c + fw > row_limit:
            cursor_r += row_h + int(_RNG.integers(2, 5))
            cursor_c = 0
            row_h = 0
        positions[idx] = (cursor_r, cursor_c)
        cursor_c += fw + int(_RNG.integers(2, 5))
        row_h = max(row_h, fh)

    assert all(pos is not None for pos in positions)
    bottom = max(int(pos[0]) + int(spec["footprint_h"]) for pos, spec in zip(positions, specs))
    right = max(int(pos[1]) + int(spec["footprint_w"]) for pos, spec in zip(positions, specs))
    if bottom > MAX_SIDE or max(dict_width, right) > MAX_SIDE:
        return None
    return [(int(r), int(c)) for r, c in positions]


def _gen_attempt() -> np.ndarray | None:
    template_count = int(_RNG.integers(3, 6))
    template_colors = list(map(int, _RNG.choice(PALETTE, size=template_count, replace=False)))
    template_dirs = {color: int(_RNG.integers(4)) for color in template_colors}
    if template_count >= 2 and len(set(template_dirs.values())) == 1:
        template_dirs[template_colors[0]] = (template_dirs[template_colors[0]] + 1) % 4

    specs = _build_region_specs(template_dirs, template_colors)
    dict_width = template_count * 4 + (template_count - 1)
    positions = _pack_regions(dict_width, specs)
    if positions is None:
        return None

    total_h = max(4, max(r + int(spec["footprint_h"]) for (r, _c), spec in zip(positions, specs)))
    total_w = max(dict_width, max(c + int(spec["footprint_w"]) for (_r, c), spec in zip(positions, specs)))
    if total_h > MAX_SIDE or total_w > MAX_SIDE:
        return None

    grid = np.full((total_h, total_w), BG_COLOR, dtype=int)
    expected = grid.copy()

    for i, color in enumerate(template_colors):
        c0 = i * 5
        _draw_frame(grid, 0, c0, 4, 4, FRAME_COLOR)
        _draw_frame(expected, 0, c0, 4, 4, FRAME_COLOR)
        mask = L_MASKS[template_dirs[color]]
        _draw_mask(grid, 1, c0 + 1, mask, color)
        _draw_mask(expected, 1, c0 + 1, mask, color)

    for (fr0, fc0), spec in zip(positions, specs):
        actual_dir = int(spec["actual_dir"])
        blank_count = int(spec["blank_count"])
        outer_h = int(spec["outer_h"])
        outer_w = int(spec["outer_w"])
        color = int(spec["color"])

        if actual_dir == 0:
            r0, c0 = fr0 + blank_count, fc0
        elif actual_dir == 1:
            r0, c0 = fr0, fc0
        elif actual_dir == 2:
            r0, c0 = fr0, fc0
        else:
            r0, c0 = fr0, fc0 + blank_count

        _draw_frame(grid, r0, c0, outer_h, outer_w, FRAME_COLOR)
        _draw_frame(expected, r0, c0, outer_h, outer_w, FRAME_COLOR)

        frag_mask = _decode_frag_mask(int(spec["frag_code"]))
        frag_r = r0 + 1 + int(spec["offset_r"])
        frag_c = c0 + 1 + int(spec["offset_c"])
        _draw_mask(grid, frag_r, frag_c, frag_mask, color)
        _draw_mask(expected, frag_r, frag_c, frag_mask, color)
        _paint_extension(expected, (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1), actual_dir, blank_count, color)

    rot_k = int(_RNG.integers(4))
    inp = np.rot90(grid, rot_k)
    out = np.rot90(expected, rot_k)
    return inp if np.array_equal(main(inp), out) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid example")
