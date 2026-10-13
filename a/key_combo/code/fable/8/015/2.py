from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BACKGROUND = 0
BOARD_COLOR = 5
TEMPLATE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

SHAPE_LIBRARY = (
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 1), (2, 2)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 1), (0, 2), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (0, 3)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    ((0, 0), (0, 1), (1, 1), (1, 2), (2, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0)),
)

MAX_ATTEMPTS = 300


def _normalize_coords(coords: np.ndarray | list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    arr = np.asarray(coords, dtype=int)
    if arr.size == 0:
        return ()
    arr = arr - arr.min(axis=0)
    return tuple(sorted((int(r), int(c)) for r, c in arr))


def _coords_to_mask(coords: tuple[tuple[int, int], ...]) -> np.ndarray:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    mask = np.zeros((max(rows) + 1, max(cols) + 1), dtype=bool)
    for r, c in coords:
        mask[r, c] = True
    return mask


def _rotation_signatures(coords: tuple[tuple[int, int], ...]) -> set[tuple[tuple[int, int], ...]]:
    mask = _coords_to_mask(coords)
    signatures = set()
    for k in range(4):
        rot = np.rot90(mask, k=k)
        signatures.add(_normalize_coords(np.argwhere(rot)))
    return signatures


CANONICAL_LIBRARY: tuple[tuple[tuple[int, int], ...], ...] = ()
AREA_GROUPS: dict[int, list[tuple[tuple[int, int], ...]]] = {}
for _shape in SHAPE_LIBRARY:
    canonical = min(_rotation_signatures(_shape))
    if canonical not in CANONICAL_LIBRARY:
        CANONICAL_LIBRARY += (canonical,)
        AREA_GROUPS.setdefault(len(canonical), []).append(canonical)


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.asarray(coords, dtype=int))
    return components


def _extract_templates(grid: np.ndarray, board_color: int) -> list[tuple[int, tuple[tuple[int, int], ...]]]:
    templates = []
    for color in sorted(int(v) for v in np.unique(grid)):
        if color in {BACKGROUND, board_color}:
            continue
        coords = np.argwhere(grid == color)
        if coords.size == 0:
            continue
        templates.append((color, _normalize_coords(coords)))
    return templates


def _find_board_bboxes(grid: np.ndarray, board_color: int) -> list[tuple[int, int, int, int]]:
    bboxes = []
    for coords in _components(grid == board_color):
        if coords.size == 0:
            continue
        bboxes.append(_bbox_from_coords(coords))
    bboxes.sort()
    return bboxes


def _dent_signature(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[tuple[int, int], ...]:
    r0, c0, r1, c1 = bbox
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    dents = np.argwhere(patch == BACKGROUND)
    return _normalize_coords(dents)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read each gray board's dent constellation and refill its holes with the matching template color when possible.

    Input structure:
    - A black background contains several large gray solid rectangles that act as boards.
    - Each board contains one or more single-cell black dents entirely inside its rectangle.
    - Along the top sit several small colored template constellations, one color per template.

    Rule:
    - For each gray board, collect the full relative dent pattern inside its bounding rectangle.
    - Compare that dent constellation against every top template under 0/90/180/270 degree rotation.
    - If exactly one template matches, recolor all dents on that board to the matched template color; otherwise refill the dents with gray.

    Key insight:
    - The black holes inside a board encode a rotatable shape rather than generic damage, so pixel count alone is insufficient when multiple templates share the same area.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    non_bg = [int(v) for v in np.unique(g) if int(v) != BACKGROUND]
    assert non_bg, "expected foreground objects"
    board_color = max(non_bg, key=lambda color: int(np.sum(g == color)))

    templates = _extract_templates(g, board_color)
    template_rotations = [(color, _rotation_signatures(shape)) for color, shape in templates]

    out = g.copy()
    for bbox in _find_board_bboxes(g, board_color):
        signature = _dent_signature(g, bbox)
        if not signature:
            continue

        matches = [color for color, rotations in template_rotations if signature in rotations]
        fill_color = int(matches[0]) if len(matches) == 1 else board_color

        r0, c0, r1, c1 = bbox
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        dents = np.argwhere(patch == BACKGROUND)
        for dr, dc in dents:
            out[r0 + int(dr), c0 + int(dc)] = fill_color

    return out


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    rows = [r for r, _ in shape]
    cols = [c for _, c in shape]
    return max(rows) + 1, max(cols) + 1


def _render_shape(color: int, shape: tuple[tuple[int, int], ...]) -> np.ndarray:
    h, w = _shape_dims(shape)
    patch = np.full((h, w), BACKGROUND, dtype=int)
    for r, c in shape:
        patch[r, c] = color
    return patch


def _choose_templates(rng: random.Random) -> list[dict]:
    same_area = rng.choice([area for area, shapes in AREA_GROUPS.items() if len(shapes) >= 2])
    first_pair = rng.sample(AREA_GROUPS[same_area], 2)

    remaining = [shape for shape in CANONICAL_LIBRARY if shape not in first_pair]
    extra_count = rng.randint(1, 2)
    extras = rng.sample(remaining, extra_count)

    shapes = first_pair + extras
    rng.shuffle(shapes)

    colors = rng.sample(TEMPLATE_COLORS, len(shapes))
    templates = []
    for color, shape in zip(colors, shapes, strict=True):
        templates.append(
            {
                "color": int(color),
                "shape": shape,
                "rotations": _rotation_signatures(shape),
                "area": len(shape),
            }
        )
    return templates


def _pick_unmatched_shape(
    rng: random.Random, templates: list[dict]
) -> tuple[tuple[tuple[int, int], ...], int]:
    template_shapes = {tpl["shape"] for tpl in templates}
    preferred_areas = [tpl["area"] for tpl in templates]
    rng.shuffle(preferred_areas)

    for area in preferred_areas:
        pool = [shape for shape in AREA_GROUPS.get(area, []) if shape not in template_shapes]
        if pool:
            shape = rng.choice(pool)
            return shape, area

    pool = [shape for shape in CANONICAL_LIBRARY if shape not in template_shapes]
    shape = rng.choice(pool)
    return shape, len(shape)


def _board_patch(
    rng: random.Random, shape: tuple[tuple[int, int], ...], board_color: int
) -> tuple[np.ndarray, tuple[tuple[int, int], ...]]:
    mask = _coords_to_mask(shape)
    k = rng.randint(0, 3)
    rot = np.rot90(mask, k=k)
    dents = _normalize_coords(np.argwhere(rot))
    dent_h, dent_w = rot.shape

    pad_top = rng.randint(1, 3)
    pad_bottom = rng.randint(1, 3)
    pad_left = rng.randint(1, 3)
    pad_right = rng.randint(1, 3)
    h = dent_h + pad_top + pad_bottom
    w = dent_w + pad_left + pad_right

    patch = np.full((h, w), board_color, dtype=int)
    placed = []
    for r, c in dents:
        rr = pad_top + r
        cc = pad_left + c
        patch[rr, cc] = BACKGROUND
        placed.append((rr, cc))
    return patch, tuple(placed)


def _try_place(
    rng: random.Random,
    canvas: np.ndarray,
    patch: np.ndarray,
    occupied: np.ndarray,
    row_min: int,
    padding: int,
) -> tuple[int, int] | None:
    h, w = canvas.shape
    ph, pw = patch.shape
    choices = []
    for r0 in range(row_min, h - ph + 1):
        for c0 in range(0, w - pw + 1):
            rr0 = max(0, r0 - padding)
            cc0 = max(0, c0 - padding)
            rr1 = min(h, r0 + ph + padding)
            cc1 = min(w, c0 + pw + padding)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            choices.append((r0, c0))
    if not choices:
        return None
    return rng.choice(choices)


def _stamp(canvas: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    canvas[top : top + h, left : left + w] = patch


def _expected_output(grid: np.ndarray, boards: list[dict]) -> np.ndarray:
    out = grid.copy()
    for board in boards:
        top = board["top"]
        left = board["left"]
        fill_color = board["fill_color"]
        for r, c in board["dents"]:
            out[top + r, left + c] = fill_color
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    templates = _choose_templates(rng)
    template_patches = [_render_shape(tpl["color"], tpl["shape"]) for tpl in templates]

    gap = rng.randint(2, 4)
    template_width = sum(int(p.shape[1]) for p in template_patches) + gap * (len(template_patches) - 1)
    max_template_h = max(int(p.shape[0]) for p in template_patches)

    board_count = rng.randint(2, 4)
    matched_count = rng.randint(1, board_count - 1)
    matched_flags = [True] * matched_count + [False] * (board_count - matched_count)
    rng.shuffle(matched_flags)

    board_specs = []
    for is_matched in matched_flags:
        if is_matched:
            template = rng.choice(templates)
            shape = template["shape"]
            fill_color = template["color"]
        else:
            shape, _area = _pick_unmatched_shape(rng, templates)
            fill_color = BOARD_COLOR
        patch, dents = _board_patch(rng, shape, BOARD_COLOR)
        board_specs.append(
            {
                "patch": patch,
                "dents": dents,
                "fill_color": int(fill_color),
            }
        )

    min_board_h = max(int(spec["patch"].shape[0]) for spec in board_specs)
    total_board_area = sum(int(spec["patch"].shape[0] * spec["patch"].shape[1]) for spec in board_specs)

    min_w = max(18, template_width + 4)
    max_w = 30
    if min_w > max_w:
        return None

    min_h = max(max_template_h + 8, min_board_h + max_template_h + 6)
    max_h = 30
    if min_h > max_h:
        return None

    for _ in range(80):
        width = rng.randint(min_w, max_w)
        height = rng.randint(min_h, max_h)
        if height * width < total_board_area + template_width + 30:
            continue

        grid = np.full((height, width), BACKGROUND, dtype=int)
        occupied = np.zeros((height, width), dtype=bool)

        start_c_max = width - template_width - 1
        if start_c_max < 1:
            continue
        cursor = rng.randint(1, start_c_max)
        template_row = rng.randint(1, 2)

        for idx, patch in enumerate(template_patches):
            ph, pw = patch.shape
            _stamp(grid, patch, template_row, cursor)
            occupied[
                max(0, template_row - 1) : min(height, template_row + ph + 1),
                max(0, cursor - 1) : min(width, cursor + pw + 1),
            ] = True
            cursor += pw + (gap if idx < len(template_patches) - 1 else 0)

        boards = []
        row_min = template_row + max_template_h + 3
        for spec in sorted(board_specs, key=lambda item: int(item["patch"].shape[0] * item["patch"].shape[1]), reverse=True):
            placement = _try_place(rng, grid, spec["patch"], occupied, row_min=row_min, padding=1)
            if placement is None:
                boards = []
                break
            top, left = placement
            ph, pw = spec["patch"].shape
            _stamp(grid, spec["patch"], top, left)
            occupied[
                max(0, top - 1) : min(height, top + ph + 1),
                max(0, left - 1) : min(width, left + pw + 1),
            ] = True
            boards.append(
                {
                    "top": top,
                    "left": left,
                    "dents": spec["dents"],
                    "fill_color": spec["fill_color"],
                }
            )
        if not boards:
            continue

        expected = _expected_output(grid, boards)
        try:
            actual = main(grid)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return grid

    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
