from collections import Counter, defaultdict
import random

import numpy as np

from arc_utils import bbox_coords, find_components, get_bg, paste, random_free_location


_DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _l_bend(body_coords: np.ndarray) -> tuple[int, int]:
    """Return the corner cell where the two one-cell-wide L legs meet."""
    coords = [tuple(map(int, rc)) for rc in body_coords]
    assert coords, "body must contain pixels"
    coord_set = set(coords)
    r0, c0, r1, c1 = bbox_coords(np.array(coords, dtype=int))
    corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}

    bends = []
    for r, c in coords:
        if (r, c) not in corners:
            continue
        nbr_dirs = [(dr, dc) for dr, dc in _DIRS4 if (r + dr, c + dc) in coord_set]
        has_vertical = any(dc == 0 for dr, dc in nbr_dirs)
        has_horizontal = any(dr == 0 for dr, dc in nbr_dirs)
        if has_vertical and has_horizontal:
            bends.append((r, c))

    assert len(bends) == 1, f"expected one L bend, got {bends}"
    return bends[0]


def _outer_diagonal_distance(
    anchor: tuple[int, int], marker: tuple[int, int], target: tuple[int, int]
) -> int:
    """Return distance when target is on the outer 45-degree ray from the L bend."""
    mr = marker[0] - anchor[0]
    mc = marker[1] - anchor[1]
    assert abs(mr) == 1 and abs(mc) == 1, "marker must be diagonally adjacent to L bend"

    tr = target[0] - anchor[0]
    tc = target[1] - anchor[1]
    dist = -tr * mr
    assert dist >= 1 and tr == -mr * dist and tc == -mc * dist, (
        "control must lie on the outer 45-degree ray from the L bend"
    )
    return dist


def main(grid: np.ndarray) -> np.ndarray:
    """Move each marked L-shaped sprite onto the separate pixel matching its marker color.

    Input structure
    - The background is the dominant color.
    - Foreground has 4-connected multicolor sprites and isolated one-pixel controls.
    - Each sprite has one repeated body color in an L shape plus one embedded marker
      pixel of a different color.

    Rule
    - If the embedded marker color also appears as a separate isolated control pixel,
      translate the whole sprite so the body L-bend lands on that control pixel.
    - Clear each moved sprite from its old position, clear its consumed control pixel,
      then paste all translated sprites; sprites without a matching control stay put.

    Key insight
    - The marker color pairs a sprite with its destination control, while the L-bend
      gives the orientation-independent anchor cell to place on that destination.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    components = find_components(g, bg=bg, connectivity=4, monochromatic=False)

    controls_by_color: dict[int, list[tuple[int, int]]] = defaultdict(list)
    sprites = []
    for comp in components:
        coords = comp["coords"]
        if len(coords) == 1:
            r, c = map(int, coords[0])
            controls_by_color[int(g[r, c])].append((r, c))
        else:
            sprites.append(comp)

    moves = []
    for sprite in sprites:
        coords = sprite["coords"]
        counts = Counter(int(g[r, c]) for r, c in coords)
        marker_colors = [color for color, count in counts.items() if count == 1]
        body_colors = [color for color, count in counts.items() if count > 1]
        assert len(counts) == 2, f"sprite must have two colors, got {counts}"
        assert len(marker_colors) == 1, f"sprite must have one embedded marker, got {counts}"
        assert len(body_colors) == 1, f"sprite must have one body color, got {counts}"

        body_color = body_colors[0]
        body_coords = np.array(
            [(int(r), int(c)) for r, c in coords if int(g[r, c]) == body_color],
            dtype=int,
        )
        anchor = _l_bend(body_coords)

        marker_color = marker_colors[0]
        marker = next(
            (int(r), int(c)) for r, c in coords if int(g[r, c]) == marker_color
        )
        controls = controls_by_color.get(marker_color, [])
        if not controls:
            continue
        assert len(controls) == 1, f"marker color {marker_color} must have one control"
        target = controls[0]
        _outer_diagonal_distance(anchor, marker, target)

        dr = target[0] - anchor[0]
        dc = target[1] - anchor[1]

        translated = []
        for r, c in coords:
            nr = int(r) + dr
            nc = int(c) + dc
            assert 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1], "move must stay in bounds"
            translated.append((nr, nc, int(g[r, c])))
        moves.append((coords, target, translated))

    assert moves, "expected at least one marker-controlled move"

    out = g.copy()
    cleared = set()
    for coords, target, _translated in moves:
        for r, c in coords:
            cleared.add((int(r), int(c)))
        cleared.add(target)
    for r, c in cleared:
        out[r, c] = bg

    occupied = set()
    for _coords, _target, translated in moves:
        for r, c, _color in translated:
            assert out[r, c] == bg, "translated sprite must land on cleared/background cells"
            assert (r, c) not in occupied, "translated sprites must not overlap"
            occupied.add((r, c))
        for r, c, color in translated:
            out[r, c] = color

    return out

# --------------- generator (attempt 2) ---------------

# Canvas and palette.
BG_COLOR = 0
GRID_ROWS = 12
GRID_COL_CHOICES = [12, 13]
COLORS = list(range(1, 10))

# L-sprite geometry.
LEG_LEN_MIN = 2
LEG_LEN_MAX = 4
MIN_BODY_PIXELS = 4
ORIENTATION_COUNT = 4
LEG_PAIRS = [
    (v_len, h_len)
    for v_len in range(LEG_LEN_MIN, LEG_LEN_MAX + 1)
    for h_len in range(LEG_LEN_MIN, LEG_LEN_MAX + 1)
    if v_len + h_len - 1 >= MIN_BODY_PIXELS
]

# Sprite/control counts and rates.
SPRITE_COUNT_MIN = 2
SPRITE_COUNT_MAX = 4
MIN_MOVED_SPRITES = 1
MAX_MOVED_SPRITES = 2
MAX_STATIONARY_SPRITES = 2
DISTRACTOR_PROBABILITY = 0.5

# Placement constraints and retry limits.
PLACEMENT_PADDING = 1
PLACEMENT_PADDING_CONNECTIVITY = 4
MIN_DIAGONAL_DISTANCE = 1
GENERATION_RETRIES = 300


def _make_l_sprite(body_color: int, marker_color: int) -> tuple[np.ndarray, tuple[int, int]]:
    v_len, h_len = random.choice(LEG_PAIRS)
    orient = random.randrange(ORIENTATION_COUNT)
    patch = np.zeros((v_len, h_len), dtype=int)

    bend_r = 0 if orient in (0, 1) else v_len - 1
    bend_c = 0 if orient in (0, 2) else h_len - 1
    patch[bend_r, :] = body_color
    patch[:, bend_c] = body_color

    marker_r = bend_r + (1 if bend_r == 0 else -1)
    marker_c = bend_c + (1 if bend_c == 0 else -1)
    patch[marker_r, marker_c] = marker_color
    return patch, (bend_r, bend_c)


def _marker_offset(
    patch: np.ndarray, bend: tuple[int, int], marker_color: int
) -> tuple[int, int]:
    marker_cells = np.argwhere(patch == marker_color)
    assert len(marker_cells) == 1
    mr, mc = map(int, marker_cells[0])
    return mr - bend[0], mc - bend[1]


def _erase_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    mask = patch != BG_COLOR
    grid[r0 : r0 + h, c0 : c0 + w][mask] = BG_COLOR


def _blocked_cells(grid: np.ndarray) -> np.ndarray:
    """Return cells where a foreground patch pixel would violate the gap rule."""
    blocked = grid != BG_COLOR
    for _ in range(PLACEMENT_PADDING):
        expanded = blocked.copy()
        expanded[1:, :] |= blocked[:-1, :]
        expanded[:-1, :] |= blocked[1:, :]
        expanded[:, 1:] |= blocked[:, :-1]
        expanded[:, :-1] |= blocked[:, 1:]
        if PLACEMENT_PADDING_CONNECTIVITY == 8:
            expanded[1:, 1:] |= blocked[:-1, :-1]
            expanded[1:, :-1] |= blocked[:-1, 1:]
            expanded[:-1, 1:] |= blocked[1:, :-1]
            expanded[:-1, :-1] |= blocked[1:, 1:]
        elif PLACEMENT_PADDING_CONNECTIVITY != 4:
            raise ValueError("placement padding connectivity must be 4 or 8")
        blocked = expanded
    return blocked


def _patch_fits(blocked: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> bool:
    rows, cols = blocked.shape
    h, w = patch.shape
    if r0 < 0 or c0 < 0 or r0 + h > rows or c0 + w > cols:
        return False
    patch_mask = patch != BG_COLOR
    return not bool((blocked[r0 : r0 + h, c0 : c0 + w] & patch_mask).any())


def _place_component(grid: np.ndarray, patch: np.ndarray) -> tuple[int, int] | None:
    try:
        r, c = random_free_location(
            grid,
            patch,
            transparent=BG_COLOR,
            padding=PLACEMENT_PADDING,
            padding_connectivity=PLACEMENT_PADDING_CONNECTIVITY,
        )
    except ValueError:
        return None
    paste(grid, patch, r, c, transparent=BG_COLOR)
    return r, c


def _choose_moved_indices(n_sprites: int) -> set[int]:
    min_moves = max(MIN_MOVED_SPRITES, n_sprites - MAX_STATIONARY_SPRITES)
    max_moves = min(MAX_MOVED_SPRITES, n_sprites)
    n_moves = random.randint(min_moves, max_moves)
    return set(random.sample(range(n_sprites), n_moves))


def _choose_marker_colors(body_colors: list[int]) -> list[int]:
    marker_colors = []
    for body in body_colors:
        choices = [c for c in COLORS if c != body and c not in marker_colors]
        marker_colors.append(random.choice(choices))
    return marker_colors


def _build_sprites(n_sprites: int) -> tuple[list[dict], list[int]]:
    body_colors = random.sample(COLORS, n_sprites)
    marker_colors = _choose_marker_colors(body_colors)
    sprites = []
    for body, marker in zip(body_colors, marker_colors):
        patch, bend = _make_l_sprite(body, marker)
        sprites.append(
            {
                "patch": patch,
                "bend": bend,
                "marker": marker,
                "marker_offset": _marker_offset(patch, bend, marker),
            }
        )
    return sprites, marker_colors


def _place_output_layout(
    rows: int, cols: int, sprites: list[dict], marker_colors: list[int]
) -> np.ndarray | None:
    """Place final sprites and annotate each with output location and target bend."""
    out = np.zeros((rows, cols), dtype=int)
    for idx in random.sample(range(len(sprites)), len(sprites)):
        loc = _place_component(out, sprites[idx]["patch"])
        if loc is None:
            return None
        r, c = loc
        br, bc = sprites[idx]["bend"]
        sprites[idx]["out_loc"] = loc
        sprites[idx]["target"] = (r + br, c + bc)

    if random.random() < DISTRACTOR_PROBABILITY:
        distractor_choices = [c for c in COLORS if c not in marker_colors]
        distractor = np.array([[random.choice(distractor_choices)]], dtype=int)
        if _place_component(out, distractor) is None:
            return None
    return out


def _place_aligned_input_sprite(grid: np.ndarray, sprite: dict) -> tuple[int, int] | None:
    patch = sprite["patch"]
    bend = sprite["bend"]
    marker_dr, marker_dc = sprite["marker_offset"]
    target_r, target_c = sprite["target"]
    rows, cols = grid.shape
    blocked = _blocked_cells(grid)

    candidates = []
    for dist in range(MIN_DIAGONAL_DISTANCE, max(rows, cols) + 1):
        bend_r = target_r + marker_dr * dist
        bend_c = target_c + marker_dc * dist
        r = bend_r - bend[0]
        c = bend_c - bend[1]
        if _patch_fits(blocked, patch, r, c):
            candidates.append((r, c))

    if not candidates:
        return None

    r, c = random.choice(candidates)
    paste(grid, patch, r, c, transparent=BG_COLOR)
    return r, c


def _derive_input_grid(out: np.ndarray, sprites: list[dict], moved: set[int]) -> np.ndarray | None:
    """Reverse selected output placements into input sprites plus control pixels."""
    inp = out.copy()
    for idx in moved:
        r, c = sprites[idx]["out_loc"]
        _erase_patch(inp, sprites[idx]["patch"], r, c)

    for idx in moved:
        r, c = sprites[idx]["target"]
        if inp[r, c] != BG_COLOR:
            return None
        inp[r, c] = sprites[idx]["marker"]

    for idx in random.sample(list(moved), len(moved)):
        if _place_aligned_input_sprite(inp, sprites[idx]) is None:
            return None
    return inp


def _oracle_checked_input(inp: np.ndarray, out: np.ndarray) -> np.ndarray | None:
    try:
        actual = main(inp)
    except AssertionError:
        return None
    return inp if np.array_equal(actual, out) else None


def _gen_attempt() -> np.ndarray | None:
    rows, cols = GRID_ROWS, random.choice(GRID_COL_CHOICES)
    n_sprites = random.randint(SPRITE_COUNT_MIN, SPRITE_COUNT_MAX)
    moved = _choose_moved_indices(n_sprites)
    sprites, marker_colors = _build_sprites(n_sprites)

    out = _place_output_layout(rows, cols, sprites, marker_colors)
    if out is None:
        return None

    inp = _derive_input_grid(out, sprites, moved)
    if inp is None:
        return None
    return _oracle_checked_input(inp, out)


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
