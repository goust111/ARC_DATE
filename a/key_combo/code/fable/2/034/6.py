import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, find_components, paste


SPRITE_SIZE = 3
HOLLOW_MASK = np.array(
    [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
    ],
    dtype=bool,
)
RING_REL = tuple(tuple(map(int, rc)) for rc in np.argwhere(HOLLOW_MASK))
DIR_BY_COLOR = {
    int(Color.GREEN): (0, -1),
    int(Color.YELLOW): (1, 0),
    int(Color.PINK): (-1, 0),
    int(Color.TEAL): (0, 1),
}
COLOR_BY_DIR = {direction: color for color, direction in DIR_BY_COLOR.items()}
OPPOSITE_DIR = {
    (0, -1): (0, 1),
    (0, 1): (0, -1),
    (-1, 0): (1, 0),
    (1, 0): (-1, 0),
}
ALL_DIRS = tuple(COLOR_BY_DIR)
SIDE_MARKER_OFFSETS = ((-1, 1), (3, 1), (1, -1), (1, 3))
PREFERRED_MARKER_OFFSET = {
    (0, -1): (1, 3),
    (0, 1): (1, -1),
    (-1, 0): (3, 1),
    (1, 0): (-1, 1),
}
GEN_RETRIES = 400
CHAIN_RETRIES = 200
DISTRACTOR_RETRIES = 250
MIN_GRID_SIZE = 16
MAX_GRID_SIZE = 30
MAX_GAP = 6

_GEN_CALL_COUNT = 0


def _sprite_patch(color: int) -> np.ndarray:
    patch = np.zeros((SPRITE_SIZE, SPRITE_SIZE), dtype=int)
    patch[HOLLOW_MASK] = int(color)
    return patch


def _bbox_from_pos(pos: tuple[int, int]) -> tuple[int, int, int, int]:
    r, c = map(int, pos)
    return r, c, r + SPRITE_SIZE - 1, c + SPRITE_SIZE - 1


def _bboxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _rects_too_close(a: tuple[int, int], b: tuple[int, int], margin: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = _bbox_from_pos(a)
    br0, bc0, br1, bc1 = _bbox_from_pos(b)
    return not (ar1 + margin < br0 or br1 + margin < ar0 or ac1 + margin < bc0 or bc1 + margin < ac0)


def _top_left_in_bounds(shape: tuple[int, int], pos: tuple[int, int]) -> bool:
    h, w = map(int, shape)
    r, c = map(int, pos)
    return 0 <= r <= h - SPRITE_SIZE and 0 <= c <= w - SPRITE_SIZE


def _occupied_cells(pos: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = map(int, pos)
    return {(r + dr, c + dc) for dr, dc in RING_REL}


def _adjacent_sprite_indices(positions: list[tuple[int, int]], marker: tuple[int, int]) -> list[int]:
    mr, mc = map(int, marker)
    neighbors = {(mr - 1, mc), (mr + 1, mc), (mr, mc - 1), (mr, mc + 1)}
    adjacent = []
    for idx, pos in enumerate(positions):
        if _occupied_cells(pos) & neighbors:
            adjacent.append(idx)
    return adjacent


def _start_index(positions: list[tuple[int, int]], marker: tuple[int, int]) -> int:
    adjacent = _adjacent_sprite_indices(positions, marker)
    assert len(adjacent) == 1, "grey trigger must touch exactly one sprite"
    return int(adjacent[0])


def _slide_to_edge(shape: tuple[int, int], pos: tuple[int, int], color: int) -> tuple[int, int]:
    dr, dc = DIR_BY_COLOR[int(color)]
    r, c = map(int, pos)
    while _top_left_in_bounds(shape, (r + dr, c + dc)):
        r += dr
        c += dc
    return r, c


def _candidate_hit_ids(
    candidate_pos: tuple[int, int],
    positions: list[tuple[int, int]],
    current_idx: int,
) -> list[int]:
    candidate_bbox = _bbox_from_pos(candidate_pos)
    candidate_cells = None
    hits = []
    for idx, pos in enumerate(positions):
        if idx == current_idx:
            continue
        if not _bboxes_overlap(candidate_bbox, _bbox_from_pos(pos)):
            continue
        if candidate_cells is None:
            candidate_cells = _occupied_cells(candidate_pos)
        if candidate_cells & _occupied_cells(pos):
            hits.append(idx)
    return hits


def _simulate_positions(
    shape: tuple[int, int],
    colors: list[int],
    positions: list[tuple[int, int]],
    marker: tuple[int, int],
) -> tuple[list[tuple[int, int]], list[int], list[tuple[int, int]], int]:
    current_positions = [tuple(map(int, pos)) for pos in positions]
    start = _start_index(current_positions, marker)
    chain = [start]
    collisions: list[tuple[int, int]] = []
    triggered = {start}
    current = start

    while True:
        dr, dc = DIR_BY_COLOR[int(colors[current])]
        hit = None
        while True:
            next_pos = (current_positions[current][0] + dr, current_positions[current][1] + dc)
            if not _top_left_in_bounds(shape, next_pos):
                break
            hits = _candidate_hit_ids(next_pos, current_positions, current)
            if hits:
                assert len(hits) == 1, "a moving sprite must hit a unique next sprite"
                hit = int(hits[0])
                assert hit not in triggered, "a sprite may not be triggered twice"
                break
            current_positions[current] = next_pos

        if hit is None:
            break

        collisions.append((current, hit))
        triggered.add(hit)
        chain.append(hit)
        current = hit

    return current_positions, chain, collisions, start


def _render_grid(
    shape: tuple[int, int],
    colors: list[int],
    positions: list[tuple[int, int]],
    marker: tuple[int, int],
) -> np.ndarray:
    out = np.full(shape, int(Color.BLACK), dtype=int)
    for color, pos in zip(colors, positions):
        paste(out, _sprite_patch(int(color)), int(pos[0]), int(pos[1]), transparent=int(Color.BLACK))
    mr, mc = map(int, marker)
    assert out[mr, mc] == int(Color.BLACK), "grey marker must remain visible in the output"
    out[mr, mc] = int(Color.GREY)
    return out


def _parse_grid(grid: np.ndarray) -> tuple[np.ndarray, list[int], list[tuple[int, int]], tuple[int, int], int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    allowed = {int(Color.BLACK), int(Color.GREEN), int(Color.YELLOW), int(Color.GREY), int(Color.PINK), int(Color.TEAL)}
    assert set(map(int, np.unique(g))) <= allowed, "unexpected color present"

    marker_coords = np.argwhere(g == int(Color.GREY))
    assert marker_coords.shape == (1, 2), "expected exactly one grey trigger marker"
    marker = tuple(map(int, marker_coords[0]))

    components = find_components(
        g,
        bg=int(Color.BLACK),
        connectivity=4,
        monochromatic=True,
        colors=[int(Color.GREEN), int(Color.YELLOW), int(Color.PINK), int(Color.TEAL)],
        return_format="dict",
    )
    assert len(components) >= 2, "expected several hollow sprites"

    colors: list[int] = []
    positions: list[tuple[int, int]] = []
    for component in components:
        color = int(component["color"])
        coords = component["coords"]
        bbox = tuple(map(int, component["bbox"]))
        r0, c0, r1, c1 = bbox
        assert (r1 - r0 + 1, c1 - c0 + 1) == (SPRITE_SIZE, SPRITE_SIZE), "sprite bbox must be 3x3"
        assert coords.shape[0] == int(HOLLOW_MASK.sum()), "sprite must have hollow 3x3 area"
        rel = coords - np.array([r0, c0], dtype=int)
        mask = np.zeros((SPRITE_SIZE, SPRITE_SIZE), dtype=bool)
        mask[rel[:, 0], rel[:, 1]] = True
        assert np.array_equal(mask, HOLLOW_MASK), "every sprite must be a hollow 3x3 square"
        colors.append(color)
        positions.append((r0, c0))

    start = _start_index(positions, marker)
    return g, colors, positions, marker, start


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide only the grey-triggered hollow sprite chain, handing motion from each collision to the next sprite.

    Input structure:
    - The background is black.
    - The foreground contains several separated 3x3 hollow square sprites colored by direction code: green=left, yellow=down, pink=up, teal=right.
    - Exactly one grey pixel is orthogonally adjacent to exactly one sprite and marks the unique starting trigger.

    Rule:
    - Identify the sprite touched by the grey pixel and trigger only that sprite first.
    - A triggered sprite slides cell by cell in the direction encoded by its own color; if the next step would overlap another sprite, it stays in the last free position flush against that sprite and transfers the trigger to the struck sprite.
    - Each sprite can be triggered at most once; untriggered sprites never move; if a triggered sprite reaches the grid edge without hitting another sprite, the chain ends. Keep the grey pixel in place.

    Key insight:
    - Color only tells each sprite which way it would move; the actual set of movers is determined by the collision chain that starts from the lone grey trigger marker.
    """
    g, colors, positions, marker, _start = _parse_grid(grid)
    final_positions, _chain, _collisions, _ = _simulate_positions(g.shape, colors, positions, marker)
    return _render_grid(g.shape, colors, final_positions, marker)


def _pick_chain_length(branch: int, rng: np.random.Generator) -> int:
    if branch == 0:
        return 1
    if branch == 1:
        return int(rng.integers(2, 4))
    if branch == 2:
        return int(rng.integers(3, 5))
    return int(rng.integers(2, 5))


def _pick_chain_directions(chain_len: int, branch: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    directions = [ALL_DIRS[int(rng.integers(len(ALL_DIRS)))]]
    need_turn = chain_len >= 3 and branch in {2, 3}

    for i in range(1, chain_len):
        prev = directions[-1]
        choices = [direction for direction in ALL_DIRS if direction != OPPOSITE_DIR[prev]]
        if need_turn:
            turn_choices = [direction for direction in choices if direction != prev]
            if turn_choices and (i == chain_len - 1 or rng.random() < 0.7):
                choices = turn_choices
        directions.append(choices[int(rng.integers(len(choices)))])

    return directions


def _place_chain_positions(
    shape: tuple[int, int],
    directions: list[tuple[int, int]],
    rng: np.random.Generator,
) -> list[tuple[int, int]] | None:
    h, w = map(int, shape)
    for _ in range(CHAIN_RETRIES):
        positions = [(int(rng.integers(1, h - SPRITE_SIZE)), int(rng.integers(1, w - SPRITE_SIZE)))]
        ok = True
        for direction in directions[:-1]:
            gap_choices = np.arange(1, MAX_GAP + 1, dtype=int)
            rng.shuffle(gap_choices)
            placed = False
            for gap in gap_choices.tolist():
                step = SPRITE_SIZE + int(gap)
                candidate = (
                    positions[-1][0] + direction[0] * step,
                    positions[-1][1] + direction[1] * step,
                )
                if not _top_left_in_bounds(shape, candidate):
                    continue
                if any(_rects_too_close(candidate, pos, margin=1) for pos in positions):
                    continue
                positions.append(candidate)
                placed = True
                break
            if not placed:
                ok = False
                break
        if not ok:
            continue
        last_color = COLOR_BY_DIR[directions[-1]]
        if _slide_to_edge(shape, positions[-1], last_color) == positions[-1]:
            continue
        return positions
    return None


def _pick_distractor_colors(chain_colors: list[int], rng: np.random.Generator) -> list[int]:
    missing = [color for color in DIR_BY_COLOR if color not in set(chain_colors)]
    count = max(1, len(missing))
    colors = list(missing)
    palette = list(DIR_BY_COLOR)
    while len(colors) < count + int(rng.integers(0, 2)):
        colors.append(int(palette[int(rng.integers(len(palette)))]))
    rng.shuffle(colors)
    return colors


def _place_distractors(
    shape: tuple[int, int],
    occupied_positions: list[tuple[int, int]],
    distractor_colors: list[int],
    rng: np.random.Generator,
) -> list[tuple[int, int]] | None:
    positions = list(occupied_positions)
    placed: list[tuple[int, int]] = []
    h, w = map(int, shape)
    for color in distractor_colors:
        success = False
        for _ in range(DISTRACTOR_RETRIES):
            candidate = (int(rng.integers(0, h - SPRITE_SIZE + 1)), int(rng.integers(0, w - SPRITE_SIZE + 1)))
            if any(_rects_too_close(candidate, pos, margin=1) for pos in positions):
                continue
            if _slide_to_edge(shape, candidate, color) == candidate:
                continue
            positions.append(candidate)
            placed.append(candidate)
            success = True
            break
        if not success:
            return None
    return placed


def _pick_marker(
    shape: tuple[int, int],
    start_pos: tuple[int, int],
    start_dir: tuple[int, int],
    positions: list[tuple[int, int]],
    rng: np.random.Generator,
) -> tuple[int, int] | None:
    preferred = PREFERRED_MARKER_OFFSET[start_dir]
    others = [offset for offset in SIDE_MARKER_OFFSETS if offset != preferred]
    rng.shuffle(others)
    for dr, dc in [preferred, *others]:
        marker = (start_pos[0] + dr, start_pos[1] + dc)
        if not (0 <= marker[0] < shape[0] and 0 <= marker[1] < shape[1]):
            continue
        if any(marker in _occupied_cells(pos) for pos in positions):
            continue
        if _adjacent_sprite_indices(positions, marker) == [0]:
            return marker
    return None


def _validate_generated_grid(grid: np.ndarray, expected_chain_len: int) -> None:
    parsed, colors, positions, marker, start = _parse_grid(grid)
    initial_positions = list(positions)
    final_positions, chain, collisions, parsed_start = _simulate_positions(parsed.shape, colors, positions, marker)
    assert parsed_start == start
    assert chain[0] == start
    assert len(chain) == expected_chain_len
    assert len(collisions) == expected_chain_len - 1
    assert len(chain) < len(colors), "there must be at least one untouched distractor"
    assert len(set(colors)) == 4, "all four direction colors should appear"
    assert any(final_positions[idx] != initial_positions[idx] for idx in chain)

    edge_only_positions = [_slide_to_edge(parsed.shape, pos, color) for pos, color in zip(initial_positions, colors)]
    assert any(edge_only_positions[idx] != final_positions[idx] for idx in range(len(colors)))

    last_triggered = chain[-1]
    for idx in range(len(colors)):
        if idx in chain:
            if idx == last_triggered:
                assert final_positions[idx] == edge_only_positions[idx], "the last triggered sprite must slide to the edge"
            else:
                assert final_positions[idx] != edge_only_positions[idx], "earlier triggered sprites must stop on collisions"
        else:
            assert final_positions[idx] == initial_positions[idx], "untouched distractors must stay put"
            assert edge_only_positions[idx] != initial_positions[idx], "distractors should expose the all-move shortcut as wrong"

    if expected_chain_len > 1:
        assert final_positions[start] != edge_only_positions[start], "the starting sprite must stop on its first collision"

    expected = _render_grid(parsed.shape, colors, final_positions, marker)
    actual = main(parsed)
    assert np.array_equal(actual, expected)
    assert actual[marker[0], marker[1]] == int(Color.GREY)
    assert not np.array_equal(actual, parsed), "the puzzle should not degenerate into a no-op"


def _gen_attempt(branch: int, rng: np.random.Generator) -> np.ndarray | None:
    chain_len = _pick_chain_length(branch, rng)
    directions = _pick_chain_directions(chain_len, branch, rng)
    chain_colors = [COLOR_BY_DIR[direction] for direction in directions]
    distractor_colors = _pick_distractor_colors(chain_colors, rng)

    min_size = max(MIN_GRID_SIZE, 11 + 2 * chain_len + len(distractor_colors))
    shape = (
        int(rng.integers(min_size, MAX_GRID_SIZE + 1)),
        int(rng.integers(min_size, MAX_GRID_SIZE + 1)),
    )

    chain_positions = _place_chain_positions(shape, directions, rng)
    if chain_positions is None:
        return None

    distractor_positions = _place_distractors(shape, chain_positions, distractor_colors, rng)
    if distractor_positions is None:
        return None

    all_colors = chain_colors + distractor_colors
    all_positions = chain_positions + distractor_positions

    marker = _pick_marker(shape, chain_positions[0], directions[0], all_positions, rng)
    if marker is None:
        return None

    try:
        final_positions, chain, collisions, start = _simulate_positions(shape, all_colors, all_positions, marker)
    except AssertionError:
        return None
    if start != 0:
        return None
    if chain != list(range(chain_len)):
        return None
    if len(collisions) != chain_len - 1:
        return None
    if any(final_positions[idx] == all_positions[idx] for idx in chain):
        return None
    if any(final_positions[idx] != all_positions[idx] for idx in range(chain_len, len(all_positions))):
        return None

    try:
        grid = _render_grid(shape, all_colors, all_positions, marker)
        _validate_generated_grid(grid, expected_chain_len=chain_len)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT

    branch = _GEN_CALL_COUNT % 4
    _GEN_CALL_COUNT += 1
    rng = np.random.default_rng()

    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(branch, rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid collision-chain input")
