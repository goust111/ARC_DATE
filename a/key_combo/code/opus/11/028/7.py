import random
from itertools import combinations
from collections import deque

import numpy as np


GRID_SIZE = 6
SEED_SIZE = 3
BACKGROUND = 0
SEED_COLOR = 5
SPRITE_COLORS = (1, 2, 3)
GENERATION_ATTEMPTS = 400


BASE_SHAPES = (
    ((0, 0), (0, 1)),                  # horizontal domino
    ((0, 0), (1, 0)),                  # vertical domino
    ((0, 0), (0, 1), (0, 2)),          # horizontal triomino
    ((0, 0), (1, 0), (2, 0)),          # vertical triomino
    ((0, 0), (1, 0), (1, 1)),          # L triomino
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
)

MODE_WEIGHTS = (
    ("independent2", 18),
    ("chain2", 28),
    ("independent3", 16),
    ("mixed3", 24),
    ("chain3", 14),
)


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _seed_patterns() -> list[np.ndarray]:
    patterns: list[np.ndarray] = []
    for bits in range(1 << (SEED_SIZE * SEED_SIZE)):
        mask = np.zeros((SEED_SIZE, SEED_SIZE), dtype=bool)
        for idx in range(SEED_SIZE * SEED_SIZE):
            if bits & (1 << idx):
                mask[idx // SEED_SIZE, idx % SEED_SIZE] = True
        grey_count = int(mask.sum())
        if not 3 <= grey_count <= 6:
            continue
        if not mask.any(axis=0).all() or not mask.any(axis=1).all():
            continue
        if not (~mask).any(axis=0).all() or not (~mask).any(axis=1).all():
            continue
        grid = np.zeros((SEED_SIZE, SEED_SIZE), dtype=int)
        grid[mask] = SEED_COLOR
        patterns.append(grid)
    return patterns


def _sprite_candidates() -> list[tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]]:
    seed_zone = {(r, c) for r in range(SEED_SIZE) for c in range(SEED_SIZE)}
    candidates = []
    seen = set()
    for base in BASE_SHAPES:
        max_r = max(r for r, _ in base)
        max_c = max(c for _, c in base)
        for r0 in range(GRID_SIZE - max_r):
            for c0 in range(GRID_SIZE - max_c):
                coords = tuple(sorted((r0 + r, c0 + c) for r, c in base))
                if any(cell in seed_zone for cell in coords):
                    continue
                if coords in seen:
                    continue
                seen.add(coords)
                candidates.append((base, coords))
    return candidates


SEED_PATTERNS = _seed_patterns()
SPRITE_CANDIDATES = _sprite_candidates()


def _invert_and_tile_seed(seed: np.ndarray) -> np.ndarray:
    swapped = np.where(seed == SEED_COLOR, BACKGROUND, SEED_COLOR).astype(int)
    return np.tile(swapped, (2, 2))


def _coords_set(coords: np.ndarray | tuple[tuple[int, int], ...]) -> set[tuple[int, int]]:
    if isinstance(coords, np.ndarray):
        return {tuple(map(int, cell)) for cell in coords}
    return set(coords)


def _sprite_occupancy(sprites: list[dict]) -> set[tuple[int, int]]:
    occ: set[tuple[int, int]] = set()
    for sprite in sprites:
        occ.update(_coords_set(sprite["coords"]))
    return occ


def _can_shift_right(coords: np.ndarray, occ: set[tuple[int, int]]) -> bool:
    own = _coords_set(coords)
    for r, c in own:
        nc = c + 1
        if nc >= GRID_SIZE:
            return False
        if (r, nc) in occ and (r, nc) not in own:
            return False
    return True


def _slide_right_terminal(sprites: list[dict]) -> list[dict]:
    moved = [{"color": int(sp["color"]), "coords": np.array(sp["coords"], dtype=int).copy()} for sp in sprites]
    for _ in range(GRID_SIZE):
        occ = _sprite_occupancy(moved)
        movable = [idx for idx, sp in enumerate(moved) if _can_shift_right(sp["coords"], occ)]
        if not movable:
            break
        for idx in movable:
            moved[idx]["coords"][:, 1] += 1
    return moved


def _sprites_from_grid(grid: np.ndarray) -> list[dict]:
    work = np.asarray(grid, dtype=int).copy()
    work[:SEED_SIZE, :SEED_SIZE] = BACKGROUND
    sprites: list[dict] = []
    for color in SPRITE_COLORS:
        mask = work == color
        for coords in _components_mask(mask):
            sprites.append({"color": color, "coords": coords})
    sprites.sort(key=lambda sp: (sp["color"], tuple(map(tuple, sp["coords"]))))
    return sprites


def _paint_sprites(canvas: np.ndarray, sprites: list[dict]) -> np.ndarray:
    out = np.array(canvas, dtype=int, copy=True)
    for sprite in sprites:
        out[sprite["coords"][:, 0], sprite["coords"][:, 1]] = int(sprite["color"])
    return out


def _row_overlap_graph(sprites: list[dict]) -> list[list[bool]]:
    n = len(sprites)
    rows = [{int(r) for r in sp["coords"][:, 0]} for sp in sprites]
    graph = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if rows[i] & rows[j]:
                graph[i][j] = graph[j][i] = True
    return graph


def _classify_settled_mode(input_sprites: list[dict], settled_sprites: list[dict]) -> str | None:
    if len(input_sprites) != len(settled_sprites):
        return None
    if not all(not np.array_equal(inp["coords"], out["coords"]) for inp, out in zip(input_sprites, settled_sprites)):
        return None

    graph = _row_overlap_graph(settled_sprites)
    n = len(settled_sprites)
    edge_count = sum(graph[i][j] for i in range(n) for j in range(i + 1, n))

    if n == 2:
        return "chain2" if edge_count == 1 else "independent2"

    if edge_count == 0:
        return "independent3"

    seen = {0}
    stack = [0]
    while stack:
        node = stack.pop()
        for nxt in range(n):
            if graph[node][nxt] and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return "chain3" if len(seen) == n else "mixed3"


def _touches_seed(coords: tuple[tuple[int, int], ...]) -> bool:
    return any(r < SEED_SIZE and c < SEED_SIZE for r, c in coords)


def _pairwise_clear(coords_list: list[tuple[tuple[int, int], ...]], require_gap: bool) -> bool:
    for i in range(len(coords_list)):
        a = set(coords_list[i])
        for j in range(i + 1, len(coords_list)):
            b = set(coords_list[j])
            if a & b:
                return False
            if require_gap:
                for r, c in a:
                    for rr, cc in b:
                        if max(abs(r - rr), abs(c - cc)) <= 1:
                            return False
    return True


def _sample_mode() -> str:
    names = [name for name, _ in MODE_WEIGHTS]
    weights = [weight for _, weight in MODE_WEIGHTS]
    return random.choices(names, weights=weights, k=1)[0]


def _valid_layouts_by_mode() -> dict[str, list[tuple[tuple[tuple[int, int], ...], ...]]]:
    grouped = {name: [] for name, _ in MODE_WEIGHTS}
    for n in (2, 3):
        for combo in combinations(SPRITE_CANDIDATES, n):
            coords_list = tuple(coords for _, coords in combo)
            if not _pairwise_clear(list(coords_list), require_gap=True):
                continue

            sprites = [{"color": idx + 1, "coords": np.array(coords, dtype=int)} for idx, coords in enumerate(coords_list)]
            occ = _sprite_occupancy(sprites)
            if not all(_can_shift_right(sp["coords"], occ) for sp in sprites):
                continue

            settled = _slide_right_terminal(sprites)
            if any(_touches_seed(tuple(map(tuple, sp["coords"]))) for sp in settled):
                continue

            settled_coords = [tuple(sorted(map(tuple, sp["coords"]))) for sp in settled]
            if not _pairwise_clear(settled_coords, require_gap=False):
                continue

            mode = _classify_settled_mode(sprites, settled)
            if mode is not None:
                grouped[mode].append(coords_list)
    return grouped


VALID_LAYOUTS = _valid_layouts_by_mode()


def _sample_layout(mode: str) -> list[tuple[tuple[int, int], ...]] | None:
    layouts = VALID_LAYOUTS.get(mode, [])
    if not layouts:
        return None
    return list(random.choice(layouts))


def _build_input(seed: np.ndarray, coords_list: list[tuple[tuple[int, int], ...]]) -> np.ndarray:
    colors = list(SPRITE_COLORS)
    random.shuffle(colors)
    if len(coords_list) == 2:
        colors = colors[:2]
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    grid[:SEED_SIZE, :SEED_SIZE] = seed
    for color, coords in zip(colors, coords_list):
        for r, c in coords:
            grid[r, c] = color
    return grid


def _stop_reasons(settled_sprites: list[dict]) -> dict[int, str]:
    occ = _sprite_occupancy(settled_sprites)
    reasons: dict[int, str] = {}
    for idx, sprite in enumerate(settled_sprites):
        own = _coords_set(sprite["coords"])
        blocker = "border"
        for r, c in own:
            nc = c + 1
            if nc >= GRID_SIZE:
                blocker = "border"
                break
            if (r, nc) in occ and (r, nc) not in own:
                blocker = "sprite"
                break
        reasons[idx] = blocker
    return reasons


def _validate_generated_case(grid: np.ndarray, mode: str) -> bool:
    if not isinstance(grid, np.ndarray) or grid.shape != (GRID_SIZE, GRID_SIZE):
        return False

    seed = grid[:SEED_SIZE, :SEED_SIZE]
    if not set(np.unique(seed)).issubset({BACKGROUND, SEED_COLOR}):
        return False
    seed_mask = seed == SEED_COLOR
    if seed_mask.sum() < 3:
        return False
    if not seed_mask.any(axis=0).all() or not seed_mask.any(axis=1).all():
        return False
    if not (~seed_mask).any(axis=0).all() or not (~seed_mask).any(axis=1).all():
        return False

    sprites = _sprites_from_grid(grid)
    if len(sprites) not in (2, 3):
        return False
    colors = [sp["color"] for sp in sprites]
    if len(colors) != len(set(colors)):
        return False
    if any(sp["coords"].shape[0] not in (2, 3) for sp in sprites):
        return False

    coords_list = [tuple(sorted(map(tuple, sp["coords"]))) for sp in sprites]
    if not _pairwise_clear(coords_list, require_gap=True):
        return False

    occ = _sprite_occupancy(sprites)
    if not all(_can_shift_right(sp["coords"], occ) for sp in sprites):
        return False

    settled = _slide_right_terminal(sprites)
    settled_mode = _classify_settled_mode(sprites, settled)
    if settled_mode != mode:
        return False

    reasons = _stop_reasons(settled)
    if mode in {"chain2", "mixed3", "chain3"} and "sprite" not in reasons.values():
        return False
    if mode in {"independent2", "independent3"} and "sprite" in reasons.values():
        return False

    background = _invert_and_tile_seed(seed)
    visible_mask = np.ones((GRID_SIZE, GRID_SIZE), dtype=bool)
    for sprite in settled:
        visible_mask[sprite["coords"][:, 0], sprite["coords"][:, 1]] = False
    visible_colors = set(map(int, np.unique(background[visible_mask])))
    if visible_colors != {BACKGROUND, SEED_COLOR}:
        return False

    expected = _paint_sprites(background, settled)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False

    if np.array_equal(actual, background):
        return False
    if np.array_equal(actual[:SEED_SIZE, :SEED_SIZE], seed):
        return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Invert the 3×3 seed into a tiled background, then overlay the terminal right-slid sprite positions.

    Input structure:
    - The grid is 6×6 with black as background.
    - The top-left 3×3 block is a binary seed made only of black and grey cells.
    - Outside that seed block are 2 or 3 small monochrome sprites in blue, red, or green.

    Rule:
    - Read the top-left 3×3 block as the seed, swapping grey with black inside that block.
    - Tile the swapped 3×3 seed in a 2×2 arrangement to form a 6×6 background layer.
    - Treat every non-seed blue/red/green connected component as a rigid sprite and slide all sprites right simultaneously until each is blocked by the right border or a settled sprite.
    - Overlay those terminal sprite positions on top of the tiled background.

    Key insight:
    - The seed and the sprites encode two independent transformations on the same input: the seed disappears into a complemented tiled background, while the colored sprites only matter through their final east-gravity resting state.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (GRID_SIZE, GRID_SIZE), "expected a 6x6 grid"

    seed = g[:SEED_SIZE, :SEED_SIZE]
    seed_colors = set(map(int, np.unique(seed)))
    assert seed_colors.issubset({BACKGROUND, SEED_COLOR}), "seed must be binary black/grey"

    background = _invert_and_tile_seed(seed)
    sprites = _sprites_from_grid(g)
    settled = _slide_right_terminal(sprites)
    return _paint_sprites(background, settled)


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        mode = _sample_mode()
        coords_list = _sample_layout(mode)
        if coords_list is None:
            continue
        seed = random.choice(SEED_PATTERNS)
        grid = _build_input(seed, coords_list)
        if _validate_generated_case(grid, mode):
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
