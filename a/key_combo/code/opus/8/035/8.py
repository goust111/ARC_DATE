from __future__ import annotations

import hashlib
import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4


_SEED_ROWS = (
    (".###", "#.#.", "####", "#.#."),
    ("####", "#.#.", "####", "#.#."),
    (".###", "#.#.", "####", "#..#"),
    ("####", "#.#.", "####", "#..#"),
    (".###", "#.#.", "###.", "#.##"),
    ("####", "#.#.", "###.", "#.##"),
    (".###", "#.#.", "####", "#.##"),
    ("####", "#.#.", "####", "#.##"),
    ("....#", ".####", "###.#", "..###", "..#.."),
    ("..#..", "..#..", ".###.", ".#.##", "####."),
    ("..#..", "..#..", ".####", "##.##", ".###."),
    ("..##.", "..###", "..#.#", "#.###", "###.."),
    (".#...", ".####", "##.#.", ".###.", "..#.."),
    ("..#..", ".###.", "##.##", ".###.", "...#."),
    ("...#.", "...#.", "#####", ".#.#.", "####."),
    ("..#..", "..#..", "####.", ".#.##", ".###."),
    ("...#.", ".#.#.", "####.", ".#.##", ".###."),
    ("..##.", "..###", "..#.#", "#####", "..##."),
    ("...#.", ".###.", "##.##", ".###.", ".#..."),
    ("..#..", "#####", ".#.#.", "####.", "#...."),
    (".###.", ".#...", "#####", ".#.#.", ".###."),
    ("#..#.", "#####", ".##.#", "..###", "...#."),
    ("..#..", "..#..", ".####", ".#.#.", "####."),
)


_FAMILY_ORDER = ("yellow_only", "yellow_red", "red_only")
_FAMILY_WEIGHTS = (0.35, 0.4, 0.25)
_CANDIDATE_POOLS: dict[str, list[dict[str, object]]] | None = None
_USED_OUTPUTS: set[str] = set()
_GEN_CALLS = 0


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _component_coords(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise AssertionError("expected a non-empty mask")
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_to_bbox(grid: np.ndarray, mask: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(mask)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _signature(arr: np.ndarray) -> str:
    return hashlib.md5(arr.tobytes() + str(arr.shape).encode()).hexdigest()


def _is_connected(mask: np.ndarray) -> bool:
    comps = _component_coords(mask)
    return len(comps) == 1


def _zero_regions(grid: np.ndarray) -> list[tuple[list[tuple[int, int]], bool]]:
    h, w = grid.shape
    zeros = grid == BLACK
    seen = np.zeros((h, w), dtype=bool)
    regions: list[tuple[list[tuple[int, int]], bool]] = []
    for r in range(h):
        for c in range(w):
            if not zeros[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            touches_border = False
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                if rr == 0 or rr == h - 1 or cc == 0 or cc == w - 1:
                    touches_border = True
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and zeros[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            regions.append((cells, touches_border))
    return regions


def _enclosed_zero_regions(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    return [cells for cells, touches_border in _zero_regions(grid) if not touches_border]


def _compose_sprite(sprite: np.ndarray) -> np.ndarray:
    n, m = sprite.shape
    if n != m:
        raise AssertionError("the cropped sprite must be square for quarter-turn quadrant tiling")
    return np.block(
        [
            [sprite, np.rot90(sprite, -1)],
            [np.rot90(sprite, 1), np.rot90(sprite, 2)],
        ]
    )


def _fill_holes_in_composite(composite: np.ndarray) -> tuple[np.ndarray, list[int]]:
    out = composite.copy()
    n = composite.shape[0] // 2
    spans: list[int] = []
    for cells in _enclosed_zero_regions(composite):
        quadrants = {
            (0 if r < n else 2) + (0 if c < n else 1)
            for r, c in cells
        }
        span = len(quadrants)
        spans.append(span)
        color = GREEN if span == 1 else YELLOW if span == 2 else RED
        for r, c in cells:
            out[r, c] = color
    return out, sorted(spans)


def _fill_quadrant_locally(tile: np.ndarray) -> np.ndarray:
    out = tile.copy()
    for cells in _enclosed_zero_regions(tile):
        for r, c in cells:
            out[r, c] = GREEN
    return out


def _naive_per_quadrant_fill(sprite: np.ndarray) -> np.ndarray:
    tl = _fill_quadrant_locally(sprite)
    tr = _fill_quadrant_locally(np.rot90(sprite, -1))
    bl = _fill_quadrant_locally(np.rot90(sprite, 1))
    br = _fill_quadrant_locally(np.rot90(sprite, 2))
    return np.block([[tl, tr], [bl, br]])


def _rows_to_mask(rows: tuple[str, ...]) -> np.ndarray:
    return np.array([[1 if ch == "#" else 0 for ch in row] for row in rows], dtype=int)


def _transforms(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[bytes] = set()
    for k in range(4):
        rotated = np.rot90(mask, k)
        for variant in (rotated, np.fliplr(rotated)):
            key = variant.tobytes()
            if key not in seen:
                seen.add(key)
                variants.append(variant.copy())
    return variants


def _sprite_family(sprite: np.ndarray) -> tuple[str | None, list[int]]:
    composite = _compose_sprite(sprite)
    _, spans = _fill_holes_in_composite(composite)
    has_yellow = 2 in spans
    has_red = any(span >= 3 for span in spans)
    if has_yellow and has_red:
        return "yellow_red", spans
    if has_yellow:
        return "yellow_only", spans
    if has_red:
        return "red_only", spans
    return None, spans


def _valid_sprite(sprite: np.ndarray) -> bool:
    n, m = sprite.shape
    if n != m:
        return False
    if not (8 <= int(sprite.sum()) <= 15):
        return False
    if not _is_connected(sprite.astype(bool)):
        return False
    if _bbox(sprite.astype(bool)) != (0, 0, n - 1, n - 1):
        return False
    if np.array_equal(sprite, np.fliplr(sprite)) or np.array_equal(sprite, np.flipud(sprite)):
        return False
    if np.array_equal(sprite, np.rot90(sprite, 1)):
        return False
    if len(_enclosed_zero_regions(sprite)) < 1:
        return False
    family, spans = _sprite_family(sprite)
    return family is not None and any(span >= 2 for span in spans)


def _candidate_pools() -> dict[str, list[dict[str, object]]]:
    global _CANDIDATE_POOLS
    if _CANDIDATE_POOLS is not None:
        return _CANDIDATE_POOLS

    pools = {name: [] for name in _FAMILY_ORDER}
    seen_outputs: set[str] = set()
    for rows in _SEED_ROWS:
        base = _rows_to_mask(rows)
        for variant in _transforms(base):
            if not _valid_sprite(variant):
                continue
            family, spans = _sprite_family(variant)
            assert family is not None
            sprite = (variant * BLUE).astype(int)
            composite = _compose_sprite(sprite)
            output, _ = _fill_holes_in_composite(composite)
            out_sig = _signature(output)
            if out_sig in seen_outputs:
                continue
            seen_outputs.add(out_sig)
            pools[family].append(
                {
                    "sprite": sprite,
                    "family": family,
                    "spans": tuple(spans),
                    "out_sig": out_sig,
                }
            )

    for family, items in pools.items():
        if not items:
            raise AssertionError(f"no candidate sprites found for {family}")
    _CANDIDATE_POOLS = pools
    return pools


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate the cropped blue sprite into a 2x2 composite and fill each enclosed black region by how many quadrants it spans.

    Input structure:
    - The grid has a black background and exactly one 4-connected blue sprite.
    - The sprite is cropped by its tight bounding box; generated instances use a square crop so the four quarter-turns tile a 2x2 square composite.
    - Black cells inside that crop can be true holes or open cavities that only become enclosed after the four rotated copies meet.

    Rule:
    - Crop the blue sprite to its tight bounding box.
    - Build a 2x2 composite: upper-left = original, upper-right = 90-degree clockwise rotation, lower-left = 90-degree counterclockwise rotation, lower-right = 180-degree rotation.
    - In the composite, find every 4-connected black region that does not touch the outer border.
    - Fill each enclosed region by the number of quadrants it occupies: 1 quadrant -> green, exactly 2 quadrants -> yellow, 3 or 4 quadrants -> red. Keep all blue pixels unchanged.

    Key insight:
    - The decisive holes are not properties of the original sprite alone: some black cavities only become closed, and only reveal their quadrant span, after the four rotated copies are stitched together.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("expected a 2D grid")

    sprite_mask = g == BLUE
    if not sprite_mask.any():
        sprite_mask = g != BLACK
    if not sprite_mask.any():
        raise AssertionError("expected a non-empty blue sprite")
    if len(_component_coords(sprite_mask)) != 1:
        raise AssertionError("expected exactly one connected sprite")

    sprite = _crop_to_bbox(g, sprite_mask)
    sprite = np.where(sprite == BLUE, BLUE, BLACK).astype(int)
    composite = _compose_sprite(sprite)
    output, _ = _fill_holes_in_composite(composite)
    return output


def _pick_family() -> str:
    global _GEN_CALLS
    if _GEN_CALLS < len(_FAMILY_ORDER):
        family = _FAMILY_ORDER[_GEN_CALLS]
    else:
        family = random.choices(_FAMILY_ORDER, weights=_FAMILY_WEIGHTS, k=1)[0]
    _GEN_CALLS += 1
    return family


def _pick_candidate(family: str) -> dict[str, object]:
    pools = _candidate_pools()
    preferred = [item for item in pools[family] if str(item["out_sig"]) not in _USED_OUTPUTS]
    if preferred:
        return random.choice(preferred)

    unseen_any = [item for fam in _FAMILY_ORDER for item in pools[fam] if str(item["out_sig"]) not in _USED_OUTPUTS]
    if unseen_any:
        return random.choice(unseen_any)

    return random.choice(pools[family])


def _embed_sprite(sprite: np.ndarray) -> np.ndarray:
    n = sprite.shape[0]
    rows = random.randint(n + 2, min(11, n + 6))
    cols = random.randint(n + 2, min(11, n + 6))
    grid = np.zeros((rows, cols), dtype=int)
    r0 = random.randint(1, rows - n - 1)
    c0 = random.randint(1, cols - n - 1)
    patch = grid[r0 : r0 + n, c0 : c0 + n]
    patch[sprite == BLUE] = BLUE
    return grid


def _validate_generated(grid: np.ndarray, candidate: dict[str, object]) -> bool:
    sprite = np.asarray(candidate["sprite"], dtype=int)
    family = str(candidate["family"])

    blue_mask = grid == BLUE
    if len(_component_coords(blue_mask)) != 1:
        return False
    r0, c0, r1, c1 = _bbox(blue_mask)
    if r0 == 0 or c0 == 0 or r1 == grid.shape[0] - 1 or c1 == grid.shape[1] - 1:
        return False

    cropped = _crop_to_bbox(grid, blue_mask)
    if not np.array_equal(cropped, sprite):
        return False
    if len(_enclosed_zero_regions(sprite)) < 1:
        return False

    expected, spans = _fill_holes_in_composite(_compose_sprite(sprite))
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False
    if not any(span >= 2 for span in spans):
        return False

    naive = _naive_per_quadrant_fill(sprite)
    if np.array_equal(actual, naive):
        return False
    if not np.any((actual != naive) & np.isin(actual, [YELLOW, RED])):
        return False

    colors = set(map(int, np.unique(actual)))
    if family == "yellow_only":
        return YELLOW in colors and RED not in colors
    if family == "yellow_red":
        return YELLOW in colors and RED in colors
    if family == "red_only":
        return RED in colors and YELLOW not in colors
    return False


def generate_input() -> np.ndarray:
    for _ in range(300):
        family = _pick_family()
        candidate = _pick_candidate(family)
        grid = _embed_sprite(np.asarray(candidate["sprite"], dtype=int))
        if _validate_generated(grid, candidate):
            _USED_OUTPUTS.add(str(candidate["out_sig"]))
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
