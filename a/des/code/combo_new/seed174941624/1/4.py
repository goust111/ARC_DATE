from __future__ import annotations

import random

import numpy as np


BG = 0
RING_SIZE = 5
PATTERN_SIZE = 3
COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 500
_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GAP_CELLS = {
    "up": (0, 2),
    "right": (2, 4),
    "down": (4, 2),
    "left": (2, 0),
}
GAP_TO_CW_TURNS = {"up": 0, "right": 1, "down": 2, "left": 3}

_RNG = random.Random()


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _components_from_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []

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
                for dr, dc in _DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(coords)

    return components


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _mask_from_coords(coords: list[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask, (r0, c0, r1, c1)


def _foreground_components(grid: np.ndarray, bg: int) -> list[dict]:
    comps: list[dict] = []
    for coords in _components_from_mask(grid != int(bg)):
        mask, box = _mask_from_coords(coords)
        comps.append({"coords": coords, "bbox": box, "mask": mask, "area": len(coords)})
    comps.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1], comp["area"]))
    return comps


def _monochrome_components(grid: np.ndarray, bg: int) -> list[dict]:
    comps: list[dict] = []
    for color_raw in np.unique(grid):
        color = int(color_raw)
        if color == int(bg):
            continue
        for coords in _components_from_mask(grid == color):
            mask, box = _mask_from_coords(coords)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": box,
                    "mask": mask,
                    "area": len(coords),
                }
            )
    comps.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1], comp["color"]))
    return comps


def _make_ring_template(gap_dir: str) -> np.ndarray:
    mask = np.zeros((RING_SIZE, RING_SIZE), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    gr, gc = GAP_CELLS[gap_dir]
    mask[gr, gc] = False
    return mask


RING_TEMPLATES = {gap_dir: _make_ring_template(gap_dir) for gap_dir in GAP_CELLS}


def _is_rotation_distinguishable(mask: np.ndarray) -> bool:
    return all(not np.array_equal(mask, np.rot90(mask, k)) for k in (1, 2, 3))


def _extract_rings(components: list[dict]) -> dict[int, dict]:
    rings: dict[int, dict] = {}
    for comp in components:
        if comp["area"] != 15 or comp["mask"].shape != (RING_SIZE, RING_SIZE):
            continue
        matched_dir = None
        for gap_dir, template in RING_TEMPLATES.items():
            if np.array_equal(comp["mask"], template):
                matched_dir = gap_dir
                break
        if matched_dir is None:
            continue
        assert comp["color"] not in rings, "ring colors must be unique"
        rings[comp["color"]] = {
            "color": comp["color"],
            "bbox": comp["bbox"],
            "gap_dir": matched_dir,
        }
    assert rings, "expected at least one 5x5 ring"
    return rings


def _extract_patterns(components: list[dict], ring_colors: set[int], bg: int) -> dict[int, dict]:
    patterns: dict[int, dict] = {}
    for comp in components:
        if comp["color"] in ring_colors:
            continue
        if comp["mask"].shape != (PATTERN_SIZE, PATTERN_SIZE):
            continue
        if not (3 <= comp["area"] <= 8):
            continue
        if not _is_rotation_distinguishable(comp["mask"]):
            continue
        patch = np.full((PATTERN_SIZE, PATTERN_SIZE), int(bg), dtype=int)
        patch[comp["mask"]] = int(comp["color"])
        assert comp["color"] not in patterns, "pattern colors must be unique"
        patterns[comp["color"]] = {
            "color": comp["color"],
            "bbox": comp["bbox"],
            "patch": patch,
            "mask": comp["mask"],
        }
    assert patterns, "expected at least one 3x3 pattern"
    return patterns


def _find_legend_mapping(
    grid: np.ndarray,
    bg: int,
    ring_colors: set[int],
    pattern_colors: set[int],
) -> dict[int, int]:
    candidates: list[dict[int, int]] = []
    expected_rows = len(ring_colors)

    for comp in _foreground_components(grid, bg):
        r0, c0, r1, c1 = comp["bbox"]
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h != expected_rows or w != 2 or comp["area"] != 2 * expected_rows:
            continue

        patch = np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)
        if np.any(patch == int(bg)):
            continue

        left = [int(x) for x in patch[:, 0]]
        right = [int(x) for x in patch[:, 1]]
        if set(left) != ring_colors or set(right) != pattern_colors:
            continue
        if len(set(left)) != expected_rows or len(set(right)) != expected_rows:
            continue

        candidates.append({left[i]: right[i] for i in range(expected_rows)})

    assert len(candidates) == 1, "expected exactly one legend strip"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each gapped square ring to a colored 3x3 pattern, rotate the pattern from the gap direction, and place it inside the ring.

    Input structure:
    - The background is the dominant color.
    - Several isolated monochrome 5x5 hollow square rings appear; each ring has exactly one missing midpoint on one side, and different rings use different colors.
    - The grid also contains the same number of isolated monochrome 3x3 patterns whose shapes are rotation-distinguishable, plus one connected 2-column legend strip.
    - Each legend row contains a ring color on the left and its matched pattern color on the right.

    Rule:
    - Detect every ring and read the side of its one-cell gap.
    - Detect the legend strip and use it to pair each ring color with one scattered pattern color.
    - Clear every scattered pattern from its original location.
    - Rotate the matched 3x3 pattern by 0, 90, 180, or 270 degrees clockwise for top, right, bottom, or left gaps respectively, then place it in the ring's central 3x3 area.
    - Keep the rings and the legend unchanged.

    Key insight:
    - The legend determines which pattern belongs to which ring, while the ring gap independently encodes the rotation angle; only combining both cues yields the final embedded pattern.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = _get_bg(g)
    components = _monochrome_components(g, bg)
    rings = _extract_rings(components)
    patterns = _extract_patterns(components, set(rings), bg)
    assert len(patterns) == len(rings), "number of rings and patterns must match"

    mapping = _find_legend_mapping(g, bg, set(rings), set(patterns))
    assert set(mapping) == set(rings)
    assert set(mapping.values()) == set(patterns)

    out = g.copy()
    for pattern in patterns.values():
        r0, c0, r1, c1 = pattern["bbox"]
        target = out[r0 : r1 + 1, c0 : c1 + 1]
        target[pattern["mask"]] = int(bg)

    for ring_color, ring in rings.items():
        pattern_color = mapping[ring_color]
        pattern = patterns[pattern_color]
        turns = GAP_TO_CW_TURNS[ring["gap_dir"]]
        rotated = np.rot90(pattern["patch"], -turns)

        r0, c0, _r1, _c1 = ring["bbox"]
        center = out[r0 + 1 : r0 + 1 + PATTERN_SIZE, c0 + 1 : c0 + 1 + PATTERN_SIZE]
        assert center.shape == (PATTERN_SIZE, PATTERN_SIZE)
        assert np.all(center == int(bg)), "ring center should be empty before insertion"
        center[rotated != int(bg)] = rotated[rotated != int(bg)]

    return out


def _mask_is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    seen = set()
    stack = [tuple(map(int, coords[0]))]
    while stack:
        rc = stack.pop()
        if rc in seen:
            continue
        seen.add(rc)
        r, c = rc
        for dr, dc in _DIRS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and (nr, nc) not in seen:
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _build_pattern_bank() -> list[np.ndarray]:
    bank: list[np.ndarray] = []
    for bits in range(1 << (PATTERN_SIZE * PATTERN_SIZE)):
        mask = np.zeros((PATTERN_SIZE, PATTERN_SIZE), dtype=bool)
        for idx in range(PATTERN_SIZE * PATTERN_SIZE):
            if (bits >> idx) & 1:
                mask[idx // PATTERN_SIZE, idx % PATTERN_SIZE] = True

        area = int(mask.sum())
        if not (4 <= area <= 6):
            continue
        if not _mask_is_connected(mask):
            continue

        rows = np.flatnonzero(mask.any(axis=1))
        cols = np.flatnonzero(mask.any(axis=0))
        if rows.size == 0 or cols.size == 0:
            continue
        if rows[0] != 0 or rows[-1] != PATTERN_SIZE - 1 or cols[0] != 0 or cols[-1] != PATTERN_SIZE - 1:
            continue
        if not _is_rotation_distinguishable(mask):
            continue

        bank.append(mask)

    bank.sort(key=lambda mask: mask.tobytes())
    assert len(bank) >= 20, "need enough 3x3 asymmetric patterns for variety"
    return bank


PATTERN_BANK = _build_pattern_bank()


def _random_grid_shape(count: int) -> tuple[int, int]:
    min_side = {2: 16, 3: 19, 4: 22}[count]
    return _RNG.randint(min_side, 30), _RNG.randint(min_side, 30)


def _pick_position(blocked: np.ndarray, h: int, w: int) -> tuple[int, int] | None:
    H, W = blocked.shape
    if h > H or w > W:
        return None

    choices: list[tuple[int, int]] = []
    for r in range(H - h + 1):
        for c in range(W - w + 1):
            r0 = max(0, r - 1)
            c0 = max(0, c - 1)
            r1 = min(H, r + h + 1)
            c1 = min(W, c + w + 1)
            if not blocked[r0:r1, c0:c1].any():
                choices.append((r, c))
    if not choices:
        return None
    return _RNG.choice(choices)


def _reserve(blocked: np.ndarray, r: int, c: int, h: int, w: int) -> None:
    H, W = blocked.shape
    blocked[max(0, r - 1) : min(H, r + h + 1), max(0, c - 1) : min(W, c + w + 1)] = True


def _render_input(shape: tuple[int, int], objects: list[dict]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for obj in objects:
        r, c = int(obj["r"]), int(obj["c"])
        if obj["kind"] == "ring":
            patch = grid[r : r + RING_SIZE, c : c + RING_SIZE]
            patch[RING_TEMPLATES[obj["gap_dir"]]] = int(obj["color"])
        elif obj["kind"] == "pattern":
            patch = grid[r : r + PATTERN_SIZE, c : c + PATTERN_SIZE]
            patch[obj["mask"]] = int(obj["color"])
        elif obj["kind"] == "legend":
            for row_idx, (ring_color, pattern_color) in enumerate(obj["rows"]):
                grid[r + row_idx, c] = int(ring_color)
                grid[r + row_idx, c + 1] = int(pattern_color)
        else:
            raise AssertionError(f"unknown object kind: {obj['kind']}")
    return grid


def _expected_output(grid: np.ndarray, objects: list[dict], ring_to_pattern: dict[int, int]) -> np.ndarray:
    out = grid.copy()
    pattern_objects = {int(obj["color"]): obj for obj in objects if obj["kind"] == "pattern"}

    for pattern in pattern_objects.values():
        r, c = int(pattern["r"]), int(pattern["c"])
        patch = out[r : r + PATTERN_SIZE, c : c + PATTERN_SIZE]
        patch[pattern["mask"]] = BG

    for ring in (obj for obj in objects if obj["kind"] == "ring"):
        turns = GAP_TO_CW_TURNS[ring["gap_dir"]]
        pattern = pattern_objects[ring_to_pattern[int(ring["color"])]]
        patch = np.full((PATTERN_SIZE, PATTERN_SIZE), BG, dtype=int)
        patch[pattern["mask"]] = int(pattern["color"])
        rotated = np.rot90(patch, -turns)
        rr, cc = int(ring["r"]) + 1, int(ring["c"]) + 1
        target = out[rr : rr + PATTERN_SIZE, cc : cc + PATTERN_SIZE]
        target[rotated != BG] = rotated[rotated != BG]

    return out


def _gen_attempt() -> np.ndarray | None:
    count = _RNG.randint(2, 4)
    ring_colors = _RNG.sample(COLOR_POOL, count)
    remaining = [color for color in COLOR_POOL if color not in ring_colors]
    pattern_colors = _RNG.sample(remaining, count)
    pattern_masks = _RNG.sample(PATTERN_BANK, count)
    directions = [_RNG.choice(tuple(GAP_CELLS)) for _ in range(count)]

    mapped_patterns = pattern_colors[:]
    _RNG.shuffle(mapped_patterns)
    ring_to_pattern = {ring_colors[i]: mapped_patterns[i] for i in range(count)}
    legend_rows = list(ring_to_pattern.items())
    _RNG.shuffle(legend_rows)

    H, W = _random_grid_shape(count)

    specs: list[dict] = []
    for color, gap_dir in zip(ring_colors, directions):
        specs.append({"kind": "ring", "color": color, "gap_dir": gap_dir, "h": RING_SIZE, "w": RING_SIZE})
    for color, mask in zip(pattern_colors, pattern_masks):
        specs.append({"kind": "pattern", "color": color, "mask": mask.copy(), "h": PATTERN_SIZE, "w": PATTERN_SIZE})
    specs.append({"kind": "legend", "rows": legend_rows, "h": count, "w": 2})

    _RNG.shuffle(specs)
    specs.sort(key=lambda spec: spec["h"] * spec["w"], reverse=True)

    blocked = np.zeros((H, W), dtype=bool)
    placed: list[dict] = []
    for spec in specs:
        pos = _pick_position(blocked, int(spec["h"]), int(spec["w"]))
        if pos is None:
            return None
        obj = dict(spec)
        obj["r"], obj["c"] = pos
        _reserve(blocked, obj["r"], obj["c"], int(obj["h"]), int(obj["w"]))
        placed.append(obj)

    grid = _render_input((H, W), placed)
    expected = _expected_output(grid, placed, ring_to_pattern)

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ring-pattern pairing puzzle")
