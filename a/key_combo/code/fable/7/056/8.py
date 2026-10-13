from __future__ import annotations

from collections import Counter
import random

import numpy as np


GREY = 5
UNKNOWN = -1
QUAD_ORDER = ("tl", "tr", "bl", "br")
STRUCTURES = {
    "mirror": ("id", "fliplr", "flipud", "rot180"),
    "rotate": ("id", "rot90cw", "rot270cw", "rot180"),
}
INVERSE_TRANSFORM = {
    "id": "id",
    "fliplr": "fliplr",
    "flipud": "flipud",
    "rot90cw": "rot270cw",
    "rot180": "rot180",
    "rot270cw": "rot90cw",
}


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return np.asarray(grid)[r0 : r1 + 1, c0 : c1 + 1].copy()


def _apply_transform(arr: np.ndarray, transform: str) -> np.ndarray:
    if transform == "id":
        return np.array(arr, copy=True)
    if transform == "fliplr":
        return np.fliplr(arr)
    if transform == "flipud":
        return np.flipud(arr)
    if transform == "rot90cw":
        return np.rot90(arr, -1)
    if transform == "rot180":
        return np.rot90(arr, 2)
    if transform == "rot270cw":
        return np.rot90(arr, 1)
    raise ValueError(f"unknown transform: {transform}")


def _split_quadrants(panel: np.ndarray) -> list[np.ndarray]:
    h, w = panel.shape
    assert h == w and h % 2 == 0
    q = h // 2
    return [
        panel[:q, :q].copy(),
        panel[:q, q:].copy(),
        panel[q:, :q].copy(),
        panel[q:, q:].copy(),
    ]


def _quadrant_color(quadrant: np.ndarray, bg: int) -> int:
    colors = sorted(set(map(int, quadrant.ravel())) - {int(bg), GREY})
    assert len(colors) == 1, "each quadrant must expose exactly one non-grey foreground color"
    return int(colors[0])


def _evaluate_candidate(
    quadrants: list[np.ndarray],
    colors: list[int],
    bg: int,
    structure_name: str,
) -> dict[str, object]:
    q = int(quadrants[0].shape[0])
    combined = np.full((q, q), UNKNOWN, dtype=int)
    known_maps: list[np.ndarray] = []
    conflicts = 0

    for quadrant, color, transform in zip(quadrants, colors, STRUCTURES[structure_name]):
        status = np.full((q, q), UNKNOWN, dtype=int)
        visible = quadrant != GREY
        status[visible & (quadrant == int(bg))] = 0
        status[visible & (quadrant == int(color))] = 1
        assert not np.any(
            visible & ~(np.isin(quadrant, [int(bg), int(color)]))
        ), "unexpected visible color inside quadrant"

        back = _apply_transform(status, INVERSE_TRANSFORM[transform])
        known = back != UNKNOWN
        conflicts += int(np.count_nonzero(known & (combined != UNKNOWN) & (combined != back)))
        fill = known & (combined == UNKNOWN)
        combined[fill] = back[fill]
        known_maps.append(known)

    return {
        "structure": structure_name,
        "mask": combined,
        "conflicts": int(conflicts),
        "unknown": int(np.count_nonzero(combined == UNKNOWN)),
        "known_maps": known_maps,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the hidden source quadrant from an occluded four-way mosaic and recolor it with the top-left quadrant color.

    Input structure:
    - The background contains one even-sided square foreground panel.
    - The panel is split into four equal quadrants, and each quadrant uses its own foreground color plus grey occlusion patches.
    - The four quadrants show the same binary source pattern under either a mirror mosaic across the center lines or a 90-degree rotation mosaic around the panel center.

    Rule:
    - Detect the square panel and split it into four quadrants.
    - Compare the visible foreground/background evidence under the two possible structural families: mirror mosaic or rotation mosaic.
    - In the unique structure with conflict-free full coverage, inverse-transform all visible evidence into the top-left source coordinate system and reconstruct the complete source pattern.
    - Output that single-quadrant reconstruction using the top-left quadrant's foreground color on the original background color.

    Key insight:
    - Because quadrant colors differ, the solver must ignore color identities and align only the visible binary mask evidence; the correct global symmetry type is the one that makes all partial observations agree and fully cover the source.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    panel_box = _bbox(g != int(bg))
    assert panel_box is not None, "expected a visible panel"
    panel = _crop(g, panel_box)
    ph, pw = panel.shape
    assert ph == pw and ph % 2 == 0, "panel must be an even square"

    quadrants = _split_quadrants(panel)
    colors = [_quadrant_color(quadrant, bg) for quadrant in quadrants]
    evaluations = [_evaluate_candidate(quadrants, colors, bg, name) for name in STRUCTURES]
    valid = [item for item in evaluations if int(item["conflicts"]) == 0 and int(item["unknown"]) == 0]
    assert len(valid) == 1, "structure must be uniquely determined"

    solved = np.asarray(valid[0]["mask"], dtype=int)
    out = np.full(solved.shape, int(bg), dtype=int)
    out[solved == 1] = int(colors[0])
    return out


def _make_source_mask(q: int, rng: random.Random) -> np.ndarray:
    for _ in range(200):
        mask = np.zeros((q, q), dtype=bool)

        # Force a visible transpose-asymmetry witness so the wrong structure conflicts.
        mask[0, 1] = True
        mask[1, 0] = False

        for c in range(q):
            mask[rng.randrange(q), c] = True
        for r in range(q):
            mask[r, rng.randrange(q)] = True

        for _ in range(rng.randint(2, 4)):
            h = rng.randint(1, max(1, q // 2))
            w = rng.randint(1, max(1, q // 2))
            r0 = rng.randint(0, q - h)
            c0 = rng.randint(0, q - w)
            mask[r0 : r0 + h, c0 : c0 + w] = rng.random() < 0.75

        mask[0, 1] = True
        mask[1, 0] = False
        mask[q - 1, q - 2] = True
        mask[q - 2, q - 1] = False

        density = float(mask.mean())
        if not (0.22 <= density <= 0.68):
            continue
        if not np.all(mask.any(axis=0)) or not np.all(mask.any(axis=1)):
            continue
        if np.array_equal(mask, mask.T):
            continue
        return mask
    raise RuntimeError("failed to generate a non-degenerate source mask")


def _build_reveal_masks(q: int, rng: random.Random) -> list[np.ndarray]:
    rh = rng.randint(2, q - 2)
    cw = rng.randint(2, q - 2)
    reveal = {name: np.ones((q, q), dtype=bool) for name in QUAD_ORDER}

    # Each quadrant misses a different opposite block, so no single quadrant is complete.
    reveal["tl"][rh:, cw:] = False
    reveal["tr"][rh:, :cw] = False
    reveal["bl"][:rh, cw:] = False
    reveal["br"][:rh, :cw] = False

    # Corner cells make every quadrant contribute unique evidence.
    unique_cells = {
        "tl": (0, 0),
        "tr": (0, q - 1),
        "bl": (q - 1, 0),
        "br": (q - 1, q - 1),
    }
    for owner, (r, c) in unique_cells.items():
        for name in QUAD_ORDER:
            if name != owner:
                reveal[name][r, c] = False

    # Keep the transpose witness visible in TL/TR/BL to disambiguate mirror vs rotation.
    for name in ("tl", "tr", "bl"):
        reveal[name][0, 1] = True
        reveal[name][1, 0] = True

    return [reveal[name] for name in QUAD_ORDER]


def _render_panel(
    source_mask: np.ndarray,
    structure_name: str,
    colors: list[int],
    reveal_masks: list[np.ndarray],
    bg: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    quadrants: list[np.ndarray] = []
    for color, transform, reveal in zip(colors, STRUCTURES[structure_name], reveal_masks):
        base_mask = _apply_transform(source_mask.astype(int), transform).astype(bool)
        visible = _apply_transform(reveal.astype(int), transform).astype(bool)

        quadrant = np.full(source_mask.shape, GREY, dtype=int)
        quadrant[visible & ~base_mask] = int(bg)
        quadrant[visible & base_mask] = int(color)
        quadrants.append(quadrant)

    q = int(source_mask.shape[0])
    panel = np.full((2 * q, 2 * q), int(bg), dtype=int)
    panel[:q, :q] = quadrants[0]
    panel[:q, q:] = quadrants[1]
    panel[q:, :q] = quadrants[2]
    panel[q:, q:] = quadrants[3]
    return panel, quadrants


def _assemble_grid(panel: np.ndarray, rng: random.Random, bg: int) -> np.ndarray:
    side = int(panel.shape[0])
    extra_h = rng.randint(8, min(14, 30 - side))
    extra_w = rng.randint(8, min(14, 30 - side))
    rows = side + extra_h
    cols = side + extra_w
    top = rng.randint(1, rows - side - 1)
    left = rng.randint(1, cols - side - 1)
    grid = np.full((rows, cols), int(bg), dtype=int)
    grid[top : top + side, left : left + side] = panel
    return grid


def _instance_semantics_ok(
    source_mask: np.ndarray,
    structure_name: str,
    quadrants: list[np.ndarray],
    colors: list[int],
    bg: int,
) -> bool:
    true_eval = _evaluate_candidate(quadrants, colors, bg, structure_name)
    other_name = "rotate" if structure_name == "mirror" else "mirror"
    other_eval = _evaluate_candidate(quadrants, colors, bg, other_name)

    if int(true_eval["conflicts"]) != 0 or int(true_eval["unknown"]) != 0:
        return False
    if int(other_eval["conflicts"]) == 0 and int(other_eval["unknown"]) == 0:
        return False
    if not np.array_equal(np.asarray(true_eval["mask"], dtype=int), source_mask.astype(int)):
        return False

    full = np.ones(source_mask.shape, dtype=bool)
    known_maps = [np.asarray(item, dtype=bool) for item in true_eval["known_maps"]]
    for known in known_maps:
        if np.array_equal(known, full):
            return False

    for idx in range(4):
        others = np.zeros(source_mask.shape, dtype=bool)
        for jdx, known in enumerate(known_maps):
            if idx != jdx:
                others |= known
        if not np.any(known_maps[idx] & ~others):
            return False

    for quadrant, color in zip(quadrants, colors):
        if not np.any(quadrant == int(color)):
            return False
        if not np.any(quadrant == int(bg)):
            return False
        if not np.any(quadrant == GREY):
            return False
    return True


def _generate_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    bg = 0
    q = rng.randint(5, 8)
    structure_name = rng.choice(list(STRUCTURES))
    colors = rng.sample([1, 2, 3, 4, 6, 7, 8, 9], 4)
    source_mask = _make_source_mask(q, rng)
    reveal_masks = _build_reveal_masks(q, rng)
    panel, quadrants = _render_panel(source_mask, structure_name, colors, reveal_masks, bg)
    assert _instance_semantics_ok(source_mask, structure_name, quadrants, colors, bg)

    grid = _assemble_grid(panel, rng, bg)
    expected = np.full(source_mask.shape, int(bg), dtype=int)
    expected[source_mask] = int(colors[0])
    assert np.array_equal(main(grid), expected)
    return grid, expected


def generate_input() -> np.ndarray:
    rng = random.Random()
    grid, _expected = _generate_case(rng)
    return grid
