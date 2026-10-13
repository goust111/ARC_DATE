from __future__ import annotations

from collections import Counter

import numpy as np


BG = 0
NONZERO_COLORS = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 400


SEED_PATTERNS = (
    np.array([[1, 1], [1, 0]], dtype=int),
    np.array([[1, 1], [0, 1]], dtype=int),
    np.array([[1, 0], [1, 1]], dtype=int),
    np.array([[0, 1], [1, 1]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=int),
    np.array([[0, 1, 1], [1, 1, 0]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[1, 0, 1], [1, 1, 1]], dtype=int),
    np.array([[1, 1, 1], [1, 0, 1]], dtype=int),
)


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(pts, dtype=int))
    return out


def _row_major_most_common_nonzero(tile: np.ndarray) -> int:
    counts: Counter[int] = Counter(int(v) for v in tile.ravel() if int(v) != BG)
    assert counts, "seed tile must contain at least one non-background color"
    best_count = max(counts.values())
    winners = {color for color, count in counts.items() if count == best_count}
    for value in tile.ravel():
        color = int(value)
        if color in winners:
            return color
    raise AssertionError("unreachable")


def _parse_rectangle(g: np.ndarray, coords: np.ndarray) -> dict:
    r0, c0, r1, c1 = _bbox_coords(coords)
    patch = g[r0 : r1 + 1, c0 : c1 + 1]
    h, w = patch.shape
    assert h >= 6 and w >= 6, "rectangles must leave room for a separated seed"
    assert np.all(patch != BG), "rectangle bbox must be fully non-background"
    assert coords.shape[0] == h * w, "component must fill its bbox"

    border = np.concatenate((patch[0, :], patch[-1, :], patch[1:-1, 0], patch[1:-1, -1]))
    border_colors = np.unique(border)
    assert len(border_colors) == 1, "rectangle border must be monochrome"
    body = int(border_colors[0])
    assert body != BG

    inner = patch[1:-1, 1:-1]
    seed_mask = inner != body
    assert seed_mask.any(), "each rectangle must contain an interior seed"

    seed_components = _components(seed_mask)
    assert len(seed_components) == 1, "seed pixels must form one 4-connected region"
    seed_coords = seed_components[0]
    sr0, sc0, sr1, sc1 = _bbox_coords(seed_coords)
    tile_h, tile_w = sr1 - sr0 + 1, sc1 - sc0 + 1
    assert (tile_h, tile_w) in {(2, 2), (2, 3)}, "seed bbox must be 2x2 or 2x3"
    assert sr0 >= 1 and sc0 >= 1, "seed must stay away from the frame"
    assert sr1 <= inner.shape[0] - 2 and sc1 <= inner.shape[1] - 2, "seed must stay away from the frame"

    tile = inner[sr0 : sr1 + 1, sc0 : sc1 + 1].copy()
    tile[tile == body] = BG
    assert np.any(tile == BG), "seed tile must include at least one gap"

    return {
        "bbox": (r0, c0, r1, c1),
        "body": body,
        "tile": tile,
        "seed_offset": (sr0, sc0),
        "inner_shape": inner.shape,
    }


def _transform_rectangle(out: np.ndarray, rect: dict) -> None:
    r0, c0, r1, c1 = rect["bbox"]
    body = int(rect["body"])
    tile = np.asarray(rect["tile"], dtype=int)
    seed_r, seed_c = rect["seed_offset"]
    inner_h, inner_w = rect["inner_shape"]

    out[r0 : r1 + 1, c0 : c1 + 1] = body
    out[r0 + 1 : r1, c0 + 1 : c1] = BG

    if (inner_h * inner_w) % 2 == 0:
        rr, cc = np.indices((inner_h, inner_w))
        tiled = tile[(rr - int(seed_r)) % tile.shape[0], (cc - int(seed_c)) % tile.shape[1]]
        out[r0 + 1 : r1, c0 + 1 : c1] = tiled
    else:
        fill_color = _row_major_most_common_nonzero(tile)
        out[r0 + 1 : r1, c0 + 1 : c1] = fill_color


def main(grid: np.ndarray) -> np.ndarray:
    """Hollow each solid rectangle and refill its cavity by seed tiling for even area or solid seed color for odd area.

    Input structure
    - A black background contains 3 to 5 separated non-black axis-aligned solid rectangles.
    - Each rectangle is filled with one body color, except for one interior 4-connected seed shape in a different color.
    - The seed shape stays strictly inside the rectangle, and its 2x2 or 2x3 bounding box may include body-colored gap cells.

    Rule
    - For each non-black connected component, treat its full bbox as one rectangle and use the bbox border color as the body color.
    - Inside the rectangle, collect all pixels whose color differs from the body; take their minimal bbox as the seed tile, but convert body-colored cells inside that tile bbox to black.
    - Hollow the rectangle to a 1-pixel frame.
    - If the cavity area `(height-2)*(width-2)` is even, tile the cavity periodically with the seed tile while preserving the seed's original row/col phase offset inside the cavity.
    - If the cavity area is odd, fill the whole cavity with the most frequent non-black color from the seed tile, breaking ties by first row-major appearance.

    Key insight
    - The seed must be extracted before hollowing, and the same extracted tile has two incompatible roles: a phase-preserved periodic template for even cavities and a color source for odd cavities.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    out = g.copy()

    rect_components = _components(g != BG)
    assert rect_components, "expected at least one rectangle"

    for coords in rect_components:
        rect = _parse_rectangle(g, coords)
        _transform_rectangle(out, rect)

    return out


def _choose_dims(
    rng: np.random.Generator,
    parity: int,
    tile_h: int,
    tile_w: int,
) -> tuple[int, int]:
    choices = []
    for inner_h in range(tile_h + 2, 9):
        for inner_w in range(tile_w + 2, 9):
            if (inner_h * inner_w) % 2 == parity:
                choices.append((inner_h + 2, inner_w + 2))
    assert choices
    weights = np.array([(h * w) ** 1.3 for h, w in choices], dtype=float)
    weights /= weights.sum()
    idx = int(rng.choice(len(choices), p=weights))
    return tuple(int(v) for v in choices[idx])


def _choose_seed_offset(
    rng: np.random.Generator,
    inner_h: int,
    inner_w: int,
    tile_h: int,
    tile_w: int,
    require_phase_shift: bool,
) -> tuple[int, int]:
    rows = list(range(1, inner_h - tile_h))
    cols = list(range(1, inner_w - tile_w))
    candidates = [(r, c) for r in rows for c in cols]
    if require_phase_shift:
        shifted = [(r, c) for r, c in candidates if (r % tile_h) != 0 or (c % tile_w) != 0]
        assert shifted, "expected at least one phase-shifted seed position"
        candidates = shifted
    return candidates[int(rng.integers(0, len(candidates)))]


def _build_patch(
    h: int,
    w: int,
    body: int,
    seed_color: int,
    tile_mask: np.ndarray,
    seed_r: int,
    seed_c: int,
) -> np.ndarray:
    patch = np.full((h, w), body, dtype=int)
    tile_h, tile_w = tile_mask.shape
    target = patch[1 + seed_r : 1 + seed_r + tile_h, 1 + seed_c : 1 + seed_c + tile_w]
    target[tile_mask.astype(bool)] = seed_color
    return patch


def _place_patches(
    rng: np.random.Generator,
    h: int,
    w: int,
    patches: list[np.ndarray],
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    grid = np.zeros((h, w), dtype=int)
    origins: list[tuple[int, int]] = [(-1, -1)] * len(patches)
    order = sorted(range(len(patches)), key=lambda idx: patches[idx].shape[0] * patches[idx].shape[1], reverse=True)

    for idx in order:
        patch = patches[idx]
        ph, pw = patch.shape
        candidates: list[tuple[int, int]] = []
        for r0 in range(1, h - ph):
            for c0 in range(1, w - pw):
                rr0 = r0 - 1
                cc0 = c0 - 1
                rr1 = r0 + ph + 1
                cc1 = c0 + pw + 1
                if not np.any(grid[rr0:rr1, cc0:cc1] != BG):
                    candidates.append((r0, c0))
        if not candidates:
            raise ValueError("could not place all rectangles")
        r0, c0 = candidates[int(rng.integers(0, len(candidates)))]
        grid[r0 : r0 + ph, c0 : c0 + pw] = patch
        origins[idx] = (r0, c0)

    return grid, origins


def _expected_from_specs(grid: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = grid.copy()
    for spec in specs:
        _transform_rectangle(out, spec)
    return out


def _build_spec(
    rng: np.random.Generator,
    body: int,
    parity: int,
    require_phase_shift: bool,
) -> tuple[np.ndarray, dict]:
    mask = SEED_PATTERNS[int(rng.integers(0, len(SEED_PATTERNS)))]
    tile_h, tile_w = mask.shape
    h, w = _choose_dims(rng, parity, tile_h, tile_w)
    inner_h, inner_w = h - 2, w - 2

    seed_choices = [color for color in NONZERO_COLORS if color != body]
    seed_color = int(seed_choices[int(rng.integers(0, len(seed_choices)))])
    seed_r, seed_c = _choose_seed_offset(rng, inner_h, inner_w, tile_h, tile_w, require_phase_shift=require_phase_shift)
    patch = _build_patch(h, w, body, seed_color, mask, seed_r, seed_c)
    spec = {
        "bbox": (0, 0, h - 1, w - 1),
        "body": body,
        "tile": (mask * seed_color).astype(int),
        "seed_offset": (seed_r, seed_c),
        "inner_shape": (inner_h, inner_w),
        "parity": parity,
        "phase_shifted": (seed_r % tile_h) != 0 or (seed_c % tile_w) != 0,
    }
    return patch, spec


def _validate_generated(inp: np.ndarray, specs: list[dict]) -> bool:
    if inp.ndim != 2 or max(inp.shape) > 30:
        return False
    if len(specs) < 3 or len(specs) > 5:
        return False
    parities = [int(spec["parity"]) for spec in specs]
    if not (0 in parities and 1 in parities):
        return False
    if not any(spec["parity"] == 0 and spec["phase_shifted"] for spec in specs):
        return False

    expected = _expected_from_specs(inp, specs)
    try:
        solved = main(inp)
    except Exception:
        return False
    if not np.array_equal(solved, expected):
        return False

    try:
        rects = [_parse_rectangle(inp, coords) for coords in _components(inp != BG)]
    except Exception:
        return False
    if len(rects) != len(specs):
        return False

    even_ok = False
    odd_ok = False
    for rect in rects:
        inner_h, inner_w = rect["inner_shape"]
        r0, c0, r1, c1 = rect["bbox"]
        cavity = solved[r0 + 1 : r1, c0 + 1 : c1]
        if (inner_h * inner_w) % 2 == 0:
            even_ok |= np.any(cavity == BG) and len(np.unique(cavity)) >= 2
        else:
            odd_ok |= len(np.unique(cavity)) == 1 and int(cavity[0, 0]) != BG
    if not (even_ok and odd_ok):
        return False
    if np.array_equal(inp, solved):
        return False
    return True


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    grid_h = int(rng.integers(22, 30))
    grid_w = int(rng.integers(22, 30))
    rect_count = int(rng.integers(3, 6))

    parity_plan = [0, 1]
    parity_plan.extend(int(rng.integers(0, 2)) for _ in range(rect_count - 2))
    rng.shuffle(parity_plan)

    phase_shift_index = next(i for i, parity in enumerate(parity_plan) if parity == 0)
    body_colors = rng.choice(np.array(NONZERO_COLORS, dtype=int), size=rect_count, replace=False).astype(int).tolist()

    patches: list[np.ndarray] = []
    specs: list[dict] = []
    for idx, (body, parity) in enumerate(zip(body_colors, parity_plan)):
        patch, spec = _build_spec(rng, int(body), int(parity), require_phase_shift=(idx == phase_shift_index))
        patches.append(patch)
        specs.append(spec)

    try:
        grid, origins = _place_patches(rng, grid_h, grid_w, patches)
    except ValueError:
        return None

    placed_specs: list[dict] = []
    for spec, origin, patch in zip(specs, origins, patches):
        r0, c0 = origin
        ph, pw = patch.shape
        placed_specs.append(
            {
                "bbox": (r0, c0, r0 + ph - 1, c0 + pw - 1),
                "body": spec["body"],
                "tile": spec["tile"],
                "seed_offset": spec["seed_offset"],
                "inner_shape": spec["inner_shape"],
                "parity": spec["parity"],
                "phase_shifted": spec["phase_shifted"],
            }
        )

    if not _validate_generated(grid, placed_specs):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for the parity-routed hollow tiling task")
