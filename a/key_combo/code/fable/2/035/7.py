from __future__ import annotations

from collections import Counter
from itertools import combinations, product

import numpy as np


BLACK = 0
GREY = 5
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIR8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 400


def _components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    dirs = DIR4 if int(connectivity) == 4 else DIR8
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in dirs:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _unique_mode(arr: np.ndarray) -> int:
    counts = Counter(map(int, np.asarray(arr).reshape(-1)))
    ranked = counts.most_common()
    assert ranked and (len(ranked) == 1 or ranked[0][1] > ranked[1][1]), "expected a unique dominant color"
    return int(ranked[0][0])


def _panel_start(grid: np.ndarray) -> int:
    for r in range(1, grid.shape[0]):
        if bool(np.all(grid[r:] != BLACK)):
            return int(r)
    raise AssertionError("expected a non-black lower panel")


def _bar_specs(top: np.ndarray) -> list[dict]:
    assert not bool(np.any((top != BLACK) & (top != GREY))), "top strip must contain only black and grey"
    bars = []
    for coords in _components_mask(top == GREY, connectivity=4):
        r0, c0, r1, c1 = _bbox_coords(coords)
        height = int(r1 - r0 + 1)
        width = int(c1 - c0 + 1)
        area = int(len(coords))
        if height == 1 and width >= 2 and area == width:
            orientation = "h"
            length = width
        elif width == 1 and height >= 2 and area == height:
            orientation = "v"
            length = height
        else:
            raise AssertionError("each grey component must be a straight bar")
        bars.append({"coords": coords, "orientation": orientation, "length": int(length)})
    assert bars, "expected at least one grey bar"
    return bars


def _scene_maps(scene: np.ndarray) -> tuple[int, dict[int, int], dict[int, int], dict[int, tuple[int, int]]]:
    bg = _unique_mode(scene)
    colors = sorted(set(map(int, np.unique(scene))) - {int(bg)})
    assert colors, "expected foreground colors in the lower panel"

    by4: dict[int, int] = {}
    by8: dict[int, int] = {}
    per_color: dict[int, tuple[int, int]] = {}
    for color in colors:
        count4 = len(_components_mask(scene == color, connectivity=4))
        count8 = len(_components_mask(scene == color, connectivity=8))
        assert count4 not in by4, "4-connected counts must identify colors uniquely"
        assert count8 not in by8, "8-connected counts must identify colors uniquely"
        by4[int(count4)] = int(color)
        by8[int(count8)] = int(color)
        per_color[int(color)] = (int(count4), int(count8))
    return int(bg), by4, by8, per_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each grey instruction bar with the uniquely matching scene color chosen by its orientation-specific component count.

    Input structure:
    - A black instruction strip sits above a non-black scene panel.
    - The strip contains separated grey horizontal and vertical bars of different lengths.
    - The scene panel has a single dominant background color and several fragment colors whose pieces may touch only diagonally.

    Rule:
    - For each horizontal grey bar of length n, count 4-connected components for every scene color and find the unique color whose count is n; recolor the whole bar to that color.
    - For each vertical grey bar of length m, count 8-connected components for every scene color and find the unique color whose count is m; recolor the whole bar to that color.
    - Leave the entire scene panel unchanged.

    Key insight:
    - Bar orientation is the selector for which connectivity notion to use: horizontal bars read 4-connectivity, while vertical bars read 8-connectivity on the same lower-panel fragments.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    panel_start = _panel_start(g)
    top = g[:panel_start]
    scene = g[panel_start:]
    _bg, by4, by8, _per_color = _scene_maps(scene)

    out = g.copy()
    for bar in _bar_specs(top):
        if bar["orientation"] == "h":
            color = by4[bar["length"]]
        else:
            color = by8[bar["length"]]
        coords = bar["coords"]
        out[coords[:, 0], coords[:, 1]] = int(color)
    return out


def _corner_filled(mask: np.ndarray, corner: str) -> bool:
    if corner == "tl":
        return bool(mask[0, 0])
    if corner == "tr":
        return bool(mask[0, -1])
    if corner == "bl":
        return bool(mask[-1, 0])
    if corner == "br":
        return bool(mask[-1, -1])
    raise ValueError("bad corner")


def _random_polyomino_mask(rng: np.random.Generator, require_corner: str | None = None) -> np.ndarray:
    for _ in range(200):
        area = int(rng.integers(2, 5))
        coords = {(0, 0)}
        while len(coords) < area:
            frontier = []
            for r, c in coords:
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) in coords:
                        continue
                    trial = coords | {(nr, nc)}
                    rows = [p[0] for p in trial]
                    cols = [p[1] for p in trial]
                    if max(rows) - min(rows) + 1 <= 4 and max(cols) - min(cols) + 1 <= 4:
                        frontier.append((nr, nc))
            if not frontier:
                break
            coords.add(frontier[int(rng.integers(len(frontier)))])
        if len(coords) != area:
            continue

        min_r = min(r for r, _ in coords)
        min_c = min(c for _, c in coords)
        norm = [(r - min_r, c - min_c) for r, c in coords]
        h = max(r for r, _ in norm) + 1
        w = max(c for _, c in norm) + 1
        mask = np.zeros((h, w), dtype=bool)
        for r, c in norm:
            mask[r, c] = True

        mask = np.rot90(mask, int(rng.integers(4)))
        if bool(rng.integers(2)):
            mask = np.flipud(mask)
        if bool(rng.integers(2)):
            mask = np.fliplr(mask)

        if require_corner is not None and not _corner_filled(mask, require_corner):
            continue
        return np.asarray(mask, dtype=bool)
    raise RuntimeError("failed to generate a valid fragment mask")


def _colorize(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    patch = np.full(mask.shape, int(bg), dtype=int)
    patch[mask] = int(color)
    return patch


def _isolated_patch(color: int, bg: int, rng: np.random.Generator) -> np.ndarray:
    return _colorize(_random_polyomino_mask(rng), color, bg)


def _pair_patch(color: int, bg: int, rng: np.random.Generator) -> np.ndarray:
    a = _random_polyomino_mask(rng, require_corner="br")
    b = _random_polyomino_mask(rng, require_corner="tl")
    h = int(a.shape[0] + b.shape[0])
    w = int(a.shape[1] + b.shape[1])
    patch = np.full((h, w), int(bg), dtype=int)
    patch[: a.shape[0], : a.shape[1]][a] = int(color)
    patch[a.shape[0] :, a.shape[1] :][b] = int(color)
    if bool(rng.integers(2)):
        patch = np.flipud(patch)
    if bool(rng.integers(2)):
        patch = np.fliplr(patch)
    assert len(_components_mask(patch == color, connectivity=4)) == 2
    assert len(_components_mask(patch == color, connectivity=8)) == 1
    return patch


def _candidate_positions(shape: tuple[int, int], patch_shape: tuple[int, int], border: int) -> list[tuple[int, int]]:
    h, w = map(int, shape)
    ph, pw = map(int, patch_shape)
    positions = []
    for r0 in range(int(border), h - ph - int(border) + 1):
        for c0 in range(int(border), w - pw - int(border) + 1):
            positions.append((int(r0), int(c0)))
    return positions


def _can_place(blocked: np.ndarray, patch_mask: np.ndarray, r0: int, c0: int) -> bool:
    h, w = patch_mask.shape
    return not bool((blocked[r0 : r0 + h, c0 : c0 + w] & patch_mask).any())


def _mark_blocked(blocked: np.ndarray, patch_mask: np.ndarray, r0: int, c0: int, padding: int = 1) -> None:
    h, w = blocked.shape
    for rr, cc in np.argwhere(patch_mask):
        gr = int(r0 + rr)
        gc = int(c0 + cc)
        blocked[max(0, gr - padding) : min(h, gr + padding + 1), max(0, gc - padding) : min(w, gc + padding + 1)] = True


def _place_patch(
    canvas: np.ndarray,
    blocked: np.ndarray,
    patch: np.ndarray,
    bg: int,
    rng: np.random.Generator,
    *,
    border: int = 1,
) -> bool:
    patch_mask = patch != int(bg)
    positions = _candidate_positions(canvas.shape, patch.shape, border)
    if not positions:
        return False
    for idx in rng.permutation(len(positions)):
        r0, c0 = positions[int(idx)]
        if not _can_place(blocked, patch_mask, r0, c0):
            continue
        view = canvas[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]]
        view[patch_mask] = patch[patch_mask]
        _mark_blocked(blocked, patch_mask, r0, c0, padding=1)
        return True
    return False


def _count_options(num_colors: int) -> list[list[tuple[int, int]]]:
    out: list[list[tuple[int, int]]] = []
    for eight_counts in combinations(range(2, 6), num_colors):
        ranges = [range(1, min(2, eight) + 1) for eight in eight_counts]
        for extras in product(*ranges):
            four_counts = [eight + extra for eight, extra in zip(eight_counts, extras)]
            if len(set(four_counts)) != num_colors or max(four_counts) > 7:
                continue
            h_conflict = any(
                any(four_counts[i] == eight_counts[j] for j in range(num_colors) if j != i)
                for i in range(num_colors)
            )
            v_conflict = any(
                any(eight_counts[i] == four_counts[j] for j in range(num_colors) if j != i)
                for i in range(num_colors)
            )
            if h_conflict and v_conflict:
                out.append([(int(eight), int(extra)) for eight, extra in zip(eight_counts, extras)])
    return out


COUNT_OPTIONS = {
    2: _count_options(2),
    3: _count_options(3),
}


def _build_scene(
    specs: list[dict],
    panel_shape: tuple[int, int],
    panel_bg: int,
    rng: np.random.Generator,
) -> np.ndarray | None:
    scene = np.full(panel_shape, int(panel_bg), dtype=int)
    blocked = np.zeros(panel_shape, dtype=bool)
    patches: list[np.ndarray] = []
    for spec in specs:
        kinds = [True] * int(spec["extra_pairs"]) + [False] * int(spec["count8"] - spec["extra_pairs"])
        rng.shuffle(kinds)
        for is_pair in kinds:
            if is_pair:
                patches.append(_pair_patch(spec["color"], panel_bg, rng))
            else:
                patches.append(_isolated_patch(spec["color"], panel_bg, rng))

    order = list(rng.permutation(len(patches)))
    order.sort(key=lambda idx: int(patches[idx].shape[0] * patches[idx].shape[1]), reverse=True)
    for idx in order:
        if not _place_patch(scene, blocked, patches[int(idx)], panel_bg, rng, border=1):
            return None
    return scene


def _build_top(specs: list[dict], shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray | None:
    top = np.full(shape, BLACK, dtype=int)
    blocked = np.zeros(shape, dtype=bool)
    bars = [("h", int(spec["count4"])) for spec in specs] + [("v", int(spec["count8"])) for spec in specs]
    rng.shuffle(bars)
    for orient, length in bars:
        patch = np.full((1, length), GREY, dtype=int) if orient == "h" else np.full((length, 1), GREY, dtype=int)
        if not _place_patch(top, blocked, patch, BLACK, rng, border=1):
            return None
    return top


def _analyze_instance(grid: np.ndarray) -> tuple[int, list[dict], dict[int, int], dict[int, int]]:
    panel_start = _panel_start(grid)
    top = grid[:panel_start]
    scene = grid[panel_start:]
    bars = _bar_specs(top)
    _bg, by4, by8, _per_color = _scene_maps(scene)
    return int(panel_start), bars, by4, by8


def _validate_instance(grid: np.ndarray) -> None:
    panel_start, bars, by4, by8 = _analyze_instance(grid)
    top = grid[:panel_start]
    scene = grid[panel_start:]

    assert np.all(scene != BLACK), "scene panel must contain no black"
    assert len(_components_mask(top == GREY, connectivity=8)) == len(bars), "bars must not touch even diagonally"
    assert any(bar["orientation"] == "h" for bar in bars), "need horizontal bars"
    assert any(bar["orientation"] == "v" for bar in bars), "need vertical bars"

    h_lengths = [int(bar["length"]) for bar in bars if bar["orientation"] == "h"]
    v_lengths = [int(bar["length"]) for bar in bars if bar["orientation"] == "v"]
    assert len(set(h_lengths)) == len(h_lengths), "horizontal lengths must be unique"
    assert len(set(v_lengths)) == len(v_lengths), "vertical lengths must be unique"
    assert all(length in by4 for length in h_lengths), "every horizontal bar must have a 4-connected match"
    assert all(length in by8 for length in v_lengths), "every vertical bar must have an 8-connected match"

    h_conflict = any(length in by8 and by8[length] != by4[length] for length in h_lengths)
    v_conflict = any(length in by4 and by4[length] != by8[length] for length in v_lengths)
    assert h_conflict and v_conflict, "orientation must change the answer"

    out = main(grid)
    assert np.array_equal(out[panel_start:], scene), "scene panel must stay unchanged"
    assert not np.array_equal(out, grid), "bars must change color"
    for bar in bars:
        coords = bar["coords"]
        expected = by4[bar["length"]] if bar["orientation"] == "h" else by8[bar["length"]]
        assert bool(np.all(out[coords[:, 0], coords[:, 1]] == expected)), "bar recolor mismatch"


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    num_colors = int(rng.choice(np.array([2, 3], dtype=int), p=np.array([0.35, 0.65])))
    counts = COUNT_OPTIONS[num_colors][int(rng.integers(len(COUNT_OPTIONS[num_colors])))]

    panel_bg = int(rng.choice(np.array(PALETTE, dtype=int)))
    scene_colors = [int(c) for c in rng.choice(np.array([c for c in PALETTE if c != panel_bg], dtype=int), size=num_colors, replace=False)]
    rng.shuffle(scene_colors)

    specs = []
    for color, (count8, extra_pairs) in zip(scene_colors, counts):
        specs.append(
            {
                "color": int(color),
                "count8": int(count8),
                "extra_pairs": int(extra_pairs),
                "count4": int(count8 + extra_pairs),
            }
        )

    max_v = max(int(spec["count8"]) for spec in specs)
    min_panel_w = max(18, max(int(spec["count4"]) for spec in specs) + 7)
    panel_w = int(rng.integers(min_panel_w, 27))
    panel_h = int(rng.integers(13, min(21, 31 - (max_v + 3))))
    top_h_min = max_v + 3
    top_h_max = min(10, 30 - panel_h)
    if top_h_min > top_h_max:
        return None
    top_h = int(rng.integers(top_h_min, top_h_max + 1))

    scene = _build_scene(specs, (panel_h, panel_w), panel_bg, rng)
    if scene is None:
        return None
    top = _build_top(specs, (top_h, panel_w), rng)
    if top is None:
        return None

    grid = np.vstack([top, scene])
    _validate_instance(grid)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        rng = np.random.default_rng()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
