# Combo variant of 1818057f (index 13)
# combo_spec: D6/2+D11/1
# Name: 连通分量大小双重编码

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.append(str(_ROOT))

from arc_utils import get_bg


# Design analysis
# a) Concrete behavior trace:
#    The local object being detected is still the same thin '+' motif: a foreground
#    center with four orthogonal foreground neighbors. Suppose one input contains four
#    disconnected yellow components with sizes 13, 27, 45, and 16, and each component
#    contains exactly one embedded '+'. The source task would detect all four centers
#    and repaint every 5-cell footprint teal. The fused task still detects the same
#    centers, but now it first measures the size of the containing 4-connected
#    component: sizes 13 and 16 become green, size 27 stays teal, and size 45 becomes
#    orange. The D11 ingredient matters only as the bookkeeping layer that says each
#    '+' must read the size of its own disconnected component; it does not introduce a
#    separate component-order palette on top of the size rule.
# b) Complexity calibration:
#    The rule uses only three size bands, so the hypothesis space stays small enough
#    for ARC. To keep the fusion visible, every generated sample contains at least four
#    plus-bearing components: one small, one medium, one large, and one extra small or
#    medium component. This guarantees all three colors appear while also showing that
#    two different disconnected components can share the same recolor when they fall in
#    the same size band. That blocks a simpler "each component gets its own color"
#    interpretation.
# c) Hidden assumptions in the original main():
#    - `fg = colors[0]` assumes a single non-background input color. The generator
#      preserves that: inputs use only bg=2 and fg=4.
#    - The original unconditional `out[...] = 8` assumed recolor was global. That is
#      no longer valid, so the solver must label components and look up a size band.
#    - The original scan tolerated repeated writes because all matches shared one
#      target color. After fusion, overlapping '+' motifs from different components
#      would be impossible anyway, but overlapping motifs inside one component could
#      still obscure reasoning. The generator therefore keeps exactly one '+' center
#      per plus-bearing component and separates all components by an orthogonal moat.
#    - Any dependency on component discovery order would be a hidden, nonvisual rule.
#      The fused solver avoids this entirely: only visible component size matters.
# d) Placement legality:
#    Each plus-bearing component is built as a 5-cell cross plus a one-cell-wide snake
#    tail of controlled length, which keeps the component 4-connected and gives an
#    exact target size. Placement checks enforce in-bounds coordinates, no overlap, and
#    at least one orthogonal background cell between any two components. Distractors
#    are short paths that are also moat-separated and explicitly rejected if they would
#    contain a '+'.
# e) Visual observability:
#    Humans can see three visibly different component scales and the repeated color on
#    two separate same-band components. Component position is randomized, so color
#    cannot be explained by top-to-bottom or left-to-right ordering. Across examples,
#    the only stable cue is the size of the disconnected foreground region containing
#    the '+'. That is exactly the intended fused concept.
# f) Visual cleanliness:
#    Components are thin snakes rather than dense blobs, so the embedded '+' remains
#    legible. The moat rule prevents touching silhouettes and keeps ownership of each
#    recolored footprint unambiguous. Distractors stay short and sparse so they add
#    noise without overwhelming the main components.
# g) Change inferability:
#    Only three output colors are introduced: green for small components, teal for
#    medium ones, and orange for large ones. Every sample uses all three colors and
#    includes at least one duplicated band across disconnected components, which gives
#    repeated structural evidence for the size-to-color mapping rather than arbitrary
#    one-off recolors.
#
# Solver assumption audit:
# - `main()` assumes component size is measured on the unrecolored foreground mask;
#   `generate_input()` always validates using the raw fg grid.
# - `main()` assumes every detected '+' belongs to one well-defined 4-connected
#   foreground component; the generator enforces orthogonal separation between all
#   components so component ids are stable.
# - `main()` does not try to resolve competing colors for overlapping '+' motifs; the
#   generator constructs one '+' center per plus-bearing component and rejects any
#   accidental extra centers in distractors.
# - `main()` assumes size thresholds alone determine output color; the generator
#   randomizes component placement so no hidden positional ordering can substitute for
#   the intended rule.


SMALL_COLOR = 3
MEDIUM_COLOR = 8
LARGE_COLOR = 7
BG = 2
FG = 4
ORTHO = ((1, 0), (-1, 0), (0, 1), (0, -1))
PLUS = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))


def _color_for_size(size: int) -> int:
    if size < 20:
        return SMALL_COLOR
    if size < 40:
        return MEDIUM_COLOR
    return LARGE_COLOR


def _band_for_size(size: int) -> str:
    if size < 20:
        return "small"
    if size < 40:
        return "medium"
    return "large"


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, list[int]]:
    H, W = mask.shape
    comp_ids = np.full((H, W), -1, dtype=int)
    sizes: list[int] = []

    for r in range(H):
        for c in range(W):
            if not bool(mask[r, c]) or int(comp_ids[r, c]) != -1:
                continue
            cid = len(sizes)
            stack = [(int(r), int(c))]
            comp_ids[r, c] = cid
            size = 0
            while stack:
                cr, cc = stack.pop()
                size += 1
                for dr, dc in ORTHO:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W and bool(mask[nr, nc]) and int(comp_ids[nr, nc]) == -1:
                        comp_ids[nr, nc] = cid
                        stack.append((int(nr), int(nc)))
            sizes.append(size)
    return comp_ids, sizes


def _find_plus_centers(mask: np.ndarray) -> list[tuple[int, int]]:
    H, W = mask.shape
    centers: list[tuple[int, int]] = []
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if not bool(mask[r, c]):
                continue
            if all(bool(mask[r + dr, c + dc]) for dr, dc in ORTHO):
                centers.append((int(r), int(c)))
    return centers


def _plus_centers_in_coords(coords: set[tuple[int, int]]) -> list[tuple[int, int]]:
    centers: list[tuple[int, int]] = []
    for r, c in coords:
        if all((r + dr, c + dc) in coords for dr, dc in ORTHO):
            centers.append((int(r), int(c)))
    return sorted(centers)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect embedded foreground '+' motifs and recolor them by the size of their own connected component.

    Input structure:
    - The background is the most common color.
    - Exactly one foreground color forms several disconnected 4-connected components.
    - Each target '+' is embedded inside a larger foreground component instead of appearing as an isolated 5-cell cross.

    Rule:
    - A '+' center is any foreground pixel whose 4 orthogonal neighbors are also foreground.
    - Label the foreground into 4-connected components and measure the size of the component containing each '+' center.
    - Recolor that center and its 4 orthogonal neighbors to green when the component size is below 20,
      to teal when the size is from 20 through 39, and to orange when the size is at least 40.
    - Leave every other pixel unchanged.

    Key insight:
    The shape test is still local, but the output color is no longer universal and is not tied to component order.
    Each '+' must inherit its recolor from a nonlocal measurement: the total size of its own disconnected foreground region.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    colors = sorted(set(map(int, g.flatten())) - {bg})
    if not colors:
        return out
    fg = int(colors[0])

    fg_mask = g == fg
    centers = _find_plus_centers(fg_mask)
    if not centers:
        return out

    comp_ids, comp_sizes = _label_components(fg_mask)
    for r, c in centers:
        color = _color_for_size(comp_sizes[int(comp_ids[r, c])])
        for dr, dc in PLUS:
            out[r + dr, c + dc] = color
    return out


def _original_style_output(grid: np.ndarray, fg: int) -> np.ndarray:
    g = np.array(grid, dtype=int)
    out = g.copy()
    H, W = g.shape
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if int(g[r, c]) != fg:
                continue
            if all(int(g[r + dr, c + dc]) == fg for dr, dc in ORTHO):
                for dr, dc in PLUS:
                    out[r + dr, c + dc] = MEDIUM_COLOR
    return out


def _rotate_coords(coords: set[tuple[int, int]], turns: int) -> set[tuple[int, int]]:
    out = set(coords)
    for _ in range(turns % 4):
        out = {(int(c), int(-r)) for r, c in out}
    return out


def _compact_path(length: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    if length <= 0:
        return []

    coords = [(0, 0)]
    remaining = length - 1
    row, col = 0, 0
    direction = 1

    while remaining > 0:
        max_run = min(6, remaining)
        min_run = 1 if remaining == 1 else min(2, max_run)
        run = int(rng.integers(min_run, max_run + 1))
        for _ in range(run):
            col += direction
            coords.append((int(row), int(col)))
        remaining -= run
        if remaining == 0:
            break

        connector = 2 if remaining >= 2 else 1
        for _ in range(connector):
            row += 1
            coords.append((int(row), int(col)))
        remaining -= connector
        direction *= -1

    return coords


def _sample_size(rng: np.random.Generator, band: str, *, prefer_compact_large: bool = False) -> int:
    if band == "small":
        choices = np.array([10, 11, 12, 14, 16, 17, 18, 19], dtype=int)
        return int(rng.choice(choices))
    if band == "medium":
        choices = np.array([20, 21, 23, 26, 30, 34, 37, 39], dtype=int)
        return int(rng.choice(choices))
    if band == "large":
        if prefer_compact_large:
            choices = np.array([40, 41, 42, 43], dtype=int)
        else:
            choices = np.array([40, 41, 43, 45], dtype=int)
        return int(rng.choice(choices))
    raise ValueError(f"unknown band: {band}")


def _make_plus_component(target_size: int, rng: np.random.Generator) -> set[tuple[int, int]] | None:
    if target_size <= 5:
        return None

    for _ in range(60):
        coords = set(PLUS)
        for pr, pc in _compact_path(target_size - 5, rng):
            coords.add((int(pr), int(pc + 2)))
        coords = _rotate_coords(coords, int(rng.integers(0, 4)))
        if _plus_centers_in_coords(coords) == [(0, 0)] and len(coords) == target_size:
            return coords
    return None


def _make_distractor(length: int, rng: np.random.Generator) -> set[tuple[int, int]]:
    coords = set(_compact_path(length, rng))
    return _rotate_coords(coords, int(rng.integers(0, 4)))


def _can_place_component(
    grid: np.ndarray,
    coords: set[tuple[int, int]],
    anchor_r: int,
    anchor_c: int,
    bg: int,
) -> bool:
    H, W = grid.shape
    global_cells = {(int(anchor_r + r), int(anchor_c + c)) for r, c in coords}

    for gr, gc in global_cells:
        if not (0 <= gr < H and 0 <= gc < W):
            return False
        if int(grid[gr, gc]) != bg:
            return False

    for gr, gc in global_cells:
        for dr, dc in ORTHO:
            nr, nc = gr + dr, gc + dc
            if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in global_cells and int(grid[nr, nc]) != bg:
                return False
    return True


def _place_component(
    grid: np.ndarray,
    coords: set[tuple[int, int]],
    value: int,
    bg: int,
    rng: np.random.Generator,
    *,
    needs_center_margin: bool,
) -> bool:
    H, W = grid.shape
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    low_r = -min(rows)
    high_r = H - 1 - max(rows)
    low_c = -min(cols)
    high_c = W - 1 - max(cols)

    if needs_center_margin:
        low_r = max(low_r, 1)
        high_r = min(high_r, H - 2)
        low_c = max(low_c, 1)
        high_c = min(high_c, W - 2)

    if low_r > high_r or low_c > high_c:
        return False

    anchors = [(int(r), int(c)) for r in range(low_r, high_r + 1) for c in range(low_c, high_c + 1)]
    rng.shuffle(anchors)
    for anchor_r, anchor_c in anchors:
        if not _can_place_component(grid, coords, anchor_r, anchor_c, bg):
            continue
        for dr, dc in coords:
            grid[anchor_r + dr, anchor_c + dc] = value
        return True
    return False


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    repeated_band = str(rng.choice(["small", "medium", "large"]))
    if repeated_band == "large":
        H = int(rng.integers(21, 23))
        W = int(rng.integers(21, 23))
    else:
        H = int(rng.integers(19, 23))
        W = int(rng.integers(19, 23))
    area = H * W
    target_fg = int(area * rng.uniform(0.28, 0.38))

    size_bands = ["small", "medium", "large", repeated_band]
    rng.shuffle(size_bands)
    prefer_compact_large = repeated_band == "large"
    sizes = [_sample_size(rng, band, prefer_compact_large=prefer_compact_large) for band in size_bands]

    if sum(sizes) >= area // 2:
        return None

    components: list[tuple[str, int, set[tuple[int, int]]]] = []
    for band, size in zip(size_bands, sizes):
        coords = _make_plus_component(size, rng)
        if coords is None:
            return None
        components.append((band, size, coords))

    grid = np.full((H, W), BG, dtype=int)
    for _, size, coords in sorted(components, key=lambda item: item[1], reverse=True):
        if not _place_component(grid, coords, FG, BG, rng, needs_center_margin=True):
            return None

    distractor_budget = max(0, target_fg - int(np.sum(grid != BG)))
    distractor_tries = 0
    while distractor_budget > 0 and distractor_tries < 12:
        distractor_tries += 1
        length = int(rng.integers(1, min(5, distractor_budget) + 1))
        coords = _make_distractor(length, rng)
        if _plus_centers_in_coords(coords):
            continue
        if _place_component(grid, coords, FG, BG, rng, needs_center_margin=False):
            distractor_budget -= len(coords)

    fg_mask = grid == FG
    centers = _find_plus_centers(fg_mask)
    if len(centers) != len(components):
        return None

    comp_ids, comp_sizes = _label_components(fg_mask)
    plus_comp_ids = {int(comp_ids[r, c]) for r, c in centers}
    if len(plus_comp_ids) != len(components):
        return None

    seen_bands = [_band_for_size(comp_sizes[int(comp_ids[r, c])]) for r, c in centers]
    if set(seen_bands) != {"small", "medium", "large"}:
        return None
    if len(seen_bands) <= len(set(seen_bands)):
        return None

    total_fg = int(np.sum(fg_mask))
    if total_fg >= area // 2:
        return None

    out = main(grid)
    added_colors = set(map(int, out.flatten())) - {BG, FG}
    if added_colors != {SMALL_COLOR, MEDIUM_COLOR, LARGE_COLOR}:
        return None
    if len(plus_comp_ids) <= len(added_colors):
        return None
    if np.array_equal(out, _original_style_output(grid, FG)):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        candidate = _gen_attempt()
        if candidate is not None:
            return candidate
    raise RuntimeError("generate_input failed after 500 attempts")
