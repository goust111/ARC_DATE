# Combo variant of d59b0160 (index 40)
# combo_spec: D12/0+D18/2
# Name: 颜色位掩码匹配+颜色0可作图例颜色

from __future__ import annotations

import math
import random

import numpy as np

from arc_utils import components_mask, paste, random_free_location


# Design analysis for combo D12/0 + D18/2
# a) Concrete behavior / trace:
#    - Base d59b0160: the 4x4 top-left legend lists target colors, and an external
#      multicolor rectangle is deleted only if its color set contains all legend colors exactly.
#    - D18/2 changes the comparison itself: each legend swatch is read as a bitmask,
#      so legend color L is satisfied when some color C inside the component has (C & L) != 0.
#    - D12/0 allows color 0 to appear in the legend. Under bitmask semantics 0 = 0000,
#      hence (C & 0) is always 0, so a legend containing 0 can never be fully satisfied.
#    - Concrete trace:
#      legend {1,4,8} and component colors {0,5,9}:
#        base rule keeps it because {1,4,8} is not a subset of {0,5,9};
#        fused rule deletes it because 5 matches 1 and 4, while 9 matches 1 and 8.
#      legend {0,1,4} and component colors {0,1,4}:
#        base rule deletes it;
#        fused rule keeps it because the 0-swatch is impossible to satisfy.
# b) Complexity calibration:
#    - To keep the hidden numeric rule from dissolving into noise, nonzero legend colors
#      are restricted to the basis-like set {1,2,4,8}. External components then use a
#      small repeated carrier palette {5,6,8,9} so the same proxy colors reappear across samples.
#    - Samples split into two clear families:
#      1) legend includes 0 -> no component is deleted;
#      2) legend excludes 0 -> at least one component is deleted by bitmask-only evidence.
# c) Hidden-assumption audit against the original main():
#    - `_find_legend(...)` may still parse the original corner frame; that assumption remains valid.
#    - `components_mask(g != bg, ...)` is no longer safe by itself because the legend may contain
#      visible non-bg swatches, including 0-adjacent distractors. The solver now masks out the
#      whole legend square before labeling components.
#    - The old predicate `legend_colors.issubset(component_colors)` is exactly what D18/2 replaces.
#      Every branch that decides deletion must therefore consult the bitmask relation instead.
#    - The solver implicitly assumes each external object remains one 4-connected component even
#      though most of its area is black 0. The generator preserves the original solid-rectangle layout,
#      so 0 continues to act as a visible object fill rather than background.
# d) Placement legality:
#    - The legend still occupies the top-left 4x4 block.
#    - A separate reservation canvas marks that whole 4x4 block as occupied before placing objects.
#      Each object patch is then placed with `random_free_location(..., padding=1)` on that canvas.
#      This guarantees in-bounds placement, no overlap, and at least one background cell between objects.
# e) Visual observability:
#    - The zero-in-legend family is intentionally stark: the output equals the input even when some
#      components visibly contain the nonzero legend swatches, because the black legend entry blocks all deletion.
#    - The zero-free family gives the opposite witness: deleted components often use proxy colors like 5 or 9
#      instead of the exact legend colors, so exact-set matching would not explain the disappearance.
# f) Visual cleanliness:
#    - The overall scene grammar stays close to the source task: one small legend, 4-6 separated rectangles,
#      orange background, black-filled objects with sparse colored accents.
#    - The carrier palette is small and repeated, preventing the board from turning into a random color soup.
# g) Inference / color-rule caveat:
#    - D18/2 inherently depends on numeric color ids, which is weaker than a purely geometric ARC cue.
#      The generator mitigates this by repeating the same carrier colors against the same basis legend colors
#      and by making the "0 disables deletion" case very visible.
# h) New solver assumptions enforced in generation:
#    - `main()` assumes the only intended legend is the canonical top-left framed square.
#    - `main()` assumes no external rectangle intrudes into that square.
#    - `main()` assumes there is a nonempty legend color set.
#    - The generator additionally rejects any sample whose fused output matches the exact-subset baseline,
#      so the produced data actually exercises the combo instead of collapsing back to the source task.


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _find_legend(g: np.ndarray, bg: int) -> tuple[int, int, set[int]]:
    """
    Detect the smallest top-left square whose bottom edge and right edge share one
    non-background border color, then read the interior swatch colors.

    Unlike the base task, color 0 is allowed to remain in the returned set.
    """
    h, w = g.shape
    for sz in range(2, min(h, w) + 1):
        border_color = int(g[sz - 1, sz - 1])
        if border_color == bg:
            continue
        if np.all(g[sz - 1, 0:sz] == border_color) and np.all(g[0:sz, sz - 1] == border_color):
            inside = g[0 : sz - 1, 0 : sz - 1]
            colors = set(map(int, np.unique(inside)))
            colors.discard(bg)
            colors.discard(border_color)
            return int(sz), int(border_color), colors
    raise AssertionError("no legend border found in grid")


def _nonlegend_mask(g: np.ndarray, bg: int, legend_size: int) -> np.ndarray:
    """Foreground mask with the legend square removed."""
    mask = (g != bg).copy()
    mask[:legend_size, :legend_size] = False
    return mask


def _matches_bitmask(component_colors: set[int], legend_colors: set[int]) -> bool:
    """True iff every legend color has some component color with nonzero bitwise AND."""
    for legend_color in legend_colors:
        if not any((component_color & legend_color) != 0 for component_color in component_colors):
            return False
    return True


def _legacy_exact_main(grid: np.ndarray) -> np.ndarray:
    """
    Exact-subset reference semantics used only for anti-degeneracy checks.

    Color 0 is read literally as part of the legend target set, and the legend square
    itself is excluded from candidate components.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    legend_size, _border_color, legend_colors = _find_legend(g, bg)
    assert legend_colors, "legend interior contains no target colors"

    out = g.copy()
    for pts in components_mask(_nonlegend_mask(g, bg, legend_size), connectivity=4):
        cols = {int(g[r, c]) for r, c in pts}
        if legend_colors.issubset(cols):
            out[pts[:, 0], pts[:, 1]] = bg
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Delete connected components whose color set bitwise-satisfies every legend swatch.

    Input structure:
    - The background color is defined by the outer frame and fills most of the grid.
    - The top-left corner contains one 4x4 legend square whose bottom row and right
      column share a green border color.
    - The legend interior lists three swatches; unlike the source task, one of them may be color 0.
    - Outside the legend, each object is a separated 4-connected rectangle made from black 0 fill
      plus a sparse set of colored accent pixels.

    Rule:
    - Read the three legend colors from the legend interior and ignore the legend square itself.
    - For each external 4-connected component, collect its set of colors.
    - Treat each legend color as a bitmask: a legend swatch L is matched when some component color C
      satisfies `(C & L) != 0`.
    - Erase a component only if all legend swatches are matched in that sense.
    - Because color 0 is the mask `0000`, a legend containing 0 can never be fully matched, so no
      external component is deleted in those cases.

    Key insight:
    - The legend is no longer an exact palette checklist. Nonzero swatches act like visible mask tests,
      while a black 0 swatch becomes a deliberate decoy that disables deletion altogether.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    legend_size, _border_color, legend_colors = _find_legend(g, bg)
    assert legend_colors, "legend interior contains no target colors"

    out = g.copy()
    for pts in components_mask(_nonlegend_mask(g, bg, legend_size), connectivity=4):
        cols = {int(g[r, c]) for r, c in pts}
        if _matches_bitmask(cols, legend_colors):
            out[pts[:, 0], pts[:, 1]] = bg
    return out


# --------------- generator ---------------

GRID_H, GRID_W = 16, 16
BG = 7
BORDER = 3
LEGEND_SIZE = 4

BASIS_COLORS = [1, 2, 4, 8]
OBJECT_COLORS = [0, 1, 2, 4, 5, 6, 8, 9]

COMP_COUNTS = [4, 5, 6]
COMP_WEIGHTS = [2, 4, 4]
KEEP_COUNT_ZERO_FREE = 3

SHAPE_WEIGHTS = [5, 3, 2]  # medium, tall, wide
TALL_H = (7, 12)
TALL_W = (2, 4)
WIDE_H = (3, 5)
WIDE_W = (5, 9)
MED_H = (3, 7)
MED_W = (2, 8)

DENSITY_LO = 0.10
DENSITY_HI = 0.42
MAX_RETRIES = 800


def _sample_targets() -> list[int]:
    """Choose either {0 + two basis colors} or three basis colors."""
    if random.random() < 0.5:
        targets = [0] + random.sample(BASIS_COLORS, 2)
    else:
        targets = random.sample(BASIS_COLORS, 3)
    random.shuffle(targets)
    return targets


def _place_legend(grid: np.ndarray, targets: list[int]) -> None:
    """Draw the 4x4 legend and place three distinct swatches in the 3x3 interior."""
    s = LEGEND_SIZE
    grid[:s, :s] = BG
    grid[s - 1, :s] = BORDER
    grid[:s, s - 1] = BORDER

    isolated = random.choice([(0, 0), (0, 1), (1, 0), (1, 1)])
    border_side = [(0, 2), (1, 2), (2, 0), (2, 1), (2, 2)]
    side_choices = [
        p for p in border_side
        if abs(p[0] - isolated[0]) + abs(p[1] - isolated[1]) != 1
    ]
    positions = [isolated] + random.sample(side_choices, 2)
    random.shuffle(positions)
    for color, (r, c) in zip(targets, positions):
        grid[r, c] = color


def _component_dims(min_required_colors: int) -> tuple[int, int]:
    """Sample a rectangle large enough to host its mandatory colored pixels."""
    min_area = max(6, min_required_colors * 3)
    for _ in range(80):
        mode = random.choices(["medium", "tall", "wide"], weights=SHAPE_WEIGHTS)[0]
        if mode == "tall":
            h, w = random.randint(*TALL_H), random.randint(*TALL_W)
        elif mode == "wide":
            h, w = random.randint(*WIDE_H), random.randint(*WIDE_W)
        else:
            h, w = random.randint(*MED_H), random.randint(*MED_W)
        if h * w >= min_area:
            return h, w
    raise ValueError("failed to sample component dimensions")


def _proxy_cover_colors(targets: tuple[int, ...]) -> list[int]:
    """Return a small carrier palette whose bitwise matches cover the requested basis targets."""
    key = tuple(sorted(targets))
    mapping = {
        (1,): [1],
        (2,): [2],
        (4,): [4],
        (8,): [8],
        (1, 2): [1, 2],
        (1, 4): [5],
        (1, 8): [9],
        (2, 4): [6],
        (2, 8): [6, 8],
        (4, 8): [6, 8],
        (1, 2, 4): [5, 6],
        (1, 2, 8): [6, 9],
        (1, 4, 8): [5, 9],
        (2, 4, 8): [6, 8],
    }
    if key not in mapping:
        raise ValueError(f"unsupported target combination: {key}")
    return list(mapping[key])


def _colors_avoiding_mask(mask_color: int) -> list[int]:
    """All nonzero object colors whose bitwise AND with mask_color is zero."""
    return [c for c in OBJECT_COLORS if c != 0 and (c & mask_color) == 0]


def _build_patch(
    h: int,
    w: int,
    required_colors: list[int],
    optional_colors: list[int],
) -> np.ndarray:
    """Create one solid-black rectangle and sprinkle colored accents."""
    area = h * w
    lo = max(len(required_colors), math.ceil(area * DENSITY_LO))
    hi = max(lo, math.floor(area * DENSITY_HI))
    n_colored = random.randint(lo, hi)

    coords = random.sample([(r, c) for r in range(h) for c in range(w)], n_colored)
    colors = list(required_colors)
    filler = optional_colors or required_colors
    while len(colors) < n_colored:
        colors.append(random.choice(filler))
    random.shuffle(colors)

    patch = np.zeros((h, w), dtype=int)
    for (r, c), color in zip(coords, colors):
        patch[r, c] = int(color)
    return patch


def _make_proxy_deleted_spec(targets: list[int]) -> dict[str, object]:
    """
    A component deleted only by bitmask semantics, not by exact-set matching.

    We require the proxy cover colors but forbid any missing exact target color from
    appearing in the accents, ensuring the exact baseline cannot delete it.
    """
    proxy = _proxy_cover_colors(tuple(sorted(targets)))
    missing_exact = [t for t in targets if t not in proxy]
    assert missing_exact, "proxy witness must omit at least one exact target color"
    optional = [c for c in OBJECT_COLORS if c != 0 and c not in missing_exact]
    return {"required": proxy, "optional": optional}


def _make_exact_zero_witness_spec(nonzero_targets: list[int]) -> dict[str, object]:
    """A component that would be deleted by exact-set semantics when 0 is in the legend."""
    optional = [c for c in OBJECT_COLORS if c != 0]
    return {"required": list(nonzero_targets), "optional": optional}


def _make_proxy_zero_witness_spec(nonzero_targets: list[int]) -> dict[str, object]:
    """A proxy-colored component that matches the nonzero legend masks but is saved by the 0 decoy."""
    proxy = _proxy_cover_colors(tuple(sorted(nonzero_targets)))
    missing_exact = [t for t in nonzero_targets if t not in proxy]
    optional = [c for c in OBJECT_COLORS if c != 0 and c not in missing_exact]
    return {"required": proxy, "optional": optional}


def _make_kept_spec(targets: list[int]) -> dict[str, object]:
    """
    A component guaranteed to miss one chosen legend mask under bitmask semantics.

    Its accent colors all satisfy `(color & missed_target) == 0`.
    """
    missed_target = random.choice(targets)
    remaining = tuple(sorted(t for t in targets if t != missed_target))
    required = _proxy_cover_colors(remaining)
    optional = _colors_avoiding_mask(missed_target)
    return {"required": required, "optional": optional}


def _component_specs(targets: list[int]) -> tuple[list[dict[str, object]], bool]:
    """
    Build a component plan and return `(specs, expect_no_deletion)`.

    If 0 is in the legend, the fused rule keeps everything.
    Otherwise, exactly `KEEP_COUNT_ZERO_FREE` components are kept and the rest are
    proxy-only deletions.
    """
    if 0 in targets:
        nonzero_targets = sorted(t for t in targets if t != 0)
        component_count = random.choices(COMP_COUNTS, weights=COMP_WEIGHTS)[0]
        specs = [
            _make_exact_zero_witness_spec(nonzero_targets),
            _make_proxy_zero_witness_spec(nonzero_targets),
        ]
        while len(specs) < component_count:
            specs.append(_make_kept_spec(nonzero_targets))
        random.shuffle(specs)
        return specs, True

    component_count = random.choices(COMP_COUNTS, weights=COMP_WEIGHTS)[0]
    deleted_count = component_count - KEEP_COUNT_ZERO_FREE
    specs = [_make_proxy_deleted_spec(targets) for _ in range(deleted_count)]
    while len(specs) < component_count:
        specs.append(_make_kept_spec(targets))
    random.shuffle(specs)
    return specs, False


def _place_components(grid: np.ndarray, reserve: np.ndarray, specs: list[dict[str, object]]) -> bool:
    """Place every component patch using the reservation canvas for collision checks."""
    for spec in specs:
        required = list(spec["required"])
        optional = list(spec["optional"])
        h, w = _component_dims(len(required))
        patch = _build_patch(h, w, required, optional)
        try:
            r, c = random_free_location(
                reserve,
                np.zeros_like(patch),
                transparent=BG,
                padding=1,
                padding_connectivity=8,
            )
        except ValueError:
            return False
        paste(grid, patch, r, c)
        paste(reserve, np.zeros_like(patch), r, c)
    return True


def _deleted_component_count(inp: np.ndarray, out: np.ndarray) -> int:
    """Count how many external components disappear entirely from input to output."""
    bg = _bg_boundary(inp)
    legend_size, _border_color, _legend_colors = _find_legend(inp, bg)
    comps = components_mask(_nonlegend_mask(inp, bg, legend_size), connectivity=4)
    deleted = 0
    for pts in comps:
        if np.all(out[pts[:, 0], pts[:, 1]] == bg):
            deleted += 1
    return int(deleted)


def _gen_attempt() -> np.ndarray | None:
    """Build one candidate input and reject degenerate samples."""
    grid = np.full((GRID_H, GRID_W), BG, dtype=int)
    reserve = np.full((GRID_H, GRID_W), BG, dtype=int)

    targets = _sample_targets()
    _place_legend(grid, targets)
    reserve[:LEGEND_SIZE, :LEGEND_SIZE] = BORDER

    specs, expect_no_deletion = _component_specs(targets)
    if not _place_components(grid, reserve, specs):
        return None

    fused_out = main(grid)
    legacy_out = _legacy_exact_main(grid)
    if np.array_equal(fused_out, legacy_out):
        return None

    deleted_now = _deleted_component_count(grid, fused_out)
    deleted_legacy = _deleted_component_count(grid, legacy_out)
    if expect_no_deletion:
        if deleted_now != 0:
            return None
        if deleted_legacy == 0:
            return None
    else:
        if deleted_now <= 0:
            return None
        if deleted_legacy != 0:
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
