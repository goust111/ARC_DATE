from __future__ import annotations

from collections import defaultdict

import numpy as np

from arc_utils import get_bg, in_bounds, paste


Template = tuple[int, tuple[tuple[int, int], ...]]


def _infer_templates(g: np.ndarray, bg: int) -> dict[int, Template]:
    candidates: list[tuple[int, int, tuple[tuple[int, int], ...]]] = []
    seen: set[tuple[int, int, tuple[tuple[int, int], ...]]] = set()
    height, width = map(int, g.shape)
    for r in range(height):
        for c in range(width):
            center = int(g[r, c])
            if center == bg:
                continue

            offsets: list[tuple[int, int]] = []
            colors: list[int] = []
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr, cc = r + dr, c + dc
                    if in_bounds(g.shape, rr, cc) and int(g[rr, cc]) != bg:
                        offsets.append((dr, dc))
                        colors.append(int(g[rr, cc]))

            if len(offsets) < 2 or len(set(colors)) != 1:
                continue
            outer = colors[0]
            if outer == center:
                continue
            offset_set = frozenset(offsets)
            if any((-dr, -dc) not in offset_set for dr, dc in offset_set):
                continue

            signature = (center, outer, tuple(sorted(offset_set)))
            if signature not in seen:
                seen.add(signature)
                candidates.append(signature)

    assert candidates, "expected at least one complete centered motif"

    by_center: dict[int, set[Template]] = defaultdict(set)
    outer_colors: set[int] = set()
    for center, outer, offsets in candidates:
        by_center[center].add((outer, offsets))
        outer_colors.add(outer)

    assert all(len(templates) == 1 for templates in by_center.values()), (
        "each center color must define a unique motif",
        by_center,
    )
    assert set(by_center).isdisjoint(outer_colors), "center and shell colors must have distinct roles"
    return {center: next(iter(templates)) for center, templates in by_center.items()}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete every partial centered motif using the complete motifs visible in the same grid.

    Input structure:
    - The background contains one or more complete 3x3 motifs, each with a center color and a symmetric same-color shell mask.
    - Other copies of those motifs are partial: either the center pixel is present without its shell, or the shell is present without its center.
    - For each center color, the complete motif uniquely determines the shell color and shell offsets.

    Rule:
    - Infer every complete motif as center color -> (shell color, shell offsets).
    - Around every pixel with a known center color, fill missing shell-offset background pixels with the shell color.
    - At every background pixel whose surrounding offsets already contain a complete shell, fill the center with that motif's center color.

    Key insight:
    The complete motifs are templates that make center-only and shell-only partial copies two views of the same local sprite.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = int(get_bg(g))
    out = g.copy()
    height, width = map(int, g.shape)
    templates = _infer_templates(g, bg)

    def paint(r: int, c: int, color: int) -> None:
        original = int(g[r, c])
        current = int(out[r, c])
        if original != bg:
            assert original == color, "cannot overwrite an input foreground pixel"
        if current != bg:
            assert current == color, "motif completions must not conflict"
        out[r, c] = color

    for center, (outer, offsets) in templates.items():
        for r, c in np.argwhere(g == center):
            r = int(r)
            c = int(c)
            for dr, dc in offsets:
                rr, cc = r + dr, c + dc
                assert in_bounds(g.shape, rr, cc), "motif copy must fit inside the grid"
                value = int(g[rr, cc])
                assert value in (bg, outer), "center-marked copies can only lack shell pixels"
                if value == bg:
                    paint(rr, cc, outer)

    for r in range(height):
        for c in range(width):
            if int(g[r, c]) != bg:
                continue
            matches: list[int] = []
            for center, (outer, offsets) in templates.items():
                if all(
                    in_bounds(g.shape, r + dr, c + dc) and int(g[r + dr, c + dc]) == outer
                    for dr, dc in offsets
                ):
                    matches.append(center)
            if matches:
                assert len(set(matches)) == 1, "shell-marked copies must choose one center color"
                paint(r, c, matches[0])

    return out

# --------------- generator (o5x) ---------------

# Shape catalogue; offsets are relative to the center pixel.
_GEN_SHAPE_OFFSETS = {
    "h_bar": ((0, -1), (0, 1)),
    "v_bar": ((-1, 0), (1, 0)),
    "plus": ((-1, 0), (0, -1), (0, 1), (1, 0)),
    "x_corners": ((-1, -1), (-1, 1), (1, -1), (1, 1)),
}

# Sampling bounds; numpy randint high values are exclusive.
_GEN_HEIGHT_BOUNDS = (13, 20)
_GEN_WIDTH_BOUNDS = (12, 23)
_GEN_TEMPLATE_COUNT_BOUNDS = (1, 4)
_GEN_TOTAL_PARTIAL_BOUNDS = (3, 7)
_GEN_PARTIALS_PER_TEMPLATE_BOUNDS = (1, 3)
_GEN_SHELL_ONLY_BOUNDS = (1, 2)
_GEN_COLOR_VALUES = np.arange(1, 10)

# Geometry and retry limits.
_GEN_PATCH_SHAPE = (3, 3)
_GEN_PATCH_CENTER = 1
_GEN_MOTIF_MARGIN = 1
_GEN_CENTER_GAP_MIN = 4
_GEN_PARTIAL_TEMPLATE_DISTANCE_MIN = 7
_GEN_PLACEMENT_NODE_BUDGET = 25000
_GEN_ATTEMPT_LIMIT = 300

_GEN_TEMPLATE_KIND = "template"
_GEN_CENTER_ONLY_KIND = "center_only"
_GEN_SHELL_ONLY_KIND = "shell_only"

_GenTemplate = tuple[int, int, tuple[tuple[int, int], ...]]
_GenMotif = tuple[int, str]

def _gen_patch(center_color: int, shell_color: int, offsets: tuple[tuple[int, int], ...]) -> np.ndarray:
    patch = np.zeros(_GEN_PATCH_SHAPE, dtype=int)
    patch[_GEN_PATCH_CENTER, _GEN_PATCH_CENTER] = int(center_color)
    for dr, dc in offsets:
        patch[_GEN_PATCH_CENTER + dr, _GEN_PATCH_CENTER + dc] = int(shell_color)
    return patch


def _gen_partial_counts(n_templates: int, total_partials: int) -> tuple[int, ...]:
    """Sample a feasible per-template partial allocation."""
    options: list[tuple[int, ...]] = []
    min_per_template, max_per_template = _GEN_PARTIALS_PER_TEMPLATE_BOUNDS

    def rec(i: int, remaining: int, acc: list[int]) -> None:
        if i == n_templates:
            if remaining == 0:
                options.append(tuple(acc))
            return
        slots_left = n_templates - i - 1
        for count in range(min_per_template, max_per_template + 1):
            if min_per_template * slots_left <= remaining - count <= max_per_template * slots_left:
                acc.append(count)
                rec(i + 1, remaining - count, acc)
                acc.pop()

    rec(0, total_partials, [])
    return options[int(np.random.randint(len(options)))]


def _gen_lattice_capacity(height: int, width: int) -> int:
    usable_h = height - 2 * _GEN_MOTIF_MARGIN
    usable_w = width - 2 * _GEN_MOTIF_MARGIN
    return ((usable_h + _GEN_CENTER_GAP_MIN - 1) // _GEN_CENTER_GAP_MIN) * ((usable_w + _GEN_CENTER_GAP_MIN - 1) // _GEN_CENTER_GAP_MIN)


def _gen_far_enough(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1])) >= _GEN_CENTER_GAP_MIN


def _gen_sample_templates(n_templates: int) -> list[_GenTemplate]:
    shape_names = list(_GEN_SHAPE_OFFSETS)
    chosen_shapes = [shape_names[int(i)] for i in np.random.choice(len(shape_names), n_templates, replace=False)]
    colors = [int(c) for c in np.random.permutation(_GEN_COLOR_VALUES)[: 2 * n_templates]]
    center_colors = colors[:n_templates]
    shell_colors = colors[n_templates:]
    return [
        (center_colors[i], shell_colors[i], _GEN_SHAPE_OFFSETS[shape_name])
        for i, shape_name in enumerate(chosen_shapes)
    ]


def _gen_build_motif_plan(n_templates: int, partial_counts: tuple[int, ...]) -> tuple[list[_GenMotif], list[_GenMotif]]:
    """Return all motif records plus the partial-only suffix."""
    partial_template_ids = [template_i for template_i, count in enumerate(partial_counts) for _ in range(count)]
    n_shell_only = int(np.random.randint(_GEN_SHELL_ONLY_BOUNDS[0], min(_GEN_SHELL_ONLY_BOUNDS[1], len(partial_template_ids)) + 1))
    shell_slot_ids = {int(i) for i in np.random.choice(len(partial_template_ids), n_shell_only, replace=False)}

    motifs = [(template_i, _GEN_TEMPLATE_KIND) for template_i in range(n_templates)]
    partial_meta = []
    for slot_i, template_i in enumerate(partial_template_ids):
        kind = _GEN_SHELL_ONLY_KIND if slot_i in shell_slot_ids else _GEN_CENTER_ONLY_KIND
        partial_meta.append((template_i, kind))
        motifs.append((template_i, kind))
    return motifs, partial_meta


def _gen_place_centers(height: int, width: int, motifs: list[_GenMotif]) -> list[tuple[int, int]] | None:
    """Backtrack centers in motif order, returning None if the node budget is exhausted."""
    candidates = [(r, c) for r in range(_GEN_MOTIF_MARGIN, height - _GEN_MOTIF_MARGIN) for c in range(_GEN_MOTIF_MARGIN, width - _GEN_MOTIF_MARGIN)]
    placements: dict[int, tuple[int, int]] = {}
    placed: list[tuple[int, int]] = []
    node_budget = [_GEN_PLACEMENT_NODE_BUDGET]

    def rec(i: int) -> bool:
        if i == len(motifs):
            return True
        node_budget[0] -= 1
        if node_budget[0] <= 0:
            return False

        order = np.random.permutation(len(candidates))
        template_i, kind = motifs[i]
        for cand_i in order:
            r, c = candidates[int(cand_i)]
            here = (int(r), int(c))
            if any(not _gen_far_enough(here, other) for other in placed):
                continue
            if kind != _GEN_TEMPLATE_KIND:
                template_center = placements[template_i]
                dist = abs(here[0] - template_center[0]) + abs(here[1] - template_center[1])
                if dist < _GEN_PARTIAL_TEMPLATE_DISTANCE_MIN:
                    continue

            placements[i] = here
            placed.append(here)
            if rec(i + 1):
                return True
            placed.pop()
            del placements[i]
        return False

    if not rec(0):
        return None
    return [placements[i] for i in range(len(motifs))]


def _gen_render_output(height: int, width: int, templates: list[_GenTemplate], motifs: list[_GenMotif], centers: list[tuple[int, int]]) -> np.ndarray:
    output = np.zeros((height, width), dtype=int)
    patches = [_gen_patch(center, shell, offsets) for center, shell, offsets in templates]
    for (template_i, _kind), (r, c) in zip(motifs, centers):
        paste(output, patches[template_i], r - _GEN_MOTIF_MARGIN, c - _GEN_MOTIF_MARGIN, transparent=0)
    return output


def _gen_derive_input(output: np.ndarray, templates: list[_GenTemplate], partial_meta: list[_GenMotif], centers: list[tuple[int, int]], n_templates: int) -> np.ndarray:
    inp = output.copy()
    for (template_i, kind), (r, c) in zip(partial_meta, centers[n_templates:]):
        _center, _shell, offsets = templates[template_i]
        if kind == _GEN_CENTER_ONLY_KIND:
            for dr, dc in offsets:
                inp[r + int(dr), c + int(dc)] = 0
        else:
            inp[r, c] = 0
    return inp


def _gen_matches_oracle(inp: np.ndarray, output: np.ndarray) -> bool:
    try:
        return bool(np.array_equal(main(inp), output))
    except Exception:
        return False


def _gen_attempt() -> np.ndarray | None:
    height = int(np.random.randint(*_GEN_HEIGHT_BOUNDS))
    width = int(np.random.randint(*_GEN_WIDTH_BOUNDS))
    n_templates = int(np.random.randint(*_GEN_TEMPLATE_COUNT_BOUNDS))
    min_total = max(_GEN_TOTAL_PARTIAL_BOUNDS[0], n_templates)
    max_total = min(_GEN_TOTAL_PARTIAL_BOUNDS[1], _GEN_PARTIALS_PER_TEMPLATE_BOUNDS[1] * n_templates)
    total_partials = int(np.random.randint(min_total, max_total + 1))
    if n_templates + total_partials > _gen_lattice_capacity(height, width):
        return None

    templates = _gen_sample_templates(n_templates)
    partial_counts = _gen_partial_counts(n_templates, total_partials)
    motifs, partial_meta = _gen_build_motif_plan(n_templates, partial_counts)
    centers = _gen_place_centers(height, width, motifs)
    if centers is None:
        return None

    output = _gen_render_output(height, width, templates, motifs, centers)
    inp = _gen_derive_input(output, templates, partial_meta, centers, n_templates)
    if _gen_matches_oracle(inp, output):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPT_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 72322fa7 input")
