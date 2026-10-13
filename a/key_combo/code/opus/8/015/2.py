from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import crop_bbox, find_components, get_bg, paste, scale_sprite


BG = 0
TEMPLATE_SHAPES = ((2, 2), (2, 3), (3, 2), (3, 3))
IDENTIFIER_COLORS = (1, 2, 3, 4)
NEUTRAL_COLORS = (5, 6, 7, 8, 9)
GRID_W_RANGE = (16, 24)
TOP_ROW_RANGE = (0, 2)
SEED_GAP_RANGE = (4, 7)
BOTTOM_MARGIN_RANGE = (1, 3)
GEN_ATTEMPTS = 500


def _dense_patch_from_component(grid: np.ndarray, comp: dict, bg: int) -> np.ndarray | None:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if (h, w) not in TEMPLATE_SHAPES:
        return None
    if int(comp["area"]) != h * w:
        return None
    patch = crop_bbox(grid, comp["bbox"])
    if np.any(patch == int(bg)):
        return None
    return patch


def _extract_templates_and_seeds(grid: np.ndarray) -> tuple[list[np.ndarray], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    components = find_components(g, bg=bg, connectivity=4, monochromatic=False)

    templates: list[dict] = []
    seeds: list[tuple[int, int]] = []
    for comp in components:
        if int(comp["area"]) == 1:
            r0, c0, _r1, _c1 = map(int, comp["bbox"])
            seeds.append((r0, c0))
            continue
        patch = _dense_patch_from_component(g, comp, bg)
        assert patch is not None, "unexpected non-template foreground object"
        templates.append({"bbox": tuple(map(int, comp["bbox"])), "patch": patch})

    assert 2 <= len(templates) <= 4, "expected 2 to 4 templates"
    assert len(seeds) == len(templates), "seed count must match template count"

    seed_rows = {int(r) for r, _c in seeds}
    assert len(seed_rows) == 1, "all seeds must lie on one row"
    seed_row = next(iter(seed_rows))
    assert all(int(t["bbox"][2]) < seed_row for t in templates), "templates must be above the seeds"

    seed_colors = [int(g[r, c]) for r, c in seeds]
    assert len(seed_colors) == len(set(seed_colors)), "seed colors must be unique"
    seed_color_set = set(seed_colors)

    color_to_template: dict[int, np.ndarray] = {}
    for template in templates:
        patch = np.asarray(template["patch"], dtype=int)
        present = set(map(int, np.unique(patch)))
        identifiers = present & seed_color_set
        assert len(identifiers) == 1, "each template must contain exactly one seed color"
        identifier = identifiers.pop()
        assert identifier not in color_to_template, "each seed color must select one template"
        color_to_template[identifier] = patch

    ordered_patches: list[np.ndarray] = []
    for r, c in sorted(seeds, key=lambda rc: int(rc[1])):
        color = int(g[r, c])
        ordered_patches.append(scale_sprite(color_to_template[color], 2))
    return ordered_patches, seeds


def _stack_chain(patches: list[np.ndarray], bg: int = BG) -> np.ndarray:
    row = 0
    col = 0
    placements: list[tuple[int, int]] = []
    out_h = 0
    out_w = 0
    for patch in patches:
        h, w = map(int, patch.shape)
        placements.append((row, col))
        out_h = max(out_h, row + h)
        out_w = max(out_w, col + w)
        row += h - 1
        col += w - 1

    out = np.full((out_h, out_w), int(bg), dtype=int)
    for patch, (r0, c0) in zip(patches, placements):
        paste(out, patch, r0, c0)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Select templates by seed color, scale them, and stack them into an overlapping diagonal chain.

    Input structure:
    - A black background grid is split into an upper template area and a lower seed row.
    - The upper area contains 2 to 4 dense multicolor rectangular templates of size 2x2, 2x3, 3x2, or 3x3.
    - The lower area contains the same number of isolated one-cell seeds on a single row, each using a color that appears in exactly one template.

    Rule:
    - Match each seed color to the unique upper template containing that color.
    - Scale every matched template by 2 in both dimensions by replacing each source cell with a 2x2 block of the same color.
    - Sort the seeds from left to right and place their scaled templates in that order: the first starts at the output's upper-left corner, and each next template starts at the previous template's lower-right corner cell, i.e. offset by (H-1, W-1).
    - When stacked templates overlap, the later template overwrites the earlier one, and the output is cropped to the tight union bbox of the whole chain.

    Key insight:
    - Seed color chooses which template to copy, while seed horizontal order chooses the chain order and therefore the visible corner-overwrite pattern.
    """
    patches, _seeds = _extract_templates_and_seeds(np.asarray(grid, dtype=int))
    return _stack_chain(patches, bg=BG)


def _random_composition(total: int, parts: int, rng: np.random.Generator) -> list[int]:
    if parts == 1:
        return [int(total)]
    cuts = sorted(int(v) for v in rng.choice(np.arange(total + parts - 1), size=parts - 1, replace=False))
    values: list[int] = []
    prev = -1
    for cut in cuts + [total + parts - 1]:
        values.append(int(cut - prev - 1))
        prev = cut
    return values


def _spaced_positions(
    lengths: list[int], min_gap: int, total_size: int, rng: np.random.Generator
) -> list[int] | None:
    required = int(sum(lengths) + max(0, len(lengths) - 1) * min_gap)
    if required > int(total_size):
        return None
    extras = _random_composition(int(total_size - required), len(lengths) + 1, rng)
    starts: list[int] = []
    cursor = int(extras[0])
    for idx, length in enumerate(lengths):
        starts.append(int(cursor))
        cursor += int(length)
        if idx != len(lengths) - 1:
            cursor += int(min_gap + extras[idx + 1])
    return starts


def _make_template(shape: tuple[int, int], identifier: int, rng: np.random.Generator) -> np.ndarray:
    h, w = map(int, shape)
    patch = np.empty((h, w), dtype=int)
    neutral_count = int(rng.integers(1, min(4, len(NEUTRAL_COLORS)) + 1))
    palette = [int(c) for c in rng.choice(NEUTRAL_COLORS, size=neutral_count, replace=False)]
    patch[:, :] = int(palette[0])
    flat = rng.choice(palette, size=h * w, replace=True).astype(int)
    patch[:, :] = flat.reshape(h, w)

    id_index = int(rng.integers(h * w))
    id_r, id_c = divmod(id_index, w)
    patch[id_r, id_c] = int(identifier)

    if len(set(map(int, patch.flatten()))) < 2:
        alt = int(palette[-1])
        patch[(id_r + 1) % h, (id_c + 1) % w] = alt
    return patch


def _render_input(
    templates: list[np.ndarray],
    template_rows: list[int],
    template_cols: list[int],
    seed_row: int,
    seed_cols: list[int],
    seed_order: list[int],
) -> np.ndarray:
    max_template_right = max(int(c + tpl.shape[1]) for tpl, c in zip(templates, template_cols))
    max_seed_col = max(int(c) for c in seed_cols)
    width = max(int(max_template_right), int(max_seed_col + 1))
    height = int(seed_row + 1)
    grid = np.full((height, width), BG, dtype=int)

    for patch, r0, c0 in zip(templates, template_rows, template_cols):
        paste(grid, patch, r0, c0)

    for col, template_idx in zip(seed_cols, seed_order):
        seed_color = int(np.unique(templates[template_idx][np.isin(templates[template_idx], IDENTIFIER_COLORS)])[0])
        grid[int(seed_row), int(col)] = seed_color
    return grid


def _chain_from_indices(templates: list[np.ndarray], order: list[int]) -> np.ndarray:
    scaled = [scale_sprite(templates[idx], 2) for idx in order]
    return _stack_chain(scaled, bg=BG)


def _identifier_of_template(template: np.ndarray) -> int:
    ids = [int(v) for v in np.unique(template) if int(v) in IDENTIFIER_COLORS]
    assert len(ids) == 1
    return int(ids[0])


def _is_semantically_strong(
    grid: np.ndarray, templates: list[np.ndarray], actual_order: list[int], expected: np.ndarray
) -> bool:
    if not np.array_equal(main(grid), expected):
        return False

    if len({tuple(map(int, tpl.shape)) for tpl in templates}) == 1:
        return False

    template_left_order = list(range(len(templates)))
    if actual_order == template_left_order:
        return False
    if np.array_equal(_chain_from_indices(templates, template_left_order), expected):
        return False

    color_order = sorted(range(len(templates)), key=lambda idx: _identifier_of_template(templates[idx]))
    if actual_order == color_order:
        return False
    if np.array_equal(_chain_from_indices(templates, color_order), expected):
        return False

    seen_outputs = {
        _chain_from_indices(templates, list(order)).tobytes()
        for order in itertools.permutations(range(len(templates)))
    }
    if len(seen_outputs) <= 1:
        return False

    parsed_patches, seeds = _extract_templates_and_seeds(grid)
    if len(seeds) != len(templates):
        return False
    if not np.array_equal(_stack_chain(parsed_patches), expected):
        return False
    return True


def _attempt_generate(rng: np.random.Generator) -> np.ndarray | None:
    count = int(rng.integers(2, 5))
    identifiers = [int(c) for c in rng.choice(IDENTIFIER_COLORS, size=count, replace=False)]
    shapes = [TEMPLATE_SHAPES[int(rng.integers(len(TEMPLATE_SHAPES)))] for _ in range(count)]
    if len(set(shapes)) == 1:
        return None

    templates = [_make_template(shape, ident, rng) for shape, ident in zip(shapes, identifiers)]
    if len({tuple(tpl.flatten()) for tpl in templates}) != len(templates):
        return None

    grid_w_min = max(
        GRID_W_RANGE[0],
        sum(int(tpl.shape[1]) for tpl in templates) + 2 * (count - 1),
        count + 2 * (count - 1),
    )
    grid_w_max = GRID_W_RANGE[1]
    if grid_w_min > grid_w_max:
        return None
    grid_w = int(rng.integers(grid_w_min, grid_w_max + 1))

    template_rows = [int(rng.integers(TOP_ROW_RANGE[0], TOP_ROW_RANGE[1] + 1)) for _ in range(count)]
    template_cols = _spaced_positions([int(tpl.shape[1]) for tpl in templates], 2, grid_w, rng)
    if template_cols is None:
        return None

    max_template_bottom = max(int(r + tpl.shape[0] - 1) for tpl, r in zip(templates, template_rows))
    seed_row = int(max_template_bottom + rng.integers(SEED_GAP_RANGE[0], SEED_GAP_RANGE[1] + 1))
    grid_h = int(seed_row + 1 + rng.integers(BOTTOM_MARGIN_RANGE[0], BOTTOM_MARGIN_RANGE[1] + 1))
    if grid_h > 30:
        return None

    seed_cols = _spaced_positions([1] * count, 2, grid_w, rng)
    if seed_cols is None:
        return None

    all_perms = [list(p) for p in itertools.permutations(range(count))]
    color_order = sorted(range(count), key=lambda idx: identifiers[idx])
    valid_orders = [perm for perm in all_perms if perm != list(range(count)) and perm != color_order]
    if not valid_orders:
        return None
    seed_order = valid_orders[int(rng.integers(len(valid_orders)))]

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    for patch, r0, c0 in zip(templates, template_rows, template_cols):
        paste(grid, patch, r0, c0)
    for col, template_idx in zip(seed_cols, seed_order):
        grid[seed_row, int(col)] = _identifier_of_template(templates[template_idx])

    expected = _chain_from_indices(templates, seed_order)
    if max(expected.shape) > 30:
        return None
    if not _is_semantically_strong(grid, templates, seed_order, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        rng = np.random.default_rng()
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for the diagonal-chain template task")
