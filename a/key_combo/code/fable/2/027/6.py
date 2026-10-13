from __future__ import annotations

import numpy as np


GRAY = 5
PALETTE = tuple(color for color in range(10) if color != GRAY)
MAX_GENERATION_ATTEMPTS = 400

TEMPLATE_BANK = tuple(
    np.array(pattern, dtype=bool)
    for pattern in (
        ((0, 1, 0), (1, 1, 1), (0, 1, 0)),
        ((1, 0, 1), (0, 1, 0), (1, 0, 1)),
        ((1, 1, 1), (0, 1, 0), (0, 1, 0)),
        ((1, 0, 0), (1, 1, 1), (0, 0, 0)),
        ((1, 1, 0), (0, 1, 1), (0, 0, 0)),
        ((1, 0, 1), (1, 0, 1), (1, 1, 1)),
        ((0, 1, 0), (1, 1, 0), (0, 1, 0)),
        ((1, 1, 0), (1, 0, 0), (1, 1, 0)),
        ((0, 1, 1), (0, 1, 0), (1, 1, 0)),
        ((1, 0, 1), (0, 0, 0), (1, 0, 1)),
        ((1, 1, 0), (0, 1, 0), (0, 1, 1)),
        ((0, 1, 1), (1, 0, 0), (1, 1, 0)),
    )
)


def _background_color(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=10)
    return int(np.argmax(counts))


def _find_separator(grid: np.ndarray) -> int:
    candidates = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GRAY)]
    assert len(candidates) == 1, "expected a unique full-height gray separator"
    return int(candidates[0])


def _parse_legend(grid: np.ndarray, separator_col: int, bg: int) -> dict[int, np.ndarray]:
    left = grid[:, :separator_col]
    entries = np.argwhere((left != bg) & (left != GRAY))
    assert entries.size > 0, "legend must contain color-code pixels"

    mapping: dict[int, np.ndarray] = {}
    for r, c in entries:
        rr, cc = int(r), int(c)
        assert 0 < rr < grid.shape[0] - 1, "legend code pixel must have room for a 3x3 symbol"
        assert cc + 3 < separator_col, "legend symbol must fit immediately to the right"
        symbol = left[rr - 1 : rr + 2, cc + 1 : cc + 4]
        gray_mask = symbol == GRAY
        assert gray_mask.any(), "legend symbol must contain gray cells"
        assert not np.any((symbol != bg) & (symbol != GRAY)), "legend symbol region must contain only bg/gray"
        color = int(left[rr, cc])
        assert color not in mapping, "legend colors must be unique"
        mapping[color] = gray_mask.copy()

    signatures = {tuple(mask.astype(int).ravel()) for mask in mapping.values()}
    assert len(signatures) == len(mapping), "legend symbols must be distinct"
    return mapping


def _scene_pixels(scene: np.ndarray, bg: int) -> np.ndarray:
    mask = (scene != bg) & (scene != GRAY)
    assert not np.any(scene == GRAY), "scene must not contain gray cells"
    coords = np.argwhere(mask)
    assert coords.size > 0, "scene must contain color-code pixels"
    return coords


def _render_scene(
    scene: np.ndarray,
    mapping: dict[int, np.ndarray],
    bg: int,
    *,
    full_block: bool = False,
    recolor: bool = True,
    shift: tuple[int, int] = (0, 0),
) -> np.ndarray:
    out = np.full(scene.shape, bg, dtype=int)
    for r, c in _scene_pixels(scene, bg):
        rr, cc = int(r), int(c)
        color = int(scene[rr, cc])
        assert color in mapping, "every scene color must be present in the legend"
        mask = np.ones((3, 3), dtype=bool) if full_block else mapping[color]
        paint_color = color if recolor else GRAY
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                if not bool(mask[dr + 1, dc + 1]):
                    continue
                pr = rr + dr + int(shift[0])
                pc = cc + dc + int(shift[1])
                assert 0 <= pr < out.shape[0] and 0 <= pc < out.shape[1], "expanded symbol left the scene bounds"
                assert out[pr, pc] == bg, "expanded symbols must not overlap"
                out[pr, pc] = paint_color
    return out


def _is_highly_symmetric(mask: np.ndarray) -> bool:
    return (
        np.array_equal(mask, np.rot90(mask, 1))
        and np.array_equal(mask, np.fliplr(mask))
        and np.array_equal(mask, np.flipud(mask))
    )


def _validate_case(grid: np.ndarray) -> None:
    bg = _background_color(grid)
    separator_col = _find_separator(grid)
    legend = _parse_legend(grid, separator_col, bg)
    scene = grid[:, separator_col + 1 :]
    coords = _scene_pixels(scene, bg)

    used_colors = {int(scene[r, c]) for r, c in coords}
    assert len(legend) >= 3, "legend should have multiple entries"
    assert len(used_colors) == len(legend) - 1, "legend must contain exactly one unused entry"
    assert len(coords) > len(used_colors), "at least one legend mapping should be reused in the scene"

    for r, c in coords:
        rr, cc = int(r), int(c)
        assert 0 < rr < scene.shape[0] - 1 and 0 < cc < scene.shape[1] - 1, "scene pixel must allow a centered 3x3 expansion"

    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            r1, c1 = map(int, coords[i])
            r2, c2 = map(int, coords[j])
            assert abs(r1 - r2) > 2 or abs(c1 - c2) > 2, "expanded 3x3 windows must be disjoint"

    used_masks = [legend[color] for color in used_colors]
    assert any(not bool(mask[1, 1]) for mask in used_masks), "at least one used symbol should remove the original pixel"
    assert any(not _is_highly_symmetric(mask) for mask in used_masks), "at least one used symbol should be orientation-sensitive"

    solved = _render_scene(scene, legend, bg)
    assert not np.array_equal(solved, scene), "solution must do more than crop the scene"
    assert not np.array_equal(solved, _render_scene(scene, legend, bg, full_block=True)), "full 3x3 expansion must be wrong"
    mirrored = {color: np.fliplr(mask) for color, mask in legend.items()}
    assert not np.array_equal(solved, _render_scene(scene, legend, bg, recolor=False)), "gray output must be wrong"
    assert not np.array_equal(solved, _render_scene(scene, mirrored, bg)), "mirrored templates must be wrong"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the in-picture legend and replace each scene color pixel with its mapped 3x3 symbol.

    Input structure:
    - A full-height gray vertical line separates a narrow legend panel on the left from a wider scene panel on the right.
    - Each legend entry contains one colored code pixel and, immediately to its right, a 3x3 gray symbol template.
    - The scene panel contains scattered colored single pixels on the background, and exactly one legend color is unused there.

    Rule:
    - Discard the legend panel and the gray separator; output only the scene panel.
    - For each colored scene pixel, look up the legend entry with the same color and place that 3x3 template centered on the pixel.
    - Draw the placed template in the scene pixel's own color, not in gray, and leave background elsewhere.

    Key insight:
    - The gray legend provides per-instance shape templates, so the solver must read the symbol table from the input and then recolor those shapes in the cropped scene.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _background_color(g)
    separator_col = _find_separator(g)
    legend = _parse_legend(g, separator_col, bg)
    scene = g[:, separator_col + 1 :]
    return _render_scene(scene, legend, bg)


def _place_scene_points(
    rng: np.random.Generator,
    height: int,
    width: int,
    count: int,
) -> list[tuple[int, int]] | None:
    candidates = [(r, c) for r in range(1, height - 1) for c in range(1, width - 1)]
    rng.shuffle(candidates)
    chosen: list[tuple[int, int]] = []
    for r, c in candidates:
        if all(abs(r - rr) > 2 or abs(c - cc) > 2 for rr, cc in chosen):
            chosen.append((r, c))
            if len(chosen) == count:
                return chosen
    return None


def _sample_templates(
    rng: np.random.Generator,
    legend_count: int,
    used_count: int,
) -> list[np.ndarray]:
    for _ in range(200):
        indices = rng.choice(len(TEMPLATE_BANK), size=legend_count, replace=False)
        templates = [TEMPLATE_BANK[int(idx)] for idx in indices]
        used_templates = templates[:used_count]
        if any(not bool(mask[1, 1]) for mask in used_templates) and any(
            not _is_highly_symmetric(mask) for mask in used_templates
        ):
            return templates
    raise RuntimeError("failed to sample diverse legend templates")


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    legend_count = int(rng.integers(3, 6))
    used_count = legend_count - 1

    code_col = int(rng.integers(0, 2))
    separator_col = code_col + 4
    scene_width_max = 30 - (separator_col + 1)
    scene_width = int(rng.integers(10, scene_width_max + 1))

    top_margin = int(rng.integers(1, 3))
    bottom_margin = int(rng.integers(1, 3))
    gaps = [int(rng.integers(1, 3)) for _ in range(legend_count - 1)]
    min_height = top_margin + 3 * legend_count + sum(gaps) + bottom_margin
    if min_height > 30:
        return None
    height = int(rng.integers(min_height, 31))

    bg = int(rng.choice([color for color in PALETTE if color != GRAY]))
    usable_colors = [color for color in PALETTE if color != bg]
    if len(usable_colors) < legend_count:
        return None
    legend_colors = [int(c) for c in rng.choice(usable_colors, size=legend_count, replace=False)]
    rng.shuffle(legend_colors)
    unused_index = int(rng.integers(0, legend_count))
    used_colors = [legend_colors[i] for i in range(legend_count) if i != unused_index]
    unused_color = legend_colors[unused_index]
    templates = _sample_templates(rng, legend_count, used_count)
    entries = list(zip(used_colors, templates[:used_count])) + [(unused_color, templates[-1])]
    rng.shuffle(entries)

    capacity = ((height - 2) + 2) // 3 * ((scene_width - 2) + 2) // 3
    if capacity <= used_count:
        return None
    max_pixels = min(8, capacity)
    total_scene_pixels = int(rng.integers(used_count + 1, max_pixels + 1))
    positions = _place_scene_points(rng, height, scene_width, total_scene_pixels)
    if positions is None:
        return None

    scene_colors = used_colors.copy()
    extras = total_scene_pixels - used_count
    for _ in range(extras):
        scene_colors.append(int(rng.choice(used_colors)))
    rng.shuffle(scene_colors)

    grid = np.full((height, separator_col + 1 + scene_width), bg, dtype=int)
    grid[:, separator_col] = GRAY

    row0 = top_margin
    for idx, (color, template) in enumerate(entries):
        center_r = row0 + 1
        grid[center_r, code_col] = int(color)
        block = grid[row0 : row0 + 3, code_col + 1 : code_col + 4]
        block[template] = GRAY
        if idx < legend_count - 1:
            row0 += 3 + gaps[idx]

    for (r, c), color in zip(positions, scene_colors):
        grid[int(r), separator_col + 1 + int(c)] = int(color)

    try:
        _validate_case(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
