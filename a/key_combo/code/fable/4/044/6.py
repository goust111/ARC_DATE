from __future__ import annotations

from collections import Counter

import numpy as np


MAX_GENERATION_ATTEMPTS = 200


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _split_panels(grid: np.ndarray) -> tuple[list[np.ndarray], int]:
    h, w = map(int, grid.shape)
    if w == 2 * h:
        n = h
        return [grid[:, :n], grid[:, n:]], n
    if h == 2 * w:
        n = w
        return [grid[:n, :], grid[n:, :]], n
    raise ValueError("input must be two equal N x N panels stacked horizontally or vertically")


def _completed_mask(panel: np.ndarray, marker: int) -> np.ndarray:
    raw = panel == marker
    return raw | raw[::-1, ::-1]


def _template_candidate(panel: np.ndarray, bg: int) -> dict[str, object]:
    colors = sorted({int(v) for v in panel.ravel() if int(v) != bg})
    best: dict[str, object] | None = None
    for marker in colors:
        allowed = {bg, marker}
        if bg != 0:
            allowed.add(0)
        extra = int(np.sum(~np.isin(panel, list(allowed))))
        mask = _completed_mask(panel, marker)
        raw_count = int(np.sum(panel == marker))
        completed_count = int(mask.sum())
        filled_holes = completed_count - raw_count
        score = (-extra, completed_count, filled_holes, raw_count)
        if best is None or score > best["score"]:
            best = {
                "marker": marker,
                "mask": mask,
                "score": score,
                "non_bg_colors": len(colors),
            }
    if best is None:
        return {
            "marker": None,
            "mask": np.zeros_like(panel, dtype=bool),
            "score": (-panel.size, 0, 0, 0),
            "non_bg_colors": 0,
        }
    return best


def _identify_panels(panels: list[np.ndarray], bg: int) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    infos = [_template_candidate(panel, bg) for panel in panels]
    scores = [
        (info["score"], -int(info["non_bg_colors"]))
        for info in infos
    ]
    template_idx = 0 if scores[0] >= scores[1] else 1
    source_idx = 1 - template_idx
    template_info = infos[template_idx]
    marker = template_info["marker"]
    if marker is None:
        raise ValueError("failed to identify template panel")
    return panels[source_idx], panels[template_idx], int(marker), np.asarray(template_info["mask"], dtype=bool)


def _stamp_output(source: np.ndarray, completed_mask: np.ndarray, bg: int) -> np.ndarray:
    n = int(source.shape[0])
    out = np.full((n * n, n * n), bg, dtype=int)
    for r in range(n):
        for c in range(n):
            color = int(source[r, c])
            if color == bg:
                continue
            block = out[r * n : (r + 1) * n, c * n : (c + 1) * n]
            block[completed_mask] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a damaged point-symmetric template, then expand it as colored block stamps.

    Input structure:
    - The input is made of two equal N x N panels placed side by side or stacked vertically on one shared background color.
    - One panel is a sparse source panel containing a few non-background colored pixels.
    - The other panel is a template panel containing one marker color plus black missing cells that hide part of a 180-degree rotationally symmetric shape.

    Rule:
    - Split the grid into its two N x N panels and identify which one is the source panel and which one is the template panel.
    - Restore the template by rotating the visible marker mask by 180 degrees and filling any black missing cell from its opposite partner.
    - Create an N^2 x N^2 background output grid, and for each non-background source pixel at (r, c), paint the restored template recolored to that pixel's color into output block (r, c).

    Key insight:
    - The black damage is resolved before stamping: the hidden master template is exactly the 180-degree symmetric closure of the visible marker cells.
    """
    g = np.asarray(grid, dtype=int)
    panels, _ = _split_panels(g)
    bg = _get_bg(g)
    source, _template, _marker, completed_mask = _identify_panels(panels, bg)
    return _stamp_output(source, completed_mask, bg)


def _orbits(n: int) -> list[list[tuple[int, int]]]:
    visited: set[tuple[int, int]] = set()
    out: list[list[tuple[int, int]]] = []
    for r in range(n):
        for c in range(n):
            if (r, c) in visited:
                continue
            rr, cc = n - 1 - r, n - 1 - c
            orbit = [(r, c)] if (rr, cc) == (r, c) else [(r, c), (rr, cc)]
            visited.update(orbit)
            out.append(orbit)
    return out


def _make_template_shape(rng: np.random.Generator, n: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    orbits = _orbits(n)
    pair_orbits = [orbit for orbit in orbits if len(orbit) == 2]
    for _ in range(100):
        mask = np.zeros((n, n), dtype=bool)
        chosen_pairs = 0
        for orbit in orbits:
            keep = rng.random() < 0.55
            if keep:
                for r, c in orbit:
                    mask[r, c] = True
                if len(orbit) == 2:
                    chosen_pairs += 1
        area = int(mask.sum())
        if 4 <= area <= max(5, (3 * n * n) // 5) and chosen_pairs >= 1 and area < n * n:
            marked_pairs = [orbit for orbit in pair_orbits if mask[orbit[0]]]
            holes_count = int(rng.integers(1, min(len(marked_pairs), max(2, len(marked_pairs) // 2 + 1)) + 1))
            hole_indices = rng.choice(len(marked_pairs), size=holes_count, replace=False)
            holes: list[tuple[int, int]] = []
            for idx in hole_indices:
                orbit = marked_pairs[int(idx)]
                holes.append(orbit[int(rng.integers(0, 2))])
            return mask, holes
    raise RuntimeError("failed to build a valid symmetric template")


def _make_source_panel(rng: np.random.Generator, n: int, bg: int, marker: int) -> np.ndarray:
    panel = np.full((n, n), bg, dtype=int)
    palette = [color for color in range(1, 10) if color not in {bg, marker}]
    count = int(rng.integers(2, min(n + 2, n * n - 1) + 1))
    colors_used = int(rng.integers(2, min(4, count, len(palette)) + 1))
    chosen_colors = list(map(int, rng.choice(palette, size=colors_used, replace=False)))
    assignments = chosen_colors + list(map(int, rng.choice(chosen_colors, size=count - colors_used, replace=True)))
    rng.shuffle(assignments)
    positions = rng.choice(n * n, size=count, replace=False)
    for idx, color in zip(positions, assignments):
        r, c = divmod(int(idx), n)
        panel[r, c] = color
    return panel


def _build_expected(source: np.ndarray, template_mask: np.ndarray, bg: int) -> np.ndarray:
    return _stamp_output(source, template_mask, bg)


def _compose_panels(
    source: np.ndarray,
    template: np.ndarray,
    horizontal: bool,
    source_first: bool,
) -> np.ndarray:
    first, second = (source, template) if source_first else (template, source)
    return np.hstack([first, second]) if horizontal else np.vstack([first, second])


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    n = int(rng.integers(3, 6))
    bg = int(rng.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
    marker_choices = [color for color in range(1, 10) if color != bg]
    marker = int(rng.choice(marker_choices))

    template_mask, holes = _make_template_shape(rng, n)
    template = np.full((n, n), bg, dtype=int)
    template[template_mask] = marker
    for r, c in holes:
        template[r, c] = 0

    source = _make_source_panel(rng, n, bg, marker)
    horizontal = bool(rng.integers(0, 2))
    source_first = bool(rng.integers(0, 2))
    inp = _compose_panels(source, template, horizontal, source_first)

    if _get_bg(inp) != bg:
        return None

    expected = _build_expected(source, template_mask, bg)
    try:
        actual = main(inp)
    except Exception:
        return None
    if actual.shape != (n * n, n * n):
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
