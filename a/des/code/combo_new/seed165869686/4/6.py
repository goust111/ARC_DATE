import random

import numpy as np


BLACK = 0
GEN_SEPARATOR = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
TEMPLATE_WIDTH_SETS = ([2, 2, 3, 3], [2, 2, 2, 3, 3])
TEMPLATE_HEIGHT_CHOICES = (2, 3)
MAX_GENERATION_ATTEMPTS = 400
MAX_GRID_SIZE = 30


def _find_separator_color(grid: np.ndarray) -> int:
    best_color = None
    best_score = -1
    for color0 in np.unique(grid):
        color = int(color0)
        full_rows = int(np.all(grid == color, axis=1).sum())
        full_cols = int(np.all(grid == color, axis=0).sum())
        if full_rows == 0 or full_cols == 0:
            continue
        score = full_rows + full_cols
        if score > best_score:
            best_color = color
            best_score = score
    assert best_color is not None, "expected a separator color forming full rows and columns"
    return best_color


def _region_bboxes(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    boxes: list[tuple[int, int, int, int]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            r0 = r1 = r
            c0 = c1 = c
            while stack:
                rr, cc = stack.pop()
                r0 = min(r0, rr)
                r1 = max(r1, rr)
                c0 = min(c0, cc)
                c1 = max(c1, cc)
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            boxes.append((r0, c0, r1, c1))
    return boxes


def _scale_mask(mask: np.ndarray, scale: int) -> np.ndarray:
    return np.kron(mask.astype(np.uint8), np.ones((scale, scale), dtype=np.uint8)).astype(bool)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each lower rectangular region with the scaled template whose color matches its seed pixel.

    Input structure:
    - Background is Black.
    - Grey line segments form a rectilinear partition of the grid into closed rectangular regions.
    - The topmost row of regions is a template library: each template region contains a small monochrome pattern in a unique non-Black, non-Grey color.
    - Every region below the library contains exactly one colored seed pixel, and the seed color matches one template color.

    Rule:
    - Identify the topmost row of rectangular regions as the template library and keep those template regions unchanged.
    - For each lower region, use its seed color to select the template of the same color from the library.
    - Uniformly scale that template pattern so it exactly fits the target region's interior size, then redraw the scaled pattern in the same color over a Black interior.
    - Keep every Grey separator line unchanged.

    Key insight:
    - The top row is not another batch of seed regions: it is a color-indexed template library, and each lower room must first be matched by color before its interior can be reconstructed by integer scaling.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    separator = _find_separator_color(g)
    regions = _region_bboxes(g != separator)
    assert regions, "expected at least one enclosed region"

    top = min(r0 for r0, _, _, _ in regions)
    library = [bbox for bbox in regions if bbox[0] == top]
    work = [bbox for bbox in regions if bbox[0] > top]
    assert library and work, "expected a top template row and at least one work region"

    templates: dict[int, np.ndarray] = {}
    for r0, c0, r1, c1 in sorted(library, key=lambda bbox: (bbox[1], bbox[0])):
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        colors = [int(color) for color in np.unique(patch) if int(color) != BLACK]
        assert len(colors) == 1, "each template region must contain exactly one non-black color"
        color = colors[0]
        assert color != separator and color not in templates, "template colors must be unique"
        templates[color] = patch == color

    out = g.copy()
    for r0, c0, r1, c1 in work:
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        colors = [int(color) for color in np.unique(patch) if int(color) != BLACK]
        assert len(colors) == 1, "each work region must contain exactly one seed color"
        color = colors[0]
        assert color in templates, "every seed color must map to a template"

        template_mask = templates[color]
        th, tw = template_mask.shape
        h, w = patch.shape
        assert h % th == 0 and w % tw == 0, "target size must be an integer multiple of the template size"
        scale_h = h // th
        scale_w = w // tw
        assert scale_h == scale_w and scale_h >= 1, "template scaling must be uniform"

        target = out[r0 : r1 + 1, c0 : c1 + 1]
        target[:] = BLACK
        target[_scale_mask(template_mask, scale_h)] = color

    return out


def _add_pattern(patterns: list[np.ndarray], seen: set[bytes], mask: np.ndarray) -> None:
    if int(mask.sum()) < 2:
        return
    key = mask.tobytes()
    if key in seen:
        return
    seen.add(key)
    patterns.append(mask.copy())


def _diag_mask(h: int, w: int, reverse: bool) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    cols = np.round(np.linspace(0, w - 1, h)).astype(int)
    for r, c in enumerate(cols):
        mask[r, w - 1 - c if reverse else c] = True
    return mask


def _pattern_catalog(h: int, w: int) -> list[np.ndarray]:
    patterns: list[np.ndarray] = []
    seen: set[bytes] = set()

    _add_pattern(patterns, seen, np.ones((h, w), dtype=bool))
    _add_pattern(patterns, seen, _diag_mask(h, w, reverse=False))
    _add_pattern(patterns, seen, _diag_mask(h, w, reverse=True))

    for corner in ("tl", "tr", "bl", "br"):
        mask = np.zeros((h, w), dtype=bool)
        if corner[0] == "t":
            mask[0, :] = True
        else:
            mask[-1, :] = True
        if corner[1] == "l":
            mask[:, 0] = True
        else:
            mask[:, -1] = True
        _add_pattern(patterns, seen, mask)

    for r in range(h):
        mask = np.zeros((h, w), dtype=bool)
        mask[r, :] = True
        _add_pattern(patterns, seen, mask)
    for c in range(w):
        mask = np.zeros((h, w), dtype=bool)
        mask[:, c] = True
        _add_pattern(patterns, seen, mask)

    if h >= 3 and w >= 3:
        mask = np.zeros((h, w), dtype=bool)
        mask[h // 2, :] = True
        mask[:, w // 2] = True
        _add_pattern(patterns, seen, mask)

        frame = np.zeros((h, w), dtype=bool)
        frame[0, :] = True
        frame[-1, :] = True
        frame[:, 0] = True
        frame[:, -1] = True
        _add_pattern(patterns, seen, frame)

    checker = (np.indices((h, w)).sum(axis=0) % 2) == 0
    _add_pattern(patterns, seen, checker)
    _add_pattern(patterns, seen, ~checker)

    return patterns


def _scale_sequences(template_h: int) -> list[list[int]]:
    candidates: list[list[int]] = []
    for seq in ([2, 3], [3, 2], [2, 2, 3], [2, 3, 2], [3, 2, 2], [2, 3, 3], [3, 2, 3], [3, 3, 2]):
        total_h = template_h * (1 + sum(seq)) + (len(seq) + 2)
        if total_h <= MAX_GRID_SIZE:
            candidates.append(list(seq))
    return candidates


def _enumerate_row_sequences(inner_width: int, template_widths: list[int], scale: int) -> list[list[int]]:
    scaled_widths = [scale * width for width in template_widths]
    results: list[list[int]] = []

    def dfs(remaining: int, seq: list[int]) -> None:
        if remaining == 0:
            if len(seq) >= 2 and len(set(seq)) >= 2:
                results.append(seq.copy())
            return
        if len(seq) >= 5:
            return
        for idx, width in enumerate(scaled_widths):
            cost = width if not seq else width + 1
            if cost <= remaining:
                seq.append(idx)
                dfs(remaining - cost, seq)
                seq.pop()

    dfs(inner_width, [])
    return results


def _choose_row_sequences(scales: list[int], choices_by_scale: dict[int, list[list[int]]], template_count: int) -> list[list[int]] | None:
    best_rows = None
    best_score = (-1, -1)
    for _ in range(80):
        rows = [random.choice(choices_by_scale[scale]) for scale in scales]
        used = set(idx for row in rows for idx in row)
        score = (len(used), sum(len(set(row)) for row in rows))
        if score > best_score:
            best_rows = rows
            best_score = score
        if len(used) == template_count:
            break
    return best_rows


def _pick_template_masks(template_h: int, template_widths: list[int]) -> list[np.ndarray] | None:
    used: set[tuple[tuple[int, int], bytes]] = set()
    masks: list[np.ndarray] = []
    for width in template_widths:
        options = _pattern_catalog(template_h, width)
        choices = [mask for mask in options if (mask.shape, mask.tobytes()) not in used]
        if not choices:
            choices = options
        if not choices:
            return None
        mask = random.choice(choices)
        used.add((mask.shape, mask.tobytes()))
        masks.append(mask)
    return masks


def _render_example(
    template_h: int,
    template_specs: list[dict[str, object]],
    scales: list[int],
    work_rows: list[list[int]],
) -> tuple[np.ndarray, np.ndarray]:
    top_widths = [int(spec["width"]) for spec in template_specs]
    inner_width = sum(top_widths) + (len(top_widths) - 1)
    total_width = inner_width + 2
    total_height = template_h * (1 + sum(scales)) + (len(scales) + 2)

    inp = np.full((total_height, total_width), GEN_SEPARATOR, dtype=int)
    out = np.full((total_height, total_width), GEN_SEPARATOR, dtype=int)

    row_top = 0
    rows = [(template_h, list(range(len(template_specs))))] + [
        (scale * template_h, seq) for scale, seq in zip(scales, work_rows)
    ]

    for row_index, (row_h, seq) in enumerate(rows):
        col = 1
        for cell_index, template_idx in enumerate(seq):
            spec = template_specs[template_idx]
            color = int(spec["color"])
            mask = np.asarray(spec["mask"], dtype=bool)
            if row_index == 0:
                cell_w = int(spec["width"])
            else:
                scale = row_h // template_h
                cell_w = scale * int(spec["width"])

            rr = slice(row_top + 1, row_top + 1 + row_h)
            cc = slice(col, col + cell_w)
            inp[rr, cc] = BLACK
            out[rr, cc] = BLACK

            if row_index == 0:
                inp[rr, cc][mask] = color
                out[rr, cc][mask] = color
            else:
                seed_r = random.randint(row_top + 1, row_top + row_h)
                seed_c = random.randint(col, col + cell_w - 1)
                inp[seed_r, seed_c] = color
                scaled = _scale_mask(mask, row_h // template_h)
                out[rr, cc][scaled] = color

            col += cell_w
            if cell_index != len(seq) - 1:
                col += 1

        row_top += row_h + 1

    return inp, out


def _gen_attempt() -> np.ndarray | None:
    template_h = random.choice(TEMPLATE_HEIGHT_CHOICES)
    template_widths = list(random.choice(TEMPLATE_WIDTH_SETS))
    random.shuffle(template_widths)

    scales = random.choice(_scale_sequences(template_h))
    inner_width = sum(template_widths) + (len(template_widths) - 1)
    total_width = inner_width + 2
    if total_width > MAX_GRID_SIZE:
        return None

    choices_by_scale: dict[int, list[list[int]]] = {}
    for scale in (2, 3):
        choices = _enumerate_row_sequences(inner_width, template_widths, scale)
        if not choices:
            return None
        choices_by_scale[scale] = choices

    work_rows = _choose_row_sequences(scales, choices_by_scale, len(template_widths))
    if work_rows is None:
        return None

    colors = random.sample(COLOR_POOL, len(template_widths))
    masks = _pick_template_masks(template_h, template_widths)
    if masks is None:
        return None

    template_specs = [
        {"width": width, "color": color, "mask": mask}
        for width, color, mask in zip(template_widths, colors, masks)
    ]

    inp, expected = _render_example(template_h, template_specs, scales, work_rows)
    if max(inp.shape) > MAX_GRID_SIZE or max(expected.shape) > MAX_GRID_SIZE:
        return None

    try:
        actual = main(inp)
    except Exception:
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
