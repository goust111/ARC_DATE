from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

# Ensure arc_utils is importable when this file is loaded directly with SourceFileLoader.
_UTILS = Path('/home/lixinyu/snap/new/solvers')
if str(_UTILS) not in sys.path:
    sys.path.insert(0, str(_UTILS))

from arc_utils import Color, colors_present, get_bg


def _find_divider(grid: np.ndarray) -> tuple[int, int]:
    h, w = grid.shape
    bg = int(get_bg(grid))
    candidates: list[tuple[int, int]] = []
    for c in range(w):
        col = grid[:, c]
        color = int(col[0])
        if np.all(col == color) and color != bg:
            candidates.append((c, color))

    assert len(candidates) == 1, "expected one non-background full-height divider column"
    return candidates[0]


def _extract_templates(left: np.ndarray, bg: int) -> dict[int, np.ndarray]:
    h, k = left.shape
    assert k > 0 and h > 0, "left panel must be non-empty"
    assert h % k == 0, "left panel height must be multiple of k"

    templates: dict[int, np.ndarray] = {}
    for r0 in range(0, h, k):
        block = left[r0 : r0 + k, :]
        colors = colors_present(block, exclude={bg})
        assert colors, "template rows must be non-empty"
        assert len(colors) == 1, "each template uses one color plus background"
        color = int(colors[0])
        assert color not in templates, "template colors must be unique"
        templates[color] = block.copy()

    return templates


def _slot_markers(slot: np.ndarray, bg: int) -> list[tuple[int, int, int]]:
    return [(int(r), int(c), int(v)) for r, c in np.argwhere(slot != bg) for v in [int(slot[r, c])]]


def _order_base_and_top(markers: list[tuple[int, int, int]]) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    m = sorted(markers, key=lambda t: (t[0], t[1]))
    assert len(m) == 2
    return m[0], m[1]


def _composite_two(base: np.ndarray, top: np.ndarray, bg: int) -> np.ndarray:
    merged = base.copy()
    overlay = top != bg
    merged[overlay] = top[overlay]
    return merged


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp templates indicated by markers into same-sized right-side slots with dual-marker layering.

    Input structure:
    - A dominant background color and one full-height divider column split the canvas.
    - Left of the divider are stacked k×k templates, each single-color with possible background holes.
    - Right of the divider is tiled into k×k slots; each slot has zero, one, or two marker pixels.
    - Marker colors are template colors.

    Rule:
    - Preserve divider, left templates, and all unmarked slots.
    - If a slot has one marker, stamp the matching template into that slot.
    - If a slot has two markers, order markers top-to-bottom (left-to-right when in same row):
      paint the first marker's template first as base, then overlay the second marker's template.
      During overlay, template background is transparent and does not cover base pixels.
    - Marker pixels disappear because stamping always overwrites slot content.

    Key insight:
    - The two-marker slot expresses a strict layer order by marker position; overlap is resolved by standard top-over-bottom compositing.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    h, w = g.shape
    bg = int(get_bg(g))
    divider_col, _ = _find_divider(g)
    assert 0 < divider_col < w - 1, "divider must separate left and right areas"

    k = divider_col
    left = g[:, :k]
    right = g[:, divider_col + 1 :]
    assert right.shape[1] % k == 0, "right panel width must be multiples of k"
    assert h % k == 0, "canvas height must be multiples of k"

    templates = _extract_templates(left, bg)
    slot_rows = h // k
    slot_cols = right.shape[1] // k
    assert len(templates) == slot_rows, "expected one template per template row"

    out = g.copy()
    for sr in range(slot_rows):
        r0 = sr * k
        r1 = r0 + k
        for sc in range(slot_cols):
            c0 = divider_col + 1 + sc * k
            c1 = c0 + k
            slot = g[r0:r1, c0:c1]
            marks = _slot_markers(slot, bg)

            if not marks:
                continue
            assert len(marks) <= 2, "slots must contain at most two markers"

            if len(marks) == 1:
                color = marks[0][2]
                assert color in templates, "single marker must match a template color"
                out[r0:r1, c0:c1] = templates[color]
                continue

            (rb, cb, base_color), (rt, ct, top_color) = _order_base_and_top(marks)
            assert base_color != top_color, "marker colors in a slot must be different"
            assert base_color in templates and top_color in templates, "double markers must reference templates"
            out[r0:r1, c0:c1] = _composite_two(templates[base_color], templates[top_color], bg)

    return out


def _random_template(k: int, color: int, bg: int) -> np.ndarray:
    while True:
        sprite = np.full((k, k), int(color), dtype=int)
        hole_prob = random.uniform(0.22, 0.52)
        holes = np.random.random((k, k)) < hole_prob
        sprite[holes] = bg
        if np.any(sprite != bg) and np.any(sprite == bg):
            return sprite


def _build_left_panel(templates: list[tuple[int, np.ndarray]]) -> np.ndarray:
    k = templates[0][1].shape[1]
    h = len(templates) * k
    panel = np.full((h, k), int(Color.BLACK), dtype=int)
    for i, (_color, patch) in enumerate(templates):
        r0 = i * k
        panel[r0 : r0 + k, :] = patch
    return panel


def _build_right_panel(
    h: int,
    slot_cols: int,
    k: int,
    template_colors: list[int],
    templates: dict[int, np.ndarray],
    bg: int,
) -> np.ndarray:
    panel = np.full((h, slot_cols * k), int(bg), dtype=int)
    slots = h // k
    total_slots = slots * slot_cols
    all_slots = list(range(total_slots))
    random.shuffle(all_slots)

    overlap_pairs: list[tuple[int, int]] = []
    for i, ca in enumerate(template_colors):
        for cb in template_colors[i + 1 :]:
            if np.any((templates[ca] != bg) & (templates[cb] != bg)):
                overlap_pairs.append((ca, cb))

    if overlap_pairs and random.random() < 0.85:
        max_double = max(1, min(2, total_slots // 2))
        n_double = random.randint(1, max_double)
    else:
        n_double = 0

    remaining = total_slots - n_double
    if remaining:
        n_single = random.randint(1 if n_double == 0 else 0, remaining)
    else:
        n_single = 0

    cursor = 0

    def slot_base(slot_idx: int) -> tuple[int, int]:
        sr, sc = divmod(slot_idx, slot_cols)
        return sr * k, sc * k

    def place_single(slot_idx: int) -> None:
        color = random.choice(template_colors)
        r0, c0 = slot_base(slot_idx)
        r = r0 + random.randrange(k)
        c = c0 + random.randrange(k)
        panel[r, c] = color

    def place_double(slot_idx: int) -> None:
        color_a, color_b = random.choice(overlap_pairs)
        r0, c0 = slot_base(slot_idx)
        same_row = random.random() < 0.35

        if same_row:
            row = random.randrange(k)
            c1, c2 = random.sample(range(k), 2)
            if c1 > c2:
                c1, c2 = c2, c1
            panel[r0 + row, c0 + c1] = color_a
            panel[r0 + row, c0 + c2] = color_b
            return

        idx1 = random.randrange(k * k)
        idx2 = random.randrange(k * k - 1)
        if idx2 >= idx1:
            idx2 += 1

        r1, c1 = divmod(idx1, k)
        r2, c2 = divmod(idx2, k)
        if r1 == r2 and c1 == c2:
            c2 = (c2 + 1) % k

        if r1 > r2:
            r1, c1, r2, c2 = r2, c2, r1, c1

        panel[r0 + r1, c0 + c1] = color_a
        panel[r0 + r2, c0 + c2] = color_b

    for _ in range(n_double):
        place_double(all_slots[cursor])
        cursor += 1

    for _ in range(n_single):
        if cursor >= total_slots:
            break
        place_single(all_slots[cursor])
        cursor += 1

    return panel


def generate_input() -> np.ndarray:
    """
    Generate valid and varied inputs for the dual-marker stamping task.
    """
    for _ in range(3000):
        k = random.randint(3, 5)
        max_templates = min(6, 30 // k)
        template_count = random.randint(2, max_templates)

        slot_cols_max = max(2, (30 - k - 1) // k)
        slot_cols = random.randint(2, slot_cols_max)

        h = template_count * k
        w = k + 1 + slot_cols * k
        if h > 30 or w > 30:
            continue

        bg = int(Color.BLACK)
        divider_color = random.choice([c for c in Color.NOT_BLACK if c != bg])
        palette = [c for c in Color.NOT_BLACK if c != bg and c != divider_color]
        if len(palette) < template_count:
            continue

        chosen_colors = random.sample(palette, template_count)
        templates = {
            color: _random_template(k, color, bg)
            for color in chosen_colors
        }

        # Guarantee at least one overlapping pair for dual-marker behavior.
        overlap_pairs: list[tuple[int, int]] = []
        for i, ca in enumerate(chosen_colors):
            for cb in chosen_colors[i + 1 :]:
                if np.any((templates[ca] != bg) & (templates[cb] != bg)):
                    overlap_pairs.append((ca, cb))
        if not overlap_pairs:
            a, b = chosen_colors[:2]
            anchors = np.argwhere(templates[a] != bg)
            if anchors.size == 0:
                continue
            ra, ca = random.choice([tuple(p) for p in anchors])
            templates[b][ra, ca] = b

        grid = np.full((h, w), bg, dtype=int)
        left = _build_left_panel([(c, templates[c]) for c in chosen_colors])
        grid[:, :k] = left
        grid[:, k] = divider_color

        right = _build_right_panel(
            h=h,
            slot_cols=slot_cols,
            k=k,
            template_colors=chosen_colors,
            templates=templates,
            bg=bg,
        )
        grid[:, k + 1 :] = right

        return grid

    raise RuntimeError("failed to generate valid input")
