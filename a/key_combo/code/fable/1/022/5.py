from __future__ import annotations

import random
from typing import Sequence

import numpy as np


State = tuple[int, ...]

BG = 0
BLUE = 1
FG_CHOICES = (2, 3, 4, 5, 6, 7, 8, 9)
MAX_ATTEMPTS = 600


def _get_bg(grid: np.ndarray) -> int:
    values = np.asarray(grid, dtype=int).ravel()
    return int(np.bincount(values).argmax())


def _compress_states(states: Sequence[State]) -> list[State]:
    compressed: list[State] = []
    for state in states:
        normalized = tuple(int(v) for v in state)
        assert normalized, "every state must contain at least one foreground offset"
        if not compressed or compressed[-1] != normalized:
            compressed.append(normalized)
    return compressed


def _is_prefix(prefix: Sequence[State], template: Sequence[State]) -> bool:
    return len(prefix) <= len(template) and all(prefix[i] == template[i] for i in range(len(prefix)))


def _column_groups(mask: Sequence[bool]) -> list[tuple[int, int]]:
    groups: list[tuple[int, int]] = []
    start: int | None = None
    for idx, flag in enumerate(mask):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            groups.append((start, idx - 1))
            start = None
    if start is not None:
        groups.append((start, len(mask) - 1))
    return groups


def _extract_template(grid: np.ndarray, bg: int) -> tuple[int, list[State]]:
    row_has_fg = np.any(grid != bg, axis=1)
    assert bool(row_has_fg[0]), "the template band must touch the top"

    band_h = 1
    while band_h < grid.shape[0] and bool(row_has_fg[band_h]):
        band_h += 1
    assert band_h < grid.shape[0], "expected a blank separator row below the top template band"

    band = grid[:band_h, :]
    raw_states: list[State] = []
    for c in range(grid.shape[1]):
        offsets = tuple(int(r) for r in np.flatnonzero(band[:, c] != bg))
        assert offsets, "every template column must contain foreground"
        raw_states.append(offsets)
    template = _compress_states(raw_states)
    assert len(template) >= 2, "the template must expose multiple compressed states"
    return band_h, template


def _find_strip_boxes(grid: np.ndarray, band_h: int, bg: int) -> list[tuple[int, int, int, int]]:
    below = grid[band_h:, :]
    col_mask = np.any(below != bg, axis=0)
    groups = _column_groups([bool(v) for v in col_mask])
    assert groups, "expected at least one vertical strip below the template band"

    boxes: list[tuple[int, int, int, int]] = []
    for c0, c1 in groups:
        rows = np.flatnonzero(np.any(grid[:, c0 : c1 + 1] != bg, axis=1))
        rows = rows[rows >= band_h]
        assert rows.size > 0
        r0, r1 = int(rows[0]), int(rows[-1])
        assert rows.size == r1 - r0 + 1, "each visible strip must occupy consecutive rows"
        boxes.append((r0, c0, r1, c1))
    return boxes


def _strip_sequences(box: np.ndarray, bg: int) -> tuple[list[State], list[State]]:
    top_to_bottom = [tuple(int(v) for v in np.flatnonzero(box[r] != bg)) for r in range(box.shape[0])]
    assert all(top_to_bottom), "every strip row must contain foreground"
    return _compress_states(top_to_bottom), _compress_states(list(reversed(top_to_bottom)))


def _extend_strip(
    out: np.ndarray,
    *,
    bg: int,
    c0: int,
    start_rows: Sequence[int],
    template: Sequence[State],
    prefix_len: int,
) -> None:
    last_index = len(template) - 1
    for step, row in enumerate(start_rows):
        state = template[min(prefix_len + step, last_index)]
        for offset in state:
            col = c0 + int(offset)
            if out[int(row), col] == bg:
                out[int(row), col] = BLUE


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each vertical strip by continuing the top template's compressed state order along its hidden writing direction.

    Input structure:
    - The background is black.
    - A complete horizontal foreground band spans the full width at the top; each column encodes one state by its occupied row offsets within the band.
    - Below it, several separated vertical strips share the same thickness as the template band height; each visible strip is a prefix of the template's compressed state sequence, but some are written top-to-bottom and some bottom-to-top.

    Rule:
    - Compress the top band column states by merging adjacent equal columns to obtain the template state sequence.
    - For each vertical strip, compress its row states both top-to-bottom and bottom-to-top; the unique orientation whose compressed sequence matches a template prefix is the strip's writing direction.
    - Keep all existing foreground cells, and extend the strip from its current tip in that writing direction with blue states: advance through the remaining template states one row per state, then keep repeating the final template state until the grid edge.

    Key insight:
    - The shared template is not the literal pixel picture but the abstract run-compressed order of offset-pattern states, so the hidden reading direction must be recovered before the missing continuation can be drawn.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    colors = sorted(set(map(int, np.unique(g))) - {bg})
    assert len(colors) == 1, "expected one foreground color in the input"

    band_h, template = _extract_template(g, bg)
    strip_boxes = _find_strip_boxes(g, band_h, bg)

    out = g.copy()
    for r0, c0, r1, c1 in strip_boxes:
        assert c1 - c0 + 1 == band_h, "strip width must match the template band height"
        box = g[r0 : r1 + 1, c0 : c1 + 1]
        down_seq, up_seq = _strip_sequences(box, bg)
        matches_down = _is_prefix(down_seq, template)
        matches_up = _is_prefix(up_seq, template)
        assert matches_down != matches_up, "strip direction must be uniquely determined"

        if matches_down:
            _extend_strip(
                out,
                bg=bg,
                c0=c0,
                start_rows=range(r1 + 1, g.shape[0]),
                template=template,
                prefix_len=len(down_seq),
            )
        else:
            _extend_strip(
                out,
                bg=bg,
                c0=c0,
                start_rows=range(r0 - 1, -1, -1),
                template=template,
                prefix_len=len(up_seq),
            )
    return out


def _contiguous_states(thickness: int) -> list[State]:
    states: list[State] = []
    for start in range(thickness):
        for end in range(start, thickness):
            states.append(tuple(range(start, end + 1)))
    return states


def _sample_template(thickness: int, n_states: int) -> list[State]:
    full = tuple(range(thickness))
    choices = [state for state in _contiguous_states(thickness) if state != full]
    while True:
        middle = random.sample(choices, n_states - 2)
        random.shuffle(middle)
        template = [full, *middle, full]
        if all(template[:k] != list(reversed(template[:k])) for k in range(2, len(template) + 1)):
            return template


def _sample_run_lengths(n_states: int, *, width_min: int, last_run_min: int) -> list[int] | None:
    runs = [random.randint(1, 3) for _ in range(n_states - 1)] + [random.randint(last_run_min, last_run_min + 4)]
    width = sum(runs)
    if width < width_min:
        runs[-1] += width_min - width
        width = sum(runs)
    if width > 30:
        return None
    return runs


def _expand_template(template: Sequence[State], runs: Sequence[int]) -> list[State]:
    raw: list[State] = []
    for state, run in zip(template, runs):
        raw.extend([state] * int(run))
    return raw


def _intervals_conflict(a0: int, width: int, b0: int, gap: int) -> bool:
    a1 = a0 + width - 1
    b1 = b0 + width - 1
    return not (a1 + gap < b0 or b1 + gap < a0)


def _pick_starts(allowed: list[int], *, count: int, width: int, gap: int) -> list[int] | None:
    starts = allowed[:]
    random.shuffle(starts)
    chosen: list[int] = []
    for start in starts:
        if all(not _intervals_conflict(start, width, other, gap) for other in chosen):
            chosen.append(start)
            if len(chosen) == count:
                return sorted(chosen)
    return None


def _render_input(
    *,
    height: int,
    width: int,
    fg: int,
    template_raw: Sequence[State],
    strip_specs: Sequence[dict],
) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    for c, state in enumerate(template_raw):
        for r in state:
            grid[int(r), c] = fg

    for spec in strip_specs:
        for local_r, state in enumerate(spec["rows"]):
            rr = int(spec["r0"] + local_r)
            for offset in state:
                grid[rr, int(spec["c0"] + offset)] = fg
    return grid


def _expected_output(
    inp: np.ndarray,
    *,
    template: Sequence[State],
    strip_specs: Sequence[dict],
) -> np.ndarray:
    out = inp.copy()
    for spec in strip_specs:
        if spec["orientation"] == "down":
            rows = range(spec["r1"] + 1, inp.shape[0])
        else:
            rows = range(spec["r0"] - 1, -1, -1)
        _extend_strip(
            out,
            bg=BG,
            c0=int(spec["c0"]),
            start_rows=rows,
            template=template,
            prefix_len=int(spec["prefix_len"]),
        )
    return out


def _validate_instance(
    inp: np.ndarray,
    expected: np.ndarray,
    *,
    template: Sequence[State],
    strip_specs: Sequence[dict],
) -> bool:
    if BLUE in inp:
        return False
    if inp.shape != expected.shape or inp.ndim != 2 or max(inp.shape) > 30:
        return False
    if not np.any(expected == BLUE):
        return False

    band_h, extracted_template = _extract_template(inp, BG)
    if extracted_template != list(template):
        return False

    boxes = sorted(_find_strip_boxes(inp, band_h, BG), key=lambda box: box[1])
    latent = sorted(strip_specs, key=lambda spec: spec["c0"])
    if len(boxes) != len(latent):
        return False

    for box, spec in zip(boxes, latent):
        r0, c0, r1, c1 = box
        if (r0, c0, r1, c1) != (spec["r0"], spec["c0"], spec["r1"], spec["c1"]):
            return False
        down_seq, up_seq = _strip_sequences(inp[r0 : r1 + 1, c0 : c1 + 1], BG)
        matches = (_is_prefix(down_seq, template), _is_prefix(up_seq, template))
        if matches != ((spec["orientation"] == "down"), (spec["orientation"] == "up")):
            return False

    try:
        actual = main(inp)
    except AssertionError:
        return False
    return np.array_equal(actual, expected)


def _gen_attempt() -> np.ndarray | None:
    thickness = random.randint(3, 5)
    n_states = random.randint(4, min(6, 2 + len(_contiguous_states(thickness)) - 1))
    template = _sample_template(thickness, n_states)

    n_strips = random.randint(2, 3)
    n_up = random.randint(1, min(2, n_strips - 1))
    n_down = n_strips - n_up
    col_gap = 1
    gap_rows = random.randint(2, 4)

    last_run_min = n_up * thickness + max(0, n_up - 1) * (col_gap + 1)
    width_min = n_strips * thickness + max(0, n_strips - 1) * (col_gap + 1)
    runs = _sample_run_lengths(n_states, width_min=width_min, last_run_min=last_run_min)
    if runs is None:
        return None
    width = sum(runs)

    last_run = runs[-1]
    last_start = width - last_run
    up_allowed = list(range(last_start, width - thickness + 1))
    up_starts = _pick_starts(up_allowed, count=n_up, width=thickness, gap=col_gap)
    if up_starts is None:
        return None

    preferred_down = [start for start in range(0, max(0, last_start - thickness + 1))]
    down_allowed = preferred_down if len(preferred_down) >= n_down else list(range(0, width - thickness + 1))
    down_allowed = [
        start
        for start in down_allowed
        if all(not _intervals_conflict(start, thickness, other, col_gap) for other in up_starts)
    ]
    down_starts = _pick_starts(down_allowed, count=n_down, width=thickness, gap=col_gap)
    if down_starts is None:
        return None

    template_raw = _expand_template(template, runs)
    fg = random.choice(FG_CHOICES)

    strip_specs: list[dict] = []
    orientations = ["up"] * n_up + ["down"] * n_down
    starts_by_orientation = {"up": up_starts[:], "down": down_starts[:]}
    random.shuffle(orientations)

    for orientation in orientations:
        if orientation == "up":
            prefix_len = random.choice([len(template) - 1, len(template)])
        else:
            prefix_len = random.randint(2, len(template))

        run_counts = [random.randint(1, 2) for _ in range(prefix_len)]
        if prefix_len == len(template):
            run_counts[-1] += random.randint(1, 3)
        rows_in_writing_order: list[State] = []
        for state, count in zip(template[:prefix_len], run_counts):
            rows_in_writing_order.extend([state] * count)
        visible_rows = rows_in_writing_order if orientation == "down" else list(reversed(rows_in_writing_order))

        strip_specs.append(
            {
                "orientation": orientation,
                "prefix_len": prefix_len,
                "rows": visible_rows,
                "height": len(visible_rows),
                "c0": starts_by_orientation[orientation].pop(0),
            }
        )

    min_height = thickness + gap_rows + max(spec["height"] for spec in strip_specs) + 3
    if min_height > 30:
        return None
    height = random.randint(max(18, min_height), 30)

    for spec in strip_specs:
        top_min = thickness + gap_rows
        top_max = height - spec["height"] - 1
        if top_min > top_max:
            return None
        spec["r0"] = random.randint(top_min, top_max)
        spec["r1"] = spec["r0"] + spec["height"] - 1
        spec["c1"] = spec["c0"] + thickness - 1

    inp = _render_input(height=height, width=width, fg=fg, template_raw=template_raw, strip_specs=strip_specs)
    expected = _expected_output(inp, template=template, strip_specs=strip_specs)
    if not _validate_instance(inp, expected, template=template, strip_specs=strip_specs):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid state-sequence strip input")
