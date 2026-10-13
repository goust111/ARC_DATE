from __future__ import annotations

import itertools
import random

import numpy as np


BG = 0
FRAME = 5
PALETTE = tuple(color for color in range(1, 10) if color != FRAME)
MAX_GENERATION_ATTEMPTS = 400


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    height, width = map(int, grid.shape)
    seen = np.zeros((height, width), dtype=bool)
    components: list[np.ndarray] = []

    for r in range(height):
        for c in range(width):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = cr + dr
                    nc = cc + dc
                    if not (0 <= nr < height and 0 <= nc < width):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != int(color):
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))

    return components


def _nonzero_runs(row: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    width = int(row.shape[0])
    c = 0
    while c < width:
        while c < width and int(row[c]) == BG:
            c += 1
        if c >= width:
            break
        start = c
        while c < width and int(row[c]) != BG:
            c += 1
        runs.append((start, c))
    return runs


def _find_template_row(grid: np.ndarray) -> tuple[int, np.ndarray, int, int]:
    candidates = [r for r in range(grid.shape[0]) if np.all(grid[r] != BG)]
    assert len(candidates) == 1, "expected one full-width template row"
    row_index = int(candidates[0])
    row = grid[row_index].copy()
    colors = sorted({int(v) for v in row.tolist()})
    assert FRAME not in colors, "template row must not use the frame color"
    assert len(colors) == 2, "template row must use exactly two colors"
    color_a = int(row[0])
    color_b = colors[0] if colors[1] == color_a else colors[1]
    assert int(np.count_nonzero(row == color_a)) >= 2
    assert int(np.count_nonzero(row == color_b)) >= 2
    return row_index, row, color_a, color_b


def _find_frames(grid: np.ndarray) -> list[dict]:
    frames: list[dict] = []
    for coords in _components_of_color(grid, FRAME):
        r0, c0, r1, c1 = _bbox(coords)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        assert 3 <= height <= 5 and 3 <= width <= 5, "frame must be 3x3 to 5x5"

        coord_set = {tuple(map(int, rc)) for rc in coords.tolist()}
        border = {
            (r, c)
            for r in range(r0, r1 + 1)
            for c in range(c0, c1 + 1)
            if r in (r0, r1) or c in (c0, c1)
        }
        assert coord_set == border, "gray component must be a one-pixel-thick rectangle border"
        assert np.all(grid[r0 + 1 : r1, c0 + 1 : c1] == BG), "frame interior must start blank"
        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "height": height,
                "width": width,
            }
        )

    assert frames, "expected at least one gray frame"
    frames.sort(key=lambda frame: frame["bbox"])
    return frames


def _find_prefix_rows(grid: np.ndarray, template_row: int) -> list[dict]:
    prefixes: list[dict] = []
    for r in range(grid.shape[0]):
        if r == template_row:
            continue
        runs = _nonzero_runs(grid[r])
        if not runs:
            continue
        start, end = runs[0]
        if start > 3 or not (2 <= end - start <= 4):
            continue
        segment = grid[r, start:end].copy()
        colors = {int(v) for v in segment.tolist()}
        if FRAME in colors or len(colors) != 2:
            continue
        tail = grid[r, end:]
        if np.any((tail != BG) & (tail != FRAME)):
            continue
        prefixes.append(
            {
                "row": int(r),
                "start": int(start),
                "end": int(end),
                "segment": segment,
            }
        )

    assert len(prefixes) >= 4, "expected several prefix rows"
    prefixes.sort(key=lambda item: item["row"])
    return prefixes


def _infer_mapping(
    template_row: np.ndarray,
    prefix: dict,
    color_a: int,
    color_b: int,
) -> dict[int, int]:
    start = int(prefix["start"])
    end = int(prefix["end"])
    template_segment = template_row[start:end]
    prefix_segment = prefix["segment"]
    assert len({int(v) for v in template_segment.tolist()}) == 2, "prefix must observe both template colors"

    mapping: dict[int, int] = {}
    inverse: dict[int, int] = {}
    for template_color, prefix_color in zip(template_segment.tolist(), prefix_segment.tolist()):
        template_color = int(template_color)
        prefix_color = int(prefix_color)
        if template_color in mapping:
            assert mapping[template_color] == prefix_color, "template-to-prefix mapping must be consistent"
        else:
            mapping[template_color] = prefix_color
        if prefix_color in inverse:
            assert inverse[prefix_color] == template_color, "prefix colors must map back one-to-one"
        else:
            inverse[prefix_color] = template_color

    assert set(mapping) == {int(color_a), int(color_b)}
    assert len(set(mapping.values())) == 2
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """Use template-aligned prefix segments to recolor each framed cavity with a checkerboard fill.

    Input structure:
    - The background is black/0, and exactly one full-width non-background row is a two-color template row.
    - Several short two-color prefix segments appear near the left edge on rows that intersect distinct gray hollow rectangles.
    - Every gray component is a closed 1-pixel-thick rectangle with a blank interior, and each rectangle's row span contains exactly one prefix row.

    Rule:
    - Identify the unique template row and its two colors; the template color appearing first from the left is color A and the other is color B.
    - For each prefix row, align its short segment with the same template columns to infer a bijection from the template's two colors to that row's two prefix colors.
    - Find the unique gray rectangle whose row span contains that prefix row, and fill the rectangle interior as a checkerboard using the inferred colors: cells with even global row+col parity get the image of color A, and odd cells get the image of color B.
    - Erase the template row and every prefix segment, keep the gray frames, and leave all other cells as background.

    Key insight:
    - The short prefix segment is not output content; it is a row-local decoder that translates the template's abstract two-color alphabet into the concrete fill colors used inside one particular enclosed region.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    template_row_index, template_row, color_a, color_b = _find_template_row(g)
    frames = _find_frames(g)
    prefixes = _find_prefix_rows(g, template_row_index)
    assert len(prefixes) == len(frames), "expected one frame per prefix row"

    out = np.zeros_like(g)
    used_frames: set[int] = set()

    for prefix in prefixes:
        frame_matches = [
            index
            for index, frame in enumerate(frames)
            if frame["bbox"][0] <= int(prefix["row"]) <= frame["bbox"][2]
        ]
        assert len(frame_matches) == 1, "each prefix row must point to exactly one frame"
        frame_index = frame_matches[0]
        assert frame_index not in used_frames, "each frame must be used exactly once"
        used_frames.add(frame_index)

        frame = frames[frame_index]
        r0, c0, r1, c1 = frame["bbox"]
        mapping = _infer_mapping(template_row, prefix, color_a, color_b)
        fill_even = int(mapping[color_a])
        fill_odd = int(mapping[color_b])

        out[r0 : r1 + 1, c0] = FRAME
        out[r0 : r1 + 1, c1] = FRAME
        out[r0, c0 : c1 + 1] = FRAME
        out[r1, c0 : c1 + 1] = FRAME

        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                out[r, c] = fill_even if (r + c) % 2 == 0 else fill_odd

    assert len(used_frames) == len(frames), "every frame must receive one fill"
    return out


def _draw_frame(grid: np.ndarray, r0: int, c0: int, height: int, width: int) -> None:
    r1 = r0 + height - 1
    c1 = c0 + width - 1
    grid[r0 : r1 + 1, c0] = FRAME
    grid[r0 : r1 + 1, c1] = FRAME
    grid[r0, c0 : c1 + 1] = FRAME
    grid[r1, c0 : c1 + 1] = FRAME


def _render_expected(spec: dict) -> np.ndarray:
    out = np.zeros((spec["height"], spec["width"]), dtype=int)
    for row_spec in spec["rows"]:
        r0 = int(row_spec["frame_r0"])
        c0 = int(row_spec["frame_c0"])
        height = int(row_spec["frame_h"])
        width = int(row_spec["frame_w"])
        _draw_frame(out, r0, c0, height, width)
        fill_even = int(row_spec["map_a"])
        fill_odd = int(row_spec["map_b"])
        for r in range(r0 + 1, r0 + height - 1):
            for c in range(c0 + 1, c0 + width - 1):
                out[r, c] = fill_even if (r + c) % 2 == 0 else fill_odd
    return out


def _build_input(spec: dict) -> np.ndarray:
    grid = np.zeros((spec["height"], spec["width"]), dtype=int)

    template_values = np.array(
        [spec["template_colors"][bit] for bit in spec["template_bits"]],
        dtype=int,
    )
    grid[int(spec["template_row"]), :] = template_values

    for row_spec in spec["rows"]:
        _draw_frame(
            grid,
            int(row_spec["frame_r0"]),
            int(row_spec["frame_c0"]),
            int(row_spec["frame_h"]),
            int(row_spec["frame_w"]),
        )
        start = int(row_spec["prefix_start"])
        end = start + int(row_spec["prefix_len"])
        grid[int(row_spec["prefix_row"]), start:end] = np.array(row_spec["prefix_segment"], dtype=int)

    return grid


def _sample_vertical_plan(frame_specs: list[dict]) -> tuple[int, int]:
    heights = [int(spec["frame_h"]) for spec in frame_specs]
    for _ in range(200):
        n_top = random.randint(2, 3)
        n_bottom = len(frame_specs) - n_top
        if n_bottom not in (2, 3):
            continue

        top_margin = 1
        bottom_margin = 1
        top_between = [random.randint(0, 1) for _ in range(max(0, n_top - 1))]
        bottom_between = [random.randint(0, 1) for _ in range(max(0, n_bottom - 1))]
        gap_above_template = 1
        gap_below_template = 1

        top_r0s: list[int] = []
        r = top_margin
        for i in range(n_top):
            top_r0s.append(r)
            r += heights[i]
            if i < n_top - 1:
                r += top_between[i]
            else:
                r += gap_above_template
        template_row = r

        bottom_r0s: list[int] = []
        r = template_row + 1 + gap_below_template
        for j in range(n_bottom):
            idx = n_top + j
            bottom_r0s.append(r)
            r += heights[idx]
            if j < n_bottom - 1:
                r += bottom_between[j]
            else:
                r += bottom_margin
        height = r
        if height > 30:
            continue

        for spec, frame_r0 in zip(frame_specs[:n_top], top_r0s):
            spec["frame_r0"] = int(frame_r0)
            spec["prefix_row"] = int(frame_r0 + spec["row_offset"])
        for spec, frame_r0 in zip(frame_specs[n_top:], bottom_r0s):
            spec["frame_r0"] = int(frame_r0)
            spec["prefix_row"] = int(frame_r0 + spec["row_offset"])
        return height, template_row

    raise RuntimeError("failed to place frames vertically")


def _template_prefix_bits(starts: list[int], lengths: list[int], forced_pair: tuple[int, int]) -> list[int]:
    window = max(start + length for start, length in zip(starts, lengths))
    candidates: list[tuple[int, ...]] = []

    for tail in itertools.product((0, 1), repeat=window - 1):
        bits = (0,) + tail
        if any(len(set(bits[start : start + length])) != 2 for start, length in zip(starts, lengths)):
            continue
        if bits[forced_pair[0]] == bits[forced_pair[1]]:
            continue
        if all(bits[i] != bits[i - 1] for i in range(1, window)):
            continue
        candidates.append(bits)

    assert candidates, "failed to build a non-degenerate template prefix"
    return list(random.choice(candidates))


def _extend_template_bits(prefix_bits: list[int], width: int) -> list[int]:
    bits = list(prefix_bits)
    while len(bits) < width:
        prev = bits[-1]
        if len(bits) >= 2 and bits[-1] != bits[-2]:
            next_bit = prev if random.random() < 0.45 else 1 - prev
        else:
            next_bit = 1 - prev if random.random() < 0.6 else prev
        bits.append(int(next_bit))

    if all(bits[i] != bits[i - 1] for i in range(1, width)):
        pivot = random.randint(len(prefix_bits), width - 1) if width > len(prefix_bits) else random.randint(1, width - 1)
        bits[pivot] = bits[pivot - 1]
    return bits


def _make_frame_specs(count: int) -> list[dict]:
    specs: list[dict] = []
    big_index = random.randrange(count)
    for i in range(count):
        if count >= 6:
            frame_h = 4 if i == big_index else random.randint(3, 4)
            frame_w = random.randint(3, 5) if i != big_index else random.randint(4, 5)
        elif i == big_index:
            frame_h = random.randint(4, 5)
            frame_w = random.randint(4, 5)
        else:
            frame_h = random.randint(3, 5)
            frame_w = random.randint(3, 5)
        specs.append({"frame_h": frame_h, "frame_w": frame_w})

    for i, spec in enumerate(specs):
        frame_h = int(spec["frame_h"])
        if i == 0:
            row_offset = 0
        elif i == 1:
            row_offset = frame_h - 1
        elif i == 2:
            row_offset = frame_h // 2
        else:
            row_offset = random.randrange(frame_h)
        spec["row_offset"] = int(row_offset)

    return specs


def _assign_prefix_controls(frame_specs: list[dict]) -> tuple[list[int], list[int], tuple[int, int]]:
    count = len(frame_specs)
    if random.random() < 0.5:
        forced_pair = (0, 2)
    else:
        forced_pair = (1, 3)

    starts = [forced_pair[0], forced_pair[1]]
    while len(starts) < count:
        starts.append(random.randint(0, 3))
    random.shuffle(starts)

    lengths = [random.randint(2, 4) for _ in range(count)]
    return starts, lengths, forced_pair


def _assign_prefix_colors(frame_specs: list[dict], prefix_pool: tuple[int, ...]) -> None:
    pairs = list(itertools.combinations(prefix_pool, 2))
    shared_pair = random.choice(pairs)

    frame_specs[0]["map_a"], frame_specs[0]["map_b"] = int(shared_pair[0]), int(shared_pair[1])
    frame_specs[1]["map_a"], frame_specs[1]["map_b"] = int(shared_pair[1]), int(shared_pair[0])

    for spec in frame_specs[2:]:
        pair = shared_pair if random.random() < 0.35 else random.choice(pairs)
        if random.random() < 0.5:
            spec["map_a"], spec["map_b"] = int(pair[0]), int(pair[1])
        else:
            spec["map_a"], spec["map_b"] = int(pair[1]), int(pair[0])


def _populate_prefix_segments(
    frame_specs: list[dict],
    template_bits: list[int],
    template_colors: tuple[int, int],
) -> None:
    color_a, color_b = template_colors
    for spec in frame_specs:
        start = int(spec["prefix_start"])
        length = int(spec["prefix_len"])
        segment: list[int] = []
        for offset in range(length):
            bit = int(template_bits[start + offset])
            if bit == 0:
                segment.append(int(spec["map_a"]))
            else:
                segment.append(int(spec["map_b"]))
        assert len(set(segment)) == 2
        spec["prefix_segment"] = tuple(segment)
        spec["template_segment"] = tuple(template_colors[bit] for bit in template_bits[start : start + length])


def _place_frame_columns(frame_specs: list[dict], width: int) -> None:
    max_prefix_end = max(int(spec["prefix_start"]) + int(spec["prefix_len"]) for spec in frame_specs)
    min_left = max_prefix_end + 2
    for spec in frame_specs:
        frame_w = int(spec["frame_w"])
        assert min_left <= width - frame_w - 1, "grid is too narrow for frame placement"
        spec["frame_c0"] = random.randint(min_left, width - frame_w - 1)


def _build_spec() -> dict:
    total_rows = random.randint(4, 6)
    frame_specs = _make_frame_specs(total_rows)
    height, template_row = _sample_vertical_plan(frame_specs)
    width = random.randint(18, 24)

    starts, lengths, forced_pair = _assign_prefix_controls(frame_specs)
    for spec, start, length in zip(frame_specs, starts, lengths):
        spec["prefix_start"] = int(start)
        spec["prefix_len"] = int(length)

    template_colors = tuple(random.sample(PALETTE, 2))
    prefix_pool = tuple(color for color in PALETTE if color not in template_colors)
    _assign_prefix_colors(frame_specs, prefix_pool)

    prefix_bits = _template_prefix_bits(starts, lengths, forced_pair)
    template_bits = _extend_template_bits(prefix_bits, width)
    _populate_prefix_segments(frame_specs, template_bits, template_colors)
    _place_frame_columns(frame_specs, width)

    for spec in frame_specs:
        spec["frame_c1"] = int(spec["frame_c0"]) + int(spec["frame_w"]) - 1
        spec["frame_r1"] = int(spec["frame_r0"]) + int(spec["frame_h"]) - 1

    return {
        "height": int(height),
        "width": int(width),
        "template_row": int(template_row),
        "template_colors": tuple(map(int, template_colors)),
        "template_bits": tuple(int(bit) for bit in template_bits),
        "rows": frame_specs,
    }


def _semantic_checks(spec: dict, grid: np.ndarray, expected: np.ndarray) -> bool:
    try:
        solved = main(grid)
    except AssertionError:
        return False
    if not np.array_equal(solved, expected):
        return False

    row_specs = spec["rows"]
    if not any(int(row["prefix_start"]) > 0 for row in row_specs):
        return False

    needs_template = any(
        int(row["prefix_segment"][0]) != int(row["map_a"])
        for row in row_specs
    )
    if not needs_template:
        return False

    first_pair = frozenset((int(row_specs[0]["map_a"]), int(row_specs[0]["map_b"])))
    second_pair = frozenset((int(row_specs[1]["map_a"]), int(row_specs[1]["map_b"])))
    if first_pair != second_pair or int(row_specs[0]["map_a"]) == int(row_specs[1]["map_a"]):
        return False

    zero_template = grid.copy()
    zero_template[int(spec["template_row"]), :] = BG
    try:
        changed = not np.array_equal(main(zero_template), expected)
    except AssertionError:
        changed = True
    if not changed:
        return False

    witness = next(row for row in row_specs if int(row["prefix_segment"][0]) != int(row["map_a"]))
    swapped = grid.copy()
    prefix_row = int(witness["prefix_row"])
    start = int(witness["prefix_start"])
    length = int(witness["prefix_len"])
    colors = tuple(sorted({int(v) for v in witness["prefix_segment"]}))
    flip = {colors[0]: colors[1], colors[1]: colors[0]}
    for c in range(start, start + length):
        swapped[prefix_row, c] = flip[int(swapped[prefix_row, c])]
    try:
        changed = not np.array_equal(main(swapped), expected)
    except AssertionError:
        changed = True
    if not changed:
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            spec = _build_spec()
            grid = _build_input(spec)
            expected = _render_expected(spec)
        except (AssertionError, RuntimeError):
            continue
        if _semantic_checks(spec, grid, expected):
            return grid
    raise RuntimeError("failed to generate a valid prefix-mapped framed-fill puzzle")
