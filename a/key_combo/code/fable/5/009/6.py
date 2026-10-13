from __future__ import annotations

from itertools import combinations
import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]

_MAX_GENERATION_ATTEMPTS = 300
_GENERATION_COUNTER = 0
_GENERATED_OUTPUTS: set[tuple[tuple[int, int], bytes]] = set()


def _background(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _bbox(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    """Return 4-connected monochromatic foreground components in (r, c) coordinates."""
    height, width = grid.shape
    seen = np.zeros(grid.shape, dtype=bool)
    result: list[dict[str, object]] = []
    for r0 in range(height):
        for c0 in range(width):
            if seen[r0, c0] or int(grid[r0, c0]) == bg:
                continue
            color = int(grid[r0, c0])
            stack = [(r0, c0)]
            seen[r0, c0] = True
            cells: list[Coord] = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if (
                        0 <= rr < height
                        and 0 <= cc < width
                        and not seen[rr, cc]
                        and int(grid[rr, cc]) == color
                    ):
                        seen[rr, cc] = True
                        stack.append((rr, cc))
            coords = np.asarray(cells, dtype=int)
            result.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": int(len(cells)),
                }
            )
    return result


def _is_rectangular_frame(component: dict[str, object]) -> bool:
    r0, c0, r1, c1 = component["bbox"]  # type: ignore[misc]
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    if height < 7 or width < 7:
        return False
    perimeter = {
        *((r0, c) for c in range(c0, c1 + 1)),
        *((r1, c) for c in range(c0, c1 + 1)),
        *((r, c0) for r in range(r0, r1 + 1)),
        *((r, c1) for r in range(r0, r1 + 1)),
    }
    coords = {tuple(map(int, rc)) for rc in component["coords"]}  # type: ignore[union-attr]
    return coords == perimeter


def _parse(grid: np.ndarray) -> tuple[int, dict[str, object], list[tuple[int, int, int]], list[dict[str, object]]]:
    bg = _background(grid)
    components = _components(grid, bg)
    frames = [component for component in components if _is_rectangular_frame(component)]
    assert len(frames) == 1, "expected one uniquely identifiable closed rectangular frame"
    frame = frames[0]
    fr0, fc0, fr1, fc1 = frame["bbox"]  # type: ignore[misc]

    placeholders: list[tuple[int, int, int]] = []
    sprites: list[dict[str, object]] = []
    for component in components:
        if component is frame:
            continue
        coords = component["coords"]  # type: ignore[assignment]
        inside = [fr0 < int(r) < fr1 and fc0 < int(c) < fc1 for r, c in coords]
        if all(inside):
            assert int(component["area"]) == 1, "every interior foreground component must be one placeholder pixel"
            r, c = map(int, coords[0])
            placeholders.append((r, c, int(component["color"])))
        else:
            assert not any(inside), "a sprite must lie wholly outside the frame"
            sprites.append(component)

    placeholders.sort(key=lambda item: (item[0], item[1]))
    sprites.sort(key=lambda component: -int(component["area"]))
    assert len(placeholders) >= 2 and len(placeholders) == len(sprites), "placeholder and sprite counts must match"
    areas = [int(component["area"]) for component in sprites]
    assert len(areas) == len(set(areas)), "sprite pixel counts must give unique ranks"
    for component in sprites:
        r0, c0, r1, c1 = component["bbox"]  # type: ignore[misc]
        assert (r1 - r0 + 1) % 2 == 1 and (c1 - c0 + 1) % 2 == 1, "sprite bboxes must have odd dimensions"
    return bg, frame, placeholders, sprites


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp area-ranked outer sprites onto reading-order-ranked inner placeholders and recolor each stamp.

    Input structure:
    - A dominant background contains one closed monochromatic rectangular frame.
    - The strict frame interior contains isolated colored one-pixel placeholders.
    - The same number of monochromatic 4-connected sprites lies wholly outside
      the frame; their pixel counts are distinct and both bbox dimensions are odd.

    Rule:
    - Rank sprites by decreasing pixel count and placeholders in row-major reading order.
    - Pair equal ranks, translate each sprite so its bbox center coincides with its
      placeholder, and recolor all stamped pixels to that placeholder's color.
    - Erase every original outer sprite while preserving the frame.

    Key insight:
    - Source color is a decoy: the correspondence is rank-to-rank between two
      independently ordered object sets, followed by center-aligned recoloring.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    bg, frame, placeholders, sprites = _parse(g)
    fr0, fc0, fr1, fc1 = frame["bbox"]  # type: ignore[misc]

    out = g.copy()
    for sprite in sprites:
        coords = sprite["coords"]  # type: ignore[assignment]
        out[coords[:, 0], coords[:, 1]] = bg

    occupied: set[Coord] = set()
    placeholder_coords = {(r, c) for r, c, _color in placeholders}
    for (target_r, target_c, target_color), sprite in zip(placeholders, sprites):
        r0, c0, r1, c1 = sprite["bbox"]  # type: ignore[misc]
        center_r = (r0 + r1) // 2
        center_c = (c0 + c1) // 2
        placed: set[Coord] = set()
        for r, c in sprite["coords"]:  # type: ignore[union-attr]
            rr = target_r + int(r) - center_r
            cc = target_c + int(c) - center_c
            assert fr0 < rr < fr1 and fc0 < cc < fc1, "a stamp must fit strictly inside the frame"
            placed.add((rr, cc))
        assert not (placed & occupied), "rank-paired stamps must not overlap"
        assert not ((placed - {(target_r, target_c)}) & placeholder_coords), "a stamp must not cover another placeholder"
        for rr, cc in placed:
            out[rr, cc] = target_color
        occupied.update(placed)
    return out


def _connected(mask: np.ndarray) -> bool:
    cells = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    if not cells:
        return False
    remaining = set(cells)
    stack = [remaining.pop()]
    while stack:
        r, c = stack.pop()
        for neighbor in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if neighbor in remaining:
                remaining.remove(neighbor)
                stack.append(neighbor)
    return not remaining


def _make_shape(area: int) -> np.ndarray:
    """Sample a connected 3x3 mask with occupied center and a true odd 3x3 bbox."""
    candidates: list[np.ndarray] = []
    cells = list(range(9))
    for chosen in combinations(cells, area):
        if 4 not in chosen:
            continue
        mask = np.zeros((3, 3), dtype=bool)
        mask.flat[list(chosen)] = True
        if not (
            np.any(mask[0])
            and np.any(mask[2])
            and np.any(mask[:, 0])
            and np.any(mask[:, 2])
            and _connected(mask)
        ):
            continue
        candidates.append(mask)
    if not candidates:
        raise AssertionError(f"no valid shape for area {area}")
    return random.choice(candidates).copy()


def _spaced_starts(length: int, count: int) -> list[int]:
    """Place 3-cell objects along one axis with at least two blank cells between bboxes."""
    minimum = 3 * count + 2 * (count - 1)
    slack = (length - 2) - minimum
    assert slack >= 0
    extras = [0] * (count + 1)
    for _ in range(slack):
        extras[random.randrange(count + 1)] += 1
    starts: list[int] = []
    cursor = 1 + extras[0]
    for i in range(count):
        starts.append(cursor)
        cursor += 3
        if i + 1 < count:
            cursor += 2 + extras[i + 1]
    return starts


def _layout(count: int) -> tuple[tuple[int, int], BBox, list[Coord]]:
    """Choose one of four outside strips, a large frame, and shuffled sprite slots."""
    side = random.choice(("left", "right", "top", "bottom"))
    if side in {"left", "right"}:
        height = random.randint(max(22, 5 * count), 30)
        width = random.randint(23, 30)
        row_starts = _spaced_starts(height, count)
        fr0 = random.randint(1, 2)
        fr1 = random.randint(max(fr0 + 15, height - 3), height - 2)
        if side == "left":
            fc0 = random.randint(5, 7)
            fc1 = random.randint(max(fc0 + 14, width - 3), width - 2)
            col = random.randint(1, fc0 - 4)
        else:
            fc1 = random.randint(width - 8, width - 6)
            fc0 = random.randint(1, min(2, fc1 - 14))
            col = random.randint(fc1 + 2, width - 4)
        slots = [(row, col) for row in row_starts]
    else:
        width = random.randint(max(22, 5 * count), 30)
        height = random.randint(23, 30)
        col_starts = _spaced_starts(width, count)
        fc0 = random.randint(1, 2)
        fc1 = random.randint(max(fc0 + 15, width - 3), width - 2)
        if side == "top":
            fr0 = random.randint(5, 7)
            fr1 = random.randint(max(fr0 + 14, height - 3), height - 2)
            row = random.randint(1, fr0 - 4)
        else:
            fr1 = random.randint(height - 8, height - 6)
            fr0 = random.randint(1, min(2, fr1 - 14))
            row = random.randint(fr1 + 2, height - 4)
        slots = [(row, col) for col in col_starts]
    return (height, width), (fr0, fc0, fr1, fc1), slots


def _target_centers(frame: BBox, count: int) -> list[Coord] | None:
    fr0, fc0, fr1, fc1 = frame
    candidates = [
        (r, c)
        for r in range(fr0 + 3, fr1 - 2)
        for c in range(fc0 + 3, fc1 - 2)
    ]
    random.shuffle(candidates)
    selected: list[Coord] = []
    for candidate in candidates:
        r, c = candidate
        if all(abs(r - rr) >= 5 or abs(c - cc) >= 5 for rr, cc in selected):
            selected.append(candidate)
            if len(selected) == count:
                return sorted(selected)
    return None


def _rank_colors(count: int, bg: int, frame_color: int, mode: int) -> tuple[list[int], list[int]]:
    palette = [color for color in range(10) if color not in {bg, frame_color}]
    placeholder = random.sample(palette, count)
    if mode == 0:
        source = placeholder[1:] + placeholder[:1]
    elif mode == 1:
        source = [placeholder[0], *placeholder[2:], placeholder[1]]
    else:
        source = placeholder[1:] + placeholder[:1]
        outside = [color for color in palette if color not in placeholder]
        source[-1] = random.choice(outside)
    assert len(source) == len(set(source))
    assert any(source[i] != placeholder[i] for i in range(count))
    assert any(color in placeholder for color in source), "each generated puzzle must contain a visible same-color decoy"
    return placeholder, source


def _paint_frame(grid: np.ndarray, frame: BBox, color: int) -> None:
    r0, c0, r1, c1 = frame
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _expected_output(
    inp: np.ndarray,
    bg: int,
    ranked_masks: list[np.ndarray],
    source_locations: list[Coord],
    targets: list[Coord],
    target_colors: list[int],
) -> np.ndarray:
    out = inp.copy()
    for rank, (r0, c0) in enumerate(source_locations):
        mask = ranked_masks[rank]
        out[r0 : r0 + 3, c0 : c0 + 3][mask] = bg
    for mask, (r, c), color in zip(ranked_masks, targets, target_colors):
        patch = out[r - 1 : r + 2, c - 1 : c + 2]
        patch[mask] = color
    return out


def _shortcut_output(
    inp: np.ndarray,
    bg: int,
    ranked_masks: list[np.ndarray],
    source_locations: list[Coord],
    source_colors: list[int],
    targets: list[Coord],
    target_colors: list[int],
) -> np.ndarray:
    """Oracle for the deliberately wrong same-color-pairing interpretation."""
    out = inp.copy()
    for rank, (r0, c0) in enumerate(source_locations):
        out[r0 : r0 + 3, c0 : c0 + 3][ranked_masks[rank]] = bg
    by_color = {color: target for color, target in zip(target_colors, targets)}
    for mask, source_color in zip(ranked_masks, source_colors):
        if source_color not in by_color:
            continue
        r, c = by_color[source_color]
        out[r - 1 : r + 2, c - 1 : c + 2][mask] = source_color
    return out


def _gen_attempt(mode: int) -> tuple[np.ndarray, np.ndarray] | None:
    count = random.randint(3, 5)
    shape, frame, slots = _layout(count)
    height, width = shape
    targets = _target_centers(frame, count)
    if targets is None:
        return None

    bg, frame_color = random.sample(range(10), 2)
    target_colors, source_colors = _rank_colors(count, bg, frame_color, mode)
    areas = sorted(random.sample(range(5, 10), count), reverse=True)
    ranked_masks = [_make_shape(area) for area in areas]

    slot_order = list(range(count))
    for _ in range(20):
        random.shuffle(slot_order)
        if slot_order != list(range(count)):
            break
    if slot_order == list(range(count)):
        return None
    source_locations: list[Coord] = [(-1, -1)] * count
    for slot, rank in zip(slots, slot_order):
        source_locations[rank] = slot

    inp = np.full((height, width), bg, dtype=int)
    _paint_frame(inp, frame, frame_color)
    for (r, c), color in zip(targets, target_colors):
        inp[r, c] = color
    for rank, (r0, c0) in enumerate(source_locations):
        inp[r0 : r0 + 3, c0 : c0 + 3][ranked_masks[rank]] = source_colors[rank]

    expected = _expected_output(inp, bg, ranked_masks, source_locations, targets, target_colors)
    try:
        parsed_bg, parsed_frame, parsed_targets, parsed_sprites = _parse(inp)
        solved = main(inp)
    except (AssertionError, IndexError, ValueError):
        return None
    if parsed_bg != bg or parsed_frame["bbox"] != frame:
        return None
    if [(r, c, color) for (r, c), color in zip(targets, target_colors)] != parsed_targets:
        return None
    if [int(sprite["area"]) for sprite in parsed_sprites] != areas:
        return None
    if not np.array_equal(solved, expected) or np.array_equal(inp, expected):
        return None

    # Semantic anti-shortcut checks: color, source reading order, ascending area,
    # source-color preservation, and failure to erase must each change the answer.
    color_shortcut = _shortcut_output(
        inp, bg, ranked_masks, source_locations, source_colors, targets, target_colors
    )
    if np.array_equal(color_shortcut, expected):
        return None

    source_reading_ranks = sorted(range(count), key=lambda rank: source_locations[rank])
    reading_masks = [ranked_masks[rank] for rank in source_reading_ranks]
    reading_out = _expected_output(inp, bg, reading_masks, [source_locations[rank] for rank in source_reading_ranks], targets, target_colors)
    if np.array_equal(reading_out, expected):
        return None

    ascending_out = _expected_output(inp, bg, list(reversed(ranked_masks)), list(reversed(source_locations)), targets, target_colors)
    if np.array_equal(ascending_out, expected):
        return None

    reverse_target_out = _expected_output(
        inp,
        bg,
        ranked_masks,
        source_locations,
        list(reversed(targets)),
        list(reversed(target_colors)),
    )
    if np.array_equal(reverse_target_out, expected):
        return None

    top_left_out = inp.copy()
    for rank, (r0, c0) in enumerate(source_locations):
        top_left_out[r0 : r0 + 3, c0 : c0 + 3][ranked_masks[rank]] = bg
    for mask, (r, c), color in zip(ranked_masks, targets, target_colors):
        top_left_out[r : r + 3, c : c + 3][mask] = color
    if np.array_equal(top_left_out, expected):
        return None

    no_recolor = _expected_output(inp, bg, ranked_masks, source_locations, targets, source_colors)
    if np.array_equal(no_recolor, expected):
        return None
    not_erased = expected.copy()
    for rank, (r0, c0) in enumerate(source_locations):
        not_erased[r0 : r0 + 3, c0 : c0 + 3][ranked_masks[rank]] = source_colors[rank]
    if np.array_equal(not_erased, expected):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    global _GENERATION_COUNTER
    mode = _GENERATION_COUNTER % 3
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        generated = _gen_attempt(mode)
        if generated is None:
            continue
        inp, expected = generated
        signature = (tuple(map(int, expected.shape)), expected.tobytes())
        if signature in _GENERATED_OUTPUTS:
            continue
        _GENERATED_OUTPUTS.add(signature)
        _GENERATION_COUNTER += 1
        return inp
    raise RuntimeError("failed to generate a valid rank-paired stamp puzzle")
