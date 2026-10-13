from __future__ import annotations

import itertools
import random

import numpy as np


BG = 0
SEPARATOR_COLOR = 5
OBSTACLE_COLOR = 9
SPRITE_SIZE = 3
SPRITE_RADIUS = 1
LEGEND_MARKER_ROW = 4
LEGEND_HEIGHT = 6
SEPARATOR_ROW = 6
MAX_ATTEMPTS = 400
PALETTE = [1, 2, 3, 4, 6, 7, 8]
SPRITE_MASKS = (
    np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=int),
    np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=int),
    np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=int),
)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _separator_row(grid: np.ndarray) -> int:
    rows = []
    for r in range(grid.shape[0]):
        row = grid[r]
        if np.all(row == row[0]) and int(row[0]) != BG:
            rows.append(r)
    assert len(rows) == 1, "expected exactly one solid non-background separator row"
    return int(rows[0])


def _legend_mapping(grid: np.ndarray, sep: int) -> dict[int, np.ndarray]:
    top = grid[:sep]
    comps = _components(top != BG)
    sprites: list[tuple[int, np.ndarray]] = []
    markers: list[tuple[int, int, int]] = []

    for coords in comps:
        area = int(coords.shape[0])
        r0, c0, r1, c1 = _bbox(coords)
        if area == 1:
            r, c = map(int, coords[0])
            markers.append((c, r, int(top[r, c])))
            continue
        assert (r1 - r0 + 1, c1 - c0 + 1) == (SPRITE_SIZE, SPRITE_SIZE), "legend sprite must be 3x3"
        assert area == SPRITE_SIZE * SPRITE_SIZE, "legend sprite must fill its 3x3 box"
        sprites.append((c0, top[r0 : r1 + 1, c0 : c1 + 1].copy()))

    assert 2 <= len(sprites) <= 3, "legend must contain two or three sprites"
    assert len(markers) == len(sprites), "legend sprites and legend markers must pair one-to-one"
    assert len({color for _c, _r, color in markers}) == len(markers), "legend marker colors must be unique"

    sprites.sort(key=lambda item: item[0])
    markers.sort(key=lambda item: item[0])

    mapping: dict[int, np.ndarray] = {}
    for (sprite_c0, sprite), (marker_c, marker_r, marker_color) in zip(sprites, markers):
        center_c = sprite_c0 + SPRITE_RADIUS
        assert marker_c == center_c, "legend marker must align with sprite center column"
        assert marker_r > SPRITE_RADIUS, "legend marker must sit below the sprite"
        mapping[marker_color] = sprite
    return mapping


def _stamp(out: np.ndarray, sprite: np.ndarray, center: tuple[int, int]) -> None:
    r, c = center
    r0 = r - SPRITE_RADIUS
    c0 = c - SPRITE_RADIUS
    r1 = r0 + SPRITE_SIZE
    c1 = c0 + SPRITE_SIZE
    assert 0 <= r0 and 0 <= c0 and r1 <= out.shape[0] and c1 <= out.shape[1], "stamp must fit"
    out[r0:r1, c0:c1] = sprite


def main(grid: np.ndarray) -> np.ndarray:
    """
    Map legend marker colors to 3x3 sprites, then stamp only the unblocked lower markers into the cropped work area.

    Input structure:
    - The grid uses black background, one solid separator row, a legend above that separator, and a work area below it.
    - Each legend entry is a 3x3 non-background sprite with one singleton marker pixel directly below its center column.
    - The work area contains singleton pixels in those legend-marker colors plus maroon obstacle pixels.

    Rule:
    - Identify the separator row and pair legend sprites with legend markers from left to right to build a marker-color -> sprite mapping.
    - Crop away the separator and legend, and erase every obstacle pixel from the work area.
    - For each work-area marker, scan upward in its column toward the separator; if any non-black cell lies in that vertical path, leave the marker as a singleton, otherwise stamp its mapped 3x3 sprite centered on that marker.
    - Apply stamps in work-area row-major order, so later stamps overwrite earlier ones where they overlap.

    Key insight:
    - The legend decides what to stamp, but a separate column-visibility test decides where stamping is allowed, so identical marker colors can behave differently at different positions.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    sep = _separator_row(g)
    mapping = _legend_mapping(g, sep)
    lower = g[sep + 1 :].copy()
    out = lower.copy()
    out[out == OBSTACLE_COLOR] = BG

    marker_colors = np.array(sorted(mapping), dtype=int)
    marker_mask = np.isin(lower, marker_colors)
    clear_markers: list[tuple[int, int, int]] = []
    for r, c in np.argwhere(marker_mask):
        color = int(lower[r, c])
        if np.any(lower[:r, c] != BG):
            continue
        clear_markers.append((int(r), int(c), color))

    clear_markers.sort()
    for r, c, color in clear_markers:
        _stamp(out, mapping[color], (r, c))
    return out


def _sprite_from_mask(base: int, accent: int, mask: np.ndarray) -> np.ndarray:
    sprite = np.full((SPRITE_SIZE, SPRITE_SIZE), base, dtype=int)
    sprite[mask == 1] = accent
    return sprite


def _legend_centers(width: int, count: int) -> tuple[int, ...] | None:
    options = [
        centers
        for centers in itertools.combinations(range(1, width - 1), count)
        if all(b - a >= 4 for a, b in zip(centers, centers[1:]))
    ]
    if not options:
        return None
    return random.choice(options)


def _box(center: tuple[int, int]) -> tuple[int, int, int, int]:
    r, c = center
    return r - 1, c - 1, r + 1, c + 1


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _inside_box(cell: tuple[int, int], box: tuple[int, int, int, int]) -> bool:
    r, c = cell
    return box[0] <= r <= box[2] and box[1] <= c <= box[3]


def _choose_clear_and_blocked(work_h: int, width: int, count: int) -> tuple[dict[int, tuple[int, int]], dict[int, tuple[int, int]], dict[int, tuple[int, int]]] | None:
    for _ in range(200):
        overlap_colors = random.sample(range(count), 2)
        base_r = random.randint(1, work_h - 3)
        base_c = random.randint(1, width - 4)
        clear = {
            overlap_colors[0]: (base_r, base_c),
            overlap_colors[1]: (base_r + 1, base_c + 2),
        }
        clear_boxes = [_box(pos) for pos in clear.values()]
        used_cols = {pos[1] for pos in clear.values()}

        ok = True
        for color_idx in range(count):
            if color_idx in clear:
                continue
            candidates = [
                (r, c)
                for r in range(1, work_h - 1)
                for c in range(1, width - 1)
                if c not in used_cols and all(not _boxes_overlap(_box((r, c)), box) for box in clear_boxes)
            ]
            if not candidates:
                ok = False
                break
            pos = random.choice(candidates)
            clear[color_idx] = pos
            clear_boxes.append(_box(pos))
            used_cols.add(pos[1])
        if not ok:
            continue

        blocked: dict[int, tuple[int, int]] = {}
        blockers: dict[int, tuple[int, int]] = {}
        occupied = set(clear.values())
        for color_idx in range(count):
            candidates = []
            for r in range(1, work_h - 1):
                for c in range(1, width - 1):
                    if c in used_cols or (r, c) in occupied:
                        continue
                    if any(_inside_box((r, c), box) for box in clear_boxes):
                        continue
                    blocker_rows = [
                        br
                        for br in range(r)
                        if (br, c) not in occupied
                        and all(not _inside_box((br, c), box) for box in clear_boxes)
                    ]
                    if blocker_rows:
                        candidates.append((r, c, blocker_rows))
            if not candidates:
                ok = False
                break
            r, c, blocker_rows = random.choice(candidates)
            blocked[color_idx] = (r, c)
            blockers[color_idx] = (random.choice(blocker_rows), c)
            occupied.add((r, c))
            occupied.add(blockers[color_idx])
            used_cols.add(c)
        if ok:
            return clear, blocked, blockers
    return None


def _render_expected(lower: np.ndarray, mapping: dict[int, np.ndarray]) -> tuple[np.ndarray, list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    out = lower.copy()
    out[out == OBSTACLE_COLOR] = BG
    marker_colors = np.array(sorted(mapping), dtype=int)
    markers = [(int(r), int(c), int(lower[r, c])) for r, c in np.argwhere(np.isin(lower, marker_colors))]
    clear_markers = [(r, c, color) for r, c, color in markers if not np.any(lower[:r, c] != BG)]
    clear_markers.sort()
    for r, c, color in clear_markers:
        _stamp(out, mapping[color], (r, c))
    return out, clear_markers, markers


def _render_with_order(lower: np.ndarray, mapping: dict[int, np.ndarray], markers: list[tuple[int, int, int]]) -> np.ndarray:
    out = lower.copy()
    out[out == OBSTACLE_COLOR] = BG
    for r, c, color in markers:
        _stamp(out, mapping[color], (r, c))
    return out


def _validate_case(
    grid: np.ndarray,
    expected: np.ndarray,
    mapping: dict[int, np.ndarray],
    clear: dict[int, tuple[int, int]],
    blocked: dict[int, tuple[int, int]],
    blockers: dict[int, tuple[int, int]],
) -> bool:
    if not np.array_equal(main(grid), expected):
        return False

    lower = grid[SEPARATOR_ROW + 1 :]
    actual_mapping = _legend_mapping(grid, _separator_row(grid))
    if sorted(actual_mapping) != sorted(mapping):
        return False
    for color in mapping:
        if not np.array_equal(actual_mapping[color], mapping[color]):
            return False

    if len({sprite.tobytes() for sprite in mapping.values()}) != len(mapping):
        return False

    expected2, clear_markers, all_markers = _render_expected(lower, mapping)
    if not np.array_equal(expected2, expected):
        return False

    per_color_clear = {color: 0 for color in mapping}
    per_color_blocked = {color: 0 for color in mapping}
    for r, c, color in all_markers:
        if np.any(lower[:r, c] != BG):
            per_color_blocked[color] += 1
        else:
            per_color_clear[color] += 1
    if any(v < 1 for v in per_color_clear.values()) or any(v < 1 for v in per_color_blocked.values()):
        return False

    for color, pos in clear.items():
        if np.any(lower[: pos[0], pos[1]] != BG):
            return False
    for color, pos in blocked.items():
        if not np.any(lower[: pos[0], pos[1]] != BG):
            return False
        if int(expected[pos]) != color:
            return False
        if blockers[color][1] != pos[1] or blockers[color][0] >= pos[0]:
            return False

    erase_only = lower.copy()
    erase_only[erase_only == OBSTACLE_COLOR] = BG
    if np.array_equal(erase_only, expected):
        return False

    stamp_all = _render_with_order(lower, mapping, sorted(all_markers))
    if np.array_equal(stamp_all, expected):
        return False

    reverse_clear = _render_with_order(lower, mapping, list(reversed(clear_markers)))
    if np.array_equal(reverse_clear, expected):
        return False

    return True


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    entry_count = random.choice((2, 3))
    width = random.randint(max(11, 2 * entry_count + 5), 15)
    work_h = random.randint(8, 13)
    legend_centers = _legend_centers(width, entry_count)
    if legend_centers is None:
        return None

    colors = random.sample(PALETTE, 2 * entry_count + 1)
    base_colors = colors[:entry_count]
    marker_colors = colors[entry_count : 2 * entry_count]
    accent = colors[-1]
    masks = random.sample(list(SPRITE_MASKS), entry_count)
    sprites = [_sprite_from_mask(base, accent, mask) for base, mask in zip(base_colors, masks)]
    mapping = {marker_color: sprite for marker_color, sprite in zip(marker_colors, sprites)}

    placement = _choose_clear_and_blocked(work_h, width, entry_count)
    if placement is None:
        return None
    clear_idx, blocked_idx, blockers_idx = placement
    clear = {marker_colors[idx]: pos for idx, pos in clear_idx.items()}
    blocked = {marker_colors[idx]: pos for idx, pos in blocked_idx.items()}
    blockers = {marker_colors[idx]: pos for idx, pos in blockers_idx.items()}

    lower = np.zeros((work_h, width), dtype=int)
    for marker_color in marker_colors:
        lower[clear[marker_color]] = marker_color
        lower[blocked[marker_color]] = marker_color
        lower[blockers[marker_color]] = OBSTACLE_COLOR

    extra_obstacle_count = random.randint(0, 2)
    clear_boxes = [_box(pos) for pos in clear.values()]
    occupied = set(clear.values()) | set(blocked.values()) | set(blockers.values())
    extra_choices = [
        (r, c)
        for r in range(work_h)
        for c in range(width)
        if (r, c) not in occupied and all(not _inside_box((r, c), box) for box in clear_boxes)
        and all(c != clear_c or r >= clear_r for clear_r, clear_c in clear.values())
    ]
    random.shuffle(extra_choices)
    for cell in extra_choices[:extra_obstacle_count]:
        lower[cell] = OBSTACLE_COLOR
        occupied.add(cell)

    legend = np.zeros((LEGEND_HEIGHT, width), dtype=int)
    for center, sprite, marker_color in zip(legend_centers, sprites, marker_colors):
        c0 = center - SPRITE_RADIUS
        legend[0:SPRITE_SIZE, c0 : c0 + SPRITE_SIZE] = sprite
        legend[LEGEND_MARKER_ROW, center] = marker_color

    separator = np.full((1, width), SEPARATOR_COLOR, dtype=int)
    grid = np.vstack([legend, separator, lower])
    expected, clear_markers, _all_markers = _render_expected(lower, mapping)

    overlap_pair = False
    for i, first in enumerate(clear_markers):
        for second in clear_markers[i + 1 :]:
            if _boxes_overlap(_box((first[0], first[1])), _box((second[0], second[1]))):
                overlap_pair = True
                break
        if overlap_pair:
            break
    if not overlap_pair:
        return None

    if not _validate_case(grid, expected, mapping, clear, blocked, blockers):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        pair = _gen_attempt()
        if pair is not None:
            return pair[0]
    raise RuntimeError("failed to generate a valid input")
