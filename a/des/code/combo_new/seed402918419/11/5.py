from __future__ import annotations

from collections import Counter
import numpy as np


BG = 0
SEPARATOR = 5
FRAGMENT = 9
FRAME_COLORS = (1, 2, 3, 4, 6, 7, 8)

_RNG = np.random.default_rng()
_GEN_RETRIES = 400
_CANDIDATE_CACHE: dict[tuple[int, int], list[dict]] = {}


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[counts.argmax()])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _coords_to_mask(coords: list[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask, (int(r0), int(c0))


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _rotation_key(mask: np.ndarray) -> tuple[int, int, bytes]:
    best: tuple[int, int, bytes] | None = None
    for k in range(4):
        rot = _crop_mask(np.rot90(mask, k))
        key = (int(rot.shape[0]), int(rot.shape[1]), rot.tobytes())
        if best is None or key < best:
            best = key
    assert best is not None
    return best


def _find_separator(grid: np.ndarray, bg: int) -> tuple[int, int]:
    candidates = []
    for r in range(grid.shape[0]):
        row = grid[r]
        if np.all(row == row[0]) and int(row[0]) != int(bg):
            candidates.append((int(r), int(row[0])))
    assert len(candidates) == 1, "expected exactly one full horizontal separator row"
    return candidates[0]


def _extract_frames(top: np.ndarray, bg: int, separator_color: int) -> list[dict]:
    mask = (top != int(bg)) & (top != int(separator_color))
    comps = _components(mask)
    assert comps, "expected at least one frame above the separator"
    frames: list[dict] = []
    for comp in comps:
        colors = {int(top[r, c]) for r, c in comp}
        assert len(colors) == 1, "each frame must be monochrome"
        color = int(next(iter(colors)))
        r0, c0, r1, c1 = _bbox(comp)
        local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in comp:
            local[r - r0, c - c0] = True

        border = np.zeros_like(local)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        assert np.all(local <= border), "frame pixels must lie on their bounding-box border"

        gap = border & (~local)
        gap_comps = _components(gap)
        assert len(gap_comps) == 1, "each frame must have exactly one connected gap"
        gap_mask, (gr0, gc0) = _coords_to_mask(gap_comps[0])
        gap_abs = [(r0 + gr0 + r, c0 + gc0 + c) for r, c in np.argwhere(gap_mask)]
        frames.append(
            {
                "color": color,
                "gap_mask": gap_mask,
                "gap_abs": gap_abs,
            }
        )
    return frames


def _extract_fragments(bottom: np.ndarray, bg: int) -> tuple[int, list[list[tuple[int, int]]]]:
    colors = [int(v) for v in np.unique(bottom) if int(v) != int(bg)]
    assert len(colors) == 1, "expected a single fragment color below the separator"
    fragment_color = colors[0]
    comps = _components(bottom == int(fragment_color))
    assert comps, "expected at least one fragment below the separator"
    return fragment_color, comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each lower fragment to refill the matching rectangular-frame gap and recolor it with the frame color.

    Input structure:
    - Background is black, and one full-width grey row splits the grid into upper and lower regions.
    - The upper region contains several single-color rectangular frames, each missing one connected border segment.
    - The lower region contains several maroon 4-connected fragments whose shapes match the frame gaps up to rotation.

    Rule:
    - Locate the separator row, then extract each upper frame as a border-only rectangle within its bounding box.
    - For every frame, compute the missing border-segment shape; for every lower fragment, compute its shape and compare up to the four 90-degree rotations.
    - Verify that the multiset of rotated fragment shapes matches the multiset of frame-gap shapes.
    - Fill every frame gap with the frame's own color, erase all lower fragments to background, and keep the separator row unchanged.

    Key insight:
    - The lower maroon pieces are not painted where they already are; they are rotation-matched to the upper border gaps, then recolored by the destination frame rather than by their original color.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    sep_row, separator_color = _find_separator(g, bg)

    top = g[:sep_row, :]
    bottom = g[sep_row + 1 :, :]
    frames = _extract_frames(top, bg, separator_color)
    fragment_color, fragments = _extract_fragments(bottom, bg)

    gap_shapes = Counter(_rotation_key(frame["gap_mask"]) for frame in frames)
    fragment_shapes = Counter(
        _rotation_key(_coords_to_mask(comp)[0]) for comp in fragments
    )
    assert gap_shapes == fragment_shapes, "fragment shapes must match frame gaps up to rotation"

    out = g.copy()
    for comp in fragments:
        for r, c in comp:
            out[sep_row + 1 + r, c] = int(bg)

    for frame in frames:
        for r, c in frame["gap_abs"]:
            out[r, c] = int(frame["color"])

    assert int(np.count_nonzero(out[sep_row + 1 :, :] == int(fragment_color))) == 0
    return out


def _border_cycle(h: int, w: int) -> list[tuple[int, int]]:
    coords = [(0, c) for c in range(w)]
    coords.extend((r, w - 1) for r in range(1, h - 1))
    coords.extend((h - 1, c) for c in range(w - 1, -1, -1))
    coords.extend((r, 0) for r in range(h - 2, 0, -1))
    return coords


def _turn_count(path: list[tuple[int, int]]) -> int:
    dirs = []
    for (r0, c0), (r1, c1) in zip(path, path[1:], strict=False):
        dirs.append((int(r1) - int(r0), int(c1) - int(c0)))
    return sum(a != b for a, b in zip(dirs, dirs[1:], strict=False))


def _gap_candidates(h: int, w: int) -> list[dict]:
    key = (int(h), int(w))
    if key in _CANDIDATE_CACHE:
        return _CANDIDATE_CACHE[key]

    cycle = _border_cycle(h, w)
    perim = len(cycle)
    max_len = min(7, perim - 5)
    out: list[dict] = []
    for start in range(perim):
        for length in range(2, max_len + 1):
            coords = [cycle[(start + i) % perim] for i in range(length)]
            coord_set = set(coords)
            if all((0, c) in coord_set for c in range(w)):
                continue
            if all((h - 1, c) in coord_set for c in range(w)):
                continue
            if all((r, 0) in coord_set for r in range(h)):
                continue
            if all((r, w - 1) in coord_set for r in range(h)):
                continue
            turns = _turn_count(coords)
            if turns > 2:
                continue
            mask, _ = _coords_to_mask(coords)
            out.append(
                {
                    "coords": tuple(coords),
                    "mask": mask,
                    "shape_key": _rotation_key(mask),
                    "turns": int(turns),
                }
            )
    _CANDIDATE_CACHE[key] = out
    return out


def _frame_mask(h: int, w: int, gap_coords: tuple[tuple[int, int], ...]) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    for r, c in gap_coords:
        mask[int(r), int(c)] = False
    return mask


def _pack_columns(widths: list[int], total_width: int, min_gap: int = 2) -> list[int] | None:
    need = sum(widths) + min_gap * (len(widths) - 1)
    if need > total_width:
        return None
    slack = total_width - need
    extras = _RNG.multinomial(slack, np.full(len(widths) + 1, 1.0 / (len(widths) + 1)))
    cols: list[int] = []
    c = int(extras[0])
    for i, width in enumerate(widths):
        cols.append(c)
        c += int(width)
        if i + 1 < len(widths):
            c += min_gap + int(extras[i + 1])
    return cols


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _place_fragments(masks: list[np.ndarray], width: int, max_height: int) -> tuple[int, list[tuple[int, int]]] | None:
    min_height = max(mask.shape[0] for mask in masks) + 2
    if min_height > max_height:
        return None

    candidate_heights = list(range(min_height, max_height + 1))
    _RNG.shuffle(candidate_heights)
    order = sorted(range(len(masks)), key=lambda i: (-int(masks[i].sum()), -max(masks[i].shape)))

    for height in candidate_heights:
        occupied = np.zeros((height, width), dtype=bool)
        blocked = np.zeros_like(occupied)
        placements: list[tuple[int, int] | None] = [None] * len(masks)
        ok = True
        for idx in order:
            mask = masks[idx]
            mh, mw = mask.shape
            if mh > height or mw > width:
                ok = False
                break

            row_choices = list(range(height - mh + 1))
            col_choices = list(range(width - mw + 1))
            _RNG.shuffle(row_choices)
            _RNG.shuffle(col_choices)

            placed = False
            for r0 in row_choices[: min(len(row_choices), 16)]:
                for c0 in col_choices[: min(len(col_choices), 24)]:
                    if np.any(blocked[r0 : r0 + mh, c0 : c0 + mw] & mask):
                        continue
                    occupied[r0 : r0 + mh, c0 : c0 + mw] |= mask
                    blocked = _dilate8(occupied)
                    placements[idx] = (int(r0), int(c0))
                    placed = True
                    break
                if placed:
                    break
            if not placed:
                ok = False
                break
        if ok and all(p is not None for p in placements):
            return int(height), [p for p in placements if p is not None]
    return None


def _sample_frame_specs() -> list[dict] | None:
    n_frames = int(_RNG.integers(2, 5))
    colors = [int(c) for c in _RNG.choice(FRAME_COLORS, size=n_frames, replace=False)]

    if n_frames == 2:
        desired_turns = [int(v) for v in _RNG.choice([0, 1, 2], size=2, replace=False)]
    elif n_frames == 3:
        desired_turns = [0, 1, 2]
    else:
        desired_turns = [0, 1, 2, int(_RNG.integers(0, 3))]
    _RNG.shuffle(desired_turns)

    specs: list[dict] = []
    used_shapes: set[tuple[int, int, bytes]] = set()
    for i in range(n_frames):
        found = None
        for _ in range(80):
            h = int(_RNG.integers(4, 9))
            w = int(_RNG.integers(5, 10))
            candidates = _gap_candidates(h, w)
            turn_order = [desired_turns[i]] + [k for k in range(3) if k != desired_turns[i]]
            for turns in turn_order:
                options = [
                    cand
                    for cand in candidates
                    if int(cand["turns"]) == int(turns) and cand["shape_key"] not in used_shapes
                ]
                if options:
                    found = dict(options[int(_RNG.integers(len(options)))])
                    found["h"] = h
                    found["w"] = w
                    found["color"] = colors[i]
                    break
            if found is not None:
                break
        if found is None:
            return None
        used_shapes.add(found["shape_key"])
        specs.append(found)

    min_width = sum(int(spec["w"]) for spec in specs) + 2 * (len(specs) - 1) + 2
    if min_width > 30:
        return None
    return specs


def _build_scene() -> tuple[np.ndarray, np.ndarray] | None:
    specs = _sample_frame_specs()
    if specs is None:
        return None

    widths = [int(spec["w"]) for spec in specs]
    max_h = max(int(spec["h"]) for spec in specs)
    min_width = sum(widths) + 2 * (len(widths) - 1) + 2
    width = int(_RNG.integers(min_width, min(31, min_width + 7)))

    packed = _pack_columns(widths, width - 2, min_gap=2)
    if packed is None:
        return None
    cols = [1 + int(c) for c in packed]

    sep_row = max_h + int(_RNG.integers(3, 6))
    fragment_masks = []
    for spec in specs:
        rot = int(_RNG.integers(0, 4))
        fragment_masks.append(_crop_mask(np.rot90(spec["mask"], rot)))

    remaining_height = 30 - (sep_row + 1)
    placed = _place_fragments(fragment_masks, width, remaining_height)
    if placed is None:
        return None
    bottom_height, fragment_positions = placed
    height = sep_row + 1 + bottom_height

    grid = np.full((height, width), BG, dtype=int)
    grid[sep_row, :] = SEPARATOR

    expected = grid.copy()
    for spec, c0 in zip(specs, cols, strict=False):
        h = int(spec["h"])
        w = int(spec["w"])
        r_max = sep_row - h - 1
        if r_max < 0:
            return None
        r0 = int(_RNG.integers(0, r_max + 1))
        frame = _frame_mask(h, w, spec["coords"])
        grid[r0 : r0 + h, c0 : c0 + w][frame] = int(spec["color"])
        expected[r0 : r0 + h, c0 : c0 + w][frame] = int(spec["color"])
        for gr, gc in spec["coords"]:
            expected[r0 + int(gr), c0 + int(gc)] = int(spec["color"])

    for mask, (r0, c0) in zip(fragment_masks, fragment_positions, strict=False):
        rr, cc = np.where(mask)
        grid[sep_row + 1 + r0 + rr, c0 + cc] = FRAGMENT

    assert height <= 30 and width <= 30
    assert len([r for r in range(height) if np.all(grid[r] == grid[r, 0]) and grid[r, 0] != BG]) == 1
    top = grid[:sep_row, :]
    top_colors = {int(v) for v in np.unique(top) if int(v) not in (BG, SEPARATOR)}
    assert len(top_colors) == len(specs)
    assert len(_components(top != BG)) == len(specs)
    assert len(_components(grid[sep_row + 1 :, :] == FRAGMENT)) == len(specs)
    assert Counter(_rotation_key(spec["mask"]) for spec in specs) == Counter(
        _rotation_key(mask) for mask in fragment_masks
    )
    assert np.array_equal(main(grid), expected)
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        built = _build_scene()
        if built is not None:
            return built[0]
    raise AssertionError("failed to generate a valid puzzle instance")
