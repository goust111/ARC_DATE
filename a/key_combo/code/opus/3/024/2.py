from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

OCCLUDER_SIZE = 4
PATTERN_COLORS = [RED, GREEN, YELLOW, GREY, ORANGE, TEAL]
GEN_ATTEMPTS = 500


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _box(r0: int, c0: int, h: int, w: int, pad: int = 0) -> tuple[int, int, int, int]:
    return r0 - pad, c0 - pad, r0 + h + pad, c0 + w + pad


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar0 < br1 and br0 < ar1 and ac0 < bc1 and bc0 < ac1


def _rot_top_left(shape: tuple[int, int], r0: int, c0: int, h: int, w: int) -> tuple[int, int]:
    rows, cols = shape
    return rows - r0 - h, cols - c0 - w


def _ring_cells(r0: int, c0: int, size: int = OCCLUDER_SIZE):
    for rr in range(r0 - 1, r0 + size + 1):
        for cc in range(c0 - 1, c0 + size + 1):
            if r0 <= rr < r0 + size and c0 <= cc < c0 + size:
                continue
            yield rr, cc


def _earliest_scan_index(grid: np.ndarray, color: int) -> int:
    matches = np.flatnonzero(grid.reshape(-1) == color)
    assert matches.size > 0, f"expected visible color {color}"
    return int(matches[0])


def _earliest_cell(grid: np.ndarray, color: int) -> tuple[int, int]:
    rows, cols = np.argwhere(grid == color)[0]
    return int(rows), int(cols)


def _marker_region_color(grid: np.ndarray, r: int, c: int) -> int:
    counts: Counter[int] = Counter()
    for rr, cc in _neighbors4(r, c):
        if not _in_bounds(grid.shape, rr, cc):
            continue
        color = int(grid[rr, cc])
        if color in {BG, MAROON, BLUE, PINK}:
            continue
        counts[color] += 1
    assert counts, "each marker must touch at least one visible pattern color"
    ranked = counts.most_common()
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "marker neighborhood vote must be unique"
    return int(ranked[0][0])


def _extract_occluders(grid: np.ndarray) -> list[dict]:
    occluders: list[dict] = []
    for color in (BLUE, PINK):
        coords = np.argwhere(grid == color)
        assert coords.shape[0] == OCCLUDER_SIZE * OCCLUDER_SIZE, f"expected one {OCCLUDER_SIZE}x{OCCLUDER_SIZE} occluder of color {color}"
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        assert int(r1 - r0 + 1) == OCCLUDER_SIZE and int(c1 - c0 + 1) == OCCLUDER_SIZE
        assert np.all(grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] == color)
        occluders.append({"color": color, "top_left": (int(r0), int(c0))})
    return sorted(occluders, key=lambda spec: (spec["top_left"][0], spec["top_left"][1], spec["color"]))


def _ring_count(grid: np.ndarray, r0: int, c0: int, color: int) -> int:
    return sum(int(grid[rr, cc] == color) for rr, cc in _ring_cells(r0, c0))


def _select_occluder_index(occluders: list[dict], grid: np.ndarray, color: int) -> int:
    counts = [_ring_count(grid, spec["top_left"][0], spec["top_left"][1], color) for spec in occluders]
    if counts[0] != counts[1]:
        return 0 if counts[0] > counts[1] else 1
    return 0


def _analyze(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] == g.shape[1] and g.shape[0] % 2 == 1

    markers = [(int(r), int(c)) for r, c in np.argwhere(g == MAROON)]
    assert 3 <= len(markers) <= 6, "expected 3-6 marker pixels"

    votes: Counter[int] = Counter()
    marker_regions: list[tuple[int, int, int]] = []
    for r, c in markers:
        region_color = _marker_region_color(g, r, c)
        marker_regions.append((r, c, region_color))
        votes[region_color] += 1

    top_vote = max(votes.values())
    tied_colors = [color for color, count in votes.items() if count == top_vote]
    winner_color = min(tied_colors, key=lambda color: _earliest_scan_index(g, color))

    occluders = _extract_occluders(g)
    boundary_counts = [
        _ring_count(g, spec["top_left"][0], spec["top_left"][1], winner_color)
        for spec in occluders
    ]
    target_idx = _select_occluder_index(occluders, g, winner_color)

    return {
        "grid": g,
        "markers": markers,
        "marker_regions": marker_regions,
        "votes": dict(votes),
        "winner_color": int(winner_color),
        "occluders": occluders,
        "boundary_counts": boundary_counts,
        "target_idx": int(target_idx),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use marker voting to choose one occluder, then restore that 4x4 block by 180-degree rotation.

    Input structure:
    - The grid is an odd square on black background with one large multi-color pattern that is 180-degree rotationally symmetric.
    - Exactly two solid 4x4 occluders are painted over the pattern: one blue block and one magenta block.
    - Three to six maroon singleton markers lie on visible pattern cells, and each marker has at least one 4-neighbor in a non-black pattern color.

    Rule:
    - For each maroon marker, inspect its 4-neighbors and keep only visible pattern colors, ignoring black, maroon, blue, and magenta; the unique majority color is that marker's region color.
    - Count marker votes by region color; the color with the most votes wins, breaking ties by the earliest appearance of the tied colors in top-to-bottom, left-to-right scan order.
    - Compare the two occluders using the one-cell-thick outer ring around each 4x4 box, and choose the occluder whose ring contains more pixels of the winning color; if tied, choose the upper occluder.
    - Rotate the entire input by 180 degrees, crop the 4x4 patch now sitting at the chosen occluder's original location, and paste that crop back into the chosen occluder position.

    Key insight:
    - The markers do not reveal the answer block directly; they only elect a pattern color, and that elected color is then used to decide which occluder must be repaired by rotational correspondence.
    """
    analysis = _analyze(np.asarray(grid, dtype=int))
    g = analysis["grid"]
    target = analysis["occluders"][analysis["target_idx"]]["top_left"]
    r0, c0 = target

    rotated = np.rot90(g, 2)
    out = g.copy()
    out[r0 : r0 + OCCLUDER_SIZE, c0 : c0 + OCCLUDER_SIZE] = rotated[
        r0 : r0 + OCCLUDER_SIZE, c0 : c0 + OCCLUDER_SIZE
    ]
    return out


def _pattern_mask(size: int) -> np.ndarray:
    center = size // 2
    mask = np.zeros((size, size), dtype=bool)

    rect_h = random.randint(size - 10, size - 6)
    rect_w = random.randint(size - 10, size - 6)
    r0 = center - rect_h // 2
    c0 = center - rect_w // 2
    mask[r0 : r0 + rect_h, c0 : c0 + rect_w] = True

    if random.random() < 0.75:
        bar_h = random.randint(rect_h, min(size - 2, rect_h + 4))
        bar_w = random.randint(1, 3)
        br0 = center - bar_h // 2
        bc0 = center - bar_w // 2
        mask[br0 : br0 + bar_h, bc0 : bc0 + bar_w] = True

    if random.random() < 0.75:
        bar_h = random.randint(1, 3)
        bar_w = random.randint(rect_w, min(size - 2, rect_w + 4))
        br0 = center - bar_h // 2
        bc0 = center - bar_w // 2
        mask[br0 : br0 + bar_h, bc0 : bc0 + bar_w] = True

    return mask


def _all_patch_positions(mask: np.ndarray, h: int, w: int) -> list[tuple[int, int]]:
    rows, cols = mask.shape
    positions: list[tuple[int, int]] = []
    for r in range(rows - h + 1):
        for c in range(cols - w + 1):
            if np.all(mask[r : r + h, c : c + w]):
                positions.append((r, c))
    return positions


def _paint_symmetric_patch(pattern: np.ndarray, mask: np.ndarray, color: int, *, force_center: bool = False) -> bool:
    rows, cols = pattern.shape
    if force_center:
        h = random.choice([3, 5])
        w = random.choice([3, 5])
        r0 = rows // 2 - h // 2
        c0 = cols // 2 - w // 2
        if r0 < 0 or c0 < 0 or r0 + h > rows or c0 + w > cols:
            return False
        if not np.all(mask[r0 : r0 + h, c0 : c0 + w]):
            return False
    else:
        h = random.randint(2, 5)
        w = random.randint(2, 5)
        positions = _all_patch_positions(mask, h, w)
        if not positions:
            return False
        r0, c0 = random.choice(positions)
    mr0, mc0 = _rot_top_left(pattern.shape, r0, c0, h, w)
    pattern[r0 : r0 + h, c0 : c0 + w] = color
    pattern[mr0 : mr0 + h, mc0 : mc0 + w] = color
    return True


def _build_pattern(size: int) -> tuple[np.ndarray, np.ndarray] | None:
    mask = _pattern_mask(size)
    palette = random.sample(PATTERN_COLORS, random.randint(4, 5))
    pattern = np.zeros((size, size), dtype=int)
    pattern[mask] = palette[0]

    for color in palette[1:]:
        if not _paint_symmetric_patch(pattern, mask, color):
            return None

    extra_patches = random.randint(6, 10)
    for _ in range(extra_patches):
        color = random.choice(palette)
        if not _paint_symmetric_patch(pattern, mask, color):
            return None

    if random.random() < 0.6:
        _paint_symmetric_patch(pattern, mask, random.choice(palette), force_center=True)

    if not np.array_equal(pattern, np.rot90(pattern, 2)):
        return None
    if len({int(color) for color in np.unique(pattern) if color != BG}) < 4:
        return None
    return pattern, mask


def _patch_is_complex(patch: np.ndarray) -> bool:
    colors = {int(color) for color in np.unique(patch)}
    if len(colors) < 3:
        return False
    changes = int(np.sum(patch[:, 1:] != patch[:, :-1]) + np.sum(patch[1:, :] != patch[:-1, :]))
    return changes >= 8


def _candidate_occluders(pattern: np.ndarray, mask: np.ndarray) -> list[tuple[int, int]]:
    rows, cols = pattern.shape
    candidates: list[tuple[int, int]] = []
    for r0 in range(1, rows - OCCLUDER_SIZE):
        for c0 in range(1, cols - OCCLUDER_SIZE):
            if not np.all(mask[r0 - 1 : r0 + OCCLUDER_SIZE + 1, c0 - 1 : c0 + OCCLUDER_SIZE + 1]):
                continue
            patch = pattern[r0 : r0 + OCCLUDER_SIZE, c0 : c0 + OCCLUDER_SIZE]
            if _patch_is_complex(patch):
                candidates.append((r0, c0))
    return candidates


def _choose_occluder_pair(pattern: np.ndarray, mask: np.ndarray) -> list[tuple[int, int]] | None:
    candidates = _candidate_occluders(pattern, mask)
    if len(candidates) < 2:
        return None

    valid_pairs: list[list[tuple[int, int]]] = []
    for i, a in enumerate(candidates):
        ar, ac = a
        a_rot = _rot_top_left(pattern.shape, ar, ac, OCCLUDER_SIZE, OCCLUDER_SIZE)
        for b in candidates[i + 1 :]:
            br, bc = b
            b_rot = _rot_top_left(pattern.shape, br, bc, OCCLUDER_SIZE, OCCLUDER_SIZE)
            if b == a_rot or a == b_rot:
                continue
            if _boxes_overlap(_box(ar, ac, OCCLUDER_SIZE, OCCLUDER_SIZE, pad=1), _box(br, bc, OCCLUDER_SIZE, OCCLUDER_SIZE, pad=1)):
                continue
            if _boxes_overlap(_box(ar, ac, OCCLUDER_SIZE, OCCLUDER_SIZE), _box(b_rot[0], b_rot[1], OCCLUDER_SIZE, OCCLUDER_SIZE)):
                continue
            if _boxes_overlap(_box(br, bc, OCCLUDER_SIZE, OCCLUDER_SIZE), _box(a_rot[0], a_rot[1], OCCLUDER_SIZE, OCCLUDER_SIZE)):
                continue
            if ar == br:
                continue
            valid_pairs.append(sorted([a, b]))

    if not valid_pairs:
        return None
    return random.choice(valid_pairs)


def _paint_occluders(grid: np.ndarray, occluders: list[tuple[int, int]], colors: list[int]) -> np.ndarray:
    out = grid.copy()
    for (r0, c0), color in zip(occluders, colors):
        out[r0 : r0 + OCCLUDER_SIZE, c0 : c0 + OCCLUDER_SIZE] = color
    return out


def _region_color_if_marked(pattern: np.ndarray, r: int, c: int) -> int | None:
    counts: Counter[int] = Counter()
    for rr, cc in _neighbors4(r, c):
        if not _in_bounds(pattern.shape, rr, cc):
            continue
        color = int(pattern[rr, cc])
        if color == BG:
            continue
        counts[color] += 1
    if not counts:
        return None
    ranked = counts.most_common()
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return None
    return int(ranked[0][0])


def _candidate_marker_sites(pattern: np.ndarray, occluders: list[tuple[int, int]]) -> dict[int, list[tuple[int, int]]]:
    blocked: set[tuple[int, int]] = set()
    for r0, c0 in occluders:
        for rr in range(r0 - 1, r0 + OCCLUDER_SIZE + 1):
            for cc in range(c0 - 1, c0 + OCCLUDER_SIZE + 1):
                blocked.add((rr, cc))
        mr0, mc0 = _rot_top_left(pattern.shape, r0, c0, OCCLUDER_SIZE, OCCLUDER_SIZE)
        for rr in range(mr0, mr0 + OCCLUDER_SIZE):
            for cc in range(mc0, mc0 + OCCLUDER_SIZE):
                blocked.add((rr, cc))

    groups: dict[int, list[tuple[int, int]]] = defaultdict(list)
    rows, cols = pattern.shape
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            if int(pattern[r, c]) == BG or (r, c) in blocked:
                continue
            region_color = _region_color_if_marked(pattern, r, c)
            if region_color is None:
                continue
            groups[region_color].append((r, c))

    for sites in groups.values():
        random.shuffle(sites)
    return dict(groups)


def _spaced(site: tuple[int, int], chosen: list[tuple[int, int]]) -> bool:
    return all(abs(site[0] - rr) + abs(site[1] - cc) > 1 for rr, cc in chosen)


def _choose_markers_for_counts(
    groups: dict[int, list[tuple[int, int]]], counts: dict[int, int]
) -> list[tuple[int, int, int]] | None:
    items = sorted(counts.items(), key=lambda item: (len(groups.get(item[0], [])), -item[1], item[0]))
    chosen: list[tuple[int, int, int]] = []

    def rec(index: int) -> bool:
        if index == len(items):
            return True

        color, need = items[index]
        sites = groups.get(color, [])
        if len(sites) < need:
            return False

        picked: list[tuple[int, int]] = []

        def pick_from(start: int) -> bool:
            if len(picked) == need:
                for r, c in picked:
                    chosen.append((r, c, color))
                if rec(index + 1):
                    return True
                del chosen[-need:]
                return False

            remaining = need - len(picked)
            for idx in range(start, len(sites)):
                if len(sites) - idx < remaining:
                    break
                site = sites[idx]
                if not _spaced(site, [(r, c) for r, c, _ in chosen]):
                    continue
                if not _spaced(site, picked):
                    continue
                picked.append(site)
                if pick_from(idx + 1):
                    return True
                picked.pop()
            return False

        return pick_from(0)

    return chosen if rec(0) else None


def _winner_marker_profiles(winner_color: int, other_colors: list[int]) -> list[dict[int, int]]:
    profiles: list[dict[int, int]] = []
    if len(other_colors) >= 1:
        profiles.append({winner_color: 2, other_colors[0]: 1})
        profiles.append({winner_color: 3, other_colors[0]: 1})
        profiles.append({winner_color: 3, other_colors[0]: 2})
    if len(other_colors) >= 2:
        profiles.append({winner_color: 2, other_colors[0]: 1, other_colors[1]: 1})
        profiles.append({winner_color: 3, other_colors[0]: 1, other_colors[1]: 1})
    return [profile for profile in profiles if 3 <= sum(profile.values()) <= 6]


def _tied_marker_profiles(winner_color: int, tie_color: int, extras: list[int]) -> list[dict[int, int]]:
    profiles: list[dict[int, int]] = [
        {winner_color: 2, tie_color: 2},
        {winner_color: 3, tie_color: 3},
    ]
    if extras:
        profiles.append({winner_color: 2, tie_color: 2, extras[0]: 1})
    return [profile for profile in profiles if 3 <= sum(profile.values()) <= 6]


def _expected_output(inp: np.ndarray, pattern: np.ndarray, target: tuple[int, int]) -> np.ndarray:
    out = inp.copy()
    r0, c0 = target
    out[r0 : r0 + OCCLUDER_SIZE, c0 : c0 + OCCLUDER_SIZE] = pattern[
        r0 : r0 + OCCLUDER_SIZE, c0 : c0 + OCCLUDER_SIZE
    ]
    return out


def _gen_attempt() -> np.ndarray | None:
    size = random.choice([17, 19, 21, 23, 25])
    built = _build_pattern(size)
    if built is None:
        return None
    pattern, mask = built

    occluders = _choose_occluder_pair(pattern, mask)
    if occluders is None:
        return None

    preview = _paint_occluders(pattern, occluders, [BLUE, PINK])
    preview_occluders = [{"top_left": top_left} for top_left in occluders]

    visible_colors = [color for color in PATTERN_COLORS if np.any(preview == color)]
    if len(visible_colors) < 3:
        return None

    select_map = {color: _select_occluder_index(preview_occluders, preview, color) for color in visible_colors}
    if len(set(select_map.values())) < 2:
        return None

    top_idx = 0
    tie_boundary_colors = [
        color
        for color in visible_colors
        if _ring_count(preview, occluders[0][0], occluders[0][1], color)
        == _ring_count(preview, occluders[1][0], occluders[1][1], color)
    ]

    if tie_boundary_colors and random.random() < 0.25:
        target_idx = top_idx
        winner_candidates = list(tie_boundary_colors)
    else:
        target_idx = random.choice([0, 1])
        winner_candidates = [
            color
            for color in visible_colors
            if select_map[color] == target_idx
            and any(select_map[other] != target_idx for other in visible_colors if other != color)
            and _ring_count(preview, occluders[0][0], occluders[0][1], color)
            != _ring_count(preview, occluders[1][0], occluders[1][1], color)
        ]
        if not winner_candidates:
            target_idx = 1 - target_idx
            winner_candidates = [
                color
                for color in visible_colors
                if select_map[color] == target_idx
                and any(select_map[other] != target_idx for other in visible_colors if other != color)
                and _ring_count(preview, occluders[0][0], occluders[0][1], color)
                != _ring_count(preview, occluders[1][0], occluders[1][1], color)
            ]
    if not winner_candidates:
        return None
    winner_color = random.choice(winner_candidates)

    marker_groups = _candidate_marker_sites(pattern, occluders)
    if winner_color not in marker_groups:
        return None

    earliest = {color: _earliest_scan_index(preview, color) for color in visible_colors}
    earliest_cells = {color: _earliest_cell(preview, color) for color in visible_colors}

    marker_plan: list[tuple[int, int, int]] | None = None
    vote_tie = random.random() < 0.35
    if vote_tie:
        tie_colors = [
            color
            for color in visible_colors
            if color != winner_color
            and color in marker_groups
            and earliest[winner_color] < earliest[color]
        ]
        tie_colors.sort(key=lambda color: (select_map[color] == target_idx, earliest[color]))
        for tie_color in tie_colors:
            filtered_groups = {color: list(sites) for color, sites in marker_groups.items()}
            filtered_groups[winner_color] = [
                site for site in filtered_groups[winner_color] if site != earliest_cells[winner_color]
            ]
            filtered_groups[tie_color] = [
                site for site in filtered_groups[tie_color] if site != earliest_cells[tie_color]
            ]
            extras = [color for color in visible_colors if color not in {winner_color, tie_color} and color in filtered_groups]
            profiles = _tied_marker_profiles(winner_color, tie_color, extras)
            random.shuffle(profiles)
            for profile in profiles:
                marker_plan = _choose_markers_for_counts(filtered_groups, profile)
                if marker_plan is not None:
                    break
            if marker_plan is not None:
                break

    if marker_plan is None:
        losers = [color for color in visible_colors if color != winner_color and color in marker_groups]
        losers.sort(key=lambda color: (select_map[color] == target_idx, earliest[color]))
        profiles = _winner_marker_profiles(winner_color, losers)
        random.shuffle(profiles)
        for profile in profiles:
            marker_plan = _choose_markers_for_counts(marker_groups, profile)
            if marker_plan is not None:
                break
    if marker_plan is None:
        return None

    inp = pattern.copy()
    for r, c, _color in marker_plan:
        inp[r, c] = MAROON

    occluder_colors = [BLUE, PINK]
    random.shuffle(occluder_colors)
    inp = _paint_occluders(inp, occluders, occluder_colors)

    analysis = _analyze(inp)
    if analysis["winner_color"] != winner_color:
        return None
    if analysis["target_idx"] != target_idx:
        return None

    target = occluders[target_idx]
    expected = _expected_output(inp, pattern, target)
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return None

    other_target = occluders[1 - target_idx]
    wrong = _expected_output(inp, pattern, other_target)
    if np.array_equal(wrong, expected):
        return None

    if np.array_equal(inp, expected):
        return None

    target_patch = pattern[target[0] : target[0] + OCCLUDER_SIZE, target[1] : target[1] + OCCLUDER_SIZE]
    if len({int(color) for color in np.unique(target_patch)}) < 3:
        return None

    if not any(
        _select_occluder_index(analysis["occluders"], inp, color) != analysis["target_idx"]
        for color in {int(color) for color in np.unique(inp)}
        if color not in {BG, BLUE, PINK, MAROON}
    ):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid marker-vote occluder sample")
