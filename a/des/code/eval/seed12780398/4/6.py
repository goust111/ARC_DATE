import itertools
from collections import Counter

import numpy as np


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(int(v) for v in np.asarray(grid).flat).most_common(1)[0][0])


def paste(dst: np.ndarray, src: np.ndarray, r0: int, c0: int, transparent: int | None = None) -> None:
    h, w = src.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    if transparent is None:
        view[:] = src
        return
    mask = src != transparent
    view[mask] = src[mask]


def find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[dict] = []
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in neighbors:
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            out.append(
                {
                    "color": color,
                    "coords": tuple(coords),
                    "bbox": (r0, c0, r1, c1),
                    "area": len(coords),
                }
            )
    return out


def component_sprite(component: dict, bg: int) -> np.ndarray:
    r0, c0, r1, c1 = component["bbox"]
    sprite = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for r, c in component["coords"]:
        sprite[r - r0, c - c0] = int(component["color"])
    return sprite


def component_rel_coords(component: dict) -> tuple[tuple[int, int], ...]:
    r0, c0, _r1, _c1 = component["bbox"]
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in component["coords"]))


def _partition_is_vertical(labels: np.ndarray) -> bool:
    h, w = labels.shape
    if w < 2:
        return False
    for left_label, right_label in ((0, 1), (1, 0)):
        cuts = []
        ok = True
        for r in range(h):
            row = labels[r]
            cut = 0
            while cut < w and int(row[cut]) == left_label:
                cut += 1
            if cut == 0 or cut == w:
                ok = False
                break
            if not np.all(row[:cut] == left_label) or not np.all(row[cut:] == right_label):
                ok = False
                break
            cuts.append(cut)
        if ok and len(set(cuts)) > 1:
            return True
    return False


def _partition_is_horizontal(labels: np.ndarray) -> bool:
    h, w = labels.shape
    if h < 2:
        return False
    for top_label, bottom_label in ((0, 1), (1, 0)):
        cuts = []
        ok = True
        for c in range(w):
            col = labels[:, c]
            cut = 0
            while cut < h and int(col[cut]) == top_label:
                cut += 1
            if cut == 0 or cut == h:
                ok = False
                break
            if not np.all(col[:cut] == top_label) or not np.all(col[cut:] == bottom_label):
                ok = False
                break
            cuts.append(cut)
        if ok and len(set(cuts)) > 1:
            return True
    return False


def _is_valid_jagged_partition(labels: np.ndarray) -> bool:
    return _partition_is_vertical(labels) or _partition_is_horizontal(labels)


def candidate_pair_patterns(component_a: dict, component_b: dict, bg: int) -> list[np.ndarray]:
    a_rel = component_rel_coords(component_a)
    b_rel = component_rel_coords(component_b)
    a_h = component_a["bbox"][2] - component_a["bbox"][0] + 1
    a_w = component_a["bbox"][3] - component_a["bbox"][1] + 1
    b_h = component_b["bbox"][2] - component_b["bbox"][0] + 1
    b_w = component_b["bbox"][3] - component_b["bbox"][1] + 1

    a_set = set(a_rel)
    out: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], tuple[int, ...]]] = set()

    for dr in range(-b_h, a_h + 1):
        for dc in range(-b_w, a_w + 1):
            b_shifted = {(r + dr, c + dc) for r, c in b_rel}
            if a_set & b_shifted:
                continue

            union = a_set | b_shifted
            rows = [r for r, _ in union]
            cols = [c for _, c in union]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if h * w != len(union):
                continue

            labels = np.full((h, w), -1, dtype=int)
            for r, c in a_set:
                labels[r - r0, c - c0] = 0
            for r, c in b_shifted:
                labels[r - r0, c - c0] = 1
            if np.any(labels < 0) or not _is_valid_jagged_partition(labels):
                continue

            pattern = np.full((h, w), bg, dtype=int)
            pattern[labels == 0] = int(component_a["color"])
            pattern[labels == 1] = int(component_b["color"])
            key = (pattern.shape, tuple(int(v) for v in pattern.flat))
            if key in seen:
                continue
            seen.add(key)
            out.append(pattern)
    return out


def solve_unique_pairing(components: list[dict], bg: int) -> list[np.ndarray]:
    count = len(components)
    assert count > 0 and count % 2 == 0, "expected an even number of fragments"

    candidate_map: dict[tuple[int, int], list[np.ndarray]] = {}
    for i in range(count):
        for j in range(i + 1, count):
            patterns = candidate_pair_patterns(components[i], components[j], bg)
            if patterns:
                candidate_map[(i, j)] = patterns

    solutions: list[list[np.ndarray]] = []

    def backtrack(remaining: tuple[int, ...], current: list[np.ndarray]) -> None:
        if len(solutions) > 1:
            return
        if not remaining:
            solutions.append([pat.copy() for pat in current])
            return
        i = remaining[0]
        rest = remaining[1:]
        for pos, j in enumerate(rest):
            patterns = candidate_map.get((i, j), [])
            if not patterns:
                continue
            next_remaining = rest[:pos] + rest[pos + 1 :]
            for pattern in patterns:
                current.append(pattern)
                backtrack(next_remaining, current)
                current.pop()
                if len(solutions) > 1:
                    return

    backtrack(tuple(range(count)), [])
    assert len(solutions) == 1, "fragment pairing must be unique"
    return solutions[0]


def stack_patterns(patterns: list[np.ndarray], bg: int) -> np.ndarray:
    assert patterns
    ordered = sorted(patterns, key=lambda arr: (-arr.size, arr.shape[0], arr.shape[1]))
    out_h = sum(arr.shape[0] for arr in ordered) + len(ordered) - 1
    out_w = max(arr.shape[1] for arr in ordered)
    out = np.full((out_h, out_w), bg, dtype=int)
    r = 0
    for arr in ordered:
        paste(out, arr, r, 0, transparent=bg)
        r += arr.shape[0] + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair complementary jagged fragments into rectangles, then stack the rectangles
    from largest area to smallest with one background row between them.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    components = find_components(g, bg)
    patterns = solve_unique_pairing(components, bg)
    return stack_patterns(patterns, bg)


_RNG = np.random.default_rng()
_PROFILE_CACHE: dict[tuple[int, int, str], list[tuple[int, ...]]] = {}
_RECT_OPTIONS = [(2, 4), (2, 5), (3, 3), (3, 4), (3, 5), (4, 4)]
_GRID_MIN = 16
_GRID_MAX = 30
_GEN_RETRIES = 400


def _profiles_for_spec(h: int, w: int, orientation: str) -> list[tuple[int, ...]]:
    key = (h, w, orientation)
    if key in _PROFILE_CACHE:
        return _PROFILE_CACHE[key]

    profiles: list[tuple[int, ...]] = []
    if orientation == "vertical":
        for cuts in itertools.product(range(1, w), repeat=h):
            area_a = sum(cuts)
            area_b = h * w - area_a
            if 4 <= area_a <= 8 and 4 <= area_b <= 8 and len(set(cuts)) > 1:
                profiles.append(tuple(int(v) for v in cuts))
    else:
        for cuts in itertools.product(range(1, h), repeat=w):
            area_a = sum(cuts)
            area_b = h * w - area_a
            if 4 <= area_a <= 8 and 4 <= area_b <= 8 and len(set(cuts)) > 1:
                profiles.append(tuple(int(v) for v in cuts))

    _PROFILE_CACHE[key] = profiles
    return profiles


def _valid_specs_by_area() -> dict[int, list[tuple[int, int, str, list[tuple[int, ...]]]]]:
    out: dict[int, list[tuple[int, int, str, list[tuple[int, ...]]]]] = {}
    for h, w in _RECT_OPTIONS:
        area = h * w
        for orientation in ("vertical", "horizontal"):
            profiles = _profiles_for_spec(h, w, orientation)
            if profiles:
                out.setdefault(area, []).append((h, w, orientation, profiles))
    return out


_SPECS_BY_AREA = _valid_specs_by_area()
_AREA_OPTIONS = tuple(sorted(_SPECS_BY_AREA))


def _mask_to_sprite(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    rows, cols = np.where(mask)
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())
    sprite = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    cropped = mask[r0 : r1 + 1, c0 : c1 + 1]
    sprite[cropped] = color
    return sprite


def _build_pair(h: int, w: int, orientation: str, profile: tuple[int, ...], color_a: int, color_b: int, bg: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask_a = np.zeros((h, w), dtype=bool)
    if orientation == "vertical":
        for r, cut in enumerate(profile):
            mask_a[r, :cut] = True
    else:
        for c, cut in enumerate(profile):
            mask_a[:cut, c] = True
    mask_b = ~mask_a

    if _RNG.random() < 0.5:
        mask_a, mask_b = mask_b, mask_a
        color_a, color_b = color_b, color_a

    composite = np.where(mask_a, color_a, color_b).astype(int)
    sprite_a = _mask_to_sprite(mask_a, color_a, bg)
    sprite_b = _mask_to_sprite(mask_b, color_b, bg)
    return sprite_a, sprite_b, composite


def _sprite_coords(sprite: np.ndarray, bg: int) -> list[tuple[int, int]]:
    return [(int(r), int(c)) for r, c in np.argwhere(sprite != bg)]


def _place_sprites(sprites: list[np.ndarray], bg: int, h: int, w: int) -> np.ndarray | None:
    grid = np.full((h, w), bg, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)
    order = list(range(len(sprites)))
    order.sort(key=lambda idx: (-len(_sprite_coords(sprites[idx], bg)), -sprites[idx].shape[0], -sprites[idx].shape[1]))

    for idx in order:
        sprite = sprites[idx]
        coords = _sprite_coords(sprite, bg)
        sh, sw = sprite.shape
        choices = []
        for r0 in range(h - sh + 1):
            for c0 in range(w - sw + 1):
                ok = True
                for rr, cc in coords:
                    if blocked[r0 + rr, c0 + cc]:
                        ok = False
                        break
                if ok:
                    choices.append((r0, c0))
        if not choices:
            return None
        r0, c0 = choices[int(_RNG.integers(len(choices)))]
        for rr, cc in coords:
            r = r0 + rr
            c = c0 + cc
            grid[r, c] = int(sprite[rr, cc])
            r_lo = max(0, r - 1)
            r_hi = min(h, r + 2)
            c_lo = max(0, c - 1)
            c_hi = min(w, c + 2)
            blocked[r_lo:r_hi, c_lo:c_hi] = True
    return grid


def _expected_output(patterns: list[np.ndarray], bg: int) -> np.ndarray:
    return stack_patterns([pat.copy() for pat in patterns], bg)


def _instance_is_valid(grid: np.ndarray, expected: np.ndarray, bg: int, expected_pair_areas: set[int]) -> bool:
    components = find_components(grid, bg)
    if len(components) % 2 != 0 or len(components) < 4:
        return False
    if {int(comp["area"]) for comp in components} - set(range(4, 9)):
        return False
    try:
        patterns = solve_unique_pairing(components, bg)
    except AssertionError:
        return False
    if {int(pat.size) for pat in patterns} != expected_pair_areas:
        return False
    return np.array_equal(stack_patterns(patterns, bg), expected)


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    pair_count = int(_RNG.integers(2, 5))
    areas = list(_RNG.choice(np.array(_AREA_OPTIONS), size=pair_count, replace=False))
    colors = [int(v) for v in _RNG.choice(np.arange(1, 10), size=2 * pair_count, replace=False)]
    _RNG.shuffle(colors)

    sprites: list[np.ndarray] = []
    expected_patterns: list[np.ndarray] = []
    for idx, area in enumerate(areas):
        options = _SPECS_BY_AREA[int(area)]
        h, w, orientation, profiles = options[int(_RNG.integers(len(options)))]
        profile = profiles[int(_RNG.integers(len(profiles)))]
        color_a = colors[2 * idx]
        color_b = colors[2 * idx + 1]
        sprite_a, sprite_b, composite = _build_pair(h, w, orientation, profile, color_a, color_b, bg)
        sprites.extend([sprite_a, sprite_b])
        expected_patterns.append(composite)

    max_sprite_h = max(sprite.shape[0] for sprite in sprites)
    max_sprite_w = max(sprite.shape[1] for sprite in sprites)
    min_h = max(_GRID_MIN, max_sprite_h + 4, pair_count * 4)
    min_w = max(_GRID_MIN, max_sprite_w + 4, pair_count * 4)
    if min_h > _GRID_MAX or min_w > _GRID_MAX:
        return None

    grid_h = int(_RNG.integers(min_h, _GRID_MAX + 1))
    grid_w = int(_RNG.integers(min_w, _GRID_MAX + 1))
    grid = _place_sprites(sprites, bg, grid_h, grid_w)
    if grid is None:
        return None

    expected = _expected_output(expected_patterns, bg)
    if not _instance_is_valid(grid, expected, bg, {int(arr.size) for arr in expected_patterns}):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid jagged-fragment instance")
