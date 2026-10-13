from __future__ import annotations

from collections import Counter

import numpy as np


PATTERN_COLOR_POOL = (1, 5, 8, 9)
TOKEN_COLOR_POOL = (2, 3, 4, 6, 7)


def _d4_orbit(r: int, c: int, n: int) -> tuple[tuple[int, int], ...]:
    last = n - 1
    pts = {
        (r, c),
        (r, last - c),
        (last - r, c),
        (last - r, last - c),
        (c, r),
        (c, last - r),
        (last - c, r),
        (last - c, last - r),
    }
    return tuple(sorted(pts))


def _is_d4_symmetric(grid: np.ndarray) -> bool:
    n = int(grid.shape[0])
    for r in range(n):
        for c in range(n):
            orbit = _d4_orbit(r, c, n)
            vals = {int(grid[rr, cc]) for rr, cc in orbit}
            if len(vals) != 1:
                return False
    return True


def _same_color_component_size(grid: np.ndarray, r: int, c: int) -> int:
    color = int(grid[r, c])
    stack = [(int(r), int(c))]
    seen = {(int(r), int(c))}
    n = int(grid.shape[0])
    while stack:
        cr, cc = stack.pop()
        for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
            if 0 <= nr < n and 0 <= nc < n and (nr, nc) not in seen and int(grid[nr, nc]) == color:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen)


def _zero_prefix(mask: np.ndarray) -> np.ndarray:
    pref = np.zeros((mask.shape[0] + 1, mask.shape[1] + 1), dtype=int)
    pref[1:, 1:] = np.cumsum(np.cumsum(mask.astype(int), axis=0), axis=1)
    return pref


def _rect_sum(pref: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> int:
    return int(pref[r1 + 1, c1 + 1] - pref[r0, c1 + 1] - pref[r1 + 1, c0] + pref[r0, c0])


def _all_zero_rectangles(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    zero = grid == 0
    pref = _zero_prefix(zero)
    n = int(grid.shape[0])
    rects: list[tuple[int, int, int, int]] = []
    for r0 in range(n):
        for c0 in range(n):
            if not zero[r0, c0]:
                continue
            for r1 in range(r0 + 1, n):
                for c1 in range(c0 + 1, n):
                    area = (r1 - r0 + 1) * (c1 - c0 + 1)
                    if area < 6:
                        continue
                    if _rect_sum(pref, r0, c0, r1, c1) == area:
                        rects.append((r0, c0, r1, c1))
    return rects


def _repair_rectangle(grid: np.ndarray, rect: tuple[int, int, int, int]) -> np.ndarray | None:
    n = int(grid.shape[0])
    r0, c0, r1, c1 = rect
    rect_cells = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
    repaired = np.array(grid, dtype=int, copy=True)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            outside_vals = [
                int(grid[rr, cc])
                for rr, cc in _d4_orbit(r, c, n)
                if (rr, cc) not in rect_cells and int(grid[rr, cc]) != 0
            ]
            if not outside_vals:
                return None
            counts = Counter(outside_vals).most_common()
            if len(counts) > 1 and counts[0][1] == counts[1][1]:
                return None
            repaired[r, c] = int(counts[0][0])
    return repaired


def _find_tokens(repaired: np.ndarray) -> list[tuple[int, int, int, int]]:
    n = int(repaired.shape[0])
    tokens: list[tuple[int, int, int, int]] = []
    seen_orbits: set[tuple[tuple[int, int], ...]] = set()
    for r in range(n):
        for c in range(n):
            color = int(repaired[r, c])
            if color == 0:
                continue
            if _same_color_component_size(repaired, r, c) != 1:
                continue
            orbit = _d4_orbit(r, c, n)
            if len(orbit) == 1:
                continue
            others = [int(repaired[rr, cc]) for rr, cc in orbit if (rr, cc) != (r, c)]
            if len(set(others)) != 1:
                continue
            underlying = int(others[0])
            if color == underlying:
                continue
            if orbit in seen_orbits:
                return []
            seen_orbits.add(orbit)
            tokens.append((r, c, color, underlying))
    return sorted(tokens)


def _strip_tokens(repaired: np.ndarray, tokens: list[tuple[int, int, int, int]]) -> np.ndarray:
    base = np.array(repaired, dtype=int, copy=True)
    for r, c, _color, underlying in tokens:
        base[r, c] = int(underlying)
    return base


def _find_missing_rectangle(grid: np.ndarray) -> tuple[int, int, int, int]:
    n = int(grid.shape[0])
    candidates: list[tuple[int, int, int, int]] = []
    for rect in _all_zero_rectangles(grid):
        repaired = _repair_rectangle(grid, rect)
        if repaired is None:
            continue
        tokens = _find_tokens(repaired)
        if not (3 <= len(tokens) <= 5):
            continue
        if len({c for _r, c, _color, _u in tokens}) != len(tokens):
            continue
        if any(r >= n // 2 for r, _c, _color, _u in tokens):
            continue
        base = _strip_tokens(repaired, tokens)
        if not _is_d4_symmetric(base):
            continue
        candidates.append(rect)
    assert len(candidates) == 1, f"expected one recoverable missing rectangle, found {len(candidates)}"
    return candidates[0]


def _drop_row(grid: np.ndarray, start_r: int, c: int) -> int:
    r = int(start_r)
    n = int(grid.shape[0])
    while r + 1 < n and int(grid[r + 1, c]) == 0:
        r += 1
    return r


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the hidden D4-symmetric terrain and then let the floating tokens settle under gravity.

    Input structure:
    - The input is a square grid whose non-token pattern is D4-symmetric under horizontal, vertical, and both diagonal reflections.
    - One rectangular patch of that pattern has been erased to black, while 3-5 isolated single-pixel tokens sit in the upper half.
    - Token colors do not belong to the symmetric terrain palette, and each token column has some eventual landing surface below.

    Rule:
    - Locate the all-black missing rectangle whose cells can be recovered from their D4 symmetry orbits.
    - Fill each missing cell with the unique non-black value supported by the rest of its orbit, restoring the full symmetric terrain.
    - Identify token pixels as isolated orbit-breaking outliers, restore their original terrain values, then drop them straight downward.
    - Process tokens from lower row to higher row; each token stops when the cell directly below is non-black terrain or an already settled token.

    Key insight:
    - The erased rectangle hides part of the terrain, so symmetry repair must happen before gravity; otherwise some tokens would fall past their true reconstructed landing surfaces.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] == g.shape[1], "expected a square 2D grid"

    rect = _find_missing_rectangle(g)
    repaired = _repair_rectangle(g, rect)
    assert repaired is not None
    tokens = _find_tokens(repaired)
    assert 3 <= len(tokens) <= 5

    out = _strip_tokens(repaired, tokens)
    for start_r, c, color, _underlying in sorted(tokens, key=lambda item: (-item[0], item[1])):
        final_r = _drop_row(out, start_r, c)
        out[final_r, c] = int(color)
    return out


def _paint_domain_orbit(domain: dict[tuple[int, int], int], center: tuple[int, int], radius: int, color: int) -> None:
    cr, cc = center
    for (r, c) in list(domain.keys()):
        if max(abs(r - cr), abs(c - cc)) <= radius:
            domain[(r, c)] = int(color)


def _domain_to_grid(domain: dict[tuple[int, int], int], n: int) -> np.ndarray:
    grid = np.zeros((n, n), dtype=int)
    for (r, c), color in domain.items():
        if color == 0:
            continue
        for rr, cc in _d4_orbit(r, c, n):
            grid[rr, cc] = int(color)
    return grid


def _generate_pattern(rng: np.random.Generator) -> np.ndarray:
    n = int(rng.choice([11, 13, 15]))
    palette = tuple(int(x) for x in rng.choice(PATTERN_COLOR_POOL, size=int(rng.integers(2, 4)), replace=False))
    m = n // 2 + 1
    reps = [(r, c) for r in range(m) for c in range(r, m)]
    domain = {coord: 0 for coord in reps}

    cluster_count = int(rng.integers(4, 8))
    for _ in range(cluster_count):
        center = reps[int(rng.integers(0, len(reps)))]
        radius = int(rng.integers(1, 3))
        color = int(rng.choice(palette))
        _paint_domain_orbit(domain, center, radius, color)

    accent_count = int(rng.integers(4, 8))
    for _ in range(accent_count):
        coord = reps[int(rng.integers(0, len(reps)))]
        color = int(rng.choice(palette))
        domain[coord] = color

    # Keep some structured empty channels so gravity paths remain readable.
    mid = n // 2
    grid = _domain_to_grid(domain, n)
    for ring in range(1, mid + 1):
        if rng.random() < 0.35:
            mask = np.fromfunction(
                lambda r, c, ring=ring: (np.maximum(np.abs(r - mid), np.abs(c - mid)) == ring)
                & (((np.abs(r - mid) + np.abs(c - mid)) % 2) == int(rng.integers(0, 2))),
                (n, n),
                dtype=int,
            )
            grid[mask.astype(bool)] = 0

    density = float(np.count_nonzero(grid)) / float(grid.size)
    used = {int(x) for x in np.unique(grid)} - {0}
    if not (0.22 <= density <= 0.62):
        raise ValueError("pattern density out of range")
    if len(used) < 2:
        raise ValueError("pattern uses too few colors")
    if not _is_d4_symmetric(grid):
        raise ValueError("pattern lost D4 symmetry")
    return grid


def _rect_cells(rect: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _candidate_rectangles(pattern: np.ndarray) -> list[tuple[int, int, int, int]]:
    n = int(pattern.shape[0])
    mid = n // 2
    rects: list[tuple[int, int, int, int]] = []
    for h in range(3, min(5, n - 2)):
        for w in range(3, min(6, n - 2)):
            for r0 in range(max(1, mid - 1), n - h):
                c_center = mid - w // 2
                for dc in (-1, 0, 1):
                    c0 = c_center + dc
                    if c0 < 1 or c0 + w >= n:
                        continue
                    r1, c1 = r0 + h - 1, c0 + w - 1
                    block = pattern[r0 : r1 + 1, c0 : c1 + 1]
                    if np.any(block == 0):
                        continue
                    # Do not erase an entire orbit.
                    rect_set = _rect_cells((r0, c0, r1, c1))
                    ok = True
                    for r in range(r0, r1 + 1):
                        for c in range(c0, c1 + 1):
                            if all((rr, cc) in rect_set for rr, cc in _d4_orbit(r, c, n)):
                                ok = False
                                break
                        if not ok:
                            break
                    if ok:
                        rects.append((r0, c0, r1, c1))
    return rects


def _list_start_candidates(
    pattern: np.ndarray,
    broken: np.ndarray,
    rect: tuple[int, int, int, int],
) -> list[dict[str, int | bool]]:
    n = int(pattern.shape[0])
    rect_set = _rect_cells(rect)
    cands: list[dict[str, int | bool]] = []
    for r in range(0, n // 2):
        for c in range(n):
            if int(pattern[r, c]) != 0:
                continue
            if (r, c) in rect_set:
                continue
            if len(_d4_orbit(r, c, n)) == 1:
                continue
            final_good = _drop_row(pattern, r, c)
            final_bad = _drop_row(broken, r, c)
            if final_good <= r:
                continue
            surface = final_good + 1
            dependent = final_good != final_bad
            cands.append(
                {
                    "start_r": int(r),
                    "col": int(c),
                    "final_r": int(final_good),
                    "broken_final_r": int(final_bad),
                    "surface_r": int(surface),
                    "dependent": bool(dependent),
                    "surface_in_rect": bool((surface, c) in rect_set),
                }
            )
    return cands


def _choose_tokens(
    rng: np.random.Generator,
    pattern: np.ndarray,
    broken: np.ndarray,
    rect: tuple[int, int, int, int],
) -> list[dict[str, int | bool]]:
    n = int(pattern.shape[0])
    cands = _list_start_candidates(pattern, broken, rect)
    dep = [cand for cand in cands if cand["dependent"] and cand["surface_in_rect"]]
    indep = [cand for cand in cands if not cand["dependent"]]
    if not dep or len(indep) < 2:
        raise ValueError("not enough token candidate diversity")

    rng.shuffle(dep)
    rng.shuffle(indep)
    token_count = int(rng.integers(3, 6))
    selected: list[dict[str, int | bool]] = []
    used_cols: set[int] = set()
    used_orbits: set[tuple[tuple[int, int], ...]] = set()

    def try_add(cand: dict[str, int | bool]) -> bool:
        start_r = int(cand["start_r"])
        col = int(cand["col"])
        orbit = _d4_orbit(start_r, col, n)
        if col in used_cols or orbit in used_orbits:
            return False
        used_cols.add(col)
        used_orbits.add(orbit)
        selected.append(cand)
        return True

    for cand in dep:
        if try_add(cand):
            break
    for cand in indep:
        if try_add(cand):
            if len(selected) >= 3:
                break
    rng.shuffle(cands)
    for cand in cands:
        if len(selected) >= token_count:
            break
        try_add(cand)

    if len(selected) < token_count:
        raise ValueError("failed to select enough non-conflicting tokens")
    if not any(bool(cand["dependent"]) for cand in selected):
        raise ValueError("missing dependency token")
    if not any(not bool(cand["dependent"]) for cand in selected):
        raise ValueError("missing original-surface token")
    if max(int(cand["final_r"]) - int(cand["start_r"]) for cand in selected) < 3:
        raise ValueError("all token drops are too short")
    return selected


def _build_input_output(
    rng: np.random.Generator,
    pattern: np.ndarray,
    rect: tuple[int, int, int, int],
    tokens: list[dict[str, int | bool]],
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.array(pattern, dtype=int, copy=True)
    out = np.array(pattern, dtype=int, copy=True)
    r0, c0, r1, c1 = rect
    inp[r0 : r1 + 1, c0 : c1 + 1] = 0

    token_colors = [int(x) for x in rng.choice(TOKEN_COLOR_POOL, size=len(tokens), replace=False)]
    for color, token in zip(token_colors, tokens):
        sr = int(token["start_r"])
        c = int(token["col"])
        fr = int(token["final_r"])
        inp[sr, c] = int(color)
        out[fr, c] = int(color)
    return inp, out


def _validate_instance(
    inp: np.ndarray,
    expected: np.ndarray,
    pattern: np.ndarray,
    rect: tuple[int, int, int, int],
    tokens: list[dict[str, int | bool]],
) -> bool:
    if not np.array_equal(main(inp), expected):
        return False
    found_rect = _find_missing_rectangle(inp)
    if found_rect != rect:
        return False
    repaired = _repair_rectangle(inp, rect)
    if repaired is None:
        return False
    found_tokens = _find_tokens(repaired)
    base = _strip_tokens(repaired, found_tokens)
    if not np.array_equal(base, pattern):
        return False
    if not _is_d4_symmetric(pattern):
        return False
    if len(tokens) != len(found_tokens):
        return False
    if len({int(token["col"]) for token in tokens}) != len(tokens):
        return False
    if not any(bool(token["dependent"]) for token in tokens):
        return False
    if not any(not bool(token["dependent"]) for token in tokens):
        return False
    if np.array_equal(expected, pattern):
        return False
    # Semantic check: at least one token truly changes landing if the rectangle is not repaired.
    broken = np.array(pattern, dtype=int, copy=True)
    r0, c0, r1, c1 = rect
    broken[r0 : r1 + 1, c0 : c1 + 1] = 0
    if not any(_drop_row(pattern, int(token["start_r"]), int(token["col"])) != _drop_row(broken, int(token["start_r"]), int(token["col"])) for token in tokens):
        return False
    return True


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(800):
        try:
            pattern = _generate_pattern(rng)
            rects = _candidate_rectangles(pattern)
            if not rects:
                continue
            rng.shuffle(rects)
            for rect in rects[:20]:
                broken = np.array(pattern, dtype=int, copy=True)
                r0, c0, r1, c1 = rect
                broken[r0 : r1 + 1, c0 : c1 + 1] = 0
                try:
                    tokens = _choose_tokens(rng, pattern, broken, rect)
                except ValueError:
                    continue
                inp, expected = _build_input_output(rng, pattern, rect, tokens)
                if _validate_instance(inp, expected, pattern, rect, tokens):
                    return inp
        except ValueError:
            continue
    raise RuntimeError("failed to generate a valid ARC instance")
