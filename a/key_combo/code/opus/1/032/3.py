import random
from typing import Callable

import numpy as np


BACKGROUND = 0
DAMAGE = 3
SEPARATOR = 7
PALETTE = (1, 2, 4, 5, 6, 8, 9)
MAX_ATTEMPTS = 400

_ORBIT_CACHE: dict[tuple[int, int], list[tuple[tuple[int, int], ...]]] = {}


def _orbit_positions(h: int, w: int, r: int, c: int) -> tuple[tuple[int, int], ...]:
    return tuple(sorted({(r, c), (h - 1 - r, c), (r, w - 1 - c), (h - 1 - r, w - 1 - c)}))


def _all_orbits(h: int, w: int) -> list[tuple[tuple[int, int], ...]]:
    key = (h, w)
    if key not in _ORBIT_CACHE:
        seen = set()
        orbits = []
        for r in range(h):
            for c in range(w):
                orbit = _orbit_positions(h, w, r, c)
                if orbit not in seen:
                    seen.add(orbit)
                    orbits.append(orbit)
        _ORBIT_CACHE[key] = orbits
    return _ORBIT_CACHE[key]


def _find_separator_col(grid: np.ndarray) -> int:
    h, w = grid.shape
    assert w % 2 == 1, "grid width must be odd"
    candidates = [c for c in range(w) if np.all(grid[:, c] == SEPARATOR) and c * 2 + 1 == w]
    assert len(candidates) == 1, "expected a unique centered separator column"
    return candidates[0]


def _compare_panels(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    assert left.shape == right.shape
    mid = np.full(left.shape, DAMAGE, dtype=int)
    same = left == right
    mid[same] = left[same]
    return mid


def _repair_from_symmetry(mid: np.ndarray) -> np.ndarray:
    h, w = mid.shape
    out = mid.copy()
    for orbit in _all_orbits(h, w):
        visible = {int(mid[r, c]) for r, c in orbit if int(mid[r, c]) != DAMAGE}
        assert len(visible) == 1, "each damaged orbit must have one unique visible color"
        color = visible.pop()
        for r, c in orbit:
            if out[r, c] == DAMAGE:
                out[r, c] = color
    assert not np.any(out == DAMAGE), "all damage placeholders must be repaired"
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compare two noisy symmetric panels, mark disagreements as damage, then repair them by double reflection symmetry.

    Input structure:
    - The grid has black background and one centered orange separator column.
    - The separator splits the grid into equal-width left and right panels.
    - Each panel shows the same underlying multicolor design with independent pixel noise, and the design is symmetric across both the horizontal and vertical axes of a single panel.

    Rule:
    - Remove the orange separator and align the two panels cell by cell.
    - If both panels show the same color at a position, keep that color in the intermediate single-panel grid.
    - If the two panels disagree, write green as a damage placeholder.
    - For every green cell, take its orbit under horizontal and vertical reflection inside the single panel and replace it with the unique non-green color visible elsewhere in that orbit.

    Key insight:
    - Panel comparison only identifies which locations are trustworthy versus damaged; the actual recovered colors come from enforcing the hidden two-axis symmetry of the underlying design.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep = _find_separator_col(g)
    left = g[:, :sep]
    right = g[:, sep + 1 :]
    assert left.shape == right.shape
    assert not np.any(left == SEPARATOR) and not np.any(right == SEPARATOR)

    mid = _compare_panels(left, right)
    return _repair_from_symmetry(mid)


def _reflect_representatives(rep: np.ndarray, h: int, w: int) -> np.ndarray:
    base = np.full((h, w), BACKGROUND, dtype=int)
    for r in range(rep.shape[0]):
        for c in range(rep.shape[1]):
            color = int(rep[r, c])
            for rr, cc in _orbit_positions(h, w, r, c):
                base[rr, cc] = color
    return base


def _paint_shape(rep: np.ndarray, color: int) -> None:
    rh, rw = rep.shape
    kind = random.choice(("rect", "rect", "hline", "vline", "point", "lshape"))
    if kind == "rect":
        r0 = random.randint(0, rh - 1)
        c0 = random.randint(0, rw - 1)
        r1 = random.randint(r0, min(rh - 1, r0 + 2))
        c1 = random.randint(c0, min(rw - 1, c0 + 2))
        rep[r0 : r1 + 1, c0 : c1 + 1] = color
    elif kind == "hline":
        r = random.randint(0, rh - 1)
        c0 = random.randint(0, rw - 1)
        c1 = random.randint(c0, rw - 1)
        rep[r, c0 : c1 + 1] = color
    elif kind == "vline":
        c = random.randint(0, rw - 1)
        r0 = random.randint(0, rh - 1)
        r1 = random.randint(r0, rh - 1)
        rep[r0 : r1 + 1, c] = color
    elif kind == "point":
        rep[random.randint(0, rh - 1), random.randint(0, rw - 1)] = color
    else:
        r = random.randint(0, rh - 1)
        c = random.randint(0, rw - 1)
        rep[r, c] = color
        if r + 1 < rh:
            rep[r + 1, c] = color
        if c + 1 < rw:
            rep[r, c + 1] = color


def _good_base(base: np.ndarray) -> bool:
    h, w = base.shape
    if not np.array_equal(base, base[::-1, :]):
        return False
    if not np.array_equal(base, base[:, ::-1]):
        return False

    colors = sorted(int(x) for x in np.unique(base) if int(x) != BACKGROUND)
    if not (2 <= len(colors) <= 3):
        return False
    for color in colors:
        if int(np.sum(base == color)) < 2:
            return False

    fg_cells = int(np.sum(base != BACKGROUND))
    bg_cells = int(np.sum(base == BACKGROUND))
    if fg_cells < 8 or fg_cells > int(h * w * 0.72):
        return False
    if bg_cells < max(6, h):
        return False

    orbits = _all_orbits(h, w)
    fg_orbits = [orbit for orbit in orbits if int(base[orbit[0]]) != BACKGROUND and len(orbit) >= 2]
    bg_orbits = [orbit for orbit in orbits if int(base[orbit[0]]) == BACKGROUND and len(orbit) >= 2]
    if len(fg_orbits) < 3 or len(bg_orbits) < 2:
        return False

    return True


def _generate_base_panel(h: int, w: int) -> np.ndarray | None:
    rh, rw = (h + 1) // 2, (w + 1) // 2
    for _ in range(120):
        n_colors = random.choice((2, 3))
        palette = random.sample(PALETTE, k=n_colors)
        rep = np.zeros((rh, rw), dtype=int)

        for color in palette:
            _paint_shape(rep, color)
        for _ in range(random.randint(n_colors + 1, n_colors + 4)):
            _paint_shape(rep, random.choice(palette))

        base = _reflect_representatives(rep, h, w)
        if _good_base(base):
            return base
    return None


def _choose_orbit(
    records: list[dict[str, object]],
    remaining: dict[int, int],
    predicate: Callable[[dict[str, object]], bool],
    excluded: set[int] | None = None,
) -> int | None:
    excluded = excluded or set()
    candidates = [
        idx
        for idx, rec in enumerate(records)
        if idx not in excluded and remaining[idx] > 0 and predicate(rec)
    ]
    return random.choice(candidates) if candidates else None


def _make_wrong_color(base_color: int, visible_colors: list[int]) -> int:
    if base_color == BACKGROUND:
        return random.choice(visible_colors)
    if random.random() < 0.5:
        return BACKGROUND
    others = [color for color in visible_colors if color != base_color]
    return random.choice(others or [BACKGROUND])


def _inject_noise(base: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    left = base.copy()
    right = base.copy()
    h, w = base.shape
    visible_colors = sorted(int(x) for x in np.unique(base) if int(x) != BACKGROUND)
    records = [
        {
            "orbit": orbit,
            "color": int(base[orbit[0]]),
            "size": len(orbit),
        }
        for orbit in _all_orbits(h, w)
    ]
    remaining = {idx: int(rec["size"]) - 1 for idx, rec in enumerate(records)}
    used_positions: set[tuple[int, int]] = set()
    damaged_per_orbit = {idx: 0 for idx in range(len(records))}
    left_count = 0
    right_count = 0

    def add_damage(orbit_idx: int, panel: str, wrong_color: int) -> None:
        nonlocal left_count, right_count
        rec = records[orbit_idx]
        base_color = int(rec["color"])
        assert wrong_color != base_color
        choices = [pos for pos in rec["orbit"] if pos not in used_positions]
        assert choices and remaining[orbit_idx] > 0
        r, c = random.choice(choices)
        target = left if panel == "left" else right
        assert target[r, c] == base_color
        target[r, c] = wrong_color
        used_positions.add((r, c))
        remaining[orbit_idx] -= 1
        damaged_per_orbit[orbit_idx] += 1
        if panel == "left":
            left_count += 1
        else:
            right_count += 1

    fg_bg_idx = _choose_orbit(
        records,
        remaining,
        lambda rec: int(rec["color"]) != BACKGROUND and int(rec["size"]) >= 2,
    )
    fg_fg_idx = _choose_orbit(
        records,
        remaining,
        lambda rec: int(rec["color"]) != BACKGROUND and int(rec["size"]) >= 2,
        excluded={fg_bg_idx} if fg_bg_idx is not None else None,
    )
    bg_fg_idx = _choose_orbit(
        records,
        remaining,
        lambda rec: int(rec["color"]) == BACKGROUND and int(rec["size"]) >= 2,
        excluded={x for x in (fg_bg_idx, fg_fg_idx) if x is not None},
    )
    multi_idx = _choose_orbit(
        records,
        remaining,
        lambda rec: int(rec["size"]) >= 4,
        excluded={x for x in (fg_bg_idx, fg_fg_idx, bg_fg_idx) if x is not None},
    )
    if None in (fg_bg_idx, fg_fg_idx, bg_fg_idx, multi_idx):
        return None

    add_damage(int(fg_bg_idx), "left", BACKGROUND)
    fg_fg_base = int(records[int(fg_fg_idx)]["color"])
    other_colors = [color for color in visible_colors if color != fg_fg_base]
    if not other_colors:
        return None
    add_damage(int(fg_fg_idx), "right", random.choice(other_colors))
    add_damage(int(bg_fg_idx), "left", random.choice(visible_colors))

    multi_base = int(records[int(multi_idx)]["color"])
    add_damage(int(multi_idx), "left", _make_wrong_color(multi_base, visible_colors))
    add_damage(int(multi_idx), "right", _make_wrong_color(multi_base, visible_colors))

    left_target = random.randint(left_count, 4)
    right_target = random.randint(right_count, 4)
    for _ in range(20):
        if left_count >= left_target and right_count >= right_target:
            break
        panel = "left" if left_count < left_target and (right_count >= right_target or random.random() < 0.6) else "right"
        orbit_idx = _choose_orbit(
            records,
            remaining,
            lambda rec: int(rec["size"]) >= 2,
        )
        if orbit_idx is None:
            return None
        base_color = int(records[int(orbit_idx)]["color"])
        add_damage(int(orbit_idx), panel, _make_wrong_color(base_color, visible_colors))
    if not (2 <= left_count <= 4 and 2 <= right_count <= 4):
        return None

    if not np.array_equal(left, base) and not np.array_equal(right, base):
        multi_damaged = sum(count >= 2 for count in damaged_per_orbit.values())
        if multi_damaged >= 1:
            return left, right
    return None


def _validate_instance(grid: np.ndarray, base: np.ndarray, left: np.ndarray, right: np.ndarray) -> bool:
    h, w = base.shape
    if grid.shape != (h, 2 * w + 1):
        return False
    if not np.array_equal(grid[:, w], np.full(h, SEPARATOR, dtype=int)):
        return False
    if np.any(left == DAMAGE) or np.any(right == DAMAGE) or np.any(left == SEPARATOR) or np.any(right == SEPARATOR):
        return False
    if not _good_base(base):
        return False

    left_err = {tuple(map(int, rc)) for rc in np.argwhere(left != base)}
    right_err = {tuple(map(int, rc)) for rc in np.argwhere(right != base)}
    if not (2 <= len(left_err) <= 4 and 2 <= len(right_err) <= 4):
        return False
    if left_err & right_err:
        return False

    mid = _compare_panels(left, right)
    if int(np.sum(mid == DAMAGE)) != len(left_err) + len(right_err):
        return False

    saw_bg_to_fg = False
    saw_fg_to_bg = False
    saw_fg_to_fg = False
    orbits_with_multiple_damage = 0
    for orbit in _all_orbits(h, w):
        base_values = {int(base[r, c]) for r, c in orbit}
        if len(base_values) != 1:
            return False
        orbit_color = next(iter(base_values))
        greens = [(r, c) for r, c in orbit if int(mid[r, c]) == DAMAGE]
        if greens:
            visible = {int(mid[r, c]) for r, c in orbit if int(mid[r, c]) != DAMAGE}
            if visible != {orbit_color}:
                return False
            if len(greens) >= len(orbit):
                return False
            if len(greens) >= 2:
                orbits_with_multiple_damage += 1
        for r, c in orbit:
            lv = int(left[r, c])
            rv = int(right[r, c])
            bv = int(base[r, c])
            if (r, c) in left_err:
                if lv == bv or rv != bv:
                    return False
                if bv == BACKGROUND and lv != BACKGROUND:
                    saw_bg_to_fg = True
                if bv != BACKGROUND and lv == BACKGROUND:
                    saw_fg_to_bg = True
                if bv != BACKGROUND and lv not in (BACKGROUND, bv):
                    saw_fg_to_fg = True
            if (r, c) in right_err:
                if rv == bv or lv != bv:
                    return False
                if bv == BACKGROUND and rv != BACKGROUND:
                    saw_bg_to_fg = True
                if bv != BACKGROUND and rv == BACKGROUND:
                    saw_fg_to_bg = True
                if bv != BACKGROUND and rv not in (BACKGROUND, bv):
                    saw_fg_to_fg = True

    if not (saw_bg_to_fg and saw_fg_to_bg and saw_fg_to_fg and orbits_with_multiple_damage >= 1):
        return False

    prefer_left_nonzero = np.where(left != BACKGROUND, left, right)
    prefer_right_nonzero = np.where(right != BACKGROUND, right, left)
    if np.array_equal(prefer_left_nonzero, base) or np.array_equal(prefer_right_nonzero, base):
        return False

    if not np.array_equal(_repair_from_symmetry(mid), base):
        return False
    if not np.array_equal(main(grid), base):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(7, 11)
    w = random.randint(5, 8)
    base = _generate_base_panel(h, w)
    if base is None:
        return None
    noisy = _inject_noise(base)
    if noisy is None:
        return None
    left, right = noisy
    grid = np.concatenate(
        [left, np.full((h, 1), SEPARATOR, dtype=int), right],
        axis=1,
    )
    if _validate_instance(grid, base, left, right):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid symmetric panel damage task")
